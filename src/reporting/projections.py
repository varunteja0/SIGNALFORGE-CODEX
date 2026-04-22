from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HorizonProjection:
    """Distribution of ending capital for a projected horizon."""

    label: str
    years: int
    periods: int
    ruin_probability: float
    p05: float
    p50: float
    p95: float


def resample_equity_to_returns(
    equity_curve: pd.Series,
    *,
    frequency: str = "1D",
) -> pd.Series:
    """Convert an equity curve into finite period returns."""
    if equity_curve is None or len(equity_curve) < 2:
        return pd.Series(dtype=float)

    equity = equity_curve.sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]

    if isinstance(equity.index, pd.DatetimeIndex):
        equity = equity.resample(frequency).last().dropna()

    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return returns.astype(float)


def _sample_block_paths(
    returns: np.ndarray,
    *,
    horizon_periods: int,
    block_size: int,
    n_sims: int,
    seed: int,
    chunk_size: int,
) -> np.ndarray:
    if horizon_periods <= 0:
        raise ValueError("horizon_periods must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")

    usable = np.asarray(returns, dtype=float)
    usable = usable[np.isfinite(usable)]
    if usable.size == 0:
        raise ValueError("returns must contain at least one finite value")

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(horizon_periods / block_size)
    offsets = np.arange(block_size, dtype=int)
    sampled = np.empty((n_sims, horizon_periods), dtype=float)

    for start in range(0, n_sims, chunk_size):
        stop = min(start + chunk_size, n_sims)
        local_size = stop - start
        starts = rng.integers(0, usable.size, size=(local_size, n_blocks))
        block_idx = (starts[..., None] + offsets) % usable.size
        chunk = usable[block_idx].reshape(local_size, n_blocks * block_size)
        sampled[start:stop] = chunk[:, :horizon_periods]

    return sampled


def project_horizon_distribution(
    period_returns: pd.Series,
    *,
    starting_capital: float,
    horizon_periods: int,
    block_size: int = 21,
    n_sims: int = 10_000,
    ruin_threshold: float = 0.25,
    seed: int = 42,
    chunk_size: int = 512,
) -> dict[str, float]:
    """Project ending-capital distribution with a circular block bootstrap.

    Ruin is path-based: a simulation is counted as ruined if equity touches
    ``starting_capital * ruin_threshold`` at any point before the horizon ends.
    """
    if starting_capital <= 0.0:
        raise ValueError("starting_capital must be positive")
    if not 0.0 < ruin_threshold < 1.0:
        raise ValueError("ruin_threshold must be between 0 and 1")

    sampled = _sample_block_paths(
        period_returns.to_numpy(dtype=float),
        horizon_periods=horizon_periods,
        block_size=block_size,
        n_sims=n_sims,
        seed=seed,
        chunk_size=chunk_size,
    )
    growth = np.cumprod(1.0 + sampled, axis=1)
    capital_paths = starting_capital * growth
    final_capitals = capital_paths[:, -1]
    ruin_level = starting_capital * ruin_threshold
    ruined = np.min(capital_paths, axis=1) <= ruin_level

    return {
        "ruin_probability": float(ruined.mean()),
        "p05": float(np.percentile(final_capitals, 5)),
        "p50": float(np.percentile(final_capitals, 50)),
        "p95": float(np.percentile(final_capitals, 95)),
    }


def project_horizon_table(
    period_returns: pd.Series,
    *,
    starting_capital: float,
    horizons_years: tuple[int, ...] = (1, 3, 10),
    periods_per_year: int = 365,
    block_size: int = 21,
    n_sims: int = 10_000,
    ruin_threshold: float = 0.25,
    seed: int = 42,
    chunk_size: int = 512,
) -> list[HorizonProjection]:
    """Project ending-capital percentiles across multiple horizons."""
    projections: list[HorizonProjection] = []
    for years in horizons_years:
        periods = int(years * periods_per_year)
        dist = project_horizon_distribution(
            period_returns,
            starting_capital=starting_capital,
            horizon_periods=periods,
            block_size=block_size,
            n_sims=n_sims,
            ruin_threshold=ruin_threshold,
            seed=seed + years,
            chunk_size=chunk_size,
        )
        projections.append(
            HorizonProjection(
                label=f"{years}y",
                years=years,
                periods=periods,
                ruin_probability=dist["ruin_probability"],
                p05=dist["p05"],
                p50=dist["p50"],
                p95=dist["p95"],
            )
        )
    return projections
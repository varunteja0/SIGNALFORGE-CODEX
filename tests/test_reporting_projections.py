from __future__ import annotations

import math

import pandas as pd
import pytest

from src.reporting.projections import project_horizon_table, resample_equity_to_returns


def test_resample_equity_to_returns_respects_daily_compounding() -> None:
    index = pd.date_range("2024-01-01", periods=49, freq="12h")
    equity = pd.Series([100.0 + idx for idx in range(len(index))], index=index)

    returns = resample_equity_to_returns(equity, frequency="1D")

    assert not returns.empty
    assert isinstance(returns.index, pd.DatetimeIndex)
    assert all(value == value for value in returns)


def test_project_horizon_table_is_deterministic_for_constant_returns() -> None:
    returns = pd.Series([0.01] * 30)

    projections = project_horizon_table(
        returns,
        starting_capital=10_000,
        horizons_years=(1,),
        periods_per_year=5,
        block_size=3,
        n_sims=128,
        ruin_threshold=0.25,
        seed=7,
        chunk_size=32,
    )

    assert len(projections) == 1
    row = projections[0]
    expected = 10_000 * math.pow(1.01, 5)
    assert row.ruin_probability == 0.0
    assert row.p05 == pytest.approx(expected)
    assert row.p50 == pytest.approx(expected)
    assert row.p95 == pytest.approx(expected)


def test_project_horizon_table_flags_pathwise_ruin() -> None:
    returns = pd.Series([-0.60] * 10)

    projections = project_horizon_table(
        returns,
        starting_capital=10_000,
        horizons_years=(1,),
        periods_per_year=1,
        block_size=1,
        n_sims=64,
        ruin_threshold=0.50,
        seed=11,
        chunk_size=16,
    )

    row = projections[0]
    assert row.ruin_probability == 1.0
    assert row.p95 == 4_000.0
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.arena.engine import (
    CandidateSpec,
    build_positions,
    evaluate_candidate,
    write_submission,
)


def _synthetic_bars() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=1200, freq="1h", tz="UTC")
    rng = np.random.default_rng(7)
    bases = {
        "BTC/USDT": 40_000.0,
        "ETH/USDT": 2_000.0,
        "SOL/USDT": 80.0,
    }
    drifts = {
        "BTC/USDT": 0.0002,
        "ETH/USDT": 0.0001,
        "SOL/USDT": -0.0001,
    }

    out: dict[str, pd.DataFrame] = {}
    for symbol in bases:
        ret = rng.normal(drifts[symbol], 0.006, len(index))
        close = bases[symbol] * np.exp(ret.cumsum())
        high = close * (1 + rng.uniform(0.0, 0.003, len(index)))
        low = close * (1 - rng.uniform(0.0, 0.003, len(index)))
        open_ = np.concatenate([[close[0]], close[:-1]])
        volume = rng.uniform(100.0, 1_000.0, len(index))
        out[symbol] = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=index,
        )
    return out


def test_build_positions_respects_bounds_and_alignment():
    bars = _synthetic_bars()
    spec = CandidateSpec(
        name="trend_test",
        family="trend",
        params={
            "fast": 24,
            "slow": 96,
            "confirm": 72,
            "enter": 0.75,
            "exit": 0.20,
            "target_vol": 0.006,
            "vol_cap": 0.03,
        },
    )

    positions = build_positions(bars, spec)
    assert set(positions) == set(bars)
    for symbol, series in positions.items():
        assert series.index.equals(bars[symbol].index)
        assert float(series.abs().max()) <= 1.0 + 1e-9


def test_position_generation_is_prefix_invariant_before_tail_change():
    bars = _synthetic_bars()
    altered = {symbol: frame.copy() for symbol, frame in bars.items()}
    for frame in altered.values():
        frame.loc[frame.index[-40:], "close"] *= 1.25
        frame.loc[frame.index[-40:], "high"] = frame.loc[frame.index[-40:], "close"] * 1.01
        frame.loc[frame.index[-40:], "low"] = frame.loc[frame.index[-40:], "close"] * 0.99

    spec = CandidateSpec(
        name="relative_test",
        family="relative",
        params={
            "lookback": 168,
            "smooth": 24,
            "enter_spread": 0.40,
            "exit_spread": 0.15,
            "switch_buffer": 0.10,
            "target_vol": 0.005,
        },
    )

    base_positions = build_positions(bars, spec)
    altered_positions = build_positions(altered, spec)
    cutoff = bars["BTC/USDT"].index[-41]
    for symbol in bars:
        pd.testing.assert_series_equal(
            base_positions[symbol].loc[:cutoff],
            altered_positions[symbol].loc[:cutoff],
        )


def test_write_submission_emits_expected_contract(tmp_path):
    bars = _synthetic_bars()
    spec = CandidateSpec(
        name="pull_test",
        family="pullback",
        params={
            "bias_fast": 24,
            "bias_slow": 168,
            "stretch_span": 24,
            "enter": 1.25,
            "exit": 0.25,
            "target_vol": 0.005,
            "vol_cap": 0.028,
        },
    )
    result = evaluate_candidate(bars, spec, n_trials=3)

    out_dir = write_submission(
        tmp_path / "submission",
        engine_name="arena-test-engine",
        best=result,
        notes="synthetic test",
        n_trials=3,
    )

    submission = json.loads((out_dir / "submission.json").read_text())
    assert submission["engine"] == "arena-test-engine"
    assert submission["n_trials"] == 3
    assert abs(sum(submission["weights"].values()) - 1.0) < 1e-9

    for symbol in bars:
        flat = symbol.replace("/", "_")
        path = out_dir / "signals" / f"{flat}.parquet"
        frame = pd.read_parquet(path)
        assert list(frame.columns) == ["position"]
        assert float(frame["position"].abs().max()) <= 1.0 + 1e-9
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert frame.index.tz is None


def test_evaluate_candidate_includes_robustness_metrics():
    bars = _synthetic_bars()
    spec = CandidateSpec(
        name="trend_test",
        family="trend",
        params={
            "fast": 24,
            "slow": 96,
            "confirm": 72,
            "enter": 0.75,
            "exit": 0.20,
            "target_vol": 0.006,
            "vol_cap": 0.03,
        },
    )

    result = evaluate_candidate(bars, spec, n_trials=5)
    metrics = result.portfolio_metrics
    for key in (
        "is_fold_sharpe_mean",
        "is_fold_sharpe_min",
        "is_fold_positive_frac",
        "is_sharpe_cost_1p5x",
        "is_sharpe_cost_2x",
        "max_weight",
    ):
        assert key in metrics
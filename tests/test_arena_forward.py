from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.arena.engine import CandidateSpec
from src.arena.forward import (
    build_paper_snapshot,
    run_forward_validation,
    write_forward_artifacts,
    write_paper_snapshot,
)


def _synthetic_bars() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=1600, freq="1h", tz="UTC")
    rng = np.random.default_rng(9)
    configs = {
        "BTC/USDT": (40_000.0, 0.0002),
        "ETH/USDT": (2_000.0, -0.00005),
        "SOL/USDT": (90.0, 0.00035),
    }
    out: dict[str, pd.DataFrame] = {}
    for symbol, (base, drift) in configs.items():
        ret = rng.normal(drift, 0.006, len(index))
        close = base * np.exp(ret.cumsum())
        high = close * (1 + rng.uniform(0.0, 0.004, len(index)))
        low = close * (1 - rng.uniform(0.0, 0.004, len(index)))
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


def _candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
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
        ),
        CandidateSpec(
            name="relative_test",
            family="relative",
            params={
                "lookback": 120,
                "smooth": 24,
                "enter_spread": 0.65,
                "exit_spread": 0.25,
                "switch_buffer": 0.10,
                "target_vol": 0.005,
            },
        ),
    ]


def test_run_forward_validation_produces_folded_result():
    bars = _synthetic_bars()
    result = run_forward_validation(
        bars,
        specs=_candidate_specs(),
        n_trials=2,
        train_bars=24 * 20,
        test_bars=24 * 10,
        step_bars=24 * 10,
        max_weight_cap=0.7,
    )
    assert len(result.folds) >= 3
    assert result.summary["n_folds"] >= 3
    assert set(result.weighted_positions) == set(bars)
    assert len(result.strategy_labels) == len(result.portfolio_returns)


def test_write_forward_artifacts_outputs_expected_files(tmp_path):
    bars = _synthetic_bars()
    result = run_forward_validation(
        bars,
        specs=_candidate_specs(),
        n_trials=2,
        train_bars=24 * 20,
        test_bars=24 * 10,
        step_bars=24 * 10,
    )
    out = write_forward_artifacts(tmp_path / "forward", result=result, bars_by_symbol=bars)
    assert (out / "report.json").exists()
    assert (out / "portfolio_returns.parquet").exists()
    assert (out / "portfolio_equity.parquet").exists()
    assert (out / "latest_targets.json").exists()
    assert (out / "shadow_ledger.json").exists()

    report = json.loads((out / "report.json").read_text())
    assert "summary" in report
    assert "folds" in report


def test_build_and_write_paper_snapshot(tmp_path):
    bars = _synthetic_bars()
    snapshot = build_paper_snapshot(
        bars,
        specs=_candidate_specs(),
        n_trials=2,
        max_weight_cap=0.7,
    )
    assert snapshot.selected_name
    assert sum(abs(v) for v in snapshot.weighted_targets.values()) <= 1.5 + 1e-9

    out = write_paper_snapshot(tmp_path / "paper", snapshot=snapshot, bars_by_symbol=bars)
    assert (out / "paper_snapshot.json").exists()
    assert (out / "paper_ledger.json").exists()
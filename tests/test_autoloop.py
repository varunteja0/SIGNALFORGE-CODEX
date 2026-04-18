"""Tests for :mod:`src.research.autoloop`."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research import (
    AutoLoopResult,
    Candidate,
    Gates,
    Hypothesis,
    compile_signal,
    deflated_sharpe,
    generate_hypotheses,
    run_auto_loop,
    synthesize_features,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _synthetic(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    ret = rng.normal(0, 0.005, n)
    close = 50_000.0 * np.exp(ret.cumsum())
    high = close * (1 + rng.uniform(0, 0.003, n))
    low = close * (1 - rng.uniform(0, 0.003, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(100, 1_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# --------------------------------------------------------------------------
# Hypothesis + compile_signal
# --------------------------------------------------------------------------
def test_hypothesis_describe_is_human_readable():
    h = Hypothesis(name="mom_long_1", feature="zmom_20", op=">", threshold=1.0, side=+1)
    desc = h.describe()
    assert "zmom_20" in desc and ">" in desc and "side=+1" in desc


def test_compile_signal_returns_zero_when_feature_missing():
    h = Hypothesis("x", "absent_feature", ">", 0.0, +1)
    sig = compile_signal(h)
    df = _synthetic(n=100)
    out = sig(df)
    assert (out == 0).all()
    assert len(out) == len(df)


def test_compile_signal_respects_op_and_side():
    df = _synthetic(n=300)
    feats = synthesize_features(df)
    h = Hypothesis("z_high", "zmom_20", ">", 0.0, +1)
    sig_long = compile_signal(h)(feats)
    # No negative positions when side=+1.
    assert (sig_long >= 0).all()
    assert sig_long.max() == 1

    h2 = Hypothesis("z_low_short", "zmom_20", "<", 0.0, -1)
    sig_short = compile_signal(h2)(feats)
    assert (sig_short <= 0).all()
    assert sig_short.min() == -1


def test_compile_signal_rejects_unknown_op():
    h = Hypothesis("bad", "zmom_20", "!=", 0.0, +1)
    df = synthesize_features(_synthetic(n=200))
    with pytest.raises(ValueError):
        compile_signal(h)(df)


# --------------------------------------------------------------------------
# Feature synthesis
# --------------------------------------------------------------------------
def test_synthesize_features_adds_expected_columns():
    df = _synthetic(n=500)
    feats = synthesize_features(df)
    expected = {"ret_1", "ret_5", "ret_20", "zmom_20", "zmom_50",
                "vol_ratio", "range_pct", "range_ratio"}
    assert expected.issubset(feats.columns)
    # Original columns preserved.
    for c in ("open", "high", "low", "close", "volume"):
        assert c in feats.columns


def test_synthesize_features_leaves_original_unchanged():
    df = _synthetic(n=200)
    before = df.copy()
    _ = synthesize_features(df)
    pd.testing.assert_frame_equal(before, df)


# --------------------------------------------------------------------------
# generate_hypotheses
# --------------------------------------------------------------------------
def test_generate_hypotheses_produces_expected_grid():
    hs = generate_hypotheses()
    # 3 mom_long + 3 mom_short + 3 mr_long + 3 mr_short + 3 volexp + 3 bo_long + 3 bo_short = 21
    assert len(hs) == 21
    # IDs are alnum/underscore/hyphen only (registry requirement).
    for h in hs:
        token = h.name.replace("_", "").replace("-", "")
        assert token.isalnum(), f"bad id: {h.name}"


def test_generate_hypotheses_subset_families():
    hs = generate_hypotheses(families=["momentum"])
    assert all(h.name.startswith("mom_") for h in hs)
    assert len(hs) == 6


# --------------------------------------------------------------------------
# Deflated Sharpe
# --------------------------------------------------------------------------
def test_deflated_sharpe_penalises_many_trials():
    one_trial = deflated_sharpe(observed_sr=1.5, n_trials=1, n_periods=1000)
    many_trials = deflated_sharpe(observed_sr=1.5, n_trials=100, n_periods=1000)
    # Same observed SR but more trials → lower deflated z-score.
    assert many_trials < one_trial


def test_deflated_sharpe_zero_sr_is_not_significant():
    z = deflated_sharpe(observed_sr=0.0, n_trials=20, n_periods=500)
    # Under many-trial adjustment, zero observed SR clearly below threshold.
    assert z < 1.0


def test_deflated_sharpe_handles_degenerate_inputs():
    assert math.isnan(deflated_sharpe(observed_sr=1.0, n_trials=0, n_periods=10))
    assert math.isnan(deflated_sharpe(observed_sr=1.0, n_trials=5, n_periods=1))


# --------------------------------------------------------------------------
# run_auto_loop end-to-end
# --------------------------------------------------------------------------
def test_run_auto_loop_returns_expected_shape():
    df = _synthetic(n=3000, seed=42)
    # Cheap gates so something passes on synthetic GBM.
    gates = Gates(
        min_oos_sharpe=-10.0,
        min_frac_positive=0.0,
        max_drawdown=1.0,
        min_trades=0,
        min_deflated_sharpe_z=-10.0,
    )
    hs = generate_hypotheses(families=["momentum"])  # smaller grid → faster
    result = run_auto_loop(df, hypotheses=hs, n_folds=3, gates=gates)
    assert isinstance(result, AutoLoopResult)
    assert result.n_candidates == len(hs)
    assert all(isinstance(c, Candidate) for c in result.candidates)
    assert all(c.verdict in {"ACCEPT", "REJECT"} for c in result.candidates)


def test_run_auto_loop_strict_gates_reject_everything_on_gbm():
    """Pure GBM has no edge → strict gates should reject every candidate."""
    df = _synthetic(n=3000, seed=7)
    hs = generate_hypotheses(families=["momentum"])
    gates = Gates(
        min_oos_sharpe=3.0,   # absurd
        min_frac_positive=0.95,
        max_drawdown=0.01,
        min_trades=10_000,
        min_deflated_sharpe_z=5.0,
    )
    result = run_auto_loop(df, hypotheses=hs, n_folds=3, gates=gates)
    assert result.n_accepted == 0
    assert all(c.verdict == "REJECT" for c in result.candidates)
    # Every rejection has a non-empty reason.
    assert all(c.reason and c.reason != "ok" for c in result.candidates)


def test_run_auto_loop_result_to_dict_is_json_serialisable():
    import json
    df = _synthetic(n=3000, seed=1)
    result = run_auto_loop(df, hypotheses=generate_hypotheses(families=["momentum"]),
                           n_folds=3, gates=Gates(min_trades=0))
    payload = json.dumps(result.to_dict(), default=str)
    assert "candidates" in payload
    assert "verdict" in payload


def test_run_auto_loop_registers_accepted_candidates(tmp_path: Path):
    df = _synthetic(n=3000, seed=3)
    hs = [Hypothesis(name="always_long", feature="ret_1", op=">=", threshold=-1e9, side=+1)]
    gates = Gates(
        min_oos_sharpe=-10.0,
        min_frac_positive=0.0,
        max_drawdown=1.0,
        min_trades=0,
        min_deflated_sharpe_z=-10.0,
    )
    reg_path = tmp_path / "registry.ndjson"
    result = run_auto_loop(df, hypotheses=hs, n_folds=3, gates=gates, registry_path=reg_path)
    assert result.n_accepted == 1
    # File exists and has a single NDJSON line.
    assert reg_path.exists()
    lines = [ln for ln in reg_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    import json
    row = json.loads(lines[0])
    assert row["strategy_id"] == "always_long"


def test_run_auto_loop_handles_very_short_frame_gracefully():
    """When n_folds × test_bars exceeds available bars, we get no folds."""
    df = _synthetic(n=1050, seed=0)  # just above min_train_bars floor
    hs = generate_hypotheses(families=["momentum"])
    # Even if folds are produced, loop should not crash.
    result = run_auto_loop(df, hypotheses=hs, n_folds=3, gates=Gates(min_trades=0))
    assert result.n_candidates == len(hs)

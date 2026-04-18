"""Tests for the dashboard data layer (pure, no Streamlit)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ops.dashboard_data import (
    DEFAULT_ASSETS,
    compute_signal_proximity,
    load_divergence,
    load_journal,
    load_market_snapshot,
    load_state,
    portfolio_summary,
    proximity_matrix,
)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def test_load_state_missing_returns_empty_dict(tmp_path: Path):
    assert load_state(tmp_path) == {}


def test_load_state_reads_json(tmp_path: Path):
    (tmp_path / "live_state.json").write_text(json.dumps({"capital": 12345}))
    assert load_state(tmp_path)["capital"] == 12345


def test_load_state_corrupted_returns_empty(tmp_path: Path):
    (tmp_path / "live_state.json").write_text("{{not json")
    assert load_state(tmp_path) == {}


def test_load_journal_list_form(tmp_path: Path):
    payload = [{"id": 1}, {"id": 2}]
    (tmp_path / "trade_journal.json").write_text(json.dumps(payload))
    assert load_journal(tmp_path) == payload


def test_load_journal_non_list_coerces_to_empty(tmp_path: Path):
    (tmp_path / "trade_journal.json").write_text(json.dumps({"oops": 1}))
    assert load_journal(tmp_path) == []


def test_load_divergence_accepts_list_and_dict(tmp_path: Path):
    (tmp_path / "divergence_log.json").write_text(json.dumps([{"x": 1}]))
    assert load_divergence(tmp_path) == [{"x": 1}]

    (tmp_path / "divergence_log.json").write_text(
        json.dumps({"comparisons": [{"x": 2}]})
    )
    assert load_divergence(tmp_path) == [{"x": 2}]


def test_load_market_snapshot_strips_timestamp(tmp_path: Path):
    snap = {"_timestamp": 12345, "BTC/USDT": {"price": 60_000}}
    (tmp_path / "market_snapshot.json").write_text(json.dumps(snap))
    out = load_market_snapshot(tmp_path)
    assert "_timestamp" not in out
    assert out["BTC/USDT"]["price"] == 60_000


# --------------------------------------------------------------------------
# Portfolio summary
# --------------------------------------------------------------------------
def test_portfolio_summary_computes_return():
    state = {
        "capital": 12_000.0,
        "initial_capital": 10_000.0,
        "open_positions": [{}, {}],
        "iteration": 4,
        "paper_mode": True,
    }
    journal = [{}, {}, {}]
    s = portfolio_summary(state, journal)
    assert s["capital"] == 12_000.0
    assert s["return_pct"] == pytest.approx(0.20)
    assert s["n_open"] == 2
    assert s["n_closed"] == 3
    assert s["iteration"] == 4
    assert s["paper_mode"] is True


def test_portfolio_summary_handles_empty():
    s = portfolio_summary({}, [])
    assert s["return_pct"] == 0.0
    assert s["n_open"] == 0
    assert s["paper_mode"] is True


def test_portfolio_summary_zero_initial_is_safe():
    s = portfolio_summary({"capital": 5_000, "initial_capital": 0}, [])
    assert s["return_pct"] == 0.0


# --------------------------------------------------------------------------
# Signal proximity
# --------------------------------------------------------------------------
def test_proximity_values_clipped_to_unit_interval():
    snap = {"funding_zscore": 99, "bb_pctile": 0, "regime": "high_volatility",
            "vol_ratio": 99, "atr_exp": 99, "price": 100, "ch_high": 110,
            "ch_low": 90}
    for sym in DEFAULT_ASSETS:
        p = compute_signal_proximity(sym, snap)
        for k, v in p.items():
            assert 0.0 <= v <= 1.0, f"{sym} {k}={v}"


def test_proximity_empty_or_error_snap_returns_zero():
    assert all(v == 0.0 for v in compute_signal_proximity("BTC/USDT", {}).values())
    assert all(
        v == 0.0
        for v in compute_signal_proximity("BTC/USDT", {"error": "boom"}).values()
    )


def test_proximity_btc_never_triggers_extreme_spike():
    snap = {"funding_zscore": 10, "regime": "high_volatility"}
    assert compute_signal_proximity("BTC/USDT", snap)["extreme_spike"] == 0.0


def test_proximity_momentum_breakout_only_on_eth():
    snap = {"price": 100, "ch_high": 110, "ch_low": 90,
            "atr_exp": 2.0, "vol_ratio": 2.0}
    for sym in ["BTC/USDT", "SOL/USDT", "XRP/USDT"]:
        assert compute_signal_proximity(sym, snap)["momentum_breakout"] == 0.0
    assert compute_signal_proximity("ETH/USDT", snap)["momentum_breakout"] > 0.0


def test_proximity_funding_mr_scales_linearly_up_to_threshold():
    p_low = compute_signal_proximity("SOL/USDT", {"funding_zscore": 1.5})
    p_at = compute_signal_proximity("SOL/USDT", {"funding_zscore": 3.0})
    p_high = compute_signal_proximity("SOL/USDT", {"funding_zscore": 6.0})
    assert p_low["funding_mr_v7"] < p_at["funding_mr_v7"]
    assert p_at["funding_mr_v7"] == 1.0
    assert p_high["funding_mr_v7"] == 1.0  # clipped


# --------------------------------------------------------------------------
# Proximity matrix
# --------------------------------------------------------------------------
def test_proximity_matrix_shape_matches_inputs():
    labels, mat, strats = proximity_matrix(
        DEFAULT_ASSETS,
        snapshots={
            "BTC/USDT": {"funding_zscore": 3.0},
            "ETH/USDT": {"funding_zscore": 2.0},
        },
    )
    assert labels == ["BTC", "ETH", "SOL", "XRP"]
    assert len(mat) == 4
    assert all(len(row) == len(strats) for row in mat)


def test_proximity_matrix_zeros_when_no_snapshots():
    labels, mat, strats = proximity_matrix(DEFAULT_ASSETS, snapshots={})
    assert len(mat) == len(DEFAULT_ASSETS)
    for row in mat:
        assert all(v == 0.0 for v in row)

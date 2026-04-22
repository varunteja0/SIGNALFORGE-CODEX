"""Tests for the dashboard data layer (pure, no Streamlit)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ops.dashboard_data import (
    DEFAULT_ASSETS,
    build_live_readiness_snapshot,
    compute_signal_proximity,
    load_capital_firewall,
    load_deployment_gate,
    load_divergence,
    load_drift_intelligence,
    load_execution_drift,
    load_health,
    load_journal,
    load_market_snapshot,
    load_production_certification,
    load_shadow_live_comparator,
    load_stress_field,
    load_state,
    load_streaming_stress_kernel,
    load_survivability,
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


def test_load_health_certification_and_drift_are_safe(tmp_path: Path):
    (tmp_path / "health.json").write_text(json.dumps({"overall_status": "ok"}))
    (tmp_path / "production_certification_status.json").write_text(json.dumps({"ready_for_live": False}))
    (tmp_path / "deployment_gate_status.json").write_text(json.dumps({"allowed_mode": "shadow_live"}))
    (tmp_path / "drift_intelligence_status.json").write_text(json.dumps({"risk_score": 42.5}))
    (tmp_path / "execution_drift_status.json").write_text(json.dumps({"execution_fidelity_score": 81.0}))
    (tmp_path / "shadow_live_comparator_status.json").write_text(json.dumps({"ready_for_capital": True, "comparator_score": 84.0}))
    (tmp_path / "capital_firewall_status.json").write_text(json.dumps({"decision": "allow_reduced_size"}))
    (tmp_path / "survivability_status.json").write_text(json.dumps({"survivability_score": 73.0}))
    (tmp_path / "streaming_stress_kernel_status.json").write_text(json.dumps({"continuous_pressure_score": 48.0}))
    (tmp_path / "stress_field_state.json").write_text(json.dumps({"phase": "paper_field", "hysteresis_score": 0.22}))

    assert load_health(tmp_path)["overall_status"] == "ok"
    assert load_production_certification(tmp_path)["ready_for_live"] is False
    assert load_deployment_gate(tmp_path)["allowed_mode"] == "shadow_live"
    assert load_drift_intelligence(tmp_path)["risk_score"] == 42.5
    assert load_execution_drift(tmp_path)["execution_fidelity_score"] == 81.0
    assert load_shadow_live_comparator(tmp_path)["ready_for_capital"] is True
    assert load_capital_firewall(tmp_path)["decision"] == "allow_reduced_size"
    assert load_survivability(tmp_path)["survivability_score"] == 73.0
    assert load_streaming_stress_kernel(tmp_path)["continuous_pressure_score"] == 48.0
    assert load_stress_field(tmp_path)["phase"] == "paper_field"


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


def test_proximity_fund_vol_squeeze_is_sol_only():
    snap = {"funding_zscore": 3.0, "bb_pctile": 5}
    assert compute_signal_proximity("SOL/USDT", snap)["fund_vol_squeeze"] > 0.0
    for sym in ["BTC/USDT", "ETH/USDT", "XRP/USDT"]:
        assert compute_signal_proximity(sym, snap)["fund_vol_squeeze"] == 0.0


def test_proximity_fund_vol_squeeze_uses_updated_thresholds():
    near = compute_signal_proximity("SOL/USDT", {"funding_zscore": 1.5, "bb_pctile": 15})
    far = compute_signal_proximity("SOL/USDT", {"funding_zscore": 1.0, "bb_pctile": 30})
    assert near["fund_vol_squeeze"] == 1.0
    assert 0.0 <= far["fund_vol_squeeze"] < near["fund_vol_squeeze"]


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


# --------------------------------------------------------------------------
# Live readiness
# --------------------------------------------------------------------------
def test_live_readiness_snapshot_defaults_to_no_go_when_artifacts_are_missing(tmp_path: Path):
    (tmp_path / "live_state.json").write_text(
        json.dumps({"capital": 10_000, "initial_capital": 10_000, "paper_mode": True})
    )
    (tmp_path / "health.json").write_text(
        json.dumps({"overall_status": "warning", "should_halt": False})
    )

    snapshot = build_live_readiness_snapshot(tmp_path)

    assert snapshot["overall_verdict"] == "no_go"
    assert snapshot["gates"]["probation_live"]["trades_remaining"] == 300
    assert snapshot["gates"]["probation_live"]["days_remaining"] == pytest.approx(3.0)
    assert snapshot["gates"]["full_live"]["trades_remaining"] == 500
    assert snapshot["gates"]["full_live"]["days_remaining"] == pytest.approx(7.0)
    assert snapshot["rollout_plan"]["current_stage"] == "paper_shadow"
    assert snapshot["rollout_plan"]["starting_size_usd"] == pytest.approx(0.0)
    assert "deployment_gate" in snapshot["missing_artifacts"]
    assert "300 fully quoted trades remaining" in snapshot["one_line_summaries"]["probation_live"]


def test_live_readiness_snapshot_builds_probation_rollout_from_green_artifacts(tmp_path: Path):
    (tmp_path / "live_state.json").write_text(
        json.dumps(
            {
                "capital": 10_000,
                "initial_capital": 10_000,
                "paper_mode": False,
                "operating_mode": "probation_live",
            }
        )
    )
    (tmp_path / "health.json").write_text(
        json.dumps({"overall_status": "ok", "should_halt": False})
    )
    (tmp_path / "paper_validation_status.json").write_text(
        json.dumps({"ready_for_live": True})
    )
    (tmp_path / "shadow_execution_status.json").write_text(
        json.dumps({"ready_for_live": True, "compared_trade_count": 6})
    )
    (tmp_path / "shadow_live_comparator_status.json").write_text(
        json.dumps(
            {
                "ready_for_capital": True,
                "entry_comparison_count": 320,
                "exit_comparison_count": 320,
                "validation_runtime_days": 3.5,
                "entry_quote_coverage_rate": 0.99,
                "exit_quote_coverage_rate": 0.98,
            }
        )
    )
    (tmp_path / "production_certification_status.json").write_text(
        json.dumps(
            {
                "current_green": True,
                "ready_for_live": False,
                "shadow_live_entry_comparison_count": 320,
                "shadow_live_exit_comparison_count": 320,
                "shadow_live_validation_runtime_days": 3.5,
            }
        )
    )
    (tmp_path / "deployment_gate_status.json").write_text(
        json.dumps(
            {
                "allowed_mode": "probation_live",
                "allow_shadow_live": True,
                "allow_probation_live": True,
                "allow_full_live": False,
                "paper_validation_ready": True,
                "shadow_ready": True,
                "shadow_compared_trade_count": 6,
                "shadow_live_entry_comparison_count": 320,
                "shadow_live_exit_comparison_count": 320,
                "shadow_live_validation_runtime_days": 3.5,
                "recommended_max_total_exposure_pct": 0.005,
                "recommended_max_per_trade_pct": 0.001,
                "reasons": [],
            }
        )
    )
    (tmp_path / "capital_firewall_status.json").write_text(
        json.dumps(
            {
                "decision": "allow_full_size",
                "max_total_exposure_pct": 0.005,
                "max_per_trade_pct": 0.001,
            }
        )
    )
    (tmp_path / "streaming_stress_kernel_status.json").write_text(
        json.dumps(
            {
                "probation_live_policy": {
                    "stage": "plm_0.50",
                    "allow_probation_live": True,
                    "allow_full_live": False,
                    "max_capital_fraction": 0.005,
                    "max_total_exposure_pct": 0.005,
                    "max_per_trade_pct": 0.001,
                }
            }
        )
    )
    (tmp_path / "survivability_status.json").write_text(
        json.dumps(
            {
                "exposure_ladder": {
                    "stage": "0.5%",
                    "max_capital_fraction": 0.005,
                    "max_total_exposure_pct": 0.005,
                    "max_per_trade_pct": 0.001,
                }
            }
        )
    )
    (tmp_path / "stress_field_state.json").write_text(
        json.dumps({"phase": "paper_field", "allow_entries": True, "should_halt": False})
    )

    snapshot = build_live_readiness_snapshot(tmp_path)

    assert snapshot["overall_verdict"] == "go_probation"
    assert snapshot["gates"]["probation_live"]["allowed"] is True
    assert snapshot["gates"]["probation_live"]["trades_remaining"] == 0
    assert snapshot["gates"]["probation_live"]["days_remaining"] == pytest.approx(0.0)
    assert snapshot["gates"]["full_live"]["trades_remaining"] == 180
    assert snapshot["gates"]["full_live"]["days_remaining"] == pytest.approx(3.5)
    assert snapshot["rollout_plan"]["current_stage"] == "plm_0.50"
    assert snapshot["rollout_plan"]["starting_size_usd"] == pytest.approx(10.0)
    assert snapshot["rollout_plan"]["max_stage_exposure_usd"] == pytest.approx(50.0)
    assert snapshot["rollout_plan"]["pause_loss_usd"] == pytest.approx(20.0)
    assert snapshot["one_line_summaries"]["probation_live"] == "GO Probation Live: gate is open."

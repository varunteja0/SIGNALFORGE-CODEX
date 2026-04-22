from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.ops.failure_drills import run_failure_drills
from src.ops.production_bridge import (
    ProductionCertificationThresholds,
    build_broker_reconciliation_report,
    build_production_certification_report,
    build_shadow_execution_report,
    build_trade_journal_parity_report,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_shadow_live_ready(path: Path) -> None:
    _write_json(
        path / "shadow_live_comparator_status.json",
        {
            "ready_for_capital": True,
            "trade_count": 320,
            "entry_comparison_count": 320,
            "exit_comparison_count": 320,
            "validation_runtime_days": 3.5,
            "avg_abs_entry_reference_gap_bps": 4.0,
            "avg_abs_exit_reference_gap_bps": 5.0,
            "avg_abs_entry_fill_gap_bps": 2.0,
            "avg_abs_exit_fill_gap_bps": 2.5,
            "comparator_score": 86.0,
            "comparator_level": "stable",
        },
    )


def test_broker_reconciliation_flags_critical_issues() -> None:
    report = build_broker_reconciliation_report(
        system_positions=[{"symbol": "BTC/USDT", "direction": 1, "qty": 1.0, "stop_order_id": "sl-1"}],
        broker_positions=[{"symbol": "ETH/USDT", "signed_size": 2.0, "size": 2.0, "side": "long"}],
        open_orders=[],
    )

    assert report.overall_status == "critical"
    assert report.missing_broker_positions == 1
    assert report.unknown_broker_positions == 1
    assert report.critical_issue_count >= 2
    assert {issue.code for issue in report.issues} >= {"missing_broker_position", "unknown_broker_position"}


def test_trade_journal_parity_uses_current_journal_shape(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "trade_journal.json",
        [
            {
                "symbol": "BTC/USDT",
                "direction": 1,
                "entry_price": 101.0,
                "exit_price": 107.0,
                "entry_expected_price": 100.0,
                "exit_expected_price": 110.0,
                "filled_size_usd": 1000.0,
            }
        ],
    )

    report = build_trade_journal_parity_report(tmp_path)

    assert report.trade_count == 1
    assert report.verdict == "FAIL"
    assert abs(report.unexplained_pnl_bps) > 6.0


def test_shadow_execution_report_requires_compared_trades(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "trade_journal.json",
        [
            {
                "symbol": "ETH/USDT",
                "direction": 1,
                "entry_price": 100.0,
                "exit_price": 105.0,
            }
        ],
    )

    report = build_shadow_execution_report(tmp_path)

    assert report.trade_count == 1
    assert report.compared_trade_count == 0
    assert report.ready_for_live is False
    assert report.reasons


def test_production_certification_requires_green_burn_in(tmp_path: Path) -> None:
    source_ts = "2026-04-21T00:00:00+00:00"
    _write_json(tmp_path / "paper_validation_status.json", {"ready_for_live": True})
    _write_json(tmp_path / "health.json", {"overall_status": "ok", "should_halt": False})
    _write_json(tmp_path / "trade_parity_status.json", {"verdict": "PASS", "unexplained_pnl_bps": 1.2})
    _write_json(tmp_path / "shadow_execution_status.json", {"ready_for_live": True})
    _write_shadow_live_ready(tmp_path)
    _write_json(tmp_path / "broker_reconciliation_status.json", {"overall_status": "ok", "critical_issue_count": 0})
    _write_json(tmp_path / "failure_drill_report.json", {"all_passed": True})
    _write_json(
        tmp_path / "drift_intelligence_status.json",
        {
            "risk_score": 12.0,
            "risk_level": "low",
            "predicted_certification_failure": False,
            "deployment_recommendation": {"mode": "paper_shadow"},
        },
    )
    _write_json(
        tmp_path / "survivability_status.json",
        {
            "survivability_score": 75.0,
            "survivability_level": "adequate",
            "regime_novelty_score": 18.0,
            "execution_stress_score": 28.0,
            "halt_latency_p95_ms": 240.0,
            "predicted_survivability_failure": False,
            "exposure_ladder": {"stage": "0.5%"},
        },
    )
    _write_json(
        tmp_path / "streaming_stress_kernel_status.json",
        {
            "continuous_pressure_score": 18.0,
            "pressure_level": "low",
            "trajectory_novelty_score": 16.0,
            "execution_friction_score": 18.0,
            "kill_switch_efficiency": 1.0,
            "predicted_pressure_failure": False,
            "source_generated_at": source_ts,
            "probation_live_policy": {"stage": "micro_live_ready"},
        },
    )
    _write_json(
        tmp_path / "stress_field_state.json",
        {
            "phase": "paper_field",
            "hysteresis_score": 0.22,
            "latency_memory": 0.14,
            "allow_entries": True,
            "should_halt": False,
            "source_generated_at": source_ts,
            "adversarial_input": {"intensity": 0.18},
        },
    )

    report = build_production_certification_report(
        tmp_path,
        ProductionCertificationThresholds(min_consecutive_green_days=30.0),
    )

    assert report.current_green is True
    assert report.ready_for_live is False
    assert any("Burn-in evidence" in reason for reason in report.reasons)

    history_path = tmp_path / "production_certification_history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "timestamp": (datetime.now(timezone.utc) - timedelta(days=31)).isoformat(),
                "current_green": True,
            }
        )
        + "\n"
    )

    ready_report = build_production_certification_report(
        tmp_path,
        ProductionCertificationThresholds(min_consecutive_green_days=30.0),
    )

    assert ready_report.current_green is True
    assert ready_report.ready_for_live is True
    assert ready_report.consecutive_green_days >= 30.0
    assert ready_report.shadow_live_ready is True
    assert ready_report.shadow_live_trade_count == 320
    assert ready_report.shadow_live_validation_runtime_days == 3.5
    assert ready_report.drift_risk_level == "low"
    assert ready_report.survivability_level == "adequate"
    assert ready_report.recommended_exposure_ladder_step == "0.5%"
    assert ready_report.recommended_probation_mode == "micro_live_ready"
    assert ready_report.continuous_pressure_score == 18.0
    assert ready_report.stress_field_phase == "paper_field"
    assert ready_report.stress_field_hysteresis == 0.22
    assert ready_report.stress_observation_consistent is True


def test_production_certification_respects_stateful_stress_gate(tmp_path: Path) -> None:
    source_ts = "2026-04-21T00:00:00+00:00"
    _write_json(tmp_path / "paper_validation_status.json", {"ready_for_live": True})
    _write_json(tmp_path / "health.json", {"overall_status": "ok", "should_halt": False})
    _write_json(tmp_path / "trade_parity_status.json", {"verdict": "PASS", "unexplained_pnl_bps": 1.0})
    _write_json(tmp_path / "shadow_execution_status.json", {"ready_for_live": True})
    _write_shadow_live_ready(tmp_path)
    _write_json(tmp_path / "broker_reconciliation_status.json", {"overall_status": "ok", "critical_issue_count": 0})
    _write_json(tmp_path / "failure_drill_report.json", {"all_passed": True})
    _write_json(
        tmp_path / "drift_intelligence_status.json",
        {
            "risk_score": 10.0,
            "risk_level": "low",
            "predicted_certification_failure": False,
            "deployment_recommendation": {"mode": "paper_shadow"},
        },
    )
    _write_json(
        tmp_path / "survivability_status.json",
        {
            "survivability_score": 80.0,
            "survivability_level": "adequate",
            "regime_novelty_score": 18.0,
            "execution_stress_score": 25.0,
            "halt_latency_p95_ms": 200.0,
            "predicted_survivability_failure": False,
            "exposure_ladder": {"stage": "0.5%"},
        },
    )
    _write_json(
        tmp_path / "streaming_stress_kernel_status.json",
        {
            "continuous_pressure_score": 18.0,
            "pressure_level": "low",
            "trajectory_novelty_score": 15.0,
            "execution_friction_score": 17.0,
            "kill_switch_efficiency": 1.0,
            "predicted_pressure_failure": False,
            "source_generated_at": source_ts,
            "probation_live_policy": {"stage": "micro_live_ready"},
        },
    )
    _write_json(
        tmp_path / "stress_field_state.json",
        {
            "phase": "probation_field_plm_0.10",
            "hysteresis_score": 0.74,
            "latency_memory": 0.51,
            "allow_entries": False,
            "should_halt": True,
            "source_generated_at": source_ts,
            "adversarial_input": {"intensity": 0.81},
        },
    )

    report = build_production_certification_report(tmp_path)

    assert report.current_green is False
    assert report.ready_for_live is False
    assert report.recommended_probation_mode == "blocked"
    assert any("Stateful stress field" in reason for reason in report.reasons)


def test_production_certification_blocks_on_mismatched_stress_observations(tmp_path: Path) -> None:
    _write_json(tmp_path / "paper_validation_status.json", {"ready_for_live": True})
    _write_json(tmp_path / "health.json", {"overall_status": "ok", "should_halt": False})
    _write_json(tmp_path / "trade_parity_status.json", {"verdict": "PASS", "unexplained_pnl_bps": 0.5})
    _write_json(tmp_path / "shadow_execution_status.json", {"ready_for_live": True})
    _write_shadow_live_ready(tmp_path)
    _write_json(tmp_path / "broker_reconciliation_status.json", {"overall_status": "ok", "critical_issue_count": 0})
    _write_json(tmp_path / "failure_drill_report.json", {"all_passed": True})
    _write_json(
        tmp_path / "drift_intelligence_status.json",
        {
            "risk_score": 10.0,
            "risk_level": "low",
            "predicted_certification_failure": False,
            "deployment_recommendation": {"mode": "paper_shadow"},
        },
    )
    _write_json(
        tmp_path / "survivability_status.json",
        {
            "survivability_score": 80.0,
            "survivability_level": "adequate",
            "regime_novelty_score": 18.0,
            "execution_stress_score": 25.0,
            "halt_latency_p95_ms": 200.0,
            "predicted_survivability_failure": False,
            "exposure_ladder": {"stage": "0.5%"},
        },
    )
    _write_json(
        tmp_path / "streaming_stress_kernel_status.json",
        {
            "continuous_pressure_score": 18.0,
            "pressure_level": "low",
            "trajectory_novelty_score": 15.0,
            "execution_friction_score": 17.0,
            "kill_switch_efficiency": 1.0,
            "predicted_pressure_failure": False,
            "source_generated_at": "2026-04-21T00:00:00+00:00",
            "probation_live_policy": {"stage": "micro_live_ready"},
        },
    )
    _write_json(
        tmp_path / "stress_field_state.json",
        {
            "phase": "paper_field",
            "hysteresis_score": 0.15,
            "latency_memory": 0.08,
            "allow_entries": True,
            "should_halt": False,
            "source_generated_at": "2026-04-21T01:00:00+00:00",
            "adversarial_input": {"intensity": 0.12},
        },
    )

    report = build_production_certification_report(tmp_path)

    assert report.current_green is False
    assert report.ready_for_live is False
    assert report.recommended_probation_mode == "blocked"
    assert report.stress_observation_consistent is False
    assert any("same source snapshot" in reason for reason in report.reasons)


def test_failure_drills_pass() -> None:
    report = run_failure_drills()

    assert report.scenario_count >= 5
    assert report.all_passed is True


def test_production_certification_blocks_when_shadow_live_comparator_missing(tmp_path: Path) -> None:
    source_ts = "2026-04-21T00:00:00+00:00"
    _write_json(tmp_path / "paper_validation_status.json", {"ready_for_live": True})
    _write_json(tmp_path / "health.json", {"overall_status": "ok", "should_halt": False})
    _write_json(tmp_path / "trade_parity_status.json", {"verdict": "PASS", "unexplained_pnl_bps": 0.5})
    _write_json(tmp_path / "shadow_execution_status.json", {"ready_for_live": True})
    _write_json(tmp_path / "broker_reconciliation_status.json", {"overall_status": "ok", "critical_issue_count": 0})
    _write_json(tmp_path / "failure_drill_report.json", {"all_passed": True})
    _write_json(
        tmp_path / "drift_intelligence_status.json",
        {
            "risk_score": 10.0,
            "risk_level": "low",
            "predicted_certification_failure": False,
            "deployment_recommendation": {"mode": "paper_shadow"},
        },
    )
    _write_json(
        tmp_path / "survivability_status.json",
        {
            "survivability_score": 80.0,
            "survivability_level": "adequate",
            "regime_novelty_score": 18.0,
            "execution_stress_score": 25.0,
            "halt_latency_p95_ms": 200.0,
            "predicted_survivability_failure": False,
            "exposure_ladder": {"stage": "0.5%"},
        },
    )
    _write_json(
        tmp_path / "streaming_stress_kernel_status.json",
        {
            "continuous_pressure_score": 18.0,
            "pressure_level": "low",
            "trajectory_novelty_score": 15.0,
            "execution_friction_score": 17.0,
            "kill_switch_efficiency": 1.0,
            "predicted_pressure_failure": False,
            "source_generated_at": source_ts,
            "probation_live_policy": {"stage": "micro_live_ready"},
        },
    )
    _write_json(
        tmp_path / "stress_field_state.json",
        {
            "phase": "paper_field",
            "hysteresis_score": 0.22,
            "latency_memory": 0.14,
            "allow_entries": True,
            "should_halt": False,
            "source_generated_at": source_ts,
            "adversarial_input": {"intensity": 0.18},
        },
    )

    report = build_production_certification_report(tmp_path)

    assert report.current_green is False
    assert report.ready_for_live is False
    assert any("shadow live comparator" in reason.lower() for reason in report.reasons)
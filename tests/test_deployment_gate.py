from __future__ import annotations

import json
from pathlib import Path

from src.ops.deployment_gate import build_deployment_gate_report


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _seed_operational_artifacts(
    tmp_path: Path,
    *,
    health_status: str = "ok",
    health_should_halt: bool = False,
    paper_ready: bool = False,
    shadow_ready: bool = False,
    compared: int = 0,
    shadow_live_ready: bool = False,
    shadow_live_entry_compared: int = 0,
    shadow_live_exit_compared: int = 0,
    shadow_live_days: float = 0.0,
    certification_green: bool = False,
    certification_ready: bool = False,
    drift_score: float = 18.0,
    survivability_score: float = 82.0,
    hysteresis: float = 0.20,
    adversary: float = 0.15,
    stress_consistent: bool = True,
    stress_halt: bool = False,
    allow_entries: bool = True,
    allow_probation: bool = True,
    allow_full_live: bool = False,
) -> None:
    _write_json(tmp_path / "health.json", {"overall_status": health_status, "should_halt": health_should_halt})
    _write_json(tmp_path / "paper_validation_status.json", {"ready_for_live": paper_ready})
    _write_json(
        tmp_path / "shadow_execution_status.json",
        {
            "ready_for_live": shadow_ready,
            "compared_trade_count": compared,
            "avg_abs_entry_delta_bps": 3.0,
            "avg_abs_pnl_delta_pct": 0.05,
        },
    )
    _write_json(
        tmp_path / "shadow_live_comparator_status.json",
        {
            "ready_for_capital": shadow_live_ready,
            "entry_comparison_count": shadow_live_entry_compared,
            "exit_comparison_count": shadow_live_exit_compared,
            "validation_runtime_days": shadow_live_days,
        },
    )
    _write_json(
        tmp_path / "production_certification_status.json",
        {
            "current_green": certification_green,
            "ready_for_live": certification_ready,
            "stress_observation_consistent": stress_consistent,
        },
    )
    _write_json(
        tmp_path / "drift_intelligence_status.json",
        {
            "risk_score": drift_score,
            "deployment_recommendation": {
                "mode": "micro_live",
                "max_total_exposure_pct": 0.010,
                "max_per_trade_pct": 0.0025,
            },
        },
    )
    _write_json(
        tmp_path / "survivability_status.json",
        {
            "survivability_score": survivability_score,
            "exposure_ladder": {
                "stage": "0.5%",
                "max_total_exposure_pct": 0.005,
                "max_per_trade_pct": 0.0010,
            },
        },
    )
    _write_json(
        tmp_path / "streaming_stress_kernel_status.json",
        {
            "probation_live_policy": {
                "allow_probation_live": allow_probation,
                "allow_full_live": allow_full_live,
                "max_total_exposure_pct": 0.0075,
                "max_per_trade_pct": 0.0015,
            }
        },
    )
    _write_json(
        tmp_path / "stress_field_state.json",
        {
            "phase": "paper_field",
            "hysteresis_score": hysteresis,
            "should_halt": stress_halt,
            "allow_entries": allow_entries,
            "adversarial_input": {"intensity": adversary},
        },
    )


def test_deployment_gate_blocks_when_health_requests_halt(tmp_path: Path) -> None:
    _seed_operational_artifacts(tmp_path, health_should_halt=True)

    report = build_deployment_gate_report(tmp_path)

    assert report.allowed_mode == "blocked"
    assert report.allow_shadow_live is False
    assert any("halt" in reason.lower() for reason in report.reasons)


def test_deployment_gate_allows_shadow_stage_before_capital(tmp_path: Path) -> None:
    _seed_operational_artifacts(
        tmp_path,
        paper_ready=False,
        shadow_ready=False,
        compared=0,
        certification_green=False,
        allow_probation=False,
    )

    report = build_deployment_gate_report(tmp_path)

    assert report.allowed_mode == "shadow_live"
    assert report.allow_shadow_live is True
    assert report.allow_probation_live is False
    assert any("paper validation" in reason.lower() for reason in report.reasons)


def test_deployment_gate_allows_probation_when_artifacts_align(tmp_path: Path) -> None:
    _seed_operational_artifacts(
        tmp_path,
        paper_ready=True,
        shadow_ready=True,
        compared=4,
        shadow_live_ready=True,
        shadow_live_entry_compared=320,
        shadow_live_exit_compared=320,
        shadow_live_days=3.5,
        certification_green=True,
        certification_ready=False,
        allow_probation=True,
        allow_full_live=False,
    )

    report = build_deployment_gate_report(tmp_path)

    assert report.allowed_mode == "probation_live"
    assert report.allow_probation_live is True
    assert report.allow_full_live is False
    assert report.recommended_max_total_exposure_pct == 0.005
    assert report.recommended_max_per_trade_pct == 0.001


def test_deployment_gate_allows_full_live_only_after_mature_shadow_and_certification(tmp_path: Path) -> None:
    _seed_operational_artifacts(
        tmp_path,
        paper_ready=True,
        shadow_ready=True,
        compared=12,
        shadow_live_ready=True,
        shadow_live_entry_compared=520,
        shadow_live_exit_compared=520,
        shadow_live_days=8.0,
        certification_green=True,
        certification_ready=True,
        drift_score=20.0,
        survivability_score=84.0,
        hysteresis=0.18,
        adversary=0.12,
        allow_probation=True,
        allow_full_live=True,
    )

    report = build_deployment_gate_report(tmp_path)

    assert report.allowed_mode == "full_live"
    assert report.allow_full_live is True
    assert report.reasons == []


def test_deployment_gate_blocks_probation_when_shadow_live_burn_in_is_immature(tmp_path: Path) -> None:
    _seed_operational_artifacts(
        tmp_path,
        paper_ready=True,
        shadow_ready=True,
        compared=8,
        shadow_live_ready=False,
        shadow_live_entry_compared=120,
        shadow_live_exit_compared=118,
        shadow_live_days=1.2,
        certification_green=False,
        certification_ready=False,
        allow_probation=True,
        allow_full_live=False,
    )

    report = build_deployment_gate_report(tmp_path)

    assert report.allowed_mode == "shadow_live"
    assert report.allow_probation_live is False
    assert any("shadow live comparator" in reason.lower() for reason in report.reasons)
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DeploymentGateThresholds:
    min_shadow_compared_trades_for_probation: int = 3
    min_shadow_compared_trades_for_live: int = 10
    min_shadow_live_entry_comparisons_for_probation: int = 300
    min_shadow_live_exit_comparisons_for_probation: int = 300
    min_shadow_live_validation_days_for_probation: float = 3.0
    min_shadow_live_entry_comparisons_for_live: int = 500
    min_shadow_live_exit_comparisons_for_live: int = 500
    min_shadow_live_validation_days_for_live: float = 7.0
    max_drift_risk_score_for_probation: float = 45.0
    max_drift_risk_score_for_live: float = 35.0
    min_survivability_score_for_probation: float = 65.0
    min_survivability_score_for_live: float = 75.0
    max_stress_hysteresis_for_probation: float = 0.55
    max_stress_hysteresis_for_live: float = 0.35
    max_stress_adversarial_intensity_for_probation: float = 0.55
    max_stress_adversarial_intensity_for_live: float = 0.35


@dataclass
class DeploymentGateReport:
    generated_at: str
    allowed_mode: str
    allow_shadow_live: bool
    allow_probation_live: bool
    allow_full_live: bool
    health_status: str
    health_should_halt: bool
    paper_validation_ready: bool
    shadow_ready: bool
    shadow_compared_trade_count: int
    shadow_live_ready: bool
    shadow_live_entry_comparison_count: int
    shadow_live_exit_comparison_count: int
    shadow_live_validation_runtime_days: float
    certification_current_green: bool
    certification_ready_for_live: bool
    drift_risk_score: float
    survivability_score: float
    stress_observation_consistent: bool
    stress_field_phase: str
    stress_field_hysteresis: float
    stress_field_adversarial_intensity: float
    recommended_max_total_exposure_pct: float
    recommended_max_per_trade_pct: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_deployment_gate_report(
    base_dir: Path,
    thresholds: DeploymentGateThresholds | None = None,
) -> DeploymentGateReport:
    thresholds = thresholds or DeploymentGateThresholds()
    health = _load_json(base_dir / "health.json", {})
    paper_validation = _load_json(base_dir / "paper_validation_status.json", {})
    shadow = _load_json(base_dir / "shadow_execution_status.json", {})
    shadow_live = _load_json(base_dir / "shadow_live_comparator_status.json", {})
    certification = _load_json(base_dir / "production_certification_status.json", {})
    drift = _load_json(base_dir / "drift_intelligence_status.json", {})
    survivability = _load_json(base_dir / "survivability_status.json", {})
    stress_kernel = _load_json(base_dir / "streaming_stress_kernel_status.json", {})
    stress_field = _load_json(base_dir / "stress_field_state.json", {})

    health_status = str(health.get("overall_status", "unknown"))
    health_should_halt = bool(health.get("should_halt"))
    paper_validation_ready = bool(paper_validation.get("ready_for_live"))
    shadow_ready = bool(shadow.get("ready_for_live")) if shadow else False
    shadow_compared_trade_count = int(shadow.get("compared_trade_count", 0) or 0)
    shadow_live_ready = bool(shadow_live.get("ready_for_capital")) if shadow_live else False
    shadow_live_entry_comparison_count = int(shadow_live.get("entry_comparison_count", 0) or 0)
    shadow_live_exit_comparison_count = int(shadow_live.get("exit_comparison_count", 0) or 0)
    shadow_live_validation_runtime_days = _float(shadow_live.get("validation_runtime_days"))
    certification_current_green = bool(certification.get("current_green"))
    certification_ready_for_live = bool(certification.get("ready_for_live"))
    drift_risk_score = _float(drift.get("risk_score"))
    survivability_score = _float(survivability.get("survivability_score"))
    stress_observation_consistent = bool(certification.get("stress_observation_consistent"))
    stress_field_phase = str(stress_field.get("phase", "unknown"))
    stress_field_hysteresis = _float(stress_field.get("hysteresis_score"))
    stress_field_adversarial_intensity = _float(
        (stress_field.get("adversarial_input") or {}).get("intensity")
        if isinstance(stress_field.get("adversarial_input"), dict)
        else 0.0
    )
    stress_field_should_halt = bool(stress_field.get("should_halt"))
    stress_field_allow_entries = bool(stress_field.get("allow_entries", True))
    probation_policy = stress_kernel.get("probation_live_policy") if isinstance(stress_kernel.get("probation_live_policy"), dict) else {}
    allow_kernel_probation = bool(probation_policy.get("allow_probation_live"))
    allow_kernel_full_live = bool(probation_policy.get("allow_full_live"))

    cap_candidates: list[dict[str, float]] = []
    deployment = drift.get("deployment_recommendation") if isinstance(drift.get("deployment_recommendation"), dict) else {}
    if deployment.get("mode") in {"micro_live", "scale_up"}:
        cap_candidates.append(
            {
                "max_total_exposure_pct": _float(deployment.get("max_total_exposure_pct")),
                "max_per_trade_pct": _float(deployment.get("max_per_trade_pct")),
            }
        )
    ladder = survivability.get("exposure_ladder") if isinstance(survivability.get("exposure_ladder"), dict) else {}
    if ladder.get("stage") not in {None, "", "shadow", "blocked"}:
        cap_candidates.append(
            {
                "max_total_exposure_pct": _float(ladder.get("max_total_exposure_pct")),
                "max_per_trade_pct": _float(ladder.get("max_per_trade_pct")),
            }
        )
    if allow_kernel_probation or allow_kernel_full_live:
        cap_candidates.append(
            {
                "max_total_exposure_pct": _float(probation_policy.get("max_total_exposure_pct")),
                "max_per_trade_pct": _float(probation_policy.get("max_per_trade_pct")),
            }
        )
    positive_exposure_caps = [
        candidate["max_total_exposure_pct"]
        for candidate in cap_candidates
        if candidate["max_total_exposure_pct"] > 0.0
    ]
    positive_trade_caps = [
        candidate["max_per_trade_pct"]
        for candidate in cap_candidates
        if candidate["max_per_trade_pct"] > 0.0
    ]
    recommended_max_total_exposure_pct = (
        min(positive_exposure_caps)
        if positive_exposure_caps
        else 0.0
    )
    recommended_max_per_trade_pct = (
        min(positive_trade_caps)
        if positive_trade_caps
        else 0.0
    )

    shadow_blockers: list[str] = []
    if health_status not in {"ok", "warning"}:
        shadow_blockers.append(f"System health is {health_status}, so even shadow deployment is blocked.")
    if health_should_halt:
        shadow_blockers.append("Health monitor currently requests a halt.")
    if not stress_kernel:
        shadow_blockers.append("Streaming stress kernel status is missing.")
    if not stress_field:
        shadow_blockers.append("Stateful stress field status is missing.")
    if stress_kernel and stress_field and not stress_observation_consistent:
        shadow_blockers.append("Stress kernel and stateful field are not aligned to the same observed snapshot.")
    if stress_field and (stress_field_should_halt or not stress_field_allow_entries):
        shadow_blockers.append("Stateful stress field is already in a halt or no-entry posture.")

    allow_shadow_live = not shadow_blockers

    probation_blockers: list[str] = []
    if not paper_validation_ready:
        probation_blockers.append("Strict paper validation has not passed.")
    if not shadow_ready:
        probation_blockers.append("Shadow execution drift is not yet within tolerance.")
    if not shadow_live:
        probation_blockers.append("Shadow live comparator status is missing.")
    elif not shadow_live_ready:
        probation_blockers.append("Shadow live comparator has not cleared broker-quoted burn-in.")
    if shadow_compared_trade_count < thresholds.min_shadow_compared_trades_for_probation:
        probation_blockers.append(
            f"Need at least {thresholds.min_shadow_compared_trades_for_probation} shadow comparisons for probation deployment."
        )
    if shadow_live_entry_comparison_count < thresholds.min_shadow_live_entry_comparisons_for_probation:
        probation_blockers.append(
            f"Need at least {thresholds.min_shadow_live_entry_comparisons_for_probation} broker-quoted entry comparisons for probation deployment."
        )
    if shadow_live_exit_comparison_count < thresholds.min_shadow_live_exit_comparisons_for_probation:
        probation_blockers.append(
            f"Need at least {thresholds.min_shadow_live_exit_comparisons_for_probation} broker-quoted exit comparisons for probation deployment."
        )
    if shadow_live_validation_runtime_days < thresholds.min_shadow_live_validation_days_for_probation:
        probation_blockers.append(
            f"Need at least {thresholds.min_shadow_live_validation_days_for_probation:.1f} shadow-live days before probation deployment."
        )
    if not certification_current_green:
        probation_blockers.append("Production certification is not currently green.")
    if drift_risk_score > thresholds.max_drift_risk_score_for_probation:
        probation_blockers.append("Drift risk is too elevated for probation deployment.")
    if survivability_score < thresholds.min_survivability_score_for_probation:
        probation_blockers.append("Survivability score is too weak for probation deployment.")
    if stress_field_hysteresis > thresholds.max_stress_hysteresis_for_probation:
        probation_blockers.append("Stress hysteresis remains too elevated for probation capital.")
    if stress_field_adversarial_intensity > thresholds.max_stress_adversarial_intensity_for_probation:
        probation_blockers.append("Adversarial field intensity remains too high for probation capital.")
    if not allow_kernel_probation:
        probation_blockers.append("Streaming stress kernel does not currently allow probation deployment.")

    allow_probation_live = allow_shadow_live and not probation_blockers

    full_live_blockers: list[str] = []
    if not certification_ready_for_live:
        full_live_blockers.append("Production certification has not fully passed yet.")
    if shadow_compared_trade_count < thresholds.min_shadow_compared_trades_for_live:
        full_live_blockers.append(
            f"Need at least {thresholds.min_shadow_compared_trades_for_live} shadow comparisons before full live deployment."
        )
    if shadow_live_entry_comparison_count < thresholds.min_shadow_live_entry_comparisons_for_live:
        full_live_blockers.append(
            f"Need at least {thresholds.min_shadow_live_entry_comparisons_for_live} broker-quoted entry comparisons before full live deployment."
        )
    if shadow_live_exit_comparison_count < thresholds.min_shadow_live_exit_comparisons_for_live:
        full_live_blockers.append(
            f"Need at least {thresholds.min_shadow_live_exit_comparisons_for_live} broker-quoted exit comparisons before full live deployment."
        )
    if shadow_live_validation_runtime_days < thresholds.min_shadow_live_validation_days_for_live:
        full_live_blockers.append(
            f"Need at least {thresholds.min_shadow_live_validation_days_for_live:.1f} shadow-live days before full live deployment."
        )
    if drift_risk_score > thresholds.max_drift_risk_score_for_live:
        full_live_blockers.append("Drift risk remains above the full-live threshold.")
    if survivability_score < thresholds.min_survivability_score_for_live:
        full_live_blockers.append("Survivability score remains below the full-live threshold.")
    if stress_field_hysteresis > thresholds.max_stress_hysteresis_for_live:
        full_live_blockers.append("Stress hysteresis remains too elevated for full live capital.")
    if stress_field_adversarial_intensity > thresholds.max_stress_adversarial_intensity_for_live:
        full_live_blockers.append("Adversarial field intensity remains too high for full live capital.")
    if not allow_kernel_full_live:
        full_live_blockers.append("Streaming stress kernel does not currently allow full live deployment.")

    allow_full_live = allow_probation_live and not full_live_blockers

    allowed_mode = "blocked"
    reasons = list(shadow_blockers)
    if allow_shadow_live:
        allowed_mode = "shadow_live"
        reasons = list(probation_blockers)
    if allow_probation_live:
        allowed_mode = "probation_live"
        reasons = list(full_live_blockers)
    if allow_full_live:
        allowed_mode = "full_live"
        reasons = []

    return DeploymentGateReport(
        generated_at=_now_iso(),
        allowed_mode=allowed_mode,
        allow_shadow_live=allow_shadow_live,
        allow_probation_live=allow_probation_live,
        allow_full_live=allow_full_live,
        health_status=health_status,
        health_should_halt=health_should_halt,
        paper_validation_ready=paper_validation_ready,
        shadow_ready=shadow_ready,
        shadow_compared_trade_count=shadow_compared_trade_count,
        shadow_live_ready=shadow_live_ready,
        shadow_live_entry_comparison_count=shadow_live_entry_comparison_count,
        shadow_live_exit_comparison_count=shadow_live_exit_comparison_count,
        shadow_live_validation_runtime_days=shadow_live_validation_runtime_days,
        certification_current_green=certification_current_green,
        certification_ready_for_live=certification_ready_for_live,
        drift_risk_score=drift_risk_score,
        survivability_score=survivability_score,
        stress_observation_consistent=stress_observation_consistent,
        stress_field_phase=stress_field_phase,
        stress_field_hysteresis=stress_field_hysteresis,
        stress_field_adversarial_intensity=stress_field_adversarial_intensity,
        recommended_max_total_exposure_pct=recommended_max_total_exposure_pct,
        recommended_max_per_trade_pct=recommended_max_per_trade_pct,
        reasons=reasons,
    )


def write_deployment_gate_report(report: DeploymentGateReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))


def format_deployment_gate_report(report: DeploymentGateReport) -> str:
    lines = [
        "Deployment Gate Report",
        f"  Allowed mode: {report.allowed_mode}",
        f"  Shadow live: {'YES' if report.allow_shadow_live else 'NO'} | probation: {'YES' if report.allow_probation_live else 'NO'} | full live: {'YES' if report.allow_full_live else 'NO'}",
        f"  Health: {report.health_status} | halt={report.health_should_halt}",
        f"  Paper validation: {'PASS' if report.paper_validation_ready else 'FAIL'}",
        f"  Shadow execution: {'PASS' if report.shadow_ready else 'FAIL'} ({report.shadow_compared_trade_count} compared)",
        f"  Shadow live comparator: {'PASS' if report.shadow_live_ready else 'FAIL'} | entry={report.shadow_live_entry_comparison_count} exit={report.shadow_live_exit_comparison_count} | runtime={report.shadow_live_validation_runtime_days:.1f}d",
        f"  Certification: green={report.certification_current_green} | ready={report.certification_ready_for_live}",
        f"  Drift/Survivability: drift={report.drift_risk_score:.1f}/100 | survivability={report.survivability_score:.1f}/100",
        f"  Stress field: phase={report.stress_field_phase} | hysteresis={report.stress_field_hysteresis:.0%} | adversary={report.stress_field_adversarial_intensity:.0%} | observation={'aligned' if report.stress_observation_consistent else 'mismatched'}",
    ]
    if report.recommended_max_total_exposure_pct > 0.0 or report.recommended_max_per_trade_pct > 0.0:
        lines.append(
            "  Recommended caps: "
            f"exposure={report.recommended_max_total_exposure_pct:.2%} | "
            f"per-trade={report.recommended_max_per_trade_pct:.2%}"
        )
    if report.reasons:
        lines.append("  Blocking reasons:")
        for reason in report.reasons:
            lines.append(f"    - {reason}")
    return "\n".join(lines)


__all__ = [
    "DeploymentGateReport",
    "DeploymentGateThresholds",
    "build_deployment_gate_report",
    "format_deployment_gate_report",
    "write_deployment_gate_report",
]
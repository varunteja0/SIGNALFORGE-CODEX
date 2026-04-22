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


def _iter_market_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in snapshot.items():
        if key.startswith("_") or not isinstance(value, dict):
            continue
        rows.append(value)
    return rows


def _min_positive(*values: float) -> float:
    usable = [value for value in values if value > 0.0]
    return min(usable) if usable else 0.0


@dataclass
class CapitalFirewallThresholds:
    reduced_size_multiplier: float = 0.35
    moderate_size_multiplier: float = 0.65
    max_stress_hysteresis_for_full_size: float = 0.35
    max_stress_hysteresis_for_reduced_size: float = 0.55
    max_adversarial_intensity_for_full_size: float = 0.35
    max_adversarial_intensity_for_reduced_size: float = 0.60
    max_collapse_probability_for_full_size: float = 0.35
    max_collapse_probability_for_reduced_size: float = 0.62
    min_collapse_horizon_for_full_size: int = 4
    min_collapse_horizon_for_reduced_size: int = 2
    min_execution_fidelity_for_full_size: float = 80.0
    min_execution_fidelity_for_reduced_size: float = 60.0
    max_market_vol_ratio_for_full_size: float = 1.60
    max_market_vol_ratio_for_reduced_size: float = 2.20
    max_market_atr_expansion_for_full_size: float = 1.50
    max_market_atr_expansion_for_reduced_size: float = 2.10
    max_corr_shift_for_full_size: float = 0.12
    max_corr_shift_for_reduced_size: float = 0.22
    max_book_spread_for_full_size_bps: float = 12.0
    max_book_spread_for_reduced_size_bps: float = 20.0
    max_book_impact_for_full_size_bps: float = 18.0
    max_book_impact_for_reduced_size_bps: float = 30.0


@dataclass
class CapitalFirewallReport:
    generated_at: str
    operating_mode: str
    enforced: bool
    deployment_allowed_mode: str
    decision: str
    allow_new_entries: bool
    base_max_total_exposure_pct: float
    base_max_per_trade_pct: float
    max_total_exposure_pct: float
    max_per_trade_pct: float
    stress_hysteresis: float
    stress_adversarial_intensity: float
    collapse_probability: float
    collapse_horizon_ticks: int
    execution_fidelity_score: float
    execution_fidelity_level: str
    execution_miss_rate: float
    avg_entry_slippage_bps: float
    avg_fill_ratio: float
    avg_book_spread_bps: float
    avg_book_impact_bps: float
    market_max_vol_ratio: float
    market_max_atr_expansion: float
    market_mean_abs_corr_48h: float
    market_corr_shift_48h: float
    market_dispersion_24h: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_capital_firewall_report(
    base_dir: Path,
    *,
    operating_mode: str = "paper",
    configured_max_total_exposure_pct: float = 0.0,
    configured_max_per_trade_pct: float = 0.0,
    thresholds: CapitalFirewallThresholds | None = None,
) -> CapitalFirewallReport:
    thresholds = thresholds or CapitalFirewallThresholds()
    deployment_gate = _load_json(base_dir / "deployment_gate_status.json", {})
    execution_drift = _load_json(base_dir / "execution_drift_status.json", {})
    stress_field = _load_json(base_dir / "stress_field_state.json", {})
    market_snapshot = _load_json(base_dir / "market_snapshot.json", {})

    deployment_allowed_mode = str(deployment_gate.get("allowed_mode", "blocked"))
    enforced = operating_mode in {"probation_live", "live"}

    gate_total = _float(deployment_gate.get("recommended_max_total_exposure_pct"))
    gate_trade = _float(deployment_gate.get("recommended_max_per_trade_pct"))
    configured_total = _float(configured_max_total_exposure_pct)
    configured_trade = _float(configured_max_per_trade_pct)
    base_max_total_exposure_pct = configured_total if configured_total > 0.0 else gate_total
    base_max_per_trade_pct = configured_trade if configured_trade > 0.0 else gate_trade

    stress_hysteresis = _float(stress_field.get("hysteresis_score"))
    stress_adversarial_intensity = _float(
        (stress_field.get("adversarial_input") or {}).get("intensity")
        if isinstance(stress_field.get("adversarial_input"), dict)
        else 0.0
    )
    collapse_probability = _float(stress_field.get("collapse_probability"))
    collapse_horizon_ticks = int(stress_field.get("collapse_horizon_ticks", 6) or 6)
    stress_should_halt = bool(stress_field.get("should_halt"))
    stress_allow_entries = bool(stress_field.get("allow_entries", True))

    execution_fidelity_score = _float(execution_drift.get("execution_fidelity_score"), 100.0)
    execution_fidelity_level = str(execution_drift.get("execution_fidelity_level", "stable"))
    execution_miss_rate = _float(execution_drift.get("miss_rate"))
    avg_entry_slippage_bps = _float(execution_drift.get("avg_entry_slippage_bps"))
    avg_fill_ratio = _float(execution_drift.get("avg_fill_ratio"), 1.0)
    avg_book_spread_bps = _float(execution_drift.get("avg_book_spread_bps"))
    avg_book_impact_bps = _float(execution_drift.get("avg_book_impact_bps"))
    reliable_for_capital = bool(execution_drift.get("reliable_for_capital", True))

    market_rows = _iter_market_rows(market_snapshot if isinstance(market_snapshot, dict) else {})
    market_max_vol_ratio = max((_float(row.get("vol_ratio"), 1.0) for row in market_rows), default=1.0)
    market_max_atr_expansion = max((_float(row.get("atr_exp"), 1.0) for row in market_rows), default=1.0)
    cross_asset = market_snapshot.get("_cross_asset") if isinstance(market_snapshot, dict) and isinstance(market_snapshot.get("_cross_asset"), dict) else {}
    market_mean_abs_corr_48h = _float(cross_asset.get("mean_abs_corr_48h"))
    market_corr_shift_48h = _float(cross_asset.get("corr_shift_48h"))
    market_dispersion_24h = _float(cross_asset.get("dispersion_24h"))

    blockers: list[str] = []
    reducers: list[str] = []
    size_multiplier = 1.0

    if stress_should_halt or not stress_allow_entries:
        blockers.append("Stress field is in halt or no-entry posture.")
    if collapse_probability >= 0.82 or collapse_horizon_ticks <= 1:
        blockers.append("Stress field predicts an imminent collapse manifold.")
    if enforced and not reliable_for_capital:
        blockers.append("Execution drift is not yet reliable enough for real capital.")
    if execution_fidelity_score < 45.0:
        blockers.append("Execution fidelity has degraded below the live-capital floor.")
    if avg_book_spread_bps > thresholds.max_book_spread_for_reduced_size_bps * 1.4:
        blockers.append("Observed execution spreads are too wide for live capital.")
    if avg_book_impact_bps > thresholds.max_book_impact_for_reduced_size_bps * 1.4:
        blockers.append("Observed execution impact is too high for live capital.")
    if (
        market_max_vol_ratio > thresholds.max_market_vol_ratio_for_reduced_size * 1.15
        and market_corr_shift_48h > thresholds.max_corr_shift_for_reduced_size
    ):
        blockers.append("Market volatility and correlation shift are both outside the live envelope.")

    if stress_hysteresis > thresholds.max_stress_hysteresis_for_full_size:
        reducers.append("Stress hysteresis remains elevated, so capital must be reduced.")
        size_multiplier = min(size_multiplier, thresholds.reduced_size_multiplier if stress_hysteresis > thresholds.max_stress_hysteresis_for_reduced_size else thresholds.moderate_size_multiplier)
    if stress_adversarial_intensity > thresholds.max_adversarial_intensity_for_full_size:
        reducers.append("Adversarial field intensity remains elevated.")
        size_multiplier = min(size_multiplier, thresholds.reduced_size_multiplier if stress_adversarial_intensity > thresholds.max_adversarial_intensity_for_reduced_size else thresholds.moderate_size_multiplier)
    if (
        collapse_probability > thresholds.max_collapse_probability_for_full_size
        or collapse_horizon_ticks < thresholds.min_collapse_horizon_for_full_size
    ):
        reducers.append("Collapse proximity remains inside the reduced-size band.")
        size_multiplier = min(size_multiplier, thresholds.reduced_size_multiplier if collapse_probability > thresholds.max_collapse_probability_for_reduced_size or collapse_horizon_ticks <= thresholds.min_collapse_horizon_for_reduced_size else thresholds.moderate_size_multiplier)
    if execution_fidelity_score < thresholds.min_execution_fidelity_for_full_size:
        reducers.append("Execution fidelity is below the full-size threshold.")
        size_multiplier = min(size_multiplier, thresholds.reduced_size_multiplier if execution_fidelity_score < thresholds.min_execution_fidelity_for_reduced_size else thresholds.moderate_size_multiplier)
    if avg_entry_slippage_bps > 8.0 or execution_miss_rate > 0.08 or avg_fill_ratio < 0.95:
        reducers.append("Paper execution drift shows elevated live-trading friction.")
        size_multiplier = min(size_multiplier, thresholds.reduced_size_multiplier if avg_entry_slippage_bps > 12.0 or execution_miss_rate > 0.15 or avg_fill_ratio < 0.85 else thresholds.moderate_size_multiplier)
    if avg_book_spread_bps > thresholds.max_book_spread_for_full_size_bps:
        reducers.append("Book spread regime is wider than the full-size budget.")
        size_multiplier = min(size_multiplier, thresholds.moderate_size_multiplier)
    if avg_book_impact_bps > thresholds.max_book_impact_for_full_size_bps:
        reducers.append("Estimated market impact is above the full-size budget.")
        size_multiplier = min(size_multiplier, thresholds.moderate_size_multiplier)
    if market_max_vol_ratio > thresholds.max_market_vol_ratio_for_full_size:
        reducers.append("Realized volume shock is above the full-size envelope.")
        size_multiplier = min(size_multiplier, thresholds.reduced_size_multiplier if market_max_vol_ratio > thresholds.max_market_vol_ratio_for_reduced_size else thresholds.moderate_size_multiplier)
    if market_max_atr_expansion > thresholds.max_market_atr_expansion_for_full_size:
        reducers.append("ATR expansion is above the full-size envelope.")
        size_multiplier = min(size_multiplier, thresholds.reduced_size_multiplier if market_max_atr_expansion > thresholds.max_market_atr_expansion_for_reduced_size else thresholds.moderate_size_multiplier)
    if market_corr_shift_48h > thresholds.max_corr_shift_for_full_size:
        reducers.append("Cross-asset correlation is shifting too quickly for full-size deployment.")
        size_multiplier = min(size_multiplier, thresholds.reduced_size_multiplier if market_corr_shift_48h > thresholds.max_corr_shift_for_reduced_size else thresholds.moderate_size_multiplier)

    if blockers:
        decision = "no_trade"
        allow_new_entries = False
        max_total_exposure_pct = 0.0
        max_per_trade_pct = 0.0
        reasons = blockers + reducers
    elif reducers:
        decision = "allow_reduced_size"
        allow_new_entries = True
        max_total_exposure_pct = base_max_total_exposure_pct * size_multiplier
        max_per_trade_pct = base_max_per_trade_pct * size_multiplier
        reasons = reducers
    else:
        decision = "allow_full_size"
        allow_new_entries = True
        max_total_exposure_pct = base_max_total_exposure_pct
        max_per_trade_pct = base_max_per_trade_pct
        reasons = []

    return CapitalFirewallReport(
        generated_at=_now_iso(),
        operating_mode=operating_mode,
        enforced=enforced,
        deployment_allowed_mode=deployment_allowed_mode,
        decision=decision,
        allow_new_entries=allow_new_entries,
        base_max_total_exposure_pct=base_max_total_exposure_pct,
        base_max_per_trade_pct=base_max_per_trade_pct,
        max_total_exposure_pct=max_total_exposure_pct,
        max_per_trade_pct=max_per_trade_pct,
        stress_hysteresis=stress_hysteresis,
        stress_adversarial_intensity=stress_adversarial_intensity,
        collapse_probability=collapse_probability,
        collapse_horizon_ticks=collapse_horizon_ticks,
        execution_fidelity_score=execution_fidelity_score,
        execution_fidelity_level=execution_fidelity_level,
        execution_miss_rate=execution_miss_rate,
        avg_entry_slippage_bps=avg_entry_slippage_bps,
        avg_fill_ratio=avg_fill_ratio,
        avg_book_spread_bps=avg_book_spread_bps,
        avg_book_impact_bps=avg_book_impact_bps,
        market_max_vol_ratio=market_max_vol_ratio,
        market_max_atr_expansion=market_max_atr_expansion,
        market_mean_abs_corr_48h=market_mean_abs_corr_48h,
        market_corr_shift_48h=market_corr_shift_48h,
        market_dispersion_24h=market_dispersion_24h,
        reasons=reasons,
    )


def write_capital_firewall_report(report: CapitalFirewallReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))


def format_capital_firewall_report(report: CapitalFirewallReport) -> str:
    lines = [
        "Capital Firewall Report",
        (
            "  Decision: "
            f"{report.decision} | enforced={'YES' if report.enforced else 'NO'} | "
            f"mode={report.operating_mode} ceiling={report.deployment_allowed_mode}"
        ),
        (
            "  Caps: "
            f"base={report.base_max_total_exposure_pct:.2%}/{report.base_max_per_trade_pct:.2%} "
            f"effective={report.max_total_exposure_pct:.2%}/{report.max_per_trade_pct:.2%}"
        ),
        (
            "  Stress: "
            f"collapse={report.collapse_probability:.0%} in ~{report.collapse_horizon_ticks} ticks | "
            f"hysteresis={report.stress_hysteresis:.0%} adversary={report.stress_adversarial_intensity:.0%}"
        ),
        (
            "  Execution: "
            f"fidelity={report.execution_fidelity_level} {report.execution_fidelity_score:.1f}/100 | "
            f"slip={report.avg_entry_slippage_bps:.2f}bps miss={report.execution_miss_rate:.0%} fill={report.avg_fill_ratio:.0%}"
        ),
        (
            "  Market: "
            f"vol={report.market_max_vol_ratio:.2f}x atr={report.market_max_atr_expansion:.2f}x "
            f"corr={report.market_mean_abs_corr_48h:.2f}/{report.market_corr_shift_48h:.2f}"
        ),
    ]
    if report.reasons:
        lines.append("  Reasons:")
        lines.extend(f"    - {reason}" for reason in report.reasons)
    return "\n".join(lines)


__all__ = [
    "CapitalFirewallReport",
    "CapitalFirewallThresholds",
    "build_capital_firewall_report",
    "format_capital_firewall_report",
    "write_capital_firewall_report",
]
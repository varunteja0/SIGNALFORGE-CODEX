from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    except OSError:
        return []
    return rows


def _iso_to_ts(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


@dataclass
class DriftIntelligenceThresholds:
    lookback_history_points: int = 96
    lookback_trade_points: int = 120
    lookback_adaptive_points: int = 96
    min_recent_green_ratio: float = 0.85
    max_state_flip_rate: float = 0.12
    max_entry_slippage_bps: float = 12.0
    max_abs_pnl_divergence_pct: float = 20.0
    max_miss_rate: float = 0.15
    max_slippage_trend_bps_per_trade: float = 0.50
    max_adaptive_pause_rate: float = 0.15
    max_allocation_shift: float = 0.35
    max_disabled_strategy_flip_rate: float = 0.15
    micro_live_min_green_days: float = 30.0
    scale_up_min_green_days: float = 60.0


@dataclass
class DeploymentRecommendation:
    mode: str = "paper_shadow"
    max_capital_fraction: float = 0.0
    max_total_exposure_pct: float = 0.0
    max_per_trade_pct: float = 0.0
    requires_manual_approval: bool = True
    reasons: list[str] = field(default_factory=list)


@dataclass
class DriftIntelligenceReport:
    generated_at: str
    current_green_snapshot: bool
    risk_score: float
    risk_level: str
    pre_kill_switch_warning: bool
    predicted_certification_failure: bool
    history_points: int
    recent_green_ratio: float
    gate_flip_count: int
    gate_flip_rate: float
    current_green_streak_days: float
    longest_green_streak_days: float
    longest_red_streak_hours: float
    avg_entry_slippage_bps: float
    avg_abs_pnl_divergence_pct: float
    miss_rate: float
    slippage_trend_bps_per_trade: float
    adaptive_pause_rate: float
    allocation_shift_score: float
    disabled_strategy_flip_rate: float
    broker_warning_pressure: float
    deployment_recommendation: DeploymentRecommendation = field(default_factory=DeploymentRecommendation)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_current_green_snapshot(base_dir: Path) -> bool:
    paper_validation = _load_json(base_dir / "paper_validation_status.json", {})
    health = _load_json(base_dir / "health.json", {})
    parity = _load_json(base_dir / "trade_parity_status.json", {})
    shadow = _load_json(base_dir / "shadow_execution_status.json", {})
    failure_drill = _load_json(base_dir / "failure_drill_report.json", {})
    broker = _load_json(base_dir / "broker_reconciliation_status.json", {})

    return (
        bool(paper_validation.get("ready_for_live"))
        and str(health.get("overall_status", "unknown")) == "ok"
        and not bool(health.get("should_halt"))
        and str(parity.get("verdict", "unknown")) == "PASS"
        and bool(shadow.get("ready_for_live"))
        and bool(failure_drill.get("all_passed"))
        and str(broker.get("overall_status", "unknown")) == "ok"
        and int(broker.get("critical_issue_count", 0) or 0) == 0
    )


def _compute_stability_metrics(
    history_rows: list[dict[str, Any]],
    *,
    current_green_snapshot: bool,
    now_iso: str,
) -> dict[str, float | int]:
    samples: list[tuple[float, bool]] = []
    for row in history_rows:
        ts = _iso_to_ts(str(row.get("timestamp", "")))
        if ts <= 0.0:
            continue
        samples.append((ts, bool(row.get("current_green"))))
    samples.append((_iso_to_ts(now_iso), current_green_snapshot))
    samples.sort(key=lambda item: item[0])

    if not samples:
        return {
            "history_points": 0,
            "recent_green_ratio": 0.0,
            "gate_flip_count": 0,
            "gate_flip_rate": 0.0,
            "current_green_streak_days": 0.0,
            "longest_green_streak_days": 0.0,
            "longest_red_streak_hours": 0.0,
        }

    states = [state for _, state in samples]
    flip_count = sum(1 for prev, cur in zip(states, states[1:]) if prev != cur)
    flip_rate = flip_count / max(len(states) - 1, 1)
    recent_green_ratio = sum(1 for state in states if state) / len(states)

    longest_green = 0.0
    longest_red = 0.0
    run_state = samples[0][1]
    run_start = samples[0][0]
    for ts, state in samples[1:]:
        if state == run_state:
            continue
        duration = max(ts - run_start, 0.0)
        if run_state:
            longest_green = max(longest_green, duration)
        else:
            longest_red = max(longest_red, duration)
        run_state = state
        run_start = ts
    final_duration = max(samples[-1][0] - run_start, 0.0)
    if run_state:
        longest_green = max(longest_green, final_duration)
    else:
        longest_red = max(longest_red, final_duration)

    current_green_streak = 0.0
    if samples[-1][1]:
        current_green_streak = max(samples[-1][0] - run_start, 0.0)

    return {
        "history_points": len(samples),
        "recent_green_ratio": recent_green_ratio,
        "gate_flip_count": flip_count,
        "gate_flip_rate": flip_rate,
        "current_green_streak_days": current_green_streak / 86400.0,
        "longest_green_streak_days": longest_green / 86400.0,
        "longest_red_streak_hours": longest_red / 3600.0,
    }


def _compute_divergence_metrics(
    divergence_rows: list[dict[str, Any]],
    *,
    thresholds: DriftIntelligenceThresholds,
) -> dict[str, float]:
    rows = [row for row in divergence_rows if isinstance(row, dict)][-thresholds.lookback_trade_points :]
    if not rows:
        return {
            "avg_entry_slippage_bps": 0.0,
            "avg_abs_pnl_divergence_pct": 0.0,
            "miss_rate": 0.0,
            "slippage_trend_bps_per_trade": 0.0,
        }

    executed = [row for row in rows if not bool(row.get("missed"))]
    missed = [row for row in rows if bool(row.get("missed"))]
    entry_slips = [abs(_float(row.get("entry_slippage_bps"))) for row in executed]
    pnl_divs = [abs(_float(row.get("pnl_divergence_pct"))) for row in executed]
    slippage_trend = 0.0
    if len(entry_slips) >= 5:
        x = np.arange(len(entry_slips), dtype=float)
        slippage_trend = float(np.polyfit(x, np.asarray(entry_slips, dtype=float), 1)[0])

    return {
        "avg_entry_slippage_bps": _mean(entry_slips),
        "avg_abs_pnl_divergence_pct": _mean(pnl_divs),
        "miss_rate": len(missed) / len(rows),
        "slippage_trend_bps_per_trade": slippage_trend,
    }


def _compute_adaptive_metrics(
    adaptive_rows: list[dict[str, Any]],
    *,
    thresholds: DriftIntelligenceThresholds,
) -> dict[str, float]:
    rows = [row for row in adaptive_rows if isinstance(row, dict)][-thresholds.lookback_adaptive_points :]
    if len(rows) < 2:
        return {
            "adaptive_pause_rate": 0.0,
            "allocation_shift_score": 0.0,
            "disabled_strategy_flip_rate": 0.0,
        }

    pause_count = sum(1 for row in rows if str(row.get("safety_action", "allow")) in {"pause_entries", "halt"})
    allocation_shifts: list[float] = []
    disabled_flips = 0
    for prev, cur in zip(rows, rows[1:]):
        prev_weights = prev.get("allocation_weights") or {}
        cur_weights = cur.get("allocation_weights") or {}
        keys = set(prev_weights) | set(cur_weights)
        shift = 0.5 * sum(abs(_float(cur_weights.get(key)) - _float(prev_weights.get(key))) for key in keys)
        allocation_shifts.append(shift)

        prev_disabled = set((prev.get("disabled_strategies") or {}).keys())
        cur_disabled = set((cur.get("disabled_strategies") or {}).keys())
        if prev_disabled != cur_disabled:
            disabled_flips += 1

    return {
        "adaptive_pause_rate": pause_count / len(rows),
        "allocation_shift_score": _mean(allocation_shifts),
        "disabled_strategy_flip_rate": disabled_flips / max(len(rows) - 1, 1),
    }


def _build_deployment_recommendation(
    *,
    current_green_snapshot: bool,
    risk_level: str,
    predicted_certification_failure: bool,
    current_green_streak_days: float,
    gate_flip_rate: float,
    thresholds: DriftIntelligenceThresholds,
) -> DeploymentRecommendation:
    if not current_green_snapshot:
        return DeploymentRecommendation(
            mode="paper_shadow",
            reasons=["Core certification snapshot is not green yet."],
        )

    if predicted_certification_failure or risk_level in {"high", "critical"}:
        return DeploymentRecommendation(
            mode="blocked",
            reasons=["Drift intelligence predicts instability under current conditions."],
        )

    if current_green_streak_days < thresholds.micro_live_min_green_days:
        return DeploymentRecommendation(
            mode="paper_shadow",
            reasons=[
                f"Burn-in streak is {current_green_streak_days:.1f} days; require {thresholds.micro_live_min_green_days:.1f} days before micro-live."
            ],
        )

    if current_green_streak_days >= thresholds.scale_up_min_green_days and gate_flip_rate <= thresholds.max_state_flip_rate * 0.5:
        return DeploymentRecommendation(
            mode="scale_up",
            max_capital_fraction=0.03,
            max_total_exposure_pct=0.03,
            max_per_trade_pct=0.005,
            reasons=["Drift is stable enough for controlled scaling beyond micro-live probation."],
        )

    return DeploymentRecommendation(
        mode="micro_live",
        max_capital_fraction=0.01,
        max_total_exposure_pct=0.01,
        max_per_trade_pct=0.0025,
        reasons=["Only micro-live probation is justified; keep real exposure minimal until more evidence accrues."],
    )


def build_drift_intelligence_report(
    base_dir: Path,
    thresholds: DriftIntelligenceThresholds | None = None,
) -> DriftIntelligenceReport:
    thresholds = thresholds or DriftIntelligenceThresholds()
    now_iso = _now_iso()

    history_rows = _load_jsonl(base_dir / "production_certification_history.jsonl")[-thresholds.lookback_history_points :]
    divergence_rows = _load_json(base_dir / "divergence_log.json", [])
    if isinstance(divergence_rows, dict):
        divergence_rows = divergence_rows.get("comparisons", [])
    if not isinstance(divergence_rows, list):
        divergence_rows = []
    adaptive_rows = _load_jsonl(base_dir / "adaptive_cycle_ledger.jsonl")
    broker = _load_json(base_dir / "broker_reconciliation_status.json", {})
    health = _load_json(base_dir / "health.json", {})
    shadow = _load_json(base_dir / "shadow_execution_status.json", {})
    parity = _load_json(base_dir / "trade_parity_status.json", {})

    current_green_snapshot = _compute_current_green_snapshot(base_dir)
    stability = _compute_stability_metrics(
        history_rows,
        current_green_snapshot=current_green_snapshot,
        now_iso=now_iso,
    )
    divergence = _compute_divergence_metrics(divergence_rows, thresholds=thresholds)
    adaptive = _compute_adaptive_metrics(adaptive_rows, thresholds=thresholds)

    broker_warning_pressure = float(int(broker.get("critical_issue_count", 0) or 0) * 3 + int(broker.get("warning_issue_count", 0) or 0))
    health_pressure = 1.0 if str(health.get("overall_status", "unknown")) == "warning" else 2.0 if str(health.get("overall_status", "unknown")) == "critical" else 0.0
    shadow_pressure = 1.0 if shadow and not bool(shadow.get("ready_for_live")) else 0.0
    parity_pressure = 1.0 if str(parity.get("verdict", "unknown")) in {"WARN", "FAIL"} else 0.0

    components = {
        "flip_rate": _clip01(_float(stability["gate_flip_rate"]) / thresholds.max_state_flip_rate),
        "green_ratio": _clip01(max(thresholds.min_recent_green_ratio - _float(stability["recent_green_ratio"]), 0.0) / max(thresholds.min_recent_green_ratio, 1e-9)),
        "entry_slippage": _clip01(divergence["avg_entry_slippage_bps"] / thresholds.max_entry_slippage_bps),
        "pnl_divergence": _clip01(divergence["avg_abs_pnl_divergence_pct"] / thresholds.max_abs_pnl_divergence_pct),
        "miss_rate": _clip01(divergence["miss_rate"] / thresholds.max_miss_rate),
        "slippage_trend": _clip01(max(divergence["slippage_trend_bps_per_trade"], 0.0) / thresholds.max_slippage_trend_bps_per_trade),
        "adaptive_pause": _clip01(adaptive["adaptive_pause_rate"] / thresholds.max_adaptive_pause_rate),
        "allocation_shift": _clip01(adaptive["allocation_shift_score"] / thresholds.max_allocation_shift),
        "disabled_flip": _clip01(adaptive["disabled_strategy_flip_rate"] / thresholds.max_disabled_strategy_flip_rate),
        "broker": _clip01(broker_warning_pressure / 4.0),
        "health": _clip01(health_pressure / 2.0),
        "shadow": _clip01(shadow_pressure),
        "parity": _clip01(parity_pressure),
    }
    weights = {
        "flip_rate": 1.2,
        "green_ratio": 1.0,
        "entry_slippage": 1.0,
        "pnl_divergence": 1.0,
        "miss_rate": 0.8,
        "slippage_trend": 0.8,
        "adaptive_pause": 1.2,
        "allocation_shift": 0.8,
        "disabled_flip": 0.8,
        "broker": 1.0,
        "health": 1.0,
        "shadow": 0.8,
        "parity": 0.8,
    }
    weighted_total = sum(weights.values())
    risk_score = 100.0 * sum(components[name] * weight for name, weight in weights.items()) / weighted_total

    if risk_score >= 75.0:
        risk_level = "critical"
    elif risk_score >= 55.0:
        risk_level = "high"
    elif risk_score >= 35.0:
        risk_level = "moderate"
    else:
        risk_level = "low"

    reasons: list[str] = []
    if _float(stability["gate_flip_rate"]) > thresholds.max_state_flip_rate:
        reasons.append(
            f"Certification gate flip rate {_float(stability['gate_flip_rate']):.0%} exceeds {_float(thresholds.max_state_flip_rate):.0%}."
        )
    if _float(stability["recent_green_ratio"]) < thresholds.min_recent_green_ratio:
        reasons.append(
            f"Recent green ratio {_float(stability['recent_green_ratio']):.0%} is below {_float(thresholds.min_recent_green_ratio):.0%}."
        )
    if divergence["avg_entry_slippage_bps"] > thresholds.max_entry_slippage_bps:
        reasons.append(
            f"Average entry slippage {divergence['avg_entry_slippage_bps']:.2f}bps exceeds {thresholds.max_entry_slippage_bps:.2f}bps."
        )
    if divergence["miss_rate"] > thresholds.max_miss_rate:
        reasons.append(
            f"Miss rate {divergence['miss_rate']:.0%} exceeds {thresholds.max_miss_rate:.0%}."
        )
    if divergence["avg_abs_pnl_divergence_pct"] > thresholds.max_abs_pnl_divergence_pct:
        reasons.append(
            f"Average absolute PnL divergence {divergence['avg_abs_pnl_divergence_pct']:.2f}% exceeds {thresholds.max_abs_pnl_divergence_pct:.2f}%."
        )
    if adaptive["adaptive_pause_rate"] > thresholds.max_adaptive_pause_rate:
        reasons.append(
            f"Adaptive pause rate {adaptive['adaptive_pause_rate']:.0%} exceeds {thresholds.max_adaptive_pause_rate:.0%}."
        )
    if adaptive["allocation_shift_score"] > thresholds.max_allocation_shift:
        reasons.append(
            f"Allocation shift score {adaptive['allocation_shift_score']:.2f} exceeds {thresholds.max_allocation_shift:.2f}."
        )
    if adaptive["disabled_strategy_flip_rate"] > thresholds.max_disabled_strategy_flip_rate:
        reasons.append(
            f"Disabled-strategy flip rate {adaptive['disabled_strategy_flip_rate']:.0%} exceeds {thresholds.max_disabled_strategy_flip_rate:.0%}."
        )
    if broker_warning_pressure > 0.0:
        reasons.append(f"Broker truth surface still carries warning pressure {broker_warning_pressure:.0f}.")
    if health_pressure > 0.0:
        reasons.append(f"System health is {health.get('overall_status', 'unknown')}, not fully nominal.")
    if shadow_pressure > 0.0:
        reasons.append("Shadow execution drift is not fully green.")
    if parity_pressure > 0.0:
        reasons.append("Parity verdict is not PASS.")

    predicted_certification_failure = (
        risk_level in {"high", "critical"}
        or _float(stability["gate_flip_rate"]) > thresholds.max_state_flip_rate
        or adaptive["adaptive_pause_rate"] > thresholds.max_adaptive_pause_rate
        or divergence["slippage_trend_bps_per_trade"] > thresholds.max_slippage_trend_bps_per_trade
    )
    pre_kill_switch_warning = predicted_certification_failure or risk_level in {"moderate", "high", "critical"}

    deployment = _build_deployment_recommendation(
        current_green_snapshot=current_green_snapshot,
        risk_level=risk_level,
        predicted_certification_failure=predicted_certification_failure,
        current_green_streak_days=_float(stability["current_green_streak_days"]),
        gate_flip_rate=_float(stability["gate_flip_rate"]),
        thresholds=thresholds,
    )

    return DriftIntelligenceReport(
        generated_at=now_iso,
        current_green_snapshot=current_green_snapshot,
        risk_score=float(risk_score),
        risk_level=risk_level,
        pre_kill_switch_warning=pre_kill_switch_warning,
        predicted_certification_failure=predicted_certification_failure,
        history_points=int(stability["history_points"]),
        recent_green_ratio=float(stability["recent_green_ratio"]),
        gate_flip_count=int(stability["gate_flip_count"]),
        gate_flip_rate=float(stability["gate_flip_rate"]),
        current_green_streak_days=float(stability["current_green_streak_days"]),
        longest_green_streak_days=float(stability["longest_green_streak_days"]),
        longest_red_streak_hours=float(stability["longest_red_streak_hours"]),
        avg_entry_slippage_bps=float(divergence["avg_entry_slippage_bps"]),
        avg_abs_pnl_divergence_pct=float(divergence["avg_abs_pnl_divergence_pct"]),
        miss_rate=float(divergence["miss_rate"]),
        slippage_trend_bps_per_trade=float(divergence["slippage_trend_bps_per_trade"]),
        adaptive_pause_rate=float(adaptive["adaptive_pause_rate"]),
        allocation_shift_score=float(adaptive["allocation_shift_score"]),
        disabled_strategy_flip_rate=float(adaptive["disabled_strategy_flip_rate"]),
        broker_warning_pressure=broker_warning_pressure,
        deployment_recommendation=deployment,
        reasons=reasons,
    )


def write_drift_intelligence_report(report: DriftIntelligenceReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))


def format_drift_intelligence_report(report: DriftIntelligenceReport) -> str:
    lines = [
        "Production Drift Intelligence Report",
        f"  Risk score: {report.risk_score:.1f}/100 ({report.risk_level})",
        f"  Pre-kill warning: {'YES' if report.pre_kill_switch_warning else 'NO'}",
        f"  Predicted certification failure: {'YES' if report.predicted_certification_failure else 'NO'}",
        f"  Gate stability: flips={report.gate_flip_count} ({report.gate_flip_rate:.0%}) | green ratio={report.recent_green_ratio:.0%}",
        f"  Current green streak: {report.current_green_streak_days:.1f}d | longest green: {report.longest_green_streak_days:.1f}d | longest red: {report.longest_red_streak_hours:.1f}h",
        f"  Divergence: slip={report.avg_entry_slippage_bps:.2f}bps | miss={report.miss_rate:.0%} | pnl={report.avg_abs_pnl_divergence_pct:.2f}% | trend={report.slippage_trend_bps_per_trade:+.2f}bps/trade",
        f"  Adaptive: pause rate={report.adaptive_pause_rate:.0%} | allocation shift={report.allocation_shift_score:.2f} | disable flips={report.disabled_strategy_flip_rate:.0%}",
        f"  Deployment mode: {report.deployment_recommendation.mode}",
    ]
    if report.deployment_recommendation.mode in {"micro_live", "scale_up"}:
        lines.append(
            f"  Recommended caps: capital={report.deployment_recommendation.max_capital_fraction:.1%} | exposure={report.deployment_recommendation.max_total_exposure_pct:.1%} | per-trade={report.deployment_recommendation.max_per_trade_pct:.2%}"
        )
    if report.reasons:
        lines.append("  Drift reasons:")
        for reason in report.reasons:
            lines.append(f"    - {reason}")
    if report.deployment_recommendation.reasons:
        lines.append("  Deployment rationale:")
        for reason in report.deployment_recommendation.reasons:
            lines.append(f"    - {reason}")
    return "\n".join(lines)


__all__ = [
    "DeploymentRecommendation",
    "DriftIntelligenceReport",
    "DriftIntelligenceThresholds",
    "build_drift_intelligence_report",
    "format_drift_intelligence_report",
    "write_drift_intelligence_report",
]
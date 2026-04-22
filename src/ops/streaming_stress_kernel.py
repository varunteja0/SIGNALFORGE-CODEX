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


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StreamingStressKernelThresholds:
    trajectory_window: int = 8
    min_history_points: int = 96
    max_trajectory_novelty_score: float = 60.0
    max_execution_friction_score: float = 55.0
    max_continuous_pressure_score: float = 60.0
    max_micro_collapse_probability: float = 0.55
    max_latency_p99_ms: float = 250.0
    max_latency_p999_ms: float = 450.0
    min_kill_switch_efficiency: float = 0.80


@dataclass
class ProbationLivePolicy:
    stage: str = "shadow"
    entry_action: str = "pause_entries"
    allow_probation_live: bool = False
    allow_full_live: bool = False
    max_capital_fraction: float = 0.0
    max_total_exposure_pct: float = 0.0
    max_per_trade_pct: float = 0.0
    entry_size_multiplier: float = 0.0
    exposure_multiplier: float = 0.0
    kill_switch_sensitivity_multiplier: float = 1.0
    synthetic_stress_intensity: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class StressKernelReport:
    generated_at: str
    history_points: int
    trajectory_window: int
    trajectory_novelty_score: float
    trajectory_novelty_level: str
    transition_stress_score: float
    micro_collapse_probability: float
    execution_friction_score: float
    partial_fill_cascade_score: float
    latency_jitter_score: float
    latency_p99_ms: float
    latency_p999_ms: float
    rejection_cluster_score: float
    liquidity_vacuum_score: float
    correlation_collapse_score: float
    kill_switch_efficiency: float
    kill_switch_event_count: int
    average_detection_to_decision_ms: float
    average_decision_to_protection_ms: float
    false_positive_rate: float
    missed_halt_count: int
    continuous_pressure_score: float
    pressure_level: str
    predicted_pressure_failure: bool
    source_generated_at: str = ""
    probation_live_policy: ProbationLivePolicy = field(default_factory=ProbationLivePolicy)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_kill_switch_event(
    path: Path,
    *,
    trigger: str,
    action: str,
    requires_protection: bool,
    protection_applied: bool,
    detection_to_decision_ms: float,
    decision_to_protection_ms: float,
    pressure_score: float = 0.0,
    false_positive: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": _now_iso(),
        "trigger": trigger,
        "action": action,
        "requires_protection": bool(requires_protection),
        "protection_applied": bool(protection_applied),
        "detection_to_decision_ms": float(detection_to_decision_ms),
        "decision_to_protection_ms": float(decision_to_protection_ms),
        "pressure_score": float(pressure_score),
        "false_positive": bool(false_positive),
        "metadata": dict(metadata or {}),
    }
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _current_market_snapshot(base_dir: Path) -> dict[str, Any]:
    snapshot = _load_json(base_dir / "market_snapshot.json", {})
    if not isinstance(snapshot, dict):
        return {}
    return snapshot


def _current_feature_vector(base_dir: Path) -> dict[str, float]:
    snapshot = _current_market_snapshot(base_dir)
    history_rows = _load_jsonl(base_dir / "regime_novelty_history.jsonl")
    if history_rows and isinstance(history_rows[-1].get("features"), dict):
        return dict(history_rows[-1]["features"])
    return {}


def _source_snapshot_timestamp(base_dir: Path) -> str:
    snapshot = _current_market_snapshot(base_dir)
    timestamp = snapshot.get("_timestamp")
    return str(timestamp) if isinstance(timestamp, str) and timestamp else ""


def _trajectory_metrics(
    history_rows: list[dict[str, Any]],
    thresholds: StreamingStressKernelThresholds,
) -> tuple[int, float, str, float, list[str]]:
    usable = [
        row.get("features")
        for row in history_rows
        if isinstance(row, dict) and isinstance(row.get("features"), dict)
    ]
    history_points = len(usable)
    window = thresholds.trajectory_window
    if history_points < max(thresholds.min_history_points, window * 3):
        return (
            history_points,
            100.0,
            "insufficient_history",
            100.0,
            [
                f"Sequence baseline has only {history_points} snapshots; require {max(thresholds.min_history_points, window * 3)} for trajectory continuity."
            ],
        )

    keys = sorted(usable[-1])
    matrix = np.asarray([[float(row.get(key, 0.0)) for key in keys] for row in usable], dtype=float)
    scale = np.std(matrix, axis=0)
    scale = np.maximum(scale, np.maximum(np.abs(np.median(matrix, axis=0)) * 0.10, 0.05))

    current = matrix[-window:]
    prior_limit = history_points - window
    candidate_distances: list[float] = []
    baseline_distances: list[float] = []
    for start in range(0, prior_limit - window + 1):
        candidate = matrix[start : start + window]
        candidate_distances.append(float(np.mean(np.abs((current - candidate) / scale))))
    for start in range(0, history_points - 2 * window):
        left = matrix[start : start + window]
        right = matrix[start + 1 : start + 1 + window]
        baseline_distances.append(float(np.mean(np.abs((left - right) / scale))))

    best_distance = min(candidate_distances) if candidate_distances else 0.0
    base_p50 = _quantile(baseline_distances, 0.50)
    base_p90 = max(_quantile(baseline_distances, 0.90), base_p50 + 1e-6)
    trajectory_score = 100.0 * _clip01((best_distance - base_p50) / (base_p90 - base_p50))

    diff_matrix = np.diff(matrix, axis=0)
    current_delta = diff_matrix[-(window - 1) :]
    delta_candidates: list[float] = []
    delta_baselines: list[float] = []
    for start in range(0, len(diff_matrix) - 2 * (window - 1) + 1):
        candidate = diff_matrix[start : start + (window - 1)]
        delta_candidates.append(float(np.mean(np.abs(candidate - current_delta))))
    for start in range(0, len(diff_matrix) - 2 * (window - 1)):
        left = diff_matrix[start : start + (window - 1)]
        right = diff_matrix[start + 1 : start + 1 + (window - 1)]
        delta_baselines.append(float(np.mean(np.abs(left - right))))

    best_delta = min(delta_candidates) if delta_candidates else 0.0
    delta_p50 = _quantile(delta_baselines, 0.50)
    delta_p90 = max(_quantile(delta_baselines, 0.90), delta_p50 + 1e-6)
    transition_score = 100.0 * _clip01((best_delta - delta_p50) / (delta_p90 - delta_p50))

    if trajectory_score >= 80.0:
        level = "critical"
    elif trajectory_score >= thresholds.max_trajectory_novelty_score:
        level = "high"
    elif trajectory_score >= 35.0:
        level = "moderate"
    else:
        level = "low"

    reasons: list[str] = []
    if trajectory_score >= thresholds.max_trajectory_novelty_score:
        reasons.append(
            f"State trajectory novelty {trajectory_score:.1f}/100 exceeds the continuity threshold."
        )
    if transition_score >= 55.0:
        reasons.append(
            f"Transition stress {transition_score:.1f}/100 suggests the current regime path is unfolding in an unfamiliar way."
        )
    return history_points, trajectory_score, level, transition_score, reasons


def _execution_friction_metrics(
    base_dir: Path,
    thresholds: StreamingStressKernelThresholds,
) -> tuple[float, float, float, float, float, float, float, list[str]]:
    journal = _load_json(base_dir / "trade_journal.json", [])
    if not isinstance(journal, list):
        journal = []
    divergence_rows = _load_json(base_dir / "divergence_log.json", [])
    if isinstance(divergence_rows, dict):
        divergence_rows = divergence_rows.get("comparisons", [])
    if not isinstance(divergence_rows, list):
        divergence_rows = []

    recent_trades = [row for row in journal if isinstance(row, dict)][-64:]
    fill_ratios = [
        _clip01(
            _float(
                row.get("fill_ratio"),
                _float(row.get("filled_size_usd")) / max(_float(row.get("requested_size_usd"), _float(row.get("size_usd"), 1.0)), 1e-9),
            )
        )
        for row in recent_trades
    ]
    partial_flags = [1 if ratio < 0.85 else 0 for ratio in fill_ratios]
    longest_partial_run = 0
    current_partial_run = 0
    for flag in partial_flags:
        if flag:
            current_partial_run += 1
            longest_partial_run = max(longest_partial_run, current_partial_run)
        else:
            current_partial_run = 0
    fill_shortfall = _mean([1.0 - ratio for ratio in fill_ratios])
    partial_fill_cascade_score = 100.0 * _clip01(fill_shortfall / 0.20 + longest_partial_run / 4.0 * 0.5)

    latencies = [
        max(_float(row.get("entry_execution_ms")), _float(row.get("exit_execution_ms")))
        for row in recent_trades
        if max(_float(row.get("entry_execution_ms")), _float(row.get("exit_execution_ms"))) > 0.0
    ]
    latency_p99 = _quantile(latencies, 0.99)
    latency_p999 = _quantile(latencies, 0.999)
    latency_jitter_score = 100.0 * max(
        _clip01(latency_p99 / thresholds.max_latency_p99_ms),
        _clip01(latency_p999 / thresholds.max_latency_p999_ms),
    )

    divergence_tail = [row for row in divergence_rows if isinstance(row, dict)][-64:]
    miss_flags = [1 if bool(row.get("missed")) else 0 for row in divergence_tail]
    longest_miss_run = 0
    current_miss_run = 0
    for flag in miss_flags:
        if flag:
            current_miss_run += 1
            longest_miss_run = max(longest_miss_run, current_miss_run)
        else:
            current_miss_run = 0
    miss_rate = sum(miss_flags) / len(miss_flags) if miss_flags else 0.0
    rejection_cluster_score = 100.0 * _clip01(max(miss_rate / 0.20, longest_miss_run / 4.0))

    spreads = [abs(_float(row.get("book_spread_bps"))) for row in recent_trades if row.get("book_spread_bps") is not None]
    impacts = [abs(_float(row.get("book_impact_bps"))) for row in recent_trades if row.get("book_impact_bps") is not None]
    liquidity_vacuum_score = 100.0 * max(
        _clip01(_mean(spreads) / 8.0),
        _clip01(_mean(impacts) / 6.0),
        _clip01((1.0 - _mean(fill_ratios or [1.0])) / 0.25),
    )

    components = {
        "partial": partial_fill_cascade_score / 100.0,
        "latency": latency_jitter_score / 100.0,
        "rejection": rejection_cluster_score / 100.0,
        "liquidity": liquidity_vacuum_score / 100.0,
    }
    weights = {"partial": 1.0, "latency": 1.2, "rejection": 1.0, "liquidity": 0.8}
    total_weight = sum(weights.values())
    friction_score = 100.0 * sum(components[name] * weight for name, weight in weights.items()) / total_weight

    reasons: list[str] = []
    if friction_score >= thresholds.max_execution_friction_score:
        reasons.append(
            f"Execution friction score {friction_score:.1f}/100 exceeds the continuity threshold."
        )
    if latency_p999 > thresholds.max_latency_p999_ms:
        reasons.append(
            f"Worst-tail latency {latency_p999:.0f}ms exceeds the {thresholds.max_latency_p999_ms:.0f}ms budget."
        )
    if rejection_cluster_score >= 55.0:
        reasons.append("Order rejections are clustering instead of staying isolated.")
    if partial_fill_cascade_score >= 55.0:
        reasons.append("Partial fills are chaining into fill-cascade behaviour.")

    return (
        friction_score,
        partial_fill_cascade_score,
        latency_jitter_score,
        latency_p99,
        latency_p999,
        rejection_cluster_score,
        liquidity_vacuum_score,
        reasons,
    )


def _correlation_collapse_score(current_features: dict[str, float]) -> float:
    mean_abs_corr = _float(current_features.get("mean_abs_corr_48h"))
    corr_shift = _float(current_features.get("corr_shift_48h"))
    dispersion = _float(current_features.get("dispersion_24h"))
    score = max(
        _clip01((mean_abs_corr - 0.65) / 0.20),
        _clip01(corr_shift / 0.30),
        _clip01(dispersion / 0.05),
    )
    return 100.0 * score


def _kill_switch_metrics(base_dir: Path) -> tuple[float, int, float, float, float, int, list[str]]:
    events = [row for row in _load_jsonl(base_dir / "kill_switch_telemetry.jsonl") if isinstance(row, dict)]
    event_count = len(events)
    if event_count == 0:
        return 1.0, 0, 0.0, 0.0, 0.0, 0, ["Kill-switch telemetry has no protection events yet; probation stage will stay capped."]

    true_positive = 0
    false_negative = 0
    false_positive = 0
    detection_lags: list[float] = []
    protection_lags: list[float] = []
    for event in events:
        requires = bool(event.get("requires_protection"))
        applied = bool(event.get("protection_applied"))
        if requires and applied:
            true_positive += 1
        elif requires and not applied:
            false_negative += 1
        elif not requires and applied:
            false_positive += 1
        if event.get("detection_to_decision_ms") is not None:
            detection_lags.append(_float(event.get("detection_to_decision_ms")))
        if event.get("decision_to_protection_ms") is not None:
            protection_lags.append(_float(event.get("decision_to_protection_ms")))

    efficiency = true_positive / max(true_positive + false_negative, 1)
    false_positive_rate = false_positive / max(event_count, 1)
    avg_detection = _mean(detection_lags)
    avg_protection = _mean(protection_lags)
    reasons: list[str] = []
    if efficiency < 0.80:
        reasons.append(f"Kill-switch efficiency is only {efficiency:.0%}.")
    if false_positive_rate > 0.20:
        reasons.append(f"Kill-switch false-positive rate is {false_positive_rate:.0%}.")
    return efficiency, event_count, avg_detection, avg_protection, false_positive_rate, false_negative, reasons


def _build_probation_live_policy(
    *,
    current_green_snapshot: bool,
    continuous_pressure_score: float,
    trajectory_novelty_score: float,
    execution_friction_score: float,
    kill_switch_efficiency: float,
    kill_switch_event_count: int,
    predicted_pressure_failure: bool,
) -> ProbationLivePolicy:
    reasons: list[str] = []
    if predicted_pressure_failure:
        return ProbationLivePolicy(
            stage="blocked",
            entry_action="halt",
            reasons=["Continuous pressure kernel predicts a live protection failure under coupled stress."],
        )

    if not current_green_snapshot:
        return ProbationLivePolicy(
            stage="shadow",
            entry_action="pause_entries",
            reasons=["Core production snapshot is not green enough for probationary capital exposure."],
        )

    if kill_switch_event_count == 0:
        reasons.append("No real protection events have been recorded yet, so probation stays capped.")

    if continuous_pressure_score < 20.0 and trajectory_novelty_score < 20.0 and execution_friction_score < 20.0 and kill_switch_efficiency >= 0.90 and kill_switch_event_count > 0:
        return ProbationLivePolicy(
            stage="micro_live_ready",
            entry_action="allow",
            allow_probation_live=True,
            allow_full_live=True,
            max_capital_fraction=0.010,
            max_total_exposure_pct=0.010,
            max_per_trade_pct=0.0025,
            entry_size_multiplier=1.0,
            exposure_multiplier=1.0,
            kill_switch_sensitivity_multiplier=1.00,
            synthetic_stress_intensity=1.00,
            reasons=["Continuous pressure is low enough for the first micro-live tranche."],
        )

    if continuous_pressure_score < 30.0 and trajectory_novelty_score < 35.0 and execution_friction_score < 35.0:
        return ProbationLivePolicy(
            stage="plm_0.50",
            entry_action="allow",
            allow_probation_live=True,
            max_capital_fraction=0.005,
            max_total_exposure_pct=0.005,
            max_per_trade_pct=0.0010,
            entry_size_multiplier=0.50,
            exposure_multiplier=0.50,
            kill_switch_sensitivity_multiplier=1.20,
            synthetic_stress_intensity=1.10,
            reasons=reasons + ["Continuous pressure supports the 0.50% probation-live tier."],
        )

    if continuous_pressure_score < 40.0:
        return ProbationLivePolicy(
            stage="plm_0.10",
            entry_action="allow",
            allow_probation_live=True,
            max_capital_fraction=0.001,
            max_total_exposure_pct=0.001,
            max_per_trade_pct=0.00025,
            entry_size_multiplier=0.25,
            exposure_multiplier=0.25,
            kill_switch_sensitivity_multiplier=1.30,
            synthetic_stress_intensity=1.15,
            reasons=reasons + ["Continuous pressure only supports a 0.10% probation-live tranche."],
        )

    if continuous_pressure_score < 50.0:
        return ProbationLivePolicy(
            stage="plm_0.05",
            entry_action="allow",
            allow_probation_live=True,
            max_capital_fraction=0.0005,
            max_total_exposure_pct=0.0005,
            max_per_trade_pct=0.000125,
            entry_size_multiplier=0.15,
            exposure_multiplier=0.15,
            kill_switch_sensitivity_multiplier=1.40,
            synthetic_stress_intensity=1.20,
            reasons=reasons + ["Only the smallest probation-live tier is justified under the current pressure field."],
        )

    return ProbationLivePolicy(
        stage="shadow",
        entry_action="pause_entries",
        reasons=["Continuous pressure remains too high for probationary real-money exposure."],
    )


def build_streaming_stress_kernel_report(
    base_dir: Path,
    thresholds: StreamingStressKernelThresholds | None = None,
) -> StressKernelReport:
    thresholds = thresholds or StreamingStressKernelThresholds()
    generated_at = _now_iso()
    source_generated_at = _source_snapshot_timestamp(base_dir) or generated_at
    history_rows = _load_jsonl(base_dir / "regime_novelty_history.jsonl")
    current_features = _current_feature_vector(base_dir)
    drift = _load_json(base_dir / "drift_intelligence_status.json", {})
    survivability = _load_json(base_dir / "survivability_status.json", {})

    history_points, trajectory_score, novelty_level, transition_score, trajectory_reasons = _trajectory_metrics(
        history_rows,
        thresholds,
    )
    (
        friction_score,
        partial_fill_score,
        latency_score,
        latency_p99,
        latency_p999,
        rejection_cluster_score,
        liquidity_vacuum_score,
        friction_reasons,
    ) = _execution_friction_metrics(base_dir, thresholds)
    correlation_score = _correlation_collapse_score(current_features)
    drift_score = _float(drift.get("risk_score"))
    survivability_score = _float(survivability.get("survivability_score"))
    kill_efficiency, kill_events, avg_detect_ms, avg_protect_ms, false_positive_rate, missed_halts, kill_reasons = _kill_switch_metrics(base_dir)

    micro_collapse_probability = _clip01(
        0.30 * trajectory_score / 100.0
        + 0.20 * transition_score / 100.0
        + 0.20 * correlation_score / 100.0
        + 0.15 * drift_score / 100.0
        + 0.15 * max(0.0, 1.0 - survivability_score / 100.0)
    )

    components = {
        "trajectory": trajectory_score / 100.0,
        "transition": transition_score / 100.0,
        "micro_collapse": micro_collapse_probability,
        "friction": friction_score / 100.0,
        "latency": latency_score / 100.0,
        "kill_deficit": max(0.0, 1.0 - kill_efficiency),
        "drift": drift_score / 100.0,
        "survivability_gap": max(0.0, 1.0 - survivability_score / 100.0),
    }
    weights = {
        "trajectory": 1.2,
        "transition": 1.0,
        "micro_collapse": 1.2,
        "friction": 1.2,
        "latency": 0.8,
        "kill_deficit": 1.2,
        "drift": 0.8,
        "survivability_gap": 0.6,
    }
    total_weight = sum(weights.values())
    pressure_score = 100.0 * sum(components[name] * weight for name, weight in weights.items()) / total_weight

    if pressure_score >= 75.0:
        pressure_level = "critical"
    elif pressure_score >= thresholds.max_continuous_pressure_score:
        pressure_level = "high"
    elif pressure_score >= 35.0:
        pressure_level = "moderate"
    else:
        pressure_level = "low"

    reasons = list(trajectory_reasons) + list(friction_reasons) + list(kill_reasons)
    if correlation_score >= 55.0:
        reasons.append("Cross-asset correlation is collapsing into a stress cluster.")
    if micro_collapse_probability > thresholds.max_micro_collapse_probability:
        reasons.append(
            f"Micro-collapse probability {micro_collapse_probability:.0%} exceeds the tolerance band."
        )

    predicted_pressure_failure = (
        history_points < thresholds.min_history_points
        or trajectory_score > thresholds.max_trajectory_novelty_score
        or friction_score > thresholds.max_execution_friction_score
        or pressure_score > thresholds.max_continuous_pressure_score
        or micro_collapse_probability > thresholds.max_micro_collapse_probability
        or latency_p999 > thresholds.max_latency_p999_ms
        or kill_efficiency < thresholds.min_kill_switch_efficiency
    )

    policy = _build_probation_live_policy(
        current_green_snapshot=bool(drift.get("current_green_snapshot")),
        continuous_pressure_score=pressure_score,
        trajectory_novelty_score=trajectory_score,
        execution_friction_score=friction_score,
        kill_switch_efficiency=kill_efficiency,
        kill_switch_event_count=kill_events,
        predicted_pressure_failure=predicted_pressure_failure,
    )

    return StressKernelReport(
        generated_at=generated_at,
        history_points=history_points,
        trajectory_window=thresholds.trajectory_window,
        trajectory_novelty_score=float(trajectory_score),
        trajectory_novelty_level=novelty_level,
        transition_stress_score=float(transition_score),
        micro_collapse_probability=float(micro_collapse_probability),
        execution_friction_score=float(friction_score),
        partial_fill_cascade_score=float(partial_fill_score),
        latency_jitter_score=float(latency_score),
        latency_p99_ms=float(latency_p99),
        latency_p999_ms=float(latency_p999),
        rejection_cluster_score=float(rejection_cluster_score),
        liquidity_vacuum_score=float(liquidity_vacuum_score),
        correlation_collapse_score=float(correlation_score),
        kill_switch_efficiency=float(kill_efficiency),
        kill_switch_event_count=int(kill_events),
        average_detection_to_decision_ms=float(avg_detect_ms),
        average_decision_to_protection_ms=float(avg_protect_ms),
        false_positive_rate=float(false_positive_rate),
        missed_halt_count=int(missed_halts),
        continuous_pressure_score=float(pressure_score),
        pressure_level=pressure_level,
        predicted_pressure_failure=predicted_pressure_failure,
        source_generated_at=source_generated_at,
        probation_live_policy=policy,
        reasons=reasons,
    )


def write_streaming_stress_kernel_report(report: StressKernelReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))


def format_streaming_stress_kernel_report(report: StressKernelReport) -> str:
    lines = [
        "Streaming Stress Kernel Report",
        f"  Observed snapshot: {report.source_generated_at or report.generated_at}",
        f"  Pressure: {report.continuous_pressure_score:.1f}/100 ({report.pressure_level})",
        f"  Trajectory novelty: {report.trajectory_novelty_score:.1f}/100 ({report.trajectory_novelty_level}) | transition={report.transition_stress_score:.1f}",
        f"  Execution friction: {report.execution_friction_score:.1f}/100 | partial={report.partial_fill_cascade_score:.1f} | rejects={report.rejection_cluster_score:.1f} | liquidity={report.liquidity_vacuum_score:.1f}",
        f"  Latency: p99={report.latency_p99_ms:.0f}ms | p999={report.latency_p999_ms:.0f}ms | jitter={report.latency_jitter_score:.1f}",
        f"  Kill switch: efficiency={report.kill_switch_efficiency:.0%} | events={report.kill_switch_event_count} | detect={report.average_detection_to_decision_ms:.0f}ms | protect={report.average_decision_to_protection_ms:.0f}ms",
        f"  PLM stage: {report.probation_live_policy.stage} | entry action={report.probation_live_policy.entry_action}",
    ]
    if report.probation_live_policy.allow_probation_live or report.probation_live_policy.allow_full_live:
        lines.append(
            f"  Caps: capital={report.probation_live_policy.max_capital_fraction:.2%} | exposure={report.probation_live_policy.max_total_exposure_pct:.2%} | per-trade={report.probation_live_policy.max_per_trade_pct:.2%}"
        )
    if report.reasons:
        lines.append("  Kernel reasons:")
        for reason in report.reasons:
            lines.append(f"    - {reason}")
    if report.probation_live_policy.reasons:
        lines.append("  Policy rationale:")
        for reason in report.probation_live_policy.reasons:
            lines.append(f"    - {reason}")
    return "\n".join(lines)


__all__ = [
    "ProbationLivePolicy",
    "StressKernelReport",
    "StreamingStressKernelThresholds",
    "build_streaming_stress_kernel_report",
    "format_streaming_stress_kernel_report",
    "record_kill_switch_event",
    "write_streaming_stress_kernel_report",
]
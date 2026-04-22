from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.fund.health import HealthMonitor


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _quantile(values: list[float], quantile: float) -> float:
    return float(np.quantile(values, quantile)) if values else 0.0


def _regime_stress_value(regime: str) -> float:
    mapping = {
        "bull_trend": 0.20,
        "sideways": 0.35,
        "bear_trend": 0.65,
        "high_volatility": 1.00,
    }
    return mapping.get(str(regime), 0.50)


@dataclass
class SurvivabilityThresholds:
    min_regime_history_points: int = 72
    max_regime_novelty_score: float = 55.0
    max_execution_stress_score: float = 55.0
    min_scenario_pass_rate: float = 0.60
    max_worst_case_drag_bps: float = 65.0
    min_worst_case_fill_ratio: float = 0.55
    max_rejection_rate: float = 0.25
    max_capital_drag_pct: float = 0.0075
    halt_latency_budget_ms: float = 500.0
    min_survivability_score: float = 65.0


@dataclass
class StressScenarioResult:
    name: str
    additional_drag_bps: float
    fill_ratio: float
    rejection_rate: float
    capital_drag_pct: float
    halt_triggered: bool
    halt_latency_ms: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class ExposureLadderStep:
    stage: str = "shadow"
    max_capital_fraction: float = 0.0
    max_total_exposure_pct: float = 0.0
    max_per_trade_pct: float = 0.0
    requires_manual_approval: bool = True
    reasons: list[str] = field(default_factory=list)


@dataclass
class SurvivabilityReport:
    generated_at: str
    history_points: int
    regime_novelty_score: float
    regime_novelty_level: str
    regime_shift_detected: bool
    execution_stress_score: float
    scenario_pass_rate: float
    scenario_count: int
    worst_case_additional_drag_bps: float
    worst_case_fill_ratio: float
    worst_case_rejection_rate: float
    worst_case_capital_drag_pct: float
    halt_latency_budget_ms: float
    halt_latency_median_ms: float
    halt_latency_p95_ms: float
    halt_latency_breached: bool
    survivability_score: float
    survivability_level: str
    predicted_survivability_failure: bool
    exposure_ladder: ExposureLadderStep = field(default_factory=ExposureLadderStep)
    scenarios: list[StressScenarioResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _market_snapshot_vector(snapshot: dict[str, Any]) -> dict[str, float]:
    asset_rows = [
        row
        for symbol, row in snapshot.items()
        if not str(symbol).startswith("_") and isinstance(row, dict) and not row.get("error")
    ]
    if not asset_rows:
        return {}

    abs_funding = [abs(_float(row.get("funding_zscore"))) for row in asset_rows]
    vol_ratios = [max(_float(row.get("vol_ratio"), 1.0), 0.0) for row in asset_rows]
    atr_expansions = [max(_float(row.get("atr_exp"), 1.0), 0.0) for row in asset_rows]
    bb_pctiles = [_clip01(_float(row.get("bb_pctile"), 50.0) / 100.0) for row in asset_rows]
    breakout_pressures: list[float] = []
    regimes: list[str] = []
    regime_stress: list[float] = []

    for row in asset_rows:
        price = _float(row.get("price"))
        ch_high = _float(row.get("ch_high"))
        ch_low = _float(row.get("ch_low"))
        if price > 0.0 and ch_high > ch_low:
            half_range = max((ch_high - ch_low) / 2.0, 1e-9)
            midpoint = (ch_high + ch_low) / 2.0
            breakout_pressures.append(abs(price - midpoint) / half_range)
        else:
            breakout_pressures.append(0.0)
        regime = str(row.get("regime", "unknown"))
        regimes.append(regime)
        regime_stress.append(_regime_stress_value(regime))

    cross_asset = snapshot.get("_cross_asset") if isinstance(snapshot.get("_cross_asset"), dict) else {}
    mean_abs_corr = _float(cross_asset.get("mean_abs_corr_48h"))
    corr_shift = _float(cross_asset.get("corr_shift_48h"))
    dispersion_24h = _float(cross_asset.get("dispersion_24h"))

    return {
        "mean_abs_funding_z": _mean(abs_funding),
        "max_abs_funding_z": max(abs_funding) if abs_funding else 0.0,
        "mean_vol_ratio": _mean(vol_ratios),
        "max_vol_ratio": max(vol_ratios) if vol_ratios else 0.0,
        "mean_atr_exp": _mean(atr_expansions),
        "max_atr_exp": max(atr_expansions) if atr_expansions else 0.0,
        "mean_bb_pctile": _mean(bb_pctiles),
        "min_bb_pctile": min(bb_pctiles) if bb_pctiles else 0.0,
        "mean_breakout_pressure": _mean(breakout_pressures),
        "max_breakout_pressure": max(breakout_pressures) if breakout_pressures else 0.0,
        "regime_stress": _mean(regime_stress),
        "regime_dispersion": len(set(regimes)) / max(len(regimes), 1),
        "mean_abs_corr_48h": mean_abs_corr,
        "corr_shift_48h": corr_shift,
        "dispersion_24h": dispersion_24h,
    }


def append_market_snapshot_history(base_dir: Path, snapshot: dict[str, Any] | None = None) -> bool:
    payload = snapshot if snapshot is not None else _load_json(base_dir / "market_snapshot.json", {})
    if not isinstance(payload, dict):
        return False
    features = _market_snapshot_vector(payload)
    if not features:
        return False

    ts = str(payload.get("_timestamp", _now_iso()))
    history_path = base_dir / "regime_novelty_history.jsonl"
    rows = _load_jsonl(history_path)
    if rows:
        last_ts = str(rows[-1].get("timestamp", ""))
        last_features = rows[-1].get("features") if isinstance(rows[-1].get("features"), dict) else {}
        if last_ts == ts and last_features == features:
            return False

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a") as handle:
        handle.write(json.dumps({"timestamp": ts, "features": features}) + "\n")
    return True


def _compute_regime_novelty(
    history_rows: list[dict[str, Any]],
    current_features: dict[str, float],
    thresholds: SurvivabilityThresholds,
) -> tuple[int, float, str, bool, list[str]]:
    if not current_features:
        return 0, 100.0, "insufficient_history", True, ["Current market snapshot is unavailable, so regime novelty cannot be assessed."]

    usable = [
        row.get("features")
        for row in history_rows
        if isinstance(row, dict) and isinstance(row.get("features"), dict)
    ]
    history_points = len(usable)
    if history_points < thresholds.min_regime_history_points:
        return (
            history_points,
            100.0,
            "insufficient_history",
            True,
            [
                f"Regime novelty baseline has only {history_points} snapshots; require {thresholds.min_regime_history_points} before capital exposure."
            ],
        )

    keys = sorted(current_features)
    matrix = np.asarray([[float(row.get(key, 0.0)) for key in keys] for row in usable], dtype=float)
    current = np.asarray([float(current_features.get(key, 0.0)) for key in keys], dtype=float)
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    scale = mad + np.maximum(np.abs(median) * 0.10, 0.10)
    robust_z = np.abs(current - median) / scale
    robust_z = np.clip(robust_z, 0.0, 10.0)
    top_count = max(3, len(keys) // 3)
    top_scores = np.sort(robust_z)[-top_count:]
    novelty_score = min(100.0, float(np.mean(top_scores) / 5.0 * 100.0))

    if novelty_score >= 80.0:
        level = "critical"
    elif novelty_score >= thresholds.max_regime_novelty_score:
        level = "high"
    elif novelty_score >= 35.0:
        level = "moderate"
    else:
        level = "low"

    regime_shift_detected = level in {"high", "critical"}
    reasons: list[str] = []
    key_to_score = dict(zip(keys, robust_z.tolist(), strict=False))
    for key, score in sorted(key_to_score.items(), key=lambda item: item[1], reverse=True)[:3]:
        if score >= 3.0:
            reasons.append(f"{key} is {score:.1f} robust-sigma away from the recent regime baseline.")
    if not reasons and regime_shift_detected:
        reasons.append("Current market state sits outside the recent regime envelope.")
    return history_points, novelty_score, level, regime_shift_detected, reasons


def _execution_baseline(base_dir: Path) -> dict[str, float]:
    journal = _load_json(base_dir / "trade_journal.json", [])
    if not isinstance(journal, list):
        journal = []
    divergence_rows = _load_json(base_dir / "divergence_log.json", [])
    if isinstance(divergence_rows, dict):
        divergence_rows = divergence_rows.get("comparisons", [])
    if not isinstance(divergence_rows, list):
        divergence_rows = []

    recent_trades = [row for row in journal if isinstance(row, dict)][-48:]
    executed_divergence = [row for row in divergence_rows if isinstance(row, dict) and not bool(row.get("missed"))][-48:]
    missed_divergence = [row for row in divergence_rows if isinstance(row, dict) and bool(row.get("missed"))][-48:]

    entry_slip = [abs(_float(row.get("entry_slippage_bps"))) for row in recent_trades if row.get("entry_slippage_bps") is not None]
    exit_slip = [abs(_float(row.get("exit_slippage_bps"))) for row in recent_trades if row.get("exit_slippage_bps") is not None]
    book_spread = [abs(_float(row.get("book_spread_bps"))) for row in recent_trades if row.get("book_spread_bps") is not None]
    book_impact = [abs(_float(row.get("book_impact_bps"))) for row in recent_trades if row.get("book_impact_bps") is not None]
    fill_ratios = [
        _clip01(
            _float(
                row.get("fill_ratio"),
                _float(row.get("filled_size_usd")) / max(_float(row.get("requested_size_usd"), _float(row.get("size_usd"), 1.0)), 1e-9),
            )
        )
        for row in recent_trades
    ]
    execution_ms = [
        max(_float(row.get("entry_execution_ms")), _float(row.get("exit_execution_ms")))
        for row in recent_trades
    ]
    size_usd = [max(_float(row.get("filled_size_usd"), _float(row.get("size_usd"))), 0.0) for row in recent_trades]

    if not entry_slip and executed_divergence:
        entry_slip = [abs(_float(row.get("entry_slippage_bps"))) for row in executed_divergence]
    if not exit_slip and executed_divergence:
        exit_slip = [abs(_float(row.get("exit_slippage_bps"))) for row in executed_divergence if row.get("exit_slippage_bps") is not None]

    rejection_rate = len(missed_divergence) / max(len(executed_divergence) + len(missed_divergence), 1)

    return {
        "sample_count": float(max(len(recent_trades), len(executed_divergence))),
        "baseline_drag_bps": _mean(entry_slip) + _mean(exit_slip) + 0.5 * _mean(book_spread),
        "baseline_impact_bps": _mean(book_impact),
        "baseline_fill_ratio": _mean(fill_ratios) if fill_ratios else 1.0,
        "baseline_rejection_rate": rejection_rate,
        "baseline_execution_ms": _mean(execution_ms),
        "avg_size_usd": _mean(size_usd) if size_usd else 0.0,
    }


def _measure_halt_latency(
    base_dir: Path,
    *,
    observation_latency_ms: float,
    stress_slippage_bps: float,
    rejection_rate: float,
) -> tuple[bool, float]:
    monitor = HealthMonitor(
        max_execution_slippage_pct=0.0035,
        critical_slippage_pct=0.0075,
        max_consecutive_errors=1,
        health_report_path=str(base_dir / ".tmp_survivability_health.json"),
        heartbeat_timeout_seconds=3600,
    )
    monitor.heartbeat()
    start = time.perf_counter()
    actual_price = 100.0 * (1.0 + max(stress_slippage_bps / 1e4, 0.0125))
    rejected = rejection_rate >= 0.20
    monitor.record_execution(
        "STRESS/USDT",
        expected_price=100.0,
        actual_price=actual_price,
        success=not rejected,
        error="stress rejection" if rejected else "",
    )
    health = monitor.check_health()
    decision_ms = (time.perf_counter() - start) * 1000.0
    effective_latency_ms = decision_ms + observation_latency_ms
    return bool(health.should_halt), effective_latency_ms


def _build_stress_scenarios(
    base_dir: Path,
    *,
    baseline: dict[str, float],
    snapshot_features: dict[str, float],
    thresholds: SurvivabilityThresholds,
) -> list[StressScenarioResult]:
    sample_count = int(baseline.get("sample_count", 0.0))
    if sample_count <= 0:
        return []

    atr_stress = max(snapshot_features.get("max_atr_exp", 1.0), 1.0)
    vol_stress = max(snapshot_features.get("max_vol_ratio", 1.0), 1.0)
    severity = min(1.75, max(0.65, 0.50 * atr_stress + 0.15 * vol_stress))
    latency_scale = min(1.50, max(0.75, 0.55 + 0.25 * atr_stress + 0.05 * max(vol_stress - 1.0, 0.0)))
    baseline_drag_bps = max(baseline.get("baseline_drag_bps", 0.0), 4.0)
    baseline_impact_bps = max(baseline.get("baseline_impact_bps", 0.0), 2.0)
    baseline_fill_ratio = _clip01(baseline.get("baseline_fill_ratio", 1.0))
    baseline_rejection_rate = _clip01(baseline.get("baseline_rejection_rate", 0.0))

    configs = [
        {"name": "spread_shock", "spread_mult": 2.4, "impact_mult": 1.3, "latency_ms": 120.0, "rejection_prob": 0.03, "fill_ceiling": 0.90},
        {"name": "partial_fill_cascade", "spread_mult": 1.9, "impact_mult": 2.1, "latency_ms": 190.0, "rejection_prob": 0.08, "fill_ceiling": 0.60},
        {"name": "api_throttle_cluster", "spread_mult": 1.6, "impact_mult": 1.4, "latency_ms": 360.0, "rejection_prob": 0.18, "fill_ceiling": 0.72},
        {"name": "rejection_burst", "spread_mult": 1.4, "impact_mult": 1.2, "latency_ms": 260.0, "rejection_prob": 0.30, "fill_ceiling": 0.68},
        {"name": "adversarial_overlap", "spread_mult": 2.8, "impact_mult": 2.5, "latency_ms": 470.0, "rejection_prob": 0.14, "fill_ceiling": 0.45},
    ]

    scenarios: list[StressScenarioResult] = []
    for config in configs:
        spread_mult = 1.0 + max(config["spread_mult"] - 1.0, 0.0) * severity
        impact_mult = 1.0 + max(config["impact_mult"] - 1.0, 0.0) * severity
        latency_ms = float(config["latency_ms"]) * latency_scale
        rejection_prob = _clip01(config["rejection_prob"] * severity)
        fill_ceiling = max(0.05, min(1.0, 1.0 - (1.0 - float(config["fill_ceiling"])) * severity))

        latency_penalty_bps = (latency_ms / 100.0) * max(0.6, atr_stress * 0.8)
        spread_penalty_bps = baseline_drag_bps * max(spread_mult - 1.0, 0.0)
        impact_penalty_bps = baseline_impact_bps * impact_mult + max(vol_stress - 1.0, 0.0) * 3.0
        rejection_penalty_bps = rejection_prob * max(12.0, baseline_drag_bps + 6.0)
        additional_drag_bps = spread_penalty_bps + impact_penalty_bps + latency_penalty_bps + rejection_penalty_bps

        fill_ratio = baseline_fill_ratio * fill_ceiling - rejection_prob * 0.12
        fill_ratio = max(0.05, min(fill_ratio, fill_ceiling))
        rejection_rate = _clip01(baseline_rejection_rate + rejection_prob)
        capital_drag_pct = additional_drag_bps / 1e4 * max(fill_ratio, 0.25) + (1.0 - fill_ratio) * 0.003

        halt_triggered, halt_latency_ms = _measure_halt_latency(
            base_dir,
            observation_latency_ms=latency_ms,
            stress_slippage_bps=additional_drag_bps,
            rejection_rate=rejection_rate,
        )

        reasons: list[str] = []
        if additional_drag_bps > thresholds.max_worst_case_drag_bps:
            reasons.append(
                f"execution drag {additional_drag_bps:.1f}bps exceeds {thresholds.max_worst_case_drag_bps:.1f}bps"
            )
        if fill_ratio < thresholds.min_worst_case_fill_ratio:
            reasons.append(
                f"fill ratio {fill_ratio:.0%} drops below {thresholds.min_worst_case_fill_ratio:.0%}"
            )
        if rejection_rate > thresholds.max_rejection_rate:
            reasons.append(
                f"rejection rate {rejection_rate:.0%} exceeds {thresholds.max_rejection_rate:.0%}"
            )
        if capital_drag_pct > thresholds.max_capital_drag_pct:
            reasons.append(
                f"capital drag {capital_drag_pct:.2%} exceeds {thresholds.max_capital_drag_pct:.2%}"
            )
        if not halt_triggered:
            reasons.append("kill switch did not trigger under stressed execution")
        elif halt_latency_ms > thresholds.halt_latency_budget_ms:
            reasons.append(
                f"halt latency {halt_latency_ms:.0f}ms exceeds {thresholds.halt_latency_budget_ms:.0f}ms budget"
            )

        scenarios.append(
            StressScenarioResult(
                name=str(config["name"]),
                additional_drag_bps=float(additional_drag_bps),
                fill_ratio=float(fill_ratio),
                rejection_rate=float(rejection_rate),
                capital_drag_pct=float(capital_drag_pct),
                halt_triggered=halt_triggered,
                halt_latency_ms=float(halt_latency_ms),
                passed=not reasons,
                reasons=reasons,
            )
        )
    return scenarios


def _execution_stress_metrics(
    scenarios: list[StressScenarioResult],
    thresholds: SurvivabilityThresholds,
) -> tuple[float, float, float, float, float, float, float, float, bool, list[str]]:
    if not scenarios:
        return 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True, ["Execution stress lab has no recent trade evidence to replay."]

    pass_rate = sum(1 for scenario in scenarios if scenario.passed) / len(scenarios)
    worst_drag = max(scenario.additional_drag_bps for scenario in scenarios)
    worst_fill = min(scenario.fill_ratio for scenario in scenarios)
    worst_rejection = max(scenario.rejection_rate for scenario in scenarios)
    worst_capital_drag = max(scenario.capital_drag_pct for scenario in scenarios)
    latencies = [scenario.halt_latency_ms for scenario in scenarios]
    latency_median = _quantile(latencies, 0.50)
    latency_p95 = _quantile(latencies, 0.95)
    halt_latency_breached = latency_p95 > thresholds.halt_latency_budget_ms

    components = {
        "pass_rate": _clip01(max(thresholds.min_scenario_pass_rate - pass_rate, 0.0) / max(thresholds.min_scenario_pass_rate, 1e-9)),
        "drag": _clip01(worst_drag / thresholds.max_worst_case_drag_bps),
        "fill": _clip01(max(thresholds.min_worst_case_fill_ratio - worst_fill, 0.0) / max(thresholds.min_worst_case_fill_ratio, 1e-9)),
        "rejection": _clip01(worst_rejection / thresholds.max_rejection_rate),
        "capital_drag": _clip01(worst_capital_drag / thresholds.max_capital_drag_pct),
        "latency": _clip01(latency_p95 / thresholds.halt_latency_budget_ms),
    }
    weights = {
        "pass_rate": 1.0,
        "drag": 1.2,
        "fill": 0.8,
        "rejection": 0.8,
        "capital_drag": 1.0,
        "latency": 1.2,
    }
    total_weight = sum(weights.values())
    stress_score = 100.0 * sum(components[name] * weight for name, weight in weights.items()) / total_weight

    reasons: list[str] = []
    if pass_rate < thresholds.min_scenario_pass_rate:
        reasons.append(
            f"Only {pass_rate:.0%} of execution stress scenarios pass; require {thresholds.min_scenario_pass_rate:.0%}."
        )
    if worst_drag > thresholds.max_worst_case_drag_bps:
        reasons.append(
            f"Worst-case execution drag is {worst_drag:.1f}bps, above the {thresholds.max_worst_case_drag_bps:.1f}bps limit."
        )
    if worst_fill < thresholds.min_worst_case_fill_ratio:
        reasons.append(
            f"Worst-case fill ratio falls to {worst_fill:.0%}, below the {thresholds.min_worst_case_fill_ratio:.0%} floor."
        )
    if worst_rejection > thresholds.max_rejection_rate:
        reasons.append(
            f"Worst-case rejection rate reaches {worst_rejection:.0%}, above the {thresholds.max_rejection_rate:.0%} cap."
        )
    if worst_capital_drag > thresholds.max_capital_drag_pct:
        reasons.append(
            f"Worst-case capital drag reaches {worst_capital_drag:.2%}, above the {thresholds.max_capital_drag_pct:.2%} cap."
        )
    if halt_latency_breached:
        reasons.append(
            f"Kill-switch p95 latency is {latency_p95:.0f}ms, above the {thresholds.halt_latency_budget_ms:.0f}ms budget."
        )

    return (
        float(stress_score),
        float(pass_rate),
        float(worst_drag),
        float(worst_fill),
        float(worst_rejection),
        float(worst_capital_drag),
        float(latency_median),
        float(latency_p95),
        halt_latency_breached,
        reasons,
    )


def _build_exposure_ladder(
    *,
    current_green_snapshot: bool,
    drift_risk_score: float,
    novelty_score: float,
    stress_score: float,
    survivability_score: float,
    predicted_failure: bool,
    halt_latency_breached: bool,
) -> ExposureLadderStep:
    if predicted_failure:
        return ExposureLadderStep(
            stage="blocked",
            reasons=["Survivability lab predicts the system will not tolerate unseen stress cleanly."],
        )

    if not current_green_snapshot:
        return ExposureLadderStep(
            stage="shadow",
            reasons=["Observed production snapshot is not yet fully green, so capital must stay in shadow."],
        )

    if halt_latency_breached or drift_risk_score > 55.0 or survivability_score < 65.0:
        return ExposureLadderStep(
            stage="shadow",
            reasons=["Observed drift or halt-latency evidence is still too weak for capital exposure."],
        )

    if survivability_score >= 88.0 and novelty_score < 25.0 and stress_score < 25.0 and drift_risk_score < 25.0:
        return ExposureLadderStep(
            stage="5%",
            max_capital_fraction=0.05,
            max_total_exposure_pct=0.05,
            max_per_trade_pct=0.010,
            reasons=["Stress resilience and regime stability are strong enough for the 5% probation tier."],
        )

    if survivability_score >= 80.0 and novelty_score < 35.0 and stress_score < 35.0 and drift_risk_score < 35.0:
        return ExposureLadderStep(
            stage="1%",
            max_capital_fraction=0.01,
            max_total_exposure_pct=0.01,
            max_per_trade_pct=0.0025,
            reasons=["System is strong enough for a tightly capped 1% live tranche."],
        )

    if survivability_score >= 72.0 and novelty_score < 45.0 and stress_score < 45.0 and drift_risk_score < 45.0:
        return ExposureLadderStep(
            stage="0.5%",
            max_capital_fraction=0.005,
            max_total_exposure_pct=0.005,
            max_per_trade_pct=0.0010,
            reasons=["System clears the minimum stress bar for a 0.5% micro-live probation step."],
        )

    return ExposureLadderStep(
        stage="0.1%",
        max_capital_fraction=0.001,
        max_total_exposure_pct=0.001,
        max_per_trade_pct=0.00025,
        reasons=["Only the smallest experimental tranche is justified under the current novelty and stress envelope."],
    )


def build_survivability_report(
    base_dir: Path,
    thresholds: SurvivabilityThresholds | None = None,
) -> SurvivabilityReport:
    thresholds = thresholds or SurvivabilityThresholds()
    snapshot = _load_json(base_dir / "market_snapshot.json", {})
    if not isinstance(snapshot, dict):
        snapshot = {}
    history_rows = _load_jsonl(base_dir / "regime_novelty_history.jsonl")
    drift = _load_json(base_dir / "drift_intelligence_status.json", {})
    current_features = _market_snapshot_vector(snapshot)

    history_points, novelty_score, novelty_level, regime_shift_detected, novelty_reasons = _compute_regime_novelty(
        history_rows,
        current_features,
        thresholds,
    )

    baseline = _execution_baseline(base_dir)
    scenarios = _build_stress_scenarios(
        base_dir,
        baseline=baseline,
        snapshot_features=current_features,
        thresholds=thresholds,
    )
    (
        execution_stress_score,
        scenario_pass_rate,
        worst_drag,
        worst_fill,
        worst_rejection,
        worst_capital_drag,
        latency_median,
        latency_p95,
        halt_latency_breached,
        stress_reasons,
    ) = _execution_stress_metrics(scenarios, thresholds)

    combined_risk = 0.45 * novelty_score + 0.55 * execution_stress_score
    survivability_score = max(0.0, min(100.0, 100.0 - combined_risk))
    if survivability_score >= 80.0:
        survivability_level = "strong"
    elif survivability_score >= thresholds.min_survivability_score:
        survivability_level = "adequate"
    elif survivability_score >= 45.0:
        survivability_level = "fragile"
    else:
        survivability_level = "critical"

    predicted_failure = (
        history_points < thresholds.min_regime_history_points
        or novelty_score > thresholds.max_regime_novelty_score
        or execution_stress_score > thresholds.max_execution_stress_score
        or scenario_pass_rate < thresholds.min_scenario_pass_rate
        or worst_fill < thresholds.min_worst_case_fill_ratio
        or worst_rejection > thresholds.max_rejection_rate
        or worst_capital_drag > thresholds.max_capital_drag_pct
        or halt_latency_breached
        or survivability_score < thresholds.min_survivability_score
    )

    exposure_ladder = _build_exposure_ladder(
        current_green_snapshot=bool(drift.get("current_green_snapshot")),
        drift_risk_score=_float(drift.get("risk_score")),
        novelty_score=novelty_score,
        stress_score=execution_stress_score,
        survivability_score=survivability_score,
        predicted_failure=predicted_failure,
        halt_latency_breached=halt_latency_breached,
    )

    reasons = list(novelty_reasons) + list(stress_reasons)
    if survivability_score < thresholds.min_survivability_score:
        reasons.append(
            f"Survivability score is {survivability_score:.1f}/100, below the {thresholds.min_survivability_score:.1f} minimum."
        )

    return SurvivabilityReport(
        generated_at=_now_iso(),
        history_points=history_points,
        regime_novelty_score=float(novelty_score),
        regime_novelty_level=novelty_level,
        regime_shift_detected=regime_shift_detected,
        execution_stress_score=float(execution_stress_score),
        scenario_pass_rate=float(scenario_pass_rate),
        scenario_count=len(scenarios),
        worst_case_additional_drag_bps=float(worst_drag),
        worst_case_fill_ratio=float(worst_fill),
        worst_case_rejection_rate=float(worst_rejection),
        worst_case_capital_drag_pct=float(worst_capital_drag),
        halt_latency_budget_ms=float(thresholds.halt_latency_budget_ms),
        halt_latency_median_ms=float(latency_median),
        halt_latency_p95_ms=float(latency_p95),
        halt_latency_breached=halt_latency_breached,
        survivability_score=float(survivability_score),
        survivability_level=survivability_level,
        predicted_survivability_failure=predicted_failure,
        exposure_ladder=exposure_ladder,
        scenarios=scenarios,
        reasons=reasons,
    )


def write_survivability_report(report: SurvivabilityReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))


def format_survivability_report(report: SurvivabilityReport) -> str:
    lines = [
        "Execution Stress & Regime Shock Report",
        f"  Survivability: {report.survivability_score:.1f}/100 ({report.survivability_level})",
        f"  Regime novelty: {report.regime_novelty_score:.1f}/100 ({report.regime_novelty_level}) | shift detected={'YES' if report.regime_shift_detected else 'NO'}",
        f"  Stress replay: {report.execution_stress_score:.1f}/100 | pass rate={report.scenario_pass_rate:.0%} over {report.scenario_count} scenarios",
        f"  Worst case: drag={report.worst_case_additional_drag_bps:.1f}bps | fill={report.worst_case_fill_ratio:.0%} | rejects={report.worst_case_rejection_rate:.0%} | cap drag={report.worst_case_capital_drag_pct:.2%}",
        f"  Halt latency: median={report.halt_latency_median_ms:.0f}ms | p95={report.halt_latency_p95_ms:.0f}ms | budget={report.halt_latency_budget_ms:.0f}ms",
        f"  Exposure ladder: {report.exposure_ladder.stage}",
    ]
    if report.exposure_ladder.stage not in {"shadow", "blocked"}:
        lines.append(
            f"  Recommended caps: capital={report.exposure_ladder.max_capital_fraction:.2%} | exposure={report.exposure_ladder.max_total_exposure_pct:.2%} | per-trade={report.exposure_ladder.max_per_trade_pct:.2%}"
        )
    if report.reasons:
        lines.append("  Survivability reasons:")
        for reason in report.reasons:
            lines.append(f"    - {reason}")
    if report.exposure_ladder.reasons:
        lines.append("  Ladder rationale:")
        for reason in report.exposure_ladder.reasons:
            lines.append(f"    - {reason}")
    return "\n".join(lines)


__all__ = [
    "ExposureLadderStep",
    "StressScenarioResult",
    "SurvivabilityReport",
    "SurvivabilityThresholds",
    "append_market_snapshot_history",
    "build_survivability_report",
    "format_survivability_report",
    "write_survivability_report",
]
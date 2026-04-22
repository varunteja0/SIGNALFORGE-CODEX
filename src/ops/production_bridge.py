from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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


def _as_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    if isinstance(obj, dict):
        return dict(obj)
    return {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BrokerReconciliationIssue:
    severity: str
    code: str
    message: str
    symbol: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerReconciliationReport:
    generated_at: str
    overall_status: str
    critical_issue_count: int
    warning_issue_count: int
    missing_broker_positions: int
    unknown_broker_positions: int
    size_mismatch_count: int
    stale_open_orders: int
    critical_symbols: list[str] = field(default_factory=list)
    issues: list[BrokerReconciliationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = [asdict(issue) for issue in self.issues]
        return payload


@dataclass
class TradeJournalParityReport:
    generated_at: str
    trade_count: int
    total_notional_usd: float
    avg_abs_entry_diff_bps: float
    avg_abs_exit_diff_bps: float
    unexplained_pnl_bps: float
    verdict: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowExecutionReport:
    generated_at: str
    trade_count: int
    compared_trade_count: int
    missing_shadow_count: int
    avg_abs_entry_delta_bps: float
    avg_abs_exit_delta_bps: float
    avg_abs_pnl_delta_pct: float
    max_abs_pnl_delta_pct: float
    ready_for_live: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProductionCertificationThresholds:
    min_consecutive_green_days: float = 30.0
    max_shadow_avg_entry_delta_bps: float = 12.0
    max_shadow_avg_pnl_delta_pct: float = 0.20
    max_parity_unexplained_pnl_bps: float = 6.0
    require_shadow_mode: bool = True
    max_broker_critical_issues: int = 0
    max_drift_risk_score: float = 55.0
    min_survivability_score: float = 65.0
    max_regime_novelty_score: float = 55.0
    max_execution_stress_score: float = 55.0
    max_halt_latency_p95_ms: float = 500.0
    max_continuous_pressure_score: float = 60.0
    min_kill_switch_efficiency: float = 0.80
    max_stress_field_hysteresis: float = 0.70


@dataclass
class ProductionCertificationReport:
    generated_at: str
    ready_for_live: bool
    current_green: bool
    consecutive_green_days: float
    required_green_days: float
    paper_validation_ready: bool
    health_status: str
    parity_verdict: str
    parity_unexplained_pnl_bps: float
    shadow_ready: bool
    shadow_live_ready: bool
    shadow_live_trade_count: int
    shadow_live_entry_comparison_count: int
    shadow_live_exit_comparison_count: int
    shadow_live_validation_runtime_days: float
    shadow_live_comparator_score: float
    shadow_live_comparator_level: str
    failure_drill_passed: bool
    drift_risk_score: float
    drift_risk_level: str
    predicted_certification_failure: bool
    survivability_score: float
    survivability_level: str
    regime_novelty_score: float
    execution_stress_score: float
    halt_latency_p95_ms: float
    recommended_exposure_ladder_step: str
    continuous_pressure_score: float
    continuous_pressure_level: str
    stress_field_phase: str
    stress_field_hysteresis: float
    stress_field_latency_memory: float
    stress_field_adversarial_intensity: float
    stress_observation_consistent: bool
    trajectory_novelty_score: float
    execution_friction_score: float
    kill_switch_efficiency: float
    recommended_probation_mode: str
    recommended_deployment_mode: str
    broker_status: str
    broker_critical_issues: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_broker_reconciliation_report(
    system_positions: Iterable[Any],
    broker_positions: Iterable[Any],
    open_orders: Iterable[Any],
    *,
    max_qty_mismatch_ratio: float = 0.05,
    max_order_age_seconds: float = 900.0,
) -> BrokerReconciliationReport:
    generated_at = _now_iso()
    issues: list[BrokerReconciliationIssue] = []

    system_map: dict[str, dict[str, Any]] = {}
    for position in system_positions:
        payload = _as_dict(position)
        symbol = str(payload.get("symbol", ""))
        if not symbol:
            continue
        qty = _float(payload.get("qty"))
        if qty <= 0.0:
            entry_price = max(_float(payload.get("entry_price")), 1e-9)
            qty = _float(payload.get("size_usd")) / entry_price
        signed_qty = qty * (1 if int(payload.get("direction", 1)) >= 0 else -1)
        agg = system_map.setdefault(
            symbol,
            {"signed_qty": 0.0, "positions": [], "stop_ids": set(), "tp_ids": set()},
        )
        agg["signed_qty"] += signed_qty
        agg["positions"].append(payload)
        if payload.get("stop_order_id"):
            agg["stop_ids"].add(str(payload["stop_order_id"]))
        if payload.get("take_profit_order_id"):
            agg["tp_ids"].add(str(payload["take_profit_order_id"]))

    broker_map: dict[str, dict[str, Any]] = {}
    for position in broker_positions:
        payload = _as_dict(position)
        symbol = str(payload.get("symbol", ""))
        if not symbol:
            continue
        broker_map[symbol] = {
            "signed_qty": _float(payload.get("signed_size") or payload.get("size")),
            "size": abs(_float(payload.get("size") or payload.get("signed_size"))),
            "side": str(payload.get("side", "")),
        }

    order_ids = set()
    stale_open_orders = 0
    now_ts = datetime.now(timezone.utc).timestamp()
    for order in open_orders:
        payload = _as_dict(order)
        order_id = str(payload.get("id", ""))
        if order_id:
            order_ids.add(order_id)
        submitted_at = str(payload.get("submitted_at", ""))
        if submitted_at and now_ts - _iso_to_ts(submitted_at) > max_order_age_seconds:
            stale_open_orders += 1

    missing_broker_positions = 0
    unknown_broker_positions = 0
    size_mismatch_count = 0

    for symbol, system_state in system_map.items():
        broker_state = broker_map.get(symbol)
        if broker_state is None:
            missing_broker_positions += 1
            issues.append(
                BrokerReconciliationIssue(
                    severity="critical",
                    code="missing_broker_position",
                    symbol=symbol,
                    message="System tracks an open position that is absent at the broker.",
                )
            )
            continue

        system_signed_qty = _float(system_state["signed_qty"])
        broker_signed_qty = _float(broker_state["signed_qty"])
        qty_diff = abs(system_signed_qty - broker_signed_qty)
        denom = max(abs(system_signed_qty), 1e-9)
        mismatch_ratio = qty_diff / denom
        if mismatch_ratio > max_qty_mismatch_ratio:
            size_mismatch_count += 1
            issues.append(
                BrokerReconciliationIssue(
                    severity="critical",
                    code="size_mismatch",
                    symbol=symbol,
                    message="Broker size diverges materially from internal state.",
                    details={
                        "system_signed_qty": system_signed_qty,
                        "broker_signed_qty": broker_signed_qty,
                        "mismatch_ratio": mismatch_ratio,
                    },
                )
            )

        for stop_order_id in sorted(system_state["stop_ids"]):
            if stop_order_id not in order_ids:
                issues.append(
                    BrokerReconciliationIssue(
                        severity="warning",
                        code="missing_stop_order",
                        symbol=symbol,
                        message="Tracked stop-loss order is not currently open at the broker.",
                        details={"stop_order_id": stop_order_id},
                    )
                )
        for take_profit_order_id in sorted(system_state["tp_ids"]):
            if take_profit_order_id not in order_ids:
                issues.append(
                    BrokerReconciliationIssue(
                        severity="warning",
                        code="missing_take_profit_order",
                        symbol=symbol,
                        message="Tracked take-profit order is not currently open at the broker.",
                        details={"take_profit_order_id": take_profit_order_id},
                    )
                )

    for symbol, broker_state in broker_map.items():
        if symbol not in system_map:
            unknown_broker_positions += 1
            issues.append(
                BrokerReconciliationIssue(
                    severity="critical",
                    code="unknown_broker_position",
                    symbol=symbol,
                    message="Broker reports an unmanaged live position.",
                    details={"side": broker_state["side"], "size": broker_state["size"]},
                )
            )

    if stale_open_orders > 0:
        issues.append(
            BrokerReconciliationIssue(
                severity="warning",
                code="stale_open_orders",
                message="Protective or working orders have been open longer than the configured SLA.",
                details={"stale_open_orders": stale_open_orders},
            )
        )

    critical_issue_count = sum(1 for issue in issues if issue.severity == "critical")
    warning_issue_count = sum(1 for issue in issues if issue.severity == "warning")
    overall_status = "critical" if critical_issue_count > 0 else "warning" if warning_issue_count > 0 else "ok"
    return BrokerReconciliationReport(
        generated_at=generated_at,
        overall_status=overall_status,
        critical_issue_count=critical_issue_count,
        warning_issue_count=warning_issue_count,
        missing_broker_positions=missing_broker_positions,
        unknown_broker_positions=unknown_broker_positions,
        size_mismatch_count=size_mismatch_count,
        stale_open_orders=stale_open_orders,
        critical_symbols=sorted({issue.symbol for issue in issues if issue.symbol and issue.severity == "critical"}),
        issues=issues,
    )


def build_trade_journal_parity_report(
    base_dir: Path,
    *,
    warn_bps: float = 3.0,
    fail_bps: float = 6.0,
) -> TradeJournalParityReport:
    journal = _load_json(base_dir / "trade_journal.json", [])
    if not isinstance(journal, list):
        journal = []

    entry_diffs: list[float] = []
    exit_diffs: list[float] = []
    total_notional = 0.0
    total_pnl_delta = 0.0

    for trade in journal:
        if not isinstance(trade, dict):
            continue
        entry_price = _float(trade.get("entry_price"))
        exit_price = _float(trade.get("exit_price"))
        expected_entry = _float(trade.get("entry_expected_price"), entry_price)
        expected_exit = _float(trade.get("exit_expected_price"), exit_price)
        size_usd = _float(trade.get("filled_size_usd"), _float(trade.get("size_usd")))
        if min(entry_price, exit_price, expected_entry, expected_exit, size_usd) <= 0.0:
            continue
        direction = 1 if int(trade.get("direction", 1)) >= 0 else -1
        entry_diff_bps = abs(_float(trade.get("entry_slippage_bps"), (entry_price - expected_entry) / expected_entry * 1e4))
        exit_diff_bps = abs(_float(trade.get("exit_slippage_bps"), (exit_price - expected_exit) / expected_exit * 1e4))
        actual_return = direction * (exit_price - entry_price) / max(entry_price, 1e-9)
        expected_return = direction * (expected_exit - expected_entry) / max(expected_entry, 1e-9)
        pnl_delta = size_usd * (actual_return - expected_return)
        entry_diffs.append(entry_diff_bps)
        exit_diffs.append(exit_diff_bps)
        total_notional += size_usd
        total_pnl_delta += pnl_delta

    unexplained_pnl_bps = (total_pnl_delta / total_notional * 1e4) if total_notional > 0.0 else 0.0
    abs_unexplained = abs(unexplained_pnl_bps)
    verdict = "FAIL" if abs_unexplained >= fail_bps else "WARN" if abs_unexplained >= warn_bps else "PASS"
    reasons: list[str] = []
    if abs_unexplained >= fail_bps:
        reasons.append(f"Unexplained execution drag {unexplained_pnl_bps:+.2f}bps exceeds fail threshold.")
    elif abs_unexplained >= warn_bps:
        reasons.append(f"Unexplained execution drag {unexplained_pnl_bps:+.2f}bps exceeds warning threshold.")

    return TradeJournalParityReport(
        generated_at=_now_iso(),
        trade_count=len(entry_diffs),
        total_notional_usd=total_notional,
        avg_abs_entry_diff_bps=sum(entry_diffs) / len(entry_diffs) if entry_diffs else 0.0,
        avg_abs_exit_diff_bps=sum(exit_diffs) / len(exit_diffs) if exit_diffs else 0.0,
        unexplained_pnl_bps=unexplained_pnl_bps,
        verdict=verdict,
        reasons=reasons,
    )


def build_shadow_execution_report(
    base_dir: Path,
    *,
    max_avg_entry_delta_bps: float = 12.0,
    max_avg_pnl_delta_pct: float = 0.20,
) -> ShadowExecutionReport:
    journal = _load_json(base_dir / "trade_journal.json", [])
    if not isinstance(journal, list):
        journal = []

    entry_deltas: list[float] = []
    exit_deltas: list[float] = []
    pnl_deltas: list[float] = []
    missing_shadow = 0

    for trade in journal:
        if not isinstance(trade, dict):
            continue
        entry_price = _float(trade.get("entry_price"))
        exit_price = _float(trade.get("exit_price"))
        shadow_entry = _float(trade.get("shadow_entry_price"))
        shadow_exit = _float(trade.get("shadow_exit_price"))
        if shadow_entry <= 0.0 or shadow_exit <= 0.0:
            missing_shadow += 1
            continue
        direction = 1 if int(trade.get("direction", 1)) >= 0 else -1
        actual_return = direction * (exit_price - entry_price) / max(entry_price, 1e-9)
        shadow_return = direction * (shadow_exit - shadow_entry) / max(shadow_entry, 1e-9)
        entry_deltas.append(abs(shadow_entry - entry_price) / max(entry_price, 1e-9) * 1e4)
        exit_deltas.append(abs(shadow_exit - exit_price) / max(exit_price, 1e-9) * 1e4)
        pnl_deltas.append(abs(shadow_return - actual_return))

    compared = len(entry_deltas)
    avg_entry_delta = sum(entry_deltas) / compared if compared else 0.0
    avg_exit_delta = sum(exit_deltas) / compared if compared else 0.0
    avg_pnl_delta = sum(pnl_deltas) / compared if compared else 0.0
    max_pnl_delta = max(pnl_deltas) if pnl_deltas else 0.0
    reasons: list[str] = []
    ready = compared > 0
    if compared == 0:
        ready = False
        reasons.append("No shadow-trade comparisons have been captured yet.")
    if avg_entry_delta > max_avg_entry_delta_bps:
        ready = False
        reasons.append(f"Average shadow entry drift {avg_entry_delta:.2f}bps exceeds threshold.")
    if avg_pnl_delta > max_avg_pnl_delta_pct:
        ready = False
        reasons.append(f"Average shadow PnL drift {avg_pnl_delta:.2%} exceeds threshold.")

    return ShadowExecutionReport(
        generated_at=_now_iso(),
        trade_count=len(journal),
        compared_trade_count=compared,
        missing_shadow_count=missing_shadow,
        avg_abs_entry_delta_bps=avg_entry_delta,
        avg_abs_exit_delta_bps=avg_exit_delta,
        avg_abs_pnl_delta_pct=avg_pnl_delta,
        max_abs_pnl_delta_pct=max_pnl_delta,
        ready_for_live=ready,
        reasons=reasons,
    )


def _compute_consecutive_green_days(
    history: list[dict[str, Any]],
    *,
    current_green: bool,
    now_iso: str,
) -> float:
    if not current_green:
        return 0.0

    rows = [row for row in history if isinstance(row, dict)]
    rows.append({"timestamp": now_iso, "current_green": True})
    rows.sort(key=lambda row: _iso_to_ts(str(row.get("timestamp", ""))))
    contiguous_start = _iso_to_ts(now_iso)
    for row in reversed(rows):
        if not bool(row.get("current_green")):
            break
        ts = _iso_to_ts(str(row.get("timestamp", "")))
        if ts > 0.0:
            contiguous_start = ts
    return max((_iso_to_ts(now_iso) - contiguous_start) / 86400.0, 0.0)


def build_production_certification_report(
    base_dir: Path,
    thresholds: ProductionCertificationThresholds | None = None,
) -> ProductionCertificationReport:
    thresholds = thresholds or ProductionCertificationThresholds()
    now_iso = _now_iso()
    paper_validation = _load_json(base_dir / "paper_validation_status.json", {})
    health = _load_json(base_dir / "health.json", {})
    parity = _load_json(base_dir / "trade_parity_status.json", {})
    shadow = _load_json(base_dir / "shadow_execution_status.json", {})
    shadow_live = _load_json(base_dir / "shadow_live_comparator_status.json", {})
    failure_drill = _load_json(base_dir / "failure_drill_report.json", {})
    drift = _load_json(base_dir / "drift_intelligence_status.json", {})
    survivability = _load_json(base_dir / "survivability_status.json", {})
    stress_kernel = _load_json(base_dir / "streaming_stress_kernel_status.json", {})
    stress_field = _load_json(base_dir / "stress_field_state.json", {})
    broker = _load_json(base_dir / "broker_reconciliation_status.json", {})
    history = _load_jsonl(base_dir / "production_certification_history.jsonl")

    paper_ready = bool(paper_validation.get("ready_for_live"))
    health_status = str(health.get("overall_status", "unknown"))
    parity_verdict = str(parity.get("verdict", "unknown"))
    parity_unexplained = _float(parity.get("unexplained_pnl_bps"))
    shadow_ready = bool(shadow.get("ready_for_live")) if shadow else False
    shadow_live_ready = bool(shadow_live.get("ready_for_capital")) if shadow_live else False
    shadow_live_trade_count = int(shadow_live.get("trade_count", 0) or 0)
    shadow_live_entry_comparison_count = int(shadow_live.get("entry_comparison_count", 0) or 0)
    shadow_live_exit_comparison_count = int(shadow_live.get("exit_comparison_count", 0) or 0)
    shadow_live_validation_runtime_days = _float(shadow_live.get("validation_runtime_days"))
    shadow_live_comparator_score = _float(shadow_live.get("comparator_score"))
    shadow_live_comparator_level = str(shadow_live.get("comparator_level", "unknown"))
    shadow_live_reasons = [
        str(reason)
        for reason in shadow_live.get("reasons", [])
        if isinstance(reason, str) and reason
    ] if isinstance(shadow_live.get("reasons"), list) else []
    failure_drill_passed = bool(failure_drill.get("all_passed"))
    drift_risk_score = _float(drift.get("risk_score"))
    drift_risk_level = str(drift.get("risk_level", "unknown"))
    predicted_certification_failure = bool(drift.get("predicted_certification_failure"))
    deployment = drift.get("deployment_recommendation") if isinstance(drift.get("deployment_recommendation"), dict) else {}
    survivability_score = _float(survivability.get("survivability_score"))
    survivability_level = str(survivability.get("survivability_level", "unknown"))
    regime_novelty_score = _float(survivability.get("regime_novelty_score"))
    execution_stress_score = _float(survivability.get("execution_stress_score"))
    halt_latency_p95_ms = _float(survivability.get("halt_latency_p95_ms"))
    exposure_ladder = survivability.get("exposure_ladder") if isinstance(survivability.get("exposure_ladder"), dict) else {}
    recommended_exposure_ladder_step = str(exposure_ladder.get("stage", "shadow"))
    continuous_pressure_score = _float(stress_kernel.get("continuous_pressure_score"))
    continuous_pressure_level = str(stress_kernel.get("pressure_level", "unknown"))
    stress_kernel_source_generated_at = str(stress_kernel.get("source_generated_at", "") or "")
    stress_field_phase = str(stress_field.get("phase", "unknown"))
    stress_field_hysteresis = _float(stress_field.get("hysteresis_score"))
    stress_field_latency_memory = _float(stress_field.get("latency_memory"))
    stress_field_adversarial_intensity = _float(
        (stress_field.get("adversarial_input") or {}).get("intensity")
        if isinstance(stress_field.get("adversarial_input"), dict)
        else 0.0
    )
    stress_field_source_generated_at = str(stress_field.get("source_generated_at", "") or "")
    stress_observation_consistent = bool(
        stress_kernel_source_generated_at
        and stress_field_source_generated_at
        and stress_kernel_source_generated_at == stress_field_source_generated_at
    )
    stress_field_should_halt = bool(stress_field.get("should_halt"))
    stress_field_allow_entries = bool(stress_field.get("allow_entries", True))
    trajectory_novelty_score = _float(stress_kernel.get("trajectory_novelty_score"))
    execution_friction_score = _float(stress_kernel.get("execution_friction_score"))
    kill_switch_efficiency = _float(stress_kernel.get("kill_switch_efficiency"), 1.0)
    probation_policy = stress_kernel.get("probation_live_policy") if isinstance(stress_kernel.get("probation_live_policy"), dict) else {}
    recommended_probation_mode = str(probation_policy.get("stage", "shadow"))
    if (
        not stress_field
        or not stress_observation_consistent
        or stress_field_should_halt
        or not stress_field_allow_entries
    ):
        recommended_probation_mode = "blocked"
    drift_mode = str(deployment.get("mode", "blocked"))
    ladder_mode = {
        "blocked": "blocked",
        "shadow": "paper_shadow",
        "0.1%": "micro_live",
        "0.5%": "micro_live",
        "1%": "micro_live",
        "5%": "scale_up",
    }.get(recommended_exposure_ladder_step, "paper_shadow")
    mode_rank = {"blocked": 0, "paper_shadow": 1, "micro_live": 2, "scale_up": 3}
    recommended_deployment_mode = min((drift_mode, ladder_mode), key=lambda mode: mode_rank.get(mode, 0))
    broker_status = str(broker.get("overall_status", "unknown"))
    broker_critical_issues = int(broker.get("critical_issue_count", 0) or 0)

    reasons: list[str] = []
    current_green = True
    if not paper_ready:
        current_green = False
        reasons.append("Strict paper validation has not passed.")
    if health_status != "ok" or bool(health.get("should_halt")):
        current_green = False
        reasons.append(f"System health is {health_status}, not ok.")
    if parity_verdict != "PASS" or abs(parity_unexplained) > thresholds.max_parity_unexplained_pnl_bps:
        current_green = False
        reasons.append("Execution parity is outside the configured tolerance.")
    if broker_status != "ok" or broker_critical_issues > thresholds.max_broker_critical_issues:
        current_green = False
        reasons.append("Broker reconciliation is not clean.")
    if thresholds.require_shadow_mode:
        if not shadow:
            current_green = False
            reasons.append("Shadow execution status is missing.")
        elif not shadow_ready:
            current_green = False
            reasons.append("Shadow execution drift is not within tolerance.")
        if not shadow_live:
            current_green = False
            reasons.append("Shadow live comparator status is missing.")
        elif not shadow_live_ready:
            current_green = False
            reasons.append("Shadow live comparator is not yet within broker-quoted tolerance.")
            reasons.extend(f"Shadow live: {reason}" for reason in shadow_live_reasons[:3])
    if not failure_drill_passed:
        current_green = False
        reasons.append("Failure drill suite has not passed.")
    if not drift:
        current_green = False
        reasons.append("Drift intelligence status is missing.")
    elif predicted_certification_failure or drift_risk_score > thresholds.max_drift_risk_score:
        current_green = False
        reasons.append("Drift intelligence predicts certification instability.")
    if not survivability:
        current_green = False
        reasons.append("Survivability report is missing.")
    elif (
        bool(survivability.get("predicted_survivability_failure"))
        or survivability_score < thresholds.min_survivability_score
        or regime_novelty_score > thresholds.max_regime_novelty_score
        or execution_stress_score > thresholds.max_execution_stress_score
        or halt_latency_p95_ms > thresholds.max_halt_latency_p95_ms
    ):
        current_green = False
        reasons.append("Survivability lab does not clear pre-live stress and novelty thresholds.")
    if not stress_kernel:
        current_green = False
        reasons.append("Streaming stress kernel status is missing.")
    elif (
        bool(stress_kernel.get("predicted_pressure_failure"))
        or continuous_pressure_score > thresholds.max_continuous_pressure_score
        or kill_switch_efficiency < thresholds.min_kill_switch_efficiency
    ):
        current_green = False
        reasons.append("Continuous pressure kernel predicts insufficient runtime continuity.")
    if not stress_field:
        current_green = False
        reasons.append("Stateful stress field status is missing.")
    elif not stress_observation_consistent:
        current_green = False
        reasons.append(
            "Stress kernel and stateful field do not reference the same source snapshot "
            f"(kernel={stress_kernel_source_generated_at or 'unknown'}, field={stress_field_source_generated_at or 'unknown'})."
        )
    elif stress_field_should_halt or not stress_field_allow_entries:
        current_green = False
        reasons.append("Stateful stress field is currently blocking entries or requiring a halt.")
    elif stress_field_hysteresis > thresholds.max_stress_field_hysteresis:
        current_green = False
        reasons.append("Stateful stress field hysteresis remains too elevated for certification.")

    consecutive_green_days = _compute_consecutive_green_days(
        history,
        current_green=current_green,
        now_iso=now_iso,
    )
    if consecutive_green_days < thresholds.min_consecutive_green_days:
        reasons.append(
            f"Burn-in evidence is only {consecutive_green_days:.1f} days; require {thresholds.min_consecutive_green_days:.1f} days."
        )

    if recommended_probation_mode != "micro_live_ready":
        reasons.append(
            f"Continuous stress kernel currently only allows {recommended_probation_mode}, not full micro-live readiness."
        )

    ready_for_live = (
        current_green
        and consecutive_green_days >= thresholds.min_consecutive_green_days
        and recommended_probation_mode == "micro_live_ready"
    )
    return ProductionCertificationReport(
        generated_at=now_iso,
        ready_for_live=ready_for_live,
        current_green=current_green,
        consecutive_green_days=consecutive_green_days,
        required_green_days=thresholds.min_consecutive_green_days,
        paper_validation_ready=paper_ready,
        health_status=health_status,
        parity_verdict=parity_verdict,
        parity_unexplained_pnl_bps=parity_unexplained,
        shadow_ready=shadow_ready,
        shadow_live_ready=shadow_live_ready,
        shadow_live_trade_count=shadow_live_trade_count,
        shadow_live_entry_comparison_count=shadow_live_entry_comparison_count,
        shadow_live_exit_comparison_count=shadow_live_exit_comparison_count,
        shadow_live_validation_runtime_days=shadow_live_validation_runtime_days,
        shadow_live_comparator_score=shadow_live_comparator_score,
        shadow_live_comparator_level=shadow_live_comparator_level,
        failure_drill_passed=failure_drill_passed,
        drift_risk_score=drift_risk_score,
        drift_risk_level=drift_risk_level,
        predicted_certification_failure=predicted_certification_failure,
        survivability_score=survivability_score,
        survivability_level=survivability_level,
        regime_novelty_score=regime_novelty_score,
        execution_stress_score=execution_stress_score,
        halt_latency_p95_ms=halt_latency_p95_ms,
        recommended_exposure_ladder_step=recommended_exposure_ladder_step,
        continuous_pressure_score=continuous_pressure_score,
        continuous_pressure_level=continuous_pressure_level,
        stress_field_phase=stress_field_phase,
        stress_field_hysteresis=stress_field_hysteresis,
        stress_field_latency_memory=stress_field_latency_memory,
        stress_field_adversarial_intensity=stress_field_adversarial_intensity,
        stress_observation_consistent=stress_observation_consistent,
        trajectory_novelty_score=trajectory_novelty_score,
        execution_friction_score=execution_friction_score,
        kill_switch_efficiency=kill_switch_efficiency,
        recommended_probation_mode=recommended_probation_mode,
        recommended_deployment_mode=recommended_deployment_mode,
        broker_status=broker_status,
        broker_critical_issues=broker_critical_issues,
        reasons=reasons,
    )


def write_production_certification_report(
    report: ProductionCertificationReport,
    path: Path,
    history_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))
    if history_path is None:
        return
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a") as handle:
        handle.write(json.dumps({
            "timestamp": report.generated_at,
            "ready_for_live": report.ready_for_live,
            "current_green": report.current_green,
            "consecutive_green_days": report.consecutive_green_days,
        }) + "\n")


def format_production_certification_report(report: ProductionCertificationReport) -> str:
    lines = [
        "Production Certification Report",
        f"  Ready for live: {'YES' if report.ready_for_live else 'NO'}",
        f"  Burn-in: {report.consecutive_green_days:.1f}/{report.required_green_days:.1f} green days",
        f"  Paper validation: {'PASS' if report.paper_validation_ready else 'FAIL'}",
        f"  Health: {report.health_status}",
        f"  Parity: {report.parity_verdict} ({report.parity_unexplained_pnl_bps:+.2f}bps)",
        f"  Shadow: {'PASS' if report.shadow_ready else 'FAIL'}",
        f"  Shadow live: {'PASS' if report.shadow_live_ready else 'FAIL'} | score={report.shadow_live_comparator_score:.1f}/100 {report.shadow_live_comparator_level} | trades={report.shadow_live_trade_count} | entry={report.shadow_live_entry_comparison_count} exit={report.shadow_live_exit_comparison_count} | runtime={report.shadow_live_validation_runtime_days:.1f}d",
        f"  Failure drills: {'PASS' if report.failure_drill_passed else 'FAIL'}",
        f"  Drift intelligence: {report.drift_risk_level} ({report.drift_risk_score:.1f}/100)",
        f"  Survivability: {report.survivability_level} ({report.survivability_score:.1f}/100) | novelty={report.regime_novelty_score:.1f} | stress={report.execution_stress_score:.1f}",
        f"  Halt latency p95: {report.halt_latency_p95_ms:.0f}ms | ladder={report.recommended_exposure_ladder_step}",
        f"  Stress continuity: {report.continuous_pressure_level} ({report.continuous_pressure_score:.1f}/100) | trajectory={report.trajectory_novelty_score:.1f} | friction={report.execution_friction_score:.1f} | kill={report.kill_switch_efficiency:.0%}",
        f"  Stress field: {report.stress_field_phase} | hysteresis={report.stress_field_hysteresis:.0%} | latency memory={report.stress_field_latency_memory:.0%} | adversary={report.stress_field_adversarial_intensity:.0%} | observation={'aligned' if report.stress_observation_consistent else 'mismatched'}",
        f"  Probation mode: {report.recommended_probation_mode}",
        f"  Broker reconciliation: {report.broker_status} ({report.broker_critical_issues} critical)",
        f"  Recommended next mode: {report.recommended_deployment_mode}",
    ]
    if report.reasons:
        lines.append("  Blocking reasons:")
        for reason in report.reasons:
            lines.append(f"    - {reason}")
    return "\n".join(lines)
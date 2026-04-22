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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _trend(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    try:
        slope, _ = np.polyfit(x, y, 1)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return 0.0
    return float(slope)


def _scaled_pressure(value: float, threshold: float, *, grace: float = 0.5) -> float:
    if threshold <= 0.0:
        return 0.0
    ratio = value / threshold
    if ratio <= grace:
        return 0.0
    return min((ratio - grace) / max(1.0 - grace, 1e-9), 2.0)


@dataclass
class ExecutionDriftThresholds:
    lookback_trade_points: int = 120
    min_compared_trades_for_capital: int = 5
    min_shadow_compared_trades_for_capital: int = 5
    max_avg_entry_slippage_bps: float = 12.0
    max_avg_exit_slippage_bps: float = 16.0
    max_abs_pnl_divergence_pct: float = 20.0
    max_miss_rate: float = 0.15
    max_partial_fill_rate: float = 0.25
    min_avg_fill_ratio: float = 0.90
    max_avg_entry_execution_ms: float = 2500.0
    max_avg_exit_execution_ms: float = 2500.0
    max_avg_book_spread_bps: float = 18.0
    max_avg_book_impact_bps: float = 25.0
    max_avg_shadow_entry_delta_bps: float = 8.0
    max_avg_shadow_exit_delta_bps: float = 10.0
    max_avg_shadow_pnl_delta_pct: float = 12.0
    min_shadow_live_entry_comparisons_for_capital: int = 300
    min_shadow_live_exit_comparisons_for_capital: int = 300
    max_avg_shadow_live_entry_reference_gap_bps: float = 10.0
    max_avg_shadow_live_exit_reference_gap_bps: float = 12.0
    max_avg_shadow_live_entry_fill_gap_bps: float = 6.0
    max_avg_shadow_live_exit_fill_gap_bps: float = 8.0


@dataclass
class ExecutionDriftReport:
    generated_at: str
    compared_trade_count: int
    missed_trade_count: int
    shadow_compared_trade_count: int
    avg_entry_slippage_bps: float
    avg_exit_slippage_bps: float
    avg_abs_pnl_divergence_pct: float
    miss_rate: float
    partial_fill_rate: float
    avg_fill_ratio: float
    avg_entry_execution_ms: float
    avg_exit_execution_ms: float
    avg_book_spread_bps: float
    avg_book_impact_bps: float
    avg_shadow_entry_delta_bps: float
    avg_shadow_exit_delta_bps: float
    avg_shadow_pnl_delta_pct: float
    shadow_live_entry_comparison_count: int
    shadow_live_exit_comparison_count: int
    avg_shadow_live_entry_reference_gap_bps: float
    avg_shadow_live_exit_reference_gap_bps: float
    avg_shadow_live_entry_fill_gap_bps: float
    avg_shadow_live_exit_fill_gap_bps: float
    slippage_trend_bps_per_trade: float
    latency_trend_ms_per_trade: float
    execution_fidelity_score: float
    execution_fidelity_level: str
    reliable_for_capital: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_execution_drift_report(
    base_dir: Path,
    thresholds: ExecutionDriftThresholds | None = None,
) -> ExecutionDriftReport:
    thresholds = thresholds or ExecutionDriftThresholds()
    divergence_payload = _load_json(base_dir / "divergence_log.json", [])
    divergence_rows = divergence_payload if isinstance(divergence_payload, list) else divergence_payload.get("comparisons", [])
    divergence_rows = [row for row in divergence_rows if isinstance(row, dict)][-thresholds.lookback_trade_points :]

    executed_rows = [row for row in divergence_rows if not bool(row.get("missed"))]
    missed_rows = [row for row in divergence_rows if bool(row.get("missed"))]

    journal_payload = _load_json(base_dir / "trade_journal.json", [])
    journal_rows = [row for row in journal_payload if isinstance(row, dict)][-thresholds.lookback_trade_points :]

    shadow = _load_json(base_dir / "shadow_execution_status.json", {})
    shadow_live = _load_json(base_dir / "shadow_live_comparator_status.json", {})

    entry_slippages = [abs(_float(row.get("entry_slippage_bps"))) for row in executed_rows]
    exit_slippages = [abs(_float(row.get("exit_slippage_bps"))) for row in executed_rows]
    pnl_divergences = [abs(_float(row.get("pnl_divergence_pct"))) for row in executed_rows]
    fill_latencies = [_float(row.get("fill_time_ms")) for row in executed_rows]
    partial_fill_rate = (
        sum(1 for row in executed_rows if bool(row.get("was_partial"))) / max(len(executed_rows), 1)
        if executed_rows
        else 0.0
    )

    fill_ratios = [_float(row.get("fill_ratio"), 1.0) for row in journal_rows]
    entry_execution_ms = [_float(row.get("entry_execution_ms")) for row in journal_rows]
    exit_execution_ms = [_float(row.get("exit_execution_ms")) for row in journal_rows]
    book_spreads = [_float(row.get("book_spread_bps")) for row in journal_rows]
    book_impacts = [_float(row.get("book_impact_bps")) for row in journal_rows]

    compared_trade_count = len(executed_rows)
    missed_trade_count = len(missed_rows)
    total_signals = len(divergence_rows)
    miss_rate = missed_trade_count / max(total_signals, 1) if total_signals else 0.0

    avg_entry_slippage_bps = _mean(entry_slippages)
    avg_exit_slippage_bps = _mean(exit_slippages)
    avg_abs_pnl_divergence_pct = _mean(pnl_divergences)
    avg_fill_ratio = _mean(fill_ratios)
    avg_entry_execution_ms = _mean(entry_execution_ms)
    avg_exit_execution_ms = _mean(exit_execution_ms)
    avg_book_spread_bps = _mean(book_spreads)
    avg_book_impact_bps = _mean(book_impacts)
    avg_shadow_entry_delta_bps = _float(shadow.get("avg_abs_entry_delta_bps"))
    avg_shadow_exit_delta_bps = _float(shadow.get("avg_abs_exit_delta_bps"))
    avg_shadow_pnl_delta_pct = _float(shadow.get("avg_abs_pnl_delta_pct")) * 100.0
    shadow_compared_trade_count = int(shadow.get("compared_trade_count", 0) or 0)
    shadow_live_entry_comparison_count = int(shadow_live.get("entry_comparison_count", 0) or 0)
    shadow_live_exit_comparison_count = int(shadow_live.get("exit_comparison_count", 0) or 0)
    avg_shadow_live_entry_reference_gap_bps = _float(shadow_live.get("avg_abs_entry_reference_gap_bps"))
    avg_shadow_live_exit_reference_gap_bps = _float(shadow_live.get("avg_abs_exit_reference_gap_bps"))
    avg_shadow_live_entry_fill_gap_bps = _float(shadow_live.get("avg_abs_entry_fill_gap_bps"))
    avg_shadow_live_exit_fill_gap_bps = _float(shadow_live.get("avg_abs_exit_fill_gap_bps"))
    slippage_trend_bps_per_trade = _trend(entry_slippages)
    latency_trend_ms_per_trade = _trend(fill_latencies)

    reasons: list[str] = []
    if compared_trade_count < thresholds.min_compared_trades_for_capital:
        reasons.append(
            f"Need at least {thresholds.min_compared_trades_for_capital} executed comparisons before capital can trust paper drift."
        )
    if shadow_compared_trade_count < thresholds.min_shadow_compared_trades_for_capital:
        reasons.append(
            f"Need at least {thresholds.min_shadow_compared_trades_for_capital} shadow comparisons before capital can trust execution drift."
        )
    if shadow_live_entry_comparison_count < thresholds.min_shadow_live_entry_comparisons_for_capital:
        reasons.append(
            f"Need at least {thresholds.min_shadow_live_entry_comparisons_for_capital} broker-quoted entry comparisons before capital can trust shadow-live execution."
        )
    if shadow_live_exit_comparison_count < thresholds.min_shadow_live_exit_comparisons_for_capital:
        reasons.append(
            f"Need at least {thresholds.min_shadow_live_exit_comparisons_for_capital} broker-quoted exit comparisons before capital can trust shadow-live execution."
        )
    if avg_entry_slippage_bps > thresholds.max_avg_entry_slippage_bps:
        reasons.append(f"Average entry slippage {avg_entry_slippage_bps:.2f}bps exceeds threshold.")
    if avg_exit_slippage_bps > thresholds.max_avg_exit_slippage_bps:
        reasons.append(f"Average exit slippage {avg_exit_slippage_bps:.2f}bps exceeds threshold.")
    if avg_abs_pnl_divergence_pct > thresholds.max_abs_pnl_divergence_pct:
        reasons.append(f"Average PnL divergence {avg_abs_pnl_divergence_pct:.2f}% exceeds threshold.")
    if miss_rate > thresholds.max_miss_rate:
        reasons.append(f"Miss rate {miss_rate:.0%} exceeds threshold.")
    if partial_fill_rate > thresholds.max_partial_fill_rate:
        reasons.append(f"Partial fill rate {partial_fill_rate:.0%} exceeds threshold.")
    if fill_ratios and avg_fill_ratio < thresholds.min_avg_fill_ratio:
        reasons.append(f"Average fill ratio {avg_fill_ratio:.0%} is below threshold.")
    if entry_execution_ms and avg_entry_execution_ms > thresholds.max_avg_entry_execution_ms:
        reasons.append(f"Average entry execution latency {avg_entry_execution_ms:.0f}ms exceeds threshold.")
    if exit_execution_ms and avg_exit_execution_ms > thresholds.max_avg_exit_execution_ms:
        reasons.append(f"Average exit execution latency {avg_exit_execution_ms:.0f}ms exceeds threshold.")
    if book_spreads and avg_book_spread_bps > thresholds.max_avg_book_spread_bps:
        reasons.append(f"Average book spread {avg_book_spread_bps:.2f}bps exceeds threshold.")
    if book_impacts and avg_book_impact_bps > thresholds.max_avg_book_impact_bps:
        reasons.append(f"Average book impact {avg_book_impact_bps:.2f}bps exceeds threshold.")
    if shadow_compared_trade_count > 0 and avg_shadow_entry_delta_bps > thresholds.max_avg_shadow_entry_delta_bps:
        reasons.append(f"Average shadow entry drift {avg_shadow_entry_delta_bps:.2f}bps exceeds threshold.")
    if shadow_compared_trade_count > 0 and avg_shadow_exit_delta_bps > thresholds.max_avg_shadow_exit_delta_bps:
        reasons.append(f"Average shadow exit drift {avg_shadow_exit_delta_bps:.2f}bps exceeds threshold.")
    if shadow_compared_trade_count > 0 and avg_shadow_pnl_delta_pct > thresholds.max_avg_shadow_pnl_delta_pct:
        reasons.append(f"Average shadow PnL drift {avg_shadow_pnl_delta_pct:.2f}% exceeds threshold.")
    if shadow_live_entry_comparison_count > 0 and avg_shadow_live_entry_reference_gap_bps > thresholds.max_avg_shadow_live_entry_reference_gap_bps:
        reasons.append(
            f"Average broker-quoted entry reference gap {avg_shadow_live_entry_reference_gap_bps:.2f}bps exceeds threshold."
        )
    if shadow_live_exit_comparison_count > 0 and avg_shadow_live_exit_reference_gap_bps > thresholds.max_avg_shadow_live_exit_reference_gap_bps:
        reasons.append(
            f"Average broker-quoted exit reference gap {avg_shadow_live_exit_reference_gap_bps:.2f}bps exceeds threshold."
        )
    if shadow_live_entry_comparison_count > 0 and avg_shadow_live_entry_fill_gap_bps > thresholds.max_avg_shadow_live_entry_fill_gap_bps:
        reasons.append(
            f"Average broker-quoted entry fill gap {avg_shadow_live_entry_fill_gap_bps:.2f}bps exceeds threshold."
        )
    if shadow_live_exit_comparison_count > 0 and avg_shadow_live_exit_fill_gap_bps > thresholds.max_avg_shadow_live_exit_fill_gap_bps:
        reasons.append(
            f"Average broker-quoted exit fill gap {avg_shadow_live_exit_fill_gap_bps:.2f}bps exceeds threshold."
        )

    penalty = 0.0
    penalty += _scaled_pressure(avg_entry_slippage_bps, thresholds.max_avg_entry_slippage_bps) * 14.0
    penalty += _scaled_pressure(avg_exit_slippage_bps, thresholds.max_avg_exit_slippage_bps) * 10.0
    penalty += _scaled_pressure(avg_abs_pnl_divergence_pct, thresholds.max_abs_pnl_divergence_pct) * 16.0
    penalty += _scaled_pressure(miss_rate, thresholds.max_miss_rate) * 12.0
    penalty += _scaled_pressure(partial_fill_rate, thresholds.max_partial_fill_rate) * 10.0
    if fill_ratios:
        fill_ratio_gap = max(thresholds.min_avg_fill_ratio - avg_fill_ratio, 0.0) / max(thresholds.min_avg_fill_ratio, 1e-9)
        penalty += min(fill_ratio_gap, 1.0) * 10.0
    if entry_execution_ms:
        penalty += _scaled_pressure(avg_entry_execution_ms, thresholds.max_avg_entry_execution_ms) * 8.0
    if book_spreads:
        penalty += _scaled_pressure(avg_book_spread_bps, thresholds.max_avg_book_spread_bps) * 8.0
    if book_impacts:
        penalty += _scaled_pressure(avg_book_impact_bps, thresholds.max_avg_book_impact_bps) * 6.0
    if shadow_compared_trade_count > 0:
        penalty += _scaled_pressure(avg_shadow_entry_delta_bps, thresholds.max_avg_shadow_entry_delta_bps) * 6.0
        penalty += _scaled_pressure(avg_shadow_pnl_delta_pct, thresholds.max_avg_shadow_pnl_delta_pct) * 6.0
    else:
        penalty += 8.0
    if shadow_live_entry_comparison_count > 0:
        penalty += _scaled_pressure(avg_shadow_live_entry_reference_gap_bps, thresholds.max_avg_shadow_live_entry_reference_gap_bps) * 10.0
        penalty += _scaled_pressure(avg_shadow_live_entry_fill_gap_bps, thresholds.max_avg_shadow_live_entry_fill_gap_bps) * 8.0
    else:
        penalty += 10.0
    if shadow_live_exit_comparison_count > 0:
        penalty += _scaled_pressure(avg_shadow_live_exit_reference_gap_bps, thresholds.max_avg_shadow_live_exit_reference_gap_bps) * 6.0
        penalty += _scaled_pressure(avg_shadow_live_exit_fill_gap_bps, thresholds.max_avg_shadow_live_exit_fill_gap_bps) * 6.0
    else:
        penalty += 8.0
    if compared_trade_count < thresholds.min_compared_trades_for_capital:
        penalty += 10.0
    if shadow_compared_trade_count < thresholds.min_shadow_compared_trades_for_capital:
        penalty += 8.0
    if shadow_live_entry_comparison_count < thresholds.min_shadow_live_entry_comparisons_for_capital:
        penalty += 8.0
    if shadow_live_exit_comparison_count < thresholds.min_shadow_live_exit_comparisons_for_capital:
        penalty += 6.0

    execution_fidelity_score = max(0.0, 100.0 - penalty)
    if execution_fidelity_score >= 80.0:
        execution_fidelity_level = "stable"
    elif execution_fidelity_score >= 60.0:
        execution_fidelity_level = "watch"
    else:
        execution_fidelity_level = "unstable"

    return ExecutionDriftReport(
        generated_at=_now_iso(),
        compared_trade_count=compared_trade_count,
        missed_trade_count=missed_trade_count,
        shadow_compared_trade_count=shadow_compared_trade_count,
        avg_entry_slippage_bps=avg_entry_slippage_bps,
        avg_exit_slippage_bps=avg_exit_slippage_bps,
        avg_abs_pnl_divergence_pct=avg_abs_pnl_divergence_pct,
        miss_rate=miss_rate,
        partial_fill_rate=partial_fill_rate,
        avg_fill_ratio=avg_fill_ratio if fill_ratios else 1.0,
        avg_entry_execution_ms=avg_entry_execution_ms,
        avg_exit_execution_ms=avg_exit_execution_ms,
        avg_book_spread_bps=avg_book_spread_bps,
        avg_book_impact_bps=avg_book_impact_bps,
        avg_shadow_entry_delta_bps=avg_shadow_entry_delta_bps,
        avg_shadow_exit_delta_bps=avg_shadow_exit_delta_bps,
        avg_shadow_pnl_delta_pct=avg_shadow_pnl_delta_pct,
        shadow_live_entry_comparison_count=shadow_live_entry_comparison_count,
        shadow_live_exit_comparison_count=shadow_live_exit_comparison_count,
        avg_shadow_live_entry_reference_gap_bps=avg_shadow_live_entry_reference_gap_bps,
        avg_shadow_live_exit_reference_gap_bps=avg_shadow_live_exit_reference_gap_bps,
        avg_shadow_live_entry_fill_gap_bps=avg_shadow_live_entry_fill_gap_bps,
        avg_shadow_live_exit_fill_gap_bps=avg_shadow_live_exit_fill_gap_bps,
        slippage_trend_bps_per_trade=slippage_trend_bps_per_trade,
        latency_trend_ms_per_trade=latency_trend_ms_per_trade,
        execution_fidelity_score=execution_fidelity_score,
        execution_fidelity_level=execution_fidelity_level,
        reliable_for_capital=not reasons,
        reasons=reasons,
    )


def write_execution_drift_report(report: ExecutionDriftReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))


def format_execution_drift_report(report: ExecutionDriftReport) -> str:
    lines = [
        "Execution Drift Report",
        (
            "  Fidelity: "
            f"{report.execution_fidelity_level} {report.execution_fidelity_score:.1f}/100 | "
            f"capital-ready={'YES' if report.reliable_for_capital else 'NO'}"
        ),
        (
            "  Trades: "
            f"compared={report.compared_trade_count} missed={report.missed_trade_count} "
            f"shadow={report.shadow_compared_trade_count}"
        ),
        (
            "  Drift: "
            f"entry={report.avg_entry_slippage_bps:.2f}bps exit={report.avg_exit_slippage_bps:.2f}bps "
            f"pnl={report.avg_abs_pnl_divergence_pct:.2f}% miss={report.miss_rate:.0%} partial={report.partial_fill_rate:.0%}"
        ),
        (
            "  Friction: "
            f"fill={report.avg_fill_ratio:.0%} lat={report.avg_entry_execution_ms:.0f}/{report.avg_exit_execution_ms:.0f}ms "
            f"spread={report.avg_book_spread_bps:.2f}bps impact={report.avg_book_impact_bps:.2f}bps"
        ),
        (
            "  Shadow: "
            f"entry={report.avg_shadow_entry_delta_bps:.2f}bps exit={report.avg_shadow_exit_delta_bps:.2f}bps "
            f"pnl={report.avg_shadow_pnl_delta_pct:.2f}%"
        ),
        (
            "  Shadow live: "
            f"entry ref={report.avg_shadow_live_entry_reference_gap_bps:.2f}bps fill={report.avg_shadow_live_entry_fill_gap_bps:.2f}bps "
            f"({report.shadow_live_entry_comparison_count} compared) | "
            f"exit ref={report.avg_shadow_live_exit_reference_gap_bps:.2f}bps fill={report.avg_shadow_live_exit_fill_gap_bps:.2f}bps "
            f"({report.shadow_live_exit_comparison_count} compared)"
        ),
    ]
    if report.reasons:
        lines.append("  Blockers:")
        lines.extend(f"    - {reason}" for reason in report.reasons)
    return "\n".join(lines)


__all__ = [
    "ExecutionDriftReport",
    "ExecutionDriftThresholds",
    "build_execution_drift_report",
    "format_execution_drift_report",
    "write_execution_drift_report",
]
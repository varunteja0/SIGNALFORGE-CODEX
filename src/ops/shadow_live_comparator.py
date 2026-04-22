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


def _mean_abs(values: list[float]) -> float:
    return _mean([abs(value) for value in values])


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def _gap_bps(price: float, benchmark: float) -> float:
    if price <= 0.0 or benchmark <= 0.0:
        return 0.0
    return (price / benchmark - 1.0) * 1e4


def _iso_to_ts(value: Any) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _scaled_pressure(value: float, threshold: float, *, grace: float = 0.5) -> float:
    if threshold <= 0.0:
        return 0.0
    ratio = value / threshold
    if ratio <= grace:
        return 0.0
    return min((ratio - grace) / max(1.0 - grace, 1e-9), 2.0)


@dataclass
class ShadowLiveComparisonObservation:
    broker: str
    symbol: str
    side: str
    direction: int
    reference_price: float
    quote_timestamp: str
    best_bid: float
    best_ask: float
    mid_price: float
    touch_price: float
    quote_spread_bps: float
    quote_impact_bps: float
    reference_gap_bps: float
    mid_gap_bps: float
    fill_price: float = 0.0
    fill_gap_bps: float = 0.0
    reduce_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def namespaced(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_quote_timestamp": self.quote_timestamp,
            f"{prefix}_touch_price": self.touch_price,
            f"{prefix}_mid_price": self.mid_price,
            f"{prefix}_reference_gap_bps": self.reference_gap_bps,
            f"{prefix}_mid_gap_bps": self.mid_gap_bps,
            f"{prefix}_fill_gap_bps": self.fill_gap_bps,
            f"{prefix}_quote_spread_bps": self.quote_spread_bps,
            f"{prefix}_quote_impact_bps": self.quote_impact_bps,
        }


@dataclass
class ShadowLiveComparatorThresholds:
    lookback_trade_points: int = 720
    min_entry_comparisons_for_capital: int = 300
    min_exit_comparisons_for_capital: int = 300
    min_validation_runtime_days_for_capital: float = 3.0
    min_quote_coverage_rate_for_capital: float = 0.95
    max_avg_entry_reference_gap_bps: float = 10.0
    max_avg_exit_reference_gap_bps: float = 12.0
    max_avg_entry_fill_gap_bps: float = 6.0
    max_avg_exit_fill_gap_bps: float = 8.0
    max_avg_quote_spread_bps: float = 15.0
    max_avg_quote_impact_bps: float = 20.0
    max_p95_entry_reference_gap_bps: float = 18.0
    max_p95_exit_reference_gap_bps: float = 20.0
    max_p95_entry_fill_gap_bps: float = 10.0
    max_p95_exit_fill_gap_bps: float = 12.0
    max_p95_quote_spread_bps: float = 24.0
    max_p95_quote_impact_bps: float = 30.0


@dataclass
class ShadowLiveComparatorReport:
    generated_at: str
    trade_count: int
    entry_comparison_count: int
    exit_comparison_count: int
    validation_start: str
    validation_end: str
    validation_runtime_days: float
    entry_quote_coverage_rate: float
    exit_quote_coverage_rate: float
    avg_abs_entry_reference_gap_bps: float
    avg_abs_exit_reference_gap_bps: float
    avg_abs_entry_mid_gap_bps: float
    avg_abs_exit_mid_gap_bps: float
    avg_abs_entry_fill_gap_bps: float
    avg_abs_exit_fill_gap_bps: float
    p95_abs_entry_reference_gap_bps: float
    p95_abs_exit_reference_gap_bps: float
    p95_abs_entry_fill_gap_bps: float
    p95_abs_exit_fill_gap_bps: float
    avg_entry_quote_spread_bps: float
    avg_exit_quote_spread_bps: float
    avg_entry_quote_impact_bps: float
    avg_exit_quote_impact_bps: float
    p95_entry_quote_spread_bps: float
    p95_exit_quote_spread_bps: float
    p95_entry_quote_impact_bps: float
    p95_exit_quote_impact_bps: float
    comparator_score: float
    comparator_level: str
    ready_for_capital: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_shadow_live_observation(
    payload: dict[str, Any] | None,
    *,
    symbol: str,
    direction: int,
    reference_price: float,
    reduce_only: bool = False,
) -> ShadowLiveComparisonObservation | None:
    if not isinstance(payload, dict):
        return None

    best_bid = _float(payload.get("best_bid"))
    best_ask = _float(payload.get("best_ask"))
    mid_price = _float(payload.get("mid_price"))
    if mid_price <= 0.0 and best_bid > 0.0 and best_ask > 0.0:
        mid_price = (best_bid + best_ask) / 2.0

    touch_price = _float(payload.get("touch_price"))
    if touch_price <= 0.0:
        if direction == 1 and best_ask > 0.0:
            touch_price = best_ask
        elif direction == -1 and best_bid > 0.0:
            touch_price = best_bid
        else:
            touch_price = mid_price or _float(payload.get("price"), _float(reference_price))

    if touch_price <= 0.0 and mid_price <= 0.0 and _float(payload.get("price")) <= 0.0:
        return None

    fill_price = _float(payload.get("price"))
    return ShadowLiveComparisonObservation(
        broker=str(payload.get("broker", "shadow")),
        symbol=symbol,
        side="buy" if direction == 1 else "sell",
        direction=direction,
        reference_price=_float(reference_price),
        quote_timestamp=str(payload.get("quote_timestamp", "")),
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        touch_price=touch_price,
        quote_spread_bps=_float(payload.get("spread_bps")),
        quote_impact_bps=_float(payload.get("impact_bps")),
        reference_gap_bps=_gap_bps(_float(reference_price), touch_price),
        mid_gap_bps=_gap_bps(_float(reference_price), mid_price),
        fill_price=fill_price,
        fill_gap_bps=_gap_bps(fill_price, touch_price),
        reduce_only=reduce_only,
    )


def build_shadow_live_comparator_report(
    base_dir: Path,
    thresholds: ShadowLiveComparatorThresholds | None = None,
) -> ShadowLiveComparatorReport:
    thresholds = thresholds or ShadowLiveComparatorThresholds()
    journal_payload = _load_json(base_dir / "trade_journal.json", [])
    journal_rows = [row for row in journal_payload if isinstance(row, dict)][-thresholds.lookback_trade_points :]

    entry_reference_gaps: list[float] = []
    exit_reference_gaps: list[float] = []
    entry_mid_gaps: list[float] = []
    exit_mid_gaps: list[float] = []
    entry_fill_gaps: list[float] = []
    exit_fill_gaps: list[float] = []
    entry_spreads: list[float] = []
    exit_spreads: list[float] = []
    entry_impacts: list[float] = []
    exit_impacts: list[float] = []
    validation_points: list[tuple[float, str]] = []

    entry_comparison_count = 0
    exit_comparison_count = 0

    for row in journal_rows:
        entry_touch = _float(row.get("shadow_live_entry_touch_price"))
        if entry_touch > 0.0:
            entry_comparison_count += 1
            entry_ts_raw = str(row.get("shadow_live_entry_quote_timestamp") or row.get("entry_time") or "")
            entry_ts = _iso_to_ts(entry_ts_raw)
            if entry_ts > 0.0:
                validation_points.append((entry_ts, entry_ts_raw))
            entry_reference_gaps.append(_float(row.get("shadow_live_entry_reference_gap_bps")))
            entry_mid_gaps.append(_float(row.get("shadow_live_entry_mid_gap_bps")))
            entry_fill_gap = _float(row.get("shadow_live_entry_fill_gap_bps"))
            if entry_fill_gap == 0.0 and _float(row.get("shadow_entry_price")) > 0.0:
                entry_fill_gap = _gap_bps(_float(row.get("shadow_entry_price")), entry_touch)
            entry_fill_gaps.append(entry_fill_gap)
            entry_spreads.append(_float(row.get("shadow_live_entry_quote_spread_bps")))
            entry_impacts.append(_float(row.get("shadow_live_entry_quote_impact_bps")))

        exit_touch = _float(row.get("shadow_live_exit_touch_price"))
        if exit_touch > 0.0:
            exit_comparison_count += 1
            exit_ts_raw = str(row.get("shadow_live_exit_quote_timestamp") or row.get("exit_time") or "")
            exit_ts = _iso_to_ts(exit_ts_raw)
            if exit_ts > 0.0:
                validation_points.append((exit_ts, exit_ts_raw))
            exit_reference_gaps.append(_float(row.get("shadow_live_exit_reference_gap_bps")))
            exit_mid_gaps.append(_float(row.get("shadow_live_exit_mid_gap_bps")))
            exit_fill_gap = _float(row.get("shadow_live_exit_fill_gap_bps"))
            if exit_fill_gap == 0.0 and _float(row.get("shadow_exit_price")) > 0.0:
                exit_fill_gap = _gap_bps(_float(row.get("shadow_exit_price")), exit_touch)
            exit_fill_gaps.append(exit_fill_gap)
            exit_spreads.append(_float(row.get("shadow_live_exit_quote_spread_bps")))
            exit_impacts.append(_float(row.get("shadow_live_exit_quote_impact_bps")))

    trade_count = len(journal_rows)
    validation_points.sort(key=lambda item: item[0])
    validation_start = validation_points[0][1] if validation_points else ""
    validation_end = validation_points[-1][1] if validation_points else ""
    validation_runtime_days = (
        max(validation_points[-1][0] - validation_points[0][0], 0.0) / 86400.0
        if len(validation_points) >= 2
        else 0.0
    )
    entry_quote_coverage_rate = entry_comparison_count / max(trade_count, 1) if trade_count else 0.0
    exit_quote_coverage_rate = exit_comparison_count / max(trade_count, 1) if trade_count else 0.0

    avg_abs_entry_reference_gap_bps = _mean_abs(entry_reference_gaps)
    avg_abs_exit_reference_gap_bps = _mean_abs(exit_reference_gaps)
    avg_abs_entry_mid_gap_bps = _mean_abs(entry_mid_gaps)
    avg_abs_exit_mid_gap_bps = _mean_abs(exit_mid_gaps)
    avg_abs_entry_fill_gap_bps = _mean_abs(entry_fill_gaps)
    avg_abs_exit_fill_gap_bps = _mean_abs(exit_fill_gaps)
    p95_abs_entry_reference_gap_bps = _percentile([abs(value) for value in entry_reference_gaps], 95.0)
    p95_abs_exit_reference_gap_bps = _percentile([abs(value) for value in exit_reference_gaps], 95.0)
    p95_abs_entry_fill_gap_bps = _percentile([abs(value) for value in entry_fill_gaps], 95.0)
    p95_abs_exit_fill_gap_bps = _percentile([abs(value) for value in exit_fill_gaps], 95.0)
    avg_entry_quote_spread_bps = _mean(entry_spreads)
    avg_exit_quote_spread_bps = _mean(exit_spreads)
    avg_entry_quote_impact_bps = _mean(entry_impacts)
    avg_exit_quote_impact_bps = _mean(exit_impacts)
    p95_entry_quote_spread_bps = _percentile(entry_spreads, 95.0)
    p95_exit_quote_spread_bps = _percentile(exit_spreads, 95.0)
    p95_entry_quote_impact_bps = _percentile(entry_impacts, 95.0)
    p95_exit_quote_impact_bps = _percentile(exit_impacts, 95.0)

    reasons: list[str] = []
    if entry_comparison_count < thresholds.min_entry_comparisons_for_capital:
        reasons.append(
            f"Need at least {thresholds.min_entry_comparisons_for_capital} entry quote comparisons before capital can trust shadow-live execution."
        )
    if exit_comparison_count < thresholds.min_exit_comparisons_for_capital:
        reasons.append(
            f"Need at least {thresholds.min_exit_comparisons_for_capital} exit quote comparisons before capital can trust shadow-live execution."
        )
    if validation_runtime_days < thresholds.min_validation_runtime_days_for_capital:
        reasons.append(
            f"Shadow-live validation covers only {validation_runtime_days:.1f} days; require {thresholds.min_validation_runtime_days_for_capital:.1f} days before capital can trust broker-quoted execution."
        )
    if entry_quote_coverage_rate < thresholds.min_quote_coverage_rate_for_capital:
        reasons.append(
            f"Entry quote coverage {entry_quote_coverage_rate:.0%} is below the {thresholds.min_quote_coverage_rate_for_capital:.0%} requirement."
        )
    if exit_quote_coverage_rate < thresholds.min_quote_coverage_rate_for_capital:
        reasons.append(
            f"Exit quote coverage {exit_quote_coverage_rate:.0%} is below the {thresholds.min_quote_coverage_rate_for_capital:.0%} requirement."
        )
    if avg_abs_entry_reference_gap_bps > thresholds.max_avg_entry_reference_gap_bps:
        reasons.append(
            f"Average entry reference gap {avg_abs_entry_reference_gap_bps:.2f}bps exceeds threshold."
        )
    if avg_abs_exit_reference_gap_bps > thresholds.max_avg_exit_reference_gap_bps:
        reasons.append(
            f"Average exit reference gap {avg_abs_exit_reference_gap_bps:.2f}bps exceeds threshold."
        )
    if avg_abs_entry_fill_gap_bps > thresholds.max_avg_entry_fill_gap_bps:
        reasons.append(f"Average entry fill gap {avg_abs_entry_fill_gap_bps:.2f}bps exceeds threshold.")
    if avg_abs_exit_fill_gap_bps > thresholds.max_avg_exit_fill_gap_bps:
        reasons.append(f"Average exit fill gap {avg_abs_exit_fill_gap_bps:.2f}bps exceeds threshold.")
    if entry_spreads and avg_entry_quote_spread_bps > thresholds.max_avg_quote_spread_bps:
        reasons.append(f"Average entry quote spread {avg_entry_quote_spread_bps:.2f}bps exceeds threshold.")
    if exit_spreads and avg_exit_quote_spread_bps > thresholds.max_avg_quote_spread_bps:
        reasons.append(f"Average exit quote spread {avg_exit_quote_spread_bps:.2f}bps exceeds threshold.")
    if entry_impacts and avg_entry_quote_impact_bps > thresholds.max_avg_quote_impact_bps:
        reasons.append(f"Average entry quote impact {avg_entry_quote_impact_bps:.2f}bps exceeds threshold.")
    if exit_impacts and avg_exit_quote_impact_bps > thresholds.max_avg_quote_impact_bps:
        reasons.append(f"Average exit quote impact {avg_exit_quote_impact_bps:.2f}bps exceeds threshold.")
    if p95_abs_entry_reference_gap_bps > thresholds.max_p95_entry_reference_gap_bps:
        reasons.append(f"Entry reference-gap p95 {p95_abs_entry_reference_gap_bps:.2f}bps exceeds stability threshold.")
    if p95_abs_exit_reference_gap_bps > thresholds.max_p95_exit_reference_gap_bps:
        reasons.append(f"Exit reference-gap p95 {p95_abs_exit_reference_gap_bps:.2f}bps exceeds stability threshold.")
    if p95_abs_entry_fill_gap_bps > thresholds.max_p95_entry_fill_gap_bps:
        reasons.append(f"Entry fill-gap p95 {p95_abs_entry_fill_gap_bps:.2f}bps exceeds stability threshold.")
    if p95_abs_exit_fill_gap_bps > thresholds.max_p95_exit_fill_gap_bps:
        reasons.append(f"Exit fill-gap p95 {p95_abs_exit_fill_gap_bps:.2f}bps exceeds stability threshold.")
    if p95_entry_quote_spread_bps > thresholds.max_p95_quote_spread_bps:
        reasons.append(f"Entry quote-spread p95 {p95_entry_quote_spread_bps:.2f}bps exceeds stability threshold.")
    if p95_exit_quote_spread_bps > thresholds.max_p95_quote_spread_bps:
        reasons.append(f"Exit quote-spread p95 {p95_exit_quote_spread_bps:.2f}bps exceeds stability threshold.")
    if p95_entry_quote_impact_bps > thresholds.max_p95_quote_impact_bps:
        reasons.append(f"Entry quote-impact p95 {p95_entry_quote_impact_bps:.2f}bps exceeds stability threshold.")
    if p95_exit_quote_impact_bps > thresholds.max_p95_quote_impact_bps:
        reasons.append(f"Exit quote-impact p95 {p95_exit_quote_impact_bps:.2f}bps exceeds stability threshold.")

    penalty = 0.0
    if trade_count == 0:
        penalty += 40.0
    else:
        entry_coverage_gap = max(thresholds.min_quote_coverage_rate_for_capital - entry_quote_coverage_rate, 0.0)
        exit_coverage_gap = max(thresholds.min_quote_coverage_rate_for_capital - exit_quote_coverage_rate, 0.0)
        penalty += (entry_coverage_gap / max(thresholds.min_quote_coverage_rate_for_capital, 1e-9)) * 16.0
        penalty += (exit_coverage_gap / max(thresholds.min_quote_coverage_rate_for_capital, 1e-9)) * 12.0
        runtime_gap = max(thresholds.min_validation_runtime_days_for_capital - validation_runtime_days, 0.0)
        penalty += (runtime_gap / max(thresholds.min_validation_runtime_days_for_capital, 1e-9)) * 18.0
    penalty += _scaled_pressure(avg_abs_entry_reference_gap_bps, thresholds.max_avg_entry_reference_gap_bps) * 16.0
    penalty += _scaled_pressure(avg_abs_exit_reference_gap_bps, thresholds.max_avg_exit_reference_gap_bps) * 10.0
    penalty += _scaled_pressure(avg_abs_entry_fill_gap_bps, thresholds.max_avg_entry_fill_gap_bps) * 14.0
    penalty += _scaled_pressure(avg_abs_exit_fill_gap_bps, thresholds.max_avg_exit_fill_gap_bps) * 10.0
    penalty += _scaled_pressure(p95_abs_entry_reference_gap_bps, thresholds.max_p95_entry_reference_gap_bps) * 10.0
    penalty += _scaled_pressure(p95_abs_exit_reference_gap_bps, thresholds.max_p95_exit_reference_gap_bps) * 8.0
    penalty += _scaled_pressure(p95_abs_entry_fill_gap_bps, thresholds.max_p95_entry_fill_gap_bps) * 10.0
    penalty += _scaled_pressure(p95_abs_exit_fill_gap_bps, thresholds.max_p95_exit_fill_gap_bps) * 8.0
    if entry_spreads:
        penalty += _scaled_pressure(avg_entry_quote_spread_bps, thresholds.max_avg_quote_spread_bps) * 8.0
        penalty += _scaled_pressure(p95_entry_quote_spread_bps, thresholds.max_p95_quote_spread_bps) * 6.0
    if exit_spreads:
        penalty += _scaled_pressure(avg_exit_quote_spread_bps, thresholds.max_avg_quote_spread_bps) * 6.0
        penalty += _scaled_pressure(p95_exit_quote_spread_bps, thresholds.max_p95_quote_spread_bps) * 4.0
    if entry_impacts:
        penalty += _scaled_pressure(avg_entry_quote_impact_bps, thresholds.max_avg_quote_impact_bps) * 6.0
        penalty += _scaled_pressure(p95_entry_quote_impact_bps, thresholds.max_p95_quote_impact_bps) * 4.0
    if exit_impacts:
        penalty += _scaled_pressure(avg_exit_quote_impact_bps, thresholds.max_avg_quote_impact_bps) * 4.0
        penalty += _scaled_pressure(p95_exit_quote_impact_bps, thresholds.max_p95_quote_impact_bps) * 3.0
    if entry_comparison_count < thresholds.min_entry_comparisons_for_capital:
        penalty += 10.0
    if exit_comparison_count < thresholds.min_exit_comparisons_for_capital:
        penalty += 8.0
    if validation_runtime_days < thresholds.min_validation_runtime_days_for_capital:
        penalty += 12.0

    comparator_score = max(0.0, 100.0 - penalty)
    if comparator_score >= 80.0:
        comparator_level = "stable"
    elif comparator_score >= 60.0:
        comparator_level = "watch"
    else:
        comparator_level = "unstable"

    return ShadowLiveComparatorReport(
        generated_at=_now_iso(),
        trade_count=trade_count,
        entry_comparison_count=entry_comparison_count,
        exit_comparison_count=exit_comparison_count,
        validation_start=validation_start,
        validation_end=validation_end,
        validation_runtime_days=validation_runtime_days,
        entry_quote_coverage_rate=entry_quote_coverage_rate,
        exit_quote_coverage_rate=exit_quote_coverage_rate,
        avg_abs_entry_reference_gap_bps=avg_abs_entry_reference_gap_bps,
        avg_abs_exit_reference_gap_bps=avg_abs_exit_reference_gap_bps,
        avg_abs_entry_mid_gap_bps=avg_abs_entry_mid_gap_bps,
        avg_abs_exit_mid_gap_bps=avg_abs_exit_mid_gap_bps,
        avg_abs_entry_fill_gap_bps=avg_abs_entry_fill_gap_bps,
        avg_abs_exit_fill_gap_bps=avg_abs_exit_fill_gap_bps,
        p95_abs_entry_reference_gap_bps=p95_abs_entry_reference_gap_bps,
        p95_abs_exit_reference_gap_bps=p95_abs_exit_reference_gap_bps,
        p95_abs_entry_fill_gap_bps=p95_abs_entry_fill_gap_bps,
        p95_abs_exit_fill_gap_bps=p95_abs_exit_fill_gap_bps,
        avg_entry_quote_spread_bps=avg_entry_quote_spread_bps,
        avg_exit_quote_spread_bps=avg_exit_quote_spread_bps,
        avg_entry_quote_impact_bps=avg_entry_quote_impact_bps,
        avg_exit_quote_impact_bps=avg_exit_quote_impact_bps,
        p95_entry_quote_spread_bps=p95_entry_quote_spread_bps,
        p95_exit_quote_spread_bps=p95_exit_quote_spread_bps,
        p95_entry_quote_impact_bps=p95_entry_quote_impact_bps,
        p95_exit_quote_impact_bps=p95_exit_quote_impact_bps,
        comparator_score=comparator_score,
        comparator_level=comparator_level,
        ready_for_capital=not reasons,
        reasons=reasons,
    )


def write_shadow_live_comparator_report(report: ShadowLiveComparatorReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))


def format_shadow_live_comparator_report(report: ShadowLiveComparatorReport) -> str:
    lines = [
        "Shadow Live Comparator Report",
        f"  Comparator: {report.comparator_score:.1f}/100 ({report.comparator_level}) | ready={'YES' if report.ready_for_capital else 'NO'}",
        f"  Burn-in: {report.validation_runtime_days:.1f}d across {report.trade_count} completed trades",
        f"  Coverage: entry={report.entry_comparison_count}/{report.trade_count} ({report.entry_quote_coverage_rate:.0%}) | exit={report.exit_comparison_count}/{report.trade_count} ({report.exit_quote_coverage_rate:.0%})",
        f"  Reference gaps: entry={report.avg_abs_entry_reference_gap_bps:.2f}bps | exit={report.avg_abs_exit_reference_gap_bps:.2f}bps",
        f"  Fill gaps: entry={report.avg_abs_entry_fill_gap_bps:.2f}bps | exit={report.avg_abs_exit_fill_gap_bps:.2f}bps",
        f"  Quote friction: spread entry={report.avg_entry_quote_spread_bps:.2f}bps exit={report.avg_exit_quote_spread_bps:.2f}bps | impact entry={report.avg_entry_quote_impact_bps:.2f}bps exit={report.avg_exit_quote_impact_bps:.2f}bps",
        f"  Tail stability: ref p95 entry={report.p95_abs_entry_reference_gap_bps:.2f}bps exit={report.p95_abs_exit_reference_gap_bps:.2f}bps | fill p95 entry={report.p95_abs_entry_fill_gap_bps:.2f}bps exit={report.p95_abs_exit_fill_gap_bps:.2f}bps | spread p95={max(report.p95_entry_quote_spread_bps, report.p95_exit_quote_spread_bps):.2f}bps | impact p95={max(report.p95_entry_quote_impact_bps, report.p95_exit_quote_impact_bps):.2f}bps",
    ]
    if report.reasons:
        lines.append("  Comparator reasons:")
        lines.extend(f"    - {reason}" for reason in report.reasons)
    return "\n".join(lines)


__all__ = [
    "ShadowLiveComparatorReport",
    "ShadowLiveComparatorThresholds",
    "ShadowLiveComparisonObservation",
    "build_shadow_live_comparator_report",
    "build_shadow_live_observation",
    "format_shadow_live_comparator_report",
    "write_shadow_live_comparator_report",
]
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PaperValidationThresholds:
    min_runtime_days: float = 14.0
    min_cycle_count: int = 100
    min_trade_count: int = 20
    max_drawdown: float = 0.08
    max_mean_tracking_error: float = 0.08
    max_avg_entry_slippage_bps: float = 8.0
    max_miss_rate: float = 0.10
    max_abs_avg_pnl_divergence_pct: float = 20.0
    max_pid_flip_rate: float = 0.35
    min_mean_edge_retention: float = 0.60
    max_fragile_edge_fraction: float = 0.15
    min_mean_objective_score: float = -0.05
    pid_flip_deadband: float = 0.05


@dataclass
class PaperValidationReport:
    ready_for_live: bool
    generated_at: str
    run_days: float
    cycle_count: int
    trade_count: int
    max_drawdown: float
    mean_tracking_error: float
    mean_edge_retention: float
    fragile_edge_fraction: float
    mean_objective_score: float
    pid_flip_rate: float
    avg_entry_slippage_bps: float
    miss_rate: float
    avg_pnl_divergence_pct: float
    latest_safety_action: str
    latest_timestamp: str = ""
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
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


def _timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _pid_flip_rate(values: list[float], deadband: float) -> float:
    signs: list[int] = []
    for value in values:
        if abs(float(value)) < deadband:
            continue
        signs.append(1 if value > 0 else -1)
    if len(signs) <= 1:
        return 0.0
    flips = sum(1 for idx in range(1, len(signs)) if signs[idx] != signs[idx - 1])
    return float(flips / (len(signs) - 1))


def build_paper_validation_report(
    base_dir: Path = Path("fund_data"),
    thresholds: PaperValidationThresholds | None = None,
) -> PaperValidationReport:
    thresholds = thresholds or PaperValidationThresholds()
    cycle_rows = _load_jsonl(base_dir / "adaptive_cycle_ledger.jsonl")
    journal = _load_json(base_dir / "trade_journal.json", [])
    state = _load_json(base_dir / "live_state.json", {})

    timestamps = [_timestamp(row.get("timestamp")) for row in cycle_rows]
    clean_timestamps = [ts for ts in timestamps if ts is not None]
    run_days = 0.0
    if len(clean_timestamps) >= 2:
        run_days = max((clean_timestamps[-1] - clean_timestamps[0]).total_seconds() / 86400.0, 0.0)

    tracking_errors = [float(row.get("volatility_tracking_error", 0.0) or 0.0) for row in cycle_rows]
    edge_retention = [float(row.get("edge_retention_ratio", 0.0) or 0.0) for row in cycle_rows]
    objective_scores = [float(row.get("portfolio_objective_score", 0.0) or 0.0) for row in cycle_rows]
    drawdowns = [float(row.get("current_drawdown", 0.0) or 0.0) for row in cycle_rows]
    pid_outputs = [float(row.get("pid_output", 0.0) or 0.0) for row in cycle_rows]
    fragile_fraction = 0.0
    if cycle_rows:
        fragile_fraction = sum(
            1
            for row in cycle_rows
            if str(row.get("edge_retention_state", "unknown")) in {"fragile", "broken"}
        ) / len(cycle_rows)

    latest_execution = {}
    if cycle_rows:
        latest_execution = dict(cycle_rows[-1].get("execution", {}) or {})
    total_exec = int(latest_execution.get("total_trades", 0) or 0)
    total_missed = int(latest_execution.get("total_missed", 0) or 0)
    miss_rate = float(total_missed / total_exec) if total_exec > 0 else 0.0
    avg_entry_slippage_bps = float(latest_execution.get("avg_entry_slippage_bps", 0.0) or 0.0)
    avg_pnl_divergence_pct = float(latest_execution.get("avg_pnl_divergence_pct", 0.0) or 0.0)
    latest_safety_action = str(cycle_rows[-1].get("safety_action", "unknown")) if cycle_rows else "unknown"
    latest_timestamp = str(cycle_rows[-1].get("timestamp", "")) if cycle_rows else str(state.get("timestamp", ""))

    report = PaperValidationReport(
        ready_for_live=True,
        generated_at=datetime.utcnow().isoformat() + "Z",
        run_days=float(run_days),
        cycle_count=len(cycle_rows),
        trade_count=len(journal) if isinstance(journal, list) else 0,
        max_drawdown=max(drawdowns) if drawdowns else 0.0,
        mean_tracking_error=(sum(tracking_errors) / len(tracking_errors)) if tracking_errors else 0.0,
        mean_edge_retention=(sum(edge_retention) / len(edge_retention)) if edge_retention else 0.0,
        fragile_edge_fraction=float(fragile_fraction),
        mean_objective_score=(sum(objective_scores) / len(objective_scores)) if objective_scores else 0.0,
        pid_flip_rate=_pid_flip_rate(pid_outputs, thresholds.pid_flip_deadband),
        avg_entry_slippage_bps=avg_entry_slippage_bps,
        miss_rate=miss_rate,
        avg_pnl_divergence_pct=avg_pnl_divergence_pct,
        latest_safety_action=latest_safety_action,
        latest_timestamp=latest_timestamp,
    )

    reasons: list[str] = []
    if report.run_days < thresholds.min_runtime_days:
        reasons.append(f"runtime {report.run_days:.1f}d < required {thresholds.min_runtime_days:.1f}d")
    if report.cycle_count < thresholds.min_cycle_count:
        reasons.append(f"cycles {report.cycle_count} < required {thresholds.min_cycle_count}")
    if report.trade_count < thresholds.min_trade_count:
        reasons.append(f"trades {report.trade_count} < required {thresholds.min_trade_count}")
    if report.max_drawdown > thresholds.max_drawdown:
        reasons.append(f"max drawdown {report.max_drawdown:.1%} > {thresholds.max_drawdown:.1%}")
    if report.mean_tracking_error > thresholds.max_mean_tracking_error:
        reasons.append(
            f"mean vol tracking error {report.mean_tracking_error:.3f} > {thresholds.max_mean_tracking_error:.3f}"
        )
    if report.mean_edge_retention < thresholds.min_mean_edge_retention:
        reasons.append(
            f"mean edge retention {report.mean_edge_retention:.2f} < {thresholds.min_mean_edge_retention:.2f}"
        )
    if report.fragile_edge_fraction > thresholds.max_fragile_edge_fraction:
        reasons.append(
            f"fragile edge fraction {report.fragile_edge_fraction:.0%} > {thresholds.max_fragile_edge_fraction:.0%}"
        )
    if report.mean_objective_score < thresholds.min_mean_objective_score:
        reasons.append(
            f"mean objective score {report.mean_objective_score:.3f} < {thresholds.min_mean_objective_score:.3f}"
        )
    if report.pid_flip_rate > thresholds.max_pid_flip_rate:
        reasons.append(
            f"pid flip rate {report.pid_flip_rate:.0%} > {thresholds.max_pid_flip_rate:.0%}"
        )
    if report.avg_entry_slippage_bps > thresholds.max_avg_entry_slippage_bps:
        reasons.append(
            f"avg entry slippage {report.avg_entry_slippage_bps:.1f}bps > {thresholds.max_avg_entry_slippage_bps:.1f}bps"
        )
    if report.miss_rate > thresholds.max_miss_rate:
        reasons.append(f"miss rate {report.miss_rate:.0%} > {thresholds.max_miss_rate:.0%}")
    if abs(report.avg_pnl_divergence_pct) > thresholds.max_abs_avg_pnl_divergence_pct:
        reasons.append(
            f"avg pnl divergence {report.avg_pnl_divergence_pct:+.1f}% > {thresholds.max_abs_avg_pnl_divergence_pct:.1f}%"
        )
    if report.latest_safety_action != "allow":
        reasons.append(f"latest safety action is {report.latest_safety_action}")

    report.reasons = reasons
    report.ready_for_live = len(reasons) == 0
    return report


def write_paper_validation_report(
    report: PaperValidationReport,
    path: Path = Path("fund_data/paper_validation_status.json"),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))


def format_paper_validation_report(report: PaperValidationReport) -> str:
    lines = [
        "=" * 60,
        "  STRICT PAPER VALIDATION REPORT",
        "=" * 60,
        f"  Ready for live:    {'YES' if report.ready_for_live else 'NO'}",
        f"  Runtime days:      {report.run_days:.1f}",
        f"  Cycle count:       {report.cycle_count}",
        f"  Trade count:       {report.trade_count}",
        f"  Max drawdown:      {report.max_drawdown:.1%}",
        f"  Mean tracking err: {report.mean_tracking_error:.3f}",
        f"  Mean edge retain:  {report.mean_edge_retention:.2f}",
        f"  Fragile edge frac: {report.fragile_edge_fraction:.0%}",
        f"  Mean objective:    {report.mean_objective_score:+.3f}",
        f"  PID flip rate:     {report.pid_flip_rate:.0%}",
        f"  Avg entry slip:    {report.avg_entry_slippage_bps:.1f}bps",
        f"  Miss rate:         {report.miss_rate:.0%}",
        f"  Avg pnl diverge:   {report.avg_pnl_divergence_pct:+.1f}%",
        f"  Latest safety:     {report.latest_safety_action}",
    ]
    if report.reasons:
        lines.append("")
        lines.append("  Blocking reasons:")
        for reason in report.reasons:
            lines.append(f"    - {reason}")
    lines.append("=" * 60)
    return "\n".join(lines)
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ops.dashboard_data import (
    DEFAULT_ASSETS,
    build_live_readiness_snapshot,
    compute_signal_proximity,
    load_health,
    load_journal,
    load_market_snapshot,
    load_state,
    portfolio_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = REPO_ROOT / "fund_data"
CONSOLE = Console()


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _compact_money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if abs(number) >= 1_000_000:
        return f"${number / 1_000_000:.2f}M"
    if abs(number) >= 1_000:
        return f"${number / 1_000:.2f}k"
    return f"${number:,.2f}"


def _pct(value: Any, *, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:+.2%}" if signed else f"{number:.2%}"


def _float_text(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _short_ts(value: Any) -> str:
    if not value:
        return "n/a"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _next_scan_text() -> str:
    seconds = int(3600 - (time.time() % 3600) + 10)
    minutes = seconds // 60
    return f"{minutes} min"


def _style_for_gate(allowed: bool) -> str:
    return "green" if allowed else "red"


def _style_for_health(status: str) -> str:
    mapping = {
        "ok": "green",
        "warning": "yellow",
        "halt": "red",
        "critical": "red",
        "unknown": "dim",
    }
    return mapping.get(status.lower(), "white")


def _style_for_action(action: str) -> str:
    mapping = {
        "allow": "green",
        "reduce": "yellow",
        "pause_entries": "yellow",
        "halt": "red",
    }
    return mapping.get(action, "white")


def _top_signal_text(proximity: dict[str, float]) -> str:
    if not proximity:
        return "n/a"
    strategy, value = max(proximity.items(), key=lambda item: item[1])
    labels = {
        "funding_mr_v7": "MR",
        "extreme_spike": "XSp",
        "fund_vol_squeeze": "Sqz",
        "momentum_breakout": "Mom",
    }
    return f"{labels.get(strategy, strategy)} {value:.0%}"


def _build_overview_panel(state: dict[str, Any], journal: list[dict[str, Any]]) -> Panel:
    summary = portfolio_summary(state, journal)
    table = Table.grid(expand=True)
    table.add_column(style="bold cyan")
    table.add_column(justify="right")

    table.add_row("Capital", _money(summary.get("capital")))
    table.add_row("Return", _pct(summary.get("return_pct"), signed=True))
    table.add_row("Open Positions", str(summary.get("n_open", 0)))
    table.add_row("Closed Trades", str(summary.get("n_closed", 0)))
    table.add_row("Iterations", str(summary.get("iteration", 0)))
    table.add_row("Mode", str(state.get("operating_mode", "paper")).upper())
    table.add_row("Next Scan", _next_scan_text())
    table.add_row("State Timestamp", _short_ts(state.get("timestamp")))

    return Panel(table, title="Overview", border_style="cyan")


def _build_readiness_panel(readiness: dict[str, Any]) -> Panel:
    gates = readiness.get("gates", {}) if isinstance(readiness.get("gates"), dict) else {}
    rollout = readiness.get("rollout_plan", {}) if isinstance(readiness.get("rollout_plan"), dict) else {}
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Gate", style="bold")
    table.add_column("Status")
    table.add_column("Trades Left", justify="right")
    table.add_column("Days Left", justify="right")

    for key, label in [
        ("shadow_live", "Shadow"),
        ("probation_live", "Probation"),
        ("full_live", "Full Live"),
    ]:
        gate = gates.get(key, {}) if isinstance(gates.get(key), dict) else {}
        allowed = bool(gate.get("allowed"))
        table.add_row(
            label,
            f"[{_style_for_gate(allowed)}]{'GO' if allowed else 'NO-GO'}[/]",
            str(int(gate.get("trades_remaining", 0) or 0)),
            f"{float(gate.get('days_remaining', 0.0) or 0.0):.1f}d",
        )

    summaries = readiness.get("one_line_summaries", {}) if isinstance(readiness.get("one_line_summaries"), dict) else {}
    details = [
        Text(str(readiness.get("headline", "Live readiness snapshot unavailable.")), style="bold"),
        Text(f"Allowed mode: {readiness.get('allowed_mode', 'blocked')}") ,
        Text(
            f"Current tranche: {rollout.get('current_stage_label', 'Paper / Shadow')} @ {_money(rollout.get('starting_size_usd', 0.0))}/trade",
            style="cyan",
        ),
    ]
    for key in ["shadow_live", "probation_live", "full_live"]:
        if summaries.get(key):
            details.append(Text(f"- {summaries[key]}", style="dim"))

    return Panel(Group(*details, table), title="Readiness", border_style="magenta")


def _build_adaptive_panel(state: dict[str, Any], health: dict[str, Any]) -> Panel:
    adaptive = state.get("adaptive_cycle", {}) if isinstance(state.get("adaptive_cycle"), dict) else {}
    validation = state.get("paper_validation", {}) if isinstance(state.get("paper_validation"), dict) else {}
    table = Table.grid(expand=True)
    table.add_column(style="bold cyan")
    table.add_column(justify="right")

    action = str(adaptive.get("safety_action", "unknown"))
    health_status = str(health.get("overall_status", "unknown"))
    table.add_row("Safety Action", f"[{_style_for_action(action)}]{action}[/]")
    table.add_row("Tracking Error", _float_text(adaptive.get("volatility_tracking_error")))
    table.add_row("Objective Score", _float_text(adaptive.get("portfolio_objective_score")))
    table.add_row("Edge Retention", _pct(adaptive.get("edge_retention_ratio")))
    table.add_row("Target Vol", _pct(adaptive.get("target_volatility")))
    table.add_row("Realized Vol", _pct(adaptive.get("realized_volatility")))
    table.add_row("Health", f"[{_style_for_health(health_status)}]{health_status}[/]")
    table.add_row("Should Halt", str(bool(health.get("should_halt"))))
    table.add_row("Validation Cycles", str(validation.get("cycle_count", 0)))
    table.add_row("Validation Trades", str(validation.get("trade_count", 0)))
    table.add_row("Validation Runtime", f"{float(validation.get('run_days', 0.0) or 0.0):.1f}d")

    notes: list[Text] = []
    safety_reasons = adaptive.get("safety_reasons", []) if isinstance(adaptive.get("safety_reasons"), list) else []
    validation_reasons = validation.get("reasons", []) if isinstance(validation.get("reasons"), list) else []
    if safety_reasons:
        notes.append(Text("Safety reasons: " + "; ".join(str(item) for item in safety_reasons[:3]), style="yellow"))
    if validation_reasons:
        notes.append(Text("Validation blockers: " + "; ".join(str(item) for item in validation_reasons[:3]), style="yellow"))
    if health.get("halt_reason"):
        notes.append(Text(f"Halt reason: {health['halt_reason']}", style="red"))

    body: list[Any] = [table]
    if notes:
        body.append(Text(""))
        body.extend(notes)

    return Panel(Group(*body), title="Adaptive & Health", border_style="green")


def _build_signals_panel(snapshots: dict[str, Any]) -> Panel:
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Asset", style="bold")
    table.add_column("Px", justify="right")
    table.add_column("Regime")
    table.add_column("FZ", justify="right")
    table.add_column("MR", justify="right")
    table.add_column("XSp", justify="right")
    table.add_column("Squeeze", justify="right")
    table.add_column("Mom", justify="right")
    table.add_column("Top Signal", justify="right")

    for symbol in DEFAULT_ASSETS:
        snapshot = snapshots.get(symbol, {}) if isinstance(snapshots.get(symbol), dict) else {}
        ticker = symbol.split("/")[0]
        if snapshot.get("error"):
            table.add_row(ticker, "n/a", str(snapshot.get("error")), "-", "-", "-", "-", "-", "error")
            continue

        proximity = compute_signal_proximity(symbol, snapshot)
        table.add_row(
            ticker,
            _compact_money(snapshot.get("price")),
            str(snapshot.get("regime", "n/a")),
            _float_text(snapshot.get("funding_zscore"), 2),
            f"{proximity.get('funding_mr_v7', 0.0):.0%}",
            f"{proximity.get('extreme_spike', 0.0):.0%}",
            f"{proximity.get('fund_vol_squeeze', 0.0):.0%}",
            f"{proximity.get('momentum_breakout', 0.0):.0%}",
            _top_signal_text(proximity),
        )

    return Panel(table, title="Signal Proximity", border_style="blue")


def _build_positions_panel(state: dict[str, Any], journal: list[dict[str, Any]], trade_limit: int) -> Panel:
    positions = state.get("open_positions", []) if isinstance(state.get("open_positions"), list) else []
    content: list[Any] = []

    if positions:
        open_table = Table(box=box.SIMPLE_HEAVY, expand=True)
        open_table.add_column("Symbol", style="bold")
        open_table.add_column("Strategy")
        open_table.add_column("Dir", justify="center")
        open_table.add_column("Size", justify="right")
        open_table.add_column("Entry", justify="right")
        open_table.add_column("uPnL", justify="right")
        for position in positions:
            direction = int(position.get("direction", 0) or 0)
            open_table.add_row(
                str(position.get("symbol", "?")),
                str(position.get("strategy", "?")),
                "LONG" if direction >= 0 else "SHORT",
                _money(position.get("size_usd", 0.0)),
                _money(position.get("entry_price", 0.0)),
                _money(position.get("unrealized_pnl", 0.0)),
            )
        content.append(Text("Open Positions", style="bold cyan"))
        content.append(open_table)
    else:
        content.append(Text("Open Positions: none", style="dim"))

    content.append(Text(""))

    recent = [row for row in journal if isinstance(row, dict)][-trade_limit:]
    recent.reverse()
    if recent:
        trade_table = Table(box=box.SIMPLE_HEAVY, expand=True)
        trade_table.add_column("Exit Time")
        trade_table.add_column("Symbol", style="bold")
        trade_table.add_column("Strategy")
        trade_table.add_column("PnL", justify="right")
        trade_table.add_column("PnL %", justify="right")
        trade_table.add_column("Reason")
        for row in recent:
            trade_table.add_row(
                _short_ts(row.get("exit_time")),
                str(row.get("symbol", "?")),
                str(row.get("strategy", "?")),
                _money(row.get("pnl", 0.0)),
                _pct(row.get("pnl_pct", 0.0), signed=True),
                str(row.get("exit_reason", "")),
            )
        content.append(Text("Recent Closed Trades", style="bold cyan"))
        content.append(trade_table)
    else:
        content.append(Text("Recent Closed Trades: none", style="dim"))

    return Panel(Group(*content), title="Positions & Trades", border_style="yellow")


def _build_footer(base_dir: Path, interval: float) -> Text:
    footer = Text()
    footer.append(f"Source: {base_dir}", style="dim")
    footer.append("  |  ", style="dim")
    footer.append(f"Refresh: {interval:.0f}s", style="dim")
    footer.append("  |  Ctrl+C to stop", style="dim")
    return footer


def render_dashboard(base_dir: Path, trade_limit: int, interval: float) -> Group:
    state = load_state(base_dir)
    journal = load_journal(base_dir)
    health = load_health(base_dir)
    readiness = build_live_readiness_snapshot(base_dir)
    snapshots = load_market_snapshot(base_dir)

    header = Text("SignalForge Terminal Dashboard", style="bold white on dark_green")
    header.append(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", style="bold")

    top = Columns(
        [
            _build_overview_panel(state, journal),
            _build_readiness_panel(readiness),
            _build_adaptive_panel(state, health),
        ],
        expand=True,
    )
    return Group(header, top, _build_signals_panel(snapshots), _build_positions_panel(state, journal, trade_limit), _build_footer(base_dir, interval))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the live dashboard directly in the terminal.")
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="Directory containing live state and dashboard artifacts",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=15.0,
        help="Refresh interval in seconds when running live",
    )
    parser.add_argument(
        "--trade-limit",
        type=int,
        default=5,
        help="How many recent closed trades to show",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render one snapshot and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    interval = max(float(args.interval), 1.0)
    trade_limit = max(int(args.trade_limit), 1)

    if args.once:
        CONSOLE.print(render_dashboard(base_dir, trade_limit, interval))
        return 0

    try:
        with Live(
            render_dashboard(base_dir, trade_limit, interval),
            console=CONSOLE,
            auto_refresh=False,
            screen=False,
            transient=False,
        ) as live:
            while True:
                time.sleep(interval)
                live.update(render_dashboard(base_dir, trade_limit, interval), refresh=True)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
Live vs Backtest Divergence Tracker
====================================
Tracks execution quality in real-time by comparing:

    1. Expected entry price vs actual fill price
    2. Expected PnL vs realized PnL
    3. Slippage drift over time
    4. Missed trades (signals that couldn't be executed)
    5. Fill rate by strategy and asset

Designed for paper trading and live trading validation.
Persists to JSON for cross-session analysis.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeComparison:
    """Single trade comparison: backtest expectation vs reality."""
    strategy: str
    symbol: str
    timestamp: str
    direction: int
    # Backtest expectations
    expected_entry: float = 0.0
    expected_exit: float = 0.0
    expected_pnl: float = 0.0
    # Actual execution
    actual_entry: float = 0.0
    actual_exit: float = 0.0
    actual_pnl: float = 0.0
    # Divergence
    entry_slippage_bps: float = 0.0
    exit_slippage_bps: float = 0.0
    pnl_divergence_pct: float = 0.0
    # Execution details
    algo_used: str = "market"
    fill_time_ms: float = 0.0
    was_partial: bool = False
    missed: bool = False
    miss_reason: str = ""


@dataclass
class DivergenceStats:
    """Aggregate divergence statistics."""
    total_trades: int = 0
    total_missed: int = 0
    avg_entry_slippage_bps: float = 0.0
    avg_exit_slippage_bps: float = 0.0
    avg_pnl_divergence_pct: float = 0.0
    # By strategy
    per_strategy: dict = field(default_factory=dict)
    # By asset
    per_asset: dict = field(default_factory=dict)
    # Trend (is slippage getting worse?)
    slippage_trend: float = 0.0  # Slope of slippage over time
    # Alerts
    alerts: list = field(default_factory=list)


class DivergenceTracker:
    """Track and analyze live vs backtest performance divergence.

    Usage (paper trading):
        tracker = DivergenceTracker()

        # When a signal fires:
        tracker.record_signal(strategy, symbol, expected_entry, direction)

        # When the trade is filled:
        tracker.record_fill(strategy, symbol, actual_entry, algo_used)

        # When the trade closes:
        tracker.record_close(strategy, symbol, expected_exit, actual_exit,
                           expected_pnl, actual_pnl)

        # Periodic check:
        stats = tracker.get_stats()
        if stats.alerts:
            print("WARNING:", stats.alerts)
    """

    def __init__(
        self,
        persist_path: str = "fund_data/divergence_log.json",
        alert_slippage_bps: float = 10.0,     # Alert if avg slippage > 10 bps
        alert_pnl_diverge_pct: float = 20.0,  # Alert if PnL diverges > 20%
        alert_miss_rate: float = 0.15,         # Alert if miss rate > 15%
        alert_trend_threshold: float = 0.5,    # Alert if slippage trending up
    ):
        self.persist_path = persist_path
        self.alert_slippage_bps = alert_slippage_bps
        self.alert_pnl_diverge_pct = alert_pnl_diverge_pct
        self.alert_miss_rate = alert_miss_rate
        self.alert_trend_threshold = alert_trend_threshold

        # Active signals awaiting fills
        self._pending: dict[str, dict] = {}  # key → signal info
        # Completed comparisons
        self.comparisons: list[TradeComparison] = []

        # Load existing log
        self._load()

    def record_signal(
        self,
        strategy: str,
        symbol: str,
        expected_entry: float,
        direction: int,
    ):
        """Record that a signal fired (before execution)."""
        key = f"{strategy}|{symbol}"
        self._pending[key] = {
            "strategy": strategy,
            "symbol": symbol,
            "expected_entry": expected_entry,
            "direction": direction,
            "signal_time": datetime.now().isoformat(),
        }

    def record_fill(
        self,
        strategy: str,
        symbol: str,
        actual_entry: float,
        algo_used: str = "market",
        fill_time_ms: float = 0,
        was_partial: bool = False,
    ):
        """Record that an order was filled."""
        key = f"{strategy}|{symbol}"
        pending = self._pending.get(key)
        if pending is None:
            return

        # Compute entry slippage
        expected = pending["expected_entry"]
        direction = pending["direction"]

        if expected > 0:
            if direction == 1:
                slippage_bps = (actual_entry / expected - 1) * 10000
            else:
                slippage_bps = (expected / actual_entry - 1) * 10000
        else:
            slippage_bps = 0

        pending["actual_entry"] = actual_entry
        pending["entry_slippage_bps"] = slippage_bps
        pending["algo_used"] = algo_used
        pending["fill_time_ms"] = fill_time_ms
        pending["was_partial"] = was_partial

    def record_miss(
        self,
        strategy: str,
        symbol: str,
        reason: str = "no fill",
    ):
        """Record a missed trade (signal fired but no execution)."""
        key = f"{strategy}|{symbol}"
        pending = self._pending.pop(key, None)

        comp = TradeComparison(
            strategy=strategy,
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            direction=pending["direction"] if pending else 0,
            expected_entry=pending["expected_entry"] if pending else 0,
            missed=True,
            miss_reason=reason,
        )
        self.comparisons.append(comp)
        self._save()

    def record_close(
        self,
        strategy: str,
        symbol: str,
        expected_exit: float,
        actual_exit: float,
        expected_pnl: float,
        actual_pnl: float,
    ):
        """Record a completed trade with full comparison."""
        key = f"{strategy}|{symbol}"
        pending = self._pending.pop(key, {})

        # Exit slippage
        direction = pending.get("direction", 1)
        if expected_exit > 0:
            if direction == 1:
                exit_slip = (expected_exit / actual_exit - 1) * 10000
            else:
                exit_slip = (actual_exit / expected_exit - 1) * 10000
        else:
            exit_slip = 0

        # PnL divergence
        if expected_pnl != 0:
            pnl_div = (actual_pnl - expected_pnl) / abs(expected_pnl) * 100
        else:
            pnl_div = 0

        comp = TradeComparison(
            strategy=strategy,
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            direction=direction,
            expected_entry=pending.get("expected_entry", 0),
            expected_exit=expected_exit,
            expected_pnl=expected_pnl,
            actual_entry=pending.get("actual_entry", 0),
            actual_exit=actual_exit,
            actual_pnl=actual_pnl,
            entry_slippage_bps=pending.get("entry_slippage_bps", 0),
            exit_slippage_bps=exit_slip,
            pnl_divergence_pct=pnl_div,
            algo_used=pending.get("algo_used", "market"),
            fill_time_ms=pending.get("fill_time_ms", 0),
            was_partial=pending.get("was_partial", False),
        )

        self.comparisons.append(comp)
        self._save()

    def get_stats(self) -> DivergenceStats:
        """Compute aggregate divergence statistics + alerts."""
        stats = DivergenceStats()

        executed = [c for c in self.comparisons if not c.missed]
        missed = [c for c in self.comparisons if c.missed]

        stats.total_trades = len(self.comparisons)
        stats.total_missed = len(missed)

        if not executed:
            return stats

        # Averages
        entry_slips = [c.entry_slippage_bps for c in executed]
        exit_slips = [c.exit_slippage_bps for c in executed]
        pnl_divs = [c.pnl_divergence_pct for c in executed if c.expected_pnl != 0]

        stats.avg_entry_slippage_bps = float(np.mean(entry_slips))
        stats.avg_exit_slippage_bps = float(np.mean(exit_slips))
        stats.avg_pnl_divergence_pct = float(np.mean(pnl_divs)) if pnl_divs else 0

        # Per strategy
        for c in executed:
            key = c.strategy
            if key not in stats.per_strategy:
                stats.per_strategy[key] = {
                    "trades": 0, "avg_entry_slip": [],
                    "avg_pnl_div": [], "misses": 0,
                }
            stats.per_strategy[key]["trades"] += 1
            stats.per_strategy[key]["avg_entry_slip"].append(c.entry_slippage_bps)
            stats.per_strategy[key]["avg_pnl_div"].append(c.pnl_divergence_pct)

        for c in missed:
            key = c.strategy
            if key not in stats.per_strategy:
                stats.per_strategy[key] = {
                    "trades": 0, "avg_entry_slip": [],
                    "avg_pnl_div": [], "misses": 0,
                }
            stats.per_strategy[key]["misses"] += 1

        # Compute per-strategy averages
        for key in stats.per_strategy:
            ps = stats.per_strategy[key]
            if ps["avg_entry_slip"]:
                ps["avg_entry_slip"] = float(np.mean(ps["avg_entry_slip"]))
            else:
                ps["avg_entry_slip"] = 0
            if ps["avg_pnl_div"]:
                ps["avg_pnl_div"] = float(np.mean(ps["avg_pnl_div"]))
            else:
                ps["avg_pnl_div"] = 0

        # Slippage trend (is it getting worse over time?)
        if len(entry_slips) >= 5:
            x = np.arange(len(entry_slips))
            slope = np.polyfit(x, entry_slips, 1)[0]
            stats.slippage_trend = float(slope)
        
        # ─── Alerts ───────────────────────────────────────────
        if stats.avg_entry_slippage_bps > self.alert_slippage_bps:
            stats.alerts.append(
                f"HIGH SLIPPAGE: avg entry slippage "
                f"{stats.avg_entry_slippage_bps:.1f} bps > "
                f"{self.alert_slippage_bps:.0f} bps threshold"
            )

        if abs(stats.avg_pnl_divergence_pct) > self.alert_pnl_diverge_pct:
            stats.alerts.append(
                f"PNL DIVERGENCE: avg {stats.avg_pnl_divergence_pct:+.1f}% "
                f"vs backtest"
            )

        miss_rate = len(missed) / len(self.comparisons) if self.comparisons else 0
        if miss_rate > self.alert_miss_rate:
            stats.alerts.append(
                f"HIGH MISS RATE: {miss_rate:.0%} of signals not executed"
            )

        if stats.slippage_trend > self.alert_trend_threshold:
            stats.alerts.append(
                f"SLIPPAGE TRENDING UP: slope={stats.slippage_trend:.2f} bps/trade"
            )

        return stats

    def format_report(self) -> str:
        """Generate human-readable divergence report."""
        stats = self.get_stats()
        lines = []
        def p(s=""):
            lines.append(s)

        p("=" * 60)
        p("  LIVE vs BACKTEST DIVERGENCE REPORT")
        p("=" * 60)
        p(f"  Total signals:     {stats.total_trades}")
        p(f"  Executed:          {stats.total_trades - stats.total_missed}")
        p(f"  Missed:            {stats.total_missed}")
        p(f"  Miss rate:         {stats.total_missed / stats.total_trades:.1%}"
          if stats.total_trades > 0 else "  Miss rate: N/A")
        p()
        p(f"  Avg entry slippage:  {stats.avg_entry_slippage_bps:+.1f} bps")
        p(f"  Avg exit slippage:   {stats.avg_exit_slippage_bps:+.1f} bps")
        p(f"  Avg PnL divergence:  {stats.avg_pnl_divergence_pct:+.1f}%")
        p(f"  Slippage trend:      {stats.slippage_trend:+.2f} bps/trade")
        p()

        if stats.per_strategy:
            p("─ PER STRATEGY ──────────────────────────────────")
            p(f"  {'Strategy':<25s} {'Trades':>7s} {'Slip bps':>9s} "
              f"{'PnL div%':>9s} {'Misses':>7s}")
            for name, ps in sorted(stats.per_strategy.items()):
                p(f"  {name:<25s} {ps['trades']:>7d} "
                  f"{ps['avg_entry_slip']:>+8.1f} "
                  f"{ps['avg_pnl_div']:>+8.1f}% "
                  f"{ps['misses']:>7d}")
            p()

        if stats.alerts:
            p("─ ALERTS ────────────────────────────────────────")
            for alert in stats.alerts:
                p(f"  ⚠ {alert}")
            p()
        else:
            p("  No alerts — execution quality is nominal.")
            p()

        p("=" * 60)
        return "\n".join(lines)

    # ─── Persistence ─────────────────────────────────────────────

    def _save(self):
        """Save comparisons to disk."""
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(c) for c in self.comparisons[-1000:]]  # Keep last 1000
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        """Load existing comparisons from disk."""
        p = Path(self.persist_path)
        if not p.exists():
            return
        try:
            with open(p) as f:
                data = json.load(f)
            self.comparisons = [TradeComparison(**d) for d in data]
        except Exception as e:
            logger.warning(f"Could not load divergence log: {e}")

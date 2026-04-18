"""
Strategy Monitor — Track live performance + detect decay.
============================================================
Watches deployed strategies in real-time:

    1. Tracks per-trade P&L against expected OOS performance
    2. Detects regime changes that invalidate signal assumptions
    3. Flags strategies for removal when edge decays
    4. Generates health reports
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.factory.deployer import DeployedStrategy

logger = logging.getLogger("factory.monitor")

MONITOR_DIR = Path("fund_data/monitor")


@dataclass
class StrategyHealth:
    """Health assessment of a deployed strategy."""
    name: str
    status: str           # "healthy", "warning", "critical", "dead"
    live_trades: int
    live_pf: float
    live_sharpe: float
    expected_pf: float    # from OOS validation
    pf_decay: float       # live_pf / expected_pf — below 0.5 is trouble
    recent_win_rate: float
    consecutive_losses: int
    message: str


@dataclass
class TradeRecord:
    """Single trade record for tracking."""
    strategy: str
    asset: str
    direction: int
    entry_price: float
    exit_price: float
    return_pct: float
    timestamp: str
    hold_bars: int


class StrategyMonitor:
    """Monitors deployed strategies for decay and anomalies."""

    def __init__(self, monitor_dir: str | Path | None = None):
        self.monitor_dir = Path(monitor_dir) if monitor_dir else MONITOR_DIR
        self.monitor_dir.mkdir(parents=True, exist_ok=True)

        self.trade_log_path = self.monitor_dir / "trade_log.json"
        self.trades: list[dict] = self._load_trades()

    def _load_trades(self) -> list[dict]:
        if self.trade_log_path.exists():
            try:
                return json.loads(self.trade_log_path.read_text())
            except Exception:
                return []
        return []

    def _save_trades(self):
        try:
            self.trade_log_path.write_text(
                json.dumps(self.trades[-10000:], indent=2, default=str)  # Keep last 10k trades
            )
        except Exception as e:
            logger.warning(f"Failed to save trade log: {e}")

    def record_trade(self, record: TradeRecord):
        """Record a completed trade."""
        self.trades.append({
            "strategy": record.strategy,
            "asset": record.asset,
            "direction": record.direction,
            "entry_price": record.entry_price,
            "exit_price": record.exit_price,
            "return_pct": record.return_pct,
            "timestamp": record.timestamp,
            "hold_bars": record.hold_bars,
        })
        self._save_trades()

    def assess_health(self, strategy: DeployedStrategy) -> StrategyHealth:
        """Assess health of a single strategy."""
        strat_trades = [t for t in self.trades if t["strategy"] == strategy.name]

        if len(strat_trades) < 5:
            return StrategyHealth(
                name=strategy.name,
                status="insufficient_data",
                live_trades=len(strat_trades),
                live_pf=0,
                live_sharpe=0,
                expected_pf=strategy.oos_pf,
                pf_decay=1.0,
                recent_win_rate=0,
                consecutive_losses=0,
                message=f"Only {len(strat_trades)} trades — need at least 5 for assessment.",
            )

        returns = [t["return_pct"] for t in strat_trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-10
        live_pf = gross_profit / gross_loss if gross_loss > 0 else 0

        live_mean = np.mean(returns)
        live_std = np.std(returns) if len(returns) > 1 else 1e-10
        live_sharpe = live_mean / live_std * np.sqrt(365 * 24 / strategy.hold_bars)

        # PF decay ratio
        pf_decay = live_pf / strategy.oos_pf if strategy.oos_pf > 0 else 0

        # Recent performance (last 10 trades)
        recent = returns[-10:]
        recent_wr = len([r for r in recent if r > 0]) / len(recent)

        # Consecutive losses
        consec_losses = 0
        for r in reversed(returns):
            if r <= 0:
                consec_losses += 1
            else:
                break

        # Status determination
        if pf_decay < 0.3 or consec_losses >= 8:
            status = "dead"
            msg = f"Edge is gone. PF decay={pf_decay:.2f}, consec losses={consec_losses}."
        elif pf_decay < 0.5 or consec_losses >= 5:
            status = "critical"
            msg = f"Severe decay. PF decay={pf_decay:.2f}, consec losses={consec_losses}."
        elif pf_decay < 0.7 or recent_wr < 0.3:
            status = "warning"
            msg = f"Edge weakening. PF decay={pf_decay:.2f}, recent WR={recent_wr:.0%}."
        else:
            status = "healthy"
            msg = f"Performing as expected. PF decay={pf_decay:.2f}."

        return StrategyHealth(
            name=strategy.name,
            status=status,
            live_trades=len(strat_trades),
            live_pf=live_pf,
            live_sharpe=live_sharpe,
            expected_pf=strategy.oos_pf,
            pf_decay=pf_decay,
            recent_win_rate=recent_wr,
            consecutive_losses=consec_losses,
            message=msg,
        )

    def get_kill_list(self, strategies: list[DeployedStrategy]) -> list[str]:
        """Return names of strategies that should be killed."""
        kills = []
        for strat in strategies:
            health = self.assess_health(strat)
            if health.status == "dead":
                kills.append(strat.name)
                logger.info(f"KILL: {strat.name} — {health.message}")
        return kills

    def report(self, strategies: list[DeployedStrategy]) -> str:
        """Generate health report for all strategies."""
        lines = []
        lines.append("=" * 60)
        lines.append("  STRATEGY HEALTH REPORT")
        lines.append(f"  {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
        lines.append("=" * 60)

        for strat in strategies:
            health = self.assess_health(strat)
            icon = {"healthy": "✓", "warning": "⚠", "critical": "✗", "dead": "☠"}.get(health.status, "?")
            lines.append(
                f"  {icon} {strat.name:<35s} "
                f"trades={health.live_trades:>3d} "
                f"PF={health.live_pf:>5.2f} "
                f"decay={health.pf_decay:>4.2f} "
                f"[{health.status}]"
            )
            if health.status in ("critical", "dead"):
                lines.append(f"    └─ {health.message}")

        lines.append("=" * 60)
        return "\n".join(lines)

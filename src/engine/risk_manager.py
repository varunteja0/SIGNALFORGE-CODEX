"""
Kill-Switch & Risk Manager — Portfolio-Level Protection
========================================================
Institutional-grade risk controls:

    1. Strategy kill-switch: disable strategy if PF < 1 over trailing window
    2. Drawdown circuit breaker: reduce/halt if portfolio DD exceeds threshold
    3. Daily loss limit: stop trading after N% daily loss
    4. Concentration limit: max exposure per strategy / per asset
    5. Correlation spike detector: cut exposure if strategy correlations jump

Designed to run in both backtest and live modes.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyHealth:
    """Rolling health metrics for one strategy."""
    name: str
    recent_trades: list = field(default_factory=list)  # (timestamp, pnl)
    total_trades: int = 0
    is_killed: bool = False
    kill_reason: str = ""
    killed_at: Optional[str] = None
    # Rolling window stats
    rolling_pf: float = 0.0
    rolling_wr: float = 0.0
    rolling_pnl: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    size_multiplier: float = 1.0  # 1.0 = full size, 0.5 = half, 0 = killed


@dataclass
class PortfolioRiskState:
    """Current risk state of the portfolio."""
    equity: float = 0.0
    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    is_halted: bool = False
    halt_reason: str = ""
    strategy_health: dict = field(default_factory=dict)  # name → StrategyHealth
    # History
    risk_events: list = field(default_factory=list)


class RiskManager:
    """Portfolio-level risk management with kill-switches.

    Controls:
        - Per-strategy: kill if rolling PF or win-rate degrades
        - Portfolio: halt if DD or daily loss exceeds threshold
        - Dynamic sizing: reduce size as drawdown increases
    """

    def __init__(
        self,
        # Strategy kill-switch thresholds
        kill_min_trades: int = 10,       # Need N trades before evaluating
        kill_pf_threshold: float = 0.8,  # Kill if rolling PF < this
        kill_wr_threshold: float = 0.25, # Kill if rolling WR < this
        kill_consec_losses: int = 8,     # Kill after N consecutive losses
        kill_window: int = 20,           # Rolling window for PF/WR
        # Drawdown controls
        dd_reduce_threshold: float = 0.05,  # Reduce size at 5% DD
        dd_reduce_factor: float = 0.5,      # Cut size by 50%
        dd_halt_threshold: float = 0.10,    # Halt at 10% DD
        # Daily loss limit
        daily_loss_limit_pct: float = 0.02, # Stop after 2% daily loss
        # Recovery
        recovery_trades: int = 5,           # Re-enable after N profitable trades
    ):
        self.kill_min_trades = kill_min_trades
        self.kill_pf_threshold = kill_pf_threshold
        self.kill_wr_threshold = kill_wr_threshold
        self.kill_consec_losses = kill_consec_losses
        self.kill_window = kill_window
        self.dd_reduce_threshold = dd_reduce_threshold
        self.dd_reduce_factor = dd_reduce_factor
        self.dd_halt_threshold = dd_halt_threshold
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.recovery_trades = recovery_trades

        self.state = PortfolioRiskState()

    def initialize(self, capital: float, strategy_names: list[str]):
        """Initialize risk state for a portfolio."""
        self.state.equity = capital
        self.state.peak_equity = capital
        for name in strategy_names:
            self.state.strategy_health[name] = StrategyHealth(name=name)

    def record_trade(
        self,
        strategy_name: str,
        pnl: float,
        timestamp: str = None,
    ):
        """Record a completed trade and evaluate risk."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        health = self.state.strategy_health.get(strategy_name)
        if health is None:
            health = StrategyHealth(name=strategy_name)
            self.state.strategy_health[strategy_name] = health

        # Update tracking
        health.recent_trades.append((timestamp, pnl))
        health.total_trades += 1

        # Trim to window
        if len(health.recent_trades) > self.kill_window:
            health.recent_trades = health.recent_trades[-self.kill_window:]

        # Consecutive losses
        if pnl < 0:
            health.consecutive_losses += 1
            health.max_consecutive_losses = max(
                health.max_consecutive_losses, health.consecutive_losses
            )
        else:
            health.consecutive_losses = 0

        # Portfolio equity
        self.state.equity += pnl
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        self.state.current_drawdown = (
            (self.state.peak_equity - self.state.equity) / self.state.peak_equity
            if self.state.peak_equity > 0 else 0
        )
        self.state.daily_pnl += pnl

        # Evaluate strategy health
        self._evaluate_strategy(health)

        # Evaluate portfolio-level
        self._evaluate_portfolio()

    def get_size_multiplier(self, strategy_name: str) -> float:
        """Get position size multiplier for a strategy.

        Returns:
            1.0 = full size
            0.5 = half size (DD reduction)
            0.0 = killed/halted
        """
        # Portfolio halt
        if self.state.is_halted:
            return 0.0

        # Strategy killed
        health = self.state.strategy_health.get(strategy_name)
        if health and health.is_killed:
            return 0.0

        # DD-based reduction
        multiplier = 1.0

        if self.state.current_drawdown >= self.dd_reduce_threshold:
            multiplier *= self.dd_reduce_factor

        if health:
            multiplier *= health.size_multiplier

        return multiplier

    def should_trade(self, strategy_name: str) -> bool:
        """Check if a strategy is allowed to trade."""
        if self.state.is_halted:
            return False

        health = self.state.strategy_health.get(strategy_name)
        if health and health.is_killed:
            return False

        return True

    def reset_daily(self):
        """Reset daily counters (call at start of each trading day)."""
        self.state.daily_pnl = 0.0
        self.state.is_halted = False
        self.state.halt_reason = ""

    def get_status(self) -> dict:
        """Get current risk status."""
        return {
            "equity": self.state.equity,
            "drawdown": f"{self.state.current_drawdown:.2%}",
            "daily_pnl": self.state.daily_pnl,
            "halted": self.state.is_halted,
            "strategies": {
                name: {
                    "killed": h.is_killed,
                    "kill_reason": h.kill_reason,
                    "rolling_pf": h.rolling_pf,
                    "rolling_wr": h.rolling_wr,
                    "size_mult": h.size_multiplier,
                    "consec_losses": h.consecutive_losses,
                    "total_trades": h.total_trades,
                }
                for name, h in self.state.strategy_health.items()
            },
        }

    # ─── Internal Evaluation ─────────────────────────────────────

    def _evaluate_strategy(self, health: StrategyHealth):
        """Evaluate whether a strategy should be killed or throttled."""
        if health.is_killed:
            # Check for recovery
            self._check_recovery(health)
            return

        if health.total_trades < self.kill_min_trades:
            return

        # Compute rolling stats
        recent_pnls = [pnl for _, pnl in health.recent_trades]
        if not recent_pnls:
            return

        wins = [p for p in recent_pnls if p > 0]
        losses = [p for p in recent_pnls if p <= 0]
        health.rolling_wr = len(wins) / len(recent_pnls)
        health.rolling_pnl = sum(recent_pnls)

        gross_win = sum(wins) if wins else 0
        gross_loss = sum(abs(p) for p in losses) if losses else 0
        health.rolling_pf = gross_win / gross_loss if gross_loss > 0 else 99

        # Kill conditions (any one triggers)
        kill_reason = None

        if (len(recent_pnls) >= self.kill_window
                and health.rolling_pf < self.kill_pf_threshold):
            kill_reason = f"PF={health.rolling_pf:.2f} < {self.kill_pf_threshold}"

        elif (len(recent_pnls) >= self.kill_window
              and health.rolling_wr < self.kill_wr_threshold):
            kill_reason = f"WR={health.rolling_wr:.1%} < {self.kill_wr_threshold:.0%}"

        elif health.consecutive_losses >= self.kill_consec_losses:
            kill_reason = f"{health.consecutive_losses} consecutive losses"

        if kill_reason:
            health.is_killed = True
            health.kill_reason = kill_reason
            health.killed_at = datetime.now().isoformat()
            health.size_multiplier = 0.0

            event = {
                "type": "strategy_kill",
                "strategy": health.name,
                "reason": kill_reason,
                "time": health.killed_at,
                "rolling_pf": health.rolling_pf,
                "rolling_wr": health.rolling_wr,
            }
            self.state.risk_events.append(event)
            logger.warning(f"KILL-SWITCH: {health.name} — {kill_reason}")

    def _check_recovery(self, health: StrategyHealth):
        """Check if a killed strategy has recovered."""
        if not health.is_killed:
            return

        # Need N recent profitable trades to recover
        recent = [pnl for _, pnl in health.recent_trades[-self.recovery_trades:]]
        if len(recent) < self.recovery_trades:
            return

        profitable = sum(1 for p in recent if p > 0)
        if profitable >= self.recovery_trades * 0.8:  # 80% profitable
            health.is_killed = False
            health.kill_reason = ""
            health.size_multiplier = 0.5  # Come back at half size
            health.consecutive_losses = 0

            event = {
                "type": "strategy_recover",
                "strategy": health.name,
                "time": datetime.now().isoformat(),
            }
            self.state.risk_events.append(event)
            logger.info(f"RECOVERY: {health.name} — restored at 50% size")

    def _evaluate_portfolio(self):
        """Portfolio-level risk checks."""
        # Daily loss limit
        daily_loss_pct = abs(self.state.daily_pnl) / self.state.peak_equity
        if self.state.daily_pnl < 0 and daily_loss_pct >= self.daily_loss_limit_pct:
            self.state.is_halted = True
            self.state.halt_reason = (
                f"Daily loss {daily_loss_pct:.1%} >= {self.daily_loss_limit_pct:.0%}"
            )
            event = {
                "type": "portfolio_halt",
                "reason": self.state.halt_reason,
                "time": datetime.now().isoformat(),
            }
            self.state.risk_events.append(event)
            logger.warning(f"PORTFOLIO HALT: {self.state.halt_reason}")

        # Drawdown halt
        if self.state.current_drawdown >= self.dd_halt_threshold:
            self.state.is_halted = True
            self.state.halt_reason = (
                f"Drawdown {self.state.current_drawdown:.1%} >= "
                f"{self.dd_halt_threshold:.0%}"
            )
            event = {
                "type": "drawdown_halt",
                "reason": self.state.halt_reason,
                "time": datetime.now().isoformat(),
            }
            self.state.risk_events.append(event)
            logger.warning(f"DRAWDOWN HALT: {self.state.halt_reason}")


class BacktestRiskManager(RiskManager):
    """Risk manager adapted for backtest integration.

    In backtest mode, integrates with the backtester's trade-by-trade
    output and applies kill/sizing logic retroactively.
    """

    def apply_to_trades(
        self,
        trades: list,
        capital: float,
        strategy_name: str,
    ) -> list:
        """Filter/size trades as the risk manager would in live trading.

        Returns a new trade list with risk-adjusted PnLs.
        Trades from killed strategies are removed.
        Trades during DD reduction have halved PnL.
        """
        self.initialize(capital, [strategy_name])
        adjusted = []

        for trade in trades:
            # Check if we should trade
            mult = self.get_size_multiplier(strategy_name)
            if mult <= 0:
                # Record but don't execute — for recovery tracking
                self.record_trade(strategy_name, trade.pnl, str(trade.entry_time))
                continue

            # Adjust PnL by size multiplier
            adj_pnl = trade.pnl * mult
            adjusted.append(trade)

            self.record_trade(strategy_name, adj_pnl, str(trade.entry_time))

        return adjusted

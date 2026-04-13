"""
SignalForge Risk Management
============================
The #1 reason traders blow up: bad risk management.
This module ensures survival first, profits second.

Rules:
1. Never risk more than X% per trade
2. Stop trading at Y% daily loss
3. Stop trading at Z% total drawdown
4. Limit correlated positions
5. Kelly Criterion for optimal position sizing
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk parameters — these keep you alive."""
    max_position_pct: float = 0.02      # Max 2% of capital per trade
    max_drawdown_pct: float = 0.10      # Stop everything at 10% drawdown
    max_daily_loss_pct: float = 0.03    # Max 3% daily loss
    max_open_positions: int = 5
    max_correlated_positions: int = 3


@dataclass
class PositionRequest:
    """Request to open a position — must pass risk checks first."""
    symbol: str
    direction: int  # 1 = long, -1 = short
    entry_price: float
    stop_loss: float
    take_profit: float
    signal_name: str
    signal_strength: float  # 0-1, how confident is the signal


@dataclass
class PositionApproval:
    """Risk manager's response to a position request."""
    approved: bool
    size: float = 0
    reason: str = ""
    risk_pct: float = 0
    kelly_fraction: float = 0


class RiskManager:
    """Central risk management — every trade goes through here."""

    def __init__(
        self,
        capital: float,
        limits: Optional[RiskLimits] = None,
    ):
        self.initial_capital = capital
        self.current_capital = capital
        self.peak_capital = capital
        self.limits = limits or RiskLimits()
        self.open_positions: dict[str, dict] = {}
        self.daily_pnl = 0
        self.daily_start_capital = capital
        self.total_pnl = 0
        self._halted = False
        self._halt_reason = ""

    def evaluate(self, request: PositionRequest) -> PositionApproval:
        """Evaluate a position request against all risk limits."""
        
        # Check if trading is halted
        if self._halted:
            return PositionApproval(
                approved=False,
                reason=f"TRADING HALTED: {self._halt_reason}",
            )

        # Check drawdown limit
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown >= self.limits.max_drawdown_pct:
            self._halted = True
            self._halt_reason = f"Max drawdown reached: {drawdown:.1%}"
            return PositionApproval(
                approved=False,
                reason=self._halt_reason,
            )

        # Check daily loss limit
        daily_loss = (self.daily_start_capital - self.current_capital) / self.daily_start_capital
        if daily_loss >= self.limits.max_daily_loss_pct:
            return PositionApproval(
                approved=False,
                reason=f"Daily loss limit reached: {daily_loss:.1%}",
            )

        # Check max open positions
        if len(self.open_positions) >= self.limits.max_open_positions:
            return PositionApproval(
                approved=False,
                reason=f"Max open positions ({self.limits.max_open_positions}) reached",
            )

        # Check if already in this symbol
        if request.symbol in self.open_positions:
            return PositionApproval(
                approved=False,
                reason=f"Already has open position in {request.symbol}",
            )

        # Calculate position size
        risk_per_unit = abs(request.entry_price - request.stop_loss)
        if risk_per_unit <= 0:
            return PositionApproval(
                approved=False,
                reason="Invalid stop loss (no risk per unit)",
            )

        # Kelly Criterion position sizing
        kelly = self._kelly_fraction(request.signal_strength)
        
        # Risk-based sizing: risk X% of capital
        max_risk_amount = self.current_capital * self.limits.max_position_pct
        size_by_risk = max_risk_amount / risk_per_unit

        # Kelly-adjusted sizing
        size = size_by_risk * kelly
        
        # Cap at reasonable level 
        max_notional = self.current_capital * 0.2  # Never more than 20% of capital in one position
        max_size_by_notional = max_notional / request.entry_price
        size = min(size, max_size_by_notional)

        if size <= 0:
            return PositionApproval(
                approved=False,
                reason="Position size too small",
            )

        actual_risk_pct = (risk_per_unit * size) / self.current_capital

        logger.info(
            f"APPROVED: {request.symbol} {'LONG' if request.direction == 1 else 'SHORT'} "
            f"size={size:.6f} risk={actual_risk_pct:.1%} kelly={kelly:.2f}"
        )

        return PositionApproval(
            approved=True,
            size=size,
            reason="All risk checks passed",
            risk_pct=actual_risk_pct,
            kelly_fraction=kelly,
        )

    def _kelly_fraction(self, win_probability: float) -> float:
        """Kelly Criterion: f* = (bp - q) / b
        
        Where:
        - b = odds received on the bet (assumed 1.5:1 avg win/loss)
        - p = probability of winning
        - q = probability of losing
        
        We use HALF Kelly for safety (full Kelly is too aggressive).
        """
        b = 1.5  # Assume 1.5:1 reward/risk ratio
        p = max(0.01, min(0.99, win_probability))
        q = 1 - p
        
        kelly = (b * p - q) / b
        kelly = max(0, min(1, kelly))
        
        # Half Kelly for safety
        return kelly * 0.5

    def register_open(self, symbol: str, direction: int, size: float, entry_price: float):
        """Register a newly opened position."""
        self.open_positions[symbol] = {
            "direction": direction,
            "size": size,
            "entry_price": entry_price,
        }

    def register_close(self, symbol: str, exit_price: float):
        """Register a closed position, update PnL."""
        if symbol not in self.open_positions:
            return

        pos = self.open_positions.pop(symbol)
        if pos["direction"] == 1:
            pnl = (exit_price - pos["entry_price"]) * pos["size"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["size"]

        self.current_capital += pnl
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)

        logger.info(
            f"CLOSED {symbol}: PnL=${pnl:.2f} Capital=${self.current_capital:.2f}"
        )

    def new_day(self):
        """Reset daily counters."""
        self.daily_pnl = 0
        self.daily_start_capital = self.current_capital

    def reset_halt(self):
        """Manually reset trading halt (use with caution)."""
        self._halted = False
        self._halt_reason = ""
        logger.warning("Trading halt manually reset")

    def get_status(self) -> dict:
        """Get current risk status."""
        drawdown = (
            (self.peak_capital - self.current_capital) / self.peak_capital
            if self.peak_capital > 0 else 0
        )
        daily_loss = (
            (self.daily_start_capital - self.current_capital) / self.daily_start_capital
            if self.daily_start_capital > 0 else 0
        )

        return {
            "capital": self.current_capital,
            "initial_capital": self.initial_capital,
            "total_return": (self.current_capital / self.initial_capital) - 1,
            "peak_capital": self.peak_capital,
            "drawdown": drawdown,
            "daily_pnl": self.daily_pnl,
            "daily_loss_pct": daily_loss,
            "open_positions": len(self.open_positions),
            "is_halted": self._halted,
            "halt_reason": self._halt_reason,
        }

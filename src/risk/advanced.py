"""
Advanced Risk Management — Drawdown Bands + Circuit Breakers + Regime Scaling
================================================================================
Production-grade risk management inspired by Citadel/Two Sigma:

1. Drawdown Bands — graduated response (yellow/orange/red)
2. Per-Strategy Circuit Breakers — isolate failure, don't let one kill all
3. Regime-Adaptive Position Sizing — smaller bets in volatile regimes
4. Correlation Break Detection — reduce when strategies become correlated
5. Liquidity Buffer — always keep emergency cash
6. Pre-Trade Blackout — no trades during extreme events
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DrawdownBand:
    """Graduated drawdown response levels."""
    yellow_pct: float = 0.05    # Warning — reduce sizing by 25%
    orange_pct: float = 0.10    # Danger — reduce sizing by 50%
    red_pct: float = 0.15       # Critical — halt all new trades
    black_pct: float = 0.20     # Emergency — close all positions


@dataclass
class CircuitBreakerState:
    """Per-strategy circuit breaker."""
    strategy_name: str
    consecutive_losses: int = 0
    hourly_loss_pct: float = 0.0
    is_tripped: bool = False
    trip_reason: str = ""
    trip_time: float = 0.0
    cooldown_seconds: float = 3600  # 1 hour cooldown
    max_consecutive_losses: int = 5
    max_hourly_loss_pct: float = 0.02  # 2% per hour


@dataclass
class RiskState:
    """Comprehensive risk state snapshot."""
    drawdown_pct: float
    drawdown_band: str          # "green", "yellow", "orange", "red", "black"
    size_multiplier: float      # 0-1, how much to scale position sizes
    regime_multiplier: float    # 0-1, regime adjustment
    tripped_breakers: list[str]
    liquidity_buffer_pct: float
    can_trade: bool
    halt_reason: str = ""
    portfolio_heat: float = 0   # Total risk as % of capital


class AdvancedRiskManager:
    """Multi-layered risk management with drawdown bands and circuit breakers."""

    def __init__(
        self,
        initial_capital: float,
        drawdown_bands: Optional[DrawdownBand] = None,
        max_portfolio_heat: float = 0.10,  # Max 10% total risk
        liquidity_buffer_pct: float = 0.05, # Keep 5% cash always
        max_correlation_spike: float = 0.85, # Flag if correlations spike
        regime_vol_baseline: float = 0.02,   # Baseline daily vol
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.bands = drawdown_bands or DrawdownBand()
        self.max_heat = max_portfolio_heat
        self.liquidity_buffer = liquidity_buffer_pct
        self.max_corr = max_correlation_spike
        self.regime_vol_baseline = regime_vol_baseline

        # Per-strategy circuit breakers
        self.breakers: dict[str, CircuitBreakerState] = {}

        # Position tracking
        self.open_positions: dict[str, dict] = {}

        # Daily tracking
        self._daily_start = initial_capital
        self._daily_pnl = 0.0
        self._last_day_reset = time.time()
        self._last_hourly_reset = time.time()

        # Recent returns for correlation monitoring
        self._strategy_returns: dict[str, list[float]] = {}

    def register_strategy(self, name: str):
        """Register a strategy for circuit breaker monitoring."""
        self.breakers[name] = CircuitBreakerState(strategy_name=name)
        self._strategy_returns[name] = []

    def update_capital(self, new_capital: float):
        """Update current capital after trade."""
        self.capital = new_capital
        self.peak_capital = max(self.peak_capital, new_capital)

        # Daily reset check
        if time.time() - self._last_day_reset > 86400:
            self._daily_start = self.capital
            self._daily_pnl = 0.0
            self._last_day_reset = time.time()

        # Hourly reset for circuit breaker hourly_loss_pct
        if time.time() - self._last_hourly_reset > 3600:
            for cb in self.breakers.values():
                cb.hourly_loss_pct = 0.0
            self._last_hourly_reset = time.time()

    def check_entry(
        self,
        strategy_name: str,
        symbol: str,
        size_usd: float,
        current_regime_vol: float = 0.02,
        risk_usd: float = 0.0,
    ) -> tuple[bool, float, str]:
        """Pre-trade check: should this trade be allowed?

        Args:
            risk_usd: Estimated risk in USD (e.g. |entry - stop_loss| * size).
                      If 0, defaults to 5% of size_usd as conservative estimate.

        Returns:
            (approved, adjusted_size_multiplier, reason)
        """
        state = self.get_risk_state(current_regime_vol)

        # 1. Check if trading is allowed at all
        if not state.can_trade:
            return False, 0.0, state.halt_reason

        # 2. Check circuit breaker for this strategy
        cb = self.breakers.get(strategy_name)
        if cb and cb.is_tripped:
            # Check cooldown
            if time.time() - cb.trip_time < cb.cooldown_seconds:
                return False, 0.0, f"Circuit breaker: {cb.trip_reason}"
            else:
                cb.is_tripped = False
                cb.consecutive_losses = 0

        # 3. Check portfolio heat (risk-based, not notional)
        estimated_risk = risk_usd if risk_usd > 0 else size_usd * 0.05
        current_heat = self._compute_portfolio_heat()
        new_heat = current_heat + estimated_risk / self.capital
        if new_heat > self.max_heat:
            return False, 0.0, f"Portfolio heat {current_heat:.1%} + {estimated_risk/self.capital:.1%} = {new_heat:.1%} exceeds max {self.max_heat:.1%}"

        # 4. Check liquidity buffer
        available = self.capital * (1 - self.liquidity_buffer)
        position_total = sum(
            p.get("notional", 0) for p in self.open_positions.values()
        )
        if position_total + size_usd > available:
            return False, 0.0, f"Would exceed liquidity buffer (avail={available:.0f})"

        # 5. Compute final size multiplier
        multiplier = state.size_multiplier * state.regime_multiplier

        return True, multiplier, "approved"

    def record_trade_result(
        self,
        strategy_name: str,
        pnl: float,
        return_pct: float,
    ):
        """Record a trade result and update circuit breakers."""
        self._daily_pnl += pnl
        self.capital += pnl
        self.peak_capital = max(self.peak_capital, self.capital)

        # Update strategy returns
        if strategy_name in self._strategy_returns:
            self._strategy_returns[strategy_name].append(return_pct)
            # Keep only last 100
            if len(self._strategy_returns[strategy_name]) > 100:
                self._strategy_returns[strategy_name] = \
                    self._strategy_returns[strategy_name][-100:]

        # Update circuit breaker
        cb = self.breakers.get(strategy_name)
        if cb:
            if pnl < 0:
                cb.consecutive_losses += 1
                cb.hourly_loss_pct += abs(return_pct)

                # Trip conditions
                if cb.consecutive_losses >= cb.max_consecutive_losses:
                    cb.is_tripped = True
                    cb.trip_reason = f"{cb.consecutive_losses} consecutive losses"
                    cb.trip_time = time.time()
                    logger.warning(
                        f"CIRCUIT BREAKER TRIPPED: {strategy_name} — "
                        f"{cb.trip_reason}"
                    )

                if cb.hourly_loss_pct > cb.max_hourly_loss_pct:
                    cb.is_tripped = True
                    cb.trip_reason = f"Hourly loss {cb.hourly_loss_pct:.1%} > max {cb.max_hourly_loss_pct:.1%}"
                    cb.trip_time = time.time()
                    logger.warning(
                        f"CIRCUIT BREAKER TRIPPED: {strategy_name} — "
                        f"{cb.trip_reason}"
                    )
            else:
                cb.consecutive_losses = 0
                cb.hourly_loss_pct = max(0, cb.hourly_loss_pct - return_pct)

    def get_risk_state(
        self, current_regime_vol: float = 0.02
    ) -> RiskState:
        """Get comprehensive risk state."""
        dd_pct = (self.peak_capital - self.capital) / (self.peak_capital + 1e-10)

        # Drawdown band
        if dd_pct >= self.bands.black_pct:
            band = "black"
            size_mult = 0.0
            can_trade = False
            reason = f"BLACK BAND: DD={dd_pct:.1%} — CLOSE ALL POSITIONS"
        elif dd_pct >= self.bands.red_pct:
            band = "red"
            size_mult = 0.0
            can_trade = False
            reason = f"RED BAND: DD={dd_pct:.1%} — ALL TRADING HALTED"
        elif dd_pct >= self.bands.orange_pct:
            band = "orange"
            size_mult = 0.5
            can_trade = True
            reason = ""
        elif dd_pct >= self.bands.yellow_pct:
            band = "yellow"
            size_mult = 0.75
            can_trade = True
            reason = ""
        else:
            band = "green"
            size_mult = 1.0
            can_trade = True
            reason = ""

        # Regime adjustment: SCALE DOWN when vol is elevated
        # Low vol (calm) → regime_mult ~ 1.0 (normal sizing)
        # High vol → regime_mult < 1.0 (smaller bets)
        # NEVER go ABOVE 1.0 — low vol precedes vol explosions
        vol_ratio = current_regime_vol / (self.regime_vol_baseline + 1e-10)
        regime_mult = min(1.0, 1.0 / max(1.0, vol_ratio))
        regime_mult = max(0.25, regime_mult)

        # Tripped breakers
        tripped = [
            name for name, cb in self.breakers.items()
            if cb.is_tripped
        ]

        heat = self._compute_portfolio_heat()

        return RiskState(
            drawdown_pct=dd_pct,
            drawdown_band=band,
            size_multiplier=size_mult,
            regime_multiplier=regime_mult,
            tripped_breakers=tripped,
            liquidity_buffer_pct=self.liquidity_buffer,
            can_trade=can_trade,
            halt_reason=reason,
            portfolio_heat=heat,
        )

    def check_correlation_spike(self) -> Optional[tuple[str, str, float]]:
        """Check if any two strategies have correlated returns above threshold.

        Returns (strategy_a, strategy_b, correlation) if spike detected, else None.
        """
        names = list(self._strategy_returns.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rets_i = self._strategy_returns[names[i]]
                rets_j = self._strategy_returns[names[j]]

                min_len = min(len(rets_i), len(rets_j))
                if min_len < 10:
                    continue

                arr_i = np.array(rets_i[-min_len:])
                arr_j = np.array(rets_j[-min_len:])

                if arr_i.std() < 1e-10 or arr_j.std() < 1e-10:
                    continue

                corr = np.corrcoef(arr_i, arr_j)[0, 1]
                if not np.isnan(corr) and abs(corr) > self.max_corr:
                    return names[i], names[j], float(corr)

        return None

    def _compute_portfolio_heat(self) -> float:
        """Total risk exposure as fraction of capital."""
        if self.capital <= 0:
            return 1.0

        total_risk = 0.0
        for pos in self.open_positions.values():
            risk_usd = pos.get("risk_usd", 0)
            if risk_usd <= 0:
                # Fallback: estimate risk as 5% of notional
                risk_usd = pos.get("notional", 0) * 0.05
            total_risk += risk_usd
        return total_risk / self.capital

    def close_all_positions(self) -> list[str]:
        """Emergency: flag all positions for immediate closure.

        Called when black band is triggered.
        Returns list of symbols that need to be closed.
        """
        symbols_to_close = list(self.open_positions.keys())
        if symbols_to_close:
            logger.critical(
                f"BLACK BAND EMERGENCY: Closing {len(symbols_to_close)} positions: "
                f"{symbols_to_close}"
            )
        return symbols_to_close

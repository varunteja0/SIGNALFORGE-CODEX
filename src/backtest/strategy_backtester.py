"""
Strategy Backtester — Advanced Exit Logic
=============================================
Purpose-built backtester for the Liquidation Reversal strategy.
Extends the base Backtester with:

1. Partial exits (50% at TP1/VWAP, rest at TP2/EMA)
2. Dynamic trailing stop (ATR-based, activates after 1.5% profit)
3. OI-based exit (OI rising fast after entry = trap forming)
4. Time-based exit (8 bars max)
5. Volatility-scaled slippage (not hardcoded)
6. Per-trade metadata (exit reason, R:R achieved, holding time)
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult, Backtester

logger = logging.getLogger(__name__)


@dataclass
class DetailedTrade:
    """Trade with full metadata for analysis."""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    symbol: str = ""
    direction: int = 0          # 1 = long, -1 = short
    entry_price: float = 0
    exit_price: float = 0       # Weighted average exit price
    size: float = 0             # Original full size
    remaining_size: float = 0   # After partial exits
    pnl: float = 0
    pnl_pct: float = 0
    commission: float = 0
    is_open: bool = True

    # Exit details
    exit_reason: str = ""       # stop_loss, tp1, tp2, trailing, time, oi_exit, reverse
    partial_exits: list = field(default_factory=list)  # [(time, price, size, reason)]
    rr_achieved: float = 0      # Actual R:R
    bars_held: int = 0
    max_favorable: float = 0    # Best price during trade (MAE/MFE analysis)
    max_adverse: float = 0      # Worst price during trade

    # Signal context
    signal_strength: float = 0
    entry_atr: float = 0
    entry_vwap: float = 0
    entry_ema20: float = 0


class StrategyBacktester:
    """Backtester with advanced exit logic for the liquidation reversal strategy.

    Key differences from base Backtester:
    - Partial exits at VWAP (TP1) and EMA (TP2)
    - Trailing stop that activates after profit threshold
    - OI-based dynamic exit
    - Volatility-scaled slippage (higher during liquidation events)
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        commission_pct: float = 0.001,
        base_slippage_pct: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.base_slippage_pct = base_slippage_pct

    def _slippage_for_bar(self, bar: pd.Series) -> float:
        """Volatility-scaled slippage. Higher during liquidation events."""
        atr_ratio = bar.get("atr_ratio", 1.0)
        vol_spike = bar.get("volume_ratio", 1.0)

        # Base slippage × sqrt(ATR ratio) × log(volume spike)
        multiplier = max(1.0, np.sqrt(atr_ratio) * np.log1p(vol_spike - 1 + 1e-10))
        return self.base_slippage_pct * min(multiplier, 3.0)  # Cap at 3× base

    def run(
        self,
        df: pd.DataFrame,
        signal_func: Callable,
        stop_loss_atr: float = 2.0,
        tp1_target: str = "vwap",    # "vwap" or "atr"
        tp2_target: str = "ema_20",  # "ema_20" or "atr"
        tp1_exit_pct: float = 0.5,   # Close 50% at TP1
        max_holding_bars: int = 8,
        trailing_activation_pct: float = 0.015,  # Activate trail after 1.5% profit
        trailing_atr_mult: float = 1.5,          # Trail at 1.5× ATR
        oi_exit_threshold: float = 0.03,         # Exit if OI rises 3% after entry
        position_size_pct: float = 0.01,
    ) -> BacktestResult:
        """Run backtest with full exit logic."""
        signals = signal_func(df)
        capital = self.initial_capital
        position: Optional[DetailedTrade] = None
        trades: list[DetailedTrade] = []
        equity = [capital]
        equity_times = [df.index[0]]

        # Pre-compute indicators if not present
        if "atr" not in df.columns and "atr_14" not in df.columns:
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df = df.copy()
            df["atr"] = tr.rolling(14).mean()

        atr_col = "atr" if "atr" in df.columns else "atr_14"

        for i in range(1, len(df)):
            bar = df.iloc[i]
            signal = signals.iloc[i]
            atr = bar.get(atr_col, bar["close"] * 0.02)
            slippage = self._slippage_for_bar(bar)

            # ═══════════════════════════════════════════════
            # MANAGE EXISTING POSITION
            # ═══════════════════════════════════════════════
            if position is not None:
                position.bars_held += 1

                # Track MAE/MFE
                if position.direction == 1:
                    position.max_favorable = max(position.max_favorable, bar["high"])
                    position.max_adverse = min(position.max_adverse, bar["low"])
                else:
                    position.max_favorable = min(position.max_favorable, bar["low"])
                    position.max_adverse = max(position.max_adverse, bar["high"])

                close_reason = None
                exit_price = bar["close"]

                # --- STOP LOSS ---
                stop = self._compute_stop(position, atr, stop_loss_atr)
                if position.direction == 1 and bar["low"] <= stop:
                    exit_price = stop
                    close_reason = "stop_loss"
                elif position.direction == -1 and bar["high"] >= stop:
                    exit_price = stop
                    close_reason = "stop_loss"

                # --- TRAILING STOP ---
                if close_reason is None:
                    trail_stop = self._check_trailing_stop(
                        position, bar, atr,
                        trailing_activation_pct, trailing_atr_mult
                    )
                    if trail_stop is not None:
                        exit_price = trail_stop
                        close_reason = "trailing_stop"

                # --- TP1: Partial exit at VWAP ---
                if close_reason is None and position.remaining_size > position.size * (1 - tp1_exit_pct + 0.01):
                    tp1_price = self._get_tp1(bar, position, atr, tp1_target)
                    if tp1_price is not None:
                        hit_tp1 = (
                            (position.direction == 1 and bar["high"] >= tp1_price) or
                            (position.direction == -1 and bar["low"] <= tp1_price)
                        )
                        if hit_tp1:
                            # Partial exit
                            partial_size = position.remaining_size * tp1_exit_pct
                            partial_price = tp1_price
                            slip = partial_price * slippage
                            if position.direction == 1:
                                partial_price -= slip
                            else:
                                partial_price += slip

                            comm = partial_price * partial_size * self.commission_pct
                            if position.direction == 1:
                                partial_pnl = (partial_price - position.entry_price) * partial_size - comm
                            else:
                                partial_pnl = (position.entry_price - partial_price) * partial_size - comm

                            position.remaining_size -= partial_size
                            position.pnl += partial_pnl
                            position.commission += comm
                            position.partial_exits.append((
                                df.index[i], partial_price, partial_size, "tp1_vwap"
                            ))
                            capital += partial_pnl

                # --- TP2: Full exit at EMA ---
                if close_reason is None and position.remaining_size > 0:
                    tp2_price = self._get_tp2(bar, position, atr, tp2_target)
                    if tp2_price is not None:
                        hit_tp2 = (
                            (position.direction == 1 and bar["high"] >= tp2_price) or
                            (position.direction == -1 and bar["low"] <= tp2_price)
                        )
                        if hit_tp2:
                            exit_price = tp2_price
                            close_reason = "tp2_ema"

                # --- OI-BASED EXIT ---
                if close_reason is None and "oi_change_1h" in bar.index:
                    oi_change = bar.get("oi_change_1h", 0)
                    if oi_change > oi_exit_threshold:
                        close_reason = "oi_rising"

                # --- TIME EXIT ---
                if close_reason is None and position.bars_held >= max_holding_bars:
                    close_reason = "time_exit"

                # --- OPPOSITE SIGNAL ---
                if close_reason is None and signal != 0 and signal != position.direction:
                    close_reason = "reverse_signal"

                # ═══════════════════════════════════════
                # CLOSE REMAINING POSITION
                # ═══════════════════════════════════════
                if close_reason:
                    remaining = position.remaining_size
                    if remaining > 0:
                        slip = exit_price * slippage
                        if position.direction == 1:
                            exit_price -= slip
                        else:
                            exit_price += slip

                        comm = exit_price * remaining * self.commission_pct
                        if position.direction == 1:
                            final_pnl = (exit_price - position.entry_price) * remaining - comm
                        else:
                            final_pnl = (position.entry_price - exit_price) * remaining - comm

                        position.pnl += final_pnl
                        position.commission += comm
                        capital += final_pnl

                    position.exit_time = df.index[i]
                    position.exit_price = exit_price
                    position.exit_reason = close_reason
                    position.remaining_size = 0
                    position.is_open = False
                    position.pnl_pct = position.pnl / (position.entry_price * position.size)

                    risk = abs(position.entry_price - self._compute_stop(
                        position, position.entry_atr, stop_loss_atr
                    ))
                    if risk > 0:
                        if position.direction == 1:
                            position.rr_achieved = (exit_price - position.entry_price) / risk
                        else:
                            position.rr_achieved = (position.entry_price - exit_price) / risk

                    trades.append(position)
                    position = None

            # ═══════════════════════════════════════════════
            # OPEN NEW POSITION
            # ═══════════════════════════════════════════════
            if signal != 0 and position is None and capital > 0:
                entry_price = bar["close"]
                slip = entry_price * slippage
                if signal == 1:
                    entry_price += slip
                else:
                    entry_price -= slip

                risk_capital = capital * position_size_pct
                size = risk_capital / entry_price
                comm = entry_price * size * self.commission_pct
                capital -= comm

                position = DetailedTrade(
                    entry_time=df.index[i],
                    symbol=bar.get("symbol", ""),
                    direction=signal,
                    entry_price=entry_price,
                    size=size,
                    remaining_size=size,
                    commission=comm,
                    signal_strength=bar.get("liq_intensity", 0),
                    entry_atr=atr,
                    entry_vwap=bar.get("vwap", entry_price),
                    entry_ema20=bar.get("ema_20", entry_price),
                    max_favorable=entry_price,
                    max_adverse=entry_price,
                )

            # Equity tracking
            unrealized = 0
            if position is not None and position.remaining_size > 0:
                if position.direction == 1:
                    unrealized = (bar["close"] - position.entry_price) * position.remaining_size
                else:
                    unrealized = (position.entry_price - bar["close"]) * position.remaining_size
            equity.append(capital + unrealized)
            equity_times.append(df.index[i])

        # Close remaining position at last bar
        if position is not None and position.remaining_size > 0:
            exit_price = df.iloc[-1]["close"]
            remaining = position.remaining_size
            comm = exit_price * remaining * self.commission_pct
            if position.direction == 1:
                final_pnl = (exit_price - position.entry_price) * remaining - comm
            else:
                final_pnl = (position.entry_price - exit_price) * remaining - comm
            position.pnl += final_pnl
            position.commission += comm
            position.exit_price = exit_price
            position.exit_time = df.index[-1]
            position.exit_reason = "end_of_data"
            position.remaining_size = 0
            position.is_open = False
            position.pnl_pct = position.pnl / (position.entry_price * position.size)
            capital += final_pnl
            trades.append(position)

        return self._compute_stats(trades, equity, equity_times)

    def _compute_stop(self, position: DetailedTrade, atr: float, atr_mult: float) -> float:
        """Compute stop loss price."""
        if position.direction == 1:
            return position.entry_price - atr_mult * atr
        else:
            return position.entry_price + atr_mult * atr

    def _check_trailing_stop(
        self,
        position: DetailedTrade,
        bar: pd.Series,
        atr: float,
        activation_pct: float,
        trail_atr_mult: float,
    ) -> Optional[float]:
        """Check if trailing stop is hit. Returns exit price or None."""
        entry = position.entry_price

        if position.direction == 1:
            profit_pct = (position.max_favorable - entry) / entry
            if profit_pct >= activation_pct:
                trail_stop = position.max_favorable - trail_atr_mult * atr
                if bar["low"] <= trail_stop:
                    return trail_stop
        else:
            profit_pct = (entry - position.max_favorable) / entry
            if profit_pct >= activation_pct:
                trail_stop = position.max_favorable + trail_atr_mult * atr
                if bar["high"] >= trail_stop:
                    return trail_stop

        return None

    def _get_tp1(
        self, bar: pd.Series, position: DetailedTrade, atr: float, target: str
    ) -> Optional[float]:
        """Get TP1 price (VWAP or ATR-based)."""
        if target == "vwap":
            vwap = bar.get("vwap", None)
            if vwap and vwap > 0:
                return vwap
        # Fallback: 1.5× ATR from entry
        if position.direction == 1:
            return position.entry_price + 1.5 * atr
        else:
            return position.entry_price - 1.5 * atr

    def _get_tp2(
        self, bar: pd.Series, position: DetailedTrade, atr: float, target: str
    ) -> Optional[float]:
        """Get TP2 price (EMA or ATR-based)."""
        if target == "ema_20":
            ema = bar.get("ema_20", None)
            if ema and ema > 0:
                return ema
        # Fallback: 2.5× ATR from entry
        if position.direction == 1:
            return position.entry_price + 2.5 * atr
        else:
            return position.entry_price - 2.5 * atr

    def _compute_stats(
        self, trades: list[DetailedTrade], equity: list, equity_times: list
    ) -> BacktestResult:
        """Compute stats using base Backtester logic + exit analysis."""
        # Build compatible Trade objects for the base stats computer
        base_backtester = Backtester(initial_capital=self.initial_capital)

        from src.backtest.engine import Trade as BaseTrade
        base_trades = []
        for t in trades:
            bt = BaseTrade(
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                symbol=t.symbol,
                direction=t.direction,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                size=t.size,
                pnl=t.pnl,
                pnl_pct=t.pnl_pct,
                commission=t.commission,
                is_open=t.is_open,
            )
            base_trades.append(bt)

        result = base_backtester._compute_stats(base_trades, equity, equity_times)

        # Replace trades with our detailed versions
        result.trades = trades

        return result

    def exit_analysis(self, result: BacktestResult) -> dict:
        """Analyze exit reasons and their profitability."""
        if not result.trades:
            return {}

        by_reason = {}
        for t in result.trades:
            reason = t.exit_reason or "unknown"
            if reason not in by_reason:
                by_reason[reason] = {"count": 0, "total_pnl": 0, "wins": 0, "losses": 0}
            by_reason[reason]["count"] += 1
            by_reason[reason]["total_pnl"] += t.pnl
            if t.pnl > 0:
                by_reason[reason]["wins"] += 1
            else:
                by_reason[reason]["losses"] += 1

        for reason, stats in by_reason.items():
            stats["win_rate"] = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
            stats["avg_pnl"] = stats["total_pnl"] / stats["count"] if stats["count"] > 0 else 0

        # R:R analysis
        rr_values = [t.rr_achieved for t in result.trades if t.rr_achieved != 0]
        bars_held = [t.bars_held for t in result.trades]

        return {
            "by_exit_reason": by_reason,
            "avg_rr_achieved": float(np.mean(rr_values)) if rr_values else 0,
            "avg_bars_held": float(np.mean(bars_held)) if bars_held else 0,
            "partial_exit_trades": sum(1 for t in result.trades if t.partial_exits),
        }

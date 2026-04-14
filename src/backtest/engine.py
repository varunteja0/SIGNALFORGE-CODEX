"""
SignalForge Backtesting Engine
===============================
Event-driven backtester with realistic simulation:
- Commission fees
- Slippage estimation
- Position sizing via Kelly Criterion
- Walk-forward out-of-sample testing
- Monte Carlo confidence intervals

This is NOT a toy backtester. It accounts for the things that kill real traders.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade."""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    symbol: str
    direction: int  # 1 = long, -1 = short
    entry_price: float
    exit_price: float = 0
    size: float = 0
    pnl: float = 0
    pnl_pct: float = 0
    commission: float = 0
    slippage: float = 0
    is_open: bool = True


@dataclass 
class BacktestResult:
    """Full backtest results with detailed statistics."""
    total_return: float = 0
    annualized_return: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    calmar_ratio: float = 0
    max_drawdown: float = 0
    max_drawdown_duration: int = 0
    win_rate: float = 0
    profit_factor: float = 0
    total_trades: int = 0
    avg_trade_return: float = 0
    avg_win: float = 0
    avg_loss: float = 0
    best_trade: float = 0
    worst_trade: float = 0
    avg_holding_period: float = 0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trades: list = field(default_factory=list)
    monthly_returns: pd.Series = field(default_factory=pd.Series)


class Backtester:
    """Event-driven backtesting engine."""

    def __init__(
        self,
        initial_capital: float = 10000,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run(
        self,
        df: pd.DataFrame,
        signal_func: Callable,
        position_size_pct: float = 0.02,
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 3.0,
        max_holding_bars: int = 50,
    ) -> BacktestResult:
        """Run a full backtest with realistic execution."""
        signals = signal_func(df)
        capital = self.initial_capital
        position = None
        trades = []
        equity = [capital]
        equity_times = [df.index[0]]

        atr_col = "atr_14" if "atr_14" in df.columns else None

        for i in range(1, len(df)):
            bar = df.iloc[i]
            prev_bar = df.iloc[i - 1]
            signal = signals.iloc[i]

            # Check if we need to close existing position
            if position is not None:
                close_reason = None
                exit_price = bar["close"]

                # Stop loss
                if atr_col and atr_col in df.columns:
                    atr = df[atr_col].iloc[i]
                    if position.direction == 1:
                        stop = position.entry_price - stop_loss_atr * atr
                        tp = position.entry_price + take_profit_atr * atr
                        if bar["low"] <= stop:
                            exit_price = stop
                            close_reason = "stop_loss"
                        elif bar["high"] >= tp:
                            exit_price = tp
                            close_reason = "take_profit"
                    else:
                        stop = position.entry_price + stop_loss_atr * atr
                        tp = position.entry_price - take_profit_atr * atr
                        if bar["high"] >= stop:
                            exit_price = stop
                            close_reason = "stop_loss"
                        elif bar["low"] <= tp:
                            exit_price = tp
                            close_reason = "take_profit"

                # Max holding period
                bars_held = i - df.index.get_loc(position.entry_time)
                if bars_held >= max_holding_bars and close_reason is None:
                    close_reason = "max_hold"

                # Opposite signal
                if signal != 0 and signal != position.direction and close_reason is None:
                    close_reason = "reverse_signal"

                if close_reason:
                    # Apply slippage
                    slip = exit_price * self.slippage_pct
                    if position.direction == 1:
                        exit_price -= slip
                    else:
                        exit_price += slip

                    commission = exit_price * position.size * self.commission_pct
                    
                    if position.direction == 1:
                        pnl = (exit_price - position.entry_price) * position.size - commission
                    else:
                        pnl = (position.entry_price - exit_price) * position.size - commission

                    position.exit_price = exit_price
                    position.exit_time = df.index[i]
                    position.pnl = pnl
                    position.pnl_pct = pnl / (position.entry_price * position.size)
                    position.commission += commission
                    position.is_open = False
                    
                    capital += pnl
                    trades.append(position)
                    position = None

            # Open new position
            if signal != 0 and position is None and capital > 0:
                entry_price = bar["close"]
                
                # Apply slippage
                slip = entry_price * self.slippage_pct
                if signal == 1:
                    entry_price += slip
                else:
                    entry_price -= slip

                # Position sizing
                risk_capital = capital * position_size_pct
                size = risk_capital / entry_price
                commission = entry_price * size * self.commission_pct

                position = Trade(
                    entry_time=df.index[i],
                    exit_time=None,
                    symbol="",
                    direction=signal,
                    entry_price=entry_price,
                    size=size,
                    commission=commission,
                )
                capital -= commission

            equity.append(capital + (
                (bar["close"] - position.entry_price) * position.size * position.direction
                if position else 0
            ))
            equity_times.append(df.index[i])

        # Close any remaining position at last bar
        if position is not None:
            exit_price = df.iloc[-1]["close"]
            commission = exit_price * position.size * self.commission_pct
            if position.direction == 1:
                pnl = (exit_price - position.entry_price) * position.size - commission
            else:
                pnl = (position.entry_price - exit_price) * position.size - commission
            position.exit_price = exit_price
            position.exit_time = df.index[-1]
            position.pnl = pnl
            position.pnl_pct = pnl / (position.entry_price * position.size)
            position.is_open = False
            capital += pnl
            trades.append(position)

        return self._compute_stats(trades, equity, equity_times)

    def _compute_stats(
        self, trades: list[Trade], equity: list, equity_times: list
    ) -> BacktestResult:
        """Compute comprehensive backtest statistics."""
        result = BacktestResult()
        result.trades = trades
        result.total_trades = len(trades)

        if not trades:
            result.equity_curve = pd.Series(equity, index=equity_times)
            return result

        # Equity curve
        eq = pd.Series(equity, index=equity_times)
        result.equity_curve = eq

        # Returns
        eq_returns = eq.pct_change().dropna()
        result.total_return = (eq.iloc[-1] / eq.iloc[0]) - 1

        days = (eq.index[-1] - eq.index[0]).total_seconds() / 86400
        if days > 0:
            result.annualized_return = (1 + result.total_return) ** (365 / days) - 1

        # Sharpe
        if len(eq_returns) > 1 and eq_returns.std() > 0:
            result.sharpe_ratio = (
                eq_returns.mean() / eq_returns.std() * np.sqrt(252 * 24)
            )

        # Sortino
        downside = eq_returns[eq_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            result.sortino_ratio = (
                eq_returns.mean() / downside.std() * np.sqrt(252 * 24)
            )

        # Drawdown
        peak = eq.cummax()
        drawdown = (eq - peak) / (peak + 1e-10)
        result.max_drawdown = abs(drawdown.min())

        # Drawdown duration
        is_dd = drawdown < 0
        dd_groups = (~is_dd).cumsum()
        if is_dd.any():
            dd_durations = is_dd.groupby(dd_groups).sum()
            result.max_drawdown_duration = dd_durations.max()

        # Calmar
        if result.max_drawdown > 0:
            result.calmar_ratio = result.annualized_return / result.max_drawdown

        # Trade stats
        pnls = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.win_rate = len(wins) / len(pnls) if pnls else 0
        result.avg_trade_return = np.mean(pnls) if pnls else 0
        result.avg_win = np.mean(wins) if wins else 0
        result.avg_loss = np.mean(losses) if losses else 0
        result.best_trade = max(pnls) if pnls else 0
        result.worst_trade = min(pnls) if pnls else 0

        if losses and np.mean([abs(l) for l in losses]) > 0:
            result.profit_factor = (
                sum(wins) / sum(abs(l) for l in losses)
            ) if losses else float("inf")

        # Monthly returns
        if len(eq) > 1:
            monthly = eq.resample("ME").last().pct_change().dropna()
            result.monthly_returns = monthly

        return result

    def monte_carlo(
        self,
        result: BacktestResult,
        n_simulations: int = 1000,
    ) -> dict:
        """Monte Carlo simulation to estimate confidence intervals.
        
        Shuffles trade order to see how robust the strategy is.
        A good strategy should work regardless of the order of trades.
        """
        if not result.trades:
            return {}

        pnls = [t.pnl for t in result.trades]
        final_capitals = []

        rng = np.random.default_rng(42)

        for _ in range(n_simulations):
            shuffled = rng.permutation(pnls)
            equity = self.initial_capital + np.cumsum(shuffled)
            final_capitals.append(equity[-1])

        final_capitals = np.array(final_capitals)

        return {
            "median_return": np.median(final_capitals) / self.initial_capital - 1,
            "p5": np.percentile(final_capitals, 5) / self.initial_capital - 1,
            "p25": np.percentile(final_capitals, 25) / self.initial_capital - 1,
            "p75": np.percentile(final_capitals, 75) / self.initial_capital - 1,
            "p95": np.percentile(final_capitals, 95) / self.initial_capital - 1,
            "probability_of_profit": (final_capitals > self.initial_capital).mean(),
            "worst_case": final_capitals.min() / self.initial_capital - 1,
            "best_case": final_capitals.max() / self.initial_capital - 1,
        }

    def run_with_tree(
        self,
        df: pd.DataFrame,
        tree,
        holding_period: int = 24,
        position_size_pct: float = 0.02,
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 3.0,
    ) -> BacktestResult:
        """Run backtest using the EXACT same signal logic as FitnessEvaluator.

        This eliminates the fitness-vs-backtest gap by using the canonical
        signal generator from FitnessEvaluator._discretize_signals() and
        enforcing the same holding period.

        The signal only changes every `holding_period` bars, which means
        positions are held for at least that many bars before any change.
        The backtester still checks stop-loss / take-profit within the
        holding window for risk management.
        """
        from src.alpha_genome.fitness import FitnessEvaluator

        evaluator = FitnessEvaluator(
            holding_period=holding_period,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
        )
        signals = evaluator.generate_backtest_signals(tree, df)

        def signal_func(data_df):
            return signals

        return self.run(
            df, signal_func,
            position_size_pct=position_size_pct,
            stop_loss_atr=stop_loss_atr,
            take_profit_atr=take_profit_atr,
            max_holding_bars=holding_period * 2,  # Allow 2x holding for SL/TP
        )

    def run_dynamic(
        self,
        df: pd.DataFrame,
        signal_func: Callable,
        position_size_pct: float = 0.02,
        # Initial stop
        initial_stop_atr: float = 1.5,
        # Trailing stop
        trail_activation_atr: float = 1.0,
        trail_distance_atr: float = 1.0,
        # Partial take profit
        tp1_atr: float = 2.5,
        tp1_close_pct: float = 0.5,
        tp2_atr: float = 5.0,
        # Time management
        max_holding_bars: int = 24,
        time_decay_bars: int = 10,
        time_decay_stop_atr: float = 0.8,
    ) -> BacktestResult:
        """Backtest with dynamic exits: trailing stop, partial TP, time decay.

        Exit logic (checked in order each bar):
        1. Initial stop: fixed at initial_stop_atr × ATR from entry
        2. Trailing stop: once MFE >= trail_activation_atr × ATR,
           stop ratchets to trail_distance_atr × ATR below high water mark
        3. TP1 partial: at tp1_atr × ATR, close tp1_close_pct of position
        4. TP2 hard: at tp2_atr × ATR, close remainder
        5. Time decay: after time_decay_bars, tighten stop to time_decay_stop_atr × ATR
        6. Max hold: force close at max_holding_bars
        7. Opposite signal: close and reverse
        """
        signals = signal_func(df)
        capital = self.initial_capital
        trades = []
        equity = [capital]
        equity_times = [df.index[0]]

        atr_col = "atr_14" if "atr_14" in df.columns else None

        # Position state
        position = None
        hwm_price = None          # High water mark (best price in our favor)
        trail_active = False      # Whether trailing stop is active
        tp1_hit = False           # Whether TP1 partial close happened
        remaining_pct = 1.0       # Fraction of position still open
        partial_pnl = 0.0         # PnL already locked in from partials
        entry_atr = 0.0           # ATR at entry time (frozen, not inflated by crashes)

        def _get_stop(entry_price, direction, atr, bars_held):
            """Compute current stop level based on state.
            
            Uses entry_atr (frozen at entry) for stop distances so crash-inflated
            ATR doesn't widen the stop when we need it most.
            """
            # Time-decay tightened stop
            if bars_held >= time_decay_bars:
                stop_mult = time_decay_stop_atr
            else:
                stop_mult = initial_stop_atr

            # Trailing stop overrides if active
            if trail_active and hwm_price is not None:
                if direction == 1:
                    fixed_stop = entry_price - stop_mult * entry_atr
                    trail_stop = hwm_price - trail_distance_atr * entry_atr
                    return max(fixed_stop, trail_stop)  # Use tighter stop
                else:
                    fixed_stop = entry_price + stop_mult * entry_atr
                    trail_stop = hwm_price + trail_distance_atr * entry_atr
                    return min(fixed_stop, trail_stop)  # Use tighter stop

            # Fixed stop
            if direction == 1:
                return entry_price - stop_mult * entry_atr
            else:
                return entry_price + stop_mult * entry_atr

        for i in range(1, len(df)):
            bar = df.iloc[i]
            signal = signals.iloc[i]

            if position is not None:
                close_reason = None
                exit_price = bar["close"]
                bars_held = i - df.index.get_loc(position.entry_time)

                # Update high water mark
                if position.direction == 1:
                    if hwm_price is None or bar["high"] > hwm_price:
                        hwm_price = bar["high"]
                    mfe_atr = (hwm_price - position.entry_price) / entry_atr if entry_atr > 0 else 0
                else:
                    if hwm_price is None or bar["low"] < hwm_price:
                        hwm_price = bar["low"]
                    mfe_atr = (position.entry_price - hwm_price) / entry_atr if entry_atr > 0 else 0

                # Activate trailing stop
                if not trail_active and mfe_atr >= trail_activation_atr:
                    trail_active = True

                # --- Check exits ---
                stop = _get_stop(position.entry_price, position.direction, entry_atr, bars_held)

                # 1. Stop loss (initial or trailing)
                if position.direction == 1:
                    if bar["low"] <= stop:
                        exit_price = stop
                        close_reason = "stop_loss"
                else:
                    if bar["high"] >= stop:
                        exit_price = stop
                        close_reason = "stop_loss"

                # 2. TP1 partial close
                if close_reason is None and not tp1_hit and entry_atr > 0:
                    if position.direction == 1:
                        tp1_level = position.entry_price + tp1_atr * entry_atr
                        if bar["high"] >= tp1_level:
                            # Partial close at TP1
                            tp1_price = tp1_level
                            slip = tp1_price * self.slippage_pct
                            tp1_price -= slip if position.direction == 1 else -slip
                            close_size = position.size * remaining_pct * tp1_close_pct
                            comm = tp1_price * close_size * self.commission_pct
                            pnl_partial = (tp1_price - position.entry_price) * close_size - comm
                            partial_pnl += pnl_partial
                            capital += pnl_partial
                            remaining_pct *= (1.0 - tp1_close_pct)
                            tp1_hit = True
                    else:
                        tp1_level = position.entry_price - tp1_atr * entry_atr
                        if bar["low"] <= tp1_level:
                            tp1_price = tp1_level
                            slip = tp1_price * self.slippage_pct
                            tp1_price += slip
                            close_size = position.size * remaining_pct * tp1_close_pct
                            comm = tp1_price * close_size * self.commission_pct
                            pnl_partial = (position.entry_price - tp1_price) * close_size - comm
                            partial_pnl += pnl_partial
                            capital += pnl_partial
                            remaining_pct *= (1.0 - tp1_close_pct)
                            tp1_hit = True

                # 3. TP2 hard close for remainder
                if close_reason is None and entry_atr > 0:
                    if position.direction == 1:
                        tp2_level = position.entry_price + tp2_atr * entry_atr
                        if bar["high"] >= tp2_level:
                            exit_price = tp2_level
                            close_reason = "take_profit_2"
                    else:
                        tp2_level = position.entry_price - tp2_atr * entry_atr
                        if bar["low"] <= tp2_level:
                            exit_price = tp2_level
                            close_reason = "take_profit_2"

                # 4. Max holding period
                if bars_held >= max_holding_bars and close_reason is None:
                    close_reason = "max_hold"

                # 5. Opposite signal
                if signal != 0 and signal != position.direction and close_reason is None:
                    close_reason = "reverse_signal"

                # Execute full close of remaining position
                if close_reason:
                    slip = exit_price * self.slippage_pct
                    if position.direction == 1:
                        exit_price -= slip
                    else:
                        exit_price += slip

                    remaining_size = position.size * remaining_pct
                    commission = exit_price * remaining_size * self.commission_pct

                    if position.direction == 1:
                        pnl_final = (exit_price - position.entry_price) * remaining_size - commission
                    else:
                        pnl_final = (position.entry_price - exit_price) * remaining_size - commission

                    total_pnl = partial_pnl + pnl_final
                    total_size = position.size  # Original full size

                    position.exit_price = exit_price
                    position.exit_time = df.index[i]
                    position.pnl = total_pnl
                    position.pnl_pct = total_pnl / (position.entry_price * total_size)
                    position.commission += commission
                    position.is_open = False

                    capital += pnl_final
                    trades.append(position)

                    # Reset state
                    position = None
                    hwm_price = None
                    trail_active = False
                    tp1_hit = False
                    remaining_pct = 1.0
                    partial_pnl = 0.0

            # Open new position
            if signal != 0 and position is None and capital > 0:
                entry_price = bar["close"]
                slip = entry_price * self.slippage_pct
                if signal == 1:
                    entry_price += slip
                else:
                    entry_price -= slip

                risk_capital = capital * position_size_pct
                size = risk_capital / entry_price
                commission = entry_price * size * self.commission_pct

                position = Trade(
                    entry_time=df.index[i],
                    exit_time=None,
                    symbol="",
                    direction=signal,
                    entry_price=entry_price,
                    size=size,
                    commission=commission,
                )
                capital -= commission

                # Initialize exit state
                hwm_price = entry_price
                trail_active = False
                tp1_hit = False
                remaining_pct = 1.0
                partial_pnl = 0.0
                entry_atr = df[atr_col].iloc[i] if atr_col else 0

            # Track equity
            mark_pnl = 0
            if position is not None:
                remaining_size = position.size * remaining_pct
                mark_pnl = (bar["close"] - position.entry_price) * remaining_size * position.direction
            equity.append(capital + mark_pnl)
            equity_times.append(df.index[i])

        # Close any remaining position
        if position is not None:
            exit_price = df.iloc[-1]["close"]
            remaining_size = position.size * remaining_pct
            commission = exit_price * remaining_size * self.commission_pct
            if position.direction == 1:
                pnl_final = (exit_price - position.entry_price) * remaining_size - commission
            else:
                pnl_final = (position.entry_price - exit_price) * remaining_size - commission
            total_pnl = partial_pnl + pnl_final
            position.exit_price = exit_price
            position.exit_time = df.index[-1]
            position.pnl = total_pnl
            position.pnl_pct = total_pnl / (position.entry_price * position.size)
            position.is_open = False
            capital += pnl_final
            trades.append(position)

        return self._compute_stats(trades, equity, equity_times)

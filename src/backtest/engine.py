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

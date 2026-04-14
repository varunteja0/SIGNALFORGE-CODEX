"""
Multi-Asset Trading Engine
=============================
Runs the Liquidation Reversal strategy across multiple symbols simultaneously.

Features:
- Per-asset signal generation and position management
- Portfolio-level risk limits (max total exposure, max correlated positions)
- Cross-asset correlation check (avoid piling into correlated longs)
- Unified equity curve and performance attribution
- Per-symbol breakdown for regime analysis

Architecture:
    For each bar:
      1. Fetch/update data for all symbols
      2. Generate signals per symbol
      3. Portfolio risk gate (total exposure, correlation)
      4. Execute approved trades
      5. Manage exits per symbol
      6. Update equity
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.backtest.strategy_backtester import DetailedTrade, StrategyBacktester
from src.backtest.engine import BacktestResult
from src.strategies.liquidation_reversal import LiquidationReversalStrategy, StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Portfolio-level risk parameters."""
    max_total_positions: int = 3            # Max positions across all assets
    max_total_exposure_pct: float = 0.06    # Max 6% capital at risk total
    max_correlation: float = 0.7            # Skip if asset correlated > 0.7 with open position
    correlation_lookback: int = 168         # 7 days for correlation (on 1h data)
    initial_capital: float = 10000
    commission_pct: float = 0.001
    base_slippage_pct: float = 0.0005


@dataclass
class PortfolioResult:
    """Multi-asset backtest results."""
    total_return: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    max_drawdown: float = 0
    total_trades: int = 0
    win_rate: float = 0
    profit_factor: float = 0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    per_symbol: dict = field(default_factory=dict)   # Per-symbol BacktestResult
    trades: list = field(default_factory=list)        # All trades across symbols
    monthly_returns: pd.Series = field(default_factory=pd.Series)


class MultiAssetEngine:
    """Runs the strategy across multiple symbols with portfolio-level risk."""

    def __init__(
        self,
        strategy_config: Optional[StrategyConfig] = None,
        portfolio_config: Optional[PortfolioConfig] = None,
    ):
        self.strategy_config = strategy_config or StrategyConfig()
        self.portfolio_config = portfolio_config or PortfolioConfig()
        self.strategy = LiquidationReversalStrategy(self.strategy_config)

    def run_multi_backtest(
        self,
        datasets: dict[str, pd.DataFrame],
        max_holding_bars: int = 8,
    ) -> PortfolioResult:
        """Run strategy across multiple symbols with portfolio risk management.

        Args:
            datasets: dict of {symbol: DataFrame} where each DataFrame has
                      OHLCV + structural data + strategy indicators pre-computed.
            max_holding_bars: max bars to hold a position.

        Returns:
            PortfolioResult with unified equity, per-symbol attribution, all trades.
        """
        pcfg = self.portfolio_config

        # Align all datasets to a common time index
        common_index = None
        for symbol, df in datasets.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)

        if common_index is None or len(common_index) == 0:
            logger.error("No overlapping data across symbols")
            return PortfolioResult()

        logger.info(f"Common index: {len(common_index)} bars across {len(datasets)} symbols")

        # Reindex all datasets
        aligned = {}
        for symbol, df in datasets.items():
            aligned[symbol] = df.reindex(common_index).fillna(method="ffill")

        # Pre-compute signals for all symbols
        all_signals = {}
        for symbol, df in aligned.items():
            indicators = self.strategy.compute_indicators(df)
            aligned[symbol] = indicators
            all_signals[symbol] = self.strategy.generate_signals(df)
            n_sig = (all_signals[symbol] != 0).sum()
            logger.info(f"  {symbol}: {n_sig} signals")

        # Compute rolling correlations between symbols
        returns = pd.DataFrame({
            sym: df["close"].pct_change() for sym, df in aligned.items()
        })

        # Portfolio simulation
        capital = pcfg.initial_capital
        positions: dict[str, DetailedTrade] = {}  # symbol → position
        all_trades: list[DetailedTrade] = []
        equity = [capital]
        equity_times = [common_index[0]]

        for i in range(1, len(common_index)):
            ts = common_index[i]

            # ═══════════════════════════════════════
            # MANAGE EXISTING POSITIONS
            # ═══════════════════════════════════════
            symbols_to_close = []
            for sym, pos in positions.items():
                bar = aligned[sym].iloc[i]
                atr = bar.get("atr", bar["close"] * 0.02)
                pos.bars_held += 1

                # Track MAE/MFE
                if pos.direction == 1:
                    pos.max_favorable = max(pos.max_favorable, bar["high"])
                    pos.max_adverse = min(pos.max_adverse, bar["low"])
                else:
                    pos.max_favorable = min(pos.max_favorable, bar["low"])
                    pos.max_adverse = max(pos.max_adverse, bar["high"])

                close_reason = None
                exit_price = bar["close"]

                # Stop loss
                if pos.direction == 1:
                    stop = pos.entry_price - self.strategy_config.stop_loss_atr_mult * atr
                    if bar["low"] <= stop:
                        exit_price = stop
                        close_reason = "stop_loss"
                else:
                    stop = pos.entry_price + self.strategy_config.stop_loss_atr_mult * atr
                    if bar["high"] >= stop:
                        exit_price = stop
                        close_reason = "stop_loss"

                # Trailing stop
                if close_reason is None:
                    if pos.direction == 1:
                        profit_pct = (pos.max_favorable - pos.entry_price) / pos.entry_price
                        if profit_pct >= 0.015:
                            trail = pos.max_favorable - 1.5 * atr
                            if bar["low"] <= trail:
                                exit_price = trail
                                close_reason = "trailing_stop"
                    else:
                        profit_pct = (pos.entry_price - pos.max_favorable) / pos.entry_price
                        if profit_pct >= 0.015:
                            trail = pos.max_favorable + 1.5 * atr
                            if bar["high"] >= trail:
                                exit_price = trail
                                close_reason = "trailing_stop"

                # TP1 partial at VWAP (50%)
                if close_reason is None and pos.remaining_size > pos.size * 0.51:
                    vwap = bar.get("vwap", None)
                    if vwap and vwap > 0:
                        hit = (
                            (pos.direction == 1 and bar["high"] >= vwap) or
                            (pos.direction == -1 and bar["low"] <= vwap)
                        )
                        if hit:
                            partial_size = pos.remaining_size * 0.5
                            partial_price = vwap
                            slip = partial_price * pcfg.base_slippage_pct
                            if pos.direction == 1:
                                partial_price -= slip
                            else:
                                partial_price += slip

                            comm = partial_price * partial_size * pcfg.commission_pct
                            if pos.direction == 1:
                                p_pnl = (partial_price - pos.entry_price) * partial_size - comm
                            else:
                                p_pnl = (pos.entry_price - partial_price) * partial_size - comm

                            pos.remaining_size -= partial_size
                            pos.pnl += p_pnl
                            pos.commission += comm
                            pos.partial_exits.append((ts, partial_price, partial_size, "tp1_vwap"))
                            capital += p_pnl

                # TP2 full exit at EMA
                if close_reason is None:
                    ema = bar.get("ema_20", None)
                    if ema and ema > 0:
                        hit = (
                            (pos.direction == 1 and bar["high"] >= ema) or
                            (pos.direction == -1 and bar["low"] <= ema)
                        )
                        if hit:
                            exit_price = ema
                            close_reason = "tp2_ema"

                # OI rising exit
                if close_reason is None:
                    oi_change = bar.get("oi_change_1h", 0)
                    if oi_change > 0.03:
                        close_reason = "oi_rising"

                # Time exit
                if close_reason is None and pos.bars_held >= max_holding_bars:
                    close_reason = "time_exit"

                if close_reason:
                    remaining = pos.remaining_size
                    if remaining > 0:
                        slip = exit_price * pcfg.base_slippage_pct
                        if pos.direction == 1:
                            exit_price -= slip
                        else:
                            exit_price += slip

                        comm = exit_price * remaining * pcfg.commission_pct
                        if pos.direction == 1:
                            pnl = (exit_price - pos.entry_price) * remaining - comm
                        else:
                            pnl = (pos.entry_price - exit_price) * remaining - comm

                        pos.pnl += pnl
                        pos.commission += comm
                        capital += pnl

                    pos.exit_time = ts
                    pos.exit_price = exit_price
                    pos.exit_reason = close_reason
                    pos.remaining_size = 0
                    pos.is_open = False
                    pos.pnl_pct = pos.pnl / (pos.entry_price * pos.size)

                    all_trades.append(pos)
                    symbols_to_close.append(sym)

            for sym in symbols_to_close:
                del positions[sym]

            # ═══════════════════════════════════════
            # OPEN NEW POSITIONS (portfolio-level gating)
            # ═══════════════════════════════════════
            for sym in aligned:
                if sym in positions:
                    continue

                signal = all_signals[sym].iloc[i]
                if signal == 0:
                    continue

                # Portfolio risk check: max positions
                if len(positions) >= pcfg.max_total_positions:
                    continue

                # Portfolio risk check: total exposure
                current_exposure = sum(
                    p.remaining_size * p.entry_price / capital
                    for p in positions.values()
                ) * self.strategy_config.base_risk_pct
                if current_exposure >= pcfg.max_total_exposure_pct:
                    continue

                # Correlation check: don't pile into correlated assets
                if positions and i >= pcfg.correlation_lookback:
                    too_correlated = False
                    for open_sym in positions:
                        corr = returns[sym].iloc[max(0, i - pcfg.correlation_lookback):i].corr(
                            returns[open_sym].iloc[max(0, i - pcfg.correlation_lookback):i]
                        )
                        if abs(corr) > pcfg.max_correlation:
                            too_correlated = True
                            break
                    if too_correlated:
                        continue

                # Open position
                bar = aligned[sym].iloc[i]
                entry_price = bar["close"]
                slip = entry_price * pcfg.base_slippage_pct
                if signal == 1:
                    entry_price += slip
                else:
                    entry_price -= slip

                size = (capital * self.strategy_config.base_risk_pct) / entry_price
                comm = entry_price * size * pcfg.commission_pct
                capital -= comm

                atr = bar.get("atr", bar["close"] * 0.02)
                positions[sym] = DetailedTrade(
                    entry_time=ts,
                    symbol=sym,
                    direction=signal,
                    entry_price=entry_price,
                    size=size,
                    remaining_size=size,
                    commission=comm,
                    entry_atr=atr,
                    entry_vwap=bar.get("vwap", entry_price),
                    entry_ema20=bar.get("ema_20", entry_price),
                    max_favorable=entry_price,
                    max_adverse=entry_price,
                )

            # Equity tracking
            unrealized = 0
            for sym, pos in positions.items():
                if pos.remaining_size > 0:
                    bar = aligned[sym].iloc[i]
                    if pos.direction == 1:
                        unrealized += (bar["close"] - pos.entry_price) * pos.remaining_size
                    else:
                        unrealized += (pos.entry_price - bar["close"]) * pos.remaining_size
            equity.append(capital + unrealized)
            equity_times.append(ts)

        # Close remaining positions
        for sym, pos in positions.items():
            bar = aligned[sym].iloc[-1]
            exit_price = bar["close"]
            remaining = pos.remaining_size
            if remaining > 0:
                comm = exit_price * remaining * pcfg.commission_pct
                if pos.direction == 1:
                    pnl = (exit_price - pos.entry_price) * remaining - comm
                else:
                    pnl = (pos.entry_price - exit_price) * remaining - comm
                pos.pnl += pnl
                pos.commission += comm
                capital += pnl

            pos.exit_time = common_index[-1]
            pos.exit_price = exit_price
            pos.exit_reason = "end_of_data"
            pos.remaining_size = 0
            pos.is_open = False
            pos.pnl_pct = pos.pnl / (pos.entry_price * pos.size)
            all_trades.append(pos)

        # Build result
        result = self._compute_portfolio_stats(all_trades, equity, equity_times)

        # Per-symbol attribution
        for sym in aligned:
            sym_trades = [t for t in all_trades if t.symbol == sym]
            result.per_symbol[sym] = {
                "trades": len(sym_trades),
                "total_pnl": sum(t.pnl for t in sym_trades),
                "win_rate": (
                    sum(1 for t in sym_trades if t.pnl > 0) / len(sym_trades)
                    if sym_trades else 0
                ),
                "avg_rr": float(np.mean([t.rr_achieved for t in sym_trades if t.rr_achieved])) if any(t.rr_achieved for t in sym_trades) else 0,
            }

        return result

    def _compute_portfolio_stats(
        self, trades: list, equity: list, equity_times: list
    ) -> PortfolioResult:
        """Compute portfolio-level statistics."""
        result = PortfolioResult()
        result.trades = trades
        result.total_trades = len(trades)

        if not equity_times:
            return result

        eq = pd.Series(equity, index=equity_times)
        result.equity_curve = eq

        eq_returns = eq.pct_change().dropna()
        result.total_return = (eq.iloc[-1] / eq.iloc[0]) - 1

        if len(eq_returns) > 1 and eq_returns.std() > 0:
            result.sharpe_ratio = eq_returns.mean() / eq_returns.std() * np.sqrt(252 * 24)

        downside = eq_returns[eq_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            result.sortino_ratio = eq_returns.mean() / downside.std() * np.sqrt(252 * 24)

        peak = eq.cummax()
        drawdown = (eq - peak) / (peak + 1e-10)
        result.max_drawdown = abs(drawdown.min())

        if trades:
            pnls = [t.pnl_pct for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            result.win_rate = len(wins) / len(pnls) if pnls else 0
            if losses:
                loss_sum = sum(abs(l) for l in losses)
                result.profit_factor = sum(wins) / loss_sum if loss_sum > 0 else float("inf")

        if len(eq) > 1:
            result.monthly_returns = eq.resample("ME").last().pct_change().dropna()

        return result


def run_multi_asset_validation(
    datasets: dict[str, pd.DataFrame],
    strategy_config: Optional[StrategyConfig] = None,
    portfolio_config: Optional[PortfolioConfig] = None,
) -> PortfolioResult:
    """Convenience function to run multi-asset backtest with logging."""
    engine = MultiAssetEngine(strategy_config, portfolio_config)
    result = engine.run_multi_backtest(datasets)

    logger.info(f"\n{'='*50}")
    logger.info("MULTI-ASSET RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Return: {result.total_return:+.1%}")
    logger.info(f"Sharpe: {result.sharpe_ratio:.2f}")
    logger.info(f"Sortino: {result.sortino_ratio:.2f}")
    logger.info(f"Max DD: {result.max_drawdown:.1%}")
    logger.info(f"Trades: {result.total_trades}")
    logger.info(f"Win Rate: {result.win_rate:.0%}")
    logger.info(f"PF: {result.profit_factor:.2f}")

    logger.info(f"\nPer-Symbol Attribution:")
    for sym, stats in result.per_symbol.items():
        logger.info(
            f"  {sym}: {stats['trades']} trades, "
            f"PnL=${stats['total_pnl']:.2f}, "
            f"WR={stats['win_rate']:.0%}"
        )

    return result

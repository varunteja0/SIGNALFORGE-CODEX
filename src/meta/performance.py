from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _stable_profit_factor(pnls: list[float]) -> float:
    wins = [value for value in pnls if value > 0]
    losses = [abs(value) for value in pnls if value <= 0]
    gross_win = float(np.sum(wins)) if wins else 0.0
    gross_loss = float(np.sum(losses)) if losses else 0.0
    if gross_loss > 0:
        return gross_win / gross_loss
    if gross_win > 0:
        return float("inf")
    return 0.0


@dataclass
class StrategyPerformanceSnapshot:
    strategy_name: str
    rolling_sharpe: float
    rolling_drawdown: float
    rolling_win_rate: float
    rolling_profit_factor: float
    expectancy: float
    total_pnl: float
    trade_count: int
    score: float
    state: str
    recommended_multiplier: float
    trailing_return: float = 0.0
    signal_strength: float = 1.0
    conviction_score: float = 0.0
    sharpe_stability: float = 0.65
    return_stability: float = 0.65
    persistence_score: float = 0.65
    annualized_return: float = 0.0
    realized_volatility: float = 0.0
    growth_score: float = 0.0
    turnover_rate: float = 0.0
    edge_retention: float = 1.0
    execution_efficiency: float = 1.0


class StrategyPerformanceTracker:
    """Track rolling strategy health for meta-allocation decisions."""

    def __init__(
        self,
        lookback_bars: int = 24 * 30,
        lookback_trades: int = 30,
        min_trades: int = 6,
        downweight_sharpe: float = 0.20,
        disable_sharpe: float = -0.25,
        downweight_drawdown: float = 0.08,
        disable_drawdown: float = 0.15,
    ):
        self.lookback_bars = lookback_bars
        self.lookback_trades = lookback_trades
        self.min_trades = min_trades
        self.downweight_sharpe = downweight_sharpe
        self.disable_sharpe = disable_sharpe
        self.downweight_drawdown = downweight_drawdown
        self.disable_drawdown = disable_drawdown
        self.history: dict[str, list[StrategyPerformanceSnapshot]] = defaultdict(list)

    def update_from_backtest(self, result) -> dict[str, StrategyPerformanceSnapshot]:
        strategy_returns = pd.DataFrame(
            {
                name: curve.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
                for name, curve in getattr(result, "strategy_equity_curves", {}).items()
                if len(curve) > 1
            }
        )
        trade_pnls: dict[str, list[float]] = defaultdict(list)
        for trade in getattr(result, "trades", []):
            strategy = getattr(trade, "strategy", None)
            if strategy is None:
                continue
            trade_pnls[strategy].append(float(getattr(trade, "pnl", 0.0) or 0.0))
        return self.update_from_streams(strategy_returns, trade_pnls)

    def update_from_streams(
        self,
        strategy_returns: pd.DataFrame,
        trade_pnls: dict[str, list[float]] | None = None,
    ) -> dict[str, StrategyPerformanceSnapshot]:
        trade_pnls = trade_pnls or {}
        snapshots: dict[str, StrategyPerformanceSnapshot] = {}
        strategy_names = sorted(set(strategy_returns.columns).union(trade_pnls))

        for strategy in strategy_names:
            returns = strategy_returns.get(strategy, pd.Series(dtype=float)).tail(self.lookback_bars)
            returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            pnls = list(trade_pnls.get(strategy, [])[-self.lookback_trades :])

            if len(returns) > 1 and float(returns.std()) > 0.0:
                sharpe = float(returns.mean() / (returns.std() + 1e-10) * np.sqrt(252 * 24))
                curve = (1.0 + returns).cumprod()
                drawdown = float(abs((curve / curve.cummax() - 1.0).min()))
                trailing_return = float(curve.iloc[-1] - 1.0)
                annualized_return = self._annualized_return(returns)
                realized_volatility = float(returns.std(ddof=0) * np.sqrt(252 * 24))
            else:
                sharpe = 0.0
                drawdown = 0.0
                trailing_return = 0.0
                annualized_return = 0.0
                realized_volatility = 0.0

            if len(returns) > 0:
                turnover_rate = float(
                    np.clip(
                        len(pnls) / max(len(returns) / 24.0, 1.0),
                        0.0,
                        3.0,
                    )
                )
            else:
                turnover_rate = 0.0

            win_rate = float(np.mean([pnl > 0 for pnl in pnls])) if pnls else 0.0
            profit_factor = _stable_profit_factor(pnls)
            expectancy = float(np.mean(pnls)) if pnls else 0.0
            total_pnl = float(np.sum(pnls)) if pnls else 0.0
            growth_score = self._growth_score(
                annualized_return=annualized_return,
                drawdown=drawdown,
                realized_volatility=realized_volatility,
            )
            score = self._score(sharpe, drawdown, win_rate, profit_factor, trailing_return)
            state, multiplier = self._classify(
                sharpe=sharpe,
                drawdown=drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                trade_count=len(pnls),
                score=score,
            )

            snapshot = StrategyPerformanceSnapshot(
                strategy_name=strategy,
                rolling_sharpe=sharpe,
                rolling_drawdown=drawdown,
                rolling_win_rate=win_rate,
                rolling_profit_factor=profit_factor,
                expectancy=expectancy,
                total_pnl=total_pnl,
                trade_count=len(pnls),
                score=score,
                state=state,
                recommended_multiplier=multiplier,
                trailing_return=trailing_return,
                annualized_return=annualized_return,
                realized_volatility=realized_volatility,
                growth_score=growth_score,
                turnover_rate=turnover_rate,
            )
            self.history[strategy].append(snapshot)
            snapshots[strategy] = snapshot

        return snapshots

    def _score(
        self,
        sharpe: float,
        drawdown: float,
        win_rate: float,
        profit_factor: float,
        trailing_return: float,
    ) -> float:
        pf_component = min(3.0, profit_factor if np.isfinite(profit_factor) else 3.0) / 3.0
        return float(
            0.45 * np.tanh(sharpe / 2.0)
            + 0.20 * (2.0 * win_rate - 1.0)
            + 0.20 * pf_component
            + 0.20 * np.tanh(trailing_return * 3.0)
            - 0.75 * drawdown
        )

    @staticmethod
    def _annualized_return(returns: pd.Series) -> float:
        clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) == 0:
            return 0.0
        wealth = float((1.0 + clean).prod())
        if wealth <= 0.0:
            return -1.0
        years = len(clean) / float(252 * 24)
        if years <= 0.0:
            return wealth - 1.0
        return float(wealth ** (1.0 / years) - 1.0)

    @staticmethod
    def _growth_score(
        *,
        annualized_return: float,
        drawdown: float,
        realized_volatility: float,
    ) -> float:
        return float(annualized_return - 1.25 * drawdown - 0.50 * realized_volatility)

    def _classify(
        self,
        *,
        sharpe: float,
        drawdown: float,
        win_rate: float,
        profit_factor: float,
        trade_count: int,
        score: float,
    ) -> tuple[str, float]:
        if trade_count < self.min_trades:
            return "warming_up", 0.80
        if (
            sharpe <= self.disable_sharpe
            or drawdown >= self.disable_drawdown
            or profit_factor < 0.80
        ):
            return "disabled", 0.0
        if (
            sharpe < self.downweight_sharpe
            or drawdown >= self.downweight_drawdown
            or win_rate < 0.45
        ):
            return "downweighted", 0.45
        if score > 0.65:
            return "boosted", 1.20
        return "active", 1.0
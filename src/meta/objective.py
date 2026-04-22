from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.meta.performance import StrategyPerformanceSnapshot


@dataclass
class PortfolioObjectiveSnapshot:
    score: float
    cagr: float
    max_drawdown: float
    volatility_tracking_error: float
    turnover: float
    components: dict[str, float] = field(default_factory=dict)


class PortfolioObjectiveScorer:
    """Score portfolio and strategy quality with explicit fund-style penalties."""

    def __init__(
        self,
        drawdown_penalty: float = 1.25,
        volatility_tracking_penalty: float = 1.10,
        turnover_penalty: float = 0.35,
        edge_retention_penalty: float = 0.40,
    ):
        self.drawdown_penalty = drawdown_penalty
        self.volatility_tracking_penalty = volatility_tracking_penalty
        self.turnover_penalty = turnover_penalty
        self.edge_retention_penalty = edge_retention_penalty

    def portfolio_score(
        self,
        *,
        cagr: float,
        max_drawdown: float,
        volatility_tracking_error: float,
        turnover: float,
    ) -> PortfolioObjectiveSnapshot:
        drawdown_cost = self.drawdown_penalty * max(0.0, float(max_drawdown))
        tracking_cost = self.volatility_tracking_penalty * max(0.0, float(volatility_tracking_error))
        turnover_cost = self.turnover_penalty * max(0.0, float(turnover))
        score = float(cagr) - drawdown_cost - tracking_cost - turnover_cost
        return PortfolioObjectiveSnapshot(
            score=score,
            cagr=float(cagr),
            max_drawdown=float(max_drawdown),
            volatility_tracking_error=float(volatility_tracking_error),
            turnover=float(turnover),
            components={
                "cagr": float(cagr),
                "drawdown_penalty": drawdown_cost,
                "volatility_tracking_penalty": tracking_cost,
                "turnover_penalty": turnover_cost,
            },
        )

    def strategy_score(
        self,
        snapshot: StrategyPerformanceSnapshot | None,
        *,
        target_volatility: float,
        edge_retention: float = 1.0,
    ) -> float:
        if snapshot is None:
            return 0.0
        tracking_error = abs(float(snapshot.realized_volatility) - max(float(target_volatility), 0.0))
        edge_penalty = self.edge_retention_penalty * max(0.0, 0.70 - float(edge_retention))
        return float(
            float(snapshot.annualized_return)
            - self.drawdown_penalty * max(0.0, float(snapshot.rolling_drawdown))
            - self.volatility_tracking_penalty * tracking_error
            - self.turnover_penalty * max(0.0, float(snapshot.turnover_rate))
            - edge_penalty
        )

    @staticmethod
    def market_scores(
        strategy_scores: dict[str, float],
        strategy_market_map: dict[str, str],
    ) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for strategy, score in strategy_scores.items():
            market = strategy_market_map.get(strategy)
            if not market:
                continue
            grouped.setdefault(market, []).append(float(score))
        return {
            market: float(np.mean(values))
            for market, values in grouped.items()
            if values
        }

    @classmethod
    def market_route_multipliers(
        cls,
        strategy_scores: dict[str, float],
        strategy_market_map: dict[str, str],
        *,
        tilt: float = 0.25,
    ) -> dict[str, float]:
        market_scores = cls.market_scores(strategy_scores, strategy_market_map)
        if not market_scores or tilt <= 0.0:
            return {
                market: 1.0
                for market in market_scores
            }
        values = np.array(list(market_scores.values()), dtype=float)
        center = float(values.mean()) if len(values) else 0.0
        scale = float(max(values.std(ddof=0), 0.05)) if len(values) else 0.05
        return {
            market: float(np.clip(1.0 + tilt * np.tanh((score - center) / scale), 0.70, 1.35))
            for market, score in market_scores.items()
        }
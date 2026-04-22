from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CapitalUtilizationDecision:
    target_utilization: float
    projected_utilization: float
    utilization_gap: float
    risk_budget_multiplier: float
    idle_capital_penalty: float
    deployment_boosts: dict[str, float] = field(default_factory=dict)
    capital_usage_by_strategy: dict[str, float] = field(default_factory=dict)


class CapitalUtilizationEngine:
    """Increase capital deployment when the portfolio is underutilized."""

    def __init__(
        self,
        target_utilization: float = 0.80,
        min_utilization: float = 0.60,
        max_risk_budget_multiplier: float = 1.60,
        idle_pressure: float = 0.35,
        conviction_boost: float = 0.30,
    ):
        self.target_utilization = target_utilization
        self.min_utilization = min_utilization
        self.max_risk_budget_multiplier = max_risk_budget_multiplier
        self.idle_pressure = idle_pressure
        self.conviction_boost = conviction_boost

    def evaluate(
        self,
        *,
        base_position_sizes: dict[str, float],
        strategy_weights: dict[str, float],
        signal_strengths: dict[str, float],
        conviction_scores: dict[str, float],
        base_risk_budget: float,
    ) -> CapitalUtilizationDecision:
        strategy_count = max(1, len(strategy_weights))
        capital_usage_by_strategy = {}
        for strategy, weight in strategy_weights.items():
            usage = (
                float(base_position_sizes.get(strategy, 0.0))
                * max(0.0, float(weight))
                * strategy_count
                * max(0.0, float(signal_strengths.get(strategy, 0.0)))
            )
            capital_usage_by_strategy[strategy] = usage

        projected_risk = float(sum(capital_usage_by_strategy.values()))
        if base_risk_budget > 0.0:
            projected_utilization = float(np.clip(projected_risk / base_risk_budget, 0.0, 2.0))
        else:
            projected_utilization = 0.0

        utilization_gap = max(self.target_utilization - projected_utilization, 0.0)
        need_pressure = projected_utilization < self.min_utilization
        if need_pressure:
            multiplier = 1.0 + utilization_gap / max(self.target_utilization, 1e-9)
        else:
            multiplier = 1.0 + 0.35 * utilization_gap / max(self.target_utilization, 1e-9)
        risk_budget_multiplier = float(np.clip(multiplier, 1.0, self.max_risk_budget_multiplier))
        idle_capital_penalty = float(np.clip(utilization_gap * self.idle_pressure, 0.0, 1.0))

        deployment_boosts: dict[str, float] = {}
        positive_convictions = {
            strategy: max(0.0, float(value))
            for strategy, value in conviction_scores.items()
        }
        total_conviction = float(sum(positive_convictions.values()))
        if total_conviction > 0.0 and utilization_gap > 0.0:
            for strategy, conviction in positive_convictions.items():
                deployment_boosts[strategy] = float(
                    self.conviction_boost
                    * utilization_gap
                    * conviction
                    / total_conviction
                )

        return CapitalUtilizationDecision(
            target_utilization=self.target_utilization,
            projected_utilization=projected_utilization,
            utilization_gap=utilization_gap,
            risk_budget_multiplier=risk_budget_multiplier,
            idle_capital_penalty=idle_capital_penalty,
            deployment_boosts=deployment_boosts,
            capital_usage_by_strategy=capital_usage_by_strategy,
        )
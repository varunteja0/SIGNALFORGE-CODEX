from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.allocation.capital_efficiency import CapitalUtilizationEngine
from src.meta.objective import PortfolioObjectiveScorer
from src.meta.performance import StrategyPerformanceSnapshot
from src.risk.portfolio import PortfolioOptimizer


@dataclass
class AllocationDecision:
    weights: dict[str, float]
    gross_exposure_scale: float
    disabled_strategies: dict[str, str] = field(default_factory=dict)
    edge_scores: dict[str, float] = field(default_factory=dict)
    performance_multipliers: dict[str, float] = field(default_factory=dict)
    regime_multipliers: dict[str, float] = field(default_factory=dict)
    correlation_penalties: dict[str, float] = field(default_factory=dict)
    signal_strengths: dict[str, float] = field(default_factory=dict)
    conviction_scores: dict[str, float] = field(default_factory=dict)
    growth_scores: dict[str, float] = field(default_factory=dict)
    persistence_scores: dict[str, float] = field(default_factory=dict)
    regime_confidences: dict[str, float] = field(default_factory=dict)
    position_size_multipliers: dict[str, float] = field(default_factory=dict)
    exploit_weights: dict[str, float] = field(default_factory=dict)
    exploration_weights: dict[str, float] = field(default_factory=dict)
    capital_usage_by_strategy: dict[str, float] = field(default_factory=dict)
    projected_utilization: float = 0.0
    target_utilization: float = 0.0
    risk_budget_multiplier: float = 1.0
    idle_capital_penalty: float = 0.0
    target_volatility: float = 0.0
    realized_volatility: float = 0.0
    volatility_multiplier: float = 1.0
    correlation_multiplier: float = 1.0
    strategy_correlation: float = 0.0
    asset_correlation: float = 0.0
    correlation_shock: bool = False
    edge_retention_scores: dict[str, float] = field(default_factory=dict)
    execution_efficiency_penalties: dict[str, float] = field(default_factory=dict)
    strategy_objective_scores: dict[str, float] = field(default_factory=dict)
    market_route_multipliers: dict[str, float] = field(default_factory=dict)
    market_objective_scores: dict[str, float] = field(default_factory=dict)
    weight_turnover: float = 0.0
    portfolio_growth_score: float = 0.0
    portfolio_objective_score: float = 0.0
    portfolio_objective_components: dict[str, float] = field(default_factory=dict)
    expected_sharpe: float = 0.0
    expected_return: float = 0.0
    expected_vol: float = 0.0


class AdaptiveAllocationEngine:
    """Allocate capital using edge, correlation, and drawdown-aware overlays."""

    def __init__(
        self,
        max_weight: float = 0.45,
        min_weight: float = 0.0,
        optimizer_method: str = "hrp",
        blend: float = 0.55,
        conviction_weight: float = 0.50,
        exploration_epsilon: float = 0.08,
        weight_inertia: float = 0.20,
        max_gross_exposure_scale: float = 1.75,
        target_drawdown: float = 0.08,
        hard_drawdown_cap: float = 0.18,
        capital_utilization_engine: CapitalUtilizationEngine | None = None,
        objective_scorer: PortfolioObjectiveScorer | None = None,
        objective_weight: float = 0.30,
        execution_efficiency_tilt: float = 0.35,
        market_routing_tilt: float = 0.25,
    ):
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.optimizer_method = optimizer_method
        self.blend = blend
        self.conviction_weight = conviction_weight
        self.exploration_epsilon = exploration_epsilon
        self.weight_inertia = weight_inertia
        self.max_gross_exposure_scale = max_gross_exposure_scale
        self.target_drawdown = target_drawdown
        self.hard_drawdown_cap = hard_drawdown_cap
        self.capital_utilization_engine = capital_utilization_engine or CapitalUtilizationEngine()
        self.objective_scorer = objective_scorer or PortfolioObjectiveScorer()
        self.objective_weight = objective_weight
        self.execution_efficiency_tilt = execution_efficiency_tilt
        self.market_routing_tilt = market_routing_tilt

    def allocate(
        self,
        strategy_returns: pd.DataFrame,
        performance_metrics: dict[str, StrategyPerformanceSnapshot],
        regime_multipliers: dict[str, float] | None = None,
        portfolio_drawdown: float = 0.0,
        signal_strengths: dict[str, float] | None = None,
        persistence_scores: dict[str, float] | None = None,
        regime_confidences: dict[str, float] | None = None,
        edge_retention_scores: dict[str, float] | None = None,
        strategy_market_map: dict[str, str] | None = None,
        base_position_sizes: dict[str, float] | None = None,
        previous_weights: dict[str, float] | None = None,
        base_risk_budget: float = 1.0,
        target_volatility: float = 0.15,
    ) -> AllocationDecision:
        regime_multipliers = regime_multipliers or {}
        signal_strengths = signal_strengths or {}
        persistence_scores = persistence_scores or {}
        regime_confidences = regime_confidences or {}
        edge_retention_scores = edge_retention_scores or {}
        strategy_market_map = strategy_market_map or {}
        base_position_sizes = base_position_sizes or {}
        previous_weights = previous_weights or {}
        strategy_names = sorted(
            set(strategy_returns.columns).union(performance_metrics).union(regime_multipliers)
        )
        disabled: dict[str, str] = {}

        if not strategy_names:
            return AllocationDecision(weights={}, gross_exposure_scale=0.0)

        if strategy_returns.empty:
            strategy_returns = pd.DataFrame(0.0, index=pd.RangeIndex(1), columns=strategy_names)

        aligned_returns = strategy_returns.reindex(columns=strategy_names, fill_value=0.0).fillna(0.0)
        corr = aligned_returns.corr().fillna(0.0) if len(aligned_returns.columns) > 1 else pd.DataFrame()

        edge_scores: dict[str, float] = {}
        performance_multipliers: dict[str, float] = {}
        regime_scale_map: dict[str, float] = {}
        correlation_penalties: dict[str, float] = {}
        conviction_scores: dict[str, float] = {}
        growth_scores: dict[str, float] = {}
        persistence_map: dict[str, float] = {}
        regime_confidence_map: dict[str, float] = {}
        signal_strength_map: dict[str, float] = {}
        position_size_multipliers: dict[str, float] = {}
        execution_efficiency_penalties: dict[str, float] = {}
        strategy_objective_scores: dict[str, float] = {}
        clamped_edge_retention: dict[str, float] = {}

        for strategy in strategy_names:
            snapshot = performance_metrics.get(strategy)
            perf_mult = snapshot.recommended_multiplier if snapshot else 1.0
            performance_multipliers[strategy] = perf_mult

            regime_mult = float(regime_multipliers.get(strategy, 1.0))
            regime_scale_map[strategy] = regime_mult

            if strategy in corr.columns:
                avg_corr = float(corr[strategy].drop(labels=[strategy], errors="ignore").abs().mean())
            else:
                avg_corr = 0.0
            corr_penalty = 1.0 / (1.0 + avg_corr)
            correlation_penalties[strategy] = corr_penalty

            signal_strength = float(np.clip(signal_strengths.get(strategy, 1.0), 0.0, 1.5))
            persistence = float(np.clip(persistence_scores.get(strategy, 0.65), 0.0, 1.25))
            regime_confidence = float(np.clip(regime_confidences.get(strategy, 0.65), 0.0, 1.0))
            signal_strength_map[strategy] = signal_strength
            persistence_map[strategy] = persistence
            regime_confidence_map[strategy] = regime_confidence

            edge_retention = float(np.clip(edge_retention_scores.get(strategy, 1.0), 0.0, 1.25))
            clamped_edge_retention[strategy] = edge_retention
            execution_efficiency_penalties[strategy] = float(
                np.clip(
                    1.0 - self.execution_efficiency_tilt * max(0.0, 0.70 - edge_retention),
                    0.45,
                    1.05,
                )
            )
            strategy_objective_scores[strategy] = self.objective_scorer.strategy_score(
                snapshot,
                target_volatility=target_volatility,
                edge_retention=edge_retention,
            ) if snapshot is not None else 0.0

            if snapshot:
                conviction = max(0.0, snapshot.rolling_sharpe) * snapshot.rolling_win_rate * signal_strength
                growth_score = float(snapshot.growth_score)
            else:
                conviction = 0.0
                growth_score = 0.0
            conviction_scores[strategy] = conviction
            growth_scores[strategy] = growth_score

            conviction_mult = 1.0 + self.conviction_weight * np.tanh(conviction)
            position_size_multipliers[strategy] = float(
                np.clip(
                    0.70
                    + 0.60 * np.tanh(conviction)
                    + 0.20 * np.tanh(growth_score / 0.20)
                    + 0.20 * persistence
                    + 0.15 * regime_confidence,
                    0.50,
                    1.80,
                )
            )

            if snapshot and snapshot.state == "disabled":
                disabled[strategy] = "performance_disabled"
                edge_scores[strategy] = 0.0
                continue
            if snapshot and edge_retention < 0.15:
                disabled[strategy] = "broken_execution_edge"
                edge_scores[strategy] = 0.0
                continue

        market_objective_scores = self.objective_scorer.market_scores(
            strategy_objective_scores,
            strategy_market_map,
        )
        market_route_multipliers = self.objective_scorer.market_route_multipliers(
            strategy_objective_scores,
            strategy_market_map,
            tilt=self.market_routing_tilt,
        )

        for strategy in strategy_names:
            if strategy in disabled and edge_scores.get(strategy, 0.0) <= 0.0:
                continue

            snapshot = performance_metrics.get(strategy)
            perf_mult = performance_multipliers[strategy]
            regime_mult = regime_scale_map[strategy]
            corr_penalty = correlation_penalties[strategy]
            signal_strength = signal_strength_map[strategy]
            persistence = persistence_map[strategy]
            regime_confidence = regime_confidence_map[strategy]
            conviction_mult = 1.0 + self.conviction_weight * np.tanh(conviction_scores[strategy])
            growth_score = growth_scores[strategy]

            score = snapshot.score if snapshot else 0.0
            objective_multiplier = float(
                np.clip(
                    1.0 + self.objective_weight * np.tanh(strategy_objective_scores.get(strategy, 0.0) / 0.10),
                    0.70,
                    1.30,
                )
            )
            market_key = strategy_market_map.get(strategy)
            market_multiplier = float(market_route_multipliers.get(market_key, 1.0))
            edge = (
                max(0.0, 0.45 + score)
                * perf_mult
                * regime_mult
                * corr_penalty
                * (0.65 + 0.35 * signal_strength)
                * (0.70 + 0.45 * persistence)
                * (0.75 + 0.35 * regime_confidence)
                * (1.0 + 0.20 * np.tanh(growth_score / 0.20))
                * conviction_mult
                * execution_efficiency_penalties.get(strategy, 1.0)
                * objective_multiplier
                * market_multiplier
            )
            edge_scores[strategy] = edge
            if edge <= 1e-8:
                disabled[strategy] = disabled.get(strategy, "no_edge")

        active = [name for name in strategy_names if edge_scores.get(name, 0.0) > 1e-8]
        if not active:
            active = list(strategy_names)
            for name in active:
                edge_scores[name] = 1.0

        raw_weights = np.array([edge_scores[name] for name in active], dtype=float)
        raw_weights = raw_weights / raw_weights.sum()

        optimizer_max = max(self.max_weight, 1.0 / len(active))
        optimizer = PortfolioOptimizer(
            method=self.optimizer_method,
            max_weight=optimizer_max,
            min_weight=self.min_weight,
            lookback_periods=min(100, len(aligned_returns)),
        )
        optimized = optimizer.optimize(aligned_returns[active])
        optimized_weights = np.array([optimized.weights.get(name, 0.0) for name in active], dtype=float)
        if optimized_weights.sum() <= 0.0:
            optimized_weights = raw_weights.copy()

        blended = self.blend * raw_weights + (1.0 - self.blend) * optimized_weights

        if previous_weights:
            previous = np.array([max(0.0, previous_weights.get(name, 0.0)) for name in active], dtype=float)
            if previous.sum() > 0.0:
                previous = previous / previous.sum()
                blended = self.weight_inertia * previous + (1.0 - self.weight_inertia) * blended

        blended = np.clip(blended, self.min_weight, optimizer_max)
        blended = blended / blended.sum()

        provisional_weights = {name: 0.0 for name in strategy_names}
        for idx, name in enumerate(active):
            provisional_weights[name] = float(blended[idx])

        utilization = self.capital_utilization_engine.evaluate(
            base_position_sizes=base_position_sizes,
            strategy_weights=provisional_weights,
            signal_strengths=signal_strength_map,
            conviction_scores=conviction_scores,
            base_risk_budget=base_risk_budget,
        )

        boosted = blended.copy()
        for idx, name in enumerate(active):
            boosted[idx] *= 1.0 + utilization.deployment_boosts.get(name, 0.0)
        boosted = boosted / boosted.sum()

        weights = {name: 0.0 for name in strategy_names}
        for idx, name in enumerate(active):
            weights[name] = float(boosted[idx])

        weight_turnover = 0.0
        if previous_weights:
            weight_turnover = float(
                sum(
                    abs(weights.get(name, 0.0) - float(previous_weights.get(name, 0.0)))
                    for name in strategy_names
                )
            )

        exploration_weights = {name: 0.0 for name in strategy_names}
        exploit_weights = dict(weights)
        explore_candidates = self._exploration_candidates(
            active,
            performance_metrics,
            signal_strength_map,
            persistence_map,
        )
        if self.exploration_epsilon > 0.0 and explore_candidates:
            explore_scores = np.array([explore_candidates[name] for name in explore_candidates], dtype=float)
            explore_scores = explore_scores / explore_scores.sum()
            epsilon = min(float(self.exploration_epsilon), 0.20)
            for idx, name in enumerate(explore_candidates):
                exploratory = min(weights[name] * 0.40, epsilon * float(explore_scores[idx]))
                exploration_weights[name] = exploratory
                exploit_weights[name] = max(0.0, exploit_weights[name] - exploratory)

        exposure_scale = 1.0
        if portfolio_drawdown > self.target_drawdown:
            drawdown_over = min(
                1.0,
                (portfolio_drawdown - self.target_drawdown)
                / max(1e-10, self.hard_drawdown_cap - self.target_drawdown),
            )
            exposure_scale = float(np.clip(1.0 - drawdown_over, 0.35, 1.0))

        avg_regime_confidence = float(
            np.mean([regime_confidence_map.get(name, 0.65) for name in active])
        ) if active else 0.65
        exposure_scale *= 0.75 + 0.45 * avg_regime_confidence
        exposure_scale *= utilization.risk_budget_multiplier
        exposure_scale = float(np.clip(exposure_scale, 0.30, self.max_gross_exposure_scale))

        weighted_cagr = float(
            sum(
                weights.get(name, 0.0) * float(performance_metrics.get(name).annualized_return)
                for name in active
                if performance_metrics.get(name) is not None
            )
        )
        weighted_drawdown = float(
            sum(
                weights.get(name, 0.0) * float(performance_metrics.get(name).rolling_drawdown)
                for name in active
                if performance_metrics.get(name) is not None
            )
        )
        weighted_tracking_error = float(
            sum(
                weights.get(name, 0.0)
                * abs(float(performance_metrics.get(name).realized_volatility) - float(target_volatility))
                for name in active
                if performance_metrics.get(name) is not None
            )
        )
        objective_snapshot = self.objective_scorer.portfolio_score(
            cagr=weighted_cagr,
            max_drawdown=weighted_drawdown,
            volatility_tracking_error=weighted_tracking_error,
            turnover=weight_turnover,
        )

        return AllocationDecision(
            weights=weights,
            gross_exposure_scale=exposure_scale,
            disabled_strategies=disabled,
            edge_scores=edge_scores,
            performance_multipliers=performance_multipliers,
            regime_multipliers=regime_scale_map,
            correlation_penalties=correlation_penalties,
            signal_strengths=signal_strength_map,
            conviction_scores=conviction_scores,
            growth_scores=growth_scores,
            persistence_scores=persistence_map,
            regime_confidences=regime_confidence_map,
            position_size_multipliers=position_size_multipliers,
            exploit_weights=exploit_weights,
            exploration_weights=exploration_weights,
            capital_usage_by_strategy=utilization.capital_usage_by_strategy,
            projected_utilization=utilization.projected_utilization,
            target_utilization=utilization.target_utilization,
            risk_budget_multiplier=utilization.risk_budget_multiplier,
            idle_capital_penalty=utilization.idle_capital_penalty,
            edge_retention_scores=clamped_edge_retention,
            execution_efficiency_penalties=execution_efficiency_penalties,
            strategy_objective_scores=strategy_objective_scores,
            market_route_multipliers=market_route_multipliers,
            market_objective_scores=market_objective_scores,
            weight_turnover=weight_turnover,
            portfolio_growth_score=float(np.mean(list(growth_scores.values()))) if growth_scores else 0.0,
            portfolio_objective_score=objective_snapshot.score,
            portfolio_objective_components=dict(objective_snapshot.components),
            expected_sharpe=optimized.expected_sharpe,
            expected_return=optimized.expected_return,
            expected_vol=optimized.expected_vol,
        )

    def apply_portfolio_objective(
        self,
        allocation_decision: AllocationDecision,
        *,
        cagr: float,
        max_drawdown: float,
        volatility_tracking_error: float,
    ):
        snapshot = self.objective_scorer.portfolio_score(
            cagr=cagr,
            max_drawdown=max_drawdown,
            volatility_tracking_error=volatility_tracking_error,
            turnover=allocation_decision.weight_turnover,
        )
        allocation_decision.portfolio_objective_score = snapshot.score
        allocation_decision.portfolio_objective_components = dict(snapshot.components)
        return snapshot

    @staticmethod
    def _exploration_candidates(
        active: list[str],
        performance_metrics: dict[str, StrategyPerformanceSnapshot],
        signal_strengths: dict[str, float],
        persistence_scores: dict[str, float],
    ) -> dict[str, float]:
        scores = {}
        for name in active:
            snapshot = performance_metrics.get(name)
            signal_strength = max(0.05, signal_strengths.get(name, 0.0))
            persistence = persistence_scores.get(name, 0.65)
            warming_bonus = 0.25 if snapshot and snapshot.state == "warming_up" else 0.0
            exploration_score = signal_strength * max(0.0, 1.05 - persistence) + warming_bonus
            if exploration_score > 0.0:
                scores[name] = float(exploration_score)

        if not scores and active:
            strongest = max(active, key=lambda name: signal_strengths.get(name, 0.0))
            scores[strongest] = max(0.05, signal_strengths.get(strongest, 0.0))
        return scores
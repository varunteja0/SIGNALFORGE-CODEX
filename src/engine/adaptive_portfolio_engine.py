from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from src.allocation.adaptive import AdaptiveAllocationEngine, AllocationDecision
from src.backtest.walk_forward import make_folds
from src.data.market_data import AssetSpec, MarketType, UnifiedDataBundle, UnifiedMarketDataAdapter
from src.engine.alpha_primitives import PrimitiveSlotSpec, get_alpha_primitive
from src.engine.portfolio_engine import PortfolioBacktestResult, PortfolioEngine, StrategySlot
from src.execution.liquidity import LiquidityConstraintDecision, LiquidityConstraintEngine
from src.execution.reality_gap import RealityGapSnapshot, RealityGapValidator
from src.execution.realism import ExecutionAdjustmentSummary, ExecutionRealismEngine
from src.learning.evolution import StrategyEvolutionEngine
from src.learning.online import LearningResult, OnlineLearningLoop
from src.meta.lifecycle import LifecycleDecision, StrategyLifecycleManager
from src.meta.persistence import EdgePersistenceScorer, PersistenceSnapshot
from src.meta.performance import StrategyPerformanceSnapshot, StrategyPerformanceTracker
from src.regime.adaptive import RegimeDetectionEngine, RegimeState
from src.risk.adaptive_controls import PortfolioRiskController, PortfolioRiskSnapshot


@dataclass
class AdaptiveCycleReport:
    bundle: UnifiedDataBundle
    base_result: object
    strategy_returns: pd.DataFrame
    performance_metrics: dict[str, StrategyPerformanceSnapshot]
    persistence_metrics: dict[str, PersistenceSnapshot]
    regime_states: dict[str, RegimeState]
    allocation_decision: AllocationDecision
    learning_result: LearningResult
    adapted_engine: PortfolioEngine
    execution_summary: ExecutionAdjustmentSummary | None = None
    risk_snapshot: PortfolioRiskSnapshot | None = None
    liquidity_snapshot: LiquidityConstraintDecision | None = None
    lifecycle_decisions: dict[str, LifecycleDecision] = field(default_factory=dict)
    reality_gap: RealityGapSnapshot | None = None
    suggested_position_sizes: dict[str, float] = field(default_factory=dict)
    adapted_primitive_specs: list[PrimitiveSlotSpec] = field(default_factory=list)


@dataclass
class AdaptiveWalkForwardFold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_summary: dict[str, float]
    test_summary: dict[str, float]
    deployed_weights: dict[str, float] = field(default_factory=dict)
    next_cycle_weights: dict[str, float] = field(default_factory=dict)
    deployed_position_sizes: dict[str, float] = field(default_factory=dict)
    next_cycle_position_sizes: dict[str, float] = field(default_factory=dict)
    disabled_strategies: dict[str, str] = field(default_factory=dict)
    projected_utilization: float = 0.0
    next_projected_utilization: float = 0.0
    execution_cost: float = 0.0
    target_volatility: float = 0.0
    realized_volatility: float = 0.0
    volatility_tracking_error: float = 0.0
    correlation_shock: bool = False
    reality_gap_fraction: float = 0.0
    edge_retention_ratio: float = 0.0
    portfolio_objective_score: float = 0.0
    retired_strategies: list[str] = field(default_factory=list)


@dataclass
class AdaptiveWalkForwardResult:
    folds: list[AdaptiveWalkForwardFold]
    summary: dict[str, float]
    portfolio_returns: pd.Series
    equity_curve: pd.Series
    final_engine: PortfolioEngine


@dataclass
class AdaptiveEngineState:
    previous_weights: dict[str, float] = field(default_factory=dict)
    learned_parameters: dict[str, dict] = field(default_factory=dict)
    performance_metrics: dict[str, StrategyPerformanceSnapshot] = field(default_factory=dict)
    persistence_metrics: dict[str, PersistenceSnapshot] = field(default_factory=dict)
    lifecycle_metrics: dict[str, LifecycleDecision] = field(default_factory=dict)


class AdaptivePortfolioEngine:
    """Adaptive wrapper on top of the existing PortfolioEngine."""

    def __init__(
        self,
        base_engine: PortfolioEngine,
        asset_specs: dict[str, AssetSpec],
        *,
        strategy_primitive_map: dict[str, str] | None = None,
        primitive_specs: list[PrimitiveSlotSpec] | None = None,
        data_adapter: UnifiedMarketDataAdapter | None = None,
        performance_tracker: StrategyPerformanceTracker | None = None,
        persistence_scorer: EdgePersistenceScorer | None = None,
        regime_engine: RegimeDetectionEngine | None = None,
        allocator: AdaptiveAllocationEngine | None = None,
        learner: OnlineLearningLoop | None = None,
        execution_realism: ExecutionRealismEngine | None = None,
        risk_controller: PortfolioRiskController | None = None,
        liquidity_engine: LiquidityConstraintEngine | None = None,
        lifecycle_manager: StrategyLifecycleManager | None = None,
        reality_gap_validator: RealityGapValidator | None = None,
        evolution_engine: StrategyEvolutionEngine | None = None,
        state: AdaptiveEngineState | None = None,
    ):
        self.base_engine = base_engine
        self.asset_specs = asset_specs
        self.primitive_specs = primitive_specs or []
        self.data_adapter = data_adapter or UnifiedMarketDataAdapter()
        self.performance_tracker = performance_tracker or StrategyPerformanceTracker()
        self.persistence_scorer = persistence_scorer or EdgePersistenceScorer()
        self.regime_engine = regime_engine or RegimeDetectionEngine()
        self.allocator = allocator or AdaptiveAllocationEngine()
        self.learner = learner or OnlineLearningLoop()
        self.execution_realism = execution_realism or ExecutionRealismEngine()
        self.risk_controller = risk_controller or PortfolioRiskController()
        self.liquidity_engine = liquidity_engine or LiquidityConstraintEngine()
        self.lifecycle_manager = lifecycle_manager or StrategyLifecycleManager()
        self.reality_gap_validator = reality_gap_validator or RealityGapValidator()
        self.evolution_engine = evolution_engine or StrategyEvolutionEngine()
        self.state = state or AdaptiveEngineState()
        self.strategy_primitive_map = strategy_primitive_map or self._infer_primitive_map(base_engine)

    @classmethod
    def from_portfolio_engine(
        cls,
        base_engine: PortfolioEngine,
        asset_specs: dict[str, AssetSpec] | None = None,
        strategy_primitive_map: dict[str, str] | None = None,
    ) -> "AdaptivePortfolioEngine":
        if asset_specs is None:
            asset_specs = {
                symbol: AssetSpec(symbol=symbol, market_type=MarketType.CRYPTO)
                for symbol in base_engine.assets
            }
        return cls(
            base_engine=base_engine,
            asset_specs=asset_specs,
            strategy_primitive_map=strategy_primitive_map,
        )

    @classmethod
    def from_primitive_specs(
        cls,
        primitive_specs: list[PrimitiveSlotSpec],
        asset_specs: list[AssetSpec] | dict[str, AssetSpec],
        *,
        capital: float = 10_000,
        data_days: int = 365,
        max_total_exposure: float = 0.10,
        max_position_notional_pct: float = 0.20,
        max_drawdown_kill: float = 0.15,
        use_risk_manager: bool = True,
        use_divergence_tracker: bool = False,
        use_market_state_brain: bool = True,
        use_execution_edge: bool = True,
        use_capital_scaling: bool = True,
    ) -> "AdaptivePortfolioEngine":
        asset_map = cls._coerce_asset_specs(asset_specs)
        slots = []
        strategy_primitive_map = {}

        for spec in primitive_specs:
            asset = asset_map[spec.asset_key]
            primitive = get_alpha_primitive(spec.primitive_name)
            signal_func = (
                lambda df, primitive=primitive, asset=asset, config=dict(spec.config):
                primitive.generate_signals(df, asset, config)
            )
            slots.append(
                StrategySlot(
                    name=spec.name,
                    signal_func=signal_func,
                    template=spec.primitive_name,
                    params=dict(spec.config),
                    allowed_assets=[asset.symbol],
                    regime_filter=None,
                    stop_loss_atr=spec.stop_loss_atr,
                    take_profit_atr=spec.take_profit_atr,
                    max_holding_bars=spec.max_holding_bars,
                    position_size_pct=spec.position_size_pct,
                    use_vwap=spec.use_vwap,
                )
            )
            strategy_primitive_map[spec.name] = spec.primitive_name

        engine = PortfolioEngine(
            slots=slots,
            assets=[asset.symbol for asset in asset_map.values()],
            capital=capital,
            data_days=data_days,
            max_total_exposure=max_total_exposure,
            max_position_notional_pct=max_position_notional_pct,
            max_drawdown_kill=max_drawdown_kill,
            use_regime_allocator=False,
            use_risk_manager=use_risk_manager,
            use_divergence_tracker=use_divergence_tracker,
            use_market_state_brain=use_market_state_brain,
            use_execution_edge=use_execution_edge,
            use_live_adaptation=False,
            use_capital_scaling=use_capital_scaling,
        )
        return cls(
            base_engine=engine,
            asset_specs={asset.symbol: asset for asset in asset_map.values()},
            strategy_primitive_map=strategy_primitive_map,
            primitive_specs=primitive_specs,
        )

    def prepare_datasets(self, datasets: dict[str, pd.DataFrame] | None = None) -> UnifiedDataBundle:
        raw = datasets or self.base_engine.load_data()
        return self.data_adapter.build_bundle(raw, self.asset_specs)

    def backtest_adaptive_cycle(
        self,
        datasets: dict[str, pd.DataFrame] | None = None,
    ) -> AdaptiveCycleReport:
        bundle = self.prepare_datasets(datasets)
        base_result = self.base_engine.backtest(bundle.datasets)
        return self._build_cycle_report(bundle, base_result)

    def walk_forward_adaptive(
        self,
        datasets: dict[str, pd.DataFrame] | None = None,
        *,
        n_folds: int = 5,
        train_bars: int | None = None,
        test_bars: int | None = None,
        anchored: bool = True,
        warmup_bars: int = 250,
    ) -> AdaptiveWalkForwardResult:
        bundle = self.prepare_datasets(datasets)
        shared_index = self._shared_index(bundle.datasets)
        effective_warmup_bars = max(int(warmup_bars), 200)
        folds = make_folds(
            shared_index,
            n_folds=n_folds,
            train_bars=train_bars,
            test_bars=test_bars,
            anchored=anchored,
            min_train_bars=effective_warmup_bars,
        )

        runner = self._spawn_walk_forward_runner()
        starting_capital = float(runner.base_engine.capital)
        current_capital = starting_capital
        current_primitive_specs = deepcopy(runner.primitive_specs)

        fold_reports: list[AdaptiveWalkForwardFold] = []
        portfolio_return_parts: list[pd.Series] = []

        for fold in folds:
            runner.base_engine.capital = current_capital
            runner.primitive_specs = deepcopy(current_primitive_specs)

            train_datasets = self._slice_datasets(bundle.datasets, fold.train_start, fold.train_end)
            train_bundle = UnifiedDataBundle(assets=bundle.assets, datasets=train_datasets)
            train_result = runner.base_engine.backtest(train_bundle.datasets)
            train_report = runner._build_cycle_report(train_bundle, train_result)

            deployed_engine = train_report.adapted_engine
            deployed_engine.capital = current_capital
            deployed_primitive_specs = deepcopy(
                train_report.adapted_primitive_specs or runner.primitive_specs
            )

            deployment_window = self._slice_datasets_with_warmup(
                bundle.datasets,
                shared_index,
                fold.test_start,
                fold.test_end,
                effective_warmup_bars,
            )
            gated_engine = runner._activation_gated_engine(
                deployed_engine,
                active_from=fold.test_start,
                capital=current_capital,
            )
            combined_test_result = gated_engine.backtest(deployment_window)
            test_result, test_returns = runner._slice_backtest_result(
                combined_test_result,
                start=fold.test_start,
                end=fold.test_end,
                initial_capital=current_capital,
            )

            runner.base_engine = deployed_engine
            runner.primitive_specs = deepcopy(deployed_primitive_specs)
            test_bundle = UnifiedDataBundle(
                assets=bundle.assets,
                datasets=self._slice_datasets(bundle.datasets, fold.test_start, fold.test_end),
            )
            test_report = runner._build_cycle_report(test_bundle, test_result)

            fold_reports.append(
                AdaptiveWalkForwardFold(
                    index=fold.index,
                    train_start=fold.train_start,
                    train_end=fold.train_end,
                    test_start=fold.test_start,
                    test_end=fold.test_end,
                    train_summary=self._summarize_backtest_result(train_report.base_result, current_capital),
                    test_summary=self._summarize_backtest_result(test_result, current_capital),
                    deployed_weights=dict(train_report.allocation_decision.weights),
                    next_cycle_weights=dict(test_report.allocation_decision.weights),
                    deployed_position_sizes=dict(train_report.suggested_position_sizes),
                    next_cycle_position_sizes=dict(test_report.suggested_position_sizes),
                    disabled_strategies=dict(train_report.allocation_decision.disabled_strategies),
                    projected_utilization=float(
                        train_report.allocation_decision.projected_utilization
                        * train_report.allocation_decision.risk_budget_multiplier
                        * train_report.allocation_decision.gross_exposure_scale
                    ),
                    next_projected_utilization=float(
                        test_report.allocation_decision.projected_utilization
                        * test_report.allocation_decision.risk_budget_multiplier
                        * test_report.allocation_decision.gross_exposure_scale
                    ),
                    execution_cost=float(
                        (train_report.execution_summary.total_execution_cost if train_report.execution_summary else 0.0)
                        + (test_report.execution_summary.total_execution_cost if test_report.execution_summary else 0.0)
                    ),
                    target_volatility=float(train_report.allocation_decision.target_volatility),
                    realized_volatility=float(train_report.allocation_decision.realized_volatility),
                    volatility_tracking_error=float(train_report.risk_snapshot.volatility_tracking_error) if train_report.risk_snapshot else 0.0,
                    correlation_shock=bool(train_report.allocation_decision.correlation_shock),
                    reality_gap_fraction=float(train_report.reality_gap.pnl_gap_fraction) if train_report.reality_gap else 0.0,
                    edge_retention_ratio=float(train_report.reality_gap.edge_retention_ratio) if train_report.reality_gap else 0.0,
                    portfolio_objective_score=float(train_report.allocation_decision.portfolio_objective_score),
                    retired_strategies=[
                        name
                        for name, decision in train_report.lifecycle_decisions.items()
                        if decision.retired
                    ],
                )
            )
            portfolio_return_parts.append(test_returns)

            if len(test_result.equity_curve) > 0:
                current_capital = float(test_result.equity_curve.iloc[-1])
            runner.base_engine = test_report.adapted_engine
            current_primitive_specs = deepcopy(
                test_report.adapted_primitive_specs or deployed_primitive_specs
            )

        if portfolio_return_parts:
            portfolio_returns = pd.concat(portfolio_return_parts).sort_index()
            portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep="last")]
        else:
            portfolio_returns = pd.Series(dtype=float)

        if len(portfolio_returns) > 0:
            equity_curve = starting_capital * (1.0 + portfolio_returns).cumprod()
        else:
            equity_curve = pd.Series(dtype=float)

        summary = self._summarize_walk_forward_result(
            folds=fold_reports,
            portfolio_returns=portfolio_returns,
            equity_curve=equity_curve,
            starting_capital=starting_capital,
        )
        return AdaptiveWalkForwardResult(
            folds=fold_reports,
            summary=summary,
            portfolio_returns=portfolio_returns,
            equity_curve=equity_curve,
            final_engine=runner.base_engine,
        )

    def _build_cycle_report(
        self,
        bundle: UnifiedDataBundle,
        base_result: PortfolioBacktestResult,
    ) -> AdaptiveCycleReport:
        adjusted_result, execution_summary = self._apply_execution_realism(bundle, base_result)
        reality_gap = self._evaluate_reality_gap(base_result, adjusted_result, execution_summary)
        edge_retention_scores = self._strategy_edge_retention_map(reality_gap)
        performance_metrics = self.performance_tracker.update_from_backtest(adjusted_result)
        persistence_metrics = self.persistence_scorer.update(performance_metrics)
        signal_strengths = self._compute_strategy_signal_strengths(bundle.datasets)
        performance_metrics = self._enrich_performance_metrics(
            performance_metrics,
            persistence_metrics,
            signal_strengths,
            edge_retention_scores,
        )
        lifecycle_decisions = (
            self.lifecycle_manager.update(performance_metrics)
            if self.lifecycle_manager is not None
            else {}
        )
        regime_states = self.regime_engine.detect_universe(bundle.datasets, self.asset_specs)
        strategy_returns = self._build_strategy_return_frame(adjusted_result)
        asset_returns = self._build_asset_return_frame(bundle.datasets)
        regime_multipliers, regime_confidences = self._compute_regime_overlays(regime_states)
        strategy_market_map = self._strategy_market_map()
        allocation_decision = self.allocator.allocate(
            strategy_returns=strategy_returns,
            performance_metrics=performance_metrics,
            regime_multipliers=regime_multipliers,
            portfolio_drawdown=float(getattr(adjusted_result, "max_drawdown", 0.0) or 0.0),
            signal_strengths=signal_strengths,
            persistence_scores={
                name: snapshot.persistence_score for name, snapshot in performance_metrics.items()
            },
            regime_confidences=regime_confidences,
            edge_retention_scores=edge_retention_scores,
            strategy_market_map=strategy_market_map,
            base_position_sizes=self._base_position_sizes(),
            previous_weights=self.state.previous_weights,
            base_risk_budget=float(self.base_engine.max_total_exposure),
            target_volatility=self._default_target_volatility(),
        )
        risk_snapshot = self._apply_portfolio_risk_controls(
            allocation_decision,
            portfolio_returns=self._portfolio_return_series(adjusted_result.equity_curve),
            strategy_returns=strategy_returns,
            asset_returns=asset_returns,
            portfolio_drawdown=float(getattr(adjusted_result, "max_drawdown", 0.0) or 0.0),
        )
        self.allocator.apply_portfolio_objective(
            allocation_decision,
            cagr=(risk_snapshot.realized_cagr if risk_snapshot is not None else self._cagr_from_equity_curve(adjusted_result.equity_curve, float(self.base_engine.capital))),
            max_drawdown=float(getattr(adjusted_result, "max_drawdown", 0.0) or 0.0),
            volatility_tracking_error=(risk_snapshot.volatility_tracking_error if risk_snapshot is not None else abs(float(allocation_decision.realized_volatility) - float(allocation_decision.target_volatility))),
        )
        learning_result = self.learner.update(
            allocation_decision,
            performance_metrics,
            strategy_configs=self._strategy_config_map(),
        )
        adapted_engine, adapted_primitive_specs = self._build_next_cycle_state(
            allocation_decision,
            learning_result,
            performance_metrics=performance_metrics,
            lifecycle_decisions=lifecycle_decisions,
        )
        adapted_engine, adapted_primitive_specs, liquidity_snapshot = self._apply_liquidity_constraints(
            bundle,
            adapted_engine,
            adapted_primitive_specs,
        )
        suggested_sizes = {slot.name: slot.position_size_pct for slot in adapted_engine.slots}
        self.state = AdaptiveEngineState(
            previous_weights=dict(allocation_decision.weights),
            learned_parameters=deepcopy(learning_result.parameter_suggestions),
            performance_metrics=deepcopy(performance_metrics),
            persistence_metrics=deepcopy(persistence_metrics),
            lifecycle_metrics=deepcopy(lifecycle_decisions),
        )
        return AdaptiveCycleReport(
            bundle=bundle,
            base_result=adjusted_result,
            strategy_returns=strategy_returns,
            performance_metrics=performance_metrics,
            persistence_metrics=persistence_metrics,
            regime_states=regime_states,
            allocation_decision=allocation_decision,
            learning_result=learning_result,
            adapted_engine=adapted_engine,
            execution_summary=execution_summary,
            risk_snapshot=risk_snapshot,
            liquidity_snapshot=liquidity_snapshot,
            lifecycle_decisions=lifecycle_decisions,
            reality_gap=reality_gap,
            suggested_position_sizes=suggested_sizes,
            adapted_primitive_specs=adapted_primitive_specs,
        )

    def build_next_cycle_engine(
        self,
        allocation_decision: AllocationDecision,
        learning_result: LearningResult,
        performance_metrics: dict[str, StrategyPerformanceSnapshot] | None = None,
    ) -> PortfolioEngine:
        adapted_engine, _ = self._build_next_cycle_state(
            allocation_decision,
            learning_result,
            performance_metrics=performance_metrics,
            lifecycle_decisions=self.state.lifecycle_metrics,
        )
        return adapted_engine

    def _build_next_cycle_state(
        self,
        allocation_decision: AllocationDecision,
        learning_result: LearningResult,
        *,
        performance_metrics: dict[str, StrategyPerformanceSnapshot] | None = None,
        lifecycle_decisions: dict[str, LifecycleDecision] | None = None,
    ) -> tuple[PortfolioEngine, list[PrimitiveSlotSpec]]:
        performance_metrics = performance_metrics or {}
        lifecycle_decisions = lifecycle_decisions or {}
        max_total_exposure, max_position_notional_pct = self._risk_budget_overrides(allocation_decision)

        if self.primitive_specs:
            updated_specs = []
            n = max(1, len(allocation_decision.weights))
            strategy_configs = self._strategy_config_map()
            for spec in self._base_primitive_specs():
                base_name = self._base_strategy_name(spec.name)
                lifecycle_decision = lifecycle_decisions.get(base_name)
                position_mult = allocation_decision.position_size_multipliers.get(base_name, 1.0)
                exploit_weight = allocation_decision.exploit_weights.get(
                    base_name,
                    allocation_decision.weights.get(base_name, 0.0),
                )
                if lifecycle_decision and lifecycle_decision.retired:
                    replacement_weight = max(
                        exploit_weight,
                        allocation_decision.exploration_weights.get(base_name, 0.0),
                        0.05 / max(1, n),
                    )
                    replacement_config = self._replacement_config(
                        base_name,
                        dict(spec.config),
                        strategy_configs,
                        lifecycle_decision,
                    )
                    replacement_config.update(
                        learning_result.parameter_suggestions.get(base_name, {})
                    )
                    updated_specs.append(
                        replace(
                            spec,
                            config=replacement_config,
                            position_size_pct=self._scaled_position_size(
                                base_size=spec.position_size_pct,
                                weight=replacement_weight,
                                strategy_count=n,
                                gross_exposure_scale=allocation_decision.gross_exposure_scale,
                                position_multiplier=min(position_mult * 0.85, 1.25),
                            ),
                        )
                    )
                    continue
                updated_config = self._promote_exploration_config(
                    base_name,
                    dict(spec.config),
                    strategy_configs,
                    performance_metrics,
                )
                updated_config.update(learning_result.parameter_suggestions.get(base_name, {}))
                updated_specs.append(
                    replace(
                        spec,
                        config=updated_config,
                        position_size_pct=self._scaled_position_size(
                            base_size=spec.position_size_pct,
                            weight=exploit_weight,
                            strategy_count=n,
                            gross_exposure_scale=allocation_decision.gross_exposure_scale,
                            position_multiplier=position_mult,
                        ),
                    )
                )

                exploration_weight = allocation_decision.exploration_weights.get(base_name, 0.0)
                if exploration_weight > 0.0:
                    variants = self._evolution_variants(
                        base_name=base_name,
                        base_config=dict(spec.config),
                        strategy_configs=strategy_configs,
                        performance_metrics=performance_metrics,
                    )
                    if not variants:
                        variants = [
                            type("_FallbackVariant", (), {
                                "name": f"{base_name}__explore",
                                "config": self._build_exploratory_config(dict(spec.config), updated_config),
                                "method": "exploration",
                            })()
                        ]
                    for variant, share in zip(variants, self._variant_weight_shares(variants)):
                        updated_specs.append(
                            replace(
                                spec,
                                name=variant.name,
                                config=dict(variant.config),
                                position_size_pct=self._scaled_position_size(
                                    base_size=spec.position_size_pct,
                                    weight=exploration_weight * share,
                                    strategy_count=n,
                                    gross_exposure_scale=allocation_decision.gross_exposure_scale,
                                    position_multiplier=min(position_mult * 0.95, 1.50),
                                ),
                            )
                        )
            adapted_engine = self.from_primitive_specs(
                updated_specs,
                self.asset_specs,
                capital=self.base_engine.capital,
                data_days=self.base_engine.data_days,
                max_total_exposure=max_total_exposure,
                max_position_notional_pct=max_position_notional_pct,
                max_drawdown_kill=self.base_engine.max_drawdown_kill,
                use_risk_manager=self.base_engine.risk_manager is not None,
                use_divergence_tracker=self.base_engine.divergence_tracker is not None,
                use_market_state_brain=self.base_engine.market_brain is not None,
                use_execution_edge=self.base_engine.exec_edge is not None,
                use_capital_scaling=self.base_engine.scaler is not None,
            ).base_engine
            return adapted_engine, updated_specs

        new_slots = []
        n = max(1, len(allocation_decision.weights))
        strategy_configs = self._strategy_config_map()
        for slot in self._base_slots():
            base_name = self._base_strategy_name(slot.name)
            lifecycle_decision = lifecycle_decisions.get(base_name)
            position_mult = allocation_decision.position_size_multipliers.get(base_name, 1.0)
            exploit_weight = allocation_decision.exploit_weights.get(
                base_name,
                allocation_decision.weights.get(base_name, 0.0),
            )
            base_params = self._default_slot_params(slot)
            if lifecycle_decision and lifecycle_decision.retired:
                replacement_weight = max(
                    exploit_weight,
                    allocation_decision.exploration_weights.get(base_name, 0.0),
                    0.05 / max(1, n),
                )
                replacement_params = self._replacement_config(
                    base_name,
                    base_params,
                    strategy_configs,
                    lifecycle_decision,
                )
                replacement_params.update(learning_result.parameter_suggestions.get(base_name, {}))
                new_slots.append(
                    self._rebind_slot(
                        slot,
                        params=replacement_params,
                        position_size_pct=self._scaled_position_size(
                            base_size=slot.position_size_pct,
                            weight=replacement_weight,
                            strategy_count=n,
                            gross_exposure_scale=allocation_decision.gross_exposure_scale,
                            position_multiplier=min(position_mult * 0.85, 1.25),
                        ),
                    )
                )
                continue
            promoted_params = self._promote_exploration_config(
                base_name,
                base_params,
                strategy_configs,
                performance_metrics,
            )
            promoted_params.update(learning_result.parameter_suggestions.get(base_name, {}))
            new_slots.append(
                self._rebind_slot(
                    slot,
                    params=promoted_params,
                    position_size_pct=self._scaled_position_size(
                        base_size=slot.position_size_pct,
                        weight=exploit_weight,
                        strategy_count=n,
                        gross_exposure_scale=allocation_decision.gross_exposure_scale,
                        position_multiplier=position_mult,
                    ),
                )
            )

            exploration_weight = allocation_decision.exploration_weights.get(base_name, 0.0)
            if exploration_weight > 0.0:
                variants = self._evolution_variants(
                    base_name=base_name,
                    base_config=base_params,
                    strategy_configs=strategy_configs,
                    performance_metrics=performance_metrics,
                )
                if not variants:
                    variants = [
                        type("_FallbackVariant", (), {
                            "name": f"{base_name}__explore",
                            "config": self._build_exploratory_config(base_params, promoted_params),
                            "method": "exploration",
                        })()
                    ]
                for variant, share in zip(variants, self._variant_weight_shares(variants)):
                    new_slots.append(
                        self._rebind_slot(
                            slot,
                            params=dict(variant.config),
                            name=variant.name,
                            position_size_pct=self._scaled_position_size(
                                base_size=slot.position_size_pct,
                                weight=exploration_weight * share,
                                strategy_count=n,
                                gross_exposure_scale=allocation_decision.gross_exposure_scale,
                                position_multiplier=min(position_mult * 0.95, 1.50),
                            ),
                        )
                    )

        return self._clone_engine(
            self.base_engine,
            slots=new_slots,
            max_total_exposure=max_total_exposure,
            max_position_notional_pct=max_position_notional_pct,
        ), []

    def _build_strategy_return_frame(self, result) -> pd.DataFrame:
        returns = {}
        for name, curve in getattr(result, "strategy_equity_curves", {}).items():
            if len(curve) <= 1:
                continue
            returns[name] = curve.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame(returns)

    def _compute_regime_overlays(
        self,
        regime_states: dict[str, RegimeState],
    ) -> tuple[dict[str, float], dict[str, float]]:
        multipliers = {}
        confidences = {}
        for slot in self.base_engine.slots:
            base_name = self._base_strategy_name(slot.name)
            primitive_name = self.strategy_primitive_map.get(base_name, "mean_reversion")
            slot_multipliers = [
                self.regime_engine.primitive_multiplier(primitive_name, regime_states[symbol])
                for symbol in slot.allowed_assets
                if symbol in regime_states
            ]
            slot_confidences = [
                float(regime_states[symbol].confidence_score)
                for symbol in slot.allowed_assets
                if symbol in regime_states
            ]
            multipliers[slot.name] = float(np.mean(slot_multipliers)) if slot_multipliers else 1.0
            confidences[slot.name] = float(np.mean(slot_confidences)) if slot_confidences else 0.65
        return multipliers, confidences

    def _strategy_config_map(self) -> dict[str, dict]:
        if self.primitive_specs:
            return {
                spec.name: {**dict(spec.config), "position_size_pct": spec.position_size_pct}
                for spec in self.primitive_specs
            }
        return {
            slot.name: {**dict(slot.params), "position_size_pct": slot.position_size_pct}
            for slot in self.base_engine.slots
        }

    @staticmethod
    def _infer_primitive_map(base_engine: PortfolioEngine) -> dict[str, str]:
        mapping = {}
        for slot in base_engine.slots:
            if "momentum" in slot.name:
                primitive = "momentum"
            elif "squeeze" in slot.name or "spike" in slot.name:
                primitive = "volatility"
            elif "contrarian" in slot.name:
                primitive = "liquidity_shock"
            else:
                primitive = "mean_reversion"
            mapping[slot.name] = primitive
        return mapping

    @staticmethod
    def _coerce_asset_specs(
        asset_specs: list[AssetSpec] | dict[str, AssetSpec],
    ) -> dict[str, AssetSpec]:
        if isinstance(asset_specs, dict):
            values = list(asset_specs.values())
        else:
            values = list(asset_specs)
        return {asset.key: asset for asset in values}

    @staticmethod
    def _clone_engine(
        engine: PortfolioEngine,
        *,
        slots: list[StrategySlot] | None = None,
        capital: float | None = None,
        max_total_exposure: float | None = None,
        max_position_notional_pct: float | None = None,
    ) -> PortfolioEngine:
        return PortfolioEngine(
            slots=list(slots or engine.slots),
            assets=list(engine.assets),
            capital=engine.capital if capital is None else capital,
            data_days=engine.data_days,
            max_total_exposure=(
                engine.max_total_exposure if max_total_exposure is None else max_total_exposure
            ),
            max_position_notional_pct=(
                engine.max_position_notional_pct
                if max_position_notional_pct is None
                else max_position_notional_pct
            ),
            max_drawdown_kill=engine.max_drawdown_kill,
            use_regime_allocator=engine.regime_allocator is not None,
            use_risk_manager=engine.risk_manager is not None,
            use_divergence_tracker=engine.divergence_tracker is not None,
            use_market_state_brain=engine.market_brain is not None,
            use_execution_edge=engine.exec_edge is not None,
            use_live_adaptation=False,
            use_capital_scaling=engine.scaler is not None,
        )

    def _spawn_walk_forward_runner(self) -> "AdaptivePortfolioEngine":
        return AdaptivePortfolioEngine(
            base_engine=deepcopy(self.base_engine),
            asset_specs=deepcopy(self.asset_specs),
            strategy_primitive_map=deepcopy(self.strategy_primitive_map),
            primitive_specs=deepcopy(self.primitive_specs),
            data_adapter=deepcopy(self.data_adapter),
            performance_tracker=deepcopy(self.performance_tracker),
            persistence_scorer=deepcopy(self.persistence_scorer),
            regime_engine=deepcopy(self.regime_engine),
            allocator=deepcopy(self.allocator),
            learner=deepcopy(self.learner),
            execution_realism=deepcopy(self.execution_realism),
            risk_controller=deepcopy(self.risk_controller),
            liquidity_engine=deepcopy(self.liquidity_engine),
            lifecycle_manager=deepcopy(self.lifecycle_manager),
            reality_gap_validator=deepcopy(self.reality_gap_validator),
            evolution_engine=deepcopy(self.evolution_engine),
            state=deepcopy(self.state),
        )

    def _apply_execution_realism(
        self,
        bundle: UnifiedDataBundle,
        result: PortfolioBacktestResult,
    ) -> tuple[PortfolioBacktestResult, ExecutionAdjustmentSummary | None]:
        if self.execution_realism is None:
            return result, None
        return self.execution_realism.adjust_backtest_result(bundle, result, bundle.assets)

    def _compute_strategy_signal_strengths(
        self,
        datasets: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        strengths: dict[str, float] = {}
        for slot in self.base_engine.slots:
            per_asset = []
            for symbol in slot.allowed_assets:
                frame = datasets.get(symbol)
                if frame is None or frame.empty:
                    continue
                signals = slot.get_signals(frame)
                if len(signals) == 0:
                    continue
                signal_abs = signals.abs()
                latest = float(signal_abs.iloc[-1])
                recent = float(signal_abs.tail(24).mean()) if len(signal_abs) else 0.0
                density = float((signal_abs.tail(72) > 0).mean()) if len(signal_abs) else 0.0
                per_asset.append(0.55 * latest + 0.30 * recent + 0.15 * density)
            strengths[slot.name] = float(np.clip(np.mean(per_asset) if per_asset else 0.0, 0.0, 1.25))
        return strengths

    @staticmethod
    def _build_asset_return_frame(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                symbol: frame["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
                for symbol, frame in datasets.items()
                if frame is not None and not frame.empty
            }
        )

    @staticmethod
    def _portfolio_return_series(equity_curve: pd.Series) -> pd.Series:
        if equity_curve is None or len(equity_curve) <= 1:
            return pd.Series(dtype=float)
        returns = equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if len(returns) > 0:
            returns.iloc[0] = 0.0
        return returns

    def _apply_portfolio_risk_controls(
        self,
        allocation_decision: AllocationDecision,
        *,
        portfolio_returns: pd.Series,
        strategy_returns: pd.DataFrame,
        asset_returns: pd.DataFrame,
        portfolio_drawdown: float,
    ) -> PortfolioRiskSnapshot | None:
        if self.risk_controller is None:
            return None
        risk_snapshot = self.risk_controller.evaluate(
            portfolio_returns,
            strategy_returns,
            asset_returns,
            portfolio_drawdown=portfolio_drawdown,
        )
        allocation_decision.target_volatility = risk_snapshot.target_volatility
        allocation_decision.realized_volatility = risk_snapshot.realized_volatility
        allocation_decision.volatility_multiplier = risk_snapshot.volatility_multiplier
        allocation_decision.correlation_multiplier = risk_snapshot.correlation_multiplier
        allocation_decision.strategy_correlation = risk_snapshot.strategy_correlation_average
        allocation_decision.asset_correlation = risk_snapshot.asset_correlation_average
        allocation_decision.correlation_shock = risk_snapshot.correlation_shock
        allocation_decision.portfolio_growth_score = risk_snapshot.growth_score
        allocation_decision.gross_exposure_scale = float(
            np.clip(
                allocation_decision.gross_exposure_scale * risk_snapshot.final_multiplier,
                0.30,
                self.allocator.max_gross_exposure_scale,
            )
        )
        return risk_snapshot

    def _apply_liquidity_constraints(
        self,
        bundle: UnifiedDataBundle,
        adapted_engine: PortfolioEngine,
        adapted_primitive_specs: list[PrimitiveSlotSpec],
    ) -> tuple[PortfolioEngine, list[PrimitiveSlotSpec], LiquidityConstraintDecision | None]:
        if self.liquidity_engine is None:
            return adapted_engine, adapted_primitive_specs, None

        liquidity_snapshot = self.liquidity_engine.evaluate(
            bundle,
            adapted_engine,
            capital=float(adapted_engine.capital),
        )
        capped_slots = [
            replace(
                slot,
                position_size_pct=liquidity_snapshot.capped_position_sizes.get(slot.name, slot.position_size_pct),
            )
            for slot in adapted_engine.slots
        ]
        capped_engine = self._clone_engine(adapted_engine, slots=capped_slots)
        capped_specs = [
            replace(
                spec,
                position_size_pct=liquidity_snapshot.capped_position_sizes.get(spec.name, spec.position_size_pct),
            )
            for spec in adapted_primitive_specs
        ] if adapted_primitive_specs else []
        return capped_engine, capped_specs, liquidity_snapshot

    def _evaluate_reality_gap(
        self,
        theoretical_result: PortfolioBacktestResult,
        adjusted_result: PortfolioBacktestResult,
        execution_summary: ExecutionAdjustmentSummary | None,
    ) -> RealityGapSnapshot | None:
        if self.reality_gap_validator is None or execution_summary is None:
            return None
        return self.reality_gap_validator.evaluate(
            theoretical_result,
            adjusted_result,
            execution_summary,
        )

    @staticmethod
    def _enrich_performance_metrics(
        performance_metrics: dict[str, StrategyPerformanceSnapshot],
        persistence_metrics: dict[str, PersistenceSnapshot],
        signal_strengths: dict[str, float],
        edge_retention_scores: dict[str, float],
    ) -> dict[str, StrategyPerformanceSnapshot]:
        enriched: dict[str, StrategyPerformanceSnapshot] = {}
        for strategy, snapshot in performance_metrics.items():
            persistence = persistence_metrics.get(strategy)
            signal_strength = float(signal_strengths.get(strategy, 1.0))
            conviction = max(0.0, snapshot.rolling_sharpe) * snapshot.rolling_win_rate * signal_strength
            edge_retention = float(np.clip(edge_retention_scores.get(strategy, 1.0), 0.0, 1.25))
            enriched[strategy] = replace(
                snapshot,
                signal_strength=signal_strength,
                conviction_score=conviction,
                sharpe_stability=persistence.sharpe_stability if persistence else 0.65,
                return_stability=persistence.return_stability if persistence else 0.65,
                persistence_score=persistence.persistence_score if persistence else 0.65,
                edge_retention=edge_retention,
                execution_efficiency=edge_retention,
            )
        return enriched

    def _base_position_sizes(self) -> dict[str, float]:
        if self.primitive_specs:
            return {
                spec.name: spec.position_size_pct
                for spec in self._base_primitive_specs()
            }
        return {
            slot.name: slot.position_size_pct
            for slot in self._base_slots()
        }

    def _risk_budget_overrides(self, allocation_decision: AllocationDecision) -> tuple[float, float]:
        risk_multiplier = max(1.0, float(allocation_decision.risk_budget_multiplier))
        max_total_exposure = float(
            np.clip(
                self.base_engine.max_total_exposure * risk_multiplier,
                self.base_engine.max_total_exposure,
                self.base_engine.max_total_exposure * 2.0,
            )
        )
        max_position_notional_pct = float(
            np.clip(
                self.base_engine.max_position_notional_pct * (1.0 + 0.35 * (risk_multiplier - 1.0)),
                self.base_engine.max_position_notional_pct,
                self.base_engine.max_position_notional_pct * 1.40,
            )
        )
        return max_total_exposure, max_position_notional_pct

    @staticmethod
    def _base_strategy_name(name: str) -> str:
        return str(name).split("__", 1)[0]

    def _base_primitive_specs(self) -> list[PrimitiveSlotSpec]:
        return [spec for spec in self.primitive_specs if self._base_strategy_name(spec.name) == spec.name]

    def _base_slots(self) -> list[StrategySlot]:
        return [slot for slot in self.base_engine.slots if self._base_strategy_name(slot.name) == slot.name]

    @staticmethod
    def _scaled_position_size(
        *,
        base_size: float,
        weight: float,
        strategy_count: int,
        gross_exposure_scale: float,
        position_multiplier: float,
    ) -> float:
        scale = max(0.0, float(weight)) * max(1, int(strategy_count)) * float(gross_exposure_scale)
        return float(max(0.0, base_size) * scale * max(0.25, float(position_multiplier)))

    def _promote_exploration_config(
        self,
        strategy_name: str,
        base_config: dict,
        strategy_configs: dict[str, dict],
        performance_metrics: dict[str, StrategyPerformanceSnapshot],
    ) -> dict:
        base_snapshot = performance_metrics.get(strategy_name)
        base_score = base_snapshot.score if base_snapshot else 0.0
        candidates: list[tuple[float, str]] = []
        for name, snapshot in performance_metrics.items():
            if self._base_strategy_name(name) != strategy_name or name == strategy_name:
                continue
            if snapshot.score <= base_score + 0.03:
                continue
            candidates.append((snapshot.score + 0.5 * snapshot.growth_score + 0.25 * snapshot.edge_retention, name))
        if not candidates:
            return dict(base_config)
        candidates.sort(reverse=True)
        promoted = dict(strategy_configs.get(candidates[0][1], base_config))
        promoted.pop("position_size_pct", None)
        return promoted

    def _replacement_config(
        self,
        strategy_name: str,
        base_config: dict,
        strategy_configs: dict[str, dict],
        lifecycle_decision: LifecycleDecision | None,
    ) -> dict:
        if lifecycle_decision and lifecycle_decision.replacement_candidate:
            promoted = dict(strategy_configs.get(lifecycle_decision.replacement_candidate, base_config))
            promoted.pop("position_size_pct", None)
            return promoted
        return self._build_exploratory_config(
            dict(base_config),
            dict(strategy_configs.get(strategy_name, base_config)),
        )

    def _evolution_variants(
        self,
        *,
        base_name: str,
        base_config: dict,
        strategy_configs: dict[str, dict],
        performance_metrics: dict[str, StrategyPerformanceSnapshot],
    ):
        if self.evolution_engine is None:
            return []
        return self.evolution_engine.build_variants(
            base_name=base_name,
            base_config=base_config,
            strategy_configs=strategy_configs,
            performance_metrics=performance_metrics,
            strategy_group_map=self._strategy_group_map(),
        )

    @staticmethod
    def _variant_weight_shares(variants) -> list[float]:
        if not variants:
            return []
        if len(variants) == 1:
            return [1.0]
        raw = np.array(
            [1.0 if getattr(variant, "method", "") == "mutation" else 0.9 for variant in variants],
            dtype=float,
        )
        raw = raw / raw.sum()
        return [float(value) for value in raw]

    def _strategy_market_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for slot in self.base_engine.slots:
            markets = [
                self.asset_specs[symbol].market_type.value
                for symbol in slot.allowed_assets
                if symbol in self.asset_specs
            ]
            market = max(set(markets), key=markets.count) if markets else "unknown"
            mapping[slot.name] = market
            mapping.setdefault(self._base_strategy_name(slot.name), market)
        return mapping

    def _strategy_group_map(self) -> dict[str, str]:
        if self.primitive_specs:
            return {
                self._base_strategy_name(spec.name): spec.primitive_name
                for spec in self.primitive_specs
            }
        return {
            self._base_strategy_name(slot.name): (slot.template or self.strategy_primitive_map.get(self._base_strategy_name(slot.name), slot.name))
            for slot in self.base_engine.slots
        }

    @staticmethod
    def _strategy_edge_retention_map(reality_gap: RealityGapSnapshot | None) -> dict[str, float]:
        if reality_gap is None:
            return {}
        return {
            name: float(snapshot.edge_retention_ratio)
            for name, snapshot in reality_gap.strategy_edge_retention.items()
        }

    def _default_target_volatility(self) -> float:
        if self.risk_controller is not None:
            return float(self.risk_controller.target_volatility)
        return 0.15

    @staticmethod
    def _build_exploratory_config(base_config: dict, exploit_config: dict) -> dict:
        exploratory = dict(exploit_config)
        for key, base_value in base_config.items():
            current_value = exploratory.get(key, base_value)
            if not isinstance(current_value, (int, float)):
                continue
            if not isinstance(base_value, (int, float)):
                continue
            delta = current_value - base_value
            if abs(delta) < 1e-9:
                delta = 0.10 * base_value if base_value != 0 else 0.10
            exploratory_value = current_value + 0.50 * delta
            if isinstance(current_value, int):
                exploratory[key] = max(1, int(round(exploratory_value)))
            else:
                exploratory[key] = float(exploratory_value)
        return exploratory

    @staticmethod
    def _default_slot_params(slot: StrategySlot) -> dict:
        params = dict(slot.params)
        defaults = {
            "funding_reversion": {
                "funding_entry_zscore": 3.0,
                "funding_lookback": 168,
                "hold_bars": 24,
                "require_price_confirmation": False,
            },
            "extreme_funding_spike": {
                "funding_z_threshold": 4.0,
                "funding_lookback": 96,
                "funding_velocity_mult": 2.0,
                "hold_bars": 8,
            },
            "funding_vol_squeeze": {
                "bb_width_percentile": 15,
                "bb_period": 20,
                "funding_z_threshold": 1.5,
                "funding_lookback": 168,
                "hold_bars": 24,
            },
            "momentum_breakout": {
                "channel_period": 30,
                "atr_expansion": 1.5,
                "volume_mult": 1.3,
                "hold_bars": 24,
            },
            "contrarian_asymmetry": {
                "funding_z_threshold": 1.5,
                "funding_lookback": 168,
                "hold_bars": 8,
                "require_volume_confirm": False,
                "volume_z_threshold": 1.0,
                "trend_filter": False,
                "trend_sma_period": 50,
            },
        }
        return {**defaults.get(slot.template, {}), **params}

    def _rebind_slot(
        self,
        slot: StrategySlot,
        *,
        params: dict,
        name: str | None = None,
        position_size_pct: float | None = None,
    ) -> StrategySlot:
        signal_func = self._build_signal_func(slot.template, params, slot.signal_func)
        return replace(
            slot,
            name=slot.name if name is None else name,
            params=dict(params),
            signal_func=signal_func,
            position_size_pct=slot.position_size_pct if position_size_pct is None else position_size_pct,
        )

    @staticmethod
    def _build_signal_func(template: str, params: dict, fallback):
        if template == "funding_reversion":
            from src.engine.strategy_factory import FundingReversionTemplate

            return lambda df, p=dict(params): FundingReversionTemplate.generate_signals(df, **p)
        if template == "extreme_funding_spike":
            from src.engine.micro_strategies import ExtremeFundingSpikeTemplate

            return lambda df, p=dict(params): ExtremeFundingSpikeTemplate.generate_signals(df, **p)
        if template == "funding_vol_squeeze":
            from src.engine.micro_strategies import FundingVolSqueezeTemplate

            return lambda df, p=dict(params): FundingVolSqueezeTemplate.generate_signals(df, **p)
        if template == "momentum_breakout":
            from src.engine.momentum_breakout import MomentumBreakoutTemplate

            return lambda df, p=dict(params): MomentumBreakoutTemplate.generate_signals(df, **p)
        if template == "contrarian_asymmetry":
            from src.engine.structural_stress import ContrarianAsymmetryEngine

            return lambda df, p=dict(params): ContrarianAsymmetryEngine.generate_signals(df, **p)
        return fallback

    @staticmethod
    def _shared_index(datasets: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        shared_index: pd.DatetimeIndex | None = None
        for frame in datasets.values():
            shared_index = frame.index if shared_index is None else shared_index.intersection(frame.index)
        if shared_index is None or len(shared_index) == 0:
            raise ValueError("Adaptive walk-forward requires a shared non-empty time index")
        return shared_index.sort_values()

    @staticmethod
    def _slice_datasets(
        datasets: dict[str, pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        sliced = {}
        for symbol, frame in datasets.items():
            window = frame.loc[start:end].copy()
            if not window.empty:
                sliced[symbol] = window
        return sliced

    @classmethod
    def _slice_datasets_with_warmup(
        cls,
        datasets: dict[str, pd.DataFrame],
        shared_index: pd.DatetimeIndex,
        start: pd.Timestamp,
        end: pd.Timestamp,
        warmup_bars: int,
    ) -> dict[str, pd.DataFrame]:
        start_idx = int(shared_index.searchsorted(start))
        warmup_start = shared_index[max(0, start_idx - warmup_bars)]
        return cls._slice_datasets(datasets, warmup_start, end)

    def _activation_gated_engine(
        self,
        engine: PortfolioEngine,
        *,
        active_from: pd.Timestamp,
        capital: float,
    ) -> PortfolioEngine:
        gated_slots = []
        for slot in engine.slots:
            original_signal_func = slot.signal_func

            def gated_signal_func(
                df: pd.DataFrame,
                original_signal_func=original_signal_func,
                active_from=active_from,
            ) -> pd.Series:
                signals = original_signal_func(df)
                if len(signals) == 0:
                    return signals
                activation_idx = int(df.index.searchsorted(active_from))
                allow_from_idx = max(0, activation_idx - 1)
                gated = signals.copy()
                gated.iloc[:allow_from_idx] = 0
                return gated

            gated_slots.append(replace(slot, signal_func=gated_signal_func))

        return self._clone_engine(engine, slots=gated_slots, capital=capital)

    def _slice_backtest_result(
        self,
        result: PortfolioBacktestResult,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        initial_capital: float,
    ) -> tuple[PortfolioBacktestResult, pd.Series]:
        sliced = PortfolioBacktestResult(weights=dict(getattr(result, "weights", {})))
        equity_curve, returns, base_equity = self._slice_curve_with_returns(
            getattr(result, "equity_curve", pd.Series(dtype=float)),
            start,
            end,
            initial_capital=initial_capital,
        )
        sliced.equity_curve = equity_curve
        sliced.sharpe = self._annualized_sharpe(returns, equity_curve.index)
        sliced.max_drawdown = self._max_drawdown(equity_curve, base_equity)
        if len(equity_curve) > 0:
            sliced.total_pnl = float(equity_curve.iloc[-1] - base_equity)

        filtered_trades = []
        for trade in getattr(result, "trades", []):
            exit_time = getattr(trade, "exit_time", None)
            if exit_time is None:
                continue
            if start <= exit_time <= end:
                filtered_trades.append(trade)

        sliced.trades = filtered_trades
        sliced.total_trades = len(filtered_trades)
        wins = [trade for trade in filtered_trades if float(getattr(trade, "pnl", 0.0)) > 0.0]
        losses = [trade for trade in filtered_trades if float(getattr(trade, "pnl", 0.0)) <= 0.0]
        gross_win = sum(float(getattr(trade, "pnl", 0.0)) for trade in wins)
        gross_loss = sum(abs(float(getattr(trade, "pnl", 0.0))) for trade in losses)
        sliced.profit_factor = float(gross_win / gross_loss) if gross_loss > 0.0 else (float("inf") if gross_win > 0.0 else 0.0)
        sliced.win_rate = len(wins) / len(filtered_trades) if filtered_trades else 0.0
        sliced.strategy_results = self._aggregate_trade_stats(filtered_trades, "strategy")
        sliced.cell_results = self._aggregate_trade_stats(filtered_trades, "cell_key")

        for name, curve in getattr(result, "strategy_equity_curves", {}).items():
            segment, _, _ = self._slice_curve_with_returns(curve, start, end)
            if not segment.empty:
                sliced.strategy_equity_curves[name] = segment

        for name, curve in getattr(result, "cell_equity_curves", {}).items():
            segment, _, _ = self._slice_curve_with_returns(curve, start, end)
            if not segment.empty:
                sliced.cell_equity_curves[name] = segment

        return sliced, returns

    @staticmethod
    def _slice_curve_with_returns(
        curve: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        initial_capital: float | None = None,
    ) -> tuple[pd.Series, pd.Series, float]:
        if curve is None or len(curve) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float), float(initial_capital or 0.0)

        upto_end = curve.loc[:end]
        test_curve = upto_end.loc[start:end]
        if test_curve.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float), float(initial_capital or 0.0)

        prev_curve = upto_end.loc[upto_end.index < start].tail(1)
        base_equity = float(prev_curve.iloc[-1]) if not prev_curve.empty else float(initial_capital or test_curve.iloc[0])
        segment = pd.concat([prev_curve, test_curve]) if not prev_curve.empty else test_curve.copy()

        returns = test_curve.pct_change()
        returns.iloc[0] = float(test_curve.iloc[0] / base_equity - 1.0) if base_equity > 0.0 else 0.0
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return segment, returns, base_equity

    @staticmethod
    def _aggregate_trade_stats(trades: list, attr_name: str) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[float]] = {}
        for trade in trades:
            key = getattr(trade, attr_name, None)
            if key is None:
                continue
            grouped.setdefault(key, []).append(float(getattr(trade, "pnl", 0.0) or 0.0))

        stats = {}
        for key, pnls in grouped.items():
            gross_win = sum(pnl for pnl in pnls if pnl > 0.0)
            gross_loss = sum(abs(pnl) for pnl in pnls if pnl <= 0.0)
            stats[key] = {
                "trades": len(pnls),
                "pf": float(gross_win / gross_loss) if gross_loss > 0.0 else (float("inf") if gross_win > 0.0 else 0.0),
                "win_rate": sum(pnl > 0.0 for pnl in pnls) / len(pnls) if pnls else 0.0,
                "net_pnl": float(sum(pnls)),
            }
        return stats

    @staticmethod
    def _annualized_sharpe(returns: pd.Series, index: pd.Index) -> float:
        clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) <= 1 or float(clean.std(ddof=0)) <= 0.0:
            return 0.0

        bars_per_year = 252.0 * 24.0
        if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
            deltas = pd.Series(index).diff().dropna()
            median_delta = deltas.median()
            if hasattr(median_delta, "total_seconds"):
                bars_per_year = 365.25 * 24.0 * 3600.0 / max(median_delta.total_seconds(), 1.0)
        return float(clean.mean() / clean.std(ddof=0) * np.sqrt(bars_per_year))

    @staticmethod
    def _max_drawdown(curve: pd.Series, base_equity: float) -> float:
        if curve is None or len(curve) == 0 or base_equity <= 0.0:
            return 0.0
        if len(curve) == 1:
            drawdown = (float(curve.iloc[0]) / base_equity) - 1.0
            return abs(min(drawdown, 0.0))
        anchor = pd.Series([base_equity], index=[curve.index[0] - pd.Timedelta(microseconds=1)])
        segment = pd.concat([anchor, curve])
        peak = segment.cummax()
        drawdown = segment / peak - 1.0
        return float(abs(drawdown.min()))

    @staticmethod
    def _summarize_backtest_result(
        result: PortfolioBacktestResult,
        capital: float,
    ) -> dict[str, float]:
        total_return = float(result.total_pnl / capital) if capital > 0.0 else 0.0
        end_capital = capital + float(result.total_pnl)
        return {
            "start_capital": float(capital),
            "end_capital": float(end_capital),
            "return": total_return,
            "net_pnl": float(result.total_pnl),
            "sharpe": float(result.sharpe),
            "profit_factor": float(result.profit_factor),
            "max_drawdown": float(result.max_drawdown),
            "total_trades": float(result.total_trades),
            "win_rate": float(result.win_rate),
        }

    def _summarize_walk_forward_result(
        self,
        *,
        folds: list[AdaptiveWalkForwardFold],
        portfolio_returns: pd.Series,
        equity_curve: pd.Series,
        starting_capital: float,
    ) -> dict[str, float]:
        fold_returns = np.array([fold.test_summary["return"] for fold in folds], dtype=float)
        fold_sharpes = np.array([fold.test_summary["sharpe"] for fold in folds], dtype=float)
        fold_utilization = np.array([fold.projected_utilization for fold in folds], dtype=float)
        execution_costs = np.array([fold.execution_cost for fold in folds], dtype=float)
        fold_realized_vol = np.array([fold.realized_volatility for fold in folds], dtype=float)
        fold_target_vol = np.array([fold.target_volatility for fold in folds], dtype=float)
        fold_tracking_error = np.array([fold.volatility_tracking_error for fold in folds], dtype=float)
        fold_reality_gap = np.array([fold.reality_gap_fraction for fold in folds], dtype=float)
        fold_edge_retention = np.array([fold.edge_retention_ratio for fold in folds], dtype=float)
        fold_objective_scores = np.array([fold.portfolio_objective_score for fold in folds], dtype=float)
        fold_shock_flags = np.array([float(fold.correlation_shock) for fold in folds], dtype=float)

        final_capital = float(equity_curve.iloc[-1]) if len(equity_curve) > 0 else float(starting_capital)
        total_return = float(final_capital / starting_capital - 1.0) if starting_capital > 0.0 else 0.0
        realized_volatility = self._annualized_volatility(portfolio_returns, equity_curve.index)

        return {
            "n_folds": float(len(folds)),
            "start_capital": float(starting_capital),
            "final_capital": final_capital,
            "total_return": total_return,
            "cagr": self._cagr_from_equity_curve(equity_curve, starting_capital),
            "sharpe": self._annualized_sharpe(portfolio_returns, equity_curve.index),
            "realized_volatility": realized_volatility,
            "max_drawdown": self._max_drawdown(equity_curve, starting_capital),
            "mean_fold_return": float(fold_returns.mean()) if len(fold_returns) else 0.0,
            "mean_fold_sharpe": float(fold_sharpes.mean()) if len(fold_sharpes) else 0.0,
            "mean_utilization": float(fold_utilization.mean()) if len(fold_utilization) else 0.0,
            "target_volatility": float(fold_target_vol.mean()) if len(fold_target_vol) else 0.0,
            "volatility_tracking_error": float(fold_tracking_error.mean()) if len(fold_tracking_error) else (float(np.mean(np.abs(fold_realized_vol - fold_target_vol))) if len(fold_realized_vol) else 0.0),
            "correlation_shock_fraction": float(fold_shock_flags.mean()) if len(fold_shock_flags) else 0.0,
            "mean_reality_gap_fraction": float(fold_reality_gap.mean()) if len(fold_reality_gap) else 0.0,
            "mean_edge_retention": float(fold_edge_retention.mean()) if len(fold_edge_retention) else 0.0,
            "mean_portfolio_objective_score": float(fold_objective_scores.mean()) if len(fold_objective_scores) else 0.0,
            "total_execution_cost": float(execution_costs.sum()) if len(execution_costs) else 0.0,
            "positive_fold_fraction": float((fold_returns > 0.0).mean()) if len(fold_returns) else 0.0,
            "best_fold_return": float(fold_returns.max()) if len(fold_returns) else 0.0,
            "worst_fold_return": float(fold_returns.min()) if len(fold_returns) else 0.0,
        }

    @staticmethod
    def _annualized_volatility(returns: pd.Series, index: pd.Index) -> float:
        clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) <= 1 or float(clean.std(ddof=0)) <= 0.0:
            return 0.0
        bars_per_year = 252.0 * 24.0
        if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
            deltas = pd.Series(index).diff().dropna()
            median_delta = deltas.median()
            if hasattr(median_delta, "total_seconds"):
                bars_per_year = 365.25 * 24.0 * 3600.0 / max(median_delta.total_seconds(), 1.0)
        return float(clean.std(ddof=0) * np.sqrt(bars_per_year))

    @staticmethod
    def _cagr_from_equity_curve(equity_curve: pd.Series, starting_capital: float) -> float:
        if equity_curve is None or len(equity_curve) == 0 or starting_capital <= 0.0:
            return 0.0
        final_capital = float(equity_curve.iloc[-1])
        if final_capital <= 0.0:
            return -1.0
        if isinstance(equity_curve.index, pd.DatetimeIndex) and len(equity_curve.index) > 1:
            span_days = max((equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 86400.0, 1.0)
            years = span_days / 365.25
        else:
            years = max(len(equity_curve) / float(252 * 24), 1.0 / float(252 * 24))
        return float((final_capital / starting_capital) ** (1.0 / years) - 1.0)
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.allocation.adaptive import AdaptiveAllocationEngine
from src.allocation.capital_efficiency import CapitalUtilizationEngine
from src.data.market_data import AssetSpec, MarketType, UnifiedMarketDataAdapter
from src.engine.adaptive_portfolio_engine import AdaptivePortfolioEngine
from src.engine.alpha_primitives import PrimitiveSlotSpec
from src.execution.reality_gap import RealityGapValidator
from src.learning.evolution import StrategyEvolutionEngine
from src.meta.objective import PortfolioObjectiveScorer
from src.meta.performance import StrategyPerformanceSnapshot
from src.meta.performance import StrategyPerformanceTracker
from src.meta.lifecycle import StrategyLifecycleManager
from src.meta.persistence import EdgePersistenceScorer
from src.regime.adaptive import RegimeDetectionEngine
from src.risk.adaptive_controls import PortfolioRiskController


def _make_dataset(
    *,
    periods: int = 320,
    seed: int = 1,
    drift: float = 0.0005,
    vol: float = 0.01,
    volume_level: float = 10_000,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = drift + rng.normal(0.0, vol, periods)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + vol * 1.5)
    low = np.minimum(open_, close) * (1.0 - vol * 1.5)
    volume = volume_level * np.clip(1.0 + rng.normal(0.0, 0.10, periods), 0.2, None)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range("2024-01-01", periods=periods, freq="h"),
    )


def test_performance_tracker_downweights_degrading_strategies() -> None:
    returns = pd.DataFrame(
        {
            "good": np.full(120, 0.0015),
            "bad": np.full(120, -0.0010),
        }
    )
    trade_pnls = {
        "good": [20.0, 18.0, 15.0, 17.0, 22.0, 16.0, 19.0],
        "bad": [-10.0, -12.0, -8.0, -15.0, -9.0, -7.0, -11.0],
    }

    tracker = StrategyPerformanceTracker(min_trades=5)
    metrics = tracker.update_from_streams(returns, trade_pnls)

    assert metrics["good"].recommended_multiplier > metrics["bad"].recommended_multiplier
    assert metrics["bad"].state in {"downweighted", "disabled"}


def test_conviction_scaling_favors_stronger_signal() -> None:
    allocator = AdaptiveAllocationEngine(
        blend=1.0,
        exploration_epsilon=0.0,
        weight_inertia=0.0,
        capital_utilization_engine=CapitalUtilizationEngine(
            target_utilization=0.10,
            min_utilization=0.0,
            max_risk_budget_multiplier=1.0,
            idle_pressure=0.0,
            conviction_boost=0.0,
        ),
    )
    strategy_returns = pd.DataFrame({"high": np.zeros(64), "low": np.zeros(64)})
    metrics = {
        "high": StrategyPerformanceSnapshot(
            strategy_name="high",
            rolling_sharpe=1.4,
            rolling_drawdown=0.02,
            rolling_win_rate=0.58,
            rolling_profit_factor=1.6,
            expectancy=8.0,
            total_pnl=60.0,
            trade_count=12,
            score=0.55,
            state="active",
            recommended_multiplier=1.0,
        ),
        "low": StrategyPerformanceSnapshot(
            strategy_name="low",
            rolling_sharpe=1.4,
            rolling_drawdown=0.02,
            rolling_win_rate=0.58,
            rolling_profit_factor=1.6,
            expectancy=8.0,
            total_pnl=60.0,
            trade_count=12,
            score=0.55,
            state="active",
            recommended_multiplier=1.0,
        ),
    }

    decision = allocator.allocate(
        strategy_returns=strategy_returns,
        performance_metrics=metrics,
        signal_strengths={"high": 1.0, "low": 0.15},
        persistence_scores={"high": 0.85, "low": 0.85},
        regime_confidences={"high": 0.8, "low": 0.8},
        base_position_sizes={"high": 0.03, "low": 0.03},
        base_risk_budget=0.10,
    )

    assert decision.conviction_scores["high"] > decision.conviction_scores["low"]
    assert decision.weights["high"] > decision.weights["low"]


def test_capital_utilization_engine_boosts_idle_portfolio() -> None:
    engine = CapitalUtilizationEngine(
        target_utilization=0.80,
        min_utilization=0.60,
        max_risk_budget_multiplier=1.60,
        idle_pressure=0.35,
        conviction_boost=0.40,
    )

    decision = engine.evaluate(
        base_position_sizes={"alpha": 0.02, "beta": 0.02},
        strategy_weights={"alpha": 0.55, "beta": 0.45},
        signal_strengths={"alpha": 0.20, "beta": 0.10},
        conviction_scores={"alpha": 1.4, "beta": 0.4},
        base_risk_budget=0.10,
    )

    assert decision.projected_utilization < decision.target_utilization
    assert decision.risk_budget_multiplier > 1.0
    assert decision.idle_capital_penalty > 0.0
    assert decision.deployment_boosts["alpha"] > decision.deployment_boosts["beta"]


def test_edge_persistence_scoring_rewards_stability() -> None:
    scorer = EdgePersistenceScorer(history_window=4, min_history=2)
    stable_scores = [1.0, 1.1, 0.95, 1.05]
    unstable_scores = [1.8, -0.7, 1.5, -0.5]
    stable_returns = [0.03, 0.028, 0.031, 0.029]
    unstable_returns = [0.09, -0.06, 0.08, -0.05]

    latest = {}
    for idx in range(4):
        latest = scorer.update(
            {
                "stable": StrategyPerformanceSnapshot(
                    strategy_name="stable",
                    rolling_sharpe=stable_scores[idx],
                    rolling_drawdown=0.03,
                    rolling_win_rate=0.57,
                    rolling_profit_factor=1.5,
                    expectancy=9.0,
                    total_pnl=80.0,
                    trade_count=14,
                    score=0.6,
                    state="active",
                    recommended_multiplier=1.0,
                    trailing_return=stable_returns[idx],
                ),
                "unstable": StrategyPerformanceSnapshot(
                    strategy_name="unstable",
                    rolling_sharpe=unstable_scores[idx],
                    rolling_drawdown=0.08,
                    rolling_win_rate=0.49,
                    rolling_profit_factor=1.1,
                    expectancy=4.0,
                    total_pnl=25.0,
                    trade_count=14,
                    score=0.2,
                    state="active",
                    recommended_multiplier=1.0,
                    trailing_return=unstable_returns[idx],
                ),
            }
        )

    assert latest["stable"].persistence_score > latest["unstable"].persistence_score
    assert latest["stable"].persistence_multiplier > latest["unstable"].persistence_multiplier


def test_portfolio_risk_controller_targets_volatility() -> None:
    index = pd.date_range("2024-01-01", periods=240, freq="h")
    high_controller = PortfolioRiskController(target_volatility=0.15)
    low_controller = PortfolioRiskController(target_volatility=0.15)

    high_vol_portfolio = pd.Series(np.resize([0.022, -0.019, 0.017, -0.021], len(index)), index=index)
    low_vol_portfolio = pd.Series(np.resize([0.0008, -0.0006, 0.0007, -0.0005], len(index)), index=index)

    high_decision = high_controller.evaluate(
        high_vol_portfolio,
        pd.DataFrame({"alpha": high_vol_portfolio, "beta": high_vol_portfolio * 0.8}, index=index),
        pd.DataFrame({"BTC": high_vol_portfolio, "ETH": high_vol_portfolio * 0.7}, index=index),
        portfolio_drawdown=0.03,
    )
    low_decision = low_controller.evaluate(
        low_vol_portfolio,
        pd.DataFrame({"alpha": low_vol_portfolio, "beta": low_vol_portfolio * 0.9}, index=index),
        pd.DataFrame({"BTC": low_vol_portfolio, "ETH": low_vol_portfolio * 0.85}, index=index),
        portfolio_drawdown=0.01,
    )

    assert high_decision.realized_volatility > high_decision.target_volatility
    assert high_decision.volatility_multiplier < 1.0
    assert high_decision.pid_output < 0.0
    assert low_decision.realized_volatility < low_decision.target_volatility
    assert low_decision.volatility_multiplier > 1.0
    assert low_decision.pid_output > 0.0


def test_portfolio_risk_controller_uses_feedback_state_to_dampen_overshoot() -> None:
    index = pd.date_range("2024-01-01", periods=240, freq="h")
    controller = PortfolioRiskController(target_volatility=0.15)
    high_vol = pd.Series(np.resize([0.0042, -0.0038, 0.0035, -0.0041], len(index)), index=index)
    calmer_vol = pd.Series(np.resize([0.0014, -0.0012, 0.0011, -0.0013], len(index)), index=index)

    first = controller.evaluate(
        high_vol,
        pd.DataFrame({"alpha": high_vol, "beta": high_vol * 0.9}, index=index),
        pd.DataFrame({"BTC": high_vol, "ETH": high_vol * 0.85}, index=index),
        portfolio_drawdown=0.03,
    )
    second = controller.evaluate(
        calmer_vol,
        pd.DataFrame({"alpha": calmer_vol, "beta": calmer_vol * 0.9}, index=index),
        pd.DataFrame({"BTC": calmer_vol, "ETH": calmer_vol * 0.85}, index=index),
        portfolio_drawdown=0.01,
    )

    assert second.volatility_multiplier > first.volatility_multiplier
    assert second.smoothed_volatility > second.realized_volatility
    assert second.derivative_error > 0.0


def test_portfolio_risk_controller_flags_correlation_shock() -> None:
    index = pd.date_range("2024-01-01", periods=180, freq="h")
    shared = pd.Series(np.resize([0.010, -0.009, 0.011, -0.008], len(index)), index=index)
    strategy_returns = pd.DataFrame(
        {
            "alpha": shared,
            "beta": shared * 0.98,
            "gamma": shared * 1.02,
        },
        index=index,
    )
    asset_returns = pd.DataFrame(
        {
            "BTC": shared,
            "ETH": shared * 0.97,
            "SPY": shared * 0.95,
        },
        index=index,
    )

    decision = PortfolioRiskController().evaluate(
        shared,
        strategy_returns,
        asset_returns,
        portfolio_drawdown=0.02,
    )

    assert decision.correlation_shock is True
    assert decision.correlation_multiplier < 1.0


def test_strategy_lifecycle_manager_retires_degraded_strategy() -> None:
    manager = StrategyLifecycleManager(
        persistence_floor=0.50,
        score_floor=0.05,
        growth_floor=-0.01,
        sustained_degradation_cycles=2,
        promotion_margin=0.05,
        promotion_persistence_floor=0.55,
    )
    metrics = {
        "alpha": StrategyPerformanceSnapshot(
            strategy_name="alpha",
            rolling_sharpe=-0.2,
            rolling_drawdown=0.09,
            rolling_win_rate=0.42,
            rolling_profit_factor=0.9,
            expectancy=-2.0,
            total_pnl=-20.0,
            trade_count=12,
            score=0.0,
            state="downweighted",
            recommended_multiplier=0.45,
            persistence_score=0.30,
            growth_score=-0.08,
        ),
        "alpha__explore": StrategyPerformanceSnapshot(
            strategy_name="alpha__explore",
            rolling_sharpe=0.9,
            rolling_drawdown=0.03,
            rolling_win_rate=0.56,
            rolling_profit_factor=1.4,
            expectancy=6.0,
            total_pnl=35.0,
            trade_count=12,
            score=0.20,
            state="active",
            recommended_multiplier=1.0,
            persistence_score=0.70,
            growth_score=0.12,
        ),
    }

    first = manager.update(metrics)
    second = manager.update(metrics)

    assert first["alpha"].retired is False
    assert second["alpha"].retired is True
    assert second["alpha"].status == "retire_replace"
    assert second["alpha"].replacement_candidate == "alpha__explore"


def test_reality_gap_validator_tracks_edge_retention_by_strategy() -> None:
    validator = RealityGapValidator()
    theoretical = SimpleNamespace(
        total_pnl=100.0,
        strategy_results={
            "alpha": {"net_pnl": 60.0},
            "beta": {"net_pnl": 40.0},
        },
    )
    adjusted = SimpleNamespace(
        total_pnl=65.0,
        strategy_results={
            "alpha": {"net_pnl": 45.0},
            "beta": {"net_pnl": 20.0},
        },
    )
    execution_summary = SimpleNamespace(total_execution_cost=35.0)

    snapshot = validator.evaluate(theoretical, adjusted, execution_summary)

    assert snapshot.edge_retention_ratio == pytest.approx(0.65)
    assert snapshot.edge_retention_state == "fragile"
    assert snapshot.strategy_edge_retention["alpha"].status == "strong"
    assert snapshot.strategy_edge_retention["beta"].status == "fragile"


def test_portfolio_objective_scorer_penalizes_drawdown_and_turnover() -> None:
    scorer = PortfolioObjectiveScorer()

    strong = scorer.portfolio_score(
        cagr=0.22,
        max_drawdown=0.05,
        volatility_tracking_error=0.02,
        turnover=0.15,
    )
    weak = scorer.portfolio_score(
        cagr=0.22,
        max_drawdown=0.10,
        volatility_tracking_error=0.06,
        turnover=0.60,
    )

    assert strong.score > weak.score


def test_adaptive_allocation_routes_capital_to_stronger_market() -> None:
    allocator = AdaptiveAllocationEngine(
        blend=1.0,
        exploration_epsilon=0.0,
        weight_inertia=0.0,
        objective_weight=0.35,
        market_routing_tilt=0.30,
        capital_utilization_engine=CapitalUtilizationEngine(
            target_utilization=0.10,
            min_utilization=0.0,
            max_risk_budget_multiplier=1.0,
            idle_pressure=0.0,
            conviction_boost=0.0,
        ),
    )
    strategy_returns = pd.DataFrame({"crypto_alpha": np.zeros(64), "equity_alpha": np.zeros(64)})
    metrics = {
        "crypto_alpha": StrategyPerformanceSnapshot(
            strategy_name="crypto_alpha",
            rolling_sharpe=1.0,
            rolling_drawdown=0.03,
            rolling_win_rate=0.55,
            rolling_profit_factor=1.4,
            expectancy=7.0,
            total_pnl=50.0,
            trade_count=12,
            score=0.55,
            state="active",
            recommended_multiplier=1.0,
            annualized_return=0.22,
            realized_volatility=0.12,
            growth_score=0.10,
            turnover_rate=0.35,
        ),
        "equity_alpha": StrategyPerformanceSnapshot(
            strategy_name="equity_alpha",
            rolling_sharpe=1.0,
            rolling_drawdown=0.03,
            rolling_win_rate=0.55,
            rolling_profit_factor=1.4,
            expectancy=7.0,
            total_pnl=50.0,
            trade_count=12,
            score=0.55,
            state="active",
            recommended_multiplier=1.0,
            annualized_return=0.06,
            realized_volatility=0.14,
            growth_score=0.02,
            turnover_rate=0.35,
        ),
    }

    decision = allocator.allocate(
        strategy_returns=strategy_returns,
        performance_metrics=metrics,
        signal_strengths={"crypto_alpha": 0.8, "equity_alpha": 0.8},
        persistence_scores={"crypto_alpha": 0.8, "equity_alpha": 0.8},
        regime_confidences={"crypto_alpha": 0.8, "equity_alpha": 0.8},
        edge_retention_scores={"crypto_alpha": 0.9, "equity_alpha": 0.9},
        strategy_market_map={"crypto_alpha": "crypto", "equity_alpha": "equities"},
        base_position_sizes={"crypto_alpha": 0.03, "equity_alpha": 0.03},
        base_risk_budget=0.10,
        target_volatility=0.15,
    )

    assert decision.market_route_multipliers["crypto"] > decision.market_route_multipliers["equities"]
    assert decision.weights["crypto_alpha"] > decision.weights["equity_alpha"]


def test_strategy_evolution_engine_emits_mutation_and_recombination() -> None:
    engine = StrategyEvolutionEngine()
    base_config = {"lookback": 48, "entry_z": 1.5, "exit_z": 0.35}
    strategy_configs = {
        "alpha": dict(base_config),
        "alpha__explore": {"lookback": 56, "entry_z": 1.35, "exit_z": 0.30},
        "beta": {"lookback": 72, "entry_z": 1.10, "exit_z": 0.25},
    }
    performance_metrics = {
        "alpha": StrategyPerformanceSnapshot(
            strategy_name="alpha",
            rolling_sharpe=0.7,
            rolling_drawdown=0.05,
            rolling_win_rate=0.53,
            rolling_profit_factor=1.2,
            expectancy=4.0,
            total_pnl=20.0,
            trade_count=10,
            score=0.15,
            state="active",
            recommended_multiplier=1.0,
            growth_score=0.02,
            edge_retention=0.75,
        ),
        "alpha__explore": StrategyPerformanceSnapshot(
            strategy_name="alpha__explore",
            rolling_sharpe=1.1,
            rolling_drawdown=0.03,
            rolling_win_rate=0.58,
            rolling_profit_factor=1.5,
            expectancy=8.0,
            total_pnl=45.0,
            trade_count=10,
            score=0.28,
            state="active",
            recommended_multiplier=1.0,
            growth_score=0.10,
            edge_retention=0.92,
        ),
        "beta": StrategyPerformanceSnapshot(
            strategy_name="beta",
            rolling_sharpe=1.0,
            rolling_drawdown=0.04,
            rolling_win_rate=0.57,
            rolling_profit_factor=1.4,
            expectancy=7.0,
            total_pnl=40.0,
            trade_count=10,
            score=0.24,
            state="active",
            recommended_multiplier=1.0,
            growth_score=0.09,
            edge_retention=0.88,
        ),
    }

    variants = engine.build_variants(
        base_name="alpha",
        base_config=base_config,
        strategy_configs=strategy_configs,
        performance_metrics=performance_metrics,
        strategy_group_map={"alpha": "mean_reversion", "beta": "mean_reversion"},
    )

    names = {variant.name for variant in variants}
    assert "alpha__mutate" in names
    assert "alpha__recombine" in names
    assert any(variant.config != base_config for variant in variants)


def test_regime_detection_engine_finds_bull_and_liquid_state() -> None:
    asset = AssetSpec(symbol="SPY", market_type=MarketType.EQUITIES)
    adapter = UnifiedMarketDataAdapter()
    df = adapter.normalize_dataset(
        _make_dataset(seed=2, drift=0.002, vol=0.003, volume_level=50_000),
        asset,
    )

    state = RegimeDetectionEngine().detect(df, asset)

    assert state.trend_regime == "bull_trend"
    assert state.liquidity_regime in {"liquid", "normal_liquidity"}
    assert 0.0 <= state.confidence_score <= 1.0


def test_adaptive_portfolio_engine_builds_allocation_from_primitives() -> None:
    assets = [
        AssetSpec(symbol="BTC/USDT", market_type=MarketType.CRYPTO),
        AssetSpec(symbol="SPY", market_type=MarketType.EQUITIES),
        AssetSpec(symbol="CL1", market_type=MarketType.COMMODITIES),
    ]
    specs = [
        PrimitiveSlotSpec(
            name="btc_mr",
            primitive_name="mean_reversion",
            asset_key="BTC/USDT",
            config={"lookback": 32, "entry_z": 1.4, "exit_z": 0.30},
            position_size_pct=0.03,
        ),
        PrimitiveSlotSpec(
            name="spy_momo",
            primitive_name="momentum",
            asset_key="SPY",
            config={"breakout_lookback": 24, "fast": 12, "slow": 48, "trend_threshold": 0.18},
            position_size_pct=0.03,
        ),
        PrimitiveSlotSpec(
            name="cl_vol",
            primitive_name="volatility",
            asset_key="CL1",
            config={"squeeze_window": 18, "expansion_mult": 1.1, "direction_lookback": 6},
            position_size_pct=0.03,
        ),
    ]
    datasets = {
        "BTC/USDT": _make_dataset(seed=10, drift=0.0002, vol=0.012, volume_level=14_000),
        "SPY": _make_dataset(seed=20, drift=0.0015, vol=0.004, volume_level=60_000),
        "CL1": _make_dataset(seed=30, drift=0.0004, vol=0.007, volume_level=20_000),
    }

    adaptive = AdaptivePortfolioEngine.from_primitive_specs(
        specs,
        assets,
        capital=10_000,
        data_days=30,
        use_risk_manager=False,
        use_divergence_tracker=False,
        use_market_state_brain=False,
        use_execution_edge=False,
        use_capital_scaling=False,
    )

    report = adaptive.backtest_adaptive_cycle(datasets)

    weights = report.allocation_decision.weights
    assert set(weights) == {"btc_mr", "spy_momo", "cl_vol"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert report.allocation_decision.gross_exposure_scale > 0.0
    assert set(report.regime_states) == {"BTC/USDT", "SPY", "CL1"}
    assert all(size >= 0.0 for size in report.suggested_position_sizes.values())
    assert report.learning_result.updates


def test_adaptive_walk_forward_runs_multiple_folds_without_mutating_source() -> None:
    assets = [
        AssetSpec(symbol="BTC/USDT", market_type=MarketType.CRYPTO),
        AssetSpec(symbol="SPY", market_type=MarketType.EQUITIES),
        AssetSpec(symbol="CL1", market_type=MarketType.COMMODITIES),
    ]
    specs = [
        PrimitiveSlotSpec(
            name="btc_mr",
            primitive_name="mean_reversion",
            asset_key="BTC/USDT",
            config={"lookback": 32, "entry_z": 1.4, "exit_z": 0.30},
            position_size_pct=0.03,
        ),
        PrimitiveSlotSpec(
            name="spy_momo",
            primitive_name="momentum",
            asset_key="SPY",
            config={"breakout_lookback": 24, "fast": 12, "slow": 48, "trend_threshold": 0.18},
            position_size_pct=0.03,
        ),
        PrimitiveSlotSpec(
            name="cl_vol",
            primitive_name="volatility",
            asset_key="CL1",
            config={"squeeze_window": 18, "expansion_mult": 1.1, "direction_lookback": 6},
            position_size_pct=0.03,
        ),
    ]
    datasets = {
        "BTC/USDT": _make_dataset(periods=720, seed=101, drift=0.0002, vol=0.012, volume_level=14_000),
        "SPY": _make_dataset(periods=720, seed=202, drift=0.0012, vol=0.004, volume_level=60_000),
        "CL1": _make_dataset(periods=720, seed=303, drift=0.0004, vol=0.007, volume_level=20_000),
    }

    adaptive = AdaptivePortfolioEngine.from_primitive_specs(
        specs,
        assets,
        capital=10_000,
        data_days=60,
        use_risk_manager=False,
        use_divergence_tracker=False,
        use_market_state_brain=False,
        use_execution_edge=False,
        use_capital_scaling=False,
    )
    original_position_sizes = {
        spec.name: spec.position_size_pct
        for spec in adaptive.primitive_specs
    }

    result = adaptive.walk_forward_adaptive(
        datasets,
        train_bars=240,
        test_bars=120,
        n_folds=3,
        anchored=False,
        warmup_bars=220,
    )

    assert len(result.folds) == 3
    assert result.summary["n_folds"] == pytest.approx(3.0)
    assert not result.portfolio_returns.empty
    assert not result.equity_curve.empty
    assert len(result.final_engine.slots) >= len(specs)
    assert result.summary["final_capital"] > 0.0

    for fold in result.folds:
        assert sum(fold.deployed_weights.values()) == pytest.approx(1.0)
        assert sum(fold.next_cycle_weights.values()) == pytest.approx(1.0)
        assert fold.test_summary["start_capital"] > 0.0

    assert {
        spec.name: spec.position_size_pct
        for spec in adaptive.primitive_specs
    } == original_position_sizes
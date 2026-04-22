from __future__ import annotations

from types import SimpleNamespace

from src.ops.adaptive_runtime import (
    AdaptiveSafetyGovernor,
    TradingCycleState,
    base_strategy_name,
    build_trading_cycle_state,
)


def test_base_strategy_name_strips_variant_suffix() -> None:
    assert base_strategy_name("funding_mr_v7") == "funding_mr_v7"
    assert base_strategy_name("funding_mr_v7__mutate") == "funding_mr_v7"
    assert base_strategy_name("momentum_breakout__recombine") == "momentum_breakout"


def test_build_trading_cycle_state_extracts_adaptive_metrics() -> None:
    report = SimpleNamespace(
        allocation_decision=SimpleNamespace(
            projected_utilization=0.55,
            risk_budget_multiplier=1.10,
            gross_exposure_scale=0.95,
            target_volatility=0.12,
            realized_volatility=0.10,
            portfolio_objective_score=0.24,
            correlation_shock=False,
            disabled_strategies={"alpha": "weak edge"},
            weights={"alpha": 0.7, "beta": 0.3},
            market_route_multipliers={"crypto": 1.1},
        ),
        risk_snapshot=SimpleNamespace(
            smoothed_volatility=0.11,
            volatility_tracking_error=0.02,
            pid_output=0.15,
        ),
        reality_gap=SimpleNamespace(
            edge_retention_ratio=0.82,
            edge_retention_state="strong",
            pnl_gap_fraction=0.08,
        ),
        regime_states={
            "BTC/USDT": SimpleNamespace(
                composite="normal_volatility|bull_trend|liquid",
                trend_regime="bull_trend",
                volatility_regime="normal_volatility",
                liquidity_regime="liquid",
                confidence_score=0.84,
            )
        },
        lifecycle_decisions={
            "alpha": SimpleNamespace(retired=True),
            "beta": SimpleNamespace(retired=False),
        },
        suggested_position_sizes={"alpha": 0.015, "beta": 0.01},
        adapted_engine=SimpleNamespace(
            slots=[
                SimpleNamespace(name="alpha"),
                SimpleNamespace(name="alpha__mutate"),
            ]
        ),
    )
    divergence_stats = SimpleNamespace(
        total_trades=6,
        total_missed=1,
        avg_entry_slippage_bps=3.5,
        avg_exit_slippage_bps=1.4,
        avg_pnl_divergence_pct=-4.2,
        slippage_trend=0.1,
        alerts=["watch drift"],
    )

    state = build_trading_cycle_state(
        report,
        divergence_stats=divergence_stats,
        capital=12_500.0,
        current_drawdown=0.03,
        daily_pnl=120.0,
        iteration=7,
        timestamp="2026-04-21T10:00:00+00:00",
        paper_mode=True,
    )

    assert state.projected_utilization == 0.55 * 1.10 * 0.95
    assert state.portfolio_objective_score == 0.24
    assert state.edge_retention_ratio == 0.82
    assert state.execution["avg_entry_slippage_bps"] == 3.5
    assert state.retired_strategies == ["alpha"]
    assert state.adapted_slots == ["alpha", "alpha__mutate"]
    assert state.regime_states["BTC/USDT"]["trend"] == "bull_trend"


def test_adaptive_safety_governor_pauses_on_execution_or_edge_break() -> None:
    state = TradingCycleState(
        iteration=1,
        timestamp="2026-04-21T10:00:00+00:00",
        paper_mode=True,
        capital=10_000.0,
        current_drawdown=0.02,
        daily_pnl=0.0,
        projected_utilization=0.4,
        gross_exposure_scale=1.0,
        risk_budget_multiplier=1.0,
        target_volatility=0.12,
        realized_volatility=0.11,
        smoothed_volatility=0.11,
        volatility_tracking_error=0.03,
        pid_output=0.0,
        edge_retention_ratio=0.30,
        edge_retention_state="fragile",
        reality_gap_fraction=0.15,
        portfolio_objective_score=0.10,
        correlation_shock=False,
        execution={
            "avg_entry_slippage_bps": 14.0,
            "avg_pnl_divergence_pct": 5.0,
        },
    )

    decision = AdaptiveSafetyGovernor().evaluate(state)

    assert decision.action == "pause_entries"
    assert decision.size_cap == 0.0
    assert any("edge retention" in reason or "slippage" in reason for reason in decision.reasons)


def test_adaptive_safety_governor_reduces_under_objective_and_corr_stress() -> None:
    state = TradingCycleState(
        iteration=2,
        timestamp="2026-04-21T11:00:00+00:00",
        paper_mode=True,
        capital=10_000.0,
        current_drawdown=0.04,
        daily_pnl=-50.0,
        projected_utilization=0.6,
        gross_exposure_scale=1.0,
        risk_budget_multiplier=1.0,
        target_volatility=0.12,
        realized_volatility=0.16,
        smoothed_volatility=0.15,
        volatility_tracking_error=0.04,
        pid_output=-0.2,
        edge_retention_ratio=0.80,
        edge_retention_state="strong",
        reality_gap_fraction=0.05,
        portfolio_objective_score=-0.30,
        correlation_shock=True,
        execution={
            "avg_entry_slippage_bps": 5.0,
            "avg_pnl_divergence_pct": 3.0,
        },
    )

    decision = AdaptiveSafetyGovernor().evaluate(state)

    assert decision.action == "reduce"
    assert decision.size_cap == 0.5
    assert len(decision.reasons) >= 1


def test_adaptive_safety_governor_allows_low_volatility_undershoot() -> None:
    state = TradingCycleState(
        iteration=13,
        timestamp="2026-04-22T03:01:45+00:00",
        paper_mode=True,
        capital=10_000.0,
        current_drawdown=0.0,
        daily_pnl=0.0,
        projected_utilization=0.0,
        gross_exposure_scale=1.75,
        risk_budget_multiplier=1.6,
        target_volatility=0.15,
        realized_volatility=0.0030618997970943895,
        smoothed_volatility=0.0030618997970943895,
        volatility_tracking_error=0.1469381002029056,
        pid_output=1.4204016352947542,
        edge_retention_ratio=0.9090046425304488,
        edge_retention_state="strong",
        reality_gap_fraction=0.09099535746955123,
        portfolio_objective_score=0.12,
        correlation_shock=False,
        execution={
            "avg_entry_slippage_bps": 0.0,
            "avg_pnl_divergence_pct": 0.0,
        },
    )

    decision = AdaptiveSafetyGovernor().evaluate(state)

    assert decision.action == "allow"
    assert decision.size_cap == 1.0
    assert decision.reasons == []


def test_adaptive_safety_governor_pauses_on_large_volatility_overshoot() -> None:
    state = TradingCycleState(
        iteration=14,
        timestamp="2026-04-22T04:01:45+00:00",
        paper_mode=True,
        capital=10_000.0,
        current_drawdown=0.0,
        daily_pnl=0.0,
        projected_utilization=0.6,
        gross_exposure_scale=1.0,
        risk_budget_multiplier=1.0,
        target_volatility=0.15,
        realized_volatility=0.29,
        smoothed_volatility=0.24,
        volatility_tracking_error=0.14,
        pid_output=-0.8,
        edge_retention_ratio=0.90,
        edge_retention_state="strong",
        reality_gap_fraction=0.02,
        portfolio_objective_score=0.18,
        correlation_shock=False,
        execution={
            "avg_entry_slippage_bps": 0.0,
            "avg_pnl_divergence_pct": 0.0,
        },
    )

    decision = AdaptiveSafetyGovernor().evaluate(state)

    assert decision.action == "pause_entries"
    assert any("tracking error" in reason for reason in decision.reasons)
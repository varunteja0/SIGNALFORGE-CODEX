#!/usr/bin/env python3
"""Demonstrate adaptive allocation for the self-improving portfolio engine."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.allocation.adaptive import AdaptiveAllocationEngine
from src.allocation.capital_efficiency import CapitalUtilizationEngine
from src.core.dataset_cache import load_or_build_datasets
from src.data.market_data import AssetSpec
from src.engine.adaptive_portfolio_engine import AdaptivePortfolioEngine
from src.engine.alpha_primitives import PrimitiveSlotSpec
from src.engine.portfolio_engine import PortfolioEngine


def load_example_config(path: str | Path) -> tuple[list[AssetSpec], list[PrimitiveSlotSpec]]:
    payload = yaml.safe_load(Path(path).read_text())
    assets = [AssetSpec(**row) for row in payload.get("assets", [])]
    strategies = [PrimitiveSlotSpec(**row) for row in payload.get("strategies", [])]
    return assets, strategies


def make_synthetic_dataset(
    *,
    periods: int,
    seed: int,
    drift: float,
    vol: float,
    volume_level: float,
    shock_every: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = drift + rng.normal(0.0, vol, periods)
    if shock_every:
        shock_idx = np.arange(shock_every, periods, shock_every)
        returns[shock_idx] += rng.normal(0.0, vol * 4.0, len(shock_idx))

    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + abs(vol) * 2.0)
    low = np.minimum(open_, close) * (1.0 - abs(vol) * 2.0)
    volume = volume_level * np.clip(1.0 + rng.normal(0.0, 0.15, periods), 0.2, None)
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range("2024-01-01", periods=periods, freq="h"),
    )
    return frame


def build_synthetic_datasets(assets: list[AssetSpec], periods: int) -> dict[str, pd.DataFrame]:
    templates = {
        "crypto": dict(drift=0.0004, vol=0.012, volume_level=12_000, shock_every=48),
        "equities": dict(drift=0.0007, vol=0.004, volume_level=40_000, shock_every=96),
        "commodities": dict(drift=0.0002, vol=0.008, volume_level=18_000, shock_every=72),
        "indices": dict(drift=0.0005, vol=0.005, volume_level=30_000, shock_every=84),
    }
    datasets = {}
    for idx, asset in enumerate(assets):
        template = templates[asset.market_type.value]
        datasets[asset.symbol] = make_synthetic_dataset(
            periods=periods,
            seed=100 + idx,
            **template,
        )
    return datasets


def summarize_report(report) -> dict:
    return {
        "portfolio": {
            "total_trades": int(report.base_result.total_trades),
            "net_pnl": round(float(report.base_result.total_pnl), 2),
            "profit_factor": round(float(report.base_result.profit_factor), 4),
            "sharpe": round(float(report.base_result.sharpe), 4),
            "max_drawdown": round(float(report.base_result.max_drawdown), 4),
        },
        "adaptive_allocation": {
            "weights": {k: round(float(v), 4) for k, v in report.allocation_decision.weights.items()},
            "gross_exposure_scale": round(float(report.allocation_decision.gross_exposure_scale), 4),
            "projected_utilization": round(float(report.allocation_decision.projected_utilization), 4),
            "effective_utilization": round(
                float(
                    report.allocation_decision.projected_utilization
                    * report.allocation_decision.risk_budget_multiplier
                    * report.allocation_decision.gross_exposure_scale
                ),
                4,
            ),
            "target_utilization": round(float(report.allocation_decision.target_utilization), 4),
            "risk_budget_multiplier": round(float(report.allocation_decision.risk_budget_multiplier), 4),
            "idle_capital_penalty": round(float(report.allocation_decision.idle_capital_penalty), 4),
            "disabled": report.allocation_decision.disabled_strategies,
            "exploration_weights": {
                k: round(float(v), 4)
                for k, v in report.allocation_decision.exploration_weights.items()
                if v > 0.0
            },
            "suggested_position_sizes": {
                k: round(float(v), 6) for k, v in report.suggested_position_sizes.items()
            },
            "weight_turnover": round(float(report.allocation_decision.weight_turnover), 4),
            "portfolio_objective_score": round(float(report.allocation_decision.portfolio_objective_score), 4),
            "market_route_multipliers": {
                k: round(float(v), 4)
                for k, v in report.allocation_decision.market_route_multipliers.items()
            },
        },
        "execution_realism": {
            "total_execution_cost": round(
                float(report.execution_summary.total_execution_cost),
                4,
            ) if report.execution_summary else 0.0,
            "average_fill_ratio": round(
                float(report.execution_summary.average_fill_ratio),
                4,
            ) if report.execution_summary else 1.0,
            "market_impact_cost": round(
                float(report.execution_summary.market_impact_cost),
                4,
            ) if report.execution_summary else 0.0,
        },
        "risk_controls": {
            "target_volatility": round(float(report.allocation_decision.target_volatility), 4),
            "realized_volatility": round(float(report.allocation_decision.realized_volatility), 4),
            "smoothed_volatility": round(float(report.risk_snapshot.smoothed_volatility), 4) if report.risk_snapshot else 0.0,
            "volatility_tracking_error": round(float(report.risk_snapshot.volatility_tracking_error), 4) if report.risk_snapshot else 0.0,
            "volatility_multiplier": round(float(report.allocation_decision.volatility_multiplier), 4),
            "pid_output": round(float(report.risk_snapshot.pid_output), 4) if report.risk_snapshot else 0.0,
            "correlation_multiplier": round(float(report.allocation_decision.correlation_multiplier), 4),
            "correlation_shock": bool(report.allocation_decision.correlation_shock),
            "strategy_correlation": round(float(report.allocation_decision.strategy_correlation), 4),
            "asset_correlation": round(float(report.allocation_decision.asset_correlation), 4),
            "portfolio_growth_score": round(float(report.allocation_decision.portfolio_growth_score), 4),
        },
        "liquidity_controls": {
            "capped": report.liquidity_snapshot.capped_strategies if report.liquidity_snapshot else {},
            "market_impact_bps": {
                k: round(float(v), 4)
                for k, v in (report.liquidity_snapshot.market_impact_bps.items() if report.liquidity_snapshot else [])
            },
            "participation": {
                k: round(float(v), 6)
                for k, v in (report.liquidity_snapshot.participation_by_strategy.items() if report.liquidity_snapshot else [])
            },
        },
        "reality_gap": {
            "theoretical_pnl": round(float(report.reality_gap.theoretical_pnl), 4) if report.reality_gap else 0.0,
            "execution_adjusted_pnl": round(float(report.reality_gap.execution_adjusted_pnl), 4) if report.reality_gap else 0.0,
            "pnl_gap_fraction": round(float(report.reality_gap.pnl_gap_fraction), 4) if report.reality_gap else 0.0,
            "execution_cost_ratio": round(float(report.reality_gap.execution_cost_ratio), 4) if report.reality_gap else 0.0,
            "edge_retention_ratio": round(float(report.reality_gap.edge_retention_ratio), 4) if report.reality_gap else 0.0,
            "edge_retention_state": report.reality_gap.edge_retention_state if report.reality_gap else "unknown",
            "fragile_edge": bool(report.reality_gap.fragile_edge) if report.reality_gap else False,
            "strategy_edge_retention": {
                name: {
                    "ratio": round(float(snapshot.edge_retention_ratio), 4),
                    "status": snapshot.status,
                }
                for name, snapshot in (report.reality_gap.strategy_edge_retention.items() if report.reality_gap else [])
            },
        },
        "regimes": {
            symbol: {
                "volatility": state.volatility_regime,
                "trend": state.trend_regime,
                "liquidity": state.liquidity_regime,
                "confidence": round(float(state.confidence_score), 4),
            }
            for symbol, state in report.regime_states.items()
        },
        "performance": {
            name: {
                "state": snapshot.state,
                "multiplier": round(float(snapshot.recommended_multiplier), 4),
                "rolling_sharpe": round(float(snapshot.rolling_sharpe), 4),
                "rolling_drawdown": round(float(snapshot.rolling_drawdown), 4),
                "win_rate": round(float(snapshot.rolling_win_rate), 4),
                "signal_strength": round(float(snapshot.signal_strength), 4),
                "conviction_score": round(float(snapshot.conviction_score), 4),
                "persistence_score": round(float(snapshot.persistence_score), 4),
                "growth_score": round(float(snapshot.growth_score), 4),
                "annualized_return": round(float(snapshot.annualized_return), 4),
                "realized_volatility": round(float(snapshot.realized_volatility), 4),
            }
            for name, snapshot in report.performance_metrics.items()
        },
        "lifecycle": {
            name: {
                "status": decision.status,
                "degradation_streak": int(decision.degradation_streak),
                "replacement_candidate": decision.replacement_candidate,
                "retired": bool(decision.retired),
            }
            for name, decision in report.lifecycle_decisions.items()
        },
        "learning_updates": {
            update.strategy_name: {
                "reward": round(float(update.reward), 4),
                "risk_bias": round(float(update.risk_bias), 4),
                "parameter_suggestions": update.parameter_suggestions,
            }
            for update in report.learning_result.updates
        },
    }


def _stress_metrics(portfolio_returns: pd.Series, datasets: dict[str, pd.DataFrame]) -> dict:
    if portfolio_returns is None or portfolio_returns.empty:
        return {
            "bar_fraction": 0.0,
            "total_return": 0.0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
        }

    asset_returns = pd.DataFrame(
        {
            symbol: frame["close"].pct_change().replace([np.inf, -np.inf], np.nan)
            for symbol, frame in datasets.items()
            if frame is not None and not frame.empty
        }
    )
    if asset_returns.empty:
        return {
            "bar_fraction": 0.0,
            "total_return": 0.0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
        }

    aligned = asset_returns.reindex(portfolio_returns.index).fillna(0.0)
    stress_signal = aligned.abs().mean(axis=1)
    threshold = float(stress_signal.quantile(0.85)) if len(stress_signal) > 0 else 0.0
    stress_mask = stress_signal >= threshold
    stress_returns = portfolio_returns.loc[stress_mask].fillna(0.0)
    if stress_returns.empty:
        return {
            "bar_fraction": 0.0,
            "total_return": 0.0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
        }

    stress_equity = (1.0 + stress_returns).cumprod()
    drawdown = float(abs((stress_equity / stress_equity.cummax() - 1.0).min())) if len(stress_equity) > 0 else 0.0
    return {
        "bar_fraction": round(float(stress_mask.mean()), 4),
        "total_return": round(float((1.0 + stress_returns).prod() - 1.0), 4),
        "hit_rate": round(float((stress_returns > 0.0).mean()), 4),
        "max_drawdown": round(drawdown, 4),
    }


def summarize_walk_forward_report(result, datasets: dict[str, pd.DataFrame] | None = None) -> dict:
    payload = {
        "walk_forward": {
            key: round(float(value), 4)
            for key, value in result.summary.items()
        },
        "folds": [
            {
                "index": fold.index,
                "train": {
                    "start": str(fold.train_start),
                    "end": str(fold.train_end),
                },
                "test": {
                    "start": str(fold.test_start),
                    "end": str(fold.test_end),
                },
                "train_summary": {
                    key: round(float(value), 4)
                    for key, value in fold.train_summary.items()
                },
                "test_summary": {
                    key: round(float(value), 4)
                    for key, value in fold.test_summary.items()
                },
                "deployed_weights": {
                    key: round(float(value), 4)
                    for key, value in fold.deployed_weights.items()
                },
                "next_cycle_weights": {
                    key: round(float(value), 4)
                    for key, value in fold.next_cycle_weights.items()
                },
                "projected_utilization": round(float(fold.projected_utilization), 4),
                "next_projected_utilization": round(float(fold.next_projected_utilization), 4),
                "execution_cost": round(float(fold.execution_cost), 4),
                "target_volatility": round(float(fold.target_volatility), 4),
                "realized_volatility": round(float(fold.realized_volatility), 4),
                "volatility_tracking_error": round(float(fold.volatility_tracking_error), 4),
                "correlation_shock": bool(fold.correlation_shock),
                "reality_gap_fraction": round(float(fold.reality_gap_fraction), 4),
                "edge_retention_ratio": round(float(fold.edge_retention_ratio), 4),
                "portfolio_objective_score": round(float(fold.portfolio_objective_score), 4),
                "retired_strategies": list(fold.retired_strategies),
            }
            for fold in result.folds
        ],
        "final_position_sizes": {
            slot.name: round(float(slot.position_size_pct), 6)
            for slot in result.final_engine.slots
        },
    }
    if datasets is not None:
        payload["stress"] = _stress_metrics(result.portfolio_returns, datasets)
    return payload


def _configure_baseline_engine(adaptive: AdaptivePortfolioEngine) -> AdaptivePortfolioEngine:
    adaptive.allocator = AdaptiveAllocationEngine(
        blend=0.55,
        conviction_weight=0.0,
        exploration_epsilon=0.0,
        weight_inertia=0.0,
        max_gross_exposure_scale=1.0,
        objective_weight=0.0,
        execution_efficiency_tilt=0.0,
        market_routing_tilt=0.0,
        capital_utilization_engine=CapitalUtilizationEngine(
            target_utilization=0.55,
            min_utilization=0.0,
            max_risk_budget_multiplier=1.0,
            idle_pressure=0.0,
            conviction_boost=0.0,
        ),
    )
    adaptive.risk_controller = None
    adaptive.liquidity_engine = None
    adaptive.lifecycle_manager = None
    return adaptive


def _compare_payloads(baseline: dict, upgraded: dict, *, walk_forward: bool) -> dict:
    base_metrics = baseline["walk_forward"] if walk_forward else baseline["portfolio"]
    upgraded_metrics = upgraded["walk_forward"] if walk_forward else upgraded["portfolio"]
    base_util = (
        baseline["walk_forward"].get("mean_utilization", 0.0)
        if walk_forward
        else baseline["adaptive_allocation"].get("effective_utilization", baseline["adaptive_allocation"].get("projected_utilization", 0.0))
    )
    upgraded_util = (
        upgraded["walk_forward"].get("mean_utilization", 0.0)
        if walk_forward
        else upgraded["adaptive_allocation"].get("effective_utilization", upgraded["adaptive_allocation"].get("projected_utilization", 0.0))
    )

    baseline_return = base_metrics.get("total_return", base_metrics.get("return", 0.0))
    upgraded_return = upgraded_metrics.get("total_return", upgraded_metrics.get("return", 0.0))
    baseline_drawdown = base_metrics.get("max_drawdown", 0.0)
    upgraded_drawdown = upgraded_metrics.get("max_drawdown", 0.0)
    baseline_edge_retention = (
        baseline.get("walk_forward", {}).get("mean_edge_retention", 0.0)
        if walk_forward
        else baseline.get("reality_gap", {}).get("edge_retention_ratio", 0.0)
    )
    upgraded_edge_retention = (
        upgraded.get("walk_forward", {}).get("mean_edge_retention", 0.0)
        if walk_forward
        else upgraded.get("reality_gap", {}).get("edge_retention_ratio", 0.0)
    )
    baseline_objective = (
        baseline.get("walk_forward", {}).get("mean_portfolio_objective_score", 0.0)
        if walk_forward
        else baseline.get("adaptive_allocation", {}).get("portfolio_objective_score", 0.0)
    )
    upgraded_objective = (
        upgraded.get("walk_forward", {}).get("mean_portfolio_objective_score", 0.0)
        if walk_forward
        else upgraded.get("adaptive_allocation", {}).get("portfolio_objective_score", 0.0)
    )

    return {
        "baseline": baseline,
        "upgraded": upgraded,
        "comparison": {
            "capital_utilization_delta": round(float(upgraded_util - base_util), 4),
            "return_delta": round(float(upgraded_return - baseline_return), 4),
            "drawdown_delta": round(float(upgraded_drawdown - baseline_drawdown), 4),
            "cagr_delta": round(
                float(
                    (upgraded.get("walk_forward", {}).get("cagr", 0.0) if walk_forward else upgraded_return)
                    - (baseline.get("walk_forward", {}).get("cagr", 0.0) if walk_forward else baseline_return)
                ),
                4,
            ),
            "volatility_tracking_delta": round(
                float(
                    upgraded.get("walk_forward", {}).get("volatility_tracking_error", 0.0)
                    - baseline.get("walk_forward", {}).get("volatility_tracking_error", 0.0)
                ),
                4,
            ) if walk_forward else 0.0,
            "stress_return_delta": round(
                float(upgraded.get("stress", {}).get("total_return", 0.0) - baseline.get("stress", {}).get("total_return", 0.0)),
                4,
            ) if walk_forward else 0.0,
            "edge_retention_delta": round(float(upgraded_edge_retention - baseline_edge_retention), 4),
            "objective_score_delta": round(float(upgraded_objective - baseline_objective), 4),
            "improved_capital_usage": bool(upgraded_util > base_util),
            "improved_return": bool(upgraded_return > baseline_return),
            "improved_cagr": bool(
                upgraded.get("walk_forward", {}).get("cagr", upgraded_return)
                > baseline.get("walk_forward", {}).get("cagr", baseline_return)
            ),
            "improved_volatility_stability": bool(
                upgraded.get("walk_forward", {}).get("volatility_tracking_error", 0.0)
                <= baseline.get("walk_forward", {}).get("volatility_tracking_error", 0.0)
            ) if walk_forward else True,
            "higher_edge_retention": bool(upgraded_edge_retention >= baseline_edge_retention),
            "better_risk_adjusted_returns": bool(upgraded_objective > baseline_objective),
            "stable_risk_profile": bool(
                upgraded.get("walk_forward", {}).get("volatility_tracking_error", 0.0)
                <= baseline.get("walk_forward", {}).get("volatility_tracking_error", 0.0)
                and upgraded_drawdown <= max(baseline_drawdown + 0.03, 0.12)
            ) if walk_forward else True,
            "stress_resilience_improved": bool(
                upgraded.get("stress", {}).get("total_return", 0.0)
                >= baseline.get("stress", {}).get("total_return", 0.0)
                and upgraded.get("stress", {}).get("max_drawdown", upgraded_drawdown)
                <= baseline.get("stress", {}).get("max_drawdown", baseline_drawdown) + 0.03
            ) if walk_forward else True,
            "controlled_drawdown": bool(upgraded_drawdown <= max(baseline_drawdown + 0.03, 0.12)),
        },
    }


def run_synthetic_demo(
    config_path: str | Path,
    periods: int,
    *,
    walk_forward: bool = False,
    train_bars: int = 220,
    test_bars: int = 80,
    folds: int = 3,
    anchored: bool = True,
    warmup_bars: int = 220,
    compare_baseline: bool = False,
) -> dict:
    assets, strategy_specs = load_example_config(config_path)
    datasets = build_synthetic_datasets(assets, periods)
    adaptive = AdaptivePortfolioEngine.from_primitive_specs(
        strategy_specs,
        assets,
        capital=10_000,
        data_days=max(30, periods // 24),
        use_risk_manager=False,
        use_divergence_tracker=False,
        use_market_state_brain=False,
        use_execution_edge=False,
        use_capital_scaling=False,
    )
    baseline_adaptive = None
    if compare_baseline:
        baseline_adaptive = _configure_baseline_engine(
            AdaptivePortfolioEngine.from_primitive_specs(
                strategy_specs,
                assets,
                capital=10_000,
                data_days=max(30, periods // 24),
                use_risk_manager=False,
                use_divergence_tracker=False,
                use_market_state_brain=False,
                use_execution_edge=False,
                use_capital_scaling=False,
            )
        )
    if walk_forward:
        result = adaptive.walk_forward_adaptive(
            datasets,
            train_bars=train_bars,
            test_bars=test_bars,
            n_folds=folds,
            anchored=anchored,
            warmup_bars=warmup_bars,
        )
        upgraded_summary = summarize_walk_forward_report(result, datasets)
        if not compare_baseline:
            return upgraded_summary
        baseline_result = baseline_adaptive.walk_forward_adaptive(
            datasets,
            train_bars=train_bars,
            test_bars=test_bars,
            n_folds=folds,
            anchored=anchored,
            warmup_bars=warmup_bars,
        )
        return _compare_payloads(
            summarize_walk_forward_report(baseline_result, datasets),
            upgraded_summary,
            walk_forward=True,
        )

    report = adaptive.backtest_adaptive_cycle(datasets)
    upgraded_summary = summarize_report(report)
    if not compare_baseline:
        return upgraded_summary
    baseline_report = baseline_adaptive.backtest_adaptive_cycle(datasets)
    return _compare_payloads(
        summarize_report(baseline_report),
        upgraded_summary,
        walk_forward=False,
    )


def run_legacy_demo(
    profile: str,
    *,
    walk_forward: bool = False,
    train_bars: int = 220,
    test_bars: int = 80,
    folds: int = 3,
    anchored: bool = True,
    warmup_bars: int = 220,
    compare_baseline: bool = False,
) -> dict:
    if profile == "compounding":
        base_engine = PortfolioEngine.compounding_focus()
    else:
        base_engine = PortfolioEngine.default()
    base_engine.capital = 10_000
    base_engine.data_days = 180 if profile == "default" else 365

    datasets, _, _ = load_or_build_datasets(
        base_engine,
        namespace=f"adaptive_demo_{profile}",
        max_age_hours=12.0,
        force_refresh=False,
    )
    adaptive = AdaptivePortfolioEngine.from_portfolio_engine(base_engine)
    baseline_adaptive = None
    if compare_baseline:
        baseline_adaptive = _configure_baseline_engine(
            AdaptivePortfolioEngine.from_portfolio_engine(deepcopy(base_engine))
        )
    if walk_forward:
        result = adaptive.walk_forward_adaptive(
            datasets,
            train_bars=train_bars,
            test_bars=test_bars,
            n_folds=folds,
            anchored=anchored,
            warmup_bars=warmup_bars,
        )
        upgraded_summary = summarize_walk_forward_report(result, datasets)
        if not compare_baseline:
            return upgraded_summary
        baseline_result = baseline_adaptive.walk_forward_adaptive(
            datasets,
            train_bars=train_bars,
            test_bars=test_bars,
            n_folds=folds,
            anchored=anchored,
            warmup_bars=warmup_bars,
        )
        return _compare_payloads(
            summarize_walk_forward_report(baseline_result, datasets),
            upgraded_summary,
            walk_forward=True,
        )

    report = adaptive.backtest_adaptive_cycle(datasets)
    upgraded_summary = summarize_report(report)
    if not compare_baseline:
        return upgraded_summary
    baseline_report = baseline_adaptive.backtest_adaptive_cycle(datasets)
    return _compare_payloads(
        summarize_report(baseline_report),
        upgraded_summary,
        walk_forward=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate adaptive allocation")
    parser.add_argument(
        "--mode",
        choices=["synthetic-primitives", "legacy"],
        default="synthetic-primitives",
        help="Demo mode",
    )
    parser.add_argument(
        "--config",
        default="config/adaptive_multi_asset_example.yaml",
        help="Primitive demo config path",
    )
    parser.add_argument(
        "--profile",
        choices=["default", "compounding"],
        default="compounding",
        help="Legacy portfolio profile",
    )
    parser.add_argument("--periods", type=int, default=480, help="Synthetic demo bars")
    parser.add_argument("--walk-forward", action="store_true", help="Run adaptive walk-forward simulation")
    parser.add_argument("--train-bars", type=int, default=220, help="Walk-forward training bars")
    parser.add_argument("--test-bars", type=int, default=80, help="Walk-forward test bars")
    parser.add_argument("--folds", type=int, default=3, help="Walk-forward fold count")
    parser.add_argument("--rolling", action="store_true", help="Use rolling instead of anchored folds")
    parser.add_argument("--warmup-bars", type=int, default=220, help="Warmup bars before each OOS fold")
    parser.add_argument("--compare-baseline", action="store_true", help="Compare upgraded engine against a baseline allocator")
    parser.add_argument("--output", type=str, default="", help="Optional output path")
    args = parser.parse_args()

    if args.mode == "synthetic-primitives":
        payload = run_synthetic_demo(
            args.config,
            args.periods,
            walk_forward=args.walk_forward,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            folds=args.folds,
            anchored=not args.rolling,
            warmup_bars=args.warmup_bars,
            compare_baseline=args.compare_baseline,
        )
    else:
        payload = run_legacy_demo(
            args.profile,
            walk_forward=args.walk_forward,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            folds=args.folds,
            anchored=not args.rolling,
            warmup_bars=args.warmup_bars,
            compare_baseline=args.compare_baseline,
        )

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


if __name__ == "__main__":
    main()
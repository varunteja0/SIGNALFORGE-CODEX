from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from src.core.dataset_cache import load_or_build_datasets
from src.engine.institutional import InstitutionalValidator, RegimeAnalyzer
from src.engine.portfolio_engine import PortfolioEngine, StrategySlot
from src.engine.structural_stress import ContrarianAsymmetryEngine


def _variant_grid(grid_mode: str) -> list[tuple[float, int, bool, bool]]:
    if grid_mode == "fast":
        return list(product(
            [1.5, 1.75],
            [8, 12],
            [False],
            [False, True],
        ))

    return list(product(
        [1.5, 1.75, 2.0],
        [8, 12],
        [False, True],
        [False, True],
    ))


def _build_variant_slot(base_slot: StrategySlot, params: dict) -> StrategySlot:
    frozen = dict(params)
    return StrategySlot(
        name=base_slot.name,
        template=base_slot.template,
        signal_func=lambda df, p=frozen: ContrarianAsymmetryEngine.generate_signals(
            df,
            funding_z_threshold=p["funding_z_threshold"],
            funding_lookback=168,
            hold_bars=p["hold_bars"],
            require_volume_confirm=p["require_volume_confirm"],
            volume_z_threshold=p["volume_z_threshold"],
            trend_filter=p["trend_filter"],
            trend_sma_period=50,
        ),
        params=frozen,
        allowed_assets=list(base_slot.allowed_assets),
        regime_filter=base_slot.regime_filter,
        stop_loss_atr=base_slot.stop_loss_atr,
        take_profit_atr=base_slot.take_profit_atr,
        max_holding_bars=base_slot.max_holding_bars,
        position_size_pct=base_slot.position_size_pct,
        use_vwap=base_slot.use_vwap,
    )


def _build_variant_engine(base_engine: PortfolioEngine, contrarian_slot: StrategySlot) -> PortfolioEngine:
    slots = list(base_engine.slots)
    idx = next(i for i, slot in enumerate(slots) if slot.name == "contrarian_asym")
    slots[idx] = contrarian_slot
    return PortfolioEngine(
        slots=slots,
        assets=list(base_engine.assets),
        capital=base_engine.capital,
        data_days=base_engine.data_days,
        max_total_exposure=base_engine.max_total_exposure,
        max_drawdown_kill=base_engine.max_drawdown_kill,
        use_regime_allocator=base_engine.regime_allocator is not None,
        use_risk_manager=base_engine.risk_manager is not None,
        use_divergence_tracker=False,
        use_market_state_brain=False,
        use_execution_edge=False,
        use_live_adaptation=False,
        use_capital_scaling=False,
    )


def _contrarian_regime_pass(slot: StrategySlot, datasets: dict) -> tuple[bool, dict]:
    breakdown = RegimeAnalyzer().analyze([slot], datasets)
    aggregated = {}
    for regime, cells in breakdown.data.items():
        gross_win = 0.0
        gross_loss = 0.0
        total_trades = 0
        for cell_key, stats in cells.items():
            if not cell_key.startswith("contrarian_asym|"):
                continue
            gross_win += float(stats.get("gross_win", 0.0))
            gross_loss += float(stats.get("gross_loss", 0.0))
            total_trades += int(stats.get("total_trades", 0))
        if total_trades == 0:
            continue
        pf = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
        aggregated[regime] = {
            "gross_win": gross_win,
            "gross_loss": gross_loss,
            "total_trades": total_trades,
            "pf": pf,
            "passes": total_trades >= 3 and pf > 1.0,
        }
    return any(stats["passes"] for stats in aggregated.values()), aggregated


def _variant_rows(
    base_engine: PortfolioEngine,
    datasets: dict,
    top_n: int,
    validator_top_n: int,
    grid_mode: str,
) -> dict:
    base_backtest = base_engine.backtest(datasets)
    base_contrarian = next(slot for slot in base_engine.slots if slot.name == "contrarian_asym")
    baseline_regime_pass, baseline_regimes = _contrarian_regime_pass(base_contrarian, datasets)
    baseline_report = InstitutionalValidator().validate(base_engine, datasets)

    rows = [
        {
            "variant": "baseline",
            "params": dict(base_contrarian.params),
            "portfolio_return": base_backtest.total_pnl / base_engine.capital,
            "portfolio_sharpe": base_backtest.sharpe,
            "portfolio_pf": base_backtest.profit_factor,
            "portfolio_max_drawdown": base_backtest.max_drawdown,
            "contrarian_pnl": base_backtest.strategy_results["contrarian_asym"]["net_pnl"],
            "contrarian_pf": base_backtest.strategy_results["contrarian_asym"]["pf"],
            "contrarian_trades": base_backtest.strategy_results["contrarian_asym"]["trades"],
            "passes_any_regime_gate": baseline_regime_pass,
            "regime_breakdown": baseline_regimes,
            "institutional_score": baseline_report.score,
            "institutional_total_tests": baseline_report.total_tests,
            "institutional_regime_verdict": baseline_report.verdicts.get("all_strats_profitable_1+_regime"),
        }
    ]

    grid = _variant_grid(grid_mode)

    for funding_z_threshold, hold_bars, require_volume_confirm, trend_filter in grid:
        params = {
            "funding_z_threshold": funding_z_threshold,
            "hold_bars": hold_bars,
            "require_volume_confirm": require_volume_confirm,
            "volume_z_threshold": 1.0,
            "trend_filter": trend_filter,
        }
        slot = _build_variant_slot(base_contrarian, params)
        engine = _build_variant_engine(base_engine, slot)
        result = engine.backtest(datasets)
        strategy = result.strategy_results["contrarian_asym"]
        passes_regime, regime_breakdown = _contrarian_regime_pass(slot, datasets)
        rows.append(
            {
                "variant": (
                    f"fz{funding_z_threshold:.2f}_hb{hold_bars}_"
                    f"vol{int(require_volume_confirm)}_trend{int(trend_filter)}"
                ),
                "params": params,
                "portfolio_return": result.total_pnl / base_engine.capital,
                "portfolio_sharpe": result.sharpe,
                "portfolio_pf": result.profit_factor,
                "portfolio_max_drawdown": result.max_drawdown,
                "contrarian_pnl": strategy["net_pnl"],
                "contrarian_pf": strategy["pf"],
                "contrarian_trades": strategy["trades"],
                "passes_any_regime_gate": passes_regime,
                "regime_breakdown": regime_breakdown,
            }
        )

    baseline = rows[0]
    ranked = sorted(
        rows,
        key=lambda row: (
            row.get("institutional_score", -1),
            row["passes_any_regime_gate"],
            row["portfolio_return"],
            row["portfolio_sharpe"],
            row["contrarian_pnl"],
        ),
        reverse=True,
    )

    rows_to_validate = []
    for row in ranked:
        if row["variant"] == "baseline":
            continue
        rows_to_validate.append(row)
        if len(rows_to_validate) == validator_top_n:
            break

    for row in rows_to_validate:
        slot = _build_variant_slot(base_contrarian, row["params"])
        engine = _build_variant_engine(base_engine, slot)
        report = InstitutionalValidator().validate(engine, datasets)
        row["institutional_score"] = report.score
        row["institutional_total_tests"] = report.total_tests
        row["institutional_regime_verdict"] = report.verdicts.get("all_strats_profitable_1+_regime")

    ranked = sorted(
        rows,
        key=lambda row: (
            row.get("institutional_score", -1),
            row["passes_any_regime_gate"],
            row["portfolio_return"],
            row["portfolio_sharpe"],
            row["contrarian_pnl"],
        ),
        reverse=True,
    )

    for row in ranked:
        row["delta_return_vs_baseline"] = row["portfolio_return"] - baseline["portfolio_return"]
        row["delta_sharpe_vs_baseline"] = row["portfolio_sharpe"] - baseline["portfolio_sharpe"]

    best = ranked[0]
    return {
        "baseline": baseline,
        "best_variant": best,
        "top_variants": ranked[:top_n],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune contrarian_asym on cached validated data")
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--validator-top-n", type=int, default=3)
    parser.add_argument(
        "--grid-mode",
        choices=["fast", "full"],
        default="fast",
        help="Use a targeted fast search or the broader full grid",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--cache-max-age-hours", type=float, default=1.0)
    parser.add_argument(
        "--output",
        default="pipeline_output/contrarian_sweep_report.json",
        help="Where to write the JSON sweep report",
    )
    args = parser.parse_args()

    base_engine = PortfolioEngine.default()
    base_engine.capital = args.capital
    base_engine.data_days = args.days

    if args.no_cache:
        datasets = base_engine.load_data()
        cache_meta = {"used_cache": False, "cache_path": None}
    else:
        datasets, cache_path, cache_hit = load_or_build_datasets(
            base_engine,
            namespace="proceed_gate_slots",
            max_age_hours=args.cache_max_age_hours,
            force_refresh=args.force_refresh,
        )
        cache_meta = {"used_cache": cache_hit, "cache_path": str(cache_path)}

    payload = _variant_rows(
        base_engine,
        datasets,
        args.top_n,
        args.validator_top_n,
        args.grid_mode,
    )
    payload["cache"] = cache_meta
    payload["grid_mode"] = args.grid_mode

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
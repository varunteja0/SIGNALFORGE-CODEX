"""
SignalForge — Pipeline Runner
================================
Complete pipeline: Fetch data -> Compute features -> Evolve strategies
                   -> Assess liquidation risk -> Score everything

Runs entirely offline (public Binance API, no keys).
"""

import sys
import json
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.fetcher import DataFetcher, compute_features
from src.alpha_genome.evolution import AlphaGenomeEngine
from src.alpha_genome.gene import ALL_FEATURE_NAMES
from src.liquidation.oracle import LiquidationOracle
from src.liquidation.cascade import CascadeSimulator
from src.liquidation.protocols import SyntheticPositionGenerator
from src.backtest.engine import Backtester
from src.alpha_genome.gene import tree_from_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger("Pipeline")


def step_fetch_data(
    symbols: list[str],
    timeframe: str = "1h",
    days: int = 730,
) -> dict[str, pd.DataFrame]:
    """Step 1: Fetch historical data for all symbols."""
    print("\n" + "=" * 60)
    print("STEP 1: FETCHING MARKET DATA")
    print("=" * 60)

    fetcher = DataFetcher()
    data = {}

    for symbol in symbols:
        print(f"\n  Fetching {symbol} {timeframe} ({days} days)...")
        try:
            df = fetcher.fetch(symbol, timeframe, days)
            if not df.empty:
                df = compute_features(df)
                df = df.dropna()
                data[symbol] = df
                print(f"  OK: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

                # Check feature coverage
                available = [f for f in ALL_FEATURE_NAMES if f in df.columns]
                print(f"  Features: {len(available)}/{len(ALL_FEATURE_NAMES)}")
            else:
                print(f"  WARN: No data returned for {symbol}")
        except Exception as e:
            print(f"  ERROR: {e}")

    return data


def step_evolve(
    data: dict[str, pd.DataFrame],
    population_size: int = 200,
    max_generations: int = 50,
    timeframe: str = "1h",
) -> list:
    """Step 2: Evolve trading strategies on real data."""
    print("\n" + "=" * 60)
    print("STEP 2: ALPHA GENOME — EVOLVING STRATEGIES")
    print("=" * 60)

    engine = AlphaGenomeEngine(
        population_size=population_size,
        max_generations=max_generations,
        tournament_size=5,
        crossover_rate=0.7,
        mutation_rate=0.2,
        elitism_count=max(2, population_size // 20),
        max_tree_depth=6,
        novelty_weight=0.2,
        walk_forward_splits=5,
        min_trades=15,
        commission_pct=0.001,
        slippage_pct=0.0005,
        output_dir="evolved_strategies",
    )

    all_strategies = []

    for symbol, df in data.items():
        print(f"\n  Evolving on {symbol} ({len(df)} bars)...")

        if len(df) < 500:
            print(f"  SKIP: Need >= 500 bars, got {len(df)}")
            continue

        def progress(gen, total, stats):
            if gen % 5 == 0 or gen == total - 1:
                print(
                    f"    Gen {gen:3d}/{total} | "
                    f"Best={stats.best_fitness:.4f} "
                    f"Sharpe={stats.best_sharpe:.2f} "
                    f"Valid={stats.valid_count}/{stats.population_size} "
                    f"Diversity={stats.diversity:.2f}"
                )

        try:
            strategies = engine.evolve(
                df,
                symbol=symbol,
                timeframe=timeframe,
                progress_callback=progress,
            )

            for s in strategies:
                s.symbol = symbol
                s.timeframe = timeframe

            all_strategies.extend(strategies)
            print(f"  Found {len(strategies)} valid strategies for {symbol}")

        except Exception as e:
            print(f"  ERROR during evolution: {e}")
            logger.exception(f"Evolution failed for {symbol}")

    print(f"\n  TOTAL: {len(all_strategies)} strategies discovered")
    return all_strategies


def step_backtest(
    strategies: list,
    data: dict[str, pd.DataFrame],
) -> list[dict]:
    """Step 3: Backtest each discovered strategy with full simulation."""
    print("\n" + "=" * 60)
    print("STEP 3: BACKTESTING STRATEGIES")
    print("=" * 60)

    backtester = Backtester(
        initial_capital=10000,
        commission_pct=0.001,
        slippage_pct=0.0005,
    )

    results = []

    for strat in strategies:
        df = data.get(strat.symbol)
        if df is None or df.empty:
            continue

        try:
            tree = tree_from_dict(strat.tree_dict)

            def signal_func(data_df, _tree=tree):
                signals = _tree.evaluate(data_df)
                return signals.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

            result = backtester.run(df, signal_func)
            mc = backtester.monte_carlo(result)

            results.append({
                "name": strat.name,
                "symbol": strat.symbol,
                "timeframe": strat.timeframe,
                "oos_sharpe": strat.fitness.oos_sharpe,
                "backtest_sharpe": result.sharpe_ratio,
                "backtest_return": result.total_return,
                "backtest_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "total_trades": result.total_trades,
                "mc_prob_profit": mc.get("probability_of_profit", 0),
                "mc_median_return": mc.get("median_return", 0),
                "novelty": strat.novelty_score,
                "formula": strat.formula[:100],
            })

            color = "+" if result.total_return > 0 else ""
            print(
                f"  {strat.name} ({strat.symbol}): "
                f"Return={color}{result.total_return:.1%} "
                f"Sharpe={result.sharpe_ratio:.2f} "
                f"DD={result.max_drawdown:.1%} "
                f"MC-Profit={mc.get('probability_of_profit', 0):.0%}"
            )

        except Exception as e:
            print(f"  ERROR backtesting {strat.name}: {e}")

    return results


def step_liquidation(symbols: list[str], data: dict[str, pd.DataFrame]) -> dict:
    """Step 4: Run liquidation risk analysis."""
    print("\n" + "=" * 60)
    print("STEP 4: LIQUIDATION ORACLE")
    print("=" * 60)

    oracle = LiquidationOracle(
        use_synthetic=True,
        synthetic_tvl=5_000_000_000,
        price_impact_bps=5.0,
    )

    risk_report = {}

    for symbol in symbols:
        df = data.get(symbol)
        if df is None or df.empty:
            continue

        asset = symbol.split("/")[0]
        current_price = float(df["close"].iloc[-1])

        print(f"\n  {symbol} @ ${current_price:,.2f}")

        risk = oracle.assess_risk(asset, current_price)
        signals = oracle.generate_signals(asset, current_price)

        risk_report[symbol] = {
            "price": current_price,
            "risk_score": risk.risk_score,
            "recommendation": risk.recommendation,
            "nearest_cliff_pct": risk.nearest_cliff_pct,
            "at_risk_usd": risk.total_at_risk_usd,
            "cascade_severity": risk.cascade_severity,
            "amplification": risk.expected_amplification,
            "signals": len(signals),
        }

        rec_color = (
            "!!" if risk.recommendation == "AVOID"
            else "!" if risk.recommendation == "CAUTIOUS"
            else ""
        )
        print(
            f"    Risk: {risk.risk_score:.0f}/100 {rec_color}"
            f"| Rec: {risk.recommendation} "
            f"| Cliff: {risk.nearest_cliff_pct:.1f}% "
            f"| Signals: {len(signals)}"
        )

        for sig in signals[:2]:
            d = "LONG" if sig.direction == 1 else "SHORT"
            print(
                f"    Signal: {sig.signal_type} {d} "
                f"entry=${sig.entry_price:,.0f} "
                f"target=${sig.target_price:,.0f} "
                f"conf={sig.confidence:.0%}"
            )

    return risk_report


def step_save_report(
    strategies: list,
    backtest_results: list[dict],
    risk_report: dict,
    output_dir: str = "pipeline_output",
):
    """Step 5: Save complete report."""
    print("\n" + "=" * 60)
    print("STEP 5: SAVING REPORT")
    print("=" * 60)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save backtest results
    if backtest_results:
        bt_df = pd.DataFrame(backtest_results)
        bt_df.to_csv(out / "backtest_results.csv", index=False)
        print(f"  Backtest results: {out / 'backtest_results.csv'}")

    # Save risk report
    with open(out / "risk_report.json", "w") as f:
        json.dump(risk_report, f, indent=2, default=str)
    print(f"  Risk report: {out / 'risk_report.json'}")

    # Summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategies_discovered": len(strategies),
        "strategies_backtested": len(backtest_results),
        "profitable_strategies": sum(
            1 for r in backtest_results if r["backtest_return"] > 0
        ),
        "avg_sharpe": (
            np.mean([r["backtest_sharpe"] for r in backtest_results])
            if backtest_results else 0
        ),
        "best_strategy": (
            max(backtest_results, key=lambda r: r["backtest_sharpe"])["name"]
            if backtest_results else "none"
        ),
        "symbols_analyzed": list(risk_report.keys()),
    }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {out / 'summary.json'}")

    return summary


def run_pipeline(
    symbols: list[str] = None,
    timeframe: str = "1h",
    days: int = 730,
    population_size: int = 200,
    max_generations: int = 50,
):
    """Run the complete SignalForge pipeline."""
    symbols = symbols or ["BTC/USDT", "ETH/USDT"]

    print("=" * 60)
    print("SIGNALFORGE COMPLETE PIPELINE")
    print(f"Symbols: {symbols}")
    print(f"Timeframe: {timeframe}, Days: {days}")
    print(f"Evolution: pop={population_size}, gens={max_generations}")
    print("=" * 60)

    start = time.time()

    # Step 1: Fetch data
    data = step_fetch_data(symbols, timeframe, days)
    if not data:
        print("\nFATAL: No data fetched. Check internet connection.")
        return

    # Step 2: Evolve strategies
    strategies = step_evolve(data, population_size, max_generations, timeframe)

    # Step 3: Backtest strategies
    backtest_results = step_backtest(strategies, data) if strategies else []

    # Step 4: Liquidation analysis
    risk_report = step_liquidation(symbols, data)

    # Step 5: Save report
    summary = step_save_report(strategies, backtest_results, risk_report)

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Time: {elapsed / 60:.1f} minutes")
    print(f"Strategies: {summary['strategies_discovered']} discovered, "
          f"{summary['profitable_strategies']} profitable")
    print(f"Best: {summary['best_strategy']} (Sharpe={summary['avg_sharpe']:.2f} avg)")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SignalForge Pipeline")
    parser.add_argument(
        "--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"],
        help="Symbols to analyze",
    )
    parser.add_argument("--timeframe", default="1h", help="Timeframe")
    parser.add_argument("--days", type=int, default=730, help="Days of history")
    parser.add_argument("--pop", type=int, default=200, help="Population size")
    parser.add_argument("--gens", type=int, default=50, help="Max generations")
    args = parser.parse_args()

    run_pipeline(
        symbols=args.symbols,
        timeframe=args.timeframe,
        days=args.days,
        population_size=args.pop,
        max_generations=args.gens,
    )

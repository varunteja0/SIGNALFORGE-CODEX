"""
SignalForge — Run full pipeline on real data
=============================================
Fetch → Features → Evolve → Backtest → Liquidation → Report
"""
import sys
import json
import warnings
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline_run.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("RealPipeline")


def main():
    from src.data.fetcher import DataFetcher, compute_features
    from src.alpha_genome.evolution import AlphaGenomeEngine
    from src.alpha_genome.gene import tree_from_dict, FEATURE_NAMES
    from src.backtest.engine import Backtester
    from src.regime.detector import RegimeDetector
    from src.liquidation.oracle import LiquidationOracle
    from src.risk.manager import RiskManager

    symbols = ["BTC/USDT", "ETH/USDT"]
    timeframe = "1h"
    days = 365  # 1 year of hourly data

    # ── STEP 1: Fetch Real Data ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1: FETCHING REAL MARKET DATA")
    print("=" * 60)

    fetcher = DataFetcher()
    print(f"  Exchange: {fetcher.exchange_id}")

    data = {}
    for symbol in symbols:
        print(f"\n  Fetching {symbol} {timeframe}, {days} days...")
        df = fetcher.fetch(symbol, timeframe, days=days)
        if df.empty:
            print(f"  SKIP: No data for {symbol}")
            continue
        df = compute_features(df).dropna()
        data[symbol] = df
        features = [f for f in FEATURE_NAMES if f in df.columns]
        print(f"  OK: {len(df)} bars | {df.index[0].date()} to {df.index[-1].date()}")
        print(f"      Price: ${df['close'].iloc[-1]:,.2f}")
        print(f"      Features: {len(features)}/{len(FEATURE_NAMES)}")

    if not data:
        print("ERROR: No data fetched. Aborting.")
        return

    # ── STEP 2: Detect Market Regime ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: MARKET REGIME DETECTION")
    print("=" * 60)

    detector = RegimeDetector(n_regimes=4, lookback_days=100)
    for symbol, df in data.items():
        detector.fit(df)
        current_regime = detector.detect(df)
        regime_stats = detector.get_regime_stats(df)
        print(f"  {symbol}: {current_regime.name}")
        for regime, stats in regime_stats.items():
            print(f"    {regime}: avg_ret={stats['avg_return']:.4f} "
                  f"vol={stats['volatility']:.4f} freq={stats['pct_of_time']:.0%}")

    # ── STEP 3: Evolve Trading Strategies ────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: ALPHA GENOME — EVOLVING STRATEGIES")
    print("=" * 60)

    engine = AlphaGenomeEngine(
        population_size=150,
        max_generations=30,
        tournament_size=5,
        crossover_rate=0.7,
        mutation_rate=0.2,
        elitism_count=8,
        max_tree_depth=5,
        novelty_weight=0.2,
        walk_forward_splits=4,
        min_trades=20,
        commission_pct=0.001,
        slippage_pct=0.0005,
        output_dir="evolved_strategies",
    )

    all_strategies = []

    for symbol, df in data.items():
        print(f"\n  Evolving on {symbol} ({len(df)} bars)...")

        def progress(gen, total, stats):
            if gen % 5 == 0 or gen == total - 1:
                print(
                    f"    Gen {gen:3d}/{total} | "
                    f"Best={stats.best_fitness:.4f} "
                    f"Sharpe={stats.best_sharpe:.2f} "
                    f"Valid={stats.valid_count}/{stats.population_size} "
                    f"Diversity={stats.diversity:.2f}"
                )

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
        print(f"  >> {len(strategies)} strategies found for {symbol}")

    print(f"\n  TOTAL STRATEGIES: {len(all_strategies)}")

    # ── STEP 4: Backtest Each Strategy ───────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4: BACKTESTING STRATEGIES")
    print("=" * 60)

    backtester = Backtester(
        initial_capital=10000,
        commission_pct=0.001,
        slippage_pct=0.0005,
    )

    # Holding period must match the fitness evaluator (default=5 bars)
    HOLDING_PERIOD = 5

    results = []

    for strat in all_strategies:
        df = data.get(strat.symbol)
        if df is None:
            continue
        try:
            tree = tree_from_dict(strat.tree_dict)

            def signal_func(data_df, _tree=tree, hp=HOLDING_PERIOD):
                """Match the fitness evaluator's signal interpretation.

                The fitness evaluator uses z-score thresholding and samples
                every `holding_period` bars. We replicate that here so the
                backtest is consistent with what evolution actually tested.
                """
                import numpy as _np
                import pandas as _pd
                raw = _tree.evaluate(data_df)
                sig_std = raw.std()
                if sig_std < 1e-10:
                    discretized = raw.apply(_np.sign)
                else:
                    zscore = (raw - raw.mean()) / sig_std
                    discretized = _pd.Series(0.0, index=raw.index)
                    discretized[zscore > 0.5] = 1.0
                    discretized[zscore < -0.5] = -1.0
                # Only emit signals every hp bars (non-overlapping windows)
                result = _pd.Series(0, index=data_df.index)
                for idx in range(0, len(data_df), hp):
                    sig = discretized.iloc[idx]
                    if sig != 0:
                        result.iloc[idx] = int(sig)
                return result

            result = backtester.run(
                df, signal_func,
                position_size_pct=0.10,       # 10% position (fitness assumes 100%)
                stop_loss_atr=5.0,            # Wider stops — fitness uses none
                take_profit_atr=8.0,
                max_holding_bars=HOLDING_PERIOD,  # Match fitness holding period
            )
            mc = backtester.monte_carlo(result, n_simulations=500)

            r = {
                "name": strat.name,
                "symbol": strat.symbol,
                "oos_sharpe": strat.fitness.oos_sharpe,
                "bt_return": result.total_return,
                "bt_sharpe": result.sharpe_ratio,
                "bt_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "total_trades": result.total_trades,
                "mc_profit_prob": mc.get("probability_of_profit", 0),
                "novelty": strat.novelty_score,
                "formula": strat.formula[:80],
            }
            results.append(r)

            sign = "+" if result.total_return > 0 else ""
            print(
                f"  {strat.name} ({strat.symbol}): "
                f"Return={sign}{result.total_return:.1%} "
                f"Sharpe={result.sharpe_ratio:.2f} "
                f"DD={result.max_drawdown:.1%} "
                f"MC={mc.get('probability_of_profit', 0):.0%} "
                f"Trades={result.total_trades}"
            )
        except Exception as e:
            print(f"  ERROR: {strat.name}: {e}")

    # ── STEP 5: Liquidation Risk ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5: LIQUIDATION ORACLE")
    print("=" * 60)

    oracle = LiquidationOracle(
        use_synthetic=True,
        synthetic_tvl=5_000_000_000,
        price_impact_bps=5.0,
    )

    for symbol, df in data.items():
        asset = symbol.split("/")[0]
        price = float(df["close"].iloc[-1])

        risk = oracle.assess_risk(asset, price)
        signals = oracle.generate_signals(asset, price)
        print(f"  {symbol} @ ${price:,.2f}")
        print(f"    Risk: {risk.risk_score}/100 ({risk.recommendation})")
        print(f"    Nearest cliff: {risk.nearest_cliff_pct:.1f}% away")
        print(f"    Cascade amp: {risk.expected_amplification:.2f}x")
        print(f"    Signals: {len(signals)}")
        for sig in signals[:3]:
            direction = "LONG" if sig.direction == 1 else "SHORT"
            print(f"      {direction} @ ${sig.entry_price:,.2f} | {sig.reasoning}")

    # ── STEP 6: Risk Check ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6: RISK MANAGEMENT")
    print("=" * 60)

    from src.risk.manager import RiskLimits
    risk_mgr = RiskManager(capital=10000, limits=RiskLimits(max_position_pct=0.1, max_drawdown_pct=0.15, max_daily_loss_pct=0.03))
    status = risk_mgr.get_status()
    print(f"  Capital: ${status['capital']:,.2f}")
    print(f"  Open positions: {status['open_positions']}")
    print(f"  Daily PnL: ${status['daily_pnl']:,.2f}")
    print(f"  Drawdown: {status['drawdown']:.1%}")

    # ── SUMMARY ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)

    print(f"\n  Data: {sum(len(df) for df in data.values())} total bars across {len(data)} symbols")
    print(f"  Strategies evolved: {len(all_strategies)}")
    print(f"  Strategies backtested: {len(results)}")

    if results:
        profitable = [r for r in results if r["bt_return"] > 0]
        print(f"  Profitable: {len(profitable)}/{len(results)}")

        best = max(results, key=lambda r: r["bt_sharpe"])
        print(f"\n  BEST STRATEGY:")
        print(f"    Name: {best['name']}")
        print(f"    Symbol: {best['symbol']}")
        print(f"    Return: {best['bt_return']:.1%}")
        print(f"    Sharpe: {best['bt_sharpe']:.2f}")
        print(f"    Drawdown: {best['bt_drawdown']:.1%}")
        print(f"    Trades: {best['total_trades']}")
        print(f"    MC P(Profit): {best['mc_profit_prob']:.0%}")
        print(f"    Formula: {best['formula']}")

        # Save results
        output_path = Path("pipeline_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")

    else:
        print("  No strategies passed validation — this is normal for first runs.")
        print("  The GP needs more data or more generations to find alpha.")


if __name__ == "__main__":
    main()

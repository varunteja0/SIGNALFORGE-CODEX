"""
SignalForge V2 — Paper Trading with Full Production Pipeline
===============================================================
Uses V2 fund manager with:
  - 130+ advanced features
  - Ensemble GP committee + liquidation oracle signals
  - HRP portfolio optimization
  - Drawdown bands + circuit breakers
  - Smart execution (TWAP, sqrt slippage, gap rejection)
  - SQLite persistence + hash-chained ledger
  - Trailing stops + regime-adaptive sizing

No API keys. No real money. Runs on public exchange data.
"""

import sys
import json
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.fetcher import DataFetcher
from src.data.features import compute_all_features
from src.alpha_genome.evolution import AlphaGenomeEngine
from src.risk.manager import RiskLimits
from src.risk.advanced import DrawdownBand
from src.fund.manager_v2 import AutonomousFundManagerV2
from src.fund.database import Database
from src.regime.detector import RegimeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("paper_trading_v2.log"),
    ],
)
logger = logging.getLogger("PaperTraderV2")


def run_paper_trading_v2(
    symbols=None,
    initial_capital=10000,
    interval_seconds=300,
    max_iterations=0,
):
    """Run V2 paper trading loop.

    Full pipeline: 130+ features → GP+ensemble signals → HRP weights →
    drawdown bands → circuit breakers → smart execution → SQLite + ledger.
    """
    symbols = symbols or ["BTC/USDT", "ETH/USDT"]

    print("=" * 60)
    print("SIGNALFORGE V2 — PAPER TRADING (PRODUCTION)")
    print(f"Symbols: {symbols}")
    print(f"Capital: ${initial_capital:,.2f}")
    print(f"Interval: {interval_seconds}s")
    print("Features: 130+ advanced | Exec: TWAP + sqrt slippage")
    print("Risk: Drawdown bands + circuit breakers | DB: SQLite WAL")
    print("=" * 60)

    # ─── Initialize V2 fund manager ───
    fund = AutonomousFundManagerV2(
        initial_capital=initial_capital,
        risk_limits=RiskLimits(
            max_position_pct=0.03,
            max_drawdown_pct=0.15,
            max_daily_loss_pct=0.05,
            max_open_positions=5,
        ),
        ledger_path="fund_data/paper_ledger_v2.json",
        db_path="fund_data/paper_v2.db",
        portfolio_method="hrp",
        drawdown_bands=DrawdownBand(
            yellow_pct=0.05,
            orange_pct=0.10,
            red_pct=0.15,
            black_pct=0.20,
        ),
        max_slippage_bps=50,
    )

    # ─── Load evolved strategies ───
    ag_engine = AlphaGenomeEngine(output_dir="evolved_strategies")
    strategies = ag_engine.load_strategies()

    if strategies:
        fund.load_strategies(strategies)
        print(f"\nLoaded {len(strategies)} evolved strategies")
        print("Portfolio weights (HRP):")
        for name, w in sorted(fund.portfolio_weights.items(), key=lambda x: -x[1]):
            print(f"  {name:30s} {w:.1%}")
    else:
        print("\nNo evolved strategies found. Using liquidation oracle only.")
        print("Run 'python scripts/run_pipeline_v2.py' first to evolve strategies.")

    # ─── Pre-load market data with advanced features ───
    fetcher = DataFetcher()
    print("\nPre-loading market data with 130+ features...")
    market_data = {}

    for symbol in symbols:
        try:
            df = fetcher.fetch(symbol, "1h", days=90)
            if not df.empty:
                df = compute_all_features(df)
                df = df.dropna()
                market_data[symbol] = df
                n_feat = len([c for c in df.columns if c not in ['open','high','low','close','volume']])
                print(f"  {symbol}: {len(df)} bars, {n_feat} features")
        except Exception as e:
            print(f"  {symbol}: ERROR - {e}")

    if not market_data:
        print("\nFATAL: No market data available.")
        return

    # ─── Paper trading loop ───
    iteration = 0
    print(f"\nStarting V2 paper trading loop (Ctrl+C to stop)...\n")

    try:
        while max_iterations == 0 or iteration < max_iterations:
            iteration += 1
            print(f"\n{'─'*50}")
            print(f"V2 Iteration {iteration} [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
            print(f"{'─'*50}")

            current_prices = {}

            for symbol in symbols:
                try:
                    # Refresh data (incremental from cache)
                    df = fetcher.fetch(symbol, "1h", days=90)
                    if df.empty or len(df) < 100:
                        continue

                    # 130+ advanced features
                    df = compute_all_features(df)
                    df = df.dropna()
                    market_data[symbol] = df

                    price = float(df["close"].iloc[-1])
                    current_prices[symbol] = price

                    # Generate signals from V2 fund manager
                    candidates = fund.generate_signals(df, symbol, price)

                    if candidates:
                        print(
                            f"  {symbol} @ ${price:,.2f} "
                            f"(regime={fund._current_regime}): "
                            f"{len(candidates)} signal(s)"
                        )

                    # Process through full V2 pipeline:
                    # risk check → Kelly size → drawdown band multiplier →
                    # smart execution → record DB + ledger
                    executed = fund.process_and_execute(candidates)

                    for trade in executed:
                        d = "LONG" if trade["direction"] == 1 else "SHORT"
                        print(
                            f"  >> TRADE: {d} {trade['asset']} "
                            f"size={trade['size']:.6f} @ ${trade['price']:.2f} "
                            f"via {trade['algo']} (slip={trade['slippage_bps']:.1f}bps) "
                            f"[{trade['strategy_name']}]"
                        )

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            # Check exits (with trailing stops)
            closed = fund.check_exits(current_prices)
            for c in closed:
                pnl_sign = "+" if c["pnl"] >= 0 else ""
                print(
                    f"  >> CLOSED: {c['asset']} "
                    f"PnL={pnl_sign}${c['pnl']:.2f} ({c['return_pct']:.2%}) "
                    f"reason={c['reason']} [{c['strategy']}] "
                    f"slip={c['slippage_bps']:.1f}bps"
                )

            # Periodic rebalance (every 24 iterations)
            if iteration % 24 == 0:
                fund.rebalance()
                print("  >> Rebalanced portfolio weights")

            # Display status
            state = fund.get_state()
            ret_sign = "+" if state.total_return_pct >= 0 else ""
            band_marker = {
                "green": "●", "yellow": "◐", "orange": "◑", "red": "○", "black": "✖"
            }.get(state.drawdown_band, "?")

            print(
                f"\n  Capital=${state.capital:,.2f} "
                f"Return={ret_sign}{state.total_return_pct:.2%} "
                f"DD={state.drawdown_pct:.2%} {band_marker}{state.drawdown_band.upper()} "
                f"Pos={len(state.open_positions)} "
                f"Ledger={state.ledger_entries}{'✓' if state.ledger_verified else '✗'} "
                f"DB={state.db_trades}"
            )

            if state.tripped_breakers:
                print(f"  !! Breakers: {', '.join(state.tripped_breakers)}")
            if state.is_halted:
                print(f"  !! HALTED: {state.halt_reason}")

            exec_q = state.execution_quality
            if exec_q.get("total_executions", 0) > 0:
                print(f"  Exec quality: avg slip={exec_q['avg_slippage_bps']:.1f}bps over {exec_q['total_executions']} fills")

            # Wait for next iteration
            if max_iterations == 0 or iteration < max_iterations:
                print(f"  Waiting {interval_seconds}s...")
                time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n\nPaper trading V2 stopped.")

    # ─── Final report ───
    print("\n" + "=" * 60)
    print("V2 PAPER TRADING — FINAL REPORT")
    print("=" * 60)

    state = fund.get_state()
    print(f"  Capital: ${state.capital:,.2f}")
    print(f"  Total Return: {state.total_return_pct:+.2%}")
    print(f"  Drawdown: {state.drawdown_pct:.2%} ({state.drawdown_band.upper()})")
    print(f"  Ledger: {state.ledger_entries} entries ({'VERIFIED' if state.ledger_verified else 'TAMPERED'})")
    print(f"  DB Trades: {state.db_trades}")

    # Strategy attribution
    attr = fund.get_strategy_attribution()
    if not attr.empty:
        print("\n  Strategy Attribution:")
        for _, row in attr.iterrows():
            sign = "+" if row["total_pnl"] >= 0 else ""
            cb = " [BREAKER]" if row["breaker_tripped"] else ""
            print(
                f"    {row['strategy']:30s} "
                f"w={row['weight']:.1%} "
                f"PnL={sign}${row['total_pnl']:.2f} ({row['pnl_pct']:+.2%}) "
                f"decay={row['decay_score']:.0f}/100{cb}"
            )

    # Execution quality
    eq = fund.smart_exec.get_execution_quality()
    if eq["total_executions"] > 0:
        print(
            f"\n  Execution: {eq['total_executions']} fills, "
            f"avg slippage={eq['avg_slippage_bps']:.1f}bps, "
            f"total slippage={eq['total_slippage_bps']:.1f}bps"
        )

    # Save trade data to database and report
    db = fund.db
    perf = db.get_strategy_performance()
    if perf:
        print("\n  DB Strategy Performance:")
        for p in perf:
            total = p.get("total_trades", 0)
            wins = p.get("wins", 0)
            wr = f"{wins/total:.0%}" if total > 0 else "N/A"
            print(
                f"    {p['strategy_name']:30s} "
                f"Trades={total} WR={wr} "
                f"PnL=${p.get('total_pnl', 0):.2f} "
                f"Avg slip={p.get('avg_slippage', 0):.1f}bps"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SignalForge V2 Paper Trading")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--interval", type=int, default=300, help="Seconds between iterations")
    parser.add_argument("--iterations", type=int, default=0, help="0 = run forever")
    args = parser.parse_args()

    run_paper_trading_v2(
        symbols=args.symbols,
        initial_capital=args.capital,
        interval_seconds=args.interval,
        max_iterations=args.iterations,
    )

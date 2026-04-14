"""
SignalForge — Paper Trading Orchestrator
==========================================
Unified paper trading loop that combines:
  Layer 1: Alpha Genome evolved strategies
  Layer 2: Liquidation Oracle signals
  Layer 3: Fund Manager with hash-chained ledger

Runs as a continuous loop, trading on testnet or paper mode.
All trades are verified and recorded.
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
from src.alpha_genome.gene import tree_from_dict
from src.liquidation.oracle import LiquidationOracle
from src.risk.manager import RiskManager, RiskLimits, PositionRequest
from src.fund.manager import AutonomousFundManager
from src.fund.ledger import VerifiableLedger
from src.regime.detector import RegimeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("paper_trading.log"),
    ],
)
logger = logging.getLogger("PaperTrader")


class PaperExecutionEngine:
    """Simulates trade execution using real price data.

    Uses fetcher to get real-time-ish prices from Binance public API.
    No API keys needed, no real money at risk.
    """

    def __init__(self, fetcher: DataFetcher, slippage_pct: float = 0.0005):
        self.fetcher = fetcher
        self.slippage_pct = slippage_pct
        self.positions: dict[str, dict] = {}
        self.capital = 0.0
        self.trade_log: list[dict] = []

    def get_current_price(self, symbol: str) -> float:
        """Get latest price from cache/exchange."""
        try:
            df = self.fetcher.fetch(symbol, "1m", days=1)
            if not df.empty:
                return float(df["close"].iloc[-1])
        except Exception:
            pass

        # Fallback: use hourly
        try:
            df = self.fetcher.fetch(symbol, "1h", days=1)
            if not df.empty:
                return float(df["close"].iloc[-1])
        except Exception:
            pass

        return 0.0

    def execute_entry(
        self, symbol: str, direction: int, size: float,
        stop_loss: float = 0, take_profit: float = 0,
    ):
        """Simulate entry with slippage."""
        price = self.get_current_price(symbol)
        if price <= 0:
            return type("Result", (), {"success": False, "price": 0, "error": "No price"})()

        # Apply slippage
        slip = price * self.slippage_pct
        exec_price = price + slip if direction == 1 else price - slip

        self.positions[symbol] = {
            "direction": direction,
            "size": size,
            "entry_price": exec_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "opened_at": time.time(),
        }

        self.trade_log.append({
            "type": "entry",
            "symbol": symbol,
            "direction": direction,
            "size": size,
            "price": exec_price,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        logger.info(
            f"PAPER ENTRY: {'LONG' if direction==1 else 'SHORT'} "
            f"{size:.6f} {symbol} @ ${exec_price:,.2f}"
        )

        return type("Result", (), {"success": True, "price": exec_price})()

    def execute_exit(self, symbol: str, size: float, direction: int):
        """Simulate exit with slippage."""
        price = self.get_current_price(symbol)
        if price <= 0:
            return type("Result", (), {"success": False, "price": 0})()

        slip = price * self.slippage_pct
        exec_price = price - slip if direction == 1 else price + slip

        pos = self.positions.pop(symbol, None)
        pnl = 0.0
        if pos:
            if pos["direction"] == 1:
                pnl = (exec_price - pos["entry_price"]) * pos["size"]
            else:
                pnl = (pos["entry_price"] - exec_price) * pos["size"]
            self.capital += pnl

        self.trade_log.append({
            "type": "exit",
            "symbol": symbol,
            "direction": direction,
            "size": size,
            "price": exec_price,
            "pnl": pnl,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        logger.info(
            f"PAPER EXIT: {symbol} @ ${exec_price:,.2f} PnL=${pnl:,.2f}"
        )

        return type("Result", (), {"success": True, "price": exec_price})()


def run_paper_trading(
    symbols: list[str] = None,
    initial_capital: float = 10000,
    interval_seconds: int = 300,
    max_iterations: int = 0,
):
    """Run the paper trading loop.

    Args:
        symbols: Trading symbols
        initial_capital: Starting capital
        interval_seconds: Seconds between iterations (default 5 min)
        max_iterations: 0 = run forever
    """
    symbols = symbols or ["BTC/USDT", "ETH/USDT"]

    print("=" * 60)
    print("SIGNALFORGE PAPER TRADING")
    print(f"Symbols: {symbols}")
    print(f"Capital: ${initial_capital:,.2f}")
    print(f"Interval: {interval_seconds}s")
    print("=" * 60)

    # Initialize components
    fetcher = DataFetcher()
    paper_exec = PaperExecutionEngine(fetcher, slippage_pct=0.0005)
    paper_exec.capital = initial_capital

    fund = AutonomousFundManager(
        initial_capital=initial_capital,
        risk_limits=RiskLimits(
            max_position_pct=0.02,
            max_drawdown_pct=0.10,
            max_daily_loss_pct=0.03,
            max_open_positions=5,
        ),
        ledger_path="fund_data/paper_ledger.json",
    )

    regime_detector = RegimeDetector()

    # Load evolved strategies
    ag_engine = AlphaGenomeEngine(output_dir="evolved_strategies")
    strategies = ag_engine.load_strategies()
    if strategies:
        fund.load_strategies(strategies)
        print(f"\nLoaded {len(strategies)} evolved strategies")
    else:
        print("\nNo evolved strategies found. Using liquidation oracle only.")

    # Pre-fetch data for regime detection
    print("\nPre-loading market data...")
    market_data = {}
    for symbol in symbols:
        try:
            df = fetcher.fetch(symbol, "1h", days=90)
            if not df.empty:
                df = compute_features(df)
                market_data[symbol] = df
                print(f"  {symbol}: {len(df)} bars loaded")
        except Exception as e:
            print(f"  {symbol}: ERROR - {e}")

    if not market_data:
        print("\nFATAL: No market data available.")
        return

    # Fit regime detector on first symbol
    first_df = next(iter(market_data.values()))
    regime_detector.fit(first_df)
    current_regime = regime_detector.detect(first_df)
    print(f"\nCurrent regime: {current_regime.value}")

    # Trading loop
    iteration = 0
    print(f"\nStarting paper trading loop (Ctrl+C to stop)...\n")

    try:
        while max_iterations == 0 or iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} [{time.strftime('%H:%M:%S')}] ---")

            current_prices = {}

            for symbol in symbols:
                try:
                    # Refresh data (incremental from cache)
                    df = fetcher.fetch(symbol, "1h", days=90)
                    if df.empty or len(df) < 100:
                        continue

                    df = compute_features(df)
                    df = df.dropna()
                    market_data[symbol] = df

                    price = float(df["close"].iloc[-1])
                    current_prices[symbol] = price
                    asset = symbol.split("/")[0]

                    # Update regime
                    regime_detector.fit(df)
                    regime = regime_detector.detect(df)

                    # Generate signals from fund manager (both AG + Liq Oracle)
                    candidates = fund.generate_signals(df, symbol, price)

                    if candidates:
                        print(
                            f"  {symbol} @ ${price:,.2f} (regime={regime.value}): "
                            f"{len(candidates)} signal(s)"
                        )

                    # Process through risk management
                    approved = fund.process_signals(candidates)

                    # Execute approved trades
                    executed = fund.execute_trades(approved, paper_exec)
                    for trade in executed:
                        d = "LONG" if trade["direction"] == 1 else "SHORT"
                        print(
                            f"  >> TRADE: {d} {symbol} "
                            f"size={trade['approved_size']:.6f} "
                            f"via {trade['strategy_name']}"
                        )

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            # Check exits
            closed = fund.check_exits(current_prices, paper_exec)
            for c in closed:
                pnl_str = f"+${c['pnl']:.2f}" if c["pnl"] > 0 else f"-${abs(c['pnl']):.2f}"
                print(f"  >> CLOSED: {c['asset']} {pnl_str} ({c['reason']})")

            # Status
            state = fund.get_state()
            ret_sign = "+" if state.total_return_pct >= 0 else ""
            print(
                f"  Capital=${state.capital:,.2f} "
                f"Return={ret_sign}{state.total_return_pct:.2%} "
                f"DD={state.drawdown_pct:.2%} "
                f"Positions={len(state.open_positions)} "
                f"Ledger={state.ledger_entries} entries "
                f"({'VERIFIED' if state.ledger_verified else 'TAMPERED!'})"
            )

            if state.is_halted:
                print(f"  !! HALTED: {state.halt_reason}")

            # Wait for next iteration
            if max_iterations == 0 or iteration < max_iterations:
                print(f"  Waiting {interval_seconds}s...")
                time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n\nPaper trading stopped by user.")

    # Final report
    print("\n" + "=" * 60)
    print("PAPER TRADING FINAL REPORT")
    print("=" * 60)

    state = fund.get_state()
    print(f"  Capital: ${state.capital:,.2f}")
    print(f"  Total Return: {state.total_return_pct:+.2%}")
    print(f"  Drawdown: {state.drawdown_pct:.2%}")
    print(f"  Ledger Entries: {state.ledger_entries}")

    is_valid, error = fund.ledger.verify_chain()
    print(f"  Ledger Integrity: {'VERIFIED' if is_valid else f'FAILED: {error}'}")

    # Strategy attribution
    attr = fund.get_strategy_attribution()
    if not attr.empty:
        print("\n  Strategy Attribution:")
        for _, row in attr.iterrows():
            sign = "+" if row["total_pnl"] >= 0 else ""
            print(
                f"    {row['strategy']:30s} {row['type']:15s} "
                f"PnL={sign}${row['total_pnl']:.2f} ({row['pnl_pct']:+.2%})"
            )

    # Save trade log
    if paper_exec.trade_log:
        log_path = Path("fund_data/paper_trade_log.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(paper_exec.trade_log, f, indent=2, default=str)
        print(f"\n  Trade log saved to {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SignalForge Paper Trading")
    parser.add_argument(
        "--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"],
    )
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--interval", type=int, default=300, help="Seconds between iterations")
    parser.add_argument("--iterations", type=int, default=0, help="0 = run forever")
    args = parser.parse_args()

    run_paper_trading(
        symbols=args.symbols,
        initial_capital=args.capital,
        interval_seconds=args.interval,
        max_iterations=args.iterations,
    )

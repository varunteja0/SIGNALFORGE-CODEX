"""
Paper Trading Loop — Liquidation Reversal v2
================================================
Multi-symbol paper trading with advanced exit logic.

Features:
- Multiple symbols (BTC, ETH, SOL)
- Partial exits at VWAP (50%), full exit at EMA
- Trailing stop (activates after 1.5% profit)
- OI-based exit (OI rising = trap forming)
- Time exit (8 bar max hold)
- Portfolio-level risk (max 3 positions, correlation check)
- Verifiable ledger logging
- State persistence (survives restarts)

Run with:
    python scripts/paper_trade_live.py
    python scripts/paper_trade_live.py --symbols BTC/USDT ETH/USDT
    python scripts/paper_trade_live.py --capital 50000 --risk 0.015
    python scripts/paper_trade_live.py --once  # Single iteration

Stop with: Ctrl+C (saves state to disk)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies.paper_trader import PaperTrader, PaperTraderConfig
from src.strategies.liquidation_reversal import StrategyConfig

# Ensure log directory exists
Path("paper_trading_logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("paper_trading_logs/paper_trade.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Paper Trade: Liquidation Reversal v2")
    parser.add_argument(
        "--symbols", nargs="+",
        default=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        help="Symbols to trade",
    )
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital")
    parser.add_argument("--risk", type=float, default=0.01, help="Risk per trade (0.01 = 1%%)")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between checks (3600 = 1h)")
    parser.add_argument("--max-dd", type=float, default=0.10, help="Max drawdown before halt (0.10 = 10%%)")
    parser.add_argument("--once", action="store_true", help="Run one iteration and exit")
    args = parser.parse_args()

    trader_config = PaperTraderConfig(
        symbols=args.symbols,
        initial_capital=args.capital,
        base_risk_pct=args.risk,
        max_drawdown_pct=args.max_dd,
    )

    strategy_config = StrategyConfig(
        base_risk_pct=args.risk,
        max_risk_pct=min(args.risk * 2, 0.03),
    )

    trader = PaperTrader(config=trader_config, strategy_config=strategy_config)

    if args.once:
        logger.info("Running single iteration...")
        actions = trader.run_once()
        trader.print_summary()
    else:
        trader.run_loop(interval_seconds=args.interval)


if __name__ == "__main__":
    main()

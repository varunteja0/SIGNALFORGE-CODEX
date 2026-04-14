"""
Strategy 1 — Validate & Go
=============================
One script. One strategy. Real data.

Usage:
    python scripts/validate_strategy1.py                    # Full validation (BTC, 1 year)
    python scripts/validate_strategy1.py --days 180         # 6 months
    python scripts/validate_strategy1.py --symbol ETH/USDT  # Test on ETH
    python scripts/validate_strategy1.py --quick             # Quick backtest only
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies.liquidation_reversal import (
    LiquidationReversalStrategy,
    StrategyConfig,
)
from src.strategies.validator import StrategyValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate Strategy 1")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital")
    parser.add_argument("--quick", action="store_true", help="Quick backtest only (no full validation)")
    parser.add_argument("--force-fetch", action="store_true", help="Force re-fetch data")
    args = parser.parse_args()

    # Initialize strategy
    strategy = LiquidationReversalStrategy(StrategyConfig())

    # Initialize validator
    validator = StrategyValidator(
        strategy=strategy,
        initial_capital=args.capital,
    )

    if args.quick:
        # Quick mode — just backtest, no validation suite
        print(f"\nQuick backtest: {args.symbol} ({args.days} days)\n")
        df = validator.load_data(
            symbol=args.symbol, days=args.days, force_fetch=args.force_fetch
        )

        result = validator.backtest(df)

        print(f"Return:      {result.total_return:.2%}")
        print(f"Sharpe:      {result.sharpe_ratio:.2f}")
        print(f"Sortino:     {result.sortino_ratio:.2f}")
        print(f"Max DD:      {result.max_drawdown:.2%}")
        print(f"Win rate:    {result.win_rate:.1%}")
        print(f"Trades:      {result.total_trades}")
        print(f"Profit factor: {result.profit_factor:.2f}")

        if result.total_trades > 0:
            print(f"\nAvg win:     {result.avg_win:.2%}")
            print(f"Avg loss:    {result.avg_loss:.2%}")
            print(f"Best trade:  {result.best_trade:.2%}")
            print(f"Worst trade: {result.worst_trade:.2%}")

        # Signal distribution
        signals = strategy.generate_signals(df)
        n_long = (signals == 1).sum()
        n_short = (signals == -1).sum()
        n_flat = (signals == 0).sum()
        print(f"\nSignals: {n_long} long, {n_short} short, {n_flat} flat")

    else:
        # Full validation suite
        results = validator.run_full_validation(
            symbol=args.symbol,
            days=args.days,
            force_fetch=args.force_fetch,
        )

        # Save results
        output_path = Path("pipeline_output/strategy1_validation.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

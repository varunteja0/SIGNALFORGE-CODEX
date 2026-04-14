"""
Run Multi-Strategy Portfolio Engine
====================================
CLI entry point for the hedge fund portfolio system.

Usage:
    python scripts/run_portfolio.py backtest          # Full portfolio backtest
    python scripts/run_portfolio.py backtest --quick   # Quick (smaller data)
    python scripts/run_portfolio.py report             # Show last results
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_backtest(args):
    """Run full portfolio backtest."""
    from src.engine.portfolio_engine import PortfolioEngine

    logger.info("Initializing Multi-Strategy Portfolio Engine...")
    engine = PortfolioEngine.default()

    if args.quick:
        engine.data_days = 180

    if args.capital:
        engine.capital = args.capital

    # Load data
    logger.info("Loading data for all assets...")
    datasets = engine.load_data()
    logger.info(f"Loaded {len(datasets)} assets")

    # Backtest
    logger.info("Running portfolio backtest...")
    result = engine.backtest(datasets)

    # Report
    out_path = "pipeline_output/portfolio_report.txt"
    report = engine.report(result, out_path=out_path)
    print(report)

    # Save config
    engine.save_config()

    logger.info(f"Done. Report: {out_path}")


def cmd_report(args):
    """Show last report."""
    path = "pipeline_output/portfolio_report.txt"
    if os.path.exists(path):
        with open(path) as f:
            print(f.read())
    else:
        print(f"No report found at {path}. Run 'backtest' first.")


def main():
    parser = argparse.ArgumentParser(description="Multi-Strategy Portfolio Engine")
    sub = parser.add_subparsers(dest="command")

    bt = sub.add_parser("backtest", help="Run portfolio backtest")
    bt.add_argument("--quick", action="store_true", help="Quick mode (180 days)")
    bt.add_argument("--capital", type=float, default=None, help="Starting capital")
    bt.set_defaults(func=cmd_backtest)

    rp = sub.add_parser("report", help="Show last report")
    rp.set_defaults(func=cmd_report)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()

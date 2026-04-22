"""
Run portfolio backtests.

Supports two engines:
    - opportunity: broad-universe institutional opportunity allocator
    - slots: existing fixed-slot multi-strategy portfolio engine
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
    if args.engine == "opportunity":
        from src.engine.opportunity_engine import OpportunityEngine

        logger.info("Initializing Institutional Opportunity Engine...")
        engine = OpportunityEngine.default()
        if args.quick:
            engine.config.data_days = 180
        if args.capital:
            engine.config.initial_capital = args.capital

        logger.info("Loading enriched data for liquid universe...")
        datasets = engine.load_data()
        logger.info(f"Loaded {len(datasets)} assets")

        logger.info("Running opportunity-engine backtest...")
        result = engine.backtest(datasets)
        report = engine.report(result)
        out_path = "pipeline_output/opportunity_report.txt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report)
        print(report)
        logger.info(f"Done. Report: {out_path}")
        return

    from src.engine.portfolio_engine import PortfolioEngine

    logger.info("Initializing Multi-Strategy Portfolio Engine...")
    engine = PortfolioEngine.default()

    if args.quick:
        engine.data_days = 180

    if args.capital:
        engine.capital = args.capital

    logger.info("Loading data for all assets...")
    datasets = engine.load_data()
    logger.info(f"Loaded {len(datasets)} assets")

    logger.info("Running portfolio backtest...")
    result = engine.backtest(datasets)

    out_path = "pipeline_output/portfolio_report.txt"
    report = engine.report(result, out_path=out_path)
    print(report)

    engine.save_config()

    logger.info(f"Done. Report: {out_path}")


def cmd_report(args):
    """Show last report."""
    path = (
        "pipeline_output/opportunity_report.txt"
        if args.engine == "opportunity"
        else "pipeline_output/portfolio_report.txt"
    )
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
    bt.add_argument(
        "--engine",
        choices=["opportunity", "slots"],
        default="slots",
        help="Which engine to run",
    )
    bt.set_defaults(func=cmd_backtest)

    rp = sub.add_parser("report", help="Show last report")
    rp.add_argument(
        "--engine",
        choices=["opportunity", "slots"],
        default="slots",
        help="Which engine report to show",
    )
    rp.set_defaults(func=cmd_report)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()

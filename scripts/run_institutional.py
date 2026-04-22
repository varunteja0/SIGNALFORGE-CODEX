"""
Run Institutional Validation — Fund-Level Portfolio Analysis
=============================================================
Runs the complete institutional validation suite:

    1. Strategy correlation matrix + rolling correlation
    2. Asset correlation matrix
    3. Diversification ratio
    4. Regime-specific performance breakdown
    5. Capacity simulation (slippage vs position size)
    6. Final scorecard with pass/fail verdicts

Usage:
    python scripts/run_institutional.py                     # Slot engine
    python scripts/run_institutional.py --engine opportunity  # Opportunity engine
    python scripts/run_institutional.py --quick             # Quick mode
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


def main():
    parser = argparse.ArgumentParser(description="Institutional Validation Suite")
    parser.add_argument("--quick", action="store_true", help="Quick mode (180 days)")
    parser.add_argument("--capital", type=float, default=100_000, help="Capital ($)")
    parser.add_argument(
        "--engine",
        choices=["opportunity", "slots"],
        default="slots",
        help="Which engine to run",
    )
    args = parser.parse_args()

    if args.engine == "opportunity":
        from src.engine.opportunity_engine import OpportunityEngine

        logger.info("Initializing Institutional Opportunity Engine...")
        engine = OpportunityEngine.default()
        engine.config.initial_capital = args.capital
        if args.quick:
            engine.config.data_days = 180

        logger.info("Loading enriched data...")
        datasets = engine.load_data()
        logger.info(f"Loaded {len(datasets)} assets")

        logger.info("Running opportunity-engine institutional backtest...")
        result = engine.backtest(datasets)
        output = engine.report(result)
        out_path = "pipeline_output/institutional_opportunity_report.txt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(output)
        print(output)
        logger.info(f"Report saved to {out_path}")
        return

    from src.engine.portfolio_engine import PortfolioEngine
    from src.engine.institutional import InstitutionalValidator

    logger.info("Initializing Multi-Strategy Portfolio Engine...")
    engine = PortfolioEngine.default()
    engine.capital = args.capital
    if args.quick:
        engine.data_days = 180

    logger.info("Loading data...")
    datasets = engine.load_data()
    logger.info(f"Loaded {len(datasets)} assets")

    logger.info("Running institutional validation suite...")
    validator = InstitutionalValidator()
    report = validator.validate(engine, datasets)

    output = validator.format_report(report, engine)
    print(output)

    out_path = "pipeline_output/institutional_report.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(output)
    logger.info(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()

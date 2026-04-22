from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.proceed_gate import (
    ProceedThresholds,
    evaluate_default_slots_engine,
    format_proceed_decision,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the validated proceed gate")
    parser.add_argument("--quick", action="store_true", help="Quick mode (180 days)")
    parser.add_argument("--capital", type=float, default=10_000, help="Starting capital")
    parser.add_argument(
        "--output",
        default="pipeline_output/proceed_gate_report.json",
        help="Path to write the JSON gate report",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the enriched dataset cache for the proceed gate",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Rebuild the enriched dataset cache even if a fresh cache exists",
    )
    parser.add_argument(
        "--cache-max-age-hours",
        type=float,
        default=1.0,
        help="Maximum age for cached enriched datasets",
    )
    parser.add_argument("--min-return", type=float, default=0.08, help="Minimum total return")
    parser.add_argument("--min-sharpe", type=float, default=1.50, help="Minimum Sharpe")
    parser.add_argument(
        "--max-drawdown", type=float, default=0.08, help="Maximum drawdown allowed"
    )
    parser.add_argument(
        "--min-profit-factor", type=float, default=1.50, help="Minimum profit factor"
    )
    parser.add_argument(
        "--min-institutional-score",
        type=int,
        default=6,
        help="Minimum institutional score required to proceed",
    )
    args = parser.parse_args()

    thresholds = ProceedThresholds(
        min_total_return=args.min_return,
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown,
        min_profit_factor=args.min_profit_factor,
        min_institutional_score=args.min_institutional_score,
    )

    logger.info("Running proceed gate backtest and institutional validation...")
    decision, cache_meta = evaluate_default_slots_engine(
        capital=args.capital,
        data_days=180 if args.quick else 365,
        thresholds=thresholds,
        use_cache=not args.no_cache,
        cache_max_age_hours=args.cache_max_age_hours,
        force_refresh=args.force_refresh,
    )
    if cache_meta["cache_path"]:
        source = "cache" if cache_meta["used_cache"] else "rebuilt data"
        logger.info("Proceed gate dataset source: %s (%s)", source, cache_meta["cache_path"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(decision.to_dict(), handle, indent=2)

    print(format_proceed_decision(decision))
    logger.info("JSON report saved to %s", args.output)


if __name__ == "__main__":
    main()
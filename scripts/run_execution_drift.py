#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ops.execution_drift import (
    ExecutionDriftThresholds,
    build_execution_drift_report,
    format_execution_drift_report,
    write_execution_drift_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper-layer execution drift engine")
    parser.add_argument("--base-dir", type=Path, default=Path("fund_data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fund_data/execution_drift_status.json"),
        help="Path to write the execution drift report",
    )
    parser.add_argument(
        "--min-compared",
        type=int,
        default=5,
        help="Minimum executed comparisons required before drift is capital-ready",
    )
    parser.add_argument(
        "--min-shadow-compared",
        type=int,
        default=5,
        help="Minimum shadow comparisons required before drift is capital-ready",
    )
    args = parser.parse_args()

    thresholds = ExecutionDriftThresholds(
        min_compared_trades_for_capital=args.min_compared,
        min_shadow_compared_trades_for_capital=args.min_shadow_compared,
    )
    report = build_execution_drift_report(args.base_dir, thresholds)
    write_execution_drift_report(report, args.output)
    print(format_execution_drift_report(report))


if __name__ == "__main__":
    main()
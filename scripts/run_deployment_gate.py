#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from src.ops.deployment_gate import (
    DeploymentGateThresholds,
    build_deployment_gate_report,
    format_deployment_gate_report,
    write_deployment_gate_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the operational capital deployment gate")
    parser.add_argument("--base-dir", type=Path, default=Path("fund_data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fund_data/deployment_gate_status.json"),
        help="Path to write the deployment gate report",
    )
    parser.add_argument(
        "--min-shadow-probation",
        type=int,
        default=3,
        help="Minimum compared shadow trades required before probation live",
    )
    parser.add_argument(
        "--min-shadow-live",
        type=int,
        default=10,
        help="Minimum compared shadow trades required before full live",
    )
    args = parser.parse_args()

    thresholds = DeploymentGateThresholds(
        min_shadow_compared_trades_for_probation=args.min_shadow_probation,
        min_shadow_compared_trades_for_live=args.min_shadow_live,
    )
    report = build_deployment_gate_report(args.base_dir, thresholds)
    write_deployment_gate_report(report, args.output)
    print(format_deployment_gate_report(report))


if __name__ == "__main__":
    main()
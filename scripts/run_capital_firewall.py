#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ops.capital_firewall import (
    CapitalFirewallThresholds,
    build_capital_firewall_report,
    format_capital_firewall_report,
    write_capital_firewall_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live capital allocation firewall")
    parser.add_argument("--base-dir", type=Path, default=Path("fund_data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fund_data/capital_firewall_status.json"),
        help="Path to write the capital firewall report",
    )
    parser.add_argument(
        "--operating-mode",
        default="paper",
        choices=["paper", "shadow_live", "probation_live", "live"],
        help="Runtime mode to evaluate against the firewall",
    )
    parser.add_argument(
        "--configured-max-exposure",
        type=float,
        default=0.0,
        help="Configured portfolio max exposure percent (decimal)",
    )
    parser.add_argument(
        "--configured-max-per-trade",
        type=float,
        default=0.0,
        help="Configured per-trade max exposure percent (decimal)",
    )
    args = parser.parse_args()

    report = build_capital_firewall_report(
        args.base_dir,
        operating_mode=args.operating_mode,
        configured_max_total_exposure_pct=args.configured_max_exposure,
        configured_max_per_trade_pct=args.configured_max_per_trade,
        thresholds=CapitalFirewallThresholds(),
    )
    write_capital_firewall_report(report, args.output)
    print(format_capital_firewall_report(report))


if __name__ == "__main__":
    main()
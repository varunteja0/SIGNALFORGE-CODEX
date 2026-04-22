#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from src.ops.survivability_lab import (
    append_market_snapshot_history,
    build_survivability_report,
    format_survivability_report,
    write_survivability_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the pre-live survivability report.")
    parser.add_argument("--base-dir", type=Path, default=Path("fund_data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fund_data/survivability_status.json"),
    )
    parser.add_argument(
        "--append-snapshot-history",
        action="store_true",
        help="Append the current market snapshot into the novelty history before scoring.",
    )
    args = parser.parse_args()

    if args.append_snapshot_history:
        append_market_snapshot_history(args.base_dir)

    report = build_survivability_report(args.base_dir)
    write_survivability_report(report, args.output)
    print(format_survivability_report(report))


if __name__ == "__main__":
    main()
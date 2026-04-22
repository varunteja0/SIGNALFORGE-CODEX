#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from src.ops.failure_drills import (
    format_failure_drill_report,
    run_failure_drills,
    write_failure_drill_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic production failure drills.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fund_data/failure_drill_report.json"),
    )
    args = parser.parse_args()

    report = run_failure_drills()
    write_failure_drill_report(report, args.output)
    print(format_failure_drill_report(report))


if __name__ == "__main__":
    main()
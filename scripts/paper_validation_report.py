#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from src.ops.paper_validation import (
    build_paper_validation_report,
    format_paper_validation_report,
    write_paper_validation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the strict paper-trading validation report.")
    parser.add_argument("--base-dir", default="fund_data", help="Directory containing live state and adaptive cycle artifacts")
    parser.add_argument(
        "--output",
        default="fund_data/paper_validation_status.json",
        help="Where to write the JSON validation report",
    )
    args = parser.parse_args()

    report = build_paper_validation_report(Path(args.base_dir))
    write_paper_validation_report(report, Path(args.output))
    print(format_paper_validation_report(report))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from src.ops.drift_intelligence import (
    build_drift_intelligence_report,
    format_drift_intelligence_report,
    write_drift_intelligence_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the production drift intelligence report.")
    parser.add_argument("--base-dir", type=Path, default=Path("fund_data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fund_data/drift_intelligence_status.json"),
    )
    args = parser.parse_args()

    report = build_drift_intelligence_report(args.base_dir)
    write_drift_intelligence_report(report, args.output)
    print(format_drift_intelligence_report(report))


if __name__ == "__main__":
    main()
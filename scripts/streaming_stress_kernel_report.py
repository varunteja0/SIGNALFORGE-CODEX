#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from src.ops.streaming_stress_kernel import (
    build_streaming_stress_kernel_report,
    format_streaming_stress_kernel_report,
    write_streaming_stress_kernel_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the streaming stress kernel report.")
    parser.add_argument("--base-dir", type=Path, default=Path("fund_data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fund_data/streaming_stress_kernel_status.json"),
    )
    args = parser.parse_args()

    report = build_streaming_stress_kernel_report(args.base_dir)
    write_streaming_stress_kernel_report(report, args.output)
    print(format_streaming_stress_kernel_report(report))


if __name__ == "__main__":
    main()
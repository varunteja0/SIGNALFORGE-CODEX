#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from src.ops.drift_intelligence import build_drift_intelligence_report, write_drift_intelligence_report
from src.ops.failure_drills import run_failure_drills, write_failure_drill_report
from src.ops.production_bridge import (
    build_production_certification_report,
    format_production_certification_report,
    write_production_certification_report,
)
from src.ops.survivability_lab import build_survivability_report, write_survivability_report


def _require_observed_stress_artifacts(base_dir: Path) -> None:
    required = [
        base_dir / "streaming_stress_kernel_status.json",
        base_dir / "stress_field_state.json",
    ]
    missing = [path for path in required if not path.exists()]
    if not missing:
        return
    raise SystemExit(
        "Production certification now samples persisted stress artifacts only. Missing: "
        + ", ".join(str(path) for path in missing)
        + ". Refresh the live/paper runtime once so it persists a matched kernel and field snapshot before running certification."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the production certification report.")
    parser.add_argument("--base-dir", type=Path, default=Path("fund_data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fund_data/production_certification_status.json"),
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=Path("fund_data/production_certification_history.jsonl"),
    )
    parser.add_argument(
        "--failure-drill-output",
        type=Path,
        default=Path("fund_data/failure_drill_report.json"),
    )
    parser.add_argument(
        "--drift-output",
        type=Path,
        default=Path("fund_data/drift_intelligence_status.json"),
    )
    parser.add_argument(
        "--survivability-output",
        type=Path,
        default=Path("fund_data/survivability_status.json"),
    )
    parser.add_argument(
        "--stress-kernel-output",
        type=Path,
        default=Path("fund_data/streaming_stress_kernel_status.json"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    _require_observed_stress_artifacts(args.base_dir)

    failure_drill_report = run_failure_drills()
    write_failure_drill_report(failure_drill_report, args.failure_drill_output)
    drift_report = build_drift_intelligence_report(args.base_dir)
    write_drift_intelligence_report(drift_report, args.drift_output)
    survivability_report = build_survivability_report(args.base_dir)
    write_survivability_report(survivability_report, args.survivability_output)
    report = build_production_certification_report(args.base_dir)
    write_production_certification_report(report, args.output, args.history_output)
    print(format_production_certification_report(report))



if __name__ == "__main__":
    main()

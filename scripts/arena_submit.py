"""Research and export a frozen-data arena submission."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arena import (
    build_default_candidate_specs,
    build_refined_candidate_specs,
    load_frozen_bars,
    run_research,
    write_research_report,
    write_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arena-root",
        default="~/SignalForge-arena",
        help="Arena root containing frozen data and judge/",
    )
    parser.add_argument(
        "--report-path",
        default="fund_data/arena_research_report.json",
        help="Where to write the research report.",
    )
    parser.add_argument(
        "--grid",
        choices=("default", "refined"),
        default="refined",
        help="Candidate grid to evaluate on frozen data.",
    )
    parser.add_argument(
        "--submission-name",
        default="codex-arena-dyn-v1",
        help="Engine name used in submission.json.",
    )
    parser.add_argument(
        "--submission-dir",
        help="Override submission directory. Defaults to <arena-root>/submissions/<submission-name>.",
    )
    parser.add_argument(
        "--write-submission",
        action="store_true",
        help="Write signals and submission.json.",
    )
    parser.add_argument(
        "--include-oos",
        action="store_true",
        help="Include OOS metrics in the local report output.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        help="Override honest total trial count written into reports/submission metadata.",
    )
    parser.add_argument(
        "--max-weight-cap",
        type=float,
        help="If set, pick the highest-ranked candidate whose max portfolio weight is <= this cap.",
    )
    args = parser.parse_args()

    arena_root = Path(args.arena_root).expanduser().resolve()
    submission_dir = (
        Path(args.submission_dir).expanduser().resolve()
        if args.submission_dir
        else arena_root / "submissions" / args.submission_name
    )
    specs = build_refined_candidate_specs() if args.grid == "refined" else build_default_candidate_specs()
    declared_trials = int(args.n_trials) if args.n_trials is not None else len(specs)

    bars = load_frozen_bars(arena_root)
    best, ranked = run_research(bars, specs=specs, n_trials=declared_trials)
    if args.max_weight_cap is not None:
        capped = [
            result
            for result in ranked
            if max(result.weights.values()) <= float(args.max_weight_cap)
        ]
        if not capped:
            raise SystemExit(
                f"No candidate satisfied max weight cap {args.max_weight_cap:.3f}"
            )
        best = capped[0]
    report_path = write_research_report(
        args.report_path,
        best=best,
        ranked_results=ranked,
        include_oos=args.include_oos,
        n_trials=declared_trials,
    )

    print(f"Selected: {best.spec.name}")
    print(
        "IS Sharpe={:.2f} IS DD={:.1%} Turnover={:.3f}".format(
            best.portfolio_metrics["is_sharpe"],
            best.portfolio_metrics["is_max_drawdown"],
            best.portfolio_metrics["mean_turnover"],
        )
    )
    if args.include_oos:
        print(
            "OOS Sharpe={:.2f} OOS DD={:.1%} OOS Ret={:.1%}".format(
                best.portfolio_metrics["oos_sharpe"],
                best.portfolio_metrics["oos_max_drawdown"],
                best.portfolio_metrics["oos_total_return"],
            )
        )
    print(f"Weights: {json.dumps(best.weights, sort_keys=True)}")
    print(f"Research report: {report_path}")

    if args.write_submission:
        notes = (
            f"Selected {best.spec.name} ({best.spec.family}) from {declared_trials} honest trials; "
            f"IS Sharpe {best.portfolio_metrics['is_sharpe']:.2f}; "
            f"IS MaxDD {best.portfolio_metrics['is_max_drawdown']:.1%}."
        )
        out = write_submission(
            submission_dir,
            engine_name=args.submission_name,
            best=best,
            notes=notes,
            n_trials=declared_trials,
        )
        print(f"Submission written: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
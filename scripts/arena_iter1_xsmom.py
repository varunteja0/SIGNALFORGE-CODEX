"""Iter 1: cross-sectional momentum (xsmom) with pre-registered gate.

Grid searches the xsmom family, evaluates every candidate on IS only,
applies gate G1 (IS fold Sharpe mean >= 1.17) and, if it passes, writes
a submission and asks the arena judge to evaluate it. If the gate
fails, the run is archived and no submission is written.
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from src.arena.engine import (
    CandidateResult,
    CandidateSpec,
    build_feature_cache,
    evaluate_candidate,
    load_frozen_bars,
    write_submission,
)

G1_FOLD_SHARPE_THRESHOLD = 1.17  # current best single on the board (dyn-v1 IS fold mean)
G1_COST_2X_SHARPE_THRESHOLD = 0.0
DEFAULT_SUBMISSION_DIR = (
    "/Users/varunteja/SignalForge-arena/submissions/codex-arena-xsmom-v1"
)


def build_xsmom_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for lookback, smooth, vol_window, target_vol, entry_thr in product(
        (48, 72, 96, 120, 144, 168, 240),
        (12, 24, 48),
        (48, 72, 168),
        (0.004, 0.005, 0.006),
        (0.0, 0.25, 0.5),
    ):
        name = (
            f"xsmom_lb{lookback}_sm{smooth}_vw{vol_window}"
            f"_tv{int(round(target_vol * 1000)):03d}"
            f"_et{int(round(entry_thr * 100)):02d}"
        )
        specs.append(
            CandidateSpec(
                name=name,
                family="xsmom",
                params={
                    "lookback": lookback,
                    "smooth": smooth,
                    "vol_window": vol_window,
                    "target_vol": target_vol,
                    "entry_threshold": entry_thr,
                    "max_scale": 1.0,
                },
                notes="Cross-sectional momentum, dollar-neutral rank weighting.",
            )
        )
    return specs


def summarize(result: CandidateResult) -> dict:
    m = result.portfolio_metrics
    return {
        "name": result.spec.name,
        "family": result.spec.family,
        "params": result.spec.params,
        "score": result.score,
        "is_sharpe": m["is_sharpe"],
        "is_fold_sharpe_mean": m["is_fold_sharpe_mean"],
        "is_fold_sharpe_min": m["is_fold_sharpe_min"],
        "is_fold_positive_frac": m["is_fold_positive_frac"],
        "is_sharpe_cost_1p5x": m["is_sharpe_cost_1p5x"],
        "is_sharpe_cost_2x": m["is_sharpe_cost_2x"],
        "is_max_drawdown": m["is_max_drawdown"],
        "is_total_return": m["is_total_return"],
        "mean_turnover": m["mean_turnover"],
        "max_weight": m["max_weight"],
        "weights": result.weights,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena-root", default="/Users/varunteja/SignalForge-arena")
    ap.add_argument(
        "--report-path", default="fund_data/arena_iter1_xsmom_report.json"
    )
    ap.add_argument("--submission-dir", default=DEFAULT_SUBMISSION_DIR)
    ap.add_argument("--engine-name", default="codex-arena-xsmom-v1")
    ap.add_argument(
        "--max-symbol-weight",
        type=float,
        default=0.50,
        help="Cap on per-symbol portfolio weight (concentration guardrail).",
    )
    ap.add_argument("--write-submission-if-gate-passes", action="store_true")
    args = ap.parse_args()

    bars = load_frozen_bars(args.arena_root)
    feature_cache = build_feature_cache(bars)

    specs = build_xsmom_specs()
    n_trials = len(specs)
    results = [
        evaluate_candidate(
            bars,
            spec,
            n_trials=n_trials,
            features_by_symbol=feature_cache,
            max_symbol_weight=args.max_symbol_weight,
        )
        for spec in specs
    ]
    results.sort(
        key=lambda r: (
            r.portfolio_metrics["is_fold_sharpe_mean"],
            r.portfolio_metrics["is_sharpe"],
        ),
        reverse=True,
    )
    best = results[0]
    summary = {
        "n_trials": n_trials,
        "top_5": [summarize(r) for r in results[:5]],
        "best": summarize(best),
        "gate_g1": {
            "fold_sharpe_threshold": G1_FOLD_SHARPE_THRESHOLD,
            "cost_2x_threshold": G1_COST_2X_SHARPE_THRESHOLD,
            "fold_sharpe_pass": best.portfolio_metrics["is_fold_sharpe_mean"]
            >= G1_FOLD_SHARPE_THRESHOLD,
            "cost_2x_pass": best.portfolio_metrics["is_sharpe_cost_2x"]
            > G1_COST_2X_SHARPE_THRESHOLD,
        },
    }
    summary["gate_g1"]["pass"] = (
        summary["gate_g1"]["fold_sharpe_pass"]
        and summary["gate_g1"]["cost_2x_pass"]
    )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary["best"], indent=2))
    print(json.dumps(summary["gate_g1"], indent=2))

    if not summary["gate_g1"]["pass"]:
        print("G1 GATE FAILED: xsmom family archived, no submission written.")
        return 0

    if not args.write_submission_if_gate_passes:
        print("G1 passed. Re-run with --write-submission-if-gate-passes to submit.")
        return 0

    notes = (
        f"Cross-sectional momentum (xsmom). Selected {best.spec.name} "
        f"from {n_trials} honest trials. "
        f"IS Sharpe {best.portfolio_metrics['is_sharpe']:.3f}, "
        f"IS fold Sharpe mean {best.portfolio_metrics['is_fold_sharpe_mean']:.3f}, "
        f"IS MaxDD {best.portfolio_metrics['is_max_drawdown']:.3f}, "
        f"cost-2x Sharpe {best.portfolio_metrics['is_sharpe_cost_2x']:.3f}."
    )
    write_submission(
        args.submission_dir,
        engine_name=args.engine_name,
        best=best,
        notes=notes,
        n_trials=n_trials,
    )
    print(f"Submission written to {args.submission_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

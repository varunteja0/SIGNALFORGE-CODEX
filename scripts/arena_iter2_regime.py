"""Iter 2: vol-regime gate on top of dyn-v1's base spec.

Base strategy is `rel_lb120_sm24_en65_sb10_tv005` (our best judged
OOS Sharpe submission). This iteration searches over regime-gate
hyperparams (vol window, regime window, quantile) and applies the
same G1 gate: IS fold Sharpe mean must meet/beat the base.
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

BASE_SPEC = {
    "family": "relative",
    "params": {
        "lookback": 120,
        "smooth": 24,
        "enter_spread": 0.65,
        "exit_spread": 0.25,
        "switch_buffer": 0.10,
        "target_vol": 0.005,
    },
}
G1_FOLD_SHARPE_FLOOR = 1.10  # small slack vs dyn-v1 IS fold Sharpe mean (~1.17)
DEFAULT_SUBMISSION_DIR = (
    "/Users/varunteja/SignalForge-arena/submissions/codex-arena-regime-v1"
)


def build_regime_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for vol_window, regime_window, regime_quantile in product(
        (48, 72, 168),
        (14 * 24, 30 * 24, 60 * 24),
        (0.80, 0.85, 0.90, 0.95),
    ):
        name = (
            f"regime_vw{vol_window}_rw{regime_window}"
            f"_q{int(round(regime_quantile * 100)):02d}"
        )
        specs.append(
            CandidateSpec(
                name=name,
                family="regime_gated",
                params={
                    "base": BASE_SPEC,
                    "vol_window": vol_window,
                    "regime_window": regime_window,
                    "regime_quantile": regime_quantile,
                },
                notes="Regime gate over dyn-v1 base: disable positions when vol is top-quantile.",
            )
        )
    return specs


def summarize(result: CandidateResult) -> dict:
    m = result.portfolio_metrics
    return {
        "name": result.spec.name,
        "params": result.spec.params,
        "score": result.score,
        "is_sharpe": m["is_sharpe"],
        "is_fold_sharpe_mean": m["is_fold_sharpe_mean"],
        "is_fold_sharpe_min": m["is_fold_sharpe_min"],
        "is_fold_positive_frac": m["is_fold_positive_frac"],
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
        "--report-path", default="fund_data/arena_iter2_regime_report.json"
    )
    ap.add_argument("--submission-dir", default=DEFAULT_SUBMISSION_DIR)
    ap.add_argument("--engine-name", default="codex-arena-regime-v1")
    ap.add_argument("--max-symbol-weight", type=float, default=0.50)
    ap.add_argument("--write-submission-if-gate-passes", action="store_true")
    args = ap.parse_args()

    bars = load_frozen_bars(args.arena_root)
    feature_cache = build_feature_cache(bars)

    # Evaluate the bare base once for comparison.
    base_spec_candidate = CandidateSpec(
        name="base_dyn_v1",
        family=BASE_SPEC["family"],
        params=dict(BASE_SPEC["params"]),
    )
    base_result = evaluate_candidate(
        bars,
        base_spec_candidate,
        n_trials=1,
        features_by_symbol=feature_cache,
        max_symbol_weight=args.max_symbol_weight,
    )

    specs = build_regime_specs()
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
    # Only keep gated variants whose IS fold Sharpe mean is >= base.
    base_fold_sharpe = base_result.portfolio_metrics["is_fold_sharpe_mean"]
    filtered = [
        r for r in results
        if r.portfolio_metrics["is_fold_sharpe_mean"] >= base_fold_sharpe
        and r.portfolio_metrics["is_max_drawdown"] >= base_result.portfolio_metrics["is_max_drawdown"]
    ]
    if not filtered:
        filtered = results
    filtered.sort(
        key=lambda r: (
            r.portfolio_metrics["is_fold_sharpe_mean"],
            -abs(r.portfolio_metrics["is_max_drawdown"]),
        ),
        reverse=True,
    )
    best = filtered[0]

    summary = {
        "n_trials": n_trials,
        "base": summarize(base_result),
        "best_gated": summarize(best),
        "gate_g1": {
            "base_fold_sharpe": base_fold_sharpe,
            "floor": G1_FOLD_SHARPE_FLOOR,
            "fold_sharpe_pass": best.portfolio_metrics["is_fold_sharpe_mean"]
            >= max(G1_FOLD_SHARPE_FLOOR, base_fold_sharpe),
            "drawdown_pass": abs(best.portfolio_metrics["is_max_drawdown"])
            <= abs(base_result.portfolio_metrics["is_max_drawdown"]),
        },
    }
    summary["gate_g1"]["pass"] = (
        summary["gate_g1"]["fold_sharpe_pass"]
        and summary["gate_g1"]["drawdown_pass"]
    )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary["base"], indent=2))
    print(json.dumps(summary["best_gated"], indent=2))
    print(json.dumps(summary["gate_g1"], indent=2))

    if not summary["gate_g1"]["pass"]:
        print("G1 GATE FAILED: regime gate did not improve on base. Archived.")
        return 0
    if not args.write_submission_if_gate_passes:
        print("G1 passed. Re-run with --write-submission-if-gate-passes to submit.")
        return 0

    notes = (
        f"Vol-regime gate on dyn-v1 base. Selected {best.spec.name} from {n_trials} trials. "
        f"Base IS fold Sharpe {base_fold_sharpe:.3f}; "
        f"gated IS fold Sharpe {best.portfolio_metrics['is_fold_sharpe_mean']:.3f}, "
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

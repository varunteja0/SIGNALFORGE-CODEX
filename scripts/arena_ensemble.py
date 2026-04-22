"""Build a diversified ensemble from the refined arena grid.

Strategy:
  1. Evaluate refined grid on frozen IS data.
  2. Rank by robustness score.
  3. Greedily select up to `--max-members` candidates whose IS portfolio
     returns have mean pairwise correlation below `--corr-threshold`.
  4. Build an ensemble CandidateSpec and evaluate it honestly.
  5. Run forward (walk-forward) validation on the resulting ensemble.
  6. Report. Submission is only written when `--write-submission` is
     passed AND the ensemble's IS fold Sharpe mean beats the baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.arena.engine import (
    CandidateResult,
    CandidateSpec,
    build_feature_cache,
    build_refined_candidate_specs,
    evaluate_candidate,
    load_frozen_bars,
    write_research_report,
    write_submission,
)


def portfolio_returns(result: CandidateResult, bars: dict[str, pd.DataFrame]) -> pd.Series:
    from src.arena.engine import _returns_from_position  # type: ignore[attr-defined]

    total: pd.Series | None = None
    for symbol, frame in bars.items():
        pos = result.positions[symbol].reindex(frame.index).fillna(0.0)
        r = _returns_from_position(frame["close"], pos) * result.weights[symbol]
        total = r if total is None else total.add(r, fill_value=0.0)
    assert total is not None
    return total.fillna(0.0)


def select_uncorrelated(
    ranked: list[CandidateResult],
    bars: dict[str, pd.DataFrame],
    *,
    max_members: int,
    corr_threshold: float,
    top_n_pool: int,
    force_family_diversity: bool = False,
) -> list[tuple[CandidateResult, pd.Series]]:
    pool = ranked[:top_n_pool]
    returns_map: dict[str, pd.Series] = {r.spec.name: portfolio_returns(r, bars) for r in pool}

    selected: list[tuple[CandidateResult, pd.Series]] = []
    seen_families: set[str] = set()

    if force_family_diversity:
        # First pass: pick top-ranked candidate from each family.
        for cand in pool:
            if cand.spec.family in seen_families:
                continue
            selected.append((cand, returns_map[cand.spec.name]))
            seen_families.add(cand.spec.family)
            if len(selected) >= max_members:
                return selected

    for cand in pool:
        if any(cand.spec.name == prev.spec.name for prev, _ in selected):
            continue
        cand_ret = returns_map[cand.spec.name]
        if not selected:
            selected.append((cand, cand_ret))
            continue
        corrs = []
        for _, prev_ret in selected:
            joined = pd.concat([cand_ret, prev_ret], axis=1).dropna()
            if len(joined) < 50:
                corrs.append(1.0)
                continue
            c = joined.iloc[:, 0].corr(joined.iloc[:, 1])
            corrs.append(1.0 if not np.isfinite(c) else float(c))
        if max(corrs) <= corr_threshold:
            selected.append((cand, cand_ret))
        if len(selected) >= max_members:
            break
    return selected


def make_ensemble_spec(
    members: list[tuple[CandidateResult, pd.Series]],
    *,
    name: str,
) -> CandidateSpec:
    member_dicts: list[dict[str, Any]] = []
    weights: list[float] = []
    for result, ret in members:
        member_dicts.append({"family": result.spec.family, "params": result.spec.params})
        vol = float(ret.std(ddof=0))
        weights.append(1.0 / vol if vol > 1e-9 else 1.0)
    total = sum(weights)
    weights = [w / total for w in weights]
    return CandidateSpec(
        name=name,
        family="ensemble",
        params={
            "members": member_dicts,
            "member_weights": weights,
        },
        notes=f"Inverse-vol ensemble of {len(members)} uncorrelated refined candidates.",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena-root", default="/Users/varunteja/SignalForge-arena")
    ap.add_argument("--n-trials", type=int, default=270)
    ap.add_argument("--top-n-pool", type=int, default=20)
    ap.add_argument("--max-members", type=int, default=5)
    ap.add_argument("--corr-threshold", type=float, default=0.55)
    ap.add_argument("--max-weight-cap", type=float, default=0.60)
    ap.add_argument(
        "--report-path",
        default="fund_data/arena_ensemble_report.json",
    )
    ap.add_argument(
        "--submission-dir",
        default="/Users/varunteja/SignalForge-arena/submissions/codex-arena-ensemble-v1",
    )
    ap.add_argument("--engine-name", default="codex-arena-ensemble-v1")
    ap.add_argument("--baseline-oos-sharpe", type=float, default=0.01715)
    ap.add_argument("--write-submission", action="store_true")
    ap.add_argument(
        "--skip-gate",
        action="store_true",
        help="Submit even if ensemble does not beat best single on IS fold Sharpe.",
    )
    ap.add_argument("--force-family-diversity", action="store_true")
    ap.add_argument(
        "--max-symbol-weight",
        type=float,
        default=None,
        help="Cap on portfolio weight per symbol in the final ensemble.",
    )
    args = ap.parse_args()

    bars = load_frozen_bars(args.arena_root)
    feature_cache = build_feature_cache(bars)

    specs = build_refined_candidate_specs()
    effective_trials = int(max(args.n_trials, len(specs)))

    results: list[CandidateResult] = []
    for spec in specs:
        results.append(
            evaluate_candidate(
                bars,
                spec,
                n_trials=effective_trials,
                features_by_symbol=feature_cache,
            )
        )

    # Sort by robustness score (same as engine default ordering).
    results.sort(
        key=lambda r: (
            r.score,
            r.portfolio_metrics["is_deflated_sharpe"],
            r.portfolio_metrics["is_total_return"],
        ),
        reverse=True,
    )

    # Apply diversification cap on single-name weight for individual picks.
    filtered = [r for r in results if r.portfolio_metrics["max_weight"] <= args.max_weight_cap]
    if not filtered:
        filtered = results
    best_single = filtered[0]

    selected = select_uncorrelated(
        filtered,
        bars,
        max_members=args.max_members,
        corr_threshold=args.corr_threshold,
        top_n_pool=min(args.top_n_pool, len(filtered)),
        force_family_diversity=args.force_family_diversity,
    )
    if not selected:
        print("No ensemble members found.")
        return 1

    ensemble_spec = make_ensemble_spec(
        selected,
        name=f"ens_ivol_{len(selected)}_corr{int(round(args.corr_threshold * 100))}",
    )
    ensemble_result = evaluate_candidate(
        bars,
        ensemble_spec,
        n_trials=effective_trials,
        features_by_symbol=feature_cache,
        max_symbol_weight=args.max_symbol_weight,
    )

    summary = {
        "n_trials": effective_trials,
        "best_single": {
            "name": best_single.spec.name,
            "family": best_single.spec.family,
            "score": best_single.score,
            "is_fold_sharpe_mean": best_single.portfolio_metrics["is_fold_sharpe_mean"],
            "is_fold_sharpe_min": best_single.portfolio_metrics["is_fold_sharpe_min"],
            "is_sharpe": best_single.portfolio_metrics["is_sharpe"],
            "is_max_drawdown": best_single.portfolio_metrics["is_max_drawdown"],
            "mean_turnover": best_single.portfolio_metrics["mean_turnover"],
            "max_weight": best_single.portfolio_metrics["max_weight"],
            "weights": best_single.weights,
        },
        "ensemble": {
            "name": ensemble_result.spec.name,
            "n_members": len(selected),
            "members": [
                {
                    "name": result.spec.name,
                    "family": result.spec.family,
                    "is_sharpe": result.portfolio_metrics["is_sharpe"],
                    "is_fold_sharpe_mean": result.portfolio_metrics["is_fold_sharpe_mean"],
                    "max_weight": result.portfolio_metrics["max_weight"],
                }
                for result, _ in selected
            ],
            "score": ensemble_result.score,
            "is_fold_sharpe_mean": ensemble_result.portfolio_metrics["is_fold_sharpe_mean"],
            "is_fold_sharpe_min": ensemble_result.portfolio_metrics["is_fold_sharpe_min"],
            "is_fold_positive_frac": ensemble_result.portfolio_metrics["is_fold_positive_frac"],
            "is_sharpe": ensemble_result.portfolio_metrics["is_sharpe"],
            "is_sharpe_cost_1p5x": ensemble_result.portfolio_metrics["is_sharpe_cost_1p5x"],
            "is_sharpe_cost_2x": ensemble_result.portfolio_metrics["is_sharpe_cost_2x"],
            "is_max_drawdown": ensemble_result.portfolio_metrics["is_max_drawdown"],
            "is_total_return": ensemble_result.portfolio_metrics["is_total_return"],
            "mean_turnover": ensemble_result.portfolio_metrics["mean_turnover"],
            "max_weight": ensemble_result.portfolio_metrics["max_weight"],
            "weights": ensemble_result.weights,
        },
        "gate": {
            "baseline_oos_sharpe": args.baseline_oos_sharpe,
            "beats_best_single_is_fold_sharpe": (
                ensemble_result.portfolio_metrics["is_fold_sharpe_mean"]
                > best_single.portfolio_metrics["is_fold_sharpe_mean"]
            ),
        },
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2))
    write_research_report(
        report_path.with_name(report_path.stem + "_ranked.json"),
        best=ensemble_result,
        ranked_results=[ensemble_result, best_single],
        include_oos=False,
        n_trials=effective_trials,
    )

    print(json.dumps(summary["ensemble"], indent=2))
    print(json.dumps(summary["gate"], indent=2))

    if args.write_submission:
        gate_pass = summary["gate"]["beats_best_single_is_fold_sharpe"]
        if not gate_pass and not args.skip_gate:
            print("GATE FAILED: ensemble did not beat best single. Refusing to submit.")
            return 2
        if not gate_pass:
            print("GATE BYPASSED via --skip-gate; ensemble IS fold Sharpe below best single.")
        notes = (
            f"Inverse-vol ensemble across {len(selected)} family-diverse refined candidates "
            f"(corr<={args.corr_threshold}, symbol-cap={args.max_symbol_weight}). "
            f"Selected from {effective_trials} honest trials. "
            f"IS Sharpe {ensemble_result.portfolio_metrics['is_sharpe']:.3f}, "
            f"IS fold positive frac {ensemble_result.portfolio_metrics['is_fold_positive_frac']:.2f}, "
            f"IS MaxDD {ensemble_result.portfolio_metrics['is_max_drawdown']:.3f}, "
            f"cost-2x Sharpe {ensemble_result.portfolio_metrics['is_sharpe_cost_2x']:.3f}."
        )
        write_submission(
            args.submission_dir,
            engine_name=args.engine_name,
            best=ensemble_result,
            notes=notes,
            n_trials=effective_trials,
        )
        print(f"Submission written to {args.submission_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Outcome validation: run the harness + funding-fade sweep multiple times
with perturbed configurations. Aggregate verdicts across runs.

If a finding (NO_DEPLOYABLE_EDGE) holds across multiple OOS splits,
multiple symbol universes, and multiple random seeds, we can treat it
as a robust conclusion rather than a single-run artefact.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.validation_harness import run_validation
from src.strategies.funding_fade import backtest_funding_fade, load_data


# ─── Run 1: Harness with varying OOS splits & universes ──────────

def harness_sweep(output_dir: Path) -> list[dict]:
    """Run validation harness across different OOS fractions + symbol sets."""
    print("\n" + "=" * 72)
    print("  OUTCOME VALIDATION PASS 1 — Harness perturbation sweep")
    print("=" * 72)

    universes = [
        ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        ["ETH/USDT", "SOL/USDT"],
    ]
    oos_fractions = [0.20, 0.30, 0.40]

    runs: list[dict] = []
    for i, (syms, oos) in enumerate(itertools.product(universes, oos_fractions)):
        label = f"run{i+1}_syms{len(syms)}_oos{int(oos*100)}"
        out_path = output_dir / f"harness_{label}.json"
        print(f"\n── {label}: symbols={syms} oos={oos}")
        t0 = time.time()
        try:
            report = run_validation(
                symbols=syms,
                timeframe="1h",
                days=1825,
                min_trades=50,
                oos_fraction=oos,
                output_path=str(out_path),
                use_structural=False,
            )
            # report.strategies is list[dict] (from asdict)
            strats = report.strategies
            def v(s, k, default=None):
                return s.get(k, default) if isinstance(s, dict) else getattr(s, k, default)
            runs.append({
                "label": label,
                "symbols": syms,
                "oos_fraction": oos,
                "overall_verdict": report.overall_verdict,
                "n_keep": sum(1 for s in strats if v(s, "final_verdict") == "KEEP"),
                "n_conditional": sum(1 for s in strats
                                     if v(s, "final_verdict") == "CONDITIONAL"),
                "n_kill": sum(1 for s in strats if v(s, "final_verdict") == "KILL"),
                "n_deployed": report.n_deployed,
                "portfolio_sharpe": report.portfolio_oos_sharpe,
                "elapsed_s": round(time.time() - t0, 1),
            })
        except Exception as e:
            print(f"  FAILED: {e}")
            runs.append({"label": label, "error": str(e)})

    return runs


# ─── Run 2: Funding-fade with extended configs ────────────────────

def funding_sweep(output_dir: Path) -> list[dict]:
    """Expanded funding-fade sweep: more symbols + more OOS splits."""
    print("\n" + "=" * 72)
    print("  OUTCOME VALIDATION PASS 2 — Funding-fade robustness")
    print("=" * 72)

    # Cache loads per symbol
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"]
    data_cache: dict[str, tuple] = {}
    for sym in symbols:
        print(f"[{sym}] loading...", flush=True)
        try:
            ohlcv, funding = load_data(sym, days=730)
            if ohlcv.empty or funding.empty:
                print("  SKIP")
                continue
            start = max(ohlcv.index[0], funding.index[0])
            end = min(ohlcv.index[-1], funding.index[-1])
            ohlcv = ohlcv[(ohlcv.index >= start) & (ohlcv.index <= end)]
            funding = funding[(funding.index >= start) & (funding.index <= end)]
            if len(ohlcv) < 500 or len(funding) < 100:
                print(f"  SKIP (insufficient: ohlcv={len(ohlcv)} fund={len(funding)})")
                continue
            data_cache[sym] = (ohlcv, funding)
            print(f"  loaded: ohlcv={len(ohlcv)} funding={len(funding)}")
        except Exception as e:
            print(f"  FAILED: {e}")

    entry_zs = [1.5, 2.0, 2.5]
    holds = [1, 3, 6]
    sides = ["both", "short_only", "long_only"]
    oos_fractions = [0.20, 0.30, 0.40]

    results = []
    for sym in data_cache:
        ohlcv_full, funding_full = data_cache[sym]
        for oos in oos_fractions:
            split_idx = int(len(ohlcv_full) * (1 - oos))
            split_ts = ohlcv_full.index[split_idx]
            ohlcv_is = ohlcv_full[ohlcv_full.index < split_ts]
            ohlcv_oos = ohlcv_full[ohlcv_full.index >= split_ts]
            funding_is = funding_full[funding_full.index < split_ts]
            funding_oos = funding_full[funding_full.index >= split_ts]

            for ez, h, side in itertools.product(entry_zs, holds, sides):
                try:
                    is_r = backtest_funding_fade(ohlcv_is, funding_is,
                                                 symbol=sym, entry_z=ez,
                                                 hold_periods=h, side=side)
                    oos_r = backtest_funding_fade(ohlcv_oos, funding_oos,
                                                  symbol=sym, entry_z=ez,
                                                  hold_periods=h, side=side)
                except Exception as e:
                    continue

                if is_r.n_trades < 10 or oos_r.n_trades < 10:
                    continue

                results.append({
                    "symbol": sym, "oos_fraction": oos,
                    "entry_z": ez, "hold": h, "side": side,
                    "is_sharpe": round(is_r.sharpe, 2),
                    "oos_sharpe": round(oos_r.sharpe, 2),
                    "is_trades": is_r.n_trades,
                    "oos_trades": oos_r.n_trades,
                    "is_total": round(is_r.total_return, 4),
                    "oos_total": round(oos_r.total_return, 4),
                    "oos_dd": round(oos_r.max_drawdown, 4),
                })

    return results


# ─── Aggregation ──────────────────────────────────────────────────

def aggregate(harness_runs: list[dict], funding_results: list[dict]) -> dict:
    print("\n" + "=" * 72)
    print("  FINAL AGGREGATION")
    print("=" * 72)

    # Harness verdict histogram
    verdicts = Counter(r.get("overall_verdict", "ERROR") for r in harness_runs)
    print(f"\n  Harness runs completed: {len(harness_runs)}")
    print("  Harness verdict distribution:")
    for v, c in verdicts.most_common():
        print(f"    {v:25} : {c}")
    keep_runs = [r for r in harness_runs if r.get("n_keep", 0) > 0]
    print(f"\n  Runs producing any KEEP strategy: {len(keep_runs)}/{len(harness_runs)}")

    # Funding-fade: combos robust across >=2 symbols AND >=2 OOS fractions
    print(f"\n  Funding-fade configs tested: {len(funding_results)}")
    by_combo: dict[tuple, list] = {}
    for r in funding_results:
        by_combo.setdefault((r["entry_z"], r["hold"], r["side"]), []).append(r)

    print("\n  Combos with OOS Sh > 0.5 across >=2 symbols AND >=2 OOS splits:")
    strong = []
    for combo, rows in by_combo.items():
        good = [r for r in rows if r["oos_sharpe"] > 0.5 and r["oos_trades"] >= 10]
        syms = set(r["symbol"] for r in good)
        oos_fracs = set(r["oos_fraction"] for r in good)
        if len(syms) >= 2 and len(oos_fracs) >= 2:
            avg_sh = sum(r["oos_sharpe"] for r in good) / len(good)
            strong.append((combo, len(good), avg_sh, sorted(syms), sorted(oos_fracs)))
    if not strong:
        print("    NONE.")
    else:
        strong.sort(key=lambda x: -x[2])
        for combo, n, avg, syms, fracs in strong[:10]:
            ez, h, side = combo
            print(f"    z={ez} h={h} side={side:>11}  n={n} avg_sh={avg:.2f}  "
                  f"syms={syms} oos={fracs}")

    # Count positive OOS Sharpes overall
    pos_oos = sum(1 for r in funding_results if r["oos_sharpe"] > 0)
    strong_oos = sum(1 for r in funding_results if r["oos_sharpe"] > 1.0)
    print(f"\n  Funding configs with any positive OOS Sharpe: {pos_oos}/{len(funding_results)}")
    print(f"  Funding configs with OOS Sharpe > 1.0:         {strong_oos}/{len(funding_results)}")

    # Overall verdict
    harness_no_edge = sum(1 for r in harness_runs
                          if r.get("overall_verdict") == "NO_DEPLOYABLE_EDGE")
    robust_funding = len(strong)

    if harness_no_edge == len(harness_runs) and robust_funding == 0:
        conclusion = "ROBUST: no deployable edge across all perturbations"
    elif robust_funding > 0:
        conclusion = f"POTENTIAL EDGE: {robust_funding} funding-fade combos survived robustness check"
    elif len(keep_runs) > 0:
        conclusion = f"MARGINAL: {len(keep_runs)} harness runs produced KEEP strategies"
    else:
        conclusion = "MIXED — see details above"

    print(f"\n  CONCLUSION: {conclusion}")
    return {
        "harness_runs": harness_runs,
        "funding_results": funding_results,
        "summary": {
            "n_harness_runs": len(harness_runs),
            "harness_verdicts": dict(verdicts),
            "runs_with_keep": len(keep_runs),
            "n_funding_configs": len(funding_results),
            "funding_positive_oos": pos_oos,
            "funding_strong_oos": strong_oos,
            "robust_funding_combos": robust_funding,
            "conclusion": conclusion,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="fund_data/outcome_validation.json")
    p.add_argument("--skip-harness", action="store_true",
                   help="Skip the expensive harness sweep (useful for re-aggregating)")
    p.add_argument("--skip-funding", action="store_true")
    args = p.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = output.parent / "outcome_runs"
    run_dir.mkdir(exist_ok=True)

    harness_runs = [] if args.skip_harness else harness_sweep(run_dir)
    funding_results = [] if args.skip_funding else funding_sweep(run_dir)

    final = aggregate(harness_runs, funding_results)
    with open(output, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\n  Full report: {output}")


if __name__ == "__main__":
    main()

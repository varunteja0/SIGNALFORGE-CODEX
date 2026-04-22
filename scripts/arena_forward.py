"""Forward validation and shadow paper-trading for arena strategies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arena import build_default_candidate_specs, build_refined_candidate_specs
from src.arena.forward import (
    build_paper_snapshot,
    load_bars_for_mode,
    run_forward_validation,
    write_forward_artifacts,
    write_paper_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("historical", "paper-once"), default="historical")
    parser.add_argument("--source", choices=("frozen", "public"), default="frozen")
    parser.add_argument("--arena-root", default="~/SignalForge-arena")
    parser.add_argument("--grid", choices=("default", "refined"), default="refined")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--train-bars", type=int, default=24 * 365)
    parser.add_argument("--test-bars", type=int, default=24 * 30)
    parser.add_argument("--step-bars", type=int)
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--max-weight-cap", type=float)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument(
        "--output-dir",
        default="fund_data/arena_forward_latest",
        help="Output directory for reports, positions, and paper/ledger artifacts.",
    )
    args = parser.parse_args()

    bars = load_bars_for_mode(source=args.source, arena_root=args.arena_root, days=args.days)
    specs = build_refined_candidate_specs() if args.grid == "refined" else build_default_candidate_specs()
    honest_trials = int(args.n_trials) if args.n_trials is not None else len(specs)

    if args.mode == "historical":
        result = run_forward_validation(
            bars,
            specs=specs,
            n_trials=honest_trials,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
            max_weight_cap=args.max_weight_cap,
        )
        out = write_forward_artifacts(
            args.output_dir,
            result=result,
            bars_by_symbol=bars,
            initial_capital=args.capital,
        )
        print(json.dumps({
            "output_dir": str(out),
            "summary": result.summary,
            "selection_counts": result.selection_counts,
            "latest_strategy": str(result.strategy_labels.iloc[-1]),
        }, indent=2))
        return 0

    snapshot = build_paper_snapshot(
        bars,
        specs=specs,
        n_trials=honest_trials,
        max_weight_cap=args.max_weight_cap,
    )
    out = write_paper_snapshot(
        args.output_dir,
        snapshot=snapshot,
        bars_by_symbol=bars,
        capital=args.capital,
    )
    print(json.dumps({"output_dir": str(out), "snapshot": snapshot.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
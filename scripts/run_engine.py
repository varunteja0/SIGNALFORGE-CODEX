#!/usr/bin/env python3
"""
SignalForge Autonomous Quant Engine — CLI
==========================================
Usage:
    # Discover: find best strategies (one-shot, no deploy)
    python scripts/run_engine.py discover

    # Run: continuous discovery → deploy → monitor loop
    python scripts/run_engine.py run

    # Status: show current engine state
    python scripts/run_engine.py status

    # Quick: fast discovery with fewer candidates
    python scripts/run_engine.py discover --quick

Options:
    --symbols BTC/USDT ETH/USDT   Target assets
    --days 365                     Data lookback
    --quick                        Fewer candidates (5 per template)
    --cycle-hours 24               Hours between cycles (run mode)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fund_data/engine.log", encoding="utf-8"),
    ],
)


def cmd_discover(args):
    """One-shot discovery: generate → test → rank → report."""
    from src.engine.autonomous import AutonomousEngine

    engine = AutonomousEngine(
        symbols=args.symbols,
        n_per_template=args.n_per_template,
        data_days=args.days,
    )

    ranked = engine.discover()

    print(f"\n{'='*70}")
    print(f"  DISCOVERY RESULTS — {len(ranked)} strategies passed filters")
    print(f"{'='*70}\n")

    if not ranked:
        print("  No viable strategies found.")
        return

    print(f"{'#':<4} {'Name':<25} {'Template':<20} {'Score':>7} "
          f"{'PF':>6} {'Trades':>7} {'Sharpe':>7} {'WR':>6}")
    print("-" * 90)

    for i, s in enumerate(ranked[:20]):
        print(f"{i+1:<4} {s.candidate.name:<25} {s.candidate.template:<20} "
              f"{s.score:>7.3f} {s.combined_pf:>6.2f} {s.total_trades:>7} "
              f"{s.avg_sharpe:>7.2f} {s.combined_wr:>5.1%}")

    print(f"\nDetailed results: pipeline_output/engine_results.json")


def cmd_run(args):
    """Continuous loop: discover → deploy → monitor → repeat."""
    from src.engine.autonomous import AutonomousEngine

    engine = AutonomousEngine(
        symbols=args.symbols,
        n_per_template=args.n_per_template,
        data_days=args.days,
        cycle_hours=args.cycle_hours,
    )

    engine.run_loop()


def cmd_status(args):
    """Show current engine state."""
    from src.engine.autonomous import EngineState

    state = EngineState.load("fund_data/engine_state.json")

    print(f"\n{'='*50}")
    print(f"  AUTONOMOUS ENGINE STATUS")
    print(f"{'='*50}\n")
    print(f"  Cycles completed:     {state.cycle_count}")
    print(f"  Candidates tested:    {state.total_candidates_tested}")
    print(f"  Strategies deployed:  {state.total_strategies_deployed}")
    print(f"  Strategies killed:    {state.total_strategies_killed}")
    print(f"  Last cycle:           {state.last_cycle_time or 'never'}")
    print(f"  Best score ever:      {state.best_score_ever:.3f}")
    print(f"  Best strategy:        {state.best_strategy_ever or 'none'}")

    if state.deployed:
        print(f"\n  Currently deployed ({len(state.deployed)}):")
        for d in state.deployed:
            print(f"    {d['name']:<25} weight={d['weight']:.1%} "
                  f"PF={d['backtest_pf']:.2f} trades={d['backtest_trades']}")

    # Show latest results if available
    results_path = Path("pipeline_output/engine_results.json")
    if results_path.exists():
        print(f"\n  Latest results: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SignalForge Autonomous Quant Engine"
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Common args
    for name in ["discover", "run"]:
        p = sub.add_parser(name)
        p.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"],
                        help="Target symbols")
        p.add_argument("--days", type=int, default=365, help="Data lookback days")
        p.add_argument("--quick", action="store_true",
                        help="Quick mode (5 candidates per template)")

    # Run-specific
    run_parser = sub.choices["run"]
    run_parser.add_argument("--cycle-hours", type=float, default=24.0,
                            help="Hours between cycles")

    # Status
    sub.add_parser("status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Handle --quick
    if hasattr(args, "quick") and args.quick:
        args.n_per_template = 5
    elif hasattr(args, "quick"):
        args.n_per_template = 20

    Path("fund_data").mkdir(exist_ok=True)

    commands = {
        "discover": cmd_discover,
        "run": cmd_run,
        "status": cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

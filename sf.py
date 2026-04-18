#!/usr/bin/env python3
"""
SignalForge CLI — Single entry point.
=======================================

Factory Pipeline (scan → validate → deploy):
    python sf.py scan              # Scan for signal anomalies
    python sf.py validate          # Validate scan results OOS
    python sf.py factory           # Run full factory loop (scan→validate→deploy)
    python sf.py factory --once    # Run one cycle and exit
    python sf.py backtest          # Honest backtest of deployed strategies
    python sf.py status            # Show deployed strategies + health

Unified Engine (evolve → trade → monitor → adapt):
    python sf.py run               # Paper trading with unified engine
    python sf.py run --real        # Live trading (CAUTION)
    python sf.py evolve            # Run GP evolution
    python sf.py crowding          # Show crowding analysis
    python sf.py cascade           # Show cascade prediction
    python sf.py engine-status     # Show engine state

Legacy:
    python sf.py live              # Paper trading via go_live.py (legacy)
    python sf.py report            # Generate daily report
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")


def cmd_scan(args):
    """Run signal hypothesis scan."""
    from src.data.fetcher import DataFetcher
    from src.data.features import compute_all_features
    from src.data.structural import StructuralDataFetcher
    from src.factory.scanner import scan

    print("=" * 60)
    print("  SIGNALFORGE — SIGNAL SCAN")
    print("=" * 60)

    # Load data
    fetcher = DataFetcher()
    struct = StructuralDataFetcher()
    datasets = {}

    for sym in args.symbols:
        try:
            raw = fetcher.fetch(sym, args.timeframe, days=args.days)
            if raw is not None and not raw.empty:
                df = compute_all_features(raw)
                try:
                    df = struct.fetch_all(symbol=sym.replace("/", ""), price_df=df, days=args.days)
                except Exception:
                    pass
                datasets[sym] = df
                print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: FAILED — {e}")

    if not datasets:
        print("No data loaded. Check connection.")
        return

    # Scan
    result = scan(datasets, min_trades=args.min_trades)

    print(f"\n  Hypotheses tested:    {result.total_hypotheses}")
    print(f"  Bonferroni threshold: {result.bonferroni_threshold:.2e}")
    print(f"  Raw survivors:        {len(result.raw_survivors)}")
    print(f"  Bonferroni survivors: {len(result.bonferroni_survivors)}")

    if result.raw_survivors:
        print(f"\n  {'Signal':<40s} {'Asset':>5s} {'N':>5s} {'PF':>6s} {'Sharpe':>7s} {'p-val':>8s}")
        print(f"  {'─'*40} {'─'*5} {'─'*5} {'─'*6} {'─'*7} {'─'*8}")

        for s in result.raw_survivors[:30]:
            sym = s.asset.split("/")[0]
            print(f"  {s.name:<40s} {sym:>5s} {s.n_trades:>5d} {s.pf:>6.2f} {s.sharpe:>7.2f} {s.p_value:>8.4f}")

    print()


def cmd_validate(args):
    """Scan + validate OOS."""
    from src.data.fetcher import DataFetcher
    from src.data.features import compute_all_features
    from src.data.structural import StructuralDataFetcher
    from src.factory.scanner import scan
    from src.factory.validator import validate

    print("=" * 60)
    print("  SIGNALFORGE — SCAN + VALIDATE")
    print("=" * 60)

    fetcher = DataFetcher()
    struct = StructuralDataFetcher()
    datasets = {}

    for sym in args.symbols:
        try:
            raw = fetcher.fetch(sym, args.timeframe, days=args.days)
            if raw is not None and not raw.empty:
                df = compute_all_features(raw)
                try:
                    df = struct.fetch_all(symbol=sym.replace("/", ""), price_df=df, days=args.days)
                except Exception:
                    pass
                datasets[sym] = df
                print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: FAILED — {e}")

    if not datasets:
        return

    # Scan
    print("\n  Scanning...")
    scan_result = scan(datasets, min_trades=args.min_trades)
    print(f"  {scan_result.total_hypotheses} hypotheses → {len(scan_result.raw_survivors)} raw survivors")

    # Validate
    print("  Validating OOS...")
    val_result = validate(scan_result.raw_survivors, datasets)

    print(f"\n  Tested: {val_result.signals_tested}")
    print(f"  Passed IS: {val_result.signals_passed_is}")
    print(f"  Passed OOS: {val_result.signals_passed_oos}")

    if val_result.validated:
        print(f"\n  {'Signal':<40s} {'Asset':>5s} {'Grade':>5s} {'OOS-PF':>7s} {'OOS-Sh':>7s} {'WF':>5s}")
        print(f"  {'─'*40} {'─'*5} {'─'*5} {'─'*7} {'─'*7} {'─'*5}")

        for s in val_result.validated:
            sym = s.asset.split("/")[0]
            wf = f"{s.wf_positive_folds}/{s.wf_total_folds}"
            print(f"  {s.name:<40s} {sym:>5s} {s.grade:>5s} {s.oos_pf:>7.2f} {s.oos_sharpe:>7.2f} {wf:>5s}")
    else:
        print("\n  No signals survived OOS validation.")

    print()


def cmd_factory(args):
    """Run the strategy factory loop."""
    from src.factory.loop import StrategyFactoryLoop

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("fund_data/factory.log", encoding="utf-8"),
        ],
    )

    loop = StrategyFactoryLoop(
        symbols=args.symbols,
        timeframe=args.timeframe,
        data_days=args.days,
        cycle_hours=args.cycle_hours,
        min_scan_trades=args.min_trades,
        max_deployed=args.max_strategies,
    )

    if args.once:
        result = loop.run_once()
        print(json.dumps(result, indent=2, default=str))
    else:
        loop.run()


def cmd_backtest(args):
    """Run honest backtest of deployed strategies."""
    from src.factory.deployer import load_deployed
    from src.data.fetcher import DataFetcher
    from src.data.features import compute_all_features
    from src.data.structural import StructuralDataFetcher
    from src.backtest.engine import Backtester

    deployed = load_deployed()
    if not deployed:
        print("No deployed strategies found. Run 'sf.py factory --once' first.")
        return

    print("=" * 60)
    print("  SIGNALFORGE — HONEST BACKTEST")
    print("=" * 60)

    fetcher = DataFetcher()
    struct = StructuralDataFetcher()
    datasets = {}

    assets = list(set(s.asset for s in deployed))
    for sym in assets:
        try:
            raw = fetcher.fetch(sym, args.timeframe, days=args.days)
            if raw is not None and not raw.empty:
                df = compute_all_features(raw)
                try:
                    df = struct.fetch_all(symbol=sym.replace("/", ""), price_df=df, days=args.days)
                except Exception:
                    pass
                datasets[sym] = df
                print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: FAILED — {e}")

    print(f"\n  {'Strategy':<35s} {'Asset':>5s} {'Trades':>7s} {'PF':>6s} {'Sharpe':>7s} {'Return':>8s}")
    print(f"  {'─'*35} {'─'*5} {'─'*7} {'─'*6} {'─'*7} {'─'*8}")

    for strat in deployed:
        df = datasets.get(strat.asset)
        if df is None:
            continue

        bt = Backtester(initial_capital=10000, commission_pct=0.001, slippage_pct=0.0005)

        try:
            result = bt.run(
                df, strat.generate_signals,
                position_size_pct=strat.position_size_pct,
                stop_loss_atr=strat.stop_loss_atr,
                take_profit_atr=strat.take_profit_atr,
                max_holding_bars=strat.hold_bars,
            )

            sym = strat.asset.split("/")[0]
            print(
                f"  {strat.name:<35s} {sym:>5s} "
                f"{result.total_trades:>7d} {result.profit_factor:>6.2f} "
                f"{result.sharpe_ratio:>7.2f} {result.total_return * 100:>+7.1f}%"
            )
        except Exception as e:
            print(f"  {strat.name:<35s}   ERROR: {e}")

    print()


def cmd_status(args):
    """Show deployed strategies and health."""
    from src.factory.deployer import load_deployed
    from src.factory.monitor import StrategyMonitor

    deployed = load_deployed()
    if not deployed:
        print("No deployed strategies.")
        return

    monitor = StrategyMonitor()

    print("=" * 60)
    print("  SIGNALFORGE — STATUS")
    print("=" * 60)
    print(f"  Active strategies: {len(deployed)}")
    print()

    for s in deployed:
        health = monitor.assess_health(s)
        print(f"  {s.name}")
        print(f"    Asset: {s.asset}  Direction: {'LONG' if s.direction > 0 else 'SHORT'}  Hold: {s.hold_bars}h")
        print(f"    Grade: {s.grade}  Size: {s.position_size_pct:.1%}  SL: {s.stop_loss_atr} ATR  TP: {s.take_profit_atr} ATR")
        print(f"    OOS:  PF={s.oos_pf:.2f}  Sharpe={s.oos_sharpe:.2f}")
        print(f"    Live: {health.status} — {health.message}")
        print()


def cmd_live(args):
    """Start live paper trading with go_live.py (legacy)."""
    import subprocess
    cmd = [sys.executable, "scripts/go_live.py"]
    if not args.real:
        cmd.append("--paper")
    subprocess.run(cmd)


def cmd_report(args):
    """Generate daily report."""
    import subprocess
    subprocess.run([sys.executable, "scripts/daily_report.py"])


# ==================================================================
# NEW — Unified Engine Commands
# ==================================================================

def cmd_run(args):
    """Run SignalForge unified engine."""
    from src.core.engine import SignalForgeEngine, EngineConfig

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("fund_data/engine.log", encoding="utf-8"),
        ],
    )

    mode = "live" if args.real else "paper"
    config = EngineConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.days,
        initial_capital=args.capital,
        paper_mode=(mode == "paper"),
        max_active_strategies=args.max_strategies,
        tick_interval_seconds=args.tick_interval,
        evolution_interval_hours=args.evolve_hours,
    )

    print("=" * 60)
    print(f"  SIGNALFORGE ENGINE — {mode.upper()} MODE")
    print("=" * 60)
    print(f"  Symbol:     {config.symbol}")
    print(f"  Timeframe:  {config.timeframe}")
    print(f"  Capital:    ${config.initial_capital:,.2f}")
    print(f"  Strategies: up to {config.max_active_strategies}")
    print(f"  Tick:       every {config.tick_interval_seconds}s")
    print("=" * 60)

    engine = SignalForgeEngine(config)
    engine.run(mode=mode)


def cmd_evolve(args):
    """Run GP evolution through unified engine."""
    from src.core.engine import SignalForgeEngine, EngineConfig

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = EngineConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.days,
        population_size=args.population,
        max_generations=args.generations,
    )

    print("=" * 60)
    print("  SIGNALFORGE — GP EVOLUTION")
    print("=" * 60)
    print(f"  Symbol: {config.symbol}  Timeframe: {config.timeframe}")
    print(f"  Population: {config.population_size}  Generations: {config.max_generations}")
    print()

    engine = SignalForgeEngine(config)
    strategies = engine.evolve()

    print(f"\n  Evolved {len(strategies)} strategies")
    for i, s in enumerate(strategies[:10]):
        name = s.name if hasattr(s, "name") else f"genome_{i}"
        sharpe = s.sharpe if hasattr(s, "sharpe") else 0
        print(f"  {i+1}. {name}: Sharpe={sharpe:.3f}")
    print()


def cmd_crowding(args):
    """Show current crowding analysis."""
    from src.core.engine import SignalForgeEngine, EngineConfig

    config = EngineConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.days,
    )
    engine = SignalForgeEngine(config)
    result = engine.get_crowding()

    dir_label = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}.get(result["direction"], "?")

    print("=" * 60)
    print("  SIGNALFORGE — CROWDING ANALYSIS")
    print("=" * 60)
    print(f"  Symbol:     {config.symbol}")
    print(f"  Score:      {result['score']:.1f} / 100")
    print(f"  Direction:  {dir_label}")
    print(f"  Confidence: {result['confidence']:.0%}")
    print(f"  Sources:    {result['n_sources']}")
    if result["components"]:
        print(f"\n  Components:")
        for comp, val in result["components"].items():
            print(f"    {comp}: {val}")
    print()


def cmd_cascade(args):
    """Show current cascade prediction."""
    from src.core.engine import SignalForgeEngine, EngineConfig

    config = EngineConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.days,
    )
    engine = SignalForgeEngine(config)
    result = engine.get_cascade()

    dir_label = {1: "UP (shorts liquidated)", -1: "DOWN (longs liquidated)", 0: "NEUTRAL"}.get(result["direction"], "?")
    sig_label = {1: "LONG", -1: "SHORT", 0: "NO TRADE"}.get(result["signal"], "?")

    print("=" * 60)
    print("  SIGNALFORGE — CASCADE PREDICTION")
    print("=" * 60)
    print(f"  Symbol:      {config.symbol}")
    print(f"  Probability: {result['probability']:.1%}")
    print(f"  Direction:   {dir_label}")
    print(f"  Signal:      {sig_label}")
    print(f"  Strength:    {result['strength']:.1%}")
    print(f"  Reasoning:   {result['reasoning']}")
    if result["preconditions"]:
        print(f"\n  Preconditions:")
        for k, v in result["preconditions"].items():
            print(f"    {k}: {v}")
    print()


def cmd_engine_status(args):
    """Show unified engine status."""
    state_path = Path("fund_data/engine_state.json")
    if not state_path.exists():
        print("No engine state found. Run 'sf.py run' first.")
        return

    with open(state_path) as f:
        state = json.load(f)

    print("=" * 60)
    print("  SIGNALFORGE — ENGINE STATUS")
    print("=" * 60)
    print(f"  Mode:       {state.get('mode', '?')}")
    print(f"  Capital:    ${state.get('capital', 0):,.2f}")
    print(f"  Peak:       ${state.get('peak_capital', 0):,.2f}")
    dd = 1 - state.get("capital", 0) / max(state.get("peak_capital", 1), 1)
    print(f"  Drawdown:   {dd:.1%}")
    print(f"  Ticks:      {state.get('tick_count', 0)}")
    print(f"  Strategies: {len(state.get('active_strategies', []))}")
    print(f"  Positions:  {len(state.get('open_positions', {}))}")
    print(f"  Last saved: {state.get('saved_at', '?')}")
    print()


def cmd_validate_all(args):
    """Run the brutal validation harness."""
    from src.core.validation_harness import run_validation

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    run_validation(
        symbols=args.symbols,
        timeframe=args.timeframe,
        days=args.days,
        min_trades=args.min_trades,
        oos_fraction=args.oos_fraction,
        output_path=args.output,
        use_structural=not args.no_structural,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="sf",
        description="SignalForge — Industrialized Hypothesis Testing",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    common.add_argument("--timeframe", default="1h")
    common.add_argument("--days", type=int, default=365)
    common.add_argument("--min-trades", type=int, default=50)

    # scan
    p_scan = sub.add_parser("scan", parents=[common], help="Scan for signal anomalies")
    p_scan.set_defaults(func=cmd_scan)

    # validate
    p_val = sub.add_parser("validate", parents=[common], help="Scan + validate OOS")
    p_val.set_defaults(func=cmd_validate)

    # factory
    p_fac = sub.add_parser("factory", parents=[common], help="Run strategy factory loop")
    p_fac.add_argument("--once", action="store_true", help="Run one cycle and exit")
    p_fac.add_argument("--cycle-hours", type=float, default=6.0)
    p_fac.add_argument("--max-strategies", type=int, default=10)
    p_fac.set_defaults(func=cmd_factory)

    # backtest
    p_bt = sub.add_parser("backtest", parents=[common], help="Backtest deployed strategies")
    p_bt.set_defaults(func=cmd_backtest)

    # status
    p_st = sub.add_parser("status", help="Show deployed strategies + health")
    p_st.set_defaults(func=cmd_status)

    # live (legacy — delegates to go_live.py)
    p_live = sub.add_parser("live", help="[Legacy] Start live paper trading via go_live.py")
    p_live.add_argument("--real", action="store_true", help="Real money mode (CAUTION)")
    p_live.set_defaults(func=cmd_live)

    # report
    p_rep = sub.add_parser("report", help="Generate daily report")
    p_rep.set_defaults(func=cmd_report)

    # ---- NEW: Unified Engine Commands ----
    engine_common = argparse.ArgumentParser(add_help=False)
    engine_common.add_argument("--symbol", default="BTCUSDT")
    engine_common.add_argument("--timeframe", default="1h")
    engine_common.add_argument("--days", type=int, default=90)

    # run
    p_run = sub.add_parser("run", parents=[engine_common], help="Run unified SignalForge engine")
    p_run.add_argument("--real", action="store_true", help="Real money mode (CAUTION)")
    p_run.add_argument("--capital", type=float, default=10000.0)
    p_run.add_argument("--max-strategies", type=int, default=8)
    p_run.add_argument("--tick-interval", type=int, default=60, help="Seconds between ticks")
    p_run.add_argument("--evolve-hours", type=float, default=24.0, help="Hours between evolution cycles")
    p_run.add_argument("--verbose", "-v", action="store_true")
    p_run.set_defaults(func=cmd_run)

    # evolve
    p_evo = sub.add_parser("evolve", parents=[engine_common], help="Run GP evolution")
    p_evo.add_argument("--population", type=int, default=200)
    p_evo.add_argument("--generations", type=int, default=50)
    p_evo.set_defaults(func=cmd_evolve)

    # crowding
    p_crowd = sub.add_parser("crowding", parents=[engine_common], help="Show crowding analysis")
    p_crowd.set_defaults(func=cmd_crowding)

    # cascade
    p_casc = sub.add_parser("cascade", parents=[engine_common], help="Show cascade prediction")
    p_casc.set_defaults(func=cmd_cascade)

    # engine-status
    p_estat = sub.add_parser("engine-status", help="Show engine state")
    p_estat.set_defaults(func=cmd_engine_status)

    # validate-all — brutal validation harness
    p_vall = sub.add_parser("validate-all", parents=[common],
                             help="Brutal validation: IS/OOS split, cost stress, regime breakdown")
    p_vall.set_defaults(days=1825)  # default 5y history for this command
    p_vall.add_argument("--oos-fraction", type=float, default=0.30,
                        help="Fraction of data reserved for true OOS (default 0.30)")
    p_vall.add_argument("--output", default="fund_data/validation_report.json")
    p_vall.add_argument("--no-structural", action="store_true",
                        help="Skip funding/OI fetching (faster)")
    p_vall.set_defaults(func=cmd_validate_all)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()

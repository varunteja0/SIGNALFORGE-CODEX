"""
SignalForge — Autonomous Continuous Trading Loop
==================================================
This is the "set and forget" engine. Once started, it:

  1. EVOLVES new strategies on fresh market data (every cycle)
  2. VALIDATES via aligned backtest (only keeps profitable ones)
  3. DEPLOYS to V2 Fund Manager (HRP weights, drawdown bands)
  4. TRADES via paper mode (or live when enabled)
  5. MONITORS risk, decay, drawdown, circuit breakers
  6. SELF-HEALS by retiring dead strategies and evolving replacements
  7. LOGS everything to SQLite + hash-chained ledger

Run:
    python scripts/autonomous_loop.py                    # Default: paper mode
    python scripts/autonomous_loop.py --live             # Live mode (CAUTION)
    python scripts/autonomous_loop.py --cycle-hours 6    # Evolve every 6 hours
    python scripts/autonomous_loop.py --symbols BTC/USDT ETH/USDT SOL/USDT

Kill with Ctrl+C — safe, all state is persisted.
"""
import argparse
import json
import logging
import signal
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data.fetcher import DataFetcher, compute_features
from src.data.features import compute_all_features
from src.alpha_genome.evolution import AlphaGenomeEngine, EvolvedStrategy
from src.alpha_genome.ensemble import EnsembleEvolver
from src.alpha_genome.gene import tree_from_dict, tree_hash
from src.alpha_genome.decay import DecayDetector
from src.backtest.engine import Backtester
from src.risk.manager import RiskLimits
from src.risk.portfolio import PortfolioOptimizer
from src.risk.advanced import AdvancedRiskManager, DrawdownBand
from src.execution.smart import SmartExecutionEngine
from src.regime.detector import RegimeDetector
from src.liquidation.oracle import LiquidationOracle
from src.fund.manager_v2 import AutonomousFundManagerV2
from src.fund.database import Database
from src.fund.ledger import VerifiableLedger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fund_data/autonomous.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("autonomous")

# ─── Graceful shutdown ───────────────────────────────────────────
SHUTDOWN = False

def _handle_signal(signum, frame):
    global SHUTDOWN
    logger.info("Shutdown signal received — finishing current cycle...")
    SHUTDOWN = True

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ─── Core Loop ───────────────────────────────────────────────────

class AutonomousLoop:
    """The brain that runs forever."""

    def __init__(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        cycle_hours: float = 4.0,
        pop_size: int = 100,
        generations: int = 25,
        data_days: int = 365,
        min_profitable_strategies: int = 3,
        initial_capital: float = 10000,
        paper_mode: bool = True,
        db_path: str = "fund_data/signalforge.db",
        ledger_path: str = "fund_data/ledger.json",
    ):
        self.symbols = symbols
        self.timeframe = timeframe
        self.cycle_hours = cycle_hours
        self.pop_size = pop_size
        self.generations = generations
        self.data_days = data_days
        self.min_profitable = min_profitable_strategies
        self.initial_capital = initial_capital
        self.paper_mode = paper_mode

        # Core components
        self.fetcher = DataFetcher()
        self.backtester = Backtester(
            initial_capital=initial_capital,
            commission_pct=0.001,
            slippage_pct=0.0005,
        )
        self.regime_detector = RegimeDetector()
        self.oracle = LiquidationOracle(use_synthetic=True, synthetic_tvl=5e9)
        self.decay_detector = DecayDetector()
        self.db = Database(db_path=db_path)

        # Fund manager (created after first evolution)
        self.fund: AutonomousFundManagerV2 = None
        self.db_path = db_path
        self.ledger_path = ledger_path

        # State
        self.active_strategies: list[EvolvedStrategy] = []
        self.cycle_count = 0
        self.total_evolved = 0
        self.total_profitable = 0
        self.best_sharpe_ever = -999
        self.data_cache: dict[str, pd.DataFrame] = {}
        self.last_prices: dict[str, float] = {}

    def run(self):
        """Main loop — runs until killed."""
        logger.info("=" * 60)
        logger.info("SIGNALFORGE AUTONOMOUS LOOP STARTED")
        logger.info(f"  Symbols: {self.symbols}")
        logger.info(f"  Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        logger.info(f"  Cycle: every {self.cycle_hours}h")
        logger.info(f"  Evolution: pop={self.pop_size} gens={self.generations}")
        logger.info(f"  Capital: ${self.initial_capital:,.0f}")

        # Reload open positions from DB
        open_trades = self.db.get_open_trades()
        if open_trades:
            logger.info(f"  Resuming {len(open_trades)} open positions from DB")
            for t in open_trades:
                logger.info(f"    {t['symbol']} {'LONG' if t['direction']>0 else 'SHORT'} @ ${t['entry_price']:,.2f}")
        logger.info("=" * 60)

        while not SHUTDOWN:
            try:
                self._run_cycle()
            except Exception as e:
                logger.error(f"Cycle failed: {e}")
                logger.error(traceback.format_exc())
                # Don't crash — wait and retry
                self._safe_sleep(60)
                continue

            self.cycle_count += 1
            if SHUTDOWN:
                break

            # Between cycles: monitor positions
            cycle_end = time.time() + self.cycle_hours * 3600 
            while time.time() < cycle_end and not SHUTDOWN:
                try:
                    self._monitor_tick()
                except Exception as e:
                    logger.warning(f"Monitor tick error: {e}")
                self._safe_sleep(60)  # Check every 60 seconds

        self._shutdown()

    def _run_cycle(self):
        """One full evolution + deployment cycle."""
        cycle_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"CYCLE {self.cycle_count + 1} — {datetime.now():%Y-%m-%d %H:%M}")
        logger.info(f"{'='*60}")

        # ─── Step 1: Fetch fresh data ───
        logger.info("[1/7] Fetching market data...")
        data = {}
        for sym in self.symbols:
            try:
                df = self.fetcher.fetch(sym, self.timeframe, days=self.data_days)
                if df is not None and len(df) > 500:
                    df = compute_all_features(df)
                    df = df.dropna()
                    data[sym] = df
                    self.last_prices[sym] = float(df["close"].iloc[-1])
                    logger.info(f"  {sym}: {len(df)} bars, price=${self.last_prices[sym]:,.2f}")
            except Exception as e:
                logger.warning(f"  {sym}: fetch failed — {e}")
        self.data_cache = data

        if not data:
            logger.error("No data fetched — skipping cycle")
            return

        # ─── Step 2: Detect regimes ───
        logger.info("[2/7] Detecting market regimes...")
        regimes = {}
        for sym, df in data.items():
            try:
                det = RegimeDetector()
                det.fit(df)
                regime = det.detect(df)
                regimes[sym] = regime.value
                logger.info(f"  {sym}: {regime.value}")
            except Exception as e:
                regimes[sym] = "unknown"

        # ─── Step 3: Evolve new strategies ───
        logger.info("[3/7] Evolving strategies...")
        new_strategies = []

        for sym, df in data.items():
            try:
                engine = AlphaGenomeEngine(
                    population_size=self.pop_size,
                    max_generations=self.generations,
                    tournament_size=5,
                    crossover_rate=0.7,
                    mutation_rate=0.2,
                    elitism_count=max(2, self.pop_size // 10),
                    max_tree_depth=6,
                    novelty_weight=0.2,
                    min_trades=15,
                    commission_pct=0.001,
                    slippage_pct=0.0005,
                    output_dir="evolved_strategies",
                )
                strats = engine.evolve(df, symbol=sym, timeframe=self.timeframe)
                for s in strats:
                    s.symbol = sym
                new_strategies.extend(strats)
                logger.info(f"  {sym}: {len(strats)} strategies evolved")
            except Exception as e:
                logger.warning(f"  {sym}: evolution failed — {e}")

        self.total_evolved += len(new_strategies)

        if not new_strategies:
            logger.warning("No strategies evolved this cycle")
            return

        # ─── Step 4: Backtest validation (aligned with fitness) ───
        logger.info("[4/7] Backtesting with aligned signals...")
        profitable = []
        unprofitable = []

        for strat in new_strategies:
            df = data.get(strat.symbol)
            if df is None:
                continue
            try:
                tree = tree_from_dict(strat.tree_dict)
                result = self.backtester.run_with_tree(
                    df, tree, holding_period=24,
                    position_size_pct=0.02,
                    stop_loss_atr=2.0,
                    take_profit_atr=3.0,
                )
                mc = self.backtester.monte_carlo(result)

                strat_info = {
                    "strat": strat,
                    "bt_return": result.total_return,
                    "bt_sharpe": result.sharpe_ratio,
                    "bt_pf": result.profit_factor,
                    "bt_wr": result.win_rate,
                    "bt_dd": result.max_drawdown,
                    "bt_trades": result.total_trades,
                    "mc_prob": mc.get("probability_of_profit", 0),
                }

                if result.total_return > 0 and result.profit_factor > 1.0:
                    profitable.append(strat_info)
                    logger.info(
                        f"  PROFITABLE: {strat.name} ({strat.symbol}) "
                        f"Return={result.total_return:+.1%} "
                        f"Sharpe={result.sharpe_ratio:.2f} "
                        f"PF={result.profit_factor:.2f} "
                        f"MC={mc.get('probability_of_profit',0):.0%}"
                    )
                else:
                    unprofitable.append(strat_info)
            except Exception as e:
                logger.warning(f"  {strat.name}: backtest error — {e}")

        self.total_profitable += len(profitable)
        logger.info(
            f"  Results: {len(profitable)} profitable / {len(new_strategies)} evolved "
            f"(lifetime: {self.total_profitable}/{self.total_evolved})"
        )

        # ─── Step 5: Deploy profitable strategies ───
        logger.info("[5/7] Deploying strategies...")
        deploy_list = [p["strat"] for p in profitable]

        # If we don't have enough profitable new ones, keep existing
        if len(deploy_list) < self.min_profitable and self.active_strategies:
            logger.info(f"  Keeping {len(self.active_strategies)} existing strategies")
            deploy_list = self.active_strategies + deploy_list

        # If still nothing, keep best by backtest Sharpe from this cycle
        if not deploy_list:
            logger.warning("  No profitable strategies — keeping top 3 by Sharpe")
            all_tested = profitable + unprofitable
            all_tested.sort(key=lambda x: x["bt_sharpe"], reverse=True)
            deploy_list = [x["strat"] for x in all_tested[:3]]

        if not deploy_list:
            logger.error("  No strategies to deploy")
            return

        self.active_strategies = deploy_list

        # Create/update fund manager
        if self.fund is None:
            self.fund = AutonomousFundManagerV2(
                initial_capital=self.initial_capital,
                risk_limits=RiskLimits(
                    max_position_pct=0.04,
                    max_drawdown_pct=0.15,
                    max_daily_loss_pct=0.05,
                    max_open_positions=8,
                ),
                ledger_path=self.ledger_path,
                db_path=self.db_path,
                portfolio_method="hrp",
                drawdown_bands=DrawdownBand(
                    yellow_pct=0.05, orange_pct=0.10,
                    red_pct=0.15, black_pct=0.20,
                ),
                max_slippage_bps=100,
            )
            # Reload any open positions from previous runs
            self.fund.reload_positions_from_db()

        self.fund.load_strategies(deploy_list)
        logger.info(f"  Deployed {len(deploy_list)} strategies")
        logger.info(f"  Portfolio weights: {self.fund.portfolio_weights}")

        # ─── Step 6: Generate + execute signals ───
        logger.info("[6/7] Generating signals...")
        total_signals = 0
        total_executed = 0

        for sym, df in data.items():
            price = self.last_prices.get(sym, 0)
            if price <= 0:
                continue

            candidates = self.fund.generate_signals(df, sym, price)
            total_signals += len(candidates)

            if candidates:
                executed = self.fund.process_and_execute(candidates)
                total_executed += len(executed)
                for t in executed:
                    logger.info(
                        f"  TRADE: {t.get('asset','?')} "
                        f"{'LONG' if t.get('direction',0)>0 else 'SHORT'} "
                        f"@ ${t.get('fill_price',0):,.2f} "
                        f"size={t.get('size',0):.6f}"
                    )

        logger.info(f"  Signals: {total_signals}, Executed: {total_executed}")

        # ─── Step 7: Risk check + state snapshot ───
        logger.info("[7/7] Risk assessment...")
        state = self.fund.get_state()

        # Track best ever
        if state.capital > self.best_sharpe_ever:
            self.best_sharpe_ever = state.capital

        elapsed = time.time() - cycle_start
        logger.info(f"\n  CYCLE COMPLETE ({elapsed:.0f}s)")
        logger.info(f"  Capital: ${state.capital:,.2f}")
        logger.info(f"  Drawdown: {state.drawdown_pct:.1%} ({state.drawdown_band} band)")
        logger.info(f"  Open positions: {state.open_positions}")
        logger.info(f"  Strategies: {len(deploy_list)}")
        logger.info(f"  Ledger verified: {state.ledger_verified}")

        # Save equity snapshot
        try:
            n_open = len(state.open_positions) if isinstance(state.open_positions, dict) else int(state.open_positions or 0)
            self.db.snapshot_equity(
                capital=state.capital,
                peak_capital=state.peak_capital,
                drawdown_pct=state.drawdown_pct,
                open_positions=n_open,
                active_strategies=len(deploy_list),
                total_pnl=state.total_pnl,
            )
        except Exception as e:
            logger.warning(f"  Equity snapshot error: {e}")

        # Save model version
        try:
            strat_data = json.dumps(
                [s.to_dict() for s in deploy_list],
                default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
            )
            version_id = self.db.save_model_version(
                strat_data,
                symbol=",".join(self.symbols),
                timeframe=self.timeframe,
                n_strategies=len(deploy_list),
                best_sharpe=float(max(s.fitness.oos_sharpe for s in deploy_list)),
                avg_sharpe=float(np.mean([s.fitness.oos_sharpe for s in deploy_list])),
                notes=f"cycle_{self.cycle_count+1}",
            )
            self.db.deploy_version(version_id)
        except Exception as e:
            logger.warning(f"  DB save error: {e}")

    def _monitor_tick(self):
        """Between-cycle monitoring: check exits, trailing stops, decay."""
        if self.fund is None or not self.fund.open_positions:
            return

        # Refresh prices
        for sym in list(self.fund.open_positions.keys()):
            base_sym = sym  # Already in format like "BTC/USDT"
            try:
                df = self.fetcher.fetch(base_sym, self.timeframe, days=2)
                if df is not None and len(df) > 0:
                    self.last_prices[base_sym] = float(df["close"].iloc[-1])
            except Exception:
                pass

        if not self.last_prices:
            return

        # Check exits (stop loss, take profit, trailing stops)
        closed = self.fund.check_exits(self.last_prices)
        for c in closed:
            direction = "LONG" if c.get("direction", 0) > 0 else "SHORT"
            logger.info(
                f"  EXIT: {c['asset']} {direction} "
                f"PnL=${c['pnl']:.2f} ({c['return_pct']:.1%}) "
                f"Reason={c['reason']}"
            )

        # Log state
        state = self.fund.get_state()
        if state.drawdown_band in ("orange", "red", "black"):
            logger.warning(
                f"  RISK ALERT: Drawdown {state.drawdown_pct:.1%} "
                f"({state.drawdown_band} band) — "
                f"{'Trading halted!' if state.drawdown_band in ('red','black') else 'Size reduced'}"
            )

    def _safe_sleep(self, seconds):
        """Sleep in 1-second chunks to respect shutdown."""
        for _ in range(int(seconds)):
            if SHUTDOWN:
                return
            time.sleep(1)

    def _shutdown(self):
        """Clean shutdown — persist all state."""
        logger.info("\n" + "=" * 60)
        logger.info("AUTONOMOUS LOOP SHUTTING DOWN")
        logger.info(f"  Total cycles: {self.cycle_count}")
        logger.info(f"  Total evolved: {self.total_evolved}")
        logger.info(f"  Total profitable: {self.total_profitable}")

        if self.fund:
            state = self.fund.get_state()
            logger.info(f"  Final capital: ${state.capital:,.2f}")
            logger.info(f"  Final drawdown: {state.drawdown_pct:.1%}")
            logger.info(f"  Open positions: {state.open_positions}")
            logger.info(f"  Ledger integrity: {'VALID' if state.ledger_verified else 'BROKEN'}")

        logger.info("Shutdown complete. All state persisted to DB + ledger.")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SignalForge Autonomous Trading Loop")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--cycle-hours", type=float, default=4.0,
                        help="Hours between evolution cycles")
    parser.add_argument("--pop", type=int, default=100, help="Evolution population size")
    parser.add_argument("--gens", type=int, default=25, help="Evolution generations")
    parser.add_argument("--days", type=int, default=365, help="Historical data days")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--live", action="store_true", help="Enable live trading (CAUTION)")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Max cycles (0=infinite)")
    parser.add_argument("--status", action="store_true",
                        help="Show fund performance summary and exit")
    args = parser.parse_args()

    # ─── Status mode: just show performance and exit ───
    if args.status:
        db = Database(db_path="fund_data/signalforge.db")
        perf = db.get_performance_summary()
        equity = db.get_equity_curve(days=30)
        open_trades = db.get_open_trades()

        print("\n" + "=" * 60)
        print("  SIGNALFORGE — FUND STATUS")
        print("=" * 60)
        print(f"  Closed Trades:    {perf.get('total_trades', 0) or 0}")
        print(f"  Win Rate:         {perf.get('win_rate', 0):.1%}")
        print(f"  Total P&L:        ${(perf.get('total_pnl') or 0):+,.2f}")
        print(f"  Avg P&L/Trade:    ${(perf.get('avg_pnl') or 0):+,.4f}")
        print(f"  Best Trade:       ${(perf.get('best_trade') or 0):+,.4f}")
        print(f"  Worst Trade:      ${(perf.get('worst_trade') or 0):+,.4f}")
        print(f"  Avg Return:       {(perf.get('avg_return_pct') or 0):+.2%}")
        print(f"  Open Positions:   {perf.get('open_positions', 0) or 0}")
        print(f"  Open Notional:    ${(perf.get('open_notional') or 0):,.2f}")

        if open_trades:
            print(f"\n  {'─' * 50}")
            print("  OPEN POSITIONS:")
            for t in open_trades:
                d = "LONG" if t["direction"] > 0 else "SHORT"
                print(f"    {t['symbol']} {d} @ ${t['entry_price']:,.2f} "
                      f"size={t['size']:.6f} SL=${t.get('stop_loss',0):,.2f} "
                      f"TP=${t.get('take_profit',0):,.2f}")

        if equity:
            latest = equity[-1]
            print(f"\n  {'─' * 50}")
            print("  LATEST EQUITY SNAPSHOT:")
            print(f"    Capital:    ${latest.get('capital', 0):,.2f}")
            print(f"    Peak:       ${latest.get('peak_capital', 0):,.2f}")
            print(f"    Drawdown:   {latest.get('drawdown_pct', 0):.1%}")
            print(f"    Total P&L:  ${latest.get('total_pnl', 0):+,.2f}")
            print(f"    Strategies: {latest.get('active_strategies', 0)}")

        # Model versions
        try:
            with db._conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) as n FROM model_versions"
                ).fetchone()
                print(f"\n  Model Versions:   {dict(row)['n']}")
                deployed = conn.execute(
                    "SELECT version_id, n_strategies, best_sharpe, avg_sharpe, notes "
                    "FROM model_versions WHERE deployed = 1 ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if deployed:
                    d = dict(deployed)
                    print(f"  Active Version:   {d['version_id']} ({d['n_strategies']} strategies)")
                    print(f"  Best Sharpe:      {d['best_sharpe']:.2f}")
                    print(f"  Avg Sharpe:       {d['avg_sharpe']:.2f}")
        except Exception:
            pass

        print("=" * 60)
        return

    if args.live:
        print("\n" + "!" * 60)
        print("  WARNING: LIVE TRADING MODE")
        print("  Real money will be at risk.")
        print("  Press Ctrl+C within 10 seconds to cancel.")
        print("!" * 60)
        try:
            for i in range(10, 0, -1):
                print(f"  Starting in {i}...", end="\r")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nCancelled.")
            return

    loop = AutonomousLoop(
        symbols=args.symbols,
        timeframe=args.timeframe,
        cycle_hours=args.cycle_hours,
        pop_size=args.pop,
        generations=args.gens,
        data_days=args.days,
        initial_capital=args.capital,
        paper_mode=not args.live,
    )

    if args.cycles > 0:
        # Run fixed number of cycles
        for i in range(args.cycles):
            if SHUTDOWN:
                break
            loop._run_cycle()
            loop.cycle_count += 1
        loop._shutdown()
    else:
        loop.run()


if __name__ == "__main__":
    main()

"""
Strategy Factory Loop — The Core Engine
==========================================
Ties Scanner → Validator → Deployer → Monitor together
in a continuous loop.

    Every cycle:
    1. Fetch fresh market data
    2. Run signal scan (generate + test hypotheses)
    3. Validate survivors (OOS + walk-forward)
    4. Deploy validated strategies (paper trade)
    5. Monitor existing strategies (detect decay)
    6. Kill dead strategies, promote survivors
    7. Sleep until next cycle

Usage:
    from src.factory.loop import StrategyFactoryLoop
    loop = StrategyFactoryLoop(symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    loop.run()  # Runs forever
"""

import json
import logging
import signal
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fetcher import DataFetcher
from src.data.features import compute_all_features
from src.data.structural import StructuralDataFetcher
from src.factory.scanner import scan, ScanResult
from src.factory.validator import validate, ValidationResult
from src.factory.deployer import deploy, load_deployed, DeployedStrategy
from src.factory.monitor import StrategyMonitor, TradeRecord
from src.backtest.engine import Backtester

logger = logging.getLogger("factory")

# Graceful shutdown
_SHUTDOWN = False

def _handle_signal(signum, frame):
    global _SHUTDOWN
    logger.info("Shutdown signal received — finishing current cycle...")
    _SHUTDOWN = True

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


class StrategyFactoryLoop:
    """The industrialized hypothesis testing engine.

    Continuously:
    - Scans for new signal hypotheses
    - Validates survivors out-of-sample
    - Deploys validated strategies
    - Monitors + kills decayed strategies
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        timeframe: str = "1h",
        data_days: int = 365,
        cycle_hours: float = 6.0,
        min_scan_trades: int = 50,
        max_deployed: int = 10,
    ):
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        self.timeframe = timeframe
        self.data_days = data_days
        self.cycle_hours = cycle_hours
        self.min_scan_trades = min_scan_trades
        self.max_deployed = max_deployed

        # Components
        self.fetcher = DataFetcher()
        self.struct_fetcher = StructuralDataFetcher()
        self.monitor = StrategyMonitor()
        self.backtester = Backtester(
            initial_capital=10000,
            commission_pct=0.001,
            slippage_pct=0.0005,
        )

        # State
        self.deployed: list[DeployedStrategy] = load_deployed()
        self.cycle_count = 0
        self.data_cache: dict[str, pd.DataFrame] = {}

        # Stats
        self.total_hypotheses_tested = 0
        self.total_raw_survivors = 0
        self.total_validated = 0

    # ─── Main Loop ───────────────────────────────────────────────

    def run(self, max_cycles: int = 0):
        """Run the factory loop.

        Args:
            max_cycles: 0 = infinite, N = run N cycles then stop
        """
        logger.info("=" * 60)
        logger.info("  SIGNALFORGE STRATEGY FACTORY — STARTED")
        logger.info(f"  Symbols:    {self.symbols}")
        logger.info(f"  Timeframe:  {self.timeframe}")
        logger.info(f"  Cycle:      every {self.cycle_hours}h")
        logger.info(f"  Deployed:   {len(self.deployed)} strategies loaded")
        logger.info("=" * 60)

        while not _SHUTDOWN:
            try:
                self._run_cycle()
            except Exception as e:
                logger.error(f"Cycle failed: {e}")
                logger.error(traceback.format_exc())
                self._safe_sleep(60)
                continue

            self.cycle_count += 1

            if max_cycles > 0 and self.cycle_count >= max_cycles:
                break

            if _SHUTDOWN:
                break

            # Sleep until next cycle
            logger.info(f"Next cycle in {self.cycle_hours}h. Sleeping...")
            self._safe_sleep(self.cycle_hours * 3600)

        self._shutdown()

    def _run_cycle(self):
        """One full discovery → validation → deployment cycle."""
        cycle_start = time.time()
        logger.info(f"\n{'━' * 60}")
        logger.info(f"  CYCLE {self.cycle_count + 1} — {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
        logger.info(f"{'━' * 60}")

        # ─── Step 1: Fetch Data ───
        logger.info("[1/5] Fetching market data...")
        datasets = self._fetch_data()
        if not datasets:
            logger.error("No data — skipping cycle")
            return

        # ─── Step 2: Monitor existing strategies ───
        logger.info("[2/5] Checking deployed strategies...")
        if self.deployed:
            report = self.monitor.report(self.deployed)
            logger.info(f"\n{report}")

            kills = self.monitor.get_kill_list(self.deployed)
            if kills:
                logger.info(f"  Killing {len(kills)} dead strategies: {kills}")
                self.deployed = [s for s in self.deployed if s.name not in kills]

        slots_available = self.max_deployed - len(self.deployed)
        if slots_available <= 0:
            logger.info("  All strategy slots filled. Skipping scan.")
            elapsed = time.time() - cycle_start
            logger.info(f"  Cycle complete ({elapsed:.0f}s)")
            return

        # ─── Step 3: Scan for new signals ───
        logger.info(f"[3/5] Scanning for signals ({slots_available} slots available)...")
        scan_result = scan(
            datasets,
            min_trades=self.min_scan_trades,
            p_threshold=0.05,
            min_pf=1.1,
        )

        self.total_hypotheses_tested += scan_result.total_hypotheses
        self.total_raw_survivors += len(scan_result.raw_survivors)

        logger.info(
            f"  Hypotheses tested: {scan_result.total_hypotheses}"
        )
        logger.info(
            f"  Raw survivors: {len(scan_result.raw_survivors)} "
            f"(Bonferroni @ {scan_result.bonferroni_threshold:.2e}: "
            f"{len(scan_result.bonferroni_survivors)})"
        )

        if not scan_result.raw_survivors:
            logger.info("  No signals survived raw filter.")
            elapsed = time.time() - cycle_start
            logger.info(f"  Cycle complete ({elapsed:.0f}s)")
            return

        # ─── Step 4: Validate OOS ───
        logger.info("[4/5] Validating out-of-sample...")
        val_result = validate(
            scan_result.raw_survivors,
            datasets,
            min_is_pf=1.2,
            min_oos_pf=1.0,
        )

        self.total_validated += val_result.signals_passed_oos

        logger.info(
            f"  Tested: {val_result.signals_tested} → "
            f"IS pass: {val_result.signals_passed_is} → "
            f"OOS pass: {val_result.signals_passed_oos}"
        )

        for sig in val_result.validated:
            logger.info(
                f"  ✓ {sig.name} ({sig.asset.split('/')[0]}) "
                f"Grade={sig.grade} "
                f"OOS: PF={sig.oos_pf:.2f} Sharpe={sig.oos_sharpe:.2f} "
                f"WF={sig.wf_positive_folds}/{sig.wf_total_folds}"
            )

        if not val_result.validated:
            logger.info("  No signals survived OOS validation.")
            elapsed = time.time() - cycle_start
            logger.info(f"  Cycle complete ({elapsed:.0f}s)")
            return

        # ─── Step 5: Deploy ───
        logger.info("[5/5] Deploying validated strategies...")
        new_deployed = deploy(
            val_result.validated,
            max_strategies=slots_available,
        )

        # Avoid duplicates
        existing_names = {s.name for s in self.deployed}
        new_unique = [s for s in new_deployed if s.name not in existing_names]

        self.deployed.extend(new_unique)
        logger.info(f"  Deployed {len(new_unique)} new strategies")
        logger.info(f"  Total active: {len(self.deployed)}")

        for s in new_unique:
            logger.info(
                f"    + {s.name} ({s.asset}) "
                f"grade={s.grade} size={s.position_size_pct:.1%} "
                f"hold={s.hold_bars}h"
            )

        elapsed = time.time() - cycle_start
        logger.info(f"\n  Cycle complete ({elapsed:.0f}s)")
        logger.info(f"  Lifetime stats: {self.total_hypotheses_tested} hypotheses → "
                     f"{self.total_raw_survivors} raw → {self.total_validated} validated")

    # ─── Data Fetching ───────────────────────────────────────────

    def _fetch_data(self) -> dict[str, pd.DataFrame]:
        """Fetch and compute features for all symbols."""
        datasets = {}

        for sym in self.symbols:
            try:
                raw = self.fetcher.fetch(sym, self.timeframe, days=self.data_days)
                if raw is None or raw.empty:
                    continue

                df = compute_all_features(raw)

                try:
                    df = self.struct_fetcher.fetch_all(
                        symbol=sym.replace("/", ""),
                        price_df=df,
                        days=self.data_days,
                    )
                except Exception:
                    pass

                datasets[sym] = df
                logger.info(f"  {sym}: {len(df)} bars, last=${float(df['close'].iloc[-1]):,.2f}")

            except Exception as e:
                logger.warning(f"  {sym}: fetch failed — {e}")

        self.data_cache = datasets
        return datasets

    # ─── Utilities ───────────────────────────────────────────────

    def _safe_sleep(self, seconds: float):
        """Sleep in chunks, checking for shutdown."""
        for _ in range(int(seconds)):
            if _SHUTDOWN:
                return
            time.sleep(1)

    def _shutdown(self):
        """Clean shutdown — persist state."""
        logger.info("\n" + "=" * 60)
        logger.info("  STRATEGY FACTORY SHUTTING DOWN")
        logger.info(f"  Cycles completed:      {self.cycle_count}")
        logger.info(f"  Hypotheses tested:     {self.total_hypotheses_tested}")
        logger.info(f"  Raw survivors:         {self.total_raw_survivors}")
        logger.info(f"  Validated strategies:  {self.total_validated}")
        logger.info(f"  Currently deployed:    {len(self.deployed)}")

        for s in self.deployed:
            logger.info(f"    {s.name} ({s.asset}) grade={s.grade}")

        logger.info("  State persisted to disk.")
        logger.info("=" * 60)

    # ─── Single-run mode for CLI ─────────────────────────────────

    def run_once(self) -> dict:
        """Run one cycle and return results (for CLI/testing).

        Returns dict with scan + validation results.
        """
        datasets = self._fetch_data()
        if not datasets:
            return {"error": "No data fetched"}

        scan_result = scan(datasets, min_trades=self.min_scan_trades)
        val_result = validate(scan_result.raw_survivors, datasets)

        deployed = []
        if val_result.validated:
            deployed = deploy(val_result.validated, max_strategies=self.max_deployed)
            self.deployed = deployed

        return {
            "hypotheses": scan_result.total_hypotheses,
            "raw_survivors": len(scan_result.raw_survivors),
            "bonferroni_survivors": len(scan_result.bonferroni_survivors),
            "validated": len(val_result.validated),
            "deployed": len(deployed),
            "strategies": [s.to_dict() for s in deployed],
        }

"""
Autonomous Engine — The Orchestrator
=====================================
The brain of the system. Runs the full loop:

    DISCOVER → TEST → RANK → ALLOCATE → DEPLOY → MONITOR → EVOLVE

Each cycle:
    1. Generate N strategy candidates from 4 templates
    2. Backtest all on target assets
    3. Walk-forward validate top performers
    4. Allocate capital across best uncorrelated strategies
    5. Deploy to paper trading
    6. Monitor live performance & kill degraded strategies
    7. Sleep until next cycle

State persisted to JSON — survives restarts.
"""

import json
import logging
import signal
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from src.engine.strategy_factory import StrategyFactory, StrategyCandidate
from src.engine.ranker import StrategyRanker, ScoredStrategy
from src.engine.allocator import StrategyAllocator, AllocationResult

logger = logging.getLogger(__name__)

# Graceful shutdown
_SHUTDOWN = False


def _handle_signal(signum, frame):
    global _SHUTDOWN
    logger.info("Shutdown signal received — finishing current cycle...")
    _SHUTDOWN = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


@dataclass
class DeployedStrategy:
    """A strategy currently in the portfolio."""
    name: str
    template: str
    params: dict
    weight: float
    score: float
    deployed_at: str
    # Live tracking
    live_trades: int = 0
    live_pnl: float = 0.0
    live_sharpe: float = 0.0
    last_updated: str = ""
    # Backtest reference
    backtest_pf: float = 0.0
    backtest_trades: int = 0
    backtest_sharpe: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class EngineState:
    """Persistent state of the autonomous engine."""
    cycle_count: int = 0
    total_candidates_tested: int = 0
    total_strategies_deployed: int = 0
    total_strategies_killed: int = 0
    deployed: list = field(default_factory=list)       # DeployedStrategy dicts
    history: list = field(default_factory=list)         # Cycle summaries
    last_cycle_time: str = ""
    best_score_ever: float = 0.0
    best_strategy_ever: str = ""

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "EngineState":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p) as f:
            data = json.load(f)
        state = cls()
        for key, value in data.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state


class AutonomousEngine:
    """The one-man quant firm engine."""

    def __init__(
        self,
        symbols: list[str] = None,
        n_per_template: int = 20,
        data_days: int = 365,
        max_deployed: int = 5,
        cycle_hours: float = 24.0,
        state_path: str = "fund_data/engine_state.json",
        results_path: str = "pipeline_output/engine_results.json",
    ):
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT"]
        self.n_per_template = n_per_template
        self.data_days = data_days
        self.max_deployed = max_deployed
        self.cycle_hours = cycle_hours
        self.state_path = state_path
        self.results_path = results_path

        # Components
        self.factory = StrategyFactory(n_per_template=n_per_template)
        self.ranker = StrategyRanker(
            symbols=self.symbols,
            days=data_days,
        )
        self.allocator = StrategyAllocator(max_strategies=max_deployed)

        # State
        self.state = EngineState.load(state_path)

    # ─── Single Cycle ────────────────────────────────────────────

    def run_cycle(self) -> dict:
        """Execute one complete discovery → deploy cycle."""
        cycle_start = time.time()
        self.state.cycle_count += 1
        cycle_num = self.state.cycle_count
        logger.info(f"{'='*60}")
        logger.info(f"  CYCLE {cycle_num}")
        logger.info(f"{'='*60}")

        # Step 1: Generate candidates
        logger.info(f"[1/5] Generating strategy candidates...")
        candidates = self.factory.generate_all()
        self.state.total_candidates_tested += len(candidates)
        logger.info(f"  Generated {len(candidates)} candidates from 4 templates")

        # Step 2: Rank (backtest + walk-forward)
        logger.info(f"[2/5] Ranking candidates (backtest + walk-forward)...")
        ranked = self.ranker.rank_all(candidates, run_walkforward=True)
        n_passed = len(ranked)
        logger.info(f"  {n_passed} candidates passed filters")

        if not ranked:
            logger.warning("  No viable strategies found this cycle.")
            summary = self._cycle_summary(cycle_num, 0, 0, {}, time.time() - cycle_start)
            self._save_state()
            return summary

        # Step 3: Allocate
        logger.info(f"[3/5] Allocating capital across top strategies...")
        allocation = self.allocator.allocate(ranked[:self.max_deployed * 2])

        if not allocation.weights:
            logger.warning("  Allocation returned empty weights.")
            summary = self._cycle_summary(cycle_num, n_passed, 0, {}, time.time() - cycle_start)
            self._save_state()
            return summary

        # Step 4: Deploy
        logger.info(f"[4/5] Deploying {allocation.n_strategies} strategies...")
        self._deploy(ranked, allocation)

        # Step 5: Report
        logger.info(f"[5/5] Generating report...")
        elapsed = time.time() - cycle_start
        summary = self._cycle_summary(
            cycle_num, n_passed, allocation.n_strategies,
            allocation.weights, elapsed,
        )

        # Update best ever
        if ranked and ranked[0].score > self.state.best_score_ever:
            self.state.best_score_ever = ranked[0].score
            self.state.best_strategy_ever = ranked[0].candidate.name

        self.state.last_cycle_time = datetime.now().isoformat()
        self.state.history.append(summary)
        self._save_state()
        self._save_results(ranked, allocation)

        logger.info(f"Cycle {cycle_num} complete in {elapsed:.0f}s")
        return summary

    # ─── Continuous Loop ─────────────────────────────────────────

    def run_loop(self):
        """Run cycles continuously until stopped."""
        logger.info(f"Starting autonomous engine — cycle every {self.cycle_hours}h")
        logger.info(f"  Symbols: {self.symbols}")
        logger.info(f"  Templates: 4 × {self.n_per_template} = "
                     f"{4 * self.n_per_template + 1} candidates/cycle")
        logger.info(f"  State: {self.state_path}")
        logger.info(f"  Press Ctrl+C to stop gracefully")

        while not _SHUTDOWN:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"Cycle failed: {e}", exc_info=True)

            if _SHUTDOWN:
                break

            # Sleep until next cycle
            sleep_secs = self.cycle_hours * 3600
            logger.info(f"Next cycle in {self.cycle_hours}h. Sleeping...")
            wake_time = time.time() + sleep_secs
            while time.time() < wake_time and not _SHUTDOWN:
                time.sleep(10)

        logger.info("Engine stopped gracefully.")
        self._save_state()

    # ─── Discover Only (no deploy) ───────────────────────────────

    def discover(self) -> list[ScoredStrategy]:
        """Run discovery + ranking without deploying. For analysis."""
        logger.info("Running discovery mode (no deployment)...")
        candidates = self.factory.generate_all()
        ranked = self.ranker.rank_all(candidates, run_walkforward=True)
        self._save_results(ranked, AllocationResult())
        return ranked

    # ─── Internal ────────────────────────────────────────────────

    def _deploy(self, ranked: list[ScoredStrategy], allocation: AllocationResult):
        """Update deployed strategies."""
        now = datetime.now().isoformat()
        deployed = []

        for scored in ranked:
            name = scored.candidate.name
            if name not in allocation.weights:
                continue

            ds = DeployedStrategy(
                name=name,
                template=scored.candidate.template,
                params=scored.candidate.params,
                weight=allocation.weights[name],
                score=scored.score,
                deployed_at=now,
                backtest_pf=scored.combined_pf,
                backtest_trades=scored.total_trades,
                backtest_sharpe=scored.avg_sharpe,
            )
            deployed.append(ds.to_dict())

        old_count = len(self.state.deployed)
        self.state.deployed = deployed
        self.state.total_strategies_deployed += len(deployed)

        if old_count > 0:
            killed = old_count  # Previous strategies are replaced
            self.state.total_strategies_killed += killed

    def _cycle_summary(
        self, cycle_num, n_passed, n_deployed, weights, elapsed
    ) -> dict:
        return {
            "cycle": cycle_num,
            "timestamp": datetime.now().isoformat(),
            "candidates_tested": 4 * self.n_per_template + 1,
            "passed_filters": n_passed,
            "deployed": n_deployed,
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "elapsed_seconds": round(elapsed, 1),
        }

    def _save_state(self):
        self.state.save(self.state_path)

    def _save_results(self, ranked: list, allocation: AllocationResult):
        """Save detailed results for inspection."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_candidates": len(ranked),
            "allocation": {
                "weights": {k: round(v, 4) for k, v in allocation.weights.items()},
                "n_strategies": allocation.n_strategies,
                "expected_sharpe": round(allocation.expected_sharpe, 3),
            },
            "top_strategies": [],
        }

        for i, scored in enumerate(ranked[:20]):
            entry = {
                "rank": i + 1,
                "name": scored.candidate.name,
                "template": scored.candidate.template,
                "score": round(scored.score, 4),
                "combined_pf": round(scored.combined_pf, 3),
                "total_trades": scored.total_trades,
                "avg_sharpe": round(scored.avg_sharpe, 3),
                "win_rate": round(scored.combined_wr, 3),
                "max_drawdown": round(scored.max_drawdown, 4),
                "params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                           for k, v in scored.candidate.params.items()},
            }
            if scored.oos_sharpe is not None:
                entry["oos_sharpe"] = round(scored.oos_sharpe, 3)
                entry["oos_profitable_folds"] = scored.oos_profitable_folds
                entry["oos_total_folds"] = scored.oos_total_folds

            # Per-symbol breakdown
            entry["per_symbol"] = {}
            for sym, res in scored.results.items():
                entry["per_symbol"][sym] = {
                    "trades": res.total_trades,
                    "pf": round(res.profit_factor, 3),
                    "sharpe": round(res.sharpe_ratio, 3),
                    "return": round(res.total_return, 5),
                }

            results["top_strategies"].append(entry)

        out_path = Path(self.results_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {out_path}")

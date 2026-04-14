"""
Signal Genome v2 — Self-Evolving Alpha Discovery + Auto-Deployment
===================================================================
Upgrades the alpha_genome engine from "find strategies" to
"continuously discover, validate, deploy, monitor, and replace strategies."

This is the autonomous loop that makes the system self-evolving:

    1. DISCOVER: Run GP evolution to find novel alphas
    2. VALIDATE: Walk-forward + institutional validation
    3. DEPLOY: Hot-swap into live portfolio with sizing from Kelly
    4. MONITOR: Track decay via DecayDetector
    5. REPLACE: When alpha dies, trigger new evolution round

The key insight: don't find THE strategy. Find a PIPELINE that
continuously generates strategies faster than they decay.

Components:
    - AlphaRegistry: tracks all discovered, deployed, and retired alphas
    - EvolutionScheduler: triggers evolution at optimal times
    - AutoDeployer: validates and adds new alphas to portfolio
    - DecayMonitor: watches deployed alphas and kills dying ones
    - GenomeOrchestrator: wires everything together
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd

from src.alpha_genome.evolution import AlphaGenomeEngine, EvolvedStrategy
from src.alpha_genome.fitness import FitnessEvaluator, FitnessResult
from src.alpha_genome.decay import DecayDetector, DecayReport
from src.alpha_genome.gene import tree_from_dict, tree_to_formula, Node

logger = logging.getLogger(__name__)


# ─── Alpha Registry ─────────────────────────────────────────────

@dataclass
class AlphaRecord:
    """Complete record of a discovered alpha."""
    name: str
    formula: str
    tree_dict: dict
    fitness: dict
    status: str = "discovered"  # discovered → validated → deployed → decayed → retired
    discovered_at: float = 0.0
    deployed_at: float = 0.0
    retired_at: float = 0.0
    # Live performance
    live_trades: int = 0
    live_pnl: float = 0.0
    live_sharpe: float = 0.0
    decay_score: float = 0.0
    # Metadata
    generation: int = 0
    symbol: str = ""
    novelty_score: float = 0.0


class AlphaRegistry:
    """Persistent registry of all discovered alphas across evolution runs."""

    def __init__(self, path: str = "evolved_strategies/alpha_registry.json"):
        self.path = path
        self.alphas: dict[str, AlphaRecord] = {}
        self._load()

    def register(self, strategy: EvolvedStrategy) -> AlphaRecord:
        """Register a newly discovered alpha."""
        record = AlphaRecord(
            name=strategy.name,
            formula=strategy.formula,
            tree_dict=strategy.tree_dict,
            fitness={
                "oos_sharpe": strategy.fitness.oos_sharpe,
                "oos_pf": strategy.fitness.oos_profit_factor,
                "oos_wr": strategy.fitness.oos_win_rate,
                "trades": strategy.fitness.total_trades,
                "consistency": strategy.fitness.consistency,
                "p_value": strategy.fitness.p_value,
                "fitness_score": strategy.fitness.fitness,
            },
            status="discovered",
            discovered_at=time.time(),
            generation=strategy.generation,
            novelty_score=strategy.novelty_score,
        )
        self.alphas[strategy.name] = record
        self._save()
        return record

    def get_deployed(self) -> list[AlphaRecord]:
        """Get all currently deployed alphas."""
        return [a for a in self.alphas.values() if a.status == "deployed"]

    def get_available(self) -> list[AlphaRecord]:
        """Get alphas that are validated but not yet deployed."""
        return [a for a in self.alphas.values() if a.status == "validated"]

    def deploy(self, name: str):
        if name in self.alphas:
            self.alphas[name].status = "deployed"
            self.alphas[name].deployed_at = time.time()
            self._save()

    def retire(self, name: str, reason: str = ""):
        if name in self.alphas:
            self.alphas[name].status = "retired"
            self.alphas[name].retired_at = time.time()
            self._save()
            logger.info(f"Alpha retired: {name} — {reason}")

    def mark_decayed(self, name: str, decay_score: float):
        if name in self.alphas:
            self.alphas[name].status = "decayed"
            self.alphas[name].decay_score = decay_score
            self._save()

    def update_live_stats(self, name: str, trades: int, pnl: float, sharpe: float):
        if name in self.alphas:
            self.alphas[name].live_trades = trades
            self.alphas[name].live_pnl = pnl
            self.alphas[name].live_sharpe = sharpe
            self._save()

    def stats(self) -> dict:
        """Summary statistics."""
        statuses = {}
        for a in self.alphas.values():
            statuses[a.status] = statuses.get(a.status, 0) + 1
        return {
            "total": len(self.alphas),
            "by_status": statuses,
            "deployed_count": len(self.get_deployed()),
        }

    def _save(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for name, rec in self.alphas.items():
            data[name] = {
                "name": rec.name,
                "formula": rec.formula,
                "tree_dict": rec.tree_dict,
                "fitness": rec.fitness,
                "status": rec.status,
                "discovered_at": rec.discovered_at,
                "deployed_at": rec.deployed_at,
                "retired_at": rec.retired_at,
                "live_trades": rec.live_trades,
                "live_pnl": rec.live_pnl,
                "live_sharpe": rec.live_sharpe,
                "decay_score": rec.decay_score,
                "generation": rec.generation,
                "novelty_score": rec.novelty_score,
            }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        p = Path(self.path)
        if not p.exists():
            return
        try:
            with open(p) as f:
                data = json.load(f)
            for name, d in data.items():
                self.alphas[name] = AlphaRecord(**d)
        except Exception as e:
            logger.warning(f"Could not load alpha registry: {e}")


# ─── Evolution Scheduler ────────────────────────────────────────

class EvolutionScheduler:
    """Decides WHEN to trigger new evolution rounds.

    Triggers:
        1. Deployed alpha count drops below minimum
        2. Average decay score rises above threshold
        3. Periodic (every N hours regardless)
        4. Portfolio Sharpe drops below threshold
    """

    def __init__(
        self,
        min_deployed_alphas: int = 3,
        max_avg_decay: float = 40.0,
        periodic_hours: float = 168,        # Weekly
        sharpe_floor: float = 0.5,
    ):
        self.min_deployed = min_deployed_alphas
        self.max_avg_decay = max_avg_decay
        self.periodic_hours = periodic_hours
        self.sharpe_floor = sharpe_floor
        self.last_evolution_time: float = 0

    def should_evolve(
        self,
        registry: AlphaRegistry,
        decay_detector: DecayDetector,
        portfolio_sharpe: float = 1.0,
    ) -> tuple[bool, str]:
        """Check if we should trigger a new evolution round."""
        deployed = registry.get_deployed()

        # 1. Not enough deployed alphas
        if len(deployed) < self.min_deployed:
            return True, f"Only {len(deployed)} deployed (min={self.min_deployed})"

        # 2. Average decay too high
        decay_scores = []
        for alpha in deployed:
            if alpha.name in decay_detector.strategies:
                report = decay_detector.check_health(alpha.name)
                decay_scores.append(report.decay_score)
        if decay_scores:
            avg_decay = np.mean(decay_scores)
            if avg_decay > self.max_avg_decay:
                return True, f"Avg decay={avg_decay:.0f} > {self.max_avg_decay}"

        # 3. Periodic trigger
        hours_since = (time.time() - self.last_evolution_time) / 3600
        if hours_since >= self.periodic_hours:
            return True, f"Periodic: {hours_since:.0f}h since last evolution"

        # 4. Portfolio Sharpe degraded
        if portfolio_sharpe < self.sharpe_floor:
            return True, f"Portfolio Sharpe={portfolio_sharpe:.2f} < {self.sharpe_floor}"

        return False, "No trigger"


# ─── Auto Deployer ───────────────────────────────────────────────

class AutoDeployer:
    """Validates and deploys evolved alphas into the portfolio.

    Validation gates:
        1. OOS Sharpe > threshold
        2. Statistical significance (p < 0.05)
        3. Minimum trades
        4. Low correlation with existing deployed alphas
        5. Consistency across walk-forward folds
    """

    def __init__(
        self,
        min_oos_sharpe: float = 0.8,
        min_profit_factor: float = 1.2,
        min_consistency: float = 0.6,
        max_correlation: float = 0.7,
        min_trades: int = 30,
        max_deployed: int = 10,
    ):
        self.min_sharpe = min_oos_sharpe
        self.min_pf = min_profit_factor
        self.min_consistency = min_consistency
        self.max_corr = max_correlation
        self.min_trades = min_trades
        self.max_deployed = max_deployed

    def validate_for_deployment(
        self,
        strategy: EvolvedStrategy,
        existing_signals: dict[str, pd.Series] = None,
        df: pd.DataFrame = None,
    ) -> tuple[bool, str]:
        """Check if a strategy passes deployment gates."""
        f = strategy.fitness

        # Gate 1: OOS Sharpe
        if f.oos_sharpe < self.min_sharpe:
            return False, f"OOS Sharpe {f.oos_sharpe:.2f} < {self.min_sharpe}"

        # Gate 2: Profit factor
        if f.oos_profit_factor < self.min_pf:
            return False, f"PF {f.oos_profit_factor:.2f} < {self.min_pf}"

        # Gate 3: Consistency
        if f.consistency < self.min_consistency:
            return False, f"Consistency {f.consistency:.0%} < {self.min_consistency:.0%}"

        # Gate 4: Trade count
        if f.total_trades < self.min_trades:
            return False, f"Trades {f.total_trades} < {self.min_trades}"

        # Gate 5: Significance
        if not f.is_significant:
            return False, f"Not statistically significant (p={f.p_value:.3f})"

        # Gate 6: Correlation with existing alphas
        if existing_signals and df is not None:
            try:
                tree = tree_from_dict(strategy.tree_dict)
                new_signals = tree.evaluate(df)
                for name, existing in existing_signals.items():
                    common = new_signals.index.intersection(existing.index)
                    if len(common) > 100:
                        corr = abs(new_signals.loc[common].corr(existing.loc[common]))
                        if corr > self.max_corr:
                            return False, (
                                f"Correlated with {name}: |r|={corr:.2f} > {self.max_corr}"
                            )
            except Exception as e:
                logger.warning(f"Correlation check failed: {e}")

        return True, "All gates passed"

    def build_signal_func(self, strategy: EvolvedStrategy) -> Callable:
        """Build a signal function from an evolved strategy for portfolio use."""
        tree_dict = strategy.tree_dict

        def signal_func(df: pd.DataFrame) -> pd.Series:
            tree = tree_from_dict(tree_dict)
            raw = tree.evaluate(df)
            # Discretize: z-score threshold at ±0.5
            std = raw.std()
            if std < 1e-10:
                return raw.apply(np.sign).astype(int)
            z = (raw - raw.mean()) / std
            signals = pd.Series(0, index=df.index, dtype=int)
            signals[z > 0.5] = 1
            signals[z < -0.5] = -1
            return signals

        return signal_func


# ─── Genome Orchestrator ─────────────────────────────────────────

class GenomeOrchestrator:
    """The autonomous alpha lifecycle manager.

    Wires together:
        AlphaGenomeEngine → AlphaRegistry → AutoDeployer → DecayDetector

    One call to `run_cycle()` executes the full loop:
        1. Check if evolution is needed
        2. Evolve new alphas
        3. Validate candidates
        4. Deploy survivors
        5. Monitor deployed alphas
        6. Retire decayed ones
    """

    def __init__(
        self,
        registry: Optional[AlphaRegistry] = None,
        decay_detector: Optional[DecayDetector] = None,
        deployer: Optional[AutoDeployer] = None,
        scheduler: Optional[EvolutionScheduler] = None,
        # Evolution params
        population_size: int = 100,
        max_generations: int = 30,
        min_trades: int = 25,
    ):
        self.registry = registry or AlphaRegistry()
        self.decay_detector = decay_detector or DecayDetector()
        self.deployer = deployer or AutoDeployer()
        self.scheduler = scheduler or EvolutionScheduler()

        self.evolution_params = {
            "population_size": population_size,
            "max_generations": max_generations,
            "min_trades": min_trades,
        }

        self._cycle_count = 0

    def run_cycle(
        self,
        df: pd.DataFrame,
        force_evolve: bool = False,
        portfolio_sharpe: float = 1.0,
        existing_signals: dict[str, pd.Series] = None,
        symbol: str = "",
    ) -> dict:
        """Execute one full alpha lifecycle cycle.

        Returns summary of what happened.
        """
        self._cycle_count += 1
        summary = {
            "cycle": self._cycle_count,
            "evolved": 0,
            "validated": 0,
            "deployed": 0,
            "decayed": 0,
            "retired": 0,
            "actions": [],
        }

        # ── Step 1: Monitor deployed alphas for decay ──
        deployed = self.registry.get_deployed()
        for alpha in deployed:
            if alpha.name in self.decay_detector.strategies:
                report = self.decay_detector.check_health(alpha.name)
                if report.kill_recommended:
                    self.registry.mark_decayed(alpha.name, report.decay_score)
                    summary["decayed"] += 1
                    summary["actions"].append(
                        f"DECAY: {alpha.name} (score={report.decay_score:.0f})"
                    )

        # ── Step 2: Retire severely decayed alphas ──
        for alpha in self.registry.get_deployed():
            if alpha.decay_score > 80:
                self.registry.retire(alpha.name, "severe decay")
                summary["retired"] += 1
                summary["actions"].append(f"RETIRE: {alpha.name}")

        # ── Step 3: Check if evolution needed ──
        should_evolve, reason = self.scheduler.should_evolve(
            self.registry, self.decay_detector, portfolio_sharpe,
        )

        if force_evolve:
            should_evolve = True
            reason = "forced"

        if should_evolve:
            summary["actions"].append(f"EVOLVE: triggered ({reason})")

            # ── Step 4: Run evolution ──
            engine = AlphaGenomeEngine(**self.evolution_params)
            strategies = engine.evolve(df, symbol=symbol)
            summary["evolved"] = len(strategies)

            # ── Step 5: Validate and register ──
            for strat in strategies:
                record = self.registry.register(strat)

                passed, gate_reason = self.deployer.validate_for_deployment(
                    strat,
                    existing_signals=existing_signals,
                    df=df,
                )

                if passed:
                    record.status = "validated"
                    summary["validated"] += 1
                    summary["actions"].append(
                        f"VALIDATED: {strat.name} "
                        f"(Sharpe={strat.fitness.oos_sharpe:.2f})"
                    )

            # ── Step 6: Deploy top validated ──
            available = self.registry.get_available()
            current_deployed = len(self.registry.get_deployed())
            slots = self.deployer.max_deployed - current_deployed

            # Sort by fitness score
            available.sort(key=lambda a: a.fitness.get("fitness_score", 0), reverse=True)

            for alpha in available[:max(0, slots)]:
                self.registry.deploy(alpha.name)
                # Register with decay detector
                self.decay_detector.register_strategy(alpha.name)
                summary["deployed"] += 1
                summary["actions"].append(f"DEPLOY: {alpha.name}")

            self.scheduler.last_evolution_time = time.time()

        # ── Summary ──
        reg_stats = self.registry.stats()
        summary["registry"] = reg_stats

        return summary

    def get_deployed_signal_funcs(self) -> dict[str, Callable]:
        """Get signal functions for all deployed alphas.

        Returns dict of {name: signal_func} ready for portfolio integration.
        """
        deployed = self.registry.get_deployed()
        funcs = {}

        for alpha in deployed:
            try:
                # Build an EvolvedStrategy-like object for the deployer
                class _FakeStrategy:
                    pass
                fake = _FakeStrategy()
                fake.tree_dict = alpha.tree_dict
                fake.fitness = type("F", (), alpha.fitness)()
                fake.name = alpha.name
                fake.formula = alpha.formula

                func = self.deployer.build_signal_func(fake)
                funcs[alpha.name] = func
            except Exception as e:
                logger.warning(f"Could not build signal func for {alpha.name}: {e}")

        return funcs

    def format_report(self) -> str:
        """Human-readable status report."""
        lines = []
        lines.append("=" * 60)
        lines.append("  SIGNAL GENOME v2 — ALPHA LIFECYCLE STATUS")
        lines.append("=" * 60)

        stats = self.registry.stats()
        lines.append(f"  Total alphas discovered: {stats['total']}")
        lines.append(f"  Currently deployed: {stats['deployed_count']}")
        for status, count in stats.get("by_status", {}).items():
            lines.append(f"    {status}: {count}")
        lines.append("")

        deployed = self.registry.get_deployed()
        if deployed:
            lines.append("─ DEPLOYED ALPHAS ──────────────────────────")
            lines.append(
                f"  {'Name':<25s} {'Sharpe':>7s} {'Trades':>7s} "
                f"{'PnL':>10s} {'Decay':>6s}"
            )
            for a in deployed:
                sharpe = a.fitness.get("oos_sharpe", 0)
                lines.append(
                    f"  {a.name:<25s} {sharpe:>7.2f} "
                    f"{a.live_trades:>7d} {a.live_pnl:>+10.2f} "
                    f"{a.decay_score:>5.0f}%"
                )
            lines.append("")

        lines.append(f"  Evolution cycles run: {self._cycle_count}")
        lines.append("=" * 60)
        return "\n".join(lines)

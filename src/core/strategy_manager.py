"""
Strategy Manager — Registry, Lifecycle, Death Autopsy
========================================================
The closed-loop learning system.

Every strategy goes through:
    BORN → DEPLOYED → MONITORED → DECAYING → KILLED → AUTOPSY → GENETIC INHERITANCE

The autopsy records WHY a strategy died:
    - What regime was active?
    - What was the crowding state?
    - Which features decayed first?
    - How did the strategy's DNA (expression tree) relate to its death?

This feeds back into evolution:
    - Bias new populations AWAY from death-causing patterns
    - Preserve DNA fragments that survived longest
    - Adapt mutation rates based on strategy lifespans

This is the edge nobody else has:
    Renaissance has GP evolution but no public evidence of systematic autopsy.
    Two Sigma has lifecycle management but manual.
    We automate the entire death → learn → re-discover loop.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyAutopsy:
    """Post-mortem of a killed strategy."""
    name: str
    born_at: str
    killed_at: str
    lifespan_hours: float
    total_trades: int
    total_pnl: float
    peak_pnl: float
    final_sharpe: float
    kill_reason: str

    # Context at death
    regime_at_death: str                # trending/ranging/volatile
    crowding_at_death: float            # 0-100
    crowding_direction_at_death: int    # +1/-1/0
    cascade_prob_at_death: float        # 0-1

    # Decay trajectory
    decay_score_at_death: float         # 0-100
    decay_components: dict              # which sub-scores were worst

    # DNA info for genetic inheritance
    tree_depth: int
    tree_features: List[str]            # features used by this strategy
    tree_hash: str                      # for deduplication
    tree_expression: str                # human-readable expression

    # Lessons
    winning_regime: str                 # regime where it performed best
    losing_regime: str                  # regime where it performed worst
    lifespan_rank: str                  # short/medium/long vs peers


@dataclass
class StrategyRecord:
    """Active strategy tracking record."""
    name: str
    tree: object                        # AlphaGene / Node expression tree
    tree_hash: str
    tree_expression: str
    features_used: List[str]
    deployed_at: datetime
    trades: int = 0
    total_pnl: float = 0.0
    peak_pnl: float = 0.0
    last_trade_at: Optional[datetime] = None
    regime_at_deploy: str = "unknown"
    status: str = "active"              # active / paused / decaying / killed

    # Performance by regime
    regime_pnl: Dict[str, float] = field(default_factory=dict)


class StrategyManager:
    """Central strategy registry with lifecycle management.

    Responsibilities:
        1. Register strategies from evolution output
        2. Track per-strategy performance and regime exposure
        3. Interface with DecayDetector for health monitoring
        4. Perform death autopsy when strategies are killed
        5. Store autopsy data for genetic inheritance
        6. Provide mutation hints to evolution engine

    Usage:
        manager = StrategyManager(max_active=8)
        manager.submit(evolved_strategies)
        signal = manager.tick(name, df)
        autopsy = manager.kill(name, reason, context)
        hints = manager.get_evolution_hints()
    """

    def __init__(
        self,
        max_active: int = 8,
        autopsy_dir: str = "fund_data/autopsies",
        min_lifespan_hours: float = 24.0,
    ):
        self.max_active = max_active
        self.autopsy_dir = Path(autopsy_dir)
        self.autopsy_dir.mkdir(parents=True, exist_ok=True)
        self.min_lifespan_hours = min_lifespan_hours

        self.strategies: Dict[str, StrategyRecord] = {}
        self.autopsies: List[StrategyAutopsy] = []
        self._load_autopsies()

    # ==================================================================
    # REGISTRATION
    # ==================================================================

    def submit(self, evolved_strategies: list) -> List[str]:
        """Submit evolved strategies for deployment.

        Returns list of names that were actually deployed.
        """
        deployed = []
        for i, strat in enumerate(evolved_strategies):
            if len(self.strategies) >= self.max_active:
                logger.info("Max active strategies reached (%d)", self.max_active)
                break

            name = strat.name if hasattr(strat, "name") else f"genome_{i}_{datetime.now().strftime('%H%M%S')}"
            tree = strat.tree if hasattr(strat, "tree") else strat

            # Extract DNA info
            tree_hash = self._hash_tree(tree)
            tree_expr = str(tree)
            features = self._extract_features(tree)

            # Check for duplicate DNA
            if any(s.tree_hash == tree_hash for s in self.strategies.values()):
                logger.info("Duplicate DNA detected, skipping: %s", name)
                continue

            # Check against autopsy history — avoid repeating deaths
            if self._is_death_pattern(features, tree_hash):
                logger.info("Death pattern detected, skipping: %s", name)
                continue

            record = StrategyRecord(
                name=name,
                tree=tree,
                tree_hash=tree_hash,
                tree_expression=tree_expr,
                features_used=features,
                deployed_at=datetime.now(),
            )
            self.strategies[name] = record
            deployed.append(name)
            logger.info("Deployed strategy: %s (features: %s)", name, features[:5])

        return deployed

    def get_active(self) -> List[str]:
        """Get names of active strategies."""
        return [n for n, s in self.strategies.items() if s.status == "active"]

    def get_record(self, name: str) -> Optional[StrategyRecord]:
        """Get strategy record by name."""
        return self.strategies.get(name)

    # ==================================================================
    # TICK — Strategy Evaluation
    # ==================================================================

    def tick(self, name: str, df) -> int:
        """Evaluate a strategy on current data.

        Args:
            name: Strategy name.
            df: Current market data DataFrame.

        Returns:
            +1 (long), -1 (short), 0 (no signal)
        """
        record = self.strategies.get(name)
        if not record or record.status != "active":
            return 0

        try:
            import pandas as pd
            import numpy as np

            signal_val = record.tree.evaluate(df)
            if isinstance(signal_val, (pd.Series, np.ndarray)):
                signal_val = signal_val.iloc[-1] if isinstance(signal_val, pd.Series) else signal_val[-1]

            if signal_val > 0:
                return 1
            elif signal_val < 0:
                return -1
            return 0
        except Exception as e:
            logger.warning("Strategy %s tick error: %s", name, e)
            return 0

    def record_trade(self, name: str, pnl: float, regime: str = "unknown"):
        """Record a trade result for a strategy."""
        record = self.strategies.get(name)
        if not record:
            return

        record.trades += 1
        record.total_pnl += pnl
        record.peak_pnl = max(record.peak_pnl, record.total_pnl)
        record.last_trade_at = datetime.now()

        # Track PnL by regime
        record.regime_pnl[regime] = record.regime_pnl.get(regime, 0) + pnl

    # ==================================================================
    # KILL — Death + Autopsy
    # ==================================================================

    def kill(
        self,
        name: str,
        reason: str,
        decay_score: float = 0,
        decay_components: Optional[dict] = None,
        regime: str = "unknown",
        crowding_score: float = 0,
        crowding_direction: int = 0,
        cascade_prob: float = 0,
    ) -> Optional[StrategyAutopsy]:
        """Kill a strategy and perform autopsy.

        Args:
            name: Strategy to kill.
            reason: Why it's being killed (decay, drawdown, adaptation, etc.)
            decay_score: Final decay score.
            decay_components: Breakdown of decay sub-scores.
            regime: Current market regime.
            crowding_score: Current crowding score.
            crowding_direction: Current crowding direction.
            cascade_prob: Current cascade probability.

        Returns:
            StrategyAutopsy record, or None if strategy not found.
        """
        record = self.strategies.pop(name, None)
        if not record:
            logger.warning("Cannot kill unknown strategy: %s", name)
            return None

        now = datetime.now()
        lifespan_hours = (now - record.deployed_at).total_seconds() / 3600

        # Determine winning/losing regimes
        winning_regime = "unknown"
        losing_regime = "unknown"
        if record.regime_pnl:
            winning_regime = max(record.regime_pnl, key=record.regime_pnl.get)
            losing_regime = min(record.regime_pnl, key=record.regime_pnl.get)

        # Lifespan ranking
        if self.autopsies:
            lifespans = [a.lifespan_hours for a in self.autopsies]
            avg_lifespan = sum(lifespans) / len(lifespans)
            if lifespan_hours > avg_lifespan * 1.5:
                lifespan_rank = "long"
            elif lifespan_hours < avg_lifespan * 0.5:
                lifespan_rank = "short"
            else:
                lifespan_rank = "medium"
        else:
            lifespan_rank = "first"

        # Compute final sharpe from PnL
        final_sharpe = record.total_pnl / max(abs(record.peak_pnl - record.total_pnl), 0.01)

        autopsy = StrategyAutopsy(
            name=name,
            born_at=record.deployed_at.isoformat(),
            killed_at=now.isoformat(),
            lifespan_hours=round(lifespan_hours, 1),
            total_trades=record.trades,
            total_pnl=round(record.total_pnl, 4),
            peak_pnl=round(record.peak_pnl, 4),
            final_sharpe=round(final_sharpe, 3),
            kill_reason=reason,
            regime_at_death=regime,
            crowding_at_death=round(crowding_score, 1),
            crowding_direction_at_death=crowding_direction,
            cascade_prob_at_death=round(cascade_prob, 3),
            decay_score_at_death=round(decay_score, 1),
            decay_components=decay_components or {},
            tree_depth=self._tree_depth(record.tree),
            tree_features=record.features_used,
            tree_hash=record.tree_hash,
            tree_expression=record.tree_expression[:500],
            winning_regime=winning_regime,
            losing_regime=losing_regime,
            lifespan_rank=lifespan_rank,
        )

        self.autopsies.append(autopsy)
        self._save_autopsy(autopsy)

        logger.info(
            "AUTOPSY: %s died after %.1fh (%d trades, $%.2f PnL). Reason: %s. "
            "Regime: %s. Crowding: %.0f. Features: %s",
            name, lifespan_hours, record.trades, record.total_pnl,
            reason, regime, crowding_score, record.features_used[:3],
        )

        return autopsy

    # ==================================================================
    # GENETIC INHERITANCE — Feedback to Evolution
    # ==================================================================

    def get_evolution_hints(self) -> dict:
        """Generate hints for the next evolution cycle based on autopsy data.

        Returns dict with:
            - avoid_features: features correlated with early death
            - preserve_features: features in long-lived strategies
            - death_patterns: tree hashes to avoid
            - regime_bias: which regime to optimize for
        """
        if not self.autopsies:
            return {"avoid_features": [], "preserve_features": [],
                    "death_patterns": set(), "regime_bias": None}

        # Feature analysis
        death_features: Dict[str, list] = {}
        survive_features: Dict[str, list] = {}

        for a in self.autopsies:
            for feat in a.tree_features:
                if a.lifespan_rank == "short":
                    death_features.setdefault(feat, []).append(a.lifespan_hours)
                elif a.lifespan_rank == "long":
                    survive_features.setdefault(feat, []).append(a.lifespan_hours)

        # Features that appear more in short-lived strategies
        avoid = []
        for feat, lifespans in death_features.items():
            if feat not in survive_features and len(lifespans) >= 3:
                avoid.append(feat)

        # Features that appear in long-lived strategies
        preserve = []
        for feat, lifespans in survive_features.items():
            if feat not in death_features:
                preserve.append(feat)

        # Death tree hashes
        death_hashes = {a.tree_hash for a in self.autopsies
                        if a.lifespan_rank == "short" and a.total_pnl < 0}

        # Regime bias — which regime killed the most strategies?
        regime_kills: Dict[str, int] = {}
        for a in self.autopsies:
            regime_kills[a.regime_at_death] = regime_kills.get(a.regime_at_death, 0) + 1

        # Bias toward the regime that kills most — evolve robustness there
        regime_bias = max(regime_kills, key=regime_kills.get) if regime_kills else None

        return {
            "avoid_features": avoid,
            "preserve_features": preserve,
            "death_patterns": death_hashes,
            "regime_bias": regime_bias,
            "total_autopsies": len(self.autopsies),
            "avg_lifespan_hours": round(
                sum(a.lifespan_hours for a in self.autopsies) / len(self.autopsies), 1
            ),
        }

    # ==================================================================
    # HELPERS
    # ==================================================================

    def _is_death_pattern(self, features: List[str], tree_hash: str) -> bool:
        """Check if this strategy matches a known death pattern."""
        # Exact DNA match with a short-lived loser
        death_hashes = {a.tree_hash for a in self.autopsies
                        if a.lifespan_rank == "short" and a.total_pnl < 0}
        if tree_hash in death_hashes:
            return True

        # Feature combination that killed >3 strategies quickly
        if len(self.autopsies) >= 5:
            feature_set = set(features)
            matches = sum(
                1 for a in self.autopsies
                if a.lifespan_rank == "short"
                and set(a.tree_features) == feature_set
            )
            if matches >= 3:
                return True

        return False

    @staticmethod
    def _hash_tree(tree) -> str:
        """Hash a strategy tree for deduplication."""
        try:
            from src.alpha_genome.gene import tree_hash
            return tree_hash(tree)
        except (ImportError, Exception):
            return str(hash(str(tree)))

    @staticmethod
    def _extract_features(tree) -> List[str]:
        """Extract feature names used by a strategy tree."""
        try:
            expr = str(tree)
            from src.alpha_genome.gene import ALL_FEATURE_NAMES
            return [f for f in ALL_FEATURE_NAMES if f in expr]
        except (ImportError, Exception):
            return []

    @staticmethod
    def _tree_depth(tree) -> int:
        """Get depth of expression tree."""
        try:
            return tree.depth() if hasattr(tree, "depth") else 0
        except Exception:
            return 0

    def _save_autopsy(self, autopsy: StrategyAutopsy):
        """Save autopsy to JSON file."""
        path = self.autopsy_dir / f"{autopsy.name}_{autopsy.killed_at[:10]}.json"
        with open(path, "w") as f:
            json.dump(asdict(autopsy), f, indent=2, default=str)

    def _load_autopsies(self):
        """Load existing autopsies from disk."""
        if not self.autopsy_dir.exists():
            return
        for path in sorted(self.autopsy_dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                self.autopsies.append(StrategyAutopsy(**data))
            except Exception as e:
                logger.warning("Failed to load autopsy %s: %s", path, e)

    def summary(self) -> dict:
        """Summary of strategy manager state."""
        return {
            "active_strategies": len([s for s in self.strategies.values() if s.status == "active"]),
            "paused_strategies": len([s for s in self.strategies.values() if s.status == "paused"]),
            "total_autopsies": len(self.autopsies),
            "avg_lifespan_hours": round(
                sum(a.lifespan_hours for a in self.autopsies) / max(len(self.autopsies), 1), 1
            ),
            "slots_free": self.max_active - len(self.strategies),
            "strategies": {
                name: {
                    "status": s.status,
                    "trades": s.trades,
                    "pnl": round(s.total_pnl, 4),
                    "deployed_at": s.deployed_at.isoformat(),
                    "features": s.features_used[:5],
                }
                for name, s in self.strategies.items()
            },
        }

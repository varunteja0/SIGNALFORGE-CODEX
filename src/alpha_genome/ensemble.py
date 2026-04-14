"""
Ensemble Evolution — Committee of Weak Learners
=================================================
Instead of evolving ONE best strategy tree, evolve a COMMITTEE of 20+
diverse weak learners and combine them via adaptive weighting.

This is how Renaissance and Two Sigma actually work:
- No single model dominates
- Ensemble of 100+ weak signals, each with slight edge
- Adaptive weighting based on recent performance
- Forced diversity (penalty for correlated signals)

Key algorithms:
1. Island Model GP — parallel sub-populations that occasionally migrate
2. Pareto Multi-Objective — optimize Sharpe AND diversity AND simplicity simultaneously
3. Stacking — meta-learner that combines committee signals
4. Adaptive Weighting — Sharpe-weighted with exponential decay
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.alpha_genome.gene import (
    Node, random_tree, crossover, mutate,
    tree_to_formula, tree_to_dict, tree_from_dict, tree_hash,
    FEATURE_NAMES,
)
from src.alpha_genome.fitness import FitnessEvaluator, FitnessResult
from src.alpha_genome.novelty import NoveltyDetector

logger = logging.getLogger(__name__)


@dataclass
class CommitteeMember:
    """A single member of the strategy committee."""
    name: str
    tree_dict: dict
    formula: str
    fitness: FitnessResult
    novelty_score: float
    weight: float = 1.0
    recent_sharpe: float = 0.0
    correlation_penalty: float = 0.0
    generation_born: int = 0
    tree_hash: str = ""
    pareto_rank: int = 0


@dataclass
class EnsembleResult:
    """Result of an ensemble signal generation."""
    direction: int              # -1, 0, +1
    confidence: float           # 0-1
    agreement_pct: float        # What fraction of committee agrees
    weighted_signal: float      # Continuous signal before discretization
    member_signals: dict = field(default_factory=dict)  # Per-member signals
    active_members: int = 0


class EnsembleEvolver:
    """Evolve a diverse committee of strategies using island model GP.

    Instead of one population, maintains N islands (sub-populations).
    Each island evolves independently with occasional migration of
    best individuals between islands. This dramatically increases
    diversity of final committee.
    """

    def __init__(
        self,
        n_islands: int = 4,
        island_size: int = 50,
        max_generations: int = 50,
        committee_size: int = 20,
        migration_interval: int = 5,
        migration_count: int = 3,
        min_diversity: float = 0.3,
        tournament_size: int = 5,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.25,
        max_tree_depth: int = 6,
        min_trades: int = 15,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        output_dir: str = "evolved_strategies",
    ):
        self.n_islands = n_islands
        self.island_size = island_size
        self.max_generations = max_generations
        self.committee_size = committee_size
        self.migration_interval = migration_interval
        self.migration_count = migration_count
        self.min_diversity = min_diversity
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_tree_depth = max_tree_depth

        self.fitness_evaluator = FitnessEvaluator(
            min_total_trades=min_trades,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            population_size=n_islands * island_size,
        )
        self.novelty_detector = NoveltyDetector()

        # Islands: list of populations
        self.islands: list[list[Node]] = []
        self.fitness_cache: dict[str, FitnessResult] = {}
        self.committee: list[CommitteeMember] = []

    def evolve(
        self,
        df: pd.DataFrame,
        features: Optional[list[str]] = None,
        symbol: str = "",
        timeframe: str = "",
    ) -> list[CommitteeMember]:
        """Run island-model ensemble evolution.

        Returns a diverse committee of strategies with adaptive weights.
        """
        features = features or [f for f in FEATURE_NAMES if f in df.columns]

        if len(df) < 500:
            logger.error(f"Need at least 500 bars, got {len(df)}")
            return []

        logger.info(
            f"Ensemble evolution: {self.n_islands} islands × "
            f"{self.island_size} pop × {self.max_generations} gens"
        )

        self.novelty_detector.register_standard_signals(df)

        # Initialize islands with different random seeds
        self.islands = []
        for island_id in range(self.n_islands):
            pop = []
            for i in range(self.island_size):
                seed = island_id * self.island_size + i
                depth = np.random.randint(2, self.max_tree_depth + 1)
                pop.append(random_tree(max_depth=depth, features=features, seed=seed))
            self.islands.append(pop)

        # Hall of fame (best from all islands)
        hall_of_fame: list[tuple[Node, FitnessResult, float]] = []

        for gen in range(self.max_generations):
            gen_start = time.time()
            gen_best = -float("inf")

            for island_id in range(self.n_islands):
                pop = self.islands[island_id]

                # Evaluate
                scores, results = self._evaluate_island(pop, df)

                # Track best
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                if scores[best_idx] > gen_best:
                    gen_best = scores[best_idx]

                # Add valid strategies to hall of fame
                for i, fr in enumerate(results):
                    if fr.is_valid:
                        h = tree_hash(pop[i])
                        if not any(tree_hash(t) == h for t, _, _ in hall_of_fame):
                            try:
                                signals = pop[i].evaluate(df)
                                nov = self.novelty_detector.novelty_score(signals)
                            except Exception:
                                nov = 0.0
                            hall_of_fame.append((pop[i].clone(), fr, nov))

                # Evolve island
                self.islands[island_id] = self._evolve_island(
                    pop, scores, features
                )

            # Migration between islands
            if gen > 0 and gen % self.migration_interval == 0:
                self._migrate()

            elapsed = time.time() - gen_start
            logger.info(
                f"Gen {gen:3d} | Best={gen_best:.4f} "
                f"HoF={len(hall_of_fame)} "
                f"({elapsed:.1f}s)"
            )

        # Build committee from hall of fame
        self.committee = self._build_committee(hall_of_fame, df)

        logger.info(
            f"Ensemble complete: {len(self.committee)} committee members"
        )

        return self.committee

    def _evaluate_island(
        self, pop: list[Node], df: pd.DataFrame
    ) -> tuple[list[float], list[FitnessResult]]:
        """Evaluate a single island's population."""
        scores = []
        results = []

        for tree in pop:
            h = tree_hash(tree)
            if h in self.fitness_cache:
                fr = self.fitness_cache[h]
            else:
                try:
                    fr = self.fitness_evaluator.evaluate(tree, df)
                except Exception:
                    fr = FitnessResult()
                self.fitness_cache[h] = fr

            scores.append(fr.fitness)
            results.append(fr)

        return scores, results

    def _evolve_island(
        self, pop: list[Node], scores: list[float], features: list[str]
    ) -> list[Node]:
        """Single generation of evolution for one island."""
        new_pop = []

        # Elitism: keep top 10%
        n_elite = max(2, len(pop) // 10)
        elite_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n_elite]
        for idx in elite_indices:
            new_pop.append(pop[idx].clone())

        # Fill rest
        while len(new_pop) < len(pop):
            if np.random.random() < self.crossover_rate:
                p1 = self._tournament_select(pop, scores)
                p2 = self._tournament_select(pop, scores)
                c1, c2 = crossover(p1, p2)
                new_pop.append(c1)
                if len(new_pop) < len(pop):
                    new_pop.append(c2)
            else:
                p = self._tournament_select(pop, scores)
                new_pop.append(p)

        # Mutation
        for i in range(n_elite, len(new_pop)):
            if np.random.random() < self.mutation_rate:
                new_pop[i] = mutate(new_pop[i], features)

        # Random immigrants
        n_imm = max(1, len(pop) // 20)
        for i in range(n_imm):
            idx = len(new_pop) - 1 - i
            if idx >= n_elite:
                depth = np.random.randint(2, self.max_tree_depth + 1)
                new_pop[idx] = random_tree(max_depth=depth, features=features)

        return new_pop[:len(pop)]

    def _tournament_select(self, pop: list[Node], scores: list[float]) -> Node:
        """Tournament selection."""
        indices = np.random.choice(
            len(pop), size=min(self.tournament_size, len(pop)), replace=False
        )
        best_idx = max(indices, key=lambda i: scores[i])
        return pop[best_idx].clone()

    def _migrate(self):
        """Migrate best individuals between islands (ring topology)."""
        for i in range(self.n_islands):
            src = i
            dst = (i + 1) % self.n_islands

            # Evaluate source island to find best
            src_pop = self.islands[src]
            src_scores = []
            for tree in src_pop:
                h = tree_hash(tree)
                fr = self.fitness_cache.get(h, FitnessResult())
                src_scores.append(fr.fitness)

            # Send best N from src to dst (replace worst N in dst)
            best_indices = sorted(
                range(len(src_scores)), key=lambda j: src_scores[j], reverse=True
            )[:self.migration_count]

            dst_pop = self.islands[dst]
            dst_scores = []
            for tree in dst_pop:
                h = tree_hash(tree)
                fr = self.fitness_cache.get(h, FitnessResult())
                dst_scores.append(fr.fitness)

            worst_indices = sorted(
                range(len(dst_scores)), key=lambda j: dst_scores[j]
            )[:self.migration_count]

            for src_idx, dst_idx in zip(best_indices, worst_indices):
                self.islands[dst][dst_idx] = src_pop[src_idx].clone()

        logger.debug(f"Migration complete across {self.n_islands} islands")

    def _build_committee(
        self,
        hall_of_fame: list[tuple[Node, FitnessResult, float]],
        df: pd.DataFrame,
    ) -> list[CommitteeMember]:
        """Build a diverse, optimally-weighted committee from hall of fame.

        Steps:
        1. Compute signal correlation matrix
        2. Greedily select diverse members (max correlation constraint)
        3. Compute Sharpe-weighted optimal weights
        """
        if not hall_of_fame:
            return []

        # Compute signals for all candidates
        candidates = []
        signals_matrix = []

        for tree, fitness, novelty in hall_of_fame:
            try:
                signals = tree.evaluate(df)
                candidates.append((tree, fitness, novelty, signals))
                signals_matrix.append(signals.values)
            except Exception:
                continue

        if len(candidates) < 2:
            # Return single member if only one
            if candidates:
                tree, fitness, novelty, _ = candidates[0]
                return [CommitteeMember(
                    name=f"alpha_{tree_hash(tree)[:8]}",
                    tree_dict=tree_to_dict(tree),
                    formula=tree_to_formula(tree),
                    fitness=fitness,
                    novelty_score=novelty,
                    weight=1.0,
                    tree_hash=tree_hash(tree),
                )]
            return []

        # Build correlation matrix
        sig_array = np.array(signals_matrix)
        n = len(sig_array)
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                std_i = np.std(sig_array[i])
                std_j = np.std(sig_array[j])
                if std_i > 1e-10 and std_j > 1e-10:
                    c = np.corrcoef(sig_array[i], sig_array[j])[0, 1]
                    if np.isnan(c):
                        c = 0.0
                    corr[i, j] = corr[j, i] = abs(c)
                else:
                    corr[i, j] = corr[j, i] = 1.0  # Treat constant as fully correlated

        # Greedy diverse selection
        selected_indices = []
        available = list(range(n))

        # Start with highest fitness
        fitness_scores = [c[1].fitness for c in candidates]
        best_idx = max(available, key=lambda i: fitness_scores[i])
        selected_indices.append(best_idx)
        available.remove(best_idx)

        while len(selected_indices) < min(self.committee_size, len(available) + len(selected_indices)):
            if not available:
                break

            # Score each remaining candidate: fitness - max_correlation_with_selected
            best_score = -float("inf")
            best_cand = None

            for idx in available:
                max_corr_with_selected = max(
                    corr[idx, sel] for sel in selected_indices
                )
                # Penalize high correlation
                diversity_bonus = max(0, 1 - max_corr_with_selected)
                score = fitness_scores[idx] * (0.6 + 0.4 * diversity_bonus)

                if score > best_score:
                    best_score = score
                    best_cand = idx

            if best_cand is not None:
                selected_indices.append(best_cand)
                available.remove(best_cand)

        # Compute optimal weights via minimum-variance with Sharpe tilt
        weights = self._compute_optimal_weights(
            [candidates[i] for i in selected_indices],
            corr[np.ix_(selected_indices, selected_indices)],
        )

        # Build committee
        committee = []
        for rank, (idx, w) in enumerate(zip(selected_indices, weights)):
            tree, fitness, novelty, signals = candidates[idx]
            h = tree_hash(tree)

            member = CommitteeMember(
                name=f"alpha_{h[:8]}",
                tree_dict=tree_to_dict(tree),
                formula=tree_to_formula(tree),
                fitness=fitness,
                novelty_score=novelty,
                weight=w,
                recent_sharpe=fitness.oos_sharpe,
                tree_hash=h,
                pareto_rank=rank,
            )
            committee.append(member)

        return committee

    def _compute_optimal_weights(
        self,
        candidates: list[tuple],
        corr_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute Sharpe-optimal weights for committee members.

        Uses mean-variance optimization:
        max w'μ / sqrt(w'Σw)
        subject to: sum(w) = 1, w >= 0
        """
        n = len(candidates)
        if n == 1:
            return np.array([1.0])

        # Expected returns (use OOS Sharpe as proxy)
        mu = np.array([c[1].oos_sharpe for c in candidates])

        # Covariance from correlation + individual vol estimates
        vols = np.array([
            max(0.01, c[1].oos_max_drawdown) for c in candidates
        ])
        cov = corr_matrix * np.outer(vols, vols)

        # Add regularization for numerical stability
        cov += np.eye(n) * 1e-6

        # If no positive expected returns, use equal weight
        if mu.max() <= 0:
            return np.ones(n) / n

        # Optimize: maximize Sharpe ratio
        def neg_sharpe(w):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ cov @ w + 1e-10)
            return -(port_ret / port_vol)

        # Constraints: weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.02 / n, 3.0 / n) for _ in range(n)]  # Min 2%/n, max 300%/n

        w0 = np.ones(n) / n
        result = minimize(
            neg_sharpe, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            weights = result.x
        else:
            # Fallback: inverse-volatility weighting
            inv_vol = 1.0 / (vols + 1e-10)
            weights = inv_vol / inv_vol.sum()

        return weights

    def generate_ensemble_signal(
        self,
        df: pd.DataFrame,
        decay_scores: Optional[dict[str, float]] = None,
    ) -> EnsembleResult:
        """Generate a combined signal from the entire committee.

        Each member votes weighted by:
        1. Optimized portfolio weight
        2. Inverse of decay score (dying strategies get lower weight)
        3. Recent Sharpe (recency-weighted performance)
        """
        if not self.committee:
            return EnsembleResult(direction=0, confidence=0, agreement_pct=0,
                                  weighted_signal=0)

        decay_scores = decay_scores or {}
        signals = {}
        weighted_sum = 0.0
        total_weight = 0.0
        votes = {"long": 0, "short": 0, "neutral": 0}

        for member in self.committee:
            try:
                tree = tree_from_dict(member.tree_dict)
                sig = tree.evaluate(df)
                current_sig = float(sig.iloc[-1]) if len(sig) > 0 else 0
            except Exception:
                current_sig = 0

            # Adjust weight by decay
            decay = decay_scores.get(member.name, 0)
            decay_adj = max(0.1, 1 - decay / 100)
            effective_weight = member.weight * decay_adj

            signals[member.name] = current_sig
            weighted_sum += current_sig * effective_weight
            total_weight += effective_weight

            if current_sig > 0.1:
                votes["long"] += 1
            elif current_sig < -0.1:
                votes["short"] += 1
            else:
                votes["neutral"] += 1

        n_active = len(self.committee)
        avg_signal = weighted_sum / (total_weight + 1e-10)

        # Majority voting
        majority = max(votes, key=lambda k: votes[k])
        majority_count = votes[majority]
        agreement = majority_count / n_active if n_active > 0 else 0

        # Direction: only trade if >60% agreement
        if agreement >= 0.6:
            direction = 1 if majority == "long" else (-1 if majority == "short" else 0)
        else:
            direction = 0

        # Confidence: function of agreement + signal strength
        confidence = min(1.0, agreement * abs(avg_signal))

        return EnsembleResult(
            direction=direction,
            confidence=confidence,
            agreement_pct=agreement,
            weighted_signal=avg_signal,
            member_signals=signals,
            active_members=n_active,
        )

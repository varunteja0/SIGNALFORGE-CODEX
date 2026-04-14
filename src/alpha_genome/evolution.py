"""
Alpha Genome — Genetic Programming Evolution Engine
=====================================================
The core engine that evolves trading strategies through:
    1. Random population initialization
    2. Fitness evaluation (walk-forward OOS Sharpe)
    3. Tournament selection
    4. Crossover (subtree swap)
    5. Mutation (random perturbation)
    6. Elitism (keep the best)
    7. Novelty pressure (reward uniqueness)

This produces "alien" strategies — mathematical expressions combining
market features in ways no human quant would conceive.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.alpha_genome.gene import (
    Node, random_tree, crossover, mutate,
    tree_to_formula, tree_to_dict, tree_from_dict, tree_hash,
    FEATURE_NAMES, ALL_FEATURE_NAMES,
)
from src.alpha_genome.fitness import FitnessEvaluator, FitnessResult
from src.alpha_genome.novelty import NoveltyDetector

logger = logging.getLogger(__name__)


@dataclass
class EvolvedStrategy:
    """A fully validated evolved trading strategy."""
    name: str
    formula: str
    tree_dict: dict
    fitness: FitnessResult
    novelty_score: float
    generation: int
    tree_hash: str
    symbol: str = ""
    timeframe: str = ""

    @staticmethod
    def _native(v):
        """Convert numpy types to native Python for JSON serialization."""
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    def to_dict(self) -> dict:
        n = self._native
        return {
            "name": self.name,
            "formula": self.formula,
            "tree": self.tree_dict,
            "fitness": {
                "oos_sharpe": n(self.fitness.oos_sharpe),
                "oos_sortino": n(self.fitness.oos_sortino),
                "oos_win_rate": n(self.fitness.oos_win_rate),
                "oos_profit_factor": n(self.fitness.oos_profit_factor),
                "oos_max_drawdown": n(self.fitness.oos_max_drawdown),
                "total_trades": n(self.fitness.total_trades),
                "p_value": n(self.fitness.p_value),
                "is_significant": n(self.fitness.is_significant),
                "consistency": n(self.fitness.consistency),
                "overfit_ratio": n(self.fitness.overfit_ratio),
                "fitness_score": n(self.fitness.fitness),
            },
            "novelty_score": n(self.novelty_score),
            "generation": n(self.generation),
            "tree_hash": self.tree_hash,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        }


@dataclass
class GenerationStats:
    """Statistics for one generation of evolution."""
    generation: int
    best_fitness: float
    avg_fitness: float
    median_fitness: float
    best_sharpe: float
    avg_sharpe: float
    valid_count: int
    population_size: int
    diversity: float
    best_formula: str
    elapsed_sec: float


class AlphaGenomeEngine:
    """The main genetic programming engine for evolving trading strategies."""

    def __init__(
        self,
        population_size: int = 200,
        max_generations: int = 50,
        tournament_size: int = 5,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.2,
        elitism_count: int = 10,
        max_tree_depth: int = 6,
        novelty_weight: float = 0.2,
        walk_forward_splits: int = 5,
        min_trades: int = 30,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        output_dir: str = "evolved_strategies",
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = min(elitism_count, population_size // 5)
        self.max_tree_depth = max_tree_depth
        self.novelty_weight = novelty_weight
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fitness_evaluator = FitnessEvaluator(
            walk_forward_splits=walk_forward_splits,
            min_total_trades=min_trades,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            population_size=population_size,
        )
        self.novelty_detector = NoveltyDetector()

        # State
        self.population: list[Node] = []
        self.fitness_cache: dict[str, FitnessResult] = {}
        self.generation_history: list[GenerationStats] = []
        self.hall_of_fame: list[EvolvedStrategy] = []

    def evolve(
        self,
        df: pd.DataFrame,
        features: Optional[list[str]] = None,
        symbol: str = "",
        timeframe: str = "",
        progress_callback=None,
    ) -> list[EvolvedStrategy]:
        """Run full evolution and return discovered strategies.

        This is the main entry point. Feed it market data with features
        already computed, and it returns novel, validated strategies.
        """
        features = features or [f for f in ALL_FEATURE_NAMES if f in df.columns]

        if len(features) < 5:
            logger.error(f"Only {len(features)} features available. Need at least 5.")
            return []

        if len(df) < 500:
            logger.error(f"Only {len(df)} bars available. Need at least 500.")
            return []

        logger.info(
            f"Starting evolution: pop={self.population_size}, "
            f"gens={self.max_generations}, features={len(features)}, "
            f"bars={len(df)}"
        )

        # Register standard signals for novelty comparison
        self.novelty_detector.register_standard_signals(df)

        # Initialize population
        self.population = self._init_population(features)
        logger.info(f"Initialized population of {len(self.population)} trees")

        best_ever_fitness = -float("inf")
        stagnation_count = 0

        for gen in range(self.max_generations):
            gen_start = time.time()

            # Evaluate fitness for all individuals
            fitness_scores, fitness_results = self._evaluate_population(df)

            # Compute novelty-adjusted fitness
            adjusted_scores = self._novelty_adjusted_fitness(
                fitness_scores, df
            )

            # Track stats
            valid_trees = sum(1 for f in fitness_results if f.is_valid)
            gen_stats = self._record_stats(
                gen, adjusted_scores, fitness_results, df, gen_start
            )
            self.generation_history.append(gen_stats)

            # Log progress
            logger.info(
                f"Gen {gen:3d} | Best={gen_stats.best_fitness:.4f} "
                f"Avg={gen_stats.avg_fitness:.4f} "
                f"Sharpe={gen_stats.best_sharpe:.2f} "
                f"Valid={valid_trees}/{len(self.population)} "
                f"Diversity={gen_stats.diversity:.2f} "
                f"({gen_stats.elapsed_sec:.1f}s)"
            )

            if progress_callback:
                progress_callback(gen, self.max_generations, gen_stats)

            # Check for new best
            if gen_stats.best_fitness > best_ever_fitness:
                best_ever_fitness = gen_stats.best_fitness
                stagnation_count = 0
                # Save to hall of fame
                self._update_hall_of_fame(
                    fitness_results, adjusted_scores, df, gen, symbol, timeframe
                )
            else:
                stagnation_count += 1

            # Early stopping if stagnated
            if stagnation_count > 15:
                logger.info(f"Stopping early: no improvement for {stagnation_count} generations")
                break

            # Last generation — don't evolve further
            if gen == self.max_generations - 1:
                break

            # Create next generation
            self.population = self._next_generation(
                adjusted_scores, features
            )

        # Final: select diverse set from hall of fame
        final_strategies = self._finalize(df, symbol, timeframe)

        # Save to disk
        self._save_strategies(final_strategies)

        logger.info(f"Evolution complete. {len(final_strategies)} strategies discovered.")
        return final_strategies

    def _init_population(self, features: list[str]) -> list[Node]:
        """Create initial random population with varied depths."""
        population = []
        for i in range(self.population_size):
            depth = np.random.randint(2, self.max_tree_depth + 1)
            tree = random_tree(max_depth=depth, features=features, seed=i)
            population.append(tree)
        return population

    def _evaluate_population(
        self, df: pd.DataFrame
    ) -> tuple[list[float], list[FitnessResult]]:
        """Evaluate fitness for entire population. Uses caching."""
        scores = []
        results = []

        for tree in self.population:
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

    def _novelty_adjusted_fitness(
        self, base_scores: list[float], df: pd.DataFrame
    ) -> list[float]:
        """Adjust fitness by novelty score to encourage diversity."""
        if self.novelty_weight <= 0:
            return base_scores

        adjusted = []
        for i, tree in enumerate(self.population):
            base = base_scores[i]
            try:
                signals = tree.evaluate(df)
                nov = self.novelty_detector.novelty_score(signals)
            except Exception:
                nov = 0.5

            adj = base * (1 - self.novelty_weight) + nov * self.novelty_weight
            adjusted.append(adj)

        return adjusted

    def _tournament_select(self, scores: list[float]) -> Node:
        """Tournament selection: pick random subset, return the best."""
        indices = np.random.choice(
            len(self.population), size=self.tournament_size, replace=False
        )
        best_idx = max(indices, key=lambda i: scores[i])
        return self.population[best_idx].clone()

    def _next_generation(
        self, scores: list[float], features: list[str]
    ) -> list[Node]:
        """Create the next generation via selection, crossover, mutation."""
        new_pop = []

        # Elitism: keep top individuals unchanged
        elite_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:self.elitism_count]
        for idx in elite_indices:
            new_pop.append(self.population[idx].clone())

        # Fill rest via crossover + mutation
        while len(new_pop) < self.population_size:
            if np.random.random() < self.crossover_rate:
                parent1 = self._tournament_select(scores)
                parent2 = self._tournament_select(scores)
                child1, child2 = crossover(parent1, parent2)
                new_pop.append(child1)
                if len(new_pop) < self.population_size:
                    new_pop.append(child2)
            else:
                parent = self._tournament_select(scores)
                new_pop.append(parent)

        # Apply mutation
        for i in range(self.elitism_count, len(new_pop)):
            if np.random.random() < self.mutation_rate:
                new_pop[i] = mutate(new_pop[i], features)

        # Inject fresh blood (random immigrants) to prevent premature convergence
        n_immigrants = max(1, self.population_size // 20)
        for i in range(n_immigrants):
            idx = len(new_pop) - 1 - i
            if idx >= self.elitism_count:
                depth = np.random.randint(2, self.max_tree_depth + 1)
                new_pop[idx] = random_tree(max_depth=depth, features=features)

        return new_pop[:self.population_size]

    def _update_hall_of_fame(
        self,
        fitness_results: list[FitnessResult],
        adjusted_scores: list[float],
        df: pd.DataFrame,
        generation: int,
        symbol: str,
        timeframe: str,
    ):
        """Add valid strategies to the hall of fame."""
        for i, fr in enumerate(fitness_results):
            if not fr.is_valid:
                continue

            tree = self.population[i]
            h = tree_hash(tree)

            # Check if already in hall of fame
            if any(s.tree_hash == h for s in self.hall_of_fame):
                continue

            try:
                signals = tree.evaluate(df)
                nov = self.novelty_detector.novelty_score(signals)
            except Exception:
                nov = 0.0

            strategy = EvolvedStrategy(
                name=f"alpha_{h[:8]}",
                formula=tree_to_formula(tree),
                tree_dict=tree_to_dict(tree),
                fitness=fr,
                novelty_score=nov,
                generation=generation,
                tree_hash=h,
                symbol=symbol,
                timeframe=timeframe,
            )
            self.hall_of_fame.append(strategy)

    def _finalize(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[EvolvedStrategy]:
        """Select final diverse set of strategies from hall of fame."""
        if not self.hall_of_fame:
            return []

        # Compute signals for each hall of fame strategy
        candidates = []
        for strategy in self.hall_of_fame:
            try:
                tree = tree_from_dict(strategy.tree_dict)
                signals = tree.evaluate(df)
                candidates.append((
                    strategy.name, signals, strategy.fitness.fitness
                ))
            except Exception:
                continue

        # Select diverse set
        diverse = self.novelty_detector.select_diverse_set(
            candidates, max_strategies=10
        )

        # Map back to strategies
        selected_names = {name for name, _, _ in diverse}
        final = [s for s in self.hall_of_fame if s.name in selected_names]
        final.sort(key=lambda s: s.fitness.fitness, reverse=True)

        return final

    def _record_stats(
        self,
        gen: int,
        scores: list[float],
        results: list[FitnessResult],
        df: pd.DataFrame,
        start_time: float,
    ) -> GenerationStats:
        """Record statistics for this generation."""
        sharpes = [r.oos_sharpe for r in results]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])

        # Diversity (sample for speed)
        sample_signals = []
        sample_indices = np.random.choice(
            len(self.population), min(30, len(self.population)), replace=False
        )
        for idx in sample_indices:
            try:
                sig = self.population[idx].evaluate(df)
                sample_signals.append(sig)
            except Exception:
                pass

        diversity = self.novelty_detector.population_diversity(sample_signals)

        return GenerationStats(
            generation=gen,
            best_fitness=max(scores),
            avg_fitness=float(np.mean(scores)),
            median_fitness=float(np.median(scores)),
            best_sharpe=max(sharpes) if sharpes else 0.0,
            avg_sharpe=float(np.mean(sharpes)) if sharpes else 0.0,
            valid_count=sum(1 for r in results if r.is_valid),
            population_size=len(self.population),
            diversity=diversity,
            best_formula=tree_to_formula(self.population[best_idx]),
            elapsed_sec=time.time() - start_time,
        )

    def _save_strategies(self, strategies: list[EvolvedStrategy]):
        """Persist evolved strategies to disk as JSON."""
        if not strategies:
            return

        output = {
            "evolved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generations_run": len(self.generation_history),
            "population_size": self.population_size,
            "strategies": [s.to_dict() for s in strategies],
            "generation_history": [
                {
                    "gen": g.generation,
                    "best_fitness": g.best_fitness,
                    "avg_fitness": g.avg_fitness,
                    "best_sharpe": g.best_sharpe,
                    "valid_count": g.valid_count,
                    "diversity": g.diversity,
                }
                for g in self.generation_history
            ],
        }

        path = self.output_dir / "latest_evolution.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Saved {len(strategies)} strategies to {path}")

    def load_strategies(self, path: Optional[str] = None) -> list[EvolvedStrategy]:
        """Load previously evolved strategies from disk."""
        path = Path(path) if path else self.output_dir / "latest_evolution.json"
        if not path.exists():
            return []

        with open(path) as f:
            data = json.load(f)

        strategies = []
        for s in data.get("strategies", []):
            fr = FitnessResult(
                oos_sharpe=s["fitness"]["oos_sharpe"],
                oos_sortino=s["fitness"]["oos_sortino"],
                oos_win_rate=s["fitness"]["oos_win_rate"],
                oos_profit_factor=s["fitness"]["oos_profit_factor"],
                oos_max_drawdown=s["fitness"]["oos_max_drawdown"],
                total_trades=s["fitness"]["total_trades"],
                p_value=s["fitness"]["p_value"],
                is_significant=s["fitness"]["is_significant"],
                consistency=s["fitness"]["consistency"],
                overfit_ratio=s["fitness"]["overfit_ratio"],
                fitness=s["fitness"]["fitness_score"],
                is_valid=True,
            )
            strategies.append(EvolvedStrategy(
                name=s["name"],
                formula=s["formula"],
                tree_dict=s["tree"],
                fitness=fr,
                novelty_score=s["novelty_score"],
                generation=s["generation"],
                tree_hash=s["tree_hash"],
                symbol=s.get("symbol", ""),
                timeframe=s.get("timeframe", ""),
            ))

        logger.info(f"Loaded {len(strategies)} strategies from {path}")
        return strategies

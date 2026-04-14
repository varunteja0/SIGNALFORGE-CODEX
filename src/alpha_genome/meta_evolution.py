"""
Meta-GP — Self-Modifying Evolution Engine
============================================
This is recursive self-improvement: the GP doesn't just evolve strategies,
it evolves ITSELF. The evolution parameters (mutation rate, tree depth,
crossover rate, feature subsets, fitness weights) are themselves evolved
based on which configurations produce winning strategies.

No institution does this. This is the ultimate edge — a system that
gets better at getting better.

Architecture:
  Meta-Population: Each individual is a "EvolutionConfig"
  Meta-Fitness: How many profitable strategies did this config produce?
  Meta-Evolution: Evolve configs → run inner GP → evaluate → select → repeat

This creates a 2-level evolutionary hierarchy:
  Level 1 (Inner): Evolve trading strategies
  Level 2 (Outer): Evolve the evolution parameters

The system discovers optimal exploration/exploitation tradeoffs,
feature subsets that contain alpha, and GP hyperparameters that
produce consistently profitable strategies.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.alpha_genome.evolution import AlphaGenomeEngine, EvolvedStrategy
from src.alpha_genome.fitness import FitnessResult

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """A candidate set of evolution parameters (the "meta-genome")."""
    config_id: str = ""

    # GP parameters
    population_size: int = 200
    max_generations: int = 50
    tournament_size: int = 5
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    elitism_count: int = 10
    max_tree_depth: int = 6
    novelty_weight: float = 0.2

    # Feature selection
    feature_subset_pct: float = 1.0  # Use 100% of features by default
    feature_groups: list = field(default_factory=lambda: [
        "returns", "volatility", "momentum", "volume",
        "microstructure", "regime", "trend",
    ])

    # Fitness weights
    sharpe_weight: float = 0.4
    consistency_weight: float = 0.2
    novelty_weight_inner: float = 0.2
    parsimony_weight: float = 0.1
    robustness_weight: float = 0.1

    # Walk-forward configuration
    walk_forward_splits: int = 5
    min_trades: int = 30

    # Meta-fitness (how good was this config?)
    meta_fitness: float = 0.0
    n_valid_strategies: int = 0
    avg_oos_sharpe: float = 0.0
    best_oos_sharpe: float = 0.0
    avg_consistency: float = 0.0
    evolution_time_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "tournament_size": self.tournament_size,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "elitism_count": self.elitism_count,
            "max_tree_depth": self.max_tree_depth,
            "novelty_weight": self.novelty_weight,
            "feature_subset_pct": self.feature_subset_pct,
            "sharpe_weight": self.sharpe_weight,
            "consistency_weight": self.consistency_weight,
            "walk_forward_splits": self.walk_forward_splits,
            "min_trades": self.min_trades,
            "meta_fitness": self.meta_fitness,
            "n_valid_strategies": self.n_valid_strategies,
            "avg_oos_sharpe": self.avg_oos_sharpe,
            "best_oos_sharpe": self.best_oos_sharpe,
        }

    def mutate(self) -> "EvolutionConfig":
        """Return a mutated copy of this EvolutionConfig."""
        child = EvolutionConfig(
            population_size=max(10, int(self.population_size * (1 + np.random.randn() * 0.05))),
            max_generations=max(1, int(self.max_generations * (1 + np.random.randn() * 0.05))),
            tournament_size=max(2, int(self.tournament_size * (1 + np.random.randn() * 0.1))),
            crossover_rate=float(np.clip(self.crossover_rate + np.random.randn() * 0.05, 0.1, 0.95)),
            mutation_rate=float(np.clip(self.mutation_rate + np.random.randn() * 0.05, 0.01, 0.6)),
            elitism_count=max(1, int(self.elitism_count * (1 + np.random.randn() * 0.1))),
            max_tree_depth=max(1, int(self.max_tree_depth + np.random.choice([-1, 0, 1]))),
            novelty_weight=float(np.clip(self.novelty_weight + np.random.randn() * 0.05, 0.0, 1.0)),
            feature_subset_pct=float(np.clip(self.feature_subset_pct + np.random.randn() * 0.05, 0.1, 1.0)),
            feature_groups=self.feature_groups.copy(),
            sharpe_weight=float(np.clip(self.sharpe_weight + np.random.randn() * 0.05, 0.0, 1.0)),
            consistency_weight=float(np.clip(self.consistency_weight + np.random.randn() * 0.05, 0.0, 1.0)),
            novelty_weight_inner=self.novelty_weight_inner,
            parsimony_weight=self.parsimony_weight,
            robustness_weight=self.robustness_weight,
            walk_forward_splits=max(2, int(self.walk_forward_splits * (1 + np.random.randn() * 0.05))),
            min_trades=max(1, int(self.min_trades * (1 + np.random.randn() * 0.05))),
        )
        return child


# Feature group definitions
FEATURE_GROUPS = {
    "returns": ["ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20", "ret_50",
                "log_ret_1", "log_ret_5", "log_ret_20"],
    "volatility": ["vol_5", "vol_10", "vol_20", "vol_50", "vol_ratio",
                   "parkinson_vol_20", "garman_klass_vol_20", "yang_zhang_vol_20",
                   "vol_of_vol_20"],
    "momentum": ["rsi_14", "rsi_7", "rsi_21", "macd", "macd_signal", "macd_hist",
                 "cci_14", "willr_14", "adx_14", "stoch_k_14", "stoch_d_14"],
    "volume": ["obv_slope_20", "mfi_14", "vwap_dev", "vol_ratio_20",
               "volume_clock_20", "participation_rate_20"],
    "microstructure": ["bar_position", "body_ratio", "upper_shadow_ratio",
                      "lower_shadow_ratio", "range_compression_10", "close_to_high"],
    "regime": ["rolling_sharpe_20", "rolling_sharpe_50", "dd_duration", "dd_pct",
              "mean_rev_pressure"],
    "trend": ["linreg_slope_10", "linreg_slope_20", "linreg_slope_50",
             "aroon_osc_14", "aroon_osc_25", "dpo_20",
             "momentum_rank_20", "momentum_rank_50"],
}


class MetaEvolutionEngine:
    """Evolves the evolution parameters themselves.

    This is recursive self-improvement: discover which GP configurations
    consistently produce profitable strategies, then focus search there.
    """

    def __init__(
        self,
        meta_population_size: int = 20,
        meta_generations: int = 10,
        inner_pop_size_range: tuple = (50, 300),
        inner_gen_range: tuple = (10, 60),
        inner_generations: Optional[int] = None,
        output_dir: str = "evolved_strategies/meta",
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ):
        self.meta_pop_size = meta_population_size
        self.meta_generations = meta_generations
        self.inner_pop_range = inner_pop_size_range
        if inner_generations is not None:
            self.inner_gen_range = (1, max(1, inner_generations))
        else:
            self.inner_gen_range = inner_gen_range
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        self.rng = np.random.RandomState(42)
        self.meta_history: list[dict] = []
        self.best_config: Optional[EvolutionConfig] = None
        self.all_strategies: list[EvolvedStrategy] = []

    def evolve(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        timeframe: str = "",
        progress_callback=None,
    ) -> tuple[EvolutionConfig, list[EvolvedStrategy]]:
        """Run meta-evolution to find optimal GP parameters.

        Returns (best_config, all_strategies_found).
        """
        logger.info(
            f"Starting meta-evolution: {self.meta_pop_size} configs × "
            f"{self.meta_generations} meta-generations"
        )

        # Initialize meta-population
        meta_pop = self._init_meta_population()

        best_meta_fitness = -float("inf")

        for meta_gen in range(self.meta_generations):
            gen_start = time.time()
            logger.info(f"\n=== Meta-Generation {meta_gen + 1}/{self.meta_generations} ===")

            # Evaluate each config by running inner GP
            for i, config in enumerate(meta_pop):
                if config.meta_fitness > 0:
                    continue  # Already evaluated (elitism)

                logger.info(
                    f"  Config {i + 1}/{len(meta_pop)}: "
                    f"pop={config.population_size}, gen={config.max_generations}, "
                    f"mut={config.mutation_rate:.2f}, depth={config.max_tree_depth}, "
                    f"feat={config.feature_subset_pct:.0%}"
                )

                # Select feature subset
                features = self._select_features(df, config)

                # Run inner GP with this config
                inner_start = time.time()
                try:
                    engine = AlphaGenomeEngine(
                        population_size=config.population_size,
                        max_generations=config.max_generations,
                        tournament_size=config.tournament_size,
                        crossover_rate=config.crossover_rate,
                        mutation_rate=config.mutation_rate,
                        elitism_count=config.elitism_count,
                        max_tree_depth=config.max_tree_depth,
                        novelty_weight=config.novelty_weight,
                        walk_forward_splits=config.walk_forward_splits,
                        min_trades=config.min_trades,
                        commission_pct=self.commission_pct,
                        slippage_pct=self.slippage_pct,
                        output_dir=str(self.output_dir / f"gen{meta_gen}_cfg{i}"),
                    )

                    strategies = engine.evolve(
                        df, features=features,
                        symbol=symbol, timeframe=timeframe,
                    )

                    config.evolution_time_sec = time.time() - inner_start
                    config.n_valid_strategies = len(strategies)

                    if strategies:
                        sharpes = [s.fitness.oos_sharpe for s in strategies]
                        consistencies = [s.fitness.consistency for s in strategies]
                        config.avg_oos_sharpe = np.mean(sharpes)
                        config.best_oos_sharpe = max(sharpes)
                        config.avg_consistency = np.mean(consistencies)
                        self.all_strategies.extend(strategies)

                except Exception as e:
                    logger.error(f"  Inner GP failed: {e}")
                    config.evolution_time_sec = time.time() - inner_start

                # Compute meta-fitness
                config.meta_fitness = self._compute_meta_fitness(config)
                config.config_id = f"meta_g{meta_gen}_c{i}"

                logger.info(
                    f"  -> {config.n_valid_strategies} strategies, "
                    f"avg_sharpe={config.avg_oos_sharpe:.2f}, "
                    f"meta_fitness={config.meta_fitness:.4f} "
                    f"({config.evolution_time_sec:.0f}s)"
                )

            # Sort by meta-fitness
            meta_pop.sort(key=lambda c: c.meta_fitness, reverse=True)

            # Track best
            if meta_pop[0].meta_fitness > best_meta_fitness:
                best_meta_fitness = meta_pop[0].meta_fitness
                self.best_config = meta_pop[0]

            # Record history
            gen_stats = {
                "meta_generation": meta_gen,
                "best_meta_fitness": meta_pop[0].meta_fitness,
                "avg_meta_fitness": np.mean([c.meta_fitness for c in meta_pop]),
                "best_config": meta_pop[0].to_dict(),
                "total_strategies": len(self.all_strategies),
                "elapsed_sec": time.time() - gen_start,
            }
            self.meta_history.append(gen_stats)

            if progress_callback:
                progress_callback(meta_gen, self.meta_generations, gen_stats)

            logger.info(
                f"  Meta-Gen {meta_gen}: Best fitness={meta_pop[0].meta_fitness:.4f}, "
                f"Total strategies={len(self.all_strategies)}"
            )

            # Last generation — don't evolve
            if meta_gen == self.meta_generations - 1:
                break

            # Meta-evolution: create next generation of configs
            meta_pop = self._evolve_configs(meta_pop)

        # Save results
        self._save_results()

        return self.best_config, self.all_strategies

    def _init_meta_population(self) -> list[EvolutionConfig]:
        """Initialize diverse meta-population."""
        configs = []

        # Config 1: Default (baseline)
        configs.append(EvolutionConfig())

        # Config 2: High exploration (large population, high mutation)
        configs.append(EvolutionConfig(
            population_size=300, max_generations=30,
            mutation_rate=0.4, crossover_rate=0.5,
            max_tree_depth=8, novelty_weight=0.4,
        ))

        # Config 3: Deep exploitation (small pop, many gens, low mutation)
        configs.append(EvolutionConfig(
            population_size=80, max_generations=60,
            mutation_rate=0.1, crossover_rate=0.8,
            max_tree_depth=4, novelty_weight=0.1,
        ))

        # Config 4: Feature-sparse (use only 50% of features)
        configs.append(EvolutionConfig(
            feature_subset_pct=0.5,
            population_size=150, max_generations=40,
        ))

        # Config 5: Robustness-focused (more walk-forward splits)
        configs.append(EvolutionConfig(
            walk_forward_splits=8, min_trades=50,
            consistency_weight=0.4, sharpe_weight=0.3,
        ))

        # Fill rest with random configs
        while len(configs) < self.meta_pop_size:
            configs.append(self._random_config())

        return configs

    def _random_config(self) -> EvolutionConfig:
        """Generate a random evolution configuration."""
        return EvolutionConfig(
            population_size=int(self.rng.uniform(*self.inner_pop_range)),
            max_generations=int(self.rng.uniform(*self.inner_gen_range)),
            tournament_size=int(self.rng.choice([3, 5, 7])),
            crossover_rate=self.rng.uniform(0.4, 0.9),
            mutation_rate=self.rng.uniform(0.05, 0.5),
            elitism_count=int(self.rng.choice([5, 10, 15, 20])),
            max_tree_depth=int(self.rng.choice([4, 5, 6, 7, 8])),
            novelty_weight=self.rng.uniform(0.0, 0.5),
            feature_subset_pct=self.rng.uniform(0.3, 1.0),
            walk_forward_splits=int(self.rng.choice([3, 5, 7, 10])),
            min_trades=int(self.rng.choice([15, 20, 30, 50])),
            sharpe_weight=self.rng.uniform(0.2, 0.6),
            consistency_weight=self.rng.uniform(0.1, 0.4),
        )

    def _compute_meta_fitness(self, config: EvolutionConfig) -> float:
        """Compute meta-fitness: how good is this evolution config?"""
        if config.n_valid_strategies == 0:
            return -1.0

        # Components of meta-fitness:
        # 1. Number of valid strategies (more = better exploration)
        n_valid_score = np.log1p(config.n_valid_strategies) / 3

        # 2. Quality of strategies (Sharpe)
        sharpe_score = np.clip(config.avg_oos_sharpe, -2, 5) / 5

        # 3. Best strategy quality
        best_score = np.clip(config.best_oos_sharpe, -2, 5) / 5

        # 4. Consistency (strategies should work across time)
        consistency_score = config.avg_consistency

        # 5. Efficiency (strategies per second of compute)
        efficiency = config.n_valid_strategies / max(config.evolution_time_sec, 1)
        efficiency_score = np.log1p(efficiency) / 3

        meta_fitness = (
            n_valid_score * 0.2
            + sharpe_score * 0.3
            + best_score * 0.2
            + consistency_score * 0.2
            + efficiency_score * 0.1
        )

        return float(meta_fitness)

    def _evolve_configs(self, configs: list[EvolutionConfig]) -> list[EvolutionConfig]:
        """Create next generation of configs via tournament + mutation."""
        new_configs = []

        # Elitism: keep top 3
        for c in configs[:3]:
            new_configs.append(c)

        # Fill rest via tournament selection + mutation
        while len(new_configs) < self.meta_pop_size:
            # Tournament select parent
            tournament = self.rng.choice(len(configs), size=3, replace=False)
            parent = max((configs[i] for i in tournament), key=lambda c: c.meta_fitness)

            # Mutate
            child = self._mutate_config(parent)
            child.meta_fitness = 0  # Reset — needs re-evaluation
            new_configs.append(child)

        return new_configs

    def _mutate_config(self, parent: EvolutionConfig) -> EvolutionConfig:
        """Mutate an evolution config."""
        child = EvolutionConfig(
            population_size=parent.population_size,
            max_generations=parent.max_generations,
            tournament_size=parent.tournament_size,
            crossover_rate=parent.crossover_rate,
            mutation_rate=parent.mutation_rate,
            elitism_count=parent.elitism_count,
            max_tree_depth=parent.max_tree_depth,
            novelty_weight=parent.novelty_weight,
            feature_subset_pct=parent.feature_subset_pct,
            walk_forward_splits=parent.walk_forward_splits,
            min_trades=parent.min_trades,
            sharpe_weight=parent.sharpe_weight,
            consistency_weight=parent.consistency_weight,
        )

        # Mutate 2-3 parameters
        n_mutations = self.rng.choice([1, 2, 3])
        params = self.rng.choice([
            "population_size", "max_generations", "crossover_rate",
            "mutation_rate", "max_tree_depth", "novelty_weight",
            "feature_subset_pct", "walk_forward_splits", "min_trades",
        ], size=n_mutations, replace=False)

        for param in params:
            if param == "population_size":
                child.population_size = int(np.clip(
                    parent.population_size + self.rng.randint(-50, 50),
                    *self.inner_pop_range
                ))
            elif param == "max_generations":
                child.max_generations = int(np.clip(
                    parent.max_generations + self.rng.randint(-10, 10),
                    *self.inner_gen_range
                ))
            elif param == "crossover_rate":
                child.crossover_rate = np.clip(
                    parent.crossover_rate + self.rng.normal(0, 0.1), 0.3, 0.95
                )
            elif param == "mutation_rate":
                child.mutation_rate = np.clip(
                    parent.mutation_rate + self.rng.normal(0, 0.1), 0.05, 0.6
                )
            elif param == "max_tree_depth":
                child.max_tree_depth = int(np.clip(
                    parent.max_tree_depth + self.rng.choice([-1, 0, 1]), 3, 10
                ))
            elif param == "novelty_weight":
                child.novelty_weight = np.clip(
                    parent.novelty_weight + self.rng.normal(0, 0.1), 0, 0.6
                )
            elif param == "feature_subset_pct":
                child.feature_subset_pct = np.clip(
                    parent.feature_subset_pct + self.rng.normal(0, 0.15), 0.2, 1.0
                )
            elif param == "walk_forward_splits":
                child.walk_forward_splits = int(np.clip(
                    parent.walk_forward_splits + self.rng.choice([-1, 0, 1]), 3, 10
                ))
            elif param == "min_trades":
                child.min_trades = int(np.clip(
                    parent.min_trades + self.rng.choice([-10, -5, 0, 5, 10]), 10, 100
                ))

        child.elitism_count = min(child.elitism_count, child.population_size // 5)
        return child

    def _select_features(self, df: pd.DataFrame, config: EvolutionConfig) -> list[str]:
        """Select feature subset based on config."""
        exclude = {"open", "high", "low", "close", "volume"}
        all_features = [c for c in df.columns if c not in exclude]

        if config.feature_subset_pct >= 0.99:
            return all_features

        # Select by feature groups
        selected = set()
        for group_name in config.feature_groups:
            if group_name in FEATURE_GROUPS:
                for f in FEATURE_GROUPS[group_name]:
                    if f in df.columns:
                        selected.add(f)

        # Then randomly sample additional to reach target percentage
        target_count = max(10, int(len(all_features) * config.feature_subset_pct))
        remaining = [f for f in all_features if f not in selected]
        n_additional = max(0, target_count - len(selected))
        if remaining and n_additional > 0:
            additional = self.rng.choice(
                remaining, size=min(n_additional, len(remaining)), replace=False
            )
            selected.update(additional)

        return list(selected)

    def _save_results(self):
        """Save meta-evolution results to disk."""
        results = {
            "best_config": self.best_config.to_dict() if self.best_config else {},
            "meta_history": self.meta_history,
            "total_strategies": len(self.all_strategies),
            "timestamp": time.time(),
        }

        output_path = self.output_dir / "meta_evolution_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Meta-evolution results saved to {output_path}")

    def get_optimal_config(self) -> Optional[EvolutionConfig]:
        """Return the best-performing evolution configuration."""
        return self.best_config

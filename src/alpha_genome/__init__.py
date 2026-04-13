"""
Alpha Genome — Self-Evolving Strategy DNA
==========================================
Genetic programming engine that INVENTS trading strategies no human
has ever conceived. Instead of testing pre-coded rules, this system
evolves mathematical expressions that combine market features in
novel, alien ways.

The output: trading signals derived from expression trees like:
    tanh(ts_std(vol_ratio_10, 5) * sin(rsi_14 / atr_pct_14)) > delay(ret_20, 3)

No human would write that. But if it makes money on unseen data, it's valid.
"""

from src.alpha_genome.gene import (
    Node,
    FeatureNode,
    ConstantNode,
    UnaryNode,
    BinaryNode,
    ComparisonNode,
    TimeSeriesNode,
    random_tree,
    crossover,
    mutate,
    tree_to_formula,
    tree_to_dict,
    tree_from_dict,
    FEATURE_NAMES,
)
from src.alpha_genome.fitness import FitnessEvaluator, FitnessResult
from src.alpha_genome.novelty import NoveltyDetector
from src.alpha_genome.evolution import AlphaGenomeEngine, EvolvedStrategy

__all__ = [
    "Node", "FeatureNode", "ConstantNode", "UnaryNode", "BinaryNode",
    "ComparisonNode", "TimeSeriesNode",
    "random_tree", "crossover", "mutate", "tree_to_formula",
    "tree_to_dict", "tree_from_dict", "FEATURE_NAMES",
    "FitnessEvaluator", "FitnessResult",
    "NoveltyDetector",
    "AlphaGenomeEngine", "EvolvedStrategy",
]

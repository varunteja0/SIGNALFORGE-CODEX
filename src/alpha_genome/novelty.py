"""
Alpha Genome — Novelty Detection
==================================
Ensures evolved strategies are GENUINELY novel — not correlated with
known signals or with each other.

This is critical because:
1. If an evolved strategy is just RSI in disguise, it adds no edge
2. If all evolved strategies are correlated, there's no diversification
3. True alpha comes from UNCORRELATED return streams
"""

import logging
import warnings

import numpy as np
import pandas as pd

from src.alpha_genome.gene import Node

logger = logging.getLogger(__name__)


class NoveltyDetector:
    """Detects whether a strategy's signals are truly novel."""

    def __init__(self, max_correlation: float = 0.7, min_novelty_score: float = 0.3):
        self.max_correlation = max_correlation
        self.min_novelty_score = min_novelty_score
        self.known_signal_series: dict[str, pd.Series] = {}

    def register_known(self, name: str, signals: pd.Series):
        """Register a known signal series for future novelty comparisons."""
        self.known_signal_series[name] = signals

    def register_standard_signals(self, df: pd.DataFrame):
        """Auto-register standard textbook signals from dataframe features.

        Anything we evolve should be different from these commoditized signals.
        """
        # RSI signals
        for col in ["rsi_7", "rsi_14", "rsi_21"]:
            if col in df.columns:
                sig = pd.Series(0.0, index=df.index)
                sig[df[col] < 30] = 1.0
                sig[df[col] > 70] = -1.0
                self.register_known(f"standard_{col}", sig)

        # MA crossover signals
        for fast, slow in [(10, 50), (20, 100), (50, 200)]:
            f_col, s_col = f"ma_{fast}", f"ma_{slow}"
            if f_col in df.columns and s_col in df.columns:
                sig = pd.Series(0.0, index=df.index)
                above = df[f_col] > df[s_col]
                sig[above] = 1.0
                sig[~above] = -1.0
                self.register_known(f"standard_ma_{fast}_{slow}", sig)

        # MACD signal
        if "macd_hist" in df.columns:
            sig = pd.Series(0.0, index=df.index)
            sig[df["macd_hist"] > 0] = 1.0
            sig[df["macd_hist"] < 0] = -1.0
            self.register_known("standard_macd", sig)

        # BB signals
        if "bb_pct_20" in df.columns:
            sig = pd.Series(0.0, index=df.index)
            sig[df["bb_pct_20"] < 0] = 1.0
            sig[df["bb_pct_20"] > 1] = -1.0
            self.register_known("standard_bb", sig)

        logger.info(f"Registered {len(self.known_signal_series)} standard signals for novelty checking")

    def novelty_score(self, candidate_signals: pd.Series) -> float:
        """Compute novelty score for a candidate signal.

        Score: 1.0 = completely novel, 0.0 = identical to a known signal.
        Uses maximum absolute correlation against all known signals.
        """
        if not self.known_signal_series:
            return 1.0

        max_corr = 0.0

        for name, known in self.known_signal_series.items():
            # Align indices
            aligned = pd.concat([candidate_signals, known], axis=1).dropna()
            if len(aligned) < 30:
                continue

            # Skip constant signals (std=0 → undefined correlation)
            if aligned.iloc[:, 0].std() < 1e-10 or aligned.iloc[:, 1].std() < 1e-10:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                corr = abs(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if np.isnan(corr):
                continue
            max_corr = max(max_corr, corr)

        # Novelty = 1 - max_correlation
        return 1.0 - max_corr

    def is_novel(self, candidate_signals: pd.Series) -> bool:
        """Check if a candidate signal meets novelty threshold."""
        score = self.novelty_score(candidate_signals)
        return score >= self.min_novelty_score

    def pairwise_correlations(self, signal_dict: dict[str, pd.Series]) -> pd.DataFrame:
        """Compute pairwise correlation matrix for a set of signals.

        Used to ensure portfolio of strategies is diversified.
        """
        df = pd.DataFrame(signal_dict)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return df.corr()

    def select_diverse_set(
        self, candidates: list[tuple[str, pd.Series, float]],
        max_strategies: int = 10,
    ) -> list[tuple[str, pd.Series, float]]:
        """Select a maximally diverse subset of strategies.

        Uses greedy forward selection: pick best, then pick next best
        that is least correlated with already selected.

        Args:
            candidates: list of (name, signal_series, fitness_score)
            max_strategies: max number to select

        Returns:
            Diverse subset sorted by fitness
        """
        if not candidates:
            return []

        # Sort by fitness descending
        ranked = sorted(candidates, key=lambda x: x[2], reverse=True)
        selected = [ranked[0]]

        for name, signals, fitness in ranked[1:]:
            if len(selected) >= max_strategies:
                break

            # Check correlation against all already-selected
            max_corr_with_selected = 0.0
            for sel_name, sel_signals, _ in selected:
                aligned = pd.concat([signals, sel_signals], axis=1).dropna()
                if len(aligned) < 30:
                    continue
                if aligned.iloc[:, 0].std() < 1e-10 or aligned.iloc[:, 1].std() < 1e-10:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    corr = abs(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                if not np.isnan(corr):
                    max_corr_with_selected = max(max_corr_with_selected, corr)

            if max_corr_with_selected < self.max_correlation:
                selected.append((name, signals, fitness))

        return selected

    def population_diversity(self, signals_list: list[pd.Series]) -> float:
        """Measure overall diversity of a population of signals.

        Returns average pairwise distance (1 - |correlation|).
        1.0 = perfectly diverse, 0.0 = all identical.
        """
        if len(signals_list) < 2:
            return 1.0

        distances = []
        n = min(len(signals_list), 50)  # Sample for speed with large populations
        indices = np.random.choice(len(signals_list), n, replace=False) if len(signals_list) > 50 else range(len(signals_list))

        for i_idx, i in enumerate(indices):
            for j in indices[i_idx + 1:]:
                aligned = pd.concat(
                    [signals_list[i], signals_list[j]], axis=1
                ).dropna()
                if len(aligned) < 30:
                    distances.append(1.0)
                    continue
                if aligned.iloc[:, 0].std() < 1e-10 or aligned.iloc[:, 1].std() < 1e-10:
                    distances.append(1.0)
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    corr = abs(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                if np.isnan(corr):
                    distances.append(1.0)
                else:
                    distances.append(1.0 - corr)

        return float(np.mean(distances)) if distances else 1.0

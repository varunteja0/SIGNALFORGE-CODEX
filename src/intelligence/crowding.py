"""
Crowding Scorer — Measures How Fragile One Side of the Market Is
===================================================================
Combines three independent data sources into a single 0–100 score:

1. Single-venue funding rate extremity (existing: structural.py)
2. Top trader vs retail L/S divergence (new: multi_venue.py)
3. Cross-venue funding disagreement (new: multi_venue.py)

When all three agree, conviction is highest.
When only funding is extreme but positioning data disagrees, conviction is lower.

Academic basis:
- Kyle (1985): informed vs uninformed flow separation
- Brunnermeier & Pedersen (2009): funding constraints → fragility
- Lo (2004): adaptive markets — crowded trades decay faster

Output: CrowdingScore per asset — used by:
- CascadePredictor (as an input)
- StrategyManager (to boost/suppress signals)
- GP evolution (as a feature column in DataFrames)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CrowdingScore:
    """Composite measure of how crowded/fragile one side is."""
    score: float            # 0–100, higher = more crowded → more fragile
    direction: int          # +1 = crowd is long, -1 = crowd is short, 0 = balanced
    funding_component: float   # 0–100, from single-venue funding z-score
    ls_component: float        # 0–100, from top-vs-retail divergence
    cross_venue_component: float  # 0–100, from cross-venue funding spread
    confidence: float       # 0–1, how many data sources are available
    n_sources: int          # number of data sources that contributed


class CrowdingScorer:
    """Scores how crowded/fragile one side of the market is.

    Uses three independent data sources. Degrades gracefully
    when not all data is available.

    Usage:
        scorer = CrowdingScorer()
        score = scorer.score(df)  # df must have relevant columns
    """

    # Columns we look for (produced by structural.py + multi_venue.py)
    FUNDING_COLS = ["fund_funding_zscore", "fund_funding_rate"]
    LS_COLS = ["top_retail_divergence_zscore", "top_retail_divergence"]
    CROSS_VENUE_COLS = ["cross_venue_funding_zscore", "cross_venue_funding_spread"]

    def __init__(
        self,
        funding_weight: float = 0.40,
        ls_weight: float = 0.35,
        cross_venue_weight: float = 0.25,
    ):
        """Initialize with component weights.

        Weights are normalized at runtime based on available data.
        """
        self.weights = {
            "funding": funding_weight,
            "ls": ls_weight,
            "cross_venue": cross_venue_weight,
        }

    def score(self, df: pd.DataFrame) -> CrowdingScore:
        """Score the current crowding level from the latest bar.

        Args:
            df: DataFrame with features from structural.py + multi_venue.py.
                Must have at least one funding rate column.

        Returns:
            CrowdingScore with composite score 0–100.
        """
        if len(df) < 20:
            return CrowdingScore(
                score=0, direction=0, funding_component=0,
                ls_component=0, cross_venue_component=0,
                confidence=0, n_sources=0,
            )

        latest = df.iloc[-1]
        components = {}
        directions = {}

        # --- Component 1: Funding rate extremity ---
        funding_z = self._get_value(latest, self.FUNDING_COLS)
        if funding_z is not None:
            # If we got raw rate instead of z-score, z-score it
            if abs(funding_z) < 0.01:  # likely raw rate, not z-score
                col = self._find_col(df, self.FUNDING_COLS)
                if col:
                    series = df[col].dropna()
                    if len(series) > 20:
                        mean = series.rolling(168, min_periods=20).mean().iloc[-1]
                        std = series.rolling(168, min_periods=20).std().iloc[-1]
                        if std > 0:
                            funding_z = (funding_z - mean) / std

            components["funding"] = self._z_to_score(abs(funding_z))
            directions["funding"] = 1 if funding_z > 0 else (-1 if funding_z < 0 else 0)
        else:
            components["funding"] = 0

        # --- Component 2: Top trader vs retail divergence ---
        ls_z = self._get_value(latest, self.LS_COLS)
        if ls_z is not None:
            components["ls"] = self._z_to_score(abs(ls_z))
            # Positive divergence = smart money more long = crowd might be wrong being short
            # Negative = smart money more short = crowd might be wrong being long
            directions["ls"] = 1 if ls_z < 0 else (-1 if ls_z > 0 else 0)
        else:
            components["ls"] = 0

        # --- Component 3: Cross-venue funding spread ---
        cv_z = self._get_value(latest, self.CROSS_VENUE_COLS)
        if cv_z is not None:
            components["cross_venue"] = self._z_to_score(abs(cv_z))
            # Large spread = venues disagree = fragility
            directions["cross_venue"] = directions.get("funding", 0)
        else:
            components["cross_venue"] = 0

        # --- Composite score ---
        col_map = {
            "funding": self.FUNDING_COLS,
            "ls": self.LS_COLS,
            "cross_venue": self.CROSS_VENUE_COLS,
        }
        available = {
            k: v for k, v in components.items()
            if self._has_col(df, col_map.get(k, []))
        }
        n_sources = len(available)

        if n_sources == 0:
            return CrowdingScore(
                score=0, direction=0, funding_component=0,
                ls_component=0, cross_venue_component=0,
                confidence=0, n_sources=0,
            )

        # Normalize weights to available sources
        total_weight = sum(self.weights[k] for k in available)
        weighted_score = sum(
            components[k] * self.weights[k] / total_weight
            for k in available
        )

        # Direction: majority vote
        dir_votes = [d for d in directions.values() if d != 0]
        if dir_votes:
            direction = 1 if sum(dir_votes) > 0 else -1
        else:
            direction = 0

        # Confidence: scales with number of agreeing sources
        confidence = n_sources / 3.0
        if n_sources >= 2 and len(set(dir_votes)) == 1:
            confidence = min(1.0, confidence + 0.2)  # bonus for agreement

        return CrowdingScore(
            score=round(float(weighted_score), 1),
            direction=direction,
            funding_component=round(float(components.get("funding", 0)), 1),
            ls_component=round(float(components.get("ls", 0)), 1),
            cross_venue_component=round(float(components.get("cross_venue", 0)), 1),
            confidence=round(float(confidence), 2),
            n_sources=n_sources,
        )

    def score_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute crowding score for every bar (for backtesting).

        Returns DataFrame with columns:
            crowding_score, crowding_direction, crowding_confidence
        """
        scores = []
        # Use a rolling window approach for efficiency
        min_bars = 50

        for i in range(len(df)):
            if i < min_bars:
                scores.append({"crowding_score": 0, "crowding_direction": 0,
                               "crowding_confidence": 0})
                continue
            window = df.iloc[max(0, i - 500):i + 1]
            cs = self.score(window)
            scores.append({
                "crowding_score": cs.score,
                "crowding_direction": cs.direction,
                "crowding_confidence": cs.confidence,
            })

        return pd.DataFrame(scores, index=df.index)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _z_to_score(z: float) -> float:
        """Convert absolute z-score to 0–100 crowding score.

        Mapping:
            z < 1.0  → 0–20  (normal)
            z 1.0–2.0 → 20–50 (elevated)
            z 2.0–3.0 → 50–75 (high)
            z > 3.0   → 75–100 (extreme)
        """
        if z < 0.5:
            return z * 20  # 0–10
        elif z < 1.0:
            return 10 + (z - 0.5) * 20  # 10–20
        elif z < 2.0:
            return 20 + (z - 1.0) * 30  # 20–50
        elif z < 3.0:
            return 50 + (z - 2.0) * 25  # 50–75
        else:
            return min(100, 75 + (z - 3.0) * 12.5)  # 75–100, capped

    @staticmethod
    def _get_value(row: pd.Series, col_names: list[str]) -> Optional[float]:
        """Get first available value from a list of column name options."""
        for col in col_names:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    return float(val)
        return None

    @staticmethod
    def _find_col(df: pd.DataFrame, col_names: list[str]) -> Optional[str]:
        """Find first available column from options."""
        for col in col_names:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _has_col(df: pd.DataFrame, col_names: list[str]) -> bool:
        """Check if any of the columns exist in the DataFrame."""
        return any(col in df.columns for col in col_names)

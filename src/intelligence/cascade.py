"""
Cascade Predictor — Forward-Looking Liquidation Cascade Detection
===================================================================
The edge that TradFi firms CANNOT replicate:

$33B in DeFi collateral is publicly visible on-chain.
Every position's liquidation price is known.
We can predict cascades BEFORE they happen.

The mechanism (Brunnermeier & Pedersen 2009):
    Extreme crowding → forced unwinds → price impact →
    more liquidations → amplified price impact → cascade

This module computes:
    1. Cascade probability (0–1) based on preconditions
    2. Nearest liquidation cluster price level
    3. Expected cascade direction

Inputs:
    - CrowdingScore (from crowding.py)
    - OI concentration (from structural.py)
    - OKX liquidation data (from multi_venue.py)
    - Funding rate velocity (from structural.py)

Output:
    CascadePrediction — used by StrategyManager for entry signals
    and by RiskManager for protective exits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CascadePrediction:
    """Forward-looking cascade risk assessment."""
    probability: float       # 0–1, cascade probability
    direction: int           # expected cascade direction (-1 = longs liquidated, +1 = shorts)
    signal: int              # -1/0/+1 trade signal (0 = no trade)
    signal_strength: float   # 0–1, conviction level
    preconditions: dict      # which fire triangle components are present
    reasoning: str           # human-readable explanation


class CascadePredictor:
    """Predicts liquidation cascades from observable preconditions.

    The "fire triangle" model:
    1. FUEL: Extreme crowding on one side (from CrowdingScorer)
    2. HEAT: High OI concentration / leverage buildup
    3. SPARK: Price approaching known liquidation clusters

    All three must be present for high-probability cascade prediction.
    Two out of three = elevated but not actionable.
    One = normal market conditions.

    Usage:
        predictor = CascadePredictor()
        prediction = predictor.predict(df, crowding_score)
    """

    def __init__(
        self,
        crowding_threshold: float = 60.0,    # Crowding score to consider "extreme"
        oi_zscore_threshold: float = 2.0,    # OI concentration z-score
        funding_velocity_z: float = 2.0,     # Funding rate acceleration
        liq_proximity_pct: float = 5.0,      # % distance to liquidation cluster
        min_probability: float = 0.30,       # Min cascade prob to generate signal
    ):
        self.crowding_threshold = crowding_threshold
        self.oi_zscore_threshold = oi_zscore_threshold
        self.funding_velocity_z = funding_velocity_z
        self.liq_proximity_pct = liq_proximity_pct
        self.min_probability = min_probability

    def predict(
        self,
        df: pd.DataFrame,
        crowding_score: Optional[float] = None,
        crowding_direction: Optional[int] = None,
    ) -> CascadePrediction:
        """Predict cascade probability from current market state.

        Args:
            df: DataFrame with structural + multi-venue features.
            crowding_score: Pre-computed crowding score (0–100).
                           If None, estimated from funding data alone.
            crowding_direction: +1 = crowd is long, -1 = crowd is short.

        Returns:
            CascadePrediction with probability and trade signal.
        """
        if len(df) < 50:
            return self._empty_prediction("Insufficient data")

        latest = df.iloc[-1]
        preconditions = {}
        reasons = []

        # =====================================================
        # COMPONENT 1: FUEL — Crowding extremity
        # =====================================================
        if crowding_score is not None:
            fuel_score = crowding_score / 100.0  # normalize to 0–1
            fuel_present = crowding_score >= self.crowding_threshold
            preconditions["fuel_crowding"] = round(crowding_score, 1)
        else:
            # Fallback: estimate from funding z-score alone
            funding_z = self._get_val(latest, [
                "fund_funding_zscore", "funding_zscore",
            ])
            if funding_z is not None:
                fuel_score = min(1.0, abs(funding_z) / 4.0)
                fuel_present = abs(funding_z) >= 2.5
                preconditions["fuel_funding_z"] = round(funding_z, 2)
                if crowding_direction is None:
                    crowding_direction = 1 if funding_z > 0 else -1
            else:
                fuel_score = 0
                fuel_present = False

        if fuel_present:
            reasons.append(f"FUEL: Crowding extreme ({preconditions})")

        # =====================================================
        # COMPONENT 2: HEAT — OI concentration / leverage
        # =====================================================
        oi_z = self._get_val(latest, [
            "oi_oi_zscore", "oi_zscore",
        ])
        oi_change = self._get_val(latest, [
            "oi_oi_change_24h", "oi_change_24h",
        ])
        leverage_heat = self._get_val(latest, ["leverage_heat"])

        heat_score = 0.0
        heat_present = False

        if oi_z is not None:
            heat_score = min(1.0, abs(oi_z) / 3.0)
            heat_present = abs(oi_z) >= self.oi_zscore_threshold
            preconditions["heat_oi_z"] = round(oi_z, 2)

        if leverage_heat is not None:
            heat_score = max(heat_score, min(1.0, leverage_heat / 3.0))
            if leverage_heat >= 2.0:
                heat_present = True
            preconditions["heat_leverage"] = round(leverage_heat, 2)

        # OI building rapidly = leverage accumulating
        if oi_change is not None and abs(oi_change) > 0.05:
            heat_score = max(heat_score, min(1.0, abs(oi_change) / 0.15))
            if abs(oi_change) > 0.10:
                heat_present = True
            preconditions["heat_oi_change_24h"] = round(oi_change, 4)

        if heat_present:
            reasons.append(f"HEAT: OI/leverage elevated")

        # =====================================================
        # COMPONENT 3: SPARK — Funding velocity + proximity
        # =====================================================
        funding_rate = self._get_val(latest, [
            "fund_funding_rate", "funding_rate",
        ])

        spark_score = 0.0
        spark_present = False

        # Funding velocity (rate of change of funding)
        if funding_rate is not None and "fund_funding_rate" in df.columns:
            funding_series = df["fund_funding_rate"].dropna()
            if len(funding_series) > 10:
                velocity = funding_series.diff(8).iloc[-1] if len(funding_series) > 8 else 0
                vel_std = funding_series.diff(8).dropna().std()
                if vel_std > 0:
                    vel_z = abs(velocity) / vel_std
                    if vel_z >= self.funding_velocity_z:
                        spark_score = min(1.0, vel_z / 4.0)
                        spark_present = True
                        preconditions["spark_velocity_z"] = round(vel_z, 2)

        # Liquidation proximity (from OKX data)
        liq_imbalance = self._get_val(latest, ["okx_liq_imbalance"])
        liq_count = self._get_val(latest, ["okx_liq_count_24h"])

        if liq_count is not None and liq_count > 10:
            # High recent liquidation activity = cascade may be starting
            spark_score = max(spark_score, min(1.0, liq_count / 100.0))
            if liq_count > 50:
                spark_present = True
            preconditions["spark_liq_count_24h"] = int(liq_count)

        if liq_imbalance is not None and abs(liq_imbalance) > 0.3:
            # Strong directional liquidation imbalance
            spark_score = max(spark_score, abs(liq_imbalance))
            preconditions["spark_liq_imbalance"] = round(liq_imbalance, 2)

        if spark_present:
            reasons.append(f"SPARK: Funding accelerating or liquidations active")

        # =====================================================
        # COMPOSITE: Fire Triangle Probability
        # =====================================================
        n_present = sum([fuel_present, heat_present, spark_present])

        # Weighted geometric mean — all components matter
        # but we degrade gracefully with missing data
        component_scores = [s for s in [fuel_score, heat_score, spark_score] if s > 0]

        if not component_scores:
            return self._empty_prediction("No cascade preconditions detected")

        # Base probability from component scores
        if n_present >= 3:
            # All three present — high conviction
            probability = 0.5 + 0.5 * np.mean(component_scores)
        elif n_present == 2:
            # Two present — moderate
            probability = 0.25 + 0.35 * np.mean(component_scores)
        elif n_present == 1:
            # Only one — low
            probability = 0.10 + 0.15 * max(component_scores)
        else:
            probability = 0.05 * np.mean(component_scores) if component_scores else 0

        probability = round(float(np.clip(probability, 0, 1)), 3)

        # Direction: cascade liquidates the CROWDED side
        # If crowd is long (direction=+1), cascade forces sell → price drops → signal SHORT
        # If crowd is short (direction=-1), cascade forces buy → price rises → signal LONG
        if crowding_direction is not None and crowding_direction != 0:
            cascade_direction = -crowding_direction  # opposite of crowd
            signal = cascade_direction if probability >= self.min_probability else 0
        else:
            cascade_direction = 0
            signal = 0

        # Signal strength scales with probability
        signal_strength = probability if signal != 0 else 0.0

        reasoning = "; ".join(reasons) if reasons else "No significant preconditions"

        return CascadePrediction(
            probability=probability,
            direction=cascade_direction,
            signal=signal,
            signal_strength=round(signal_strength, 3),
            preconditions=preconditions,
            reasoning=reasoning,
        )

    def predict_series(self, df: pd.DataFrame,
                       crowding_scores: Optional[pd.DataFrame] = None,
                       ) -> pd.DataFrame:
        """Compute cascade predictions for every bar (for backtesting).

        Args:
            df: Full OHLCV + structural features DataFrame.
            crowding_scores: DataFrame from CrowdingScorer.score_series()
                            with columns: crowding_score, crowding_direction.

        Returns:
            DataFrame with columns: cascade_probability, cascade_signal,
            cascade_direction, cascade_strength.
        """
        results = []
        min_bars = 50

        for i in range(len(df)):
            if i < min_bars:
                results.append({
                    "cascade_probability": 0.0,
                    "cascade_signal": 0,
                    "cascade_direction": 0,
                    "cascade_strength": 0.0,
                })
                continue

            window = df.iloc[max(0, i - 500):i + 1]

            cs = None
            cd = None
            if crowding_scores is not None and i < len(crowding_scores):
                cs = crowding_scores.iloc[i].get("crowding_score", None)
                cd = crowding_scores.iloc[i].get("crowding_direction", None)

            pred = self.predict(window, cs, cd)
            results.append({
                "cascade_probability": pred.probability,
                "cascade_signal": pred.signal,
                "cascade_direction": pred.direction,
                "cascade_strength": pred.signal_strength,
            })

        return pd.DataFrame(results, index=df.index)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_val(row: pd.Series, col_names: list[str]) -> Optional[float]:
        """Get first available value from column name options."""
        for col in col_names:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    return float(val)
        return None

    @staticmethod
    def _empty_prediction(reason: str) -> CascadePrediction:
        return CascadePrediction(
            probability=0.0,
            direction=0,
            signal=0,
            signal_strength=0.0,
            preconditions={},
            reasoning=reason,
        )

"""
Dynamic Regime Allocator — Capital Rotation Based on Market State
=================================================================
Instead of static weights, shifts capital between strategies based on
detected regime. Validated result: funding_mr_v7 dominates high-vol,
fund_vol_squeeze dominates sideways.

    High Volatility → funding_mr_v7 (60%), extreme_spike (30%), squeeze (10%)
    Sideways        → fund_vol_squeeze (50%), funding_mr_v7 (30%), spike (20%)
    Bull Trend      → funding_mr_v7 (40%), momentum (40%), squeeze (20%)
    Bear Trend      → funding_mr_v7 (40%), extreme_spike (40%), squeeze (20%)

Weights are calibrated from institutional validation regime breakdown.
"""

import logging
from typing import Optional

import pandas as pd

from src.regime.detector import RegimeDetector

logger = logging.getLogger(__name__)


# ─── Regime Weight Profiles ─────────────────────────────────────

# Calibrated from institutional validation:
#   high_vol:  funding_mr_v7 PF=1.95, extreme_spike PF=1.40
#   sideways:  fund_vol_squeeze PF=2.01, funding_mr_v7 PF=0.98
DEFAULT_REGIME_WEIGHTS = {
    "high_volatility": {
        "funding_mr_v7":    0.50,
        "extreme_spike":    0.35,
        "fund_vol_squeeze": 0.10,
        "momentum_breakout": 0.05,
    },
    "sideways": {
        "funding_mr_v7":    0.20,
        "extreme_spike":    0.10,
        "fund_vol_squeeze": 0.50,
        "momentum_breakout": 0.20,
    },
    "bull_trend": {
        "funding_mr_v7":    0.30,
        "extreme_spike":    0.10,
        "fund_vol_squeeze": 0.20,
        "momentum_breakout": 0.40,
    },
    "bear_trend": {
        "funding_mr_v7":    0.35,
        "extreme_spike":    0.35,
        "fund_vol_squeeze": 0.15,
        "momentum_breakout": 0.15,
    },
}


class DynamicRegimeAllocator:
    """Dynamically shift capital allocation based on detected regime.

    Instead of equal-weighting strategies:
        1. Detect current regime
        2. Look up optimal weight profile for that regime
        3. Apply weights to position sizing

    The weight profiles are calibrated from backtested regime performance.
    """

    def __init__(
        self,
        regime_weights: dict = None,
        detector: Optional[RegimeDetector] = None,
        smoothing_bars: int = 24,  # Don't flip weights on every bar
    ):
        self.regime_weights = regime_weights or DEFAULT_REGIME_WEIGHTS
        self.detector = detector
        self.smoothing_bars = smoothing_bars
        self._fitted = False
        self._current_regime: str = "sideways"
        self._regime_history: Optional[pd.Series] = None
        self._bars_since_change: int = 0

    def fit(self, df: pd.DataFrame) -> "DynamicRegimeAllocator":
        """Fit regime detector on data."""
        if self.detector is None:
            self.detector = RegimeDetector()
        self.detector.fit(df)
        self._regime_history = self.detector.get_regime_history(df)
        self._fitted = True
        return self

    def detect_current(self, df: pd.DataFrame) -> str:
        """Detect current regime (with smoothing)."""
        if not self._fitted:
            self.fit(df)

        regime = self.detector.detect(df)
        new_regime = regime.value if hasattr(regime, 'value') else str(regime)

        # Smoothing: don't flip regimes faster than smoothing_bars
        if new_regime != self._current_regime:
            self._bars_since_change += 1
            if self._bars_since_change >= self.smoothing_bars:
                self._current_regime = new_regime
                self._bars_since_change = 0
        else:
            self._bars_since_change = 0

        return self._current_regime

    def get_weights(self, regime: str = None) -> dict:
        """Get strategy weights for a given regime."""
        if regime is None:
            regime = self._current_regime

        weights = self.regime_weights.get(regime)
        if weights is None:
            # Unknown regime → equal weight
            return {}

        return dict(weights)

    def get_position_size(
        self,
        strategy_name: str,
        base_size_pct: float,
        regime: str = None,
    ) -> float:
        """Get regime-adjusted position size for a strategy.

        Instead of uniform 1% sizing, scales by regime weight:
            adjusted = base_size × regime_weight × n_strategies
        So the total exposure stays the same, but allocation shifts.
        """
        weights = self.get_weights(regime)
        if not weights:
            return base_size_pct

        w = weights.get(strategy_name, 0)
        n = len(weights)
        # Scale so total exposure = n × base_size_pct (same as uniform)
        return base_size_pct * w * n

    def get_regime_timeline(self, df: pd.DataFrame) -> pd.Series:
        """Get per-bar regime labels for the full dataset."""
        if not self._fitted:
            self.fit(df)
        return self._regime_history

    def get_weight_timeline(self, df: pd.DataFrame, strategy_name: str) -> pd.Series:
        """Get per-bar weight for a strategy (for backtest integration)."""
        if not self._fitted:
            self.fit(df)

        regimes = self._regime_history
        weights = pd.Series(0.0, index=regimes.index)

        for i in range(len(regimes)):
            regime = str(regimes.iloc[i])
            w = self.regime_weights.get(regime, {}).get(strategy_name, 0.25)
            n = len(self.regime_weights.get(regime, {"_": 1}))
            weights.iloc[i] = w * n

        # Apply smoothing (rolling mean to prevent whiplash)
        if self.smoothing_bars > 1:
            weights = weights.rolling(
                self.smoothing_bars, min_periods=1
            ).mean()

        return weights

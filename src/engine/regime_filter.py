"""
Regime Filter — Trade Only When Conditions Favor Your Edge
==========================================================
Wraps any signal function and zeroes out signals in unfavorable regimes.

Usage:
    rf = RegimeFilter(allowed_regimes=["high_volatility"])
    filtered = rf.filter(df, raw_signals)

Can also be used as a factory wrapper:
    filtered_func = rf.wrap(original_signal_func)
    signals = filtered_func(df)
"""

import logging
from typing import Callable, Optional

import pandas as pd

from src.regime.detector import RegimeDetector, MarketRegime

logger = logging.getLogger(__name__)


class RegimeFilter:
    """Filter signals by market regime.

    The detector classifies bars into regimes (high_volatility, sideways,
    bull_trend, bear_trend). Only signals in allowed regimes pass through.
    """

    def __init__(
        self,
        allowed_regimes: list[str] = None,
        detector: Optional[RegimeDetector] = None,
    ):
        # Default: allow high_volatility + sideways (where funding_mr_v7 works)
        if allowed_regimes is None:
            allowed_regimes = ["high_volatility", "sideways"]

        self.allowed_regimes = set(allowed_regimes)
        self.detector = detector
        self._fitted = False
        self._regime_cache: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "RegimeFilter":
        """Fit the regime detector on historical data."""
        if self.detector is None:
            self.detector = RegimeDetector()
        self.detector.fit(df)
        self._regime_cache = self.detector.get_regime_history(df)
        self._fitted = True
        return self

    def filter(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Zero out signals that fall in disallowed regimes."""
        if not self._fitted:
            self.fit(df)

        regimes = self._regime_cache
        if regimes is None:
            regimes = self.detector.get_regime_history(df)

        # Align indices — regimes may have fewer rows (feature NaN drop)
        filtered = signals.copy()
        for i in range(len(filtered)):
            idx = filtered.index[i]
            if filtered.iloc[i] == 0:
                continue

            # Find regime for this bar
            if idx in regimes.index:
                regime = regimes.loc[idx]
            else:
                # Find nearest regime
                pos = regimes.index.searchsorted(idx)
                if pos >= len(regimes):
                    pos = len(regimes) - 1
                regime = regimes.iloc[pos]

            if str(regime) not in self.allowed_regimes:
                filtered.iloc[i] = 0

        return filtered

    def wrap(self, signal_func: Callable) -> Callable:
        """Wrap a signal function to auto-filter by regime.

        Returns a new function with the same signature that filters signals.
        """
        def wrapped(df: pd.DataFrame, **kwargs) -> pd.Series:
            raw = signal_func(df, **kwargs)
            return self.filter(df, raw)
        return wrapped

    def get_regime_stats(self, df: pd.DataFrame, signals: pd.Series) -> dict:
        """Report how many signals per regime (for diagnostics)."""
        if not self._fitted:
            self.fit(df)

        regimes = self._regime_cache
        if regimes is None:
            regimes = self.detector.get_regime_history(df)

        stats = {}
        active = signals[signals != 0]
        for idx in active.index:
            if idx in regimes.index:
                rn = str(regimes.loc[idx])
            else:
                pos = regimes.index.searchsorted(idx)
                if pos >= len(regimes):
                    pos = len(regimes) - 1
                rn = str(regimes.iloc[pos])

            stats.setdefault(rn, {"total": 0, "allowed": 0, "blocked": 0})
            stats[rn]["total"] += 1
            if rn in self.allowed_regimes:
                stats[rn]["allowed"] += 1
            else:
                stats[rn]["blocked"] += 1

        return stats

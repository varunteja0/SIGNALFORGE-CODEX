"""
SignalForge Regime Detection
=============================
Markets behave differently in different regimes (trending up, trending down,
ranging/choppy). A strategy that kills it in a trend will get destroyed in
a range. This module detects the current regime so we only run strategies
that match the current market state.

Uses Hidden Markov Model approach via Gaussian Mixture Models.
"""

import logging
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


class RegimeDetector:
    """Detects market regime using statistical clustering."""

    def __init__(self, n_regimes: int = 3, lookback_days: int = 90):
        self.n_regimes = n_regimes
        self.lookback_days = lookback_days
        self.model = None
        self.scaler = StandardScaler()
        self.regime_labels = {}

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build regime features from OHLCV data."""
        features = pd.DataFrame(index=df.index)

        # Trend features
        features["ret_20"] = df["close"].pct_change(20)
        features["ret_50"] = df["close"].pct_change(50)

        # Volatility features
        features["vol_20"] = df["close"].pct_change().rolling(20).std()
        features["vol_ratio"] = (
            df["close"].pct_change().rolling(10).std()
            / (df["close"].pct_change().rolling(50).std() + 1e-10)
        )

        # Mean reversion vs trend
        ma20 = df["close"].rolling(20).mean()
        features["price_vs_ma20"] = (df["close"] - ma20) / (ma20 + 1e-10)

        # Volume trend
        features["vol_trend"] = (
            df["volume"].rolling(10).mean()
            / (df["volume"].rolling(50).mean() + 1e-10)
        )

        return features.dropna()

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """Fit regime model on historical data."""
        features = self._build_features(df)
        if len(features) < 100:
            logger.warning("Not enough data to fit regime model")
            return self

        X = self.scaler.fit_transform(features.values)

        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            n_init=10,
            random_state=42,
        )
        self.model.fit(X)

        # Label regimes based on cluster characteristics
        labels = self.model.predict(X)
        features["regime"] = labels

        for regime_id in range(self.n_regimes):
            mask = features["regime"] == regime_id
            avg_ret = features.loc[mask, "ret_20"].mean()
            avg_vol = features.loc[mask, "vol_20"].mean()

            if avg_ret > 0.02 and avg_vol < features["vol_20"].quantile(0.7):
                self.regime_labels[regime_id] = MarketRegime.BULL_TREND
            elif avg_ret < -0.02 and avg_vol < features["vol_20"].quantile(0.7):
                self.regime_labels[regime_id] = MarketRegime.BEAR_TREND
            elif avg_vol >= features["vol_20"].quantile(0.7):
                self.regime_labels[regime_id] = MarketRegime.HIGH_VOLATILITY
            else:
                self.regime_labels[regime_id] = MarketRegime.SIDEWAYS

        logger.info(f"Regime model fit. Labels: {self.regime_labels}")
        return self

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime."""
        if self.model is None:
            logger.warning("Model not fitted, returning SIDEWAYS")
            return MarketRegime.SIDEWAYS

        features = self._build_features(df)
        if features.empty:
            return MarketRegime.SIDEWAYS

        X = self.scaler.transform(features.iloc[[-1]].values)
        regime_id = self.model.predict(X)[0]

        regime = self.regime_labels.get(regime_id, MarketRegime.SIDEWAYS)
        logger.info(f"Current regime: {regime.value}")
        return regime

    def get_regime_history(self, df: pd.DataFrame) -> pd.Series:
        """Get regime classification for each bar in the dataframe."""
        if self.model is None:
            return pd.Series(MarketRegime.SIDEWAYS.value, index=df.index)

        features = self._build_features(df)
        X = self.scaler.transform(features.values)
        labels = self.model.predict(X)

        regimes = pd.Series(
            [self.regime_labels.get(l, MarketRegime.SIDEWAYS).value for l in labels],
            index=features.index,
        )
        return regimes

    def get_regime_stats(self, df: pd.DataFrame) -> dict:
        """Get statistics for each regime period."""
        regimes = self.get_regime_history(df)
        returns = df["close"].pct_change()

        # Align indices — regimes may have fewer rows due to feature NaN drops
        common_idx = regimes.index.intersection(returns.index)
        regimes = regimes.loc[common_idx]
        returns = returns.loc[common_idx]

        stats = {}
        for regime in MarketRegime:
            mask = regimes == regime.value
            if mask.sum() == 0:
                continue

            r = returns[mask].dropna()
            stats[regime.value] = {
                "pct_of_time": mask.mean(),
                "avg_return": r.mean(),
                "volatility": r.std(),
                "sharpe": r.mean() / (r.std() + 1e-10) * np.sqrt(252 * 24),
                "bars": mask.sum(),
            }

        return stats

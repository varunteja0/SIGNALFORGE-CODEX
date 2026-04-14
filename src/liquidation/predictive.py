"""
Predictive Liquidation Timing — Front-Run the Overleveraging
===============================================================
Don't just map where liquidations ARE — predict when whales will
ADD leverage. Front-run the overleveraging, not just the liquidation.

Core insight: Leverage cycles follow patterns:
  1. Market pumps → leverage increases (greed)
  2. Funding rates spike → overcrowded longs
  3. A small dip triggers cascade → forced selling amplifies the drop
  4. Maximum fear → bottom → cycle repeats

This module:
  - Models leverage cycles with feature engineering
  - Predicts upcoming liquidation probability
  - Identifies optimal entry/exit timing around cascades
  - Uses historical liquidation events for pattern recognition
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


@dataclass
class LiquidationTimingPrediction:
    """Prediction about upcoming liquidation cascade."""
    timestamp: float
    asset: str

    # Probability estimates
    cascade_prob_1h: float = 0       # P(cascade in next 1 hour)
    cascade_prob_4h: float = 0       # P(cascade in next 4 hours)
    cascade_prob_24h: float = 0      # P(cascade in next 24 hours)

    # Leverage cycle position (0-1, where 1 = peak leverage)
    leverage_cycle_position: float = 0.5

    @property
    def probability_1h(self) -> float:
        return self.cascade_prob_1h

    @property
    def probability_4h(self) -> float:
        return self.cascade_prob_4h

    @property
    def probability_24h(self) -> float:
        return self.cascade_prob_24h

    @property
    def recommended_action(self) -> str:
        return self.action

    @property
    def recommended_reasoning(self) -> str:
        return self.reasoning

    # Expected cascade magnitude (% price drop)
    expected_cascade_pct: float = 0

    # Timing signals
    overleveraged: bool = False      # Market is dangerously leveraged
    cascade_imminent: bool = False   # Cascade likely within hours
    post_cascade_bounce: bool = False  # Bounce opportunity after cascade

    # Recommended action
    action: str = "HOLD"             # HOLD / SHORT_HEDGE / AGGRESSIVE_SHORT / BUY_DIP
    confidence: float = 0
    reasoning: str = ""

    # Feature contributions
    top_signals: dict = field(default_factory=dict)


class LiquidationTimingPredictor:
    """Predicts WHEN liquidation cascades will occur.

    Uses a combination of:
    1. Leverage cycle features (funding, OI, vol)
    2. Historical cascade pattern matching
    3. Gradient boosting for probability estimation
    4. Regime context for calibration
    """

    def __init__(
        self,
        cascade_threshold_pct: float = 5.0,   # Drop > 5% = cascade
        training_lookback_days: int = 365,
        prediction_horizons: list = None,
    ):
        self.cascade_threshold = cascade_threshold_pct
        self.training_lookback = training_lookback_days
        self.horizons = prediction_horizons or [1, 4, 24]  # hours

        self.scaler = StandardScaler()
        self.cascade_models: dict[int, GradientBoostingClassifier] = {}
        self.magnitude_model: Optional[GradientBoostingRegressor] = None

        self._is_trained = False
        self._feature_names: list[str] = []
        self._leverage_history: list[dict] = []

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build leverage cycle features from OHLCV data.

        These features capture the dynamics of leverage accumulation
        and the preconditions for cascades.
        """
        feat = pd.DataFrame(index=df.index)
        close = df["close"]
        volume = df.get("volume", pd.Series(1, index=df.index))

        # ============================================================
        # 1. VOLATILITY REGIME (cascades happen after vol compression)
        # ============================================================
        ret = close.pct_change()
        feat["realized_vol_1h"] = ret.rolling(1).std()  # Instant vol
        feat["realized_vol_4h"] = ret.rolling(4).std()
        feat["realized_vol_24h"] = ret.rolling(24).std()
        feat["vol_compression"] = (
            feat["realized_vol_4h"] / (feat["realized_vol_24h"] + 1e-10)
        )
        feat["vol_expansion_rate"] = feat["realized_vol_4h"].pct_change(4)

        # Garman-Klass vol (more efficient estimate)
        log_hl = np.log(df["high"] / df["low"]) ** 2
        log_co = np.log(close / df["open"]) ** 2
        feat["gk_vol_20"] = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(20).mean()

        # ============================================================
        # 2. MOMENTUM EXHAUSTION (cascades happen at extremes)
        # ============================================================
        feat["ret_1h"] = ret
        feat["ret_4h"] = close.pct_change(4)
        feat["ret_24h"] = close.pct_change(24)
        feat["ret_72h"] = close.pct_change(72)

        # RSI divergence
        rsi_14 = df.get("rsi_14", self._compute_rsi(close, 14))
        feat["rsi_14"] = rsi_14
        feat["rsi_overbought"] = (rsi_14 > 70).astype(float)
        feat["rsi_oversold"] = (rsi_14 < 30).astype(float)
        feat["rsi_momentum"] = rsi_14.diff(4)

        # Price vs moving averages (extended = vulnerable)
        ma_20 = close.rolling(20).mean()
        ma_50 = close.rolling(50).mean()
        feat["price_vs_ma20_pct"] = (close - ma_20) / (ma_20 + 1e-10)
        feat["price_vs_ma50_pct"] = (close - ma_50) / (ma_50 + 1e-10)

        # ============================================================
        # 3. VOLUME DYNAMICS (cascades preceded by volume patterns)
        # ============================================================
        vol_ma = volume.rolling(20).mean()
        feat["volume_ratio"] = volume / (vol_ma + 1e-10)
        feat["volume_trend"] = vol_ma.pct_change(10)
        feat["volume_spike"] = (volume / (vol_ma + 1e-10) > 2.0).astype(float)

        # Selling pressure
        feat["selling_pressure"] = np.where(
            close < df["open"],
            volume * (df["open"] - close) / (df["high"] - df["low"] + 1e-10),
            0
        )
        feat["selling_pressure_ma"] = pd.Series(
            feat["selling_pressure"], index=df.index
        ).rolling(10).mean()

        # ============================================================
        # 4. LEVERAGE PROXY FEATURES
        # ============================================================
        # Funding rate features (if available)
        if "avg_funding_rate" in df.columns:
            feat["funding_rate"] = df["avg_funding_rate"]
            feat["funding_zscore"] = (
                (feat["funding_rate"] - feat["funding_rate"].rolling(30).mean())
                / (feat["funding_rate"].rolling(30).std() + 1e-10)
            )
            feat["funding_extreme"] = (abs(feat["funding_zscore"]) > 2).astype(float)
            feat["funding_momentum"] = feat["funding_rate"].diff(8)
        else:
            # Proxy: use vol + returns to estimate leverage
            feat["funding_rate"] = 0
            feat["funding_zscore"] = 0
            feat["funding_extreme"] = 0
            feat["funding_momentum"] = 0

        # Open interest proxy
        if "open_interest_usd" in df.columns:
            feat["oi"] = df["open_interest_usd"]
            feat["oi_change_pct"] = feat["oi"].pct_change(1)
            feat["oi_zscore"] = (
                (feat["oi"] - feat["oi"].rolling(30).mean())
                / (feat["oi"].rolling(30).std() + 1e-10)
            )
        else:
            feat["oi"] = 0
            feat["oi_change_pct"] = 0
            feat["oi_zscore"] = 0

        # ============================================================
        # 5. CASCADE PRECONDITION SIGNALS
        # ============================================================
        # Bollinger Band squeeze (vol compression → breakout → cascade)
        bb_std = close.rolling(20).std()
        bb_upper = ma_20 + 2 * bb_std
        bb_lower = ma_20 - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / (ma_20 + 1e-10)
        feat["bb_width"] = bb_width
        feat["bb_squeeze"] = (bb_width < bb_width.rolling(50).quantile(0.1)).astype(float)

        # ATR contraction
        atr_14 = self._compute_atr(df, 14)
        atr_50 = self._compute_atr(df, 50)
        feat["atr_ratio"] = atr_14 / (atr_50 + 1e-10)
        feat["atr_contracting"] = (feat["atr_ratio"] < 0.7).astype(float)

        # Cumulative selling imbalance
        signed_vol = np.where(close > df["open"], volume, -volume)
        feat["cum_imbalance_20"] = pd.Series(signed_vol, index=df.index).rolling(20).sum()
        feat["imbalance_zscore"] = (
            (feat["cum_imbalance_20"] - feat["cum_imbalance_20"].rolling(50).mean())
            / (feat["cum_imbalance_20"].rolling(50).std() + 1e-10)
        )

        # ============================================================
        # 6. HISTORICAL CASCADE PATTERN FEATURES
        # ============================================================
        # Maximum drawdown over recent windows
        for window in [12, 24, 48]:
            rolling_max = close.rolling(window).max()
            feat[f"max_dd_{window}h"] = (close - rolling_max) / (rolling_max + 1e-10)

        # Speed of decline (how fast is it dropping?)
        feat["decline_speed"] = ret.rolling(4).apply(
            lambda x: x[x < 0].sum() if len(x) > 0 else 0
        )

        # Consecutive red candles
        red = (close < df["open"]).astype(int)
        feat["consecutive_red"] = red.groupby(
            (red != red.shift()).cumsum()
        ).cumsum() * red

        return feat.replace([np.inf, -np.inf], 0).fillna(0)

    def _compute_rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _create_labels(self, df: pd.DataFrame, horizon_hours: int) -> pd.Series:
        """Create binary labels: 1 = cascade happened within horizon."""
        close = df["close"]
        # Forward-looking minimum price within horizon
        future_min = close.shift(-horizon_hours).rolling(horizon_hours).min()
        # Did price drop more than threshold?
        drop_pct = (future_min - close) / close * 100
        cascade = (drop_pct < -self.cascade_threshold).astype(int)
        return cascade

    def train(self, df: pd.DataFrame):
        """Train cascade prediction models on historical data."""
        logger.info("Training liquidation timing predictor...")

        features = self._build_features(df)
        self._feature_names = list(features.columns)

        # Scale features
        X = self.scaler.fit_transform(features.values)

        # Train a model for each prediction horizon
        for horizon in self.horizons:
            labels = self._create_labels(df, horizon)

            # Remove NaN labels (future data not available)
            valid = labels.notna() & (labels.index.isin(features.index))
            X_valid = X[valid.values[:len(X)]]
            y_valid = labels[valid].values[:len(X_valid)]

            if len(X_valid) < 100 or y_valid.sum() < 10:
                logger.warning(
                    f"Insufficient cascade events for {horizon}h horizon: "
                    f"{y_valid.sum()} events in {len(y_valid)} samples"
                )
                continue

            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            best_model = None
            best_auc = 0

            for train_idx, val_idx in tscv.split(X_valid):
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                )
                model.fit(X_valid[train_idx], y_valid[train_idx])

                # Evaluate on validation
                if len(val_idx) > 0:
                    probs = model.predict_proba(X_valid[val_idx])
                    if probs.shape[1] > 1:
                        from sklearn.metrics import roc_auc_score
                        try:
                            auc = roc_auc_score(y_valid[val_idx], probs[:, 1])
                            if auc > best_auc:
                                best_auc = auc
                                best_model = model
                        except ValueError:
                            best_model = model

            if best_model:
                self.cascade_models[horizon] = best_model
                logger.info(f"  {horizon}h model: AUC={best_auc:.3f}")

        # Magnitude predictor
        close = df["close"]
        for horizon in [24]:
            future_min = close.shift(-horizon).rolling(horizon).min()
            magnitude = ((future_min - close) / close * 100).clip(-30, 0)

            valid = magnitude.notna()
            X_mag = X[valid.values[:len(X)]]
            y_mag = magnitude[valid].values[:len(X_mag)]

            if len(X_mag) > 100:
                self.magnitude_model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=4,
                    learning_rate=0.05, random_state=42,
                )
                self.magnitude_model.fit(X_mag, y_mag)

        self._is_trained = True
        logger.info("Liquidation timing predictor trained")

    def predict(self, df: pd.DataFrame, asset: str = "BTC") -> LiquidationTimingPrediction:
        """Predict upcoming cascade probability and timing."""
        features = self._build_features(df)

        if features.empty:
            return LiquidationTimingPrediction(
                timestamp=time.time(), asset=asset
            )

        X_latest = features.iloc[[-1]].values

        # Scale
        if self._is_trained:
            X_scaled = self.scaler.transform(X_latest)
        else:
            X_scaled = X_latest

        pred = LiquidationTimingPrediction(
            timestamp=time.time(), asset=asset,
        )

        # Get cascade probabilities from trained models
        if self._is_trained:
            for horizon, model in self.cascade_models.items():
                try:
                    probs = model.predict_proba(X_scaled)
                    cascade_prob = float(probs[0, 1]) if probs.shape[1] > 1 else 0
                except Exception:
                    cascade_prob = 0

                if horizon == 1:
                    pred.cascade_prob_1h = cascade_prob
                elif horizon == 4:
                    pred.cascade_prob_4h = cascade_prob
                elif horizon == 24:
                    pred.cascade_prob_24h = cascade_prob

            # Magnitude prediction
            if self.magnitude_model:
                pred.expected_cascade_pct = abs(float(
                    self.magnitude_model.predict(X_scaled)[0]
                ))

            # Feature importance
            if self.cascade_models:
                model = list(self.cascade_models.values())[0]
                importances = model.feature_importances_
                top_idx = np.argsort(importances)[-5:]
                pred.top_signals = {
                    self._feature_names[i]: float(importances[i])
                    for i in top_idx if i < len(self._feature_names)
                }
        else:
            # Heuristic fallback
            pred = self._heuristic_predict(features, df, asset)

        # Determine leverage cycle position
        latest = features.iloc[-1]
        pred.leverage_cycle_position = self._estimate_cycle_position(latest)

        # Generate action recommendation
        self._determine_action(pred)

        return pred

    def _heuristic_predict(
        self, features: pd.DataFrame, df: pd.DataFrame, asset: str
    ) -> LiquidationTimingPrediction:
        """Fallback prediction using heuristics when model not trained."""
        latest = features.iloc[-1]
        pred = LiquidationTimingPrediction(
            timestamp=time.time(), asset=asset,
        )

        # Heuristic cascade probability based on feature values
        risk_score = 0

        # High RSI + vol compression = danger
        rsi = latest.get("rsi_14", 50)
        if rsi > 75:
            risk_score += 0.2
        elif rsi < 25:
            risk_score += 0.1  # Already in cascade territory

        # Volume spike with decline
        if latest.get("volume_spike", 0) > 0 and latest.get("ret_4h", 0) < -0.02:
            risk_score += 0.2

        # BB squeeze about to break
        if latest.get("bb_squeeze", 0) > 0:
            risk_score += 0.15

        # Funding extreme
        if abs(latest.get("funding_zscore", 0)) > 2:
            risk_score += 0.15

        # Consecutive red candles
        if latest.get("consecutive_red", 0) >= 4:
            risk_score += 0.15

        # Price extended from MA
        if abs(latest.get("price_vs_ma20_pct", 0)) > 0.1:
            risk_score += 0.1

        pred.cascade_prob_4h = min(0.9, risk_score)
        pred.cascade_prob_1h = pred.cascade_prob_4h * 0.6
        pred.cascade_prob_24h = min(0.95, pred.cascade_prob_4h * 1.3)

        return pred

    def _estimate_cycle_position(self, features: pd.Series) -> float:
        """Estimate where we are in the leverage cycle (0=trough, 1=peak)."""
        signals = []

        # RSI position
        rsi = features.get("rsi_14", 50)
        signals.append(rsi / 100)

        # Price extension from MA
        ext = features.get("price_vs_ma20_pct", 0)
        signals.append(np.clip(ext * 5 + 0.5, 0, 1))

        # Volume trend
        vol_trend = features.get("volume_trend", 0)
        signals.append(np.clip(vol_trend + 0.5, 0, 1))

        # Funding
        funding_z = features.get("funding_zscore", 0)
        signals.append(np.clip(funding_z / 4 + 0.5, 0, 1))

        return float(np.mean(signals))

    def _determine_action(self, pred: LiquidationTimingPrediction):
        """Determine recommended action from prediction."""
        # Cascade imminent
        if pred.cascade_prob_4h > 0.7:
            pred.cascade_imminent = True
            pred.action = "AGGRESSIVE_SHORT"
            pred.confidence = pred.cascade_prob_4h
            pred.reasoning = (
                f"Cascade probability {pred.cascade_prob_4h:.0%} in 4h. "
                f"Expected drop: {pred.expected_cascade_pct:.1f}%"
            )
        elif pred.cascade_prob_24h > 0.6:
            pred.overleveraged = True
            pred.action = "SHORT_HEDGE"
            pred.confidence = pred.cascade_prob_24h * 0.8
            pred.reasoning = (
                f"Market overleveraged. {pred.cascade_prob_24h:.0%} chance of "
                f"cascade in 24h. Consider hedging."
            )
        elif pred.leverage_cycle_position < 0.2:
            pred.post_cascade_bounce = True
            pred.action = "BUY_DIP"
            pred.confidence = 0.6
            pred.reasoning = (
                "Leverage cycle at trough — past cascade. "
                "Bounce opportunity if fundamentals intact."
            )
        else:
            pred.action = "HOLD"
            pred.confidence = 0.5
            pred.reasoning = "No extreme leverage conditions detected."

    def compute_features(self, df: pd.DataFrame, asset: str = "BTC") -> dict:
        """Compute predictive liquidation features for GP engine."""
        pred = self.predict(df, asset)

        features = {
            "liq_cascade_prob_1h": pred.cascade_prob_1h,
            "liq_cascade_prob_4h": pred.cascade_prob_4h,
            "liq_cascade_prob_24h": pred.cascade_prob_24h,
            "liq_cycle_position": pred.leverage_cycle_position,
            "liq_expected_magnitude": pred.expected_cascade_pct / 30,  # Normalize
            "liq_imminent": 1.0 if pred.cascade_imminent else 0.0,
            "liq_overleveraged": 1.0 if pred.overleveraged else 0.0,
            "liq_post_cascade_bounce": 1.0 if pred.post_cascade_bounce else 0.0,
            "liq_short_signal": pred.confidence if pred.action in ("AGGRESSIVE_SHORT", "SHORT_HEDGE") else 0.0,
            "liq_buy_signal": pred.confidence if pred.action == "BUY_DIP" else 0.0,
            "pred_liq_1h": pred.cascade_prob_1h,
            "pred_liq_4h": pred.cascade_prob_4h,
            "pred_liq_24h": pred.cascade_prob_24h,
        }
        return features

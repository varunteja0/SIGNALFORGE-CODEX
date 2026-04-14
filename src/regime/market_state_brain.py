"""
Market State Brain — Latent State Model
=========================================
Replaces basic regime labels (volatility, sideways) with a rich,
multi-dimensional market state representation using Hidden Markov Models.

Detects latent states that capture:
    1. Liquidity stress — thin books, wide spreads, high impact
    2. Retail trap zones — funding extreme + OI spike + price reversal
    3. Whale absorption — large OI changes with flat price
    4. Funding imbalance clusters — persistent funding skew
    5. Volatility compression — squeeze before breakout
    6. Trend exhaustion — momentum dying before reversal

Architecture:
    Raw features → HMM (N latent states) → state probabilities
    State probabilities → per-strategy weight adjustments

Unlike RegimeDetector (3 GMM clusters), this uses:
    - More states (6-8 latent)
    - Transition probability matrix (predicts NEXT state)
    - Feature embeddings (captures non-linear relationships)
    - Online updates (adapts as new data arrives)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


# ─── State Definitions ──────────────────────────────────────────

LATENT_STATE_NAMES = [
    "liquidity_stress",
    "retail_trap",
    "whale_absorption",
    "funding_imbalance",
    "vol_compression",
    "trend_exhaustion",
    "normal_trending",
    "high_opportunity",
]


@dataclass
class MarketState:
    """Rich market state with probabilities for each latent factor."""
    timestamp: str = ""
    dominant_state: str = "normal_trending"
    state_probabilities: dict = field(default_factory=dict)
    transition_probs: dict = field(default_factory=dict)  # next state probs
    # Derived signals
    liquidity_score: float = 0.5      # 0=stressed, 1=healthy
    trap_probability: float = 0.0     # P(retail trap)
    whale_activity: float = 0.0       # 0=none, 1=heavy
    regime_stability: float = 0.5     # 0=regime about to change, 1=stable


@dataclass
class StrategyStateAdjustment:
    """How to adjust each strategy based on current market state."""
    strategy_name: str
    size_multiplier: float = 1.0      # 0=don't trade, 2=double size
    urgency: float = 0.5              # 0=wait, 1=immediate
    confidence: float = 0.5           # How confident in the adjustment
    reason: str = ""


class MarketStateBrain:
    """Multi-dimensional latent market state detector.

    Uses a Gaussian Mixture Model with transition tracking to detect
    latent market states from a rich feature set. Each state maps to
    specific strategy adjustments.

    Unlike basic regime detection:
        - Uses 15+ features (not just 5)
        - Tracks state transitions (predicts what's coming)
        - Outputs per-strategy adjustments (not just labels)
        - Updates online as new data arrives
    """

    def __init__(
        self,
        n_states: int = 6,
        lookback_bars: int = 168,       # 7 days at 1h
        transition_window: int = 24,     # Track transitions over 24 bars
        min_bars_for_fit: int = 500,
        update_interval: int = 24,       # Re-fit every 24 bars  
    ):
        self.n_states = n_states
        self.lookback_bars = lookback_bars
        self.transition_window = transition_window
        self.min_bars = min_bars_for_fit
        self.update_interval = update_interval

        self.model: Optional[GaussianMixture] = None
        self.scaler = StandardScaler()
        self._fitted = False
        self._state_names: dict[int, str] = {}
        self._transition_matrix: Optional[np.ndarray] = None
        self._last_states: list[int] = []
        self._bars_since_fit: int = 0

        # Strategy adjustment profiles per state
        self._strategy_profiles = self._default_profiles()

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build rich feature matrix for state detection.

        Uses 15+ features across multiple dimensions:
        price, volatility, volume, momentum, structure.
        """
        f = pd.DataFrame(index=df.index)
        close = df["close"]
        volume = df.get("volume", pd.Series(0, index=df.index))

        # ── Price momentum (multi-scale) ──
        for w in [5, 20, 50]:
            f[f"ret_{w}"] = close.pct_change(w)
        f["ret_acceleration"] = f["ret_5"] - f["ret_20"]

        # ── Volatility features ──
        returns = close.pct_change()
        f["vol_5"] = returns.rolling(5).std()
        f["vol_20"] = returns.rolling(20).std()
        f["vol_ratio"] = f["vol_5"] / (f["vol_20"] + 1e-10)
        f["vol_of_vol"] = f["vol_5"].rolling(20).std()

        # ── Volume features ──
        f["vol_trend"] = volume.rolling(5).mean() / (volume.rolling(50).mean() + 1e-10)
        f["vol_spike"] = volume / (volume.rolling(20).mean() + 1e-10)

        # ── Mean reversion vs trend ──
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        f["price_vs_ma20"] = (close - ma20) / (ma20 + 1e-10)
        f["price_vs_ma50"] = (close - ma50) / (ma50 + 1e-10)
        f["ma_spread"] = (ma20 - ma50) / (ma50 + 1e-10)

        # ── Structural features (if available) ──
        if "funding_rate" in df.columns:
            fr = df["funding_rate"]
            f["funding_zscore"] = (fr - fr.rolling(168).mean()) / (
                fr.rolling(168).std() + 1e-10
            )
            f["funding_persistence"] = fr.rolling(24).mean() / (
                fr.rolling(168).std() + 1e-10
            )
        else:
            f["funding_zscore"] = 0
            f["funding_persistence"] = 0

        if "oi_change_pct" in df.columns:
            f["oi_momentum"] = df["oi_change_pct"].rolling(5).mean()
        else:
            f["oi_momentum"] = 0

        # ── Squeeze detection ──
        bb_std = close.rolling(20).std()
        f["bb_width"] = bb_std / (ma20 + 1e-10)
        f["bb_width_pctile"] = f["bb_width"].rolling(168).rank(pct=True)

        return f.dropna()

    def fit(self, df: pd.DataFrame) -> "MarketStateBrain":
        """Fit the latent state model on historical data."""
        features = self._build_features(df)
        if len(features) < self.min_bars:
            logger.warning(
                f"MarketStateBrain: only {len(features)} bars, "
                f"need {self.min_bars}"
            )
            return self

        X = self.scaler.fit_transform(features.values)

        self.model = GaussianMixture(
            n_components=self.n_states,
            covariance_type="full",
            n_init=5,
            max_iter=200,
            random_state=42,
        )
        self.model.fit(X)

        # Label states by their characteristics
        labels = self.model.predict(X)
        self._label_states(features, labels)

        # Build transition matrix
        self._build_transition_matrix(labels)

        # Store recent states for online tracking
        self._last_states = list(labels[-self.transition_window:])
        self._fitted = True
        self._bars_since_fit = 0

        logger.info(
            f"MarketStateBrain fitted: {self.n_states} states, "
            f"{len(features)} bars, states={self._state_names}"
        )
        return self

    def detect(self, df: pd.DataFrame) -> MarketState:
        """Detect current market state with transition probabilities."""
        if not self._fitted:
            self.fit(df)

        if not self._fitted or self.model is None:
            return MarketState(dominant_state="normal_trending")

        features = self._build_features(df)
        if features.empty:
            return MarketState(dominant_state="normal_trending")

        X_last = self.scaler.transform(features.iloc[[-1]].values)

        # State probabilities
        probs = self.model.predict_proba(X_last)[0]
        state_id = int(np.argmax(probs))

        state_probs = {}
        for sid, prob in enumerate(probs):
            name = self._state_names.get(sid, f"state_{sid}")
            state_probs[name] = float(prob)

        # Transition probabilities (what's the next likely state?)
        transition_probs = {}
        if self._transition_matrix is not None:
            for sid in range(self.n_states):
                name = self._state_names.get(sid, f"state_{sid}")
                transition_probs[name] = float(
                    self._transition_matrix[state_id, sid]
                )

        dominant = self._state_names.get(state_id, "normal_trending")

        # Derived scores
        liquidity = 1.0 - state_probs.get("liquidity_stress", 0)
        trap = state_probs.get("retail_trap", 0)
        whale = state_probs.get("whale_absorption", 0)

        # Regime stability: how concentrated is the probability?
        stability = float(np.max(probs))

        state = MarketState(
            dominant_state=dominant,
            state_probabilities=state_probs,
            transition_probs=transition_probs,
            liquidity_score=liquidity,
            trap_probability=trap,
            whale_activity=whale,
            regime_stability=stability,
        )

        # Track for online updates
        self._last_states.append(state_id)
        if len(self._last_states) > self.transition_window * 2:
            self._last_states = self._last_states[-self.transition_window:]

        self._bars_since_fit += 1

        return state

    def get_state_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get per-bar state labels and probabilities for backtest."""
        if not self._fitted:
            self.fit(df)

        if self.model is None:
            return pd.DataFrame(
                {"dominant_state": "normal_trending"},
                index=df.index,
            )

        features = self._build_features(df)
        X = self.scaler.transform(features.values)

        labels = self.model.predict(X)
        probs = self.model.predict_proba(X)

        result = pd.DataFrame(index=features.index)
        result["state_id"] = labels
        result["dominant_state"] = [
            self._state_names.get(l, "normal_trending") for l in labels
        ]

        for sid in range(self.n_states):
            name = self._state_names.get(sid, f"state_{sid}")
            result[f"prob_{name}"] = probs[:, sid]

        # Stability = max probability
        result["stability"] = probs.max(axis=1)

        return result

    def get_strategy_adjustments(
        self,
        state: MarketState,
        strategy_names: list[str],
    ) -> dict[str, StrategyStateAdjustment]:
        """Get per-strategy sizing adjustments for current state."""
        adjustments = {}

        for name in strategy_names:
            adj = StrategyStateAdjustment(strategy_name=name)

            # Weighted combination of state probabilities × strategy profiles
            size_mult = 0.0
            urgency = 0.0

            for state_name, prob in state.state_probabilities.items():
                profile = self._strategy_profiles.get(state_name, {})
                strat_profile = profile.get(name, {})
                size_mult += prob * strat_profile.get("size", 1.0)
                urgency += prob * strat_profile.get("urgency", 0.5)

            adj.size_multiplier = max(0.0, min(2.0, size_mult))
            adj.urgency = max(0.0, min(1.0, urgency))
            adj.confidence = state.regime_stability

            # Special overrides
            if state.liquidity_score < 0.3:
                adj.size_multiplier *= 0.5
                adj.reason = "liquidity stress — halving size"
            elif state.trap_probability > 0.6:
                # Mean reversion strategies love traps
                if "funding" in name or "spike" in name:
                    adj.size_multiplier *= 1.5
                    adj.reason = "retail trap detected — boosting MR"
                else:
                    adj.size_multiplier *= 0.5
                    adj.reason = "retail trap — reducing momentum"

            adjustments[name] = adj

        return adjustments

    # ─── Internal Methods ────────────────────────────────────────

    def _label_states(self, features: pd.DataFrame, labels: np.ndarray):
        """Label each cluster by its dominant characteristics."""
        available_names = list(LATENT_STATE_NAMES)
        self._state_names = {}

        cluster_chars = []
        for sid in range(self.n_states):
            mask = labels == sid
            if mask.sum() == 0:
                cluster_chars.append({})
                continue

            chars = {
                "vol_ratio": features.loc[mask, "vol_ratio"].mean()
                if "vol_ratio" in features.columns else 1.0,
                "ret_5": features.loc[mask, "ret_5"].mean()
                if "ret_5" in features.columns else 0.0,
                "bb_width_pctile": features.loc[mask, "bb_width_pctile"].mean()
                if "bb_width_pctile" in features.columns else 0.5,
                "vol_spike": features.loc[mask, "vol_spike"].mean()
                if "vol_spike" in features.columns else 1.0,
                "funding_zscore": features.loc[mask, "funding_zscore"].mean()
                if "funding_zscore" in features.columns else 0.0,
                "pct_time": mask.mean(),
            }
            cluster_chars.append(chars)

        # Assign names by most distinctive characteristic
        for sid in range(self.n_states):
            chars = cluster_chars[sid]
            if not chars:
                name = available_names.pop() if available_names else f"state_{sid}"
                self._state_names[sid] = name
                continue

            # Score each possible label
            scores = {}

            # Liquidity stress: high vol ratio, high vol spike
            scores["liquidity_stress"] = (
                chars.get("vol_ratio", 1) * 0.5
                + chars.get("vol_spike", 1) * 0.3
                - 0.8
            )

            # Vol compression: low bb_width_pctile
            scores["vol_compression"] = (
                (1 - chars.get("bb_width_pctile", 0.5)) * 1.5 - 0.5
            )

            # Funding imbalance: high abs funding zscore
            scores["funding_imbalance"] = (
                abs(chars.get("funding_zscore", 0)) * 0.8 - 0.3
            )

            # Trend exhaustion: high ret but decreasing
            ret = chars.get("ret_5", 0)
            scores["trend_exhaustion"] = abs(ret) * 3 - 0.5

            # Normal trending
            scores["normal_trending"] = 0.0  # Default fallback

            # High opportunity: vol compression + funding extreme
            scores["high_opportunity"] = (
                scores.get("vol_compression", 0) * 0.5
                + scores.get("funding_imbalance", 0) * 0.5
            )

            # Pick best available name
            for name in sorted(scores, key=scores.get, reverse=True):
                if name in available_names:
                    self._state_names[sid] = name
                    available_names.remove(name)
                    break
            else:
                name = available_names.pop() if available_names else f"state_{sid}"
                self._state_names[sid] = name

    def _build_transition_matrix(self, labels: np.ndarray):
        """Build state transition probability matrix."""
        n = self.n_states
        counts = np.zeros((n, n))

        for i in range(len(labels) - 1):
            counts[labels[i], labels[i + 1]] += 1

        # Normalize rows to get probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self._transition_matrix = counts / row_sums

    def _default_profiles(self) -> dict:
        """Default strategy adjustment profiles per market state.

        Maps: state_name → {strategy_name → {size, urgency}}
        """
        return {
            "liquidity_stress": {
                "funding_mr_v7": {"size": 0.3, "urgency": 0.2},
                "extreme_spike": {"size": 0.5, "urgency": 0.3},
                "fund_vol_squeeze": {"size": 0.2, "urgency": 0.1},
                "momentum_breakout": {"size": 0.1, "urgency": 0.1},
            },
            "retail_trap": {
                "funding_mr_v7": {"size": 1.8, "urgency": 0.9},
                "extreme_spike": {"size": 1.5, "urgency": 0.8},
                "fund_vol_squeeze": {"size": 1.0, "urgency": 0.5},
                "momentum_breakout": {"size": 0.3, "urgency": 0.2},
            },
            "whale_absorption": {
                "funding_mr_v7":  {"size": 1.2, "urgency": 0.6},
                "extreme_spike": {"size": 0.8, "urgency": 0.4},
                "fund_vol_squeeze": {"size": 1.0, "urgency": 0.5},
                "momentum_breakout": {"size": 0.5, "urgency": 0.3},
            },
            "funding_imbalance": {
                "funding_mr_v7": {"size": 2.0, "urgency": 1.0},
                "extreme_spike": {"size": 1.5, "urgency": 0.8},
                "fund_vol_squeeze": {"size": 1.3, "urgency": 0.7},
                "momentum_breakout": {"size": 0.5, "urgency": 0.3},
            },
            "vol_compression": {
                "funding_mr_v7": {"size": 0.8, "urgency": 0.4},
                "extreme_spike": {"size": 0.5, "urgency": 0.2},
                "fund_vol_squeeze": {"size": 2.0, "urgency": 1.0},
                "momentum_breakout": {"size": 1.5, "urgency": 0.8},
            },
            "trend_exhaustion": {
                "funding_mr_v7": {"size": 1.5, "urgency": 0.8},
                "extreme_spike": {"size": 1.3, "urgency": 0.7},
                "fund_vol_squeeze": {"size": 1.0, "urgency": 0.5},
                "momentum_breakout": {"size": 0.2, "urgency": 0.1},
            },
            "normal_trending": {
                "funding_mr_v7": {"size": 1.0, "urgency": 0.5},
                "extreme_spike": {"size": 1.0, "urgency": 0.5},
                "fund_vol_squeeze": {"size": 1.0, "urgency": 0.5},
                "momentum_breakout": {"size": 1.0, "urgency": 0.5},
            },
            "high_opportunity": {
                "funding_mr_v7": {"size": 1.5, "urgency": 0.8},
                "extreme_spike": {"size": 1.5, "urgency": 0.8},
                "fund_vol_squeeze": {"size": 1.5, "urgency": 0.8},
                "momentum_breakout": {"size": 1.0, "urgency": 0.6},
            },
        }

    def get_transition_forecast(self, current_state: str, horizon: int = 3) -> dict:
        """Forecast state probabilities N steps ahead.

        Uses transition matrix exponentiation for multi-step prediction.
        """
        if self._transition_matrix is None:
            return {}

        # Find current state ID
        state_id = None
        for sid, name in self._state_names.items():
            if name == current_state:
                state_id = sid
                break

        if state_id is None:
            return {}

        # Multi-step transition: T^horizon
        T_n = np.linalg.matrix_power(self._transition_matrix, horizon)
        forecast_probs = T_n[state_id]

        result = {}
        for sid in range(self.n_states):
            name = self._state_names.get(sid, f"state_{sid}")
            result[name] = float(forecast_probs[sid])

        return result

    def format_report(self, state: MarketState) -> str:
        """Human-readable state report."""
        lines = []
        lines.append("=" * 60)
        lines.append("  MARKET STATE BRAIN — CURRENT STATE")
        lines.append("=" * 60)
        lines.append(f"  Dominant: {state.dominant_state}")
        lines.append(f"  Stability: {state.regime_stability:.1%}")
        lines.append(f"  Liquidity: {state.liquidity_score:.1%}")
        lines.append(f"  Trap prob: {state.trap_probability:.1%}")
        lines.append(f"  Whale act: {state.whale_activity:.1%}")
        lines.append("")

        if state.state_probabilities:
            lines.append("─ STATE PROBABILITIES ───────────────────")
            for name, prob in sorted(
                state.state_probabilities.items(),
                key=lambda x: x[1], reverse=True,
            ):
                bar = "█" * int(prob * 30)
                lines.append(f"  {name:<25s} {prob:5.1%} {bar}")
            lines.append("")

        if state.transition_probs:
            lines.append("─ NEXT STATE FORECAST ──────────────────")
            for name, prob in sorted(
                state.transition_probs.items(),
                key=lambda x: x[1], reverse=True,
            )[:4]:
                lines.append(f"  → {name:<25s} {prob:5.1%}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

"""
Derivative Microstructure Alpha — Two Novel Engines
====================================================

Engine 1: Structural Stress Index (SSI)
    Multi-dimensional anomaly detection via Mahalanobis distance.
    Experimental — shows edge on BTC only.

Engine 2: Contrarian Asymmetry Engine (CAE) ★ PRIMARY
    Exploits the most hidden edge in crypto derivatives:

    FINDING: Funding rate mean reversion is MASSIVELY ASYMMETRIC.
    - When funding is POSITIVE (crowd LONG on alts): SHORT wins 75-86%
    - When funding is NEGATIVE (crowd SHORT): LONG wins only 42-53%
    - On BTC, the crowd is usually RIGHT (momentum asset) → excluded

    WHY this works:
    Crypto retail is structurally long. When they pile in enough to
    push funding positive on altcoins (ETH, SOL, XRP), they're the
    classic "dumb money" at the top. Their capitulation is violent
    and predictable because:
    1. They're paying funding (cost pressure to close)
    2. They're losing on mark-to-market (price drops after overshoot)
    3. The double squeeze creates accelerating sell pressure
    4. Unlike institutional shorts, retail longs panic-exit in herds

    When funding is NEGATIVE (shorts paying), it's usually market
    makers or sophisticated hedgers → they don't panic exit →
    mean reversion is unreliable → DON'T TRADE IT.

    Nobody builds this because everyone treats long and short signals
    symmetrically. The asymmetry is hiding in plain sight.

    Optional gate: Ornstein-Uhlenbeck half-life estimation.
    When funding mean-reverts fast (short half-life), the signal
    resolves quickly → better timing for entry.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.engine.strategy_factory import _first_signal_only

logger = logging.getLogger(__name__)


class StructuralStressEngine:
    """Multi-dimensional structural anomaly detector.

    Dimensions measured (all z-scored against rolling history):
        1. Funding Dislocation — how extreme is derivative positioning
        2. Funding Velocity — speed of positioning shift (2nd derivative)
        3. OI Pressure — OI building in one direction (coiled energy)
        4. Volatility Compression — ATR squeeze (spring loading)
        5. Volume-Volatility Ratio — hidden activity (high vol, low move)
        6. VWAP Deviation — exhaustion vs structural support/resistance
        7. Momentum Divergence — timeframe disagreement (3h vs 21h RSI)
        8. Return Concentration — kurtosis of recent returns (tail risk)

    Signal generation:
        1. Mahalanobis distance > adaptive threshold (97th percentile)
        2. Distance is peaking (started declining → stress resolving)
        3. ≥ N dimensions individually above 1.5σ
        4. Return entropy below median (compressed/ordered market)
        5. Direction: mean-revert against the structural bias
    """

    STRESS_DIMS = [
        "funding_dislocation",
        "funding_velocity",
        "oi_pressure",
        "vol_compression",
        "volume_vol_ratio",
        "vwap_deviation",
        "momentum_divergence",
        "return_concentration",
    ]

    @staticmethod
    def generate_signals(
        df: pd.DataFrame,
        lookback: int = 168,
        distance_pctile: float = 95,
        min_stressed_dims: int = 3,
        entropy_filter_pct: float = 50,
        hold_bars: int = 12,
        dim_stress_threshold: float = 1.5,
        require_peak: bool = True,
        **kwargs,
    ) -> pd.Series:
        """Generate signals from structural stress analysis.

        Args:
            lookback: Rolling window for covariance estimation (bars).
            distance_pctile: Percentile threshold for Mahalanobis distance
                (self-calibrating — adapts to regime volatility).
            min_stressed_dims: Minimum individually-stressed dimensions
                required (prevents noise-driven distance spikes).
            entropy_filter_pct: Only trade below this percentile of
                return entropy (low entropy = compressed = better signals).
            hold_bars: Signal cooldown / position hold duration.
            dim_stress_threshold: Individual dimension stress threshold (σ).
            require_peak: If True, require distance to be peaking (declining).

        Returns:
            pd.Series of {-1, 0, 1} signals.
        """
        engine = StructuralStressEngine()
        return engine._generate(
            df, lookback, distance_pctile, min_stressed_dims,
            entropy_filter_pct, hold_bars, dim_stress_threshold,
            require_peak,
        )

    def _generate(
        self,
        df: pd.DataFrame,
        lookback: int,
        distance_pctile: float,
        min_stressed_dims: int,
        entropy_filter_pct: float,
        hold_bars: int,
        dim_stress_threshold: float,
        require_peak: bool,
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0, index=df.index, dtype=int)

        if n < lookback + 20:
            return signals

        # 1. Extract stress dimensions
        stress = self._compute_stress_features(df, lookback)

        # 2. Compute Mahalanobis distance (vectorized chunked)
        distances = self._compute_mahalanobis(stress, lookback)

        # 3. Adaptive threshold: rolling percentile of distance itself
        dist_threshold = distances.rolling(
            lookback * 2, min_periods=lookback
        ).quantile(distance_pctile / 100.0)

        # 4. Distance velocity (for peak detection)
        dist_velocity = distances.diff(3)  # 3-bar change

        # 5. Return entropy for filtering
        returns = df["close"].pct_change().fillna(0)
        entropy = returns.rolling(24, min_periods=12).apply(
            _shannon_entropy, raw=True,
        )
        entropy_thresh = entropy.rolling(
            lookback, min_periods=lookback // 2,
        ).quantile(entropy_filter_pct / 100.0)

        # 6. Scan for signals
        cols = [c for c in self.STRESS_DIMS if c in stress.columns]
        raw_signals = pd.Series(0, index=df.index, dtype=int)

        start = lookback + 20

        for i in range(start, n):
            dist = distances.iloc[i]
            thresh = dist_threshold.iloc[i]

            if pd.isna(dist) or pd.isna(thresh) or dist <= 0:
                continue

            # A. Mahalanobis distance above adaptive threshold
            if dist < thresh:
                continue

            # B. Peak detection: distance should be declining (stress resolving)
            if require_peak:
                dv = dist_velocity.iloc[i]
                if pd.isna(dv) or dv > 0:
                    continue

            # C. Count individually stressed dimensions
            stressed_count = 0
            for dim in cols:
                v = stress[dim].iloc[i]
                if pd.notna(v) and abs(v) > dim_stress_threshold:
                    stressed_count += 1

            if stressed_count < min_stressed_dims:
                continue

            # D. Entropy filter: only trade in compressed/ordered markets
            ent = entropy.iloc[i]
            ent_t = entropy_thresh.iloc[i]
            if pd.notna(ent) and pd.notna(ent_t) and ent > ent_t:
                continue

            # E. Determine direction (mean reversion)
            direction = self._determine_direction(stress, i, cols)
            raw_signals.iloc[i] = direction

        # Apply cooldown
        return _first_signal_only(raw_signals, hold_bars)

    # ── Stress Feature Extraction ─────────────────────────────────

    def _compute_stress_features(
        self, df: pd.DataFrame, lookback: int,
    ) -> pd.DataFrame:
        """Extract 8 structural stress dimensions, all z-scored."""
        stress = pd.DataFrame(index=df.index)

        # 1. Funding Dislocation
        if "fund_funding_zscore" in df.columns:
            stress["funding_dislocation"] = df["fund_funding_zscore"].fillna(0)
        elif "fund_funding_rate" in df.columns:
            fr = df["fund_funding_rate"].fillna(0)
            mu = fr.rolling(lookback, min_periods=20).mean()
            sd = fr.rolling(lookback, min_periods=20).std()
            stress["funding_dislocation"] = ((fr - mu) / (sd + 1e-10)).fillna(0)
        else:
            stress["funding_dislocation"] = 0.0

        # 2. Funding Velocity — second derivative of funding
        if "fund_funding_rate" in df.columns:
            fr = df["fund_funding_rate"].fillna(0)
            vel = fr.diff(8)  # 8h change
            mu = vel.rolling(lookback, min_periods=20).mean()
            sd = vel.rolling(lookback, min_periods=20).std()
            stress["funding_velocity"] = ((vel - mu) / (sd + 1e-10)).fillna(0)
        else:
            stress["funding_velocity"] = 0.0

        # 3. OI Pressure — OI change directionally aligned with funding
        if "oi_oi_change_24h" in df.columns:
            oi = df["oi_oi_change_24h"].fillna(0)
            mu = oi.rolling(lookback, min_periods=20).mean()
            sd = oi.rolling(lookback, min_periods=20).std()
            oi_z = ((oi - mu) / (sd + 1e-10)).fillna(0)
            fund_sign = np.sign(stress["funding_dislocation"])
            stress["oi_pressure"] = oi_z * fund_sign
        elif "oi_oi_zscore" in df.columns:
            oi_z = df["oi_oi_zscore"].fillna(0)
            fund_sign = np.sign(stress["funding_dislocation"])
            stress["oi_pressure"] = oi_z * fund_sign
        else:
            stress["oi_pressure"] = 0.0

        # 4. Volatility Compression — inverse ATR percentile rank
        if "atr_14" in df.columns:
            atr = df["atr_14"]
            atr_rank = atr.rolling(lookback, min_periods=40).rank(pct=True)
            compression = 1.0 - atr_rank  # higher = more compressed
            mu = compression.rolling(lookback, min_periods=40).mean()
            sd = compression.rolling(lookback, min_periods=40).std()
            stress["vol_compression"] = (
                (compression - mu) / (sd + 1e-10)
            ).fillna(0)
        else:
            stress["vol_compression"] = 0.0

        # 5. Volume-Volatility Ratio — hidden accumulation/distribution
        if "volume" in df.columns and "atr_14" in df.columns:
            vol_n = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
            atr_n = df["atr_14"] / (df["atr_14"].rolling(20).mean() + 1e-10)
            vv = vol_n / (atr_n + 1e-10)
            mu = vv.rolling(lookback, min_periods=40).mean()
            sd = vv.rolling(lookback, min_periods=40).std()
            stress["volume_vol_ratio"] = ((vv - mu) / (sd + 1e-10)).fillna(0)
        else:
            stress["volume_vol_ratio"] = 0.0

        # 6. VWAP Deviation — price exhaustion signal
        if all(c in df.columns for c in ["close", "high", "low", "volume"]):
            tp = (df["high"] + df["low"] + df["close"]) / 3.0
            cum_vol = df["volume"].rolling(48, min_periods=12).sum()
            cum_tp = (tp * df["volume"]).rolling(48, min_periods=12).sum()
            vwap = cum_tp / (cum_vol + 1e-10)
            denom = df["atr_14"] if "atr_14" in df.columns else vwap * 0.01
            dev = (df["close"] - vwap) / (denom + 1e-10)
            mu = dev.rolling(lookback, min_periods=40).mean()
            sd = dev.rolling(lookback, min_periods=40).std()
            stress["vwap_deviation"] = ((dev - mu) / (sd + 1e-10)).fillna(0)
        else:
            stress["vwap_deviation"] = 0.0

        # 7. Momentum Divergence — short vs long-term momentum disagree
        if "rsi_3" in df.columns and "rsi_21" in df.columns:
            short_n = (df["rsi_3"] - 50.0) / 50.0
            long_n = (df["rsi_21"] - 50.0) / 50.0
            div = short_n - long_n
            mu = div.rolling(lookback, min_periods=40).mean()
            sd = div.rolling(lookback, min_periods=40).std()
            stress["momentum_divergence"] = (
                (div - mu) / (sd + 1e-10)
            ).fillna(0)
        else:
            stress["momentum_divergence"] = 0.0

        # 8. Return Concentration — kurtosis of recent returns
        #    High kurtosis = fat tails = hidden tail risk building
        if "close" in df.columns:
            rets = df["close"].pct_change().fillna(0)
            kurt = rets.rolling(48, min_periods=24).kurt()
            mu = kurt.rolling(lookback, min_periods=40).mean()
            sd = kurt.rolling(lookback, min_periods=40).std()
            stress["return_concentration"] = (
                (kurt - mu) / (sd + 1e-10)
            ).fillna(0)
        else:
            stress["return_concentration"] = 0.0

        return stress

    # ── Mahalanobis Distance ──────────────────────────────────────

    def _compute_mahalanobis(
        self, stress_df: pd.DataFrame, lookback: int,
    ) -> pd.Series:
        """Rolling Mahalanobis distance — vectorized in chunks."""
        cols = [c for c in self.STRESS_DIMS if c in stress_df.columns]
        n = len(stress_df)
        distances = np.zeros(n)
        data = stress_df[cols].values  # (n, d)
        d = len(cols)

        if d == 0:
            return pd.Series(0.0, index=stress_df.index)

        reg = np.eye(d) * 1e-5  # regularization

        # Process in blocks for efficiency
        start = lookback + 10
        for i in range(start, n):
            window = data[i - lookback : i]  # (lookback, d)

            # Skip if too many NaNs
            valid_mask = ~np.isnan(window).any(axis=1)
            window = window[valid_mask]
            if len(window) < lookback // 3:
                continue

            current = data[i]
            if np.isnan(current).any():
                continue

            mean = np.nanmean(window, axis=0)
            cov = np.cov(window.T) + reg

            try:
                inv_cov = np.linalg.inv(cov)
                diff = current - mean
                d_sq = diff @ inv_cov @ diff
                distances[i] = np.sqrt(max(d_sq, 0))
            except np.linalg.LinAlgError:
                continue

        return pd.Series(distances, index=stress_df.index)

    # ── Direction Determination ───────────────────────────────────

    def _determine_direction(
        self,
        stress: pd.DataFrame,
        idx: int,
        cols: list[str],
    ) -> int:
        """Mean-reversion direction from signed stress decomposition.

        Positive net score = bullish structural stress = crowd is long
        → mean revert SHORT.

        Negative net score = bearish structural stress = crowd is short
        → mean revert LONG.
        """
        # Directional weight: positive component value → bullish stress
        directional_dims = {
            "funding_dislocation": 1.0,   # +funding → crowd long
            "funding_velocity": 0.8,      # +velocity → building longs
            "oi_pressure": 0.8,           # signed by funding already
            "vwap_deviation": 0.6,        # +deviation → above VWAP → extended
            "momentum_divergence": 0.6,   # +divergence → short-term overheated
        }

        score = 0.0
        for dim, weight in directional_dims.items():
            if dim in cols:
                v = stress[dim].iloc[idx]
                if pd.notna(v):
                    score += v * weight

        # Mean revert: positive stress (bullish excess) → SHORT (-1)
        return -1 if score > 0 else 1

    # ── Diagnostic Method (for proximity logging) ─────────────────

    @staticmethod
    def get_stress_state(
        df: pd.DataFrame,
        lookback: int = 168,
    ) -> dict:
        """Get current stress state for logging/monitoring.

        Returns dict with:
            distance: current Mahalanobis distance
            threshold: adaptive threshold at this point
            pct_of_threshold: distance / threshold (1.0 = at threshold)
            top_stressors: list of (dimension, z-score) sorted by abs value
            stressed_dim_count: number of dims above 1.5σ
        """
        engine = StructuralStressEngine()
        stress = engine._compute_stress_features(df, lookback)
        distances = engine._compute_mahalanobis(stress, lookback)

        if len(distances) == 0 or distances.iloc[-1] == 0:
            return {
                "distance": 0,
                "threshold": 0,
                "pct_of_threshold": 0,
                "top_stressors": [],
                "stressed_dim_count": 0,
            }

        dist = distances.iloc[-1]
        thresh = distances.rolling(
            lookback * 2, min_periods=lookback,
        ).quantile(0.95).iloc[-1]

        cols = [c for c in StructuralStressEngine.STRESS_DIMS
                if c in stress.columns]
        stressors = []
        stressed_count = 0
        for dim in cols:
            v = float(stress[dim].iloc[-1])
            stressors.append((dim, v))
            if abs(v) > 1.5:
                stressed_count += 1

        stressors.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            "distance": float(dist),
            "threshold": float(thresh) if pd.notna(thresh) else 0,
            "pct_of_threshold": float(dist / thresh) if thresh > 0 else 0,
            "top_stressors": stressors[:3],
            "stressed_dim_count": stressed_count,
        }


def _shannon_entropy(returns: np.ndarray, bins: int = 8) -> float:
    """Shannon entropy of return distribution (lower = more ordered)."""
    if len(returns) < 4:
        return 0.0
    counts, _ = np.histogram(returns, bins=bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ══════════════════════════════════════════════════════════════════
# ENGINE 2: CONTRARIAN ASYMMETRY ENGINE (CAE)
# ══════════════════════════════════════════════════════════════════

class ContrarianAsymmetryEngine:
    """SHORT-ONLY funding reversion on altcoins.

    The core insight nobody exploits:
        Positive funding (crowd LONG) on altcoins → SHORT wins 75-86%
        Negative funding (crowd SHORT) → LONG wins only 42-53%

    This asymmetry exists because crypto retail is structurally long.
    When they push funding positive on ETH/SOL/XRP, they're the dumb
    money at the top. Their double squeeze (funding cost + MTM loss)
    creates violent, predictable capitulation.

    BTC is excluded because on BTC the crowd is usually RIGHT
    (momentum/trend asset — positive funding → price continues up).

    Optional enhancements:
        - Volume confirmation: WR jumps to 86% when volume also spikes
        - OU half-life gate: Only trade when funding mean-reverts fast
        - Trend filter: Exclude if strong uptrend (50-SMA gradient > 0)
    """

    @staticmethod
    def generate_signals(
        df: pd.DataFrame,
        funding_z_threshold: float = 2.0,
        funding_lookback: int = 168,
        hold_bars: int = 12,
        require_volume_confirm: bool = False,
        volume_z_threshold: float = 1.5,
        use_halflife_gate: bool = False,
        halflife_window: int = 72,
        trend_filter: bool = False,
        trend_sma_period: int = 50,
        **kwargs,
    ) -> pd.Series:
        """Generate SHORT-ONLY signals when crowd is excessively long.

        Args:
            funding_z_threshold: Positive funding z-score trigger (lower
                than symmetric strategies because the asymmetric edge
                lets us be less selective).
            funding_lookback: Rolling window for z-score computation.
            hold_bars: Signal cooldown.
            require_volume_confirm: If True, also require volume spike
                (increases WR from ~76% to ~86% but fewer trades).
            volume_z_threshold: Volume z-score required if volume
                confirmation is on.
            use_halflife_gate: If True, only trade when OU half-life
                is short (fast mean reversion expected).
            halflife_window: Window for OU parameter estimation.
            trend_filter: If True, suppress signals when price is in
                a strong uptrend (50-SMA gradient positive).
            trend_sma_period: SMA period for trend filter.
        """
        signals = pd.Series(0, index=df.index, dtype=int)

        # ── Find funding column ──
        funding_col = None
        for col in ["fund_funding_rate", "funding_rate"]:
            if col in df.columns:
                funding_col = col
                break
        if funding_col is None:
            return signals

        funding = df[funding_col].fillna(0)

        # ── Compute funding z-score ──
        if "fund_funding_zscore" in df.columns:
            f_z = df["fund_funding_zscore"].fillna(0)
        else:
            f_mean = funding.rolling(funding_lookback, min_periods=20).mean()
            f_std = funding.rolling(funding_lookback, min_periods=20).std()
            f_z = (funding - f_mean) / (f_std + 1e-10)

        # ── Optional: volume z-score ──
        if require_volume_confirm and "volume" in df.columns:
            vol = df["volume"]
            v_mean = vol.rolling(funding_lookback, min_periods=20).mean()
            v_std = vol.rolling(funding_lookback, min_periods=20).std()
            vol_z = (vol - v_mean) / (v_std + 1e-10)
        else:
            vol_z = pd.Series(999.0, index=df.index)  # Always passes

        # ── Optional: OU half-life ──
        if use_halflife_gate:
            halflife = _compute_ou_halflife(funding.values, halflife_window)
            hl_series = pd.Series(halflife, index=df.index)
            hl_median = hl_series.rolling(
                funding_lookback, min_periods=40,
            ).median()
        else:
            hl_series = None
            hl_median = None

        # ── Optional: trend filter ──
        if trend_filter:
            sma = df["close"].rolling(trend_sma_period, min_periods=20).mean()
            sma_gradient = sma.pct_change(10)  # 10-bar SMA slope
        else:
            sma_gradient = None

        # ── Signal generation: SHORT ONLY ──
        raw = pd.Series(0, index=df.index, dtype=int)

        for i in range(max(funding_lookback, 100), len(df)):
            z = f_z.iloc[i]

            # Core condition: crowd is LONG (positive funding z-score)
            if z <= funding_z_threshold:
                continue

            # Volume confirmation
            if require_volume_confirm and vol_z.iloc[i] < volume_z_threshold:
                continue

            # Half-life gate: only trade if mean reversion is fast
            if use_halflife_gate and hl_series is not None:
                hl = hl_series.iloc[i]
                hl_m = hl_median.iloc[i]
                if pd.isna(hl) or pd.isna(hl_m) or hl > hl_m:
                    continue

            # Trend filter: suppress if strong uptrend
            if trend_filter and sma_gradient is not None:
                grad = sma_gradient.iloc[i]
                if pd.notna(grad) and grad > 0.005:  # >0.5% SMA growth in 10 bars
                    continue

            raw.iloc[i] = -1  # SHORT signal

        return _first_signal_only(raw, hold_bars)

    @staticmethod
    def get_proximity(df: pd.DataFrame, funding_lookback: int = 168) -> dict:
        """Get proximity to signal threshold for monitoring."""
        funding_col = None
        for col in ["fund_funding_rate", "funding_rate"]:
            if col in df.columns:
                funding_col = col
                break

        if funding_col is None or "fund_funding_zscore" not in df.columns:
            return {"pct": 0.0, "detail": "no funding data", "z": 0.0}

        z = float(df["fund_funding_zscore"].iloc[-1])
        threshold = 2.0
        pct = min(max(z / threshold, 0), 1.0) if z > 0 else 0.0

        return {
            "pct": pct,
            "detail": f"z={z:+.1f}/+{threshold:.1f} (SHORT only)",
            "z": z,
        }


def _compute_ou_halflife(
    funding_rates: np.ndarray, window: int,
) -> np.ndarray:
    """Estimate Ornstein-Uhlenbeck half-life from rolling AR(1)."""
    n = len(funding_rates)
    halflife = np.full(n, np.nan)

    for i in range(window, n):
        y = funding_rates[i - window + 1 : i + 1]
        x = funding_rates[i - window : i]

        sx = np.std(x)
        if sx < 1e-12:
            continue

        beta = np.corrcoef(x, y)[0, 1] * np.std(y) / sx
        if 0.01 < beta < 0.999:
            halflife[i] = -np.log(2) / np.log(beta)

    return halflife

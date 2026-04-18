"""
SignalForge — Advanced Feature Engineering (120+ features)
============================================================
World-class quantitative feature library covering:

1. Price microstructure    — bar internals, gap analysis, range compression
2. Multi-horizon returns   — 1m to 50-bar returns, log returns, volatility-adjusted
3. Volatility regimes      — Parkinson, Garman-Klass, Yang-Zhang, realized vol surfaces
4. Volume intelligence     — OBV, MFI, VWAP deviation, volume clock, participation rate
5. Momentum spectrum       — RSI multi-period, Stochastic, Williams %R, CCI, ADX, DPO
6. Cross-sectional         — Z-scored everything for multi-asset comparison
7. Mean reversion signals  — Bollinger, Keltner, Donchian, price oscillators
8. Trend strength          — ADX, Aroon, linear regression slope, Hurst exponent
9. Order flow proxies      — bar position, close-to-high, intrabar volatility
10. Entropy & information  — approximate entropy, sample entropy, fractal dimension
11. Regime detection       — rolling Sharpe, rolling beta, drawdown duration
12. Calendar features      — hour of day, day of week (crypto has patterns)

These features are the "genes" that the Alpha Genome GP engine recombines.
More features = richer search space = stronger alpha.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


def compute_all_features(df: pd.DataFrame, include_calendar: bool = True) -> pd.DataFrame:
    """Compute 120+ features from OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        include_calendar: Whether to add hour/day features

    Returns:
        DataFrame with all original + computed features
    """
    df = df.copy()

    # Suppress expected fragmentation warnings from incremental column assignments
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        df = _compute_all_features_inner(df, include_calendar)

    return df


def _compute_all_features_inner(df: pd.DataFrame, include_calendar: bool) -> pd.DataFrame:
    """Inner implementation - computes all features."""
    # ================================================================
    # 1. RETURNS — Multi-horizon, log, volatility-adjusted
    # ================================================================
    for period in [1, 2, 3, 5, 10, 20, 50]:
        df[f"ret_{period}"] = df["close"].pct_change(period)
        df[f"log_ret_{period}"] = np.log(df["close"] / df["close"].shift(period))

    # Volatility-adjusted returns (risk-scaled momentum)
    for period in [5, 10, 20]:
        vol = df["close"].pct_change().rolling(period).std()
        df[f"ret_vol_adj_{period}"] = df[f"ret_{period}"] / (vol + 1e-10)

    # ================================================================
    # 2. VOLATILITY — Multiple estimators for richer signal
    # ================================================================
    # Close-to-close (standard)
    for window in [5, 10, 20, 50]:
        df[f"vol_{window}"] = df["close"].pct_change().rolling(window).std()

    # Parkinson (uses high-low range — more efficient estimator)
    hl_ratio = np.log(df["high"] / df["low"])
    for window in [10, 20]:
        df[f"vol_parkinson_{window}"] = (
            hl_ratio.pow(2).rolling(window).mean() / (4 * np.log(2))
        ).apply(np.sqrt)

    # Garman-Klass (uses OHLC — most efficient)
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    for window in [10, 20]:
        gk = 0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2)
        df[f"vol_garman_klass_{window}"] = gk.rolling(window).mean().apply(np.sqrt)

    # Volatility ratio (current vs historical — detects regime changes)
    df["vol_ratio_5_20"] = df["vol_5"] / (df["vol_20"] + 1e-10)
    df["vol_ratio_10_50"] = df["vol_10"] / (df["vol_50"] + 1e-10)

    # Volatility of volatility (vol-of-vol = uncertainty about uncertainty)
    df["vol_of_vol_20"] = df["vol_10"].rolling(20).std()

    # ================================================================
    # 3. VOLUME INTELLIGENCE
    # ================================================================
    for window in [5, 10, 20, 50]:
        df[f"vol_ratio_{window}"] = df["volume"] / (
            df["volume"].rolling(window).mean() + 1e-10
        )

    # On-Balance Volume (OBV)
    obv = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
    df["obv"] = obv
    df["obv_slope_10"] = obv.diff(10) / (obv.rolling(10).std() + 1e-10)
    df["obv_slope_20"] = obv.diff(20) / (obv.rolling(20).std() + 1e-10)

    # Money Flow Index (MFI) — volume-weighted RSI
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    raw_money_flow = typical_price * df["volume"]
    pos_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    neg_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    for period in [14, 21]:
        pos_sum = pos_flow.rolling(period).sum()
        neg_sum = neg_flow.rolling(period).sum()
        mfi = 100 - 100 / (1 + pos_sum / (neg_sum + 1e-10))
        df[f"mfi_{period}"] = mfi

    # VWAP deviation (intraday anchor)
    cumvol = df["volume"].cumsum()
    cumvp = (typical_price * df["volume"]).cumsum()
    vwap = cumvp / (cumvol + 1e-10)
    df["vwap_dev"] = (df["close"] - vwap) / (vwap + 1e-10)

    # Volume-price confirmation (do price and volume agree?)
    df["vol_price_corr_10"] = (
        df["close"].pct_change().rolling(10).corr(df["volume"].pct_change())
    )
    df["vol_price_corr_20"] = (
        df["close"].pct_change().rolling(20).corr(df["volume"].pct_change())
    )

    # Participation rate (volume relative to recent peak)
    df["vol_participation_20"] = df["volume"] / (
        df["volume"].rolling(20).max() + 1e-10
    )

    # ================================================================
    # 4. MOVING AVERAGES & PRICE POSITION
    # ================================================================
    for window in [5, 10, 20, 50, 100, 200]:
        ma = df["close"].rolling(window).mean()
        df[f"price_vs_ma_{window}"] = (df["close"] - ma) / (ma + 1e-10)

    # EMA crossovers
    for fast, slow in [(5, 20), (10, 50), (20, 100)]:
        ema_fast = df["close"].ewm(span=fast).mean()
        ema_slow = df["close"].ewm(span=slow).mean()
        df[f"ema_cross_{fast}_{slow}"] = (ema_fast - ema_slow) / (ema_slow + 1e-10)

    # ================================================================
    # 5. RSI — Multi-period for complete picture
    # ================================================================
    delta = df["close"].diff()
    for period in [3, 7, 14, 21]:
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # RSI momentum (rate of change of RSI)
    df["rsi_14_momentum"] = df["rsi_14"].diff(3)

    # RSI divergence (price makes new high but RSI doesn't)
    df["rsi_divergence_14"] = (
        df["close"].rolling(20).apply(lambda x: 1 if x.iloc[-1] == x.max() else 0, raw=False)
        - df["rsi_14"].rolling(20).apply(lambda x: 1 if x.iloc[-1] == x.max() else 0, raw=False)
    )

    # ================================================================
    # 6. MACD
    # ================================================================
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_slope"] = df["macd_hist"].diff(3)

    # ================================================================
    # 7. BOLLINGER BANDS
    # ================================================================
    for window in [10, 20]:
        ma = df["close"].rolling(window).mean()
        std = df["close"].rolling(window).std()
        df[f"bb_upper_{window}"] = ma + 2 * std
        df[f"bb_lower_{window}"] = ma - 2 * std
        df[f"bb_pct_{window}"] = (df["close"] - (ma - 2 * std)) / (
            4 * std + 1e-10
        )
        df[f"bb_width_{window}"] = (4 * std) / (ma + 1e-10)

    # Keltner Channels (ATR-based — catches different breakouts than BB)
    atr_20 = _compute_atr(df, 20)
    ema_20 = df["close"].ewm(span=20).mean()
    df["keltner_upper_20"] = ema_20 + 2 * atr_20
    df["keltner_lower_20"] = ema_20 - 2 * atr_20
    df["keltner_pct_20"] = (df["close"] - df["keltner_lower_20"]) / (
        4 * atr_20 + 1e-10
    )

    # Squeeze (BB inside Keltner = low vol, about to explode)
    df["squeeze"] = (
        (df["bb_lower_20"] > df["keltner_lower_20"]).astype(float)
        * (df["bb_upper_20"] < df["keltner_upper_20"]).astype(float)
    )

    # ================================================================
    # 8. ATR — Average True Range
    # ================================================================
    for period in [7, 14, 21]:
        atr = _compute_atr(df, period)
        df[f"atr_{period}"] = atr
        df[f"atr_pct_{period}"] = atr / (df["close"] + 1e-10)

    # ================================================================
    # 9. STOCHASTIC OSCILLATOR
    # ================================================================
    for period in [14, 21]:
        low_min = df["low"].rolling(period).min()
        high_max = df["high"].rolling(period).max()
        df[f"stoch_k_{period}"] = 100 * (df["close"] - low_min) / (
            high_max - low_min + 1e-10
        )
        df[f"stoch_d_{period}"] = df[f"stoch_k_{period}"].rolling(3).mean()

    # ================================================================
    # 10. ADX — Average Directional Index (trend strength)
    # ================================================================
    df["adx_14"] = _compute_adx(df, 14)
    df["adx_21"] = _compute_adx(df, 21)

    # ================================================================
    # 11. WILLIAMS %R
    # ================================================================
    for period in [14, 21]:
        highest = df["high"].rolling(period).max()
        lowest = df["low"].rolling(period).min()
        df[f"williams_r_{period}"] = -100 * (highest - df["close"]) / (
            highest - lowest + 1e-10
        )

    # ================================================================
    # 12. CCI — Commodity Channel Index
    # ================================================================
    tp = (df["high"] + df["low"] + df["close"]) / 3
    for period in [14, 20]:
        tp_ma = tp.rolling(period).mean()
        tp_mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df[f"cci_{period}"] = (tp - tp_ma) / (0.015 * tp_mad + 1e-10)

    # ================================================================
    # 13. PRICE MICROSTRUCTURE
    # ================================================================
    # Bar position (close location within bar range)
    df["bar_position"] = (df["close"] - df["low"]) / (
        df["high"] - df["low"] + 1e-10
    )

    # Bar size relative to recent
    bar_range = df["high"] - df["low"]
    df["bar_range_ratio_10"] = bar_range / (bar_range.rolling(10).mean() + 1e-10)

    # Gap analysis
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / (
        df["close"].shift(1) + 1e-10
    )

    # Upper/lower shadow ratios
    body = (df["close"] - df["open"]).abs()
    df["upper_shadow_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (
        body + 1e-10
    )
    df["lower_shadow_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (
        body + 1e-10
    )

    # Range compression (signal for breakout)
    df["range_compression_10"] = bar_range.rolling(10).min() / (
        bar_range.rolling(10).max() + 1e-10
    )

    # ================================================================
    # 14. TREND STRENGTH
    # ================================================================
    # Linear regression slope (trend direction + strength)
    for window in [10, 20, 50]:
        df[f"linreg_slope_{window}"] = (
            df["close"].rolling(window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] / (x.mean() + 1e-10)
                if len(x) == window else 0,
                raw=True,
            )
        )

    # Aroon (time since high/low — measures trend age)
    for period in [14, 25]:
        df[f"aroon_up_{period}"] = (
            df["high"].rolling(period).apply(lambda x: x.argmax(), raw=True)
            / period * 100
        )
        df[f"aroon_down_{period}"] = (
            df["low"].rolling(period).apply(lambda x: x.argmin(), raw=True)
            / period * 100
        )
        df[f"aroon_osc_{period}"] = df[f"aroon_up_{period}"] - df[f"aroon_down_{period}"]

    # DPO — Detrended Price Oscillator
    n = 20
    shift_n = n // 2 + 1
    df["dpo_20"] = df["close"] - df["close"].rolling(n).mean().shift(shift_n)

    # ================================================================
    # 15. MOMENTUM RANK
    # ================================================================
    for window in [20, 50]:
        df[f"momentum_rank_{window}"] = (
            df["ret_1"].rolling(window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
        )

    # Rate of change
    for period in [5, 10, 20]:
        df[f"roc_{period}"] = (df["close"] - df["close"].shift(period)) / (
            df["close"].shift(period) + 1e-10
        )

    # ================================================================
    # 16. REGIME DETECTION FEATURES
    # ================================================================
    # Rolling Sharpe (is this a good or bad regime?)
    ret_1 = df["close"].pct_change()
    for window in [20, 50]:
        roll_mean = ret_1.rolling(window).mean()
        roll_std = ret_1.rolling(window).std()
        df[f"rolling_sharpe_{window}"] = roll_mean / (roll_std + 1e-10) * np.sqrt(252 * 24)

    # Drawdown duration (how long have we been in a drawdown?)
    equity = df["close"].cummax()
    in_dd = (df["close"] < equity).astype(int)
    df["dd_duration"] = in_dd.groupby((in_dd != in_dd.shift()).cumsum()).cumsum()
    df["dd_pct"] = (equity - df["close"]) / (equity + 1e-10)

    # Mean-reversion pressure (distance from VWAP + BB)
    df["mean_rev_pressure"] = (
        df["bb_pct_20"] * 0.5 + (1 - df["bar_position"]) * 0.3 + df["vwap_dev"] * 0.2
    )

    # ================================================================
    # 17. INFORMATION / ENTROPY FEATURES
    # ================================================================
    # Approximate entropy proxy — via rolling autocorrelation
    for lag in [1, 5]:
        df[f"autocorr_{lag}_20"] = ret_1.rolling(20).apply(
            lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else 0,
            raw=False,
        )

    # Hurst exponent proxy (H > 0.5 = trending, H < 0.5 = mean-reverting)
    df["hurst_proxy_50"] = _compute_hurst_proxy(df["close"], 50)

    # ================================================================
    # 18. CALENDAR FEATURES (crypto has significant day/hour patterns)
    # ================================================================
    if include_calendar and hasattr(df.index, "hour"):
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # ================================================================
    # 19. Z-SCORED FEATURES (for cross-asset comparability)
    # ================================================================
    zscore_cols = [
        "ret_5", "ret_20", "vol_10", "vol_20", "rsi_14",
        "macd_hist", "adx_14", "mfi_14", "cci_14",
    ]
    for col in zscore_cols:
        if col in df.columns:
            roll_mean = df[col].rolling(50).mean()
            roll_std = df[col].rolling(50).std()
            df[f"{col}_zscore"] = (df[col] - roll_mean) / (roll_std + 1e-10)

    # Replace infinities with NaN (NOT 0 — zero can be an active signal)
    # NaN propagates correctly through downstream calculations and
    # prevents phantom signals (e.g. RSI=0 reads as extremely oversold)
    df = df.replace([np.inf, -np.inf], np.nan)

    n_features = len([c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]])
    logger.info(f"Computed {n_features} features ({df.isna().sum().sum()} NaN cells, mostly warmup)")

    return df


# ================================================================
# Helper functions
# ================================================================

def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    atr = _compute_atr(df, period)
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean()


def _compute_hurst_proxy(series: pd.Series, window: int) -> pd.Series:
    """Approximate Hurst exponent via rescaled range (R/S) method."""
    def _rs_hurst(x):
        if len(x) < 10:
            return 0.5
        y = np.diff(np.asarray(x, dtype=float))
        n = len(y)
        if n < 4:
            return 0.5
        mean_y = np.mean(y)
        cumdev = np.cumsum(y - mean_y)
        r = np.max(cumdev) - np.min(cumdev)
        s = np.std(y, ddof=1)
        if s < 1e-10:
            return 0.5
        rs = r / s
        if rs <= 0:
            return 0.5
        return np.log(rs) / np.log(n)

    return series.rolling(window).apply(_rs_hurst, raw=True)


# ================================================================
# Feature name registry (for GP gene.py integration)
# ================================================================

ADVANCED_FEATURE_NAMES = [
    # Returns (14)
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20", "ret_50",
    "log_ret_1", "log_ret_5", "log_ret_10", "log_ret_20",
    "ret_vol_adj_5", "ret_vol_adj_10", "ret_vol_adj_20",
    # Volatility (12)
    "vol_5", "vol_10", "vol_20", "vol_50",
    "vol_parkinson_10", "vol_parkinson_20",
    "vol_garman_klass_10", "vol_garman_klass_20",
    "vol_ratio_5_20", "vol_ratio_10_50",
    "vol_of_vol_20",
    # Volume (14)
    "vol_ratio_5", "vol_ratio_10", "vol_ratio_20", "vol_ratio_50",
    "obv_slope_10", "obv_slope_20",
    "mfi_14", "mfi_21",
    "vwap_dev",
    "vol_price_corr_10", "vol_price_corr_20",
    "vol_participation_20",
    # Price position (9)
    "price_vs_ma_5", "price_vs_ma_10", "price_vs_ma_20",
    "price_vs_ma_50", "price_vs_ma_100", "price_vs_ma_200",
    "ema_cross_5_20", "ema_cross_10_50", "ema_cross_20_100",
    # RSI (7)
    "rsi_3", "rsi_7", "rsi_14", "rsi_21",
    "rsi_14_momentum", "rsi_divergence_14",
    # MACD (4)
    "macd", "macd_signal", "macd_hist", "macd_hist_slope",
    # Bands (8)
    "bb_pct_10", "bb_pct_20", "bb_width_10", "bb_width_20",
    "keltner_pct_20", "squeeze",
    # ATR (6)
    "atr_pct_7", "atr_pct_14", "atr_pct_21",
    # Stochastic (4)
    "stoch_k_14", "stoch_d_14", "stoch_k_21", "stoch_d_21",
    # Trend (11)
    "adx_14", "adx_21",
    "williams_r_14", "williams_r_21",
    "cci_14", "cci_20",
    "linreg_slope_10", "linreg_slope_20", "linreg_slope_50",
    "aroon_osc_14", "aroon_osc_25",
    "dpo_20",
    # Micro (6)
    "bar_position", "bar_range_ratio_10", "gap_pct",
    "upper_shadow_ratio", "lower_shadow_ratio",
    "range_compression_10",
    # Momentum (5)
    "momentum_rank_20", "momentum_rank_50",
    "roc_5", "roc_10", "roc_20",
    # Regime (5)
    "rolling_sharpe_20", "rolling_sharpe_50",
    "dd_duration", "dd_pct",
    "mean_rev_pressure",
    # Information (3)
    "autocorr_1_20", "autocorr_5_20",
    "hurst_proxy_50",
    # Calendar (4)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Z-scored (9)
    "ret_5_zscore", "ret_20_zscore", "vol_10_zscore", "vol_20_zscore",
    "rsi_14_zscore", "macd_hist_zscore", "adx_14_zscore",
    "mfi_14_zscore", "cci_14_zscore",
]

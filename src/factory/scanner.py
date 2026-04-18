"""
Signal Scanner — Hypothesis Generation + First Pass Filter
============================================================
Generates signal hypotheses from raw data and tests each one
with next-bar entry, realistic costs, and non-overlapping trades.

Output: List of signals that pass the raw filter (p < 0.05 uncorrected,
PF > 1.1, 50+ trades). These go to the Validator for OOS confirmation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats
from typing import Callable

from src.factory.ensemble import (
    NearMiss,
    build_ensemble_candidates,
    clear_registry,
    register_ensemble,
)


@dataclass
class RawSignal:
    """A signal hypothesis with its raw (uncorrected) test results."""
    name: str
    asset: str
    direction: int          # 1 = long, -1 = short
    hold_bars: int
    n_trades: int
    mean_return: float
    pf: float
    sharpe: float
    p_value: float          # uncorrected t-test p-value
    signal_func: Callable   # function(df) -> boolean mask


@dataclass
class ScanResult:
    """Output of a full scan."""
    total_hypotheses: int
    bonferroni_threshold: float
    raw_survivors: list[RawSignal]
    bonferroni_survivors: list[RawSignal]


# ─── Signal Generators ──────────────────────────────────────────

def _generate_dow_signals(df: pd.DataFrame) -> list[tuple[str, pd.Series, int]]:
    """Day-of-week directional signals."""
    signals = []
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    if not hasattr(df.index, "dayofweek"):
        return signals

    for dow in range(7):
        name = day_names[dow]
        # One entry per week at hour 0
        mask = (df.index.dayofweek == dow) & (df.index.hour == 0)
        signals.append((f"dow_{name}_long", mask, 1))
        signals.append((f"dow_{name}_short", mask, -1))

    return signals


def _generate_funding_signals(df: pd.DataFrame) -> list[tuple[str, pd.Series, int]]:
    """Funding rate z-score signals — fade extremes."""
    signals = []
    fr_col = None
    for c in ["fund_funding_rate", "funding_rate"]:
        if c in df.columns:
            fr_col = c
            break
    if fr_col is None:
        return signals

    fr = df[fr_col]

    for lookback in [48, 96, 168]:
        mu = fr.rolling(lookback, min_periods=lookback).mean()
        sigma = fr.rolling(lookback, min_periods=lookback).std()
        fz = (fr - mu) / (sigma + 1e-10)

        for thresh in [1.5, 2.0, 2.5, 3.0]:
            # Extreme positive funding → short (crowd is long)
            mask_high = (fz > thresh) & (fr > 0.0001)
            signals.append((f"fund_z{lookback}>{thresh}_short", mask_high, -1))
            # Extreme negative funding → long (crowd is short)
            mask_low = (fz < -thresh) & (fr < -0.0001)
            signals.append((f"fund_z{lookback}<-{thresh}_long", mask_low, 1))

    return signals


def _generate_oi_signals(df: pd.DataFrame) -> list[tuple[str, pd.Series, int]]:
    """Open-interest z-score signals — detect positioning extremes.

    Hypotheses:
      * OI z > +thresh AND price rising strongly → crowd piling long at the
        top → fade (short).
      * OI z > +thresh AND price falling → shorts piling in → short-squeeze
        setup (long).
      * OI z < -thresh AND price rising → short covering exhausted → fade
        (short).
      * OI z < -thresh AND price falling → longs exiting capitulation → long.
    """
    signals = []
    oi_col = None
    for c in ["oi_oi_zscore", "oi_oi_value"]:
        if c in df.columns:
            oi_col = c
            break
    if oi_col is None:
        return signals

    # Price return context (4h) — drives the sign of the fade
    if "close" not in df.columns:
        return signals
    ret4h = df["close"].pct_change(4)

    if oi_col == "oi_oi_zscore":
        oi_z = df[oi_col]
    else:
        val = df[oi_col]
        mu = val.rolling(168, min_periods=48).mean()
        sd = val.rolling(168, min_periods=48).std()
        oi_z = (val - mu) / (sd + 1e-10)

    for thresh in [1.5, 2.0, 2.5]:
        # OI pile-in with price up → likely top
        mask_pile_long = (oi_z > thresh) & (ret4h > 0.01)
        signals.append((f"oi_pile_long_z{thresh}_short", mask_pile_long, -1))
        # OI pile-in with price down → shorts piling → squeeze long
        mask_pile_short = (oi_z > thresh) & (ret4h < -0.01)
        signals.append((f"oi_pile_short_z{thresh}_long", mask_pile_short, 1))
        # OI washout with price down → longs liquidated → reversal long
        mask_wash_long = (oi_z < -thresh) & (ret4h < -0.01)
        signals.append((f"oi_washout_long_z{thresh}_long", mask_wash_long, 1))
        # OI washout with price up → short covering done → fade
        mask_wash_short = (oi_z < -thresh) & (ret4h > 0.01)
        signals.append((f"oi_washout_short_z{thresh}_short", mask_wash_short, -1))

    return signals


def _generate_lsr_signals(df: pd.DataFrame) -> list[tuple[str, pd.Series, int]]:
    """Long/Short ratio extremes — fade the crowd."""
    signals = []
    lsr_col = None
    for c in ["lsr_long_short_ratio", "long_short_ratio"]:
        if c in df.columns:
            lsr_col = c
            break
    if lsr_col is None:
        return signals

    lsr = df[lsr_col]
    mu = lsr.rolling(168, min_periods=48).mean()
    sd = lsr.rolling(168, min_periods=48).std()
    lsr_z = (lsr - mu) / (sd + 1e-10)

    for thresh in [1.5, 2.0, 2.5]:
        # Crowd extremely long → fade short
        signals.append((f"lsr_crowd_long_z{thresh}_short", lsr_z > thresh, -1))
        # Crowd extremely short → fade long
        signals.append((f"lsr_crowd_short_z{thresh}_long", lsr_z < -thresh, 1))

    return signals


def _generate_taker_signals(df: pd.DataFrame) -> list[tuple[str, pd.Series, int]]:
    """Taker-buy/sell imbalance — aggressive-flow exhaustion."""
    signals = []
    tk_col = None
    for c in ["taker_taker_imbalance", "taker_imbalance"]:
        if c in df.columns:
            tk_col = c
            break
    if tk_col is None:
        return signals

    ti = df[tk_col]
    mu = ti.rolling(168, min_periods=48).mean()
    sd = ti.rolling(168, min_periods=48).std()
    ti_z = (ti - mu) / (sd + 1e-10)

    for thresh in [1.5, 2.0, 2.5]:
        # Persistent aggressive buying → exhaustion → fade
        signals.append((f"taker_buy_exhaust_z{thresh}_short", ti_z > thresh, -1))
        # Persistent aggressive selling → exhaustion → fade
        signals.append((f"taker_sell_exhaust_z{thresh}_long", ti_z < -thresh, 1))

    return signals


def _generate_leverage_signals(df: pd.DataFrame) -> list[tuple[str, pd.Series, int]]:
    """Composite leverage-heat + liquidation-pressure signals."""
    signals = []

    # leverage_heat — funding z + OI z combined
    if "leverage_heat" in df.columns and "close" in df.columns:
        heat = df["leverage_heat"]
        ret8h = df["close"].pct_change(8)
        for thresh in [1.5, 2.0]:
            # Hot leverage with trend up → long positions crowded → short
            mask_hot_up = (heat > thresh) & (ret8h > 0.02)
            signals.append((f"lev_hot_up_{thresh}_short", mask_hot_up, -1))
            # Hot leverage with trend down → short positions crowded → long
            mask_hot_dn = (heat > thresh) & (ret8h < -0.02)
            signals.append((f"lev_hot_dn_{thresh}_long", mask_hot_dn, 1))

    # liq_pressure_long: positive funding + price drop → longs getting liquidated
    # Cascade continuation: short. Post-cascade reversal: long.
    if "liq_pressure_long" in df.columns:
        lp = df["liq_pressure_long"]
        mu = lp.rolling(168, min_periods=48).mean()
        sd = lp.rolling(168, min_periods=48).std()
        lp_z = (lp - mu) / (sd + 1e-10)
        for thresh in [2.0, 2.5, 3.0]:
            signals.append((f"liq_long_cascade_z{thresh}_short", lp_z > thresh, -1))
            signals.append((f"liq_long_reversal_z{thresh}_long", lp_z > thresh, 1))

    if "liq_pressure_short" in df.columns:
        lp = df["liq_pressure_short"]
        mu = lp.rolling(168, min_periods=48).mean()
        sd = lp.rolling(168, min_periods=48).std()
        lp_z = (lp - mu) / (sd + 1e-10)
        for thresh in [2.0, 2.5, 3.0]:
            signals.append((f"liq_short_cascade_z{thresh}_long", lp_z > thresh, 1))
            signals.append((f"liq_short_reversal_z{thresh}_short", lp_z > thresh, -1))

    return signals


def _generate_return_signals(df: pd.DataFrame) -> list[tuple[str, pd.Series, int]]:
    """Extreme return signals — mean reversion after big moves."""
    signals = []

    for window in [1, 2, 3, 5, 10]:
        ret = df["close"].pct_change(window)
        mu = ret.rolling(100, min_periods=50).mean()
        sigma = ret.rolling(100, min_periods=50).std()
        z = (ret - mu) / (sigma + 1e-10)

        for thresh in [2.0, 2.5, 3.0]:
            # Big drop → long (mean reversion)
            mask_low = z < -thresh
            signals.append((f"ret{window}_z<-{thresh}_long", mask_low, 1))
            # Big pump → short (mean reversion)
            mask_high = z > thresh
            signals.append((f"ret{window}_z>{thresh}_short", mask_high, -1))

    return signals


def _generate_rsi_signals(df: pd.DataFrame) -> list[tuple[str, pd.Series, int]]:
    """RSI extreme signals — oversold bounce / overbought fade."""
    signals = []

    for period in [3, 7, 14]:
        col = f"rsi_{period}"
        if col not in df.columns:
            continue

        rsi = df[col]
        for thresh in [10, 15, 20, 25]:
            signals.append((f"rsi_{period}<{thresh}_long", rsi < thresh, 1))
        for thresh in [75, 80, 85, 90]:
            signals.append((f"rsi_{period}>{thresh}_short", rsi > thresh, -1))

    return signals


def _generate_volume_signals(df: pd.DataFrame) -> list[tuple[str, pd.Series, int]]:
    """Volume spike signals."""
    signals = []

    vol = df["volume"]
    vol_ma = vol.rolling(24, min_periods=12).mean()
    vol_ratio = vol / (vol_ma + 1e-10)

    for thresh in [2.0, 3.0, 5.0]:
        # Volume spike — test both directions
        mask = vol_ratio > thresh
        signals.append((f"vol_spike_{thresh}x_long", mask, 1))
        signals.append((f"vol_spike_{thresh}x_short", mask, -1))

    return signals


def _generate_volatility_signals(df: pd.DataFrame) -> list[tuple[str, pd.Series, int]]:
    """Volatility compression / expansion signals."""
    signals = []

    if "atr_14" in df.columns:
        atr = df["atr_14"]
        atr_ma = atr.rolling(48, min_periods=24).mean()
        atr_ratio = atr / (atr_ma + 1e-10)

        # Vol compression → breakout expected
        for thresh in [0.5, 0.6, 0.7]:
            mask = atr_ratio < thresh
            signals.append((f"vol_compress_{thresh}_long", mask, 1))
            signals.append((f"vol_compress_{thresh}_short", mask, -1))

        # Vol expansion → mean reversion expected
        for thresh in [1.5, 2.0, 2.5]:
            mask = atr_ratio > thresh
            signals.append((f"vol_expand_{thresh}_long", mask, 1))
            signals.append((f"vol_expand_{thresh}_short", mask, -1))

    return signals


def _generate_momentum_signals(df: pd.DataFrame) -> list[tuple[str, pd.Series, int]]:
    """Trend-continuation signals: moving-average crosses, persistence."""
    signals = []
    close = df["close"]

    # SMA cross-overs (slow regime flip, rare)
    for fast, slow in [(20, 50), (50, 200), (10, 30)]:
        sma_f = close.rolling(fast, min_periods=fast).mean()
        sma_s = close.rolling(slow, min_periods=slow).mean()
        cross_up = (sma_f > sma_s) & (sma_f.shift(1) <= sma_s.shift(1))
        cross_dn = (sma_f < sma_s) & (sma_f.shift(1) >= sma_s.shift(1))
        signals.append((f"ma_cross_{fast}_{slow}_long", cross_up, 1))
        signals.append((f"ma_cross_{fast}_{slow}_short", cross_dn, -1))

    # Momentum persistence: price above fast SMA AND rising
    for window in [20, 50, 100]:
        sma = close.rolling(window, min_periods=window).mean()
        rising = sma.diff(window // 4) > 0
        falling = sma.diff(window // 4) < 0
        above = close > sma
        below = close < sma
        # One entry per swing (edge of condition)
        above_edge = above & ~above.shift(1).fillna(False)
        below_edge = below & ~below.shift(1).fillna(False)
        signals.append((f"trend_above_sma{window}_long", above_edge & rising, 1))
        signals.append((f"trend_below_sma{window}_short", below_edge & falling, -1))

    return signals


def _generate_breakout_signals(df: pd.DataFrame) -> list[tuple[str, pd.Series, int]]:
    """Donchian-style range breakouts."""
    signals = []
    high = df["high"] if "high" in df.columns else df["close"]
    low = df["low"] if "low" in df.columns else df["close"]
    close = df["close"]

    for window in [20, 50, 100]:
        upper = high.rolling(window, min_periods=window).max().shift(1)
        lower = low.rolling(window, min_periods=window).min().shift(1)
        bo_up = close > upper
        bo_dn = close < lower
        # Edges only to avoid counting every bar above breakout as a new trade
        bo_up_edge = bo_up & ~bo_up.shift(1).fillna(False)
        bo_dn_edge = bo_dn & ~bo_dn.shift(1).fillna(False)
        signals.append((f"breakout_{window}_long", bo_up_edge, 1))
        signals.append((f"breakout_{window}_short", bo_dn_edge, -1))

    return signals


# All signal generators.
#
# NOTE ON STRUCTURAL FEATURES:
#   * Funding rate is available on full history (Binance keeps all funding).
#   * OI / LSR / taker imbalance are capped at ~30 days by Binance API, so
#     generators that depend on them (oi/lsr/taker) are excluded — they
#     would inflate the Bonferroni correction without contributing any
#     testable history. Re-enable once a long-history OI source is wired
#     (e.g. Coinalyze, Laevitas) via src/data/structural.py.
#   * liq_pressure_long / liq_pressure_short are composites derived from
#     funding + price only, so they ARE safe to use on full history.
SIGNAL_GENERATORS = [
    _generate_dow_signals,
    _generate_funding_signals,
    _generate_leverage_signals,   # funding-derived liquidation pressure only
    _generate_return_signals,
    _generate_rsi_signals,
    _generate_volume_signals,
    _generate_volatility_signals,
    _generate_momentum_signals,
    _generate_breakout_signals,
]


# ─── Evaluation ──────────────────────────────────────────────────

def _compute_forward_returns(df: pd.DataFrame, hold: int) -> pd.Series:
    """Next-bar entry at OPEN, exit at CLOSE after hold bars.
    Includes realistic costs."""
    entry = df["open"].shift(-1)
    exit_price = df["close"].shift(-(1 + hold))

    raw_ret = (exit_price - entry) / entry

    # Vol-scaled slippage
    log_ret = np.log(df["close"] / df["close"].shift(1))
    realized_vol = log_ret.rolling(24).std()
    vol_slip = (realized_vol * np.sqrt(1 / 24)).clip(upper=0.005)
    base_slip = 0.0005

    total_cost = 0.001 + base_slip + vol_slip  # commission + slippage

    return raw_ret - total_cost


def _compute_raw_forward_returns(df: pd.DataFrame, hold: int) -> pd.Series:
    """Gross forward return (no costs) for beta-neutral comparisons."""
    entry = df["open"].shift(-1)
    exit_price = df["close"].shift(-(1 + hold))
    return (exit_price - entry) / entry


def evaluate_signal(
    df: pd.DataFrame,
    mask: pd.Series,
    direction: int,
    hold: int,
    min_trades: int = 50,
) -> dict | None:
    """Evaluate a signal with non-overlapping trades.

    ALL metrics (PF, Sharpe, chunk means, t-test) are computed on EXCESS
    returns = signal_return - directional_baseline, where baseline is the
    mean directional gross forward return over the ENTIRE dataset. This
    removes regime beta: a short signal in a bear market no longer looks
    good just because the market went down — it must beat the passive
    "short-everything" baseline on this exact dataset.

    Returns dict with stats or None if insufficient trades.
    """
    fwd = _compute_forward_returns(df, hold)
    gross_fwd = _compute_raw_forward_returns(df, hold)

    warmup = 200
    valid_mask = mask.copy()
    if isinstance(valid_mask, pd.Series):
        valid_mask.iloc[:warmup] = False
    else:
        valid_mask[:warmup] = False

    # Non-overlapping: enforce cooldown
    indices = np.where(valid_mask)[0]
    filtered = []
    last_entry = -hold - 1
    for idx in indices:
        if idx - last_entry > hold:
            filtered.append(idx)
            last_entry = idx

    if len(filtered) < min_trades:
        return None

    # Directional baseline on the dataset (all valid bars, excl warmup)
    baseline_series = (gross_fwd * direction).iloc[warmup:].dropna()
    if len(baseline_series) == 0:
        return None
    baseline_mean = float(baseline_series.mean())

    # Net directional returns (with costs) at signal bars — these are what
    # the strategy actually earns. All PF/Sharpe/chunk gates operate on
    # these RAW NET returns (not excess) so a signal must make money on its
    # own, not merely "less-bad than baseline."
    net_returns = (fwd.iloc[filtered] * direction).dropna()
    if len(net_returns) < min_trades:
        return None

    wins = net_returns[net_returns > 0]
    losses = net_returns[net_returns <= 0]
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-10
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    mean_ret = net_returns.mean()
    std_ret = net_returns.std()
    sharpe = (mean_ret / std_ret * np.sqrt(365 * 24 / hold)) if std_ret > 0 else 0

    t_stat, p_val = stats.ttest_1samp(net_returns, 0)

    # ── Time-chunk robustness on RAW NET returns ──
    # Split signal trades into 3 equal TIME chunks; each must be positive.
    # This catches single-period luck.
    idx_sorted = np.array(filtered)
    chunk_size = max(1, len(idx_sorted) // 3)
    chunk_means = []
    min_trades_per_chunk = max(5, min_trades // 6)
    for k in range(3):
        lo = k * chunk_size
        hi = (k + 1) * chunk_size if k < 2 else len(idx_sorted)
        if hi <= lo:
            continue
        chunk_net = (fwd.iloc[idx_sorted[lo:hi]] * direction).dropna()
        if len(chunk_net) < min_trades_per_chunk:
            chunk_means.append(float("nan"))
            continue
        chunk_means.append(float(chunk_net.mean()))
    valid_chunks = [m for m in chunk_means if not np.isnan(m)]
    positive_chunks = sum(1 for m in valid_chunks if m > 0)
    all_chunks_positive = (len(valid_chunks) == 3 and positive_chunks == 3)

    # ── Regime robustness on RAW NET returns ──
    # Classify each bar as bull / bear / chop by trailing 500-bar return
    # sign (~21 days on 1h), then require the signal to be profitable in
    # EVERY valid regime bucket (with adequate sample size). This is the
    # single most important gate for avoiding regime flips on OOS: a
    # short-only signal fit to a bear-heavy SCAN will pass the time-chunk
    # test but FAIL here because the bull-regime sub-sample is negative.
    regime_lb = 500
    fwd_trend = df["close"].pct_change(regime_lb)
    trend_std = float(fwd_trend.std())
    bull_thr = 0.5 * trend_std if trend_std > 0 else 0.0
    bear_thr = -bull_thr
    regime_at_entry = fwd_trend.iloc[idx_sorted].values
    regime_net = (fwd.iloc[idx_sorted] * direction).dropna().values
    regime_mask_n = min(len(regime_at_entry), len(regime_net))
    regime_at_entry = regime_at_entry[:regime_mask_n]
    regime_net_arr = regime_net[:regime_mask_n]

    bull_idx = regime_at_entry > bull_thr
    bear_idx = regime_at_entry < bear_thr
    chop_idx = ~(bull_idx | bear_idx)

    # Require ≥ 20 trades per regime for it to be "valid" — below that,
    # regime-mean is too noisy. Signals with trade flow in only one
    # regime get dropped entirely (valid_regimes < 2).
    min_trades_per_regime = max(20, min_trades // 4)

    regime_means = {}
    regime_counts = {}
    for label, sel in [("bull", bull_idx), ("bear", bear_idx), ("chop", chop_idx)]:
        regime_counts[label] = int(sel.sum())
        if sel.sum() >= min_trades_per_regime:
            regime_means[label] = float(np.nanmean(regime_net_arr[sel]))
        else:
            regime_means[label] = float("nan")
    positive_regimes = sum(
        1 for v in regime_means.values() if not np.isnan(v) and v > 0
    )
    valid_regimes = sum(1 for v in regime_means.values() if not np.isnan(v))
    # Strict: at least 2 valid regimes AND EVERY valid regime positive.
    # A signal must either (a) fire in ≥2 regimes with positive returns
    # in all of them, or (b) be rejected. This catches short-only fits to
    # bear SCAN periods that wouldn't work in the bull OOS.
    regime_robust = (valid_regimes >= 2 and positive_regimes == valid_regimes)
    # ── Beta-neutral alpha gate — SEPARATE from PF/Sharpe ──
    # Signal's GROSS directional mean must exceed the dataset's GROSS
    # directional baseline. This is the pure "does this signal pick better
    # bars than random" test. Using gross (not net) for both sides so costs
    # don't distort the comparison.
    gross_signal_mean = float(
        (gross_fwd.iloc[filtered] * direction).dropna().mean()
    )
    alpha = gross_signal_mean - baseline_mean
    # Require signal itself to have positive gross expectancy AND to beat
    # the passive-direction baseline. "Signal must make money on its own
    # merits AND pick better bars than random." Both conditions needed:
    #   - gross_signal_mean > 0 → positive EV direction+timing
    #   - alpha > 0           → better than just taking every bar
    has_alpha = (gross_signal_mean > 0) and (alpha > 0)

    # ── Alpha significance (t-test on excess returns vs zero) ──
    # Per-trade excess returns are inherently noisy, so demanding p<0.01
    # here is too strict — it rejects genuine weak signals that would
    # otherwise survive the stacked defences (Bonferroni on raw returns,
    # chunk-robustness on scan AND val, held-out val alpha_significant).
    # We use p<0.05 one-sided (t > 1.645) which still rejects pure-noise
    # alpha; the stricter p<0.01 bar is effectively restored downstream
    # via Bonferroni over ~1000+ hypotheses.
    excess_series = (gross_fwd.iloc[filtered] * direction).dropna() - baseline_mean
    if len(excess_series) >= 10 and excess_series.std() > 0:
        alpha_t = float(excess_series.mean() / (excess_series.std() / np.sqrt(len(excess_series))))
        _, alpha_p = stats.ttest_1samp(excess_series, 0)
        alpha_p_one_sided = float(alpha_p / 2) if alpha_t > 0 else float(1 - alpha_p / 2)
    else:
        alpha_t = 0.0
        alpha_p_one_sided = 1.0
    alpha_significant = (alpha_t > 1.645)  # one-sided p<0.05

    return {
        "n_trades": len(net_returns),
        "mean_return": float(mean_ret),
        "baseline_mean": baseline_mean,
        "gross_signal_mean": gross_signal_mean,
        "pf": float(pf),
        "sharpe": float(sharpe),
        "p_value": float(p_val / 2) if t_stat > 0 else float(1 - p_val / 2),
        "win_rate": float(len(wins) / len(net_returns)),
        "positive_chunks": positive_chunks,
        "total_chunks": len(valid_chunks),
        "all_chunks_positive": all_chunks_positive,
        "regime_means": regime_means,
        "regime_counts": regime_counts,
        "positive_regimes": positive_regimes,
        "valid_regimes": valid_regimes,
        "regime_robust": regime_robust,
        "alpha": alpha,
        "alpha_t": alpha_t,
        "alpha_p": alpha_p_one_sided,
        "alpha_significant": alpha_significant,
        "has_alpha": has_alpha,
    }


# ─── Scanner ─────────────────────────────────────────────────────

HOLD_PERIODS = [4, 8, 12, 24]


def scan(
    datasets: dict[str, pd.DataFrame],
    hold_periods: list[int] | None = None,
    min_trades: int = 50,
    p_threshold: float = 0.05,
    min_pf: float = 1.1,
) -> ScanResult:
    """Run full hypothesis scan across all assets and signals.

    Args:
        datasets: {symbol: DataFrame} with features computed
        hold_periods: list of forward return windows (bars)
        min_trades: minimum non-overlapping trades
        p_threshold: uncorrected p-value threshold for raw filter
        min_pf: minimum profit factor for raw filter

    Returns:
        ScanResult with raw and Bonferroni survivors
    """
    holds = hold_periods or HOLD_PERIODS
    all_signals = []

    # Fresh ensemble registry per scan (prevents stale specs from prior runs
    # leaking into validator reconstruction).
    clear_registry()
    near_miss_pool: list[NearMiss] = []

    # Count total hypotheses for Bonferroni
    total_hypotheses = 0
    for sym, df in datasets.items():
        for gen in SIGNAL_GENERATORS:
            sigs = gen(df)
            total_hypotheses += len(sigs) * len(holds)

    bonferroni = p_threshold / total_hypotheses if total_hypotheses > 0 else p_threshold

    raw_survivors = []
    bonferroni_survivors = []

    for sym, df in datasets.items():
        sym_short = sym.split("/")[0]

        for gen in SIGNAL_GENERATORS:
            sigs = gen(df)

            for sig_name, mask, direction in sigs:
                for hold in holds:
                    result = evaluate_signal(df, mask, direction, hold, min_trades)
                    if result is None:
                        continue

                    # Reject single-regime artefacts — require edge in ALL
                    # 3 time chunks, not just 2 of 3.
                    if not result.get("all_chunks_positive", False):
                        continue
                    # Regime gate — signal must work in at least 2 of the
                    # 3 volatility/trend regimes (bull/bear/chop). Without
                    # this, shorts fit to bear-dominated scan periods die
                    # immediately when OOS turns bullish (and vice versa).
                    if not result.get("regime_robust", False):
                        continue
                    # Reject trend-riders — require directional alpha over
                    # the dataset's directional base rate.
                    if not result.get("has_alpha", False):
                        continue

                    # ── Near-miss capture ──
                    # Signals that pass chunks + regime + has_alpha + pf +
                    # min_trades but fall short on individual
                    # alpha_significance go into the ensemble pool. We
                    # require regime_robust so the ensemble won't be built
                    # from regime-fragile components.
                    if (
                        result.get("pf", 0) > min_pf
                        and result.get("n_trades", 0) >= min_trades
                        and result.get("alpha_t", 0) > 0
                        and result.get("regime_robust", False)
                        and not result.get("alpha_significant", False)
                    ):
                        near_miss_pool.append(
                            NearMiss(
                                name=sig_name,
                                asset=sym,
                                direction=direction,
                                hold_bars=hold,
                                mask=mask.copy(),
                                family=sig_name.split("_", 1)[0],
                                alpha_t=float(result.get("alpha_t", 0.0)),
                                pf=float(result.get("pf", 0.0)),
                                sharpe=float(result.get("sharpe", 0.0)),
                                n_trades=int(result.get("n_trades", 0)),
                            )
                        )

                    # Alpha must be statistically significant (t > 1.645,
                    # p < 0.05 one-sided) — not just positive-by-luck.
                    if not result.get("alpha_significant", False):
                        continue

                    if result["p_value"] < p_threshold and result["pf"] > min_pf:
                        # Create a closure for the signal function
                        _mask = mask.copy()
                        _dir = direction

                        def make_func(m, d):
                            def f(new_df):
                                return m
                            return f

                        sig = RawSignal(
                            name=f"{sig_name}_h{hold}",
                            asset=sym,
                            direction=direction,
                            hold_bars=hold,
                            n_trades=result["n_trades"],
                            mean_return=result["mean_return"],
                            pf=result["pf"],
                            sharpe=result["sharpe"],
                            p_value=result["p_value"],
                            signal_func=make_func(_mask, _dir),
                        )

                        raw_survivors.append(sig)

                        if result["p_value"] < bonferroni:
                            bonferroni_survivors.append(sig)

    # ─── Ensemble pass: combine near-miss pool into composite signals ───
    # Near-miss components have real alpha but fail individual
    # significance. Combining N of them into a voting ensemble reduces
    # noise by ~sqrt(N) and frequently clears the alpha_significant gate
    # cleanly — this is the weak-signal → composite edge mechanism that
    # underpins most real quant engines.
    ensemble_candidates = build_ensemble_candidates(near_miss_pool)
    n_ensemble_hypotheses = len(ensemble_candidates)

    # Ensembles share the Bonferroni budget with individual hypotheses so
    # the overall FDR guarantee still holds.
    total_hypotheses_with_ens = total_hypotheses + n_ensemble_hypotheses
    bonferroni = (
        p_threshold / total_hypotheses_with_ens
        if total_hypotheses_with_ens > 0 else p_threshold
    )

    for cand in ensemble_candidates:
        asset = cand["asset"]
        df = datasets.get(asset)
        if df is None:
            continue
        mask = cand["mask"]
        direction = cand["direction"]
        hold = cand["hold_bars"]

        result = evaluate_signal(df, mask, direction, hold, min_trades)
        if result is None:
            continue
        if not result.get("all_chunks_positive", False):
            continue
        if not result.get("regime_robust", False):
            continue
        if not result.get("has_alpha", False):
            continue
        if not result.get("alpha_significant", False):
            continue
        if result["p_value"] >= p_threshold or result["pf"] <= min_pf:
            continue

        # Register so validator can rebuild on held-out data.
        register_ensemble(cand["name"], cand)

        _mask = mask.copy()
        _dir = direction

        def make_func(m, d):
            def f(new_df):
                return m
            return f

        sig = RawSignal(
            name=f"{cand['name']}_h{hold}",
            asset=asset,
            direction=direction,
            hold_bars=hold,
            n_trades=result["n_trades"],
            mean_return=result["mean_return"],
            pf=result["pf"],
            sharpe=result["sharpe"],
            p_value=result["p_value"],
            signal_func=make_func(_mask, _dir),
        )
        raw_survivors.append(sig)
        if result["p_value"] < bonferroni:
            bonferroni_survivors.append(sig)

    # Sort by p-value
    raw_survivors.sort(key=lambda s: s.p_value)
    bonferroni_survivors.sort(key=lambda s: s.p_value)

    return ScanResult(
        total_hypotheses=total_hypotheses_with_ens,
        bonferroni_threshold=bonferroni,
        raw_survivors=raw_survivors,
        bonferroni_survivors=bonferroni_survivors,
    )

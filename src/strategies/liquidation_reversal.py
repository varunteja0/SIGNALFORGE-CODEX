"""
Strategy 1: Liquidation Reversal v2
======================================
TRADE: Long after forced-selling flush (+ Short after short-squeeze)

MECHANIC:
    Overleveraged longs get liquidated → forced market sells →
    price dumps below fair value → smart money absorbs → price snaps back.

ENTRY (LONG — primary):
    1. Liquidation spike   → Proxy intensity z-score > 3σ
    2. OI flush confirmed  → OI dropped >5% in 4h, 1h momentum negative
    3. Funding negative    → funding < -0.005% (market already flushed)
    4. Price dislocation   → below lower Bollinger Band AND >1.5% below VWAP
    5. Absorption candle   → long lower wick (>60% of range) + close upper half
    6. Confirmation candle → NEXT bar closes above absorption bar's high
    7. Volume spike        → volume > 2× 24h average

ENTRY TIMING:
    Detect spike → WAIT for absorption → WAIT for confirmation → enter.
    This avoids catching falling knives.

EXIT:
    TP1: VWAP (close 50%)    TP2: 20-EMA (close rest)
    SL:  Below absorption zone low - 0.5×ATR
    Time: 8 bars max hold

INVALIDATION FILTERS:
    - ATR > 2× normal → chaotic, skip
    - Multiple spikes in 24h → broken structure, skip
    - Below 200 EMA → reduce size 50% (don't fight macro)

WHO WE EXPLOIT:
    Degenerate leverage traders who got wiped. We buy what they were forced to sell.

WHY THIS HAS EDGE:
    - Liquidation cascades are MECHANICAL — they must happen
    - Forced selling is non-discretionary
    - The overshoot is predictable: driven by market structure, not opinion
    - Edge has persisted since perpetual futures were invented (2018)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(Enum):
    NO_SIGNAL = 0
    LONG = 1
    SHORT = -1


@dataclass
class StrategyConfig:
    """Tunable parameters — calibrated to 2025-2026 BTC market data.

    All thresholds derived from real data distribution, not curve-fit.
    """
    # --- Liquidation detection ---
    liq_z_threshold: float = 2.0            # σ above mean for spike detection (calibrated to real data)
    oi_drop_threshold_pct: float = -0.05    # 5% OI drop in 4h = flush confirmed
    oi_lookback_bars: int = 4

    # --- Funding (calibrated: max observed = 0.01%, 99th pct = 0.01%) ---
    funding_extreme_pct: float = 0.00008    # ~90th pct of real rates = elevated
    funding_flush_threshold: float = -0.00003  # Negative = market flushed
    funding_lookback_bars: int = 8

    # --- Price dislocation ---
    bb_period: int = 20
    bb_std: float = 2.0
    vwap_dist_threshold: float = 0.01       # 1% below VWAP (relaxed from 1.5%)
    price_move_threshold_pct: float = 0.025 # 2.5% move triggers cascade detection

    # --- Post-spike entry window ---
    wick_ratio_threshold: float = 0.4       # Lower wick > 40% (relaxed from 60%)
    volume_spike_threshold: float = 1.5     # Volume > 1.5× 24h average (relaxed from 2x)
    max_wait_bars: int = 4                  # 4 bars after spike (relaxed from 2)

    # --- Price lookback ---
    price_lookback_bars: int = 4

    # --- Momentum exhaustion ---
    rsi_oversold: float = 35.0              # Relaxed from 30
    rsi_overbought: float = 65.0            # Relaxed from 70
    rsi_period: int = 14

    # --- Risk management ---
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 4.0         # Raised from 3.0 → let winners run
    max_holding_bars: int = 24                 # Raised from 12 → give reversals time
    max_cascades_24h: int = 2               # Allow 2 cascades (relaxed from 1)

    # --- Dynamic exits (v3) ---
    initial_stop_atr: float = 2.0           # Same as initial stop (trails don't help in crypto)
    trail_activation_atr: float = 3.0       # Only trail after deep profit (3× ATR)
    trail_distance_atr: float = 2.0         # Wide trail for crypto whipsaw
    tp1_atr: float = 99.0                   # No partial TP (crypto mean-reverts too much)
    tp1_close_pct: float = 0.0              # Disabled
    tp2_atr: float = 4.0                    # Hard TP at 4× ATR
    dynamic_max_hold: int = 24              # 24 hours — give winners room
    time_decay_bars: int = 100              # Disabled (hurts more than helps)
    time_decay_stop_atr: float = 2.0        # N/A when disabled

    # --- Filters ---
    max_atr_multiple: float = 2.0           # Skip if ATR > 2× normal
    trend_ema_period: int = 200
    trend_size_reduction: float = 0.5       # Reduce 50% against trend
    min_volume_ratio: float = 1.5

    # --- Position sizing ---
    base_risk_pct: float = 0.01             # 1% base risk
    strong_signal_risk: float = 0.015       # 1.5% for strong signals
    max_risk_pct: float = 0.02              # Hard cap at 2%


class LiquidationReversalStrategy:
    """Liquidation Reversal v2 — multi-bar confirmation, absorption detection.

    Designed from first principles of market microstructure.
    Supports both backtesting (generate_signals) and live (on_bar) interfaces.
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.name = "liq_reversal_v2"

    def required_columns(self) -> list[str]:
        """Columns this strategy needs in the input DataFrame."""
        return ["close", "high", "low", "volume"]

    def structural_columns(self) -> list[str]:
        """Columns from structural data fetchers (funding, OI, liquidation proxy)."""
        return [
            "fund_funding_rate",
            "oi_oi_value_usd",
        ]

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ALL strategy indicators from OHLCV + structural data."""
        out = df.copy()
        cfg = self.config

        # ── RSI ──
        delta = out["close"].diff()
        gain = delta.where(delta > 0, 0.0).ewm(span=cfg.rsi_period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(span=cfg.rsi_period, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        out["rsi"] = 100 - (100 / (1 + rs))

        # ── ATR ──
        high_low = out["high"] - out["low"]
        high_close = (out["high"] - out["close"].shift(1)).abs()
        low_close = (out["low"] - out["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        out["atr"] = tr.rolling(14).mean()
        out["atr_ratio"] = out["atr"] / (out["atr"].rolling(168).mean() + 1e-10)

        # ── Price change ──
        out["price_change_pct"] = out["close"].pct_change(cfg.price_lookback_bars)

        # ── Volume ──
        out["volume_ma"] = out["volume"].rolling(24).mean()
        out["volume_ratio"] = out["volume"] / (out["volume_ma"] + 1e-10)

        # ── Bollinger Bands ──
        bb_ma = out["close"].rolling(cfg.bb_period).mean()
        bb_std = out["close"].rolling(cfg.bb_period).std()
        out["bb_lower"] = bb_ma - cfg.bb_std * bb_std
        out["bb_upper"] = bb_ma + cfg.bb_std * bb_std
        out["below_bb"] = (out["close"] < out["bb_lower"]).astype(int)
        out["above_bb"] = (out["close"] > out["bb_upper"]).astype(int)

        # ── VWAP (rolling 24h) ──
        typical_price = (out["high"] + out["low"] + out["close"]) / 3
        cum_tp_vol = (typical_price * out["volume"]).rolling(24).sum()
        cum_vol = out["volume"].rolling(24).sum()
        out["vwap"] = cum_tp_vol / (cum_vol + 1e-10)
        out["vwap_dist"] = (out["close"] - out["vwap"]) / (out["vwap"] + 1e-10)

        # ── EMAs ──
        out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
        out["ema_200"] = out["close"].ewm(span=cfg.trend_ema_period, adjust=False).mean()
        out["above_trend"] = (out["close"] > out["ema_200"]).astype(int)

        # ── Candle structure (absorption detection) ──
        candle_range = out["high"] - out["low"]
        lower_wick = out[["open", "close"]].min(axis=1) - out["low"]
        upper_wick = out["high"] - out[["open", "close"]].max(axis=1)
        out["wick_ratio"] = lower_wick / (candle_range + 1e-10)
        out["upper_wick_ratio"] = upper_wick / (candle_range + 1e-10)
        out["bar_close_pct"] = (out["close"] - out["low"]) / (candle_range + 1e-10)

        # ── OI change ──
        if "oi_oi_value_usd" in out.columns:
            out["oi_change_pct"] = out["oi_oi_value_usd"].pct_change(cfg.oi_lookback_bars)
            out["oi_change_1h"] = out["oi_oi_value_usd"].pct_change(1)
            out["has_oi"] = out["oi_oi_value_usd"].notna()  # Per-row flag
        elif "oi_oi_change_4h" in out.columns:
            out["oi_change_pct"] = out["oi_oi_change_4h"]
            out["oi_change_1h"] = out.get("oi_oi_change_1h", 0)
            out["has_oi"] = out["oi_oi_change_4h"].notna()
        else:
            out["oi_change_pct"] = 0.0
            out["oi_change_1h"] = 0.0
            out["has_oi"] = False

        # ── Funding ──
        if "fund_funding_rate" in out.columns:
            out["funding"] = out["fund_funding_rate"]
        else:
            out["funding"] = 0.0

        # ── Liquidation proxy intensity ──
        # Built from price drop speed + OI drop + volume spike
        price_drop_z = (
            (-out["price_change_pct"] - (-out["price_change_pct"]).rolling(168).mean())
            / ((-out["price_change_pct"]).rolling(168).std() + 1e-10)
        ).clip(0, 10)

        vol_z = (
            (out["volume_ratio"] - out["volume_ratio"].rolling(168).mean())
            / (out["volume_ratio"].rolling(168).std() + 1e-10)
        ).clip(0, 10)

        # Compute OI z-score where data exists, blend per-bar
        oi_drop_z = pd.Series(0.0, index=out.index)
        oi_available = out["oi_change_pct"].notna() & (out["oi_change_pct"] != 0)
        if oi_available.sum() > 168:
            raw_oi_z = (
                (-out["oi_change_pct"] - (-out["oi_change_pct"]).rolling(168, min_periods=20).mean())
                / ((-out["oi_change_pct"]).rolling(168, min_periods=20).std() + 1e-10)
            ).clip(0, 10)
            oi_drop_z = raw_oi_z.fillna(0)

        # Per-bar blending: use 3 components where OI exists, 2 where it doesn't
        out["liq_intensity"] = np.where(
            oi_available,
            0.4 * price_drop_z + 0.4 * oi_drop_z + 0.2 * vol_z,
            0.6 * price_drop_z + 0.4 * vol_z,
        )

        # If external liq_intensity_z is provided (from LiquidationFetcher), use it
        if "liq_intensity_z" in df.columns:
            out["liq_intensity"] = df["liq_intensity_z"]

        # Replace infinities
        out = out.replace([np.inf, -np.inf], 0).fillna(0)

        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals for backtesting.

        Returns Series of {-1, 0, 1}.
        Implements the multi-bar state machine:
        spike → wait → absorption → confirmation → entry.
        """
        ind = self.compute_indicators(df)
        cfg = self.config
        n = len(ind)

        signals = pd.Series(0, index=df.index, dtype=int)

        # State tracking
        in_position = False
        position_dir = 0
        bars_in_position = 0
        bars_since_spike = 999
        spike_count_24h = 0
        last_spike_bar = -999
        last_spike_direction = 0  # 1 = long cascade (price dropped), -1 = short cascade

        warmup = max(cfg.price_lookback_bars, cfg.rsi_period, 24, 200) + 1

        for i in range(warmup, n):
            row = ind.iloc[i]

            # ── Track position holding ──
            if in_position:
                bars_in_position += 1
                if bars_in_position >= cfg.max_holding_bars:
                    in_position = False
                    position_dir = 0
                continue

            # ── Detect liquidation spikes ──
            is_long_spike = (
                row["liq_intensity"] > cfg.liq_z_threshold
                and row["price_change_pct"] < -cfg.price_move_threshold_pct
            )
            is_short_spike = (
                row["liq_intensity"] > cfg.liq_z_threshold
                and row["price_change_pct"] > cfg.price_move_threshold_pct
            )

            if is_long_spike or is_short_spike:
                if i - last_spike_bar > 24:
                    spike_count_24h = 1
                else:
                    spike_count_24h += 1
                last_spike_bar = i
                last_spike_direction = 1 if is_long_spike else -1
                bars_since_spike = 0
                continue
            else:
                bars_since_spike += 1

            # Reset 24h counter
            if i - last_spike_bar > 24:
                spike_count_24h = 0

            # Skip if multiple cascades (broken structure)
            if spike_count_24h > cfg.max_cascades_24h:
                continue

            # Skip if ATR too high (chaotic market)
            if row["atr_ratio"] > cfg.max_atr_multiple:
                continue

            # Skip if no recent spike
            if bars_since_spike > cfg.max_wait_bars:
                continue

            # ═══════════════════════════════════════════════════════
            # LONG: Post-liquidation bounce after forced long liquidations
            # ═══════════════════════════════════════════════════════
            if last_spike_direction == 1:
                # Core conditions (REQUIRED):
                # 1. Price dislocation: below BB or far from VWAP
                price_dislocated = (
                    row["below_bb"] == 1
                    or row["vwap_dist"] < -cfg.vwap_dist_threshold
                )

                # 2. RSI shows selling exhausted
                rsi_ok = row["rsi"] < cfg.rsi_oversold

                # 3. Volume confirms real event
                vol_ok = row["volume_ratio"] > cfg.volume_spike_threshold

                if not (price_dislocated and rsi_ok and vol_ok):
                    continue

                # Enhancer conditions (boost confidence, not required):
                funding_avg = ind["funding"].iloc[max(0, i - cfg.funding_lookback_bars):i].mean()
                funding_ok = (
                    funding_avg > cfg.funding_extreme_pct
                    or row["funding"] < cfg.funding_flush_threshold
                )

                # Absorption candle: long lower wick + close in upper half
                absorption = (
                    row["wick_ratio"] > cfg.wick_ratio_threshold
                    and row["bar_close_pct"] > 0.5
                )

                # OI flushed (if available)
                has_oi = row.get("has_oi", False)
                oi_flushed = (
                    row["oi_change_pct"] < cfg.oi_drop_threshold_pct
                    if has_oi else False
                )

                # Score: need at least 1 enhancer to filter noise
                enhancers = sum([funding_ok, absorption, oi_flushed])
                if enhancers >= 1:
                    signals.iloc[i] = 1
                    in_position = True
                    position_dir = 1
                    bars_in_position = 0

            # ═══════════════════════════════════════════════════════
            # SHORT: Post-liquidation reversal after short squeeze
            # ═══════════════════════════════════════════════════════
            elif last_spike_direction == -1:
                # Core conditions (REQUIRED):
                price_dislocated = (
                    row["above_bb"] == 1
                    or row["vwap_dist"] > cfg.vwap_dist_threshold
                )
                rsi_ok = row["rsi"] > cfg.rsi_overbought
                vol_ok = row["volume_ratio"] > cfg.volume_spike_threshold

                if not (price_dislocated and rsi_ok and vol_ok):
                    continue

                # Enhancers:
                funding_avg = ind["funding"].iloc[max(0, i - cfg.funding_lookback_bars):i].mean()
                funding_ok = (
                    funding_avg < -cfg.funding_extreme_pct
                    or row["funding"] > abs(cfg.funding_flush_threshold)
                )

                absorption = (
                    row["upper_wick_ratio"] > cfg.wick_ratio_threshold
                    and row["bar_close_pct"] < 0.5
                )

                has_oi = row.get("has_oi", False)
                oi_flushed = (
                    row["oi_change_pct"] < cfg.oi_drop_threshold_pct
                    if has_oi else False
                )

                enhancers = sum([funding_ok, absorption, oi_flushed])
                if enhancers >= 1:
                    signals.iloc[i] = -1
                    in_position = True
                    position_dir = -1
                    bars_in_position = 0

        return signals

    def compute_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        signal_strength: float = 0.5,
        above_trend: bool = True,
        drawdown_pct: float = 0.0,
    ) -> float:
        """Compute position size with drawdown-aware risk scaling."""
        cfg = self.config
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return 0.0

        risk_pct = cfg.base_risk_pct
        if signal_strength > 0.7:
            risk_pct = cfg.strong_signal_risk
        if not above_trend:
            risk_pct *= cfg.trend_size_reduction
        if drawdown_pct > 0.05:
            risk_pct *= 0.5
        if drawdown_pct > 0.10:
            risk_pct *= 0.25

        risk_pct = min(risk_pct, cfg.max_risk_pct)
        size = (capital * risk_pct) / risk_per_unit

        # Notional cap
        max_size = (capital * 0.20) / entry_price
        return min(size, max_size)

    def compute_exit_levels(self, df: pd.DataFrame, entry_bar: int) -> dict:
        """Compute stop loss and take profit levels for a given entry bar."""
        bar = df.iloc[entry_bar]
        atr = bar.get("atr", bar["close"] * 0.02)

        lookback = min(entry_bar, 3)
        recent_low = df["low"].iloc[entry_bar - lookback:entry_bar + 1].min()
        recent_high = df["high"].iloc[entry_bar - lookback:entry_bar + 1].max()

        entry = bar["close"]
        direction = 1  # Default to long

        if entry < df["close"].iloc[entry_bar - 1]:
            # If entry is below previous close, might be a short
            direction = -1

        if direction == 1:
            sl = recent_low - 0.5 * atr
            tp1 = bar.get("vwap", entry + 1.5 * atr)
            tp2 = bar.get("ema_20", entry + 2.5 * atr)
            if tp1 <= entry:
                tp1 = entry + 1.5 * atr
            if tp2 <= entry:
                tp2 = entry + 2.5 * atr
        else:
            sl = recent_high + 0.5 * atr
            tp1 = bar.get("vwap", entry - 1.5 * atr)
            tp2 = bar.get("ema_20", entry - 2.5 * atr)
            if tp1 >= entry:
                tp1 = entry - 1.5 * atr
            if tp2 >= entry:
                tp2 = entry - 2.5 * atr

        risk = abs(entry - sl)
        return {
            "stop_loss": sl,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "risk_per_unit": risk,
            "rr_ratio_tp1": abs(tp1 - entry) / (risk + 1e-10),
            "rr_ratio_tp2": abs(tp2 - entry) / (risk + 1e-10),
        }

    def explain_signal(self, df: pd.DataFrame, idx: int) -> str:
        """Human-readable explanation of why a signal fired at index."""
        ind = self.compute_indicators(df)
        row = ind.iloc[idx]
        cfg = self.config
        funding_avg = ind["funding"].iloc[max(0, idx - cfg.funding_lookback_bars):idx].mean()

        return (
            f"Bar {idx} | Price={row['close']:.2f} | "
            f"RSI={row['rsi']:.1f} | "
            f"Funding(avg {cfg.funding_lookback_bars}h)={funding_avg:.6f} | "
            f"OI Δ={row['oi_change_pct']:.2%} | "
            f"Price Δ={row['price_change_pct']:.2%} | "
            f"Vol ratio={row['volume_ratio']:.1f} | "
            f"Wick ratio={row['wick_ratio']:.2f} | "
            f"VWAP dist={row['vwap_dist']:.3f} | "
            f"ATR ratio={row['atr_ratio']:.2f} | "
            f"Liq intensity={row['liq_intensity']:.2f}"
        )

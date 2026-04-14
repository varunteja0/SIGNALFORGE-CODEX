"""
Micro-Strategies — Three Orthogonal Edges from Structural Data
===============================================================
Split from the monolithic funding_mr_v7 into 3 uncorrelated strategies:

    1. Extreme Funding Spike Reversal — high z-score + funding velocity
    2. Post-Liquidation Bounce — OI collapse → price overshoot → bounce
    3. Funding + Vol Compression — coiled spring: squeeze + funding extreme

Each exploits a DIFFERENT market microstructure inefficiency.
Together they triple trade count and diversify PnL concentration.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.engine.strategy_factory import StrategyCandidate, _first_signal_only

logger = logging.getLogger(__name__)


# ─── Strategy 1: Extreme Funding Spike Reversal ─────────────────

class ExtremeFundingSpikeTemplate:
    """Trade only the MOST extreme funding dislocations.

    Difference from funding_mr_v7:
        - Higher z-score threshold (3.5+) — only the most extreme
        - Adds funding VELOCITY check (how fast funding is changing)
        - Shorter hold (8-16 bars) — catch the snap, get out fast
        - Tighter stops (1.5 ATR)

    This catches fewer but HIGHER quality trades.
    """

    PARAM_GRID = {
        "funding_z_threshold": [3.5, 4.0, 4.5, 5.0],
        "funding_lookback": [48, 96, 168],
        "funding_velocity_mult": [1.5, 2.0, 2.5],  # velocity z > this
        "hold_bars": [6, 8, 12, 16],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "tp_atr_mult": [2.0, 3.0, 4.0],
    }

    @staticmethod
    def generate_signals(
        df: pd.DataFrame,
        funding_z_threshold: float = 4.0,
        funding_lookback: int = 96,
        funding_velocity_mult: float = 2.0,
        hold_bars: int = 8,
        **kwargs,
    ) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)

        # Find funding rate column
        funding_col = None
        for col in ["fund_funding_rate", "funding_rate"]:
            if col in df.columns:
                funding_col = col
                break
        if funding_col is None:
            return signals

        funding = df[funding_col].fillna(0)

        # Z-score of funding rate
        f_mean = funding.rolling(funding_lookback, min_periods=20).mean()
        f_std = funding.rolling(funding_lookback, min_periods=20).std()
        f_z = (funding - f_mean) / (f_std + 1e-10)

        # Funding VELOCITY — how fast is funding changing?
        f_velocity = funding.diff(3).abs()  # 3-bar change in funding
        v_mean = f_velocity.rolling(funding_lookback, min_periods=20).mean()
        v_std = f_velocity.rolling(funding_lookback, min_periods=20).std()
        v_z = (f_velocity - v_mean) / (v_std + 1e-10)

        for i in range(funding_lookback, len(df)):
            z = f_z.iloc[i]
            vz = v_z.iloc[i]

            # Need BOTH extreme level AND velocity
            if abs(z) < funding_z_threshold:
                continue
            if vz < funding_velocity_mult:
                continue

            if z > 0:
                signals.iloc[i] = -1  # Extreme positive funding → short
            else:
                signals.iloc[i] = 1   # Extreme negative funding → long

        return _first_signal_only(signals, hold_bars)

    @staticmethod
    def generate_candidates(n_random: int = 15):
        grid = ExtremeFundingSpikeTemplate.PARAM_GRID
        rng = np.random.default_rng(51)
        candidates = []

        for i in range(n_random):
            params = {key: rng.choice(values) for key, values in grid.items()}
            frozen = dict(params)
            candidates.append(StrategyCandidate(
                name=f"extreme_spike_v{i}",
                template="extreme_funding_spike",
                params=frozen,
                signal_func=lambda df, p=frozen: ExtremeFundingSpikeTemplate.generate_signals(df, **p),
                stop_loss_atr=frozen["stop_atr_mult"],
                take_profit_atr=frozen["tp_atr_mult"],
                max_holding_bars=frozen["hold_bars"],
                description=f"ExtSpike: z={frozen['funding_z_threshold']}, "
                            f"vel={frozen['funding_velocity_mult']}, "
                            f"hold={frozen['hold_bars']}",
            ))

        return candidates


# ─── Strategy 2: Post-Liquidation Bounce ────────────────────────

class PostLiquidationBounceTemplate:
    """Enter AFTER a liquidation cascade completes.

    Mechanic:
        1. Detect OI drop > threshold in N bars (liquidation event)
        2. Wait for price to stabilize (stop making new lows for M bars)
        3. Enter long (or short after short squeeze)
        4. The market overshoots during cascades → mean-revert

    Different from LiquidationReversalTemplate:
        - Uses OI collapse as PRIMARY trigger (not RSI/z-score)
        - Waits for stabilization before entry (not during cascade)
        - Works on the AFTERMATH, not the event itself
    """

    PARAM_GRID = {
        "oi_drop_pct": [5.0, 8.0, 10.0, 15.0],  # % OI drop to detect cascade
        "oi_lookback": [6, 12, 24],  # bars to measure OI drop
        "stabilize_bars": [2, 3, 4, 6],  # bars of no new low before entry
        "hold_bars": [12, 24, 36],
        "stop_atr_mult": [1.5, 2.0, 2.5],
        "tp_atr_mult": [3.0, 4.0, 5.0],
    }

    @staticmethod
    def generate_signals(
        df: pd.DataFrame,
        oi_drop_pct: float = 10.0,
        oi_lookback: int = 12,
        stabilize_bars: int = 3,
        hold_bars: int = 24,
        **kwargs,
    ) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)

        # Find OI column
        oi_col = None
        for col in ["fund_open_interest", "open_interest", "oi"]:
            if col in df.columns:
                oi_col = col
                break

        # Fall back to synthetic OI from volume if no OI column
        if oi_col is None:
            # Use volume surge as proxy for liquidation events
            if "volume" not in df.columns:
                return signals
            vol = df["volume"]
            vol_ma = vol.rolling(48).mean()
            vol_z = (vol - vol_ma) / (vol_ma + 1e-10)
        else:
            oi = df[oi_col].fillna(method="ffill")
            # OI % change over lookback
            oi_pct_change = (oi - oi.shift(oi_lookback)) / (oi.shift(oi_lookback) + 1e-10) * 100

        close = df["close"]
        rsi = df.get("rsi_14")

        start = max(oi_lookback + stabilize_bars, 50)

        for i in range(start, len(df)):
            # Step 1: Detect liquidation cascade
            if oi_col is not None:
                oi_drop = oi_pct_change.iloc[i]
            else:
                # Volume spike as proxy
                oi_drop = -vol_z.iloc[i] * 5  # Fake, but directionally correct

            # Step 2: Check if recent price action shows overshoot
            if oi_col is not None and oi_drop < -oi_drop_pct:
                # OI dropped sharply → long liquidations happened
                # Check price stabilization: no new low in last M bars
                recent_lows = close.iloc[i - stabilize_bars:i + 1]
                if recent_lows.iloc[-1] > recent_lows.min() * 0.999:
                    # Price has stopped making new lows → bounce likely
                    if rsi is not None and rsi.iloc[i] < 45:
                        signals.iloc[i] = 1  # Long bounce
                    elif rsi is None:
                        signals.iloc[i] = 1

            elif oi_col is not None and oi_drop > oi_drop_pct:
                # OI increased sharply (short squeeze liquidations)
                recent_highs = close.iloc[i - stabilize_bars:i + 1]
                if recent_highs.iloc[-1] < recent_highs.max() * 1.001:
                    if rsi is not None and rsi.iloc[i] > 55:
                        signals.iloc[i] = -1  # Short the overextension
                    elif rsi is None:
                        signals.iloc[i] = -1

            elif oi_col is None:
                # Volume proxy path
                v = vol_z.iloc[i] if 'vol_z' in dir() else 0
                if v > 3.0:  # Volume spike
                    price_drop = (close.iloc[i] - close.iloc[i - oi_lookback]) / close.iloc[i - oi_lookback]
                    if price_drop < -0.03:  # Price dropped 3%+
                        recent_lows = close.iloc[i - stabilize_bars:i + 1]
                        if recent_lows.iloc[-1] > recent_lows.min() * 0.999:
                            signals.iloc[i] = 1

        return _first_signal_only(signals, hold_bars)

    @staticmethod
    def generate_candidates(n_random: int = 15):
        grid = PostLiquidationBounceTemplate.PARAM_GRID
        rng = np.random.default_rng(52)
        candidates = []

        for i in range(n_random):
            params = {key: rng.choice(values) for key, values in grid.items()}
            frozen = dict(params)
            candidates.append(StrategyCandidate(
                name=f"liq_bounce_v{i}",
                template="post_liq_bounce",
                params=frozen,
                signal_func=lambda df, p=frozen: PostLiquidationBounceTemplate.generate_signals(df, **p),
                stop_loss_atr=frozen["stop_atr_mult"],
                take_profit_atr=frozen["tp_atr_mult"],
                max_holding_bars=frozen["hold_bars"],
                description=f"LiqBounce: oi_drop={frozen['oi_drop_pct']}%, "
                            f"stab={frozen['stabilize_bars']}, "
                            f"hold={frozen['hold_bars']}",
            ))

        return candidates


# ─── Strategy 3: Funding + Volatility Compression ───────────────

class FundingVolSqueezeTemplate:
    """The coiled spring: funding extreme + volatility compression.

    Mechanic:
        When volatility is compressed (BB squeeze) AND funding is extreme,
        the eventual unwind is VIOLENT. This is the highest-conviction setup.

        - BB width at historic low → energy is stored
        - Funding extreme → there's a directional bias
        - The squeeze resolves in the OPPOSITE direction of funding
          (because funding extreme = crowded positioning that will unwind)

    Difference from VolSqueeze:
        - Requires funding extreme as DIRECTIONAL filter
        - Doesn't need price breakout — the squeeze + funding IS the signal

    Difference from FundingReversion:
        - Only trades when vol is compressed (higher conviction)
        - Longer holds (squeeze breakouts trend for days)
    """

    PARAM_GRID = {
        "bb_width_percentile": [5, 10, 15],
        "bb_period": [20, 30],
        "funding_z_threshold": [1.5, 2.0, 2.5],
        "funding_lookback": [96, 168],
        "hold_bars": [16, 24, 36, 48],
        "stop_atr_mult": [1.5, 2.0, 2.5],
        "tp_atr_mult": [3.0, 4.0, 5.0, 6.0],
    }

    @staticmethod
    def generate_signals(
        df: pd.DataFrame,
        bb_width_percentile: int = 10,
        bb_period: int = 20,
        funding_z_threshold: float = 2.0,
        funding_lookback: int = 168,
        hold_bars: int = 24,
        **kwargs,
    ) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)

        # Funding rate
        funding_col = None
        for col in ["fund_funding_rate", "funding_rate"]:
            if col in df.columns:
                funding_col = col
                break
        if funding_col is None:
            return signals

        funding = df[funding_col].fillna(0)

        # Funding z-score
        f_mean = funding.rolling(funding_lookback, min_periods=20).mean()
        f_std = funding.rolling(funding_lookback, min_periods=20).std()
        f_z = (funding - f_mean) / (f_std + 1e-10)

        # Bollinger Band width
        close = df["close"]
        sma = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        bb_width = (2 * std) / (sma + 1e-10)

        # Rolling percentile of BB width
        bb_rank = bb_width.rolling(200, min_periods=50).rank(pct=True) * 100

        start = max(funding_lookback, 200)

        for i in range(start, len(df)):
            # Condition 1: Volatility is compressed
            if bb_rank.iloc[i] > bb_width_percentile:
                continue

            # Condition 2: Funding is extreme
            z = f_z.iloc[i]
            if abs(z) < funding_z_threshold:
                continue

            # Direction: OPPOSITE to funding (crowded trade will unwind)
            if z > 0:
                signals.iloc[i] = -1  # Crowded longs → short
            else:
                signals.iloc[i] = 1   # Crowded shorts → long

        return _first_signal_only(signals, hold_bars)

    @staticmethod
    def generate_candidates(n_random: int = 15):
        grid = FundingVolSqueezeTemplate.PARAM_GRID
        rng = np.random.default_rng(53)
        candidates = []

        for i in range(n_random):
            params = {key: rng.choice(values) for key, values in grid.items()}
            frozen = dict(params)
            candidates.append(StrategyCandidate(
                name=f"fund_vol_squeeze_v{i}",
                template="funding_vol_squeeze",
                params=frozen,
                signal_func=lambda df, p=frozen: FundingVolSqueezeTemplate.generate_signals(df, **p),
                stop_loss_atr=frozen["stop_atr_mult"],
                take_profit_atr=frozen["tp_atr_mult"],
                max_holding_bars=frozen["hold_bars"],
                description=f"FundVolSqz: bb_pct={frozen['bb_width_percentile']}, "
                            f"f_z={frozen['funding_z_threshold']}, "
                            f"hold={frozen['hold_bars']}",
            ))

        return candidates


# ─── Convenience: generate all micro-strategy candidates ────────

def generate_all_micro_candidates(n_per: int = 15) -> list[StrategyCandidate]:
    """Generate candidates from all 3 micro-strategy templates."""
    candidates = []
    candidates.extend(ExtremeFundingSpikeTemplate.generate_candidates(n_per))
    candidates.extend(PostLiquidationBounceTemplate.generate_candidates(n_per))
    candidates.extend(FundingVolSqueezeTemplate.generate_candidates(n_per))
    logger.info(f"Generated {len(candidates)} micro-strategy candidates "
                f"from 3 templates ({n_per} each)")
    return candidates

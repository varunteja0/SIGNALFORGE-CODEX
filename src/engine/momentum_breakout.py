"""
Momentum Breakout Strategy — Orthogonal Alpha Family
=====================================================
Complements the mean-reversion/funding strategies by trading WITH
the trend instead of against it.

Mechanic:
    Price breaks above/below a channel → momentum continuation.
    Uses Donchian channels + ATR expansion + volume confirmation.

Why this is orthogonal:
    - Mean reversion: buy dips, short rips → works in sideways/choppy
    - Momentum breakout: buy breakouts, short breakdowns → works in trends
    - Correlation between them should be NEGATIVE → perfect diversifier

This is NOT the same as MomentumExhaustionTemplate (which trades
reversals FROM momentum). This trades in the DIRECTION of momentum.
"""

import logging

import numpy as np
import pandas as pd

from src.engine.strategy_factory import StrategyCandidate, _first_signal_only

logger = logging.getLogger(__name__)


class MomentumBreakoutTemplate:
    """Donchian channel breakout with ATR expansion and volume confirmation.

    Entry logic:
        LONG:  close > highest high of N bars AND
               ATR expanding (current ATR > avg ATR) AND
               volume above average
        SHORT: close < lowest low of N bars AND same filters

    Exit: ATR-based trailing stop (let winners run).

    This captures trend starts — the worst time for mean reversion
    and the best time for momentum.
    """

    PARAM_GRID = {
        "channel_period": [20, 30, 48, 72],    # Donchian channel lookback
        "atr_expansion": [1.2, 1.5, 1.8],      # ATR must be > avg × this
        "volume_mult": [1.0, 1.3, 1.5, 2.0],   # Volume must be > avg × this
        "hold_bars": [12, 24, 36, 48],
        "stop_atr_mult": [1.5, 2.0, 2.5, 3.0],
        "tp_atr_mult": [3.0, 4.0, 5.0, 6.0],
    }

    @staticmethod
    def generate_signals(
        df: pd.DataFrame,
        channel_period: int = 30,
        atr_expansion: float = 1.5,
        volume_mult: float = 1.3,
        hold_bars: int = 24,
        **kwargs,
    ) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Donchian channel
        dc_high = high.rolling(channel_period).max()
        dc_low = low.rolling(channel_period).min()

        # ATR for expansion check
        if "atr_14" in df.columns:
            atr = df["atr_14"]
        else:
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()

        atr_avg = atr.rolling(channel_period).mean()

        # Volume
        if "volume" in df.columns:
            vol = df["volume"]
            vol_avg = vol.rolling(channel_period).mean()
        else:
            vol = None

        start = channel_period + 14

        for i in range(start, len(df)):
            # Must have ATR expansion (volatility increasing = breakout)
            if atr.iloc[i] < atr_avg.iloc[i] * atr_expansion:
                continue

            # Volume confirmation
            if vol is not None and vol.iloc[i] < vol_avg.iloc[i] * volume_mult:
                continue

            # Breakout detection
            if close.iloc[i] > dc_high.iloc[i - 1]:
                # Price broke above channel → LONG
                signals.iloc[i] = 1

            elif close.iloc[i] < dc_low.iloc[i - 1]:
                # Price broke below channel → SHORT
                signals.iloc[i] = -1

        return _first_signal_only(signals, hold_bars)

    @staticmethod
    def generate_candidates(n_random: int = 15):
        grid = MomentumBreakoutTemplate.PARAM_GRID
        rng = np.random.default_rng(60)
        candidates = []

        for i in range(n_random):
            params = {key: rng.choice(values) for key, values in grid.items()}
            frozen = dict(params)
            candidates.append(StrategyCandidate(
                name=f"momentum_bo_v{i}",
                template="momentum_breakout",
                params=frozen,
                signal_func=lambda df, p=frozen: MomentumBreakoutTemplate.generate_signals(df, **p),
                stop_loss_atr=frozen["stop_atr_mult"],
                take_profit_atr=frozen["tp_atr_mult"],
                max_holding_bars=frozen["hold_bars"],
                description=f"MomBO: ch={frozen['channel_period']}, "
                            f"atr_exp={frozen['atr_expansion']}, "
                            f"vol={frozen['volume_mult']}, "
                            f"hold={frozen['hold_bars']}",
            ))

        return candidates

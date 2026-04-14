"""
Liquidity Sweep + Reversal Strategy
=====================================
Catches the most profitable pattern in crypto: smart money hunting stops.

WHAT IS A LIQUIDITY SWEEP:
    Price spikes through a known support/resistance level (where stops cluster),
    triggers liquidations/stop-losses, then REVERSES sharply. The "sweep" is
    the wick that goes through the level. The reversal is the trade.

    This is what smart money does:
    1. Drive price into obvious stop cluster
    2. Trigger cascade of liquidations
    3. Absorb cheap liquidity
    4. Reverse price in the intended direction

WHY THIS IS ORTHOGONAL:
    - Mean reversion: trades statistical extremes (funding z-score)
    - Momentum: trades breakouts (Donchian)
    - Vol squeeze: trades compressed ranges
    - Liquidity sweep: trades FAKEOUTS — failed breakouts that reverse
      Negative correlation with momentum breakout = perfect diversifier

DETECTION LOGIC:
    1. Identify key levels (recent swing highs/lows, round numbers)
    2. Detect wick THROUGH level with close BACK inside (sweep pattern)
    3. Confirm with: volume spike, funding imbalance, wick-to-body ratio
    4. Enter reversal direction with tight stop beyond sweep wick

This captures the pattern that wrecks retail traders:
    "I got stopped out, then price went exactly where I expected"
"""

import logging

import numpy as np
import pandas as pd

from src.engine.strategy_factory import StrategyCandidate, _first_signal_only

logger = logging.getLogger(__name__)


class LiquiditySweepTemplate:
    """Detects liquidity sweeps (stop hunts) and trades the reversal.

    Entry logic (LONG — sweep below support):
        1. Price wicks below recent swing low (lookback period)
        2. Close is ABOVE the swing low (wick = sweep, body = rejection)
        3. Lower shadow is large relative to body (wick_ratio > threshold)
        4. Volume spike confirms liquidity was grabbed
        5. Optional: funding rate shows shorts are overextended

    Entry logic (SHORT — sweep above resistance):
        Mirror of above — wick above swing high, close below, upper shadow large.

    Exit: Tight stops (the thesis is invalidated if sweep continues).
    """

    PARAM_GRID = {
        "swing_lookback": [24, 48, 72, 96],    # Hours to find swing levels
        "wick_ratio": [1.5, 2.0, 2.5, 3.0],    # Min wick-to-body ratio
        "volume_mult": [1.3, 1.5, 2.0],         # Volume must be above avg
        "sweep_margin_pct": [0.001, 0.002, 0.003],  # How far past level = sweep
        "hold_bars": [8, 12, 16, 24],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "tp_atr_mult": [2.0, 3.0, 4.0, 5.0],
        "funding_confirm": [True, False],         # Require funding confirmation
    }

    @staticmethod
    def generate_signals(
        df: pd.DataFrame,
        swing_lookback: int = 48,
        wick_ratio: float = 2.0,
        volume_mult: float = 1.5,
        sweep_margin_pct: float = 0.002,
        hold_bars: int = 12,
        funding_confirm: bool = True,
        **kwargs,
    ) -> pd.Series:
        """Generate liquidity sweep reversal signals.

        Args:
            swing_lookback: bars to look back for swing levels
            wick_ratio: minimum wick-to-body ratio for rejection candle
            volume_mult: minimum volume relative to moving average
            sweep_margin_pct: how far price must go past level (0.2% = clear sweep)
            hold_bars: position holding period
            funding_confirm: require funding rate to confirm counter-positioning
        """
        signals = pd.Series(0, index=df.index, dtype=int)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        # Candle body and shadows
        body = (close - open_).abs()
        body = body.clip(lower=1e-10)  # Avoid division by zero

        upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
        lower_shadow = pd.concat([close, open_], axis=1).min(axis=1) - low

        # Volume
        if "volume" in df.columns:
            vol = df["volume"]
            vol_avg = vol.rolling(swing_lookback).mean()
        else:
            vol = None

        # Funding z-score for confirmation
        if "fund_funding_zscore" in df.columns:
            funding_z = df["fund_funding_zscore"]
        else:
            funding_z = None

        # Swing levels — rolling min/max of CLOSE (not wick)
        # These are where stops cluster: just below support, just above resistance
        swing_low = close.rolling(swing_lookback).min()
        swing_high = close.rolling(swing_lookback).max()

        start = swing_lookback + 5

        for i in range(start, len(df)):
            # The swing level is from 1 bar ago (don't include current bar)
            sl = swing_low.iloc[i - 1]
            sh = swing_high.iloc[i - 1]

            # Current candle properties
            c_body = body.iloc[i]
            c_lower = lower_shadow.iloc[i]
            c_upper = upper_shadow.iloc[i]
            c_close = close.iloc[i]
            c_low = low.iloc[i]
            c_high = high.iloc[i]

            # ── LONG: Sweep below support ──
            # 1. Low goes BELOW swing low (the sweep)
            sweep_threshold_low = sl * (1 - sweep_margin_pct)
            if c_low < sweep_threshold_low:
                # 2. Close is ABOVE swing low (rejection — price came back)
                if c_close > sl:
                    # 3. Lower shadow is large relative to body
                    if c_lower > c_body * wick_ratio:
                        # 4. Volume spike
                        vol_ok = True
                        if vol is not None:
                            vol_ok = vol.iloc[i] > vol_avg.iloc[i] * volume_mult

                        # 5. Funding confirmation (shorts overextended = long setup)
                        fund_ok = True
                        if funding_confirm and funding_z is not None:
                            fund_ok = funding_z.iloc[i] < -0.5  # Shorts paying longs

                        if vol_ok and fund_ok:
                            signals.iloc[i] = 1

            # ── SHORT: Sweep above resistance ──
            sweep_threshold_high = sh * (1 + sweep_margin_pct)
            if c_high > sweep_threshold_high:
                if c_close < sh:
                    if c_upper > c_body * wick_ratio:
                        vol_ok = True
                        if vol is not None:
                            vol_ok = vol.iloc[i] > vol_avg.iloc[i] * volume_mult

                        fund_ok = True
                        if funding_confirm and funding_z is not None:
                            fund_ok = funding_z.iloc[i] > 0.5  # Longs paying shorts

                        if vol_ok and fund_ok:
                            signals.iloc[i] = -1

        return _first_signal_only(signals, hold_bars)

    @staticmethod
    def generate_candidates(n_random: int = 20):
        """Generate random parameter combinations for grid search."""
        grid = LiquiditySweepTemplate.PARAM_GRID
        rng = np.random.default_rng(77)
        candidates = []

        for i in range(n_random):
            params = {key: rng.choice(values) for key, values in grid.items()}
            frozen = dict(params)
            candidates.append(StrategyCandidate(
                name=f"liq_sweep_v{i}",
                template="liquidity_sweep",
                params=frozen,
                signal_func=lambda df, p=frozen: LiquiditySweepTemplate.generate_signals(df, **p),
                stop_loss_atr=frozen["stop_atr_mult"],
                take_profit_atr=frozen["tp_atr_mult"],
                max_holding_bars=frozen["hold_bars"],
                description=f"LiqSweep: swing={frozen['swing_lookback']}, "
                            f"wick={frozen['wick_ratio']}, "
                            f"vol={frozen['volume_mult']}, "
                            f"fund={'Y' if frozen['funding_confirm'] else 'N'}",
            ))

        return candidates

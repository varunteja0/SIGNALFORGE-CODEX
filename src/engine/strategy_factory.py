"""
Strategy Factory — Systematic Generation of Trading Strategy Variants
=====================================================================
Generates candidates from multiple strategy templates with parameter
variations. Each template exploits a different market microstructure
inefficiency.

Templates:
    1. Liquidation Reversal  — Proven: PF 1.50 on BTC+ETH
    2. Funding Rate Reversion — Exploit mechanical funding normalization
    3. Volatility Squeeze     — BB squeeze → breakout momentum
    4. Momentum Exhaustion    — RSI divergence at extremes

Each template produces signals in [-1, 0, 1] format compatible with
Backtester.run().
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────

@dataclass
class StrategyCandidate:
    """A specific strategy configuration ready for backtesting."""
    name: str
    template: str
    params: dict
    signal_func: Callable[[pd.DataFrame], pd.Series]
    # Exit params for backtester
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 4.0
    max_holding_bars: int = 24
    position_size_pct: float = 0.01
    description: str = ""

    def __repr__(self):
        return f"Candidate({self.name}, template={self.template})"


# ─── Helper ──────────────────────────────────────────────────────

def _first_signal_only(signals: pd.Series, cooldown: int) -> pd.Series:
    """Keep only the first signal in each cluster, enforce cooldown."""
    result = np.zeros(len(signals), dtype=int)
    last_idx = -cooldown - 1

    for i in range(len(signals)):
        if signals.iloc[i] != 0 and (i - last_idx) >= cooldown:
            result[i] = signals.iloc[i]
            last_idx = i

    return pd.Series(result, index=signals.index)


# ─── Template 1: Liquidation Reversal (proven) ──────────────────

class LiquidationReversalTemplate:
    """Liquidation cascade reversal with structural data.

    This is the proven template (PF 1.50 on BTC+ETH).
    Variations sweep the key parameters around the calibrated center.
    """

    PARAM_GRID = {
        "liq_z_threshold": [1.5, 2.0, 2.5, 3.0],
        "rsi_oversold": [30.0, 35.0, 40.0],
        "rsi_overbought": [60.0, 65.0, 70.0],
        "max_wait_bars": [2, 4, 6],
        "stop_loss_atr_mult": [1.5, 2.0, 2.5],
        "take_profit_atr_mult": [3.0, 4.0, 5.0],
        "max_holding_bars": [12, 24, 36],
    }

    @staticmethod
    def generate_candidates(n_random: int = 20, include_best: bool = True):
        from src.strategies.liquidation_reversal import (
            LiquidationReversalStrategy,
            StrategyConfig,
        )

        candidates = []

        # The proven best configuration
        if include_best:
            strategy = LiquidationReversalStrategy(StrategyConfig())
            cfg = strategy.config
            candidates.append(StrategyCandidate(
                name="liq_reversal_best",
                template="liquidation_reversal",
                params={k: getattr(cfg, k) for k in LiquidationReversalTemplate.PARAM_GRID},
                signal_func=lambda df, s=strategy: s.generate_signals(df),
                stop_loss_atr=cfg.stop_loss_atr_mult,
                take_profit_atr=cfg.take_profit_atr_mult,
                max_holding_bars=cfg.max_holding_bars,
                position_size_pct=cfg.base_risk_pct,
                description="Proven best: SL=2, TP=4, MH=24",
            ))

        # Random parameter variations
        rng = np.random.default_rng(42)
        grid = LiquidationReversalTemplate.PARAM_GRID

        for i in range(n_random):
            params = {key: rng.choice(values) for key, values in grid.items()}
            config = StrategyConfig()
            for key, value in params.items():
                setattr(config, key, value)
            strategy = LiquidationReversalStrategy(config)

            candidates.append(StrategyCandidate(
                name=f"liq_reversal_v{i}",
                template="liquidation_reversal",
                params=params,
                signal_func=lambda df, s=strategy: s.generate_signals(df),
                stop_loss_atr=params["stop_loss_atr_mult"],
                take_profit_atr=params["take_profit_atr_mult"],
                max_holding_bars=params["max_holding_bars"],
                description=f"liq_z={params['liq_z_threshold']}, "
                            f"rsi={params['rsi_oversold']}, "
                            f"sl={params['stop_loss_atr_mult']}, "
                            f"tp={params['take_profit_atr_mult']}",
            ))

        return candidates


# ─── Template 2: Funding Rate Mean Reversion ────────────────────

class FundingReversionTemplate:
    """Exploit mechanical funding rate normalization.

    Mechanic: When funding rate is extreme, it must revert.
    Extreme positive → short (longs overpaying → will reduce)
    Extreme negative → long (shorts overpaying → will reduce)

    This is MECHANICAL: funding rates are bounded and mean-revert by design.
    """

    PARAM_GRID = {
        "funding_entry_zscore": [1.5, 2.0, 2.5, 3.0],
        "funding_lookback": [48, 96, 168],
        "hold_bars": [4, 8, 12, 24],
        "stop_atr_mult": [1.5, 2.0, 2.5],
        "tp_atr_mult": [2.0, 3.0, 4.0],
        "require_price_confirmation": [True, False],
    }

    @staticmethod
    def generate_signals(
        df: pd.DataFrame,
        funding_entry_zscore: float = 2.0,
        funding_lookback: int = 96,
        hold_bars: int = 8,
        require_price_confirmation: bool = True,
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

        # RSI for confirmation
        rsi = df.get("rsi_14")

        for i in range(funding_lookback, len(df)):
            z = f_z.iloc[i]

            if z > funding_entry_zscore:
                # Funding extremely positive → short
                if require_price_confirmation and rsi is not None:
                    if rsi.iloc[i] > 60:
                        signals.iloc[i] = -1
                else:
                    signals.iloc[i] = -1

            elif z < -funding_entry_zscore:
                # Funding extremely negative → long
                if require_price_confirmation and rsi is not None:
                    if rsi.iloc[i] < 40:
                        signals.iloc[i] = 1
                else:
                    signals.iloc[i] = 1

        return _first_signal_only(signals, hold_bars)

    @staticmethod
    def generate_candidates(n_random: int = 20):
        grid = FundingReversionTemplate.PARAM_GRID
        rng = np.random.default_rng(43)
        candidates = []

        for i in range(n_random):
            params = {key: rng.choice(values) for key, values in grid.items()}
            frozen = dict(params)
            candidates.append(StrategyCandidate(
                name=f"funding_mr_v{i}",
                template="funding_reversion",
                params=frozen,
                signal_func=lambda df, p=frozen: FundingReversionTemplate.generate_signals(df, **p),
                stop_loss_atr=frozen["stop_atr_mult"],
                take_profit_atr=frozen["tp_atr_mult"],
                max_holding_bars=frozen["hold_bars"],
                description=f"FundMR: z={frozen['funding_entry_zscore']}, "
                            f"lb={frozen['funding_lookback']}, "
                            f"hold={frozen['hold_bars']}",
            ))

        return candidates


# ─── Template 3: Volatility Squeeze Breakout ────────────────────

class VolSqueezeTemplate:
    """Bollinger Band squeeze → breakout.

    Mechanic: Low-volatility (compressed BB) periods precede explosive moves.
    Enter when BB width is at historical low and price breaks out.
    """

    PARAM_GRID = {
        "bb_width_percentile": [5, 10, 15, 20],
        "bb_period": [14, 20, 30],
        "bb_std": [1.5, 2.0, 2.5],
        "confirm_close_outside": [True, False],
        "hold_bars": [6, 12, 24],
        "stop_atr_mult": [1.5, 2.0],
        "tp_atr_mult": [2.0, 3.0, 4.0],
    }

    @staticmethod
    def generate_signals(
        df: pd.DataFrame,
        bb_width_percentile: int = 10,
        bb_period: int = 20,
        bb_std: float = 2.0,
        confirm_close_outside: bool = True,
        hold_bars: int = 12,
        **kwargs,
    ) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)
        close = df["close"]

        # Bollinger Bands
        sma = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        upper = sma + bb_std * std
        lower = sma - bb_std * std
        bb_width = (upper - lower) / (sma + 1e-10)

        # Rolling percentile of BB width
        bb_rank = bb_width.rolling(200, min_periods=50).rank(pct=True) * 100

        for i in range(200, len(df)):
            if bb_rank.iloc[i] > bb_width_percentile:
                continue

            # Squeeze detected → check for breakout
            if confirm_close_outside:
                if close.iloc[i] > upper.iloc[i]:
                    signals.iloc[i] = 1
                elif close.iloc[i] < lower.iloc[i]:
                    signals.iloc[i] = -1
            else:
                if df["high"].iloc[i] > upper.iloc[i]:
                    signals.iloc[i] = 1
                elif df["low"].iloc[i] < lower.iloc[i]:
                    signals.iloc[i] = -1

        return _first_signal_only(signals, hold_bars)

    @staticmethod
    def generate_candidates(n_random: int = 15):
        grid = VolSqueezeTemplate.PARAM_GRID
        rng = np.random.default_rng(44)
        candidates = []

        for i in range(n_random):
            params = {key: rng.choice(values) for key, values in grid.items()}
            frozen = dict(params)
            candidates.append(StrategyCandidate(
                name=f"vol_squeeze_v{i}",
                template="vol_squeeze",
                params=frozen,
                signal_func=lambda df, p=frozen: VolSqueezeTemplate.generate_signals(df, **p),
                stop_loss_atr=frozen["stop_atr_mult"],
                take_profit_atr=frozen["tp_atr_mult"],
                max_holding_bars=frozen["hold_bars"],
                description=f"VolSqueeze: pct={frozen['bb_width_percentile']}, "
                            f"bb={frozen['bb_period']}, hold={frozen['hold_bars']}",
            ))

        return candidates


# ─── Template 4: Momentum Exhaustion ────────────────────────────

class MomentumExhaustionTemplate:
    """RSI divergence at price extremes.

    Mechanic: Price makes new high/low but RSI doesn't confirm →
    momentum exhaustion → mean reversion.
    """

    PARAM_GRID = {
        "rsi_period": [10, 14, 21],
        "divergence_lookback": [10, 20, 30],
        "rsi_threshold_low": [25, 30, 35],
        "rsi_threshold_high": [65, 70, 75],
        "require_volume_drop": [True, False],
        "hold_bars": [4, 8, 16],
        "stop_atr_mult": [1.5, 2.0, 2.5],
        "tp_atr_mult": [2.0, 3.0, 4.0],
    }

    @staticmethod
    def generate_signals(
        df: pd.DataFrame,
        rsi_period: int = 14,
        divergence_lookback: int = 20,
        rsi_threshold_low: int = 30,
        rsi_threshold_high: int = 70,
        require_volume_drop: bool = True,
        hold_bars: int = 8,
        **kwargs,
    ) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)
        close = df["close"]

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # Volume baseline
        vol_ma = df["volume"].rolling(24).mean() if "volume" in df.columns else None

        start = divergence_lookback + rsi_period
        for i in range(start, len(df)):
            w = slice(i - divergence_lookback, i + 1)
            price_w = close.iloc[w]
            rsi_w = rsi.iloc[w]

            # Bullish divergence: price new low but RSI higher low
            if (close.iloc[i] <= price_w.min() * 1.001
                    and rsi.iloc[i] > rsi_w.min()
                    and rsi.iloc[i] < rsi_threshold_low):
                if require_volume_drop and vol_ma is not None:
                    if df["volume"].iloc[i] < vol_ma.iloc[i]:
                        signals.iloc[i] = 1
                else:
                    signals.iloc[i] = 1

            # Bearish divergence: price new high but RSI lower high
            elif (close.iloc[i] >= price_w.max() * 0.999
                  and rsi.iloc[i] < rsi_w.max()
                  and rsi.iloc[i] > rsi_threshold_high):
                if require_volume_drop and vol_ma is not None:
                    if df["volume"].iloc[i] < vol_ma.iloc[i]:
                        signals.iloc[i] = -1
                else:
                    signals.iloc[i] = -1

        return _first_signal_only(signals, hold_bars)

    @staticmethod
    def generate_candidates(n_random: int = 15):
        grid = MomentumExhaustionTemplate.PARAM_GRID
        rng = np.random.default_rng(45)
        candidates = []

        for i in range(n_random):
            params = {key: rng.choice(values) for key, values in grid.items()}
            frozen = dict(params)
            candidates.append(StrategyCandidate(
                name=f"momentum_exh_v{i}",
                template="momentum_exhaustion",
                params=frozen,
                signal_func=lambda df, p=frozen: MomentumExhaustionTemplate.generate_signals(df, **p),
                stop_loss_atr=frozen["stop_atr_mult"],
                take_profit_atr=frozen["tp_atr_mult"],
                max_holding_bars=frozen["hold_bars"],
                description=f"MomExh: rsi={frozen['rsi_period']}, "
                            f"div={frozen['divergence_lookback']}, hold={frozen['hold_bars']}",
            ))

        return candidates


# ─── Factory ─────────────────────────────────────────────────────

class StrategyFactory:
    """Generate strategy candidates from all templates."""

    def __init__(self, n_per_template: int = 20):
        self.n_per_template = n_per_template

    def generate_all(self) -> list[StrategyCandidate]:
        """Generate candidates from all templates."""
        candidates = []

        # Template 1: Liquidation Reversal (proven edge)
        liq = LiquidationReversalTemplate.generate_candidates(
            n_random=self.n_per_template, include_best=True
        )
        candidates.extend(liq)
        logger.info(f"Liquidation Reversal: {len(liq)} candidates")

        # Template 2: Funding Rate Mean Reversion
        fund = FundingReversionTemplate.generate_candidates(self.n_per_template)
        candidates.extend(fund)
        logger.info(f"Funding Reversion: {len(fund)} candidates")

        # Template 3: Volatility Squeeze
        squeeze = VolSqueezeTemplate.generate_candidates(self.n_per_template)
        candidates.extend(squeeze)
        logger.info(f"Volatility Squeeze: {len(squeeze)} candidates")

        # Template 4: Momentum Exhaustion
        mom = MomentumExhaustionTemplate.generate_candidates(self.n_per_template)
        candidates.extend(mom)
        logger.info(f"Momentum Exhaustion: {len(mom)} candidates")

        logger.info(f"Total candidates: {len(candidates)}")
        return candidates

    def generate_template(self, template_name: str, n: int = 20) -> list[StrategyCandidate]:
        """Generate candidates from a single template."""
        templates = {
            "liquidation_reversal": lambda: LiquidationReversalTemplate.generate_candidates(n, True),
            "funding_reversion": lambda: FundingReversionTemplate.generate_candidates(n),
            "vol_squeeze": lambda: VolSqueezeTemplate.generate_candidates(n),
            "momentum_exhaustion": lambda: MomentumExhaustionTemplate.generate_candidates(n),
        }
        gen = templates.get(template_name)
        if gen is None:
            raise ValueError(f"Unknown template: {template_name}. "
                             f"Available: {list(templates.keys())}")
        return gen()

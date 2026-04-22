from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from src.data.market_data import AssetSpec, MarketType, coerce_market_type


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window, min_periods=max(5, window // 3)).std()
    return (series - mean) / (std + 1e-10)


def _apply_position_logic(
    long_entry: pd.Series,
    short_entry: pd.Series,
    exit_signal: pd.Series,
) -> pd.Series:
    position = 0
    out = pd.Series(0, index=long_entry.index, dtype=int)

    for idx in long_entry.index:
        if position == 0:
            if bool(long_entry.loc[idx]):
                position = 1
            elif bool(short_entry.loc[idx]):
                position = -1
        else:
            if bool(exit_signal.loc[idx]):
                position = 0
            elif position == 1 and bool(short_entry.loc[idx]):
                position = -1
            elif position == -1 and bool(long_entry.loc[idx]):
                position = 1
        out.loc[idx] = position

    return out


@dataclass(frozen=True)
class PrimitiveSlotSpec:
    name: str
    primitive_name: str
    asset_key: str
    config: dict = field(default_factory=dict)
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 4.0
    max_holding_bars: int = 24
    position_size_pct: float = 0.02
    use_vwap: bool = True


class AlphaPrimitive:
    primitive_name: ClassVar[str] = "base"
    DEFAULT_MARKET_CONFIGS: ClassVar[dict[MarketType, dict]] = {}

    def get_market_config(self, market_type: MarketType | str) -> dict:
        market = coerce_market_type(market_type)
        return dict(self.DEFAULT_MARKET_CONFIGS.get(market, {}))

    def resolve_config(self, asset: AssetSpec, overrides: dict | None = None) -> dict:
        config = self.get_market_config(asset.market_type)
        config.update(overrides or {})
        return config

    def generate_signals(
        self,
        df: pd.DataFrame,
        asset: AssetSpec,
        overrides: dict | None = None,
    ) -> pd.Series:
        raise NotImplementedError


class MeanReversionPrimitive(AlphaPrimitive):
    primitive_name = "mean_reversion"
    DEFAULT_MARKET_CONFIGS = {
        MarketType.CRYPTO: {"lookback": 48, "entry_z": 1.7, "exit_z": 0.40, "funding_weight": 0.35},
        MarketType.EQUITIES: {"lookback": 20, "entry_z": 1.25, "exit_z": 0.30, "funding_weight": 0.0},
        MarketType.COMMODITIES: {"lookback": 30, "entry_z": 1.40, "exit_z": 0.35, "funding_weight": 0.0},
        MarketType.INDICES: {"lookback": 24, "entry_z": 1.15, "exit_z": 0.25, "funding_weight": 0.0},
    }

    def generate_signals(self, df: pd.DataFrame, asset: AssetSpec, overrides: dict | None = None) -> pd.Series:
        config = self.resolve_config(asset, overrides)
        lookback = int(config["lookback"])
        funding_weight = float(config.get("funding_weight", 0.0))

        price_z = _rolling_zscore(df["close"], lookback)
        funding = df.get("fund_funding_rate", pd.Series(0.0, index=df.index))
        funding_z = _rolling_zscore(funding.fillna(0.0), lookback)
        composite = price_z + funding_weight * funding_z
        trend_filter = df.get("trend_strength_20_60", pd.Series(0.0, index=df.index)).abs()

        long_entry = (composite < -float(config["entry_z"])) & (trend_filter < 1.5)
        short_entry = (composite > float(config["entry_z"])) & (trend_filter < 1.5)
        exit_signal = composite.abs() <= float(config["exit_z"])
        return _apply_position_logic(long_entry.fillna(False), short_entry.fillna(False), exit_signal.fillna(False))


class MomentumPrimitive(AlphaPrimitive):
    primitive_name = "momentum"
    DEFAULT_MARKET_CONFIGS = {
        MarketType.CRYPTO: {"breakout_lookback": 30, "fast": 20, "slow": 60, "trend_threshold": 0.45, "exit_threshold": 0.10},
        MarketType.EQUITIES: {"breakout_lookback": 55, "fast": 20, "slow": 80, "trend_threshold": 0.35, "exit_threshold": 0.08},
        MarketType.COMMODITIES: {"breakout_lookback": 35, "fast": 18, "slow": 72, "trend_threshold": 0.30, "exit_threshold": 0.06},
        MarketType.INDICES: {"breakout_lookback": 40, "fast": 16, "slow": 64, "trend_threshold": 0.25, "exit_threshold": 0.05},
    }

    def generate_signals(self, df: pd.DataFrame, asset: AssetSpec, overrides: dict | None = None) -> pd.Series:
        config = self.resolve_config(asset, overrides)
        breakout = int(config["breakout_lookback"])
        fast = int(config["fast"])
        slow = int(config["slow"])

        fast_ema = df["close"].ewm(span=fast, adjust=False).mean()
        slow_ema = df["close"].ewm(span=slow, adjust=False).mean()
        trend_strength = (fast_ema - slow_ema) / (df.get("atr_14", pd.Series(1.0, index=df.index)) + 1e-10)
        high_break = df["high"].rolling(breakout, min_periods=max(5, breakout // 3)).max().shift(1)
        low_break = df["low"].rolling(breakout, min_periods=max(5, breakout // 3)).min().shift(1)

        long_entry = (df["close"] > high_break) & (trend_strength > float(config["trend_threshold"]))
        short_entry = (df["close"] < low_break) & (trend_strength < -float(config["trend_threshold"]))
        exit_signal = trend_strength.abs() < float(config["exit_threshold"])
        return _apply_position_logic(long_entry.fillna(False), short_entry.fillna(False), exit_signal.fillna(False))


class VolatilityExpansionPrimitive(AlphaPrimitive):
    primitive_name = "volatility"
    DEFAULT_MARKET_CONFIGS = {
        MarketType.CRYPTO: {"squeeze_window": 20, "squeeze_quantile": 0.25, "expansion_mult": 1.25, "direction_lookback": 12},
        MarketType.EQUITIES: {"squeeze_window": 20, "squeeze_quantile": 0.30, "expansion_mult": 1.15, "direction_lookback": 10},
        MarketType.COMMODITIES: {"squeeze_window": 24, "squeeze_quantile": 0.30, "expansion_mult": 1.20, "direction_lookback": 12},
        MarketType.INDICES: {"squeeze_window": 18, "squeeze_quantile": 0.35, "expansion_mult": 1.10, "direction_lookback": 8},
    }

    def generate_signals(self, df: pd.DataFrame, asset: AssetSpec, overrides: dict | None = None) -> pd.Series:
        config = self.resolve_config(asset, overrides)
        window = int(config["squeeze_window"])
        returns = df["close"].pct_change()
        width = returns.rolling(window, min_periods=max(5, window // 3)).std()
        squeeze_thresh = width.rolling(window, min_periods=max(5, window // 3)).quantile(float(config["squeeze_quantile"]))
        squeeze = width <= squeeze_thresh
        expansion = df.get("vol_ratio_20_100", pd.Series(1.0, index=df.index)) > float(config["expansion_mult"])
        direction = df["close"].pct_change(int(config["direction_lookback"]))

        long_entry = squeeze.shift(1).fillna(False) & expansion & (direction > 0)
        short_entry = squeeze.shift(1).fillna(False) & expansion & (direction < 0)
        exit_signal = (~expansion) | (direction.abs() < 0.0025)
        return _apply_position_logic(long_entry.fillna(False), short_entry.fillna(False), exit_signal.fillna(False))


class LiquidityShockPrimitive(AlphaPrimitive):
    primitive_name = "liquidity_shock"
    DEFAULT_MARKET_CONFIGS = {
        MarketType.CRYPTO: {"volume_spike": 2.0, "shock_return": 0.02, "follow_through": False, "exit_return": 0.004},
        MarketType.EQUITIES: {"volume_spike": 1.8, "shock_return": 0.012, "follow_through": True, "exit_return": 0.003},
        MarketType.COMMODITIES: {"volume_spike": 1.7, "shock_return": 0.015, "follow_through": True, "exit_return": 0.004},
        MarketType.INDICES: {"volume_spike": 1.6, "shock_return": 0.008, "follow_through": True, "exit_return": 0.0025},
    }

    def generate_signals(self, df: pd.DataFrame, asset: AssetSpec, overrides: dict | None = None) -> pd.Series:
        config = self.resolve_config(asset, overrides)
        returns = df["close"].pct_change().fillna(0.0)
        volume_ratio = df["volume"] / (df["volume"].rolling(20, min_periods=5).median() + 1e-10)
        liquidity_score = df.get("liquidity_score", pd.Series(1.0, index=df.index))
        shock = (volume_ratio > float(config["volume_spike"])) & (returns.abs() > float(config["shock_return"]))
        stressed = liquidity_score > 1.2
        follow_through = bool(config.get("follow_through", False))

        if follow_through:
            long_entry = shock & stressed & (returns > 0)
            short_entry = shock & stressed & (returns < 0)
        else:
            long_entry = shock & stressed & (returns < 0)
            short_entry = shock & stressed & (returns > 0)

        exit_signal = returns.abs() < float(config["exit_return"])
        return _apply_position_logic(long_entry.fillna(False), short_entry.fillna(False), exit_signal.fillna(False))


ALPHA_PRIMITIVES = {
    MeanReversionPrimitive.primitive_name: MeanReversionPrimitive(),
    MomentumPrimitive.primitive_name: MomentumPrimitive(),
    VolatilityExpansionPrimitive.primitive_name: VolatilityExpansionPrimitive(),
    LiquidityShockPrimitive.primitive_name: LiquidityShockPrimitive(),
}


def get_alpha_primitive(name: str) -> AlphaPrimitive:
    primitive = ALPHA_PRIMITIVES.get(name)
    if primitive is None:
        raise KeyError(f"Unknown alpha primitive: {name}")
    return primitive
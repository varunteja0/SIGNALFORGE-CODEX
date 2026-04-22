from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.market_data import AssetSpec


@dataclass(frozen=True)
class RegimeState:
    asset: str
    market_type: str
    volatility_regime: str
    trend_regime: str
    liquidity_regime: str
    composite: str
    volatility_score: float
    trend_score: float
    liquidity_score: float
    confidence_score: float
    uncertainty_score: float


class RegimeDetectionEngine:
    """Detect volatility, trend, and liquidity regimes across markets."""

    def __init__(
        self,
        high_vol_threshold: float = 1.25,
        low_vol_threshold: float = 0.75,
        trend_threshold: float = 0.35,
        liquid_threshold: float = 1.20,
        illiquid_threshold: float = 0.80,
    ):
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.trend_threshold = trend_threshold
        self.liquid_threshold = liquid_threshold
        self.illiquid_threshold = illiquid_threshold

    def detect(self, df: pd.DataFrame, asset: AssetSpec) -> RegimeState:
        vol_ratio = float(df.get("vol_ratio_20_100", pd.Series(1.0, index=df.index)).iloc[-1])
        trend_score = float(df.get("trend_strength_20_60", pd.Series(0.0, index=df.index)).iloc[-1])
        liquidity_score = float(df.get("liquidity_score", pd.Series(1.0, index=df.index)).iloc[-1])

        if vol_ratio >= self.high_vol_threshold:
            volatility_regime = "high_volatility"
        elif vol_ratio <= self.low_vol_threshold:
            volatility_regime = "low_volatility"
        else:
            volatility_regime = "normal_volatility"

        if trend_score >= self.trend_threshold:
            trend_regime = "bull_trend"
        elif trend_score <= -self.trend_threshold:
            trend_regime = "bear_trend"
        else:
            trend_regime = "range"

        if liquidity_score >= self.liquid_threshold:
            liquidity_regime = "liquid"
        elif liquidity_score <= self.illiquid_threshold:
            liquidity_regime = "illiquid"
        else:
            liquidity_regime = "normal_liquidity"

        confidence = self._confidence_score(
            vol_ratio=vol_ratio,
            trend_score=trend_score,
            liquidity_score=liquidity_score,
            volatility_regime=volatility_regime,
            trend_regime=trend_regime,
            liquidity_regime=liquidity_regime,
        )

        return RegimeState(
            asset=asset.symbol,
            market_type=asset.market_type.value,
            volatility_regime=volatility_regime,
            trend_regime=trend_regime,
            liquidity_regime=liquidity_regime,
            composite=f"{volatility_regime}|{trend_regime}|{liquidity_regime}",
            volatility_score=vol_ratio,
            trend_score=trend_score,
            liquidity_score=liquidity_score,
            confidence_score=confidence,
            uncertainty_score=float(np.clip(1.0 - confidence, 0.0, 1.0)),
        )

    def detect_universe(
        self,
        datasets: dict[str, pd.DataFrame],
        asset_specs: dict[str, AssetSpec],
    ) -> dict[str, RegimeState]:
        states = {}
        for symbol, asset in asset_specs.items():
            df = datasets.get(symbol)
            if df is None or df.empty:
                continue
            states[symbol] = self.detect(df, asset)
        return states

    def primitive_multiplier(self, primitive_name: str, state: RegimeState) -> float:
        multiplier = 1.0

        if primitive_name == "mean_reversion":
            if state.trend_regime == "range":
                multiplier += 0.20
            else:
                multiplier -= 0.45
            if state.volatility_regime == "high_volatility":
                multiplier += 0.15
        elif primitive_name == "momentum":
            if state.trend_regime in {"bull_trend", "bear_trend"}:
                multiplier += 0.25
            else:
                multiplier -= 0.45
            if state.volatility_regime == "low_volatility":
                multiplier -= 0.10
        elif primitive_name == "volatility":
            if state.volatility_regime == "high_volatility":
                multiplier += 0.25
            elif state.volatility_regime == "low_volatility":
                multiplier -= 0.20
            if state.trend_regime == "range":
                multiplier += 0.05
        elif primitive_name == "liquidity_shock":
            if state.liquidity_regime != "normal_liquidity":
                multiplier += 0.25
            if state.volatility_regime == "high_volatility":
                multiplier += 0.10

        multiplier *= 0.70 + 0.60 * state.confidence_score
        return float(np.clip(multiplier, 0.0, 1.6))

    def _confidence_score(
        self,
        *,
        vol_ratio: float,
        trend_score: float,
        liquidity_score: float,
        volatility_regime: str,
        trend_regime: str,
        liquidity_regime: str,
    ) -> float:
        if volatility_regime == "high_volatility":
            vol_conf = abs(vol_ratio - self.high_vol_threshold) / max(self.high_vol_threshold, 1e-9)
        elif volatility_regime == "low_volatility":
            vol_conf = abs(self.low_vol_threshold - vol_ratio) / max(self.low_vol_threshold, 1e-9)
        else:
            vol_conf = 1.0 - abs(vol_ratio - 1.0) / max(self.high_vol_threshold - self.low_vol_threshold, 1e-9)

        trend_conf = abs(trend_score) / max(self.trend_threshold * 2.0, 1e-9)
        if trend_regime == "range":
            trend_conf = 1.0 - min(trend_conf, 1.0)

        if liquidity_regime == "liquid":
            liquidity_conf = (liquidity_score - self.liquid_threshold) / max(self.liquid_threshold, 1e-9)
        elif liquidity_regime == "illiquid":
            liquidity_conf = (self.illiquid_threshold - liquidity_score) / max(self.illiquid_threshold, 1e-9)
        else:
            liquidity_conf = 1.0 - abs(liquidity_score - 1.0) / max(self.liquid_threshold - self.illiquid_threshold, 1e-9)

        confidence = 0.40 * vol_conf + 0.40 * trend_conf + 0.20 * liquidity_conf
        return float(np.clip(confidence, 0.0, 1.0))
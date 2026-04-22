from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.market_data import MarketType, coerce_market_type


@dataclass
class LiquidityConstraintDecision:
    liquidity_caps: dict[str, float] = field(default_factory=dict)
    requested_position_sizes: dict[str, float] = field(default_factory=dict)
    capped_position_sizes: dict[str, float] = field(default_factory=dict)
    participation_by_strategy: dict[str, float] = field(default_factory=dict)
    market_impact_bps: dict[str, float] = field(default_factory=dict)
    capped_strategies: dict[str, bool] = field(default_factory=dict)


class LiquidityConstraintEngine:
    """Constrain position sizing by average traded volume participation."""

    DEFAULT_MAX_ADV_FRACTION = {
        MarketType.CRYPTO: 0.00010,
        MarketType.EQUITIES: 0.00008,
        MarketType.COMMODITIES: 0.00010,
        MarketType.INDICES: 0.00012,
    }
    DEFAULT_IMPACT_BPS = {
        MarketType.CRYPTO: 12.0,
        MarketType.EQUITIES: 4.0,
        MarketType.COMMODITIES: 6.0,
        MarketType.INDICES: 3.0,
    }

    def __init__(
        self,
        max_adv_fraction_by_market: dict[MarketType, float] | None = None,
        impact_bps_by_market: dict[MarketType, float] | None = None,
        adv_bars: int = 24,
    ):
        self.max_adv_fraction_by_market = max_adv_fraction_by_market or dict(self.DEFAULT_MAX_ADV_FRACTION)
        self.impact_bps_by_market = impact_bps_by_market or dict(self.DEFAULT_IMPACT_BPS)
        self.adv_bars = adv_bars

    def evaluate(
        self,
        bundle,
        engine,
        *,
        capital: float,
    ) -> LiquidityConstraintDecision:
        decision = LiquidityConstraintDecision()
        for slot in engine.slots:
            requested_size = max(0.0, float(slot.position_size_pct))
            caps: list[float] = []
            impacts: list[float] = []
            participations: list[float] = []

            for symbol in slot.allowed_assets:
                frame = bundle.datasets.get(symbol)
                asset = bundle.assets.get(symbol)
                if frame is None or frame.empty or asset is None:
                    continue
                market_type = coerce_market_type(asset.market_type)
                adv = self._average_daily_dollar_volume(frame)
                liquidity_cap_pct = adv * self.max_adv_fraction_by_market[market_type] / max(capital, 1e-9)
                requested_notional = capital * requested_size
                participation = requested_notional / max(adv, 1e-9)
                impact_bps = self.impact_bps_by_market[market_type] * np.sqrt(max(participation, 0.0))
                caps.append(float(liquidity_cap_pct))
                impacts.append(float(impact_bps))
                participations.append(float(participation))

            cap_pct = min(caps) if caps else requested_size
            capped_size = min(requested_size, cap_pct)
            decision.liquidity_caps[slot.name] = float(cap_pct)
            decision.requested_position_sizes[slot.name] = float(requested_size)
            decision.capped_position_sizes[slot.name] = float(capped_size)
            decision.participation_by_strategy[slot.name] = float(np.mean(participations)) if participations else 0.0
            decision.market_impact_bps[slot.name] = float(np.mean(impacts)) if impacts else 0.0
            decision.capped_strategies[slot.name] = bool(capped_size + 1e-12 < requested_size)
        return decision

    def _average_daily_dollar_volume(self, frame: pd.DataFrame) -> float:
        dollar_volume = frame.get("dollar_volume")
        if dollar_volume is None:
            dollar_volume = frame["close"].astype(float) * frame["volume"].astype(float)
        clean = pd.to_numeric(dollar_volume, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return 0.0
        recent = clean.tail(self.adv_bars)
        return float(recent.sum())
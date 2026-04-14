"""
Liquidation Oracle — Main Prediction Engine
=============================================
Combines protocol adapters + cascade simulator to produce actionable
trading signals based on liquidation risk.

Core capabilities:
1. Map all visible leveraged positions across protocols
2. Identify prices where cascades trigger (cliff edges)
3. Score current market's liquidation risk
4. Generate trading signals: SHORT before cascade, BUY at cascade bottom
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.liquidation.protocols import (
    LeveragedPosition,
    ProtocolAdapter,
    AaveV3Adapter,
    CompoundV3Adapter,
    SyntheticPositionGenerator,
)
from src.liquidation.cascade import CascadeSimulator, CascadeResult

logger = logging.getLogger(__name__)


@dataclass
class LiquidationRiskScore:
    """Overall liquidation risk assessment for an asset."""
    asset: str
    current_price: float
    risk_score: float              # 0-100 (100 = extreme danger)
    nearest_cliff_pct: float       # Distance to nearest cascade cliff edge
    total_at_risk_usd: float       # Value at risk within 20% drop
    cascade_severity: str          # low / medium / high / extreme
    expected_amplification: float  # Average cascade amplification
    recommendation: str            # AVOID / CAUTIOUS / OPPORTUNITY / NEUTRAL
    timestamp: float = 0.0

    def __repr__(self):
        return (
            f"LiqRisk({self.asset} score={self.risk_score:.0f}/100 "
            f"cliff={self.nearest_cliff_pct:.1f}% "
            f"at_risk=${self.total_at_risk_usd:,.0f} "
            f"rec={self.recommendation})"
        )


@dataclass
class LiquidationSignal:
    """Actionable trading signal from liquidation analysis."""
    asset: str
    direction: int                 # 1 = long (buy cascade bottom), -1 = short (front-run cascade)
    signal_type: str               # "cascade_short" or "cascade_bounce"
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float              # 0-1
    reasoning: str
    cascade_data: Optional[CascadeResult] = None


class LiquidationOracle:
    """Main engine for liquidation-based trading intelligence."""

    def __init__(
        self,
        adapters: Optional[list[ProtocolAdapter]] = None,
        price_impact_bps: float = 5.0,
        use_synthetic: bool = True,
        synthetic_tvl: float = 5_000_000_000,
    ):
        self.adapters = adapters or []
        self.simulator = CascadeSimulator(
            price_impact_bps_per_million=price_impact_bps
        )
        self.use_synthetic = use_synthetic
        self.synthetic_tvl = synthetic_tvl
        self.synthetic_gen = SyntheticPositionGenerator()

        # Position cache
        self._position_cache: dict[str, list[LeveragedPosition]] = {}
        self._cache_time: dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes

    def fetch_positions(self, asset: str, current_price: float) -> list[LeveragedPosition]:
        """Fetch all leveraged positions for an asset across protocols."""
        cache_key = asset
        if (
            cache_key in self._position_cache
            and time.time() - self._cache_time.get(cache_key, 0) < self._cache_ttl
        ):
            return self._position_cache[cache_key]

        all_positions = []

        # Fetch from real protocols
        for adapter in self.adapters:
            try:
                positions = adapter.fetch_positions(asset)
                all_positions.extend(positions)
                logger.info(
                    f"Fetched {len(positions)} positions from "
                    f"{adapter.get_protocol_name()} ({adapter.get_chain()})"
                )
            except Exception as e:
                logger.error(f"Error fetching from {adapter.get_protocol_name()}: {e}")

        # Use synthetic data for development/testing
        if self.use_synthetic and not all_positions:
            all_positions = self.synthetic_gen.generate(
                asset=asset,
                current_price=current_price,
                n_positions=1000,
                total_tvl_usd=self.synthetic_tvl,
            )
            logger.info(
                f"Generated {len(all_positions)} synthetic positions "
                f"(TVL=${self.synthetic_tvl:,.0f})"
            )

        self._position_cache[cache_key] = all_positions
        self._cache_time[cache_key] = time.time()
        return all_positions

    def assess_risk(self, asset: str, current_price: float) -> LiquidationRiskScore:
        """Comprehensive liquidation risk assessment for an asset."""
        positions = self.fetch_positions(asset, current_price)

        if not positions:
            return LiquidationRiskScore(
                asset=asset, current_price=current_price,
                risk_score=0, nearest_cliff_pct=100,
                total_at_risk_usd=0, cascade_severity="none",
                expected_amplification=1.0,
                recommendation="NEUTRAL",
                timestamp=time.time(),
            )

        # Scan for cascade behavior across drop levels
        scan = self.simulator.scan_trigger_levels(
            positions, current_price, drop_range_pct=(1, 30), steps=60
        )

        # Find cliff edges
        cliffs = self.simulator.find_cliff_edges(positions, current_price)

        # Total value at risk within 20% drop
        at_risk_20 = sum(
            p.collateral_usd for p in positions
            if p.distance_to_liq_pct < 20
        )

        # Nearest cliff
        nearest_cliff_pct = 100.0
        if cliffs:
            nearest_cliff_pct = min(c["trigger_drop_pct"] for c in cliffs)

        # Average amplification for moderate drops (5-15%)
        moderate = scan[
            (scan["trigger_drop_pct"] >= 5) & (scan["trigger_drop_pct"] <= 15)
        ]
        avg_amp = float(moderate["amplification"].mean()) if len(moderate) > 0 else 1.0

        # Max severity
        worst = scan["severity"].iloc[-1] if len(scan) > 0 else "low"

        # Risk score (0-100)
        risk_score = self._compute_risk_score(
            nearest_cliff_pct, at_risk_20, avg_amp, current_price
        )

        # Recommendation
        if risk_score > 70:
            recommendation = "AVOID"  # High liquidation risk – reduce exposure
        elif risk_score > 50:
            recommendation = "CAUTIOUS"
        elif cliffs and nearest_cliff_pct < 10:
            recommendation = "OPPORTUNITY"  # Close to cliff = trading opportunity
        else:
            recommendation = "NEUTRAL"

        return LiquidationRiskScore(
            asset=asset,
            current_price=current_price,
            risk_score=risk_score,
            nearest_cliff_pct=nearest_cliff_pct,
            total_at_risk_usd=at_risk_20,
            cascade_severity=worst,
            expected_amplification=avg_amp,
            recommendation=recommendation,
            timestamp=time.time(),
        )

    def generate_signals(
        self, asset: str, current_price: float
    ) -> list[LiquidationSignal]:
        """Generate trading signals from liquidation analysis."""
        positions = self.fetch_positions(asset, current_price)
        if not positions:
            return []

        signals = []

        # 1a. Cascade short signals from cliff edges
        cliffs = self.simulator.find_cliff_edges(positions, current_price)

        for cliff in cliffs:
            if cliff["trigger_drop_pct"] > 20:
                continue

            cliff_price = cliff["cliff_price"]
            entry_price = cliff_price * 1.02

            # Only if cliff is close enough to be actionable
            if cliff["trigger_drop_pct"] < 15:
                cascade = self.simulator.simulate(
                    positions, current_price, cliff["trigger_drop_pct"]
                )

                signals.append(LiquidationSignal(
                    asset=asset,
                    direction=-1,  # SHORT
                    signal_type="cascade_short",
                    entry_price=entry_price,
                    target_price=cascade.final_price,
                    stop_loss=entry_price * 1.03,  # 3% stop
                    confidence=min(0.9, cliff["total_amplification"] / 5),
                    reasoning=(
                        f"Cascade cliff at {cliff['trigger_drop_pct']:.1f}% drop. "
                        f"Amplification: {cliff['total_amplification']:.1f}x. "
                        f"${cliff['liquidation_volume_usd']:,.0f} at risk."
                    ),
                    cascade_data=cascade,
                ))

        # 1b. Risk-concentration short: heavy liquidation volume at nearby levels
        at_risk_positions = [p for p in positions if p.distance_to_liq_pct < 15]
        at_risk_value = sum(p.collateral_usd for p in at_risk_positions)
        total_value = sum(p.collateral_usd for p in positions)

        if total_value > 0 and at_risk_value / total_value > 0.05:
            # > 5% of total collateral is within 15% of liquidation
            nearest_liq = min(p.distance_to_liq_pct for p in at_risk_positions) if at_risk_positions else 100
            cascade_10 = self.simulator.simulate(positions, current_price, 10)

            signals.append(LiquidationSignal(
                asset=asset,
                direction=-1,
                signal_type="cascade_short",
                entry_price=current_price * (1 - nearest_liq / 200),
                target_price=cascade_10.final_price,
                stop_loss=current_price * 1.03,
                confidence=min(0.85, at_risk_value / total_value * 3),
                reasoning=(
                    f"High liquidation concentration: "
                    f"${at_risk_value/1e6:,.0f}M ({at_risk_value/total_value:.1%}) "
                    f"within 15% of liquidation. "
                    f"10% drop -> {cascade_10.total_drop_pct:.1f}% total "
                    f"({cascade_10.amplification_factor:.2f}x amp)."
                ),
                cascade_data=cascade_10,
            ))

        # 2. Cascade bounce signals (buy at predicted cascade bottom)
        for trigger_pct in [5, 10, 15, 20]:
            cascade = self.simulator.simulate(
                positions, current_price, trigger_pct
            )

            if cascade.amplification_factor > 1.1:
                # Cascade causes overshoot — buy the bottom
                bounce_entry = cascade.final_price * 1.01
                overshoot_pct = cascade.total_drop_pct - trigger_pct
                bounce_target = cascade.final_price * (
                    1 + overshoot_pct / 100 * 0.7
                )

                if bounce_target > bounce_entry * 1.01:
                    signals.append(LiquidationSignal(
                        asset=asset,
                        direction=1,  # LONG
                        signal_type="cascade_bounce",
                        entry_price=bounce_entry,
                        target_price=bounce_target,
                        stop_loss=cascade.final_price * 0.95,
                        confidence=min(0.8, (cascade.amplification_factor - 1) / 4),
                        reasoning=(
                            f"After {trigger_pct}% drop: cascade amplifies to "
                            f"{cascade.total_drop_pct:.1f}% ({cascade.amplification_factor:.1f}x). "
                            f"Forced selling exhausts at ${cascade.final_price:,.0f}. "
                            f"Bounce target: ${bounce_target:,.0f}."
                        ),
                        cascade_data=cascade,
                    ))

        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals

    def _compute_risk_score(
        self,
        nearest_cliff_pct: float,
        at_risk_usd: float,
        avg_amplification: float,
        current_price: float,
    ) -> float:
        """Compute 0-100 risk score."""
        score = 0.0

        # Proximity to cliff (0-40 points)
        if nearest_cliff_pct < 5:
            score += 40
        elif nearest_cliff_pct < 10:
            score += 30
        elif nearest_cliff_pct < 20:
            score += 15

        # Value at risk (0-30 points)
        if at_risk_usd > 1_000_000_000:
            score += 30
        elif at_risk_usd > 500_000_000:
            score += 20
        elif at_risk_usd > 100_000_000:
            score += 10

        # Amplification (0-30 points)
        if avg_amplification > 5:
            score += 30
        elif avg_amplification > 3:
            score += 20
        elif avg_amplification > 1.5:
            score += 10

        return min(100, score)

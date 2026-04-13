"""
Liquidation Oracle — Cascade Simulator
========================================
Models HOW liquidation cascades propagate:

    Price drops 5%
    → $50M in positions get liquidated (forced selling)
    → That selling pushes price down another 3%
    → Another $80M gets liquidated
    → Price drops 4% more
    → ... until no more liquidations trigger

This is the weapon: by mapping ALL positions, we can predict EXACTLY
how deep a cascade will go for any initial price shock.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.liquidation.protocols import LeveragedPosition

logger = logging.getLogger(__name__)


@dataclass
class LiquidationWave:
    """One wave of a liquidation cascade."""
    wave_number: int
    price_before: float
    price_after: float
    price_drop_pct: float
    positions_liquidated: int
    volume_liquidated_usd: float
    cumulative_liquidated_usd: float


@dataclass
class CascadeResult:
    """Full simulation result of a liquidation cascade."""
    asset: str
    initial_price: float
    trigger_drop_pct: float          # Initial price shock
    final_price: float
    total_drop_pct: float            # Total drop including cascade
    amplification_factor: float      # total_drop / initial_drop
    total_liquidated_usd: float
    total_positions_liquidated: int
    waves: list[LiquidationWave] = field(default_factory=list)
    price_levels: list[float] = field(default_factory=list)
    volume_at_levels: list[float] = field(default_factory=list)

    @property
    def is_cascade(self) -> bool:
        """True if the liquidations amplified the initial shock."""
        return self.amplification_factor > 1.2

    @property
    def severity(self) -> str:
        if self.amplification_factor < 1.5:
            return "low"
        elif self.amplification_factor < 3.0:
            return "medium"
        elif self.amplification_factor < 5.0:
            return "high"
        return "extreme"


class CascadeSimulator:
    """Simulates liquidation cascades given a set of leveraged positions.

    The key insight: order book depth determines how much price impact
    each wave of liquidations has. We model this with a price impact
    function calibrated to typical crypto market depth.
    """

    def __init__(
        self,
        price_impact_bps_per_million: float = 5.0,
        max_waves: int = 20,
        liquidation_penalty: float = 0.05,
    ):
        """
        Args:
            price_impact_bps_per_million: How many basis points price moves
                per $1M of selling. BTC ~2-5 bps, altcoins ~10-50 bps.
            max_waves: Maximum cascade depth to simulate.
            liquidation_penalty: Protocol liquidation penalty (5% for Aave).
        """
        self.price_impact_bps_per_million = price_impact_bps_per_million
        self.max_waves = max_waves
        self.liquidation_penalty = liquidation_penalty

    def simulate(
        self,
        positions: list[LeveragedPosition],
        initial_price: float,
        trigger_drop_pct: float,
    ) -> CascadeResult:
        """Simulate a liquidation cascade from an initial price shock.

        Process:
        1. Apply initial price drop
        2. Find positions now below liquidation threshold
        3. Compute forced selling volume
        4. Compute price impact of that selling
        5. Repeat until no more liquidations trigger
        """
        current_price = initial_price * (1 - trigger_drop_pct / 100)
        total_liquidated_usd = 0.0
        total_positions_liquidated = 0
        remaining = list(positions)  # Don't modify original
        waves = []

        for wave_num in range(1, self.max_waves + 1):
            # Find positions that get liquidated at current price
            liquidated = []
            still_alive = []

            for pos in remaining:
                if current_price <= pos.liquidation_price:
                    liquidated.append(pos)
                else:
                    still_alive.append(pos)

            if not liquidated:
                break

            # Volume of forced selling
            wave_volume = sum(p.collateral_usd for p in liquidated)
            wave_volume_net = wave_volume * (1 - self.liquidation_penalty)

            # Square-root price impact model (industry standard):
            # Impact = bps_per_million * sqrt(volume_in_millions)
            # This captures that order books absorb small volume easily
            # but large volume has diminishing depth to hit.
            volume_millions = wave_volume_net / 1_000_000
            impact_bps = self.price_impact_bps_per_million * np.sqrt(volume_millions)
            impact_pct = impact_bps / 10_000  # bps to fraction

            price_before = current_price
            current_price *= (1 - impact_pct)
            current_price = max(current_price, initial_price * 0.01)  # Floor at 99% drop

            total_liquidated_usd += wave_volume
            total_positions_liquidated += len(liquidated)

            waves.append(LiquidationWave(
                wave_number=wave_num,
                price_before=price_before,
                price_after=current_price,
                price_drop_pct=impact_pct,
                positions_liquidated=len(liquidated),
                volume_liquidated_usd=wave_volume,
                cumulative_liquidated_usd=total_liquidated_usd,
            ))

            remaining = still_alive

        total_drop_pct = (initial_price - current_price) / initial_price * 100
        amp = total_drop_pct / trigger_drop_pct if trigger_drop_pct > 0 else 1.0

        result = CascadeResult(
            asset=positions[0].collateral_asset if positions else "unknown",
            initial_price=initial_price,
            trigger_drop_pct=trigger_drop_pct,
            final_price=current_price,
            total_drop_pct=total_drop_pct,
            amplification_factor=amp,
            total_liquidated_usd=total_liquidated_usd,
            total_positions_liquidated=total_positions_liquidated,
            waves=waves,
        )

        return result

    def scan_trigger_levels(
        self,
        positions: list[LeveragedPosition],
        current_price: float,
        drop_range_pct: tuple[float, float] = (1.0, 30.0),
        steps: int = 60,
    ) -> pd.DataFrame:
        """Scan across multiple trigger levels to build a complete cascade map.

        Returns a DataFrame showing: for each % drop, how much liquidation
        volume triggers, cascade amplification, and final price.
        """
        drops = np.linspace(drop_range_pct[0], drop_range_pct[1], steps)
        results = []

        for drop in drops:
            cascade = self.simulate(positions, current_price, drop)
            results.append({
                "trigger_drop_pct": drop,
                "final_drop_pct": cascade.total_drop_pct,
                "amplification": cascade.amplification_factor,
                "liquidated_usd": cascade.total_liquidated_usd,
                "positions_liquidated": cascade.total_positions_liquidated,
                "n_waves": len(cascade.waves),
                "severity": cascade.severity,
                "final_price": cascade.final_price,
            })

        return pd.DataFrame(results)

    def find_cliff_edges(
        self,
        positions: list[LeveragedPosition],
        current_price: float,
    ) -> list[dict]:
        """Find 'cliff edges' — price levels where small additional drops
        trigger disproportionately large cascades.

        These are the most profitable levels to trade around. Position SHORT
        just above the cliff; the cascade does the rest.
        """
        scan = self.scan_trigger_levels(positions, current_price, steps=100)

        cliff_edges = []
        for i in range(1, len(scan)):
            prev_amp = scan.iloc[i - 1]["amplification"]
            curr_amp = scan.iloc[i]["amplification"]

            # Cliff = amplification jumps significantly
            if curr_amp > prev_amp * 1.5 and curr_amp > 2.0:
                trigger_pct = scan.iloc[i]["trigger_drop_pct"]
                cliff_price = current_price * (1 - trigger_pct / 100)

                cliff_edges.append({
                    "trigger_drop_pct": trigger_pct,
                    "cliff_price": cliff_price,
                    "amplification_jump": curr_amp - prev_amp,
                    "total_amplification": curr_amp,
                    "liquidation_volume_usd": scan.iloc[i]["liquidated_usd"],
                    "severity": scan.iloc[i]["severity"],
                })

        return cliff_edges

    def liquidation_heatmap(
        self, positions: list[LeveragedPosition], current_price: float
    ) -> pd.DataFrame:
        """Build a price-level heatmap of liquidation density.

        Shows how much collateral is liquidatable at each price level,
        creating a 'liquidation density' map.
        """
        if not positions:
            return pd.DataFrame()

        liq_prices = [p.liquidation_price for p in positions]
        min_price = max(min(liq_prices), current_price * 0.5)
        max_price = current_price * 1.01

        bins = np.linspace(min_price, max_price, 50)
        volumes = []

        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            mid = (low + high) / 2

            vol = sum(
                p.collateral_usd
                for p in positions
                if low <= p.liquidation_price < high
            )
            count = sum(
                1 for p in positions
                if low <= p.liquidation_price < high
            )

            volumes.append({
                "price_level": mid,
                "price_drop_pct": (current_price - mid) / current_price * 100,
                "liquidation_volume_usd": vol,
                "position_count": count,
            })

        return pd.DataFrame(volumes)

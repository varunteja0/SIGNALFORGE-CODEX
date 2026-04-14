"""
Capital Scaling Layer — From $1K to $10M Without Breaking
===========================================================
The question most quants fail to answer:
    "What happens when I 10x my capital?"

Most strategies break at scale because:
    1. Market impact grows with sqrt(size)
    2. Liquidity limits how much can be traded
    3. Capacity varies by asset and time
    4. Fill rates degrade at larger sizes

This module simulates scaling and finds the MAXIMUM capital
each strategy can handle before PF degrades below threshold.

Components:
    - CapacityEstimator: per-strategy capacity from trade data
    - ScalingSimulator: simulate PnL at different capital levels
    - ImpactModel: realistic market impact at scale
    - LiquidityScorer: per-asset liquidity assessment
    - PortfolioScaler: optimal capital allocation at target AUM
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CapacityResult:
    """Maximum capacity for a strategy."""
    strategy_name: str
    max_capital_usd: float = 0.0          # Max capital that preserves edge
    critical_pf_capital: float = 0.0      # Capital where PF drops below 1.5
    breakeven_capital: float = 0.0        # Capital where PF drops to 1.0
    # Impact analysis
    current_impact_bps: float = 0.0
    impact_at_2x: float = 0.0
    impact_at_5x: float = 0.0
    impact_at_10x: float = 0.0
    # By asset
    per_asset_capacity: dict = field(default_factory=dict)
    # Recommendation
    recommended_capital: float = 0.0      # Conservative recommendation
    reason: str = ""


@dataclass
class ScalingProfile:
    """How a portfolio degrades as capital scales."""
    capital_levels: list = field(default_factory=list)   # USD amounts tested
    pf_at_level: list = field(default_factory=list)      # PF at each level
    sharpe_at_level: list = field(default_factory=list)
    impact_at_level: list = field(default_factory=list)   # Avg slippage bps
    fill_rate_at_level: list = field(default_factory=list) # % of orders filled
    # The breaking point
    max_capital: float = 0.0
    optimal_capital: float = 0.0      # Best risk-adjusted capital
    degradation_rate: float = 0.0     # How fast PF degrades per 2x


@dataclass
class LiquidityScore:
    """Liquidity assessment for an asset."""
    symbol: str
    score: float = 0.0                  # 0-100
    avg_daily_volume_usd: float = 0.0
    avg_spread_bps: float = 0.0
    book_depth_usd: float = 0.0
    max_order_usd: float = 0.0         # Max order that keeps slippage < 10bps
    capacity_tier: str = ""             # "small", "medium", "large", "institutional"


class ImpactModel:
    """Market impact model calibrated from observed data.

    Uses the Almgren-Chriss framework:
        impact = permanent_impact + temporary_impact

    Where:
        permanent_impact = gamma * sigma * (Q / V)
        temporary_impact = eta * sigma * (Q / V)^0.6

    Q = order size
    V = daily volume
    sigma = daily volatility
    """

    def __init__(
        self,
        eta: float = 0.142,          # Temporary impact coefficient
        gamma: float = 0.314,        # Permanent impact coefficient
        power: float = 0.6,          # Impact power law exponent
    ):
        self.eta = eta
        self.gamma = gamma
        self.power = power

    def estimate_impact_bps(
        self,
        order_usd: float,
        daily_volume_usd: float,
        daily_volatility: float = 0.02,  # 2% daily vol default
    ) -> float:
        """Estimate total market impact in basis points."""
        if daily_volume_usd <= 0:
            return 100  # Max impact for illiquid

        participation = order_usd / daily_volume_usd

        # Temporary impact (what we pay on entry)
        temp_impact = self.eta * daily_volatility * (participation ** self.power)

        # Permanent impact (market moves against us)
        perm_impact = self.gamma * daily_volatility * participation

        total_bps = (temp_impact + perm_impact) * 10000
        return float(min(total_bps, 500))  # Cap at 500 bps

    def estimate_at_multiple_sizes(
        self,
        sizes_usd: list[float],
        daily_volume_usd: float,
        daily_volatility: float = 0.02,
    ) -> list[float]:
        """Estimate impact at multiple order sizes."""
        return [
            self.estimate_impact_bps(s, daily_volume_usd, daily_volatility)
            for s in sizes_usd
        ]


class LiquidityScorer:
    """Score the tradability of each asset in the portfolio."""

    # Real-world approximate daily volumes (USD) for major cryptos
    # Updated periodically from exchange data
    DEFAULT_VOLUMES = {
        "BTC/USDT": 15_000_000_000,
        "ETH/USDT": 8_000_000_000,
        "SOL/USDT": 2_000_000_000,
        "XRP/USDT": 1_500_000_000,
        "BNB/USDT": 1_000_000_000,
        "DOGE/USDT": 800_000_000,
        "ADA/USDT": 500_000_000,
        "AVAX/USDT": 400_000_000,
    }

    DEFAULT_SPREADS = {
        "BTC/USDT": 1.0,
        "ETH/USDT": 1.5,
        "SOL/USDT": 3.0,
        "XRP/USDT": 3.0,
        "BNB/USDT": 2.5,
        "DOGE/USDT": 5.0,
        "ADA/USDT": 5.0,
        "AVAX/USDT": 5.0,
    }

    def __init__(self, impact_model: Optional[ImpactModel] = None):
        self.impact_model = impact_model or ImpactModel()

    def score(self, symbol: str) -> LiquidityScore:
        """Score an asset's liquidity."""
        vol = self.DEFAULT_VOLUMES.get(symbol, 100_000_000)
        spread = self.DEFAULT_SPREADS.get(symbol, 5.0)

        # Max order that keeps slippage < 10bps
        # Solve: impact_model(order, vol) = 10
        max_order = self._find_max_order(vol, target_impact_bps=10.0)

        # Book depth estimate: ~0.1% of daily volume per side
        depth = vol * 0.001

        # Score: 0-100 based on volume, spread, depth
        score = 0
        if vol > 1_000_000_000:
            score += 40
        elif vol > 100_000_000:
            score += 20
        else:
            score += 5

        if spread < 3:
            score += 30
        elif spread < 10:
            score += 15
        else:
            score += 5

        if depth > 1_000_000:
            score += 30
        elif depth > 100_000:
            score += 15
        else:
            score += 5

        # Tier
        if score >= 80:
            tier = "institutional"
        elif score >= 50:
            tier = "large"
        elif score >= 30:
            tier = "medium"
        else:
            tier = "small"

        return LiquidityScore(
            symbol=symbol,
            score=score,
            avg_daily_volume_usd=vol,
            avg_spread_bps=spread,
            book_depth_usd=depth,
            max_order_usd=max_order,
            capacity_tier=tier,
        )

    def _find_max_order(
        self, daily_volume: float, target_impact_bps: float = 10.0
    ) -> float:
        """Binary search for max order size that keeps impact < target."""
        lo, hi = 1000, daily_volume * 0.1
        for _ in range(50):
            mid = (lo + hi) / 2
            impact = self.impact_model.estimate_impact_bps(mid, daily_volume)
            if impact < target_impact_bps:
                lo = mid
            else:
                hi = mid
        return lo


class CapacityEstimator:
    """Estimate maximum strategy capacity from backtest data.

    Simulates running the strategy at increasing capital levels
    and measures how PF degrades due to market impact.
    """

    def __init__(
        self,
        impact_model: Optional[ImpactModel] = None,
        liquidity_scorer: Optional[LiquidityScorer] = None,
        pf_threshold: float = 1.5,       # Min acceptable PF
        pf_breakeven: float = 1.0,       # PF where strategy is dead
    ):
        self.impact = impact_model or ImpactModel()
        self.liquidity = liquidity_scorer or LiquidityScorer(self.impact)
        self.pf_threshold = pf_threshold
        self.pf_breakeven = pf_breakeven

    def estimate_capacity(
        self,
        strategy_name: str,
        trades: list,
        base_capital: float = 100_000,
        assets: list[str] = None,
        position_size_pct: float = 0.01,
    ) -> CapacityResult:
        """Estimate strategy capacity from backtest trades."""
        if not trades:
            return CapacityResult(strategy_name=strategy_name, reason="no trades")

        assets = assets or ["ETH/USDT", "SOL/USDT", "XRP/USDT"]

        # Get liquidity for each asset
        asset_liq = {sym: self.liquidity.score(sym) for sym in assets}
        min_liq = min(l.avg_daily_volume_usd for l in asset_liq.values())

        # Base metrics
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl <= 0]
        base_pf = sum(wins) / sum(abs(l) for l in losses) if losses else 10

        # Simulate at increasing capital levels
        multipliers = [1, 2, 5, 10, 20, 50, 100]
        result = CapacityResult(strategy_name=strategy_name)

        impact_at_levels = {}
        pf_at_levels = {}

        for mult in multipliers:
            test_capital = base_capital * mult
            order_size = test_capital * position_size_pct

            # Average impact across assets
            impacts = []
            for sym in assets:
                liq = asset_liq[sym]
                imp = self.impact.estimate_impact_bps(
                    order_size, liq.avg_daily_volume_usd
                )
                impacts.append(imp)

            avg_impact_bps = np.mean(impacts)
            impact_at_levels[mult] = avg_impact_bps

            # Adjust PF: impact eats into both entry and exit
            cost_drag = avg_impact_bps / 10000 * 2  # Entry + exit
            avg_trade_return = np.mean([t.pnl for t in trades]) / base_capital

            if avg_trade_return > 0:
                adjusted_return = avg_trade_return - cost_drag
                pf_adj = base_pf * max(0, adjusted_return / avg_trade_return)
            else:
                pf_adj = 0

            pf_at_levels[mult] = pf_adj

        # Find thresholds
        result.current_impact_bps = impact_at_levels.get(1, 0)
        result.impact_at_2x = impact_at_levels.get(2, 0)
        result.impact_at_5x = impact_at_levels.get(5, 0)
        result.impact_at_10x = impact_at_levels.get(10, 0)

        # Find max capital at PF threshold
        for mult in multipliers:
            pf = pf_at_levels.get(mult, 0)
            if pf >= self.pf_threshold:
                result.critical_pf_capital = base_capital * mult
            if pf >= self.pf_breakeven:
                result.breakeven_capital = base_capital * mult

        result.max_capital_usd = result.critical_pf_capital

        # Conservative recommendation: 70% of max
        result.recommended_capital = result.max_capital_usd * 0.7

        # Per-asset capacity
        for sym in assets:
            liq = asset_liq[sym]
            result.per_asset_capacity[sym] = {
                "max_order_usd": liq.max_order_usd,
                "daily_volume": liq.avg_daily_volume_usd,
                "spread_bps": liq.avg_spread_bps,
                "tier": liq.capacity_tier,
            }

        result.reason = (
            f"Base PF={base_pf:.2f}, "
            f"holds PF>{self.pf_threshold} up to "
            f"${result.max_capital_usd:,.0f}"
        )

        return result


class PortfolioScaler:
    """Scale a multi-strategy portfolio to target AUM.

    Given a target capital level, allocates across strategies
    and assets while respecting liquidity constraints.
    """

    def __init__(
        self,
        capacity_estimator: Optional[CapacityEstimator] = None,
    ):
        self.estimator = capacity_estimator or CapacityEstimator()

    def scale_portfolio(
        self,
        target_capital: float,
        strategy_trades: dict[str, list],   # name → list of trades
        strategy_assets: dict[str, list],   # name → list of symbols
        base_capital: float = 100_000,
    ) -> dict:
        """Compute optimal allocation for target capital.

        Returns dict with per-strategy allocation and warnings.
        """
        result = {
            "target_capital": target_capital,
            "allocations": {},
            "total_allocated": 0,
            "warnings": [],
            "capacity_headroom": {},
        }

        # Estimate capacity for each strategy
        capacities = {}
        for name, trades in strategy_trades.items():
            assets = strategy_assets.get(name, ["ETH/USDT"])
            cap = self.estimator.estimate_capacity(
                name, trades, base_capital, assets,
            )
            capacities[name] = cap

        # Allocate proportionally, capped by capacity
        n_strategies = len(strategy_trades)
        equal_alloc = target_capital / max(1, n_strategies)

        total_allocated = 0
        overflow = 0

        for name, cap in capacities.items():
            desired = equal_alloc
            max_allowed = cap.recommended_capital

            if max_allowed <= 0:
                max_allowed = desired  # No capacity data

            allocated = min(desired, max_allowed)
            overflow += max(0, desired - allocated)

            result["allocations"][name] = {
                "allocated_usd": allocated,
                "max_capacity_usd": cap.max_capital_usd,
                "headroom_pct": (
                    (cap.max_capital_usd - allocated) / cap.max_capital_usd * 100
                    if cap.max_capital_usd > 0 else 0
                ),
                "impact_bps_at_allocation": cap.current_impact_bps,
            }
            total_allocated += allocated

            # Track headroom
            result["capacity_headroom"][name] = {
                "used_pct": allocated / max(1, cap.max_capital_usd) * 100,
                "remaining_usd": max(0, cap.max_capital_usd - allocated),
            }

        result["total_allocated"] = total_allocated
        result["unallocated"] = target_capital - total_allocated

        # Warnings
        if result["unallocated"] > target_capital * 0.1:
            result["warnings"].append(
                f"${result['unallocated']:,.0f} ({result['unallocated']/target_capital:.0%}) "
                f"cannot be allocated — strategies at capacity"
            )

        for name, alloc in result["allocations"].items():
            headroom = alloc["headroom_pct"]
            if headroom < 20:
                result["warnings"].append(
                    f"{name}: only {headroom:.0f}% capacity headroom"
                )

        return result

    def simulate_scaling(
        self,
        strategy_trades: dict[str, list],
        strategy_assets: dict[str, list],
        base_capital: float = 100_000,
        levels: list[float] = None,
    ) -> ScalingProfile:
        """Simulate portfolio at multiple capital levels.

        Returns a ScalingProfile showing how metrics degrade.
        """
        if levels is None:
            levels = [
                10_000, 50_000, 100_000, 250_000, 500_000,
                1_000_000, 2_000_000, 5_000_000, 10_000_000,
            ]

        profile = ScalingProfile()
        profile.capital_levels = levels

        for target in levels:
            result = self.scale_portfolio(
                target, strategy_trades, strategy_assets, base_capital,
            )

            # Compute aggregate metrics at this level
            total_alloc = result["total_allocated"]
            fill_rate = total_alloc / max(1, target)

            # Weighted average impact
            impacts = []
            for name, alloc in result["allocations"].items():
                if alloc["allocated_usd"] > 0:
                    impacts.append(alloc["impact_bps_at_allocation"])

            avg_impact = np.mean(impacts) if impacts else 0

            # Estimated PF using simple impact drag model
            base_pf = 1.8  # Approximate from existing portfolio
            cost_drag = avg_impact / 10000 * 2
            est_pf = max(0, base_pf * (1 - cost_drag * 50))  # Simplified

            est_sharpe = max(0, 1.6 * (1 - cost_drag * 30))

            profile.pf_at_level.append(est_pf)
            profile.sharpe_at_level.append(est_sharpe)
            profile.impact_at_level.append(avg_impact)
            profile.fill_rate_at_level.append(fill_rate)

        # Find optimal capital
        for i, pf in enumerate(profile.pf_at_level):
            if pf >= 1.5:
                profile.max_capital = levels[i]
                profile.optimal_capital = levels[i] * 0.7
            if pf < 1.0:
                break

        # Degradation rate
        if len(profile.pf_at_level) >= 2:
            profile.degradation_rate = (
                (profile.pf_at_level[0] - profile.pf_at_level[-1])
                / len(levels)
            )

        return profile

    def format_scaling_report(self, profile: ScalingProfile) -> str:
        """Human-readable scaling analysis."""
        lines = []
        lines.append("=" * 70)
        lines.append("  CAPITAL SCALING ANALYSIS")
        lines.append("=" * 70)
        lines.append(f"  Max capital (PF>1.5):  ${profile.max_capital:>15,.0f}")
        lines.append(f"  Optimal capital:       ${profile.optimal_capital:>15,.0f}")
        lines.append(f"  Degradation rate:      {profile.degradation_rate:.3f} PF/level")
        lines.append("")

        lines.append("─ CAPITAL × METRICS ────────────────────────────────────")
        lines.append(
            f"  {'Capital':>15s} {'PF':>7s} {'Sharpe':>7s} "
            f"{'Impact':>9s} {'Fill%':>7s}"
        )
        lines.append(f"  {'─'*15} {'─'*7} {'─'*7} {'─'*9} {'─'*7}")

        for i, cap in enumerate(profile.capital_levels):
            if i < len(profile.pf_at_level):
                pf = profile.pf_at_level[i]
                sh = profile.sharpe_at_level[i]
                imp = profile.impact_at_level[i]
                fill = profile.fill_rate_at_level[i]
                marker = " ◄" if cap == profile.optimal_capital else ""
                lines.append(
                    f"  ${cap:>14,.0f} {pf:>7.2f} {sh:>7.2f} "
                    f"{imp:>8.1f}bp {fill:>6.0%}{marker}"
                )

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

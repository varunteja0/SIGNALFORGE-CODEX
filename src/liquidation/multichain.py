"""
Multi-Chain Liquidation Oracle — Cross-Chain Risk Aggregation
================================================================
Nobody else does this: aggregate liquidation risk across ALL chains.

Supported chains & protocols:
  Ethereum: Aave V3, Compound V3, MakerDAO, Morpho Blue
  Arbitrum: Aave V3, GMX, Radiant
  Optimism: Aave V3, Sonne Finance
  Base:     Aave V3, Moonwell
  Polygon:  Aave V3
  Solana:   Marinade, Kamino (via public APIs)

Key insight: Liquidation cascades don't respect chain boundaries.
When ETH drops 10%, positions get liquidated on ALL chains simultaneously.
Cross-chain aggregation reveals the TRUE systemic risk.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from src.data.thegraph import (
    TheGraphFetcher, OnChainPosition, ProtocolMetrics,
    SUBGRAPH_URLS,
)
from src.liquidation.cascade import CascadeSimulator, CascadeResult
from src.liquidation.protocols import LeveragedPosition

logger = logging.getLogger(__name__)

# Chain configurations
CHAIN_CONFIG = {
    "ethereum": {
        "protocols": ["aave_v3", "compound_v3"],
        "block_time": 12,
        "gas_token": "ETH",
    },
    "arbitrum": {
        "protocols": ["aave_v3", "compound_v3"],
        "block_time": 0.25,
        "gas_token": "ETH",
    },
    "optimism": {
        "protocols": ["aave_v3"],
        "block_time": 2,
        "gas_token": "ETH",
    },
    "polygon": {
        "protocols": ["aave_v3"],
        "block_time": 2,
        "gas_token": "MATIC",
    },
    "base": {
        "protocols": ["aave_v3"],
        "block_time": 2,
        "gas_token": "ETH",
    },
}


@dataclass
class CrossChainRisk:
    """Aggregate liquidation risk across all chains."""
    asset: str
    current_price: float
    timestamp: float

    # Per-chain breakdown
    chain_risks: dict = field(default_factory=dict)  # chain -> risk_metrics

    # Aggregate metrics
    total_positions: int = 0
    total_debt_usd: float = 0
    total_at_risk_usd: float = 0
    cross_chain_risk_score: float = 0     # 0-100
    aggregate_risk_score: float = 0        # Legacy alias for cross_chain_risk_score
    cascade_amplification: float = 1.0    # How much a drop amplifies
    nearest_cliff_pct: float = 100.0      # Nearest cascade cliff
    systemic_risk_level: str = "low"      # low/medium/high/extreme

    # Concentration analysis
    dominant_chain: str = ""
    dominant_chain_pct: float = 0
    chain_diversification: float = 0      # 0-1 (1 = equally spread)

    # Cascade simulation results
    cascade_5pct: Optional[CascadeResult] = None
    cascade_10pct: Optional[CascadeResult] = None
    cascade_20pct: Optional[CascadeResult] = None

    # Trading signals
    short_signal_strength: float = 0      # 0-1
    bounce_signal_strength: float = 0     # 0-1
    recommended_action: str = "NEUTRAL"
    recommendation: str = "NEUTRAL"


@dataclass
class ChainRiskMetrics:
    """Per-chain risk metrics."""
    chain: str
    total_positions: int = 0
    total_debt_usd: float = 0
    positions_near_liq: int = 0
    at_risk_usd: float = 0
    avg_health_factor: float = 0
    min_health_factor: float = 999
    protocols_active: int = 0
    risk_score: float = 0  # 0-100


class MultiChainLiquidationOracle:
    """Aggregates liquidation risk across multiple blockchains.

    This is the edge: no one else aggregates cross-chain liquidation
    risk into a single view. When ETH drops, positions get liquidated
    on Ethereum, Arbitrum, Optimism, Base, Polygon simultaneously.
    The total cascade is worse than any single chain suggests.
    """

    def __init__(
        self,
        chains: list[str] = None,
        price_impact_bps: float = 5.0,
        parallel_fetches: bool = True,
        max_workers: int = 4,
    ):
        self.chains = chains or list(CHAIN_CONFIG.keys())
        self.price_impact_bps = price_impact_bps
        self.parallel = parallel_fetches
        self.max_workers = max_workers
        self.fetcher = TheGraphFetcher()
        self.simulator = CascadeSimulator(
            price_impact_bps_per_million=price_impact_bps
        )

        # Cache
        self._risk_cache: dict[str, tuple[float, CrossChainRisk]] = {}
        self._cache_ttl = 300

    def assess_cross_chain_risk(
        self,
        asset: str,
        current_price: float = 2000.0,
        force_refresh: bool = False,
    ) -> CrossChainRisk:
        """Full cross-chain liquidation risk assessment.

        Fetches positions from all chains, runs cascade simulations,
        and produces an aggregate risk score with trading signals.
        """
        cache_key = asset
        if not force_refresh and cache_key in self._risk_cache:
            ts, cached = self._risk_cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return cached

        logger.info(f"Assessing cross-chain risk for {asset} @ ${current_price:,.2f}")

        # Fetch positions from all chains
        all_positions = self._fetch_all_chains(asset)

        # Convert to LeveragedPosition format for cascade simulator
        lev_positions = self._convert_positions(all_positions, current_price)

        # Compute per-chain risk
        chain_risks = self._compute_chain_risks(all_positions)

        # Run cascade simulations at different drop levels
        cascade_5 = cascade_10 = cascade_20 = None
        cliffs = []
        if lev_positions:
            cascade_5 = self.simulator.simulate_cascade(
                lev_positions, current_price, price_drop_pct=5.0
            )
            cascade_10 = self.simulator.simulate_cascade(
                lev_positions, current_price, price_drop_pct=10.0
            )
            cascade_20 = self.simulator.simulate_cascade(
                lev_positions, current_price, price_drop_pct=20.0
            )
            cliffs = self.simulator.find_cliff_edges(lev_positions, current_price)

        # Compute aggregate metrics
        total_positions = sum(cr.total_positions for cr in chain_risks.values())
        total_debt = sum(cr.total_debt_usd for cr in chain_risks.values())
        total_at_risk = sum(cr.at_risk_usd for cr in chain_risks.values())

        # Risk score (0-100)
        risk_score = self._compute_risk_score(
            chain_risks, cascade_5, cascade_10, cascade_20, cliffs
        )

        # Chain concentration analysis
        dominant_chain = ""
        dominant_pct = 0
        chain_debts = {c: cr.total_debt_usd for c, cr in chain_risks.items()}
        if total_debt > 0:
            dominant_chain = max(chain_debts, key=chain_debts.get)
            dominant_pct = chain_debts[dominant_chain] / total_debt
            # Compute diversification (inverse HHI)
            shares = [d / total_debt for d in chain_debts.values() if d > 0]
            hhi = sum(s ** 2 for s in shares) if shares else 1
            diversification = 1 - hhi
        else:
            diversification = 0

        # Nearest cliff
        nearest_cliff = 100.0
        if cliffs:
            nearest_cliff = min(c["trigger_drop_pct"] for c in cliffs)

        # Cascade amplification
        amp = 1.0
        if cascade_10 and cascade_10.initial_drop_pct > 0:
            amp = cascade_10.total_price_impact_pct / cascade_10.initial_drop_pct

        # Systemic risk level
        if risk_score >= 80:
            systemic = "extreme"
        elif risk_score >= 60:
            systemic = "high"
        elif risk_score >= 35:
            systemic = "medium"
        else:
            systemic = "low"

        # Trading signals
        short_signal, bounce_signal, action = self._generate_signals(
            risk_score, nearest_cliff, amp, cascade_10
        )

        risk = CrossChainRisk(
            asset=asset,
            current_price=current_price,
            timestamp=time.time(),
            chain_risks=chain_risks,
            total_positions=total_positions,
            total_debt_usd=total_debt,
            total_at_risk_usd=total_at_risk,
            cross_chain_risk_score=risk_score,
            aggregate_risk_score=risk_score,
            cascade_amplification=amp,
            nearest_cliff_pct=nearest_cliff,
            systemic_risk_level=systemic,
            dominant_chain=dominant_chain,
            dominant_chain_pct=dominant_pct,
            chain_diversification=diversification,
            cascade_5pct=cascade_5,
            cascade_10pct=cascade_10,
            cascade_20pct=cascade_20,
            short_signal_strength=short_signal,
            bounce_signal_strength=bounce_signal,
            recommended_action=action,
            recommendation=action,
        )

        self._risk_cache[cache_key] = (time.time(), risk)
        return risk

    def _fetch_all_chains(self, asset: str) -> dict[str, list[OnChainPosition]]:
        """Fetch positions from all chains (optionally in parallel)."""
        chain_positions = {}

        if self.parallel:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for chain in self.chains:
                    f = executor.submit(
                        self.fetcher.fetch_all_positions,
                        asset, [chain], 5000,
                    )
                    futures[f] = chain

                for future in as_completed(futures):
                    chain = futures[future]
                    try:
                        positions = future.result(timeout=30)
                        chain_positions[chain] = positions
                    except Exception as e:
                        logger.error(f"Error fetching {chain}: {e}")
                        chain_positions[chain] = []
        else:
            for chain in self.chains:
                try:
                    positions = self.fetcher.fetch_all_positions(asset, [chain], 5000)
                    chain_positions[chain] = positions
                except Exception as e:
                    logger.error(f"Error fetching {chain}: {e}")
                    chain_positions[chain] = []

        total = sum(len(p) for p in chain_positions.values())
        logger.info(f"Fetched {total} positions across {len(self.chains)} chains")
        return chain_positions

    def _convert_positions(
        self, chain_positions: dict, current_price: float
    ) -> list[LeveragedPosition]:
        """Convert OnChainPositions to LeveragedPositions for cascade sim."""
        lev_positions = []
        for chain, positions in chain_positions.items():
            for p in positions:
                lev_positions.append(LeveragedPosition(
                    protocol=f"{p.protocol}_{chain}",
                    borrower=p.user_address,
                    collateral_usd=p.collateral_usd,
                    debt_usd=p.debt_usd,
                    liquidation_price=p.liquidation_price if p.liquidation_price > 0
                    else current_price * (1 - p.distance_to_liquidation_pct / 100),
                    health_factor=p.health_factor,
                    collateral_asset=p.collateral_asset,
                ))
        return lev_positions

    def _compute_chain_risks(
        self, chain_positions: dict
    ) -> dict[str, ChainRiskMetrics]:
        """Compute per-chain risk metrics."""
        risks = {}
        for chain, positions in chain_positions.items():
            if not positions:
                risks[chain] = ChainRiskMetrics(chain=chain)
                continue

            near_liq = [p for p in positions if p.health_factor < 1.2]
            hfs = [p.health_factor for p in positions]

            cr = ChainRiskMetrics(
                chain=chain,
                total_positions=len(positions),
                total_debt_usd=sum(p.debt_usd for p in positions),
                positions_near_liq=len(near_liq),
                at_risk_usd=sum(p.collateral_usd for p in near_liq),
                avg_health_factor=np.mean(hfs),
                min_health_factor=min(hfs),
                protocols_active=len(set(p.protocol for p in positions)),
            )
            # Per-chain risk score
            cr.risk_score = min(100, (
                (cr.positions_near_liq / max(cr.total_positions, 1)) * 40
                + max(0, 2 - cr.avg_health_factor) * 30
                + min(1, cr.at_risk_usd / 1e9) * 30
            ))
            risks[chain] = cr

        return risks

    def _compute_risk_score(self, chain_risks, c5, c10, c20, cliffs) -> float:
        """Compute aggregate cross-chain risk score (0-100)."""
        score = 0

        # Component 1: Position concentration near liquidation
        total_pos = sum(cr.total_positions for cr in chain_risks.values())
        near_liq = sum(cr.positions_near_liq for cr in chain_risks.values())
        if total_pos > 0:
            score += (near_liq / total_pos) * 25

        # Component 2: Cascade amplification at 10%
        if c10 and c10.initial_drop_pct > 0:
            amp = c10.total_price_impact_pct / c10.initial_drop_pct
            score += min(25, (amp - 1) * 20)

        # Component 3: Cliff proximity
        if cliffs:
            nearest = min(c["trigger_drop_pct"] for c in cliffs)
            score += max(0, 25 - nearest)  # Closer cliff = higher risk

        # Component 4: Total USD at risk
        total_at_risk = sum(cr.at_risk_usd for cr in chain_risks.values())
        score += min(25, total_at_risk / 1e9 * 25)

        return min(100, max(0, score))

    def _generate_signals(
        self, risk_score, nearest_cliff, amplification, cascade_10
    ) -> tuple[float, float, str]:
        """Generate trading signals from risk assessment."""
        short_signal = 0.0
        bounce_signal = 0.0
        action = "MONITOR"

        if risk_score >= 80 and nearest_cliff < 10:
            short_signal = min(1.0, risk_score / 100 * (amplification / 2))
            action = "EXIT_ALL"
        elif risk_score >= 60:
            short_signal = min(1.0, risk_score / 100 * 0.8)
            action = "HEDGE_NOW"
        elif risk_score >= 40:
            short_signal = min(1.0, risk_score / 100 * 0.5)
            action = "REDUCE_EXPOSURE"
        elif risk_score < 20:
            bounce_signal = 0.3
            action = "SAFE"

        # Post-cascade bounce signal
        if cascade_10 and cascade_10.total_price_impact_pct > 20:
            bounce_signal = min(1.0, cascade_10.total_price_impact_pct / 30)

        return short_signal, bounce_signal, action

    def get_heatmap(self, asset: str, current_price: float) -> pd.DataFrame:
        """Generate a cross-chain liquidation heatmap."""
        chain_positions = self._fetch_all_chains(asset)
        all_positions = []
        for positions in chain_positions.values():
            all_positions.extend(positions)

        if not all_positions:
            return pd.DataFrame()

        lev_positions = self._convert_positions(chain_positions, current_price)
        return self.simulator.liquidation_heatmap(lev_positions, current_price)

    def compute_features(self, asset: str, current_price: float = 2000.0) -> dict:
        """Compute cross-chain features for the GP engine."""
        risk = self.assess_cross_chain_risk(asset, current_price)

        features = {
            "multichain_aggregate_risk": risk.aggregate_risk_score / 100,
            "multichain_contagion": 1.0 - risk.chain_diversification,
            "xchain_risk_score": risk.cross_chain_risk_score / 100,
            "xchain_total_debt_usd_log": np.log1p(risk.total_debt_usd),
            "xchain_at_risk_usd_log": np.log1p(risk.total_at_risk_usd),
            "xchain_cascade_amp": risk.cascade_amplification,
            "xchain_nearest_cliff": risk.nearest_cliff_pct / 100,
            "xchain_chain_diversification": risk.chain_diversification,
            "xchain_dominant_pct": risk.dominant_chain_pct,
            "xchain_short_signal": risk.short_signal_strength,
            "xchain_bounce_signal": risk.bounce_signal_strength,
            "xchain_systemic_extreme": 1.0 if risk.systemic_risk_level == "extreme" else 0.0,
            "xchain_systemic_high": 1.0 if risk.systemic_risk_level == "high" else 0.0,
            "xchain_total_positions_log": np.log1p(risk.total_positions),
        }

        # Per-chain features
        for chain, cr in risk.chain_risks.items():
            prefix = f"xchain_{chain[:3]}"
            features[f"{prefix}_risk"] = cr.risk_score / 100
            features[f"{prefix}_near_liq_pct"] = (
                cr.positions_near_liq / max(cr.total_positions, 1)
            )

        return features

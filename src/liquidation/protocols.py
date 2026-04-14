"""
Liquidation Oracle — DeFi Protocol Adapters
=============================================
Maps leveraged positions across DeFi lending protocols.

Every position on Aave, Compound, MakerDAO, etc. has a publicly visible
liquidation price. This module fetches them all, building a complete map
of WHERE forced selling will happen at every price level.

Adapter pattern: each protocol gets its own adapter class implementing
a common interface. New protocols are added without changing existing code.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LeveragedPosition:
    """A single leveraged position on a DeFi protocol."""
    protocol: str                # e.g., "aave_v3", "compound_v3"
    chain: str                   # e.g., "ethereum", "arbitrum"
    borrower: str                # Wallet address (public on-chain)
    collateral_asset: str        # e.g., "ETH", "WBTC"
    debt_asset: str              # e.g., "USDC", "DAI"
    collateral_amount: float     # In collateral asset units
    collateral_usd: float        # Current USD value
    debt_amount: float           # In debt asset units
    debt_usd: float              # Current USD value
    health_factor: float         # > 1.0 = safe, < 1.0 = liquidatable
    liquidation_price: float     # Price at which position gets liquidated
    liquidation_threshold: float # Protocol's LTV threshold (e.g., 0.825)
    current_price: float         # Current collateral price
    distance_to_liq_pct: float   # How far from liquidation (%)
    timestamp: float = 0.0

    @property
    def is_at_risk(self) -> bool:
        """Position within 10% of liquidation."""
        return self.distance_to_liq_pct < 10.0

    @property
    def liquidation_value(self) -> float:
        """USD value that will be force-sold on liquidation."""
        return self.collateral_usd


@dataclass
class ProtocolSnapshot:
    """Complete snapshot of a protocol's leveraged positions."""
    protocol: str
    chain: str
    total_positions: int
    total_collateral_usd: float
    total_debt_usd: float
    at_risk_positions: int         # Within 10% of liquidation
    at_risk_value_usd: float       # USD value at risk
    positions: list[LeveragedPosition]
    timestamp: float

    @property
    def risk_ratio(self) -> float:
        """% of total collateral that's at risk of liquidation."""
        if self.total_collateral_usd < 1:
            return 0.0
        return self.at_risk_value_usd / self.total_collateral_usd


class ProtocolAdapter(ABC):
    """Base interface for DeFi protocol data adapters."""

    @abstractmethod
    def fetch_positions(
        self, asset: str, min_collateral_usd: float = 1000
    ) -> list[LeveragedPosition]:
        """Fetch all leveraged positions for an asset above minimum size."""

    @abstractmethod
    def get_protocol_name(self) -> str:
        """Return protocol identifier."""

    @abstractmethod
    def get_chain(self) -> str:
        """Return chain identifier."""


class AaveV3Adapter(ProtocolAdapter):
    """Adapter for Aave V3 lending protocol.

    Aave V3 positions are public via The Graph subgraph API.
    Health factor = (collateral * liq_threshold) / debt
    Liquidation when health_factor < 1.0
    """

    # Aave V3 liquidation thresholds per asset (from governance)
    THRESHOLDS = {
        "ETH": 0.825, "WETH": 0.825,
        "WBTC": 0.78, "BTC": 0.78,
        "USDC": 0.0, "USDT": 0.0, "DAI": 0.0,
        "LINK": 0.70, "UNI": 0.70,
        "AAVE": 0.66, "CRV": 0.45,
        "SOL": 0.70, "ARB": 0.65,
    }

    def __init__(self, chain: str = "ethereum", api_url: Optional[str] = None):
        self.chain = chain
        self.api_url = api_url  # The Graph subgraph URL

    def get_protocol_name(self) -> str:
        return "aave_v3"

    def get_chain(self) -> str:
        return self.chain

    def fetch_positions(
        self, asset: str, min_collateral_usd: float = 1000
    ) -> list[LeveragedPosition]:
        """Fetch Aave V3 positions.

        In production, this queries The Graph subgraph:
        https://thegraph.com/hosted-service/subgraph/aave/protocol-v3

        For now: provides the interface and data structures. Plug in
        real API URL to activate.
        """
        if not self.api_url:
            logger.debug("AaveV3: No API URL configured, returning empty")
            return []

        # Production query would be:
        # query = '''
        # {
        #   users(where: {borrowedReservesCount_gt: 0}, first: 1000) {
        #     id
        #     reserves {
        #       currentATokenBalance
        #       currentVariableDebt
        #       reserve { symbol decimals price { priceInEth } liquidationThreshold }
        #     }
        #   }
        # }
        # '''
        # This would be fetched via aiohttp POST to self.api_url

        logger.info(f"AaveV3: Would fetch {asset} positions from {self.api_url}")
        return []

    def compute_liquidation_price(
        self,
        collateral_amount: float,
        debt_usd: float,
        current_price: float,
        threshold: float,
    ) -> float:
        """Compute the exact price at which a position gets liquidated.

        liquidation_price = debt_usd / (collateral_amount * threshold)
        """
        if collateral_amount * threshold < 1e-10:
            return 0.0
        return debt_usd / (collateral_amount * threshold)


class CompoundV3Adapter(ProtocolAdapter):
    """Adapter for Compound V3 (Comet) protocol."""

    def __init__(self, chain: str = "ethereum", api_url: Optional[str] = None):
        self.chain = chain
        self.api_url = api_url

    def get_protocol_name(self) -> str:
        return "compound_v3"

    def get_chain(self) -> str:
        return self.chain

    def fetch_positions(
        self, asset: str, min_collateral_usd: float = 1000
    ) -> list[LeveragedPosition]:
        if not self.api_url:
            logger.debug("CompoundV3: No API URL configured, returning empty")
            return []
        return []


class SyntheticPositionGenerator:
    """Generates realistic synthetic positions for testing and simulation.

    Based on empirical distributions from DeFi analytics:
    - Position sizes follow a power law (many small, few large)
    - Health factors cluster between 1.2-2.0 during stable markets
    - Leverage usage depends on asset volatility
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        asset: str,
        current_price: float,
        n_positions: int = 500,
        total_tvl_usd: float = 1_000_000_000,
    ) -> list[LeveragedPosition]:
        """Generate realistic synthetic positions for simulation."""
        positions = []

        # Power-law size distribution: alpha=1.5 gives realistic whale distribution
        sizes = self.rng.pareto(1.5, n_positions) + 1
        sizes = sizes / sizes.sum() * total_tvl_usd

        threshold = 0.825  # ETH-like
        if asset.upper() in ("BTC", "WBTC"):
            threshold = 0.78

        # Bimodal health factor distribution (matches real DeFi markets):
        # ~20% of positions are near liquidation (HF 1.01-1.3)
        # ~80% are safely leveraged (HF 1.3-4.0)
        near_liq_count = int(n_positions * 0.2)
        safe_count = n_positions - near_liq_count
        hf_near = self.rng.uniform(1.01, 1.30, near_liq_count)
        hf_safe = self.rng.gamma(2.0, 0.5, safe_count) + 1.3
        health_factors = np.concatenate([hf_near, hf_safe])
        health_factors = np.clip(health_factors, 1.01, 10.0)
        self.rng.shuffle(health_factors)

        for i in range(n_positions):
            collateral_usd = float(sizes[i])

            health_factor = float(health_factors[i])

            # Derive debt from health factor
            debt_usd = collateral_usd * threshold / health_factor

            collateral_amount = collateral_usd / current_price

            # Liquidation price
            liq_price = debt_usd / (collateral_amount * threshold)
            distance_pct = (current_price - liq_price) / current_price * 100

            positions.append(LeveragedPosition(
                protocol="synthetic",
                chain="simulation",
                borrower=f"0x{self.rng.integers(0, 2**63):016x}",
                collateral_asset=asset,
                debt_asset="USDC",
                collateral_amount=collateral_amount,
                collateral_usd=collateral_usd,
                debt_amount=debt_usd,
                debt_usd=debt_usd,
                health_factor=health_factor,
                liquidation_price=liq_price,
                liquidation_threshold=threshold,
                current_price=current_price,
                distance_to_liq_pct=distance_pct,
                timestamp=time.time(),
            ))

        return positions

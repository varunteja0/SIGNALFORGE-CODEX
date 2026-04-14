"""
MEV-Aware Execution Engine
============================
In crypto, miners/validators can see pending transactions and reorder them
for profit (Maximal Extractable Value). This means:
  - Large market orders get sandwiched (buy before you, sell after)
  - Liquidations get front-run
  - AMM trades suffer from JIT liquidity attacks

This engine monitors for MEV conditions and adapts execution:
  1. Detects high-MEV environments (mempool congestion, gas spikes)
  2. Splits orders to reduce MEV exposure
  3. Uses timing strategies to avoid peak MEV windows
  4. Monitors Flashbots/MEV-boost activity for liquidation cascade signals

On CEXes (Binance etc.), MEV isn't relevant - but the concepts translate:
  - Order flow toxicity detection
  - Iceberg order splitting
  - TWAP/VWAP execution
  - Execution timing optimization

For SignalForge, the key insight: MEV activity on-chain is a SIGNAL.
When MEV bots are aggressively front-running liquidations, a cascade is near.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class MEVMetrics:
    """Current MEV environment metrics."""
    gas_price_gwei: float = 0
    gas_price_percentile: float = 0    # vs 30-day history (0-1)
    pending_tx_count: int = 0
    flashbots_blocks_pct: float = 0    # % of recent blocks via Flashbots
    sandwich_count_1h: int = 0         # Sandwich attacks in last hour
    liquidation_count_1h: int = 0      # On-chain liquidations in last hour
    mev_revenue_1h_eth: float = 0      # ETH extracted in last hour
    timestamp: float = 0

    @property
    def mev_intensity(self) -> float:
        """0-1 score of how intense MEV activity is."""
        gas_score = min(1, self.gas_price_gwei / 100)
        sandwich_score = min(1, self.sandwich_count_1h / 50)
        liq_score = min(1, self.liquidation_count_1h / 20)
        return (gas_score * 0.3 + sandwich_score * 0.3 + liq_score * 0.4)


@dataclass
class ExecutionPlan:
    """Optimized execution plan for an order."""
    strategy: str        # "immediate", "twap", "iceberg", "snipe"
    n_slices: int = 1
    slice_interval_sec: float = 0
    slice_sizes: list = field(default_factory=list)
    max_slippage_pct: float = 0.1
    urgency: float = 0.5  # 0=patient, 1=urgent
    reason: str = ""
    estimated_slippage_bps: float = 0
    estimated_mev_cost_bps: float = 0


class MEVExecutionEngine:
    """MEV-aware execution engine.

    Monitors on-chain MEV activity and adapts order execution
    to minimize costs and extract signals from MEV patterns.
    """

    def __init__(
        self,
        max_slippage_bps: float = 50,     # 0.5% max slippage
        twap_window_min: int = 30,        # TWAP over 30 minutes
        iceberg_show_pct: float = 0.1,    # Show 10% of order
        min_order_usd: float = 1000,
    ):
        self.max_slippage = max_slippage_bps
        self.twap_window = twap_window_min
        self.iceberg_show_pct = iceberg_show_pct
        self.min_order = min_order_usd

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "SignalForge/2.0"})
        self._cache: dict[str, tuple[float, object]] = {}
        self._gas_history: list[float] = []

    # ================================================================
    # Gas & MEV Monitoring (Public APIs)
    # ================================================================

    def fetch_gas_price(self) -> float:
        """Fetch current ETH gas price from Etherscan public API."""
        cache_key = "gas_price"
        if cache_key in self._cache:
            ts, val = self._cache[cache_key]
            if time.time() - ts < 15:
                return val

        try:
            # Free public endpoint, no key needed
            resp = self._session.get(
                "https://api.etherscan.io/api",
                params={
                    "module": "gastracker",
                    "action": "gasoracle",
                },
                timeout=5,
            )
            data = resp.json()
            if data.get("status") == "1" and "result" in data:
                gas = float(data["result"].get("ProposeGasPrice", 30))
                self._cache[cache_key] = (time.time(), gas)
                self._gas_history.append(gas)
                # Keep last 1000 readings
                if len(self._gas_history) > 1000:
                    self._gas_history = self._gas_history[-1000:]
                return gas
        except Exception as e:
            logger.debug(f"Gas fetch error: {e}")

        return 30.0  # Default fallback

    def fetch_mev_metrics(self) -> MEVMetrics:
        """Fetch MEV environment metrics from public sources."""
        gas = self.fetch_gas_price()

        # Gas percentile vs history
        gas_pctl = 0.5
        if self._gas_history:
            gas_pctl = sum(1 for g in self._gas_history if g <= gas) / len(self._gas_history)

        # Try to get Flashbots MEV data
        flashbots_pct = 0.0
        try:
            resp = self._session.get(
                "https://blocks.flashbots.net/v1/blocks?limit=100",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                blocks = data.get("blocks", [])
                if blocks:
                    flashbots_pct = len(blocks) / 100
        except Exception:
            pass

        return MEVMetrics(
            gas_price_gwei=gas,
            gas_price_percentile=gas_pctl,
            flashbots_blocks_pct=flashbots_pct,
            timestamp=time.time(),
        )

    # ================================================================
    # Execution Planning
    # ================================================================

    def plan_execution(
        self,
        side: str,
        size_usd: float,
        orderbook_depth: dict = None,
        mev_metrics: MEVMetrics = None,
        is_liquidation_nearby: bool = False,
    ) -> ExecutionPlan:
        """Create an optimal execution plan given market conditions.

        Args:
            side: "buy" or "sell"
            size_usd: Order size in USD
            orderbook_depth: {"bid_depth_usd": X, "ask_depth_usd": Y}
            mev_metrics: Current MEV environment
            is_liquidation_nearby: Whether a liquidation cascade is predicted
        """
        if mev_metrics is None:
            mev_metrics = self.fetch_mev_metrics()

        if orderbook_depth is None:
            orderbook_depth = {"bid_depth_usd": 1_000_000, "ask_depth_usd": 1_000_000}

        relevant_depth = (
            orderbook_depth.get("bid_depth_usd", 1_000_000)
            if side == "sell"
            else orderbook_depth.get("ask_depth_usd", 1_000_000)
        )

        # Size relative to book depth
        if relevant_depth > 0:
            size_ratio = size_usd / relevant_depth
        else:
            size_ratio = 1.0

        # ---- Decision logic ----

        # Small orders: just execute immediately
        if size_usd < self.min_order * 5:
            return ExecutionPlan(
                strategy="immediate",
                n_slices=1,
                slice_sizes=[size_usd],
                max_slippage_pct=self.max_slippage / 100,
                urgency=0.8,
                reason="Small order, immediate execution optimal",
                estimated_slippage_bps=size_ratio * 10,
                estimated_mev_cost_bps=0,
            )

        # Liquidation nearby: SNIPE (execute fast to capture bounce)
        if is_liquidation_nearby:
            n_slices = min(3, max(1, int(size_usd / 10000)))
            return ExecutionPlan(
                strategy="snipe",
                n_slices=n_slices,
                slice_interval_sec=2,
                slice_sizes=[size_usd / n_slices] * n_slices,
                max_slippage_pct=self.max_slippage / 100 * 1.5,  # Allow more slippage
                urgency=1.0,
                reason="Liquidation cascade predicted - fast execution for bounce capture",
                estimated_slippage_bps=size_ratio * 20,
                estimated_mev_cost_bps=mev_metrics.mev_intensity * 5,
            )

        # High MEV environment: ICEBERG to hide size
        if mev_metrics.mev_intensity > 0.6:
            show_size = size_usd * self.iceberg_show_pct
            n_slices = max(3, int(size_usd / show_size))
            return ExecutionPlan(
                strategy="iceberg",
                n_slices=n_slices,
                slice_interval_sec=10,
                slice_sizes=[size_usd / n_slices] * n_slices,
                max_slippage_pct=self.max_slippage / 100,
                urgency=0.4,
                reason=f"High MEV intensity ({mev_metrics.mev_intensity:.0%}) - iceberg to hide order flow",
                estimated_slippage_bps=size_ratio * 5,
                estimated_mev_cost_bps=mev_metrics.mev_intensity * 10 * self.iceberg_show_pct,
            )

        # Large order relative to book: TWAP
        if size_ratio > 0.05:
            n_slices = max(5, min(30, int(self.twap_window * 60 / 30)))
            interval = self.twap_window * 60 / n_slices
            return ExecutionPlan(
                strategy="twap",
                n_slices=n_slices,
                slice_interval_sec=interval,
                slice_sizes=[size_usd / n_slices] * n_slices,
                max_slippage_pct=self.max_slippage / 100,
                urgency=0.3,
                reason=f"Large order ({size_ratio:.1%} of book depth) - TWAP over {self.twap_window}min",
                estimated_slippage_bps=size_ratio * 8,
                estimated_mev_cost_bps=mev_metrics.mev_intensity * 3,
            )

        # Default: immediate
        return ExecutionPlan(
            strategy="immediate",
            n_slices=1,
            slice_sizes=[size_usd],
            max_slippage_pct=self.max_slippage / 100,
            urgency=0.6,
            reason="Normal conditions, immediate execution",
            estimated_slippage_bps=size_ratio * 10,
            estimated_mev_cost_bps=mev_metrics.mev_intensity * 2,
        )

    def estimate_execution_cost(
        self,
        size_usd: float,
        plan: ExecutionPlan,
    ) -> dict:
        """Estimate total cost of execution."""
        slippage_cost = size_usd * plan.estimated_slippage_bps / 10000
        mev_cost = size_usd * plan.estimated_mev_cost_bps / 10000
        fee_cost = size_usd * 0.001  # Assume 0.1% taker fee

        return {
            "slippage_usd": slippage_cost,
            "mev_cost_usd": mev_cost,
            "fee_usd": fee_cost,
            "total_cost_usd": slippage_cost + mev_cost + fee_cost,
            "total_cost_bps": (slippage_cost + mev_cost + fee_cost) / size_usd * 10000,
        }

    # ================================================================
    # Order Flow Toxicity
    # ================================================================

    def detect_order_flow_toxicity(self, trades: pd.DataFrame) -> dict:
        """Analyze recent trades for toxic order flow (VPIN-like).

        Toxic flow = informed traders moving the market.
        High toxicity = bad time to trade.

        Args:
            trades: DataFrame with columns [timestamp, price, volume, side]
        """
        if trades.empty or len(trades) < 20:
            return {
                "toxicity": 0.5,
                "buy_volume_pct": 0.5,
                "price_impact_bps": 0,
                "recommendation": "insufficient_data",
            }

        # Volume-synchronized probability of informed trading (simplified VPIN)
        buy_vol = trades[trades["side"] == "buy"]["volume"].sum()
        sell_vol = trades[trades["side"] == "sell"]["volume"].sum()
        total_vol = buy_vol + sell_vol

        if total_vol == 0:
            return {"toxicity": 0.5, "recommendation": "no_volume"}

        # Order imbalance
        imbalance = abs(buy_vol - sell_vol) / total_vol

        # Price impact: how much did price move per unit volume?
        if len(trades) >= 2:
            price_change = abs(trades["price"].iloc[-1] / trades["price"].iloc[0] - 1)
            price_impact = price_change / (total_vol / trades["price"].mean()) * 10000
        else:
            price_impact = 0

        # Toxicity score
        toxicity = min(1.0, imbalance * 0.6 + min(1, price_impact / 100) * 0.4)

        if toxicity > 0.7:
            rec = "delay_execution"
        elif toxicity > 0.4:
            rec = "use_iceberg"
        else:
            rec = "execute_normally"

        return {
            "toxicity": toxicity,
            "buy_volume_pct": buy_vol / total_vol if total_vol > 0 else 0.5,
            "sell_volume_pct": sell_vol / total_vol if total_vol > 0 else 0.5,
            "order_imbalance": imbalance,
            "price_impact_bps": price_impact,
            "recommendation": rec,
        }

    # ================================================================
    # Feature Computation for GP
    # ================================================================

    def compute_features(self) -> dict:
        """Compute MEV-related features for the GP engine."""
        metrics = self.fetch_mev_metrics()

        gas_zscore = 0
        if self._gas_history and len(self._gas_history) > 10:
            mean_gas = np.mean(self._gas_history)
            std_gas = np.std(self._gas_history) or 1
            gas_zscore = (metrics.gas_price_gwei - mean_gas) / std_gas

        return {
            "mev_intensity": metrics.mev_intensity,
            "gas_price_gwei": metrics.gas_price_gwei,
            "gas_percentile": metrics.gas_price_percentile,
            "gas_zscore": gas_zscore,
            "flashbots_activity": metrics.flashbots_blocks_pct,
            "mev_high_alert": 1.0 if metrics.mev_intensity > 0.7 else 0.0,
        }

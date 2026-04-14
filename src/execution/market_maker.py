"""
Synthetic Market Maker Engine
================================
On thin order-book altcoins, spreads are wide and liquidity is poor.
Most fund strategies avoid these because execution costs eat alpha.

But what if your 120+ feature model KNOWS price direction? Then you can:
  1. Quote tight spreads WITH directional bias (informed quoting)
  2. Earn the spread while the edge tilts in your favor
  3. Pull quotes when regime detector says "danger"
  4. Layer orders to create synthetic depth

This is Citadel-style market making: not neutral, but informed.

Key features:
  - Inventory-aware quoting (Avellaneda-Stoikov model)
  - Signal-informed spread skewing
  - Regime-adaptive quote widths
  - Inventory limits and auto-hedging
  - Performance tracking (PnL decomposition: spread vs inventory)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """A two-sided quote."""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    mid_price: float
    spread_bps: float
    skew_bps: float
    timestamp: float = 0
    reason: str = ""


@dataclass
class MMInventory:
    """Market maker inventory state."""
    position: float = 0          # Current position (positive = long)
    avg_entry: float = 0         # Average entry price
    max_position: float = 1.0    # Max allowed position
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    spread_pnl: float = 0       # PnL from spread capture
    inventory_pnl: float = 0    # PnL from inventory directional moves
    n_fills: int = 0
    n_quotes: int = 0


@dataclass
class MMConfig:
    """Market maker configuration."""
    base_spread_bps: float = 20    # Base spread in bps
    min_spread_bps: float = 5      # Floor spread
    max_spread_bps: float = 100    # Max spread in high-vol
    max_position: float = 10000    # Max position in USD
    order_size_usd: float = 500    # Default order size
    gamma: float = 0.1             # Risk aversion (Avellaneda-Stoikov)
    kappa: float = 1.5             # Order arrival rate parameter
    signal_skew_multiplier: float = 1.0  # How much signal affects spread skew
    inventory_skew_bps: float = 5  # BPS skew per unit inventory
    regime_vol_multiplier: float = 2.0   # Widen spread in high vol regimes
    quote_refresh_sec: float = 5


class SyntheticMarketMaker:
    """Informed synthetic market maker using Avellaneda-Stoikov framework.

    Uses the GP signal model to skew quotes toward expected price direction,
    capturing spread while maintaining directional edge.
    """

    def __init__(self, config: MMConfig = None):
        self.config = config or MMConfig()
        self.inventory = MMInventory(max_position=self.config.max_position)

        # State
        self._last_mid: float = 0
        self._vol_estimate: float = 0.001  # Running vol estimate
        self._quote_history: list[Quote] = []
        self._fill_history: list[dict] = []

    # ================================================================
    # Avellaneda-Stoikov Optimal Quoting
    # ================================================================

    def compute_optimal_spread(
        self,
        mid_price: float,
        volatility: float,
        time_horizon: float = 1.0,
    ) -> float:
        """Avellaneda-Stoikov optimal spread.

        The theoretical optimal spread balances:
          - Earning the spread (wider = more per trade)
          - Getting filled (narrower = more fills)
          - Inventory risk (wider when holding large inventory)

        Formula: spread = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)
        """
        gamma = self.config.gamma
        kappa = self.config.kappa
        sigma = volatility

        # Optimal spread from A-S model
        optimal_spread = gamma * sigma**2 * time_horizon + (2 / gamma) * np.log(1 + gamma / kappa)

        # Convert to bps
        spread_bps = optimal_spread / mid_price * 10000

        # Apply floor and ceiling
        spread_bps = max(self.config.min_spread_bps, min(self.config.max_spread_bps, spread_bps))

        return spread_bps

    def compute_inventory_skew(self) -> float:
        """How much to skew quotes based on current inventory.

        When we're long, we want to sell more than buy:
          - Make ask slightly cheaper (tighter)
          - Make bid slightly wider
        This naturally reduces inventory toward zero.
        """
        if self.inventory.max_position == 0:
            return 0

        # Inventory ratio: -1 (max short) to +1 (max long)
        inv_ratio = self.inventory.position / self.inventory.max_position

        # Skew in bps: positive = skew toward selling
        return inv_ratio * self.config.inventory_skew_bps

    def generate_quote(
        self,
        mid_price: float,
        signal_score: float = 0,     # -1 to +1, GP model signal
        volatility: float = 0.001,   # Current realized vol
        regime: str = "normal",      # From regime detector
        orderbook_imbalance: float = 0,  # -1 to +1, bid/ask ratio
    ) -> Quote:
        """Generate an informed two-sided quote.

        Args:
            mid_price: Current mid price
            signal_score: GP model signal (-1 bearish, +1 bullish)
            volatility: Estimated volatility (1-period returns std)
            regime: Current market regime
            orderbook_imbalance: Book imbalance (-1=all bids, +1=all asks)
        """
        self._vol_estimate = 0.9 * self._vol_estimate + 0.1 * volatility

        # 1. Base spread from A-S model
        spread_bps = self.compute_optimal_spread(mid_price, volatility)

        # 2. Regime adjustment
        if regime in ("crisis", "high_vol"):
            spread_bps *= self.config.regime_vol_multiplier
        elif regime == "trending":
            spread_bps *= 1.3  # Slightly wider in trends

        # 3. Inventory skew
        inv_skew = self.compute_inventory_skew()

        # 4. Signal skew: if bullish, make bid tighter (more willing to buy)
        signal_skew = signal_score * self.config.signal_skew_multiplier * 5  # bps

        # 5. Order book imbalance skew
        book_skew = -orderbook_imbalance * 2  # bps

        total_skew = inv_skew + signal_skew + book_skew

        # Compute bid/ask
        half_spread = spread_bps / 2 / 10000 * mid_price
        skew_adj = total_skew / 10000 * mid_price

        bid = mid_price - half_spread + skew_adj
        ask = mid_price + half_spread + skew_adj

        # Size: reduce when inventory is large
        inv_ratio = abs(self.inventory.position) / self.inventory.max_position if self.inventory.max_position > 0 else 0
        base_size = self.config.order_size_usd / mid_price
        bid_size = base_size * (1 - inv_ratio * 0.5) if self.inventory.position > 0 else base_size
        ask_size = base_size * (1 - inv_ratio * 0.5) if self.inventory.position < 0 else base_size

        quote = Quote(
            bid_price=bid,
            ask_price=ask,
            bid_size=max(0, bid_size),
            ask_size=max(0, ask_size),
            mid_price=mid_price,
            spread_bps=spread_bps,
            skew_bps=total_skew,
            timestamp=time.time(),
            reason=f"regime={regime} signal={signal_score:.2f} inv={self.inventory.position:.2f}",
        )

        self._quote_history.append(quote)
        self.inventory.n_quotes += 1
        self._last_mid = mid_price

        return quote

    # ================================================================
    # Fill Processing
    # ================================================================

    def process_fill(self, side: str, price: float, size: float):
        """Process a quote fill."""
        usd_value = price * size

        if side == "buy":
            new_position = self.inventory.position + size
            if abs(new_position) > self.inventory.max_position / price:
                logger.warning("Fill would exceed max position, adjusting")
                size = max(0, self.inventory.max_position / price - abs(self.inventory.position))
                new_position = self.inventory.position + size

            # Update avg entry
            if self.inventory.position >= 0:
                total = self.inventory.position * self.inventory.avg_entry + size * price
                self.inventory.avg_entry = total / (self.inventory.position + size) if (self.inventory.position + size) > 0 else price
            else:
                # Closing short position - realize PnL
                close_size = min(size, abs(self.inventory.position))
                self.inventory.realized_pnl += close_size * (self.inventory.avg_entry - price)

            self.inventory.position = new_position

        elif side == "sell":
            new_position = self.inventory.position - size

            if self.inventory.position > 0:
                close_size = min(size, self.inventory.position)
                self.inventory.realized_pnl += close_size * (price - self.inventory.avg_entry)

            if new_position < 0 and self.inventory.position >= 0:
                self.inventory.avg_entry = price

            self.inventory.position = new_position

        # Track spread PnL
        if self._last_mid > 0:
            if side == "buy":
                self.inventory.spread_pnl += (self._last_mid - price) * size
            else:
                self.inventory.spread_pnl += (price - self._last_mid) * size

        self.inventory.n_fills += 1
        self._fill_history.append({
            "side": side,
            "price": price,
            "size": size,
            "usd": usd_value,
            "position_after": self.inventory.position,
            "timestamp": time.time(),
        })

    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized PnL at current price."""
        if self.inventory.position != 0 and self.inventory.avg_entry > 0:
            self.inventory.unrealized_pnl = (
                self.inventory.position * (current_price - self.inventory.avg_entry)
            )
            self.inventory.inventory_pnl = self.inventory.unrealized_pnl

    # ================================================================
    # Performance Analytics
    # ================================================================

    def get_performance(self, current_price: float = None) -> dict:
        """Get market making performance metrics."""
        if current_price:
            self.update_unrealized_pnl(current_price)

        total_pnl = self.inventory.realized_pnl + self.inventory.unrealized_pnl

        # Fill rate
        fill_rate = self.inventory.n_fills / max(1, self.inventory.n_quotes)

        # Average spread captured
        avg_spread = 0
        if self._quote_history:
            avg_spread = np.mean([q.spread_bps for q in self._quote_history[-100:]])

        return {
            "total_pnl": total_pnl,
            "realized_pnl": self.inventory.realized_pnl,
            "unrealized_pnl": self.inventory.unrealized_pnl,
            "spread_pnl": self.inventory.spread_pnl,
            "inventory_pnl": self.inventory.inventory_pnl,
            "position": self.inventory.position,
            "n_fills": self.inventory.n_fills,
            "n_quotes": self.inventory.n_quotes,
            "fill_rate": fill_rate,
            "avg_spread_bps": avg_spread,
        }

    def should_quote(self, regime: str = "normal") -> bool:
        """Whether we should be quoting at all.

        Pull quotes in crisis regimes or when inventory is maxed out.
        """
        if regime in ("crisis",):
            return False

        inv_ratio = abs(self.inventory.position) / max(1, self.inventory.max_position)
        if inv_ratio > 0.95:
            return False

        return True

    # ================================================================
    # Feature Computation for GP
    # ================================================================

    def compute_features(self, current_price: float = 0) -> dict:
        """Compute market-making features for the GP engine."""
        if current_price:
            self.update_unrealized_pnl(current_price)

        inv_ratio = self.inventory.position / max(1, self.inventory.max_position)

        recent_spreads = [q.spread_bps for q in self._quote_history[-50:]] if self._quote_history else [0]
        recent_skews = [q.skew_bps for q in self._quote_history[-50:]] if self._quote_history else [0]

        return {
            "mm_inventory_ratio": inv_ratio,
            "mm_spread_avg": np.mean(recent_spreads),
            "mm_spread_trend": (np.mean(recent_spreads[-10:]) - np.mean(recent_spreads)) if len(recent_spreads) > 10 else 0,
            "mm_skew_avg": np.mean(recent_skews),
            "mm_fill_rate": self.inventory.n_fills / max(1, self.inventory.n_quotes),
            "mm_pnl_total": self.inventory.realized_pnl + self.inventory.unrealized_pnl,
            "mm_is_active": 1.0 if self.inventory.n_quotes > 0 else 0.0,
        }

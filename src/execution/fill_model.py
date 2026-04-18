"""
Execution fill model
=====================

A lightweight fill simulator bolted onto the backtester's next-bar entry.
Captures the execution effects that move a paper-PnL away from a live PnL:

- **Partial fills** when order size exceeds the top-of-book liquidity
  available in a single bar.
- **Maker vs. taker** regimes: taker pays the crossed spread + fee;
  maker earns the rebate (or pays the lower maker fee) but takes
  queue-position risk — modeled as a fill probability.
- **Per-venue fee schedules** with default tables for Binance, Bybit,
  and OKX perps (configurable).

Scope
-----
Deterministic, stateless, vectorised where possible. Not a
microstructure simulator — it's a better mark than ``price * (1+slip)``
without pretending to model the LOB. Use when you need a quick
capacity / slippage reality check on a candidate strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Fee schedules
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FeeSchedule:
    """Per-venue fee model (all values are fractions, not percent)."""

    maker_bps: float
    taker_bps: float
    maker_rebate_bps: float = 0.0  # set >0 for rebate venues

    @property
    def maker(self) -> float:
        # Rebate is income (negative fee).
        return (self.maker_bps - self.maker_rebate_bps) / 1e4

    @property
    def taker(self) -> float:
        return self.taker_bps / 1e4


# 2025-2026 published retail tier schedules; override per-account as needed.
VENUES: dict[str, FeeSchedule] = {
    "binance_perp":  FeeSchedule(maker_bps=2.0, taker_bps=5.0),
    "bybit_perp":    FeeSchedule(maker_bps=2.0, taker_bps=5.5),
    "okx_perp":      FeeSchedule(maker_bps=2.0, taker_bps=5.0),
    # Conservative default — use when the operator hasn't specified.
    "default":       FeeSchedule(maker_bps=2.0, taker_bps=5.0),
}


# --------------------------------------------------------------------------
# Order types
# --------------------------------------------------------------------------
class OrderKind(str, Enum):
    MARKET = "market"    # taker — always fills, pays spread
    LIMIT = "limit"      # maker — fills probabilistically


@dataclass(frozen=True)
class Order:
    ts: pd.Timestamp
    side: int            # +1 buy, -1 sell
    qty: float           # units of base asset (positive)
    kind: OrderKind = OrderKind.MARKET
    limit_price: float | None = None


@dataclass
class Fill:
    ts: pd.Timestamp
    side: int
    qty: float           # quantity filled this event
    price: float         # average execution price after slippage
    fee: float           # USD fee paid (positive) or rebate (negative)
    is_partial: bool = False
    remaining: float = 0.0


@dataclass
class FillResult:
    order: Order
    fills: list[Fill] = field(default_factory=list)

    @property
    def filled_qty(self) -> float:
        return sum(f.qty for f in self.fills)

    @property
    def unfilled_qty(self) -> float:
        return max(0.0, self.order.qty - self.filled_qty)

    @property
    def avg_price(self) -> float:
        q = self.filled_qty
        if q <= 0:
            return 0.0
        return sum(f.qty * f.price for f in self.fills) / q

    @property
    def total_fee(self) -> float:
        return sum(f.fee for f in self.fills)

    @property
    def is_fully_filled(self) -> bool:
        return self.unfilled_qty <= 1e-9


# --------------------------------------------------------------------------
# Core model
# --------------------------------------------------------------------------
@dataclass
class FillModel:
    """Per-bar fill simulator.

    Parameters
    ----------
    venue :
        Key into :data:`VENUES`. Pass a custom :class:`FeeSchedule` via
        ``fee_schedule`` for non-default accounts.
    participation_cap :
        Fraction of bar's traded volume available to *this* order.
        A large single order cannot consume more than this, the rest
        becomes a partial fill (or rolls onto subsequent bars if the
        caller chooses).
    spread_bps :
        Half-spread paid by takers, in basis points. Represents the
        cost of crossing; maker orders don't pay this.
    impact_k :
        Square-root impact coefficient. Market impact on a fill is
        ``price * impact_k * sigma * sqrt(qty_usd / bar_volume_usd)``.
    maker_fill_prob_base :
        Probability a resting maker order fills in a single bar when
        price does NOT trade through the limit. Queue-position proxy.
    rng_seed :
        Controls the stochastic maker fills. Same seed → same results.
    """

    venue: str = "default"
    fee_schedule: FeeSchedule | None = None
    participation_cap: float = 0.01
    spread_bps: float = 2.0
    impact_k: float = 1.0
    maker_fill_prob_base: float = 0.35
    rng_seed: int = 1

    def __post_init__(self) -> None:
        self.fee_schedule = self.fee_schedule or VENUES.get(self.venue, VENUES["default"])
        self._rng = np.random.default_rng(self.rng_seed)

    # ----- helpers --------------------------------------------------------
    @staticmethod
    def _bar_volume_usd(bar: pd.Series, price: float) -> float:
        vol = float(bar.get("volume", 0.0) or 0.0)
        return max(1.0, vol * price)

    def _impact_price(self, qty_usd: float, bar: pd.Series, price: float) -> float:
        """Return the adverse impact in price units (always positive)."""
        vol_usd = self._bar_volume_usd(bar, price)
        # Daily-sigma proxy — we don't have it inside a single bar, so use
        # ATR-ratio as a stand-in. 1.0 when calm, >1 when vol spikes.
        sigma_proxy = float(bar.get("atr_ratio", 1.0) or 1.0)
        sigma_proxy = max(1.0, sigma_proxy)
        frac = qty_usd / vol_usd
        return price * self.impact_k * (sigma_proxy - 1.0 + 1.0) * np.sqrt(max(frac, 0.0)) * 0.01

    # ----- fills ----------------------------------------------------------
    def fill_market(self, order: Order, bar: pd.Series) -> FillResult:
        """Execute a market (taker) order against a single bar."""
        if order.side not in (-1, 1):
            raise ValueError("order.side must be ±1")
        if order.qty <= 0:
            raise ValueError("order.qty must be positive")

        mid = float(bar.get("open", bar.get("close", 0.0)))
        if mid <= 0:
            raise ValueError("bar has no usable price")

        # Taker crosses the spread.
        cross = mid * self.spread_bps / 1e4
        # Impact based on total order notional.
        order_usd = order.qty * mid
        impact = self._impact_price(order_usd, bar, mid)

        # Available liquidity this bar.
        vol_usd = self._bar_volume_usd(bar, mid)
        max_fillable_usd = vol_usd * self.participation_cap
        fill_usd = min(order_usd, max_fillable_usd)
        fill_qty = fill_usd / mid

        exec_price = mid + order.side * (cross + impact)
        assert self.fee_schedule is not None
        fee = fill_usd * self.fee_schedule.taker

        fill = Fill(
            ts=order.ts,
            side=order.side,
            qty=fill_qty,
            price=exec_price,
            fee=fee,
            is_partial=fill_qty < order.qty - 1e-9,
            remaining=order.qty - fill_qty,
        )
        return FillResult(order=order, fills=[fill])

    def fill_limit(self, order: Order, bar: pd.Series) -> FillResult:
        """Execute a limit (maker) order against a single bar.

        Fills when:
        - The bar's range trades through the limit (guaranteed fill at
          the limit price), OR
        - The order sits on the quote and gets matched with probability
          ``maker_fill_prob_base × participation`` — a crude queue model.
        """
        if order.limit_price is None:
            raise ValueError("limit orders require limit_price")
        if order.side not in (-1, 1):
            raise ValueError("order.side must be ±1")

        high = float(bar["high"])
        low = float(bar["low"])
        lp = float(order.limit_price)

        traded_through = (
            (order.side == 1 and low <= lp)        # buy fills if price trades ≤ limit
            or (order.side == -1 and high >= lp)   # sell fills if price trades ≥ limit
        )

        assert self.fee_schedule is not None
        exec_price = lp  # maker always fills at limit
        mid = float(bar.get("open", bar.get("close", 0.0)))
        # Size economics computed against the **limit price** — that's
        # the notional the user committed to at placement.
        order_usd = order.qty * lp
        vol_usd = self._bar_volume_usd(bar, mid or lp)
        max_fillable_usd = vol_usd * self.participation_cap

        if traded_through:
            fill_usd = min(order_usd, max_fillable_usd)
            fill_qty = fill_usd / lp
        else:
            # Queue-position proxy — base probability is the chance a
            # resting quote matches during this bar; participation then
            # caps the fillable quantity. Size does not reduce the
            # *probability* of a fill, only the amount that fills.
            p_fill = self.maker_fill_prob_base
            if self._rng.random() < p_fill:
                fill_usd = min(order_usd, max_fillable_usd)
                fill_qty = fill_usd / lp
            else:
                return FillResult(order=order, fills=[])

        fee = fill_qty * exec_price * self.fee_schedule.maker

        fill = Fill(
            ts=order.ts,
            side=order.side,
            qty=fill_qty,
            price=exec_price,
            fee=fee,
            is_partial=fill_qty < order.qty - 1e-9,
            remaining=order.qty - fill_qty,
        )
        return FillResult(order=order, fills=[fill])

    def fill(self, order: Order, bar: pd.Series) -> FillResult:
        """Dispatch on order kind."""
        if order.kind == OrderKind.MARKET:
            return self.fill_market(order, bar)
        if order.kind == OrderKind.LIMIT:
            return self.fill_limit(order, bar)
        raise ValueError(f"Unknown order kind: {order.kind}")

    def fill_many(
        self, orders: Iterable[Order], bars: pd.DataFrame
    ) -> list[FillResult]:
        """Fill a stream of orders against a bar index."""
        out: list[FillResult] = []
        for o in orders:
            if o.ts not in bars.index:
                raise KeyError(f"no bar at {o.ts}")
            out.append(self.fill(o, bars.loc[o.ts]))
        return out


__all__ = [
    "FeeSchedule",
    "VENUES",
    "OrderKind",
    "Order",
    "Fill",
    "FillResult",
    "FillModel",
]

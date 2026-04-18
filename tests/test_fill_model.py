"""Tests for the execution fill model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.execution.fill_model import (
    VENUES,
    FeeSchedule,
    FillModel,
    Order,
    OrderKind,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _bar(price: float = 100.0, volume: float = 1e5, atr_ratio: float = 1.0) -> pd.Series:
    return pd.Series(
        {
            "open": price,
            "high": price * 1.005,
            "low": price * 0.995,
            "close": price,
            "volume": volume,
            "atr_ratio": atr_ratio,
        }
    )


# --------------------------------------------------------------------------
# Fee schedule
# --------------------------------------------------------------------------
def test_fee_schedule_rebate_makes_maker_negative():
    fs = FeeSchedule(maker_bps=2.0, taker_bps=5.0, maker_rebate_bps=3.0)
    assert fs.maker < 0
    assert fs.taker > 0


def test_known_venues_present():
    for key in ("binance_perp", "bybit_perp", "okx_perp", "default"):
        assert key in VENUES


# --------------------------------------------------------------------------
# Market (taker) fills
# --------------------------------------------------------------------------
def test_market_buy_pays_spread_and_fee():
    model = FillModel(spread_bps=2.0, participation_cap=1.0, impact_k=0.0)
    order = Order(ts=pd.Timestamp("2025-01-01"), side=1, qty=1.0, kind=OrderKind.MARKET)
    result = model.fill(order, _bar(price=100.0, volume=1e6))
    assert result.is_fully_filled
    # Taker buy fills above mid (spread × side).
    assert result.avg_price > 100.0
    assert result.total_fee > 0


def test_market_sell_earns_below_mid():
    model = FillModel(spread_bps=2.0, participation_cap=1.0, impact_k=0.0)
    order = Order(ts=pd.Timestamp("2025-01-01"), side=-1, qty=1.0, kind=OrderKind.MARKET)
    result = model.fill(order, _bar(price=100.0, volume=1e6))
    assert result.avg_price < 100.0


def test_market_partial_fill_when_order_exceeds_participation_cap():
    # Bar turnover = 100 * 1000 = $100k, cap = 1% → $1k fillable.
    # Order of 100 units × $100 = $10k — 90% should be unfilled.
    model = FillModel(participation_cap=0.01, impact_k=0.0)
    order = Order(ts=pd.Timestamp("2025-01-01"), side=1, qty=100.0)
    result = model.fill(order, _bar(price=100.0, volume=1000))
    assert not result.is_fully_filled
    assert result.fills[0].is_partial
    assert result.unfilled_qty > 0


def test_market_impact_grows_with_order_size():
    model = FillModel(participation_cap=1.0, spread_bps=0.0, impact_k=1.0)
    small = model.fill(
        Order(ts=pd.Timestamp("2025-01-01"), side=1, qty=1.0),
        _bar(price=100.0, volume=1000),
    )
    big = model.fill(
        Order(ts=pd.Timestamp("2025-01-01"), side=1, qty=50.0),
        _bar(price=100.0, volume=1000),
    )
    assert big.avg_price > small.avg_price


# --------------------------------------------------------------------------
# Limit (maker) fills
# --------------------------------------------------------------------------
def test_limit_buy_fills_when_traded_through():
    model = FillModel(participation_cap=1.0)
    bar = _bar(price=100.0, volume=1e6)
    # Limit at mid; bar low is 99.5 → trades through, fills.
    order = Order(
        ts=pd.Timestamp("2025-01-01"), side=1, qty=1.0,
        kind=OrderKind.LIMIT, limit_price=99.8,
    )
    result = model.fill(order, bar)
    assert result.is_fully_filled
    # Maker fills at limit, not mid.
    assert result.fills[0].price == pytest.approx(99.8)


def test_limit_sell_fills_when_traded_through():
    model = FillModel(participation_cap=1.0)
    bar = _bar(price=100.0, volume=1e6)
    order = Order(
        ts=pd.Timestamp("2025-01-01"), side=-1, qty=1.0,
        kind=OrderKind.LIMIT, limit_price=100.3,
    )
    result = model.fill(order, bar)
    assert result.is_fully_filled
    assert result.fills[0].price == pytest.approx(100.3)


def test_limit_not_traded_through_sometimes_fills_via_queue():
    # Extremely high base probability + large participation → near-certain queue fill.
    model = FillModel(participation_cap=1.0, maker_fill_prob_base=1.0, rng_seed=42)
    bar = _bar(price=100.0, volume=1e6)
    # Buy at 99.0 — bar low is 99.5, so NOT traded through.
    order = Order(
        ts=pd.Timestamp("2025-01-01"), side=1, qty=10.0,
        kind=OrderKind.LIMIT, limit_price=99.0,
    )
    result = model.fill(order, bar)
    # With p=1.0 this should fill (at the limit).
    assert result.fills
    assert result.fills[0].price == pytest.approx(99.0)


def test_limit_never_fills_when_prob_zero_and_not_traded_through():
    model = FillModel(participation_cap=1.0, maker_fill_prob_base=0.0)
    bar = _bar(price=100.0, volume=1e6)
    order = Order(
        ts=pd.Timestamp("2025-01-01"), side=1, qty=1.0,
        kind=OrderKind.LIMIT, limit_price=99.0,
    )
    result = model.fill(order, bar)
    assert result.fills == []
    assert result.unfilled_qty == pytest.approx(order.qty)


def test_maker_fee_smaller_than_taker_fee():
    bar = _bar(price=100.0, volume=1e6)
    taker_model = FillModel(spread_bps=2.0, participation_cap=1.0, impact_k=0.0)
    maker_model = FillModel(spread_bps=2.0, participation_cap=1.0, impact_k=0.0)
    t = taker_model.fill(
        Order(ts=pd.Timestamp("2025-01-01"), side=1, qty=1.0, kind=OrderKind.MARKET),
        bar,
    )
    m = maker_model.fill(
        Order(
            ts=pd.Timestamp("2025-01-01"), side=1, qty=1.0,
            kind=OrderKind.LIMIT, limit_price=99.8,
        ),
        bar,
    )
    assert m.total_fee < t.total_fee


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def test_invalid_side_rejected():
    model = FillModel()
    bar = _bar()
    with pytest.raises(ValueError):
        model.fill(Order(ts=pd.Timestamp("2025-01-01"), side=0, qty=1.0), bar)


def test_non_positive_qty_rejected():
    model = FillModel()
    bar = _bar()
    with pytest.raises(ValueError):
        model.fill(Order(ts=pd.Timestamp("2025-01-01"), side=1, qty=0.0), bar)


def test_limit_without_price_rejected():
    model = FillModel()
    bar = _bar()
    with pytest.raises(ValueError):
        model.fill(
            Order(
                ts=pd.Timestamp("2025-01-01"), side=1, qty=1.0, kind=OrderKind.LIMIT
            ),
            bar,
        )


def test_reproducibility_with_fixed_seed():
    bar = _bar(price=100.0, volume=1e6)
    order = Order(
        ts=pd.Timestamp("2025-01-01"), side=1, qty=100.0,
        kind=OrderKind.LIMIT, limit_price=99.0,
    )
    m1 = FillModel(participation_cap=0.01, maker_fill_prob_base=0.5, rng_seed=7)
    m2 = FillModel(participation_cap=0.01, maker_fill_prob_base=0.5, rng_seed=7)
    # Run 20 orders through each; results must be identical.
    r1 = [m1.fill(order, bar).filled_qty for _ in range(20)]
    r2 = [m2.fill(order, bar).filled_qty for _ in range(20)]
    assert r1 == r2


def test_fill_many_fills_multiple_orders():
    model = FillModel(participation_cap=1.0, impact_k=0.0)
    idx = pd.date_range("2025-01-01", periods=3, freq="1h")
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [100.5, 101.5, 102.5],
            "low": [99.5, 100.5, 101.5],
            "close": [100.2, 101.2, 102.2],
            "volume": [1e6, 1e6, 1e6],
            "atr_ratio": [1.0, 1.0, 1.0],
        },
        index=idx,
    )
    orders = [Order(ts=idx[i], side=1, qty=1.0) for i in range(3)]
    results = model.fill_many(orders, bars)
    assert len(results) == 3
    assert all(r.is_fully_filled for r in results)

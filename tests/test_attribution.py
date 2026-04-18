"""Tests for :mod:`src.audit.attribution`."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.audit.attribution import (
    Attribution,
    AttributionReport,
    attribute_round_trip,
    attribute_trades,
)
from src.audit.parity import JournalRecord, TradeRoundTrip


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _rt(
    *,
    direction: int = 1,
    qty: float = 1.0,
    entry_ref: float = 50_000.0,
    entry_exec: float = 50_000.0,
    entry_fee: float = 12.5,
    exit_ref: float = 50_500.0,
    exit_exec: float = 50_500.0,
    exit_fee: float = 12.625,
    holding_hours: float = 8.0,
    pnl: float | None = None,
    asset: str = "BTC/USDT",
    strategy: str = "alpha",
) -> TradeRoundTrip:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    if pnl is None:
        gross = (exit_exec - entry_exec) * direction * qty
        pnl = gross - (entry_fee + exit_fee)
    entry = JournalRecord(
        ts=base, event="entry", strategy=strategy, asset=asset,
        direction=direction, qty=qty, price=entry_exec, fee=entry_fee,
        reference_price=entry_ref, slippage_bps=0.0,
    )
    exit_rec = JournalRecord(
        ts=base + timedelta(hours=holding_hours), event="exit", strategy=strategy,
        asset=asset, direction=direction, qty=qty, price=exit_exec, fee=exit_fee,
        reference_price=exit_ref, slippage_bps=0.0, pnl=pnl,
    )
    return TradeRoundTrip(entry=entry, exit=exit_rec)


# --------------------------------------------------------------------------
# Single-trip attribution
# --------------------------------------------------------------------------
def test_attribution_zero_frictions_puts_everything_in_signal():
    """Ref == exec, zero fee, zero funding → all P&L in the 'signal' bucket."""
    rt = _rt(entry_exec=50_000.0, exit_exec=50_500.0, entry_fee=0.0, exit_fee=0.0)
    a = attribute_round_trip(rt)
    # Realised = signal = 500.0
    assert a.signal == pytest.approx(500.0)
    assert a.slippage == pytest.approx(0.0)
    assert a.fee == pytest.approx(0.0)
    assert a.funding == pytest.approx(0.0)
    assert a.drift == pytest.approx(0.0, abs=1e-9)


def test_attribution_buckets_sum_to_realised():
    rt = _rt()  # default with fees and typical execution
    a = attribute_round_trip(rt, funding_rate=0.0001, bar_volume_usd=1e8)
    bucket_sum = a.signal - a.slippage - a.fee - a.funding + a.drift
    assert bucket_sum == pytest.approx(a.realised_pnl, abs=1e-9)


def test_attribution_adverse_slippage_is_positive():
    """Live paid 10bps worse on each leg → slippage > 0 (cost to trader)."""
    rt = _rt(
        direction=1, qty=1.0,
        entry_ref=50_000.0, entry_exec=50_050.0,   # paid 50 higher on buy
        exit_ref=50_500.0, exit_exec=50_450.0,     # got 50 lower on sell
        entry_fee=0.0, exit_fee=0.0,
    )
    a = attribute_round_trip(rt)
    # Both legs adverse: (50_050-50_000)*1*1 + (50_500-50_450)*1*1 = 100.
    assert a.slippage == pytest.approx(100.0)


def test_attribution_short_direction_flips_slippage_sign_correctly():
    """On a short: buying lower on entry = favourable; buying higher on exit = adverse."""
    rt = _rt(
        direction=-1, qty=1.0,
        entry_ref=50_000.0, entry_exec=49_950.0,   # sold at 49950 (worse for shortseller)
        exit_ref=49_500.0, exit_exec=49_550.0,     # covered at 49550 (worse for shortseller)
        entry_fee=0.0, exit_fee=0.0,
    )
    a = attribute_round_trip(rt)
    # entry: (49_950-50_000)*(-1)*1 = +50
    # exit:  (49_500-49_550)*(-1)*1 = +50
    assert a.slippage == pytest.approx(100.0)


def test_attribution_funding_cost_long_pays_positive_rate():
    rt = _rt(direction=+1, qty=1.0, holding_hours=24.0,
             entry_exec=50_000.0, exit_exec=50_000.0,
             entry_ref=50_000.0, exit_ref=50_000.0,
             entry_fee=0.0, exit_fee=0.0)
    # funding_rate=0.0001 per 8h interval, 3 intervals in 24h.
    a = attribute_round_trip(rt, funding_rate=0.0001, funding_interval_hours=8.0)
    expected = 0.0001 * 50_000.0 * 3.0 * 1  # positive cost for long
    assert a.funding == pytest.approx(expected)


def test_attribution_funding_short_receives_positive_rate():
    rt = _rt(direction=-1, qty=1.0, holding_hours=24.0,
             entry_exec=50_000.0, exit_exec=50_000.0,
             entry_ref=50_000.0, exit_ref=50_000.0,
             entry_fee=0.0, exit_fee=0.0)
    a = attribute_round_trip(rt, funding_rate=0.0001, funding_interval_hours=8.0)
    # Short with positive funding: funding bucket is negative cost (= gain).
    assert a.funding < 0


def test_attribution_impact_is_non_negative_and_bounded():
    rt = _rt(
        entry_ref=50_000.0, entry_exec=50_050.0,
        exit_ref=50_500.0, exit_exec=50_450.0,
        entry_fee=0.0, exit_fee=0.0,
    )
    a = attribute_round_trip(rt, bar_volume_usd=1e7, impact_k=1.0)
    assert a.impact >= 0.0
    # Impact is an attribution component of slippage; for small orders it
    # should be bounded by |slippage|.
    assert a.impact <= abs(a.slippage) + 1e-6


def test_attribution_drift_captures_mismatched_realised():
    """If the journal's realised pnl disagrees with buckets, drift absorbs it."""
    rt = _rt(
        entry_exec=50_000.0, exit_exec=50_500.0,
        entry_fee=0.0, exit_fee=0.0,
        pnl=600.0,   # 100 more than signal-implied 500
    )
    a = attribute_round_trip(rt)
    assert a.drift == pytest.approx(100.0, abs=1e-9)


# --------------------------------------------------------------------------
# Batch report
# --------------------------------------------------------------------------
def test_attribute_trades_aggregates_per_asset_and_strategy():
    trips = [
        _rt(asset="BTC/USDT", strategy="a"),
        _rt(asset="BTC/USDT", strategy="b"),
        _rt(asset="ETH/USDT", strategy="a"),
    ]
    report = attribute_trades(trips)
    assert report.n_trades == 3
    assert set(report.per_asset) == {"BTC/USDT", "ETH/USDT"}
    assert set(report.per_strategy) == {"a", "b"}
    assert report.per_asset["BTC/USDT"]["n"] == 2
    assert report.per_strategy["a"]["n"] == 2


def test_attribute_trades_totals_add_up():
    trips = [_rt() for _ in range(5)]
    report = attribute_trades(trips)
    # total_realised should equal sum of per-trip realised.
    expected = sum(t.realised_pnl for t in trips)
    assert report.total_realised == pytest.approx(expected)


def test_attribute_trades_empty_returns_zero_report():
    report = attribute_trades([])
    assert report.n_trades == 0
    assert report.total_realised == 0.0


def test_attribute_trades_respects_per_asset_funding_rates():
    trips = [
        _rt(asset="BTC/USDT", holding_hours=8.0),
        _rt(asset="ETH/USDT", holding_hours=8.0),
    ]
    report = attribute_trades(
        trips,
        funding_rates={"BTC/USDT": 0.0001, "ETH/USDT": 0.0},
    )
    # Only BTC trip should have non-zero funding.
    btc_funding = report.per_asset["BTC/USDT"]["funding"]
    eth_funding = report.per_asset["ETH/USDT"]["funding"]
    assert btc_funding != 0.0
    assert eth_funding == 0.0


def test_attribution_to_dict_is_serialisable():
    import json
    rt = _rt()
    a = attribute_round_trip(rt)
    payload = json.dumps(a.as_dict(), default=str)
    assert "realised_pnl" in payload

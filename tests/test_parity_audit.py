"""Tests for ``src.audit.parity``."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.audit.parity import (
    JournalRecord,
    ParityDelta,
    ParityReport,
    audit_parity,
    load_journal,
    pair_round_trips,
    reconstruct_live_equity,
)
from src.execution.fill_model import FillModel


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _rec(
    ts: datetime,
    event: str,
    asset: str = "BTC/USDT",
    strategy: str = "alpha_v1",
    direction: int = 1,
    qty: float = 1.0,
    price: float = 50_000.0,
    fee: float = 12.5,           # 50k * 1.0 * 5bps
    reference_price: float | None = None,
    slippage_bps: float = 2.0,
    pnl: float = 0.0,
) -> dict:
    return {
        "ts": ts.isoformat(),
        "event": event,
        "strategy": strategy,
        "asset": asset,
        "direction": direction,
        "qty": qty,
        "price": price,
        "fee": fee,
        "reference_price": reference_price if reference_price is not None else price,
        "slippage_bps": slippage_bps,
        "reason": "signal",
        "pnl": pnl,
    }


def _write_journal(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "paper_journal.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def _ideal_round_trip(tmp_path: Path, n: int = 6) -> Path:
    """Build a journal where live execs match the model (ref_price == fill, zero adverse)."""
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    rows: list[dict] = []
    # Use a market-neutral model: reference_price == executed price, live_fee == 5bps of notional.
    # Fill model with participation_cap=1.0 and spread_bps=0, impact_k=0 gives model_price == ref.
    for i in range(n):
        t_in = base + timedelta(hours=i * 2)
        t_out = base + timedelta(hours=i * 2 + 1)
        qty = 0.1
        price = 50_000.0 + i
        # 5 bps taker fee: qty * price * 5e-4
        fee = qty * price * 5e-4
        rows.append(_rec(t_in, "entry", qty=qty, price=price, fee=fee, reference_price=price))
        rows.append(
            _rec(t_out, "exit", qty=qty, price=price + 10, fee=(qty * (price + 10) * 5e-4),
                 reference_price=(price + 10), pnl=qty * 10)
        )
    return _write_journal(tmp_path, rows)


# --------------------------------------------------------------------------
# load_journal / pair_round_trips / equity
# --------------------------------------------------------------------------
def test_load_journal_parses_valid_rows(tmp_path: Path):
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    rows = [
        _rec(base, "entry", qty=0.5, price=60_000.0),
        _rec(base + timedelta(hours=1), "exit", qty=0.5, price=60_500.0, pnl=250.0),
    ]
    p = _write_journal(tmp_path, rows)
    recs = load_journal(p)
    assert len(recs) == 2
    assert recs[0].event == "entry"
    assert recs[1].event == "exit"
    assert recs[1].pnl == pytest.approx(250.0)


def test_load_journal_missing_file_returns_empty(tmp_path: Path):
    assert load_journal(tmp_path / "nope.jsonl") == []


def test_load_journal_skips_malformed_lines(tmp_path: Path):
    p = tmp_path / "j.jsonl"
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    good = json.dumps(_rec(base, "entry"))
    p.write_text(f"{good}\nnot-json\n\n{{\"partial\": true}}\n{good}\n")
    recs = load_journal(p)
    assert len(recs) == 2


def test_pair_round_trips_matches_entries_to_exits():
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    recs = [
        JournalRecord(ts=base, event="entry", strategy="a", asset="BTC/USDT",
                      direction=1, qty=1.0, price=50_000.0, fee=25.0,
                      reference_price=50_000.0, slippage_bps=0.0),
        JournalRecord(ts=base + timedelta(hours=1), event="exit", strategy="a",
                      asset="BTC/USDT", direction=1, qty=1.0, price=50_500.0,
                      fee=25.25, reference_price=50_500.0, slippage_bps=0.0,
                      pnl=499.75),
    ]
    pairs = pair_round_trips(recs)
    assert len(pairs) == 1
    assert pairs[0].entry.event == "entry"
    assert pairs[0].exit.event == "exit"
    assert pairs[0].realised_pnl == pytest.approx(499.75)


def test_pair_round_trips_drops_unmatched_entries():
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    recs = [
        JournalRecord(ts=base, event="entry", strategy="a", asset="BTC/USDT",
                      direction=1, qty=1.0, price=50_000.0, fee=0.0,
                      reference_price=50_000.0, slippage_bps=0.0),
    ]
    assert pair_round_trips(recs) == []


def test_pair_round_trips_skip_events_are_ignored():
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    recs = [
        JournalRecord(ts=base, event="skip", strategy="a", asset="BTC/USDT",
                      direction=1, qty=0.0, price=0.0, fee=0.0,
                      reference_price=0.0, slippage_bps=0.0),
    ]
    assert pair_round_trips(recs) == []


def test_reconstruct_live_equity_accumulates_pnl(tmp_path: Path):
    p = _ideal_round_trip(tmp_path, n=3)
    recs = load_journal(p)
    eq = reconstruct_live_equity(recs, initial_capital=10_000.0)
    assert isinstance(eq, pd.Series)
    assert len(eq) == 3
    # Each round trip makes qty * 10 = 0.1 * 10 = 1 of PnL (minus fees on both sides).
    assert eq.iloc[-1] > 10_000.0 - 10.0  # fees shouldn't blow it up


def test_reconstruct_live_equity_empty_returns_initial_only():
    eq = reconstruct_live_equity([], initial_capital=5_000.0)
    assert len(eq) == 1
    assert eq.iloc[0] == 5_000.0


# --------------------------------------------------------------------------
# audit_parity
# --------------------------------------------------------------------------
def _frictionless_model() -> FillModel:
    """FillModel that returns exec_price == mid, fee == taker_bps * notional, no partial."""
    return FillModel(
        venue="binance_perp",
        participation_cap=10.0,   # huge — no partials
        spread_bps=0.0,
        impact_k=0.0,
        rng_seed=7,
    )


def test_audit_parity_ideal_case_is_pass(tmp_path: Path):
    """Live journal that exactly matches the model → verdict PASS, ~0 unexplained bps."""
    p = _ideal_round_trip(tmp_path, n=10)
    report = audit_parity(p, fill_model=_frictionless_model())
    assert report.n_events == 20  # 10 entries + 10 exits
    assert report.n_entries == 10
    assert report.n_exits == 10
    assert report.verdict == "PASS"
    assert abs(report.unexplained_pnl_bps) < 1e-6
    assert abs(report.mean_price_diff_bps) < 1e-6


def test_audit_parity_flags_systematic_adverse_slippage(tmp_path: Path):
    """If every live fill is 20 bps worse than reference, verdict FAIL."""
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    rows: list[dict] = []
    for i in range(6):
        t_in = base + timedelta(hours=i * 2)
        t_out = base + timedelta(hours=i * 2 + 1)
        ref = 50_000.0
        adverse = ref * 20e-4  # 20 bps
        rows.append(_rec(t_in, "entry", qty=0.2, price=ref + adverse, fee=5.0, reference_price=ref))
        rows.append(_rec(t_out, "exit", qty=0.2, price=ref - adverse, fee=5.0,
                         reference_price=ref, pnl=-adverse * 0.2 * 2))
    p = _write_journal(tmp_path, rows)
    report = audit_parity(p, fill_model=_frictionless_model(), warn_bps=2.5, fail_bps=5.0)
    assert report.verdict == "FAIL"
    # Unexplained P&L impact should be negative-trader-perspective (live costs more).
    assert report.total_pnl_impact_usd > 0


def test_audit_parity_per_asset_and_per_strategy_aggregates(tmp_path: Path):
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    rows = [
        _rec(base, "entry", asset="BTC/USDT", strategy="a", qty=0.1, price=50_000.0, fee=2.5, reference_price=50_000.0),
        _rec(base + timedelta(hours=1), "exit", asset="BTC/USDT", strategy="a", qty=0.1, price=50_100.0, fee=2.5, reference_price=50_100.0, pnl=10.0),
        _rec(base + timedelta(hours=2), "entry", asset="ETH/USDT", strategy="b", qty=1.0, price=3_000.0, fee=1.5, reference_price=3_000.0),
        _rec(base + timedelta(hours=3), "exit", asset="ETH/USDT", strategy="b", qty=1.0, price=3_010.0, fee=1.5, reference_price=3_010.0, pnl=10.0),
    ]
    p = _write_journal(tmp_path, rows)
    report = audit_parity(p, fill_model=_frictionless_model())
    assert set(report.per_asset) == {"BTC/USDT", "ETH/USDT"}
    assert set(report.per_strategy) == {"a", "b"}
    assert report.per_asset["BTC/USDT"]["n"] == 2
    assert report.per_strategy["b"]["n"] == 2


def test_audit_parity_empty_journal(tmp_path: Path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    report = audit_parity(p, fill_model=_frictionless_model())
    assert report.n_events == 0
    assert report.verdict == "PASS"
    assert report.unexplained_pnl_bps == 0.0


def test_audit_parity_to_dict_is_json_serialisable(tmp_path: Path):
    p = _ideal_round_trip(tmp_path, n=2)
    report = audit_parity(p, fill_model=_frictionless_model())
    blob = json.dumps(report.to_dict(), default=str)
    assert "verdict" in blob


def test_audit_parity_warn_threshold(tmp_path: Path):
    """Small adverse slippage lands in WARN band between warn_bps and fail_bps."""
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    rows: list[dict] = []
    for i in range(4):
        t_in = base + timedelta(hours=i * 2)
        t_out = base + timedelta(hours=i * 2 + 1)
        ref = 50_000.0
        adverse = ref * 3.5e-4   # 3.5 bps each side
        rows.append(_rec(t_in, "entry", qty=0.2, price=ref + adverse, fee=5.0, reference_price=ref))
        rows.append(_rec(t_out, "exit", qty=0.2, price=ref - adverse, fee=5.0,
                         reference_price=ref))
    p = _write_journal(tmp_path, rows)
    report = audit_parity(p, fill_model=_frictionless_model(), warn_bps=2.5, fail_bps=10.0)
    assert report.verdict == "WARN"


def test_audit_parity_accepts_in_memory_records():
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    recs = [
        JournalRecord(ts=base, event="entry", strategy="a", asset="BTC/USDT",
                      direction=1, qty=0.1, price=50_000.0, fee=2.5,
                      reference_price=50_000.0, slippage_bps=0.0),
        JournalRecord(ts=base + timedelta(hours=1), event="exit", strategy="a",
                      asset="BTC/USDT", direction=1, qty=0.1, price=50_100.0,
                      fee=2.505, reference_price=50_100.0, slippage_bps=0.0, pnl=10.0),
    ]
    report = audit_parity(recs, fill_model=_frictionless_model())
    assert report.n_events == 2
    assert report.verdict == "PASS"


def test_parity_delta_price_diff_bps():
    d = ParityDelta(
        ts=datetime(2026, 4, 1, tzinfo=timezone.utc),
        asset="BTC/USDT", strategy="a", event="entry", direction=1, qty=1.0,
        live_price=50_005.0, model_price=50_000.0,
        live_fee=25.0, model_fee=25.0, reference_price=50_000.0,
    )
    assert d.price_diff_bps == pytest.approx(1.0)     # 5/50_000 = 1 bps
    assert d.fee_diff_usd == 0.0
    # Long direction: pnl_impact = (live-model)*qty = 5
    assert d.pnl_impact_usd == pytest.approx(5.0)


def test_parity_delta_short_direction_flips_pnl_sign():
    d = ParityDelta(
        ts=datetime(2026, 4, 1, tzinfo=timezone.utc),
        asset="BTC/USDT", strategy="a", event="entry", direction=-1, qty=1.0,
        live_price=50_005.0, model_price=50_000.0,
        live_fee=25.0, model_fee=25.0, reference_price=50_000.0,
    )
    # Short filled 5 higher = BETTER for shortseller → negative impact on trader.
    assert d.pnl_impact_usd == pytest.approx(-5.0)


# --------------------------------------------------------------------------
# CLI smoke
# --------------------------------------------------------------------------
def test_cli_runs_and_writes_output(tmp_path: Path, capsys):
    from src.audit.parity import _main
    journal = _ideal_round_trip(tmp_path, n=2)
    out = tmp_path / "report.json"
    # Default FillModel has a 2bps spread the ideal journal doesn't show, so
    # allow a generous tolerance here — the test is about CLI mechanics.
    code = _main([
        str(journal),
        "--warn-bps", "10",
        "--fail-bps", "20",
        "--output", str(out),
    ])
    captured = capsys.readouterr()
    assert "verdict=PASS" in captured.out
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["verdict"] == "PASS"
    assert payload["n_events"] == 4

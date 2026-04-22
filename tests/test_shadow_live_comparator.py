from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.ops.shadow_live_comparator import (
    build_shadow_live_comparator_report,
    build_shadow_live_observation,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2))


def test_shadow_live_observation_uses_touch_and_fill_gaps() -> None:
    observation = build_shadow_live_observation(
        {
            "broker": "bybit",
            "best_bid": 99.5,
            "best_ask": 100.5,
            "mid_price": 100.0,
            "touch_price": 100.5,
            "price": 100.7,
            "spread_bps": 10.0,
            "impact_bps": 6.0,
            "quote_timestamp": "2026-04-21T00:00:00+00:00",
        },
        symbol="BTC/USDT",
        direction=1,
        reference_price=100.0,
    )

    assert observation is not None
    assert observation.reference_gap_bps < 0.0
    assert observation.fill_gap_bps > 0.0
    namespaced = observation.namespaced("shadow_live_entry")
    assert namespaced["shadow_live_entry_touch_price"] == 100.5


def test_shadow_live_comparator_report_marks_ready_when_quote_gaps_are_clean(tmp_path: Path) -> None:
    start = datetime(2026, 4, 21, tzinfo=timezone.utc)
    _write_json(
        tmp_path / "trade_journal.json",
        [
            {
                "entry_time": (start + timedelta(minutes=20 * index)).isoformat(),
                "exit_time": (start + timedelta(minutes=20 * index + 5)).isoformat(),
                "shadow_live_entry_quote_timestamp": (start + timedelta(minutes=20 * index)).isoformat(),
                "shadow_live_entry_touch_price": 100.5,
                "shadow_live_entry_mid_price": 100.0,
                "shadow_live_entry_reference_gap_bps": -4.9,
                "shadow_live_entry_fill_gap_bps": 2.5,
                "shadow_live_entry_quote_spread_bps": 10.0,
                "shadow_live_entry_quote_impact_bps": 4.0,
                "shadow_live_exit_quote_timestamp": (start + timedelta(minutes=20 * index + 5)).isoformat(),
                "shadow_live_exit_touch_price": 104.5,
                "shadow_live_exit_mid_price": 104.0,
                "shadow_live_exit_reference_gap_bps": 3.8,
                "shadow_live_exit_fill_gap_bps": 2.0,
                "shadow_live_exit_quote_spread_bps": 9.0,
                "shadow_live_exit_quote_impact_bps": 5.0,
            }
            for index in range(300)
        ],
    )

    report = build_shadow_live_comparator_report(tmp_path)

    assert report.ready_for_capital is True
    assert report.comparator_level == "stable"
    assert report.entry_comparison_count == 300
    assert report.exit_comparison_count == 300
    assert report.validation_runtime_days >= 4.0
    assert report.reasons == []


def test_shadow_live_comparator_report_blocks_when_tail_risk_breaks_stability(tmp_path: Path) -> None:
    start = datetime(2026, 4, 21, tzinfo=timezone.utc)
    _write_json(
        tmp_path / "trade_journal.json",
        [
            {
                "entry_time": (start + timedelta(minutes=20 * index)).isoformat(),
                "exit_time": (start + timedelta(minutes=20 * index + 5)).isoformat(),
                "shadow_live_entry_quote_timestamp": (start + timedelta(minutes=20 * index)).isoformat(),
                "shadow_live_entry_touch_price": 100.5,
                "shadow_live_entry_reference_gap_bps": 4.0 if index < 280 else 22.0,
                "shadow_live_entry_fill_gap_bps": 2.0 if index < 280 else 14.0,
                "shadow_live_entry_quote_spread_bps": 10.0 if index < 280 else 34.0,
                "shadow_live_entry_quote_impact_bps": 5.0 if index < 280 else 38.0,
                "shadow_live_exit_quote_timestamp": (start + timedelta(minutes=20 * index + 5)).isoformat(),
                "shadow_live_exit_touch_price": 104.5,
                "shadow_live_exit_reference_gap_bps": 4.5 if index < 280 else 24.0,
                "shadow_live_exit_fill_gap_bps": 2.5 if index < 280 else 16.0,
                "shadow_live_exit_quote_spread_bps": 9.0 if index < 280 else 32.0,
                "shadow_live_exit_quote_impact_bps": 6.0 if index < 280 else 36.0,
            }
            for index in range(300)
        ],
    )

    report = build_shadow_live_comparator_report(tmp_path)

    assert report.ready_for_capital is False
    assert report.comparator_level == "unstable"
    assert any("p95" in reason.lower() for reason in report.reasons)
    assert any("stability" in reason.lower() for reason in report.reasons)
from __future__ import annotations

import json
from pathlib import Path

from src.ops.execution_drift import build_execution_drift_report


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2))


def test_execution_drift_report_marks_clean_paper_layer_ready(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "divergence_log.json",
        [
            {
                "strategy": "funding_mr_v7",
                "symbol": "BTC/USDT",
                "entry_slippage_bps": 4.0,
                "exit_slippage_bps": 5.0,
                "pnl_divergence_pct": 8.0,
                "fill_time_ms": 120.0,
                "was_partial": False,
                "missed": False,
            }
            for _ in range(6)
        ],
    )
    _write_json(
        tmp_path / "trade_journal.json",
        [
            {
                "fill_ratio": 0.98,
                "entry_execution_ms": 140.0,
                "exit_execution_ms": 160.0,
                "book_spread_bps": 3.0,
                "book_impact_bps": 4.0,
            }
            for _ in range(6)
        ],
    )
    _write_json(
        tmp_path / "shadow_execution_status.json",
        {
            "compared_trade_count": 6,
            "avg_abs_entry_delta_bps": 3.5,
            "avg_abs_exit_delta_bps": 4.0,
            "avg_abs_pnl_delta_pct": 0.04,
        },
    )
    _write_json(
        tmp_path / "shadow_live_comparator_status.json",
        {
            "entry_comparison_count": 300,
            "exit_comparison_count": 300,
            "avg_abs_entry_reference_gap_bps": 4.0,
            "avg_abs_exit_reference_gap_bps": 5.0,
            "avg_abs_entry_fill_gap_bps": 2.0,
            "avg_abs_exit_fill_gap_bps": 2.5,
        },
    )

    report = build_execution_drift_report(tmp_path)

    assert report.reliable_for_capital is True
    assert report.execution_fidelity_level == "stable"
    assert report.compared_trade_count == 6
    assert report.shadow_compared_trade_count == 6
    assert report.shadow_live_entry_comparison_count == 300
    assert report.shadow_live_exit_comparison_count == 300
    assert report.reasons == []


def test_execution_drift_report_blocks_when_friction_and_misses_rise(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "divergence_log.json",
        [
            {
                "strategy": "momentum_breakout",
                "symbol": "ETH/USDT",
                "entry_slippage_bps": 18.0,
                "exit_slippage_bps": 21.0,
                "pnl_divergence_pct": 28.0,
                "fill_time_ms": 3400.0,
                "was_partial": True,
                "missed": False,
            },
            {
                "strategy": "momentum_breakout",
                "symbol": "ETH/USDT",
                "missed": True,
            },
            {
                "strategy": "momentum_breakout",
                "symbol": "ETH/USDT",
                "missed": True,
            },
        ],
    )
    _write_json(
        tmp_path / "trade_journal.json",
        [
            {
                "fill_ratio": 0.72,
                "entry_execution_ms": 3200.0,
                "exit_execution_ms": 3600.0,
                "book_spread_bps": 26.0,
                "book_impact_bps": 34.0,
            }
        ],
    )
    _write_json(
        tmp_path / "shadow_execution_status.json",
        {
            "compared_trade_count": 1,
            "avg_abs_entry_delta_bps": 12.0,
            "avg_abs_exit_delta_bps": 15.0,
            "avg_abs_pnl_delta_pct": 0.18,
        },
    )
    _write_json(
        tmp_path / "shadow_live_comparator_status.json",
        {
            "entry_comparison_count": 1,
            "exit_comparison_count": 0,
            "avg_abs_entry_reference_gap_bps": 18.0,
            "avg_abs_exit_reference_gap_bps": 0.0,
            "avg_abs_entry_fill_gap_bps": 11.0,
            "avg_abs_exit_fill_gap_bps": 0.0,
        },
    )

    report = build_execution_drift_report(tmp_path)

    assert report.reliable_for_capital is False
    assert report.execution_fidelity_level == "unstable"
    assert any("miss rate" in reason.lower() for reason in report.reasons)
    assert any("slippage" in reason.lower() for reason in report.reasons)
    assert any("shadow" in reason.lower() for reason in report.reasons)
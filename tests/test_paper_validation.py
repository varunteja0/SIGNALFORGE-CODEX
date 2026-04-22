from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.ops.paper_validation import (
    PaperValidationThresholds,
    build_paper_validation_report,
    format_paper_validation_report,
    write_paper_validation_report,
)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_build_paper_validation_report_ready_for_live(tmp_path):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    cycle_rows = [
        {
            "timestamp": (start + timedelta(days=offset)).isoformat(),
            "volatility_tracking_error": 0.03,
            "edge_retention_ratio": 0.82,
            "edge_retention_state": "healthy",
            "portfolio_objective_score": 0.18,
            "current_drawdown": 0.03,
            "pid_output": pid_output,
            "execution": {
                "total_trades": 12,
                "total_missed": 0,
                "avg_entry_slippage_bps": 3.5,
                "avg_pnl_divergence_pct": 4.0,
            },
            "safety_action": "allow",
        }
        for offset, pid_output in [(0, 0.10), (7, 0.12), (15, 0.08)]
    ]
    _write_jsonl(tmp_path / "adaptive_cycle_ledger.jsonl", cycle_rows)
    _write_json(tmp_path / "trade_journal.json", [{"id": 1}, {"id": 2}, {"id": 3}])
    _write_json(tmp_path / "live_state.json", {"timestamp": cycle_rows[-1]["timestamp"]})

    thresholds = PaperValidationThresholds(
        min_runtime_days=14.0,
        min_cycle_count=3,
        min_trade_count=3,
        max_drawdown=0.05,
        max_mean_tracking_error=0.05,
        max_avg_entry_slippage_bps=5.0,
        max_miss_rate=0.05,
        max_abs_avg_pnl_divergence_pct=10.0,
        max_pid_flip_rate=0.4,
        min_mean_edge_retention=0.70,
        max_fragile_edge_fraction=0.10,
        min_mean_objective_score=0.05,
    )

    report = build_paper_validation_report(tmp_path, thresholds)

    assert report.ready_for_live is True
    assert report.run_days >= 15.0
    assert report.trade_count == 3
    assert report.miss_rate == 0.0
    assert report.reasons == []


def test_build_paper_validation_report_blocks_on_instability(tmp_path):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    cycle_rows = [
        {
            "timestamp": (start + timedelta(days=offset)).isoformat(),
            "volatility_tracking_error": tracking_error,
            "edge_retention_ratio": edge_retention,
            "edge_retention_state": edge_state,
            "portfolio_objective_score": objective,
            "current_drawdown": drawdown,
            "pid_output": pid_output,
            "execution": execution,
            "safety_action": safety_action,
        }
        for offset, tracking_error, edge_retention, edge_state, objective, drawdown, pid_output, execution, safety_action in [
            (0, 0.12, 0.45, "broken", -0.20, 0.05, 0.20, {}, "allow"),
            (1, 0.11, 0.40, "fragile", -0.22, 0.12, -0.20, {}, "reduce"),
            (
                2,
                0.10,
                0.35,
                "broken",
                -0.25,
                0.18,
                0.25,
                {
                    "total_trades": 10,
                    "total_missed": 3,
                    "avg_entry_slippage_bps": 12.0,
                    "avg_pnl_divergence_pct": 30.0,
                },
                "halt",
            ),
        ]
    ]
    _write_jsonl(tmp_path / "adaptive_cycle_ledger.jsonl", cycle_rows)
    _write_json(tmp_path / "trade_journal.json", [{"id": 1}])
    _write_json(tmp_path / "live_state.json", {"timestamp": cycle_rows[-1]["timestamp"]})

    thresholds = PaperValidationThresholds(
        min_runtime_days=7.0,
        min_cycle_count=5,
        min_trade_count=2,
        max_drawdown=0.08,
        max_mean_tracking_error=0.08,
        max_avg_entry_slippage_bps=8.0,
        max_miss_rate=0.10,
        max_abs_avg_pnl_divergence_pct=20.0,
        max_pid_flip_rate=0.35,
        min_mean_edge_retention=0.60,
        max_fragile_edge_fraction=0.15,
        min_mean_objective_score=-0.05,
    )

    report = build_paper_validation_report(tmp_path, thresholds)

    assert report.ready_for_live is False
    assert any("runtime" in reason for reason in report.reasons)
    assert any("cycles" in reason for reason in report.reasons)
    assert any("avg entry slippage" in reason for reason in report.reasons)
    assert any("latest safety action is halt" == reason for reason in report.reasons)


def test_write_and_format_paper_validation_report(tmp_path):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "timestamp": start.isoformat(),
            "volatility_tracking_error": 0.02,
            "edge_retention_ratio": 0.80,
            "edge_retention_state": "healthy",
            "portfolio_objective_score": 0.10,
            "current_drawdown": 0.01,
            "pid_output": 0.10,
            "execution": {
                "total_trades": 4,
                "total_missed": 0,
                "avg_entry_slippage_bps": 2.0,
                "avg_pnl_divergence_pct": 3.0,
            },
            "safety_action": "allow",
        },
        {
            "timestamp": (start + timedelta(days=14)).isoformat(),
            "volatility_tracking_error": 0.02,
            "edge_retention_ratio": 0.80,
            "edge_retention_state": "healthy",
            "portfolio_objective_score": 0.10,
            "current_drawdown": 0.01,
            "pid_output": 0.10,
            "execution": {
                "total_trades": 4,
                "total_missed": 0,
                "avg_entry_slippage_bps": 2.0,
                "avg_pnl_divergence_pct": 3.0,
            },
            "safety_action": "allow",
        },
    ]
    _write_jsonl(tmp_path / "adaptive_cycle_ledger.jsonl", rows)
    _write_json(tmp_path / "trade_journal.json", [{"id": 1}])
    _write_json(tmp_path / "live_state.json", {"timestamp": rows[-1]["timestamp"]})

    report = build_paper_validation_report(
        tmp_path,
        PaperValidationThresholds(min_cycle_count=2, min_trade_count=1),
    )
    output_path = tmp_path / "paper_validation_status.json"
    write_paper_validation_report(report, output_path)

    assert output_path.exists()
    formatted = format_paper_validation_report(report)
    assert "STRICT PAPER VALIDATION REPORT" in formatted
    assert ("Ready for live:    YES" in formatted) or ("Ready for live:    NO" in formatted)
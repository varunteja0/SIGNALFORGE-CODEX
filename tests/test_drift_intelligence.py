from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.ops.drift_intelligence import build_drift_intelligence_report


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))


def _seed_green_statuses(base_dir: Path) -> None:
    _write_json(base_dir / "paper_validation_status.json", {"ready_for_live": True})
    _write_json(base_dir / "health.json", {"overall_status": "ok", "should_halt": False})
    _write_json(base_dir / "trade_parity_status.json", {"verdict": "PASS", "unexplained_pnl_bps": 0.5})
    _write_json(base_dir / "shadow_execution_status.json", {"ready_for_live": True})
    _write_json(base_dir / "failure_drill_report.json", {"all_passed": True})
    _write_json(base_dir / "broker_reconciliation_status.json", {"overall_status": "ok", "critical_issue_count": 0, "warning_issue_count": 0})


def test_drift_intelligence_recommends_micro_live_after_stable_burn_in(tmp_path: Path) -> None:
    _seed_green_statuses(tmp_path)
    _write_json(tmp_path / "divergence_log.json", [])
    _write_jsonl(
        tmp_path / "production_certification_history.jsonl",
        [
            {
                "timestamp": (datetime.now(timezone.utc) - timedelta(days=31)).isoformat(),
                "current_green": True,
            }
        ],
    )
    _write_jsonl(
        tmp_path / "adaptive_cycle_ledger.jsonl",
        [
            {
                "timestamp": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
                "safety_action": "allow",
                "allocation_weights": {"funding_mr_v7": 0.5, "momentum_breakout": 0.5},
                "disabled_strategies": {},
            },
            {
                "timestamp": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                "safety_action": "allow",
                "allocation_weights": {"funding_mr_v7": 0.48, "momentum_breakout": 0.52},
                "disabled_strategies": {},
            },
        ],
    )

    report = build_drift_intelligence_report(tmp_path)

    assert report.current_green_snapshot is True
    assert report.risk_level == "low"
    assert report.deployment_recommendation.mode == "micro_live"
    assert report.deployment_recommendation.max_total_exposure_pct > 0.0


def test_drift_intelligence_blocks_when_instability_clusters(tmp_path: Path) -> None:
    _seed_green_statuses(tmp_path)
    now = datetime.now(timezone.utc)
    _write_json(
        tmp_path / "divergence_log.json",
        [
            {
                "missed": False,
                "entry_slippage_bps": 24.0,
                "pnl_divergence_pct": 35.0,
            },
            {
                "missed": False,
                "entry_slippage_bps": 28.0,
                "pnl_divergence_pct": 40.0,
            },
            {
                "missed": False,
                "entry_slippage_bps": 31.0,
                "pnl_divergence_pct": 32.0,
            },
            {
                "missed": False,
                "entry_slippage_bps": 35.0,
                "pnl_divergence_pct": 45.0,
            },
            {
                "missed": False,
                "entry_slippage_bps": 39.0,
                "pnl_divergence_pct": 41.0,
            },
            {
                "missed": True,
            },
        ],
    )
    _write_jsonl(
        tmp_path / "production_certification_history.jsonl",
        [
            {"timestamp": (now - timedelta(hours=6)).isoformat(), "current_green": True},
            {"timestamp": (now - timedelta(hours=5)).isoformat(), "current_green": False},
            {"timestamp": (now - timedelta(hours=4)).isoformat(), "current_green": True},
            {"timestamp": (now - timedelta(hours=3)).isoformat(), "current_green": False},
            {"timestamp": (now - timedelta(hours=2)).isoformat(), "current_green": True},
            {"timestamp": (now - timedelta(hours=1)).isoformat(), "current_green": False},
        ],
    )
    _write_jsonl(
        tmp_path / "adaptive_cycle_ledger.jsonl",
        [
            {
                "timestamp": (now - timedelta(hours=4)).isoformat(),
                "safety_action": "allow",
                "allocation_weights": {"funding_mr_v7": 0.9, "momentum_breakout": 0.1},
                "disabled_strategies": {},
            },
            {
                "timestamp": (now - timedelta(hours=3)).isoformat(),
                "safety_action": "pause_entries",
                "allocation_weights": {"funding_mr_v7": 0.1, "momentum_breakout": 0.9},
                "disabled_strategies": {"funding_mr_v7": "edge retention broken"},
            },
            {
                "timestamp": (now - timedelta(hours=2)).isoformat(),
                "safety_action": "halt",
                "allocation_weights": {"funding_mr_v7": 0.85, "momentum_breakout": 0.15},
                "disabled_strategies": {},
            },
            {
                "timestamp": (now - timedelta(hours=1)).isoformat(),
                "safety_action": "pause_entries",
                "allocation_weights": {"funding_mr_v7": 0.15, "momentum_breakout": 0.85},
                "disabled_strategies": {"momentum_breakout": "tracking error"},
            },
        ],
    )

    report = build_drift_intelligence_report(tmp_path)

    assert report.risk_level in {"high", "critical"}
    assert report.predicted_certification_failure is True
    assert report.deployment_recommendation.mode == "blocked"
    assert report.reasons
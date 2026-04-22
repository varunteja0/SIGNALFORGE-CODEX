from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.ops.streaming_stress_kernel import build_streaming_stress_kernel_report


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))


def _history_rows(now: datetime, *, points: int, shock: bool = False) -> list[dict]:
    rows = []
    for idx in range(points):
        wiggle = ((idx % 5) - 2) / 100.0
        rows.append(
            {
                "timestamp": (now - timedelta(hours=points - idx)).isoformat(),
                "features": {
                    "mean_abs_funding_z": 1.00 + wiggle,
                    "max_abs_funding_z": 1.15 + wiggle,
                    "mean_vol_ratio": 1.08 + wiggle,
                    "max_vol_ratio": 1.16 + wiggle,
                    "mean_atr_exp": 1.02 + wiggle,
                    "max_atr_exp": 1.08 + wiggle,
                    "mean_bb_pctile": 0.42 + wiggle,
                    "min_bb_pctile": 0.36 + wiggle,
                    "mean_breakout_pressure": 0.05 + wiggle,
                    "max_breakout_pressure": 0.10 + wiggle,
                    "regime_stress": 0.35,
                    "regime_dispersion": 0.50,
                    "mean_abs_corr_48h": 0.38 + wiggle,
                    "corr_shift_48h": 0.03 if not shock else 0.55,
                    "dispersion_24h": 0.012 if not shock else 0.070,
                },
            }
        )
    return rows


def _trade(fill_ratio: float, latency_ms: float, spread_bps: float = 2.0, impact_bps: float = 1.5) -> dict:
    return {
        "fill_ratio": fill_ratio,
        "entry_execution_ms": latency_ms,
        "exit_execution_ms": latency_ms,
        "book_spread_bps": spread_bps,
        "book_impact_bps": impact_bps,
        "filled_size_usd": 1000.0,
        "requested_size_usd": 1000.0,
        "size_usd": 1000.0,
    }


def test_streaming_stress_kernel_allows_probation_when_pressure_is_low(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    snapshot_ts = now.isoformat()
    _write_json(tmp_path / "market_snapshot.json", {"_timestamp": snapshot_ts})
    _write_jsonl(tmp_path / "regime_novelty_history.jsonl", _history_rows(now, points=120))
    _write_json(tmp_path / "trade_journal.json", [_trade(0.99, 35.0), _trade(0.98, 42.0), _trade(0.97, 48.0)])
    _write_json(tmp_path / "divergence_log.json", [{"missed": False}, {"missed": False}])
    _write_json(
        tmp_path / "drift_intelligence_status.json",
        {"current_green_snapshot": True, "risk_score": 18.0},
    )
    _write_json(
        tmp_path / "survivability_status.json",
        {"survivability_score": 82.0},
    )
    _write_jsonl(
        tmp_path / "kill_switch_telemetry.jsonl",
        [
            {
                "requires_protection": True,
                "protection_applied": True,
                "detection_to_decision_ms": 42.0,
                "decision_to_protection_ms": 18.0,
            }
        ],
    )

    report = build_streaming_stress_kernel_report(tmp_path)

    assert report.predicted_pressure_failure is False
    assert report.probation_live_policy.allow_probation_live is True
    assert report.probation_live_policy.stage in {"plm_0.50", "micro_live_ready", "plm_0.10"}
    assert report.source_generated_at == snapshot_ts


def test_streaming_stress_kernel_blocks_when_path_and_execution_cluster(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    _write_jsonl(tmp_path / "regime_novelty_history.jsonl", _history_rows(now, points=120, shock=True))
    _write_json(tmp_path / "trade_journal.json", [_trade(0.55, 420.0, 9.0, 8.0), _trade(0.50, 520.0, 11.0, 9.0)])
    _write_json(tmp_path / "divergence_log.json", [{"missed": True}, {"missed": True}, {"missed": True}])
    _write_json(
        tmp_path / "drift_intelligence_status.json",
        {"current_green_snapshot": True, "risk_score": 62.0},
    )
    _write_json(
        tmp_path / "survivability_status.json",
        {"survivability_score": 38.0},
    )
    _write_jsonl(
        tmp_path / "kill_switch_telemetry.jsonl",
        [
            {
                "requires_protection": True,
                "protection_applied": False,
                "detection_to_decision_ms": 380.0,
                "decision_to_protection_ms": 250.0,
            }
        ],
    )

    report = build_streaming_stress_kernel_report(tmp_path)

    assert report.predicted_pressure_failure is True
    assert report.probation_live_policy.stage == "blocked"
    assert report.reasons
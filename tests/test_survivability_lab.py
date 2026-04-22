from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.ops.survivability_lab import build_survivability_report


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))


def _history_rows(now: datetime, *, points: int, shock: bool = False) -> list[dict]:
    rows = []
    for idx in range(points):
        wiggle = ((idx % 7) - 3) / 100.0
        rows.append(
            {
                "timestamp": (now - timedelta(hours=points - idx)).isoformat(),
                "features": {
                    "mean_abs_funding_z": 1.00 + wiggle,
                    "max_abs_funding_z": 1.12 + wiggle,
                    "mean_vol_ratio": 1.09 + wiggle,
                    "max_vol_ratio": 1.16 + wiggle,
                    "mean_atr_exp": 1.01 + wiggle,
                    "max_atr_exp": 1.05 + wiggle,
                    "mean_bb_pctile": 0.40 + wiggle,
                    "min_bb_pctile": 0.36 + wiggle,
                    "mean_breakout_pressure": 0.03 + wiggle,
                    "max_breakout_pressure": 0.08 + wiggle,
                    "regime_stress": 0.35,
                    "regime_dispersion": 0.50,
                    "mean_abs_corr_48h": 0.40 + wiggle,
                    "corr_shift_48h": 0.03 if not shock else 0.55,
                    "dispersion_24h": 0.011 if not shock else 0.060,
                },
            }
        )
    return rows


def _stable_snapshot(now: datetime) -> dict:
    return {
        "BTC/USDT": {
            "price": 70000.0,
            "funding_zscore": 0.9,
            "regime": "sideways",
            "bb_pctile": 42.0,
            "vol_ratio": 1.10,
            "ch_high": 71000.0,
            "ch_low": 69000.0,
            "atr_exp": 1.02,
        },
        "ETH/USDT": {
            "price": 3200.0,
            "funding_zscore": 1.1,
            "regime": "sideways",
            "bb_pctile": 38.0,
            "vol_ratio": 1.08,
            "ch_high": 3260.0,
            "ch_low": 3140.0,
            "atr_exp": 1.00,
        },
        "_cross_asset": {
            "mean_abs_corr_48h": 0.40,
            "corr_shift_48h": 0.03,
            "dispersion_24h": 0.011,
        },
        "_timestamp": now.isoformat(),
    }


def _stress_snapshot(now: datetime) -> dict:
    return {
        "BTC/USDT": {
            "price": 70000.0,
            "funding_zscore": 5.2,
            "regime": "high_volatility",
            "bb_pctile": 4.0,
            "vol_ratio": 4.5,
            "ch_high": 76000.0,
            "ch_low": 64000.0,
            "atr_exp": 2.8,
        },
        "ETH/USDT": {
            "price": 3200.0,
            "funding_zscore": 4.8,
            "regime": "high_volatility",
            "bb_pctile": 6.0,
            "vol_ratio": 3.9,
            "ch_high": 3700.0,
            "ch_low": 2800.0,
            "atr_exp": 2.5,
        },
        "_cross_asset": {
            "mean_abs_corr_48h": 0.92,
            "corr_shift_48h": 0.51,
            "dispersion_24h": 0.075,
        },
        "_timestamp": now.isoformat(),
    }


def _trade_row(entry_slip: float, exit_slip: float, fill_ratio: float, execution_ms: float) -> dict:
    return {
        "entry_slippage_bps": entry_slip,
        "exit_slippage_bps": exit_slip,
        "book_spread_bps": 2.0,
        "book_impact_bps": 1.5,
        "fill_ratio": fill_ratio,
        "entry_execution_ms": execution_ms,
        "exit_execution_ms": execution_ms,
        "filled_size_usd": 1000.0,
        "requested_size_usd": 1000.0,
        "size_usd": 1000.0,
    }


def test_survivability_lab_allows_micro_probation_when_history_and_execution_are_clean(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    _write_jsonl(tmp_path / "regime_novelty_history.jsonl", _history_rows(now, points=80))
    _write_json(tmp_path / "market_snapshot.json", _stable_snapshot(now))
    _write_json(
        tmp_path / "drift_intelligence_status.json",
        {"current_green_snapshot": True, "risk_score": 22.0},
    )
    _write_json(
        tmp_path / "trade_journal.json",
        [
            _trade_row(1.5, 1.1, 0.99, 45.0),
            _trade_row(1.4, 1.2, 0.98, 55.0),
            _trade_row(1.8, 1.3, 0.97, 65.0),
        ],
    )
    _write_json(tmp_path / "divergence_log.json", [{"missed": False, "entry_slippage_bps": 1.6}])

    report = build_survivability_report(tmp_path)

    assert report.predicted_survivability_failure is False
    assert report.survivability_score >= 65.0
    assert report.regime_novelty_level == "low"
    assert report.exposure_ladder.stage in {"0.5%", "1%", "5%"}


def test_survivability_lab_blocks_on_regime_shock_and_execution_fragility(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    _write_jsonl(tmp_path / "regime_novelty_history.jsonl", _history_rows(now, points=80, shock=True))
    _write_json(tmp_path / "market_snapshot.json", _stress_snapshot(now))
    _write_json(
        tmp_path / "drift_intelligence_status.json",
        {"current_green_snapshot": True, "risk_score": 48.0},
    )
    _write_json(
        tmp_path / "trade_journal.json",
        [
            _trade_row(18.0, 12.0, 0.72, 180.0),
            _trade_row(22.0, 15.0, 0.68, 220.0),
            _trade_row(25.0, 17.0, 0.64, 260.0),
        ],
    )
    _write_json(
        tmp_path / "divergence_log.json",
        [
            {"missed": False, "entry_slippage_bps": 24.0},
            {"missed": True},
            {"missed": True},
        ],
    )

    report = build_survivability_report(tmp_path)

    assert report.predicted_survivability_failure is True
    assert report.regime_shift_detected is True
    assert report.exposure_ladder.stage == "blocked"
    assert report.reasons
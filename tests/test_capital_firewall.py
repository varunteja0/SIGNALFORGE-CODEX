from __future__ import annotations

import json
from pathlib import Path

from src.ops.capital_firewall import build_capital_firewall_report


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _seed_firewall_inputs(
    tmp_path: Path,
    *,
    fidelity: float = 88.0,
    reliable_for_capital: bool = True,
    miss_rate: float = 0.03,
    avg_entry_slippage_bps: float = 4.0,
    avg_fill_ratio: float = 0.98,
    avg_book_spread_bps: float = 4.0,
    avg_book_impact_bps: float = 5.0,
    hysteresis: float = 0.18,
    adversary: float = 0.20,
    collapse_probability: float = 0.20,
    collapse_horizon_ticks: int = 5,
    should_halt: bool = False,
    allow_entries: bool = True,
    max_vol_ratio: float = 1.20,
    max_atr_expansion: float = 1.15,
    corr_shift: float = 0.05,
    deployment_mode: str = "shadow_live",
) -> None:
    _write_json(
        tmp_path / "execution_drift_status.json",
        {
            "execution_fidelity_score": fidelity,
            "execution_fidelity_level": "stable" if fidelity >= 80 else "watch" if fidelity >= 60 else "unstable",
            "reliable_for_capital": reliable_for_capital,
            "miss_rate": miss_rate,
            "avg_entry_slippage_bps": avg_entry_slippage_bps,
            "avg_fill_ratio": avg_fill_ratio,
            "avg_book_spread_bps": avg_book_spread_bps,
            "avg_book_impact_bps": avg_book_impact_bps,
        },
    )
    _write_json(
        tmp_path / "stress_field_state.json",
        {
            "hysteresis_score": hysteresis,
            "collapse_probability": collapse_probability,
            "collapse_horizon_ticks": collapse_horizon_ticks,
            "should_halt": should_halt,
            "allow_entries": allow_entries,
            "adversarial_input": {"intensity": adversary},
        },
    )
    _write_json(
        tmp_path / "market_snapshot.json",
        {
            "BTC/USDT": {"vol_ratio": max_vol_ratio, "atr_exp": max_atr_expansion},
            "ETH/USDT": {"vol_ratio": 1.10, "atr_exp": 1.05},
            "_cross_asset": {
                "mean_abs_corr_48h": 0.42,
                "corr_shift_48h": corr_shift,
                "dispersion_24h": 0.015,
            },
        },
    )
    _write_json(
        tmp_path / "deployment_gate_status.json",
        {
            "allowed_mode": deployment_mode,
            "recommended_max_total_exposure_pct": 0.010,
            "recommended_max_per_trade_pct": 0.0025,
        },
    )


def test_capital_firewall_allows_full_size_when_inputs_are_clean(tmp_path: Path) -> None:
    _seed_firewall_inputs(tmp_path)

    report = build_capital_firewall_report(
        tmp_path,
        operating_mode="live",
        configured_max_total_exposure_pct=0.012,
        configured_max_per_trade_pct=0.003,
    )

    assert report.decision == "allow_full_size"
    assert report.allow_new_entries is True
    assert report.max_total_exposure_pct == 0.012
    assert report.max_per_trade_pct == 0.003


def test_capital_firewall_reduces_size_when_stress_and_drift_are_elevated(tmp_path: Path) -> None:
    _seed_firewall_inputs(
        tmp_path,
        fidelity=68.0,
        miss_rate=0.09,
        avg_entry_slippage_bps=9.5,
        avg_fill_ratio=0.92,
        avg_book_spread_bps=14.0,
        hysteresis=0.42,
        adversary=0.46,
        collapse_probability=0.44,
        collapse_horizon_ticks=3,
        max_vol_ratio=1.75,
        max_atr_expansion=1.65,
        corr_shift=0.14,
    )

    report = build_capital_firewall_report(
        tmp_path,
        operating_mode="probation_live",
        configured_max_total_exposure_pct=0.010,
        configured_max_per_trade_pct=0.0025,
    )

    assert report.decision == "allow_reduced_size"
    assert report.max_total_exposure_pct < 0.010
    assert report.max_per_trade_pct < 0.0025
    assert report.allow_new_entries is True
    assert report.reasons


def test_capital_firewall_halts_on_imminent_collapse_even_if_deployment_gate_is_green(tmp_path: Path) -> None:
    _seed_firewall_inputs(
        tmp_path,
        fidelity=82.0,
        reliable_for_capital=True,
        collapse_probability=0.88,
        collapse_horizon_ticks=1,
        should_halt=True,
        deployment_mode="full_live",
    )

    report = build_capital_firewall_report(
        tmp_path,
        operating_mode="live",
        configured_max_total_exposure_pct=0.010,
        configured_max_per_trade_pct=0.0025,
    )

    assert report.decision == "no_trade"
    assert report.allow_new_entries is False
    assert report.max_total_exposure_pct == 0.0
    assert any("collapse" in reason.lower() or "halt" in reason.lower() for reason in report.reasons)


def test_capital_firewall_decision_is_independent_from_deployment_stage_label(tmp_path: Path) -> None:
    _seed_firewall_inputs(tmp_path, deployment_mode="shadow_live")

    report = build_capital_firewall_report(
        tmp_path,
        operating_mode="live",
        configured_max_total_exposure_pct=0.010,
        configured_max_per_trade_pct=0.0025,
    )

    assert report.deployment_allowed_mode == "shadow_live"
    assert report.decision == "allow_full_size"
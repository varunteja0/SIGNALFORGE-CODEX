from __future__ import annotations

import json
from pathlib import Path

from scripts import alerts


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_prime_state_suppresses_existing_alerts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(alerts, "STATE", tmp_path / "live_state.json")
    monkeypatch.setattr(alerts, "JOURNAL", tmp_path / "trade_journal.json")
    monkeypatch.setattr(alerts, "DIVERGENCE", tmp_path / "divergence_log.json")
    monkeypatch.setattr(alerts, "STRESS_KERNEL", tmp_path / "streaming_stress_kernel_status.json")
    monkeypatch.setattr(alerts, "STRESS_FIELD", tmp_path / "stress_field_state.json")
    monkeypatch.setattr(alerts, "STRESS_CONTEXT", tmp_path / "stress_context_status.json")
    monkeypatch.setattr(alerts, "DEPLOYMENT_GATE", tmp_path / "deployment_gate_status.json")
    monkeypatch.setattr(alerts, "EXECUTION_DRIFT", tmp_path / "execution_drift_status.json")
    monkeypatch.setattr(alerts, "CAPITAL_FIREWALL", tmp_path / "capital_firewall_status.json")

    _write(alerts.STATE, {"capital": 10_000, "initial_capital": 10_000, "open_positions": []})
    _write(alerts.JOURNAL, [])
    _write(alerts.DIVERGENCE, {"comparisons": [{"strategy": "funding_mr_v7", "timestamp": "2026-04-24T00:00:00Z"}]})
    _write(alerts.STRESS_KERNEL, {"pressure_level": "moderate", "continuous_pressure_score": 51.2, "probation_live_policy": {"stage": "blocked"}})
    _write(alerts.STRESS_CONTEXT, {"collapse_horizon_ticks": 4, "collapse_probability": 0.63})
    _write(alerts.STRESS_FIELD, {"adversarial_input": {"intensity": 0.61}})
    _write(alerts.DEPLOYMENT_GATE, {"allowed_mode": "shadow_live", "reasons": ["still paper"]})
    _write(alerts.EXECUTION_DRIFT, {"reliable_for_capital": False, "execution_fidelity_level": "unstable"})
    _write(alerts.CAPITAL_FIREWALL, {"decision": "no_trade", "max_total_exposure_pct": 0.0, "max_per_trade_pct": 0.0, "reasons": ["blocked"]})

    sent: list[tuple[str, str, str]] = []
    monkeypatch.setattr(alerts, "alert", lambda title, message, level="info", sound="Ping": sent.append((title, message, level)))

    monitor = alerts.AlertMonitor()
    monitor.prime_state()
    monitor.tick()

    assert sent == []


def test_tick_alerts_on_gate_change_after_prime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(alerts, "STATE", tmp_path / "live_state.json")
    monkeypatch.setattr(alerts, "JOURNAL", tmp_path / "trade_journal.json")
    monkeypatch.setattr(alerts, "DIVERGENCE", tmp_path / "divergence_log.json")
    monkeypatch.setattr(alerts, "STRESS_KERNEL", tmp_path / "streaming_stress_kernel_status.json")
    monkeypatch.setattr(alerts, "STRESS_FIELD", tmp_path / "stress_field_state.json")
    monkeypatch.setattr(alerts, "STRESS_CONTEXT", tmp_path / "stress_context_status.json")
    monkeypatch.setattr(alerts, "DEPLOYMENT_GATE", tmp_path / "deployment_gate_status.json")
    monkeypatch.setattr(alerts, "EXECUTION_DRIFT", tmp_path / "execution_drift_status.json")
    monkeypatch.setattr(alerts, "CAPITAL_FIREWALL", tmp_path / "capital_firewall_status.json")

    _write(alerts.STATE, {"capital": 10_000, "initial_capital": 10_000, "open_positions": []})
    _write(alerts.JOURNAL, [])
    _write(alerts.DIVERGENCE, {"comparisons": []})
    _write(alerts.STRESS_KERNEL, {"pressure_level": "moderate", "continuous_pressure_score": 51.2, "probation_live_policy": {"stage": "blocked"}})
    _write(alerts.STRESS_CONTEXT, {"collapse_horizon_ticks": 4, "collapse_probability": 0.63})
    _write(alerts.STRESS_FIELD, {"adversarial_input": {"intensity": 0.20}})
    _write(alerts.DEPLOYMENT_GATE, {"allowed_mode": "shadow_live", "reasons": []})
    _write(alerts.EXECUTION_DRIFT, {"reliable_for_capital": False, "execution_fidelity_level": "unstable"})
    _write(alerts.CAPITAL_FIREWALL, {"decision": "allow_reduced_size", "max_total_exposure_pct": 0.01, "max_per_trade_pct": 0.0025, "reasons": []})

    sent: list[tuple[str, str, str]] = []
    monkeypatch.setattr(alerts, "alert", lambda title, message, level="info", sound="Ping": sent.append((title, message, level)))

    monitor = alerts.AlertMonitor()
    monitor.prime_state()

    _write(alerts.DEPLOYMENT_GATE, {"allowed_mode": "probation_live", "reasons": []})
    monitor.tick()

    assert sent == [("Deployment Gate Shift", "Operational capital gate moved to probation_live", "info")]
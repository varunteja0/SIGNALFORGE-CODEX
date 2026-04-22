from __future__ import annotations

from pathlib import Path

from src.ops.stress_field_engine import (
    StressFieldEngine,
    load_stress_field_state,
    project_stress_context,
    write_stress_field_state,
)
from src.ops.streaming_stress_kernel import ProbationLivePolicy, StressKernelReport


def _report(
    *,
    generated_at: str,
    pressure_score: float,
    pressure_level: str,
    trajectory_score: float,
    transition_score: float,
    friction_score: float,
    liquidity_score: float,
    rejection_score: float,
    latency_score: float,
    latency_p999_ms: float,
    micro_collapse_probability: float,
    kill_switch_efficiency: float,
    stage: str,
    entry_action: str,
    synthetic_stress_intensity: float = 1.0,
    kill_switch_sensitivity_multiplier: float = 1.0,
    history_points: int = 144,
    trajectory_level: str | None = None,
) -> StressKernelReport:
    return StressKernelReport(
        generated_at=generated_at,
        history_points=history_points,
        trajectory_window=8,
        trajectory_novelty_score=trajectory_score,
        trajectory_novelty_level=trajectory_level or pressure_level,
        transition_stress_score=transition_score,
        micro_collapse_probability=micro_collapse_probability,
        execution_friction_score=friction_score,
        partial_fill_cascade_score=friction_score * 0.7,
        latency_jitter_score=latency_score,
        latency_p99_ms=max(latency_p999_ms * 0.6, 1.0),
        latency_p999_ms=latency_p999_ms,
        rejection_cluster_score=rejection_score,
        liquidity_vacuum_score=liquidity_score,
        correlation_collapse_score=trajectory_score * 0.7,
        kill_switch_efficiency=kill_switch_efficiency,
        kill_switch_event_count=5,
        average_detection_to_decision_ms=45.0,
        average_decision_to_protection_ms=20.0,
        false_positive_rate=0.04,
        missed_halt_count=0,
        continuous_pressure_score=pressure_score,
        pressure_level=pressure_level,
        predicted_pressure_failure=pressure_level in {"high", "critical"},
        probation_live_policy=ProbationLivePolicy(
            stage=stage,
            entry_action=entry_action,
            allow_probation_live=entry_action == "allow",
            allow_full_live=stage == "micro_live_ready",
            max_capital_fraction=0.01,
            max_total_exposure_pct=0.01,
            max_per_trade_pct=0.0025,
            entry_size_multiplier=0.7,
            exposure_multiplier=0.7,
            kill_switch_sensitivity_multiplier=kill_switch_sensitivity_multiplier,
            synthetic_stress_intensity=synthetic_stress_intensity,
            reasons=[stage],
        ),
        reasons=[pressure_level],
    )


def test_stress_field_engine_retains_hysteresis_across_ticks() -> None:
    engine = StressFieldEngine(paper_mode=True, probation_mode=False)
    high = _report(
        generated_at="2026-04-21T00:00:00+00:00",
        pressure_score=78.0,
        pressure_level="high",
        trajectory_score=82.0,
        transition_score=74.0,
        friction_score=76.0,
        liquidity_score=65.0,
        rejection_score=61.0,
        latency_score=72.0,
        latency_p999_ms=620.0,
        micro_collapse_probability=0.79,
        kill_switch_efficiency=0.74,
        stage="blocked",
        entry_action="halt",
        synthetic_stress_intensity=1.20,
        kill_switch_sensitivity_multiplier=1.35,
    )
    calm = _report(
        generated_at="2026-04-21T01:00:00+00:00",
        pressure_score=18.0,
        pressure_level="low",
        trajectory_score=16.0,
        transition_score=14.0,
        friction_score=12.0,
        liquidity_score=14.0,
        rejection_score=10.0,
        latency_score=14.0,
        latency_p999_ms=120.0,
        micro_collapse_probability=0.12,
        kill_switch_efficiency=0.96,
        stage="micro_live_ready",
        entry_action="allow",
    )

    first = engine.evolve(high, source_generated_at="tick-1")
    second = engine.evolve(calm, source_generated_at="tick-2")

    assert first.should_halt is True
    assert second.hysteresis_score > 0.0
    assert second.latency_memory > 0.0
    assert second.pressure_score > calm.continuous_pressure_score


def test_probation_phase_is_more_sensitive_than_paper() -> None:
    report = _report(
        generated_at="2026-04-21T00:00:00+00:00",
        pressure_score=42.0,
        pressure_level="moderate",
        trajectory_score=44.0,
        transition_score=38.0,
        friction_score=49.0,
        liquidity_score=46.0,
        rejection_score=41.0,
        latency_score=52.0,
        latency_p999_ms=420.0,
        micro_collapse_probability=0.48,
        kill_switch_efficiency=0.87,
        stage="plm_0.10",
        entry_action="allow",
        synthetic_stress_intensity=1.15,
        kill_switch_sensitivity_multiplier=1.30,
    )

    paper_state = StressFieldEngine(paper_mode=True, probation_mode=False).evolve(report, source_generated_at="same")
    probation_state = StressFieldEngine(paper_mode=False, probation_mode=True).evolve(report, source_generated_at="same")

    assert probation_state.phase.startswith("probation_field")
    assert probation_state.propagation_speed > paper_state.propagation_speed
    assert probation_state.execution_inertia > paper_state.execution_inertia
    assert probation_state.execution_profile.latency_multiplier > paper_state.execution_profile.latency_multiplier


def test_paper_warmup_ignores_bootstrap_only_policy_halt() -> None:
    report = _report(
        generated_at="2026-04-21T00:00:00+00:00",
        pressure_score=36.0,
        pressure_level="moderate",
        trajectory_score=100.0,
        transition_score=100.0,
        friction_score=0.0,
        liquidity_score=0.0,
        rejection_score=0.0,
        latency_score=0.0,
        latency_p999_ms=0.0,
        micro_collapse_probability=0.878,
        kill_switch_efficiency=1.0,
        stage="blocked",
        entry_action="halt",
        history_points=1,
        trajectory_level="insufficient_history",
    )

    paper_state = StressFieldEngine(paper_mode=True, probation_mode=False).evolve(report, source_generated_at="tick-1")
    probation_state = StressFieldEngine(paper_mode=False, probation_mode=True).evolve(report, source_generated_at="tick-1")

    assert paper_state.policy_stage == "paper_warmup"
    assert paper_state.should_halt is False
    assert paper_state.allow_entries is True
    assert paper_state.entry_action == "allow"
    assert probation_state.should_halt is True


def test_stress_field_state_round_trips_and_projects(tmp_path: Path) -> None:
    report = _report(
        generated_at="2026-04-21T00:00:00+00:00",
        pressure_score=33.0,
        pressure_level="moderate",
        trajectory_score=30.0,
        transition_score=26.0,
        friction_score=36.0,
        liquidity_score=34.0,
        rejection_score=28.0,
        latency_score=40.0,
        latency_p999_ms=300.0,
        micro_collapse_probability=0.34,
        kill_switch_efficiency=0.91,
        stage="plm_0.50",
        entry_action="allow",
        synthetic_stress_intensity=1.05,
    )
    state = StressFieldEngine(paper_mode=True, probation_mode=True).evolve(report, source_generated_at="tick-3")

    path = tmp_path / "stress_field_state.json"
    write_stress_field_state(state, path)
    restored = load_stress_field_state(path)

    assert restored is not None
    assert restored.phase == state.phase
    assert restored.hysteresis_score == state.hysteresis_score

    context = project_stress_context(restored)
    assert context.policy_stage == state.policy_stage
    assert context.collapse_horizon_ticks == state.collapse_horizon_ticks
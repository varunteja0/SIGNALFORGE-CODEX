from __future__ import annotations

import numpy as np

from src.execution.smart import SmartExecutionEngine
from src.ops.stress_context import build_stress_context
from src.ops.streaming_stress_kernel import ProbationLivePolicy, StressKernelReport


def _report(
    *,
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
    entry_size_multiplier: float = 1.0,
    exposure_multiplier: float = 1.0,
    synthetic_stress_intensity: float = 1.0,
    kill_switch_sensitivity_multiplier: float = 1.0,
) -> StressKernelReport:
    return StressKernelReport(
        generated_at="2026-04-21T00:00:00+00:00",
        history_points=144,
        trajectory_window=8,
        trajectory_novelty_score=trajectory_score,
        trajectory_novelty_level=pressure_level,
        transition_stress_score=transition_score,
        micro_collapse_probability=micro_collapse_probability,
        execution_friction_score=friction_score,
        partial_fill_cascade_score=friction_score * 0.8,
        latency_jitter_score=latency_score,
        latency_p99_ms=max(latency_p999_ms * 0.6, 1.0),
        latency_p999_ms=latency_p999_ms,
        rejection_cluster_score=rejection_score,
        liquidity_vacuum_score=liquidity_score,
        correlation_collapse_score=max(trajectory_score * 0.75, 0.0),
        kill_switch_efficiency=kill_switch_efficiency,
        kill_switch_event_count=6,
        average_detection_to_decision_ms=45.0,
        average_decision_to_protection_ms=20.0,
        false_positive_rate=0.05,
        missed_halt_count=0,
        continuous_pressure_score=pressure_score,
        pressure_level=pressure_level,
        predicted_pressure_failure=pressure_level in {"high", "critical"} and micro_collapse_probability >= 0.70,
        probation_live_policy=ProbationLivePolicy(
            stage=stage,
            entry_action=entry_action,
            allow_probation_live=entry_action == "allow",
            allow_full_live=stage == "micro_live_ready",
            max_capital_fraction=0.01,
            max_total_exposure_pct=0.01,
            max_per_trade_pct=0.0025,
            entry_size_multiplier=entry_size_multiplier,
            exposure_multiplier=exposure_multiplier,
            kill_switch_sensitivity_multiplier=kill_switch_sensitivity_multiplier,
            synthetic_stress_intensity=synthetic_stress_intensity,
            reasons=[f"policy={stage}"],
        ),
        reasons=[f"pressure={pressure_level}"],
    )


def test_stress_context_predicts_collapse_topology() -> None:
    report = _report(
        pressure_score=88.0,
        pressure_level="critical",
        trajectory_score=91.0,
        transition_score=84.0,
        friction_score=79.0,
        liquidity_score=76.0,
        rejection_score=74.0,
        latency_score=82.0,
        latency_p999_ms=720.0,
        micro_collapse_probability=0.91,
        kill_switch_efficiency=0.58,
        stage="blocked",
        entry_action="halt",
        synthetic_stress_intensity=1.20,
        kill_switch_sensitivity_multiplier=1.40,
    )

    context = build_stress_context(report, paper_mode=True, probation_mode=True)

    assert context.should_halt is True
    assert context.allow_entries is False
    assert context.collapse_horizon_ticks == 1
    assert context.execution_profile.preferred_algo in {"vwap", "twap"}
    assert context.execution_profile.slippage_multiplier > 1.0
    assert context.execution_profile.book_depth_multiplier < 1.0


def test_smart_execution_runs_inside_stress_context() -> None:
    report = _report(
        pressure_score=56.0,
        pressure_level="moderate",
        trajectory_score=58.0,
        transition_score=47.0,
        friction_score=71.0,
        liquidity_score=68.0,
        rejection_score=60.0,
        latency_score=62.0,
        latency_p999_ms=480.0,
        micro_collapse_probability=0.63,
        kill_switch_efficiency=0.82,
        stage="plm_0.10",
        entry_action="allow",
        entry_size_multiplier=0.25,
        exposure_multiplier=0.25,
        synthetic_stress_intensity=1.15,
        kill_switch_sensitivity_multiplier=1.30,
    )
    context = build_stress_context(report, paper_mode=True, probation_mode=True)

    plain = SmartExecutionEngine(paper_mode=True, max_slippage_bps=250)
    stressed = SmartExecutionEngine(paper_mode=True, max_slippage_bps=250)
    stressed.set_stress_context(context)

    np.random.seed(7)
    plain_result = plain.execute_entry(
        "BTC/USDT",
        1,
        6.0,
        50_000.0,
        49_000.0,
        52_000.0,
        50_000.0,
        atr=500.0,
        book_depth_usd=500_000.0,
    )
    np.random.seed(7)
    stressed_result = stressed.execute_entry(
        "BTC/USDT",
        1,
        6.0,
        50_000.0,
        49_000.0,
        52_000.0,
        50_000.0,
        atr=500.0,
        book_depth_usd=500_000.0,
    )

    assert plain_result.success is True
    assert stressed_result.success is True
    assert stressed_result.slippage_bps > plain_result.slippage_bps
    assert stressed_result.execution_ms > plain_result.execution_ms
    assert stressed_result.size < plain_result.size
    assert stressed_result.metadata["stress_context"]["policy_stage"] == "plm_0.10"


def test_stress_context_can_bias_algorithm_choice() -> None:
    report = _report(
        pressure_score=49.0,
        pressure_level="moderate",
        trajectory_score=44.0,
        transition_score=40.0,
        friction_score=65.0,
        liquidity_score=72.0,
        rejection_score=55.0,
        latency_score=48.0,
        latency_p999_ms=420.0,
        micro_collapse_probability=0.57,
        kill_switch_efficiency=0.86,
        stage="plm_0.50",
        entry_action="allow",
        entry_size_multiplier=0.50,
        exposure_multiplier=0.50,
        synthetic_stress_intensity=1.10,
    )
    context = build_stress_context(report, paper_mode=True, probation_mode=True)
    engine = SmartExecutionEngine(paper_mode=True)
    engine.set_stress_context(context)

    algo = engine.choose_algo(order_notional=15_000.0, urgency="normal", book_depth_usd=500_000.0)

    assert algo in {"limit_bias", "vwap", "twap"}
    assert algo != "market"

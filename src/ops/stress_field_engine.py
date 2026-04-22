from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ops.stress_context import StressContext, StressExecutionProfile, StressFieldTensor
from src.ops.streaming_stress_kernel import ProbationLivePolicy, StressKernelReport


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _score_level(score: float) -> str:
    if score >= 75.0:
        return "critical"
    if score >= 60.0:
        return "high"
    if score >= 35.0:
        return "moderate"
    return "low"


@dataclass(frozen=True)
class AdversarialStressInput:
    intensity: float = 0.0
    liquidity_impulse: float = 0.0
    latency_impulse: float = 0.0
    regime_impulse: float = 0.0
    friction_impulse: float = 0.0
    collapse_impulse: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StressFieldPhaseConfig:
    name: str
    persistence: float
    hysteresis_decay: float
    hysteresis_gain: float
    propagation_base: float
    latency_feedback: float
    adversarial_gain: float
    execution_inertia_base: float
    pause_pressure_score: float
    halt_collapse_probability: float
    halt_horizon_ticks: int


@dataclass(frozen=True)
class StressFieldState:
    generated_at: str
    source_generated_at: str
    paper_mode: bool
    probation_mode: bool
    tick_index: int = 0
    phase: str = "paper_reality_lab"
    policy_stage: str = "shadow"
    pressure_score: float = 0.0
    pressure_level: str = "low"
    collapse_probability: float = 0.0
    collapse_horizon_ticks: int = 6
    hysteresis_score: float = 0.0
    propagation_speed: float = 1.0
    execution_inertia: float = 0.0
    latency_memory: float = 0.0
    should_halt: bool = False
    allow_entries: bool = True
    entry_action: str = "allow"
    tensor: StressFieldTensor = field(default_factory=StressFieldTensor)
    execution_profile: StressExecutionProfile = field(default_factory=StressExecutionProfile)
    adversarial_input: AdversarialStressInput = field(default_factory=AdversarialStressInput)
    reasons: list[str] = field(default_factory=list)
    policy_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StressFieldEngine:
    def __init__(
        self,
        *,
        paper_mode: bool,
        probation_mode: bool,
        initial_state: StressFieldState | None = None,
    ) -> None:
        self.paper_mode = paper_mode
        self.probation_mode = probation_mode
        self.state = initial_state if self._matches_mode(initial_state) else None

    def _matches_mode(self, state: StressFieldState | None) -> bool:
        return bool(
            state is not None
            and state.paper_mode == self.paper_mode
            and state.probation_mode == self.probation_mode
        )

    def evolve(
        self,
        report: StressKernelReport,
        *,
        source_generated_at: str | None = None,
        generated_at: str | None = None,
    ) -> StressFieldState:
        source_generated_at = source_generated_at or report.generated_at
        if self.state is not None and self.state.source_generated_at == source_generated_at:
            return self.state

        phase = _phase_for_mode(
            report.probation_live_policy,
            paper_mode=self.paper_mode,
            probation_mode=self.probation_mode,
        )
        previous = self.state if self._matches_mode(self.state) else None
        adversarial = _build_adversarial_input(report, previous, phase)
        state = _evolve_state(
            report,
            previous,
            phase,
            adversarial,
            paper_mode=self.paper_mode,
            probation_mode=self.probation_mode,
            source_generated_at=source_generated_at,
            generated_at=generated_at or _now_iso(),
        )
        self.state = state
        return state

    def project_context(self) -> StressContext | None:
        if self.state is None:
            return None
        return project_stress_context(self.state)


def _phase_for_mode(
    policy: ProbationLivePolicy,
    *,
    paper_mode: bool,
    probation_mode: bool,
) -> StressFieldPhaseConfig:
    if probation_mode:
        persistence = 0.84
        hysteresis_decay = 0.90
        hysteresis_gain = 0.56
        propagation_base = 1.35
        latency_feedback = 0.58
        adversarial_gain = 0.52
        execution_inertia_base = 0.36
        pause_pressure_score = 46.0
        halt_collapse_probability = 0.82
        name = "probation_field"
    elif paper_mode:
        persistence = 0.72
        hysteresis_decay = 0.82
        hysteresis_gain = 0.44
        propagation_base = 1.15
        latency_feedback = 0.42
        adversarial_gain = 0.44
        execution_inertia_base = 0.26
        pause_pressure_score = 54.0
        halt_collapse_probability = 0.88
        name = "paper_field"
    else:
        persistence = 0.64
        hysteresis_decay = 0.76
        hysteresis_gain = 0.36
        propagation_base = 1.00
        latency_feedback = 0.36
        adversarial_gain = 0.28
        execution_inertia_base = 0.18
        pause_pressure_score = 58.0
        halt_collapse_probability = 0.86
        name = "live_field"

    sensitivity = max(float(policy.kill_switch_sensitivity_multiplier) - 1.0, 0.0)
    persistence = _bounded(persistence + sensitivity * 0.10, 0.55, 0.94)
    hysteresis_decay = _bounded(hysteresis_decay + sensitivity * 0.08, 0.60, 0.96)
    propagation_base = _bounded(propagation_base + sensitivity * 0.35, 0.80, 2.20)
    latency_feedback = _bounded(latency_feedback + sensitivity * 0.15, 0.20, 0.85)
    adversarial_gain = _bounded(adversarial_gain + max(float(policy.synthetic_stress_intensity) - 1.0, 0.0) * 0.40, 0.10, 0.90)
    execution_inertia_base = _bounded(execution_inertia_base + sensitivity * 0.10, 0.05, 0.65)

    if str(policy.stage).startswith("plm_"):
        name = f"{name}:{policy.stage}"
    elif policy.stage == "micro_live_ready":
        name = f"{name}:micro_live"

    return StressFieldPhaseConfig(
        name=name,
        persistence=persistence,
        hysteresis_decay=hysteresis_decay,
        hysteresis_gain=hysteresis_gain,
        propagation_base=propagation_base,
        latency_feedback=latency_feedback,
        adversarial_gain=adversarial_gain,
        execution_inertia_base=execution_inertia_base,
        pause_pressure_score=pause_pressure_score,
        halt_collapse_probability=halt_collapse_probability,
        halt_horizon_ticks=1,
    )


def _build_adversarial_input(
    report: StressKernelReport,
    previous: StressFieldState | None,
    phase: StressFieldPhaseConfig,
) -> AdversarialStressInput:
    prev_latency = previous.latency_memory if previous is not None else 0.0
    prev_collapse = previous.tensor.collapse_hazard if previous is not None else 0.0
    prev_hysteresis = previous.hysteresis_score if previous is not None else 0.0
    policy = report.probation_live_policy

    base_intensity = _clip01(
        0.18 * report.trajectory_novelty_score / 100.0
        + 0.14 * report.transition_stress_score / 100.0
        + 0.18 * report.execution_friction_score / 100.0
        + 0.16 * report.latency_jitter_score / 100.0
        + 0.10 * report.rejection_cluster_score / 100.0
        + 0.10 * report.liquidity_vacuum_score / 100.0
        + 0.08 * max(float(policy.synthetic_stress_intensity) - 1.0, 0.0)
        + 0.06 * prev_latency
    )
    intensity = _clip01(
        base_intensity * (1.0 + phase.adversarial_gain)
        + prev_collapse * 0.12
        + prev_hysteresis * 0.10
    )

    liquidity_impulse = _clip01(
        0.70 * report.liquidity_vacuum_score / 100.0
        + 0.18 * report.rejection_cluster_score / 100.0
        + 0.20 * intensity
    )
    latency_impulse = _clip01(
        0.62 * report.latency_jitter_score / 100.0
        + 0.18 * min(report.latency_p999_ms / 700.0, 1.0)
        + 0.14 * prev_latency
        + 0.16 * intensity
    )
    regime_impulse = _clip01(
        0.45 * report.trajectory_novelty_score / 100.0
        + 0.30 * report.transition_stress_score / 100.0
        + 0.15 * report.correlation_collapse_score / 100.0
        + 0.10 * intensity
    )
    friction_impulse = _clip01(
        0.58 * report.execution_friction_score / 100.0
        + 0.14 * report.partial_fill_cascade_score / 100.0
        + 0.10 * report.rejection_cluster_score / 100.0
        + 0.18 * intensity
    )
    collapse_impulse = _clip01(
        0.42 * report.micro_collapse_probability
        + 0.18 * report.continuous_pressure_score / 100.0
        + 0.16 * max(0.0, 1.0 - report.kill_switch_efficiency)
        + 0.12 * prev_collapse
        + 0.12 * intensity
    )

    reasons: list[str] = []
    if intensity >= 0.70:
        reasons.append("Adversarial injector is actively amplifying the local stress field.")
    if latency_impulse >= 0.65:
        reasons.append("Latency shocks are feeding the next stress propagation step.")
    if liquidity_impulse >= 0.65:
        reasons.append("Liquidity distortion is being treated as an active adversarial force.")

    return AdversarialStressInput(
        intensity=intensity,
        liquidity_impulse=liquidity_impulse,
        latency_impulse=latency_impulse,
        regime_impulse=regime_impulse,
        friction_impulse=friction_impulse,
        collapse_impulse=collapse_impulse,
        reasons=reasons,
    )


def _evolve_state(
    report: StressKernelReport,
    previous: StressFieldState | None,
    phase: StressFieldPhaseConfig,
    adversarial: AdversarialStressInput,
    *,
    paper_mode: bool,
    probation_mode: bool,
    source_generated_at: str,
    generated_at: str,
) -> StressFieldState:
    prev_tensor = previous.tensor if previous is not None else StressFieldTensor()
    prev_tick = previous.tick_index if previous is not None else 0
    prev_pressure = previous.pressure_score / 100.0 if previous is not None else 0.0
    prev_hysteresis = previous.hysteresis_score if previous is not None else 0.0
    prev_latency_memory = previous.latency_memory if previous is not None else 0.0
    prev_inertia = previous.execution_inertia if previous is not None else 0.0

    hysteresis_score = _clip01(
        prev_hysteresis * phase.hysteresis_decay
        + abs(prev_pressure - report.continuous_pressure_score / 100.0) * phase.hysteresis_gain
        + prev_latency_memory * 0.08
    )
    latency_memory = _clip01(
        prev_latency_memory * phase.hysteresis_decay
        + adversarial.latency_impulse * (1.0 - phase.hysteresis_decay)
        + prev_tensor.latency_amplification * phase.latency_feedback * 0.20
        + report.latency_jitter_score / 100.0 * 0.18
    )

    liquidity = _clip01(
        prev_tensor.liquidity_distortion * phase.persistence
        + adversarial.liquidity_impulse * (1.0 - phase.persistence)
        + latency_memory * phase.latency_feedback * 0.08
        + hysteresis_score * 0.06
    )
    regime = _clip01(
        prev_tensor.regime_volatility_curvature * phase.persistence
        + adversarial.regime_impulse * (1.0 - phase.persistence)
        + hysteresis_score * 0.10
        + report.trajectory_novelty_score / 100.0 * 0.08
    )
    friction = _clip01(
        prev_tensor.execution_friction_density * phase.persistence
        + adversarial.friction_impulse * (1.0 - phase.persistence)
        + latency_memory * 0.12
        + liquidity * 0.10
    )
    collapse = _clip01(
        prev_tensor.collapse_hazard * phase.persistence
        + adversarial.collapse_impulse * (1.0 - phase.persistence)
        + 0.14 * latency_memory
        + 0.12 * friction
        + 0.10 * regime
        + 0.08 * hysteresis_score
    )

    propagation_speed = _bounded(
        phase.propagation_base * (1.0 + 0.50 * latency_memory + 0.32 * hysteresis_score + 0.35 * adversarial.intensity),
        0.8,
        4.5,
    )
    execution_inertia = _clip01(
        prev_inertia * phase.persistence
        + phase.execution_inertia_base * (0.55 + 0.45 * phase.persistence)
        + 0.16 * latency_memory
        + 0.14 * friction
        + 0.08 * hysteresis_score
    )

    pressure_score = 100.0 * _clip01(
        0.14 * report.continuous_pressure_score / 100.0
        + 0.14 * liquidity
        + 0.14 * latency_memory
        + 0.14 * regime
        + 0.16 * friction
        + 0.16 * collapse
        + 0.06 * hysteresis_score
        + 0.06 * adversarial.intensity
    )
    pressure_level = _score_level(pressure_score)
    collapse_probability = _clip01(
        0.38 * collapse
        + 0.18 * latency_memory
        + 0.14 * friction
        + 0.10 * regime
        + 0.10 * hysteresis_score
        + 0.10 * adversarial.intensity
    )

    if collapse_probability >= 0.90:
        collapse_horizon_ticks = 1
    elif collapse_probability >= 0.78 or propagation_speed >= 2.80:
        collapse_horizon_ticks = 2
    elif collapse_probability >= 0.64 or propagation_speed >= 2.20:
        collapse_horizon_ticks = 3
    elif collapse_probability >= 0.48:
        collapse_horizon_ticks = 4
    else:
        collapse_horizon_ticks = 6
    collapse_horizon_ticks = max(
        1,
        collapse_horizon_ticks - int(latency_memory >= 0.70) - int(hysteresis_score >= 0.72),
    )

    policy = report.probation_live_policy
    warmup_policy_override = (
        paper_mode
        and not probation_mode
        and str(policy.stage) == "blocked"
        and str(policy.entry_action) == "halt"
        and str(report.trajectory_novelty_level) == "insufficient_history"
    )
    policy_stage = "paper_warmup" if warmup_policy_override else str(policy.stage)
    preferred_algo = "auto"
    if liquidity >= 0.72 or friction >= 0.72:
        preferred_algo = "twap" if collapse_probability >= 0.72 else "vwap"
    elif liquidity >= 0.48 or latency_memory >= 0.55:
        preferred_algo = "limit_bias"

    entry_size_multiplier = _bounded(
        float(policy.entry_size_multiplier or 1.0)
        * (1.0 - 0.18 * hysteresis_score - 0.22 * latency_memory - 0.12 * execution_inertia),
        0.0,
        1.0,
    )
    exposure_multiplier = _bounded(
        float(policy.exposure_multiplier or 1.0)
        * (1.0 - 0.16 * hysteresis_score - 0.18 * collapse_probability - 0.10 * execution_inertia),
        0.0,
        1.0,
    )
    execution_profile = StressExecutionProfile(
        slippage_multiplier=_bounded(1.0 + 1.10 * friction + 0.72 * liquidity + 0.30 * regime + 0.16 * adversarial.intensity, 1.0, 5.5),
        latency_multiplier=_bounded(1.0 + 1.28 * latency_memory + 0.22 * execution_inertia + 0.18 * (propagation_speed - 1.0), 1.0, 6.0),
        book_depth_multiplier=_bounded(1.0 - 0.48 * liquidity - 0.18 * friction - 0.10 * regime - 0.08 * adversarial.intensity, 0.05, 1.0),
        fill_ratio_multiplier=_bounded(1.0 - 0.34 * friction - 0.22 * liquidity - 0.16 * latency_memory - 0.10 * execution_inertia, 0.12, 1.0),
        price_gap_multiplier=_bounded(1.0 - 0.28 * regime - 0.20 * hysteresis_score - 0.14 * collapse_probability, 0.20, 1.0),
        entry_size_multiplier=entry_size_multiplier,
        exposure_multiplier=exposure_multiplier,
        preferred_algo=preferred_algo,
    )

    entry_action = "allow" if warmup_policy_override else str(policy.entry_action)
    allow_entries = entry_action == "allow"
    should_halt = entry_action == "halt"
    reasons = list(report.reasons) + list(adversarial.reasons)
    if should_halt:
        allow_entries = False
        reasons.append("Kernel policy already requires an immediate halt under the current field topology.")
    elif warmup_policy_override:
        reasons.append("Paper-mode warmup is observing trades while the stress trajectory baseline is still bootstrapping.")
    elif collapse_probability >= phase.halt_collapse_probability or collapse_horizon_ticks <= phase.halt_horizon_ticks:
        entry_action = "halt"
        allow_entries = False
        should_halt = True
        reasons.append(f"Stress field predicts collapse within {collapse_horizon_ticks} ticks.")
    elif pressure_score >= phase.pause_pressure_score or hysteresis_score >= 0.72 or latency_memory >= 0.76:
        entry_action = "pause_entries"
        allow_entries = False
        reasons.append("Stress field hysteresis remains elevated, so entries stay paused.")

    return StressFieldState(
        generated_at=generated_at,
        source_generated_at=source_generated_at,
        paper_mode=paper_mode,
        probation_mode=probation_mode,
        tick_index=prev_tick + 1,
        phase=phase.name,
        policy_stage=policy_stage,
        pressure_score=float(pressure_score),
        pressure_level=pressure_level,
        collapse_probability=float(collapse_probability),
        collapse_horizon_ticks=collapse_horizon_ticks,
        hysteresis_score=float(hysteresis_score),
        propagation_speed=float(propagation_speed),
        execution_inertia=float(execution_inertia),
        latency_memory=float(latency_memory),
        should_halt=should_halt,
        allow_entries=allow_entries,
        entry_action=entry_action,
        tensor=StressFieldTensor(
            liquidity_distortion=float(liquidity),
            latency_amplification=float(latency_memory),
            regime_volatility_curvature=float(regime),
            execution_friction_density=float(friction),
            collapse_hazard=float(collapse),
        ),
        execution_profile=execution_profile,
        adversarial_input=adversarial,
        reasons=reasons,
        policy_reasons=list(policy.reasons),
    )


def project_stress_context(state: StressFieldState) -> StressContext:
    return StressContext(
        generated_at=state.generated_at,
        source_generated_at=state.source_generated_at,
        paper_mode=state.paper_mode,
        probation_mode=state.probation_mode,
        pressure_score=state.pressure_score,
        pressure_level=state.pressure_level,
        collapse_probability=state.collapse_probability,
        collapse_horizon_ticks=state.collapse_horizon_ticks,
        should_halt=state.should_halt,
        allow_entries=state.allow_entries,
        entry_action=state.entry_action,
        policy_stage=state.policy_stage,
        tensor=state.tensor,
        execution_profile=state.execution_profile,
        reasons=list(state.reasons),
        policy_reasons=list(state.policy_reasons),
    )


def format_stress_field_state(state: StressFieldState) -> str:
    lines = [
        "Stress Field State",
        f"  Phase: {state.phase} | policy={state.policy_stage} | tick={state.tick_index}",
        f"  Pressure: {state.pressure_level} {state.pressure_score:.1f}/100 | collapse={state.collapse_probability:.0%} in ~{state.collapse_horizon_ticks} ticks",
        f"  Memory: hysteresis={state.hysteresis_score:.0%} | latency={state.latency_memory:.0%} | propagation={state.propagation_speed:.2f}x | inertia={state.execution_inertia:.0%}",
        f"  Tensor: liquidity={state.tensor.liquidity_distortion:.0%} | friction={state.tensor.execution_friction_density:.0%} | regime={state.tensor.regime_volatility_curvature:.0%} | collapse={state.tensor.collapse_hazard:.0%}",
        f"  Adversary: intensity={state.adversarial_input.intensity:.0%} | latency={state.adversarial_input.latency_impulse:.0%} | liquidity={state.adversarial_input.liquidity_impulse:.0%}",
        f"  Execution: slip={state.execution_profile.slippage_multiplier:.2f}x | latency={state.execution_profile.latency_multiplier:.2f}x | depth={state.execution_profile.book_depth_multiplier:.2f}x | fill={state.execution_profile.fill_ratio_multiplier:.2f}x | algo={state.execution_profile.preferred_algo}",
    ]
    if state.reasons:
        lines.append("  Reasons:")
        for reason in state.reasons[:5]:
            lines.append(f"    - {reason}")
    return "\n".join(lines)


def _state_from_dict(payload: dict[str, Any]) -> StressFieldState:
    return StressFieldState(
        generated_at=str(payload.get("generated_at", "")),
        source_generated_at=str(payload.get("source_generated_at", payload.get("generated_at", ""))),
        paper_mode=bool(payload.get("paper_mode", True)),
        probation_mode=bool(payload.get("probation_mode", False)),
        tick_index=int(payload.get("tick_index", 0) or 0),
        phase=str(payload.get("phase", "paper_reality_lab")),
        policy_stage=str(payload.get("policy_stage", "shadow")),
        pressure_score=float(payload.get("pressure_score", 0.0) or 0.0),
        pressure_level=str(payload.get("pressure_level", "low")),
        collapse_probability=float(payload.get("collapse_probability", 0.0) or 0.0),
        collapse_horizon_ticks=int(payload.get("collapse_horizon_ticks", 6) or 6),
        hysteresis_score=float(payload.get("hysteresis_score", 0.0) or 0.0),
        propagation_speed=float(payload.get("propagation_speed", 1.0) or 1.0),
        execution_inertia=float(payload.get("execution_inertia", 0.0) or 0.0),
        latency_memory=float(payload.get("latency_memory", 0.0) or 0.0),
        should_halt=bool(payload.get("should_halt", False)),
        allow_entries=bool(payload.get("allow_entries", True)),
        entry_action=str(payload.get("entry_action", "allow")),
        tensor=StressFieldTensor(**(payload.get("tensor") or {})),
        execution_profile=StressExecutionProfile(**(payload.get("execution_profile") or {})),
        adversarial_input=AdversarialStressInput(**(payload.get("adversarial_input") or {})),
        reasons=list(payload.get("reasons", []) or []),
        policy_reasons=list(payload.get("policy_reasons", []) or []),
    )


def load_stress_field_state(path: Path) -> StressFieldState | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return _state_from_dict(payload)
    except Exception:
        return None


def write_stress_field_state(state: StressFieldState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2))


def append_stress_field_state(state: StressFieldState, path: Path) -> None:
    payload = state.to_dict()
    last_payload = None
    if path.exists():
        try:
            lines = path.read_text().splitlines()
            if lines:
                last_payload = json.loads(lines[-1])
        except (OSError, json.JSONDecodeError):
            last_payload = None
    if last_payload == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


__all__ = [
    "AdversarialStressInput",
    "StressFieldEngine",
    "StressFieldPhaseConfig",
    "StressFieldState",
    "append_stress_field_state",
    "format_stress_field_state",
    "load_stress_field_state",
    "project_stress_context",
    "write_stress_field_state",
]

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.ops.streaming_stress_kernel import StressKernelReport


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class StressFieldTensor:
    liquidity_distortion: float = 0.0
    latency_amplification: float = 0.0
    regime_volatility_curvature: float = 0.0
    execution_friction_density: float = 0.0
    collapse_hazard: float = 0.0


@dataclass(frozen=True)
class StressExecutionProfile:
    slippage_multiplier: float = 1.0
    latency_multiplier: float = 1.0
    book_depth_multiplier: float = 1.0
    fill_ratio_multiplier: float = 1.0
    price_gap_multiplier: float = 1.0
    entry_size_multiplier: float = 1.0
    exposure_multiplier: float = 1.0
    preferred_algo: str = "auto"


@dataclass(frozen=True)
class StressContext:
    generated_at: str
    source_generated_at: str
    paper_mode: bool
    probation_mode: bool
    pressure_score: float
    pressure_level: str
    collapse_probability: float
    collapse_horizon_ticks: int
    should_halt: bool
    allow_entries: bool
    entry_action: str
    policy_stage: str
    tensor: StressFieldTensor = field(default_factory=StressFieldTensor)
    execution_profile: StressExecutionProfile = field(default_factory=StressExecutionProfile)
    reasons: list[str] = field(default_factory=list)
    policy_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def adjusted_book_depth(self, base_book_depth: float) -> float:
        return max(base_book_depth * self.execution_profile.book_depth_multiplier, 1_000.0)

    def adjusted_slippage_bps(self, base_slippage_bps: float) -> float:
        return max(base_slippage_bps * self.execution_profile.slippage_multiplier, 0.0)

    def adjusted_latency_ms(self, base_latency_ms: float) -> float:
        return max(base_latency_ms * self.execution_profile.latency_multiplier, 0.0)

    def adjusted_fill_ratio(self, base_fill_ratio: float) -> float:
        return _bounded(base_fill_ratio * self.execution_profile.fill_ratio_multiplier, 0.05, 1.0)

    def adjusted_price_gap_pct(self, base_gap_pct: float) -> float:
        return max(base_gap_pct * self.execution_profile.price_gap_multiplier, 0.10)

    def select_algo(self, default_algo: str, urgency: str) -> str:
        preferred = self.execution_profile.preferred_algo
        if preferred == "auto" or urgency == "high":
            return default_algo
        if preferred == "limit_bias" and urgency == "normal" and default_algo == "market":
            return "limit_bias"
        if preferred == "vwap" and default_algo == "market":
            return "vwap"
        if preferred == "twap" and default_algo in {"market", "vwap"}:
            return "twap"
        return default_algo

    def execution_metadata(self) -> dict[str, Any]:
        return {
            "pressure_score": self.pressure_score,
            "pressure_level": self.pressure_level,
            "collapse_probability": self.collapse_probability,
            "collapse_horizon_ticks": self.collapse_horizon_ticks,
            "entry_action": self.entry_action,
            "policy_stage": self.policy_stage,
            "slippage_multiplier": self.execution_profile.slippage_multiplier,
            "latency_multiplier": self.execution_profile.latency_multiplier,
            "book_depth_multiplier": self.execution_profile.book_depth_multiplier,
            "fill_ratio_multiplier": self.execution_profile.fill_ratio_multiplier,
        }


def build_stress_context(
    report: StressKernelReport,
    *,
    paper_mode: bool,
    probation_mode: bool,
) -> StressContext:
    policy = report.probation_live_policy
    stress_overlay = 1.0 + max(float(policy.synthetic_stress_intensity) - 1.0, 0.0) * 0.60
    if paper_mode:
        stress_overlay *= 1.10
    if probation_mode:
        stress_overlay *= max(1.0, float(policy.kill_switch_sensitivity_multiplier) * 0.95)

    liquidity = _clip01(
        max(report.liquidity_vacuum_score, report.rejection_cluster_score * 0.70) / 100.0 * stress_overlay
    )
    latency = _clip01(
        max(report.latency_jitter_score / 100.0, report.latency_p999_ms / 600.0) * stress_overlay
    )
    regime = _clip01(
        (
            0.45 * report.trajectory_novelty_score
            + 0.30 * report.transition_stress_score
            + 0.25 * report.correlation_collapse_score
        )
        / 100.0
        * stress_overlay
    )
    friction = _clip01(report.execution_friction_score / 100.0 * stress_overlay)
    collapse = _clip01(
        max(
            report.micro_collapse_probability,
            0.45 * report.continuous_pressure_score / 100.0
            + 0.25 * max(0.0, 1.0 - report.kill_switch_efficiency)
            + 0.30 * regime,
        )
        * stress_overlay
    )

    if collapse >= 0.92:
        collapse_horizon_ticks = 1
    elif collapse >= 0.78:
        collapse_horizon_ticks = 2
    elif collapse >= 0.62:
        collapse_horizon_ticks = 3
    elif collapse >= 0.46:
        collapse_horizon_ticks = 4
    else:
        collapse_horizon_ticks = 6

    preferred_algo = "auto"
    if liquidity >= 0.70 or friction >= 0.70:
        preferred_algo = "twap" if collapse >= 0.75 else "vwap"
    elif liquidity >= 0.45 and report.pressure_level in {"moderate", "high"}:
        preferred_algo = "limit_bias"

    entry_action = str(policy.entry_action)
    should_halt = bool(entry_action == "halt" or collapse_horizon_ticks <= 1)
    allow_entries = bool(entry_action not in {"pause_entries", "halt"} and not should_halt)
    if should_halt:
        entry_action = "halt"
        allow_entries = False

    execution_profile = StressExecutionProfile(
        slippage_multiplier=_bounded(1.0 + 1.20 * friction + 0.75 * liquidity + 0.30 * regime, 1.0, 4.5),
        latency_multiplier=_bounded(1.0 + 1.40 * latency + 0.45 * collapse, 1.0, 5.0),
        book_depth_multiplier=_bounded(1.0 - 0.55 * liquidity - 0.20 * friction - 0.10 * regime, 0.10, 1.0),
        fill_ratio_multiplier=_bounded(1.0 - 0.40 * friction - 0.30 * liquidity - 0.15 * collapse, 0.20, 1.0),
        price_gap_multiplier=_bounded(1.0 - 0.35 * regime - 0.15 * collapse, 0.30, 1.0),
        entry_size_multiplier=_bounded(float(policy.entry_size_multiplier or 1.0), 0.0, 1.0),
        exposure_multiplier=_bounded(float(policy.exposure_multiplier or 1.0), 0.0, 1.0),
        preferred_algo=preferred_algo,
    )

    reasons = list(report.reasons)
    if should_halt and collapse_horizon_ticks <= 1:
        reasons.append("Stress field topology implies a collapse manifold inside the next tick.")
    elif collapse_horizon_ticks <= 2:
        reasons.append(f"Stress field topology implies a collapse manifold within {collapse_horizon_ticks} ticks.")

    return StressContext(
        generated_at=report.generated_at,
        source_generated_at=report.generated_at,
        paper_mode=paper_mode,
        probation_mode=probation_mode,
        pressure_score=float(report.continuous_pressure_score),
        pressure_level=str(report.pressure_level),
        collapse_probability=float(collapse),
        collapse_horizon_ticks=collapse_horizon_ticks,
        should_halt=should_halt,
        allow_entries=allow_entries,
        entry_action=entry_action,
        policy_stage=str(policy.stage),
        tensor=StressFieldTensor(
            liquidity_distortion=liquidity,
            latency_amplification=latency,
            regime_volatility_curvature=regime,
            execution_friction_density=friction,
            collapse_hazard=collapse,
        ),
        execution_profile=execution_profile,
        reasons=reasons,
        policy_reasons=list(policy.reasons),
    )


__all__ = [
    "StressContext",
    "StressExecutionProfile",
    "StressFieldTensor",
    "build_stress_context",
]

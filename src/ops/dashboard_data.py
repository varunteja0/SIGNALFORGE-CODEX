"""
Pure data layer for the live dashboard.

All functions in this module are side-effect-free (apart from
filesystem reads) and contain no Streamlit calls — so they can be
imported and tested without spinning up a Streamlit session.

``scripts/live_dashboard.py`` is a thin rendering shell that imports
from here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ops.deployment_gate import DeploymentGateThresholds
from src.ops.shadow_live_comparator import ShadowLiveComparatorThresholds

# Resolve dashboard artifacts from the repository root, not the caller cwd.
_DEFAULT_BASE = Path(__file__).resolve().parents[2] / "fund_data"

DEFAULT_ASSETS: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]

_ROLLOUT_STAGE_ORDER: list[str] = [
    "paper_shadow",
    "plm_0.05",
    "plm_0.10",
    "plm_0.50",
    "micro_live_ready",
    "full_live_gate",
]

_ROLLOUT_STAGE_SPECS: dict[str, dict[str, Any]] = {
    "paper_shadow": {
        "label": "Paper / Shadow",
        "max_capital_fraction": 0.0,
        "max_total_exposure_pct": 0.0,
        "max_per_trade_pct": 0.0,
    },
    "plm_0.05": {
        "label": "Probation 0.05%",
        "max_capital_fraction": 0.0005,
        "max_total_exposure_pct": 0.0005,
        "max_per_trade_pct": 0.000125,
    },
    "plm_0.10": {
        "label": "Probation 0.10%",
        "max_capital_fraction": 0.001,
        "max_total_exposure_pct": 0.001,
        "max_per_trade_pct": 0.00025,
    },
    "plm_0.50": {
        "label": "Probation 0.50%",
        "max_capital_fraction": 0.005,
        "max_total_exposure_pct": 0.005,
        "max_per_trade_pct": 0.0010,
    },
    "micro_live_ready": {
        "label": "Probation 1.00%",
        "max_capital_fraction": 0.010,
        "max_total_exposure_pct": 0.010,
        "max_per_trade_pct": 0.0025,
    },
    "full_live_gate": {
        "label": "Full Live Gate",
        "max_capital_fraction": 0.0,
        "max_total_exposure_pct": 0.0,
        "max_per_trade_pct": 0.0,
    },
}

_PAUSE_AFTER_CONSECUTIVE_LOSSES = 3
_STEP_BACK_AFTER_CONSECUTIVE_LOSSES = 4
_HARD_HALT_AFTER_CONSECUTIVE_LOSSES = 5


# --------------------------------------------------------------------------
# Loaders — each returns a "safe" default on any error.
# --------------------------------------------------------------------------
def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _positive_min(*values: Any) -> float:
    usable = [_float(value) for value in values if _float(value) > 0.0]
    return min(usable) if usable else 0.0


def _remaining_int(current: Any, required: int) -> int:
    return max(int(required) - int(_float(current)), 0)


def _remaining_float(current: Any, required: float) -> float:
    return max(float(required) - _float(current), 0.0)


def _dedupe_reasons(reasons: list[Any]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for reason in reasons:
        text = str(reason or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _first_reasons(reasons: list[Any], *, limit: int = 2) -> list[str]:
    return _dedupe_reasons(list(reasons))[:limit]


def _safe_pct(value: Any) -> float:
    return max(_float(value), 0.0)


def _stage_rank(stage: str) -> int:
    try:
        return _ROLLOUT_STAGE_ORDER.index(stage)
    except ValueError:
        return 0


def _map_probation_stage(stage: str) -> str:
    mapping = {
        "blocked": "paper_shadow",
        "shadow": "paper_shadow",
        "plm_0.05": "plm_0.05",
        "plm_0.10": "plm_0.10",
        "plm_0.50": "plm_0.50",
        "micro_live_ready": "micro_live_ready",
    }
    return mapping.get(str(stage), "paper_shadow")


def _infer_probation_stage(capital_fraction: float) -> str:
    if capital_fraction >= 0.010:
        return "micro_live_ready"
    if capital_fraction >= 0.005:
        return "plm_0.50"
    if capital_fraction >= 0.001:
        return "plm_0.10"
    if capital_fraction >= 0.0005:
        return "plm_0.05"
    return "paper_shadow"


def _format_gate_summary(
    label: str,
    *,
    allowed: bool,
    trades_remaining: int,
    days_remaining: float,
    blockers: list[str],
) -> str:
    prefix = "GO" if allowed else "NO-GO"
    if allowed:
        return f"{prefix} {label}: gate is open."

    fragments: list[str] = []
    if trades_remaining > 0:
        fragments.append(f"{trades_remaining} fully quoted trades remaining")
    if days_remaining > 0.0:
        fragments.append(f"{days_remaining:.1f} days remaining")
    if blockers:
        fragments.append(f"blockers: {'; '.join(_first_reasons(blockers))}")
    if not fragments:
        fragments.append("current readiness artifacts are incomplete")
    return f"{prefix} {label}: " + " | ".join(fragments)


def _rollout_stage_rows(
    *,
    capital: float,
    current_stage: str,
    effective_capital_fraction: float,
    effective_total_exposure_pct: float,
    effective_per_trade_pct: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_rank = _stage_rank(current_stage)

    for idx, stage in enumerate(_ROLLOUT_STAGE_ORDER):
        spec = dict(_ROLLOUT_STAGE_SPECS[stage])
        if stage == "full_live_gate":
            spec["max_capital_fraction"] = effective_capital_fraction
            spec["max_total_exposure_pct"] = effective_total_exposure_pct
            spec["max_per_trade_pct"] = effective_per_trade_pct

        max_capital_fraction = _safe_pct(spec["max_capital_fraction"])
        max_total_exposure_pct = _safe_pct(spec["max_total_exposure_pct"])
        max_per_trade_pct = _safe_pct(spec["max_per_trade_pct"])
        starting_size_usd = capital * max_per_trade_pct
        max_stage_exposure_usd = capital * max_total_exposure_pct
        pause_loss_usd = min(2.0 * starting_size_usd, max_stage_exposure_usd)
        step_back_loss_usd = min(3.0 * starting_size_usd, max_stage_exposure_usd)
        hard_stop_loss_usd = max_stage_exposure_usd

        if stage == current_stage:
            status = "current"
        elif _stage_rank(stage) < current_rank:
            status = "cleared"
        elif _stage_rank(stage) == current_rank + 1:
            status = "next"
        else:
            status = "locked"

        next_stage = _ROLLOUT_STAGE_ORDER[idx + 1] if idx + 1 < len(_ROLLOUT_STAGE_ORDER) else ""
        next_label = _ROLLOUT_STAGE_SPECS[next_stage]["label"] if next_stage else "manual review"

        if stage == "paper_shadow":
            scale_up_rule = "Advance only after the probation gate turns GO and the capital firewall stops vetoing capital."
            stop_conditions = "Stay paper-only until shadow-live, certification, and deployment artifacts are present and green."
        elif stage == "full_live_gate":
            scale_up_rule = "Advance only with explicit manual approval after the full-live gate and capital firewall are both green."
            stop_conditions = (
                "Halt immediately on health halt, stress-field halt, deployment-gate downgrade, or capital-firewall no_trade."
            )
        else:
            scale_up_rule = (
                f"Advance to {next_label} only when the deployment gate, survivability ladder, stress-kernel policy, and capital firewall all support the higher tier."
            )
            stop_conditions = (
                f"Pause after ${pause_loss_usd:,.2f} realized loss or {_PAUSE_AFTER_CONSECUTIVE_LOSSES} consecutive live losses; "
                f"step back after ${step_back_loss_usd:,.2f} or {_STEP_BACK_AFTER_CONSECUTIVE_LOSSES} losses; "
                f"hard halt at ${hard_stop_loss_usd:,.2f}, {_HARD_HALT_AFTER_CONSECUTIVE_LOSSES} losses, health halt, or firewall no_trade."
            )

        rows.append(
            {
                "stage": stage,
                "label": spec["label"],
                "status": status,
                "max_capital_fraction": max_capital_fraction,
                "max_total_exposure_pct": max_total_exposure_pct,
                "max_per_trade_pct": max_per_trade_pct,
                "starting_size_usd": starting_size_usd,
                "max_stage_exposure_usd": max_stage_exposure_usd,
                "pause_loss_usd": pause_loss_usd,
                "step_back_loss_usd": step_back_loss_usd,
                "hard_stop_loss_usd": hard_stop_loss_usd,
                "scale_up_rule": scale_up_rule,
                "stop_conditions": stop_conditions,
            }
        )

    return rows


def load_state(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    return _load_json(base_dir / "live_state.json", {})


def load_journal(base_dir: Path = _DEFAULT_BASE) -> list[dict[str, Any]]:
    data = _load_json(base_dir / "trade_journal.json", [])
    return data if isinstance(data, list) else []


def load_divergence(base_dir: Path = _DEFAULT_BASE) -> list[dict[str, Any]]:
    data = _load_json(base_dir / "divergence_log.json", [])
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("comparisons", [])
    return []


def load_market_snapshot(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    data = _load_json(base_dir / "market_snapshot.json", {})
    if isinstance(data, dict):
        data = dict(data)  # don't mutate cached value
        data.pop("_timestamp", None)
        return data
    return {}


def load_health(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    data = _load_json(base_dir / "health.json", {})
    return data if isinstance(data, dict) else {}


def load_production_certification(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    data = _load_json(base_dir / "production_certification_status.json", {})
    return data if isinstance(data, dict) else {}


def load_deployment_gate(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    data = _load_json(base_dir / "deployment_gate_status.json", {})
    return data if isinstance(data, dict) else {}


def load_drift_intelligence(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    data = _load_json(base_dir / "drift_intelligence_status.json", {})
    return data if isinstance(data, dict) else {}


def load_execution_drift(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    data = _load_json(base_dir / "execution_drift_status.json", {})
    return data if isinstance(data, dict) else {}


def load_shadow_live_comparator(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    data = _load_json(base_dir / "shadow_live_comparator_status.json", {})
    return data if isinstance(data, dict) else {}


def load_capital_firewall(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    data = _load_json(base_dir / "capital_firewall_status.json", {})
    return data if isinstance(data, dict) else {}


def load_survivability(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    data = _load_json(base_dir / "survivability_status.json", {})
    return data if isinstance(data, dict) else {}


def load_streaming_stress_kernel(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    data = _load_json(base_dir / "streaming_stress_kernel_status.json", {})
    return data if isinstance(data, dict) else {}


def load_stress_field(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    data = _load_json(base_dir / "stress_field_state.json", {})
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------
# Derived metrics
# --------------------------------------------------------------------------
def portfolio_summary(state: dict[str, Any], journal: list[dict[str, Any]]) -> dict[str, Any]:
    """Condense live-state + journal into a single metrics bundle."""
    capital = float(state.get("capital", 10_000.0))
    initial = float(state.get("initial_capital", 10_000.0))
    ret = (capital - initial) / initial if initial > 0 else 0.0
    return {
        "capital": capital,
        "initial_capital": initial,
        "return_pct": ret,
        "n_open": len(state.get("open_positions", []) or []),
        "n_closed": len(journal or []),
        "iteration": int(state.get("iteration", 0)),
        "paper_mode": bool(state.get("paper_mode", True)),
    }


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def compute_signal_proximity(symbol: str, snap: dict[str, Any]) -> dict[str, float]:
    """How close each strategy is to firing on this asset (∈ [0, 1]).

    The values here are **UI heuristics**, not real trigger probabilities —
    they're scaled so that ``1.0`` means "at threshold". The definitions
    mirror the production strategies as of v4.0.
    """
    if not isinstance(snap, dict) or snap.get("error"):
        return {
            "funding_mr_v7": 0.0,
            "extreme_spike": 0.0,
            "fund_vol_squeeze": 0.0,
            "momentum_breakout": 0.0,
        }

    fz = abs(float(snap.get("funding_zscore", 0) or 0))
    regime = str(snap.get("regime", "") or "")
    bb = float(snap.get("bb_pctile", 50) or 50)
    vol = float(snap.get("vol_ratio", 1) or 1)
    atr_exp = float(snap.get("atr_exp", 1) or 1)
    price = float(snap.get("price", 0) or 0)
    ch_high = float(snap.get("ch_high", 0) or 0)
    ch_low = float(snap.get("ch_low", 0) or 0)

    prox: dict[str, float] = {}

    # funding_mr_v7: needs |z| >= 3.0 on any asset.
    prox["funding_mr_v7"] = _clip(fz / 3.0)

    # extreme_spike: needs |z| >= 4.0 + high_vol regime. BTC excluded.
    base = _clip(fz / 4.0)
    regime_ok = 1.0 if regime == "high_volatility" else 0.5
    prox["extreme_spike"] = 0.0 if symbol == "BTC/USDT" else base * regime_ok

    # fund_vol_squeeze: needs bb_pctile <= 15 + |z| >= 1.5; SOL only.
    if symbol != "SOL/USDT":
        prox["fund_vol_squeeze"] = 0.0
    else:
        sq = 1.0 if bb <= 15.0 else _clip(15.0 / max(bb, 1e-10))
        fz2 = _clip(fz / 1.5)
        prox["fund_vol_squeeze"] = min(sq, fz2)

    # momentum_breakout: breakout + ATR expansion + volume; ETH only.
    if symbol == "ETH/USDT" and ch_high > ch_low:
        # Distance from extremes, scaled to 1.0 at the breakout levels.
        edge = max(price - ch_low, ch_high - price) / max(ch_high - ch_low, 1e-10)
        breakout = (
            1.0
            if (price > ch_high * 0.999 or price < ch_low * 1.001)
            else _clip(edge * 2)
        )
        atr_pct = _clip(atr_exp / 1.5)
        vol_pct = _clip(vol / 1.3)
        prox["momentum_breakout"] = min(breakout, atr_pct, vol_pct)
    else:
        prox["momentum_breakout"] = 0.0

    return prox


def proximity_matrix(
    assets: list[str],
    snapshots: dict[str, Any],
    strategies: list[str] | None = None,
) -> tuple[list[str], list[list[float]], list[str]]:
    """Build a (assets × strategies) grid of proximity scores.

    Returns ``(asset_labels, matrix, strategy_keys)`` — ``matrix[i][j]`` is
    the proximity of strategy ``j`` on asset ``i``.
    """
    strategies = strategies or [
        "funding_mr_v7",
        "extreme_spike",
        "fund_vol_squeeze",
        "momentum_breakout",
    ]
    labels: list[str] = []
    matrix: list[list[float]] = []
    for sym in assets:
        labels.append(sym.split("/")[0])
        prox = compute_signal_proximity(sym, snapshots.get(sym, {}))
        matrix.append([prox.get(s, 0.0) for s in strategies])
    return labels, matrix, strategies


def build_live_readiness_snapshot(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    state = load_state(base_dir)
    journal = load_journal(base_dir)
    summary = portfolio_summary(state, journal)
    health = load_health(base_dir)
    certification = load_production_certification(base_dir)
    deployment_gate = load_deployment_gate(base_dir)
    execution_drift = load_execution_drift(base_dir)
    shadow_live = load_shadow_live_comparator(base_dir)
    capital_firewall = load_capital_firewall(base_dir)
    survivability = load_survivability(base_dir)
    stress_kernel = load_streaming_stress_kernel(base_dir)
    stress_field = load_stress_field(base_dir)
    shadow_execution = _load_json(base_dir / "shadow_execution_status.json", {})
    paper_validation = _load_json(base_dir / "paper_validation_status.json", {})

    gate_thresholds = DeploymentGateThresholds()
    comparator_thresholds = ShadowLiveComparatorThresholds()

    health_status = str(health.get("overall_status", "unknown")) if health else "unknown"
    operating_mode = str(
        state.get("operating_mode")
        or ("paper" if summary.get("paper_mode", True) else "unknown")
    )
    paper_validation_ready = bool(
        paper_validation.get("ready_for_live", deployment_gate.get("paper_validation_ready", False))
    )
    shadow_ready = bool(
        shadow_execution.get("ready_for_live", deployment_gate.get("shadow_ready", False))
    )
    shadow_compared_trade_count = int(
        shadow_execution.get(
            "compared_trade_count",
            deployment_gate.get("shadow_compared_trade_count", 0),
        )
        or 0
    )
    shadow_live_entry_comparison_count = int(
        shadow_live.get(
            "entry_comparison_count",
            deployment_gate.get(
                "shadow_live_entry_comparison_count",
                certification.get("shadow_live_entry_comparison_count", 0),
            ),
        )
        or 0
    )
    shadow_live_exit_comparison_count = int(
        shadow_live.get(
            "exit_comparison_count",
            deployment_gate.get(
                "shadow_live_exit_comparison_count",
                certification.get("shadow_live_exit_comparison_count", 0),
            ),
        )
        or 0
    )
    shadow_live_validation_runtime_days = _float(
        shadow_live.get(
            "validation_runtime_days",
            deployment_gate.get(
                "shadow_live_validation_runtime_days",
                certification.get("shadow_live_validation_runtime_days", 0.0),
            ),
        )
    )
    entry_quote_coverage_rate = _float(shadow_live.get("entry_quote_coverage_rate"))
    exit_quote_coverage_rate = _float(shadow_live.get("exit_quote_coverage_rate"))

    allow_shadow_live = bool(deployment_gate.get("allow_shadow_live"))
    allow_probation_live = bool(deployment_gate.get("allow_probation_live"))
    allow_full_live = bool(deployment_gate.get("allow_full_live"))
    capital_firewall_decision = str(capital_firewall.get("decision", "unknown"))

    shadow_blockers = list(deployment_gate.get("reasons", []))
    if not deployment_gate:
        shadow_blockers.append("Deployment gate status is missing.")
    if not health:
        shadow_blockers.append("Health status is missing.")
    elif bool(health.get("should_halt")):
        shadow_blockers.append("Health monitor currently requests a halt.")
    if not stress_kernel:
        shadow_blockers.append("Streaming stress kernel status is missing.")
    if not stress_field:
        shadow_blockers.append("Stress field status is missing.")

    probation_blockers = list(deployment_gate.get("reasons", []))
    if not paper_validation_ready:
        probation_blockers.append("Strict paper validation has not passed yet.")
    if not shadow_ready:
        probation_blockers.append("Shadow execution drift is not yet within tolerance.")
    if not shadow_live:
        probation_blockers.append("Shadow live comparator status is missing.")
    if not certification:
        probation_blockers.append("Production certification status is missing.")
    if not deployment_gate:
        probation_blockers.append("Deployment gate status is missing.")
    if entry_quote_coverage_rate and entry_quote_coverage_rate < comparator_thresholds.min_quote_coverage_rate_for_capital:
        probation_blockers.append(
            f"Entry quote coverage is {entry_quote_coverage_rate:.0%}, below the {comparator_thresholds.min_quote_coverage_rate_for_capital:.0%} requirement."
        )
    if exit_quote_coverage_rate and exit_quote_coverage_rate < comparator_thresholds.min_quote_coverage_rate_for_capital:
        probation_blockers.append(
            f"Exit quote coverage is {exit_quote_coverage_rate:.0%}, below the {comparator_thresholds.min_quote_coverage_rate_for_capital:.0%} requirement."
        )

    full_live_blockers = list(deployment_gate.get("reasons", []))
    if not certification:
        full_live_blockers.append("Production certification status is missing.")
    elif not bool(certification.get("ready_for_live")):
        full_live_blockers.append("Production certification has not fully passed yet.")
    if not deployment_gate:
        full_live_blockers.append("Deployment gate status is missing.")
    if capital_firewall_decision == "no_trade":
        full_live_blockers.append("Capital firewall still blocks new live entries.")
    elif not capital_firewall:
        full_live_blockers.append("Capital firewall status is missing.")

    probation_trades_remaining = max(
        _remaining_int(
            shadow_compared_trade_count,
            gate_thresholds.min_shadow_compared_trades_for_probation,
        ),
        _remaining_int(
            shadow_live_entry_comparison_count,
            gate_thresholds.min_shadow_live_entry_comparisons_for_probation,
        ),
        _remaining_int(
            shadow_live_exit_comparison_count,
            gate_thresholds.min_shadow_live_exit_comparisons_for_probation,
        ),
    )
    probation_days_remaining = _remaining_float(
        shadow_live_validation_runtime_days,
        gate_thresholds.min_shadow_live_validation_days_for_probation,
    )
    full_live_trades_remaining = max(
        _remaining_int(
            shadow_compared_trade_count,
            gate_thresholds.min_shadow_compared_trades_for_live,
        ),
        _remaining_int(
            shadow_live_entry_comparison_count,
            gate_thresholds.min_shadow_live_entry_comparisons_for_live,
        ),
        _remaining_int(
            shadow_live_exit_comparison_count,
            gate_thresholds.min_shadow_live_exit_comparisons_for_live,
        ),
    )
    full_live_days_remaining = _remaining_float(
        shadow_live_validation_runtime_days,
        gate_thresholds.min_shadow_live_validation_days_for_live,
    )

    probation_policy = (
        stress_kernel.get("probation_live_policy")
        if isinstance(stress_kernel.get("probation_live_policy"), dict)
        else {}
    )
    exposure_ladder = (
        survivability.get("exposure_ladder")
        if isinstance(survivability.get("exposure_ladder"), dict)
        else {}
    )

    overall_verdict = "no_go"
    if allow_full_live and capital_firewall_decision in {"allow_full_size", "allow_reduced_size"}:
        overall_verdict = "go_full_live"
    elif allow_probation_live and capital_firewall_decision in {"allow_full_size", "allow_reduced_size"}:
        overall_verdict = "go_probation"
    elif allow_shadow_live:
        overall_verdict = "shadow_only"

    if overall_verdict in {"go_probation", "go_full_live"}:
        effective_total_exposure_pct = _positive_min(
            probation_policy.get("max_total_exposure_pct"),
            exposure_ladder.get("max_total_exposure_pct"),
            deployment_gate.get("recommended_max_total_exposure_pct"),
            capital_firewall.get("max_total_exposure_pct"),
        )
        effective_per_trade_pct = _positive_min(
            probation_policy.get("max_per_trade_pct"),
            exposure_ladder.get("max_per_trade_pct"),
            deployment_gate.get("recommended_max_per_trade_pct"),
            capital_firewall.get("max_per_trade_pct"),
        )
        effective_capital_fraction = _positive_min(
            probation_policy.get("max_capital_fraction"),
            exposure_ladder.get("max_capital_fraction"),
            effective_total_exposure_pct,
        )
    else:
        effective_total_exposure_pct = 0.0
        effective_per_trade_pct = 0.0
        effective_capital_fraction = 0.0

    if overall_verdict == "go_full_live":
        current_stage = "full_live_gate"
    elif overall_verdict == "go_probation":
        current_stage = _map_probation_stage(str(probation_policy.get("stage", "")))
        if current_stage == "paper_shadow":
            current_stage = _infer_probation_stage(effective_capital_fraction)
    else:
        current_stage = "paper_shadow"

    rollout_plan = {
        "current_stage": current_stage,
        "current_stage_label": _ROLLOUT_STAGE_SPECS[current_stage]["label"],
        "effective_capital_fraction": effective_capital_fraction,
        "effective_total_exposure_pct": effective_total_exposure_pct,
        "effective_per_trade_pct": effective_per_trade_pct,
        "starting_size_usd": summary["capital"] * effective_per_trade_pct,
        "max_stage_exposure_usd": summary["capital"] * effective_total_exposure_pct,
        "pause_loss_usd": min(summary["capital"] * effective_per_trade_pct * 2.0, summary["capital"] * effective_total_exposure_pct),
        "step_back_loss_usd": min(summary["capital"] * effective_per_trade_pct * 3.0, summary["capital"] * effective_total_exposure_pct),
        "hard_stop_loss_usd": summary["capital"] * effective_total_exposure_pct,
        "pause_after_consecutive_losses": _PAUSE_AFTER_CONSECUTIVE_LOSSES,
        "step_back_after_consecutive_losses": _STEP_BACK_AFTER_CONSECUTIVE_LOSSES,
        "hard_halt_after_consecutive_losses": _HARD_HALT_AFTER_CONSECUTIVE_LOSSES,
        "stages": _rollout_stage_rows(
            capital=summary["capital"],
            current_stage=current_stage,
            effective_capital_fraction=effective_capital_fraction,
            effective_total_exposure_pct=effective_total_exposure_pct,
            effective_per_trade_pct=effective_per_trade_pct,
        ),
    }

    shadow_gate = {
        "allowed": allow_shadow_live,
        "trades_remaining": 0,
        "days_remaining": 0.0,
        "top_blockers": _first_reasons(shadow_blockers),
        "summary": _format_gate_summary(
            "Shadow Live",
            allowed=allow_shadow_live,
            trades_remaining=0,
            days_remaining=0.0,
            blockers=shadow_blockers,
        ),
    }
    probation_gate = {
        "allowed": allow_probation_live,
        "trades_remaining": probation_trades_remaining,
        "days_remaining": probation_days_remaining,
        "shadow_trade_comparisons_remaining": _remaining_int(
            shadow_compared_trade_count,
            gate_thresholds.min_shadow_compared_trades_for_probation,
        ),
        "entry_comparisons_remaining": _remaining_int(
            shadow_live_entry_comparison_count,
            gate_thresholds.min_shadow_live_entry_comparisons_for_probation,
        ),
        "exit_comparisons_remaining": _remaining_int(
            shadow_live_exit_comparison_count,
            gate_thresholds.min_shadow_live_exit_comparisons_for_probation,
        ),
        "top_blockers": _first_reasons(probation_blockers),
        "summary": _format_gate_summary(
            "Probation Live",
            allowed=allow_probation_live,
            trades_remaining=probation_trades_remaining,
            days_remaining=probation_days_remaining,
            blockers=probation_blockers,
        ),
    }
    full_live_gate = {
        "allowed": allow_full_live and capital_firewall_decision in {"allow_full_size", "allow_reduced_size"},
        "trades_remaining": full_live_trades_remaining,
        "days_remaining": full_live_days_remaining,
        "shadow_trade_comparisons_remaining": _remaining_int(
            shadow_compared_trade_count,
            gate_thresholds.min_shadow_compared_trades_for_live,
        ),
        "entry_comparisons_remaining": _remaining_int(
            shadow_live_entry_comparison_count,
            gate_thresholds.min_shadow_live_entry_comparisons_for_live,
        ),
        "exit_comparisons_remaining": _remaining_int(
            shadow_live_exit_comparison_count,
            gate_thresholds.min_shadow_live_exit_comparisons_for_live,
        ),
        "top_blockers": _first_reasons(full_live_blockers),
        "summary": _format_gate_summary(
            "Full Live",
            allowed=allow_full_live and capital_firewall_decision in {"allow_full_size", "allow_reduced_size"},
            trades_remaining=full_live_trades_remaining,
            days_remaining=full_live_days_remaining,
            blockers=full_live_blockers,
        ),
    }

    missing_artifacts = [
        name
        for name, payload in {
            "health": health,
            "paper_validation": paper_validation,
            "shadow_execution": shadow_execution,
            "shadow_live_comparator": shadow_live,
            "production_certification": certification,
            "deployment_gate": deployment_gate,
            "capital_firewall": capital_firewall,
            "streaming_stress_kernel": stress_kernel,
            "survivability": survivability,
            "stress_field": stress_field,
        }.items()
        if not payload
    ]

    if overall_verdict == "go_full_live":
        headline = (
            "GO: full-live gate is open. "
            f"Current caps are {effective_total_exposure_pct:.2%} total exposure and {effective_per_trade_pct:.2%} per trade."
        )
    elif overall_verdict == "go_probation":
        headline = (
            "GO: probation live is open. "
            f"Start at ${rollout_plan['starting_size_usd']:,.2f} per trade with ${rollout_plan['max_stage_exposure_usd']:,.2f} max live exposure."
        )
    elif overall_verdict == "shadow_only":
        headline = (
            "NO-GO for capital: stay in paper or shadow. "
            f"Probation still needs {probation_trades_remaining} fully quoted trades and {probation_days_remaining:.1f} days."
        )
    else:
        blocker = _first_reasons(shadow_blockers, limit=1)
        headline = "NO-GO: live readiness artifacts do not yet justify leaving paper."
        if blocker:
            headline += f" {blocker[0]}"

    return {
        "overall_verdict": overall_verdict,
        "headline": headline,
        "operating_mode": operating_mode,
        "health_status": health_status,
        "paper_mode": summary.get("paper_mode", True),
        "allowed_mode": str(deployment_gate.get("allowed_mode", "blocked") or "blocked"),
        "capital_firewall_decision": capital_firewall_decision,
        "missing_artifacts": missing_artifacts,
        "counts": {
            "closed_trades": summary["n_closed"],
            "shadow_compared_trade_count": shadow_compared_trade_count,
            "shadow_live_entry_comparison_count": shadow_live_entry_comparison_count,
            "shadow_live_exit_comparison_count": shadow_live_exit_comparison_count,
            "shadow_live_validation_runtime_days": shadow_live_validation_runtime_days,
        },
        "gates": {
            "shadow_live": shadow_gate,
            "probation_live": probation_gate,
            "full_live": full_live_gate,
        },
        "one_line_summaries": {
            "shadow_live": shadow_gate["summary"],
            "probation_live": probation_gate["summary"],
            "full_live": full_live_gate["summary"],
        },
        "rollout_plan": rollout_plan,
        "sources": {
            "paper_validation_ready": paper_validation_ready,
            "shadow_ready": shadow_ready,
            "entry_quote_coverage_rate": entry_quote_coverage_rate,
            "exit_quote_coverage_rate": exit_quote_coverage_rate,
            "execution_fidelity_score": _float(execution_drift.get("execution_fidelity_score")),
        },
    }


__all__ = [
    "DEFAULT_ASSETS",
    "load_state",
    "load_journal",
    "load_divergence",
    "load_deployment_gate",
    "load_drift_intelligence",
    "load_execution_drift",
    "load_shadow_live_comparator",
    "load_capital_firewall",
    "load_health",
    "load_market_snapshot",
    "load_production_certification",
    "load_stress_field",
    "load_streaming_stress_kernel",
    "load_survivability",
    "portfolio_summary",
    "compute_signal_proximity",
    "proximity_matrix",
    "build_live_readiness_snapshot",
]

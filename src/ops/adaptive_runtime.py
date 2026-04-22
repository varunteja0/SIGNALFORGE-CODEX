from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.engine.adaptive_portfolio_engine import AdaptiveCycleReport
    from src.engine.divergence_tracker import DivergenceStats


def base_strategy_name(name: str) -> str:
    return str(name).split("__", 1)[0]


@dataclass
class AdaptiveSafetyDecision:
    action: str = "allow"
    size_cap: float = 1.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class TradingCycleState:
    iteration: int
    timestamp: str
    paper_mode: bool
    capital: float
    current_drawdown: float
    daily_pnl: float
    projected_utilization: float
    gross_exposure_scale: float
    risk_budget_multiplier: float
    target_volatility: float
    realized_volatility: float
    smoothed_volatility: float
    volatility_tracking_error: float
    pid_output: float
    edge_retention_ratio: float
    edge_retention_state: str
    reality_gap_fraction: float
    portfolio_objective_score: float
    correlation_shock: bool
    disabled_strategies: dict[str, str] = field(default_factory=dict)
    retired_strategies: list[str] = field(default_factory=list)
    allocation_weights: dict[str, float] = field(default_factory=dict)
    suggested_position_sizes: dict[str, float] = field(default_factory=dict)
    market_route_multipliers: dict[str, float] = field(default_factory=dict)
    regime_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    adapted_slots: list[str] = field(default_factory=list)
    safety_action: str = "allow"
    safety_size_cap: float = 1.0
    safety_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdaptiveSafetyGovernor:
    """Convert adaptive health signals into entry gating decisions."""

    def __init__(
        self,
        *,
        hard_drawdown_limit: float = 0.15,
        edge_retention_floor: float = 0.45,
        objective_floor: float = -0.25,
        max_tracking_error: float = 0.10,
        max_avg_entry_slippage_bps: float = 12.0,
        max_avg_pnl_divergence_pct: float = 25.0,
    ):
        self.hard_drawdown_limit = hard_drawdown_limit
        self.edge_retention_floor = edge_retention_floor
        self.objective_floor = objective_floor
        self.max_tracking_error = max_tracking_error
        self.max_avg_entry_slippage_bps = max_avg_entry_slippage_bps
        self.max_avg_pnl_divergence_pct = max_avg_pnl_divergence_pct

    def evaluate(self, cycle_state: TradingCycleState) -> AdaptiveSafetyDecision:
        if cycle_state.current_drawdown >= self.hard_drawdown_limit:
            return AdaptiveSafetyDecision(
                action="halt",
                size_cap=0.0,
                reasons=[
                    f"drawdown {cycle_state.current_drawdown:.1%} >= {self.hard_drawdown_limit:.1%}",
                ],
            )

        pause_reasons: list[str] = []
        avg_entry_slippage = float(cycle_state.execution.get("avg_entry_slippage_bps", 0.0) or 0.0)
        avg_pnl_divergence = float(cycle_state.execution.get("avg_pnl_divergence_pct", 0.0) or 0.0)

        if cycle_state.edge_retention_state == "broken":
            pause_reasons.append("edge retention broken")
        elif cycle_state.edge_retention_state != "unknown" and cycle_state.edge_retention_ratio < self.edge_retention_floor:
            pause_reasons.append(
                f"edge retention {cycle_state.edge_retention_ratio:.2f} < {self.edge_retention_floor:.2f}"
            )

        if cycle_state.volatility_tracking_error > self.max_tracking_error:
            pause_reasons.append(
                f"tracking error {cycle_state.volatility_tracking_error:.3f} > {self.max_tracking_error:.3f}"
            )

        if avg_entry_slippage > self.max_avg_entry_slippage_bps:
            pause_reasons.append(
                f"avg entry slippage {avg_entry_slippage:.1f}bps > {self.max_avg_entry_slippage_bps:.1f}bps"
            )

        if abs(avg_pnl_divergence) > self.max_avg_pnl_divergence_pct:
            pause_reasons.append(
                f"avg pnl divergence {avg_pnl_divergence:+.1f}% > {self.max_avg_pnl_divergence_pct:.1f}%"
            )

        if pause_reasons:
            return AdaptiveSafetyDecision(
                action="pause_entries",
                size_cap=0.0,
                reasons=pause_reasons,
            )

        reduce_reasons: list[str] = []
        if cycle_state.portfolio_objective_score < self.objective_floor:
            reduce_reasons.append(
                f"objective score {cycle_state.portfolio_objective_score:.3f} < {self.objective_floor:.3f}"
            )

        if cycle_state.correlation_shock and cycle_state.realized_volatility > max(
            cycle_state.target_volatility * 1.20,
            cycle_state.target_volatility + 0.02,
        ):
            reduce_reasons.append("correlation shock with elevated realized volatility")

        if reduce_reasons:
            return AdaptiveSafetyDecision(
                action="reduce",
                size_cap=0.5,
                reasons=reduce_reasons,
            )

        return AdaptiveSafetyDecision()


def build_trading_cycle_state(
    report: AdaptiveCycleReport,
    *,
    divergence_stats: DivergenceStats,
    capital: float,
    current_drawdown: float,
    daily_pnl: float,
    iteration: int,
    timestamp: str,
    paper_mode: bool,
) -> TradingCycleState:
    allocation = report.allocation_decision
    risk_snapshot = report.risk_snapshot
    reality_gap = report.reality_gap
    projected_utilization = float(
        allocation.projected_utilization
        * allocation.risk_budget_multiplier
        * allocation.gross_exposure_scale
    )
    regime_states = {
        symbol: {
            "composite": state.composite,
            "trend": state.trend_regime,
            "volatility": state.volatility_regime,
            "liquidity": state.liquidity_regime,
            "confidence": float(state.confidence_score),
        }
        for symbol, state in report.regime_states.items()
    }
    retired = [name for name, decision in report.lifecycle_decisions.items() if decision.retired]
    execution = {
        "total_trades": int(divergence_stats.total_trades),
        "total_missed": int(divergence_stats.total_missed),
        "avg_entry_slippage_bps": float(divergence_stats.avg_entry_slippage_bps),
        "avg_exit_slippage_bps": float(divergence_stats.avg_exit_slippage_bps),
        "avg_pnl_divergence_pct": float(divergence_stats.avg_pnl_divergence_pct),
        "slippage_trend": float(divergence_stats.slippage_trend),
        "alerts": list(divergence_stats.alerts),
    }
    return TradingCycleState(
        iteration=iteration,
        timestamp=timestamp,
        paper_mode=paper_mode,
        capital=float(capital),
        current_drawdown=float(current_drawdown),
        daily_pnl=float(daily_pnl),
        projected_utilization=projected_utilization,
        gross_exposure_scale=float(allocation.gross_exposure_scale),
        risk_budget_multiplier=float(allocation.risk_budget_multiplier),
        target_volatility=float(allocation.target_volatility),
        realized_volatility=float(allocation.realized_volatility),
        smoothed_volatility=float(risk_snapshot.smoothed_volatility) if risk_snapshot else float(allocation.realized_volatility),
        volatility_tracking_error=float(risk_snapshot.volatility_tracking_error) if risk_snapshot else abs(float(allocation.target_volatility) - float(allocation.realized_volatility)),
        pid_output=float(risk_snapshot.pid_output) if risk_snapshot else 0.0,
        edge_retention_ratio=float(reality_gap.edge_retention_ratio) if reality_gap else 1.0,
        edge_retention_state=str(reality_gap.edge_retention_state) if reality_gap else "unknown",
        reality_gap_fraction=float(reality_gap.pnl_gap_fraction) if reality_gap else 0.0,
        portfolio_objective_score=float(allocation.portfolio_objective_score),
        correlation_shock=bool(allocation.correlation_shock),
        disabled_strategies=dict(allocation.disabled_strategies),
        retired_strategies=retired,
        allocation_weights={k: float(v) for k, v in allocation.weights.items()},
        suggested_position_sizes={k: float(v) for k, v in report.suggested_position_sizes.items()},
        market_route_multipliers={k: float(v) for k, v in allocation.market_route_multipliers.items()},
        regime_states=regime_states,
        execution=execution,
        adapted_slots=[slot.name for slot in report.adapted_engine.slots],
    )
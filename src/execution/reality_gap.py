from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StrategyEdgeRetentionSnapshot:
    theoretical_pnl: float
    execution_adjusted_pnl: float
    edge_retention_ratio: float
    status: str


@dataclass
class RealityGapSnapshot:
    theoretical_pnl: float
    execution_adjusted_pnl: float
    pnl_gap: float
    pnl_gap_fraction: float
    execution_cost_ratio: float
    edge_retention_ratio: float
    edge_retention_state: str
    fragile_edge: bool
    strategy_edge_retention: dict[str, StrategyEdgeRetentionSnapshot] = field(default_factory=dict)


class RealityGapValidator:
    """Flag strategies whose theoretical edge degrades materially after execution costs."""

    def __init__(self, fragility_threshold: float = 0.30):
        self.fragility_threshold = fragility_threshold

    def evaluate(self, theoretical_result, adjusted_result, execution_summary) -> RealityGapSnapshot:
        theoretical_pnl = float(getattr(theoretical_result, "total_pnl", 0.0) or 0.0)
        execution_adjusted_pnl = float(getattr(adjusted_result, "total_pnl", 0.0) or 0.0)
        pnl_gap = theoretical_pnl - execution_adjusted_pnl
        if abs(theoretical_pnl) > 1e-9:
            pnl_gap_fraction = pnl_gap / abs(theoretical_pnl)
        else:
            pnl_gap_fraction = 1.0 if abs(pnl_gap) > 0.0 else 0.0
        total_execution_cost = float(getattr(execution_summary, "total_execution_cost", 0.0) or 0.0)
        execution_cost_ratio = total_execution_cost / max(abs(theoretical_pnl), 1e-9)
        edge_retention_ratio, edge_retention_state = self._edge_retention(theoretical_pnl, execution_adjusted_pnl)
        strategy_edge_retention = self._strategy_edge_retention(theoretical_result, adjusted_result)
        fragile_edge = bool(
            (theoretical_pnl > 0.0 and execution_adjusted_pnl <= 0.0)
            or pnl_gap_fraction > self.fragility_threshold
            or execution_cost_ratio > self.fragility_threshold
            or edge_retention_state != "strong"
        )
        return RealityGapSnapshot(
            theoretical_pnl=theoretical_pnl,
            execution_adjusted_pnl=execution_adjusted_pnl,
            pnl_gap=pnl_gap,
            pnl_gap_fraction=float(pnl_gap_fraction),
            execution_cost_ratio=float(execution_cost_ratio),
            edge_retention_ratio=edge_retention_ratio,
            edge_retention_state=edge_retention_state,
            fragile_edge=fragile_edge,
            strategy_edge_retention=strategy_edge_retention,
        )

    @staticmethod
    def _edge_retention(theoretical_pnl: float, execution_adjusted_pnl: float) -> tuple[float, str]:
        if theoretical_pnl <= 0.0:
            ratio = 1.0 if execution_adjusted_pnl > 0.0 else 0.0
        else:
            ratio = execution_adjusted_pnl / max(theoretical_pnl, 1e-9)
        if ratio > 0.70:
            state = "strong"
        elif ratio >= 0.40:
            state = "fragile"
        else:
            state = "broken"
        return float(ratio), state

    def _strategy_edge_retention(self, theoretical_result, adjusted_result) -> dict[str, StrategyEdgeRetentionSnapshot]:
        theoretical = self._strategy_pnls(theoretical_result)
        adjusted = self._strategy_pnls(adjusted_result)
        snapshots = {}
        for strategy in sorted(set(theoretical).union(adjusted)):
            theoretical_pnl = float(theoretical.get(strategy, 0.0))
            adjusted_pnl = float(adjusted.get(strategy, 0.0))
            ratio, state = self._edge_retention(theoretical_pnl, adjusted_pnl)
            snapshots[strategy] = StrategyEdgeRetentionSnapshot(
                theoretical_pnl=theoretical_pnl,
                execution_adjusted_pnl=adjusted_pnl,
                edge_retention_ratio=ratio,
                status=state,
            )
        return snapshots

    @staticmethod
    def _strategy_pnls(result) -> dict[str, float]:
        raw = getattr(result, "strategy_results", {}) or {}
        out = {}
        for name, stats in raw.items():
            if isinstance(stats, dict):
                value = stats.get("net_pnl", 0.0)
            else:
                value = getattr(stats, "net_pnl", 0.0)
            out[str(name)] = float(value or 0.0)
        return out
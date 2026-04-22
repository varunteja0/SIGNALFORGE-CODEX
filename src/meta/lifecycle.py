from __future__ import annotations

from dataclasses import dataclass

from src.meta.performance import StrategyPerformanceSnapshot


@dataclass
class LifecycleDecision:
    strategy_name: str
    degradation_streak: int
    persistence_score: float
    score: float
    growth_score: float
    edge_retention: float
    status: str
    replacement_candidate: str | None = None
    retired: bool = False


class StrategyLifecycleManager:
    """Retire strategies with sustained degradation and promote exploratory replacements."""

    def __init__(
        self,
        persistence_floor: float = 0.40,
        score_floor: float = 0.05,
        growth_floor: float = -0.05,
        edge_retention_floor: float = 0.40,
        sustained_degradation_cycles: int = 2,
        promotion_margin: float = 0.05,
        promotion_persistence_floor: float = 0.50,
    ):
        self.persistence_floor = persistence_floor
        self.score_floor = score_floor
        self.growth_floor = growth_floor
        self.edge_retention_floor = edge_retention_floor
        self.sustained_degradation_cycles = sustained_degradation_cycles
        self.promotion_margin = promotion_margin
        self.promotion_persistence_floor = promotion_persistence_floor
        self.degradation_streaks: dict[str, int] = {}

    def update(
        self,
        performance_metrics: dict[str, StrategyPerformanceSnapshot],
    ) -> dict[str, LifecycleDecision]:
        decisions: dict[str, LifecycleDecision] = {}
        base_names = sorted({self._base_strategy_name(name) for name in performance_metrics})
        for base_name in base_names:
            base_snapshot = performance_metrics.get(base_name)
            if base_snapshot is None:
                continue

            degraded = self._is_degraded(base_snapshot)
            streak = self.degradation_streaks.get(base_name, 0) + 1 if degraded else 0
            self.degradation_streaks[base_name] = streak
            replacement_candidate = self._replacement_candidate(base_name, performance_metrics, base_snapshot)
            retired = streak >= self.sustained_degradation_cycles

            status = "active"
            if retired:
                status = "retire_replace"
            elif replacement_candidate is not None:
                status = "promotion_candidate"
            elif degraded:
                status = "monitor"

            decisions[base_name] = LifecycleDecision(
                strategy_name=base_name,
                degradation_streak=streak,
                persistence_score=float(base_snapshot.persistence_score),
                score=float(base_snapshot.score),
                growth_score=float(base_snapshot.growth_score),
                edge_retention=float(base_snapshot.edge_retention),
                status=status,
                replacement_candidate=replacement_candidate,
                retired=retired,
            )
        return decisions

    def _is_degraded(self, snapshot: StrategyPerformanceSnapshot) -> bool:
        if snapshot.state == "warming_up":
            return False
        return bool(
            snapshot.state == "disabled"
            or snapshot.persistence_score < self.persistence_floor
            or snapshot.score < self.score_floor
            or snapshot.growth_score < self.growth_floor
            or snapshot.edge_retention < self.edge_retention_floor
        )

    def _replacement_candidate(
        self,
        base_name: str,
        performance_metrics: dict[str, StrategyPerformanceSnapshot],
        base_snapshot: StrategyPerformanceSnapshot,
    ) -> str | None:
        candidates: list[tuple[float, str]] = []
        for name, snapshot in performance_metrics.items():
            if self._base_strategy_name(name) != base_name or name == base_name:
                continue
            if snapshot.persistence_score < self.promotion_persistence_floor:
                continue
            if snapshot.edge_retention < self.edge_retention_floor:
                continue
            if snapshot.score <= base_snapshot.score + self.promotion_margin:
                continue
            candidates.append((snapshot.score + 0.5 * snapshot.growth_score, name))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    @staticmethod
    def _base_strategy_name(name: str) -> str:
        return str(name).split("__", 1)[0]
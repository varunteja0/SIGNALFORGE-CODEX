from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src.meta.performance import StrategyPerformanceSnapshot


@dataclass
class PersistenceSnapshot:
    strategy_name: str
    sharpe_stability: float
    return_stability: float
    positive_fold_fraction: float
    persistence_score: float
    persistence_multiplier: float
    history_length: int


class EdgePersistenceScorer:
    """Measure whether a strategy's edge remains stable across cycles/folds."""

    def __init__(
        self,
        history_window: int = 6,
        min_history: int = 2,
        multiplier_floor: float = 0.60,
        multiplier_ceiling: float = 1.30,
    ):
        self.history_window = history_window
        self.min_history = min_history
        self.multiplier_floor = multiplier_floor
        self.multiplier_ceiling = multiplier_ceiling
        self.history: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def update(
        self,
        performance_metrics: dict[str, StrategyPerformanceSnapshot],
    ) -> dict[str, PersistenceSnapshot]:
        snapshots: dict[str, PersistenceSnapshot] = {}

        for strategy, snapshot in performance_metrics.items():
            history = self.history[strategy]
            history.append((float(snapshot.rolling_sharpe), float(snapshot.trailing_return)))
            if len(history) > self.history_window:
                del history[:-self.history_window]

            sharpes = np.array([item[0] for item in history], dtype=float)
            returns = np.array([item[1] for item in history], dtype=float)
            if len(history) < self.min_history:
                sharpe_stability = 0.65
                return_stability = 0.65
                positive_fraction = 1.0 if returns[-1] >= 0.0 else 0.5
            else:
                sharpe_stability = self._stability_from_series(sharpes)
                return_stability = self._stability_from_series(returns)
                positive_fraction = float((returns > 0.0).mean())

            persistence_score = float(
                np.clip(
                    0.40 * sharpe_stability
                    + 0.40 * return_stability
                    + 0.20 * positive_fraction,
                    0.0,
                    1.0,
                )
            )
            persistence_multiplier = float(
                np.clip(
                    0.70 + 0.60 * persistence_score,
                    self.multiplier_floor,
                    self.multiplier_ceiling,
                )
            )
            snapshots[strategy] = PersistenceSnapshot(
                strategy_name=strategy,
                sharpe_stability=sharpe_stability,
                return_stability=return_stability,
                positive_fold_fraction=positive_fraction,
                persistence_score=persistence_score,
                persistence_multiplier=persistence_multiplier,
                history_length=len(history),
            )

        return snapshots

    @staticmethod
    def _stability_from_series(values: np.ndarray) -> float:
        if len(values) <= 1:
            return 0.65
        mean_abs = max(abs(float(values.mean())), 0.25)
        dispersion = float(values.std(ddof=0)) / mean_abs
        return float(np.clip(1.0 / (1.0 + dispersion), 0.0, 1.0))
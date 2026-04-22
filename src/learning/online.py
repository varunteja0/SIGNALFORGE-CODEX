from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.allocation.adaptive import AllocationDecision
from src.meta.performance import StrategyPerformanceSnapshot


@dataclass
class StrategyLearningState:
    strategy_name: str
    expected_score: float = 0.0
    reward_ema: float = 0.0
    risk_bias: float = 1.0


@dataclass
class LearningUpdate:
    strategy_name: str
    realized_score: float
    previous_expected_score: float
    reward: float
    new_expected_score: float
    risk_bias: float
    allocation_weight: float = 0.0
    exploration_weight: float = 0.0
    parameter_suggestions: dict = field(default_factory=dict)


@dataclass
class LearningResult:
    updates: list[LearningUpdate]
    parameter_suggestions: dict[str, dict]


class OnlineLearningLoop:
    """Simple reinforcement-style update loop for next-cycle tuning."""

    THRESHOLD_KEYS = {"entry_z", "trend_threshold", "shock_return", "volume_spike", "exit_z"}
    LOOKBACK_KEYS = {"lookback", "breakout_lookback", "squeeze_window", "fast", "slow"}
    RISK_KEYS = {"position_size_pct"}

    def __init__(
        self,
        learning_rate: float = 0.20,
        parameter_step: float = 0.05,
        history_path: str = "fund_data/adaptive_learning_history.jsonl",
    ):
        self.learning_rate = learning_rate
        self.parameter_step = parameter_step
        self.history_path = Path(history_path)
        self.strategy_states: dict[str, StrategyLearningState] = {}

    def update(
        self,
        allocation_decision: AllocationDecision,
        performance_metrics: dict[str, StrategyPerformanceSnapshot],
        strategy_configs: dict[str, dict] | None = None,
    ) -> LearningResult:
        strategy_configs = strategy_configs or {}
        updates: list[LearningUpdate] = []
        parameter_suggestions: dict[str, dict] = {}
        records = []

        for strategy, snapshot in performance_metrics.items():
            state = self.strategy_states.setdefault(
                strategy,
                StrategyLearningState(strategy_name=strategy),
            )
            previous_expected = state.expected_score
            realized = snapshot.score
            reward = realized - previous_expected
            state.expected_score = (1.0 - self.learning_rate) * previous_expected + self.learning_rate * realized
            state.reward_ema = (1.0 - self.learning_rate) * state.reward_ema + self.learning_rate * reward
            state.risk_bias = float(np.clip(state.risk_bias * (1.0 + self.learning_rate * reward), 0.50, 1.50))

            suggestions = self._suggest_parameters(strategy_configs.get(strategy, {}), reward)
            if suggestions:
                parameter_suggestions[strategy] = suggestions

            update = LearningUpdate(
                strategy_name=strategy,
                realized_score=realized,
                previous_expected_score=previous_expected,
                reward=reward,
                new_expected_score=state.expected_score,
                risk_bias=state.risk_bias,
                allocation_weight=allocation_decision.weights.get(strategy, 0.0),
                exploration_weight=allocation_decision.exploration_weights.get(strategy, 0.0),
                parameter_suggestions=suggestions,
            )
            updates.append(update)
            records.append(
                {
                    "strategy_name": strategy,
                    "realized_score": realized,
                    "previous_expected_score": previous_expected,
                    "reward": reward,
                    "new_expected_score": state.expected_score,
                    "risk_bias": state.risk_bias,
                    "allocation_weight": allocation_decision.weights.get(strategy, 0.0),
                    "exploration_weight": allocation_decision.exploration_weights.get(strategy, 0.0),
                    "parameter_suggestions": suggestions,
                }
            )

        if records:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

        return LearningResult(updates=updates, parameter_suggestions=parameter_suggestions)

    def _suggest_parameters(self, config: dict, reward: float) -> dict:
        if not config or abs(reward) < 0.02:
            return {}

        step = self.parameter_step
        direction = -1.0 if reward > 0 else 1.0
        suggestions = {}

        for key, value in config.items():
            if not isinstance(value, (int, float)):
                continue

            updated = float(value)
            if key in self.THRESHOLD_KEYS:
                updated = value * (1.0 + direction * step)
            elif key in self.LOOKBACK_KEYS:
                updated = value * (1.0 + direction * step * 0.5)
            elif key in self.RISK_KEYS:
                updated = value * (1.0 - direction * step * 0.5)
            else:
                continue

            if isinstance(value, int):
                updated = max(1, int(round(updated)))
            suggestions[key] = updated

        return suggestions
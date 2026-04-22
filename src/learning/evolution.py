from __future__ import annotations

from dataclasses import dataclass

from src.meta.performance import StrategyPerformanceSnapshot


@dataclass(frozen=True)
class EvolutionVariant:
    name: str
    method: str
    source_strategy: str
    donor_strategy: str | None
    config: dict
    lineage_score: float


class StrategyEvolutionEngine:
    """Generate deterministic mutation and recombination variants per strategy."""

    THRESHOLD_KEYS = {"entry_z", "trend_threshold", "shock_return", "volume_spike", "exit_z"}
    LOOKBACK_KEYS = {
        "lookback",
        "breakout_lookback",
        "squeeze_window",
        "fast",
        "slow",
        "confirm",
        "hold_bars",
    }
    RISK_KEYS = {"position_size_pct", "target_vol"}

    def __init__(
        self,
        mutation_strength: float = 0.10,
        recombination_blend: float = 0.50,
    ):
        self.mutation_strength = mutation_strength
        self.recombination_blend = recombination_blend

    def build_variants(
        self,
        *,
        base_name: str,
        base_config: dict,
        strategy_configs: dict[str, dict],
        performance_metrics: dict[str, StrategyPerformanceSnapshot],
        strategy_group_map: dict[str, str],
    ) -> list[EvolutionVariant]:
        base_strategy = self._base_strategy_name(base_name)
        candidate_names = [
            name
            for name in strategy_configs
            if self._base_strategy_name(name) == base_strategy
        ]
        ranked_variants = sorted(
            candidate_names,
            key=lambda name: self._variant_score(name, performance_metrics),
            reverse=True,
        )
        anchor_name = ranked_variants[0] if ranked_variants else base_name
        anchor_config = dict(strategy_configs.get(anchor_name, base_config))
        anchor_config.pop("position_size_pct", None)
        anchor_score = self._variant_score(anchor_name, performance_metrics)

        variants = [
            EvolutionVariant(
                name=f"{base_strategy}__mutate",
                method="mutation",
                source_strategy=anchor_name,
                donor_strategy=None,
                config=self._mutate(anchor_config, base_config, anchor_name),
                lineage_score=anchor_score,
            )
        ]

        donor_name = self._select_donor(
            base_strategy=base_strategy,
            performance_metrics=performance_metrics,
            strategy_group_map=strategy_group_map,
        )
        if donor_name is not None:
            donor_config = dict(strategy_configs.get(donor_name, {}))
            donor_config.pop("position_size_pct", None)
            if donor_config:
                variants.append(
                    EvolutionVariant(
                        name=f"{base_strategy}__recombine",
                        method="recombination",
                        source_strategy=anchor_name,
                        donor_strategy=donor_name,
                        config=self._recombine(anchor_config, donor_config, base_strategy),
                        lineage_score=0.5 * (anchor_score + self._variant_score(donor_name, performance_metrics)),
                    )
                )
        return variants

    def _select_donor(
        self,
        *,
        base_strategy: str,
        performance_metrics: dict[str, StrategyPerformanceSnapshot],
        strategy_group_map: dict[str, str],
    ) -> str | None:
        group = strategy_group_map.get(base_strategy)
        candidates = []
        for name, snapshot in performance_metrics.items():
            if self._base_strategy_name(name) == base_strategy:
                continue
            if group is not None and strategy_group_map.get(self._base_strategy_name(name)) != group:
                continue
            candidates.append((self._variant_score(name, performance_metrics), name, snapshot))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return str(candidates[0][1])

    def _mutate(self, anchor_config: dict, base_config: dict, anchor_name: str) -> dict:
        mutated = dict(anchor_config)
        for key, value in anchor_config.items():
            if not isinstance(value, (int, float)):
                continue
            base_value = base_config.get(key, value)
            if not isinstance(base_value, (int, float)):
                base_value = value
            signature = sum(ord(char) for char in f"{anchor_name}:{key}")
            direction = 1.0 if signature % 2 == 0 else -1.0
            if key in self.THRESHOLD_KEYS:
                scale = 1.0 + direction * self.mutation_strength
            elif key in self.LOOKBACK_KEYS:
                scale = 1.0 + direction * self.mutation_strength * 0.50
            elif key in self.RISK_KEYS:
                scale = 1.0 + direction * self.mutation_strength * 0.35
            else:
                scale = 1.0 + direction * self.mutation_strength * 0.25
            proposal = float(value) * scale
            if abs(float(value) - float(base_value)) < 1e-9:
                proposal = float(base_value) * scale
            mutated[key] = self._cast_like(value, proposal)
        return mutated

    def _recombine(self, anchor_config: dict, donor_config: dict, base_strategy: str) -> dict:
        child = dict(anchor_config)
        for key, value in anchor_config.items():
            donor_value = donor_config.get(key)
            if donor_value is None:
                continue
            if not isinstance(value, (int, float)) or not isinstance(donor_value, (int, float)):
                continue
            signature = sum(ord(char) for char in f"{base_strategy}:{key}")
            blend = self.recombination_blend if signature % 2 == 0 else (1.0 - self.recombination_blend)
            proposal = blend * float(value) + (1.0 - blend) * float(donor_value)
            child[key] = self._cast_like(value, proposal)
        return child

    def _variant_score(
        self,
        name: str,
        performance_metrics: dict[str, StrategyPerformanceSnapshot],
    ) -> float:
        snapshot = performance_metrics.get(name)
        if snapshot is None:
            return 0.0
        return float(snapshot.score + 0.50 * snapshot.growth_score + 0.25 * snapshot.edge_retention)

    @staticmethod
    def _cast_like(reference, value: float):
        if isinstance(reference, int):
            return max(1, int(round(value)))
        return float(value)

    @staticmethod
    def _base_strategy_name(name: str) -> str:
        return str(name).split("__", 1)[0]
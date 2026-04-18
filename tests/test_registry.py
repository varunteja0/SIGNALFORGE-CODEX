"""Tests for the strategy/model registry."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.registry import RegistryEntry, StrategyRegistry


@pytest.fixture()
def tmp_registry(tmp_path: Path) -> StrategyRegistry:
    return StrategyRegistry(tmp_path / "registry.ndjson")


def test_empty_registry_returns_empty_list(tmp_registry: StrategyRegistry):
    assert tmp_registry.all() == []
    assert tmp_registry.history("anything") == []
    assert tmp_registry.latest("anything") is None


def test_register_creates_version_one(tmp_registry: StrategyRegistry):
    entry = tmp_registry.register(
        "liq_reversal_btc",
        params={"atr_mult": 2.0, "window": 14},
        trained_on={"symbol": "BTC/USDT", "days": 1825},
        notes="initial deploy",
    )
    assert entry.version == 1
    assert entry.strategy_id == "liq_reversal_btc"
    assert len(entry.config_hash) == 64
    assert entry.deployed_at.endswith("Z")


def test_versions_monotonically_increase(tmp_registry: StrategyRegistry):
    tmp_registry.register("s1", {"a": 1})
    tmp_registry.register("s1", {"a": 2})
    tmp_registry.register("s1", {"a": 3})
    hist = tmp_registry.history("s1")
    assert [e.version for e in hist] == [1, 2, 3]
    assert tmp_registry.latest("s1").version == 3


def test_independent_strategy_ids_have_independent_versions(
    tmp_registry: StrategyRegistry,
):
    tmp_registry.register("strat_a", {"p": 1})
    tmp_registry.register("strat_b", {"p": 1})
    tmp_registry.register("strat_a", {"p": 2})
    assert tmp_registry.latest("strat_a").version == 2
    assert tmp_registry.latest("strat_b").version == 1


def test_config_hash_is_stable_across_key_order(tmp_registry: StrategyRegistry):
    e1 = tmp_registry.register("s", {"a": 1, "b": 2})
    e2 = tmp_registry.register("s", {"b": 2, "a": 1})
    assert e1.config_hash == e2.config_hash


def test_config_hash_differs_on_param_change(tmp_registry: StrategyRegistry):
    e1 = tmp_registry.register("s", {"a": 1})
    e2 = tmp_registry.register("s", {"a": 2})
    assert e1.config_hash != e2.config_hash


def test_invalid_strategy_id_raises(tmp_registry: StrategyRegistry):
    with pytest.raises(ValueError):
        tmp_registry.register("has spaces", {})
    with pytest.raises(ValueError):
        tmp_registry.register("", {})


def test_validation_hash_recorded_when_file_supplied(
    tmp_path: Path, tmp_registry: StrategyRegistry
):
    val_file = tmp_path / "validation.json"
    val_file.write_text(json.dumps({"keeps": ["a", "b"]}))
    entry = tmp_registry.register(
        "s", {"p": 1}, validation_file=val_file, notes="with validation"
    )
    assert entry.validation_file == str(val_file)
    assert entry.validation_hash is not None
    assert len(entry.validation_hash) == 64


def test_ndjson_is_append_only(tmp_registry: StrategyRegistry, tmp_path: Path):
    tmp_registry.register("s", {"p": 1})
    tmp_registry.register("s", {"p": 2})
    lines = tmp_registry.path.read_text().strip().splitlines()
    assert len(lines) == 2
    # Each line must be valid JSON and round-trip through RegistryEntry.
    for line in lines:
        RegistryEntry.from_json(line)


def test_ids_returns_sorted_unique(tmp_registry: StrategyRegistry):
    tmp_registry.register("zzz", {})
    tmp_registry.register("aaa", {})
    tmp_registry.register("mmm", {})
    tmp_registry.register("aaa", {"v": 2})
    assert tmp_registry.ids() == ["aaa", "mmm", "zzz"]

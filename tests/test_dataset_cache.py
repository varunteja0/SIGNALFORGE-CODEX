from pathlib import Path

import pandas as pd

from src.core.dataset_cache import dataset_cache_path, load_or_build_datasets


class _FakeEngine:
    def __init__(self) -> None:
        self.assets = ["BTC/USDT", "ETH/USDT"]
        self.data_days = 180
        self.calls = 0

    def load_data(self):
        self.calls += 1
        return {
            "BTC/USDT": pd.DataFrame({"close": [1.0, 2.0]}),
            "ETH/USDT": pd.DataFrame({"close": [3.0, 4.0]}),
        }


def test_dataset_cache_path_uses_namespace_and_days(tmp_path: Path) -> None:
    path = dataset_cache_path(
        assets=["BTC/USDT", "ETH/USDT"],
        data_days=180,
        namespace="unit_test",
        cache_dir=tmp_path,
    )

    assert path.parent == tmp_path
    assert path.name.startswith("unit_test_180d_")
    assert path.suffix == ".pkl"


def test_load_or_build_datasets_reuses_fresh_cache(tmp_path: Path) -> None:
    engine = _FakeEngine()

    datasets_1, path, cache_hit_1 = load_or_build_datasets(
        engine,
        namespace="unit_test",
        cache_dir=tmp_path,
        max_age_hours=1.0,
    )
    datasets_2, _, cache_hit_2 = load_or_build_datasets(
        engine,
        namespace="unit_test",
        cache_dir=tmp_path,
        max_age_hours=1.0,
    )

    assert path.exists()
    assert cache_hit_1 is False
    assert cache_hit_2 is True
    assert engine.calls == 1
    assert list(datasets_1.keys()) == list(datasets_2.keys())
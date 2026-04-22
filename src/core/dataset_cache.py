from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import pandas as pd


CACHE_DIR = Path("data/cache/enriched")


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def dataset_cache_path(
    *,
    assets: list[str],
    data_days: int,
    namespace: str,
    cache_dir: str | Path | None = None,
) -> Path:
    root = Path(cache_dir) if cache_dir else CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)

    asset_token = "-".join(_safe_symbol(asset) for asset in assets)
    digest = hashlib.sha1(asset_token.encode("ascii", "ignore")).hexdigest()[:8]
    name = f"{namespace}_{data_days}d_{digest}.pkl"
    return root / name


def load_cached_datasets(
    path: Path,
    *,
    max_age_hours: float = 1.0,
) -> dict[str, pd.DataFrame] | None:
    if not path.exists():
        return None

    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    if age_hours > max_age_hours:
        return None

    data = pd.read_pickle(path)
    if isinstance(data, dict):
        return data
    return None


def save_cached_datasets(path: Path, datasets: dict[str, pd.DataFrame]) -> None:
    pd.to_pickle(datasets, path)


def load_or_build_datasets(
    engine: Any,
    *,
    namespace: str,
    max_age_hours: float = 1.0,
    force_refresh: bool = False,
    cache_dir: str | Path | None = None,
) -> tuple[dict[str, pd.DataFrame], Path, bool]:
    path = dataset_cache_path(
        assets=list(getattr(engine, "assets", [])),
        data_days=int(getattr(engine, "data_days", 0)),
        namespace=namespace,
        cache_dir=cache_dir,
    )

    if not force_refresh:
        cached = load_cached_datasets(path, max_age_hours=max_age_hours)
        if cached is not None:
            return cached, path, True

    datasets = engine.load_data()
    save_cached_datasets(path, datasets)
    return datasets, path, False
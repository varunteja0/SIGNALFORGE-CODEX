from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import structural as structural_module
from src.data.multi_venue import MultiVenueFetcher


def test_cross_venue_funding_returns_none_for_empty_binance_data(tmp_path) -> None:
    fetcher = MultiVenueFetcher(cache_dir=str(tmp_path))

    bybit_idx = pd.date_range("2024-01-01", periods=3, freq="8h", tz="UTC")
    bybit_df = pd.DataFrame(
        {"bybit_funding_rate": [0.0001, 0.0002, 0.00015]},
        index=bybit_idx,
    )

    fetcher.fetch_bybit_funding = lambda symbol, days: bybit_df
    fetcher.fetch_okx_funding = lambda symbol, days: None

    result = fetcher.fetch_cross_venue_funding(
        "BTC/USDT",
        days=30,
        binance_funding_df=pd.DataFrame(),
    )

    assert result is None


def test_fetch_all_keeps_other_features_when_cross_venue_fails(tmp_path) -> None:
    fetcher = MultiVenueFetcher(cache_dir=str(tmp_path))

    divergence_idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    divergence_df = pd.DataFrame(
        {
            "top_trader_ls_ratio": [1.1, 1.2, 1.15, 1.18],
            "global_ls_ratio": [0.9, 0.95, 1.0, 1.02],
            "top_retail_divergence": [0.2, 0.25, 0.15, 0.16],
            "top_retail_divergence_zscore": [0.0, 1.0, -0.5, 0.25],
        },
        index=divergence_idx,
    )

    price_idx = pd.date_range("2024-01-01", periods=4, freq="1h")
    price_df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1_000.0, 1_000.0, 1_000.0, 1_000.0],
        },
        index=price_idx,
    )

    fetcher.fetch_top_retail_divergence = lambda symbol, days: divergence_df

    def _cross_venue_failure(symbol: str, days: int) -> pd.DataFrame:
        raise AttributeError("'NaTType' object has no attribute 'unit'")

    fetcher.fetch_cross_venue_funding = _cross_venue_failure
    fetcher.fetch_okx_liquidations = lambda symbol: None

    result = fetcher.fetch_all(symbol="BTC/USDT", days=30, price_df=price_df)

    assert "top_retail_divergence_zscore" in result.columns
    assert "cross_venue_funding_zscore" not in result.columns
    assert result["top_retail_divergence_zscore"].notna().any()


def test_fetch_all_normalizes_merge_index_resolution(tmp_path) -> None:
    fetcher = MultiVenueFetcher(cache_dir=str(tmp_path))

    divergence_idx = pd.DatetimeIndex(
        np.array(
            [
                "2024-01-01T00:00:00.000",
                "2024-01-01T01:00:00.000",
                "2024-01-01T02:00:00.000",
                "2024-01-01T03:00:00.000",
            ],
            dtype="datetime64[ms]",
        )
    )
    divergence_df = pd.DataFrame(
        {
            "top_trader_ls_ratio": [1.1, 1.2, 1.15, 1.18],
            "global_ls_ratio": [0.9, 0.95, 1.0, 1.02],
            "top_retail_divergence": [0.2, 0.25, 0.15, 0.16],
            "top_retail_divergence_zscore": [0.0, 1.0, -0.5, 0.25],
        },
        index=divergence_idx,
    )

    price_idx = pd.date_range("2024-01-01", periods=4, freq="1h")
    price_df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1_000.0, 1_000.0, 1_000.0, 1_000.0],
        },
        index=price_idx,
    )

    fetcher.fetch_top_retail_divergence = lambda symbol, days: divergence_df
    fetcher.fetch_cross_venue_funding = lambda symbol, days: None
    fetcher.fetch_okx_liquidations = lambda symbol: None

    result = fetcher.fetch_all(symbol="BTC/USDT", days=30, price_df=price_df)

    assert "top_retail_divergence_zscore" in result.columns
    assert str(result.index.dtype) == "datetime64[ns]"
    assert result["top_retail_divergence_zscore"].notna().any()


def test_cross_venue_funding_normalizes_binance_symbol(monkeypatch, tmp_path) -> None:
    fetcher = MultiVenueFetcher(cache_dir=str(tmp_path))
    calls: dict[str, str] = {}

    def _fake_fetch_funding(self, symbol: str, days: int) -> pd.DataFrame:
        calls["symbol"] = symbol
        idx = pd.date_range("2024-01-01", periods=3, freq="8h")
        return pd.DataFrame({"funding_rate": [0.0001, 0.0002, 0.00015]}, index=idx)

    monkeypatch.setattr(
        structural_module.StructuralDataFetcher,
        "fetch_funding_rate_history",
        _fake_fetch_funding,
    )
    fetcher.fetch_bybit_funding = lambda symbol, days: None
    fetcher.fetch_okx_funding = lambda symbol, days: None

    result = fetcher.fetch_cross_venue_funding("BTC/USDT", days=30)

    assert calls["symbol"] == "BTCUSDT"
    assert result is not None
    assert "binance_funding_rate" in result.columns
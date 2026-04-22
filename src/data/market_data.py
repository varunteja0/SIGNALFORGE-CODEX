from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class MarketType(str, Enum):
    CRYPTO = "crypto"
    EQUITIES = "equities"
    COMMODITIES = "commodities"
    INDICES = "indices"


def coerce_market_type(value: MarketType | str) -> MarketType:
    if isinstance(value, MarketType):
        return value
    return MarketType(str(value).lower())


@dataclass(frozen=True)
class AssetSpec:
    """Normalized description of one tradable asset."""

    symbol: str
    market_type: MarketType | str = MarketType.CRYPTO
    dataset_key: str | None = None
    quote_currency: str = "USD"
    contract_multiplier: float = 1.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "market_type", coerce_market_type(self.market_type))

    @property
    def key(self) -> str:
        return self.dataset_key or self.symbol


@dataclass
class UnifiedDataBundle:
    assets: dict[str, AssetSpec]
    datasets: dict[str, pd.DataFrame]


class UnifiedMarketDataAdapter:
    """Normalize heterogeneous market datasets into one portfolio-ready schema."""

    VOL_TARGETS = {
        MarketType.CRYPTO: 0.035,
        MarketType.EQUITIES: 0.015,
        MarketType.COMMODITIES: 0.020,
        MarketType.INDICES: 0.012,
    }

    def build_bundle(
        self,
        datasets: dict[str, pd.DataFrame],
        asset_specs: dict[str, AssetSpec],
    ) -> UnifiedDataBundle:
        normalized = {}
        for key, asset in asset_specs.items():
            frame = datasets.get(asset.symbol)
            if frame is None:
                frame = datasets.get(asset.key)
            if frame is None:
                continue
            normalized[asset.symbol] = self.normalize_dataset(frame, asset)
        return UnifiedDataBundle(
            assets={asset.symbol: asset for asset in asset_specs.values()},
            datasets=normalized,
        )

    def normalize_dataset(self, frame: pd.DataFrame, asset: AssetSpec) -> pd.DataFrame:
        df = frame.copy()
        df.columns = [str(col).lower() for col in df.columns]
        df = df.sort_index()

        required = {"open", "high", "low", "close"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Dataset for {asset.symbol} missing required columns: {sorted(missing)}")

        if "volume" not in df.columns:
            df["volume"] = 0.0

        df["close"] = df["close"].astype(float)
        for col in ("open", "high", "low", "volume"):
            df[col] = df[col].astype(float)

        if "ret_1" not in df.columns:
            df["ret_1"] = df["close"].pct_change()

        if "atr_14" not in df.columns:
            tr_components = pd.concat(
                [
                    df["high"] - df["low"],
                    (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"] - df["close"].shift(1)).abs(),
                ],
                axis=1,
            )
            df["atr_14"] = tr_components.max(axis=1).rolling(14, min_periods=1).mean()

        df["atr_pct_14"] = df.get("atr_pct_14", df["atr_14"] / (df["close"].abs() + 1e-10))
        df["realized_vol_20"] = df.get(
            "realized_vol_20",
            df["ret_1"].rolling(20, min_periods=5).std(),
        )
        df["realized_vol_100"] = df.get(
            "realized_vol_100",
            df["ret_1"].rolling(100, min_periods=20).std(),
        )
        df["vol_ratio_20_100"] = df["realized_vol_20"] / (df["realized_vol_100"] + 1e-10)

        dollar_volume = df["close"] * df["volume"] * asset.contract_multiplier
        df["dollar_volume"] = df.get("dollar_volume", dollar_volume)
        df["liquidity_score"] = df["dollar_volume"] / (
            df["dollar_volume"].rolling(50, min_periods=5).median() + 1e-10
        )

        fast_ema = df["close"].ewm(span=20, adjust=False).mean()
        slow_ema = df["close"].ewm(span=60, adjust=False).mean()
        df["trend_strength_20_60"] = (fast_ema - slow_ema) / (df["atr_14"] + 1e-10)

        target = self.VOL_TARGETS[asset.market_type]
        df["cross_market_vol_adj"] = df["realized_vol_20"] / (target + 1e-10)

        if "fund_funding_rate" not in df.columns:
            df["fund_funding_rate"] = 0.0
        if "macro_signal" not in df.columns:
            df["macro_signal"] = 0.0

        return df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
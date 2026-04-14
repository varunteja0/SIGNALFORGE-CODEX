"""
Structural Crypto Data Fetcher
================================
Fetches the data that actually drives crypto price moves:
- Funding rates (where leveraged money is positioned)
- Open interest (total leverage in the system)
- Liquidation levels (where forced selling will happen)

These are PUBLIC APIs — no keys needed.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/structural")


@dataclass
class StructuralSnapshot:
    """Point-in-time structural data for an asset."""
    timestamp: float
    symbol: str
    funding_rate: float           # Current funding rate (8h)
    funding_rate_annualized: float
    open_interest_usd: float      # Total OI in USD
    oi_change_1h_pct: float       # OI change last hour
    oi_change_4h_pct: float       # OI change last 4 hours
    long_short_ratio: float       # Global long/short ratio
    mark_price: float
    index_price: float
    basis_pct: float              # (mark - index) / index — shows leverage bias


class StructuralDataFetcher:
    """Fetches structural market data from public exchange APIs.

    No API keys required — all endpoints are public.
    Focuses on Binance Futures (most liquid, most data).
    """

    BASE_URL = "https://fapi.binance.com"

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "SignalForge/2.0"})
        self._last_request_time = 0.0
        self._min_interval = 0.2  # 200ms between requests

    def _rate_limited_get(self, url: str, params: dict = None) -> Optional[dict]:
        """Rate-limited GET. Returns parsed JSON or None."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

        try:
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Request failed: {url} — {e}")
            return None

    # ================================================================
    # Funding Rate
    # ================================================================

    def fetch_funding_rate_history(
        self, symbol: str = "BTCUSDT", days: int = 90
    ) -> pd.DataFrame:
        """Fetch historical funding rates (8h intervals).

        This is the single most important structural indicator.
        High positive = overleveraged longs (short opportunity).
        High negative = overleveraged shorts (long opportunity).
        """
        cache_path = self.cache_dir / f"funding_{symbol}_{days}d.parquet"

        # Use cache if fresh (< 1 hour old)
        if cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours < 1:
                return pd.read_parquet(cache_path)

        all_data = []
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 3600 * 1000)
        current = start_time

        while current < end_time:
            data = self._rate_limited_get(
                f"{self.BASE_URL}/fapi/v1/fundingRate",
                params={
                    "symbol": symbol,
                    "startTime": current,
                    "limit": 1000,
                },
            )
            if not data:
                break

            all_data.extend(data)

            if len(data) < 1000:
                break
            current = data[-1]["fundingTime"] + 1

        if not all_data:
            logger.error(f"No funding rate data for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["funding_rate"] = df["fundingRate"].astype(float)
        df["mark_price"] = df["markPrice"].astype(float)
        df = df[["timestamp", "funding_rate", "mark_price"]].set_index("timestamp")
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # Derived features
        df["funding_annualized"] = df["funding_rate"] * 3 * 365  # 3x per day
        df["funding_cumsum"] = df["funding_rate"].cumsum()
        df["funding_ma_7d"] = df["funding_rate"].rolling(21).mean()    # 21 = 7 days * 3/day
        df["funding_ma_30d"] = df["funding_rate"].rolling(90).mean()   # 90 = 30 * 3
        df["funding_zscore"] = (
            (df["funding_rate"] - df["funding_ma_30d"])
            / (df["funding_rate"].rolling(90).std() + 1e-10)
        )

        df.to_parquet(cache_path)
        logger.info(f"Fetched {len(df)} funding rate records for {symbol}")
        return df

    # ================================================================
    # Open Interest
    # ================================================================

    def fetch_open_interest_history(
        self, symbol: str = "BTCUSDT", period: str = "1h", days: int = 90
    ) -> pd.DataFrame:
        """Fetch historical open interest.

        Rising OI + rising price = new longs (trend continues)
        Rising OI + falling price = new shorts (squeeze risk)
        Falling OI = positions closing (trend exhaustion)

        Note: Binance limits OI history to ~30 days.
        """
        # Binance only provides ~30 days of OI history
        effective_days = min(days, 30)
        cache_path = self.cache_dir / f"oi_{symbol}_{period}_{effective_days}d.parquet"

        if cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours < 1:
                return pd.read_parquet(cache_path)

        all_data = []
        end_time = int(time.time() * 1000)
        start_time = end_time - (effective_days * 24 * 3600 * 1000)
        current = start_time

        while current < end_time:
            data = self._rate_limited_get(
                f"{self.BASE_URL}/futures/data/openInterestHist",
                params={
                    "symbol": symbol,
                    "period": period,
                    "startTime": current,
                    "limit": 500,
                },
            )
            if not data:
                break

            all_data.extend(data)

            if len(data) < 500:
                break
            current = data[-1]["timestamp"] + 1

        if not all_data:
            logger.error(f"No OI data for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open_interest"] = df["sumOpenInterest"].astype(float)
        df["oi_value_usd"] = df["sumOpenInterestValue"].astype(float)
        df = df[["timestamp", "open_interest", "oi_value_usd"]].set_index("timestamp")
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # Derived features
        df["oi_change_1h"] = df["oi_value_usd"].pct_change(1)
        df["oi_change_4h"] = df["oi_value_usd"].pct_change(4)
        df["oi_change_24h"] = df["oi_value_usd"].pct_change(24)
        df["oi_ma_24h"] = df["oi_value_usd"].rolling(24).mean()
        df["oi_zscore"] = (
            (df["oi_value_usd"] - df["oi_ma_24h"])
            / (df["oi_value_usd"].rolling(24).std() + 1e-10)
        )

        df.to_parquet(cache_path)
        logger.info(f"Fetched {len(df)} OI records for {symbol}")
        return df

    # ================================================================
    # Long/Short Ratio
    # ================================================================

    def fetch_long_short_ratio(
        self, symbol: str = "BTCUSDT", period: str = "1h", days: int = 30
    ) -> pd.DataFrame:
        """Fetch global long/short account ratio.

        When everyone is long → contrarian short signal.
        When everyone is short → contrarian long signal.
        """
        cache_path = self.cache_dir / f"lsr_{symbol}_{period}_{days}d.parquet"

        if cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours < 1:
                return pd.read_parquet(cache_path)

        all_data = []
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 3600 * 1000)
        current = start_time

        while current < end_time:
            data = self._rate_limited_get(
                f"{self.BASE_URL}/futures/data/globalLongShortAccountRatio",
                params={
                    "symbol": symbol,
                    "period": period,
                    "startTime": current,
                    "limit": 500,
                },
            )
            if not data:
                break

            all_data.extend(data)

            if len(data) < 500:
                break
            current = data[-1]["timestamp"] + 1

        if not all_data:
            logger.error(f"No long/short ratio data for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["long_short_ratio"] = df["longShortRatio"].astype(float)
        df["long_account_pct"] = df["longAccount"].astype(float)
        df["short_account_pct"] = df["shortAccount"].astype(float)
        df = df[["timestamp", "long_short_ratio", "long_account_pct", "short_account_pct"]]
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # Derived
        df["lsr_ma_24h"] = df["long_short_ratio"].rolling(24).mean()
        df["lsr_zscore"] = (
            (df["long_short_ratio"] - df["lsr_ma_24h"])
            / (df["long_short_ratio"].rolling(24).std() + 1e-10)
        )

        df.to_parquet(cache_path)
        logger.info(f"Fetched {len(df)} L/S ratio records for {symbol}")
        return df

    # ================================================================
    # Taker Buy/Sell Volume
    # ================================================================

    def fetch_taker_volume(
        self, symbol: str = "BTCUSDT", period: str = "1h", days: int = 30
    ) -> pd.DataFrame:
        """Fetch taker buy/sell volume ratio.

        Taker buys = aggressive buying (market orders hitting asks).
        High taker buy ratio during OI spike = real demand.
        High taker buy ratio during OI drop = short covering (fake rally).
        """
        cache_path = self.cache_dir / f"taker_{symbol}_{period}_{days}d.parquet"

        if cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours < 1:
                return pd.read_parquet(cache_path)

        all_data = []
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 3600 * 1000)
        current = start_time

        while current < end_time:
            data = self._rate_limited_get(
                f"{self.BASE_URL}/futures/data/takerlongshortRatio",
                params={
                    "symbol": symbol,
                    "period": period,
                    "startTime": current,
                    "limit": 500,
                },
            )
            if not data:
                break

            all_data.extend(data)

            if len(data) < 500:
                break
            current = data[-1]["timestamp"] + 1

        if not all_data:
            logger.error(f"No taker volume data for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["buy_sell_ratio"] = df["buySellRatio"].astype(float)
        df["buy_vol"] = df["buyVol"].astype(float)
        df["sell_vol"] = df["sellVol"].astype(float)
        df = df[["timestamp", "buy_sell_ratio", "buy_vol", "sell_vol"]]
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # Derived
        df["net_taker_vol"] = df["buy_vol"] - df["sell_vol"]
        df["taker_imbalance"] = df["net_taker_vol"] / (df["buy_vol"] + df["sell_vol"] + 1e-10)
        df["taker_imbalance_ma"] = df["taker_imbalance"].rolling(24).mean()

        df.to_parquet(cache_path)
        logger.info(f"Fetched {len(df)} taker volume records for {symbol}")
        return df

    # ================================================================
    # Combine All Structural Data
    # ================================================================

    def fetch_all(
        self,
        symbol: str = "BTCUSDT",
        price_df: Optional[pd.DataFrame] = None,
        days: int = 90,
    ) -> pd.DataFrame:
        """Fetch and merge all structural data into one DataFrame.

        If price_df is provided, merges structural data onto price index.
        This is the primary method to call — gives you everything.
        """
        logger.info(f"Fetching all structural data for {symbol} ({days} days)")

        funding = self.fetch_funding_rate_history(symbol, days=days)
        oi = self.fetch_open_interest_history(symbol, period="1h", days=days)
        lsr = self.fetch_long_short_ratio(symbol, period="1h", days=min(days, 30))
        taker = self.fetch_taker_volume(symbol, period="1h", days=min(days, 30))

        # Prefix columns to avoid conflicts
        funding = funding.add_prefix("fund_")
        oi = oi.add_prefix("oi_")
        lsr = lsr.add_prefix("lsr_")
        taker = taker.add_prefix("taker_")

        if price_df is not None and not price_df.empty:
            # Merge everything onto price index using forward-fill
            result = price_df.copy()
            for struct_df in [funding, oi, lsr, taker]:
                if not struct_df.empty:
                    result = pd.merge_asof(
                        result.sort_index(),
                        struct_df.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction="backward",
                    )
            # Composite features
            result = self._add_composite_features(result)
            return result
        else:
            # Just merge structural data together
            result = oi if not oi.empty else pd.DataFrame()
            for struct_df in [funding, lsr, taker]:
                if not struct_df.empty and not result.empty:
                    result = pd.merge_asof(
                        result.sort_index(),
                        struct_df.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction="backward",
                    )
            return result

    def _add_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composite structural features that combine multiple data sources."""
        # Leverage heat: high OI + extreme funding = overleveraged market
        if "oi_oi_zscore" in df.columns and "fund_funding_zscore" in df.columns:
            df["leverage_heat"] = (
                df["oi_oi_zscore"].abs() + df["fund_funding_zscore"].abs()
            ) / 2

        # Liquidation pressure: high leverage + price moving against positions
        if "fund_funding_rate" in df.columns and "close" in df.columns:
            price_ret_4h = df["close"].pct_change(4)
            # Positive funding = longs dominate; if price drops, longs get liquidated
            df["liq_pressure_long"] = (
                df["fund_funding_rate"].clip(lower=0) * (-price_ret_4h).clip(lower=0) * 10000
            )
            # Negative funding = shorts dominate; if price rises, shorts get squeezed
            df["liq_pressure_short"] = (
                (-df["fund_funding_rate"]).clip(lower=0) * price_ret_4h.clip(lower=0) * 10000
            )
            df["liq_pressure"] = df["liq_pressure_long"] + df["liq_pressure_short"]

        # Smart money indicator: taker flow vs crowd positioning
        if "taker_taker_imbalance" in df.columns and "lsr_long_short_ratio" in df.columns:
            # When taker flow diverges from crowd positioning → smart money signal
            df["smart_money_divergence"] = (
                df["taker_taker_imbalance"] - (df["lsr_long_short_ratio"] - 1)
            )

        return df

"""
Open Interest Fetcher — Real Exchange Data
=============================================
Fetches historical and current open interest from Binance and Bybit futures.
PUBLIC endpoints — no API keys required.

Open interest is critical for:
- Measuring leverage in the system (high OI = crowded, flush incoming)
- Confirming liquidation cascades (OI dropping = leverage being purged)
- Gauging cascade fuel (high OI after spike = more liquidations possible)
"""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Binance public endpoint for OI history (no auth needed)
BINANCE_OI_URL = "https://fapi.binance.com/futures/data/openInterestHist"
BINANCE_OI_CURRENT_URL = "https://fapi.binance.com/fapi/v1/openInterest"

# Bybit public endpoint
BYBIT_OI_URL = "https://api.bybit.com/v5/market/open-interest"


def _symbol_to_binance(symbol: str) -> str:
    """Convert 'BTC/USDT' → 'BTCUSDT'."""
    return symbol.replace("/", "").replace(":USDT", "")


def _symbol_to_bybit(symbol: str) -> str:
    """Convert 'BTC/USDT' → 'BTCUSDT'."""
    return symbol.replace("/", "").replace(":USDT", "")


class OpenInterestFetcher:
    """Fetches real open interest data from crypto exchanges."""

    def __init__(self, exchange_id: str = "binance"):
        self.exchange_id = exchange_id

    def fetch_current(self, symbol: str) -> Optional[dict]:
        """Fetch latest open interest for a symbol."""
        try:
            if self.exchange_id == "binance":
                return self._binance_current(symbol)
            else:
                return self._bybit_current(symbol)
        except Exception as e:
            logger.error(f"Failed to fetch current OI for {symbol}: {e}")
            return None

    def fetch_history(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 90,
    ) -> pd.DataFrame:
        """Fetch historical open interest data.

        Args:
            symbol: e.g. "BTC/USDT"
            timeframe: "5m", "15m", "30m", "1h", "2h", "4h", "1d"
            days: how many days of history

        Returns:
            DataFrame indexed by timestamp with columns:
                oi_value (in USDT), oi_contracts
        """
        try:
            if self.exchange_id == "binance":
                return self._binance_history(symbol, timeframe, days)
            else:
                return self._bybit_history(symbol, timeframe, days)
        except Exception as e:
            logger.error(f"Failed to fetch OI history for {symbol}: {e}")
            return pd.DataFrame()

    def _binance_current(self, symbol: str) -> dict:
        """Fetch current OI from Binance."""
        pair = _symbol_to_binance(symbol)
        resp = requests.get(
            BINANCE_OI_CURRENT_URL,
            params={"symbol": pair},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "symbol": symbol,
            "oi_contracts": float(data["openInterest"]),
            "timestamp": int(data.get("time", time.time() * 1000)),
        }

    def _binance_history(
        self, symbol: str, timeframe: str, days: int
    ) -> pd.DataFrame:
        """Binance open interest history (max 30 days per request)."""
        pair = _symbol_to_binance(symbol)

        # Map timeframe to Binance period format
        tf_map = {
            "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1d",
        }
        period = tf_map.get(timeframe, "1h")

        all_records = []
        start_ms = int((time.time() - days * 86400) * 1000)
        now_ms = int(time.time() * 1000)

        while start_ms < now_ms:
            # Binance limits to 500 records per request
            end_ms = min(start_ms + 30 * 86400 * 1000, now_ms)
            try:
                resp = requests.get(
                    BINANCE_OI_URL,
                    params={
                        "symbol": pair,
                        "period": period,
                        "startTime": start_ms,
                        "endTime": end_ms,
                        "limit": 500,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()

                if not data:
                    break

                all_records.extend(data)
                start_ms = int(data[-1]["timestamp"]) + 1
                time.sleep(0.3)

            except Exception as e:
                logger.warning(f"Binance OI fetch chunk failed: {e}")
                break

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.rename(columns={
            "sumOpenInterest": "oi_contracts",
            "sumOpenInterestValue": "oi_value",
        })
        df["oi_contracts"] = df["oi_contracts"].astype(float)
        df["oi_value"] = df["oi_value"].astype(float)
        df = df[["oi_contracts", "oi_value"]]
        df = df[~df.index.duplicated(keep="last")].sort_index()

        logger.info(f"Fetched {len(df)} OI records for {symbol}")
        return df

    def _bybit_current(self, symbol: str) -> dict:
        """Fetch current OI from Bybit."""
        pair = _symbol_to_bybit(symbol)
        resp = requests.get(
            BYBIT_OI_URL,
            params={
                "category": "linear",
                "symbol": pair,
                "intervalTime": "1h",
                "limit": 1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("result", {}).get("list"):
            item = data["result"]["list"][0]
            return {
                "symbol": symbol,
                "oi_contracts": float(item.get("openInterest", 0)),
                "timestamp": int(item.get("timestamp", time.time() * 1000)),
            }
        return {"symbol": symbol, "oi_contracts": 0, "timestamp": 0}

    def _bybit_history(
        self, symbol: str, timeframe: str, days: int
    ) -> pd.DataFrame:
        """Bybit open interest history."""
        pair = _symbol_to_bybit(symbol)

        tf_map = {
            "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "4h": "4h", "1d": "1d",
        }
        interval = tf_map.get(timeframe, "1h")

        all_records = []
        # Bybit uses cursor-based pagination
        cursor = None
        start_ms = int((time.time() - days * 86400) * 1000)

        for _ in range(100):  # Safety limit
            params = {
                "category": "linear",
                "symbol": pair,
                "intervalTime": interval,
                "startTime": start_ms,
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                resp = requests.get(BYBIT_OI_URL, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                items = data.get("result", {}).get("list", [])

                if not items:
                    break

                all_records.extend(items)
                cursor = data.get("result", {}).get("nextPageCursor")

                if not cursor:
                    break

                time.sleep(0.3)

            except Exception as e:
                logger.warning(f"Bybit OI fetch failed: {e}")
                break

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df.set_index("timestamp", inplace=True)
        df["oi_contracts"] = df["openInterest"].astype(float)
        df["oi_value"] = df["oi_contracts"]  # Bybit may not provide USDT value directly
        df = df[["oi_contracts", "oi_value"]]
        df = df[~df.index.duplicated(keep="last")].sort_index()

        logger.info(f"Fetched {len(df)} OI records for {symbol} from Bybit")
        return df

    def resample_to_ohlcv(
        self,
        oi_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align OI data to OHLCV timeframe via forward-fill.

        Returns DataFrame with oi_value and oi_change_pct columns
        aligned to ohlcv_df.index.
        """
        if oi_df.empty:
            return pd.DataFrame({
                "oi_value": 0.0,
                "oi_change_pct": 0.0,
            }, index=ohlcv_df.index)

        aligned = oi_df.reindex(ohlcv_df.index, method="ffill").fillna(method="bfill")

        result = pd.DataFrame(index=ohlcv_df.index)
        result["oi_value"] = aligned["oi_value"].fillna(0)
        result["oi_change_pct"] = result["oi_value"].pct_change().fillna(0)

        return result

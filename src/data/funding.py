"""
Funding Rate Fetcher — Real Exchange Data
==========================================
Fetches historical and current funding rates from Binance and Bybit futures.
These are PUBLIC endpoints — no API keys required.

Funding rates are critical for:
- Detecting crowded trades (extreme funding = leverage flush incoming)
- Timing entries (enter AFTER funding payment when pain is fresh)
- Confirming liquidation setups (negative funding = market already flushed)
"""

import logging
import time
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FundingRateFetcher:
    """Fetches real funding rate data from crypto exchanges.

    Binance/Bybit futures publish funding rates every 8 hours.
    Payment times: 00:00, 08:00, 16:00 UTC.
    """

    def __init__(self, exchange_id: str = "binance"):
        self.exchange_id = exchange_id
        self._init_exchange()

    def _init_exchange(self):
        """Initialize exchange with futures market type."""
        config = {
            "enableRateLimit": True,
            "timeout": 15000,
            "options": {"defaultType": "future"},
        }
        try:
            self.exchange = getattr(ccxt, self.exchange_id)(config)
            self.exchange.load_markets()
            logger.info(f"FundingRateFetcher connected to {self.exchange_id}")
        except Exception as e:
            logger.warning(f"{self.exchange_id} failed, falling back to bybit: {e}")
            self.exchange_id = "bybit"
            self.exchange = ccxt.bybit(config)
            self.exchange.load_markets()

    def fetch_current(self, symbol: str) -> Optional[dict]:
        """Fetch the current/latest funding rate for a symbol.

        Returns:
            dict with keys: symbol, funding_rate, next_funding_time, timestamp
        """
        try:
            # ccxt unified method
            result = self.exchange.fetch_funding_rate(symbol)
            return {
                "symbol": symbol,
                "funding_rate": result.get("fundingRate", 0),
                "next_funding_time": result.get("fundingTimestamp"),
                "timestamp": result.get("timestamp", int(time.time() * 1000)),
                "mark_price": result.get("markPrice", 0),
                "index_price": result.get("indexPrice", 0),
            }
        except Exception as e:
            logger.error(f"Failed to fetch current funding for {symbol}: {e}")
            return None

    def fetch_history(
        self,
        symbol: str,
        days: int = 90,
        since_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch historical funding rate data.

        Returns DataFrame indexed by timestamp with columns:
            funding_rate, timestamp
        """
        if since_ms is None:
            since_ms = int((time.time() - days * 86400) * 1000)

        all_rates = []

        try:
            while True:
                rates = self.exchange.fetch_funding_rate_history(
                    symbol, since=since_ms, limit=1000
                )
                if not rates:
                    break

                all_rates.extend(rates)
                last_ts = rates[-1]["timestamp"]

                # Stop if we've reached current time
                if last_ts >= int(time.time() * 1000) - 3_600_000:
                    break

                since_ms = last_ts + 1
                time.sleep(0.2)  # Rate limit

        except Exception as e:
            logger.error(f"Failed to fetch funding history for {symbol}: {e}")

        if not all_rates:
            logger.warning(f"No funding rate history for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(all_rates)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
        df = df[~df.index.duplicated(keep="last")].sort_index()

        logger.info(f"Fetched {len(df)} funding rate records for {symbol}")
        return df

    def fetch_multi(
        self,
        symbols: list[str],
        days: int = 90,
    ) -> dict[str, pd.DataFrame]:
        """Fetch funding history for multiple symbols."""
        result = {}
        for symbol in symbols:
            result[symbol] = self.fetch_history(symbol, days)
            time.sleep(0.3)
        return result

    def resample_to_ohlcv(
        self,
        funding_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
    ) -> pd.Series:
        """Align funding rates to OHLCV timeframe via forward-fill.

        Funding rates are published every 8 hours.
        This forward-fills them to match any OHLCV timeframe (1h, 4h, etc).

        Returns a Series aligned to ohlcv_df.index.
        """
        if funding_df.empty:
            return pd.Series(0.0, index=ohlcv_df.index, name="funding_rate")

        # Reindex to OHLCV timestamps, forward-fill
        aligned = funding_df["funding_rate"].reindex(
            ohlcv_df.index, method="ffill"
        ).fillna(0)

        return aligned

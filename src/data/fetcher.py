"""
SignalForge — Offline Data Fetcher
====================================
Fetches OHLCV data from PUBLIC crypto exchange APIs (no API keys required).
Supports multiple exchanges with automatic fallback: Bybit → KuCoin → Binance.
Handles pagination, caching, and incremental updates.

Stores data as parquet files for fast loading.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")

# Ordered by reliability; first working exchange wins
EXCHANGE_CHAIN = [
    ("bybit", {"enableRateLimit": True, "timeout": 15000}),
    ("kucoin", {"enableRateLimit": True, "timeout": 15000}),
    ("binance", {"enableRateLimit": True, "timeout": 15000, "options": {"defaultType": "future"}}),
]


class DataFetcher:
    """Fetches historical OHLCV from public exchange APIs. No API keys needed."""

    def __init__(self, cache_dir: Optional[str] = None, exchange_id: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.exchange = None
        self.exchange_id = None

        if exchange_id:
            # Use specific exchange
            self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 15000})
            self.exchange_id = exchange_id
        else:
            # Auto-detect working exchange
            self._connect_best_exchange()

    def _connect_best_exchange(self):
        """Try exchanges in order until one responds."""
        for ex_id, config in EXCHANGE_CHAIN:
            try:
                ex = getattr(ccxt, ex_id)(config)
                # Quick connectivity check — fetch 1 candle
                ex.fetch_ohlcv("BTC/USDT", "1h", limit=1)
                self.exchange = ex
                self.exchange_id = ex_id
                logger.info(f"Connected to {ex_id}")
                return
            except Exception as e:
                logger.warning(f"{ex_id} failed: {e}")
                continue
        raise ConnectionError(
            "All exchanges unreachable. Check internet connection."
        )

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        safe_name = symbol.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_name}_{timeframe}.parquet"

    def fetch(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 730,
        force: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV data, using cache when available.

        Args:
            symbol: e.g. "BTC/USDT"
            timeframe: e.g. "1h", "4h", "1d"
            days: how many days of history to fetch
            force: if True, ignore cache and re-fetch
        """
        cache_path = self._cache_path(symbol, timeframe)

        # Try to load cache
        if not force and cache_path.exists():
            cached = pd.read_parquet(cache_path)
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600

            if age_hours < 1:
                logger.info(
                    f"Using fresh cache for {symbol} {timeframe} "
                    f"({len(cached)} bars, {age_hours:.1f}h old)"
                )
                return cached

            # Incremental update: fetch only new bars
            last_ts = int(cached.index[-1].timestamp() * 1000) + 1
            new_bars = self._fetch_from_exchange(symbol, timeframe, since_ms=last_ts)

            if not new_bars.empty:
                df = pd.concat([cached, new_bars])
                df = df[~df.index.duplicated(keep="last")].sort_index()
                df.to_parquet(cache_path)
                logger.info(
                    f"Updated cache: +{len(new_bars)} bars for {symbol} {timeframe}"
                )
                return df

            return cached

        # Full fetch
        logger.info(f"Fetching {days} days of {symbol} {timeframe} from {self.exchange_id}...")
        since_ms = int((time.time() - days * 86400) * 1000)
        df = self._fetch_from_exchange(symbol, timeframe, since_ms=since_ms)

        if not df.empty:
            df.to_parquet(cache_path)
            logger.info(f"Cached {len(df)} bars to {cache_path}")

        return df

    def _fetch_from_exchange(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        max_bars: int = 100_000,
    ) -> pd.DataFrame:
        """Paginated fetch with rate limiting."""
        all_candles = []
        # Bybit max = 1000, KuCoin = 1500, Binance = 1000
        batch_limits = {"bybit": 1000, "kucoin": 1500, "binance": 1000}
        batch_size = batch_limits.get(self.exchange_id, 200)
        now_ms = int(time.time() * 1000)

        while len(all_candles) < max_bars:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ms, limit=batch_size
                )
            except Exception as e:
                logger.error(f"Fetch error for {symbol} {timeframe}: {e}")
                break

            if not candles:
                break

            all_candles.extend(candles)
            last_ts = candles[-1][0]
            since_ms = last_ts + 1

            # Progress logging every 5k bars
            if len(all_candles) % 5000 < len(candles):
                logger.info(f"  ... {len(all_candles)} bars fetched so far")

            # Stop if we've reached current time
            if last_ts >= now_ms - 3_600_000:
                break

            # Rate-limit courtesy
            time.sleep(0.15)

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="last")].sort_index()

        return df

    def fetch_multi(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        days: int = 730,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.fetch(symbol, timeframe, days)
                logger.info(f"  {symbol}: {len(result[symbol])} bars")
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                result[symbol] = pd.DataFrame()
        return result


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features needed by Alpha Genome.

    This is a standalone version that matches DataEngine.compute_features
    exactly, but doesn't require an exchange connection.
    """
    if df.empty:
        return df

    df = df.copy()

    # Returns at multiple horizons
    for period in [1, 3, 5, 10, 20, 50]:
        df[f"ret_{period}"] = df["close"].pct_change(period)

    # Volatility (realized)
    for window in [10, 20, 50]:
        df[f"vol_{window}"] = df["close"].pct_change().rolling(window).std()

    # Volume features
    for window in [10, 20, 50]:
        df[f"vol_ratio_{window}"] = df["volume"] / (
            df["volume"].rolling(window).mean() + 1e-10
        )

    # Price vs moving averages
    for window in [10, 20, 50, 100, 200]:
        ma = df["close"].rolling(window).mean()
        df[f"ma_{window}"] = ma
        df[f"price_vs_ma_{window}"] = (df["close"] - ma) / (ma + 1e-10)

    # RSI
    for period in [7, 14, 21]:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    for window in [20]:
        ma = df["close"].rolling(window).mean()
        std = df["close"].rolling(window).std()
        df[f"bb_upper_{window}"] = ma + 2 * std
        df[f"bb_lower_{window}"] = ma - 2 * std
        df[f"bb_pct_{window}"] = (df["close"] - df[f"bb_lower_{window}"]) / (
            df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"] + 1e-10
        )

    # ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    for period in [14, 21]:
        df[f"atr_{period}"] = true_range.rolling(period).mean()
        df[f"atr_pct_{period}"] = df[f"atr_{period}"] / (df["close"] + 1e-10)

    # Order flow proxy
    df["bar_position"] = (df["close"] - df["low"]) / (
        df["high"] - df["low"] + 1e-10
    )

    # Momentum rank
    for window in [20, 50]:
        df[f"momentum_rank_{window}"] = (
            df["ret_1"]
            .rolling(window)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        )

    return df

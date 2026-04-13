"""
SignalForge Data Ingestion Engine
=================================
Pulls OHLCV, order book, and funding rate data from exchanges via CCXT.
Handles rate limiting, caching, and multi-timeframe alignment.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class DataEngine:
    """Fetches and manages market data from crypto exchanges."""

    def __init__(self, exchange_name: str = "binance", testnet: bool = True):
        self.exchange_name = exchange_name
        self.testnet = testnet
        self.exchange = self._init_exchange()
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _init_exchange(self) -> ccxt.Exchange:
        exchange_class = getattr(ccxt, self.exchange_name)
        config = {
            "apiKey": os.getenv("BINANCE_API_KEY"),
            "secret": os.getenv("BINANCE_SECRET"),
            "enableRateLimit": True,
            "options": {"defaultType": "future"},  # Use futures for leverage + shorting
        }

        exchange = exchange_class(config)

        if self.testnet:
            exchange.set_sandbox_mode(True)
            logger.info("TESTNET MODE ACTIVE — no real money at risk")

        return exchange

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 1000,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV candlestick data."""
        logger.info(f"Fetching {symbol} {timeframe} x{limit}")

        all_candles = []
        fetched = 0

        while fetched < limit:
            batch_limit = min(1000, limit - fetched)
            candles = self.exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=batch_limit
            )

            if not candles:
                break

            all_candles.extend(candles)
            fetched += len(candles)
            since = candles[-1][0] + 1  # Next ms after last candle

            if len(candles) < batch_limit:
                break

        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        logger.info(f"Got {len(df)} candles for {symbol} {timeframe}")
        return df

    def fetch_order_book(self, symbol: str, depth: int = 50) -> dict:
        """Fetch L2 order book — reveals hidden supply/demand zones."""
        book = self.exchange.fetch_order_book(symbol, limit=depth)
        return {
            "bids": pd.DataFrame(book["bids"], columns=["price", "quantity"]),
            "asks": pd.DataFrame(book["asks"], columns=["price", "quantity"]),
            "timestamp": book.get("timestamp"),
            "bid_total": sum(b[1] for b in book["bids"]),
            "ask_total": sum(a[1] for a in book["asks"]),
            "imbalance": (
                sum(b[1] for b in book["bids"]) - sum(a[1] for a in book["asks"])
            )
            / (sum(b[1] for b in book["bids"]) + sum(a[1] for a in book["asks"]) + 1e-10),
        }

    def fetch_funding_rate(self, symbol: str) -> dict:
        """Fetch funding rate — tells you where leveraged traders are positioned."""
        try:
            funding = self.exchange.fetch_funding_rate(symbol)
            return {
                "rate": funding.get("fundingRate", 0),
                "timestamp": funding.get("fundingTimestamp"),
                "next_timestamp": funding.get("nextFundingTimestamp"),
            }
        except Exception as e:
            logger.warning(f"Could not fetch funding rate for {symbol}: {e}")
            return {"rate": 0, "timestamp": None, "next_timestamp": None}

    def fetch_all_data(
        self, symbols: list, timeframes: list, limit: int = 1000
    ) -> dict:
        """Fetch OHLCV for all symbol/timeframe combos. Returns nested dict."""
        data = {}
        total = len(symbols) * len(timeframes)
        fetched = 0

        for symbol in symbols:
            data[symbol] = {}
            for tf in timeframes:
                try:
                    data[symbol][tf] = self.fetch_ohlcv(symbol, tf, limit)
                    fetched += 1
                    logger.info(f"Progress: {fetched}/{total}")
                except Exception as e:
                    logger.error(f"Failed {symbol} {tf}: {e}")
                    data[symbol][tf] = pd.DataFrame()

        return data

    def compute_features(self, df: pd.DataFrame, advanced: bool = True) -> pd.DataFrame:
        """Add computed features to OHLCV data — these become signal ingredients.

        Args:
            df: OHLCV DataFrame
            advanced: If True, uses the 120+ feature engine. If False, uses legacy 32 features.
        """
        if df.empty:
            return df

        if advanced:
            from src.data.features import compute_all_features
            return compute_all_features(df)

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

        # ATR (Average True Range) — for stop loss calculation
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        for period in [14, 21]:
            df[f"atr_{period}"] = true_range.rolling(period).mean()
            df[f"atr_pct_{period}"] = df[f"atr_{period}"] / (df["close"] + 1e-10)

        # Order flow proxy: close position within bar
        df["bar_position"] = (df["close"] - df["low"]) / (
            df["high"] - df["low"] + 1e-10
        )

        # Momentum rank (percentile of return over lookback)
        for window in [20, 50]:
            df[f"momentum_rank_{window}"] = (
                df["ret_1"].rolling(window).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )
            )

        return df

    def save_cache(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Cache data locally to avoid re-fetching."""
        fname = f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        path = self.cache_dir / fname
        df.to_parquet(path)
        logger.info(f"Cached {symbol} {timeframe} -> {path}")

    def load_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load cached data if available and recent."""
        fname = f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        path = self.cache_dir / fname

        if path.exists():
            df = pd.read_parquet(path)
            age_hours = (time.time() - path.stat().st_mtime) / 3600
            if age_hours < 1:  # Cache valid for 1 hour
                logger.info(f"Using cached {symbol} {timeframe} (age: {age_hours:.1f}h)")
                return df

        return None

    def enrich_with_onchain(
        self, df: pd.DataFrame, asset: str = "ETH", days: int = 90
    ) -> pd.DataFrame:
        """Merge on-chain features into market data for GP evolution.

        Fetches historical on-chain snapshots and merges derived features
        into the price DataFrame. Columns that don't exist in the on-chain
        data are silently skipped (FeatureNode returns 0 for missing cols).
        """
        from src.data.onchain import OnChainFetcher

        fetcher = OnChainFetcher(use_live=True)
        onchain_df = fetcher.fetch_historical_snapshots(asset=asset, days=days)

        if onchain_df.empty:
            logger.info("No on-chain data available; skipping enrichment")
            return df

        # Resample on-chain data to match market data frequency
        onchain_df.index = pd.to_datetime(onchain_df.index)
        df_idx = pd.to_datetime(df.index)

        # Forward-fill on-chain data to market frequency
        combined = df.copy()
        for col in onchain_df.columns:
            if col not in combined.columns:
                resampled = onchain_df[col].reindex(df_idx, method="ffill")
                combined[col] = resampled

        # Compute derived on-chain features
        combined = fetcher.compute_onchain_features(combined)

        n_new = len([c for c in combined.columns if c not in df.columns])
        logger.info(f"Enriched with {n_new} on-chain features for {asset}")
        return combined

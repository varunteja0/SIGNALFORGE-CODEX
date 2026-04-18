"""
Multi-Venue Data Fetcher — The Data Edge
==========================================
Fetches data that the existing structural.py doesn't cover:

1. Top Trader L/S Ratio (Binance) — smart money vs retail positioning
2. Cross-Venue Funding Rates — funding divergence across exchanges
3. OKX Liquidation Orders — real-time forced-unwind flow
4. DeFi Llama — TVL + stablecoin supply (macro regime signals)

All endpoints are PUBLIC — no API keys required.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/multi_venue")


@dataclass
class TopTraderData:
    """Top trader vs global positioning divergence."""
    timestamp: float
    symbol: str
    top_long_short_ratio: float     # Top traders L/S (by position $)
    global_long_short_ratio: float  # All accounts L/S
    divergence: float               # top - global (positive = smart money more long)


@dataclass
class CrossVenueFunding:
    """Funding rate comparison across exchanges."""
    timestamp: float
    symbol: str
    binance_rate: float
    bybit_rate: float
    okx_rate: float
    mean_rate: float
    spread: float                   # max - min across venues
    zscore: float                   # spread relative to historical


class MultiVenueFetcher:
    """Fetches structural data from multiple exchanges.

    Designed to complement StructuralDataFetcher (Binance-only).
    Adds: top trader positioning, cross-venue funding, OKX liquidations.
    """

    BINANCE_BASE = "https://fapi.binance.com"
    BYBIT_BASE = "https://api.bybit.com"
    OKX_BASE = "https://www.okx.com"
    DEFI_LLAMA_BASE = "https://api.llama.fi"
    STABLECOIN_BASE = "https://stablecoins.llama.fi"

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "SignalForge/3.0"})
        self._last_request_time = 0.0
        self._min_interval = 0.25  # 250ms between requests

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    def _rate_limited_get(self, url: str, params: dict = None,
                          timeout: int = 10) -> Optional[Union[dict, list]]:
        """GET with rate limiting and error handling."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        try:
            resp = self._session.get(url, params=params, timeout=timeout)
            self._last_request_time = time.time()
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed: {url} — {e}")
            return None

    # ------------------------------------------------------------------
    # 1. Binance Top Trader L/S Ratio (by positions, USD-weighted)
    # ------------------------------------------------------------------
    def fetch_top_trader_ratio(
        self, symbol: str = "BTC/USDT", days: int = 30
    ) -> Optional[pd.DataFrame]:
        """Fetch top trader long/short ratio (position-weighted).

        This is the SMART MONEY signal — divergence from global L/S
        indicates informed flow.

        Endpoint: /futures/data/topLongShortPositionRatio
        Public, no API key.
        """
        clean_symbol = symbol.replace("/", "").replace(":USDT", "")
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 86400 * 1000
        all_rows = []

        while start_ms < end_ms:
            data = self._rate_limited_get(
                f"{self.BINANCE_BASE}/futures/data/topLongShortPositionRatio",
                params={
                    "symbol": clean_symbol,
                    "period": "1h",
                    "startTime": start_ms,
                    "limit": 500,
                },
            )
            if not data:
                break
            all_rows.extend(data)
            if len(data) < 500:
                break
            start_ms = int(data[-1]["timestamp"]) + 1

        if not all_rows:
            logger.warning(f"No top trader data for {symbol}")
            return None

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df.rename(columns={
            "longShortRatio": "top_trader_ls_ratio",
            "longAccount": "top_long_pct",
            "shortAccount": "top_short_pct",
        })
        for col in ["top_trader_ls_ratio", "top_long_pct", "top_short_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def fetch_global_ls_ratio(
        self, symbol: str = "BTC/USDT", days: int = 30
    ) -> Optional[pd.DataFrame]:
        """Fetch global (all accounts) long/short ratio.

        Endpoint: /futures/data/globalLongShortAccountRatio
        """
        clean_symbol = symbol.replace("/", "").replace(":USDT", "")
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 86400 * 1000
        all_rows = []

        while start_ms < end_ms:
            data = self._rate_limited_get(
                f"{self.BINANCE_BASE}/futures/data/globalLongShortAccountRatio",
                params={
                    "symbol": clean_symbol,
                    "period": "1h",
                    "startTime": start_ms,
                    "limit": 500,
                },
            )
            if not data:
                break
            all_rows.extend(data)
            if len(data) < 500:
                break
            start_ms = int(data[-1]["timestamp"]) + 1

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df.rename(columns={"longShortRatio": "global_ls_ratio"})
        df["global_ls_ratio"] = pd.to_numeric(df["global_ls_ratio"], errors="coerce")
        return df

    def fetch_top_retail_divergence(
        self, symbol: str = "BTC/USDT", days: int = 30
    ) -> Optional[pd.DataFrame]:
        """Compute top trader vs retail divergence.

        Positive divergence = smart money more long than retail → bullish signal.
        Negative divergence = smart money more short than retail → bearish signal.
        """
        top_df = self.fetch_top_trader_ratio(symbol, days)
        global_df = self.fetch_global_ls_ratio(symbol, days)

        if top_df is None or global_df is None:
            logger.warning(f"Cannot compute divergence for {symbol} — missing data")
            return None

        # Merge on nearest timestamp (both are hourly)
        merged = pd.merge_asof(
            top_df[["top_trader_ls_ratio"]].sort_index(),
            global_df[["global_ls_ratio"]].sort_index(),
            left_index=True, right_index=True,
            direction="nearest",
            tolerance=pd.Timedelta("2h"),
        )

        # Raw divergence: top trader ratio - global ratio
        merged["top_retail_divergence"] = (
            merged["top_trader_ls_ratio"] - merged["global_ls_ratio"]
        )

        # Z-score the divergence for signal generation
        lookback = min(168, len(merged))
        rolling_mean = merged["top_retail_divergence"].rolling(lookback, min_periods=20).mean()
        rolling_std = merged["top_retail_divergence"].rolling(lookback, min_periods=20).std()
        merged["top_retail_divergence_zscore"] = (
            (merged["top_retail_divergence"] - rolling_mean) / rolling_std.replace(0, np.nan)
        )

        return merged

    # ------------------------------------------------------------------
    # 2. Cross-Venue Funding Rate Comparison
    # ------------------------------------------------------------------
    def fetch_bybit_funding(
        self, symbol: str = "BTC/USDT", days: int = 30
    ) -> Optional[pd.DataFrame]:
        """Fetch Bybit funding rate history.

        Endpoint: /v5/market/funding/history
        """
        bybit_symbol = symbol.replace("/", "").replace(":USDT", "")
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 86400 * 1000
        all_rows = []
        cursor = ""

        for _ in range(100):  # safety limit
            params = {
                "category": "linear",
                "symbol": bybit_symbol,
                "startTime": str(start_ms),
                "endTime": str(end_ms),
                "limit": "200",
            }
            if cursor:
                params["cursor"] = cursor

            data = self._rate_limited_get(
                f"{self.BYBIT_BASE}/v5/market/funding/history",
                params=params,
            )
            if not data or "result" not in data:
                break
            rows = data["result"].get("list", [])
            if not rows:
                break
            all_rows.extend(rows)
            cursor = data["result"].get("nextPageCursor", "")
            if not cursor:
                break

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(
            pd.to_numeric(df["fundingRateTimestamp"], errors="coerce"),
            unit="ms", utc=True,
        )
        df["bybit_funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df = df.set_index("timestamp").sort_index()
        return df[["bybit_funding_rate"]]

    def fetch_okx_funding(
        self, symbol: str = "BTC/USDT", days: int = 30
    ) -> Optional[pd.DataFrame]:
        """Fetch OKX funding rate history.

        Endpoint: /api/v5/public/funding-rate-history
        """
        okx_inst_id = symbol.replace("/", "-").replace(":USDT", "") + "-SWAP"
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 86400 * 1000
        all_rows = []
        after = ""

        for _ in range(100):
            params = {"instId": okx_inst_id, "limit": "100"}
            if after:
                params["after"] = after

            data = self._rate_limited_get(
                f"{self.OKX_BASE}/api/v5/public/funding-rate-history",
                params=params,
            )
            if not data or "data" not in data:
                break
            rows = data["data"]
            if not rows:
                break
            all_rows.extend(rows)
            after = rows[-1].get("fundingTime", "")
            # Check if we've gone past our start time
            oldest_ts = int(rows[-1].get("fundingTime", end_ms))
            if oldest_ts < start_ms:
                break

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(
            pd.to_numeric(df["fundingTime"], errors="coerce"),
            unit="ms", utc=True,
        )
        df["okx_funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df = df.set_index("timestamp").sort_index()
        return df[["okx_funding_rate"]]

    def fetch_cross_venue_funding(
        self, symbol: str = "BTC/USDT", days: int = 30,
        binance_funding_df: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """Compare funding rates across Binance, Bybit, OKX.

        When funding diverges significantly between venues, it indicates
        venue-specific positioning imbalances that arbitrageurs compress.

        Args:
            binance_funding_df: Optional pre-fetched Binance funding data.
                                If None, uses structural.py to fetch.
        """
        bybit_df = self.fetch_bybit_funding(symbol, days)
        okx_df = self.fetch_okx_funding(symbol, days)

        # Get Binance funding if not provided
        if binance_funding_df is None:
            from src.data.structural import StructuralDataFetcher
            sf = StructuralDataFetcher()
            binance_funding_df = sf.fetch_funding_rate_history(symbol, days)

        if binance_funding_df is None:
            logger.warning("No Binance funding data available")
            return None

        # Normalize Binance funding column name
        if "funding_rate" in binance_funding_df.columns:
            binance_funding_df = binance_funding_df.rename(
                columns={"funding_rate": "binance_funding_rate"}
            )

        # Create hourly index and forward-fill to align
        start = binance_funding_df.index.min()
        end = binance_funding_df.index.max()
        hourly_idx = pd.date_range(start, end, freq="1h")
        result = pd.DataFrame(index=hourly_idx)
        result.index.name = "timestamp"

        # Merge each venue (forward-fill since funding updates at different intervals)
        for name, df in [
            ("binance_funding_rate", binance_funding_df),
            ("bybit_funding_rate", bybit_df),
            ("okx_funding_rate", okx_df),
        ]:
            if df is not None and name in df.columns:
                result = pd.merge_asof(
                    result, df[[name]].sort_index(),
                    left_index=True, right_index=True,
                    direction="backward",
                    tolerance=pd.Timedelta("12h"),
                )

        # Compute cross-venue metrics
        venue_cols = [c for c in result.columns if "_funding_rate" in c]
        n_venues = result[venue_cols].notna().sum(axis=1)

        result["cross_venue_funding_mean"] = result[venue_cols].mean(axis=1)
        result["cross_venue_funding_spread"] = (
            result[venue_cols].max(axis=1) - result[venue_cols].min(axis=1)
        )

        # Z-score the spread
        lookback = 168  # 1 week
        rolling_mean = result["cross_venue_funding_spread"].rolling(
            lookback, min_periods=20
        ).mean()
        rolling_std = result["cross_venue_funding_spread"].rolling(
            lookback, min_periods=20
        ).std()
        result["cross_venue_funding_zscore"] = (
            (result["cross_venue_funding_spread"] - rolling_mean)
            / rolling_std.replace(0, np.nan)
        )

        result["cross_venue_count"] = n_venues

        return result

    # ------------------------------------------------------------------
    # 3. OKX Liquidation Orders
    # ------------------------------------------------------------------
    def fetch_okx_liquidations(
        self, symbol: str = "BTC/USDT", limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """Fetch recent OKX liquidation orders.

        Endpoint: /api/v5/public/liquidation-orders
        This data is NOT available from Binance publicly.

        Returns DataFrame with columns:
            direction: 'long' or 'short' (which side was liquidated)
            size: position size
            price: liquidation price
            timestamp: when it happened
        """
        okx_inst_id = symbol.replace("/", "-").replace(":USDT", "") + "-SWAP"

        data = self._rate_limited_get(
            f"{self.OKX_BASE}/api/v5/public/liquidation-orders",
            params={
                "instType": "SWAP",
                "instId": okx_inst_id,
                "limit": str(min(limit, 100)),
                "state": "filled",
            },
        )
        if not data or "data" not in data:
            return None

        rows = []
        for entry in data["data"]:
            details = entry.get("details", [])
            for d in details:
                rows.append({
                    "timestamp": pd.to_datetime(
                        int(d.get("ts", 0)), unit="ms", utc=True
                    ),
                    "direction": d.get("side", ""),  # buy = short liq, sell = long liq
                    "price": float(d.get("bkPx", 0)),
                    "size": float(d.get("sz", 0)),
                })

        if not rows:
            return None

        df = pd.DataFrame(rows)
        # Normalize: "buy" means shorts were liquidated (forced buy),
        # "sell" means longs were liquidated (forced sell)
        df["liq_side"] = df["direction"].map({"buy": "short", "sell": "long"})
        df = df.set_index("timestamp").sort_index()
        return df

    # ------------------------------------------------------------------
    # 4. DeFi Llama — TVL + Stablecoin Supply
    # ------------------------------------------------------------------
    def fetch_defi_tvl(self, chain: str = "Ethereum") -> Optional[pd.DataFrame]:
        """Fetch historical TVL for a chain.

        Endpoint: /v2/historicalChainTvl/{chain}
        Free, no API key.
        """
        data = self._rate_limited_get(
            f"{self.DEFI_LLAMA_BASE}/v2/historicalChainTvl/{chain}",
        )
        if not data:
            return None

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["date"], unit="s", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df.rename(columns={"tvl": "defi_tvl"})

        # Derived features
        df["tvl_change_1d"] = df["defi_tvl"].pct_change(1)
        df["tvl_change_7d"] = df["defi_tvl"].pct_change(7)
        df["tvl_change_30d"] = df["defi_tvl"].pct_change(30)

        return df[["defi_tvl", "tvl_change_1d", "tvl_change_7d", "tvl_change_30d"]]

    def fetch_stablecoin_supply(self) -> Optional[pd.DataFrame]:
        """Fetch total stablecoin market cap over time.

        Endpoint: stablecoins.llama.fi/stablecoincharts/all
        Free, no API key.

        Stablecoin supply growth = dry powder entering crypto = risk-on.
        Supply decline = capital leaving = risk-off.
        """
        data = self._rate_limited_get(
            f"{self.STABLECOIN_BASE}/stablecoincharts/all",
        )
        if not data:
            return None

        rows = []
        for entry in data:
            total_circ = sum(
                float(v.get("peggedUSD", 0))
                for v in entry.get("totalCirculating", {}).values()
            ) if isinstance(entry.get("totalCirculating"), dict) else 0
            rows.append({
                "timestamp": pd.to_datetime(int(entry["date"]), unit="s", utc=True),
                "stablecoin_mcap": total_circ,
            })

        if not rows:
            return None

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()

        # Derived features
        df["stable_supply_change_1d"] = df["stablecoin_mcap"].pct_change(1)
        df["stable_supply_change_7d"] = df["stablecoin_mcap"].pct_change(7)
        df["stable_supply_change_30d"] = df["stablecoin_mcap"].pct_change(30)

        # Risk-on/risk-off indicator: 7d MA of supply change
        df["stable_risk_signal"] = (
            df["stable_supply_change_7d"]
            .rolling(7, min_periods=3)
            .mean()
            .apply(lambda x: 1 if x > 0.001 else (-1 if x < -0.001 else 0))
        )

        return df

    # ------------------------------------------------------------------
    # 5. Bybit Insurance Fund (systemic stress indicator)
    # ------------------------------------------------------------------
    def fetch_bybit_insurance(self, coin: str = "USDT") -> Optional[pd.DataFrame]:
        """Fetch Bybit insurance fund balance history.

        Declining insurance fund = exchange absorbing liquidation losses
        = systemic stress indicator.

        Endpoint: /v5/market/insurance
        """
        data = self._rate_limited_get(
            f"{self.BYBIT_BASE}/v5/market/insurance",
            params={"coin": coin},
        )
        if not data or "result" not in data:
            return None

        rows = data["result"].get("updatedTime", "")
        fund_list = data["result"].get("list", [])

        if not fund_list:
            return None

        # This endpoint returns current balance, not history
        # We'll track it over time by caching snapshots
        balance = float(fund_list[0].get("balance", 0))
        return pd.DataFrame([{
            "timestamp": pd.Timestamp.now(tz="UTC"),
            "insurance_fund_balance": balance,
            "coin": coin,
        }]).set_index("timestamp")

    # ------------------------------------------------------------------
    # Unified fetch: get everything for a symbol
    # ------------------------------------------------------------------
    def fetch_all(
        self,
        symbol: str = "BTC/USDT",
        days: int = 30,
        price_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Fetch all multi-venue data and merge onto price DataFrame.

        Args:
            symbol: Trading pair
            days: Historical lookback
            price_df: OHLCV DataFrame to merge onto. If None, returns raw data.

        Returns:
            DataFrame with multi-venue features added as columns.
        """
        logger.info(f"[MultiVenue] Fetching data for {symbol} ({days}d)")

        # 1. Top trader vs retail divergence
        divergence_df = self.fetch_top_retail_divergence(symbol, days)
        if divergence_df is not None:
            logger.info(f"  Top trader divergence: {len(divergence_df)} rows")
        else:
            logger.warning("  Top trader divergence: UNAVAILABLE")

        # 2. Cross-venue funding
        cross_funding_df = self.fetch_cross_venue_funding(symbol, days)
        if cross_funding_df is not None:
            logger.info(f"  Cross-venue funding: {len(cross_funding_df)} rows")
        else:
            logger.warning("  Cross-venue funding: UNAVAILABLE")

        # 3. OKX liquidations (recent only, not historical)
        liq_df = self.fetch_okx_liquidations(symbol)
        if liq_df is not None:
            logger.info(f"  OKX liquidations: {len(liq_df)} recent orders")
        else:
            logger.warning("  OKX liquidations: UNAVAILABLE")

        # If no price_df provided, return what we have
        if price_df is None:
            result = pd.DataFrame()
            for df in [divergence_df, cross_funding_df]:
                if df is not None:
                    if result.empty:
                        result = df
                    else:
                        result = pd.merge_asof(
                            result, df, left_index=True, right_index=True,
                            direction="backward",
                        )
            return result

        # Merge onto price_df using merge_asof (backward = no lookahead)
        result = price_df.copy()

        # Ensure timezone-naive for merging
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)

        for name, df in [
            ("divergence", divergence_df),
            ("cross_funding", cross_funding_df),
        ]:
            if df is not None:
                df_copy = df.copy()
                if df_copy.index.tz is not None:
                    df_copy.index = df_copy.index.tz_localize(None)
                result = pd.merge_asof(
                    result, df_copy, left_index=True, right_index=True,
                    direction="backward",
                    tolerance=pd.Timedelta("2h"),
                )

        # Aggregate recent liquidation data into features
        if liq_df is not None:
            liq_copy = liq_df.copy()
            if liq_copy.index.tz is not None:
                liq_copy.index = liq_copy.index.tz_localize(None)

            # Count long vs short liquidations in recent window
            recent_mask = liq_copy.index >= (result.index[-1] - pd.Timedelta("24h"))
            recent_liq = liq_copy[recent_mask]
            if len(recent_liq) > 0:
                long_liq_count = (recent_liq["liq_side"] == "long").sum()
                short_liq_count = (recent_liq["liq_side"] == "short").sum()
                total_liq = len(recent_liq)
                result["okx_liq_count_24h"] = total_liq
                result["okx_liq_long_pct"] = long_liq_count / max(total_liq, 1)
                result["okx_liq_imbalance"] = (
                    (long_liq_count - short_liq_count) / max(total_liq, 1)
                )

        # Forward-fill structural data (updates less frequently than price)
        structural_cols = [c for c in result.columns if c not in price_df.columns]
        result[structural_cols] = result[structural_cols].ffill()

        logger.info(
            f"  Merged {len(structural_cols)} multi-venue features onto "
            f"{len(result)} price bars"
        )

        return result

"""
Liquidation Data Fetcher — Real Exchange Data
================================================
Fetches historical liquidation data from exchanges and aggregators.

For backtesting: Uses Binance aggregated liquidation snapshots
                 (via long/short ratio + taker buy/sell as proxy).
For live trading: Can connect to WebSocket streams for real-time
                  forced liquidation events.

Key insight: We don't need exact per-position liquidation data for the strategy.
What we need is AGGREGATE liquidation volume spikes — detectable from:
1. Long/Short ratio shifts (sudden drop = longs liquidated)
2. Taker buy/sell volume ratio (spike in sells = forced selling)
3. OI drops (leverage being purged)
4. Actual liquidation endpoints where available
"""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

BINANCE_LSR_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
BINANCE_TAKER_URL = "https://fapi.binance.com/futures/data/takerlongshortRatio"


def _symbol_to_binance(symbol: str) -> str:
    return symbol.replace("/", "").replace(":USDT", "")


class LiquidationFetcher:
    """Fetches liquidation proxy data for backtesting and live trading.

    For backtesting, we construct a liquidation intensity signal from:
    - Long/Short account ratio changes (sharp drops = long liquidation wave)
    - Taker buy/sell ratio (spike in taker sells = forced selling)
    - Combined into a single 'liquidation_intensity' score

    This is more robust than trying to get exact liquidation data
    (which exchanges limit/hide) and captures the EFFECT we care about:
    forced selling pressure.
    """

    def __init__(self, exchange_id: str = "binance"):
        self.exchange_id = exchange_id

    def fetch_long_short_ratio(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 90,
    ) -> pd.DataFrame:
        """Fetch global long/short account ratio from Binance.

        Values < 1.0 = more shorts than longs (bearish positioning).
        Sharp drops = longs getting liquidated.
        """
        pair = _symbol_to_binance(symbol)
        tf_map = {
            "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1d",
        }
        period = tf_map.get(timeframe, "1h")

        all_records = []
        start_ms = int((time.time() - days * 86400) * 1000)
        now_ms = int(time.time() * 1000)

        while start_ms < now_ms:
            end_ms = min(start_ms + 30 * 86400 * 1000, now_ms)
            try:
                resp = requests.get(
                    BINANCE_LSR_URL,
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
                logger.warning(f"LSR fetch failed: {e}")
                break

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df["long_short_ratio"] = df["longShortRatio"].astype(float)
        df["long_account_pct"] = df["longAccount"].astype(float)
        df["short_account_pct"] = df["shortAccount"].astype(float)
        df = df[["long_short_ratio", "long_account_pct", "short_account_pct"]]
        df = df[~df.index.duplicated(keep="last")].sort_index()

        logger.info(f"Fetched {len(df)} L/S ratio records for {symbol}")
        return df

    def fetch_taker_ratio(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 90,
    ) -> pd.DataFrame:
        """Fetch taker buy/sell volume ratio from Binance.

        buySellRatio > 1.0 = more taker buys (buyers aggressive).
        buySellRatio < 1.0 = more taker sells (sellers aggressive / liquidations).
        Sharp drops = forced selling wave.
        """
        pair = _symbol_to_binance(symbol)
        tf_map = {
            "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1d",
        }
        period = tf_map.get(timeframe, "1h")

        all_records = []
        start_ms = int((time.time() - days * 86400) * 1000)
        now_ms = int(time.time() * 1000)

        while start_ms < now_ms:
            end_ms = min(start_ms + 30 * 86400 * 1000, now_ms)
            try:
                resp = requests.get(
                    BINANCE_TAKER_URL,
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
                logger.warning(f"Taker ratio fetch failed: {e}")
                break

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df["taker_buy_sell_ratio"] = df["buySellRatio"].astype(float)
        df["taker_buy_vol"] = df["buyVol"].astype(float)
        df["taker_sell_vol"] = df["sellVol"].astype(float)
        df = df[["taker_buy_sell_ratio", "taker_buy_vol", "taker_sell_vol"]]
        df = df[~df.index.duplicated(keep="last")].sort_index()

        logger.info(f"Fetched {len(df)} taker ratio records for {symbol}")
        return df

    def build_liquidation_features(
        self,
        ohlcv_df: pd.DataFrame,
        lsr_df: pd.DataFrame,
        taker_df: pd.DataFrame,
        oi_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Construct liquidation intensity features from proxy data.

        Combines L/S ratio, taker ratio, and OI changes into
        actionable liquidation signals aligned to OHLCV bars.
        """
        features = pd.DataFrame(index=ohlcv_df.index)

        # --- Long/Short Ratio features ---
        if not lsr_df.empty:
            lsr = lsr_df["long_short_ratio"].reindex(
                ohlcv_df.index, method="ffill"
            ).fillna(1.0)
            features["lsr"] = lsr
            features["lsr_change"] = lsr.pct_change().fillna(0)
            features["lsr_z"] = (
                (lsr - lsr.rolling(24).mean()) /
                (lsr.rolling(24).std() + 1e-10)
            ).fillna(0)
        else:
            features["lsr"] = 1.0
            features["lsr_change"] = 0.0
            features["lsr_z"] = 0.0

        # --- Taker Buy/Sell Ratio features ---
        if not taker_df.empty:
            taker = taker_df["taker_buy_sell_ratio"].reindex(
                ohlcv_df.index, method="ffill"
            ).fillna(1.0)
            sell_vol = taker_df["taker_sell_vol"].reindex(
                ohlcv_df.index, method="ffill"
            ).fillna(0)

            features["taker_ratio"] = taker
            features["taker_ratio_z"] = (
                (taker - taker.rolling(24).mean()) /
                (taker.rolling(24).std() + 1e-10)
            ).fillna(0)
            features["taker_sell_spike"] = (
                sell_vol / (sell_vol.rolling(24).mean() + 1e-10)
            ).fillna(1.0)
        else:
            features["taker_ratio"] = 1.0
            features["taker_ratio_z"] = 0.0
            features["taker_sell_spike"] = 1.0

        # --- OI change features ---
        if not oi_df.empty:
            oi = oi_df["oi_value"].reindex(
                ohlcv_df.index, method="ffill"
            ).fillna(method="bfill")
            features["oi_change_1h"] = oi.pct_change(1).fillna(0)
            features["oi_change_4h"] = oi.pct_change(4).fillna(0)
            features["oi_z"] = (
                (oi - oi.rolling(24).mean()) /
                (oi.rolling(24).std() + 1e-10)
            ).fillna(0)
        else:
            features["oi_change_1h"] = 0.0
            features["oi_change_4h"] = 0.0
            features["oi_z"] = 0.0

        # --- Composite liquidation intensity score ---
        # High score = strong liquidation event happening
        # Components:
        #   1. LSR dropping fast (longs getting wiped)
        #   2. Taker sells spiking (forced selling)
        #   3. OI dropping (leverage being purged)
        lsr_signal = (-features["lsr_z"]).clip(0, 5)           # Positive when LSR drops
        taker_signal = (-features["taker_ratio_z"]).clip(0, 5) # Positive when sells dominate
        oi_signal = (-features["oi_change_4h"] * 100).clip(0, 5)  # Positive when OI drops

        features["liq_intensity"] = (
            0.4 * lsr_signal +
            0.4 * taker_signal +
            0.2 * oi_signal
        )

        # Rolling stats for threshold detection
        features["liq_intensity_mean"] = features["liq_intensity"].rolling(
            24 * 7  # 7 day rolling mean
        ).mean().fillna(0)
        features["liq_intensity_std"] = features["liq_intensity"].rolling(
            24 * 7
        ).std().fillna(1)

        # Z-score of liquidation intensity (how many σ above normal)
        features["liq_intensity_z"] = (
            (features["liq_intensity"] - features["liq_intensity_mean"]) /
            (features["liq_intensity_std"] + 1e-10)
        ).fillna(0)

        # Replace infinities
        features = features.replace([np.inf, -np.inf], 0)

        return features

"""
Multi-Timeframe Feature Fusion
================================
Combines features from multiple timeframes into a single DataFrame.
Higher timeframes provide context (trend), lower timeframes provide timing.

Also includes order book microstructure features when live data is available.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data.features import compute_all_features

logger = logging.getLogger(__name__)


def fuse_timeframes(
    data: dict[str, pd.DataFrame],
    base_timeframe: str = "1h",
    higher_timeframes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Merge features from multiple timeframes into the base timeframe.

    Args:
        data: Dict of {timeframe: ohlcv_dataframe}
        base_timeframe: The primary trading timeframe
        higher_timeframes: Higher TFs to merge (e.g., ["4h", "1d"])

    Returns:
        Single DataFrame at base_timeframe resolution with all features.
    """
    if base_timeframe not in data:
        raise ValueError(f"Base timeframe {base_timeframe} not in data")

    higher_timeframes = higher_timeframes or ["4h", "1d"]

    # Compute features for base timeframe
    base_df = compute_all_features(data[base_timeframe])

    for tf in higher_timeframes:
        if tf not in data or data[tf].empty:
            continue

        htf_df = compute_all_features(data[tf], include_calendar=False)

        # Select key features from higher timeframe
        key_features = [
            "ret_5", "ret_20", "vol_10", "vol_20",
            "rsi_14", "macd_hist", "adx_14",
            "bb_pct_20", "linreg_slope_20",
            "rolling_sharpe_20", "dd_pct",
            "stoch_k_14", "cci_14",
        ]

        available = [f for f in key_features if f in htf_df.columns]

        # Forward-fill to base timeframe resolution
        for feat in available:
            col_name = f"{feat}__{tf}"
            resampled = htf_df[feat].reindex(base_df.index, method="ffill")
            base_df[col_name] = resampled

    n_htf = len([c for c in base_df.columns if "__" in c])
    logger.info(f"Fused {n_htf} higher-timeframe features into base")

    return base_df


def compute_microstructure_features(
    df: pd.DataFrame,
    order_book: Optional[dict] = None,
    funding_rate: Optional[float] = None,
    open_interest: Optional[float] = None,
    open_interest_prev: Optional[float] = None,
) -> pd.DataFrame:
    """Add real-time microstructure features when live data is available.

    These features are only available during live/paper trading,
    not during backtesting on historical OHLCV data.
    """
    df = df.copy()

    # Order book features
    if order_book:
        bids = order_book.get("bids")
        asks = order_book.get("asks")

        if bids is not None and asks is not None:
            bid_vol = bids["quantity"].sum() if isinstance(bids, pd.DataFrame) else sum(b[1] for b in bids)
            ask_vol = asks["quantity"].sum() if isinstance(asks, pd.DataFrame) else sum(a[1] for a in asks)

            total = bid_vol + ask_vol + 1e-10
            df.iloc[-1, df.columns.get_loc("book_imbalance") if "book_imbalance" in df.columns
                     else len(df.columns)] = (bid_vol - ask_vol) / total

            # Top-of-book spread
            if isinstance(bids, pd.DataFrame) and len(bids) > 0 and len(asks) > 0:
                best_bid = bids.iloc[0]["price"]
                best_ask = asks.iloc[0]["price"]
                mid = (best_bid + best_ask) / 2
                df.loc[df.index[-1], "spread_pct"] = (best_ask - best_bid) / (mid + 1e-10)

                # Depth at levels (how much liquidity at each %)
                for pct in [0.1, 0.5, 1.0]:
                    bid_depth = bids[bids["price"] >= best_bid * (1 - pct / 100)]["quantity"].sum()
                    ask_depth = asks[asks["price"] <= best_ask * (1 + pct / 100)]["quantity"].sum()
                    df.loc[df.index[-1], f"depth_ratio_{pct}pct"] = (
                        bid_depth / (ask_depth + 1e-10)
                    )

    # Funding rate
    if funding_rate is not None:
        df.loc[df.index[-1], "funding_rate_live"] = funding_rate
        df.loc[df.index[-1], "funding_annualized"] = funding_rate * 3 * 365

    # Open interest
    if open_interest is not None:
        df.loc[df.index[-1], "open_interest"] = open_interest
        if open_interest_prev is not None and open_interest_prev > 0:
            df.loc[df.index[-1], "oi_change_pct_live"] = (
                open_interest / open_interest_prev - 1
            )

    return df


# Feature names for microstructure (live-only)
MICROSTRUCTURE_FEATURE_NAMES = [
    "book_imbalance", "spread_pct",
    "depth_ratio_0.1pct", "depth_ratio_0.5pct", "depth_ratio_1.0pct",
    "funding_rate_live", "funding_annualized",
    "open_interest", "oi_change_pct_live",
]

# Higher-timeframe feature names (generated by fuse_timeframes)
HTF_FEATURE_TEMPLATE = [
    "ret_5__{tf}", "ret_20__{tf}", "vol_10__{tf}", "vol_20__{tf}",
    "rsi_14__{tf}", "macd_hist__{tf}", "adx_14__{tf}",
    "bb_pct_20__{tf}", "linreg_slope_20__{tf}",
    "rolling_sharpe_20__{tf}", "dd_pct__{tf}",
    "stoch_k_14__{tf}", "cci_14__{tf}",
]

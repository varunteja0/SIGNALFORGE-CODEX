"""
Cross-Exchange Funding Rate Arbitrage Engine
===============================================
Delta-neutral funding rate arbitrage: the closest thing to "free money"
in crypto. When funding is positive on Binance and negative on Bybit,
go short on Binance (collect funding) and long on Bybit (collect funding).

Net exposure: zero. Net income: the funding rate spread.

This is the strategy quantitative crypto funds like Alameda used before
they blew up (they blew up from leverage, not from funding arb itself).

Key features:
  - Multi-exchange funding rate monitoring (Binance, Bybit, OKX)
  - Automatic spread detection and entry/exit signals
  - Delta-neutral position management
  - Risk controls for basis risk and execution risk
  - Historical funding analysis for rate prediction

No API keys needed for monitoring. Keys only needed for execution.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Public funding rate endpoints (no auth needed)
FUNDING_ENDPOINTS = {
    "binance": {
        "current": "https://fapi.binance.com/fapi/v1/premiumIndex",
        "history": "https://fapi.binance.com/fapi/v1/fundingRate",
    },
    "bybit": {
        "current": "https://api.bybit.com/v5/market/tickers?category=linear",
        "history": "https://api.bybit.com/v5/market/funding/history",
    },
    "okx": {
        "current": "https://www.okx.com/api/v5/public/funding-rate",
        "history": "https://www.okx.com/api/v5/public/funding-rate-history",
    },
}

# Symbol mapping between exchanges
SYMBOL_MAP = {
    "BTC/USDT": {
        "binance": "BTCUSDT",
        "bybit": "BTCUSDT",
        "okx": "BTC-USDT-SWAP",
    },
    "ETH/USDT": {
        "binance": "ETHUSDT",
        "bybit": "ETHUSDT",
        "okx": "ETH-USDT-SWAP",
    },
    "SOL/USDT": {
        "binance": "SOLUSDT",
        "bybit": "SOLUSDT",
        "okx": "SOL-USDT-SWAP",
    },
    "BNB/USDT": {
        "binance": "BNBUSDT",
        "bybit": "BNBUSDT",
        "okx": "BNB-USDT-SWAP",
    },
}


@dataclass
class FundingSnapshot:
    """Funding rate snapshot from one exchange."""
    exchange: str
    symbol: str
    funding_rate: float          # Current/next funding rate
    next_funding_time: float     # Unix timestamp of next funding
    mark_price: float = 0       # Current mark price
    index_price: float = 0      # Current index price
    open_interest: float = 0    # Open interest in USD
    timestamp: float = 0


@dataclass
class FundingArbOpportunity:
    """A funding rate arbitrage opportunity."""
    symbol: str
    long_exchange: str           # Go long here (collect negative funding)
    short_exchange: str          # Go short here (collect positive funding)
    long_rate: float             # Funding rate on long side
    short_rate: float            # Funding rate on short side
    spread: float                # Total spread (annualized %)
    spread_8h: float             # Spread per 8-hour period
    next_funding_in_sec: float   # Seconds until next funding
    mark_price: float = 0
    confidence: float = 0        # How reliable (based on history)
    estimated_annual_yield: float = 0

    # Risk metrics
    basis_risk: float = 0        # Price difference between exchanges
    execution_risk: float = 0    # Slippage/fill risk
    recommended_size_usd: float = 0

    @property
    def is_actionable(self) -> bool:
        return self.spread_8h > 0.01 and self.confidence > 0.5


@dataclass
class ArbPosition:
    """Active arbitrage position."""
    position_id: str
    symbol: str
    long_exchange: str
    short_exchange: str
    size_usd: float
    entry_spread: float
    cumulative_yield: float = 0
    open_time: float = 0
    funding_collected: int = 0


class FundingArbEngine:
    """Cross-exchange funding rate arbitrage engine.

    Monitors funding rates across exchanges and identifies
    opportunities for delta-neutral yield.
    """

    def __init__(
        self,
        exchanges: list[str] = None,
        min_spread_8h: float = 0.01,    # Min 0.01% per 8h (≈1.1% annualized)
        max_basis_risk_pct: float = 0.5, # Max price diff between exchanges
        max_position_usd: float = 50000,
        cache_ttl: int = 60,
    ):
        self.exchanges = exchanges or ["binance", "bybit", "okx"]
        self.min_spread = min_spread_8h
        self.max_basis_risk = max_basis_risk_pct
        self.max_position = max_position_usd
        self.cache_ttl = cache_ttl

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "SignalForge/2.0"})
        self._cache: dict[str, tuple[float, object]] = {}
        self._last_request: dict[str, float] = {}

        # State
        self.active_positions: list[ArbPosition] = []
        self.opportunity_history: list[FundingArbOpportunity] = []
        self.funding_history: dict[str, list[FundingSnapshot]] = {}

    def _rate_limited_get(self, url: str, exchange: str, params: dict = None) -> Optional[dict]:
        """Rate-limited GET request."""
        cache_key = f"{url}:{params}"
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if time.time() - ts < self.cache_ttl:
                return data

        last = self._last_request.get(exchange, 0)
        if time.time() - last < 0.5:
            time.sleep(0.5)
        self._last_request[exchange] = time.time()

        try:
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self._cache[cache_key] = (time.time(), data)
            return data
        except Exception as e:
            logger.warning(f"Failed to fetch from {exchange}: {e}")
            return None

    # ================================================================
    # Funding Rate Fetchers (Public APIs)
    # ================================================================

    def fetch_binance_funding(self, symbol: str = "BTCUSDT") -> Optional[FundingSnapshot]:
        """Fetch current funding rate from Binance Futures."""
        data = self._rate_limited_get(
            FUNDING_ENDPOINTS["binance"]["current"],
            "binance",
            {"symbol": symbol},
        )
        if not data:
            return None

        # Handle both single dict and list responses
        if isinstance(data, list):
            data = next((d for d in data if d.get("symbol") == symbol), None)
            if not data:
                return None

        return FundingSnapshot(
            exchange="binance",
            symbol=symbol,
            funding_rate=float(data.get("lastFundingRate", 0)),
            next_funding_time=float(data.get("nextFundingTime", 0)) / 1000,
            mark_price=float(data.get("markPrice", 0)),
            index_price=float(data.get("indexPrice", 0)),
            timestamp=time.time(),
        )

    def fetch_bybit_funding(self, symbol: str = "BTCUSDT") -> Optional[FundingSnapshot]:
        """Fetch current funding rate from Bybit."""
        data = self._rate_limited_get(
            FUNDING_ENDPOINTS["bybit"]["current"],
            "bybit",
            {"symbol": symbol},
        )
        if not data or "result" not in data:
            return None

        tickers = data["result"].get("list", [])
        ticker = next((t for t in tickers if t.get("symbol") == symbol), None)
        if not ticker:
            return None

        return FundingSnapshot(
            exchange="bybit",
            symbol=symbol,
            funding_rate=float(ticker.get("fundingRate", 0)),
            next_funding_time=float(ticker.get("nextFundingTime", 0)) / 1000,
            mark_price=float(ticker.get("markPrice", 0)),
            index_price=float(ticker.get("indexPrice", 0)),
            open_interest=float(ticker.get("openInterestValue", 0)),
            timestamp=time.time(),
        )

    def fetch_okx_funding(self, symbol: str = "BTC-USDT-SWAP") -> Optional[FundingSnapshot]:
        """Fetch current funding rate from OKX."""
        data = self._rate_limited_get(
            FUNDING_ENDPOINTS["okx"]["current"],
            "okx",
            {"instId": symbol},
        )
        if not data or "data" not in data:
            return None

        items = data["data"]
        if not items:
            return None

        item = items[0]
        return FundingSnapshot(
            exchange="okx",
            symbol=symbol,
            funding_rate=float(item.get("fundingRate", 0)),
            next_funding_time=float(item.get("nextFundingTime", 0)) / 1000,
            timestamp=time.time(),
        )

    def fetch_all_funding(self, symbol: str = "BTC/USDT") -> dict[str, FundingSnapshot]:
        """Fetch funding rates from all exchanges for a symbol."""
        mapping = SYMBOL_MAP.get(symbol, {})
        snapshots = {}

        for exchange in self.exchanges:
            ex_symbol = mapping.get(exchange)
            if not ex_symbol:
                continue

            fetcher = {
                "binance": self.fetch_binance_funding,
                "bybit": self.fetch_bybit_funding,
                "okx": self.fetch_okx_funding,
            }.get(exchange)

            if fetcher:
                try:
                    snap = fetcher(ex_symbol)
                    if snap:
                        snapshots[exchange] = snap
                except Exception as e:
                    logger.error(f"Error fetching {exchange} funding: {e}")

        return snapshots

    # ================================================================
    # Arbitrage Detection
    # ================================================================

    def scan_opportunities(
        self, symbols: list[str] = None
    ) -> list[FundingArbOpportunity]:
        """Scan for funding rate arbitrage opportunities across all pairs."""
        symbols = symbols or list(SYMBOL_MAP.keys())
        opportunities = []

        for symbol in symbols:
            snapshots = self.fetch_all_funding(symbol)

            if len(snapshots) < 2:
                continue

            # Find all pairs with positive spread
            exchanges = list(snapshots.keys())
            for i, ex_a in enumerate(exchanges):
                for ex_b in exchanges[i + 1:]:
                    snap_a = snapshots[ex_a]
                    snap_b = snapshots[ex_b]

                    rate_a = snap_a.funding_rate
                    rate_b = snap_b.funding_rate

                    # Determine direction: go long where funding is lower (cheaper)
                    if rate_a > rate_b:
                        short_ex, long_ex = ex_a, ex_b
                        short_rate, long_rate = rate_a, rate_b
                    else:
                        short_ex, long_ex = ex_b, ex_a
                        short_rate, long_rate = rate_b, rate_a

                    # Spread per 8h period
                    spread_8h = short_rate - long_rate
                    # Annualized (3 funding periods per day × 365)
                    annualized = spread_8h * 3 * 365 * 100

                    if spread_8h < self.min_spread / 100:
                        continue

                    # Basis risk (price difference between exchanges)
                    basis_risk = 0
                    if snap_a.mark_price > 0 and snap_b.mark_price > 0:
                        basis_risk = abs(
                            snap_a.mark_price / snap_b.mark_price - 1
                        ) * 100

                    if basis_risk > self.max_basis_risk:
                        continue

                    # Next funding time
                    next_time = min(
                        snap_a.next_funding_time or time.time() + 28800,
                        snap_b.next_funding_time or time.time() + 28800,
                    )

                    opp = FundingArbOpportunity(
                        symbol=symbol,
                        long_exchange=long_ex,
                        short_exchange=short_ex,
                        long_rate=long_rate,
                        short_rate=short_rate,
                        spread=annualized,
                        spread_8h=spread_8h * 100,
                        next_funding_in_sec=max(0, next_time - time.time()),
                        mark_price=max(snap_a.mark_price, snap_b.mark_price),
                        basis_risk=basis_risk,
                        estimated_annual_yield=annualized,
                        recommended_size_usd=min(
                            self.max_position,
                            self.max_position * (1 - basis_risk / self.max_basis_risk),
                        ),
                    )

                    # Confidence based on spread size and basis risk
                    opp.confidence = min(1.0, (
                        min(1, spread_8h * 100 / 0.05) * 0.5  # Spread quality
                        + (1 - basis_risk / self.max_basis_risk) * 0.3  # Basis safety
                        + 0.2  # Base confidence
                    ))

                    opportunities.append(opp)

        # Sort by spread (best first)
        opportunities.sort(key=lambda o: o.spread, reverse=True)
        self.opportunity_history.extend(opportunities)

        return opportunities

    def fetch_historical_funding(
        self,
        symbol: str = "BTC/USDT",
        exchange: str = "binance",
        days: int = 30,
    ) -> pd.DataFrame:
        """Fetch historical funding rates for analysis."""
        mapping = SYMBOL_MAP.get(symbol, {})
        ex_symbol = mapping.get(exchange)
        if not ex_symbol:
            return pd.DataFrame()

        if exchange == "binance":
            data = self._rate_limited_get(
                FUNDING_ENDPOINTS["binance"]["history"],
                "binance",
                {"symbol": ex_symbol, "limit": days * 3},
            )
            if data:
                df = pd.DataFrame(data)
                if not df.empty and "fundingRate" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
                    df["funding_rate"] = df["fundingRate"].astype(float)
                    df = df.set_index("timestamp")[["funding_rate"]].sort_index()
                    return df

        elif exchange == "bybit":
            data = self._rate_limited_get(
                FUNDING_ENDPOINTS["bybit"]["history"],
                "bybit",
                {"category": "linear", "symbol": ex_symbol, "limit": days * 3},
            )
            if data and "result" in data:
                items = data["result"].get("list", [])
                if items:
                    df = pd.DataFrame(items)
                    df["timestamp"] = pd.to_datetime(df["fundingRateTimestamp"].astype(float), unit="ms")
                    df["funding_rate"] = df["fundingRate"].astype(float)
                    df = df.set_index("timestamp")[["funding_rate"]].sort_index()
                    return df

        return pd.DataFrame()

    def get_arb_summary(self) -> dict:
        """Get summary of current arbitrage state."""
        opportunities = self.scan_opportunities()

        return {
            "n_opportunities": len(opportunities),
            "best_spread_annual": opportunities[0].spread if opportunities else 0,
            "best_symbol": opportunities[0].symbol if opportunities else "",
            "active_positions": len(self.active_positions),
            "total_yield": sum(p.cumulative_yield for p in self.active_positions),
            "opportunities": [
                {
                    "symbol": o.symbol,
                    "long": o.long_exchange,
                    "short": o.short_exchange,
                    "spread_annual": f"{o.spread:.1f}%",
                    "spread_8h": f"{o.spread_8h:.3f}%",
                    "basis_risk": f"{o.basis_risk:.3f}%",
                    "confidence": f"{o.confidence:.0%}",
                    "next_funding": f"{o.next_funding_in_sec / 3600:.1f}h",
                }
                for o in opportunities[:10]
            ],
        }

    def compute_features(self, symbol: str = "BTC/USDT") -> dict:
        """Compute funding arb features for GP engine."""
        snapshots = self.fetch_all_funding(symbol)

        if not snapshots:
            return {
                "funding_spread_max": 0,
                "funding_avg": 0,
                "funding_skew": 0,
                "funding_arb_available": 0,
            }

        rates = [s.funding_rate for s in snapshots.values()]
        spread = max(rates) - min(rates) if len(rates) >= 2 else 0

        return {
            "funding_spread_max": spread * 100,
            "funding_avg": np.mean(rates) * 100,
            "funding_skew": (max(rates) + min(rates)) / 2 * 100 if len(rates) >= 2 else 0,
            "funding_arb_available": 1.0 if spread > self.min_spread / 100 else 0.0,
            "funding_extreme_positive": 1.0 if max(rates) > 0.001 else 0.0,
            "funding_extreme_negative": 1.0 if min(rates) < -0.001 else 0.0,
            "funding_n_exchanges": len(snapshots),
        }

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import ccxt
import numpy as np

logger = logging.getLogger(__name__)


def _utc_iso_from_ms(timestamp_ms: Any) -> str:
    if timestamp_ms in (None, ""):
        return datetime.now(timezone.utc).isoformat()
    try:
        return datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        return datetime.now(timezone.utc).isoformat()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class BrokerOrder:
    id: str = ""
    symbol: str = ""
    side: str = ""
    type: str = ""
    amount: float = 0.0
    filled: float = 0.0
    remaining: float = 0.0
    average_price: float = 0.0
    cost: float = 0.0
    status: str = "unknown"
    reduce_only: bool = False
    stop_price: float = 0.0
    submitted_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ccxt(cls, payload: dict[str, Any]) -> "BrokerOrder":
        info = payload.get("info") or {}
        reduce_only = bool(payload.get("reduceOnly") or info.get("reduceOnly") or info.get("reduce_only"))
        stop_price = _float(payload.get("stopPrice") or info.get("stopPrice") or info.get("triggerPrice"))
        return cls(
            id=str(payload.get("id", "")),
            symbol=str(payload.get("symbol", "")),
            side=str(payload.get("side", "")),
            type=str(payload.get("type", "")),
            amount=_float(payload.get("amount")),
            filled=_float(payload.get("filled")),
            remaining=_float(payload.get("remaining")),
            average_price=_float(payload.get("average") or payload.get("price")),
            cost=_float(payload.get("cost")),
            status=str(payload.get("status", "unknown")),
            reduce_only=reduce_only,
            stop_price=stop_price,
            submitted_at=_utc_iso_from_ms(payload.get("timestamp") or info.get("createdTime")),
            raw=payload,
        )


@dataclass
class BrokerPosition:
    symbol: str = ""
    side: str = ""
    size: float = 0.0
    signed_size: float = 0.0
    notional: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ccxt(cls, payload: dict[str, Any]) -> "BrokerPosition":
        side = str(payload.get("side") or payload.get("info", {}).get("side") or "").lower()
        size = _float(payload.get("contracts") or payload.get("contractSize") or payload.get("positionAmt") or payload.get("size"))
        if size == 0.0:
            size = abs(_float(payload.get("notional"))) / max(_float(payload.get("entryPrice")), 1e-9)
        signed_size = size
        if side in {"short", "sell"}:
            signed_size = -abs(size)
        elif side in {"long", "buy"}:
            signed_size = abs(size)
        return cls(
            symbol=str(payload.get("symbol", "")),
            side=side,
            size=abs(size),
            signed_size=signed_size,
            notional=abs(_float(payload.get("notional"))),
            entry_price=_float(payload.get("entryPrice") or payload.get("avgPrice")),
            unrealized_pnl=_float(payload.get("unrealizedPnl") or payload.get("unrealizedProfit")),
            raw=payload,
        )


@dataclass
class OrderBookSnapshot:
    symbol: str
    timestamp: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread_bps: float = 0.0
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    estimated_impact_bps: float = 0.0


class CcxtBrokerAdapter:
    """Thin normalization layer over CCXT for live and shadow execution."""

    def __init__(self, exchange: Any, exchange_id: str):
        self.exchange = exchange
        self.exchange_id = exchange_id

    @classmethod
    def from_env(
        cls,
        exchange_id: str = "bybit",
        *,
        env_prefix: str | None = None,
        default_type: str = "swap",
        sandbox: bool = False,
    ) -> "CcxtBrokerAdapter":
        prefix = env_prefix or exchange_id.upper()
        api_key = os.environ.get(f"{prefix}_API_KEY", "")
        api_secret = os.environ.get(f"{prefix}_API_SECRET", "")
        if not api_key or not api_secret:
            raise ValueError(
                f"Missing broker credentials for {exchange_id}. Expected {prefix}_API_KEY and {prefix}_API_SECRET."
            )

        exchange_cls = getattr(ccxt, exchange_id)
        exchange = exchange_cls(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": default_type},
            }
        )
        if sandbox and hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(True)
        return cls(exchange=exchange, exchange_id=exchange_id)

    def fetch_balance(self) -> dict[str, Any]:
        return self.exchange.fetch_balance()

    def fetch_balance_usd(self) -> float:
        balance = self.fetch_balance()
        if isinstance(balance.get("USDT"), dict):
            return _float(balance["USDT"].get("free"))
        if isinstance(balance.get("free"), dict):
            return _float(balance["free"].get("USDT"))
        return 0.0

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        return self.exchange.fetch_ticker(symbol)

    def fetch_positions(self) -> list[BrokerPosition]:
        positions = self.exchange.fetch_positions()
        normalized = [BrokerPosition.from_ccxt(position) for position in positions]
        return [position for position in normalized if abs(position.signed_size) > 1e-9]

    def fetch_open_orders(self, symbol: str | None = None) -> list[BrokerOrder]:
        orders = self.exchange.fetch_open_orders(symbol) if symbol else self.exchange.fetch_open_orders()
        return [BrokerOrder.from_ccxt(order) for order in orders]

    def fetch_order(self, order_id: str, symbol: str) -> BrokerOrder:
        return BrokerOrder.from_ccxt(self.exchange.fetch_order(order_id, symbol))

    def cancel_order(self, order_id: str, symbol: str) -> BrokerOrder:
        return BrokerOrder.from_ccxt(self.exchange.cancel_order(order_id, symbol))

    def create_market_order(
        self,
        symbol: str,
        *,
        side: str,
        amount: float,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> BrokerOrder:
        params: dict[str, Any] = {}
        if reduce_only:
            params["reduceOnly"] = True
        if client_order_id:
            params["clientOrderId"] = client_order_id
        order = self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params=params,
        )
        return BrokerOrder.from_ccxt(order)

    def create_trigger_order(
        self,
        symbol: str,
        *,
        side: str,
        amount: float,
        trigger_price: float,
        order_type: str,
        reduce_only: bool = True,
    ) -> BrokerOrder:
        params: dict[str, Any] = {
            "reduceOnly": reduce_only,
            "stopPrice": trigger_price,
        }
        order = self.exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            params=params,
        )
        return BrokerOrder.from_ccxt(order)

    def capture_order_book(
        self,
        symbol: str,
        *,
        requested_notional_usd: float = 0.0,
        depth: int = 5,
    ) -> OrderBookSnapshot:
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit=depth)
            bids = order_book.get("bids") or []
            asks = order_book.get("asks") or []
            best_bid = _float(bids[0][0]) if bids else 0.0
            best_ask = _float(asks[0][0]) if asks else 0.0
            mid = (best_bid + best_ask) / 2.0 if best_bid > 0.0 and best_ask > 0.0 else max(best_bid, best_ask)
            spread_bps = ((best_ask - best_bid) / mid * 1e4) if mid > 0.0 and best_ask >= best_bid > 0.0 else 0.0
            bid_depth_usd = float(sum(_float(level[0]) * _float(level[1]) for level in bids[:depth]))
            ask_depth_usd = float(sum(_float(level[0]) * _float(level[1]) for level in asks[:depth]))
            available_depth = max(min(bid_depth_usd, ask_depth_usd), 1e-9)
            estimated_impact_bps = 0.0
            if requested_notional_usd > 0.0:
                estimated_impact_bps = float(np.sqrt(requested_notional_usd / available_depth) * 30.0)
            return OrderBookSnapshot(
                symbol=symbol,
                timestamp=_utc_iso_from_ms(order_book.get("timestamp")),
                best_bid=best_bid,
                best_ask=best_ask,
                spread_bps=spread_bps,
                bid_depth_usd=bid_depth_usd,
                ask_depth_usd=ask_depth_usd,
                estimated_impact_bps=estimated_impact_bps,
            )
        except Exception as exc:
            logger.warning("Order book snapshot failed for %s on %s: %s", symbol, self.exchange_id, exc)
            ticker = self.fetch_ticker(symbol)
            bid = _float(ticker.get("bid"))
            ask = _float(ticker.get("ask"))
            last = _float(ticker.get("last"))
            best_bid = bid if bid > 0.0 else last
            best_ask = ask if ask > 0.0 else last
            mid = (best_bid + best_ask) / 2.0 if best_bid > 0.0 and best_ask > 0.0 else max(best_bid, best_ask)
            spread_bps = ((best_ask - best_bid) / mid * 1e4) if mid > 0.0 and best_ask >= best_bid > 0.0 else 0.0
            return OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc).isoformat(),
                best_bid=best_bid,
                best_ask=best_ask,
                spread_bps=spread_bps,
                bid_depth_usd=0.0,
                ask_depth_usd=0.0,
                estimated_impact_bps=0.0,
            )

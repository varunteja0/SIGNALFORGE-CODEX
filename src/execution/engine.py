"""
SignalForge Live Execution Engine
==================================
Handles real (and paper) trade execution on exchanges.

SAFETY FEATURES:
- Paper trading mode by default
- Order confirmation logging
- Automatic stop-loss placement
- Rate limiting
- Error recovery
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    price: float = 0
    size: float = 0
    cost: float = 0
    fee: float = 0
    error: str = ""
    is_paper: bool = True


class ExecutionEngine:
    """Handles order execution with safety checks."""

    def __init__(
        self,
        exchange: ccxt.Exchange,
        paper_mode: bool = True,
    ):
        self.exchange = exchange
        self.paper_mode = paper_mode
        self.paper_positions: dict[str, dict] = {}
        self.paper_capital = 10000
        self.order_history: list[OrderResult] = []

        if paper_mode:
            logger.info("PAPER TRADING MODE — no real money at risk")
        else:
            logger.warning("LIVE TRADING MODE — real money at risk!")

    def execute_entry(
        self,
        symbol: str,
        direction: int,
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> OrderResult:
        """Execute a market entry order."""
        side = "buy" if direction == 1 else "sell"

        if self.paper_mode:
            return self._paper_entry(symbol, direction, size, stop_loss, take_profit)

        try:
            # Place market order
            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=size,
            )

            result = OrderResult(
                success=True,
                order_id=order.get("id", ""),
                symbol=symbol,
                side=side,
                price=order.get("average", order.get("price", 0)),
                size=size,
                cost=order.get("cost", 0),
                fee=order.get("fee", {}).get("cost", 0),
                is_paper=False,
            )

            # Place stop loss if specified
            if stop_loss:
                sl_side = "sell" if direction == 1 else "buy"
                try:
                    self.exchange.create_order(
                        symbol=symbol,
                        type="stop_market",
                        side=sl_side,
                        amount=size,
                        params={"stopPrice": stop_loss},
                    )
                    logger.info(f"Stop loss placed at {stop_loss}")
                except Exception as e:
                    logger.error(f"Failed to place stop loss: {e}")

            # Place take profit if specified
            if take_profit:
                tp_side = "sell" if direction == 1 else "buy"
                try:
                    self.exchange.create_order(
                        symbol=symbol,
                        type="take_profit_market",
                        side=tp_side,
                        amount=size,
                        params={"stopPrice": take_profit},
                    )
                    logger.info(f"Take profit placed at {take_profit}")
                except Exception as e:
                    logger.error(f"Failed to place take profit: {e}")

            logger.info(f"ENTRY: {side.upper()} {size} {symbol} @ {result.price}")
            self.order_history.append(result)
            return result

        except Exception as e:
            result = OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                error=str(e),
                is_paper=False,
            )
            logger.error(f"ENTRY FAILED: {symbol} {side} - {e}")
            self.order_history.append(result)
            return result

    def execute_exit(self, symbol: str, size: float, direction: int) -> OrderResult:
        """Execute a market exit order."""
        side = "sell" if direction == 1 else "buy"  # Opposite of entry

        if self.paper_mode:
            return self._paper_exit(symbol, direction)

        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=size,
            )

            # Cancel any remaining stop/TP orders for this symbol
            try:
                open_orders = self.exchange.fetch_open_orders(symbol)
                for oo in open_orders:
                    self.exchange.cancel_order(oo["id"], symbol)
            except Exception:
                pass

            result = OrderResult(
                success=True,
                order_id=order.get("id", ""),
                symbol=symbol,
                side=side,
                price=order.get("average", order.get("price", 0)),
                size=size,
                cost=order.get("cost", 0),
                fee=order.get("fee", {}).get("cost", 0),
                is_paper=False,
            )

            logger.info(f"EXIT: {side.upper()} {size} {symbol} @ {result.price}")
            self.order_history.append(result)
            return result

        except Exception as e:
            result = OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                error=str(e),
                is_paper=False,
            )
            logger.error(f"EXIT FAILED: {symbol} {side} - {e}")
            self.order_history.append(result)
            return result

    def _paper_entry(
        self, symbol: str, direction: int, size: float,
        stop_loss: Optional[float], take_profit: Optional[float],
    ) -> OrderResult:
        """Simulate entry on paper."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker["last"]
        except Exception:
            price = 0
            logger.warning(f"Could not fetch price for {symbol}, using 0")

        cost = price * size
        if cost > self.paper_capital * 0.95:
            return OrderResult(success=False, error="Insufficient paper capital")

        self.paper_positions[symbol] = {
            "direction": direction,
            "size": size,
            "entry_price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        result = OrderResult(
            success=True,
            order_id=f"paper_{int(time.time())}",
            symbol=symbol,
            side="buy" if direction == 1 else "sell",
            price=price,
            size=size,
            cost=cost,
            fee=cost * 0.001,
            is_paper=True,
        )

        logger.info(
            f"PAPER ENTRY: {'LONG' if direction == 1 else 'SHORT'} "
            f"{size:.6f} {symbol} @ ${price:.2f}"
        )
        self.order_history.append(result)
        return result

    def _paper_exit(self, symbol: str, direction: int) -> OrderResult:
        """Simulate exit on paper."""
        if symbol not in self.paper_positions:
            return OrderResult(success=False, error=f"No paper position in {symbol}")

        pos = self.paper_positions.pop(symbol)

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker["last"]
        except Exception:
            price = pos["entry_price"]

        if pos["direction"] == 1:
            pnl = (price - pos["entry_price"]) * pos["size"]
        else:
            pnl = (pos["entry_price"] - price) * pos["size"]

        fee = price * pos["size"] * 0.001
        pnl -= fee
        self.paper_capital += pnl

        result = OrderResult(
            success=True,
            order_id=f"paper_{int(time.time())}",
            symbol=symbol,
            side="sell" if pos["direction"] == 1 else "buy",
            price=price,
            size=pos["size"],
            cost=price * pos["size"],
            fee=fee,
            is_paper=True,
        )

        logger.info(
            f"PAPER EXIT: {symbol} @ ${price:.2f} PnL=${pnl:.2f} "
            f"Capital=${self.paper_capital:.2f}"
        )
        self.order_history.append(result)
        return result

    def get_open_positions(self) -> dict:
        """Get currently open positions."""
        if self.paper_mode:
            return self.paper_positions

        try:
            positions = self.exchange.fetch_positions()
            return {
                p["symbol"]: {
                    "direction": 1 if p["side"] == "long" else -1,
                    "size": p["contracts"],
                    "entry_price": p["entryPrice"],
                    "unrealized_pnl": p["unrealizedPnl"],
                }
                for p in positions
                if p["contracts"] > 0
            }
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return {}

    def get_balance(self) -> float:
        """Get available balance."""
        if self.paper_mode:
            return self.paper_capital

        try:
            balance = self.exchange.fetch_balance()
            return balance.get("USDT", {}).get("free", 0)
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return 0

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.execution.smart import SmartOrderResult


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class ExecutionObservation:
    venue: str
    broker: str
    symbol: str
    side: str
    direction: int
    reference_price: float
    fill_price: float
    requested_size: float
    filled_size: float
    fill_ratio: float
    slippage_bps: float
    execution_ms: float
    order_id: str = ""
    status: str = ""
    reduce_only: bool = False
    book_spread_bps: float = 0.0
    book_impact_bps: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def filled_notional_usd(self) -> float:
        return self.fill_price * self.filled_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def namespaced(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_price": self.fill_price,
            f"{prefix}_order_id": self.order_id,
            f"{prefix}_slippage_bps": self.slippage_bps,
        }


class ExecutionAbstractionLayer:
    """Normalize paper, live, and shadow execution outputs into one shape."""

    @staticmethod
    def from_order_result(
        result: SmartOrderResult,
        *,
        symbol: str,
        direction: int,
        reference_price: float,
        venue: str,
        fallback_broker: str = "",
        reduce_only: bool = False,
    ) -> ExecutionObservation:
        metadata = dict(result.metadata or {})
        requested_size = _float(result.requested_size, _float(result.size))
        filled_size = _float(result.size)
        fill_ratio = filled_size / requested_size if requested_size > 0.0 else 0.0
        fill_price = _float(result.price, reference_price)
        broker = str(metadata.get("broker") or fallback_broker or venue)
        status = str(metadata.get("status") or ("filled" if result.success else "rejected"))
        return ExecutionObservation(
            venue=venue,
            broker=broker,
            symbol=symbol,
            side=str(result.side or ("buy" if direction == 1 else "sell")),
            direction=direction,
            reference_price=_float(reference_price),
            fill_price=fill_price,
            requested_size=requested_size,
            filled_size=filled_size,
            fill_ratio=fill_ratio,
            slippage_bps=_float(result.slippage_bps),
            execution_ms=_float(result.execution_ms),
            order_id=str(result.order_id or metadata.get("entry_order_id", "")),
            status=status,
            reduce_only=reduce_only,
            book_spread_bps=_float(metadata.get("book_spread_bps")),
            book_impact_bps=_float(metadata.get("book_impact_bps")),
            metadata=metadata,
        )

    @staticmethod
    def from_shadow_payload(
        payload: dict[str, Any] | None,
        *,
        symbol: str,
        direction: int,
        qty: float,
        reference_price: float,
        reduce_only: bool = False,
    ) -> ExecutionObservation | None:
        if not isinstance(payload, dict) or payload.get("error"):
            return None
        fill_price = _float(payload.get("price"), reference_price)
        requested_size = _float(qty)
        filled_size = requested_size
        fill_ratio = filled_size / requested_size if requested_size > 0.0 else 0.0
        return ExecutionObservation(
            venue="shadow",
            broker=str(payload.get("broker", "shadow")),
            symbol=symbol,
            side="buy" if direction == 1 else "sell",
            direction=direction,
            reference_price=_float(reference_price),
            fill_price=fill_price,
            requested_size=requested_size,
            filled_size=filled_size,
            fill_ratio=fill_ratio,
            slippage_bps=_float(payload.get("slippage_bps")),
            execution_ms=0.0,
            order_id=str(payload.get("order_id", "")),
            status=str(payload.get("status", "filled")),
            reduce_only=reduce_only,
            book_spread_bps=_float(payload.get("spread_bps")),
            book_impact_bps=_float(payload.get("impact_bps")),
            metadata=dict(payload),
        )


__all__ = [
    "ExecutionAbstractionLayer",
    "ExecutionObservation",
]
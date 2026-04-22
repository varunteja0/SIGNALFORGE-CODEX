from __future__ import annotations

from src.execution.abstraction import ExecutionAbstractionLayer
from src.execution.smart import SmartOrderResult


def test_execution_abstraction_normalizes_order_result() -> None:
    result = SmartOrderResult(
        success=True,
        order_id="ord-123",
        symbol="BTC/USDT",
        side="buy",
        price=60_120.0,
        size=0.2,
        requested_size=0.25,
        slippage_bps=4.2,
        execution_ms=180.0,
        is_paper=False,
        algo="market",
        metadata={
            "broker": "bybit",
            "status": "filled",
            "book_spread_bps": 3.0,
            "book_impact_bps": 4.5,
        },
    )

    observation = ExecutionAbstractionLayer.from_order_result(
        result,
        symbol="BTC/USDT",
        direction=1,
        reference_price=60_000.0,
        venue="live",
        fallback_broker="bybit",
    )

    assert observation.broker == "bybit"
    assert observation.order_id == "ord-123"
    assert observation.fill_ratio == 0.8
    assert observation.book_spread_bps == 3.0
    assert observation.filled_notional_usd == 60_120.0 * 0.2


def test_execution_abstraction_normalizes_shadow_payload() -> None:
    observation = ExecutionAbstractionLayer.from_shadow_payload(
        {
            "broker": "bybit-shadow",
            "order_id": "shadow-1",
            "price": 2500.0,
            "status": "filled",
            "slippage_bps": 6.0,
            "spread_bps": 2.0,
            "impact_bps": 3.5,
        },
        symbol="ETH/USDT",
        direction=-1,
        qty=1.5,
        reference_price=2490.0,
        reduce_only=True,
    )

    assert observation is not None
    assert observation.venue == "shadow"
    assert observation.side == "sell"
    assert observation.reduce_only is True
    assert observation.namespaced("shadow_exit") == {
        "shadow_exit_price": 2500.0,
        "shadow_exit_order_id": "shadow-1",
        "shadow_exit_slippage_bps": 6.0,
    }
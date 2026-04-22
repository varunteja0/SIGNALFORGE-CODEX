from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd

from scripts.go_live import LiveTrader


def test_log_proximity_uses_base_strategy_metrics_for_mutated_slots(caplog) -> None:
    trader = LiveTrader.__new__(LiveTrader)
    trader.slots = [
        SimpleNamespace(
            name="momentum_breakout__mutate",
            allowed_assets=["ETH/USDT"],
        )
    ]
    datasets = {
        "ETH/USDT": pd.DataFrame(
            {
                "atr_14": [1.0] * 29 + [2.0],
                "volume": [1.0] * 29 + [2.0],
            }
        )
    }

    with caplog.at_level(logging.INFO, logger="GoLive"):
        LiveTrader._log_proximity(trader, datasets)

    assert "momentum_breakout__mutate" in caplog.text
    assert "atr=" in caplog.text
    assert "(ETH)" in caplog.text
    assert "  0%" not in caplog.text
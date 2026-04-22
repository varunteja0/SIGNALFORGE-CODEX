from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.opportunity_engine import OpportunityEngine, OpportunityEngineConfig


def test_build_weights_respects_caps_and_budget() -> None:
    engine = OpportunityEngine(
        OpportunityEngineConfig(
            gross_limit=1.0,
            net_limit=0.30,
            per_asset_cap=0.25,
            max_positions=4,
            max_longs=3,
            max_shorts=3,
        )
    )
    snapshot = pd.DataFrame(
        {
            "alpha": [0.9, 0.7, -0.8, -0.4, 0.1],
            "opportunity": [0.8, 0.7, 0.9, 0.4, 0.05],
            "realized_vol": [0.02, 0.03, 0.02, 0.04, 0.01],
            "liquidity": [0.8, 0.8, 0.9, 0.7, 0.9],
            "data_quality": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"],
    )

    weights = engine._build_weights(snapshot, gross_budget=1.0)

    assert float(weights.abs().sum()) <= 1.0 + 1e-9
    assert abs(float(weights.sum())) <= 0.30 + 1e-9
    assert (weights.abs() <= 0.25 + 1e-9).all()
    assert weights["BTC/USDT"] > 0.0
    assert weights["SOL/USDT"] < 0.0
    assert weights["DOGE/USDT"] == 0.0


def _synthetic_dataset(direction: float, n: int = 320) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    base_ret = np.full(n, direction)
    close = 100.0 * np.cumprod(1.0 + base_ret)
    return pd.DataFrame(
        {
            "close": close,
            "volume": np.full(n, 1_000_000.0),
            "vol_20": np.full(n, 0.005),
            "vol_50": np.full(n, 0.005),
            "vol_of_vol_20": np.full(n, 0.001),
            "atr_14": np.full(n, 1.0),
            "ret_vol_adj_20": np.full(n, np.sign(direction) * 1.2),
            "ema_cross_20_100": np.full(n, np.sign(direction) * 0.03),
            "macd_hist_slope": np.full(n, np.sign(direction) * 0.20),
            "adx_14": np.full(n, 28.0),
            "price_vs_ma_50": np.full(n, np.sign(direction) * 0.04),
            "fund_funding_zscore": np.full(n, -np.sign(direction) * 0.8),
            "lsr_lsr_zscore": np.full(n, -np.sign(direction) * 0.5),
            "top_retail_divergence_zscore": np.full(n, np.sign(direction) * 0.6),
            "liq_pressure_short": np.full(n, 2.0 if direction > 0 else 0.1),
            "liq_pressure_long": np.full(n, 0.1 if direction > 0 else 2.0),
            "smart_money_divergence": np.full(n, np.sign(direction) * 0.7),
            "cross_venue_funding_zscore": np.full(n, -np.sign(direction) * 0.3),
            "squeeze": np.zeros(n),
            "ret_3": np.full(n, np.sign(direction) * 0.01),
            "taker_taker_imbalance": np.full(n, np.sign(direction) * 0.2),
        },
        index=idx,
    )


def test_backtest_compounds_on_clear_long_short_opportunities() -> None:
    engine = OpportunityEngine(
        OpportunityEngineConfig(
            assets=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            initial_capital=10_000.0,
            gross_limit=1.0,
            net_limit=0.20,
            per_asset_cap=0.40,
            max_positions=2,
            max_longs=1,
            max_shorts=1,
            turnover_cost_bps=0.0,
            use_multi_venue=False,
        )
    )

    datasets = {
        "BTC/USDT": _synthetic_dataset(0.0015),
        "ETH/USDT": _synthetic_dataset(-0.0012),
        "SOL/USDT": _synthetic_dataset(0.0),
    }
    result = engine.backtest(datasets)

    assert result.total_return > 0.0
    assert result.sharpe > 0.0
    assert result.max_drawdown < 0.10
    assert not result.weights.empty
    assert float(result.weights.abs().sum(axis=1).max()) <= 1.0 + 1e-9
    assert float(result.weights.sum(axis=1).abs().max()) <= 0.20 + 1e-9

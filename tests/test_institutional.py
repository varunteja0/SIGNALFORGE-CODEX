from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.institutional import CorrelationEngine, _profit_factor


def test_correlation_engine_handles_read_only_corr_arrays() -> None:
    index = pd.date_range("2024-01-01", periods=100, freq="1h")

    base = np.linspace(10_000.0, 11_000.0, len(index))
    hedge = np.linspace(10_000.0, 10_400.0, len(index))
    alt = np.linspace(10_000.0, 10_800.0, len(index))

    equity_curves = {
        "funding_mr_v7|BTC/USDT": pd.Series(base, index=index),
        "momentum_breakout|ETH/USDT": pd.Series(hedge, index=index),
        "contrarian_asym|SOL/USDT": pd.Series(alt, index=index),
    }

    report = CorrelationEngine(rolling_window=24).analyze(equity_curves)

    assert not report.strategy_corr.empty
    assert not report.asset_corr.empty
    assert np.isfinite(report.max_strategy_corr)
    assert np.isfinite(report.max_asset_corr)


def test_profit_factor_handles_no_loss_samples() -> None:
    assert _profit_factor(10.0, 5.0) == 2.0
    assert np.isinf(_profit_factor(10.0, 0.0))
    assert _profit_factor(0.0, 0.0) == 0.0

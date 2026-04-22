from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.engine.portfolio_engine import PortfolioEngine, StrategySlot, _profit_factor


def test_profit_factor_handles_no_loss_samples() -> None:
    assert _profit_factor(10.0, 5.0) == 2.0
    assert np.isinf(_profit_factor(10.0, 0.0))
    assert _profit_factor(0.0, 0.0) == 0.0


def test_monthly_challenge_profile_is_concentrated_and_levered() -> None:
    engine = PortfolioEngine.monthly_challenge()

    assert [slot.name for slot in engine.slots] == [
        "momentum_breakout_swing",
        "momentum_breakout_fast",
    ]
    assert all(slot.position_size_pct == pytest.approx(1.0) for slot in engine.slots)
    assert all(
        slot.allowed_assets == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        for slot in engine.slots
    )
    assert engine.data_days == 30
    assert engine.max_position_notional_pct == pytest.approx(18.035)
    assert engine.max_total_exposure == pytest.approx(18.035)
    assert engine.risk_manager is None
    assert engine.regime_allocator is None


def test_compounding_focus_profile_matches_validated_winner() -> None:
    engine = PortfolioEngine.compounding_focus()

    assert [slot.name for slot in engine.slots] == [
        "funding_mr_v7",
        "extreme_spike",
        "momentum_breakout",
        "contrarian_asym",
    ]
    assert engine.assets == ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
    assert engine.max_total_exposure == pytest.approx(0.10)
    assert engine.max_position_notional_pct == pytest.approx(0.20)

    assets_by_slot = {slot.name: slot.allowed_assets for slot in engine.slots}
    assert assets_by_slot == {
        "funding_mr_v7": ["ETH/USDT", "XRP/USDT"],
        "extreme_spike": ["XRP/USDT"],
        "momentum_breakout": ["ETH/USDT"],
        "contrarian_asym": ["XRP/USDT"],
    }


def test_backtest_splits_total_exposure_across_duplicate_cells() -> None:
    index = pd.date_range("2024-01-01", periods=260, freq="h")
    open_prices = np.linspace(100.0, 130.0, len(index))
    close_prices = open_prices + 0.25
    dataset = pd.DataFrame(
        {
            "open": open_prices,
            "high": close_prices + 0.25,
            "low": open_prices - 0.25,
            "close": close_prices,
            "atr_14": np.full(len(index), 0.5),
            "atr_ratio": np.ones(len(index)),
            "volume_ratio": np.ones(len(index)),
            "fund_funding_rate": np.zeros(len(index)),
        },
        index=index,
    )

    def long_after_warmup(df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)
        signals.iloc[201:] = 1
        return signals

    slot = StrategySlot(
        name="dup_test",
        template="test",
        signal_func=long_after_warmup,
        allowed_assets=["ETH/USDT"],
        stop_loss_atr=2.0,
        take_profit_atr=100.0,
        max_holding_bars=10_000,
        position_size_pct=1.0,
    )

    single = PortfolioEngine(
        slots=[slot],
        assets=["ETH/USDT"],
        capital=10_000,
        data_days=30,
        max_total_exposure=1.0,
        max_position_notional_pct=1.0,
        use_regime_allocator=False,
        use_risk_manager=False,
        use_divergence_tracker=False,
        use_market_state_brain=False,
        use_execution_edge=False,
        use_live_adaptation=False,
        use_capital_scaling=False,
    ).backtest({"ETH/USDT": dataset})

    duplicate = PortfolioEngine(
        slots=[slot, slot],
        assets=["ETH/USDT"],
        capital=10_000,
        data_days=30,
        max_total_exposure=1.0,
        max_position_notional_pct=1.0,
        use_regime_allocator=False,
        use_risk_manager=False,
        use_divergence_tracker=False,
        use_market_state_brain=False,
        use_execution_edge=False,
        use_live_adaptation=False,
        use_capital_scaling=False,
    ).backtest({"ETH/USDT": dataset})

    assert single.total_trades == 1
    assert duplicate.total_trades == 2
    assert duplicate.total_pnl == pytest.approx(single.total_pnl, rel=1e-6)

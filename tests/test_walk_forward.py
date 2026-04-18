"""Tests for the walk-forward harness."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Backtester
from src.backtest.walk_forward import Fold, make_folds, walk_forward


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _synthetic(n: int = 3000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.lognormal(mean=10.0, sigma=0.3, size=n)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h")
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([open_, close, high]),
            "low": np.minimum.reduce([open_, close, low]),
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14, min_periods=1).mean()
    df["atr_ratio"] = df["atr_14"] / df["atr_14"].rolling(50, min_periods=1).mean()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(50, min_periods=1).mean()
    return df


def _long_every_50(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(0, index=df.index, dtype=int)
    s.iloc[::50] = 1
    return s


# --------------------------------------------------------------------------
# make_folds
# --------------------------------------------------------------------------
def test_make_folds_anchored_produces_growing_train_window():
    idx = pd.date_range("2020-01-01", periods=3000, freq="1h")
    folds = make_folds(idx, n_folds=4, anchored=True, min_train_bars=500)
    assert len(folds) == 4
    # Train always starts at index[0] when anchored.
    for f in folds:
        assert f.train_start == idx[0]
    # Test windows are strictly non-overlapping and in order.
    for a, b in zip(folds, folds[1:]):
        assert a.test_end < b.test_start
        assert b.test_start > a.test_end


def test_make_folds_rolling_has_fixed_train_length():
    idx = pd.date_range("2020-01-01", periods=3000, freq="1h")
    folds = make_folds(idx, n_folds=4, anchored=False, train_bars=800,
                       test_bars=300, min_train_bars=500)
    lengths = [(f.train_end - f.train_start).total_seconds() for f in folds]
    # All train windows should be the same length (± 1h).
    assert max(lengths) - min(lengths) <= 3600 * 2


def test_make_folds_refuses_too_short_index():
    idx = pd.date_range("2020-01-01", periods=100, freq="1h")
    with pytest.raises(ValueError):
        make_folds(idx, n_folds=3, min_train_bars=500)


def test_make_folds_rejects_non_monotonic_index():
    idx = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-01", "2020-01-03"]))
    with pytest.raises(ValueError):
        make_folds(idx, n_folds=2, min_train_bars=1, test_bars=1)


def test_make_folds_rejects_non_datetime_index():
    with pytest.raises(TypeError):
        make_folds(pd.Index(range(1000)), n_folds=3)  # type: ignore[arg-type]


def test_folds_test_windows_cover_disjoint_bars():
    idx = pd.date_range("2020-01-01", periods=3000, freq="1h")
    folds = make_folds(idx, n_folds=5, anchored=True, min_train_bars=500)
    spans = [(f.test_start, f.test_end) for f in folds]
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert e1 < s2


# --------------------------------------------------------------------------
# walk_forward integration
# --------------------------------------------------------------------------
def test_walk_forward_runs_and_aggregates():
    df = _synthetic(n=3000)
    result = walk_forward(
        df,
        _long_every_50,
        n_folds=3,
        anchored=True,
        backtester=Backtester(initial_capital=10_000),
        position_size_pct=0.02,
        stop_loss_atr=2.0,
        take_profit_atr=3.0,
        max_holding_bars=30,
    )
    assert result.n_folds >= 2
    assert len(result.fold_sharpes) == result.n_folds
    agg = result.aggregate
    for key in ("sharpe_mean", "sharpe_std", "pooled_return", "frac_positive", "worst_max_dd"):
        assert key in agg
    # Pooled return is bounded (no NaNs, no infinities).
    assert np.isfinite(agg["pooled_return"])
    assert 0.0 <= agg["frac_positive"] <= 1.0
    assert 0.0 <= agg["worst_max_dd"] <= 1.0


def test_walk_forward_is_reproducible():
    df = _synthetic(n=3000, seed=11)
    r1 = walk_forward(df, _long_every_50, n_folds=3)
    r2 = walk_forward(df, _long_every_50, n_folds=3)
    assert r1.fold_sharpes == r2.fold_sharpes
    assert r1.fold_returns == r2.fold_returns


def test_walk_forward_summary_contains_expected_fields():
    df = _synthetic(n=3000)
    result = walk_forward(df, _long_every_50, n_folds=3)
    s = result.summary()
    assert "Walk-forward:" in s
    assert "Sharpe" in s
    assert "Positive folds" in s

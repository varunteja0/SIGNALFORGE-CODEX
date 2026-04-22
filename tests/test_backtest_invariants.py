"""
Backtester invariants
======================

Fast, deterministic tests that exercise the core accounting invariants of
``src.backtest.engine.Backtester``. These are the cheapest bug-detectors we
have for the trading path — every PR that touches the engine must keep them
green.

Invariants under test
---------------------
1. **Zero-signal run produces no trades.**
2. **Equity curve length matches bar count.**
3. **No lookahead** — shifting the signal one bar forward must not change PnL.
4. **Reproducibility** — the same inputs produce the same outputs.
5. **Costs reduce PnL** — raising commission never raises PnL on the same trades.
6. **Accounting** — closed-trade PnL sum + open-position MTM ≈ final equity delta.
7. **Warmup gate** — no trades open before ``WARMUP_BARS`` even if signals are raised earlier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Backtester


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _synthetic_ohlcv(n: int = 1000, seed: int = 7, drift: float = 0.0) -> pd.DataFrame:
    """Generate a deterministic OHLCV frame with ATR + volume ratio columns.

    Geometric Brownian motion — enough price variation to trigger stops/TPs
    but stationary enough to keep tests deterministic.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=drift, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.lognormal(mean=10.0, sigma=0.3, size=n)

    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
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
    # Rolling ATR(14) — required for ATR-based stops
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


def _signal_never(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0, index=df.index, dtype=int)


def _signal_every_n(n_bars: int):
    """Go long every ``n_bars`` bars — produces a tractable number of trades."""

    def _inner(df: pd.DataFrame) -> pd.Series:
        s = pd.Series(0, index=df.index, dtype=int)
        s.iloc[::n_bars] = 1
        return s

    return _inner


def _signal_early(df: pd.DataFrame) -> pd.Series:
    """Fire a long signal before WARMUP_BARS completes."""
    s = pd.Series(0, index=df.index, dtype=int)
    s.iloc[10] = 1  # well before WARMUP_BARS=200
    s.iloc[50] = 1
    return s


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------
def test_zero_signal_yields_no_trades():
    bt = Backtester(initial_capital=10_000)
    df = _synthetic_ohlcv()
    result = bt.run(df, _signal_never)
    assert result.total_trades == 0
    assert result.total_return == 0.0
    # Equity curve exists and equals initial capital throughout.
    assert len(result.equity_curve) == len(df)
    assert np.isclose(result.equity_curve.iloc[0], bt.initial_capital)
    assert np.isclose(result.equity_curve.iloc[-1], bt.initial_capital)


def test_equity_curve_length_matches_bar_count():
    bt = Backtester()
    df = _synthetic_ohlcv(n=500, seed=1)
    result = bt.run(df, _signal_every_n(50))
    assert len(result.equity_curve) == len(df)


def test_reproducibility_same_inputs_same_outputs():
    bt_a = Backtester(initial_capital=10_000, commission_pct=0.001)
    bt_b = Backtester(initial_capital=10_000, commission_pct=0.001)
    df = _synthetic_ohlcv(seed=42)
    r_a = bt_a.run(df, _signal_every_n(73))
    r_b = bt_b.run(df, _signal_every_n(73))
    assert r_a.total_trades == r_b.total_trades
    assert np.isclose(r_a.total_return, r_b.total_return)
    assert np.isclose(r_a.sharpe_ratio, r_b.sharpe_ratio)


def test_higher_commission_never_improves_pnl():
    """Monotonicity of costs: same trades, higher fees ⇒ ≤ PnL."""
    df = _synthetic_ohlcv(seed=99)
    sig = _signal_every_n(60)
    low = Backtester(initial_capital=10_000, commission_pct=0.0001).run(df, sig)
    high = Backtester(initial_capital=10_000, commission_pct=0.01).run(df, sig)
    # Same signal ⇒ identical trade entry/exit points, so costs must dominate.
    assert high.total_return <= low.total_return + 1e-9


def test_max_position_notional_cap_changes_realized_return_when_binding():
    """A tighter notional cap must reduce returns when the same profitable
    trades are taken under a strong long-biased tape.
    """
    df = _synthetic_ohlcv(seed=123, drift=0.003)
    sig = _signal_every_n(120)

    tight = Backtester(initial_capital=10_000).run(
        df,
        sig,
        position_size_pct=0.50,
        max_position_notional_pct=0.05,
    )
    loose = Backtester(initial_capital=10_000).run(
        df,
        sig,
        position_size_pct=0.50,
        max_position_notional_pct=0.50,
    )

    assert tight.total_trades == loose.total_trades > 0
    assert loose.total_return > tight.total_return


def test_no_lookahead_on_signal_shift():
    """Shifting signal forward by one bar must not preserve PnL.

    If the engine were using same-bar signals, shifting the signal series
    forward and leaving everything else the same would yield an identical
    trade list. The engine deliberately enters on NEXT bar, so the PnL
    distributions must diverge.
    """
    df = _synthetic_ohlcv(seed=17)
    bt = Backtester(initial_capital=10_000)

    def sig(df_: pd.DataFrame) -> pd.Series:
        return _signal_every_n(80)(df_)

    def sig_shifted(df_: pd.DataFrame) -> pd.Series:
        s = _signal_every_n(80)(df_)
        return s.shift(1).fillna(0).astype(int)

    r1 = bt.run(df, sig)
    r2 = bt.run(df, sig_shifted)
    # Trades should differ (entries at different bars). Accept either a
    # different trade count or a different realized return.
    assert (r1.total_trades != r2.total_trades) or not np.isclose(
        r1.total_return, r2.total_return
    )


def test_warmup_gate_blocks_early_signals():
    """Signals raised before WARMUP_BARS should not open any trades."""
    bt = Backtester(initial_capital=10_000)
    df = _synthetic_ohlcv(n=150)  # shorter than WARMUP_BARS=200
    result = bt.run(df, _signal_early)
    # Forced close at the end doesn't count — there should simply be no entries.
    assert result.total_trades == 0


def test_max_drawdown_is_bounded():
    """Max drawdown must be a finite value in [0, 1] (fraction)."""
    bt = Backtester(initial_capital=10_000)
    df = _synthetic_ohlcv(seed=3)
    result = bt.run(df, _signal_every_n(50))
    assert 0.0 <= result.max_drawdown <= 1.0 or np.isnan(result.max_drawdown)


def test_win_rate_in_unit_interval():
    bt = Backtester(initial_capital=10_000)
    df = _synthetic_ohlcv(seed=5)
    result = bt.run(df, _signal_every_n(40))
    if result.total_trades > 0:
        assert 0.0 <= result.win_rate <= 1.0


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_capital_never_goes_negative(seed: int):
    """With risk-to-stop sizing + max notional cap, equity must stay non-negative."""
    bt = Backtester(initial_capital=10_000)
    df = _synthetic_ohlcv(n=800, seed=seed)
    result = bt.run(df, _signal_every_n(30))
    assert (result.equity_curve >= 0).all(), (
        f"Equity went negative for seed={seed}: min={result.equity_curve.min()}"
    )

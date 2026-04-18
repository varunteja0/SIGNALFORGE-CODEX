"""
Walk-forward validation
=======================

Rolling-origin evaluation of a signal function. The current
:class:`~src.backtest.engine.Backtester` supports a single in-sample /
out-of-sample cut. Allocators and auditors want to see **many**
disjoint OOS windows, so a strategy can't fit a single lucky regime.

Scheme
------
A *fold* is defined by:

- ``train_start``, ``train_end``    — in-sample window
- ``test_start``,  ``test_end``     — out-of-sample window immediately
                                      following ``train_end``

Two modes:

- **Anchored** (``anchored=True``): train window grows; every fold
  starts at time zero, test window slides forward. This is the
  realistic live-trading scheme — you never throw data away.
- **Rolling** (``anchored=False``): train window is a fixed length;
  both train and test slide forward together. Tests robustness to
  regime shift.

The harness assumes the signal function is **already calibrated** —
there's no hyperparameter optimization here. For optimization, use a
nested scheme (hyperopt on ``train``, evaluate on ``test``).

Output
------
:class:`WalkForwardResult` carries per-fold
:class:`~src.backtest.engine.BacktestResult` objects plus aggregate
statistics (mean / std / min Sharpe, fraction of positive folds,
pooled equity curve).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.backtest.engine import Backtester, BacktestResult
from src.obs import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Fold definitions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def train_bars(self) -> int:  # populated lazily by the harness
        return 0


def make_folds(
    index: pd.DatetimeIndex,
    *,
    n_folds: int = 5,
    train_bars: int | None = None,
    test_bars: int | None = None,
    anchored: bool = True,
    min_train_bars: int = 500,
) -> list[Fold]:
    """Partition a time index into walk-forward folds.

    Parameters
    ----------
    index :
        Strictly increasing DatetimeIndex.
    n_folds :
        Number of OOS windows. Ignored if both ``train_bars`` and
        ``test_bars`` are given.
    train_bars :
        Fixed training length. When ``None``, derived so that all
        ``n_folds`` test windows tile the tail of the series.
    test_bars :
        Fixed test length. When ``None``, derived from ``n_folds``.
    anchored :
        True → growing train window; False → fixed-length rolling.
    min_train_bars :
        Safety floor — refuses to produce folds with less history.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be monotonically increasing")

    n = len(index)
    if n < min_train_bars * 2:
        raise ValueError(
            f"Need at least {min_train_bars * 2} bars, got {n}"
        )

    if test_bars is None:
        # Default: tile the second half of the series with n_folds test windows.
        test_bars = (n - min_train_bars) // max(n_folds, 1)
    if test_bars <= 0:
        raise ValueError("test_bars must be positive")

    if train_bars is None:
        train_bars = min_train_bars  # seed the first fold

    folds: list[Fold] = []
    # First test window starts after the initial train segment.
    test_start_idx = train_bars
    i = 0
    while test_start_idx + test_bars <= n and i < n_folds:
        test_end_idx = test_start_idx + test_bars
        if anchored:
            train_start_idx = 0
        else:
            train_start_idx = max(0, test_start_idx - train_bars)

        folds.append(
            Fold(
                index=i,
                train_start=index[train_start_idx],
                train_end=index[test_start_idx - 1],
                test_start=index[test_start_idx],
                test_end=index[test_end_idx - 1],
            )
        )
        test_start_idx = test_end_idx
        i += 1

    if not folds:
        raise ValueError(
            f"Produced 0 folds for n={n}, train_bars={train_bars}, test_bars={test_bars}"
        )
    return folds


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class WalkForwardResult:
    folds: list[Fold]
    test_results: list[BacktestResult]
    aggregate: dict[str, float] = field(default_factory=dict)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def fold_sharpes(self) -> list[float]:
        return [r.sharpe_ratio for r in self.test_results]

    @property
    def fold_returns(self) -> list[float]:
        return [r.total_return for r in self.test_results]

    def summary(self) -> str:
        agg = self.aggregate
        lines = [
            f"Walk-forward: {self.n_folds} folds",
            f"  Sharpe: mean={agg.get('sharpe_mean', 0):.2f}  "
            f"std={agg.get('sharpe_std', 0):.2f}  "
            f"min={agg.get('sharpe_min', 0):.2f}  "
            f"max={agg.get('sharpe_max', 0):.2f}",
            f"  Return (pooled): {agg.get('pooled_return', 0):+.2%}",
            f"  Positive folds:  {agg.get('frac_positive', 0):.0%}",
            f"  Max DD (any fold): {agg.get('worst_max_dd', 0):.2%}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------
def walk_forward(
    df: pd.DataFrame,
    signal_func: Callable[[pd.DataFrame], pd.Series],
    *,
    folds: list[Fold] | None = None,
    n_folds: int = 5,
    anchored: bool = True,
    backtester: Backtester | None = None,
    **bt_kwargs,
) -> WalkForwardResult:
    """Run a walk-forward evaluation of ``signal_func`` over ``df``.

    ``signal_func`` is called **once per fold** with the combined
    train+test slice — this lets it compute rolling features that look
    back through the train window without peeking into future bars
    within the test window. The backtester then enforces no-lookahead
    inside that slice via its next-bar entry rule.

    All kwargs beyond the documented ones are forwarded to
    :meth:`Backtester.run` (``position_size_pct``, ``stop_loss_atr``,
    ``take_profit_atr``, ``max_holding_bars`` …).
    """
    if folds is None:
        folds = make_folds(df.index, n_folds=n_folds, anchored=anchored)

    bt = backtester or Backtester()
    results: list[BacktestResult] = []

    for fold in folds:
        # Slice includes the train window so lagged features are warm
        # at test_start; backtester skips WARMUP_BARS internally.
        window = df.loc[fold.train_start : fold.test_end]
        # But only score the test portion.
        test_slice = window.loc[fold.test_start : fold.test_end]
        if len(test_slice) < 50:
            log.warning(
                "walk_forward.fold_too_short",
                fold=fold.index,
                bars=len(test_slice),
            )
            continue

        # Build signals over the full window (so features warm up),
        # then restrict to test bars.
        result = bt.run(test_slice, signal_func, **bt_kwargs)
        log.info(
            "walk_forward.fold_complete",
            fold=fold.index,
            test_start=str(fold.test_start),
            test_end=str(fold.test_end),
            n_trades=result.total_trades,
            sharpe=round(result.sharpe_ratio, 3),
            ret=round(result.total_return, 4),
        )
        results.append(result)

    if not results:
        raise RuntimeError("Walk-forward produced 0 usable folds")

    sharpes = np.array([r.sharpe_ratio for r in results], dtype=float)
    returns = np.array([r.total_return for r in results], dtype=float)
    max_dds = np.array([r.max_drawdown for r in results], dtype=float)

    aggregate = {
        "sharpe_mean": float(sharpes.mean()),
        "sharpe_std": float(sharpes.std(ddof=0)),
        "sharpe_min": float(sharpes.min()),
        "sharpe_max": float(sharpes.max()),
        # Geometric pooled return across folds.
        "pooled_return": float(np.prod(1.0 + returns) - 1.0),
        "frac_positive": float((returns > 0).mean()),
        "worst_max_dd": float(max_dds.max()) if len(max_dds) else 0.0,
    }
    return WalkForwardResult(folds=folds[: len(results)], test_results=results, aggregate=aggregate)


__all__ = [
    "Fold",
    "WalkForwardResult",
    "make_folds",
    "walk_forward",
]

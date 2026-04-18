#!/usr/bin/env python3
"""
SignalForge Walk-Forward Stress Test
=====================================

For each KEEP strategy in ``fund_data/validation_v20.json``, re-measure
its OOS Sharpe / return / trades across 5 non-overlapping rolling OOS
windows carved from the full cached history. A KEEP that is genuinely
edge-bearing (not regime-lucky) should show positive Sharpe in >= 3
of the 5 windows. Strategies that only produce positive numbers in
one window are flagged as REGIME_LUCKY.

Method
------
1. Load the cached full dataset for each asset (the harness already
   persisted it; we only read).
2. Partition the last ``total_oos_fraction`` of each series into 5
   equal non-overlapping windows, leaving the earlier 70% as the
   "sufficient context" window so every signal has the 200/336-bar
   lookback it needs.
3. For each KEEP, for each window, run the ``Backtester`` with the
   strategy's already-sized ``position_size_pct`` and ``hold_bars``.
4. Aggregate per-strategy: mean Sharpe, std Sharpe, min Sharpe,
   positive-window count, worst-window drawdown.
5. Classify:
     STABLE         positive-window count >= 4, min sharpe >= 0.0
     ROBUST         positive-window count >= 3, mean sharpe >= 0.5
     REGIME_LUCKY   positive-window count <= 1 OR (max-min sharpe spread > 2.0)
     MARGINAL       everything else

Output: ``fund_data/walk_forward_stress_v20.json``. The paper trader
does NOT auto-consume this (it's an offline diagnostic) but the
REGIME_LUCKY list is the shortlist of strategies to monitor closely
in live paper trading.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.engine import Backtester
from src.data.fetcher import DataFetcher
from src.factory.deployer import DeployedStrategy, set_universe

logger = logging.getLogger("walk_forward_stress")

REPORT_PATH = Path("fund_data/validation_v20.json")
OUTPUT_PATH = Path("fund_data/walk_forward_stress_v20.json")

UNIVERSE_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "SOL/USDT",
]

# Of the full cached history, reserve the first TRAIN_FRACTION for
# providing sufficient lookback context; carve the remainder into
# N_WINDOWS equal OOS slices.
TRAIN_FRACTION = 0.50
N_WINDOWS = 5


@dataclass
class WindowResult:
    window_idx: int
    start: str
    end: str
    trades: int
    sharpe: float
    total_return: float
    max_dd: float


@dataclass
class StrategyStability:
    name: str
    asset: str
    mean_sharpe: float
    std_sharpe: float
    min_sharpe: float
    max_sharpe: float
    positive_windows: int
    worst_dd: float
    total_trades: int
    classification: str
    windows: list[WindowResult]


def _recover_signal_name(strategy_name: str) -> str:
    """Invert the deployer naming convention — identical to paper trader."""
    parts = strategy_name.split("_")
    if len(parts) < 4 or parts[0] != "sf":
        return strategy_name
    inner = parts[2:-1]
    return "_".join(inner)


def load_keeps(report_path: Path) -> list[DeployedStrategy]:
    data = json.loads(report_path.read_text())
    keeps: list[DeployedStrategy] = []
    for row in data.get("strategies", []):
        if row.get("final_verdict") != "KEEP":
            continue
        keeps.append(DeployedStrategy(
            name=row["name"],
            asset=row["asset"],
            direction=int(row["direction"]),
            hold_bars=int(row["hold_bars"]),
            signal_name=_recover_signal_name(row["name"]),
            position_size_pct=float(row["position_size_pct"]),
            stop_loss_atr=2.5,
            take_profit_atr=4.0,
            oos_pf=float(row.get("oos_pf", 1.0)),
            oos_sharpe=float(row.get("oos_sharpe", 0.0)),
            grade=row.get("grade", "B"),
            deployed_at=data.get("timestamp", ""),
        ))
    return keeps


def fetch_cached_universe() -> dict[str, pd.DataFrame]:
    """Use same DataFetcher path as the harness — hits cache when fresh."""
    fetcher = DataFetcher()
    out: dict[str, pd.DataFrame] = {}
    for sym in UNIVERSE_SYMBOLS:
        try:
            df = fetcher.fetch(sym, timeframe="1h", days=1825)
            if df is not None and not df.empty:
                out[sym] = df
        except Exception as exc:
            logger.warning("fetch %s failed: %s", sym, exc)
    return out


def make_windows(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Carve the last (1 - TRAIN_FRACTION) of ``df`` into N_WINDOWS
    non-overlapping slices, preserving the preceding TRAIN_FRACTION
    as context so the strategy's lookback indicators have valid input.

    Each returned DataFrame starts at the training-region boundary and
    extends through the end of its OOS slice, ensuring the
    backtester's own warmup consumes the training region (it won't
    produce trades there because the signal gen sees fresh data from
    bar 1). This matches how the harness evaluates on the OOS slice.
    """
    n = len(df)
    train_end = int(n * TRAIN_FRACTION)
    remaining = n - train_end
    if remaining < N_WINDOWS * 200:
        return []

    window_len = remaining // N_WINDOWS
    windows = []
    for i in range(N_WINDOWS):
        oos_start = train_end + i * window_len
        oos_end = train_end + (i + 1) * window_len if i < N_WINDOWS - 1 else n
        # Include the full history up to oos_end so lookbacks are valid.
        windows.append(df.iloc[:oos_end].copy())
    return windows


def evaluate_window(strat: DeployedStrategy, window_df: pd.DataFrame,
                     oos_start_idx: int) -> WindowResult:
    """Run backtester over the full window; report metrics only on the
    OOS portion. The backtester's warmup is consumed by the pre-OOS
    region so indicator lookbacks are stable when OOS starts."""
    bt = Backtester(initial_capital=10_000, commission_pct=0.0005, slippage_pct=0.0005)
    result = bt.run(
        window_df, strat.generate_signals,
        position_size_pct=strat.position_size_pct,
        stop_loss_atr=strat.stop_loss_atr,
        take_profit_atr=strat.take_profit_atr,
        max_holding_bars=strat.hold_bars,
    )
    # Filter the equity curve to the OOS portion only.
    eq = getattr(result, "equity_curve", None)
    oos_start = window_df.index[oos_start_idx]
    if eq is not None and not eq.empty:
        oos_eq = eq[eq.index >= oos_start]
        if len(oos_eq) >= 2:
            rets = oos_eq.pct_change().dropna()
            mu, sigma = rets.mean(), rets.std()
            sharpe = float(mu / sigma * np.sqrt(24 * 365)) if sigma > 0 else 0.0
            total_ret = float((1 + rets).prod() - 1)
            eq_max = oos_eq.cummax()
            dd = float(-(oos_eq / eq_max - 1).min())
        else:
            sharpe = total_ret = dd = 0.0
    else:
        sharpe = total_ret = dd = 0.0

    return WindowResult(
        window_idx=-1,  # filled by caller
        start=str(window_df.index[oos_start_idx]),
        end=str(window_df.index[-1]),
        trades=int(result.total_trades),
        sharpe=sharpe,
        total_return=total_ret,
        max_dd=dd,
    )


def classify(mean_s: float, min_s: float, max_s: float,
             positive: int) -> str:
    spread = max_s - min_s
    if positive >= 4 and min_s >= 0.0:
        return "STABLE"
    if positive >= 3 and mean_s >= 0.5:
        return "ROBUST"
    if positive <= 1 or spread > 2.0:
        return "REGIME_LUCKY"
    return "MARGINAL"


def stress_test(keeps: list[DeployedStrategy],
                universe: dict[str, pd.DataFrame]) -> list[StrategyStability]:
    set_universe(universe)
    results: list[StrategyStability] = []

    # Precompute windows per asset.
    windows_by_asset: dict[str, list[pd.DataFrame]] = {}
    oos_start_idx_by_asset: dict[str, list[int]] = {}
    for asset, df in universe.items():
        wins = make_windows(df)
        windows_by_asset[asset] = wins
        n = len(df)
        train_end = int(n * TRAIN_FRACTION)
        remaining = n - train_end
        window_len = remaining // N_WINDOWS
        starts = [train_end + i * window_len for i in range(len(wins))]
        oos_start_idx_by_asset[asset] = starts

    for strat in keeps:
        wins = windows_by_asset.get(strat.asset, [])
        starts = oos_start_idx_by_asset.get(strat.asset, [])
        if not wins:
            logger.warning("no windows for %s (%s)", strat.name, strat.asset)
            continue
        win_results: list[WindowResult] = []
        for i, (win_df, oos_idx) in enumerate(zip(wins, starts)):
            try:
                wr = evaluate_window(strat, win_df, oos_idx)
                wr.window_idx = i
                win_results.append(wr)
            except Exception:
                logger.exception("eval failed for %s window %d", strat.name, i)

        if not win_results:
            continue

        sharpes = [w.sharpe for w in win_results]
        mean_s = float(np.mean(sharpes))
        std_s = float(np.std(sharpes))
        min_s = float(np.min(sharpes))
        max_s = float(np.max(sharpes))
        pos = sum(1 for s in sharpes if s > 0)
        worst_dd = float(np.max([w.max_dd for w in win_results]))
        total_tr = int(sum(w.trades for w in win_results))
        cls = classify(mean_s, min_s, max_s, pos)

        results.append(StrategyStability(
            name=strat.name, asset=strat.asset,
            mean_sharpe=mean_s, std_sharpe=std_s,
            min_sharpe=min_s, max_sharpe=max_s,
            positive_windows=pos, worst_dd=worst_dd,
            total_trades=total_tr,
            classification=cls, windows=win_results,
        ))
    return results


def write_report(results: list[StrategyStability]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_windows": N_WINDOWS,
        "train_fraction": TRAIN_FRACTION,
        "strategies": [
            {
                **{k: v for k, v in asdict(r).items() if k != "windows"},
                "windows": [asdict(w) for w in r.windows],
            }
            for r in results
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(f"loading KEEPs from {args.report}...")
    keeps = load_keeps(args.report)
    print(f"  {len(keeps)} KEEP strategies")
    print(f"fetching cached universe...")
    universe = fetch_cached_universe()
    print(f"  {len(universe)} assets loaded")
    print(f"running {N_WINDOWS}-window stress test...")
    results = stress_test(keeps, universe)
    write_report(results)

    # Console summary.
    print()
    print(f"{'strategy':<55s}{'class':>14s}{'n+':>4s}{'mean_sh':>9s}"
          f"{'std':>7s}{'min':>7s}{'max':>7s}{'trades':>8s}")
    print("-" * 111)
    for r in sorted(results, key=lambda x: (-x.positive_windows, -x.mean_sharpe)):
        print(f"{r.name[:55]:<55s}{r.classification:>14s}"
              f"{r.positive_windows:>4d}{r.mean_sharpe:>9.2f}"
              f"{r.std_sharpe:>7.2f}{r.min_sharpe:>7.2f}"
              f"{r.max_sharpe:>7.2f}{r.total_trades:>8d}")

    # Summary buckets.
    buckets: dict[str, int] = {}
    for r in results:
        buckets[r.classification] = buckets.get(r.classification, 0) + 1
    print()
    print("classification summary:", buckets)
    print(f"full report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SignalForge Kill-Switch — Live-vs-Backtest Sharpe Monitor
==========================================================

Reads ``fund_data/paper_journal_v20.jsonl``, computes realized PnL per
strategy over a rolling window, compares to the backtested Sharpe from
``fund_data/validation_v20.json``, and writes a disabled list that the
paper trader consults before opening new positions.

Decision rule (institutional-grade, documented):

    For each strategy with >= MIN_TRADES closed paper trades,
    compute live_sharpe over the last N completed trades:

        live_sharpe < 0.25 × backtest_sharpe  → HALVE size (warning)
        live_sharpe < 0.00                    → PAUSE strategy
        live_sharpe >= 0.50 × backtest_sharpe → restore full size

    Strategies with < MIN_TRADES are left alone — we need enough trades
    to reject the null "this is noise".

The output is ``fund_data/strategy_overrides.json`` of the form::

    {
      "sf_tf_breakout_50_long_h24_XRP": {"size_mult": 0.5, "reason": "..."},
      "sf_ll_leadlag_btc_short_t1_p150_SOL": {"size_mult": 0.0, "reason": "..."}
    }

Run via cron / launchd every hour after the paper trader has emitted
its events. Idempotent.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger("kill_switch")

REPORT_PATH = Path("fund_data/validation_v20.json")
JOURNAL_PATH = Path("fund_data/paper_journal_v20.jsonl")
OVERRIDES_PATH = Path("fund_data/strategy_overrides.json")

# How many complete paper trades before kill-switch will act on a
# strategy. Below this, the live sample is too small to reject noise
# (empirically ~12-20 gets you a loose 90% CI on Sharpe).
MIN_TRADES = 15

# Rolling window of trades to use for the live Sharpe estimate. Shorter
# → faster regime-change response, noisier. 30 is a reasonable compromise.
ROLLING_WINDOW = 30

# Annualization factor for intraday Sharpe (1h bars, crypto 24/7).
BARS_PER_YEAR = 24 * 365


@dataclass
class StrategyStats:
    name: str
    n_trades: int
    live_sharpe: float
    backtest_sharpe: float
    size_mult: float
    reason: str


def _load_journal() -> list[dict]:
    if not JOURNAL_PATH.exists():
        return []
    lines = JOURNAL_PATH.read_text().strip().splitlines()
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _load_backtest_sharpes(report_path: Path) -> dict[str, float]:
    data = json.loads(report_path.read_text())
    return {
        row["name"]: float(row.get("oos_sharpe", 0.0))
        for row in data.get("strategies", [])
        if row.get("final_verdict") == "KEEP"
    }


def _live_sharpe(trade_returns: list[float]) -> float:
    """Sharpe from a series of per-trade returns.

    Approximates annualization by scaling with sqrt(trades_per_year).
    For hourly-horizon strategies that close roughly once per day on
    average, trades_per_year ~ 250; we use the more conservative 200.
    """
    if len(trade_returns) < 2:
        return 0.0
    mu = float(np.mean(trade_returns))
    sigma = float(np.std(trade_returns, ddof=1))
    if sigma <= 0:
        return 0.0
    return mu / sigma * np.sqrt(200)


def compute_overrides() -> dict[str, StrategyStats]:
    """Scan the journal, return per-strategy size multipliers."""
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"backtest report missing: {REPORT_PATH}")
    backtest_sharpes = _load_backtest_sharpes(REPORT_PATH)
    journal = _load_journal()
    if not journal:
        logger.info("empty journal — no decisions to make")
        return {}

    # Build trade returns per strategy. A "trade" is one entry + its
    # matching exit. We use the exit event's `pnl` divided by the
    # implied entry notional.
    trades_by_strat: dict[str, list[float]] = {}
    # Track last entry price/qty per (strategy, asset) to compute return
    open_by_key: dict[tuple[str, str], dict] = {}
    for ev in journal:
        strat = ev.get("strategy")
        asset = ev.get("asset")
        if not strat:
            continue
        key = (strat, asset)
        if ev.get("event") == "entry":
            open_by_key[key] = {
                "price": float(ev.get("price", 0.0)),
                "qty": float(ev.get("qty", 0.0)),
            }
        elif ev.get("event") == "exit":
            ref = open_by_key.pop(key, None)
            if ref is None or ref["qty"] <= 0 or ref["price"] <= 0:
                continue
            notional = ref["qty"] * ref["price"]
            if notional <= 0:
                continue
            trade_ret = float(ev.get("pnl", 0.0)) / notional
            trades_by_strat.setdefault(strat, []).append(trade_ret)

    overrides: dict[str, StrategyStats] = {}
    for strat, rets in trades_by_strat.items():
        n = len(rets)
        bt = backtest_sharpes.get(strat, 0.0)
        if n < MIN_TRADES:
            # Not enough evidence; leave strategy alone.
            continue
        recent = rets[-ROLLING_WINDOW:]
        live_sh = _live_sharpe(recent)

        # Decision rule.
        if live_sh < 0.0:
            mult = 0.0
            reason = f"paused: live_sharpe={live_sh:.2f} < 0"
        elif bt > 0 and live_sh < 0.25 * bt:
            mult = 0.5
            reason = (
                f"halved: live_sharpe={live_sh:.2f} < 0.25×backtest "
                f"({0.25*bt:.2f})"
            )
        elif bt > 0 and live_sh >= 0.50 * bt:
            mult = 1.0
            reason = f"ok: live_sharpe={live_sh:.2f} >= 0.50×backtest ({0.50*bt:.2f})"
        else:
            # Between 0 and 0.25×backtest — borderline, keep at half.
            mult = 0.5
            reason = (
                f"borderline: live_sharpe={live_sh:.2f} in "
                f"[0, 0.25×backtest={0.25*bt:.2f})"
            )

        overrides[strat] = StrategyStats(
            name=strat,
            n_trades=n,
            live_sharpe=live_sh,
            backtest_sharpe=bt,
            size_mult=mult,
            reason=reason,
        )
    return overrides


def write_overrides(overrides: dict[str, StrategyStats]) -> None:
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "strategies": {
            name: {
                "size_mult": s.size_mult,
                "live_sharpe": round(s.live_sharpe, 4),
                "backtest_sharpe": round(s.backtest_sharpe, 4),
                "n_trades": s.n_trades,
                "reason": s.reason,
            }
            for name, s in overrides.items()
        },
    }
    OVERRIDES_PATH.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    overrides = compute_overrides()
    write_overrides(overrides)

    if not overrides:
        print("no strategies have enough trades yet for kill-switch action")
        return

    print(f"updated {OVERRIDES_PATH}")
    print(f"{'strategy':<55s}{'n':>5s}{'live_sh':>9s}{'bt_sh':>9s}{'mult':>7s}  reason")
    for s in sorted(overrides.values(), key=lambda x: x.size_mult):
        print(f"{s.name[:55]:<55s}{s.n_trades:>5d}"
              f"{s.live_sharpe:>9.2f}{s.backtest_sharpe:>9.2f}"
              f"{s.size_mult:>7.2f}  {s.reason}")


if __name__ == "__main__":
    main()

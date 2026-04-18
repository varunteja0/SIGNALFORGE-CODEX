#!/usr/bin/env python3
"""
SignalForge Paper Trader — v20 multi-factor portfolio
======================================================

Consumes the KEEP strategies from a validation report (default:
fund_data/validation_v20.json), reconstructs each one as a
``DeployedStrategy``, fetches live-ish OHLCV via the public ccxt
pipeline, generates signals, and executes orders through the
:class:`FillModel` — the same fill simulator unit-tested in the
execution module.

Deliberately minimal. One iteration per invocation (use cron /
``--loop`` for hourly runs). Writes to:

    fund_data/paper_journal_v20.jsonl       one JSON line per fill event
    fund_data/paper_positions_v20.json      current open positions
    fund_data/paper_drift_v20.json          live-vs-backtest slippage tracker

Usage
-----
    # One iteration (fetch, decide, fill, persist)
    python scripts/paper_trade_v20.py

    # Single iteration with a specific report / capital
    python scripts/paper_trade_v20.py --report fund_data/validation_v20.json \\
        --capital 10000

    # Dry-run replay against the most recent OOS bar in the validation
    # dataset — for smoke-testing the whole pipeline offline.
    python scripts/paper_trade_v20.py --dry-run

    # Continuous hourly loop (aligns to 00:00:05 of each hour)
    python scripts/paper_trade_v20.py --loop

The "drift tracker" diffs the FillModel's realized fill price against
the next-bar open the backtester assumed, and accumulates a rolling
mean absolute slippage in basis points. This is the metric used to
decide whether to flip to live capital.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Make sibling packages importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.fetcher import DataFetcher
from src.execution.fill_model import FillModel, Order, OrderKind
from src.factory.deployer import DeployedStrategy, set_universe

logger = logging.getLogger("paper_trade_v20")

# Paths — single source of truth.
REPORT_PATH = Path("fund_data/validation_v20.json")
JOURNAL_PATH = Path("fund_data/paper_journal_v20.jsonl")
POSITIONS_PATH = Path("fund_data/paper_positions_v20.json")
DRIFT_PATH = Path("fund_data/paper_drift_v20.json")
OVERRIDES_PATH = Path("fund_data/strategy_overrides.json")

# Universe that needs to be in the registry for xsec/leadlag signals to
# generate on any single asset. Must match what the harness trained on.
UNIVERSE_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "SOL/USDT",
]

# How many bars of recent history to request per asset. Needs to cover
# the longest lookback used by any deployed signal (MA_200, xsec lb336,
# regime pct_change(200)) plus a safety margin.
HISTORY_BARS = 500


# ─── Data classes ───────────────────────────────────────────────────────

@dataclass
class Position:
    """An open paper position."""

    strategy: str
    asset: str
    direction: int
    qty: float
    entry_price: float
    entry_ts: str
    hold_bars_remaining: int
    stop_loss_price: float
    take_profit_price: float


@dataclass
class JournalEntry:
    """A single event written to the journal."""

    ts: str
    event: str              # "entry" | "exit" | "skip"
    strategy: str
    asset: str
    direction: int
    qty: float
    price: float
    fee: float
    # Backtester would have filled at bar open — store the drift.
    reference_price: float = 0.0
    slippage_bps: float = 0.0
    reason: str = ""
    pnl: float = 0.0


# ─── Persistence ────────────────────────────────────────────────────────

def _load_positions() -> list[Position]:
    if not POSITIONS_PATH.exists():
        return []
    try:
        data = json.loads(POSITIONS_PATH.read_text())
        return [Position(**d) for d in data]
    except Exception as exc:  # pragma: no cover - user-facing recovery
        logger.warning("failed to load positions: %s", exc)
        return []


def _save_positions(positions: list[Position]) -> None:
    POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    POSITIONS_PATH.write_text(json.dumps([asdict(p) for p in positions], indent=2))


def _append_journal(entries: list[JournalEntry]) -> None:
    if not entries:
        return
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL_PATH.open("a") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry)) + "\n")


def _update_drift(entries: list[JournalEntry]) -> None:
    """Accumulate live-vs-backtest slippage as a rolling mean."""
    fills = [e for e in entries if e.event in ("entry", "exit") and e.reference_price > 0]
    if not fills:
        return
    prior = {}
    if DRIFT_PATH.exists():
        try:
            prior = json.loads(DRIFT_PATH.read_text())
        except Exception:
            prior = {}
    n_prior = int(prior.get("n_fills", 0))
    mean_prior = float(prior.get("mean_abs_slippage_bps", 0.0))
    max_prior = float(prior.get("max_abs_slippage_bps", 0.0))

    new_slips = [abs(e.slippage_bps) for e in fills]
    n_new = n_prior + len(new_slips)
    mean_new = (mean_prior * n_prior + sum(new_slips)) / max(n_new, 1)
    max_new = max(max_prior, max(new_slips))

    DRIFT_PATH.write_text(json.dumps({
        "n_fills": n_new,
        "mean_abs_slippage_bps": round(mean_new, 4),
        "max_abs_slippage_bps": round(max_new, 4),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


# ─── Strategy loading ───────────────────────────────────────────────────

def load_keeps(report_path: Path) -> list[DeployedStrategy]:
    """Parse the validation JSON and reconstruct KEEP strategies.

    We reuse DeployedStrategy directly — its ``generate_signals`` is
    the single source of truth shared with the harness. The dataclass
    is permissive enough to reconstruct from the nested strategy rows
    in the report with a small adapter.
    """
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
            # signal_name isn't in the harness report — derive it from
            # the deploy naming convention: sf_{family_prefix}_{rest}
            # maps back to ``{rest}`` for non-ensemble families. For
            # ensembles the row carries the full base name already.
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


def _recover_signal_name(strategy_name: str) -> str:
    """Invert the deployer naming convention.

    Deploy layer builds names of the form:
        sf_{family}_{signal_name}_{asset_base}
    where ``family`` is one of tf/sm/mr/xs/ll and ``asset_base`` is
    the first segment of the asset (e.g. BTC, ETH). We need to strip
    the ``sf_{family}_`` prefix and the trailing ``_{asset_base}``.
    """
    parts = strategy_name.split("_")
    if len(parts) < 4 or parts[0] != "sf":
        return strategy_name
    # Drop the leading "sf_{family}" and the trailing asset base.
    inner = parts[2:-1]
    return "_".join(inner)


# ─── Market data ────────────────────────────────────────────────────────

def fetch_universe(days: int = 60) -> dict[str, pd.DataFrame]:
    """Fetch recent 1h OHLCV for the full universe.

    Uses the same cached ccxt pipeline as the validation harness, so
    the data semantics match exactly.
    """
    fetcher = DataFetcher()
    out: dict[str, pd.DataFrame] = {}
    for sym in UNIVERSE_SYMBOLS:
        try:
            df = fetcher.fetch(sym, timeframe="1h", days=days)
            if df is None or df.empty:
                logger.warning("no data for %s", sym)
                continue
            # Ensure we carry enough history for the longest lookback
            # (xsec 336-bar, MA 200-bar).
            if len(df) < HISTORY_BARS:
                logger.warning("%s only has %d bars (<%d)", sym, len(df), HISTORY_BARS)
            out[sym] = df
        except Exception as exc:
            logger.error("fetch failed for %s: %s", sym, exc)
    return out


# ─── Core loop ──────────────────────────────────────────────────────────

def _check_exits(
    positions: list[Position],
    bar_by_asset: dict[str, pd.Series],
    fill_model: FillModel,
    ts: pd.Timestamp,
) -> tuple[list[Position], list[JournalEntry]]:
    """Advance each open position by one bar; exit on SL/TP/time."""
    still_open: list[Position] = []
    journal: list[JournalEntry] = []
    for pos in positions:
        bar = bar_by_asset.get(pos.asset)
        if bar is None:
            still_open.append(pos)
            continue

        high = float(bar["high"])
        low = float(bar["low"])

        exit_reason = ""
        if pos.direction > 0:
            if low <= pos.stop_loss_price:
                exit_reason = "stop"
            elif high >= pos.take_profit_price:
                exit_reason = "target"
        else:
            if high >= pos.stop_loss_price:
                exit_reason = "stop"
            elif low <= pos.take_profit_price:
                exit_reason = "target"

        pos.hold_bars_remaining -= 1
        if not exit_reason and pos.hold_bars_remaining <= 0:
            exit_reason = "time"

        if not exit_reason:
            still_open.append(pos)
            continue

        order = Order(
            ts=ts,
            side=-pos.direction,  # closing order opposite to position
            qty=pos.qty,
            kind=OrderKind.MARKET,
        )
        result = fill_model.fill_market(order, bar)
        exec_price = result.avg_price if result.filled_qty > 0 else float(bar["open"])
        reference = float(bar["open"])
        # pnl in $: qty × (exit - entry) × direction − fees
        pnl = pos.qty * (exec_price - pos.entry_price) * pos.direction - result.total_fee
        journal.append(JournalEntry(
            ts=str(ts),
            event="exit",
            strategy=pos.strategy,
            asset=pos.asset,
            direction=pos.direction,
            qty=result.filled_qty,
            price=exec_price,
            fee=result.total_fee,
            reference_price=reference,
            slippage_bps=(exec_price - reference) / reference * 1e4 if reference else 0.0,
            reason=exit_reason,
            pnl=pnl,
        ))
    return still_open, journal


def _load_overrides() -> dict[str, float]:
    """Load per-strategy size multipliers written by the kill switch.

    Returns a dict mapping strategy name → size_mult in [0.0, 1.0].
    Missing strategies default to 1.0 (full size). An absent overrides
    file means no kill-switch is active yet — treat everything as 1.0.
    """
    if not OVERRIDES_PATH.exists():
        return {}
    try:
        data = json.loads(OVERRIDES_PATH.read_text())
        return {
            name: float(payload.get("size_mult", 1.0))
            for name, payload in data.get("strategies", {}).items()
        }
    except Exception as exc:
        logger.warning("failed to load overrides: %s", exc)
        return {}


def _maybe_enter(
    strategies: list[DeployedStrategy],
    data_by_asset: dict[str, pd.DataFrame],
    open_keys: set[tuple[str, str]],
    fill_model: FillModel,
    capital: float,
    ts: pd.Timestamp,
    size_overrides: dict[str, float] | None = None,
) -> tuple[list[Position], list[JournalEntry]]:
    new_positions: list[Position] = []
    journal: list[JournalEntry] = []
    size_overrides = size_overrides or {}

    for strat in strategies:
        key = (strat.name, strat.asset)
        if key in open_keys:
            # One position per strategy at a time — canonical.
            continue
        # Kill-switch override.
        size_mult = size_overrides.get(strat.name, 1.0)
        if size_mult <= 0.0:
            journal.append(JournalEntry(
                ts=str(ts), event="skip", strategy=strat.name, asset=strat.asset,
                direction=strat.direction, qty=0.0, price=0.0, fee=0.0,
                reason="kill_switch_paused",
            ))
            continue
        df = data_by_asset.get(strat.asset)
        if df is None or df.empty:
            continue

        # Signal computed up to the latest closed bar; entry executes
        # on the NEXT bar's open — but in live we only have one bar at
        # a time, so we treat the latest bar as the entry bar. That's
        # a 1-bar look-ahead bias vs. backtest; we explicitly track the
        # drift between ``bar.open`` and the realized fill price.
        try:
            signals = strat.generate_signals(df)
        except Exception as exc:
            logger.exception("signal gen failed for %s: %s", strat.name, exc)
            continue
        if signals.empty or signals.iloc[-1] == 0:
            continue
        if int(signals.iloc[-1]) != strat.direction:
            continue

        bar = df.iloc[-1]
        reference = float(bar["open"])
        notional = capital * float(strat.position_size_pct) * size_mult
        qty = notional / reference if reference > 0 else 0.0
        if qty <= 0:
            continue

        order = Order(ts=ts, side=strat.direction, qty=qty, kind=OrderKind.MARKET)
        result = fill_model.fill_market(order, bar)
        if result.filled_qty <= 0:
            journal.append(JournalEntry(
                ts=str(ts), event="skip", strategy=strat.name, asset=strat.asset,
                direction=strat.direction, qty=0.0, price=0.0, fee=0.0,
                reason="no_liquidity",
            ))
            continue

        exec_price = result.avg_price
        # ATR-based SL/TP — approximate ATR off the last 14 bars.
        tr = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.tail(14).mean())
        sl = exec_price - strat.direction * strat.stop_loss_atr * atr
        tp = exec_price + strat.direction * strat.take_profit_atr * atr

        new_positions.append(Position(
            strategy=strat.name,
            asset=strat.asset,
            direction=strat.direction,
            qty=result.filled_qty,
            entry_price=exec_price,
            entry_ts=str(ts),
            hold_bars_remaining=strat.hold_bars,
            stop_loss_price=sl,
            take_profit_price=tp,
        ))
        journal.append(JournalEntry(
            ts=str(ts), event="entry", strategy=strat.name, asset=strat.asset,
            direction=strat.direction, qty=result.filled_qty, price=exec_price,
            fee=result.total_fee, reference_price=reference,
            slippage_bps=(exec_price - reference) / reference * 1e4 if reference else 0.0,
            reason="signal",
        ))
        open_keys.add(key)

    return new_positions, journal


def run_iteration(
    report_path: Path = REPORT_PATH,
    capital: float = 10_000.0,
    dry_run: bool = False,
    venue: str = "binance",
) -> dict:
    """Execute one paper-trading iteration and persist state."""
    strategies = load_keeps(report_path)
    if not strategies:
        logger.warning("no KEEP strategies in %s", report_path)
        return {"kept": 0, "entries": 0, "exits": 0}

    data_by_asset = fetch_universe(days=max(30, HISTORY_BARS // 24 + 2))
    if not data_by_asset:
        raise RuntimeError("universe fetch returned no assets")
    set_universe(data_by_asset)

    # Use the latest closed bar's timestamp as the decision time.
    latest_ts = max(df.index[-1] for df in data_by_asset.values())
    bar_by_asset = {a: df.loc[df.index == latest_ts].iloc[0]
                    for a, df in data_by_asset.items() if latest_ts in df.index}

    fill_model = FillModel(venue=venue, rng_seed=int(latest_ts.value) & 0xFFFF)

    positions = _load_positions()
    positions, exit_journal = _check_exits(positions, bar_by_asset, fill_model, latest_ts)

    open_keys = {(p.strategy, p.asset) for p in positions}
    size_overrides = _load_overrides()
    new_positions, entry_journal = _maybe_enter(
        strategies, data_by_asset, open_keys, fill_model, capital, latest_ts,
        size_overrides=size_overrides,
    )
    positions.extend(new_positions)

    if not dry_run:
        _save_positions(positions)
        _append_journal(exit_journal + entry_journal)
        _update_drift(exit_journal + entry_journal)

    summary = {
        "ts": str(latest_ts),
        "kept": len(strategies),
        "entries": len(entry_journal),
        "exits": len(exit_journal),
        "open_positions": len(positions),
    }
    logger.info("iteration done: %s", summary)
    return summary


# ─── CLI ────────────────────────────────────────────────────────────────

def _sleep_until_next_hour() -> None:
    now = datetime.now(timezone.utc)
    nxt = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
    time.sleep(max(1.0, (nxt - now).total_seconds()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--venue", type=str, default="binance")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute but don't persist")
    parser.add_argument("--loop", action="store_true",
                        help="run forever, one iteration per hour")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.loop:
        while True:
            try:
                run_iteration(args.report, args.capital, args.dry_run, args.venue)
            except Exception:
                logger.exception("iteration failed; continuing")
            _sleep_until_next_hour()
    else:
        summary = run_iteration(args.report, args.capital, args.dry_run, args.venue)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

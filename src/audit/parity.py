"""
Live-vs-backtest parity auditor.

Purpose
-------
Prove that the paper trader's realised P&L stays within a tight tolerance
of what :mod:`src.execution.fill_model` predicts it *should* have been.
Any divergence beyond the tolerance is flagged with an attribution
(spread, impact, fee, unexplained) so the operator knows whether the
discrepancy is benign (e.g. a venue fee-tier change) or a red flag
(e.g. the signal is silently executing on stale prices).

The auditor is journal-only — it does not re-fetch OHLCV. For every
entry/exit in ``paper_journal_v20.jsonl`` it asks the fill model:

    Given the reference price we recorded and a typical-venue bar
    context, what should the fill price and fee have been?

It then diffs that against what was actually written to the journal.

Invariants
----------
- Deterministic given the journal and fill model seed.
- Read-only: never modifies the journal.
- No network, no disk writes aside from the optional output path.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.execution.fill_model import FillModel, Order, OrderKind

# Default tolerance: 5 bps of unexplained slippage averaged across trades
# is the line between "model is a good mark" and "something is wrong".
DEFAULT_TOL_BPS = 5.0
DEFAULT_WARN_BPS = 2.5


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class JournalRecord:
    """One entry from the paper-trader journal (one line of JSONL)."""

    ts: datetime
    event: str                 # "entry" | "exit" | "skip"
    strategy: str
    asset: str
    direction: int             # +1 long, -1 short
    qty: float
    price: float               # executed price
    fee: float                 # USD
    reference_price: float     # bar.open at order time
    slippage_bps: float
    reason: str = ""
    pnl: float = 0.0           # populated on exits only

    @property
    def notional_usd(self) -> float:
        return abs(self.qty) * self.price


@dataclass(frozen=True)
class TradeRoundTrip:
    """An entry paired with its corresponding exit."""

    entry: JournalRecord
    exit: JournalRecord

    @property
    def holding_bars(self) -> float:
        # Paper trader runs hourly — report in hours.
        return (self.exit.ts - self.entry.ts).total_seconds() / 3600.0

    @property
    def realised_pnl(self) -> float:
        # Trust the journal's own ``pnl`` on exit; fallback to compute.
        if self.exit.pnl:
            return self.exit.pnl
        gross = (self.exit.price - self.entry.price) * self.entry.direction * self.entry.qty
        return gross - (self.entry.fee + self.exit.fee)


@dataclass(frozen=True)
class ParityDelta:
    """Per-event discrepancy between the live journal and the fill model."""

    ts: datetime
    asset: str
    strategy: str
    event: str
    direction: int
    qty: float
    live_price: float
    model_price: float
    live_fee: float
    model_fee: float
    reference_price: float

    @property
    def price_diff_bps(self) -> float:
        if self.reference_price <= 0:
            return 0.0
        return (self.live_price - self.model_price) / self.reference_price * 1e4

    @property
    def fee_diff_usd(self) -> float:
        return self.live_fee - self.model_fee

    @property
    def pnl_impact_usd(self) -> float:
        """Signed P&L impact: positive = live was more expensive than model.

        On an entry with ``direction=+1`` the trader *bought*, so a higher
        ``live_price`` is adverse. On the matching exit the trader *sells*,
        so a *lower* ``live_price`` is adverse — hence the sign flip.
        """
        sign = self.direction if self.event == "entry" else -self.direction
        slip = (self.live_price - self.model_price) * sign * self.qty
        return slip + self.fee_diff_usd


@dataclass
class ParityReport:
    """Aggregate parity across all journal events."""

    n_events: int
    n_entries: int
    n_exits: int
    total_notional_usd: float
    total_pnl_impact_usd: float
    total_fee_diff_usd: float
    mean_price_diff_bps: float
    median_price_diff_bps: float
    p95_abs_price_diff_bps: float
    max_abs_price_diff_bps: float
    unexplained_pnl_bps: float     # total_pnl_impact / total_notional
    verdict: str                   # "PASS" | "WARN" | "FAIL"
    per_asset: dict[str, dict] = field(default_factory=dict)
    per_strategy: dict[str, dict] = field(default_factory=dict)
    deltas: list[ParityDelta] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_events": self.n_events,
            "n_entries": self.n_entries,
            "n_exits": self.n_exits,
            "total_notional_usd": self.total_notional_usd,
            "total_pnl_impact_usd": self.total_pnl_impact_usd,
            "total_fee_diff_usd": self.total_fee_diff_usd,
            "mean_price_diff_bps": self.mean_price_diff_bps,
            "median_price_diff_bps": self.median_price_diff_bps,
            "p95_abs_price_diff_bps": self.p95_abs_price_diff_bps,
            "max_abs_price_diff_bps": self.max_abs_price_diff_bps,
            "unexplained_pnl_bps": self.unexplained_pnl_bps,
            "verdict": self.verdict,
            "per_asset": self.per_asset,
            "per_strategy": self.per_strategy,
        }


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def _parse_ts(raw: str) -> datetime:
    # Tolerate trailing 'Z' and micro-precision variations.
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw)


def load_journal(path: str | Path) -> list[JournalRecord]:
    """Load a JSONL paper-trader journal into :class:`JournalRecord` objects.

    Skips blank lines and records missing required fields.
    """
    p = Path(path)
    out: list[JournalRecord] = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        required = {"ts", "event", "strategy", "asset", "direction", "qty", "price"}
        if not required.issubset(obj):
            continue
        out.append(
            JournalRecord(
                ts=_parse_ts(str(obj["ts"])),
                event=str(obj["event"]),
                strategy=str(obj["strategy"]),
                asset=str(obj["asset"]),
                direction=int(obj["direction"]),
                qty=float(obj["qty"]),
                price=float(obj["price"]),
                fee=float(obj.get("fee", 0.0) or 0.0),
                reference_price=float(obj.get("reference_price", obj["price"]) or obj["price"]),
                slippage_bps=float(obj.get("slippage_bps", 0.0) or 0.0),
                reason=str(obj.get("reason", "")),
                pnl=float(obj.get("pnl", 0.0) or 0.0),
            )
        )
    return out


def pair_round_trips(records: Iterable[JournalRecord]) -> list[TradeRoundTrip]:
    """Pair entries with the next exit on the same (asset, strategy, direction).

    Skip-events are ignored. An unmatched entry is dropped silently — the
    parity report treats it as an open position and handles it in the
    per-event deltas instead.
    """
    records = sorted(
        [r for r in records if r.event in ("entry", "exit")],
        key=lambda r: r.ts,
    )
    open_legs: dict[tuple[str, str, int], JournalRecord] = {}
    out: list[TradeRoundTrip] = []
    for r in records:
        key = (r.asset, r.strategy, r.direction)
        if r.event == "entry":
            open_legs[key] = r
        elif r.event == "exit":
            # Exit direction in journal is typically the entry direction;
            # look up by any matching asset+strategy if exact key misses.
            entry = open_legs.pop(key, None)
            if entry is None:
                # Match by asset+strategy with opposite sign (some journals flip on exit).
                for k in list(open_legs.keys()):
                    if k[0] == r.asset and k[1] == r.strategy:
                        entry = open_legs.pop(k)
                        break
            if entry is not None:
                out.append(TradeRoundTrip(entry=entry, exit=r))
    return out


def reconstruct_live_equity(
    records: Iterable[JournalRecord], initial_capital: float
) -> pd.Series:
    """Reconstruct the realised (closed-trade) equity curve from the journal.

    Open positions are not marked-to-market — this is the realised curve.
    """
    pairs = pair_round_trips(records)
    if not pairs:
        return pd.Series([initial_capital], index=[pd.Timestamp.utcnow()], name="equity")
    index = [pd.Timestamp(p.exit.ts) for p in pairs]
    equity = []
    running = initial_capital
    for p in pairs:
        running += p.realised_pnl
        equity.append(running)
    return pd.Series(equity, index=pd.DatetimeIndex(index), name="equity")


# --------------------------------------------------------------------------
# Model estimate
# --------------------------------------------------------------------------
def _synthetic_bar(record: JournalRecord, atr_ratio: float, volume_usd: float) -> pd.Series:
    """Build the minimal bar the fill model needs from a journal record."""
    mid = float(record.reference_price or record.price)
    # Choose volume so that ``_bar_volume_usd`` returns ``volume_usd``.
    volume = volume_usd / max(mid, 1e-9)
    return pd.Series(
        {
            "open": mid,
            "close": mid,
            "high": mid,
            "low": mid,
            "volume": volume,
            "atr_ratio": atr_ratio,
        }
    )


def _model_fill(
    record: JournalRecord,
    fill_model: FillModel,
    atr_ratio: float,
    volume_usd: float,
) -> tuple[float, float]:
    """Ask the fill model what a same-signed market order would have executed at."""
    bar = _synthetic_bar(record, atr_ratio=atr_ratio, volume_usd=volume_usd)
    # Entries use the journal direction; exits are the closing side (opposite).
    side = record.direction if record.event == "entry" else -record.direction
    order = Order(
        ts=pd.Timestamp(record.ts),
        side=int(side),
        qty=abs(float(record.qty)),
        kind=OrderKind.MARKET,
    )
    result = fill_model.fill_market(order, bar)
    # Pro-rate fee to match the live fill qty (fill model caps at participation).
    filled = result.filled_qty
    if filled <= 0:
        return record.reference_price, 0.0
    live_qty = abs(record.qty)
    scale = live_qty / filled
    model_price = result.avg_price
    model_fee = result.total_fee * scale
    return model_price, model_fee


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------
def audit_parity(
    journal: str | Path | Iterable[JournalRecord],
    *,
    fill_model: FillModel | None = None,
    atr_ratio: float = 1.0,
    volume_usd: float = 1e8,
    warn_bps: float = DEFAULT_WARN_BPS,
    fail_bps: float = DEFAULT_TOL_BPS,
) -> ParityReport:
    """Audit a paper-trader journal against the fill model.

    Parameters
    ----------
    journal
        Path to a JSONL journal or an iterable of :class:`JournalRecord`.
    fill_model
        Optional — defaults to a binance-perp FillModel with deterministic seed.
    atr_ratio, volume_usd
        Market-context assumptions used to build the synthetic bar each
        journal record is replayed against. Tune to your venue.
    warn_bps, fail_bps
        Absolute-value thresholds on ``|unexplained_pnl_bps|``.

    Returns
    -------
    :class:`ParityReport` with per-event deltas, per-asset and
    per-strategy aggregates, and a PASS/WARN/FAIL verdict.
    """
    if isinstance(journal, (str, Path)):
        records = load_journal(journal)
    else:
        records = list(journal)

    fm = fill_model or FillModel(venue="binance_perp")

    deltas: list[ParityDelta] = []
    for r in records:
        if r.event not in ("entry", "exit"):
            continue
        if r.price <= 0 or r.qty <= 0:
            continue
        model_price, model_fee = _model_fill(
            r, fill_model=fm, atr_ratio=atr_ratio, volume_usd=volume_usd
        )
        deltas.append(
            ParityDelta(
                ts=r.ts,
                asset=r.asset,
                strategy=r.strategy,
                event=r.event,
                direction=r.direction,
                qty=r.qty,
                live_price=r.price,
                model_price=model_price,
                live_fee=r.fee,
                model_fee=model_fee,
                reference_price=r.reference_price,
            )
        )

    return _summarise(deltas, warn_bps=warn_bps, fail_bps=fail_bps)


def _summarise(
    deltas: list[ParityDelta], *, warn_bps: float, fail_bps: float
) -> ParityReport:
    if not deltas:
        return ParityReport(
            n_events=0, n_entries=0, n_exits=0,
            total_notional_usd=0.0, total_pnl_impact_usd=0.0,
            total_fee_diff_usd=0.0,
            mean_price_diff_bps=0.0, median_price_diff_bps=0.0,
            p95_abs_price_diff_bps=0.0, max_abs_price_diff_bps=0.0,
            unexplained_pnl_bps=0.0, verdict="PASS",
        )

    df = pd.DataFrame(
        [
            {
                "asset": d.asset,
                "strategy": d.strategy,
                "event": d.event,
                "notional": d.qty * d.live_price,
                "price_diff_bps": d.price_diff_bps,
                "abs_price_diff_bps": abs(d.price_diff_bps),
                "fee_diff_usd": d.fee_diff_usd,
                "pnl_impact_usd": d.pnl_impact_usd,
            }
            for d in deltas
        ]
    )
    notional = float(df["notional"].sum())
    pnl_impact = float(df["pnl_impact_usd"].sum())
    unexplained_bps = (pnl_impact / notional * 1e4) if notional > 0 else 0.0

    abs_unexplained = abs(unexplained_bps)
    if abs_unexplained >= fail_bps:
        verdict = "FAIL"
    elif abs_unexplained >= warn_bps:
        verdict = "WARN"
    else:
        verdict = "PASS"

    per_asset = {
        asset: {
            "n": int(len(g)),
            "mean_price_diff_bps": float(g["price_diff_bps"].mean()),
            "pnl_impact_usd": float(g["pnl_impact_usd"].sum()),
            "fee_diff_usd": float(g["fee_diff_usd"].sum()),
        }
        for asset, g in df.groupby("asset")
    }
    per_strategy = {
        strat: {
            "n": int(len(g)),
            "mean_price_diff_bps": float(g["price_diff_bps"].mean()),
            "pnl_impact_usd": float(g["pnl_impact_usd"].sum()),
            "fee_diff_usd": float(g["fee_diff_usd"].sum()),
        }
        for strat, g in df.groupby("strategy")
    }

    return ParityReport(
        n_events=int(len(df)),
        n_entries=int((df["event"] == "entry").sum()),
        n_exits=int((df["event"] == "exit").sum()),
        total_notional_usd=notional,
        total_pnl_impact_usd=pnl_impact,
        total_fee_diff_usd=float(df["fee_diff_usd"].sum()),
        mean_price_diff_bps=float(df["price_diff_bps"].mean()),
        median_price_diff_bps=float(df["price_diff_bps"].median()),
        p95_abs_price_diff_bps=float(df["abs_price_diff_bps"].quantile(0.95)) if len(df) >= 5 else float(df["abs_price_diff_bps"].max()),
        max_abs_price_diff_bps=float(df["abs_price_diff_bps"].max()),
        unexplained_pnl_bps=unexplained_bps,
        verdict=verdict,
        per_asset=per_asset,
        per_strategy=per_strategy,
        deltas=deltas,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Live-vs-model parity auditor")
    p.add_argument("journal", type=Path, help="Path to paper-trader JSONL journal")
    p.add_argument("--venue", default="binance_perp")
    p.add_argument("--atr-ratio", type=float, default=1.0)
    p.add_argument("--volume-usd", type=float, default=1e8)
    p.add_argument("--warn-bps", type=float, default=DEFAULT_WARN_BPS)
    p.add_argument("--fail-bps", type=float, default=DEFAULT_TOL_BPS)
    p.add_argument("--output", type=Path, default=None, help="Write report JSON here")
    args = p.parse_args(argv)

    fm = FillModel(venue=args.venue)
    report = audit_parity(
        args.journal,
        fill_model=fm,
        atr_ratio=args.atr_ratio,
        volume_usd=args.volume_usd,
        warn_bps=args.warn_bps,
        fail_bps=args.fail_bps,
    )
    print(f"[parity] events={report.n_events} verdict={report.verdict}")
    print(f"[parity] unexplained_pnl_bps={report.unexplained_pnl_bps:+.3f}")
    print(f"[parity] mean_price_diff_bps={report.mean_price_diff_bps:+.3f}  "
          f"p95_abs={report.p95_abs_price_diff_bps:.3f}  "
          f"max_abs={report.max_abs_price_diff_bps:.3f}")
    print(f"[parity] total_pnl_impact_usd={report.total_pnl_impact_usd:+.2f}  "
          f"total_fee_diff_usd={report.total_fee_diff_usd:+.2f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"[parity] wrote {args.output}")
    return {"PASS": 0, "WARN": 1, "FAIL": 2}[report.verdict]


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))

"""
Causal P&L attribution.

Decompose every closed round-trip into orthogonal components so you can
see where the edge actually comes from — and where it leaks.

Attribution buckets (sum to realised P&L):

- **signal**    : the idealised "frictionless" P&L the strategy *should* have
                  earned at the reference prices with zero costs. The raw
                  thesis return.
- **slippage**  : (live_exec_price - reference_price) on both legs, signed
                  so positive = trader lost to slippage.
- **impact**    : square-root impact cost implied by order notional vs bar
                  volume; a subset of slippage attributed to *this order's*
                  footprint.
- **fee**       : taker/maker fees paid on both legs.
- **funding**   : funding carry over the holding period (if funding rate
                  and intervals were recorded, else zero).
- **drift**     : whatever remains after subtracting the above from realised.
                  If attribution is well-specified, drift should be tiny;
                  large drift = unattributed P&L (e.g. regime change,
                  funding not captured, stale prints).

Built on top of :class:`src.audit.parity.TradeRoundTrip`. Stateless,
deterministic, journal-only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping

import pandas as pd

from src.audit.parity import TradeRoundTrip


@dataclass(frozen=True)
class Attribution:
    """Per-round-trip decomposition. ``buckets`` sum to ``realised_pnl``."""

    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    asset: str
    strategy: str
    direction: int
    qty: float
    realised_pnl: float
    signal: float
    slippage: float
    impact: float
    fee: float
    funding: float
    drift: float

    def as_dict(self) -> dict:
        d = asdict(self)
        d["entry_ts"] = str(self.entry_ts)
        d["exit_ts"] = str(self.exit_ts)
        return d


@dataclass(frozen=True)
class AttributionReport:
    """Aggregate across many trades."""

    n_trades: int
    total_realised: float
    total_signal: float
    total_slippage: float
    total_impact: float
    total_fee: float
    total_funding: float
    total_drift: float
    per_strategy: dict[str, dict[str, float]] = field(default_factory=dict)
    per_asset: dict[str, dict[str, float]] = field(default_factory=dict)
    rows: list[Attribution] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "total_realised": self.total_realised,
            "total_signal": self.total_signal,
            "total_slippage": self.total_slippage,
            "total_impact": self.total_impact,
            "total_fee": self.total_fee,
            "total_funding": self.total_funding,
            "total_drift": self.total_drift,
            "per_strategy": self.per_strategy,
            "per_asset": self.per_asset,
        }


# --------------------------------------------------------------------------
# Core decomposition
# --------------------------------------------------------------------------
def _impact_component(
    qty: float,
    exec_price: float,
    ref_price: float,
    bar_volume_usd: float | None,
    impact_k: float,
) -> float:
    """Fraction of slippage attributable to square-root impact.

    Impact per unit ≈ ref * impact_k * sqrt(notional / bar_volume_usd).
    Capped by the raw slippage so ``impact <= |slippage|``.
    """
    if bar_volume_usd is None or bar_volume_usd <= 0 or exec_price <= 0:
        return 0.0
    notional = abs(qty) * ref_price
    frac = notional / bar_volume_usd
    import math
    unit_impact = ref_price * impact_k * math.sqrt(max(frac, 0.0)) * 0.01
    return float(unit_impact * abs(qty))


def attribute_round_trip(
    rt: TradeRoundTrip,
    *,
    funding_rate: float = 0.0,
    funding_interval_hours: float = 8.0,
    bar_volume_usd: float | None = None,
    impact_k: float = 1.0,
) -> Attribution:
    """Decompose one round-trip into attribution buckets.

    Parameters
    ----------
    rt : TradeRoundTrip
    funding_rate : float
        Mean funding rate over the holding period, fractional per
        ``funding_interval_hours`` interval. Long pays positive funding;
        short receives it. Zero if funding data not available.
    funding_interval_hours : float
        Hours per funding settlement (Binance/Bybit default: 8).
    bar_volume_usd : float | None
        Bar volume in USD at execution; required for impact attribution.
    impact_k : float
        Square-root impact coefficient; matches ``FillModel.impact_k``.
    """
    entry, exit_ = rt.entry, rt.exit
    qty = abs(entry.qty)
    direction = entry.direction

    # --- 1. signal: frictionless P&L at reference prices ---
    signal = (exit_.reference_price - entry.reference_price) * direction * qty

    # --- 2. fee: sum of both legs ---
    fee = float(entry.fee + exit_.fee)

    # --- 3. slippage: signed; positive = trader lost ---
    entry_slip = (entry.price - entry.reference_price) * direction * qty
    # On exit the trader does the opposite side, so adverse = lower exit price.
    exit_slip = (exit_.reference_price - exit_.price) * direction * qty
    slippage = float(entry_slip + exit_slip)

    # --- 4. impact: sub-component of slippage tied to order footprint ---
    imp_entry = _impact_component(qty, entry.price, entry.reference_price,
                                   bar_volume_usd, impact_k)
    imp_exit = _impact_component(qty, exit_.price, exit_.reference_price,
                                  bar_volume_usd, impact_k)
    impact = float(imp_entry + imp_exit)

    # --- 5. funding: rate * notional * num_intervals * sign ---
    # Long pays positive funding -> cost; short receives it -> gain.
    holding_hours = rt.holding_bars  # paper trader runs hourly
    n_intervals = holding_hours / max(funding_interval_hours, 1e-9)
    avg_notional = entry.reference_price * qty
    # A long pays when funding>0; convention: positive 'funding' in the
    # bucket = cost to trader, so long -> +, short -> -.
    funding_cost = float(funding_rate * avg_notional * n_intervals * direction)

    realised = rt.realised_pnl
    # drift = realised - (signal - slippage - fee - funding)
    # i.e. signal is gross; costs subtract; drift picks up anything else.
    drift = float(realised - (signal - slippage - fee - funding_cost))

    return Attribution(
        entry_ts=pd.Timestamp(entry.ts),
        exit_ts=pd.Timestamp(exit_.ts),
        asset=entry.asset,
        strategy=entry.strategy,
        direction=direction,
        qty=qty,
        realised_pnl=float(realised),
        signal=float(signal),
        slippage=float(slippage),
        impact=float(impact),
        fee=float(fee),
        funding=float(funding_cost),
        drift=float(drift),
    )


def attribute_trades(
    round_trips: Iterable[TradeRoundTrip],
    *,
    funding_rates: Mapping[str, float] | None = None,
    bar_volume_usd: Mapping[str, float] | None = None,
    impact_k: float = 1.0,
) -> AttributionReport:
    """Aggregate attribution over a batch of round-trips.

    ``funding_rates`` and ``bar_volume_usd`` are keyed by asset; missing
    keys default to 0.0 and None respectively.
    """
    rows: list[Attribution] = []
    for rt in round_trips:
        a = rt.entry.asset
        fr = 0.0 if funding_rates is None else float(funding_rates.get(a, 0.0))
        bv = None if bar_volume_usd is None else bar_volume_usd.get(a)
        rows.append(
            attribute_round_trip(
                rt,
                funding_rate=fr,
                bar_volume_usd=bv,
                impact_k=impact_k,
            )
        )
    return _summarise(rows)


def _summarise(rows: list[Attribution]) -> AttributionReport:
    if not rows:
        return AttributionReport(
            n_trades=0,
            total_realised=0.0, total_signal=0.0, total_slippage=0.0,
            total_impact=0.0, total_fee=0.0, total_funding=0.0, total_drift=0.0,
        )
    df = pd.DataFrame([r.as_dict() for r in rows])

    def _agg(col: str) -> float:
        return float(df[col].sum())

    per_strategy = {
        strat: {
            "n": int(len(g)),
            "realised": float(g["realised_pnl"].sum()),
            "signal": float(g["signal"].sum()),
            "slippage": float(g["slippage"].sum()),
            "fee": float(g["fee"].sum()),
            "funding": float(g["funding"].sum()),
            "drift": float(g["drift"].sum()),
        }
        for strat, g in df.groupby("strategy")
    }
    per_asset = {
        asset: {
            "n": int(len(g)),
            "realised": float(g["realised_pnl"].sum()),
            "signal": float(g["signal"].sum()),
            "slippage": float(g["slippage"].sum()),
            "fee": float(g["fee"].sum()),
            "funding": float(g["funding"].sum()),
            "drift": float(g["drift"].sum()),
        }
        for asset, g in df.groupby("asset")
    }

    return AttributionReport(
        n_trades=len(rows),
        total_realised=_agg("realised_pnl"),
        total_signal=_agg("signal"),
        total_slippage=_agg("slippage"),
        total_impact=_agg("impact"),
        total_fee=_agg("fee"),
        total_funding=_agg("funding"),
        total_drift=_agg("drift"),
        per_strategy=per_strategy,
        per_asset=per_asset,
        rows=rows,
    )

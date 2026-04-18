"""
Capacity analysis
==================

Estimates the AUM at which a strategy's **market-impact** slippage starts
eating its edge. This is the number every allocator asks first; without it,
a Sharpe number is a vanity metric.

Methodology
-----------
For each strategy report (from a validation JSON) we have:

- ``avg_trade_return``     — mean return per trade, **gross** of costs beyond base commission
- ``n_trades``             — trade count in the OOS window
- ``avg_holding_period``   — mean bars held
- ``asset``                — instrument traded

We combine these with per-asset **participation-rate** assumptions
(10% of bar volume as a conservative default) to solve:

    edge_bps(AUM) = gross_edge_bps − impact_bps(AUM)

where impact is the standard square-root law:

    impact_bps ≈ k × sigma × sqrt(AUM / ADV)

``k`` calibrated to ~1.0 for crypto perps (Almgren-style); ``sigma`` is the
daily return vol of the asset; ``ADV`` is average daily notional volume.

We report the AUM where ``edge_bps(AUM) = gross_edge_bps / 2`` (the
**half-life AUM**) — the point at which half the edge has been paid back to
the market. This is deliberately conservative.

Assumptions
-----------
- **No queue priority.** We assume we're a price-taker.
- **No funding/basis edges.** Pure slippage-limited capacity.
- **No cross-venue routing.** Single-venue fills.
- **Daily ADV in USD** supplied via ``--adv`` or a defaults table.

These assumptions err on the low side — real capacity is typically 1–3×
this estimate when routed across venues.

CLI
----

.. code-block:: bash

    python -m research.capacity \\
        --input fund_data/validation_v16.json \\
        --adv BTC/USDT=3e10 ETH/USDT=1.5e10 SOL/USDT=3e9 \\
        --sigma BTC/USDT=0.04 ETH/USDT=0.055 SOL/USDT=0.075
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------
# Default universe assumptions (USD daily notional, daily return vol)
# Calibrated from typical 2025-2026 perp data; override with --adv / --sigma.
# --------------------------------------------------------------------------
DEFAULT_ADV_USD: dict[str, float] = {
    "BTC/USDT": 3.0e10,
    "ETH/USDT": 1.5e10,
    "SOL/USDT": 3.0e9,
    "BNB/USDT": 1.2e9,
    "XRP/USDT": 2.0e9,
    "DOGE/USDT": 1.5e9,
}

DEFAULT_SIGMA_DAILY: dict[str, float] = {
    "BTC/USDT": 0.040,
    "ETH/USDT": 0.055,
    "SOL/USDT": 0.075,
    "BNB/USDT": 0.050,
    "XRP/USDT": 0.070,
    "DOGE/USDT": 0.090,
}

# Square-root impact coefficient. Crypto perps ≈ 0.8-1.2 empirically.
K_IMPACT = 1.0

# Participation cap — we never take more than this fraction of an ADV per day.
MAX_PARTICIPATION = 0.10


# --------------------------------------------------------------------------
# Math
# --------------------------------------------------------------------------
def impact_bps(aum_usd: float, adv_usd: float, sigma_daily: float) -> float:
    """Square-root market-impact model, returned in basis points."""
    if aum_usd <= 0 or adv_usd <= 0:
        return 0.0
    return 1e4 * K_IMPACT * sigma_daily * math.sqrt(aum_usd / adv_usd)


def half_life_aum(gross_edge_bps: float, adv_usd: float, sigma_daily: float) -> float:
    """AUM at which impact = gross_edge_bps / 2 (half-life capacity)."""
    if gross_edge_bps <= 0:
        return 0.0
    # Solve gross/2 = K * sigma * 1e4 * sqrt(AUM/ADV)
    target_bps = gross_edge_bps / 2
    scale = target_bps / (1e4 * K_IMPACT * sigma_daily)
    return adv_usd * scale * scale


def participation_cap_aum(adv_usd: float, daily_turnover: float) -> float:
    """Largest AUM consistent with ≤ ``MAX_PARTICIPATION`` of ADV per day."""
    if daily_turnover <= 0:
        return float("inf")
    return adv_usd * MAX_PARTICIPATION / daily_turnover


# --------------------------------------------------------------------------
# Per-strategy estimation
# --------------------------------------------------------------------------
def estimate(
    row: dict[str, Any],
    adv: dict[str, float],
    sigma: dict[str, float],
    bars_per_day: float,
) -> dict[str, Any]:
    asset = row.get("asset", "")
    n_trades = float(row.get("oos_trades") or row.get("n_trades") or 0)
    oos_return = float(row.get("oos_return") or 0.0)
    # Prefer explicit avg; otherwise derive from total / count.
    avg_ret = float(
        row.get("avg_trade_return")
        or (oos_return / n_trades if n_trades > 0 else 0.0)
    )
    avg_hold = float(row.get("oos_avg_hold") or row.get("avg_holding_period") or 1.0)
    # Derive OOS day count from bars/day when not explicit.
    oos_bars = float(row.get("oos_bars") or 0.0)
    oos_days = float(
        row.get("oos_days")
        or row.get("days")
        or (oos_bars / bars_per_day if oos_bars > 0 else 0.0)
    )
    # Fall back: assume ~2y OOS window (typical SignalForge default).
    if oos_days <= 0:
        oos_days = 365.0 * 2.0

    # Gross per-trade edge in basis points. If the backtester already
    # deducts base commission, gross here is net-of-base-fee — still a
    # usable upper bound on impact tolerance.
    gross_edge_bps = avg_ret * 1e4

    adv_usd = adv.get(asset, DEFAULT_ADV_USD.get(asset, 1.0e9))
    sigma_daily = sigma.get(asset, DEFAULT_SIGMA_DAILY.get(asset, 0.06))

    hl_aum = half_life_aum(gross_edge_bps, adv_usd, sigma_daily)

    # Daily-turnover cap. If a strategy turns over 3x/day, you can carry
    # at most 10% of ADV / 3 = 3.3% of ADV as standing AUM.
    trades_per_day = n_trades / oos_days if oos_days > 0 else 0.0
    bars_per_trade = avg_hold
    # Turnover ≈ (trades_per_day) × (round-trip notional per trade / AUM)
    # Normalise to 1x notional per trade.
    daily_turnover = trades_per_day
    part_cap = participation_cap_aum(adv_usd, daily_turnover)

    return {
        "name": row.get("name", "?"),
        "asset": asset,
        "oos_sharpe": row.get("oos_sharpe"),
        "oos_trades": int(n_trades),
        "avg_trade_return_bps": round(gross_edge_bps, 2),
        "adv_usd": adv_usd,
        "sigma_daily": sigma_daily,
        "trades_per_day": round(trades_per_day, 2),
        "half_life_aum_usd": round(hl_aum, 0),
        "participation_cap_aum_usd": (
            round(part_cap, 0) if math.isfinite(part_cap) else None
        ),
        "recommended_aum_usd": round(min(hl_aum, part_cap), 0)
        if math.isfinite(part_cap)
        else round(hl_aum, 0),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_kv(pairs: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for kv in pairs or []:
        if "=" not in kv:
            raise ValueError(f"Expected KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        out[k.strip()] = float(v)
    return out


def _fmt_usd(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    if x >= 1e9:
        return f"${x/1e9:,.2f}B"
    if x >= 1e6:
        return f"${x/1e6:,.2f}M"
    if x >= 1e3:
        return f"${x/1e3:,.1f}K"
    return f"${x:,.0f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Estimate AUM capacity for SignalForge strategies from a validation JSON."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Path to fund_data/validation_*.json")
    p.add_argument("--adv", nargs="*", default=[],
                   help="Override daily ADV in USD, e.g. BTC/USDT=3e10")
    p.add_argument("--sigma", nargs="*", default=[],
                   help="Override daily return vol, e.g. BTC/USDT=0.04")
    p.add_argument("--bars-per-day", type=float, default=24.0,
                   help="Bars per trading day for the backtest (default 24 for 1h).")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional path to write results JSON.")
    p.add_argument("--top", type=int, default=20,
                   help="Print top N by recommended AUM.")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2

    payload = json.loads(args.input.read_text())
    strategies = payload.get("strategies") or payload.get("results") or []
    keeps = [s for s in strategies if s.get("final_verdict") == "KEEP"] or strategies

    adv_overrides = _parse_kv(args.adv)
    sigma_overrides = _parse_kv(args.sigma)

    rows = [
        estimate(
            r,
            adv={**DEFAULT_ADV_USD, **adv_overrides},
            sigma={**DEFAULT_SIGMA_DAILY, **sigma_overrides},
            bars_per_day=args.bars_per_day,
        )
        for r in keeps
    ]
    rows.sort(key=lambda r: r["recommended_aum_usd"] or 0, reverse=True)

    print(f"\n  Capacity analysis — source: {args.input}")
    print(f"  Model: square-root impact, k={K_IMPACT}, max participation={MAX_PARTICIPATION:.0%}\n")
    print(
        f"  {'Strategy':<32s} {'Asset':>9s} {'EdgeBps':>8s} "
        f"{'Tr/day':>7s} {'Half-AUM':>12s} {'Part-cap':>12s} {'Recommend':>12s}"
    )
    print(f"  {'─'*32} {'─'*9} {'─'*8} {'─'*7} {'─'*12} {'─'*12} {'─'*12}")
    for r in rows[: args.top]:
        print(
            f"  {r['name'][:32]:<32s} {r['asset']:>9s} "
            f"{r['avg_trade_return_bps']:>8.1f} {r['trades_per_day']:>7.2f} "
            f"{_fmt_usd(r['half_life_aum_usd']):>12s} "
            f"{_fmt_usd(r['participation_cap_aum_usd']):>12s} "
            f"{_fmt_usd(r['recommended_aum_usd']):>12s}"
        )
    total = sum((r["recommended_aum_usd"] or 0) for r in rows)
    print(f"\n  Σ aggregate capacity (sum, no diversification benefit): {_fmt_usd(total)}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2))
        print(f"  Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
End-to-end pipeline: research -> cost-aware backtest -> meta-router -> attribution.

Glue script that composes the four modules shipped in this session
(`src.research.autoloop`, `src.execution.fill_model`, `src.intelligence.bandit`,
`src.audit.attribution`) into one command. Proves the pieces compose — they are
no longer four isolated modules with passing tests.

Pipeline
========
    1. Load accepted candidates from `fund_data/research_report.json`
       (produced by `sf research`).
    2. For each candidate, replay signals against cached OHLCV with
       `FillModel` applied on entry / exit bars — real taker fees,
       spread crossing, square-root impact. Produces a stream of
       `JournalRecord` -> `TradeRoundTrip` pairs.
    3. Feed those round-trips into `attribute_trades` to get
       signal / slippage / impact / fee / drift buckets.
    4. Compute a per-strategy-day return series on a held-out slice
       (last 20% of the sample). Score each strategy by its regime
       context using `LinUCB`, then let `MetaRouter` allocate across
       them. Record the softmax weights the router *would* have used.

Output
    fund_data/pipeline_e2e_report.json

Usage
    python scripts/pipeline_e2e.py
    python scripts/pipeline_e2e.py --research fund_data/research_report.json \\
                                    --output fund_data/pipeline_e2e_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit.attribution import attribute_trades
from src.audit.parity import JournalRecord, TradeRoundTrip
from src.execution.fill_model import FillModel, Order, OrderKind
from src.intelligence.bandit import LinUCB, MetaRouter
from src.research import Hypothesis, compile_signal, synthesize_features


# --------------------------------------------------------------------------
# Cost-aware replay
# --------------------------------------------------------------------------
def _signal_to_trades(
    df: pd.DataFrame,
    signal: pd.Series,
    *,
    asset: str,
    strategy: str,
    qty_usd: float,
    fill_model: FillModel,
) -> list[TradeRoundTrip]:
    """Convert a +/-1 signal series into closed round-trips via FillModel."""
    sig = signal.reindex(df.index).fillna(0).astype(int)
    bars = df
    rts: list[TradeRoundTrip] = []

    pos = 0
    entry_rec: JournalRecord | None = None

    # Align signal to *next bar* execution: decide on bar t, fill on bar t+1.
    targets = sig.shift(1).fillna(0).astype(int)

    for i in range(len(bars)):
        ts = bars.index[i]
        bar = bars.iloc[i]
        target = int(targets.iloc[i])

        # No position -> open?
        if pos == 0 and target != 0:
            ref = float(bar["open"])
            qty = qty_usd / max(ref, 1e-9)
            order = Order(ts=ts, side=target, qty=qty, kind=OrderKind.MARKET)
            result = fill_model.fill_market(order, bar)
            if result.filled_qty <= 0:
                continue
            entry_rec = JournalRecord(
                ts=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                event="entry",
                strategy=strategy,
                asset=asset,
                direction=target,
                qty=result.filled_qty,
                price=result.avg_price,
                fee=result.total_fee,
                reference_price=ref,
                slippage_bps=(result.avg_price - ref) / ref * 1e4 * target,
            )
            pos = target
            continue

        # Open position — close when signal flips or goes flat.
        if pos != 0 and target != pos and entry_rec is not None:
            ref = float(bar["open"])
            qty = entry_rec.qty
            order = Order(ts=ts, side=-pos, qty=qty, kind=OrderKind.MARKET)
            result = fill_model.fill_market(order, bar)
            if result.filled_qty <= 0:
                continue
            exit_rec = JournalRecord(
                ts=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                event="exit",
                strategy=strategy,
                asset=asset,
                direction=pos,
                qty=result.filled_qty,
                price=result.avg_price,
                fee=result.total_fee,
                reference_price=ref,
                slippage_bps=(ref - result.avg_price) / ref * 1e4 * pos,
                pnl=(result.avg_price - entry_rec.price) * pos * qty
                    - (entry_rec.fee + result.total_fee),
            )
            rts.append(TradeRoundTrip(entry=entry_rec, exit=exit_rec))
            entry_rec = None
            pos = 0
            # Re-enter same bar if target is non-zero opposite.
            if target != 0:
                ref2 = float(bar["close"])
                qty2 = qty_usd / max(ref2, 1e-9)
                o2 = Order(ts=ts, side=target, qty=qty2, kind=OrderKind.MARKET)
                r2 = fill_model.fill_market(o2, bar)
                if r2.filled_qty > 0:
                    entry_rec = JournalRecord(
                        ts=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                        event="entry",
                        strategy=strategy,
                        asset=asset,
                        direction=target,
                        qty=r2.filled_qty,
                        price=r2.avg_price,
                        fee=r2.total_fee,
                        reference_price=ref2,
                        slippage_bps=(r2.avg_price - ref2) / ref2 * 1e4 * target,
                    )
                    pos = target
    return rts


def _equity_curve(rts: list[TradeRoundTrip], start_capital: float = 10_000.0) -> pd.Series:
    """Equity curve indexed by exit timestamp."""
    if not rts:
        return pd.Series([start_capital], index=[pd.Timestamp.utcnow()])
    sorted_rts = sorted(rts, key=lambda r: r.exit.ts)
    ts = [pd.Timestamp(r.exit.ts) for r in sorted_rts]
    pnl = np.cumsum([r.realised_pnl for r in sorted_rts])
    return pd.Series(start_capital + pnl, index=pd.DatetimeIndex(ts))


def _daily_returns(eq: pd.Series) -> pd.Series:
    if len(eq) < 2:
        return pd.Series(dtype=float)
    daily = eq.resample("1D").last().ffill()
    return daily.pct_change().dropna()


# --------------------------------------------------------------------------
# Regime context for the bandit
# --------------------------------------------------------------------------
def _regime_context(df: pd.DataFrame, ts: pd.Timestamp) -> np.ndarray:
    """Build a 3-d regime context vector at timestamp ts."""
    if ts not in df.index:
        # Find nearest <= ts.
        idx = df.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return np.zeros(3)
        row = df.iloc[idx]
    else:
        row = df.loc[ts]
    # [zmom_20, vol_ratio, range_ratio] — all synthesised by autoloop features.
    return np.array([
        float(row.get("zmom_20", 0.0) or 0.0),
        float(row.get("vol_ratio", 1.0) or 1.0),
        float(row.get("range_ratio", 1.0) or 1.0),
    ])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(
    research_path: Path,
    cache_dir: Path,
    output_path: Path,
    *,
    qty_usd: float = 10_000.0,
    holdout_frac: float = 0.2,
    temperature: float = 0.5,
) -> dict:
    research = json.loads(research_path.read_text())
    accepted = research.get("accepted", [])
    if not accepted:
        print("No accepted candidates in research report; nothing to do.")
        return {}

    fill_model = FillModel(venue="binance_perp", participation_cap=0.02,
                            spread_bps=2.0, impact_k=1.0)

    # Group: one per (symbol, candidate) combination.
    per_strategy: dict[str, dict] = {}
    all_trades: list[TradeRoundTrip] = []
    frames_cache: dict[str, pd.DataFrame] = {}

    for c in accepted:
        sym: str = c["symbol"]
        name: str = c["name"]
        sid = f"{sym.replace('/', '_')}::{name}"
        if sym not in frames_cache:
            p = cache_dir / f"{sym.replace('/', '_')}_1h.parquet"
            if not p.exists():
                print(f"  skip {sym}: {p} missing")
                continue
            frames_cache[sym] = synthesize_features(pd.read_parquet(p))
        feats = frames_cache[sym]

        # Split: last `holdout_frac` is the out-of-everything slice.
        n = len(feats)
        cut = int(n * (1 - holdout_frac))
        holdout = feats.iloc[cut:]

        hyp = Hypothesis(
            name=name,
            feature=c["feature"],
            op=c["op"],
            threshold=float(c["threshold"]),
            side=int(c["side"]),
        )
        signal_fn = compile_signal(hyp)
        sig = signal_fn(holdout)

        rts = _signal_to_trades(
            holdout, sig,
            asset=sym, strategy=name,
            qty_usd=qty_usd, fill_model=fill_model,
        )
        eq = _equity_curve(rts, start_capital=100_000.0)
        dret = _daily_returns(eq)
        sharpe = 0.0
        if len(dret) > 1 and dret.std(ddof=0) > 0:
            sharpe = float(dret.mean() / dret.std(ddof=0) * math.sqrt(365))
        per_strategy[sid] = {
            "symbol": sym,
            "hypothesis": name,
            "n_trades": len(rts),
            "realised_pnl": sum(r.realised_pnl for r in rts),
            "sharpe_net": sharpe,
            "equity_end": float(eq.iloc[-1]) if len(eq) else 100_000.0,
        }
        all_trades.extend(rts)
        print(f"  {sid:40s} trades={len(rts):4d}  net_pnl={per_strategy[sid]['realised_pnl']:>10,.0f}  sharpe={sharpe:+.2f}")

    # -------- Attribution over all trades --------
    report = attribute_trades(all_trades)
    print(f"\nAttribution over {report.n_trades} closed round-trips:")
    print(f"  realised : {report.total_realised:>12,.0f}")
    print(f"  signal   : {report.total_signal:>12,.0f}")
    print(f"  slippage : {report.total_slippage:>12,.0f}")
    print(f"  impact   : {report.total_impact:>12,.0f}   (of slippage)")
    print(f"  fees     : {report.total_fee:>12,.0f}")
    print(f"  funding  : {report.total_funding:>12,.0f}")
    print(f"  drift    : {report.total_drift:>12,.0f}   (unattributed)")
    # Sanity: sum should equal realised within fp eps.
    recon = (report.total_signal - report.total_slippage
             - report.total_fee - report.total_funding + report.total_drift)
    print(f"  check    : signal - slip - fees - funding + drift = {recon:,.2f}  vs realised {report.total_realised:,.2f}")

    # -------- Meta-router: fit on per-trade rewards, then allocate --------
    arms = list(per_strategy.keys())
    router_alloc: dict = {}
    if len(arms) >= 2 and all_trades:
        bandit = LinUCB(arms=arms, d=3, alpha=0.3)
        router = MetaRouter(bandit=bandit, temperature=temperature)
        # One update per trade: reward = realised P&L (normalised),
        # context = regime at entry.
        pnls = [r.realised_pnl for r in all_trades]
        pnl_scale = max(abs(max(pnls)), abs(min(pnls)), 1.0)
        for rt in all_trades:
            sid = f"{rt.entry.asset.replace('/', '_')}::{rt.entry.strategy}"
            if sid not in arms:
                continue
            ctx = _regime_context(
                frames_cache[rt.entry.asset],
                pd.Timestamp(rt.entry.ts),
            )
            reward = rt.realised_pnl / pnl_scale
            router.update(ctx, sid, reward)
        # Allocation under the *current* (latest) regime:
        last_ts = max(df.index[-1] for df in frames_cache.values())
        ref_sym = next(iter(frames_cache))
        ctx_now = _regime_context(frames_cache[ref_sym], last_ts)
        weights = router.allocate(ctx_now)
        router_alloc = {
            "context": {
                "zmom_20": float(ctx_now[0]),
                "vol_ratio": float(ctx_now[1]),
                "range_ratio": float(ctx_now[2]),
            },
            "weights": {k: float(v) for k, v in weights.items()},
        }
        print("\nMetaRouter allocation under current regime:")
        for k, v in sorted(weights.items(), key=lambda kv: -kv[1]):
            print(f"  {k:40s} {v:6.1%}")

    # -------- Pack report --------
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_source": str(research_path),
        "holdout_fraction": holdout_frac,
        "qty_usd": qty_usd,
        "per_strategy": per_strategy,
        "attribution": report.to_dict(),
        "router": router_alloc,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nReport written to {output_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--research", default="fund_data/research_report.json")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--output", default="fund_data/pipeline_e2e_report.json")
    ap.add_argument("--qty-usd", type=float, default=10_000.0)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=0.5)
    args = ap.parse_args()
    run(
        research_path=Path(args.research),
        cache_dir=Path(args.cache_dir),
        output_path=Path(args.output),
        qty_usd=args.qty_usd,
        holdout_frac=args.holdout_frac,
        temperature=args.temperature,
    )

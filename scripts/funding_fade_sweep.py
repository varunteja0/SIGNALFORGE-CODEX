"""Fast sweep: load data once per symbol, then iterate params in-memory."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies.funding_fade import (
    backtest_funding_fade,
    load_data,
)


def run(symbols: list[str], days: int, oos_fraction: float, output: str) -> None:
    print(f"Symbols: {symbols}  history={days}d  oos={oos_fraction}\n")

    cached: dict[str, dict] = {}
    for sym in symbols:
        print(f"[{sym}] fetching...", flush=True)
        ohlcv, funding = load_data(sym, days=days)
        if ohlcv.empty or funding.empty:
            print("  SKIP (no data)")
            continue
        start = max(ohlcv.index[0], funding.index[0])
        end = min(ohlcv.index[-1], funding.index[-1])
        ohlcv = ohlcv[(ohlcv.index >= start) & (ohlcv.index <= end)]
        funding = funding[(funding.index >= start) & (funding.index <= end)]
        if len(ohlcv) < 500 or len(funding) < 100:
            print(f"  SKIP (insufficient: ohlcv={len(ohlcv)} funding={len(funding)})")
            continue
        split_idx = int(len(ohlcv) * (1 - oos_fraction))
        split_ts = ohlcv.index[split_idx]
        cached[sym] = {
            "ohlcv_is":    ohlcv[ohlcv.index < split_ts],
            "ohlcv_oos":   ohlcv[ohlcv.index >= split_ts],
            "funding_is":  funding[funding.index < split_ts],
            "funding_oos": funding[funding.index >= split_ts],
            "split_ts":    split_ts,
        }
        print(f"  loaded: ohlcv={len(ohlcv)} funding={len(funding)} split={split_ts}")

    if not cached:
        print("No data loaded, aborting.")
        return

    entry_zs = [1.5, 2.0, 2.5, 3.0]
    holds = [1, 2, 3, 6]
    sides = ["both", "short_only", "long_only"]

    results = []
    header = (f"\n{'sym':7} {'z':>4} {'h':>3} {'side':>11} | "
              f"{'IS trd':>6} {'IS Sh':>6} {'IS PF':>6} {'IS ret':>7} | "
              f"{'OOS trd':>7} {'OOS Sh':>6} {'OOS PF':>6} {'OOS ret':>7} {'OOS dd':>6}")
    print(header)
    print("-" * (len(header) - 1))

    for sym, ez, h, side in itertools.product(cached.keys(), entry_zs, holds, sides):
        c = cached[sym]
        is_r = backtest_funding_fade(c["ohlcv_is"], c["funding_is"],
                                     symbol=sym, entry_z=ez,
                                     hold_periods=h, side=side)
        oos_r = backtest_funding_fade(c["ohlcv_oos"], c["funding_oos"],
                                      symbol=sym, entry_z=ez,
                                      hold_periods=h, side=side)
        if is_r.n_trades < 10 or oos_r.n_trades < 10:
            continue
        print(f"{sym.split('/')[0]:7} {ez:>4.1f} {h:>3} {side:>11} | "
              f"{is_r.n_trades:>6} {is_r.sharpe:>6.2f} {is_r.profit_factor:>6.2f} "
              f"{is_r.total_return*100:>6.2f}% | "
              f"{oos_r.n_trades:>7} {oos_r.sharpe:>6.2f} {oos_r.profit_factor:>6.2f} "
              f"{oos_r.total_return*100:>6.2f}% {oos_r.max_drawdown*100:>5.2f}%")
        results.append({
            "symbol": sym, "entry_z": ez, "hold": h, "side": side,
            "is": {"trades": is_r.n_trades, "sharpe": round(is_r.sharpe, 2),
                   "pf": round(is_r.profit_factor, 2),
                   "total": round(is_r.total_return, 4),
                   "dd": round(is_r.max_drawdown, 4)},
            "oos": {"trades": oos_r.n_trades, "sharpe": round(oos_r.sharpe, 2),
                    "pf": round(oos_r.profit_factor, 2),
                    "total": round(oos_r.total_return, 4),
                    "dd": round(oos_r.max_drawdown, 4)},
        })

    print("\n" + "=" * 72)
    print("  BEST OOS per symbol (by OOS Sharpe)")
    print("=" * 72)
    best_by_sym: dict[str, dict] = {}
    for r in results:
        cur = best_by_sym.get(r["symbol"])
        if cur is None or r["oos"]["sharpe"] > cur["oos"]["sharpe"]:
            best_by_sym[r["symbol"]] = r
    for sym, r in best_by_sym.items():
        print(f"  {sym}:  z={r['entry_z']} hold={r['hold']} side={r['side']}  "
              f"OOS Sh={r['oos']['sharpe']}  trades={r['oos']['trades']}  "
              f"ret={r['oos']['total']*100:.2f}%  dd={r['oos']['dd']*100:.2f}%")

    print("\n" + "=" * 72)
    print("  ROBUSTNESS — combos working on >=2 symbols (OOS Sh > 0.5, >=10 trades)")
    print("=" * 72)
    by_combo: dict[tuple, list] = {}
    for r in results:
        by_combo.setdefault((r["entry_z"], r["hold"], r["side"]), []).append(r)
    robust = []
    for combo, rows in by_combo.items():
        good = [r for r in rows if r["oos"]["sharpe"] > 0.5 and r["oos"]["trades"] >= 10]
        if len(good) >= 2:
            avg_sh = sum(r["oos"]["sharpe"] for r in good) / len(good)
            robust.append((combo, len(good), avg_sh, good))
    robust.sort(key=lambda x: -x[2])
    if not robust:
        print("  NONE. No param combo generalises across >=2 symbols.")
    else:
        for combo, _, avg, good in robust[:10]:
            ez, h, side = combo
            syms = [r["symbol"].split("/")[0] for r in good]
            print(f"  z={ez} hold={h} side={side:>11}  symbols={syms}  avg_OOS_Sh={avg:.2f}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"sweep": results, "best_by_symbol": best_by_sym},
                  f, indent=2, default=str)
    print(f"\nReport: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+",
                   default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--oos-fraction", type=float, default=0.30)
    p.add_argument("--output", default="fund_data/funding_fade_sweep.json")
    args = p.parse_args()
    run(args.symbols, args.days, args.oos_fraction, args.output)

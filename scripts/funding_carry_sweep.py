"""Funding carry parameter sweep: find entry/exit thresholds that survive OOS."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies.funding_carry import backtest_funding_carry, load_data


def run(symbols, days, oos_fraction, output):
    cached = {}
    print(f"Loading {len(symbols)} symbols...")
    for s in symbols:
        print(f"  [{s}]", flush=True)
        ohlcv, funding = load_data(s, days=days)
        if ohlcv.empty or funding.empty or len(funding) < 100:
            print("    SKIP")
            continue
        start = max(ohlcv.index[0], funding.index[0])
        end = min(ohlcv.index[-1], funding.index[-1])
        ohlcv = ohlcv[(ohlcv.index >= start) & (ohlcv.index <= end)]
        funding = funding[(funding.index >= start) & (funding.index <= end)]
        split_idx = int(len(ohlcv) * (1 - oos_fraction))
        split_ts = ohlcv.index[split_idx]
        cached[s] = {
            "o_is":  ohlcv[ohlcv.index < split_ts],
            "o_oos": ohlcv[ohlcv.index >= split_ts],
            "f_is":  funding[funding.index < split_ts],
            "f_oos": funding[funding.index >= split_ts],
        }

    entry_thresholds = [0.00005, 0.00010, 0.00015, 0.00020, 0.00030]
    exit_thresholds  = [0.0, 0.00002, 0.00005]
    exit_windows     = [3, 9, 15]
    min_holds        = [3, 6, 12]

    print(f"\n{'sym':7} {'entry':>7} {'exit':>7} {'win':>4} {'hold':>4} | "
          f"{'IS trd':>6} {'IS ann':>7} {'IS Sh':>6} {'IS dd':>6} | "
          f"{'OOS trd':>7} {'OOS ann':>8} {'OOS Sh':>7} {'OOS dd':>6} {'OOS wr':>6}")
    print("-" * 130)
    results = []
    for sym, ent, ex, win, mh in itertools.product(
            cached, entry_thresholds, exit_thresholds, exit_windows, min_holds):
        c = cached[sym]
        is_r = backtest_funding_carry(c["o_is"], c["f_is"], sym,
                                      min_funding_threshold=ent,
                                      exit_when_below=ex,
                                      exit_window=win,
                                      min_hold_periods=mh)
        oos_r = backtest_funding_carry(c["o_oos"], c["f_oos"], sym,
                                       min_funding_threshold=ent,
                                       exit_when_below=ex,
                                       exit_window=win,
                                       min_hold_periods=mh)
        if is_r.n_trades < 3 or oos_r.n_trades < 2:
            continue
        print(f"{sym.split('/')[0]:7} {ent*100:>6.3f}% {ex*100:>6.3f}% "
              f"{win:>4d} {mh:>4d} | "
              f"{is_r.n_trades:>6d} {is_r.annualised_return_on_capital*100:>6.2f}% "
              f"{is_r.sharpe:>6.2f} {is_r.max_drawdown*100:>5.2f}% | "
              f"{oos_r.n_trades:>7d} {oos_r.annualised_return_on_capital*100:>7.2f}% "
              f"{oos_r.sharpe:>7.2f} {oos_r.max_drawdown*100:>5.2f}% "
              f"{oos_r.win_rate*100:>5.1f}%")
        results.append({
            "symbol": sym, "entry": ent, "exit": ex, "win": win, "hold": mh,
            "is_ann": is_r.annualised_return_on_capital,
            "oos_ann": oos_r.annualised_return_on_capital,
            "is_sharpe": is_r.sharpe, "oos_sharpe": oos_r.sharpe,
            "is_dd": is_r.max_drawdown, "oos_dd": oos_r.max_drawdown,
            "is_trades": is_r.n_trades, "oos_trades": oos_r.n_trades,
            "oos_wr": oos_r.win_rate,
        })

    print("\n" + "=" * 110)
    print("  ROBUSTNESS: combos where OOS ann > 2% AND OOS Sh > 1 AND OOS dd < 5%")
    print("  across >=3 symbols")
    print("=" * 110)
    by_combo = {}
    for r in results:
        key = (r["entry"], r["exit"], r["win"], r["hold"])
        by_combo.setdefault(key, []).append(r)
    robust = []
    for combo, rows in by_combo.items():
        good = [r for r in rows
                if r["oos_ann"] > 0.02 and r["oos_sharpe"] > 1.0
                and r["oos_dd"] < 0.05]
        if len(good) >= 3:
            avg_ann = sum(r["oos_ann"] for r in good) / len(good)
            robust.append((combo, len(good), avg_ann, good))
    robust.sort(key=lambda x: -x[2])
    if not robust:
        print("  NONE.")
        # Fall back: show combos working on >= 2 symbols with any positive OOS
        print("\n  Fallback — combos with OOS ann > 0 on >=3 symbols:")
        for combo, rows in by_combo.items():
            good = [r for r in rows if r["oos_ann"] > 0]
            if len(good) >= 3:
                avg_ann = sum(r["oos_ann"] for r in good) / len(good)
                ent, ex, win, mh = combo
                syms = [r["symbol"].split("/")[0] for r in good]
                print(f"    entry={ent*100:.3f}% exit={ex*100:.3f}% win={win} hold={mh}  "
                      f"symbols={syms} avg_OOS_ann={avg_ann*100:.2f}%")
    else:
        for combo, n, avg_ann, good in robust[:10]:
            ent, ex, win, mh = combo
            syms = [r["symbol"].split("/")[0] for r in good]
            print(f"  entry={ent*100:.3f}% exit={ex*100:.3f}% win={win} hold={mh}  "
                  f"syms={syms} avg_OOS_ann={avg_ann*100:.2f}%")

    print("\n" + "=" * 110)
    print("  BEST OOS per symbol (maximise OOS annualised return with Sh > 1, dd < 5%)")
    print("=" * 110)
    best_by_sym = {}
    for r in results:
        if r["oos_sharpe"] < 1.0 or r["oos_dd"] > 0.05:
            continue
        cur = best_by_sym.get(r["symbol"])
        if cur is None or r["oos_ann"] > cur["oos_ann"]:
            best_by_sym[r["symbol"]] = r
    for sym, r in best_by_sym.items():
        print(f"  {sym:12}: entry={r['entry']*100:.3f}% exit={r['exit']*100:.3f}% "
              f"win={r['win']} hold={r['hold']}  "
              f"OOS ann={r['oos_ann']*100:.2f}%  Sh={r['oos_sharpe']:.2f}  "
              f"dd={r['oos_dd']*100:.2f}%  trd={r['oos_trades']}  wr={r['oos_wr']*100:.0f}%")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"sweep": results,
                   "best_by_symbol": best_by_sym,
                   "robust_combos": [
                       {"entry": c[0], "exit": c[1], "win": c[2], "hold": c[3],
                        "n_symbols": n, "avg_oos_ann": avg}
                       for c, n, avg, _ in robust]},
                  f, indent=2, default=str)
    print(f"\nReport: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+",
                   default=["BTC/USDT", "ETH/USDT", "SOL/USDT",
                            "XRP/USDT", "BNB/USDT"])
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--oos-fraction", type=float, default=0.30)
    p.add_argument("--output", default="fund_data/funding_carry_sweep.json")
    args = p.parse_args()
    run(args.symbols, args.days, args.oos_fraction, args.output)

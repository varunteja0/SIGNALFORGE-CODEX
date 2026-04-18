"""Run funding carry backtest across multiple symbols with IS/OOS split."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies.funding_carry import (
    CarryResult,
    backtest_funding_carry,
    load_data,
)


def fmt_result(label: str, r: CarryResult) -> str:
    return (f"  {label:4s}: trades={r.n_trades:>4d} "
            f"periods_held={r.n_periods_in_position:>5d}/{r.n_periods_total:<5d} "
            f"net_notional={r.net_return_on_notional*100:>6.2f}% "
            f"net_capital={r.net_return_on_capital*100:>6.2f}% "
            f"ann={r.annualised_return_on_capital*100:>6.2f}% "
            f"Sh={r.sharpe:>5.2f} dd={r.max_drawdown*100:>5.2f}% "
            f"wr={r.win_rate*100:>4.1f}% "
            f"pct_pos_funding={r.pct_periods_positive_funding*100:>4.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+",
                   default=["BTC/USDT", "ETH/USDT", "SOL/USDT",
                            "XRP/USDT", "BNB/USDT"])
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--oos-fraction", type=float, default=0.30)
    p.add_argument("--min-funding", type=float, default=0.00005,
                   help="Min funding rate to enter (default 0.005% per 8h)")
    p.add_argument("--exit-at", type=float, default=0.0,
                   help="Exit when funding drops to this level")
    p.add_argument("--output", default="fund_data/funding_carry_report.json")
    args = p.parse_args()

    print("=" * 110)
    print("  FUNDING CARRY (DELTA-NEUTRAL) BACKTEST")
    print("=" * 110)
    print(f"  Symbols: {args.symbols}")
    print(f"  History: {args.days}d  OOS: {args.oos_fraction}")
    print(f"  Entry funding >= {args.min_funding*100:.4f}%  "
          f"Exit funding <= {args.exit_at*100:.4f}%")
    print(f"  Costs: spot {100*0.0010:.2f}% comm + {100*0.0005:.2f}% slip | "
          f"perp {100*0.0005:.2f}% comm + {100*0.0003:.2f}% slip per leg entry+exit")
    print("=" * 110)

    all_results = {}
    is_rollup = {"net_not": 0.0, "net_cap": 0.0, "n": 0, "dd": 0.0}
    oos_rollup = {"net_not": 0.0, "net_cap": 0.0, "n": 0, "dd": 0.0}

    for sym in args.symbols:
        print(f"\n[{sym}] loading...", flush=True)
        ohlcv, funding = load_data(sym, days=args.days)
        if ohlcv.empty or funding.empty:
            print(f"  SKIP (no data)")
            continue
        start = max(ohlcv.index[0], funding.index[0])
        end = min(ohlcv.index[-1], funding.index[-1])
        ohlcv = ohlcv[(ohlcv.index >= start) & (ohlcv.index <= end)]
        funding = funding[(funding.index >= start) & (funding.index <= end)]
        if len(funding) < 100:
            print(f"  SKIP (insufficient funding: {len(funding)})")
            continue

        split_idx = int(len(ohlcv) * (1 - args.oos_fraction))
        split_ts = ohlcv.index[split_idx]

        is_r = backtest_funding_carry(
            ohlcv[ohlcv.index < split_ts],
            funding[funding.index < split_ts],
            symbol=sym,
            min_funding_threshold=args.min_funding,
            exit_when_below=args.exit_at,
        )
        oos_r = backtest_funding_carry(
            ohlcv[ohlcv.index >= split_ts],
            funding[funding.index >= split_ts],
            symbol=sym,
            min_funding_threshold=args.min_funding,
            exit_when_below=args.exit_at,
        )

        print(fmt_result("IS ", is_r))
        print(fmt_result("OOS", oos_r))
        print(f"    funding stats (full):  "
              f"mean={is_r.funding_stats.get('mean', 0)*100:.5f}%  "
              f"p90={is_r.funding_stats.get('p90', 0)*100:.5f}%  "
              f"p99={is_r.funding_stats.get('p99', 0)*100:.5f}%")

        # Clean for JSON
        def scrub(r: CarryResult) -> dict:
            d = {k: v for k, v in asdict(r).items() if k != "trades"}
            d["n_trades"] = r.n_trades
            return d

        all_results[sym] = {
            "split_ts": str(split_ts),
            "is": scrub(is_r),
            "oos": scrub(oos_r),
        }

        if is_r.n_trades > 0:
            is_rollup["net_not"] += is_r.net_return_on_notional
            is_rollup["net_cap"] += is_r.net_return_on_capital
            is_rollup["dd"] = max(is_rollup["dd"], is_r.max_drawdown)
            is_rollup["n"] += 1
        if oos_r.n_trades > 0:
            oos_rollup["net_not"] += oos_r.net_return_on_notional
            oos_rollup["net_cap"] += oos_r.net_return_on_capital
            oos_rollup["dd"] = max(oos_rollup["dd"], oos_r.max_drawdown)
            oos_rollup["n"] += 1

    # Aggregate verdict
    print("\n" + "=" * 110)
    print("  PORTFOLIO (equal-weight across symbols)")
    print("=" * 110)
    n_is = max(is_rollup["n"], 1)
    n_oos = max(oos_rollup["n"], 1)
    print(f"  IS  : avg_net_capital_return = {is_rollup['net_cap']/n_is*100:>6.2f}%  "
          f"worst_symbol_dd = {is_rollup['dd']*100:>5.2f}%  symbols={n_is}")
    print(f"  OOS : avg_net_capital_return = {oos_rollup['net_cap']/n_oos*100:>6.2f}%  "
          f"worst_symbol_dd = {oos_rollup['dd']*100:>5.2f}%  symbols={n_oos}")

    # Positive-OOS check
    pos_syms = [s for s, r in all_results.items()
                if r["oos"]["net_return_on_capital"] > 0]
    print(f"\n  Symbols with POSITIVE OOS return: {len(pos_syms)}/{len(all_results)}"
          f"  ->  {pos_syms}")

    # Verdict
    avg_oos = oos_rollup["net_cap"] / n_oos
    if avg_oos > 0.02 and len(pos_syms) >= len(all_results) * 0.6:
        verdict = "DEPLOYABLE"
    elif avg_oos > 0 and len(pos_syms) >= 2:
        verdict = "MARGINAL"
    else:
        verdict = "NO EDGE"
    print(f"\n  VERDICT: {verdict}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "config": vars(args),
            "results": all_results,
            "rollup": {
                "is_avg_capital_return": is_rollup["net_cap"] / n_is,
                "oos_avg_capital_return": oos_rollup["net_cap"] / n_oos,
                "is_worst_dd": is_rollup["dd"],
                "oos_worst_dd": oos_rollup["dd"],
                "positive_oos_symbols": pos_syms,
                "verdict": verdict,
            },
        }, f, indent=2, default=str)
    print(f"\n  Report: {args.output}")


if __name__ == "__main__":
    main()

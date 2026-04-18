"""Run Cross-Sectional Funding Carry backtest across a broad universe."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies.csfc import backtest_csfc, load_universe

DEFAULT_UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "MATIC/USDT",
    "LTC/USDT", "DOT/USDT", "TRX/USDT", "ATOM/USDT", "ETC/USDT",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--oos-fraction", type=float, default=0.30)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--min-funding", type=float, default=0.00010,
                   help="Min funding rate (0.01% per 8h = ~11%/yr raw)")
    p.add_argument("--rebalance-every", type=int, default=1)
    p.add_argument("--output", default="fund_data/csfc_report.json")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    print("=" * 100)
    print("  CROSS-SECTIONAL FUNDING CARRY (CSFC)")
    print("=" * 100)
    print(f"  Universe: {len(args.symbols)} symbols")
    print(f"  Top-K:    {args.top_k}  min_funding: {args.min_funding*100:.4f}%/8h")
    print(f"  Rebalance every {args.rebalance_every} funding period(s)")
    print(f"  Round-trip cost per symbol rotation: "
          f"{2*(0.00075+0.00020+0.00018+0.00010)*100:.3f}% of notional")
    print("=" * 100)

    print("\nLoading universe funding data...")
    universe = load_universe(args.symbols, days=args.days)
    print(f"Loaded: {list(universe.keys())}")

    if len(universe) < 3:
        print("Too few symbols with data. Aborting.")
        return

    # Split into IS / OOS by global timeline
    all_ts = sorted(set().union(*[df.index for df in universe.values()]))
    split_idx = int(len(all_ts) * (1 - args.oos_fraction))
    split_ts = all_ts[split_idx]
    is_universe = {s: df[df.index < split_ts] for s, df in universe.items()}
    oos_universe = {s: df[df.index >= split_ts] for s, df in universe.items()}
    print(f"\nSplit ts: {split_ts}")

    def fmt(label, r):
        return (f"  {label:4s}: periods={r.n_periods:>4d} deployed={r.n_periods_deployed:<4d} "
                f"ret_cap={r.total_return_on_capital*100:>6.2f}% "
                f"ann={r.annualised_return_on_capital*100:>6.2f}% "
                f"Sh={r.sharpe:>6.2f} Sor={r.sortino:>6.2f} "
                f"dd={r.max_drawdown*100:>4.2f}% "
                f"win%={r.period_win_rate*100:>4.1f}% "
                f"fail%={r.failure_rate*100:>4.1f}%  "
                f"gross={r.gross_funding*100:>5.2f}% cost={r.total_costs*100:>5.2f}%")

    if args.sweep:
        print("\n=== PARAMETER SWEEP ===")
        print(f"{'K':>3} {'min_f%':>7} {'reb':>4} {'win':>4} {'marg%':>6} | "
              f"{'IS ann%':>7} {'IS Sh':>6} {'IS dd%':>6} {'IS fail%':>8} | "
              f"{'OOSann%':>7} {'OOSSh':>6} {'OOSdd%':>6} {'OOSfail%':>9}")
        print("-" * 120)
        sweep_results = []
        for k, mf, reb, win, marg in itertools.product(
                [2, 3, 5],
                [0.00005, 0.00010, 0.00015],
                [1, 3],
                [3, 6, 12],
                [0.00005, 0.00010, 0.00020]):
            is_r = backtest_csfc(is_universe, top_k=k, min_funding=mf,
                                 rebalance_every=reb, rank_window=win,
                                 rotation_margin=marg)
            oos_r = backtest_csfc(oos_universe, top_k=k, min_funding=mf,
                                  rebalance_every=reb, rank_window=win,
                                  rotation_margin=marg)
            if is_r.n_periods < 50 or oos_r.n_periods < 20:
                continue
            print(f"{k:>3} {mf*100:>6.3f}% {reb:>4} {win:>4} {marg*100:>5.3f}% | "
                  f"{is_r.annualised_return_on_capital*100:>6.2f}% {is_r.sharpe:>6.2f} "
                  f"{is_r.max_drawdown*100:>5.2f}% {is_r.failure_rate*100:>7.2f}% | "
                  f"{oos_r.annualised_return_on_capital*100:>6.2f}% {oos_r.sharpe:>6.2f} "
                  f"{oos_r.max_drawdown*100:>5.2f}% {oos_r.failure_rate*100:>8.2f}%")
            sweep_results.append({
                "top_k": k, "min_funding": mf, "rebalance_every": reb,
                "rank_window": win, "rotation_margin": marg,
                "is_ann": is_r.annualised_return_on_capital,
                "oos_ann": oos_r.annualised_return_on_capital,
                "is_sharpe": is_r.sharpe, "oos_sharpe": oos_r.sharpe,
                "is_dd": is_r.max_drawdown, "oos_dd": oos_r.max_drawdown,
                "is_fail": is_r.failure_rate, "oos_fail": oos_r.failure_rate,
                "is_periods": is_r.n_periods, "oos_periods": oos_r.n_periods,
                "oos_gross": is_r.gross_funding, "oos_cost": is_r.total_costs,
            })

        print("\n=== TOP 20 by OOS annualised return (OOS dd < 5%, Sh > 0) ===")
        filtered = [r for r in sweep_results
                    if r["oos_dd"] < 0.05 and r["oos_sharpe"] > 0.0
                    and r["is_ann"] > 0 and r["oos_ann"] > 0]
        filtered.sort(key=lambda r: -r["oos_ann"])
        for r in filtered[:20]:
            print(f"  K={r['top_k']} min_f={r['min_funding']*100:.3f}% "
                  f"reb={r['rebalance_every']} win={r['rank_window']} "
                  f"marg={r['rotation_margin']*100:.3f}%  "
                  f"OOS ann={r['oos_ann']*100:.2f}% Sh={r['oos_sharpe']:.2f} "
                  f"dd={r['oos_dd']*100:.2f}% fail={r['oos_fail']*100:.1f}%  "
                  f"IS ann={r['is_ann']*100:.2f}% Sh={r['is_sharpe']:.2f}")

        with open(args.output, "w") as f:
            json.dump({"config": vars(args), "sweep": sweep_results}, f, indent=2,
                      default=str)
    else:
        is_r = backtest_csfc(is_universe, top_k=args.top_k,
                             min_funding=args.min_funding,
                             rebalance_every=args.rebalance_every)
        oos_r = backtest_csfc(oos_universe, top_k=args.top_k,
                              min_funding=args.min_funding,
                              rebalance_every=args.rebalance_every)

        print("\n" + fmt("IS ", is_r))
        print(fmt("OOS", oos_r))

        print(f"\n  IS top held symbols:")
        for s, t in sorted(is_r.symbol_time_in_book.items(),
                           key=lambda x: -x[1])[:10]:
            print(f"    {s:12} held in {t} periods ({t/is_r.n_periods*100:.1f}%)")
        print(f"\n  OOS top held symbols:")
        for s, t in sorted(oos_r.symbol_time_in_book.items(),
                           key=lambda x: -x[1])[:10]:
            print(f"    {s:12} held in {t} periods ({t/oos_r.n_periods*100:.1f}%)")

        # Verdict
        oos_ann = oos_r.annualised_return_on_capital
        if oos_ann > 0.05 and oos_r.sharpe > 2 and oos_r.max_drawdown < 0.05:
            verdict = "DEPLOYABLE — strong cross-sectional edge"
        elif oos_ann > 0.02 and oos_r.sharpe > 1 and oos_r.max_drawdown < 0.05:
            verdict = "MARGINAL — acceptable edge, low failure rate"
        elif oos_ann > 0:
            verdict = "WEAK — edge exists but thin"
        else:
            verdict = "NO EDGE"
        print(f"\n  VERDICT: {verdict}")

        def scrub(r):
            d = {k: v for k, v in asdict(r).items() if k != "periods"}
            return d

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "config": vars(args),
                "split_ts": str(split_ts),
                "is": scrub(is_r), "oos": scrub(oos_r),
                "verdict": verdict,
            }, f, indent=2, default=str)
    print(f"\n  Report: {args.output}")


if __name__ == "__main__":
    main()

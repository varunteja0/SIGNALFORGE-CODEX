"""Runner: backtest funding-fade on BTC/ETH/SOL, IS/OOS split, report honest numbers."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.strategies.funding_fade import evaluate_is_oos, FundingFadeResult


def _summarize(r: FundingFadeResult) -> dict:
    return {
        "n_trades": r.n_trades,
        "n_long": r.n_long,
        "n_short": r.n_short,
        "win_rate": round(r.win_rate, 4),
        "avg_return_per_trade": round(r.avg_return, 5),
        "total_return": round(r.total_return, 4),
        "sharpe_annualised": round(r.sharpe, 2),
        "sortino_annualised": round(r.sortino, 2),
        "max_drawdown": round(r.max_drawdown, 4),
        "profit_factor": round(r.profit_factor, 2),
    }


def run(symbols: list[str], days: int, oos_fraction: float,
        entry_z: float, hold_periods: int, output: str) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    print("=" * 72)
    print("  FUNDING FADE — FOCUSED STRATEGY BACKTEST")
    print("=" * 72)
    print(f"  Symbols:        {symbols}")
    print(f"  History:        {days} days (~{days/365:.1f}y)")
    print(f"  OOS fraction:   {oos_fraction}")
    print(f"  Entry |z|:      {entry_z}")
    print(f"  Hold periods:   {hold_periods}  (~{hold_periods * 8}h)")
    print("=" * 72)

    report: dict = {"params": {"days": days, "oos_fraction": oos_fraction,
                               "entry_z": entry_z, "hold_periods": hold_periods},
                    "per_symbol": {}}

    for sym in symbols:
        print(f"\n[{sym}] loading data ...")
        out = evaluate_is_oos(
            sym,
            days=days,
            oos_fraction=oos_fraction,
            entry_z=entry_z,
            hold_periods=hold_periods,
        )
        if "error" in out:
            print(f"  SKIP — {out['error']}")
            report["per_symbol"][sym] = {"error": out["error"]}
            continue

        is_s = _summarize(out["is"])
        oos_s = _summarize(out["oos"])

        print(f"  IS  : trades={is_s['n_trades']:>4} "
              f"(L{is_s['n_long']}/S{is_s['n_short']}) "
              f"wr={is_s['win_rate']:.2f} "
              f"sh={is_s['sharpe_annualised']:>5.2f} "
              f"pf={is_s['profit_factor']:>4.2f} "
              f"tot={is_s['total_return']*100:>6.2f}% "
              f"dd={is_s['max_drawdown']*100:>5.2f}%")
        print(f"  OOS : trades={oos_s['n_trades']:>4} "
              f"(L{oos_s['n_long']}/S{oos_s['n_short']}) "
              f"wr={oos_s['win_rate']:.2f} "
              f"sh={oos_s['sharpe_annualised']:>5.2f} "
              f"pf={oos_s['profit_factor']:>4.2f} "
              f"tot={oos_s['total_return']*100:>6.2f}% "
              f"dd={oos_s['max_drawdown']*100:>5.2f}%")

        report["per_symbol"][sym] = {
            "split_ts": out["split_ts"],
            "is": is_s,
            "oos": oos_s,
        }

    # Aggregate verdict
    print("\n" + "=" * 72)
    print("  AGGREGATE VERDICT")
    print("=" * 72)
    oos_sharpes = []
    oos_total_returns = []
    oos_trades_total = 0
    for sym, d in report["per_symbol"].items():
        if "error" in d:
            continue
        oos_sharpes.append(d["oos"]["sharpe_annualised"])
        oos_total_returns.append(d["oos"]["total_return"])
        oos_trades_total += d["oos"]["n_trades"]

    if oos_sharpes:
        avg_oos_sharpe = sum(oos_sharpes) / len(oos_sharpes)
        # Simple equal-weight portfolio total return (arithmetic avg on log-like proxy)
        avg_oos_return = sum(oos_total_returns) / len(oos_total_returns)
        print(f"  Avg OOS Sharpe:       {avg_oos_sharpe:.2f}")
        print(f"  Avg OOS Total Return: {avg_oos_return*100:.2f}%")
        print(f"  OOS trades (total):   {oos_trades_total}")
        if avg_oos_sharpe > 1.0 and oos_trades_total >= 30:
            verdict = "DEPLOYABLE (passes paper-trade bar)"
        elif avg_oos_sharpe > 0.3 and oos_trades_total >= 30:
            verdict = "MARGINAL (needs tuning or more data)"
        elif oos_trades_total < 30:
            verdict = "INSUFFICIENT DATA (not enough OOS trades)"
        else:
            verdict = "NO EDGE"
        print(f"  VERDICT: {verdict}")
        report["verdict"] = verdict
        report["avg_oos_sharpe"] = round(avg_oos_sharpe, 2)
        report["oos_trades_total"] = oos_trades_total

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report: {output}")
    print("=" * 72)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+",
                   default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--oos-fraction", type=float, default=0.30)
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--hold-periods", type=int, default=3)
    p.add_argument("--output", default="fund_data/funding_fade_report.json")
    args = p.parse_args()
    run(
        symbols=args.symbols,
        days=args.days,
        oos_fraction=args.oos_fraction,
        entry_z=args.entry_z,
        hold_periods=args.hold_periods,
        output=args.output,
    )

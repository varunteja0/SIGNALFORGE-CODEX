#!/usr/bin/env python3
"""
SignalForge — Daily Report
============================
Run once per day to get a complete status of the trading system.
Shows: P&L, strategy health, divergence, go/no-go verdict.

Usage:
    python scripts/daily_report.py

Can be run as a cron job:
    0 0 * * * cd /Users/varunteja/SignalForge && python scripts/daily_report.py >> daily_reports.log
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np

JOURNAL_PATH = Path("fund_data/trade_journal.json")
STATE_PATH = Path("fund_data/live_state.json")
DIVERGENCE_PATH = Path("fund_data/divergence_log.json")


def load_json(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def main():
    now = datetime.now(timezone.utc)
    print(f"\n{'='*65}")
    print(f"  SIGNALFORGE — DAILY REPORT")
    print(f"  {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*65}")

    state = load_json(STATE_PATH) or {}
    journal = load_json(JOURNAL_PATH) or []
    div_data = load_json(DIVERGENCE_PATH)

    # ─── Portfolio Status ────────────────────────────────────
    capital = state.get("capital", 10000)
    initial = state.get("initial_capital", 10000)
    ret = (capital - initial) / initial if initial > 0 else 0
    iteration = state.get("iteration", 0)
    n_open = len(state.get("open_positions", []))

    print(f"\n  ─── PORTFOLIO ───")
    print(f"  Capital:    ${capital:,.2f} ({ret:+.2%})")
    print(f"  Iterations: {iteration}")
    print(f"  Open:       {n_open} positions")
    print(f"  Mode:       {'PAPER' if state.get('paper_mode', True) else 'LIVE'}")

    # ─── Trade Stats ─────────────────────────────────────────
    print(f"\n  ─── TRADES ───")
    if not journal:
        print(f"  No trades yet. System waiting for signals.")
        # Show what we're waiting for
        print(f"\n  ─── SIGNAL REQUIREMENTS ───")
        print(f"  funding_mr_v7:     Funding z-score must reach ±3.0")
        print(f"  extreme_spike:     Funding z-score must reach ±4.0 + high vol regime")
        print(f"  fund_vol_squeeze:  SOL-only BB squeeze <15th pctile + funding z ±1.5")
        print(f"  momentum_breakout: Donchian breakout (ETH) + ATR expansion + volume")
        print(f"\n  These are HIGH-CONVICTION strategies. They fire rarely = good.")
        print(f"  From backtest: ~131 trades over 365 days = ~2.5 trades/week average.")
    else:
        n = len(journal)
        pnls = [t["pnl"] for t in journal]
        wins = sum(1 for p in pnls if p > 0)
        gw = sum(p for p in pnls if p > 0)
        gl = sum(abs(p) for p in pnls if p <= 0)
        pf = gw / gl if gl > 0 else float("inf")

        print(f"  Total trades: {n}")
        print(f"  Win rate:     {wins/n:.0%}")
        print(f"  Profit factor:{pf:.2f}")
        print(f"  Total PnL:    ${sum(pnls):+,.2f}")
        print(f"  Avg PnL:      ${np.mean(pnls):+,.2f}")
        print(f"  Best trade:   ${max(pnls):+,.2f}")
        print(f"  Worst trade:  ${min(pnls):,.2f}")

        # Today's trades
        today = now.strftime("%Y-%m-%d")
        today_trades = [t for t in journal if t.get("exit_time", "").startswith(today)]
        if today_trades:
            today_pnl = sum(t["pnl"] for t in today_trades)
            print(f"\n  Today: {len(today_trades)} trades, ${today_pnl:+,.2f}")

        # Last 7 days
        week_ago = (now - timedelta(days=7)).isoformat()
        week_trades = [t for t in journal if t.get("exit_time", "") >= week_ago]
        if week_trades:
            week_pnl = sum(t["pnl"] for t in week_trades)
            print(f"  Last 7d: {len(week_trades)} trades, ${week_pnl:+,.2f}")

        # Per strategy
        print(f"\n  ─── STRATEGY BREAKDOWN ───")
        strats = {}
        for t in journal:
            strats.setdefault(t["strategy"], []).append(t["pnl"])

        for name, pnls in sorted(strats.items()):
            w = sum(1 for p in pnls if p > 0)
            gw = sum(p for p in pnls if p > 0)
            gl = sum(abs(p) for p in pnls if p <= 0)
            pf = gw / gl if gl > 0 else float("inf")
            total = sum(pnls)
            print(f"  {name:<25s} N={len(pnls):>3d} WR={w/len(pnls):.0%} PF={pf:.2f} PnL=${total:>+8.2f}")

    # ─── Divergence ──────────────────────────────────────────
    if div_data:
        comparisons = div_data if isinstance(div_data, list) else div_data.get("comparisons", [])
        executed = [d for d in comparisons if not d.get("missed", False)]
        missed = [d for d in comparisons if d.get("missed", False)]

        if executed:
            print(f"\n  ─── DIVERGENCE (Backtest vs Live) ───")
            avg_slip = np.mean([d.get("entry_slippage_bps", 0) for d in executed])
            avg_div = np.mean([d.get("pnl_divergence_pct", 0) for d in executed])
            print(f"  Tracked:       {len(executed)} trades")
            print(f"  Missed:        {len(missed)} signals")
            print(f"  Avg slip:      {avg_slip:.1f} bps {'✓' if avg_slip < 10 else '⚠ HIGH'}")
            print(f"  Avg PnL div:   {avg_div:+.1f}% {'✓' if abs(avg_div) < 20 else '⚠ DRIFT'}")

    # ─── Safety Rails ────────────────────────────────────────
    print(f"\n  ─── SAFETY STATUS ───")
    dd = max(0, (initial - capital) / initial) if initial > 0 else 0
    print(f"  Drawdown:    {dd:.1%} / 15% kill {'✓' if dd < 0.15 else '✗ HALTED'}")

    if journal:
        recent = journal[-8:]
        consec = 0
        for t in reversed(recent):
            if t.get("pnl", 0) < 0:
                consec += 1
            else:
                break
        print(f"  Consec loss: {consec} / 8 halt {'✓' if consec < 8 else '✗ PAUSED'}")
    else:
        print(f"  Consec loss: 0 / 8 halt ✓")

    # ─── Verdict ─────────────────────────────────────────────
    print(f"\n  ─── VERDICT ───")
    if not journal:
        n_needed = 30
        print(f"  📊 COLLECTING DATA — need {n_needed} trades for reliable assessment")
        print(f"     Current: 0/{n_needed}")
        print(f"     Expected: ~2-3 weeks at current signal frequency")
    elif len(journal) < 30:
        n_needed = 30
        cur_pf = gw / gl if gl > 0 else 0
        print(f"  📊 EARLY STAGE — {len(journal)}/{n_needed} trades collected")
        print(f"     Live PF: {cur_pf:.2f} (need 30+ trades to trust)")
    else:
        cur_pf = gw / gl if gl > 0 else 0
        if cur_pf > 1.5 and (dd < 0.10):
            print(f"  ✅ GO-LIVE READY — PF={cur_pf:.2f}, DD={dd:.1%}")
            print(f"     Consider: python scripts/go_live.py --live --capital 1000")
        elif cur_pf > 1.0:
            print(f"  🟡 MARGINAL — PF={cur_pf:.2f}")
            print(f"     Need more data or edge refinement")
        else:
            print(f"  🔴 NO EDGE IN LIVE — PF={cur_pf:.2f}")
            print(f"     Stop trading. Debug divergence. Check regime shift.")

    # ─── Next Actions ────────────────────────────────────────
    print(f"\n  ─── NEXT ACTIONS ───")
    if not journal:
        print(f"  1. Keep paper trader running (scripts/go_live.py)")
        print(f"  2. Check scan.py --signals daily for proximity")
        print(f"  3. Wait for first trades (patience = edge)")
    elif len(journal) < 30:
        print(f"  1. Keep running — {30 - len(journal)} more trades needed")
        print(f"  2. Check divergence_log.json for slippage drift")
        print(f"  3. DO NOT change strategies mid-validation")
    else:
        print(f"  1. Review strategy-level PF vs backtest")
        print(f"  2. Decision point: go live or debug")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()

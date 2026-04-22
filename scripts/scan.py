#!/usr/bin/env python3
"""
SignalForge — Signal Scanner & Trade Inspector
================================================
Shows EXACTLY what's happening right now:
  - Current funding rates, z-scores, regime
  - How close each strategy is to triggering
  - Recent trade journal analysis
  - Per-strategy edge breakdown

Usage:
    python scripts/inspect.py              # Full scan
    python scripts/inspect.py --journal    # Analyze trade journal only
    python scripts/inspect.py --signals    # Signal proximity only
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.fetcher import DataFetcher
from src.data.features import compute_all_features
from src.data.structural import StructuralDataFetcher
from src.regime.detector import RegimeDetector

import logging
logging.basicConfig(level=logging.WARNING)

ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
JOURNAL_PATH = Path("fund_data/trade_journal.json")


def scan_signals():
    """Show current market state and signal proximity for each strategy."""
    print("=" * 70)
    print("  SIGNALFORGE — LIVE SIGNAL SCANNER")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    fetcher = DataFetcher()
    struct = StructuralDataFetcher()
    detector = RegimeDetector()

    for sym in ASSETS:
        try:
            pdf = compute_all_features(fetcher.fetch(sym, "1h", days=365))
            df = struct.fetch_all(symbol=sym.replace("/", ""), price_df=pdf, days=365)
        except Exception as e:
            print(f"\n  {sym}: ERROR — {e}")
            continue

        price = float(df["close"].iloc[-1])
        atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else price * 0.02

        # Funding data
        fr = float(df["fund_funding_rate"].iloc[-1]) if "fund_funding_rate" in df.columns else 0
        fr_zscore = float(df["fund_funding_zscore"].iloc[-1]) if "fund_funding_zscore" in df.columns else 0

        # Regime
        try:
            detector.fit(df)
            regime = detector.detect(df)
            regime_str = regime.value if hasattr(regime, "value") else str(regime)
        except Exception:
            regime_str = "unknown"

        # Bollinger width for squeeze detection
        if "bb_width_20" in df.columns:
            bb_width = float(df["bb_width_20"].iloc[-1])
            bb_pctile = float((df["bb_width_20"] < bb_width).mean() * 100)
        else:
            bb_width = 0
            bb_pctile = 50

        # Volume
        vol_sma = float(df["volume"].rolling(20).mean().iloc[-1]) if "volume" in df.columns else 0
        vol_now = float(df["volume"].iloc[-1])
        vol_ratio = vol_now / vol_sma if vol_sma > 0 else 1

        # Donchian channel for momentum
        ch_high = float(df["high"].rolling(30).max().iloc[-1])
        ch_low = float(df["low"].rolling(30).min().iloc[-1])

        print(f"\n  {sym} @ ${price:,.2f}")
        print(f"  {'─'*50}")
        print(f"  Regime:        {regime_str}")
        print(f"  Funding:       {fr:.6f} (z={fr_zscore:+.2f})")
        print(f"  ATR:           ${atr:,.2f} ({atr/price*100:.2f}%)")
        print(f"  BB Width:      {bb_width:.4f} (pctile={bb_pctile:.0f}%)")
        print(f"  Volume:        {vol_ratio:.2f}x (vs 20-SMA)")
        print(f"  Donchian:      ${ch_low:,.2f} — ${ch_high:,.2f}")

        # Signal proximity for each strategy
        print(f"\n  Signal Proximity:")

        # 1. funding_mr_v7: needs |z| >= 3.0
        z_need = 3.0
        z_pct = min(abs(fr_zscore) / z_need * 100, 100)
        z_bar = "█" * int(z_pct / 5) + "░" * (20 - int(z_pct / 5))
        z_dir = "SHORT" if fr_zscore > 0 else "LONG"
        z_ready = "✓ ACTIVE" if abs(fr_zscore) >= z_need else f"{100-z_pct:.0f}% away"
        print(f"    funding_mr_v7:     [{z_bar}] {z_pct:5.1f}% ({z_dir}) — {z_ready}")

        # 2. extreme_spike: needs |z| >= 4.0 + velocity + high_volatility regime
        z4_pct = min(abs(fr_zscore) / 4.0 * 100, 100)
        z4_bar = "█" * int(z4_pct / 5) + "░" * (20 - int(z4_pct / 5))
        regime_ok = "✓" if regime_str == "high_volatility" else "✗"
        z4_ready = "✓ ACTIVE" if abs(fr_zscore) >= 4.0 and regime_str == "high_volatility" else f"z={z4_pct:.0f}% regime={regime_ok}"
        if sym == "BTC/USDT":
            z4_ready = "N/A (BTC excluded)"
        print(f"    extreme_spike:     [{z4_bar}] {z4_pct:5.1f}% — {z4_ready}")

        # 3. fund_vol_squeeze: needs bb_pctile <= 15 + |funding_z| >= 1.5 on SOL only
        sq_pct = 100.0 if bb_pctile <= 15 else min(15.0 / max(bb_pctile, 1e-10) * 100, 100)
        fz2_pct = min(abs(fr_zscore) / 1.5 * 100, 100)
        combo = min(sq_pct, fz2_pct)
        sq_bar = "█" * int(combo / 5) + "░" * (20 - int(combo / 5))
        sq_ready = "✓ ACTIVE" if sym == "SOL/USDT" and bb_pctile <= 15 and abs(fr_zscore) >= 1.5 else f"squeeze={sq_pct:.0f}% funding={fz2_pct:.0f}%"
        if sym != "SOL/USDT":
            sq_ready = "N/A (SOL only)"
            combo = 0.0
            sq_bar = "░" * 20
        print(f"    fund_vol_squeeze:  [{sq_bar}] {combo:5.1f}% — {sq_ready}")

        # 4. momentum_breakout: needs breakout + ATR expansion + volume
        breakout_pct = (price - ch_low) / (ch_high - ch_low) * 100 if ch_high > ch_low else 50
        atr_exp = float(df["atr_14"].iloc[-1] / df["atr_14"].rolling(30).mean().iloc[-1]) if "atr_14" in df.columns else 1
        atr_exp_pct = min(atr_exp / 1.5 * 100, 100)
        vol_pct = min(vol_ratio / 1.3 * 100, 100)
        mb_combo = min(atr_exp_pct, vol_pct)
        mb_bar = "█" * int(mb_combo / 5) + "░" * (20 - int(mb_combo / 5))
        near_break = breakout_pct > 95 or breakout_pct < 5
        mb_ready = "✓ NEAR BREAKOUT" if near_break and atr_exp >= 1.5 and vol_ratio >= 1.3 else f"atr={atr_exp_pct:.0f}% vol={vol_pct:.0f}%"
        if sym == "XRP/USDT":
            mb_ready = "N/A (XRP excluded)"
        print(f"    momentum_breakout: [{mb_bar}] {mb_combo:5.1f}% — {mb_ready}")


def analyze_journal():
    """Analyze the trade journal for edge insights."""
    if not JOURNAL_PATH.exists():
        print("No trade journal found. Run go_live.py first.")
        return

    trades = json.loads(JOURNAL_PATH.read_text())
    if not trades:
        print("Trade journal is empty.")
        return

    print("=" * 70)
    print("  TRADE JOURNAL ANALYSIS")
    print(f"  {len(trades)} trades recorded")
    print("=" * 70)

    # Per-strategy breakdown
    strats = {}
    for t in trades:
        name = t["strategy"]
        strats.setdefault(name, []).append(t)

    print(f"\n  {'Strategy':<25s} {'N':>5s} {'Win':>5s} {'PF':>7s} {'Avg PnL':>10s} {'Total':>10s}")
    print(f"  {'─'*25} {'─'*5} {'─'*5} {'─'*7} {'─'*10} {'─'*10}")

    for name, ts in sorted(strats.items()):
        pnls = [t["pnl"] for t in ts]
        wins = sum(1 for p in pnls if p > 0)
        gw = sum(p for p in pnls if p > 0)
        gl = sum(abs(p) for p in pnls if p <= 0)
        pf = gw / gl if gl > 0 else float("inf")
        total = sum(pnls)
        avg = np.mean(pnls)
        wr = wins / len(pnls) if pnls else 0
        print(f"  {name:<25s} {len(ts):>5d} {wr:>4.0%} {pf:>7.2f} ${avg:>+9.2f} ${total:>+9.2f}")

    # Exit reason analysis
    print(f"\n  Exit Reasons:")
    reasons = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        reasons.setdefault(r, []).append(t["pnl"])

    for reason, pnls in sorted(reasons.items()):
        avg = np.mean(pnls)
        print(f"    {reason:<10s} N={len(pnls):>3d} avg=${avg:+.2f} total=${sum(pnls):+.2f}")

    # Recent trades
    print(f"\n  Last 10 Trades:")
    for t in trades[-10:]:
        dir_str = "L" if t["direction"] == 1 else "S"
        sym = t["symbol"].split("/")[0]
        pnl_str = f"${t['pnl']:+.2f}"
        print(
            f"    {t['id']} {t['strategy']:<20s} {dir_str} {sym:<4s} "
            f"${t['entry_price']:>10,.2f}→${t['exit_price']:>10,.2f} "
            f"{pnl_str:>10s} ({t['exit_reason']}) {t['bars_held']}bars"
        )

    # Cumulative P&L
    cum_pnl = np.cumsum([t["pnl"] for t in trades])
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak
    max_dd = dd.min()

    print(f"\n  Cumulative P&L: ${cum_pnl[-1]:+,.2f}")
    print(f"  Max Drawdown:   ${max_dd:,.2f}")
    print(f"  Best Trade:     ${max(t['pnl'] for t in trades):+,.2f}")
    print(f"  Worst Trade:    ${min(t['pnl'] for t in trades):,.2f}")


def main():
    parser = argparse.ArgumentParser(description="SignalForge Inspector")
    parser.add_argument("--journal", action="store_true", help="Analyze trade journal only")
    parser.add_argument("--signals", action="store_true", help="Signal scanner only")
    args = parser.parse_args()

    if args.journal:
        analyze_journal()
    elif args.signals:
        scan_signals()
    else:
        scan_signals()
        print()
        analyze_journal()


if __name__ == "__main__":
    main()

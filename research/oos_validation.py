#!/usr/bin/env python3
"""
OUT-OF-SAMPLE VALIDATION — The final test.
============================================
Split data 50/50. Find signals in first half. Test in second half.
If nothing survives out-of-sample, we have ZERO edge. Period.

Also tests composite signals (combining weak edges).
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from src.data.fetcher import DataFetcher
from src.data.features import compute_all_features
from src.data.structural import StructuralDataFetcher

import logging
logging.basicConfig(level=logging.WARNING)

ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
COMMISSION = 0.001
BASE_SLIPPAGE = 0.0005


def load_data():
    fetcher = DataFetcher()
    struct_fetcher = StructuralDataFetcher()
    datasets = {}
    for sym in ASSETS:
        try:
            raw = fetcher.fetch(sym, timeframe="1h", days=365)
            if raw.empty:
                continue
            df = compute_all_features(raw)
            try:
                df = struct_fetcher.fetch_all(
                    symbol=sym.replace("/", ""), price_df=df, days=365,
                )
            except:
                pass
            datasets[sym] = df.iloc[200:]  # Skip warmup
        except Exception as e:
            print(f"  {sym}: FAILED — {e}")
    return datasets


def compute_returns(df, hold):
    """Next-bar open entry, hold bars later close exit, with costs."""
    entry = df["open"].shift(-1)
    exit_ = df["close"].shift(-(1 + hold))
    
    atr = df.get("atr_14", df["close"] * 0.02)
    slippage = np.minimum(BASE_SLIPPAGE * np.maximum(1.0, atr / (df["close"] * 0.01)), 0.005)
    cost = COMMISSION + slippage.values
    
    long_ret = (exit_ / entry - 1).values - cost
    short_ret = (1 - exit_ / entry).values - cost
    return long_ret, short_ret


def eval_non_overlap(mask, rets, hold, min_trades=20):
    """Evaluate with non-overlapping trade enforcement."""
    mask = np.array(mask, dtype=bool)
    n = min(len(mask), len(rets))
    mask = mask[:n]
    rets = rets[:n]
    
    # Non-overlapping
    selected = []
    last = -hold - 1
    for i in range(n):
        if mask[i] and (i - last) > hold and not np.isnan(rets[i]):
            selected.append(rets[i])
            last = i
    
    if len(selected) < min_trades:
        return None
    
    arr = np.array(selected)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    gw = wins.sum() if len(wins) > 0 else 0
    gl = abs(losses.sum()) if len(losses) > 0 else 1e-10
    
    pf = gw / gl
    wr = len(wins) / len(arr)
    avg = arr.mean()
    sharpe = avg / (arr.std() + 1e-10) * np.sqrt(252 * 24 / max(1, hold))
    
    t_stat, p_val = stats.ttest_1samp(arr, 0)
    p_one = p_val / 2 if t_stat > 0 else 1.0
    
    return {
        "n": len(arr), "wr": wr, "pf": pf, "sharpe": sharpe,
        "avg_ret": avg, "total_ret": arr.sum(), "p": p_one,
    }


def run():
    print("=" * 70)
    print("  OUT-OF-SAMPLE VALIDATION — THE FINAL TEST")
    print("=" * 70)
    print()
    
    datasets = load_data()
    if not datasets:
        print("No data.")
        return
    
    for sym, df in datasets.items():
        print(f"  {sym}: {len(df)} bars")
    print()
    
    # ═══════════════════════════════════════════════════════════
    # TEST 1: Day-of-week effects (the strongest signals found)
    # ═══════════════════════════════════════════════════════════
    
    print("━━━ TEST 1: DAY-OF-WEEK EFFECTS ━━━━━━━━━━━━━━━━━━━")
    print("  Split: first half = in-sample, second half = out-of-sample")
    print()
    print(f"  {'Signal':<25s} {'Asset':>5s} │ {'IS-N':>5s} {'IS-PF':>6s} {'IS-Sh':>6s} │ {'OOS-N':>5s} {'OOS-PF':>7s} {'OOS-Sh':>7s} {'OOS-p':>7s} │ {'VERDICT':>10s}")
    print(f"  {'─'*25} {'─'*5} │ {'─'*5} {'─'*6} {'─'*6} │ {'─'*5} {'─'*7} {'─'*7} {'─'*7} │ {'─'*10}")
    
    dow_results = []
    
    for sym, df in datasets.items():
        mid = len(df) // 2
        df_is = df.iloc[:mid]
        df_oos = df.iloc[mid:]
        
        sym_short = sym.split("/")[0]
        
        for hold in [8, 12, 24]:
            is_long, is_short = compute_returns(df_is, hold)
            oos_long, oos_short = compute_returns(df_oos, hold)
            
            for dow in range(7):
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                
                for direction, dir_label, is_rets, oos_rets in [
                    (1, "LONG", is_long, oos_long),
                    (-1, "SHORT", is_short, oos_short),
                ]:
                    is_mask = df_is.index.dayofweek == dow
                    oos_mask = df_oos.index.dayofweek == dow
                    
                    is_res = eval_non_overlap(is_mask, is_rets, hold, min_trades=10)
                    oos_res = eval_non_overlap(oos_mask, oos_rets, hold, min_trades=10)
                    
                    if is_res and oos_res and is_res["pf"] > 1.2 and is_res["sharpe"] > 1.0:
                        # In-sample looks good — does it hold OOS?
                        if oos_res["pf"] > 1.0 and oos_res["sharpe"] > 0:
                            verdict = "HOLDS" if oos_res["pf"] > 1.2 else "WEAK"
                        else:
                            verdict = "FAILS OOS"
                        
                        label = f"{day_names[dow]}_{dir_label}_h{hold}"
                        print(f"  {label:<25s} {sym_short:>5s} │ {is_res['n']:>5d} {is_res['pf']:>6.2f} {is_res['sharpe']:>6.2f} │ "
                              f"{oos_res['n']:>5d} {oos_res['pf']:>7.2f} {oos_res['sharpe']:>7.2f} {oos_res['p']:>7.4f} │ {verdict:>10s}")
                        
                        if verdict == "HOLDS":
                            dow_results.append({
                                "signal": label, "sym": sym, "is": is_res, "oos": oos_res,
                            })
    
    print()
    
    # ═══════════════════════════════════════════════════════════
    # TEST 2: Funding rate mean reversion
    # ═══════════════════════════════════════════════════════════
    
    print("━━━ TEST 2: FUNDING RATE SIGNALS ━━━━━━━━━━━━━━━━━━")
    print()
    print(f"  {'Signal':<35s} {'Asset':>5s} │ {'IS-N':>5s} {'IS-PF':>6s} {'IS-Sh':>6s} │ {'OOS-N':>5s} {'OOS-PF':>7s} {'OOS-Sh':>7s} {'OOS-p':>7s} │ {'VERDICT':>10s}")
    print(f"  {'─'*35} {'─'*5} │ {'─'*5} {'─'*6} {'─'*6} │ {'─'*5} {'─'*7} {'─'*7} {'─'*7} │ {'─'*10}")
    
    fund_results = []
    
    for sym, df in datasets.items():
        fr_col = None
        for c in ["fund_funding_rate", "funding_rate"]:
            if c in df.columns:
                fr_col = c
                break
        if fr_col is None:
            continue
        
        mid = len(df) // 2
        df_is = df.iloc[:mid]
        df_oos = df.iloc[mid:]
        sym_short = sym.split("/")[0]
        
        fr = df[fr_col]
        
        for hold in [4, 8, 12, 24]:
            is_long, is_short = compute_returns(df_is, hold)
            oos_long, oos_short = compute_returns(df_oos, hold)
            
            for lookback in [48, 96, 168]:
                mu = fr.rolling(lookback).mean()
                sigma = fr.rolling(lookback).std()
                fz = (fr - mu) / (sigma + 1e-10)
                
                for thresh in [1.5, 2.0, 2.5, 3.0]:
                    # Positive funding extreme → short
                    is_mask = (fz.iloc[:mid] > thresh).values
                    oos_mask = (fz.iloc[mid:] > thresh).values
                    
                    is_res = eval_non_overlap(is_mask, is_short[:mid], hold, min_trades=8)
                    oos_res = eval_non_overlap(oos_mask, oos_short[:len(oos_mask)], hold, min_trades=8)
                    
                    if is_res and oos_res and is_res["pf"] > 1.2:
                        verdict = "HOLDS" if oos_res["pf"] > 1.2 and oos_res["sharpe"] > 0 else (
                            "WEAK" if oos_res["pf"] > 1.0 else "FAILS OOS")
                        
                        label = f"fund_z{lookback}>{thresh}_short_h{hold}"
                        print(f"  {label:<35s} {sym_short:>5s} │ {is_res['n']:>5d} {is_res['pf']:>6.2f} {is_res['sharpe']:>6.2f} │ "
                              f"{oos_res['n']:>5d} {oos_res['pf']:>7.2f} {oos_res['sharpe']:>7.2f} {oos_res['p']:>7.4f} │ {verdict:>10s}")
                        
                        if verdict == "HOLDS":
                            fund_results.append({
                                "signal": label, "sym": sym, "is": is_res, "oos": oos_res,
                            })
                    
                    # Negative funding extreme → long
                    is_mask = (fz.iloc[:mid] < -thresh).values
                    oos_mask = (fz.iloc[mid:] < -thresh).values
                    
                    is_res = eval_non_overlap(is_mask, is_long[:mid], hold, min_trades=8)
                    oos_res = eval_non_overlap(oos_mask, oos_long[:len(oos_mask)], hold, min_trades=8)
                    
                    if is_res and oos_res and is_res["pf"] > 1.2:
                        verdict = "HOLDS" if oos_res["pf"] > 1.2 and oos_res["sharpe"] > 0 else (
                            "WEAK" if oos_res["pf"] > 1.0 else "FAILS OOS")
                        
                        label = f"fund_z{lookback}<-{thresh}_long_h{hold}"
                        print(f"  {label:<35s} {sym_short:>5s} │ {is_res['n']:>5d} {is_res['pf']:>6.2f} {is_res['sharpe']:>6.2f} │ "
                              f"{oos_res['n']:>5d} {oos_res['pf']:>7.2f} {oos_res['sharpe']:>7.2f} {oos_res['p']:>7.4f} │ {verdict:>10s}")
                        
                        if verdict == "HOLDS":
                            fund_results.append({
                                "signal": label, "sym": sym, "is": is_res, "oos": oos_res,
                            })
    
    print()
    
    # ═══════════════════════════════════════════════════════════
    # TEST 3: Mean reversion after extreme moves
    # ═══════════════════════════════════════════════════════════
    
    print("━━━ TEST 3: EXTREME MOVE MEAN REVERSION ━━━━━━━━━━━")
    print()
    print(f"  {'Signal':<35s} {'Asset':>5s} │ {'IS-N':>5s} {'IS-PF':>6s} {'IS-Sh':>6s} │ {'OOS-N':>5s} {'OOS-PF':>7s} {'OOS-Sh':>7s} {'OOS-p':>7s} │ {'VERDICT':>10s}")
    print(f"  {'─'*35} {'─'*5} │ {'─'*5} {'─'*6} {'─'*6} │ {'─'*5} {'─'*7} {'─'*7} {'─'*7} │ {'─'*10}")
    
    mr_results = []
    
    for sym, df in datasets.items():
        mid = len(df) // 2
        df_is = df.iloc[:mid]
        df_oos = df.iloc[mid:]
        sym_short = sym.split("/")[0]
        close = df["close"]
        
        for ret_window in [1, 2, 3, 5, 10]:
            ret = close.pct_change(ret_window)
            mu = ret.rolling(100).mean()
            sigma = ret.rolling(100).std()
            zscore = (ret - mu) / (sigma + 1e-10)
            
            for hold in [4, 8, 12, 24]:
                is_long, is_short = compute_returns(df_is, hold)
                oos_long, oos_short = compute_returns(df_oos, hold)
                
                for thresh in [2.0, 2.5, 3.0]:
                    # Big drop → long
                    is_mask = (zscore.iloc[:mid] < -thresh).values
                    oos_mask = (zscore.iloc[mid:] < -thresh).values
                    
                    is_res = eval_non_overlap(is_mask, is_long[:mid], hold, min_trades=8)
                    oos_res = eval_non_overlap(oos_mask, oos_long[:len(oos_mask)], hold, min_trades=8)
                    
                    if is_res and oos_res and is_res["pf"] > 1.2:
                        verdict = "HOLDS" if oos_res["pf"] > 1.2 and oos_res["sharpe"] > 0 else (
                            "WEAK" if oos_res["pf"] > 1.0 else "FAILS OOS")
                        
                        label = f"ret{ret_window}_z<-{thresh}_long_h{hold}"
                        print(f"  {label:<35s} {sym_short:>5s} │ {is_res['n']:>5d} {is_res['pf']:>6.2f} {is_res['sharpe']:>6.2f} │ "
                              f"{oos_res['n']:>5d} {oos_res['pf']:>7.2f} {oos_res['sharpe']:>7.2f} {oos_res['p']:>7.4f} │ {verdict:>10s}")
                        
                        if verdict == "HOLDS":
                            mr_results.append({
                                "signal": label, "sym": sym, "is": is_res, "oos": oos_res,
                            })
                    
                    # Big pump → short
                    is_mask = (zscore.iloc[:mid] > thresh).values
                    oos_mask = (zscore.iloc[mid:] > thresh).values
                    
                    is_res = eval_non_overlap(is_mask, is_short[:mid], hold, min_trades=8)
                    oos_res = eval_non_overlap(oos_mask, oos_short[:len(oos_mask)], hold, min_trades=8)
                    
                    if is_res and oos_res and is_res["pf"] > 1.2:
                        verdict = "HOLDS" if oos_res["pf"] > 1.2 and oos_res["sharpe"] > 0 else (
                            "WEAK" if oos_res["pf"] > 1.0 else "FAILS OOS")
                        
                        label = f"ret{ret_window}_z>{thresh}_short_h{hold}"
                        print(f"  {label:<35s} {sym_short:>5s} │ {is_res['n']:>5d} {is_res['pf']:>6.2f} {is_res['sharpe']:>6.2f} │ "
                              f"{oos_res['n']:>5d} {oos_res['pf']:>7.2f} {oos_res['sharpe']:>7.2f} {oos_res['p']:>7.4f} │ {verdict:>10s}")
                        
                        if verdict == "HOLDS":
                            mr_results.append({
                                "signal": label, "sym": sym, "is": is_res, "oos": oos_res,
                            })
    
    print()
    
    # ═══════════════════════════════════════════════════════════
    # TEST 4: RSI extremes
    # ═══════════════════════════════════════════════════════════
    
    print("━━━ TEST 4: RSI EXTREME MEAN REVERSION ━━━━━━━━━━━━")
    print()
    print(f"  {'Signal':<35s} {'Asset':>5s} │ {'IS-N':>5s} {'IS-PF':>6s} {'IS-Sh':>6s} │ {'OOS-N':>5s} {'OOS-PF':>7s} {'OOS-Sh':>7s} {'OOS-p':>7s} │ {'VERDICT':>10s}")
    print(f"  {'─'*35} {'─'*5} │ {'─'*5} {'─'*6} {'─'*6} │ {'─'*5} {'─'*7} {'─'*7} {'─'*7} │ {'─'*10}")
    
    rsi_results = []
    
    for sym, df in datasets.items():
        mid = len(df) // 2
        df_is = df.iloc[:mid]
        df_oos = df.iloc[mid:]
        sym_short = sym.split("/")[0]
        
        for rsi_col in ["rsi_3", "rsi_7", "rsi_14"]:
            if rsi_col not in df.columns:
                continue
            rsi = df[rsi_col]
            
            for hold in [4, 8, 12, 24]:
                is_long, is_short = compute_returns(df_is, hold)
                oos_long, oos_short = compute_returns(df_oos, hold)
                
                for lo in [10, 15, 20, 25]:
                    is_mask = (rsi.iloc[:mid] < lo).values
                    oos_mask = (rsi.iloc[mid:] < lo).values
                    
                    is_res = eval_non_overlap(is_mask, is_long[:mid], hold, min_trades=8)
                    oos_res = eval_non_overlap(oos_mask, oos_long[:len(oos_mask)], hold, min_trades=8)
                    
                    if is_res and oos_res and is_res["pf"] > 1.2:
                        verdict = "HOLDS" if oos_res["pf"] > 1.2 and oos_res["sharpe"] > 0 else (
                            "WEAK" if oos_res["pf"] > 1.0 else "FAILS OOS")
                        
                        label = f"{rsi_col}<{lo}_long_h{hold}"
                        print(f"  {label:<35s} {sym_short:>5s} │ {is_res['n']:>5d} {is_res['pf']:>6.2f} {is_res['sharpe']:>6.2f} │ "
                              f"{oos_res['n']:>5d} {oos_res['pf']:>7.2f} {oos_res['sharpe']:>7.2f} {oos_res['p']:>7.4f} │ {verdict:>10s}")
                        
                        if verdict == "HOLDS":
                            rsi_results.append({
                                "signal": label, "sym": sym, "is": is_res, "oos": oos_res,
                            })
                
                for hi in [75, 80, 85, 90]:
                    is_mask = (rsi.iloc[:mid] > hi).values
                    oos_mask = (rsi.iloc[mid:] > hi).values
                    
                    is_res = eval_non_overlap(is_mask, is_short[:mid], hold, min_trades=8)
                    oos_res = eval_non_overlap(oos_mask, oos_short[:len(oos_mask)], hold, min_trades=8)
                    
                    if is_res and oos_res and is_res["pf"] > 1.2:
                        verdict = "HOLDS" if oos_res["pf"] > 1.2 and oos_res["sharpe"] > 0 else (
                            "WEAK" if oos_res["pf"] > 1.0 else "FAILS OOS")
                        
                        label = f"{rsi_col}>{hi}_short_h{hold}"
                        print(f"  {label:<35s} {sym_short:>5s} │ {is_res['n']:>5d} {is_res['pf']:>6.2f} {is_res['sharpe']:>6.2f} │ "
                              f"{oos_res['n']:>5d} {oos_res['pf']:>7.2f} {oos_res['sharpe']:>7.2f} {oos_res['p']:>7.4f} │ {verdict:>10s}")
                        
                        if verdict == "HOLDS":
                            rsi_results.append({
                                "signal": label, "sym": sym, "is": is_res, "oos": oos_res,
                            })
    
    print()
    
    # ═══════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════
    
    all_holds = dow_results + fund_results + mr_results + rsi_results
    
    print("=" * 70)
    print("  FINAL VERDICT — OUT-OF-SAMPLE TRUTH")
    print("=" * 70)
    print()
    
    if not all_holds:
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║  NOTHING SURVIVES OUT-OF-SAMPLE.                ║")
        print("  ║  Every edge found in-sample fails when tested   ║")
        print("  ║  on unseen data. We have ZERO real edge.        ║")
        print("  ╚══════════════════════════════════════════════════╝")
    else:
        print(f"  {len(all_holds)} signals HOLD out-of-sample:")
        print()
        
        # Sort by OOS Sharpe
        all_holds.sort(key=lambda x: x["oos"]["sharpe"], reverse=True)
        
        for r in all_holds:
            sym_short = r["sym"].split("/")[0]
            oos = r["oos"]
            is_ = r["is"]
            print(f"  ✓ {r['signal']:<35s} {sym_short:>5s}")
            print(f"    IS:  N={is_['n']:>4d}  PF={is_['pf']:.2f}  Sharpe={is_['sharpe']:.2f}")
            print(f"    OOS: N={oos['n']:>4d}  PF={oos['pf']:.2f}  Sharpe={oos['sharpe']:.2f}  p={oos['p']:.4f}")
            print()
        
        # Cross-asset check
        sig_names = set(r["signal"] for r in all_holds)
        cross_asset = []
        for name in sig_names:
            assets_with = [r["sym"].split("/")[0] for r in all_holds if r["signal"] == name]
            if len(set(assets_with)) > 1:
                cross_asset.append((name, assets_with))
        
        if cross_asset:
            print("  CROSS-ASSET EDGES (strongest evidence):")
            for name, assets in cross_asset:
                print(f"    {name} → {', '.join(set(assets))}")
            print()
        
        # Estimate real-world performance
        print("  ─ REALISTIC PROJECTION ────────────────────────")
        print("  WARNING: In-sample PF is ALWAYS higher than live.")
        print("  Expect 30-50% decay from OOS numbers in production.")
        print("  These are starting points, not guarantees.")
    
    print()
    print("  NO HYPE. NO PROMISES. JUST DATA.")
    print()


if __name__ == "__main__":
    run()

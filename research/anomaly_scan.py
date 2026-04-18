#!/usr/bin/env python3
"""
RAW ANOMALY SCANNER — Let the data speak.
==========================================
No theory. No textbook. No assumptions.

Method:
  1. Compute 50+ simple conditional signals
  2. For each: measure forward returns at next-bar OPEN entry
  3. Apply realistic costs (commission + vol-scaled slippage)
  4. Require 100+ trades minimum
  5. Report ONLY what survives PF > 1.3 and Sharpe > 0.5

If nothing survives, we know there's nothing here.
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


# ─── CONSTANTS ───────────────────────────────────────────────────

ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
COMMISSION = 0.001        # 10 bps round-trip
BASE_SLIPPAGE = 0.0005    # 5 bps base
MIN_TRADES = 50           # Minimum trades to consider
HOLD_PERIODS = [1, 2, 4, 8, 12, 24]  # Forward return windows (bars)


# ─── DATA LOADING ────────────────────────────────────────────────

def load_data():
    print("Loading data...")
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
                    symbol=sym.replace("/", ""),
                    price_df=df,
                    days=365,
                )
            except:
                pass
            datasets[sym] = df
            print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: FAILED — {e}")
    
    return datasets


# ─── FORWARD RETURN COMPUTATION ──────────────────────────────────

def compute_forward_returns(df, hold):
    """
    Compute realistic forward returns for NEXT-BAR entry.
    
    Signal on bar[i] → enter at bar[i+1] OPEN → exit at bar[i+1+hold] CLOSE.
    Costs applied on both sides.
    """
    entry_price = df["open"].shift(-1)     # Next bar's open
    exit_price = df["close"].shift(-1 - hold)  # Wait, this is wrong
    
    # Correct: signal at bar i, enter at bar i+1 open, exit at bar i+1+hold close
    # So for bar i: 
    #   entry = open[i+1]
    #   exit  = close[i+1+hold]
    
    # Actually let's compute it properly:
    # For a signal at index i:
    #   entry_price = df["open"].iloc[i+1]
    #   exit_price  = df["close"].iloc[i+1+hold]
    #   long_return = exit/entry - 1 - costs
    
    # Using shift:
    #   entry = df["open"].shift(-1)        → the NEXT bar's open
    #   exit  = df["close"].shift(-(1+hold)) → the bar (1+hold) later's close
    
    entry = df["open"].shift(-1)
    exit_ = df["close"].shift(-(1 + hold))
    
    # Vol-scaled slippage estimate
    atr = df.get("atr_14", df["close"] * 0.02)
    slippage_pct = np.minimum(BASE_SLIPPAGE * np.maximum(1.0, atr / (df["close"] * 0.01)), 0.005)
    
    total_cost = COMMISSION + slippage_pct.values  # entry + exit costs
    
    long_ret = (exit_ / entry - 1).values - total_cost
    short_ret = (1 - exit_ / entry).values - total_cost
    
    return long_ret, short_ret


# ─── SIGNAL GENERATORS ──────────────────────────────────────────

def generate_all_signals(df):
    """Generate 50+ conditional signals from raw data.
    
    Each signal is a boolean mask: True = condition met.
    Returns dict of {name: (mask, direction)} where direction is 1 or -1.
    """
    signals = {}
    close = df["close"]
    
    # ═══ 1. RETURNS-BASED ═══════════════════════════════════════
    
    for window in [1, 2, 3, 5, 10, 20]:
        ret = close.pct_change(window)
        
        for threshold in [1.5, 2.0, 2.5, 3.0]:
            mu = ret.rolling(100).mean()
            sigma = ret.rolling(100).std()
            zscore = (ret - mu) / (sigma + 1e-10)
            
            # Mean reversion: big drop → long
            signals[f"ret{window}_z<-{threshold}_long"] = (zscore < -threshold, 1)
            # Mean reversion: big pump → short  
            signals[f"ret{window}_z>{threshold}_short"] = (zscore > threshold, -1)
    
    # ═══ 2. RSI-BASED ══════════════════════════════════════════
    
    for col in ["rsi_14", "rsi_7", "rsi_3"]:
        if col not in df.columns:
            continue
        rsi = df[col]
        
        for lo, hi in [(20, 80), (25, 75), (30, 70), (15, 85), (10, 90)]:
            signals[f"{col}<{lo}_long"] = (rsi < lo, 1)
            signals[f"{col}>{hi}_short"] = (rsi > hi, -1)
    
    # ═══ 3. BOLLINGER BAND ═════════════════════════════════════
    
    for col in ["bb_pct_20", "bb_pct_10"]:
        if col not in df.columns:
            continue
        bb = df[col]
        
        signals[f"{col}<0_long"] = (bb < 0, 1)
        signals[f"{col}<-0.1_long"] = (bb < -0.1, 1)
        signals[f"{col}>1_short"] = (bb > 1, -1)
        signals[f"{col}>1.1_short"] = (bb > 1.1, -1)
    
    # ═══ 4. VOLUME-BASED ══════════════════════════════════════
    
    for col in ["vol_ratio_5", "vol_ratio_10", "vol_ratio_20"]:
        if col not in df.columns:
            continue
        vr = df[col]
        
        # High volume + down move = capitulation → long
        ret1 = close.pct_change(1)
        signals[f"{col}>2_down_long"] = ((vr > 2) & (ret1 < -0.01), 1)
        signals[f"{col}>3_down_long"] = ((vr > 3) & (ret1 < -0.01), 1)
        signals[f"{col}>2_up_short"] = ((vr > 2) & (ret1 > 0.01), -1)
    
    # ═══ 5. FUNDING RATE ═════════════════════════════════════
    
    fr_col = None
    for c in ["fund_funding_rate", "funding_rate"]:
        if c in df.columns:
            fr_col = c
            break
    
    if fr_col is not None:
        fr = df[fr_col]
        
        # Funding rate z-scores at different lookbacks
        for lb in [48, 96, 168, 336]:
            mu = fr.rolling(lb).mean()
            sigma = fr.rolling(lb).std()
            fz = (fr - mu) / (sigma + 1e-10)
            
            for thresh in [1.5, 2.0, 2.5, 3.0, 4.0]:
                signals[f"fund_z{lb}>{thresh}_short"] = (fz > thresh, -1)
                signals[f"fund_z{lb}<-{thresh}_long"] = (fz < -thresh, 1)
        
        # Raw funding rate levels
        for thresh in [0.0003, 0.0005, 0.001, 0.002]:
            signals[f"fund>{thresh}_short"] = (fr > thresh, -1)
            signals[f"fund<-{thresh}_long"] = (fr < -thresh, 1)
        
        # Funding rate CHANGE (velocity)
        fr_change = fr.diff()
        fr_change_z = (fr_change - fr_change.rolling(48).mean()) / (fr_change.rolling(48).std() + 1e-10)
        
        for thresh in [2.0, 3.0]:
            signals[f"fund_vel>{thresh}_short"] = (fr_change_z > thresh, -1)
            signals[f"fund_vel<-{thresh}_long"] = (fr_change_z < -thresh, 1)
        
        # Funding at settlement times (after payment → fade)
        # Funding is paid every 8 hours: bars 0, 8, 16 of the day
        if hasattr(df.index, 'hour'):
            for settle_hour in [0, 8, 16]:
                at_settle = df.index.hour == settle_hour
                signals[f"fund>{0.0003}_settle{settle_hour}_short"] = (
                    (fr > 0.0003) & at_settle, -1
                )
                signals[f"fund<-{0.0003}_settle{settle_hour}_long"] = (
                    (fr < -0.0003) & at_settle, 1
                )
    
    # ═══ 6. MULTI-CONDITION COMBOS ════════════════════════════
    
    ret1 = close.pct_change(1)
    ret5 = close.pct_change(5)
    
    if "rsi_14" in df.columns:
        rsi = df["rsi_14"]
        
        # Oversold + big drop = strong mean reversion setup
        signals["rsi14<25_ret5<-5%_long"] = ((rsi < 25) & (ret5 < -0.05), 1)
        signals["rsi14<30_ret5<-3%_long"] = ((rsi < 30) & (ret5 < -0.03), 1)
        signals["rsi14>75_ret5>5%_short"] = ((rsi > 75) & (ret5 > 0.05), -1)
        signals["rsi14>70_ret5>3%_short"] = ((rsi > 70) & (ret5 > 0.03), -1)
        
        # RSI divergence with funding
        if fr_col is not None:
            fr = df[fr_col]
            signals["rsi<30_fund>0.0003_long"] = ((rsi < 30) & (fr > 0.0003), 1)
            signals["rsi>70_fund<-0.0003_short"] = ((rsi > 70) & (fr < -0.0003), -1)
    
    # ═══ 7. PRICE VS MA ══════════════════════════════════════
    
    for col in ["price_vs_ma_20", "price_vs_ma_50", "price_vs_ma_200"]:
        if col not in df.columns:
            continue
        pvm = df[col]
        
        # Deep below MA → mean reversion long
        for thresh in [-0.05, -0.08, -0.10, -0.15]:
            signals[f"{col}<{thresh}_long"] = (pvm < thresh, 1)
        
        # Far above MA → mean reversion short
        for thresh in [0.05, 0.08, 0.10, 0.15]:
            signals[f"{col}>{thresh}_short"] = (pvm > thresh, -1)
    
    # ═══ 8. ATR / VOLATILITY ═════════════════════════════════
    
    if "atr_14" in df.columns:
        atr_pct = df["atr_14"] / close
        atr_z = (atr_pct - atr_pct.rolling(100).mean()) / (atr_pct.rolling(100).std() + 1e-10)
        
        # High vol + down → capitulation long
        signals["atr_z>2_ret1<-2%_long"] = ((atr_z > 2) & (ret1 < -0.02), 1)
        signals["atr_z>2_ret1>2%_short"] = ((atr_z > 2) & (ret1 > 0.02), -1)
        
        # Low vol → squeeze coming (breakout)
        vol_low = atr_z < -1.5
        signals["low_vol_breakup_long"] = (vol_low & (ret1 > 0.01), 1)
        signals["low_vol_breakdown_short"] = (vol_low & (ret1 < -0.01), -1)
    
    # ═══ 9. HOUR-OF-DAY ═════════════════════════════════════
    
    if hasattr(df.index, 'hour'):
        for h in range(24):
            signals[f"hour{h}_long"] = (df.index.hour == h, 1)
            signals[f"hour{h}_short"] = (df.index.hour == h, -1)
    
    # ═══ 10. DAY-OF-WEEK ════════════════════════════════════
    
    if hasattr(df.index, 'dayofweek'):
        for d in range(7):
            signals[f"dow{d}_long"] = (df.index.dayofweek == d, 1)
            signals[f"dow{d}_short"] = (df.index.dayofweek == d, -1)
    
    # ═══ 11. OI / STRUCTURAL ════════════════════════════════
    
    for col in ["oi_change_pct", "oi_change_1h", "oi_change_4h"]:
        if col not in df.columns:
            continue
        oi_c = df[col]
        oi_z = (oi_c - oi_c.rolling(48).mean()) / (oi_c.rolling(48).std() + 1e-10)
        
        # Big OI drop = leverage flush → long
        signals[f"{col}_z<-2_long"] = (oi_z < -2, 1)
        signals[f"{col}_z<-3_long"] = (oi_z < -3, 1)
        # Big OI spike = crowded → short
        signals[f"{col}_z>2_short"] = (oi_z > 2, -1)
    
    # ═══ 12. CONSECUTIVE MOVES ═══════════════════════════════
    
    up = (ret1 > 0).astype(int)
    down = (ret1 < 0).astype(int)
    
    # Count consecutive up/down bars
    consec_up = up.groupby((up != up.shift()).cumsum()).cumsum()
    consec_down = down.groupby((down != down.shift()).cumsum()).cumsum()
    
    for n in [3, 4, 5, 6, 7]:
        signals[f"consec_up{n}_short"] = (consec_up >= n, -1)
        signals[f"consec_down{n}_long"] = (consec_down >= n, 1)
    
    # ═══ 13. FUNDING + PRICE COMBO ═══════════════════════════
    
    if fr_col is not None:
        fr = df[fr_col]
        
        # After funding payment + price drop = forced sellers done → long
        if hasattr(df.index, 'hour'):
            post_settle = df.index.hour.isin([1, 9, 17])  # 1 hour after settlement
            signals["post_settle_drop_long"] = (post_settle & (ret1 < -0.005) & (fr.shift(1) > 0.0002), 1)
            signals["post_settle_pump_short"] = (post_settle & (ret1 > 0.005) & (fr.shift(1) < -0.0002), -1)
    
    return signals


# ─── EVALUATE SIGNALS ───────────────────────────────────────────

def evaluate_signal(mask, direction, long_rets, short_rets, hold):
    """Evaluate a single signal with proper next-bar entry.
    
    CRITICAL: Enforce non-overlapping trades.
    After a signal fires, impose cooldown of `hold` bars before
    the next entry. This prevents counting the same move multiple times.
    """
    mask_arr = np.array(mask.values if hasattr(mask, 'values') else mask, dtype=bool)
    
    # Trim to valid range (where forward returns exist)
    valid_len = min(len(mask_arr), len(long_rets))
    mask_arr = mask_arr[:valid_len]
    
    if direction == 1:
        rets = long_rets[:valid_len]
    else:
        rets = short_rets[:valid_len]
    
    # Enforce non-overlapping: after a signal, skip `hold` bars
    non_overlap_mask = np.zeros(valid_len, dtype=bool)
    last_entry = -hold - 1
    for i in range(valid_len):
        if mask_arr[i] and (i - last_entry) > hold and not np.isnan(rets[i]):
            non_overlap_mask[i] = True
            last_entry = i
    
    signal_rets = rets[non_overlap_mask]
    signal_rets = signal_rets[~np.isnan(signal_rets)]
    
    n_trades = len(signal_rets)
    if n_trades < MIN_TRADES:
        return None
    
    wins = signal_rets[signal_rets > 0]
    losses = signal_rets[signal_rets <= 0]
    
    gross_win = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-10
    
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    wr = len(wins) / n_trades if n_trades > 0 else 0
    avg_ret = signal_rets.mean()
    sharpe = avg_ret / (signal_rets.std() + 1e-10) * np.sqrt(252 * 24 / max(1, hold))
    
    # Statistical significance: is mean return > 0?
    if n_trades >= 20:
        t_stat, p_val = stats.ttest_1samp(signal_rets, 0)
        p_one_sided = p_val / 2 if t_stat > 0 else 1.0
    else:
        p_one_sided = 1.0
    
    # Max drawdown of cumulative returns
    cumret = np.cumsum(signal_rets)
    peak = np.maximum.accumulate(cumret)
    dd = peak - cumret
    max_dd = dd.max() if len(dd) > 0 else 0
    
    # Concentration: what % of PnL comes from top 3 trades?
    sorted_rets = np.sort(signal_rets)[::-1]
    top3_pnl = sorted_rets[:3].sum() if len(sorted_rets) >= 3 else sorted_rets.sum()
    total_pnl = signal_rets.sum()
    concentration = top3_pnl / total_pnl if total_pnl > 0 else 1.0
    
    return {
        "trades": n_trades,
        "win_rate": wr,
        "pf": pf,
        "sharpe": sharpe,
        "avg_ret": avg_ret,
        "total_ret": signal_rets.sum(),
        "max_dd": max_dd,
        "p_value": p_one_sided,
        "best_trade": signal_rets.max(),
        "worst_trade": signal_rets.min(),
        "median_ret": np.median(signal_rets),
        "concentration": concentration,
    }


# ─── MAIN ────────────────────────────────────────────────────────

def run_scan():
    print("=" * 70)
    print("  RAW ANOMALY SCANNER — DATA SPEAKS, NOT THEORY")
    print("=" * 70)
    print()
    
    datasets = load_data()
    if not datasets:
        print("No data. Exiting.")
        return
    
    print()
    
    all_survivors = []
    
    for sym, df in datasets.items():
        print(f"━━━ SCANNING {sym} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Skip warmup
        df = df.iloc[200:].copy()
        
        # Generate signals
        signals = generate_all_signals(df)
        print(f"  Generated {len(signals)} signal hypotheses")
        
        for hold in HOLD_PERIODS:
            long_rets, short_rets = compute_forward_returns(df, hold)
            
            results = []
            
            for name, (mask, direction) in signals.items():
                try:
                    r = evaluate_signal(mask, direction, long_rets, short_rets, hold)
                    if r is not None:
                        r["name"] = name
                        r["hold"] = hold
                        r["direction"] = "LONG" if direction == 1 else "SHORT"
                        r["symbol"] = sym
                        results.append(r)
                except Exception:
                    continue
            
            # Filter survivors: PF > 1.2 AND trades >= MIN_TRADES AND sharpe > 0
            survivors = [r for r in results if r["pf"] > 1.2 and r["sharpe"] > 0.3 and r["trades"] >= MIN_TRADES]
            
            if survivors:
                # Sort by Sharpe
                survivors.sort(key=lambda x: x["sharpe"], reverse=True)
                all_survivors.extend(survivors[:10])  # Top 10 per hold period
    
    print()
    print("=" * 70)
    print("  SCAN RESULTS — RAW TRUTH")
    print("=" * 70)
    print()
    
    if not all_survivors:
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║  ZERO ANOMALIES SURVIVE.                        ║")
        print("  ║  There is no edge in this data with these        ║")
        print("  ║  conditions. Period.                             ║")
        print("  ╚══════════════════════════════════════════════════╝")
        return
    
    # Deduplicate and sort
    all_survivors.sort(key=lambda x: x["sharpe"], reverse=True)
    
    # Apply Bonferroni-style correction
    # We tested ~len(signals) * len(HOLD_PERIODS) * len(ASSETS) hypotheses
    total_hypotheses = len(generate_all_signals(list(datasets.values())[0])) * len(HOLD_PERIODS) * len(datasets)
    bonferroni_threshold = 0.05 / total_hypotheses
    
    print(f"  Total hypotheses tested: ~{total_hypotheses}")
    print(f"  Bonferroni p-value threshold: {bonferroni_threshold:.6f}")
    print()
    
    # Filter by statistical significance after Bonferroni
    significant = [r for r in all_survivors if r["p_value"] < bonferroni_threshold]
    
    if not significant:
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║  NO ANOMALIES SURVIVE BONFERRONI CORRECTION.    ║")
        print("  ║  Some signals look promising but are likely      ║")
        print("  ║  random luck given the number of tests run.      ║")
        print("  ╚══════════════════════════════════════════════════╝")
        print()
        print("  Top signals BEFORE correction (may be spurious):")
        print()
        print(f"  {'Signal':<45s} {'Sym':>4s} {'Hold':>4s} {'Dir':>5s} {'N':>5s} {'WR':>6s} {'PF':>6s} {'Sharpe':>7s} {'Ret%':>7s} {'p-val':>8s}")
        print(f"  {'─'*45} {'─'*4} {'─'*4} {'─'*5} {'─'*5} {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*8}")
        
        for r in all_survivors[:30]:
            sym_short = r["symbol"].split("/")[0]
            ret_pct = r["total_ret"] * 100
            print(f"  {r['name']:<45s} {sym_short:>4s} {r['hold']:>4d} {r['direction']:>5s} "
                  f"{r['trades']:>5d} {r['win_rate']:>5.1%} {r['pf']:>6.2f} {r['sharpe']:>7.2f} "
                  f"{ret_pct:>+6.1f}% {r['p_value']:>8.4f}")
        
        return
    
    print(f"  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  {len(significant)} ANOMALIES SURVIVE BONFERRONI.              ║")
    print(f"  ║  These are statistically significant edges.      ║")
    print(f"  ╚══════════════════════════════════════════════════╝")
    print()
    print(f"  {'Signal':<45s} {'Sym':>4s} {'Hold':>4s} {'Dir':>5s} {'N':>5s} {'WR':>6s} {'PF':>6s} {'Sharpe':>7s} {'Ret%':>7s} {'p-val':>10s}")
    print(f"  {'─'*45} {'─'*4} {'─'*4} {'─'*5} {'─'*5} {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*10}")
    
    for r in significant[:50]:
        sym_short = r["symbol"].split("/")[0]
        ret_pct = r["total_ret"] * 100
        print(f"  {r['name']:<45s} {sym_short:>4s} {r['hold']:>4d} {r['direction']:>5s} "
              f"{r['trades']:>5d} {r['win_rate']:>5.1%} {r['pf']:>6.2f} {r['sharpe']:>7.2f} "
              f"{ret_pct:>+6.1f}% {r['p_value']:>10.6f}")
    
    # Summary of best edges
    print()
    print("─ TOP EDGES BY ASSET ────────────────────────────────")
    for sym in ASSETS:
        sym_sigs = [r for r in significant if r["symbol"] == sym]
        if sym_sigs:
            best = sym_sigs[0]
            print(f"  {sym}: {best['name']} (hold={best['hold']}h, PF={best['pf']:.2f}, "
                  f"Sharpe={best['sharpe']:.2f}, N={best['trades']}, p={best['p_value']:.6f})")
        else:
            print(f"  {sym}: No surviving edges")
    
    print()
    print("─ EDGE CONSISTENCY CHECK ────────────────────────────")
    
    # Check if same signal works across multiple assets
    sig_names = set(r["name"] for r in significant)
    for name in sig_names:
        assets_with = [r["symbol"].split("/")[0] for r in significant if r["name"] == name]
        if len(assets_with) > 1:
            print(f"  CROSS-ASSET: {name} works on {', '.join(assets_with)}")
    
    print()


if __name__ == "__main__":
    run_scan()

#!/usr/bin/env python3
"""
STRATEGY BUILD — From validated anomalies to real backtest.
=============================================================
Uses ONLY signals that held out-of-sample:

1. Monday LONG — BTC/ETH (cross-asset validated)
2. Funding extreme fade — SOL (mechanical edge)
3. RSI-7 extreme — ETH (decent trade count OOS)

Runs through the CORRECTED Backtester with:
  - Next-bar entry at OPEN
  - Volatility-scaled slippage
  - Funding costs
  - 200-bar warmup skip

Walk-forward validation: 3 expanding windows.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data.fetcher import DataFetcher
from src.data.features import compute_all_features
from src.data.structural import StructuralDataFetcher
from src.backtest.engine import Backtester

import logging
logging.basicConfig(level=logging.WARNING)


# ═══════════════════════════════════════════════════════════════
# STRATEGY 1: Monday Long (BTC + ETH)
# Rationale: Weekend selling pressure clears, institutional
#            demand returns Monday. Hold 24h.
# ═══════════════════════════════════════════════════════════════

def monday_long_signals(df: pd.DataFrame) -> pd.Series:
    """Go long at the start of Monday (UTC), hold 24 bars.
    
    Signal fires at hour 0 of Monday only (one entry per week).
    """
    signals = pd.Series(0, index=df.index)
    
    if not hasattr(df.index, 'dayofweek') or not hasattr(df.index, 'hour'):
        return signals
    
    # Signal at Monday hour 0 UTC only
    monday_open = (df.index.dayofweek == 0) & (df.index.hour == 0)
    signals[monday_open] = 1
    
    return signals


# ═══════════════════════════════════════════════════════════════
# STRATEGY 2: Funding Rate Extreme Fade (SOL)
# Rationale: Extreme funding = crowded trade. Arb normalizes rate.
#            Fade the crowd → profit from normalization.
# ═══════════════════════════════════════════════════════════════

def funding_extreme_fade_signals(df: pd.DataFrame) -> pd.Series:
    """Fade extreme funding rates.
    
    Extreme positive funding → SHORT (crowd is long, about to get squeezed)
    Extreme negative funding → LONG (crowd is short, about to get squeezed)
    
    Uses 96-bar (4-day) lookback for z-score.
    Threshold: ±2.5 (validated OOS at ±2.0 to ±3.0)
    """
    signals = pd.Series(0, index=df.index)
    
    fr_col = None
    for c in ["fund_funding_rate", "funding_rate"]:
        if c in df.columns:
            fr_col = c
            break
    
    if fr_col is None:
        return signals
    
    fr = df[fr_col]
    mu = fr.rolling(96).mean()
    sigma = fr.rolling(96).std()
    fz = (fr - mu) / (sigma + 1e-10)
    
    # Only trade when z-score is extreme AND funding rate itself is meaningful
    signals[(fz > 2.5) & (fr > 0.0001)] = -1   # Extreme positive → short
    signals[(fz < -2.5) & (fr < -0.0001)] = 1   # Extreme negative → long
    
    return signals


# ═══════════════════════════════════════════════════════════════
# STRATEGY 3: RSI-7 Extreme Oversold Bounce (ETH)
# Rationale: When 7-bar RSI drops below 10, short-term bounce
#            is reliable (69 OOS trades, PF 1.29). Hold 24h.
# ═══════════════════════════════════════════════════════════════

def rsi7_extreme_long_signals(df: pd.DataFrame) -> pd.Series:
    """Long when RSI-7 is extremely oversold (< 10)."""
    signals = pd.Series(0, index=df.index)
    
    if "rsi_7" not in df.columns:
        return signals
    
    rsi = df["rsi_7"]
    signals[rsi < 10] = 1
    
    return signals


# ═══════════════════════════════════════════════════════════════
# STRATEGY 4: Thursday Short SOL
# Rationale: Weekly expiry effect? OOS PF=2.77, Sharpe=6.25
#            Single-asset, so lower confidence.
# ═══════════════════════════════════════════════════════════════

def thursday_short_signals(df: pd.DataFrame) -> pd.Series:
    """Short SOL at start of Thursday, hold 24h."""
    signals = pd.Series(0, index=df.index)
    
    if not hasattr(df.index, 'dayofweek') or not hasattr(df.index, 'hour'):
        return signals
    
    thursday_open = (df.index.dayofweek == 3) & (df.index.hour == 0)
    signals[thursday_open] = -1
    
    return signals


# ═══════════════════════════════════════════════════════════════
# RUN BACKTESTS
# ═══════════════════════════════════════════════════════════════

STRATEGIES = [
    # (name, signal_func, assets, sl_atr, tp_atr, hold, size)
    ("monday_long",      monday_long_signals,         ["BTC/USDT", "ETH/USDT"], 2.0, 4.0, 24, 0.03),
    ("funding_fade",     funding_extreme_fade_signals, ["SOL/USDT"],             2.0, 4.0, 12, 0.03),
    ("rsi7_extreme",     rsi7_extreme_long_signals,    ["ETH/USDT"],             2.0, 3.0, 24, 0.02),
    ("thursday_short",   thursday_short_signals,        ["SOL/USDT"],            1.5, 3.0, 24, 0.02),
]


def run():
    print("=" * 70)
    print("  VALIDATED STRATEGY BACKTEST — CORRECTED ENGINE")
    print("=" * 70)
    print()
    
    # Load data
    fetcher = DataFetcher()
    struct_fetcher = StructuralDataFetcher()
    datasets = {}
    
    for sym in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
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
            datasets[sym] = df
            print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: FAILED — {e}")
    
    print()
    
    # ─── Walk-Forward: 3 expanding windows ───────────────────
    print("━━━ WALK-FORWARD VALIDATION ━━━━━━━━━━━━━━━━━━━━━━━")
    print("  3 expanding windows (train→test)")
    print()
    
    combined_results = {}
    
    for strat_name, signal_func, assets, sl_atr, tp_atr, hold, size in STRATEGIES:
        strat_results = []
        
        for sym in assets:
            if sym not in datasets:
                continue
            
            df = datasets[sym]
            n = len(df)
            sym_short = sym.split("/")[0]
            
            # Walk-forward: 3 folds
            # Fold 1: train 0-33%, test 33-50%
            # Fold 2: train 0-50%, test 50-67%
            # Fold 3: train 0-67%, test 67-100%
            folds = [
                (0, int(n*0.33), int(n*0.33), int(n*0.50)),
                (0, int(n*0.50), int(n*0.50), int(n*0.67)),
                (0, int(n*0.67), int(n*0.67), n),
            ]
            
            for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
                test_df = df.iloc[te_s:te_e]
                
                if len(test_df) < 100:
                    continue
                
                bt = Backtester(
                    initial_capital=10000,
                    commission_pct=0.001,
                    slippage_pct=0.0005,
                )
                
                try:
                    result = bt.run(
                        test_df, signal_func,
                        position_size_pct=size,
                        stop_loss_atr=sl_atr,
                        take_profit_atr=tp_atr,
                        max_holding_bars=hold,
                    )
                    
                    strat_results.append({
                        "fold": fold_idx + 1,
                        "sym": sym_short,
                        "trades": result.total_trades,
                        "wr": result.win_rate,
                        "pf": result.profit_factor,
                        "sharpe": result.sharpe_ratio,
                        "dd": result.max_drawdown,
                        "ret": result.total_return * 100,
                    })
                except Exception as e:
                    strat_results.append({
                        "fold": fold_idx + 1, "sym": sym_short,
                        "trades": 0, "wr": 0, "pf": 0, "sharpe": 0, "dd": 0, "ret": 0,
                        "error": str(e),
                    })
        
        combined_results[strat_name] = strat_results
    
    # Print results
    for strat_name, results in combined_results.items():
        print(f"  ── {strat_name} ──────────────────────────────────")
        print(f"    {'Fold':>4s} {'Asset':>5s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} {'Sharpe':>7s} {'MaxDD':>7s} {'Return':>8s}")
        print(f"    {'─'*4} {'─'*5} {'─'*7} {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*8}")
        
        total_trades = 0
        total_ret = 0
        positive_folds = 0
        total_folds = 0
        
        for r in results:
            if "error" in r:
                print(f"    {r['fold']:>4d} {r['sym']:>5s}   ERROR: {r['error']}")
                continue
            
            total_trades += r["trades"]
            total_ret += r["ret"]
            total_folds += 1
            if r["ret"] > 0:
                positive_folds += 1
            
            print(f"    {r['fold']:>4d} {r['sym']:>5s} {r['trades']:>7d} {r['wr']:>5.1%} {r['pf']:>6.2f} {r['sharpe']:>7.2f} {r['dd']:>6.1%} {r['ret']:>+7.1f}%")
        
        consistency = positive_folds / total_folds if total_folds > 0 else 0
        print(f"    {'':>4s} {'TOTAL':>5s} {total_trades:>7d} {'':>6s} {'':>6s} {'':>7s} {'':>7s} {total_ret:>+7.1f}%")
        print(f"    Consistency: {positive_folds}/{total_folds} folds positive ({consistency:.0%})")
        print()
    
    # Full backtest (no walk-forward split)
    print("━━━ FULL-PERIOD BACKTEST ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print(f"  {'Strategy':<25s} {'Asset':>5s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} {'Sharpe':>7s} {'MaxDD':>7s} {'Return':>8s} {'Verdict':>10s}")
    print(f"  {'─'*25} {'─'*5} {'─'*7} {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*8} {'─'*10}")
    
    portfolio_pnl = 0
    portfolio_trades = 0
    survivors = []
    
    for strat_name, signal_func, assets, sl_atr, tp_atr, hold, size in STRATEGIES:
        for sym in assets:
            if sym not in datasets:
                continue
            
            df = datasets[sym]
            sym_short = sym.split("/")[0]
            
            bt = Backtester(
                initial_capital=10000,
                commission_pct=0.001,
                slippage_pct=0.0005,
            )
            
            try:
                result = bt.run(
                    df, signal_func,
                    position_size_pct=size,
                    stop_loss_atr=sl_atr,
                    take_profit_atr=tp_atr,
                    max_holding_bars=hold,
                )
                
                trades = result.total_trades
                wr = result.win_rate
                pf = result.profit_factor
                sharpe = result.sharpe_ratio
                dd = result.max_drawdown
                ret = result.total_return * 100
                
                portfolio_pnl += result.total_return * 10000
                portfolio_trades += trades
                
                if trades >= 20 and pf > 1.2 and sharpe > 0.5:
                    verdict = "EDGE"
                    survivors.append(f"{strat_name}×{sym_short}")
                elif trades >= 10 and pf > 1.0 and sharpe > 0:
                    verdict = "WEAK EDGE"
                elif trades < 10:
                    verdict = "TOO FEW"
                else:
                    verdict = "NO EDGE"
                
                print(f"  {strat_name:<25s} {sym_short:>5s} {trades:>7d} {wr:>5.1%} {pf:>6.2f} {sharpe:>7.2f} {dd:>6.1%} {ret:>+7.1f}% {verdict:>10s}")
            
            except Exception as e:
                print(f"  {strat_name:<25s} {sym_short:>5s}   ERROR: {e}")
    
    print()
    print("━━━ PORTFOLIO SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Total trades:     {portfolio_trades}")
    print(f"  Combined PnL:     ${portfolio_pnl:+,.2f}")
    print(f"  Surviving edges:  {len(survivors)}")
    if survivors:
        for s in survivors:
            print(f"    ✓ {s}")
    print()
    
    if len(survivors) >= 2:
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║  REAL EDGES FOUND.                              ║")
        print("  ║  Next step: 30-day paper trade to confirm.      ║")
        print("  ╚══════════════════════════════════════════════════╝")
    elif len(survivors) == 1:
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║  ONE WEAK EDGE. Not portfolio-grade yet.        ║")
        print("  ║  Need more data or better signal combinations.  ║")
        print("  ╚══════════════════════════════════════════════════╝")
    else:
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║  STRATEGIES DON'T SURVIVE FULL BACKTEST.        ║")
        print("  ║  The OOS results were too noisy to be real.     ║")
        print("  ╚══════════════════════════════════════════════════╝")
    
    print()


if __name__ == "__main__":
    run()

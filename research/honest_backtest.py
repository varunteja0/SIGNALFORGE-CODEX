#!/usr/bin/env python3
"""
HONEST BACKTEST — No lies, no inflation, no excuses.
=====================================================
Runs every strategy through the CORRECTED engine with:
  - Next-bar entry at OPEN (not same-bar close)
  - Volatility-scaled slippage (10x cap)
  - Funding cost model for perp futures
  - 200-bar warmup skip
  - No fillna(0) phantom signals

Reports RAW numbers. If nothing survives, we know we have nothing.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.fetcher import DataFetcher
from src.data.features import compute_all_features
from src.data.structural import StructuralDataFetcher
from src.backtest.engine import Backtester

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("HonestBacktest")


# ─── Strategy Signal Generators ──────────────────────────────────

def funding_mr_v7_signals(df: pd.DataFrame) -> pd.Series:
    """Funding rate mean reversion — enter when funding is extreme."""
    signals = pd.Series(0, index=df.index)
    
    col = None
    for c in ["fund_funding_zscore", "funding_zscore", "funding_rate_zscore"]:
        if c in df.columns:
            col = c
            break
    
    if col is None:
        # Compute from raw funding rate if available
        for c in ["fund_funding_rate", "funding_rate"]:
            if c in df.columns:
                fr = df[c]
                mu = fr.rolling(168).mean()
                sigma = fr.rolling(168).std()
                zscore = (fr - mu) / (sigma + 1e-10)
                signals[zscore > 3.0] = -1   # Extreme positive funding → short
                signals[zscore < -3.0] = 1   # Extreme negative funding → long
                return signals
        return signals
    
    zscore = df[col]
    signals[zscore > 3.0] = -1
    signals[zscore < -3.0] = 1
    return signals


def extreme_spike_signals(df: pd.DataFrame) -> pd.Series:
    """Extreme funding spike — only very high conviction."""
    signals = pd.Series(0, index=df.index)
    
    col = None
    for c in ["fund_funding_zscore", "funding_zscore", "funding_rate_zscore"]:
        if c in df.columns:
            col = c
            break
    
    if col is None:
        for c in ["fund_funding_rate", "funding_rate"]:
            if c in df.columns:
                fr = df[c]
                mu = fr.rolling(96).mean()
                sigma = fr.rolling(96).std()
                zscore = (fr - mu) / (sigma + 1e-10)
                signals[zscore > 4.0] = -1
                signals[zscore < -4.0] = 1
                return signals
        return signals
    
    zscore = df[col]
    signals[zscore > 4.0] = -1
    signals[zscore < -4.0] = 1
    return signals


def momentum_breakout_signals(df: pd.DataFrame) -> pd.Series:
    """Channel breakout with volume confirmation."""
    signals = pd.Series(0, index=df.index)
    
    if "close" not in df.columns:
        return signals
    
    close = df["close"]
    high_30 = close.rolling(30).max()
    low_30 = close.rolling(30).min()
    
    atr_col = "atr_14" if "atr_14" in df.columns else "atr"
    if atr_col not in df.columns:
        return signals
    
    atr = df[atr_col]
    
    vol_col = None
    for c in ["vol_ratio_5", "volume_ratio", "vol_ratio_10"]:
        if c in df.columns:
            vol_col = c
            break
    
    vol_confirm = df[vol_col] > 1.3 if vol_col else pd.Series(True, index=df.index)
    
    # Breakout above channel high + ATR expansion
    atr_pct = atr / close
    atr_expanding = atr_pct > atr_pct.rolling(20).mean() * 1.5
    
    signals[(close > high_30.shift(1)) & vol_confirm & atr_expanding] = 1
    signals[(close < low_30.shift(1)) & vol_confirm & atr_expanding] = -1
    
    return signals


def contrarian_asym_signals(df: pd.DataFrame) -> pd.Series:
    """Contrarian asymmetry — SHORT when funding persistently positive."""
    signals = pd.Series(0, index=df.index)
    
    col = None
    for c in ["fund_funding_zscore", "funding_zscore"]:
        if c in df.columns:
            col = c
            break
    
    if col is None:
        for c in ["fund_funding_rate", "funding_rate"]:
            if c in df.columns:
                fr = df[c]
                mu = fr.rolling(168).mean()
                sigma = fr.rolling(168).std()
                zscore = (fr - mu) / (sigma + 1e-10)
                # Short when crowd is aggressively long (positive funding)
                signals[zscore > 2.0] = -1
                return signals
        return signals
    
    zscore = df[col]
    signals[zscore > 2.0] = -1
    return signals


def simple_rsi_mr_signals(df: pd.DataFrame) -> pd.Series:
    """Simple RSI mean reversion — baseline benchmark."""
    signals = pd.Series(0, index=df.index)
    
    rsi_col = None
    for c in ["rsi_14", "rsi"]:
        if c in df.columns:
            rsi_col = c
            break
    
    if rsi_col is None:
        return signals
    
    rsi = df[rsi_col]
    signals[rsi < 25] = 1    # Oversold → long
    signals[rsi > 75] = -1   # Overbought → short
    return signals


# ─── MAIN ────────────────────────────────────────────────────────

STRATEGIES = {
    "funding_mr_v7": {
        "func": funding_mr_v7_signals,
        "sl_atr": 2.0, "tp_atr": 4.0, "hold": 24, "size": 0.02,
    },
    "extreme_spike": {
        "func": extreme_spike_signals,
        "sl_atr": 1.5, "tp_atr": 3.0, "hold": 8, "size": 0.02,
    },
    "momentum_breakout": {
        "func": momentum_breakout_signals,
        "sl_atr": 2.0, "tp_atr": 4.0, "hold": 24, "size": 0.02,
    },
    "contrarian_asym": {
        "func": contrarian_asym_signals,
        "sl_atr": 1.5, "tp_atr": 3.0, "hold": 24, "size": 0.02,
    },
    "rsi_mr_baseline": {
        "func": simple_rsi_mr_signals,
        "sl_atr": 2.0, "tp_atr": 3.0, "hold": 12, "size": 0.02,
    },
}

ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def run_honest():
    print("=" * 70)
    print("  HONEST BACKTEST — CORRECTED ENGINE — NO LIES")
    print("=" * 70)
    print()
    print("  Engine corrections applied:")
    print("    ✓ Next-bar entry at OPEN (not same-bar close)")
    print("    ✓ Volatility-scaled slippage (up to 50 bps)")
    print("    ✓ Funding cost model for perp holding")
    print("    ✓ 200-bar warmup skip")
    print("    ✓ inf→NaN (not inf→0)")
    print()

    # Fetch data
    print("─ FETCHING DATA ─────────────────────────────────────")
    fetcher = DataFetcher()
    struct_fetcher = StructuralDataFetcher()
    datasets = {}

    for sym in ASSETS:
        try:
            raw = fetcher.fetch(sym, timeframe="1h", days=365)
            if raw.empty:
                print(f"  {sym}: NO DATA")
                continue
            df = compute_all_features(raw)
            # Fetch structural data (funding rates etc)
            try:
                df = struct_fetcher.fetch_all(
                    symbol=sym.replace("/", ""),
                    price_df=df,
                    days=365,
                )
            except Exception as e:
                print(f"  {sym}: structural data failed ({e}), using OHLCV features only")
            
            datasets[sym] = df
            print(f"  {sym}: {len(df)} bars, {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            
            # Check if funding data exists
            has_funding = any(c for c in df.columns if "funding" in c.lower() or "fund_" in c.lower())
            print(f"         Funding data: {'YES' if has_funding else 'NO'}")
            
        except Exception as e:
            print(f"  {sym}: FAILED — {e}")

    if not datasets:
        print("\n  FATAL: No data fetched. Cannot backtest. Network issue?")
        return

    print()

    # Run each strategy on each asset
    print("─ RESULTS ───────────────────────────────────────────")
    print(f"  {'Strategy × Asset':<35s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'PnL%':>8s} {'VERDICT':>10s}")
    print(f"  {'─'*35} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*10}")

    total_trades = 0
    total_pnl = 0
    surviving = []
    dead = []

    for strat_name, strat_cfg in STRATEGIES.items():
        for sym in ASSETS:
            if sym not in datasets:
                continue

            df = datasets[sym]
            
            try:
                bt = Backtester(
                    initial_capital=10000,
                    commission_pct=0.001,       # 10 bps commission
                    slippage_pct=0.0005,        # 5 bps base slippage
                )

                result = bt.run(
                    df,
                    strat_cfg["func"],
                    position_size_pct=strat_cfg["size"],
                    stop_loss_atr=strat_cfg["sl_atr"],
                    take_profit_atr=strat_cfg["tp_atr"],
                    max_holding_bars=strat_cfg["hold"],
                )

                trades = result.total_trades
                wr = result.win_rate
                pf = result.profit_factor
                sharpe = result.sharpe_ratio
                dd = result.max_drawdown
                pnl_pct = result.total_return * 100

                total_trades += trades
                total_pnl += (10000 * result.total_return)

                # Verdict
                if trades < 20:
                    verdict = "FEW TRADES"
                elif pf > 1.3 and sharpe > 0.5 and wr > 0.4:
                    verdict = "SURVIVES"
                    surviving.append(f"{strat_name} × {sym}")
                elif pf > 1.0:
                    verdict = "MARGINAL"
                else:
                    verdict = "DEAD"
                    dead.append(f"{strat_name} × {sym}")

                label = f"{strat_name} × {sym.split('/')[0]}"
                print(f"  {label:<35s} {trades:>7d} {wr:>6.1%} {pf:>7.2f} {sharpe:>7.2f} {dd:>6.1%} {pnl_pct:>+7.1f}% {verdict:>10s}")

            except Exception as e:
                label = f"{strat_name} × {sym.split('/')[0]}"
                print(f"  {label:<35s}   ERROR: {e}")

    print()
    print("─ HONEST VERDICT ────────────────────────────────────")
    print(f"  Total strategy×asset cells tested: {len(STRATEGIES) * len([s for s in ASSETS if s in datasets])}")
    print(f"  Total trades across all:           {total_trades}")
    print(f"  Combined PnL:                      ${total_pnl:+,.2f}")
    print()
    
    if surviving:
        print(f"  SURVIVING ({len(surviving)}):")
        for s in surviving:
            print(f"    ✓ {s}")
    else:
        print("  SURVIVING: NONE")
    
    print()
    
    if dead:
        print(f"  DEAD ({len(dead)}):")
        for s in dead:
            print(f"    ✗ {s}")
    
    print()
    
    if not surviving:
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║  CONCLUSION: NO EDGE FOUND.                     ║")
        print("  ║  All strategies fail under honest conditions.    ║")
        print("  ║  The previous results were inflated by biases.   ║")
        print("  ║  We need to find a REAL edge or stop pretending. ║")
        print("  ╚══════════════════════════════════════════════════╝")
    elif len(surviving) <= 2:
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║  CONCLUSION: WEAK EDGE.                         ║")
        print("  ║  A few cells survive but not portfolio-grade.    ║")
        print("  ║  Need more work before risking real capital.     ║")
        print("  ╚══════════════════════════════════════════════════╝")
    else:
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║  CONCLUSION: POTENTIAL EDGE EXISTS.              ║")
        print("  ║  Multiple cells survive honest conditions.       ║")
        print("  ║  Next: paper trade for 30 days to confirm.       ║")
        print("  ╚══════════════════════════════════════════════════╝")

    print()
    print("  NO HYPE. NO PROMISES. JUST NUMBERS.")
    print()


if __name__ == "__main__":
    run_honest()

#!/usr/bin/env python3
"""Test the liquidity sweep strategy across all parameter combos."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np, pandas as pd
from src.data.fetcher import DataFetcher
from src.data.features import compute_all_features
from src.data.structural import StructuralDataFetcher
from src.backtest.engine import Backtester
from src.engine.liquidity_sweep import LiquiditySweepTemplate

fetcher = DataFetcher()
struct = StructuralDataFetcher()
assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']

datasets = {}
for sym in assets:
    pdf = compute_all_features(fetcher.fetch(sym, '1h', days=365))
    df = struct.fetch_all(sym.replace('/', ''), pdf, days=365)
    datasets[sym] = df
    print(f"Loaded {sym}: {len(df)} bars")

configs = [
    dict(swing_lookback=48, wick_ratio=2.0, volume_mult=1.5, sweep_margin_pct=0.002, hold_bars=12, funding_confirm=True, label='default'),
    dict(swing_lookback=48, wick_ratio=1.5, volume_mult=1.0, sweep_margin_pct=0.001, hold_bars=12, funding_confirm=False, label='relaxed'),
    dict(swing_lookback=72, wick_ratio=2.5, volume_mult=1.5, sweep_margin_pct=0.003, hold_bars=16, funding_confirm=True, label='strict'),
    dict(swing_lookback=24, wick_ratio=1.5, volume_mult=1.3, sweep_margin_pct=0.001, hold_bars=8, funding_confirm=True, label='fast'),
    dict(swing_lookback=48, wick_ratio=2.0, volume_mult=1.3, sweep_margin_pct=0.002, hold_bars=24, funding_confirm=True, label='long_hold'),
    dict(swing_lookback=48, wick_ratio=2.0, volume_mult=1.5, sweep_margin_pct=0.002, hold_bars=12, funding_confirm=False, label='no_fund'),
]

print(f"\n{'Config':<15s} {'Asset':<8s} {'N':>4s} {'PF':>6s} {'WR':>6s} {'PnL':>10s}")
print('-' * 52)

best_pf = 0
best_info = None

for cfg in configs:
    label = cfg.pop('label')
    for sym in assets:
        df = datasets[sym]
        sig = LiquiditySweepTemplate.generate_signals(df, **cfg)
        n_sig = (sig != 0).sum()
        if n_sig < 3:
            continue
        
        bt = Backtester(commission_pct=0.001, slippage_pct=0.0005)
        res = bt.run(df, lambda d, s=sig: s, position_size_pct=0.01,
                     stop_loss_atr=1.5, take_profit_atr=3.0,
                     max_holding_bars=cfg.get('hold_bars', 12))
        
        if res.total_trades >= 5:
            pf = res.profit_factor
            wr = res.win_rate
            pnl = sum(t.pnl for t in res.trades)
            s = sym.split('/')[0]
            marker = ' *' if pf > 1.5 and res.total_trades >= 8 else ''
            print(f'  {label:<13s} {s:<8s} {res.total_trades:>4d} {pf:>6.2f} {wr:>5.0%} {pnl:>+10.2f}{marker}')
            
            if pf > best_pf and res.total_trades >= 8:
                best_pf = pf
                best_info = f"{label} x {sym} PF={pf:.2f} N={res.total_trades}"
    cfg['label'] = label

if best_info:
    print(f"\nBest: {best_info}")
else:
    print("\nNo config with PF > 1.0 and N >= 8 found")
    print("Liquidity sweep does NOT have robust edge on 1h timeframe")
    print("This is an honest result — don't force a strategy that doesn't work")

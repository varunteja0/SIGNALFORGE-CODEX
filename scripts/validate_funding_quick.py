"""Quick validation — writes results to file."""
import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.data.fetcher import DataFetcher
from src.data.features import compute_all_features
from src.data.structural import StructuralDataFetcher
from src.backtest.engine import Backtester
from src.engine.strategy_factory import FundingReversionTemplate
from src.regime.detector import RegimeDetector

PARAMS = {
    'funding_entry_zscore': 3.0, 'funding_lookback': 168, 'hold_bars': 24,
    'stop_atr_mult': 2.0, 'tp_atr_mult': 4.0, 'require_price_confirmation': False,
}
EDGE = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']

out = open('pipeline_output/funding_validation.txt', 'w')
def p(s=''):
    out.write(str(s) + '\n')
    out.flush()

def sigs(df, delay=0):
    s = FundingReversionTemplate.generate_signals(df, **PARAMS)
    if delay > 0:
        s = s.shift(delay).fillna(0).astype(int)
    return s

def bt_all(datasets, slip=0.0005, comm=0.001, delay=0):
    trades = []
    sharpes = []
    for sym, df in datasets.items():
        b = Backtester(initial_capital=10000, commission_pct=comm, slippage_pct=slip)
        sg = sigs(df, delay)
        r = b.run(df, lambda d, s=sg: s, position_size_pct=0.01,
                  stop_loss_atr=PARAMS['stop_atr_mult'],
                  take_profit_atr=PARAMS['tp_atr_mult'],
                  max_holding_bars=PARAMS['hold_bars'])
        trades.extend(r.trades)
        if r.total_trades > 0:
            sharpes.append(r.sharpe_ratio)
    w = [t for t in trades if t.pnl > 0]
    l = [t for t in trades if t.pnl <= 0]
    gw = sum(t.pnl for t in w)
    gl = sum(abs(t.pnl) for t in l)
    pf = gw/gl if gl > 0 else 0
    wr = len(w)/len(trades) if trades else 0
    pnl = sum(t.pnl for t in trades)
    sh = np.mean(sharpes) if sharpes else 0
    return len(trades), wr, pf, pnl/10000, sh

# Load data
p('Loading data...')
fetcher = DataFetcher()
struct = StructuralDataFetcher()
datasets = {}
for sym in EDGE:
    price_df = fetcher.fetch(sym, timeframe='1h', days=365)
    price_df = compute_all_features(price_df)
    datasets[sym] = struct.fetch_all(symbol=sym.replace('/', ''), price_df=price_df, days=365)
p('Data loaded.')

scorecard = {}

# TEST: Slippage
p('\n=== SLIPPAGE STRESS ===')
for m in [1.0, 2.0, 3.0, 5.0, 10.0]:
    n, wr, pf, ret, sh = bt_all(datasets, slip=0.0005*m)
    e = 'PASS' if pf > 1.0 else 'FAIL'
    p(f'  {m:.0f}x: trades={n}, WR={wr:.1%}, PF={pf:.2f}, ret={ret:+.2%}, sharpe={sh:+.2f} [{e}]')
    if m == 3.0:
        scorecard['slippage_3x'] = pf > 1.0

# TEST: Latency
p('\n=== LATENCY ===')
for d in [0, 1, 2, 3]:
    n, wr, pf, ret, sh = bt_all(datasets, delay=d)
    e = 'PASS' if pf > 1.0 else 'FAIL'
    p(f'  {d}bar: trades={n}, WR={wr:.1%}, PF={pf:.2f}, ret={ret:+.2%}, sharpe={sh:+.2f} [{e}]')
    if d == 1:
        scorecard['latency_1bar'] = pf > 1.0

# TEST: Commission
p('\n=== COMMISSION STRESS ===')
for m in [1.0, 2.0, 3.0, 5.0]:
    n, wr, pf, ret, sh = bt_all(datasets, comm=0.001*m)
    e = 'PASS' if pf > 1.0 else 'FAIL'
    p(f'  {m:.0f}x: trades={n}, WR={wr:.1%}, PF={pf:.2f}, ret={ret:+.2%} [{e}]')
    if m == 3.0:
        scorecard['commission_3x'] = pf > 1.0

# TEST: Walk-forward
p('\n=== WALK-FORWARD OOS ===')
oos_sharpes = []
for sym, df in datasets.items():
    n = len(df)
    min_train = n // 3
    step = (n - min_train) // 5
    sym_sharpes = []
    for fold in range(5):
        ts = min_train + fold * step
        te = min(ts + step, n)
        tdf = df.iloc[ts:te]
        if len(tdf) < 50:
            continue
        b = Backtester(initial_capital=10000)
        sg = sigs(tdf)
        r = b.run(tdf, lambda d, s=sg: s, position_size_pct=0.01,
                  stop_loss_atr=PARAMS['stop_atr_mult'],
                  take_profit_atr=PARAMS['tp_atr_mult'],
                  max_holding_bars=PARAMS['hold_bars'])
        sym_sharpes.append(r.sharpe_ratio)
        oos_sharpes.append(r.sharpe_ratio)
    prof = sum(1 for s in sym_sharpes if s > 0)
    p(f'  {sym}: {len(sym_sharpes)} folds, {prof}/{len(sym_sharpes)} profitable, avg_sharpe={np.mean(sym_sharpes):.2f}')

prof_folds = sum(1 for s in oos_sharpes if s > 0)
p(f'  TOTAL: {prof_folds}/{len(oos_sharpes)} profitable, avg_sharpe={np.mean(oos_sharpes):.2f}')
scorecard['walkforward'] = prof_folds / len(oos_sharpes) > 0.5 if oos_sharpes else False

# TEST: Regime
p('\n=== REGIME ANALYSIS ===')
regime_trades = {}
for sym, df in datasets.items():
    try:
        det = RegimeDetector()
        det.fit(df)
        regimes = det.predict(df)
    except Exception:
        continue
    b = Backtester(initial_capital=10000)
    sg = sigs(df)
    r = b.run(df, lambda d, s=sg: s, position_size_pct=0.01,
              stop_loss_atr=PARAMS['stop_atr_mult'],
              take_profit_atr=PARAMS['tp_atr_mult'],
              max_holding_bars=PARAMS['hold_bars'])
    for t in r.trades:
        idx = df.index.get_loc(t.entry_time)
        rg = regimes.iloc[idx] if idx < len(regimes) else "unknown"
        rn = rg.name if hasattr(rg, 'name') else str(rg)
        regime_trades.setdefault(rn, []).append(t)

prof_regimes = 0
for rn, tds in sorted(regime_trades.items()):
    w = [t for t in tds if t.pnl > 0]
    l = [t for t in tds if t.pnl <= 0]
    gw = sum(t.pnl for t in w)
    gl = sum(abs(t.pnl) for t in l)
    pf = gw/gl if gl > 0 else 0
    wr = len(w)/len(tds) if tds else 0
    e = 'PASS' if pf > 1.0 else 'FAIL'
    p(f'  {rn:<20s}: trades={len(tds)}, WR={wr:.1%}, PF={pf:.2f} [{e}]')
    if pf > 1.0:
        prof_regimes += 1
scorecard['regime'] = prof_regimes >= 2

# TEST: Clustering
p('\n=== TRADE CLUSTERING ===')
all_trades = []
for sym, df in datasets.items():
    b = Backtester(initial_capital=10000)
    sg = sigs(df)
    r = b.run(df, lambda d, s=sg: s, position_size_pct=0.01,
              stop_loss_atr=PARAMS['stop_atr_mult'],
              take_profit_atr=PARAMS['tp_atr_mult'],
              max_holding_bars=PARAMS['hold_bars'])
    all_trades.extend(r.trades)

pnls = sorted([t.pnl for t in all_trades], reverse=True)
total_pnl = sum(pnls)
top3 = sum(pnls[:3])
rest = total_pnl - top3
p(f'  Total PnL: ${total_pnl:.2f}')
p(f'  Top 3 PnL: ${top3:.2f}')
p(f'  Without top 3: ${rest:.2f}')
p(f'  Profitable without top 3: {"YES" if rest > 0 else "NO"}')
scorecard['clustering'] = rest > 0

# TEST: Monte Carlo
p('\n=== MONTE CARLO ===')
pnl_arr = np.array([t.pnl for t in all_trades])
rng = np.random.default_rng(42)
finals = []
for _ in range(5000):
    shuffled = rng.permutation(pnl_arr)
    eq = 10000 + np.cumsum(shuffled)
    finals.append(eq[-1])
finals = np.array(finals)
prob = (finals > 10000).mean()
p(f'  P(profit): {prob:.1%}')
p(f'  Median return: {np.median(finals)/10000 - 1:+.2%}')
p(f'  5th pct: {np.percentile(finals, 5)/10000 - 1:+.2%}')
p(f'  95th pct: {np.percentile(finals, 95)/10000 - 1:+.2%}')
scorecard['monte_carlo'] = prob > 0.6

# SCORECARD
p('\n' + '=' * 70)
p('  FINAL SCORECARD')
p('=' * 70)
passed = 0
for test, result in scorecard.items():
    status = 'PASS' if result else 'FAIL'
    p(f'  [{status}] {test}')
    if result:
        passed += 1

total = len(scorecard)
p(f'\n  Score: {passed}/{total}')
if passed >= 6:
    verdict = 'DEPLOY'
elif passed >= 4:
    verdict = 'PROMISING'
elif passed >= 2:
    verdict = 'WEAK'
else:
    verdict = 'REJECT'
p(f'  Verdict: {verdict}')

out.close()
print(f'Results written to pipeline_output/funding_validation.txt')

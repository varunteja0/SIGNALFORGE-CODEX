"""Institution-level validation of funding_mr_v7."""
import warnings; warnings.filterwarnings('ignore')
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.fetcher import DataFetcher
from src.data.features import compute_all_features
from src.data.structural import StructuralDataFetcher
from src.backtest.engine import Backtester
from src.engine.strategy_factory import FundingReversionTemplate
from src.regime.detector import RegimeDetector

PARAMS = dict(funding_entry_zscore=3.0, funding_lookback=168, hold_bars=24,
              stop_atr_mult=2.0, tp_atr_mult=4.0, require_price_confirmation=False)
EDGE_ASSETS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'pipeline_output', 'funding_validation.txt')

def sigs(df, delay=0):
    s = FundingReversionTemplate.generate_signals(df, **PARAMS)
    if delay > 0:
        s = s.shift(delay).fillna(0).astype(int)
    return s

def bt_all(datasets, slip=0.0005, comm=0.001, delay=0):
    trades, sharpes = [], []
    for sym, df in datasets.items():
        b = Backtester(initial_capital=10000, commission_pct=comm, slippage_pct=slip)
        sg = sigs(df, delay)
        r = b.run(df, lambda d, s=sg: s, position_size_pct=0.01,
                  stop_loss_atr=2.0, take_profit_atr=4.0, max_holding_bars=24)
        trades.extend(r.trades)
        if r.total_trades > 0:
            sharpes.append(r.sharpe_ratio)
    w = [t for t in trades if t.pnl > 0]
    l = [t for t in trades if t.pnl <= 0]
    gw = sum(t.pnl for t in w)
    gl = sum(abs(t.pnl) for t in l)
    pf = gw / gl if gl > 0 else 0
    wr = len(w) / len(trades) if trades else 0
    net = sum(t.pnl for t in trades) / 10000
    sh = np.mean(sharpes) if sharpes else 0
    return len(trades), wr, pf, net, sh, trades

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    lines = []
    def p(s=''):
        print(s, flush=True)
        lines.append(s)

    p('FUNDING_MR_V7 INSTITUTION-LEVEL VALIDATION')
    p('=' * 60)

    # Load data
    p('Loading data...')
    fetcher = DataFetcher()
    struct = StructuralDataFetcher()
    ds = {}
    for sym in EDGE_ASSETS:
        pdf = compute_all_features(fetcher.fetch(sym, timeframe='1h', days=365))
        ds[sym] = struct.fetch_all(symbol=sym.replace('/', ''), price_df=pdf, days=365)
        p(f'  {sym}: {len(ds[sym])} bars')

    R = []  # (test_name, passed_bool)

    # 1. BASELINE
    p()
    p('=== BASELINE ===')
    n, wr, pf, ret, sh, all_trades = bt_all(ds)
    p(f'  Trades={n}, WR={wr:.1%}, PF={pf:.2f}, Ret={ret:+.2%}, Sharpe={sh:.2f}')

    # 2. SLIPPAGE STRESS
    p()
    p('=== SLIPPAGE STRESS ===')
    for m in [1, 2, 3, 5, 10]:
        n2, wr2, pf2, ret2, sh2, _ = bt_all(ds, slip=0.0005 * m)
        tag = 'PASS' if pf2 > 1 else 'FAIL'
        p(f'  {m}x (={0.0005*m:.4f}): PF={pf2:.2f} WR={wr2:.1%} trades={n2} ret={ret2:+.2%} [{tag}]')
        if m == 3:
            R.append(('slippage_3x', pf2 > 1))

    # 3. LATENCY
    p()
    p('=== LATENCY (entry delay) ===')
    for d in [0, 1, 2, 3]:
        n2, wr2, pf2, ret2, sh2, _ = bt_all(ds, delay=d)
        tag = 'PASS' if pf2 > 1 else 'FAIL'
        p(f'  {d}-bar delay: PF={pf2:.2f} WR={wr2:.1%} trades={n2} [{tag}]')
        if d == 1:
            R.append(('latency_1bar', pf2 > 1))

    # 4. COMMISSION
    p()
    p('=== COMMISSION STRESS ===')
    for m in [1, 2, 3, 5]:
        n2, wr2, pf2, ret2, sh2, _ = bt_all(ds, comm=0.001 * m)
        tag = 'PASS' if pf2 > 1 else 'FAIL'
        p(f'  {m}x (={0.001*m:.3f}): PF={pf2:.2f} WR={wr2:.1%} [{tag}]')
        if m == 3:
            R.append(('commission_3x', pf2 > 1))

    # 5. WALK-FORWARD OOS
    p()
    p('=== WALK-FORWARD OOS (5-fold each asset) ===')
    all_oos = []
    for sym, df in ds.items():
        n_rows = len(df)
        min_train = n_rows // 3
        step = (n_rows - min_train) // 5
        folds = []
        for f in range(5):
            ts = min_train + f * step
            te = min(min_train + (f + 1) * step, n_rows)
            tdf = df.iloc[ts:te]
            if len(tdf) < 50:
                continue
            sg = sigs(tdf)
            r = Backtester(10000).run(tdf, lambda d, s=sg: s, position_size_pct=0.01,
                                      stop_loss_atr=2.0, take_profit_atr=4.0, max_holding_bars=24)
            folds.append(r.sharpe_ratio)
            all_oos.append(r.sharpe_ratio)
        prof = sum(1 for x in folds if x > 0)
        avg = np.mean(folds) if folds else 0
        p(f'  {sym}: {prof}/{len(folds)} profitable, avg_sharpe={avg:.2f}')
    pf_total = sum(1 for x in all_oos if x > 0)
    avg_total = np.mean(all_oos) if all_oos else 0
    p(f'  TOTAL: {pf_total}/{len(all_oos)} profitable, avg_oos_sharpe={avg_total:+.2f}')
    R.append(('walkforward_>50%', pf_total / len(all_oos) > 0.5 if all_oos else False))

    # 6. REGIME ANALYSIS
    p()
    p('=== REGIME ANALYSIS ===')
    regime_trades = {}
    for sym, df in ds.items():
        det = RegimeDetector()
        det.fit(df)
        regimes = det.get_regime_history(df)
        sg = sigs(df)
        r = Backtester(10000).run(df, lambda d, s=sg: s, position_size_pct=0.01,
                                  stop_loss_atr=2.0, take_profit_atr=4.0, max_holding_bars=24)
        for t in r.trades:
            idx = df.index.get_indexer([t.entry_time], method='nearest')[0]
            if 0 <= idx < len(regimes):
                rn = str(regimes.iloc[idx])
            else:
                rn = 'unknown'
            regime_trades.setdefault(rn, []).append(t)

    profitable_regimes = 0
    for rn, tds in sorted(regime_trades.items()):
        w = [t for t in tds if t.pnl > 0]
        l = [t for t in tds if t.pnl <= 0]
        gw = sum(t.pnl for t in w)
        gl = sum(abs(t.pnl) for t in l)
        rpf = gw / gl if gl > 0 else 0
        tag = 'PASS' if rpf > 1 else 'FAIL'
        p(f'  {rn:<20s}: trades={len(tds):3d}, PF={rpf:.2f}, WR={len(w)/len(tds):.1%} [{tag}]')
        if rpf > 1:
            profitable_regimes += 1
    R.append(('regime_profitable_2+', profitable_regimes >= 2))

    # 7. TRADE CLUSTERING (top-3 removal)
    p()
    p('=== TRADE CLUSTERING ===')
    pnls = sorted([t.pnl for t in all_trades], reverse=True)
    total_pnl = sum(pnls)
    top3_pnl = sum(pnls[:3])
    rest_pnl = total_pnl - top3_pnl
    p(f'  Total PnL: ${total_pnl:.2f}')
    p(f'  Top 3 trades: ${top3_pnl:.2f} ({top3_pnl/total_pnl*100:.0f}% of total)' if total_pnl > 0 else f'  Top 3 trades: ${top3_pnl:.2f}')
    p(f'  Remaining: ${rest_pnl:.2f}')
    p(f'  Edge survives without top 3: {"YES" if rest_pnl > 0 else "NO"}')
    R.append(('clustering_survives', rest_pnl > 0))

    # 8. MONTE CARLO
    p()
    p('=== MONTE CARLO (5000 shuffles) ===')
    pa = np.array([t.pnl for t in all_trades])
    rng = np.random.default_rng(42)
    finals = []
    for _ in range(5000):
        shuffled = rng.permutation(pa)
        eq = 10000 + np.cumsum(shuffled)
        finals.append(eq[-1])
    finals = np.array(finals)
    prob_profit = (finals > 10000).mean()
    median_ret = np.median(finals) / 10000 - 1
    p5 = np.percentile(finals, 5) / 10000 - 1
    p95 = np.percentile(finals, 95) / 10000 - 1
    max_dd_pcts = []
    for _ in range(1000):
        eq = 10000 + np.cumsum(rng.permutation(pa))
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        max_dd_pcts.append(dd.max())
    worst_dd = np.percentile(max_dd_pcts, 95)
    p(f'  P(profit): {prob_profit:.1%}')
    p(f'  Median return: {median_ret:+.2%}')
    p(f'  5th-95th pctile: {p5:+.2%} to {p95:+.2%}')
    p(f'  95th pctile max DD: {worst_dd:.1%}')
    R.append(('monte_carlo_>60%', prob_profit > 0.6))

    # SCORECARD
    p()
    p('=' * 60)
    p('  FINAL SCORECARD')
    p('=' * 60)
    passed = 0
    for name, ok in R:
        tag = 'PASS' if ok else 'FAIL'
        p(f'  [{tag}] {name}')
        if ok:
            passed += 1
    p(f'')
    p(f'  Score: {passed}/{len(R)}')
    if passed >= 6:
        verdict = 'DEPLOY — Edge is real. Ready for paper trading.'
    elif passed >= 4:
        verdict = 'PROMISING — Edge likely real but needs more data/tuning.'
    elif passed >= 2:
        verdict = 'WEAK — Some signal but high risk of overfitting.'
    else:
        verdict = 'REJECT — Insufficient evidence of real edge.'
    p(f'  Verdict: {verdict}')
    p('=' * 60)

    with open(OUT, 'w') as f:
        f.write('\n'.join(lines))
    p(f'\nResults saved to {OUT}')

if __name__ == '__main__':
    main()

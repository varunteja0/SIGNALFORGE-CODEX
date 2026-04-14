"""Institution-level validation of funding_mr_v7."""
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
    'funding_entry_zscore': 3.0,
    'funding_lookback': 168,
    'hold_bars': 24,
    'stop_atr_mult': 2.0,
    'tp_atr_mult': 4.0,
    'require_price_confirmation': False,
}
EDGE_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']


def make_signals(df, delay=0):
    s = FundingReversionTemplate.generate_signals(df, **PARAMS)
    if delay > 0:
        s = s.shift(delay).fillna(0).astype(int)
    return s


def load_datasets():
    fetcher = DataFetcher()
    struct = StructuralDataFetcher()
    datasets = {}
    for symbol in EDGE_SYMBOLS:
        price_df = fetcher.fetch(symbol, timeframe='1h', days=365)
        price_df = compute_all_features(price_df)
        datasets[symbol] = struct.fetch_all(
            symbol=symbol.replace('/', ''), price_df=price_df, days=365
        )
    return datasets


def run_combined(datasets, slip_pct=0.0005, comm_pct=0.001, delay=0):
    all_trades = []
    sharpes = []
    for symbol, df in datasets.items():
        bt = Backtester(initial_capital=10000, commission_pct=comm_pct, slippage_pct=slip_pct)
        signals = make_signals(df, delay)
        result = bt.run(
            df, lambda d, s=signals: s,
            position_size_pct=0.01, stop_loss_atr=PARAMS['stop_atr_mult'],
            take_profit_atr=PARAMS['tp_atr_mult'],
            max_holding_bars=PARAMS['hold_bars'],
        )
        all_trades.extend(result.trades)
        if result.total_trades > 0:
            sharpes.append(result.sharpe_ratio)

    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]
    gw = sum(t.pnl for t in wins)
    gl = sum(abs(t.pnl) for t in losses)
    pf = gw / gl if gl > 0 else 0
    wr = len(wins) / len(all_trades) if all_trades else 0
    total_pnl = sum(t.pnl for t in all_trades)
    avg_sh = np.mean(sharpes) if sharpes else 0
    return len(all_trades), wr, pf, total_pnl / 10000, avg_sh


def test_slippage(datasets):
    print('=' * 70)
    print('  TEST 2: Slippage Stress (4 edge assets)')
    print('=' * 70)
    print(f'{"Mult":>8s} {"Trades":>7s} {"WR":>7s} {"PF":>7s} {"Return":>8s} {"Sharpe":>8s}')
    print('-' * 50)
    results = []
    for m in [1.0, 2.0, 3.0, 5.0, 10.0]:
        n, wr, pf, ret, sh = run_combined(datasets, slip_pct=0.0005 * m)
        e = '✓' if pf > 1.0 else '✗'
        print(f'{e} {m:>5.0f}x {n:>7d} {wr:>6.1%} {pf:>7.2f} {ret:>+7.2%} {sh:>+8.2f}')
        results.append(pf > 1.0)
    return results


def test_latency(datasets):
    print()
    print('=' * 70)
    print('  TEST 3: Entry Latency (4 edge assets)')
    print('=' * 70)
    print(f'{"Delay":>8s} {"Trades":>7s} {"WR":>7s} {"PF":>7s} {"Return":>8s} {"Sharpe":>8s}')
    print('-' * 50)
    results = []
    for d in [0, 1, 2, 3]:
        n, wr, pf, ret, sh = run_combined(datasets, delay=d)
        e = '✓' if pf > 1.0 else '✗'
        print(f'{e} {d:>5d}bar {n:>7d} {wr:>6.1%} {pf:>7.2f} {ret:>+7.2%} {sh:>+8.2f}')
        results.append(pf > 1.0)
    return results


def test_commission(datasets):
    print()
    print('=' * 70)
    print('  TEST 4: Commission Stress (4 edge assets)')
    print('=' * 70)
    print(f'{"Mult":>8s} {"Trades":>7s} {"WR":>7s} {"PF":>7s} {"Return":>8s}')
    print('-' * 42)
    results = []
    for m in [1.0, 2.0, 3.0, 5.0]:
        n, wr, pf, ret, sh = run_combined(datasets, comm_pct=0.001 * m)
        e = '✓' if pf > 1.0 else '✗'
        print(f'{e} {m:>5.0f}x {n:>7d} {wr:>6.1%} {pf:>7.2f} {ret:>+7.2%}')
        results.append(pf > 1.0)
    return results


def test_walkforward(datasets, n_splits=5):
    print()
    print('=' * 70)
    print('  TEST 5: Walk-Forward OOS Validation')
    print('=' * 70)
    all_oos_sharpes = []
    all_oos_pfs = []

    for symbol, df in datasets.items():
        n = len(df)
        min_train = n // 3
        step = (n - min_train) // n_splits
        fold_results = []

        for fold in range(n_splits):
            test_start = min_train + fold * step
            test_end = min(test_start + step, n)
            test_df = df.iloc[test_start:test_end]
            if len(test_df) < 50:
                continue

            bt = Backtester(initial_capital=10000)
            signals = make_signals(test_df)
            result = bt.run(
                test_df, lambda d, s=signals: s,
                position_size_pct=0.01, stop_loss_atr=PARAMS['stop_atr_mult'],
                take_profit_atr=PARAMS['tp_atr_mult'],
                max_holding_bars=PARAMS['hold_bars'],
            )
            all_oos_sharpes.append(result.sharpe_ratio)
            all_oos_pfs.append(result.profit_factor)
            fold_results.append(result)

        sym_sharpes = [r.sharpe_ratio for r in fold_results]
        sym_profitable = sum(1 for s in sym_sharpes if s > 0)
        print(f'  {symbol}: {len(fold_results)} folds, '
              f'{sym_profitable}/{len(fold_results)} profitable, '
              f'avg Sharpe={np.mean(sym_sharpes):.2f}')

    profitable_folds = sum(1 for s in all_oos_sharpes if s > 0)
    total_folds = len(all_oos_sharpes)
    avg_oos_sharpe = np.mean(all_oos_sharpes) if all_oos_sharpes else 0
    avg_oos_pf = np.mean(all_oos_pfs) if all_oos_pfs else 0

    print(f'\n  OOS Summary:')
    print(f'    Avg OOS Sharpe:  {avg_oos_sharpe:+.3f}')
    print(f'    Avg OOS PF:      {avg_oos_pf:.2f}')
    print(f'    Profitable folds: {profitable_folds}/{total_folds}')
    return profitable_folds, total_folds, avg_oos_sharpe


def test_regime(datasets):
    print()
    print('=' * 70)
    print('  TEST 6: Regime Analysis')
    print('=' * 70)
    regime_trades = {}

    for symbol, df in datasets.items():
        try:
            detector = RegimeDetector()
            detector.fit(df)
            regimes = detector.predict(df)
        except Exception:
            continue

        bt = Backtester(initial_capital=10000)
        signals = make_signals(df)
        result = bt.run(
            df, lambda d, s=signals: s,
            position_size_pct=0.01, stop_loss_atr=PARAMS['stop_atr_mult'],
            take_profit_atr=PARAMS['tp_atr_mult'],
            max_holding_bars=PARAMS['hold_bars'],
        )

        for trade in result.trades:
            entry_idx = df.index.get_loc(trade.entry_time)
            regime = regimes.iloc[entry_idx] if entry_idx < len(regimes) else "unknown"
            regime_name = regime.name if hasattr(regime, 'name') else str(regime)
            if regime_name not in regime_trades:
                regime_trades[regime_name] = []
            regime_trades[regime_name].append(trade)

    print(f'  {"Regime":<20s} {"Trades":>7s} {"WR":>7s} {"PF":>7s} {"AvgPnL":>8s}')
    print('  ' + '-' * 55)

    profitable_regimes = 0
    for regime, trades in sorted(regime_trades.items()):
        if not trades:
            continue
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gw = sum(t.pnl for t in wins)
        gl = sum(abs(t.pnl) for t in losses)
        pf_r = gw / gl if gl > 0 else 0
        wr = len(wins) / len(trades) if trades else 0
        avg_pnl = np.mean([t.pnl_pct for t in trades])
        status = '✓' if pf_r > 1.0 else '✗'
        print(f'  {status} {regime:<18s} {len(trades):>7d} {wr:>6.1%} {pf_r:>7.2f} {avg_pnl:>+7.2%}')
        if pf_r > 1.0:
            profitable_regimes += 1

    return profitable_regimes, len(regime_trades)


def test_clustering(datasets):
    print()
    print('=' * 70)
    print('  TEST 7: Trade Clustering (Luck Test)')
    print('=' * 70)
    all_trades = []
    for symbol, df in datasets.items():
        bt = Backtester(initial_capital=10000)
        signals = make_signals(df)
        result = bt.run(
            df, lambda d, s=signals: s,
            position_size_pct=0.01, stop_loss_atr=PARAMS['stop_atr_mult'],
            take_profit_atr=PARAMS['tp_atr_mult'],
            max_holding_bars=PARAMS['hold_bars'],
        )
        all_trades.extend(result.trades)

    if not all_trades:
        print('  No trades to analyze.')
        return False

    pnls = sorted([t.pnl for t in all_trades], reverse=True)
    total_pnl = sum(pnls)
    print(f'  Total PnL: ${total_pnl:.2f}')

    # Remove top 3 trades
    top3_pnl = sum(pnls[:3])
    rest_pnl = total_pnl - top3_pnl
    print(f'  Top 3 trades PnL: ${top3_pnl:.2f}')
    print(f'  PnL without top 3: ${rest_pnl:.2f}')
    profitable_without = rest_pnl > 0
    print(f'  Profitable without top 3: {"YES ✓" if profitable_without else "NO ✗"}')

    # Top 20% concentration
    n_top = max(1, len(pnls) // 5)
    top20_pnl = sum(pnls[:n_top])
    concentration = top20_pnl / total_pnl * 100 if total_pnl > 0 else 0
    print(f'  Top 20% concentration: {concentration:.0f}%')

    return profitable_without


def test_monte_carlo(datasets, n_sims=5000):
    print()
    print('=' * 70)
    print('  TEST 8: Monte Carlo Simulation')
    print('=' * 70)
    all_trades = []
    for symbol, df in datasets.items():
        bt = Backtester(initial_capital=10000)
        signals = make_signals(df)
        result = bt.run(
            df, lambda d, s=signals: s,
            position_size_pct=0.01, stop_loss_atr=PARAMS['stop_atr_mult'],
            take_profit_atr=PARAMS['tp_atr_mult'],
            max_holding_bars=PARAMS['hold_bars'],
        )
        all_trades.extend(result.trades)

    if not all_trades:
        return 0

    pnls = np.array([t.pnl for t in all_trades])
    rng = np.random.default_rng(42)
    finals = []
    for _ in range(n_sims):
        shuffled = rng.permutation(pnls)
        equity = 10000 + np.cumsum(shuffled)
        finals.append(equity[-1])

    finals = np.array(finals)
    prob_profit = (finals > 10000).mean()
    print(f'  Simulations: {n_sims}')
    print(f'  P(profit):   {prob_profit:.1%}')
    print(f'  Median return: {np.median(finals)/10000 - 1:+.2%}')
    print(f'  5th percentile: {np.percentile(finals, 5)/10000 - 1:+.2%}')
    print(f'  95th percentile: {np.percentile(finals, 95)/10000 - 1:+.2%}')
    return prob_profit


def main():
    import sys

    outfile = open('pipeline_output/funding_mr_v7_validation.txt', 'w')

    def p(*args, **kwargs):
        s = ' '.join(str(a) for a in args)
        outfile.write(s + '\n')
        outfile.flush()
        sys.stdout.write(s + '\n')
        sys.stdout.flush()

    p()
    p('=' * 70)
    p('  INSTITUTION-LEVEL VALIDATION: funding_mr_v7')
    p('  Strategy: Funding Rate Mean Reversion')
    p('  Assets: BTC, ETH, SOL, XRP (4 edge assets)')
    p('=' * 70)
    p()

    datasets = load_datasets()

    # Run all tests
    scorecard = {}

    # Test 2: Slippage
    slip_results = test_slippage(datasets)
    scorecard['slippage_3x'] = slip_results[2]  # Survives 3x?

    # Test 3: Latency
    lat_results = test_latency(datasets)
    scorecard['latency_1bar'] = lat_results[1]  # Survives 1-bar delay?

    # Test 4: Commission
    comm_results = test_commission(datasets)
    scorecard['commission_3x'] = comm_results[2]  # Survives 3x?

    # Test 5: Walk-forward
    wf_profitable, wf_total, wf_sharpe = test_walkforward(datasets)
    scorecard['walkforward'] = (wf_profitable / wf_total > 0.5) if wf_total > 0 else False

    # Test 6: Regime
    regime_profitable, regime_total = test_regime(datasets)
    scorecard['regime'] = regime_profitable >= 2  # Profitable in ≥2 regimes

    # Test 7: Clustering
    not_lucky = test_clustering(datasets)
    scorecard['clustering'] = not_lucky

    # Test 8: Monte Carlo
    mc_prob = test_monte_carlo(datasets)
    scorecard['monte_carlo'] = mc_prob > 0.6

    # ─── FINAL SCORECARD ────────────────────────────────────────
    print()
    print('╔' + '═' * 68 + '╗')
    print('║  FINAL SCORECARD                                                  ║')
    print('╚' + '═' * 68 + '╝')
    passed = 0
    total = len(scorecard)
    for test, result in scorecard.items():
        status = '✓ PASS' if result else '✗ FAIL'
        print(f'  {status}  {test}')
        if result:
            passed += 1

    print(f'\n  Score: {passed}/{total}')
    if passed >= 6:
        verdict = 'DEPLOY — Real edge confirmed under stress'
    elif passed >= 4:
        verdict = 'PROMISING — Edge exists but has weaknesses'
    elif passed >= 2:
        verdict = 'WEAK — May be overfitting, proceed with caution'
    else:
        verdict = 'REJECT — No robust edge'
    print(f'  Verdict: {verdict}')

    # Save to file
    with open('pipeline_output/funding_mr_v7_validation.txt', 'w') as f:
        f.write(output.getvalue())


if __name__ == '__main__':
    main()

"""V2.0 Deep Validation — tests all new modules end-to-end."""
import sys, warnings, os
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

print('=== DEEP FUNCTIONAL TESTS ===')
print()

# 1. Feature engine: 120+ features from OHLCV
from src.data.features import compute_all_features
dates = pd.date_range('2024-01-01', periods=1000, freq='1h')
price = 50000 + np.cumsum(np.random.randn(1000) * 100)
df = pd.DataFrame({
    'open': price + np.random.randn(1000) * 10,
    'high': price + abs(np.random.randn(1000) * 50),
    'low': price - abs(np.random.randn(1000) * 50),
    'close': price,
    'volume': abs(np.random.randn(1000) * 1e6) + 1000
}, index=dates)
df_feat = compute_all_features(df)
n_feat = len([c for c in df_feat.columns if c not in ['open', 'high', 'low', 'close', 'volume']])
nan_pct = df_feat.iloc[200:].isna().mean().mean() * 100
print(f'1. Features:     {n_feat} computed, NaN%={nan_pct:.1f}% (after warmup)')
assert n_feat > 100, f'Expected 100+ features, got {n_feat}'
assert nan_pct < 5, f'Too many NaN: {nan_pct:.1f}%'
print('   PASS')

# 2. Portfolio optimizer (all 4 methods)
from src.risk.portfolio import PortfolioOptimizer
returns = pd.DataFrame(np.random.randn(200, 6) * 0.01, columns=[f's{i}' for i in range(6)])
for method in ['markowitz', 'risk_parity', 'cvar', 'hrp']:
    opt = PortfolioOptimizer(method=method)
    pw = opt.optimize(returns)
    assert abs(sum(pw.weights.values()) - 1.0) < 0.01, f'{method}: weights dont sum to 1'
    assert all(w >= 0 for w in pw.weights.values()), f'{method}: negative weight'
print('2. Portfolio:    All 4 methods OK (markowitz/risk_parity/cvar/hrp)')
print('   PASS')

# 3. Advanced risk manager
from src.risk.advanced import AdvancedRiskManager, DrawdownBand
arm = AdvancedRiskManager(initial_capital=10000)
arm.register_strategy('alpha_1')
arm.register_strategy('alpha_2')
ok, mult, reason = arm.check_entry('alpha_1', 'BTC/USDT', 500)
assert ok, f'Should allow trade: {reason}'
# Simulate drawdown to orange band
arm.capital = 8900
arm.peak_capital = 10000
state = arm.get_risk_state()
assert state.drawdown_band == 'orange', f'Expected orange, got {state.drawdown_band}'
assert state.size_multiplier == 0.5
# Test circuit breaker
for _ in range(5):
    arm.record_trade_result('alpha_1', -50, -0.01)
cb = arm.breakers['alpha_1']
assert cb.is_tripped, 'Circuit breaker should have tripped after 5 losses'
print('3. Risk:         Drawdown bands + circuit breakers OK')
print('   PASS')

# 4. Smart execution
from src.execution.smart import SmartExecutionEngine
se = SmartExecutionEngine(paper_mode=True)
# Small order = market
r1 = se.execute_entry('BTC/USDT', 1, 0.01, 50000, 49000, 52000, 50000, atr=500)
assert r1.success and r1.algo == 'market'
# Large order = TWAP
r2 = se.execute_entry('ETH/USDT', 1, 50, 2000, 1900, 2200, 2000, atr=40)
assert r2.success and r2.algo == 'twap'
# Stale signal rejection
r3 = se.execute_entry('SOL/USDT', 1, 1, 150, 140, 170, 100, atr=5)
assert not r3.success and 'gap' in r3.error.lower()
quality = se.get_execution_quality()
avg_slip = quality['avg_slippage_bps']
print(f'4. Execution:    Market/TWAP routing + gap reject OK (avg slip={avg_slip:.1f}bps)')
print('   PASS')

# 5. Database persistence
from src.fund.database import Database
test_db = 'fund_data/test_validation.db'
if os.path.exists(test_db):
    os.remove(test_db)
db = Database(db_path=test_db)
tid = db.record_trade_open('alpha_1', 'BTC/USDT', 1, 50000, 0.1, signal_strength=0.8)
db.record_trade_close(tid, 51000, 100, 0.02, 'take_profit', slippage_bps=5)
vid = db.save_model_version('{"test": true}', 'BTC/USDT', '1h', 3, 1.5, 1.2)
db.deploy_version(vid)
deployed = db.get_deployed_version('BTC/USDT')
assert deployed is not None and deployed['version_id'] == vid
db.log_risk_event('circuit_breaker', 'warning', 'alpha_1', '5 consecutive losses')
db.snapshot_equity(10100, 10100, 0, 0, 2, 100)
curve = db.get_equity_curve(days=1)
assert len(curve) == 1
perf = db.get_strategy_performance()
assert len(perf) == 1 and perf[0]['total_pnl'] == 100
os.remove(test_db)
print('5. Database:     Trades + versions + risk events + equity + analytics OK')
print('   PASS')

# 6. Ensemble evolution (quick smoke test)
from src.alpha_genome.ensemble import EnsembleEvolver
df_clean = df_feat.dropna()
evolver = EnsembleEvolver(
    n_islands=2, island_size=10, max_generations=3,
    committee_size=5, min_trades=5, output_dir='test_evolved'
)
committee = evolver.evolve(df_clean, symbol='BTC/USDT', timeframe='1h')
signal = evolver.generate_ensemble_signal(df_clean)
print(f'6. Ensemble:     {len(committee)} committee members, signal_dir={signal.direction}')
print('   PASS')

# 7. Original test suite
print()
print('--- Running original 61-test suite ---')

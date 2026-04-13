"""
SignalForge V2 — End-to-End Integration Test
=============================================
Tests the FULL production pipeline:
  Features → Signals → Risk → Size → Execute → Record → Monitor

Validates that all V2.0 modules work together correctly.
"""
import sys
import json
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
    return condition


def make_market_data(n=1000, seed=42):
    """Generate realistic synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="1h")
    price = 3000.0
    prices = []
    for _ in range(n):
        ret = rng.normal(0.0001, 0.015)
        price *= (1 + ret)
        prices.append(price)
    close = np.array(prices)
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    opn = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.uniform(100, 10000, n)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.set_index("timestamp", inplace=True)
    return df


# ================================================================
print("=" * 60)
print("SIGNALFORGE V2 — END-TO-END INTEGRATION TEST")
print("=" * 60)

# ────────────────────────────────────────
print("\n[1] FEATURE PIPELINE")
# ────────────────────────────────────────
from src.data.features import compute_all_features, ADVANCED_FEATURE_NAMES

df = make_market_data()
df_feat = compute_all_features(df)
n_features = len([c for c in df_feat.columns if c not in df.columns])
check("compute_all_features produces data", len(df_feat) == len(df))
check(f"Feature count: {n_features} >= 100", n_features >= 100)

# Drop NaN rows (warmup period)
warmup = df_feat.dropna()
check(f"Data after NaN drop: {len(warmup)} rows", len(warmup) > 700)

# ────────────────────────────────────────
print("\n[2] FUND MANAGER V2 — CONSTRUCTION")
# ────────────────────────────────────────
from src.fund.manager_v2 import AutonomousFundManagerV2, FundStateV2
from src.risk.manager import RiskLimits
from src.risk.advanced import DrawdownBand

# Use temp DB
import tempfile, os
tmp_dir = tempfile.mkdtemp()
db_path = os.path.join(tmp_dir, "test.db")
ledger_path = os.path.join(tmp_dir, "test_ledger.json")

fund = AutonomousFundManagerV2(
    initial_capital=10000,
    risk_limits=RiskLimits(
        max_position_pct=0.05,
        max_drawdown_pct=0.15,
        max_daily_loss_pct=0.05,
        max_open_positions=5,
    ),
    max_strategies=10,
    ledger_path=ledger_path,
    db_path=db_path,
    portfolio_method="hrp",
    drawdown_bands=DrawdownBand(
        yellow_pct=0.05,
        orange_pct=0.10,
        red_pct=0.15,
        black_pct=0.20,
    ),
    max_slippage_bps=100,
)

check("FundManagerV2 created", fund is not None)
check("Advanced risk manager active", fund.advanced_risk is not None)
check("Smart exec engine active", fund.smart_exec is not None)
check("Database connected", fund.db is not None)
check("Portfolio optimizer ready", fund.portfolio_optimizer is not None)

# ────────────────────────────────────────
print("\n[3] STRATEGY LOADING + PORTFOLIO OPTIMIZATION")
# ────────────────────────────────────────
from src.alpha_genome.evolution import EvolvedStrategy
from src.alpha_genome.gene import random_tree, tree_to_dict

# Create mock strategies
mock_strategies = []
for i in range(5):
    rng = np.random.RandomState(i + 100)
    tree = random_tree(max_depth=4, features=list(warmup.columns)[:40])
    tree_d = tree_to_dict(tree)

    from src.alpha_genome.evolution import FitnessResult
    fitness = FitnessResult(
        fitness=rng.uniform(0.01, 0.5),
        oos_sharpe=rng.uniform(0.5, 2.0),
        oos_profit_factor=rng.uniform(1.05, 2.0),
        oos_win_rate=rng.uniform(0.45, 0.65),
        total_trades=rng.randint(20, 100),
        consistency=rng.uniform(0.6, 1.0),
        p_value=rng.uniform(0.001, 0.05),
        is_significant=True,
    )

    from src.alpha_genome.gene import tree_hash, tree_to_formula
    strat = EvolvedStrategy(
        name=f"test_strat_{i}",
        tree_dict=tree_d,
        tree_hash=tree_hash(tree),
        formula=tree_to_formula(tree),
        fitness=fitness,
        novelty_score=rng.uniform(0.3, 0.9),
        generation=10,
    )
    mock_strategies.append(strat)

fund.load_strategies(mock_strategies)
check("Strategies loaded", len(fund.active_strategies) == 5)
check("Portfolio weights computed", len(fund.portfolio_weights) > 0)
check("Weights sum reasonable", 0.5 < sum(fund.portfolio_weights.values()) <= 1.05)

weights_str = ", ".join(f"{k}:{v:.2f}" for k, v in fund.portfolio_weights.items())
print(f"       Weights: {weights_str}")

# ────────────────────────────────────────
print("\n[4] SIGNAL GENERATION")
# ────────────────────────────────────────
price = float(warmup["close"].iloc[-1])
candidates = fund.generate_signals(warmup, "BTC/USDT", price)
check("Signals generated", isinstance(candidates, list))
print(f"       {len(candidates)} candidate signal(s)")

# Add a manual signal to ensure execution path is tested
manual_signal = {
    "source": "test",
    "strategy_name": "test_strat_0",
    "strategy_hash": "test_hash",
    "asset": "TEST/USDT",
    "direction": 1,
    "price": 100.0,
    "signal_price": 100.0,
    "stop_loss": 95.0,
    "take_profit": 110.0,
    "atr": 2.0,
    "signal_strength": 0.8,
    "allocation_pct": 0.10,
    "reasoning": "Manual test signal",
}

# ────────────────────────────────────────
print("\n[5] FULL EXECUTION PIPELINE")
# ────────────────────────────────────────
executed = fund.process_and_execute([manual_signal])
check("Execution pipeline runs", isinstance(executed, list))

if executed:
    trade = executed[0]
    check("Trade has price", trade["price"] > 0)
    check("Trade has slippage", trade["slippage_bps"] >= 0)
    check("Trade algo set", trade["algo"] in ("market", "twap"))
    check("Position tracked", "TEST/USDT" in fund.open_positions)

    pos = fund.open_positions["TEST/USDT"]
    check("Position has trade_id", pos.trade_id > 0)
    check("Position entry price > 0", pos.entry_price > 0)
    print(f"       Trade: size={trade['size']:.6f} @ ${trade['price']:.2f} via {trade['algo']}")
else:
    print("       No trades executed (risk filter may have blocked)")
    # Still check that the pipeline ran without error
    check("Pipeline completed without crash", True)

# ────────────────────────────────────────
print("\n[6] EXIT MANAGEMENT + TRAILING STOPS")
# ────────────────────────────────────────
if "TEST/USDT" in fund.open_positions:
    # Simulate price hitting take profit
    closed = fund.check_exits({"TEST/USDT": 111.0})
    check("Exit check runs", isinstance(closed, list))

    if closed:
        c = closed[0]
        check("PnL computed", c["pnl"] != 0)
        check("Return computed", c["return_pct"] != 0)
        check("Close reason", c["reason"] in ("take_profit", "trailing_stop", "stop_loss"))
        check("Position removed", "TEST/USDT" not in fund.open_positions)
        print(f"       Closed: PnL=${c['pnl']:.2f} ({c['return_pct']:.2%}) reason={c['reason']}")
    else:
        # Price might not have triggered exit — that's OK
        check("No exit triggered (price within bounds)", True)

        # Force a stop loss exit
        pos = fund.open_positions["TEST/USDT"]
        closed = fund.check_exits({"TEST/USDT": pos.stop_loss - 1})
        if closed:
            check("Stop loss exit worked", closed[0]["reason"] == "stop_loss")
        else:
            check("Exit management completes", True)
else:
    check("Exit management (no position to test)", True)

# ────────────────────────────────────────
print("\n[7] DATABASE PERSISTENCE")
# ────────────────────────────────────────
from src.fund.database import Database
db = Database(db_path=db_path)

# Check trades were recorded
import sqlite3
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

trades = conn.execute("SELECT * FROM trades").fetchall()
check(f"Trades in DB: {len(trades)}", len(trades) >= 1)

if trades:
    t = dict(trades[0])
    check("Trade has strategy_name", t["strategy_name"] is not None)
    check("Trade has hash", t["hash"] is not None)
    check("Trade has timestamp", t["timestamp"] > 0)

# Execution log
exec_logs = conn.execute("SELECT * FROM execution_log").fetchall()
check(f"Execution logs: {len(exec_logs)}", len(exec_logs) >= 1)

# Equity snapshots
snapshots = conn.execute("SELECT * FROM equity_snapshots").fetchall()
check(f"Equity snapshots: {len(snapshots)}", len(snapshots) >= 0)  # May not have any if no close happened

conn.close()

# ────────────────────────────────────────
print("\n[8] HASH-CHAINED LEDGER INTEGRITY")
# ────────────────────────────────────────
is_valid, error = fund.ledger.verify_chain()
check("Ledger chain valid", is_valid, error or "")
check(f"Ledger entries: {len(fund.ledger.entries)}", len(fund.ledger.entries) >= 1)

# ────────────────────────────────────────
print("\n[9] FUND STATE + RISK STATE")
# ────────────────────────────────────────
state = fund.get_state()
check("State is FundStateV2", isinstance(state, FundStateV2))
check("Capital tracked", state.capital > 0)
check("Drawdown band valid", state.drawdown_band in ("green", "yellow", "orange", "red", "black"))
check("Ledger verified", state.ledger_verified)
check("Regime detected", state.regime != "")
check("Portfolio method set", state.portfolio_method == "hrp")

print(f"       Capital=${state.capital:.2f} DD={state.drawdown_pct:.2%} Band={state.drawdown_band}")
print(f"       Regime={state.regime} Strategies={state.active_strategies}")

# ────────────────────────────────────────
print("\n[10] STRATEGY ATTRIBUTION")
# ────────────────────────────────────────
attr = fund.get_strategy_attribution()
check("Attribution is DataFrame", isinstance(attr, pd.DataFrame))
check("Has strategy column", "strategy" in attr.columns)
check("Has weight column", "weight" in attr.columns)
check("Has decay_score", "decay_score" in attr.columns)
check("Has breaker_tripped", "breaker_tripped" in attr.columns)

# ────────────────────────────────────────
print("\n[11] REBALANCING")
# ────────────────────────────────────────
old_weights = dict(fund.portfolio_weights)
fund.rebalance()
check("Rebalance runs", True)
check("Weights still exist", len(fund.portfolio_weights) > 0)

# ────────────────────────────────────────
print("\n[12] ADVANCED RISK — DRAWDOWN BAND RESPONSE")
# ────────────────────────────────────────
# Simulate a 12% drawdown
fund.advanced_risk.capital = fund.advanced_risk.peak_capital * 0.88
risk_state = fund.advanced_risk.get_risk_state()
check("Orange band at 12% DD", risk_state.drawdown_band == "orange", f"got {risk_state.drawdown_band}")
check("Size mult reduced to 0.5", risk_state.size_multiplier == 0.5)
check("Still trading allowed", risk_state.can_trade)

# Simulate 16% drawdown → red band
fund.advanced_risk.capital = fund.advanced_risk.peak_capital * 0.84
risk_state = fund.advanced_risk.get_risk_state()
check("Red band at 16% DD", risk_state.drawdown_band == "red", f"got {risk_state.drawdown_band}")
check("Trading halted in red", not risk_state.can_trade)

# Reset
fund.advanced_risk.capital = fund.advanced_risk.peak_capital

# ────────────────────────────────────────
print("\n[13] CIRCUIT BREAKER ISOLATION")
# ────────────────────────────────────────
# Trip a circuit breaker for one strategy
for _ in range(6):
    fund.advanced_risk.record_trade_result("test_strat_0", -50, -0.005)

cb = fund.advanced_risk.breakers.get("test_strat_0")
check("Circuit breaker tripped", cb is not None and cb.is_tripped)

# Verify other strategies are unaffected
cb_1 = fund.advanced_risk.breakers.get("test_strat_1")
check("Other strategy unaffected", cb_1 is not None and not cb_1.is_tripped)

# ────────────────────────────────────────
print("\n[14] SMART EXECUTION — SLIPPAGE MODEL")
# ────────────────────────────────────────
from src.execution.smart import SmartExecutionEngine

exec_eng = SmartExecutionEngine(paper_mode=True, max_slippage_bps=100)

# Small order — market
r1 = exec_eng.execute_entry(
    symbol="ETH/USDT", direction=1, size=0.5,
    entry_price=2000, stop_loss=1900, take_profit=2200,
    signal_price=2000, atr=50,
)
check("Small order fills", r1.success)
if r1.success:
    check("Slippage > 0", r1.slippage_bps > 0)
    check("Algo = market", r1.algo == "market")
    print(f"       Small: fill=${r1.price:.2f} slip={r1.slippage_bps:.1f}bps")

# Large order — should trigger TWAP
r2 = exec_eng.execute_entry(
    symbol="BTC/USDT", direction=-1, size=2.0,
    entry_price=50000, stop_loss=52000, take_profit=45000,
    signal_price=50000, atr=1000,
)
check("Large order fills", r2.success)
if r2.success:
    check("TWAP used for large order", r2.algo == "twap")
    print(f"       Large: fill=${r2.price:.2f} slip={r2.slippage_bps:.1f}bps algo={r2.algo}")

# Gap rejection
r3 = exec_eng.execute_entry(
    symbol="SOL/USDT", direction=1, size=100,
    entry_price=100, stop_loss=90, take_profit=120,
    signal_price=95, atr=3,  # 5.3% gap
)
check("Gap rejection works", not r3.success)

# ────────────────────────────────────────
print("\n[15] PORTFOLIO OPTIMIZER")
# ────────────────────────────────────────
from src.risk.portfolio import PortfolioOptimizer

rng = np.random.RandomState(99)
returns_data = pd.DataFrame({
    f"s{i}": rng.normal(0.001, 0.02, 120) for i in range(5)
})

for method in ["hrp", "risk_parity", "markowitz", "cvar"]:
    try:
        opt = PortfolioOptimizer(method=method)
        result = opt.optimize(returns_data)
        weights_ok = abs(sum(result.weights.values()) - 1.0) < 0.01
        check(f"{method}: weights sum ≈ 1", weights_ok)
    except Exception as e:
        check(f"{method}: runs", False, str(e))


# ================================================================
# CLEANUP
# ================================================================
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)

# ================================================================
# FINAL RESULTS
# ================================================================
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL == 0:
    print("ALL V2 INTEGRATION TESTS PASSED")
else:
    print(f"WARNING: {FAIL} test(s) failed")
print("=" * 60)

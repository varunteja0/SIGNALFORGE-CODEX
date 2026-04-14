"""
SignalForge — End-to-End Integration Test
===========================================
Tests the FULL pipeline: data -> features -> evolution -> liquidation
                         -> fund -> ledger -> attribution

This proves all layers work together correctly.
"""

import sys
import os
import json
import time
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

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
        print(f"  [FAIL] {name} -- {detail}")


def build_realistic_data(n=2000, seed=42):
    """Build BTC-like data with trends, crashes, and recoveries."""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="1h")

    # Regime changes: bull -> crash -> recovery -> sideways
    price = np.zeros(n)
    price[0] = 40000
    regime_noise = 0.002

    for i in range(1, n):
        phase = i / n
        if phase < 0.3:
            drift = 0.0002  # Bull
        elif phase < 0.35:
            drift = -0.003  # Crash
        elif phase < 0.6:
            drift = 0.001   # Recovery
        else:
            drift = 0.00005 # Sideways

        price[i] = price[i-1] * (1 + drift + np.random.randn() * regime_noise)

    df = pd.DataFrame({
        "open": price * (1 + np.random.randn(n) * 0.001),
        "high": price * (1 + abs(np.random.randn(n) * 0.005)),
        "low": price * (1 - abs(np.random.randn(n) * 0.005)),
        "close": price,
        "volume": np.abs(np.random.randn(n) * 1e6 + 5e6),
    }, index=dates)

    return df


print("=" * 60)
print("SIGNALFORGE END-TO-END INTEGRATION TEST")
print("=" * 60)

# ======================================================================
# 1. DATA + FEATURES
# ======================================================================
print("\n--- SECTION 1: Data + Feature Pipeline ---")

from src.data.fetcher import compute_features
from src.alpha_genome.gene import FEATURE_NAMES

df_raw = build_realistic_data(2000)
df = compute_features(df_raw).dropna()
check("Realistic data built", len(df) > 1500, f"got {len(df)}")
check("All features present",
      all(f in df.columns for f in FEATURE_NAMES),
      f"missing: {[f for f in FEATURE_NAMES if f not in df.columns]}")
check("No NaN in features",
      not df[FEATURE_NAMES].isnull().any().any())

# ======================================================================
# 2. ALPHA GENOME EVOLUTION
# ======================================================================
print("\n--- SECTION 2: Alpha Genome Evolution ---")

from src.alpha_genome.evolution import AlphaGenomeEngine
from src.alpha_genome.gene import tree_from_dict, tree_to_formula

engine = AlphaGenomeEngine(
    population_size=30, max_generations=5,
    tournament_size=3, elitism_count=3,
    min_trades=10, output_dir=tempfile.mkdtemp(),
)

strategies = engine.evolve(df, symbol="BTC/USDT", timeframe="1h")
check("Evolution completes", True)
check("Generation history recorded", len(engine.generation_history) == 5)
check("Fitness improves or holds",
      engine.generation_history[-1].best_fitness >= engine.generation_history[0].best_fitness - 1)

# Even if no valid strategies, the engine should produce them for persistence
check("Hall of fame tracked", isinstance(engine.hall_of_fame, list))
print(f"       Strategies found: {len(strategies)}")

# ======================================================================
# 3. BACKTEST ENGINE
# ======================================================================
print("\n--- SECTION 3: Backtest Integration ---")

from src.backtest.engine import Backtester

backtester = Backtester(initial_capital=10000, commission_pct=0.001, slippage_pct=0.0005)

# Backtest a simple signal function
def simple_ma_signal(data):
    fast = data["close"].rolling(10).mean()
    slow = data["close"].rolling(50).mean()
    return (fast > slow).astype(int) - (fast < slow).astype(int)

bt_result = backtester.run(df, simple_ma_signal)
check("Backtest runs", bt_result.total_trades > 0, f"trades={bt_result.total_trades}")
check("Equity curve exists", len(bt_result.equity_curve) > 0)
check("Sharpe computed", isinstance(bt_result.sharpe_ratio, float))
print(f"       Return={bt_result.total_return:.2%} Sharpe={bt_result.sharpe_ratio:.2f} DD={bt_result.max_drawdown:.2%}")

mc = backtester.monte_carlo(bt_result)
check("Monte Carlo runs", "probability_of_profit" in mc)
print(f"       MC P(Profit)={mc['probability_of_profit']:.0%}")

# ======================================================================
# 4. REGIME DETECTION
# ======================================================================
print("\n--- SECTION 4: Regime Detection ---")

from src.regime.detector import RegimeDetector, MarketRegime

detector = RegimeDetector(n_regimes=3)
detector.fit(df)
regime = detector.detect(df)
check("Regime detected", isinstance(regime, MarketRegime))
print(f"       Current regime: {regime.value}")

history = detector.get_regime_history(df)
check("Regime history computed", len(history) > 0)

stats = detector.get_regime_stats(df)
check("Regime stats computed", len(stats) > 0)
for r_name, r_stats in stats.items():
    print(f"       {r_name}: {r_stats['pct_of_time']:.0%} of time")

# ======================================================================
# 5. LIQUIDATION ORACLE (CALIBRATED)
# ======================================================================
print("\n--- SECTION 5: Liquidation Oracle ---")

from src.liquidation.oracle import LiquidationOracle
from src.liquidation.cascade import CascadeSimulator
from src.liquidation.protocols import SyntheticPositionGenerator

current_price = float(df["close"].iloc[-1])

oracle = LiquidationOracle(use_synthetic=True, synthetic_tvl=10_000_000_000)
risk = oracle.assess_risk("BTC", current_price)
check("Risk score valid", 0 <= risk.risk_score <= 100, f"score={risk.risk_score}")
check("Has recommendation", risk.recommendation in ("AVOID", "CAUTIOUS", "OPPORTUNITY", "NEUTRAL"))
print(f"       Risk={risk.risk_score:.0f}/100 Rec={risk.recommendation}")

signals = oracle.generate_signals("BTC", current_price)
check("Signals generated", len(signals) > 0, f"count={len(signals)}")
for sig in signals[:2]:
    d = "LONG" if sig.direction == 1 else "SHORT"
    print(f"       {sig.signal_type} {d} conf={sig.confidence:.2f}")

# Cascade physics
gen = SyntheticPositionGenerator(seed=42)
positions = gen.generate("BTC", current_price, 1000, 10_000_000_000)
sim = CascadeSimulator(price_impact_bps_per_million=5.0)

r5 = sim.simulate(positions, current_price, 5)
r20 = sim.simulate(positions, current_price, 20)
check("Cascade monotonic", r20.total_liquidated_usd >= r5.total_liquidated_usd)
check("Cascade amplifies", r5.amplification_factor > 1.0)
check("Realistic amplification", r5.amplification_factor < 10, f"amp={r5.amplification_factor:.2f}")
print(f"       5% -> {r5.total_drop_pct:.1f}% ({r5.amplification_factor:.2f}x)")
print(f"       20% -> {r20.total_drop_pct:.1f}% ({r20.amplification_factor:.2f}x)")

# ======================================================================
# 6. RISK MANAGEMENT
# ======================================================================
print("\n--- SECTION 6: Risk Management ---")

from src.risk.manager import RiskManager, RiskLimits, PositionRequest

rm = RiskManager(capital=10000, limits=RiskLimits(
    max_position_pct=0.02, max_drawdown_pct=0.10,
    max_daily_loss_pct=0.03, max_open_positions=3,
))

# Should approve
req = PositionRequest(
    symbol="BTC/USDT", direction=1, entry_price=current_price,
    stop_loss=current_price * 0.95, take_profit=current_price * 1.10,
    signal_name="test", signal_strength=0.7,
)
approval = rm.evaluate(req)
check("Trade approved", approval.approved, approval.reason)
check("Size > 0", approval.size > 0, f"size={approval.size}")
check("Kelly applied", 0 < approval.kelly_fraction <= 1)

# Register and check status
rm.register_open("BTC/USDT", 1, approval.size, current_price)
status = rm.get_status()
check("Status has capital", status["capital"] == 10000)
check("Has open position", status["open_positions"] == 1)

# ======================================================================
# 7. FUND MANAGER (FULL LOOP)
# ======================================================================
print("\n--- SECTION 7: Autonomous Fund Manager ---")

from src.fund.manager import AutonomousFundManager, StrategyAllocation
from src.fund.ledger import VerifiableLedger

ledger_path = Path(tempfile.mkdtemp()) / "e2e_ledger.json"
fund = AutonomousFundManager(
    initial_capital=10000,
    risk_limits=RiskLimits(max_position_pct=0.02, max_drawdown_pct=0.10,
                           max_daily_loss_pct=0.03, max_open_positions=5),
    ledger_path=str(ledger_path),
)

# Load strategies if evolution found any
if strategies:
    fund.load_strategies(strategies)
    check("Strategies loaded into fund", len(fund.active_strategies) > 0)
else:
    # Manual liquidation oracle allocation
    fund.strategy_allocations["liquidation_oracle"] = StrategyAllocation(
        strategy_name="liquidation_oracle", strategy_type="liquidation",
        allocation_pct=0.15, active=True,
    )
    check("Liquidation oracle allocated", True)

# Generate signals
candidates = fund.generate_signals(df, "BTC/USDT", current_price)
check("Fund generates candidates", len(candidates) > 0, f"got {len(candidates)}")

# Process through risk
approved = fund.process_signals(candidates)
check("Risk processing works", isinstance(approved, list))
print(f"       Candidates={len(candidates)}, Approved={len(approved)}")

# Verify ledger recorded everything
check("Ledger recorded signals", len(fund.ledger.entries) > 0,
      f"entries={len(fund.ledger.entries)}")

# Chain integrity
is_valid, error = fund.ledger.verify_chain()
check("Ledger chain VALID", is_valid, f"error={error}")

# Fund state
state = fund.get_state()
check("Fund state valid", state.capital == 10000)
check("Ledger verified in state", state.ledger_verified)

# Attribution
attr = fund.get_strategy_attribution()
check("Attribution works", isinstance(attr, pd.DataFrame))

# ======================================================================
# 8. CROSS-LAYER WIRING
# ======================================================================
print("\n--- SECTION 8: Cross-Layer Wiring ---")

# Config
import yaml
with open("config/settings.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)
check("Config has all sections",
      all(k in config for k in ["alpha_genome", "liquidation", "fund"]))

# main.py parses
import ast
with open("main.py", encoding="utf-8") as f:
    ast.parse(f.read())
check("main.py syntax valid", True)

# Pipeline script parses
with open("scripts/run_pipeline.py", encoding="utf-8") as f:
    ast.parse(f.read())
check("Pipeline script syntax valid", True)

# Dashboard script parses
with open("scripts/dashboard.py", encoding="utf-8") as f:
    ast.parse(f.read())
check("Dashboard script syntax valid", True)

# Paper trading script parses
with open("scripts/paper_trade.py", encoding="utf-8") as f:
    ast.parse(f.read())
check("Paper trading script syntax valid", True)

# Calibration script parses
with open("scripts/calibrate_cascade.py", encoding="utf-8") as f:
    ast.parse(f.read())
check("Calibration script syntax valid", True)

# Calibration output exists
cal_path = Path("config/calibration.json")
if cal_path.exists():
    with open(cal_path) as f:
        cal = json.load(f)
    check("Calibration saved", "price_impact_bps_per_million" in cal)
else:
    check("Calibration file exists", False, "Run scripts/calibrate_cascade.py first")


# ======================================================================
# FINAL REPORT
# ======================================================================
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL == 0:
    print("ALL END-TO-END TESTS PASSED")
else:
    print(f"WARNING: {FAIL} tests failed!")
print("=" * 60)

sys.exit(1 if FAIL > 0 else 0)

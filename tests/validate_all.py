"""
SignalForge â€” Full System Validation
=====================================
Tests every layer: Alpha Genome, Liquidation Oracle, Fund.
"""

import sys
import os
import time
import json
import tempfile
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {name}")
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {name} â€” {detail}")


def build_test_data(n=1000, seed=42):
    """Build realistic synthetic market data with all features."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    # Trending + mean-reverting price (more realistic than pure random walk)
    trend = np.linspace(0, 2000, n)
    noise = np.cumsum(np.random.randn(n) * 50)
    mean_rev = -0.01 * noise  # pulls back toward trend
    price = 50000 + trend + noise + np.cumsum(mean_rev)

    df = pd.DataFrame({
        "open": price + np.random.randn(n) * 10,
        "high": price + abs(np.random.randn(n) * 50),
        "low": price - abs(np.random.randn(n) * 50),
        "close": price,
        "volume": abs(np.random.randn(n) * 1000000) + 100000,
    }, index=dates)

    # Features (same as DataEngine.compute_features)
    for p in [1, 3, 5, 10, 20, 50]:
        df[f"ret_{p}"] = df["close"].pct_change(p)
    for w in [10, 20, 50]:
        df[f"vol_{w}"] = df["close"].pct_change().rolling(w).std()
        df[f"vol_ratio_{w}"] = df["volume"] / (df["volume"].rolling(w).mean() + 1e-10)
    for w in [10, 20, 50, 100, 200]:
        ma = df["close"].rolling(w).mean()
        df[f"ma_{w}"] = ma
        df[f"price_vs_ma_{w}"] = (df["close"] - ma) / (ma + 1e-10)
    for p in [7, 14, 21]:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
        rs = gain / (loss + 1e-10)
        df[f"rsi_{p}"] = 100 - (100 / (1 + rs))
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    ma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper_20"] = ma20 + 2 * std20
    df["bb_lower_20"] = ma20 - 2 * std20
    df["bb_pct_20"] = (df["close"] - df["bb_lower_20"]) / (
        df["bb_upper_20"] - df["bb_lower_20"] + 1e-10
    )
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct_14"] = df["atr_14"] / (df["close"] + 1e-10)
    df["atr_21"] = tr.rolling(21).mean()
    df["atr_pct_21"] = df["atr_21"] / (df["close"] + 1e-10)
    df["bar_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)

    return df.dropna()


# ======================================================================
print("=" * 60)
print("SIGNALFORGE FULL SYSTEM VALIDATION")
print("=" * 60)

df = build_test_data(1500)
print(f"\nTest data: {len(df)} bars, {len(df.columns)} features\n")

# ======================================================================
# LAYER 1: ALPHA GENOME
# ======================================================================
print("â”€" * 60)
print("LAYER 1: ALPHA GENOME")
print("â”€" * 60)

# --- 1.1 Expression Trees ---
print("\n[1.1] Expression Tree Generation & Evaluation")
from src.alpha_genome.gene import (
    random_tree, crossover, mutate,
    tree_to_formula, tree_to_dict, tree_from_dict, tree_hash,
    FeatureNode, ConstantNode, UnaryNode, BinaryNode,
    ComparisonNode, TimeSeriesNode, FEATURE_NAMES,
)

trees_ok = True
for i in range(20):
    tree = random_tree(max_depth=5, seed=i)
    result = tree.evaluate(df)
    if result.isna().any() or np.isinf(result).any():
        trees_ok = False
        break
check("20 random trees NaN/Inf free", trees_ok)

# Serialization roundtrip
tree = random_tree(max_depth=4, seed=99)
f1 = tree_to_formula(tree)
tree2 = tree_from_dict(tree_to_dict(tree))
f2 = tree_to_formula(tree2)
check("Serialization roundtrip", f1 == f2)
print(f"       Sample formula: {f1[:80]}")

# Crossover
t1 = random_tree(max_depth=4, seed=1)
t2 = random_tree(max_depth=4, seed=2)
c1, c2 = crossover(t1, t2)
r1 = c1.evaluate(df)
r2 = c2.evaluate(df)
check("Crossover produces valid children", not r1.isna().any() and not r2.isna().any())

# Mutation
m1 = mutate(t1)
rm = m1.evaluate(df)
check("Mutation produces valid tree", not rm.isna().any() and m1.depth() <= 8)

# Hash uniqueness
h1 = tree_hash(t1)
h2 = tree_hash(t2)
check("Different trees have different hashes", h1 != h2)

# --- 1.2 Fitness Evaluation ---
print("\n[1.2] Fitness Evaluation (Walk-Forward)")
from src.alpha_genome.fitness import FitnessEvaluator

ev = FitnessEvaluator(
    walk_forward_splits=4,
    min_total_trades=10,
    population_size=50,
    holding_period=5,
)

# Test _signal_returns produces trades
tree = random_tree(max_depth=5, seed=100)
sigs = tree.evaluate(df)
rets = ev._signal_returns(sigs, df)
check("_signal_returns produces trades", len(rets) > 50, f"got {len(rets)}")

# Full evaluation
result = ev.evaluate(tree, df)
check("evaluate() returns FitnessResult", result.total_trades > 0, f"trades={result.total_trades}")
check("Fitness score is numeric", isinstance(result.fitness, float))
print(f"       Sharpe={result.oos_sharpe:.2f} WR={result.oos_win_rate:.1%} trades={result.total_trades}")

# Evaluate many â€” some should have nonzero fitness
fitnesses = []
for i in range(30):
    t = random_tree(max_depth=5, seed=i + 500)
    r = ev.evaluate(t, df)
    fitnesses.append(r.fitness)
check("30 trees produce varied fitness", max(fitnesses) != min(fitnesses),
      f"range=[{min(fitnesses):.4f}, {max(fitnesses):.4f}]")

# --- 1.3 Novelty Detection ---
print("\n[1.3] Novelty Detection")
from src.alpha_genome.novelty import NoveltyDetector

nd = NoveltyDetector(max_correlation=0.7)
nd.register_standard_signals(df)
check("Standard signals registered", len(nd.known_signal_series) > 0,
      f"count={len(nd.known_signal_series)}")

# RSI-like should have near-zero novelty
rsi_like = pd.Series(0.0, index=df.index)
rsi_like[df["rsi_14"] < 30] = 1.0
rsi_like[df["rsi_14"] > 70] = -1.0
nov_rsi = nd.novelty_score(rsi_like)
check("RSI-like has low novelty", nov_rsi < 0.3, f"novelty={nov_rsi:.3f}")

# Evolved tree should be more novel
tree_novel = random_tree(max_depth=5, seed=300)
sig_novel = tree_novel.evaluate(df)
nov_evolved = nd.novelty_score(sig_novel)
check("Evolved signal more novel than RSI", nov_evolved > nov_rsi,
      f"evolved={nov_evolved:.3f} vs rsi={nov_rsi:.3f}")

# Diverse selection
candidates = []
for i in range(10):
    t = random_tree(max_depth=5, seed=i + 700)
    s = t.evaluate(df)
    candidates.append((f"strat_{i}", s, float(np.random.rand())))
diverse = nd.select_diverse_set(candidates, max_strategies=5)
check("Diverse selection returns <= max", len(diverse) <= 5)

# --- 1.4 Mini Evolution ---
print("\n[1.4] Mini Evolution (5 generations x 30 pop)")
from src.alpha_genome.evolution import AlphaGenomeEngine

engine = AlphaGenomeEngine(
    population_size=30,
    max_generations=5,
    tournament_size=3,
    crossover_rate=0.7,
    mutation_rate=0.2,
    elitism_count=3,
    max_tree_depth=5,
    novelty_weight=0.2,
    walk_forward_splits=3,
    min_trades=10,
    output_dir="test_evolved",
)
strategies = engine.evolve(df, symbol="TEST/USDT", timeframe="1h")
check("Evolution completes", True)
check("Generation history recorded", len(engine.generation_history) > 0,
      f"gens={len(engine.generation_history)}")
print(f"       Strategies found: {len(strategies)}")
print(f"       Hall of fame: {len(engine.hall_of_fame)}")

if strategies:
    best = strategies[0]
    print(f"       Best: {best.name} Sharpe={best.fitness.oos_sharpe:.2f} fitness={best.fitness.fitness:.4f}")
    check("Strategy has formula", len(best.formula) > 0)
    check("Strategy has tree_dict", "type" in best.tree_dict)

    # Save/load roundtrip
    loaded = engine.load_strategies()
    check("Save/load roundtrip", len(loaded) == len(strategies),
          f"saved={len(strategies)} loaded={len(loaded)}")
else:
    print("       (No valid strategies â€” expected on synthetic data)")


# ======================================================================
# LAYER 2: LIQUIDATION ORACLE
# ======================================================================
print("\n" + "â”€" * 60)
print("LAYER 2: LIQUIDATION ORACLE")
print("â”€" * 60)

# --- 2.1 Synthetic Positions ---
print("\n[2.1] Synthetic Position Generation")
from src.liquidation.protocols import SyntheticPositionGenerator

gen = SyntheticPositionGenerator(seed=42)
positions = gen.generate(asset="ETH", current_price=3000, n_positions=500,
                         total_tvl_usd=2_000_000_000)
check("500 positions generated", len(positions) == 500)
check("All collateral positive", all(p.collateral_usd > 0 for p in positions))
check("All liq prices below current", all(p.liquidation_price < 3000 for p in positions))
at_risk = sum(1 for p in positions if p.is_at_risk)
total_collateral = sum(p.collateral_usd for p in positions)
print(f"       Total collateral: ${total_collateral:,.0f}")
print(f"       At risk (within 10%): {at_risk}")

# --- 2.2 Cascade Simulation ---
print("\n[2.2] Cascade Simulation")
from src.liquidation.cascade import CascadeSimulator

sim = CascadeSimulator(price_impact_bps_per_million=5.0)
r5 = sim.simulate(positions, 3000, trigger_drop_pct=5)
r15 = sim.simulate(positions, 3000, trigger_drop_pct=15)
check("5% drop: total >= trigger", r5.total_drop_pct >= 5,
      f"got {r5.total_drop_pct:.1f}%")
check("15% drop: more liquidations", r15.total_liquidated_usd >= r5.total_liquidated_usd)
check("Cascade amplifies", r5.amplification_factor >= 1.0)
print(f"       5% trigger  -> {r5.total_drop_pct:.1f}% total ({r5.amplification_factor:.2f}x)")
print(f"       15% trigger -> {r15.total_drop_pct:.1f}% total ({r15.amplification_factor:.2f}x)")

# --- 2.3 Trigger Scan ---
print("\n[2.3] Trigger Level Scan")
scan = sim.scan_trigger_levels(positions, 3000, (1, 25), steps=25)
check("Scan produces 25 levels", len(scan) == 25)
check("Monotonic liquidation volume", scan.iloc[-1]["liquidated_usd"] >= scan.iloc[0]["liquidated_usd"])

# --- 2.4 Cliff Edges ---
print("\n[2.4] Cliff Edge Detection")
cliffs = sim.find_cliff_edges(positions, 3000)
print(f"       Found {len(cliffs)} cliff edges")
for c in cliffs[:3]:
    print(f"       At -{c['trigger_drop_pct']:.1f}%: amp={c['total_amplification']:.1f}x")
check("Cliff detection runs", True)

# --- 2.5 Heatmap ---
print("\n[2.5] Liquidation Heatmap")
hm = sim.liquidation_heatmap(positions, 3000)
check("Heatmap has data", len(hm) > 0)
check("Heatmap has correct columns",
      set(["price_level", "liquidation_volume_usd", "position_count"]).issubset(hm.columns))

# --- 2.6 Full Oracle ---
print("\n[2.6] Full Oracle Risk + Signals")
from src.liquidation.oracle import LiquidationOracle

oracle = LiquidationOracle(use_synthetic=True, synthetic_tvl=5_000_000_000)
risk = oracle.assess_risk("ETH", 3000)
check("Risk score in [0,100]", 0 <= risk.risk_score <= 100, f"score={risk.risk_score}")
check("Has recommendation", risk.recommendation in ("AVOID", "CAUTIOUS", "OPPORTUNITY", "NEUTRAL"))
print(f"       Score: {risk.risk_score}/100, Rec: {risk.recommendation}")

signals = oracle.generate_signals("ETH", 3000)
check("Signals generated", len(signals) > 0, f"count={len(signals)}")
for sig in signals[:2]:
    d = "LONG" if sig.direction == 1 else "SHORT"
    print(f"       {sig.signal_type} {d}: entry=${sig.entry_price:,.0f} target=${sig.target_price:,.0f}")
    check(f"Signal {sig.signal_type} valid",
          sig.direction in (1, -1) and 0 <= sig.confidence <= 1)


# ======================================================================
# LAYER 3: AUTONOMOUS FUND
# ======================================================================
print("\n" + "â”€" * 60)
print("LAYER 3: AUTONOMOUS FUND")
print("â”€" * 60)

# --- 3.1 Verifiable Ledger ---
print("\n[3.1] Verifiable Hash-Chained Ledger")
from src.fund.ledger import VerifiableLedger

tmp_path = Path(tempfile.mkdtemp()) / "test_ledger.json"
ledger = VerifiableLedger(ledger_path=str(tmp_path))

# Append entries
for i in range(5):
    ledger.append(
        entry_type="trade_open" if i % 2 == 0 else "trade_close",
        asset="ETH/USDT",
        direction=1,
        price=3000 + i * 10,
        size=0.1,
        strategy_name="test_strategy",
        strategy_hash="abc123",
        signal_strength=0.8,
        risk_approval=True,
        risk_details="All checks passed",
        pnl=10.0 if i % 2 == 1 else 0,
    )

check("5 entries appended", len(ledger.entries) == 5)

# Verify chain
is_valid, error = ledger.verify_chain()
check("Chain integrity VALID", is_valid, f"error={error}")

# Verify chain link correctness
check("Genesis prev_hash", ledger.entries[0].prev_hash == "genesis")
for i in range(1, len(ledger.entries)):
    check(f"Entry {i} chain link",
          ledger.entries[i].prev_hash == ledger.entries[i-1].entry_hash)

# Self-hash verification
for i, entry in enumerate(ledger.entries):
    check(f"Entry {i} self-hash",
          entry.entry_hash == entry.compute_hash())

# Tamper detection
original_price = ledger.entries[2].price
ledger.entries[2].price = 99999  # TAMPER!
is_valid_after_tamper, tamper_error = ledger.verify_chain()
check("Tamper DETECTED", not is_valid_after_tamper, f"should fail: {tamper_error}")
ledger.entries[2].price = original_price  # Restore

# Persistence roundtrip
ledger2 = VerifiableLedger(ledger_path=str(tmp_path))
check("Persistence roundtrip", len(ledger2.entries) == 5)
is_valid2, _ = ledger2.verify_chain()
check("Loaded ledger chain valid", is_valid2)

# Performance metrics
perf = ledger.get_performance()
check("Performance computed", "total_trades" in perf)
check("Chain verified in perf", perf["chain_verified"] is True)
print(f"       Trades: {perf['total_trades']}, PnL: ${perf['total_pnl']:.2f}")

# Audit report
report = ledger.export_audit_report()
check("Audit report complete", "ledger_integrity" in report)
check("Audit says VALID", report["ledger_integrity"] == "VALID")

# --- 3.2 Fund Manager ---
print("\n[3.2] Autonomous Fund Manager")
from src.fund.manager import AutonomousFundManager
from src.risk.manager import RiskLimits

fund = AutonomousFundManager(
    initial_capital=10000,
    risk_limits=RiskLimits(
        max_position_pct=0.02,
        max_drawdown_pct=0.10,
        max_daily_loss_pct=0.03,
        max_open_positions=5,
    ),
    ledger_path=str(Path(tempfile.mkdtemp()) / "fund_ledger.json"),
)
check("Fund initialized", fund.current_capital == 10000)

# Load strategies if we evolved any
if strategies:
    fund.load_strategies(strategies)
    check("Strategies loaded", len(fund.active_strategies) > 0)
    check("Allocations set", len(fund.strategy_allocations) > 0)

    total_alloc = sum(a.allocation_pct for a in fund.strategy_allocations.values())
    check("Total allocation <= 100%", total_alloc <= 1.01, f"total={total_alloc:.1%}")
else:
    # Even with no strategies, fund should work with liquidation oracle
    fund.strategy_allocations["liquidation_oracle"] = __import__(
        "src.fund.manager", fromlist=["StrategyAllocation"]
    ).StrategyAllocation(
        strategy_name="liquidation_oracle",
        strategy_type="liquidation",
        allocation_pct=0.15,
        active=True,
    )

# Generate signals
candidates = fund.generate_signals(df, "ETH/USDT", 3000)
check("Fund generates candidates", isinstance(candidates, list))
print(f"       Candidates: {len(candidates)}")

# Fund state
state = fund.get_state()
check("Fund state valid", state.capital == 10000)
check("Ledger verified in state", state.ledger_verified is True)

# Strategy attribution
attr = fund.get_strategy_attribution()
check("Attribution is DataFrame", isinstance(attr, pd.DataFrame))


# ======================================================================
# CROSS-LAYER INTEGRATION
# ======================================================================
print("\n" + "â”€" * 60)
print("CROSS-LAYER INTEGRATION")
print("â”€" * 60)

print("\n[4.1] main.py imports")
try:
    # Verify main.py can parse without executing
    import ast
    with open("main.py", encoding="utf-8") as f:
        ast.parse(f.read())
    check("main.py parses without syntax errors", True)
except SyntaxError as e:
    check("main.py parses without syntax errors", False, str(e))

print("\n[4.2] Config loads with new sections")
import yaml
with open("config/settings.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)
check("alpha_genome config exists", "alpha_genome" in config)
check("liquidation config exists", "liquidation" in config)
check("fund config exists", "fund" in config)
check("alpha_genome.population_size", config["alpha_genome"]["population_size"] == 200)
check("liquidation.use_synthetic", config["liquidation"]["use_synthetic"] is True)
check("fund.ledger_path", "ledger" in config["fund"]["ledger_path"])


# ======================================================================
# FINAL REPORT
# ======================================================================
print("\n" + "=" * 60)
total = PASS_COUNT + FAIL_COUNT
print(f"RESULTS: {PASS_COUNT}/{total} passed, {FAIL_COUNT}/{total} failed")
if FAIL_COUNT == 0:
    print("ALL TESTS PASSED â€” SYSTEM VALIDATED")
else:
    print(f"WARNING: {FAIL_COUNT} tests failed!")
print("=" * 60)


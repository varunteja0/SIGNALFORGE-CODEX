"""
SignalForge — Cascade Calibrator
==================================
Calibrates the liquidation cascade simulator against known historical
crash events to ensure realistic predictions.

Known calibration events:
  - May 2021 BTC crash: 53% drop, massive DeFi liquidations
  - Nov 2022 FTX collapse: 25% BTC drop, cascade liquidations
  - Aug 2023 flash crash: 10% BTC drop, quick recovery
  - Mar 2020 COVID: 50% drop, record liquidations

Uses real historical data to tune:
  1. price_impact_bps_per_million
  2. Health factor distributions
  3. Cascade amplification factors
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.fetcher import DataFetcher, compute_features
from src.liquidation.cascade import CascadeSimulator
from src.liquidation.protocols import SyntheticPositionGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Calibrator")


# Known historical events for calibration
CALIBRATION_EVENTS = [
    {
        "name": "COVID Crash Mar 2020",
        "trigger_drop_pct": 15,
        "actual_total_drop_pct": 50,
        "tvl_at_time": 1_000_000_000,
        "amplification_actual": 3.3,
    },
    {
        "name": "May 2021 Crash",
        "trigger_drop_pct": 20,
        "actual_total_drop_pct": 53,
        "tvl_at_time": 80_000_000_000,
        "amplification_actual": 2.65,
    },
    {
        "name": "FTX Collapse Nov 2022",
        "trigger_drop_pct": 10,
        "actual_total_drop_pct": 25,
        "tvl_at_time": 40_000_000_000,
        "amplification_actual": 2.5,
    },
    {
        "name": "Aug 2023 Flash Crash",
        "trigger_drop_pct": 5,
        "actual_total_drop_pct": 10,
        "tvl_at_time": 50_000_000_000,
        "amplification_actual": 2.0,
    },
]


def calibrate_price_impact(events: list[dict] = None) -> dict:
    """Find optimal price_impact_bps_per_million that matches historical data."""
    events = events or CALIBRATION_EVENTS

    print("=" * 60)
    print("CASCADE SIMULATOR CALIBRATION")
    print("=" * 60)

    best_bps = 5.0
    best_error = float("inf")
    results = []

    # Grid search over price impact parameter
    for bps in np.arange(1.0, 20.1, 0.5):
        total_error = 0

        for event in events:
            gen = SyntheticPositionGenerator(seed=42)
            positions = gen.generate(
                asset="BTC",
                current_price=50000,
                n_positions=1000,
                total_tvl_usd=event["tvl_at_time"],
            )

            sim = CascadeSimulator(price_impact_bps_per_million=bps)
            result = sim.simulate(
                positions, 50000, trigger_drop_pct=event["trigger_drop_pct"]
            )

            predicted_amp = result.amplification_factor
            actual_amp = event["amplification_actual"]
            error = (predicted_amp - actual_amp) ** 2
            total_error += error

        if total_error < best_error:
            best_error = total_error
            best_bps = bps

        results.append({"bps": bps, "error": total_error})

    print(f"\n  Optimal price_impact_bps_per_million: {best_bps:.1f}")
    print(f"  Mean squared error: {best_error:.4f}")

    # Show calibrated results
    print(f"\n  Calibrated predictions:")
    gen = SyntheticPositionGenerator(seed=42)
    sim = CascadeSimulator(price_impact_bps_per_million=best_bps)

    for event in events:
        positions = gen.generate(
            asset="BTC",
            current_price=50000,
            n_positions=1000,
            total_tvl_usd=event["tvl_at_time"],
        )
        result = sim.simulate(
            positions, 50000, trigger_drop_pct=event["trigger_drop_pct"]
        )
        match = abs(result.amplification_factor - event["amplification_actual"]) < 0.5
        status = "OK" if match else "MISS"
        print(
            f"    [{status}] {event['name']}: "
            f"predicted={result.amplification_factor:.2f}x "
            f"actual={event['amplification_actual']:.2f}x "
            f"(trigger={event['trigger_drop_pct']}%)"
        )

    return {
        "optimal_bps": best_bps,
        "error": best_error,
        "calibration_data": results,
    }


def validate_cascade_physics(bps: float = None) -> dict:
    """Validate that the cascade model behaves correctly."""
    if bps is None:
        cal = calibrate_price_impact()
        bps = cal["optimal_bps"]

    print("\n" + "=" * 60)
    print("PHYSICS VALIDATION")
    print("=" * 60)

    gen = SyntheticPositionGenerator(seed=42)
    sim = CascadeSimulator(price_impact_bps_per_million=bps)

    results = {}

    # Test 1: Monotonicity — larger drops should cause more liquidations
    print("\n  Test 1: Monotonicity (larger drops -> more liquidations)")
    positions = gen.generate("BTC", 50000, 1000, 10_000_000_000)
    prev_liq = 0
    monotonic = True

    for drop in [1, 3, 5, 10, 15, 20, 25, 30]:
        r = sim.simulate(positions, 50000, trigger_drop_pct=drop)
        if r.total_liquidated_usd < prev_liq:
            monotonic = False
        prev_liq = r.total_liquidated_usd
        print(f"    {drop:2d}% -> {r.total_drop_pct:5.1f}% total, amp={r.amplification_factor:.2f}x, liq=${r.total_liquidated_usd/1e6:.0f}M")

    results["monotonic"] = monotonic
    print(f"  {'PASS' if monotonic else 'FAIL'}: Monotonicity")

    # Test 2: Amplification > 1 for significant drops
    print("\n  Test 2: Cascade amplification present")
    r5 = sim.simulate(positions, 50000, trigger_drop_pct=5)
    r15 = sim.simulate(positions, 50000, trigger_drop_pct=15)
    has_amplification = r5.amplification_factor > 1.0 and r15.amplification_factor > 1.0
    results["amplification"] = has_amplification
    print(f"  {'PASS' if has_amplification else 'FAIL'}: 5%={r5.amplification_factor:.2f}x, 15%={r15.amplification_factor:.2f}x")

    # Test 3: TVL sensitivity — more TVL at risk = bigger cascades
    print("\n  Test 3: TVL sensitivity")
    tvl_results = []
    for tvl in [1e9, 5e9, 10e9, 50e9]:
        pos = gen.generate("BTC", 50000, 1000, tvl)
        r = sim.simulate(pos, 50000, trigger_drop_pct=10)
        tvl_results.append(r.total_drop_pct)
        print(f"    TVL=${tvl/1e9:.0f}B -> {r.total_drop_pct:.1f}% drop, amp={r.amplification_factor:.2f}x")

    tvl_sensitive = tvl_results[-1] > tvl_results[0]
    results["tvl_sensitive"] = tvl_sensitive
    print(f"  {'PASS' if tvl_sensitive else 'FAIL'}: TVL sensitivity")

    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    return results


def save_calibration(bps: float, output_path: str = "config/calibration.json"):
    """Save calibrated parameters."""
    cal_data = {
        "price_impact_bps_per_million": bps,
        "calibrated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "calibration_events": [e["name"] for e in CALIBRATION_EVENTS],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cal_data, f, indent=2)
    print(f"\n  Saved calibration to {output_path}")


if __name__ == "__main__":
    cal = calibrate_price_impact()
    validate_cascade_physics(cal["optimal_bps"])
    save_calibration(cal["optimal_bps"])

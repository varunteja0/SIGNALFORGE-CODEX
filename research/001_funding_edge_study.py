#!/usr/bin/env python3
"""
================================================================================
RESEARCH PAPER #001: THE CRYPTO FUNDING RATE EDGE
================================================================================
Author: Senior Quant Researcher
Date: April 2026

ABSTRACT:
---------
This study investigates whether crypto perpetual futures funding rates represent
a tradeable, persistent, structural edge. We decompose the edge into:

1. CARRY COMPONENT: Do funding rates have a persistent positive bias?
   (i.e., does holding short perp + long spot generate consistent income?)

2. TIMING COMPONENT: Can extreme funding rates predict future returns?
   (i.e., does z-score > 2 predict reversals?)

3. BASIS COMPONENT: Does the perp premium/discount predict returns?

4. CROSS-ASSET STRUCTURE: Do different assets have different funding dynamics?

5. REGIME DEPENDENCY: Does the edge persist across bull/bear/sideways markets?

HYPOTHESES:
-----------
H1: Crypto funding rates have a persistent POSITIVE bias (retail is structurally long)
    → This makes delta-neutral short carry a positive-EV strategy

H2: Extreme positive funding (z > 2) predicts negative forward returns (12-48h)
    → Crowded longs get liquidated

H3: The effect is ASYMMETRIC — positive extremes predict better than negative
    → Because retail longs outnumber shorts structurally

H4: Altcoins have stronger effects than BTC
    → Less institutional absorption, more retail-dominated

H5: The carry is large enough to compound meaningfully after costs
    → Need > 15% annualized net of fees and hedging costs

METHODOLOGY:
------------
- Data: Binance Futures, all available history (up to 2 years)
- Assets: BTC, ETH, SOL, XRP, BNB, DOGE (covering top liquid perps)
- Forward returns: 8h, 24h, 48h, 72h (measured from funding snapshot)
- Controls: Transaction costs (0.1%), slippage (0.05%), funding costs
- Statistical tests: t-test, bootstrap confidence intervals, regime splits
================================================================================
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetcher import DataFetcher
from src.data.structural import StructuralDataFetcher

# ============================================================
# Configuration
# ============================================================

SYMBOLS_PERP = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]
SYMBOLS_SPOT = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "DOGE/USDT"]
FETCH_DAYS = 365  # 1 year of funding data
OHLCV_DAYS = 400  # Extra for warmup

# Forward return horizons (in hours)
FWD_HORIZONS = [8, 24, 48, 72]

# Transaction costs
TAKER_FEE = 0.001       # 0.1% taker fee
MAKER_FEE = 0.0002      # 0.02% maker fee (for limit orders)
SLIPPAGE = 0.0005        # 0.05% estimated slippage


def section_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def subsection(title):
    print(f"\n--- {title} ---\n")


# ============================================================
# PHASE 1: DATA COLLECTION
# ============================================================

def fetch_all_data():
    """Fetch funding rates and price data for all symbols."""
    section_header("PHASE 1: DATA COLLECTION")

    struct_fetcher = StructuralDataFetcher()
    price_fetcher = DataFetcher()

    data = {}

    for sym_perp, sym_spot in zip(SYMBOLS_PERP, SYMBOLS_SPOT):
        name = sym_perp.replace("USDT", "")
        print(f"Fetching {name}...")

        try:
            # Funding rates (8h intervals, up to 1yr)
            funding = struct_fetcher.fetch_funding_rate_history(sym_perp, days=FETCH_DAYS)

            # Price data (1h OHLCV)
            price = price_fetcher.fetch(sym_spot, timeframe="1h", days=OHLCV_DAYS)

            if funding.empty or price.empty:
                print(f"  SKIP: No data for {name}")
                continue

            # Normalize timezone info
            if hasattr(funding.index, 'tz') and funding.index.tz is not None:
                funding.index = funding.index.tz_localize(None)
            if hasattr(price.index, 'tz') and price.index.tz is not None:
                price.index = price.index.tz_localize(None)

            data[name] = {
                "funding": funding,
                "price": price,
            }

            print(f"  Funding: {len(funding)} records, {funding.index[0]} to {funding.index[-1]}")
            print(f"  Price: {len(price)} bars, {price.index[0]} to {price.index[-1]}")

        except Exception as e:
            print(f"  ERROR fetching {name}: {e}")
            continue

    print(f"\nLoaded {len(data)} assets: {list(data.keys())}")
    return data


# ============================================================
# PHASE 2: STUDY 1 — FUNDING RATE STRUCTURAL BIAS
# ============================================================

def study_1_funding_bias(data):
    """H1: Do funding rates have a persistent positive bias?"""
    section_header("STUDY 1: FUNDING RATE STRUCTURAL BIAS")

    print("QUESTION: Is there a persistent positive funding rate (retail long bias)?")
    print("If YES → delta-neutral carry is a viable base strategy.\n")

    results = []

    for name, d in data.items():
        fr = d["funding"]["funding_rate"]

        # Basic statistics
        mean_rate = fr.mean()
        median_rate = fr.median()
        std_rate = fr.std()
        pct_positive = (fr > 0).mean() * 100
        annualized = mean_rate * 3 * 365  # 3 funding periods per day
        annualized_pct = annualized * 100

        # Statistical significance: is the mean significantly > 0?
        t_stat, p_value = stats.ttest_1samp(fr.dropna(), 0)

        # Confidence interval
        ci_low, ci_high = stats.t.interval(
            0.95, len(fr) - 1,
            loc=fr.mean(),
            scale=stats.sem(fr.dropna())
        )
        ci_low_ann = ci_low * 3 * 365 * 100
        ci_high_ann = ci_high * 3 * 365 * 100

        # Monthly breakdown
        monthly = fr.resample("ME").agg(["mean", "count"])
        pct_months_positive = (monthly["mean"] > 0).mean() * 100

        results.append({
            "Asset": name,
            "Mean FR (bps)": f"{mean_rate * 10000:.2f}",
            "Median FR (bps)": f"{median_rate * 10000:.2f}",
            "Annualized %": f"{annualized_pct:.1f}%",
            "% Positive": f"{pct_positive:.1f}%",
            "t-stat": f"{t_stat:.2f}",
            "p-value": f"{p_value:.4f}",
            "95% CI (ann)": f"[{ci_low_ann:.1f}%, {ci_high_ann:.1f}%]",
            "% Months +": f"{pct_months_positive:.0f}%",
            "N": len(fr),
        })

        print(f"{name}:")
        print(f"  Mean funding rate: {mean_rate * 10000:.2f} bps per 8h")
        print(f"  Annualized carry: {annualized_pct:.1f}%")
        print(f"  Positive {pct_positive:.1f}% of the time")
        print(f"  t-stat: {t_stat:.2f}, p-value: {p_value:.4f}")
        print(f"  95% CI annualized: [{ci_low_ann:.1f}%, {ci_high_ann:.1f}%]")
        print(f"  Positive months: {pct_months_positive:.0f}%")
        print()

    # Summary table
    df_results = pd.DataFrame(results)
    print("\n=== SUMMARY TABLE ===")
    print(df_results.to_string(index=False))

    # VERDICT
    print("\n=== VERDICT ON H1 ===")
    significant = sum(1 for r in results if float(r["p-value"]) < 0.05 and float(r["t-stat"]) > 0)
    print(f"Assets with significant positive bias: {significant}/{len(results)}")
    if significant >= len(results) // 2:
        print("✓ H1 SUPPORTED: Funding rates have a structural positive bias")
        print("  → Delta-neutral carry (long spot, short perp) is positive EV")
    else:
        print("✗ H1 REJECTED: No consistent positive bias across assets")

    return results


# ============================================================
# PHASE 3: STUDY 2 — EXTREME FUNDING PREDICTS RETURNS
# ============================================================

def study_2_extreme_funding_prediction(data):
    """H2 + H3: Do extreme funding rates predict forward returns?"""
    section_header("STUDY 2: EXTREME FUNDING → FORWARD RETURNS")

    print("QUESTION: When funding is extreme (z > 2), do prices reverse?")
    print("Sub-question: Is the effect asymmetric (positive vs negative extremes)?\n")

    all_results = []

    for name, d in data.items():
        fr = d["funding"]
        price = d["price"]

        # Align funding to hourly price data (forward-fill)
        if hasattr(fr.index, 'tz') and fr.index.tz is not None:
            fr.index = fr.index.tz_localize(None)
        if hasattr(price.index, 'tz') and price.index.tz is not None:
            price.index = price.index.tz_localize(None)

        # Create funding z-score on price timeframe
        funding_aligned = fr["funding_rate"].reindex(price.index, method="ffill")
        funding_ma = funding_aligned.rolling(90 * 3, min_periods=30).mean()  # ~30d MA
        funding_std = funding_aligned.rolling(90 * 3, min_periods=30).std()
        funding_z = (funding_aligned - funding_ma) / (funding_std + 1e-10)

        # Forward returns at different horizons
        fwd_returns = {}
        for h in FWD_HORIZONS:
            fwd_returns[f"fwd_{h}h"] = price["close"].pct_change(h).shift(-h)

        subsection(f"{name} — Extreme Funding Analysis")

        for z_threshold in [1.5, 2.0, 2.5]:
            print(f"  Z-threshold: ±{z_threshold}")

            # Positive extreme: crowd is VERY LONG → expect reversal DOWN
            pos_extreme = funding_z > z_threshold
            # Negative extreme: crowd is VERY SHORT → expect reversal UP
            neg_extreme = funding_z < -z_threshold

            for horizon_name, fwd_ret in fwd_returns.items():
                # Positive extreme → SHORT signal
                pos_returns = fwd_ret[pos_extreme].dropna()
                neg_returns = fwd_ret[neg_extreme].dropna()
                all_returns = fwd_ret.dropna()

                if len(pos_returns) < 5:
                    pos_mean = np.nan
                    pos_wr = np.nan
                    pos_t = np.nan
                    pos_p = np.nan
                else:
                    # For positive extreme, we'd SHORT, so our return = -fwd_return
                    short_returns = -pos_returns
                    pos_mean = short_returns.mean() * 100
                    pos_wr = (short_returns > 0).mean() * 100
                    pos_t, pos_p = stats.ttest_1samp(short_returns, 0)

                if len(neg_returns) < 5:
                    neg_mean = np.nan
                    neg_wr = np.nan
                    neg_t = np.nan
                    neg_p = np.nan
                else:
                    # For negative extreme, we'd LONG
                    neg_mean = neg_returns.mean() * 100
                    neg_wr = (neg_returns > 0).mean() * 100
                    neg_t, neg_p = stats.ttest_1samp(neg_returns, 0)

                baseline_mean = all_returns.mean() * 100

                all_results.append({
                    "Asset": name,
                    "Z": z_threshold,
                    "Horizon": horizon_name,
                    "SHORT_n": len(pos_returns),
                    "SHORT_mean%": round(pos_mean, 3) if not np.isnan(pos_mean) else None,
                    "SHORT_WR%": round(pos_wr, 1) if not np.isnan(pos_wr) else None,
                    "SHORT_t": round(pos_t, 2) if not np.isnan(pos_t) else None,
                    "SHORT_p": round(pos_p, 4) if not np.isnan(pos_p) else None,
                    "LONG_n": len(neg_returns),
                    "LONG_mean%": round(neg_mean, 3) if not np.isnan(neg_mean) else None,
                    "LONG_WR%": round(neg_wr, 1) if not np.isnan(neg_wr) else None,
                    "LONG_t": round(neg_t, 2) if not np.isnan(neg_t) else None,
                    "LONG_p": round(neg_p, 4) if not np.isnan(neg_p) else None,
                    "Baseline%": round(baseline_mean, 3),
                })

            # Print best horizon results
            best = [r for r in all_results if r["Asset"] == name and r["Z"] == z_threshold]
            for r in best:
                short_sig = "**" if r["SHORT_p"] is not None and r["SHORT_p"] < 0.05 else ""
                long_sig = "**" if r["LONG_p"] is not None and r["LONG_p"] < 0.05 else ""
                print(f"    {r['Horizon']}: SHORT(n={r['SHORT_n']}, "
                      f"mean={r['SHORT_mean%']}%, WR={r['SHORT_WR%']}%){short_sig} | "
                      f"LONG(n={r['LONG_n']}, mean={r['LONG_mean%']}%, "
                      f"WR={r['LONG_WR%']}%){long_sig}")

    # Summary
    df_res = pd.DataFrame(all_results)
    print("\n=== SIGNIFICANT RESULTS (p < 0.05) ===\n")
    sig_short = df_res[(df_res["SHORT_p"].notna()) & (df_res["SHORT_p"] < 0.05)]
    sig_long = df_res[(df_res["LONG_p"].notna()) & (df_res["LONG_p"] < 0.05)]

    if not sig_short.empty:
        print("SHORT signals (positive extreme funding → short):")
        print(sig_short[["Asset", "Z", "Horizon", "SHORT_n", "SHORT_mean%",
                         "SHORT_WR%", "SHORT_t", "SHORT_p"]].to_string(index=False))

    if not sig_long.empty:
        print("\nLONG signals (negative extreme funding → long):")
        print(sig_long[["Asset", "Z", "Horizon", "LONG_n", "LONG_mean%",
                         "LONG_WR%", "LONG_t", "LONG_p"]].to_string(index=False))

    print("\n=== VERDICT ON H2/H3 ===")
    n_sig_short = len(sig_short)
    n_sig_long = len(sig_long)
    print(f"Significant SHORT signals: {n_sig_short}")
    print(f"Significant LONG signals: {n_sig_long}")
    if n_sig_short > n_sig_long:
        print("✓ H3 SUPPORTED: Positive extreme funding (crowd long) is more predictive")
        print("  → SHORT-side timing alpha is stronger")
    elif n_sig_long > n_sig_short:
        print("✗ H3 REJECTED: Negative extreme is more predictive (unexpected)")
    else:
        print("≈ H3 INCONCLUSIVE: Both sides similar")

    return df_res


# ============================================================
# PHASE 4: STUDY 3 — CARRY RETURN SIMULATION
# ============================================================

def study_3_carry_simulation(data):
    """H5: Is the carry large enough to compound after costs?"""
    section_header("STUDY 3: DELTA-NEUTRAL CARRY SIMULATION")

    print("QUESTION: What are the ACTUAL returns from delta-neutral carry?")
    print("Strategy: Long spot, short perp, earn funding rate payments.")
    print("Costs: Maker fees on entry/exit, slippage, spot-perp tracking error.\n")

    for name, d in data.items():
        fr = d["funding"]["funding_rate"]
        price = d["price"]["close"]

        subsection(f"{name} — Carry Simulation")

        # Simulate collecting every funding payment over the period
        total_funding_payments = fr.sum()
        n_payments = len(fr)
        n_days = (fr.index[-1] - fr.index[0]).days

        # Costs per round-trip (enter + exit)
        # Spot: maker buy + maker sell = 2 * 0.02% = 0.04%
        # Perp: maker short + maker cover = 2 * 0.02% = 0.04%
        # Slippage: 2 * 0.05% = 0.10%
        # Total round-trip: 0.18%
        rt_cost = 2 * (MAKER_FEE + SLIPPAGE)  # for spot + perp combined = ~0.3%

        # Assume we hold position for different durations
        for hold_days in [30, 90, 180, 365]:
            effective_days = min(hold_days, n_days)
            n_periods = int(effective_days * 3)  # 3 funding payments per day

            # Take the most recent data
            recent_fr = fr.tail(n_periods)
            if len(recent_fr) < 10:
                continue

            gross_carry = recent_fr.sum()
            gross_carry_pct = gross_carry * 100
            annualized = (gross_carry / effective_days) * 365 * 100

            # Cost: one entry + one exit (spread across hold period)
            net_carry = gross_carry - rt_cost
            net_carry_pct = net_carry * 100
            net_annualized = (net_carry / effective_days) * 365 * 100

            # Volatility of daily funding
            daily_funding = recent_fr.resample("D").sum()
            funding_vol = daily_funding.std()
            sharpe = (daily_funding.mean() / (funding_vol + 1e-10)) * np.sqrt(365)

            # Max drawdown of cumulative funding
            cum_funding = recent_fr.cumsum()
            max_dd = (cum_funding - cum_funding.cummax()).min() * 100

            # What % of days were negative?
            pct_neg_days = (daily_funding < 0).mean() * 100

            print(f"  Hold {hold_days}d ({effective_days}d actual):")
            print(f"    Gross carry: {gross_carry_pct:.2f}% ({annualized:.1f}% ann)")
            print(f"    Net carry:   {net_carry_pct:.2f}% ({net_annualized:.1f}% ann)")
            print(f"    Funding Sharpe: {sharpe:.2f}")
            print(f"    Max funding DD: {max_dd:.2f}%")
            print(f"    Negative days: {pct_neg_days:.1f}%")

        # Study: funding rate autocorrelation (is it persistent or mean-reverting?)
        print(f"\n  Autocorrelation:")
        for lag in [1, 3, 7, 21]:  # In funding periods (8h each)
            ac = fr.autocorr(lag=lag)
            print(f"    Lag {lag} ({lag*8}h): {ac:.3f}")

        # Distribution analysis
        print(f"\n  Distribution:")
        print(f"    Skew: {fr.skew():.3f}")
        print(f"    Kurtosis: {fr.kurtosis():.3f}")
        percentiles = fr.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        print(f"    1st pct: {percentiles.iloc[0]*10000:.2f} bps")
        print(f"    5th pct: {percentiles.iloc[1]*10000:.2f} bps")
        print(f"    25th pct: {percentiles.iloc[2]*10000:.2f} bps")
        print(f"    Median:  {percentiles.iloc[3]*10000:.2f} bps")
        print(f"    75th pct: {percentiles.iloc[4]*10000:.2f} bps")
        print(f"    95th pct: {percentiles.iloc[5]*10000:.2f} bps")
        print(f"    99th pct: {percentiles.iloc[6]*10000:.2f} bps")


# ============================================================
# PHASE 5: STUDY 4 — CROSS-EXCHANGE FUNDING SPREAD
# ============================================================

def study_4_cross_exchange(data):
    """Can we capture spread between exchanges' funding rates?"""
    section_header("STUDY 4: CROSS-EXCHANGE FUNDING SPREAD (CONCEPTUAL)")

    print("NOTE: This study requires data from multiple exchanges.")
    print("We currently only have Binance data. This section provides the")
    print("framework for when we add Bybit/OKX/dYdX data.\n")

    print("HYPOTHESIS: Different exchanges have different funding rates because")
    print("they have different user bases (retail vs institutional mix).")
    print("")
    print("POTENTIAL STRATEGY:")
    print("  - Short perp on exchange with HIGHEST positive funding")
    print("  - Long perp on exchange with LOWEST (or negative) funding")
    print("  - Net: earn the spread, delta-neutral")
    print("")
    print("EDGES vs single-exchange carry:")
    print("  + Higher yield (capturing the spread)")
    print("  + Less capital (no spot position needed)")
    print("  - Higher complexity (2 exchange accounts)")
    print("  - Counterparty risk on 2 exchanges")
    print("  - Need to monitor both positions")
    print("")
    print("TO DO: Add Bybit funding rate fetcher and compare.")

    # At minimum, we can check if Binance funding varies by time-of-day
    for name, d in data.items():
        fr = d["funding"]
        if "funding_rate" not in fr.columns:
            continue
        fr_copy = fr.copy()
        fr_copy["hour"] = fr_copy.index.hour

        print(f"\n{name} — Funding by Time of Day (UTC):")
        for hour in [0, 8, 16]:
            h_data = fr_copy[fr_copy["hour"] == hour]["funding_rate"]
            if len(h_data) > 0:
                print(f"  {hour:02d}:00 UTC: mean={h_data.mean()*10000:.2f} bps, "
                      f"median={h_data.median()*10000:.2f} bps, n={len(h_data)}")


# ============================================================
# PHASE 6: STUDY 5 — REGIME ANALYSIS
# ============================================================

def study_5_regime_analysis(data):
    """Does the edge persist across market regimes?"""
    section_header("STUDY 5: REGIME DEPENDENCY")

    print("QUESTION: Does funding carry work in ALL market regimes,")
    print("or only in specific conditions?\n")

    for name, d in data.items():
        price = d["price"]["close"]
        fr = d["funding"]["funding_rate"]

        subsection(f"{name} — Regime Split")

        # Simple regime classification using 30-day returns
        monthly_ret = price.resample("ME").last().pct_change()
        if len(monthly_ret) < 4:
            print(f"  Insufficient data for regime analysis")
            continue

        # Classify months
        bull_months = monthly_ret[monthly_ret > 0.05].index     # > 5% monthly gain
        bear_months = monthly_ret[monthly_ret < -0.05].index    # > 5% monthly loss
        sideways_months = monthly_ret[
            (monthly_ret >= -0.05) & (monthly_ret <= 0.05)
        ].index

        # Get funding for each regime
        for regime_name, months in [("BULL", bull_months), ("BEAR", bear_months),
                                     ("SIDEWAYS", sideways_months)]:
            regime_funding = []
            for m in months:
                # Get all funding rates in this month
                month_mask = (fr.index.year == m.year) & (fr.index.month == m.month)
                regime_funding.extend(fr[month_mask].tolist())

            if len(regime_funding) < 10:
                print(f"  {regime_name}: insufficient data (n={len(regime_funding)})")
                continue

            regime_funding = np.array(regime_funding)
            mean_ann = regime_funding.mean() * 3 * 365 * 100
            pct_pos = (regime_funding > 0).mean() * 100

            print(f"  {regime_name} ({len(months)} months, {len(regime_funding)} periods):")
            print(f"    Mean funding (ann): {mean_ann:.1f}%")
            print(f"    Positive: {pct_pos:.1f}%")


# ============================================================
# PHASE 7: STUDY 6 — COMBINED EDGE ESTIMATION
# ============================================================

def study_6_combined_edge(data):
    """What's the TOTAL realistic edge combining carry + timing?"""
    section_header("STUDY 6: COMBINED EDGE ESTIMATION")

    print("QUESTION: What can we realistically compound annually?")
    print("Combining: base carry + timing alpha - costs - risk adjustments\n")

    for name, d in data.items():
        funding_df = d["funding"]
        fr = funding_df["funding_rate"]
        price = d["price"]["close"]

        subsection(f"{name} — Combined Edge")

        n_days = (fr.index[-1] - fr.index[0]).days
        if n_days < 90:
            print(f"  Insufficient history: {n_days} days")
            continue

        # --- BASE CARRY ---
        # Simple: collect all funding payments
        daily_funding = fr.resample("D").sum()
        annual_carry_gross = daily_funding.mean() * 365 * 100
        carry_sharpe = (daily_funding.mean() / (daily_funding.std() + 1e-10)) * np.sqrt(365)

        # --- COSTS ---
        # Entry/exit: assume 4 round-trips per year (quarterly rebalancing)
        annual_rt_costs = 4 * 2 * (MAKER_FEE + SLIPPAGE) * 100  # 4 round-trips

        # Spot-perp tracking: basis can move against us
        # Estimate from actual basis movement
        if "mark_price" in funding_df.columns:
            mark = funding_df["mark_price"]
            # Approximate index price from spot
            spot_aligned = price.reindex(mark.index, method="ffill")
            basis = (mark - spot_aligned) / spot_aligned
            basis_vol = basis.dropna().std() * np.sqrt(3 * 365) * 100  # Annualized
        else:
            basis_vol = 5.0  # Conservative estimate

        annual_carry_net = annual_carry_gross - annual_rt_costs
        carry_max_dd = (daily_funding.cumsum() - daily_funding.cumsum().cummax()).min() * 100

        print(f"  BASE CARRY:")
        print(f"    Gross annual: {annual_carry_gross:.1f}%")
        print(f"    Costs (4 RT): -{annual_rt_costs:.2f}%")
        print(f"    Net annual:   {annual_carry_net:.1f}%")
        print(f"    Carry Sharpe: {carry_sharpe:.2f}")
        print(f"    Max DD:       {carry_max_dd:.2f}%")
        print(f"    Basis vol:    {basis_vol:.1f}% ann")

        # --- TIMING ALPHA ---
        # From Study 2 results: add timing on top of carry
        funding_aligned = fr.reindex(price.index, method="ffill")
        funding_z = (
            (funding_aligned - funding_aligned.rolling(270, min_periods=30).mean())
            / (funding_aligned.rolling(270, min_periods=30).std() + 1e-10)
        )

        # When z > 2, over-weight short perp position (1.5x vs 1.0x base)
        # When z < -2, reduce short (0.5x vs 1.0x)
        # Else: 1.0x (base carry only)
        timing_multiplier = pd.Series(1.0, index=price.index)
        timing_multiplier[funding_z > 2.0] = 1.5
        timing_multiplier[funding_z < -2.0] = 0.5

        # The extra return from timing is proportional to funding * (multiplier - 1)
        timing_extra = (funding_aligned * (timing_multiplier - 1)).resample("D").sum()
        annual_timing = timing_extra.mean() * 365 * 100

        print(f"\n  TIMING ALPHA (z-based sizing):")
        print(f"    Annual contribution: {annual_timing:.2f}%")

        # --- COMBINED ---
        total_edge = annual_carry_net + annual_timing
        print(f"\n  COMBINED EDGE:")
        print(f"    Total annual (est): {total_edge:.1f}%")

        # --- CAPACITY ---
        # How much capital before we move markets?
        # BTC: ~$20B OI → can do 0.1% = $20M without impact
        # ETH: ~$8B OI → can do 0.1% = $8M
        # Alts: ~$1-3B OI → can do $1-3M
        if name == "BTC":
            capacity = 20_000_000
        elif name == "ETH":
            capacity = 8_000_000
        else:
            capacity = 2_000_000

        print(f"\n  CAPACITY:")
        print(f"    Estimated max: ${capacity/1e6:.0f}M (0.1% of typical OI)")
        print(f"    At max capital, annual $: ${capacity * total_edge / 100:,.0f}")


# ============================================================
# PHASE 8: STUDY 7 — RISK ANALYSIS
# ============================================================

def study_7_risks(data):
    """What can go wrong?"""
    section_header("STUDY 7: RISK ANALYSIS")

    print("Every edge has kill scenarios. Here are ours:\n")

    print("1. FUNDING RATE COMPRESSION")
    print("   Risk: As more people do this trade, funding rates drop toward 0")
    print("   Mitigation: Monitor rolling average, exit if mean drops below 5% ann")
    print()

    print("2. EXCHANGE COUNTERPARTY RISK")
    print("   Risk: Exchange insolvency (FTX 2022)")
    print("   Mitigation: Max 25% of capital per exchange, spread across 3+")
    print()

    print("3. FUNDING RATE INVERSION (PERSISTENT NEGATIVE)")
    print("   Risk: Bear market → funding goes deeply negative → we PAY funding")
    print("   Mitigation: Close position when funding z < -2 for > 72h")

    # Actually check how often this happened
    for name, d in data.items():
        fr = d["funding"]["funding_rate"]
        # Find longest streak of negative funding
        neg_streak = 0
        max_neg_streak = 0
        for rate in fr:
            if rate < 0:
                neg_streak += 1
                max_neg_streak = max(max_neg_streak, neg_streak)
            else:
                neg_streak = 0
        print(f"   {name}: longest negative streak = {max_neg_streak} periods "
              f"({max_neg_streak * 8}h = {max_neg_streak * 8 / 24:.1f} days)")

    print()
    print("4. SPOT-PERP BASIS DIVERGENCE")
    print("   Risk: Perp trades at persistent discount to spot → hedging loss")
    print("   Mitigation: Monitor basis, close if basis < -1%")
    print()

    print("5. LIQUIDATION RISK")
    print("   Risk: Spot and perp positions margin-called during extreme volatility")
    print("   Mitigation: Max 2x leverage on perp, 50% of capital in spot (no leverage)")
    print()

    print("6. CORRELATION RISK")
    print("   Risk: All crypto assets move together, all funding rates spike negative")

    # Check correlation of funding rates
    funding_df = pd.DataFrame()
    for name, d in data.items():
        fr = d["funding"]["funding_rate"]
        daily_fr = fr.resample("D").mean()
        funding_df[name] = daily_fr

    if len(funding_df.columns) > 1:
        corr = funding_df.corr()
        print("   Funding rate correlations:")
        print(corr.to_string())

    print()
    print("7. REGULATORY RISK")
    print("   Risk: Perp futures banned in your jurisdiction")
    print("   Mitigation: Use offshore exchanges where legal, stay compliant")


# ============================================================
# MAIN
# ============================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         RESEARCH PAPER #001: THE CRYPTO FUNDING RATE EDGE       ║
║                                                                  ║
║  Senior Quant Research — Empirical Investigation                 ║
║  Date: April 2026                                                ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # Phase 1: Collect data
    data = fetch_all_data()

    if not data:
        print("\nFATAL: No data collected. Check internet connection.")
        return

    # Phase 2-7: Run all studies
    study_1_funding_bias(data)
    study_2_extreme_funding_prediction(data)
    study_3_carry_simulation(data)
    study_4_cross_exchange(data)
    study_5_regime_analysis(data)
    study_6_combined_edge(data)
    study_7_risks(data)

    # Final Summary
    section_header("RESEARCH CONCLUSIONS")
    print("""
FINDINGS SUMMARY:
=================

The studies above answer these questions:

1. IS THERE A STRUCTURAL POSITIVE FUNDING BIAS?
   → See Study 1 results

2. CAN EXTREME FUNDING PREDICT RETURNS?
   → See Study 2 results (with p-values)

3. IS THE CARRY LARGE ENOUGH AFTER COSTS?
   → See Study 3 simulation

4. WHICH REGIMES DOES IT WORK IN?
   → See Study 5 regime analysis

5. WHAT'S THE REALISTIC COMBINED EDGE?
   → See Study 6 combined estimation

6. WHAT CAN KILL THIS EDGE?
   → See Study 7 risk analysis

NEXT STEPS (if research supports the edge):
=============================================
- Engineer role: Design the production system architecture
- Developer role: Implement the system
- Quant role: Optimize parameters, size positions, monitor decay
    """)


if __name__ == "__main__":
    main()

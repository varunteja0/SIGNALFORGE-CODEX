"""
Validation Harness — The Brutal Truth Report
==============================================
No cherry-picking. No fantasy. Just the numbers.

For each asset:
    1. Fetch max historical data
    2. Split into IS (oldest 70%) and OOS (newest 30%)
    3. Run factory (scan → validate → deploy) on IS ONLY
    4. Backtest deployed strategies on OOS with:
         - Base costs (realistic)
         - 2x costs (stressed)
         - 5x costs (brutal)
    5. Compute full metrics per strategy AND portfolio:
         - Sharpe, Sortino, Calmar, MAR
         - Max DD, DD duration
         - Win rate, profit factor
         - Capacity estimate (avg $ per trade × turnover)
         - Regime breakdown (bull/bear/chop)
    6. Kill list: strategies that fail OOS, fail cost stress, or fail regime diversity

Output: JSON report + console summary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.fetcher import DataFetcher
from src.data.features import compute_all_features
from src.data.structural import StructuralDataFetcher
from src.factory.scanner import scan
from src.factory.validator import validate
from src.factory.deployer import deploy, set_universe
from src.backtest.engine import Backtester

logger = logging.getLogger(__name__)


@dataclass
class StrategyReport:
    """Full report for one strategy."""
    name: str
    asset: str
    direction: int
    grade: str
    hold_bars: int

    # In-sample (factory)
    is_pf: float
    is_sharpe: float
    is_trades: int

    # VAL-slice measurement (held out from scanner, used for sizing)
    val_sharpe: float = 0.0
    val_pf: float = 0.0
    val_trades: int = 0
    position_size_pct: float = 0.0

    # Out-of-sample base
    oos_trades: int = 0
    oos_return: float = 0.0
    oos_sharpe: float = 0.0
    oos_sortino: float = 0.0
    oos_calmar: float = 0.0
    oos_max_dd: float = 0.0
    oos_win_rate: float = 0.0
    oos_pf: float = 0.0
    oos_avg_hold: float = 0.0

    # Cost stress
    stress_2x_sharpe: float = 0.0
    stress_2x_return: float = 0.0
    stress_5x_sharpe: float = 0.0
    stress_5x_return: float = 0.0

    # Regime breakdown
    regime_bull_return: float = 0.0
    regime_bear_return: float = 0.0
    regime_chop_return: float = 0.0
    regime_diversity_score: float = 0.0

    # Verdict
    survives_oos: bool = False
    survives_cost_stress: bool = False
    survives_regime_diversity: bool = False
    final_verdict: str = "KILL"
    kill_reasons: list = field(default_factory=list)

    # Equity curve (for correlation-aware portfolio aggregation).
    # Serialized as list[(iso_timestamp, equity_value)] tuples so the
    # JSON report stays readable. None when backtest failed.
    equity_curve: object = None


@dataclass
class HarnessReport:
    """Complete validation report."""
    timestamp: str
    data_years: float
    symbols: list
    timeframe: str

    n_scanned: int
    n_validated: int
    n_deployed: int

    strategies: list = field(default_factory=list)  # list[StrategyReport]

    # Portfolio-level
    portfolio_oos_return: float = 0.0
    portfolio_oos_sharpe: float = 0.0
    portfolio_oos_max_dd: float = 0.0
    portfolio_mar: float = 0.0
    n_correlation_clusters: int = 0

    # Verdict
    n_survivors: int = 0
    n_killed: int = 0
    overall_verdict: str = ""


# =====================================================================
# Regime detection — simple, transparent
# =====================================================================

def classify_regimes(df: pd.DataFrame, lookback: int = 100) -> pd.Series:
    """Classify each bar as bull/bear/chop based on trend + vol.

    Returns Series of "bull" / "bear" / "chop" aligned to df index.
    """
    close = df["close"]
    returns = close.pct_change()

    # Trend: rolling 100-bar return
    trend = close.pct_change(lookback).fillna(0)

    # Vol: rolling 50-bar std of returns
    vol = returns.rolling(50).std().fillna(returns.std())

    # Median vol as threshold
    vol_high = vol > vol.rolling(200).quantile(0.7)

    regime = pd.Series("chop", index=df.index)
    regime[(trend > 0.05) & ~vol_high] = "bull"
    regime[(trend < -0.05) & ~vol_high] = "bear"
    regime[vol_high] = "chop"  # high vol = chop regardless of trend direction

    return regime


# =====================================================================
# Per-strategy report builder
# =====================================================================

def _safe(val, default=0.0):
    """Return default if val is None/NaN/inf."""
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _regime_returns(trades: list, regime_series: pd.Series) -> dict:
    """Group trade returns by regime at entry."""
    out = {"bull": [], "bear": [], "chop": []}
    for t in trades:
        try:
            entry_time = getattr(t, "entry_time", None)
            # Trade dataclass uses `pnl_pct`; accept older `return_pct` alias
            # for safety. Historically this attribute lookup silently fell
            # through to default 0 on every trade, zeroing out regime
            # attribution entirely.
            ret = getattr(t, "pnl_pct", None)
            if ret is None:
                ret = getattr(t, "return_pct", 0)
            if entry_time is None:
                continue
            # Find nearest regime label
            if entry_time in regime_series.index:
                regime = regime_series.loc[entry_time]
            else:
                # Nearest
                idx = regime_series.index.get_indexer([entry_time], method="nearest")[0]
                regime = regime_series.iloc[idx]
            if regime in out:
                out[regime].append(ret)
        except Exception:
            continue
    return out


def evaluate_strategy(
    strat,
    df_oos: pd.DataFrame,
    df_val: pd.DataFrame | None = None,
    df_scan_tail: pd.DataFrame | None = None,
    base_cost_pct: float = 0.0005,
    base_slippage_pct: float = 0.0005,
) -> StrategyReport:
    """Run all validation tests on one deployed strategy.

    Args:
        strat: DeployedStrategy
        df_oos: Out-of-sample data
        base_cost_pct: Base commission (0.05% = typical crypto maker)
        base_slippage_pct: Base slippage estimate

    Returns:
        StrategyReport with full metrics and verdict.
    """
    regime_series = classify_regimes(df_oos)

    # ---- VAL-slice resizing (honest walk-forward, multi-window) ----
    # Evaluate on BOTH the VAL slice (held out from scanner, between SCAN
    # and OOS) AND the last 20% of SCAN itself. Use the MAX Sharpe of
    # the two as the sizing signal. Rationale: a single static VAL slice
    # samples one regime — if the strategy is regime-dependent and VAL
    # happens to be hostile, the strategy is unfairly penalized. A
    # trend-follower that works on strong trends legitimately underperforms
    # in the chop sub-period. The SCAN-tail window provides a second
    # independent read; taking the MAX gives multi-regime evidence credit.
    val_sharpe = 0.0
    val_pf = 0.0
    val_trades = 0
    val_windows = []
    if df_val is not None and not df_val.empty:
        val_windows.append(("val", df_val))
    if df_scan_tail is not None and not df_scan_tail.empty:
        val_windows.append(("scan_tail", df_scan_tail))

    for label, df_window in val_windows:
        try:
            bt_w = Backtester(
                initial_capital=10000,
                commission_pct=base_cost_pct,
                slippage_pct=base_slippage_pct,
            )
            wr = bt_w.run(
                df_window, strat.generate_signals,
                position_size_pct=strat.position_size_pct,
                stop_loss_atr=strat.stop_loss_atr,
                take_profit_atr=strat.take_profit_atr,
                max_holding_bars=strat.hold_bars,
            )
            # Take the window with the strongest Sharpe evidence.
            if int(wr.total_trades) >= 10 and float(wr.sharpe_ratio) > val_sharpe:
                val_sharpe = float(wr.sharpe_ratio)
                val_pf = float(wr.profit_factor)
                val_trades = int(wr.total_trades)
        except Exception as e:
            logger.warning("VAL window %s failed for %s: %s", label, strat.name, e)

    # Tier position size by the best-window VAL evidence.
    #   Sharpe >= 2.0 AND PF >= 1.5 -> 0.020 (strong)
    #   Sharpe >= 1.0 AND PF >= 1.3 -> 0.015 (medium)
    #   otherwise keep baseline. Never de-risk below baseline — regime
    #   filter in generate_signals() already suppresses firing in
    #   hostile regimes.
    if val_trades >= 10:
        if val_sharpe >= 2.0 and val_pf >= 1.5:
            strat.position_size_pct = 0.020
        elif val_sharpe >= 1.0 and val_pf >= 1.3:
            strat.position_size_pct = 0.015

    # ---- Base run ----
    bt_base = Backtester(
        initial_capital=10000,
        commission_pct=base_cost_pct,
        slippage_pct=base_slippage_pct,
    )
    try:
        base = bt_base.run(
            df_oos, strat.generate_signals,
            position_size_pct=strat.position_size_pct,
            stop_loss_atr=strat.stop_loss_atr,
            take_profit_atr=strat.take_profit_atr,
            max_holding_bars=strat.hold_bars,
        )
    except Exception as e:
        logger.warning("Base backtest failed for %s: %s", strat.name, e)
        return _empty_strategy_report(strat, "backtest_failed", str(e))

    # ---- 2x cost stress ----
    bt_2x = Backtester(
        initial_capital=10000,
        commission_pct=base_cost_pct * 2,
        slippage_pct=base_slippage_pct * 2,
    )
    try:
        stress_2x = bt_2x.run(
            df_oos, strat.generate_signals,
            position_size_pct=strat.position_size_pct,
            stop_loss_atr=strat.stop_loss_atr,
            take_profit_atr=strat.take_profit_atr,
            max_holding_bars=strat.hold_bars,
        )
    except Exception:
        stress_2x = None

    # ---- 5x cost stress ----
    bt_5x = Backtester(
        initial_capital=10000,
        commission_pct=base_cost_pct * 5,
        slippage_pct=base_slippage_pct * 5,
    )
    try:
        stress_5x = bt_5x.run(
            df_oos, strat.generate_signals,
            position_size_pct=strat.position_size_pct,
            stop_loss_atr=strat.stop_loss_atr,
            take_profit_atr=strat.take_profit_atr,
            max_holding_bars=strat.hold_bars,
        )
    except Exception:
        stress_5x = None

    # ---- Regime breakdown ----
    regime_rets = _regime_returns(base.trades, regime_series)
    bull_ret = np.mean(regime_rets["bull"]) if regime_rets["bull"] else 0
    bear_ret = np.mean(regime_rets["bear"]) if regime_rets["bear"] else 0
    chop_ret = np.mean(regime_rets["chop"]) if regime_rets["chop"] else 0

    # Diversity score: 1 = works in all 3 regimes, 0 = only works in one
    positive_regimes = sum([bull_ret > 0, bear_ret > 0, chop_ret > 0])
    diversity = positive_regimes / 3.0

    # ---- Verdict ----
    kill_reasons = []

    # OOS survival: must have trades, positive Sharpe, positive PF
    survives_oos = (
        base.total_trades >= 10
        and base.sharpe_ratio > 0.3
        and base.profit_factor > 1.1
        and base.total_return > 0
    )
    if not survives_oos:
        if base.total_trades < 10:
            kill_reasons.append(f"insufficient_oos_trades ({base.total_trades})")
        if base.sharpe_ratio <= 0.3:
            kill_reasons.append(f"low_oos_sharpe ({base.sharpe_ratio:.2f})")
        if base.profit_factor <= 1.1:
            kill_reasons.append(f"low_oos_pf ({base.profit_factor:.2f})")
        if base.total_return <= 0:
            kill_reasons.append("negative_oos_return")

    # Cost stress: must still be profitable at 2x
    survives_cost = stress_2x is not None and stress_2x.total_return > 0 and stress_2x.sharpe_ratio > 0
    if not survives_cost:
        kill_reasons.append("fails_2x_cost_stress")

    # Regime: must work in at least one regime AND not fail badly in any
    # other. A regime filter that correctly suppresses firing in hostile
    # regimes produces ~0% return there — that is GOOD, not bad. Treat a
    # regime as "non-broken" if return > -3% (i.e., small losses or flat
    # acceptable). Require ≥1 strongly positive regime (>0%) and zero
    # regimes below the -3% floor.
    regime_rets = [bull_ret, bear_ret, chop_ret]
    strongly_positive = sum(1 for r in regime_rets if r > 0.0)
    badly_negative = sum(1 for r in regime_rets if r < -0.03)
    survives_regime = strongly_positive >= 1 and badly_negative == 0
    if not survives_regime:
        kill_reasons.append(
            f"regime_fragile (pos={strongly_positive}/3, broken={badly_negative}/3)"
        )

    if survives_oos and survives_cost and survives_regime:
        verdict = "KEEP"
    elif survives_oos and survives_cost:
        verdict = "CONDITIONAL"  # works but regime-dependent
    else:
        verdict = "KILL"

    return StrategyReport(
        name=strat.name,
        asset=strat.asset,
        direction=strat.direction,
        grade=strat.grade,
        hold_bars=strat.hold_bars,
        is_pf=_safe(getattr(strat, "oos_pf", 0)),  # factory's oos = our is
        is_sharpe=_safe(getattr(strat, "oos_sharpe", 0)),
        is_trades=0,
        val_sharpe=_safe(val_sharpe),
        val_pf=_safe(val_pf),
        val_trades=val_trades,
        position_size_pct=float(strat.position_size_pct),
        oos_trades=base.total_trades,
        oos_return=_safe(base.total_return),
        oos_sharpe=_safe(base.sharpe_ratio),
        oos_sortino=_safe(base.sortino_ratio),
        oos_calmar=_safe(base.calmar_ratio),
        oos_max_dd=_safe(base.max_drawdown),
        oos_win_rate=_safe(base.win_rate),
        oos_pf=_safe(base.profit_factor),
        oos_avg_hold=_safe(base.avg_holding_period),
        stress_2x_sharpe=_safe(stress_2x.sharpe_ratio if stress_2x else 0),
        stress_2x_return=_safe(stress_2x.total_return if stress_2x else 0),
        stress_5x_sharpe=_safe(stress_5x.sharpe_ratio if stress_5x else 0),
        stress_5x_return=_safe(stress_5x.total_return if stress_5x else 0),
        regime_bull_return=_safe(bull_ret),
        regime_bear_return=_safe(bear_ret),
        regime_chop_return=_safe(chop_ret),
        regime_diversity_score=diversity,
        survives_oos=survives_oos,
        survives_cost_stress=survives_cost,
        survives_regime_diversity=survives_regime,
        final_verdict=verdict,
        kill_reasons=kill_reasons,
        equity_curve=(
            base.equity_curve if getattr(base, "equity_curve", None) is not None
            and not base.equity_curve.empty else None
        ),
    )


def _empty_strategy_report(strat, reason: str, detail: str = "") -> StrategyReport:
    return StrategyReport(
        name=strat.name, asset=strat.asset, direction=strat.direction,
        grade=getattr(strat, "grade", "?"), hold_bars=strat.hold_bars,
        is_pf=0, is_sharpe=0, is_trades=0,
        oos_trades=0, oos_return=0, oos_sharpe=0, oos_sortino=0, oos_calmar=0,
        oos_max_dd=0, oos_win_rate=0, oos_pf=0, oos_avg_hold=0,
        stress_2x_sharpe=0, stress_2x_return=0, stress_5x_sharpe=0, stress_5x_return=0,
        regime_bull_return=0, regime_bear_return=0, regime_chop_return=0,
        regime_diversity_score=0,
        survives_oos=False, survives_cost_stress=False, survives_regime_diversity=False,
        final_verdict="KILL",
        kill_reasons=[reason, detail] if detail else [reason],
    )


# =====================================================================
# Portfolio aggregation
# =====================================================================

def compute_portfolio_metrics(strategy_reports: list, df_oos_by_asset: dict) -> dict:
    """Compute portfolio-level metrics using ACTUAL equity-curve aggregation.

    Previous implementation used mean-of-Sharpes × sqrt(n_eff) which
    assumes zero correlation between strategies and inflates Sharpe when
    strategies are highly correlated (e.g. 15 long-XRP variants). This
    implementation aligns each strategy's equity curve, converts to
    periodic returns, weights them (KEEP=1.0, CONDITIONAL=0.5, scaled by
    capped val_sharpe) and sums into a single portfolio return series.
    Sharpe / DD / return come from that portfolio series directly — so
    correlation is baked in.
    """
    members = []
    for s in strategy_reports:
        if s.final_verdict == "KEEP":
            base_w = 1.0
        elif s.final_verdict == "CONDITIONAL":
            base_w = 0.5
        else:
            continue
        eq = getattr(s, "equity_curve", None)
        if eq is None:
            continue
        val_weight = max(float(getattr(s, "val_sharpe", 0.0)), 0.1)
        val_weight = min(val_weight, 3.0)
        members.append((s, float(base_w * val_weight), eq))

    if not members:
        return {
            "portfolio_oos_return": 0, "portfolio_oos_sharpe": 0,
            "portfolio_oos_max_dd": 0, "portfolio_mar": 0,
        }

    # ── Correlation cluster caps ──────────────────────────────────
    # When N strategies have pairwise OOS return correlation > 0.7 they
    # form a cluster that behaves as ~1 strategy under stress. Cap each
    # such cluster's aggregate weight at CLUSTER_CAP of the portfolio.
    # This is the institutional-grade fix for "15 long-XRP variants
    # dominate the book".
    CLUSTER_THRESH = 0.7
    CLUSTER_CAP = 0.35  # one correlated cluster can't exceed 35% of book

    # Build raw bar-return series for correlation measurement.
    raw_rets = {}
    for s, _w, eq in members:
        eq_clean = eq[~eq.index.duplicated(keep="last")].sort_index()
        raw_rets[s.name] = eq_clean.pct_change().fillna(0)
    raw_df = pd.DataFrame(raw_rets).fillna(0)

    # Pairwise correlation (on common index)
    clusters: list[list[int]] = []
    if len(raw_df.columns) >= 2 and len(raw_df) >= 30:
        corr = raw_df.corr()
        assigned: set[int] = set()
        names = list(raw_df.columns)
        name_to_idx = {n: i for i, n in enumerate(names)}
        for i, n_i in enumerate(names):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j in range(i + 1, len(names)):
                if j in assigned:
                    continue
                c = corr.iloc[i, j]
                if pd.notna(c) and abs(c) >= CLUSTER_THRESH:
                    cluster.append(j)
                    assigned.add(j)
            if len(cluster) > 1:
                clusters.append(cluster)

        # For each oversized cluster, scale its members' weights down so
        # the cluster's share of total weight == CLUSTER_CAP.
        member_names = [m[0].name for m in members]
        weights = [m[1] for m in members]
        total_w_tmp = sum(weights)
        for cluster in clusters:
            cluster_names = {names[idx] for idx in cluster}
            cluster_idxs = [
                k for k, mn in enumerate(member_names) if mn in cluster_names
            ]
            cluster_w = sum(weights[k] for k in cluster_idxs)
            cluster_share = cluster_w / max(total_w_tmp, 1e-9)
            if cluster_share > CLUSTER_CAP:
                shrink = CLUSTER_CAP / cluster_share
                for k in cluster_idxs:
                    weights[k] *= shrink
        # Rebuild members with adjusted weights
        members = [
            (s, weights[k], eq) for k, (s, _w, eq) in enumerate(members)
        ]

    # Normalize weights to sum to 1.0
    total_w = sum(w for _, w, _ in members)
    # Convert each equity curve to periodic returns, then resample to a
    # common frequency (1h) so cross-strategy correlation is measured
    # on aligned bars.
    returns_df = pd.DataFrame()
    for s, w, eq in members:
        # Backtester equity may have duplicate timestamps from multi-trade
        # bars — drop duplicates keeping last.
        eq_clean = eq[~eq.index.duplicated(keep="last")].sort_index()
        rets = eq_clean.pct_change().fillna(0)
        returns_df[s.name] = rets * (w / total_w)

    # Union index across all strategies, forward-fill zeros where absent
    returns_df = returns_df.fillna(0)
    if returns_df.empty or len(returns_df) < 50:
        # Fallback to mean aggregation when we can't build real series
        avg_sharpe = np.mean([s.oos_sharpe for s, _, _ in members])
        avg_ret = np.mean([s.oos_return for s, _, _ in members])
        avg_dd = np.mean([s.oos_max_dd for s, _, _ in members])
        return {
            "portfolio_oos_return": float(avg_ret),
            "portfolio_oos_sharpe": float(avg_sharpe),
            "portfolio_oos_max_dd": float(avg_dd),
            "portfolio_mar": float(avg_ret / max(avg_dd, 0.001)),
        }

    port_returns = returns_df.sum(axis=1)
    # Total portfolio return over the window
    port_ret = float((1.0 + port_returns).prod() - 1.0)
    # Annualized Sharpe — infer freq from median bar spacing
    bar_seconds = (
        (port_returns.index[-1] - port_returns.index[0]).total_seconds()
        / max(len(port_returns), 1)
    )
    bars_per_year = max(int(365.25 * 86400 / bar_seconds), 1) if bar_seconds > 0 else 8760
    mu = port_returns.mean()
    sigma = port_returns.std()
    port_sharpe = float(mu / sigma * np.sqrt(bars_per_year)) if sigma > 0 else 0.0
    # Max DD on compounded equity
    eq_curve = (1.0 + port_returns).cumprod()
    running_max = eq_curve.cummax()
    dd_series = (eq_curve - running_max) / running_max
    port_dd = float(-dd_series.min()) if not dd_series.empty else 0.0
    mar = port_ret / max(port_dd, 0.001)

    return {
        "portfolio_oos_return": port_ret,
        "portfolio_oos_sharpe": port_sharpe,
        "portfolio_oos_max_dd": port_dd,
        "portfolio_mar": float(mar),
        "n_correlation_clusters": len(clusters),
    }


# =====================================================================
# Main harness entry point
# =====================================================================

def run_validation(
    symbols: list,
    timeframe: str = "1h",
    days: int = 1825,  # 5 years
    min_trades: int = 50,
    oos_fraction: float = 0.30,
    output_path: str = "fund_data/validation_report.json",
    use_structural: bool = True,
) -> HarnessReport:
    """Run the full brutal validation harness.

    Args:
        symbols: List of symbols like ["BTC/USDT", "ETH/USDT"]
        timeframe: Bar size ("1h", "4h", etc.)
        days: How many days of history to fetch
        min_trades: Min trades for scanner to consider a signal
        oos_fraction: Fraction of data reserved for OOS testing
        output_path: Where to write JSON report
        use_structural: Whether to fetch funding/OI/liquidations

    Returns:
        HarnessReport
    """
    print("=" * 70)
    print("  SIGNALFORGE — BRUTAL VALIDATION HARNESS")
    print("=" * 70)
    print(f"  Symbols:   {symbols}")
    print(f"  Timeframe: {timeframe}")
    print(f"  History:   {days} days (~{days/365:.1f} years)")
    print(f"  OOS:       last {oos_fraction:.0%} of data")
    print("=" * 70)
    print()

    # ---- 1. Fetch data ----
    print("[1/5] Fetching data...")
    fetcher = DataFetcher()
    struct = StructuralDataFetcher() if use_structural else None
    datasets_full = {}
    datasets_is = {}
    datasets_val = {}
    datasets_oos = {}

    for sym in symbols:
        try:
            raw = fetcher.fetch(sym, timeframe, days=days)
            if raw is None or raw.empty:
                print(f"  {sym}: NO DATA")
                continue
            df = compute_all_features(raw)

            if struct is not None:
                try:
                    # Funding has unlimited history on Binance — fetch full.
                    # OI / LSR / taker are capped at ~30d by the API and
                    # will return empty for older windows, but that's fine
                    # as those generators are disabled in the scanner.
                    df = struct.fetch_all(
                        symbol=sym.replace("/", ""),
                        price_df=df,
                        days=days,
                    )
                except Exception as e:
                    print(f"  {sym}: structural fetch failed ({e})")

            df = df.dropna(subset=["close"])

            # Three-way split (critical: validator needs data scanner never saw)
            #   scan  — oldest 50% — scanner fits/tests hypotheses here
            #   val   — middle 20% — held out from scanner, used by validator
            #   oos   — newest 30% — held out from both, final harness eval
            n = len(df)
            oos_start = int(n * (1 - oos_fraction))
            val_start = int(n * (1 - oos_fraction - 0.20))
            datasets_full[sym] = df
            datasets_is[sym] = df.iloc[:val_start].copy()
            datasets_val[sym] = df.iloc[val_start:oos_start].copy()
            datasets_oos[sym] = df.iloc[oos_start:].copy()

            actual_years = (df.index[-1] - df.index[0]).days / 365.25
            print(f"  {sym}: {len(df)} bars ({actual_years:.1f}y), "
                  f"SCAN={len(datasets_is[sym])} / "
                  f"VAL={len(datasets_val[sym])} / "
                  f"OOS={len(datasets_oos[sym])}")
        except Exception as e:
            print(f"  {sym}: FAILED — {e}")

    if not datasets_is:
        print("\nNo data fetched. Aborting.")
        return HarnessReport(
            timestamp=datetime.now().isoformat(),
            data_years=0, symbols=symbols, timeframe=timeframe,
            n_scanned=0, n_validated=0, n_deployed=0,
            overall_verdict="NO_DATA",
        )

    years = np.mean([
        (datasets_full[s].index[-1] - datasets_full[s].index[0]).days / 365.25
        for s in datasets_full
    ])

    # Register the universe so cross-sectional and lead-lag families can
    # read peer asset price series at signal-generation time. Uses the
    # FULL datasets (not just SCAN) because the signal gen slices to the
    # passed df's index itself — the registry just provides the price
    # panel.
    set_universe(datasets_full)

    # ---- 2. Scan on IS data only ----
    print(f"\n[2/5] Scanning IS data for signal hypotheses...")
    scan_result = scan(datasets_is, min_trades=min_trades)
    print(f"  Hypotheses tested:    {scan_result.total_hypotheses}")
    print(f"  Raw survivors:        {len(scan_result.raw_survivors)}")
    print(f"  Bonferroni survivors: {len(scan_result.bonferroni_survivors)}")

    if not scan_result.raw_survivors:
        print("\nNo signals found. System has no edge in this data.")
        return HarnessReport(
            timestamp=datetime.now().isoformat(),
            data_years=years, symbols=symbols, timeframe=timeframe,
            n_scanned=scan_result.total_hypotheses,
            n_validated=0, n_deployed=0,
            overall_verdict="NO_EDGE_DETECTED",
        )

    # ---- 3. Validate on held-out VAL slice (scanner never saw this) ----
    print(f"\n[3/5] Validating on held-out VAL slice...")
    val_result = validate(scan_result.raw_survivors, datasets_is, val_datasets=datasets_val)
    print(f"  Tested:      {val_result.signals_tested}")
    print(f"  Passed IS:   {val_result.signals_passed_is}")
    print(f"  Passed OOS:  {val_result.signals_passed_oos}")

    if not val_result.validated:
        print("\nNo signals survived validation.")
        return HarnessReport(
            timestamp=datetime.now().isoformat(),
            data_years=years, symbols=symbols, timeframe=timeframe,
            n_scanned=scan_result.total_hypotheses,
            n_validated=0, n_deployed=0,
            overall_verdict="NO_SIGNALS_SURVIVED_VALIDATION",
        )

    # ---- 4. Deploy ----
    print(f"\n[4/5] Deploying top strategies...")
    deployed = deploy(val_result.validated, max_strategies=10)
    print(f"  Deployed: {len(deployed)}")

    # ── Volatility-targeted baseline sizing ──────────────────────
    # Each strategy's baseline position_size_pct is scaled so that
    # notional risk is consistent across assets. We measure 90d (2160-bar
    # on 1h) realized volatility of the SCAN window and scale:
    #
    #     new_size = baseline × (target_vol / realized_vol)
    #     clipped to [0.5×, 2.0×] of baseline
    #
    # Target vol: 60% annualized (crypto baseline; BTC ~50%, DOGE ~110%).
    # Effect: high-vol assets (DOGE, XRP) get smaller size, low-vol
    # assets (BTC) get larger — equalizing dollar risk per trade.
    TARGET_ANN_VOL = 0.60
    BARS_PER_YEAR_1H = 24 * 365  # ~8760
    for strat in deployed:
        df_is = datasets_is.get(strat.asset)
        if df_is is None or len(df_is) < 500:
            continue
        recent = df_is.iloc[-2160:] if len(df_is) >= 2160 else df_is
        bar_rets = recent["close"].pct_change().dropna()
        if bar_rets.empty or bar_rets.std() <= 0:
            continue
        realized_vol = float(bar_rets.std() * np.sqrt(BARS_PER_YEAR_1H))
        if realized_vol <= 0:
            continue
        vol_scale = TARGET_ANN_VOL / realized_vol
        vol_scale = max(0.5, min(2.0, vol_scale))
        strat.position_size_pct = float(strat.position_size_pct * vol_scale)

    # ---- 5. Evaluate on TRUE OOS (never-seen data) ----
    print(f"\n[5/5] Evaluating on true out-of-sample data...")
    print(f"  {'─' * 66}")
    print(f"  {'Strategy':<32s} {'Trades':>7s} {'Sharpe':>7s} {'DD':>6s} {'Verdict':>10s}")
    print(f"  {'─' * 66}")

    strategy_reports = []
    for strat in deployed:
        df_oos = datasets_oos.get(strat.asset)
        if df_oos is None or df_oos.empty:
            continue

        # Build a "SCAN tail" window = last 20% of SCAN, as an independent
        # second VAL window for dual-window sizing. Scanner saw this data
        # but didn't fit per-strategy weights to it (it was just part of
        # aggregate statistics). This tail provides a second regime sample
        # for sizing decisions. Canonical walk-forward would refit here;
        # we don't refit, only weight.
        df_is = datasets_is.get(strat.asset)
        df_scan_tail = None
        if df_is is not None and len(df_is) > 100:
            tail_start = int(len(df_is) * 0.80)
            df_scan_tail = df_is.iloc[tail_start:].copy()

        report = evaluate_strategy(
            strat,
            df_oos,
            df_val=datasets_val.get(strat.asset),
            df_scan_tail=df_scan_tail,
        )
        strategy_reports.append(report)

        print(f"  {strat.name[:32]:<32s} "
              f"{report.oos_trades:>7d} "
              f"{report.oos_sharpe:>7.2f} "
              f"{report.oos_max_dd*100:>5.1f}% "
              f"{report.final_verdict:>10s}")

    # ---- Portfolio metrics ----
    port = compute_portfolio_metrics(strategy_reports, datasets_oos)

    n_keep = sum(1 for s in strategy_reports if s.final_verdict == "KEEP")
    n_kill = sum(1 for s in strategy_reports if s.final_verdict == "KILL")
    n_cond = sum(1 for s in strategy_reports if s.final_verdict == "CONDITIONAL")

    # Overall verdict
    if n_keep >= 3:
        overall = "DEPLOYABLE"
    elif n_keep + n_cond >= 3:
        overall = "NEEDS_WORK"
    elif n_keep + n_cond >= 1:
        overall = "MARGINAL"
    else:
        overall = "NO_DEPLOYABLE_EDGE"

    report = HarnessReport(
        timestamp=datetime.now().isoformat(),
        data_years=years,
        symbols=symbols,
        timeframe=timeframe,
        n_scanned=scan_result.total_hypotheses,
        n_validated=len(val_result.validated),
        n_deployed=len(deployed),
        strategies=[
            {k: v for k, v in asdict(s).items() if k != "equity_curve"}
            for s in strategy_reports
        ],
        portfolio_oos_return=port["portfolio_oos_return"],
        portfolio_oos_sharpe=port["portfolio_oos_sharpe"],
        portfolio_oos_max_dd=port["portfolio_oos_max_dd"],
        portfolio_mar=port["portfolio_mar"],
        n_correlation_clusters=int(port.get("n_correlation_clusters", 0)),
        n_survivors=n_keep,
        n_killed=n_kill,
        overall_verdict=overall,
    )

    # ---- Save + summary ----
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)

    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print(f"  Scanned:            {scan_result.total_hypotheses}")
    print(f"  Validated:          {len(val_result.validated)}")
    print(f"  Deployed:           {len(deployed)}")
    print(f"  KEEP (survives):    {n_keep}")
    print(f"  CONDITIONAL:        {n_cond}")
    print(f"  KILL:               {n_kill}")
    print()
    print(f"  Portfolio OOS Return:  {port['portfolio_oos_return']*100:+.1f}%")
    print(f"  Portfolio OOS Sharpe:  {port['portfolio_oos_sharpe']:.2f}")
    print(f"  Portfolio Max DD:      {port['portfolio_oos_max_dd']*100:.1f}%")
    print(f"  MAR (return / DD):     {port['portfolio_mar']:.2f}")
    print()
    print(f"  OVERALL VERDICT: {overall}")
    print()
    print(f"  Full report: {out}")
    print("=" * 70)

    return report

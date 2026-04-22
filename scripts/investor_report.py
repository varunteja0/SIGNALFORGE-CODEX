#!/usr/bin/env python3
"""SignalForge investor report.

Runs a portfolio backtest, institutional validation, and long-horizon
block-bootstrap projections, then renders a text report.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.dataset_cache import load_or_build_datasets
from src.engine.institutional import InstitutionalValidator
from src.engine.portfolio_engine import PortfolioEngine
from src.reporting import project_horizon_table, resample_equity_to_returns


PROFILE_TITLES = {
    "default": "Validated Default Portfolio",
    "compounding": "Compounding Focus Portfolio",
}

STRATEGY_DESCRIPTIONS = {
    "funding_mr_v7": "Funding mean reversion on structurally stretched perp pricing.",
    "extreme_spike": "Convex volatility sleeve for extreme spike dislocations.",
    "funding_vol_squeeze": "Funding-volatility squeeze entries after compression.",
    "momentum_breakout": "Trend-following breakout sleeve with structural confirmation.",
    "contrarian_asym": "Asymmetric contrarian sleeve targeting crowded overextension.",
}

INSTITUTIONAL_LABELS = {
    "strategy_corr_<0.7": "Strategy correlation < 0.70",
    "asset_corr_<0.8": "Asset correlation < 0.80",
    "diversification_ratio_>1": "Diversification ratio > 1.0",
    "all_strats_profitable_1+_regime": "All strategies profitable in 1+ regime",
    "pf>1_at_2%_size": "PF > 1.0 at 2% position size",
    "pf>1_at_5%_size": "PF > 1.0 at 5% position size",
    "max_viable_>=3%": "Max viable position >= 3%",
}

INSTITUTIONAL_ORDER = [
    "strategy_corr_<0.7",
    "asset_corr_<0.8",
    "diversification_ratio_>1",
    "all_strats_profitable_1+_regime",
    "pf>1_at_2%_size",
    "pf>1_at_5%_size",
    "max_viable_>=3%",
]


def build_engine(profile: str) -> PortfolioEngine:
    if profile == "default":
        return PortfolioEngine.default()
    if profile == "compounding":
        return PortfolioEngine.compounding_focus()
    raise ValueError(f"Unsupported profile: {profile}")


def format_metric(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    value = float(value)
    if np.isinf(value):
        return "inf"
    return f"{value:.{decimals}f}"


def run_backtest(
    *,
    profile: str,
    capital: float,
    data_days: int,
    use_cache: bool,
    cache_max_age_hours: float,
    force_refresh: bool,
) -> tuple[PortfolioEngine, dict[str, pd.DataFrame], object, object, dict[str, object]]:
    engine = build_engine(profile)
    engine.capital = capital
    engine.data_days = data_days

    if use_cache:
        datasets, cache_path, used_cache = load_or_build_datasets(
            engine,
            namespace=f"investor_report_{profile}",
            max_age_hours=cache_max_age_hours,
            force_refresh=force_refresh,
        )
    else:
        datasets = engine.load_data()
        cache_path = None
        used_cache = False

    result = engine.backtest(datasets)
    institutional_report = InstitutionalValidator().validate(engine, datasets)
    cache_meta = {
        "used_cache": used_cache,
        "cache_path": str(cache_path) if cache_path else None,
    }
    return engine, datasets, result, institutional_report, cache_meta


def compute_monthly_pnl(trades: list) -> OrderedDict[str, tuple[float, int]]:
    monthly: dict[str, tuple[float, int]] = {}
    for trade in trades:
        realized_at = getattr(trade, "exit_time", None) or getattr(trade, "entry_time", None)
        if realized_at is None:
            continue
        month = pd.Timestamp(realized_at).strftime("%Y-%m")
        pnl, count = monthly.get(month, (0.0, 0))
        monthly[month] = (pnl + float(getattr(trade, "pnl", 0.0) or 0.0), count + 1)
    return OrderedDict(sorted(monthly.items()))


def compute_horizon_rows(
    result,
    *,
    capital: float,
    n_sims: int,
    block_size: int,
    ruin_threshold: float,
) -> list:
    period_returns = resample_equity_to_returns(result.equity_curve, frequency="1D")
    if period_returns.empty:
        return []
    return project_horizon_table(
        period_returns,
        starting_capital=capital,
        block_size=block_size,
        n_sims=n_sims,
        ruin_threshold=ruin_threshold,
    )


def print_report(
    engine: PortfolioEngine,
    result,
    institutional_report,
    monthly_pnl: OrderedDict[str, tuple[float, int]],
    horizon_rows: list,
    *,
    profile: str,
    cache_meta: dict[str, object],
    block_size: int,
    n_sims: int,
    ruin_threshold: float,
    output_file: str | None = None,
) -> None:
    lines: list[str] = []

    def p(text: str = "") -> None:
        lines.append(text)

    def section(title: str) -> None:
        p()
        p("=" * 74)
        p(title)
        p("=" * 74)

    now = datetime.now().strftime("%B %d, %Y")
    profile_title = PROFILE_TITLES.get(profile, profile.replace("_", " ").title())
    source_label = "cache" if cache_meta.get("used_cache") else "fresh data build"
    active_assets = sorted(
        {asset.split("/")[0] for slot in engine.slots for asset in slot.allowed_assets}
    )

    all_pnls = [float(getattr(trade, "pnl", 0.0) or 0.0) for trade in result.trades]
    if not all_pnls:
        all_pnls = [0.0]
    wins = [pnl for pnl in all_pnls if pnl > 0]
    losses = [pnl for pnl in all_pnls if pnl <= 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = abs(float(np.mean(losses))) if losses else 0.0
    payoff = avg_win / avg_loss if avg_loss > 0 else (float("inf") if avg_win > 0 else 0.0)
    expectancy = float(np.mean(all_pnls))
    total_return = result.total_pnl / engine.capital if engine.capital > 0 else 0.0
    calmar = total_return / result.max_drawdown if result.max_drawdown > 0 else float("inf")

    total_months = len(monthly_pnl)
    profitable_months = sum(1 for pnl, _ in monthly_pnl.values() if pnl > 0)
    strategy_count = len(result.strategy_results)
    profitable_strategies = sum(
        1 for stats in result.strategy_results.values() if stats.get("net_pnl", 0.0) > 0
    )

    verdict_status = (
        "PASS"
        if institutional_report.score == institutional_report.total_tests
        else "PARTIAL"
    )
    correlation = institutional_report.correlation
    capacity = institutional_report.capacity
    verdict_keys = [
        key for key in INSTITUTIONAL_ORDER if key in institutional_report.verdicts
    ] + sorted(
        key for key in institutional_report.verdicts if key not in INSTITUTIONAL_ORDER
    )
    horizon_by_year = {row.years: row for row in horizon_rows}
    one_year = horizon_by_year.get(1)
    three_year = horizon_by_year.get(3)
    ten_year = horizon_by_year.get(10)

    p("SignalForge Investor Report")
    p(profile_title)
    p(f"Report date: {now}")
    p("-" * 74)

    section("1. Executive Summary")
    p("SignalForge runs a multi-sleeve systematic crypto portfolio with shared")
    p("portfolio exposure controls, institutional validation, and long-horizon")
    p("block-bootstrap projections built from realized equity-path returns.")
    p()
    p("Key points:")
    p(
        f"- {profitable_strategies}/{strategy_count} active sleeves profitable in the current profile"
    )
    p(f"- Profit factor {format_metric(result.profit_factor)}, Sharpe {format_metric(result.sharpe)}")
    p(f"- {result.win_rate:.1%} win rate across {result.total_trades} trades")
    if total_months > 0:
        p(f"- {profitable_months}/{total_months} profitable months")
    p(f"- Max drawdown {result.max_drawdown:.2%}")
    p(
        f"- Institutional scorecard {institutional_report.score}/"
        f"{institutional_report.total_tests} {verdict_status}"
    )
    if one_year is not None:
        p(
            f"- 1y median ending capital ${one_year.p50:,.0f}; "
            f"P(ruin) {one_year.ruin_probability:.1%}"
        )
    p(f"- Dataset source: {source_label}")

    section(f"2. Audited Performance Metrics ({engine.data_days}-Day Backtest)")
    p(f"Capital:              ${engine.capital:,.0f}")
    p(f"Assets:               {', '.join(active_assets)}")
    p(f"Dataset source:       {source_label}")
    if cache_meta.get("cache_path"):
        p(f"Cache path:           {cache_meta['cache_path']}")
    p(f"Total trades:         {result.total_trades}")
    p(f"Net PnL:              ${result.total_pnl:+,.2f}")
    p(f"Return:               {total_return:+.2%}")
    p(f"Win rate:             {result.win_rate:.1%}")
    p(f"Profit factor:        {format_metric(result.profit_factor)}")
    p(f"Sharpe:               {format_metric(result.sharpe)}")
    p(f"Max drawdown:         {result.max_drawdown:.2%}")
    p(f"Calmar:               {format_metric(calmar, 1)}")
    p(f"Payoff ratio:         {format_metric(payoff)}")
    p(f"Expectancy/trade:     ${expectancy:+.2f}")
    p(f"Best trade:           ${max(all_pnls):+,.2f}")
    p(f"Worst trade:          ${min(all_pnls):+,.2f}")

    section("3. Strategy Breakdown")
    p(f"{'Strategy':<25s} {'Trades':>7s} {'PF':>7s} {'WR':>7s} {'PnL':>12s}")
    p(f"{'-' * 25} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 12}")
    for name, stats in sorted(result.strategy_results.items()):
        p(
            f"{name:<25s} {stats['trades']:>7d} {format_metric(stats['pf']):>7s} "
            f"{stats['win_rate']:>6.1%} {stats['net_pnl']:>+12.2f}"
        )
    p()
    p("Strategy descriptions:")
    for name in sorted(result.strategy_results):
        p(f"- {name}: {STRATEGY_DESCRIPTIONS.get(name, 'Description unavailable.')}")

    section("4. Monthly PnL Consistency")
    if monthly_pnl:
        p(f"{'Month':<10s} {'PnL':>12s} {'Trades':>7s} {'Result':>10s}")
        p(f"{'-' * 10} {'-' * 12} {'-' * 7} {'-' * 10}")
        for month, (pnl, trade_count) in monthly_pnl.items():
            label = "PROFIT" if pnl > 0 else "LOSS"
            p(f"{month:<10s} ${pnl:>+10.2f} {trade_count:>7d} {label:>10s}")
        p()
        p(f"Profitable months:    {profitable_months}/{total_months}")
        p(f"Best month:           ${max(pnl for pnl, _ in monthly_pnl.values()):+,.2f}")
        p(f"Worst month:          ${min(pnl for pnl, _ in monthly_pnl.values()):+,.2f}")
    else:
        p("No completed monthly buckets were available in this run.")

    section(
        "5. Institutional Validation "
        f"({institutional_report.score}/{institutional_report.total_tests} {verdict_status})"
    )
    for key in verdict_keys:
        label = INSTITUTIONAL_LABELS.get(key, key)
        status = "PASS" if institutional_report.verdicts.get(key) else "FAIL"
        p(f"- {label:<45s} {status}")
    p()
    p("Key institutional metrics:")
    p(f"- Max strategy correlation: {format_metric(correlation.max_strategy_corr, 3)}")
    p(f"- Max asset correlation:    {format_metric(correlation.max_asset_corr, 3)}")
    p(f"- Diversification ratio:    {format_metric(correlation.diversification_ratio)}")
    p(f"- Max viable position size: {capacity.max_viable_pct:.1%}")
    if 0.02 in capacity.curve:
        p(f"- PF at 2% size:           {format_metric(capacity.curve[0.02].get('pf'))}")
    if 0.05 in capacity.curve:
        p(f"- PF at 5% size:           {format_metric(capacity.curve[0.05].get('pf'))}")

    section("6. Horizon Risk Projections")
    if horizon_rows:
        p(f"{'Horizon':<10s} {'P(ruin)':>10s} {'p05':>14s} {'p50':>14s} {'p95':>14s}")
        p(f"{'-' * 10} {'-' * 10} {'-' * 14} {'-' * 14} {'-' * 14}")
        for row in horizon_rows:
            p(
                f"{row.label:<10s} {row.ruin_probability:>9.1%} "
                f"${row.p05:>12,.0f} ${row.p50:>12,.0f} ${row.p95:>12,.0f}"
            )
        p()
        p(
            f"Ruin is defined as equity touching {ruin_threshold:.0%} of starting capital "
            "at any point in the simulated path."
        )
        p(
            f"Method: {n_sims:,} circular block-bootstrap simulations on daily returns "
            f"using {block_size}-day blocks."
        )
    else:
        p("Horizon projections unavailable because the equity curve could not be resampled.")

    section("7. Compounding Benchmarks")
    if one_year and three_year and ten_year:
        one_year_mult = one_year.p50 / engine.capital
        three_year_mult = three_year.p50 / engine.capital
        ten_year_mult = ten_year.p50 / engine.capital
        for start_capital in (10_000, 100_000, 1_000_000):
            p(
                f"Start ${start_capital:>10,.0f}: "
                f"1y ${start_capital * one_year_mult:>12,.0f}  "
                f"3y ${start_capital * three_year_mult:>12,.0f}  "
                f"10y ${start_capital * ten_year_mult:>13,.0f}"
            )
    else:
        p("Compounding benchmarks unavailable.")

    section("8. Risk Controls And Scaling")
    p("- Portfolio-wide exposure budgeting enforces one shared risk pool.")
    p("- Capacity simulation checks whether edge survives larger position sizes.")
    p("- Drawdown controls stay active in the compounding profile; growth does not")
    p("  override capital preservation.")
    p("- Scale decisions should be tied to fresh outcome validation on the same")
    p("  profile, not inherited from older books.")

    section("9. Summary")
    p(
        f"- Edge: {profitable_strategies}/{strategy_count} sleeves profitable, "
        f"profit factor {format_metric(result.profit_factor)}, Sharpe {format_metric(result.sharpe)}"
    )
    if total_months > 0:
        p(
            f"- Consistency: {profitable_months}/{total_months} profitable months, "
            f"max drawdown {result.max_drawdown:.2%}"
        )
    p(
        f"- Validation: institutional score {institutional_report.score}/"
        f"{institutional_report.total_tests}, max strategy correlation "
        f"{format_metric(correlation.max_strategy_corr, 3)}"
    )
    if one_year and ten_year:
        p(
            f"- Horizon view: 1y p50 ${one_year.p50:,.0f}, 10y p50 ${ten_year.p50:,.0f}, "
            f"10y P(ruin) {ten_year.ruin_probability:.1%}"
        )
    p("- This report is generated from current backtest outputs and live validation metrics.")

    report = "\n".join(lines)
    print(report)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"\nReport saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SignalForge investor report")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_TITLES),
        default="compounding",
        help="Portfolio profile to report on",
    )
    parser.add_argument("--capital", type=float, default=10_000, help="Starting capital")
    parser.add_argument("--days", type=int, default=365, help="Backtest lookback window")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cached enriched datasets for this run",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Rebuild cached enriched datasets even if a fresh cache exists",
    )
    parser.add_argument(
        "--cache-max-age-hours",
        type=float,
        default=12.0,
        help="Maximum accepted age for cached enriched datasets",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=10_000,
        help="Number of block-bootstrap simulations for horizon projections",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=21,
        help="Daily block size for the circular block bootstrap",
    )
    parser.add_argument(
        "--ruin-threshold",
        type=float,
        default=0.25,
        help="Ruin threshold as a fraction of starting capital",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="pipeline_output/investor_report.txt",
        help="Output file path",
    )
    args = parser.parse_args()

    print("\nGenerating investor report...\n")
    print("[1/4] Running portfolio backtest and institutional validation...")
    engine, datasets, result, institutional_report, cache_meta = run_backtest(
        profile=args.profile,
        capital=args.capital,
        data_days=args.days,
        use_cache=not args.no_cache,
        cache_max_age_hours=args.cache_max_age_hours,
        force_refresh=args.force_refresh,
    )

    print("[2/4] Computing monthly consistency...")
    monthly_pnl = compute_monthly_pnl(result.trades)

    print("[3/4] Building horizon risk projections...")
    horizon_rows = compute_horizon_rows(
        result,
        capital=engine.capital,
        n_sims=args.n_sims,
        block_size=args.block_size,
        ruin_threshold=args.ruin_threshold,
    )

    print("[4/4] Rendering investor report...")
    print_report(
        engine,
        result,
        institutional_report,
        monthly_pnl,
        horizon_rows,
        profile=args.profile,
        cache_meta=cache_meta,
        block_size=args.block_size,
        n_sims=args.n_sims,
        ruin_threshold=args.ruin_threshold,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
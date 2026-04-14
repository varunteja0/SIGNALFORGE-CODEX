"""
Institutional Validation Suite
===============================
Hedge-fund-grade analytics for a multi-strategy portfolio:

    1. Strategy correlation matrix (return-level, not signal-level)
    2. Rolling correlation (detect convergence over time)
    3. Regime-specific performance breakdown per strategy × asset
    4. Capacity simulation (slippage vs position size curve)
    5. Diversification ratio (portfolio vol / sum of component vols)
    6. Marginal contribution to risk per strategy

Used by the portfolio engine and the run_institutional.py CLI.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────

@dataclass
class CorrelationReport:
    """Strategy and asset correlation analysis."""
    strategy_corr: pd.DataFrame = field(default_factory=pd.DataFrame)
    asset_corr: pd.DataFrame = field(default_factory=pd.DataFrame)
    rolling_corr: dict = field(default_factory=dict)  # {pair: pd.Series}
    max_strategy_corr: float = 0.0
    max_asset_corr: float = 0.0
    diversification_ratio: float = 0.0
    # True if all pairwise correlations < 0.7
    is_diversified: bool = False


@dataclass
class RegimeBreakdown:
    """Per-regime performance for each strategy × asset."""
    # {regime: {strategy|asset: {pf, sharpe, trades, win_rate}}}
    data: dict = field(default_factory=dict)


@dataclass
class CapacityResult:
    """How PF/Sharpe degrade as position size increases."""
    # {position_pct: {pf, sharpe, trades, net_pnl, avg_slippage_bps}}
    curve: dict = field(default_factory=dict)
    max_viable_pct: float = 0.0  # Largest size with PF > 1.0
    optimal_pct: float = 0.0     # Size that maximizes risk-adj return


@dataclass
class InstitutionalReport:
    """Combined institutional validation output."""
    correlation: CorrelationReport = field(default_factory=CorrelationReport)
    regime_breakdown: RegimeBreakdown = field(default_factory=RegimeBreakdown)
    capacity: CapacityResult = field(default_factory=CapacityResult)
    # Pass/fail verdicts
    verdicts: dict = field(default_factory=dict)
    score: int = 0
    total_tests: int = 0


# ─── Correlation Engine ─────────────────────────────────────────

class CorrelationEngine:
    """Compute strategy and asset return correlations."""

    def __init__(self, rolling_window: int = 720):  # 30 days of hourly bars
        self.rolling_window = rolling_window

    def analyze(
        self,
        equity_curves: dict[str, pd.Series],
        strategy_names: list[str] = None,
    ) -> CorrelationReport:
        """Compute full correlation analysis from equity curves.

        equity_curves: {cell_key: equity_curve} where cell_key = "strategy|ASSET/USDT"
        """
        report = CorrelationReport()

        if len(equity_curves) < 2:
            report.is_diversified = True
            report.diversification_ratio = 1.0
            return report

        # Build returns per strategy (aggregate across assets)
        strat_returns = {}
        asset_returns = {}

        for key, curve in equity_curves.items():
            if len(curve) < 50:
                continue

            parts = key.split("|")
            strat = parts[0] if len(parts) > 1 else key
            asset = parts[1] if len(parts) > 1 else "unknown"

            ret = curve.pct_change().fillna(0)

            # Aggregate by strategy
            if strat not in strat_returns:
                strat_returns[strat] = ret
            else:
                # Align and average
                common = strat_returns[strat].index.intersection(ret.index)
                if len(common) > 0:
                    strat_returns[strat] = (
                        strat_returns[strat].reindex(common) + ret.reindex(common)
                    ) / 2

            # Aggregate by asset
            if asset not in asset_returns:
                asset_returns[asset] = ret
            else:
                common = asset_returns[asset].index.intersection(ret.index)
                if len(common) > 0:
                    asset_returns[asset] = (
                        asset_returns[asset].reindex(common) + ret.reindex(common)
                    ) / 2

        # Strategy correlation matrix
        if len(strat_returns) >= 2:
            strat_df = pd.DataFrame(strat_returns)
            strat_df = strat_df.dropna()
            if len(strat_df) > 50:
                report.strategy_corr = strat_df.corr()

                # Max pairwise correlation (off-diagonal)
                corr_vals = report.strategy_corr.values
                np.fill_diagonal(corr_vals, 0)
                report.max_strategy_corr = float(np.abs(corr_vals).max())

                # Rolling correlation for each pair
                cols = list(strat_df.columns)
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        pair_key = f"{cols[i]} × {cols[j]}"
                        rc = strat_df[cols[i]].rolling(self.rolling_window).corr(
                            strat_df[cols[j]]
                        )
                        report.rolling_corr[pair_key] = rc.dropna()

        # Asset correlation matrix
        if len(asset_returns) >= 2:
            asset_df = pd.DataFrame(asset_returns)
            asset_df = asset_df.dropna()
            if len(asset_df) > 50:
                report.asset_corr = asset_df.corr()
                corr_vals = report.asset_corr.values
                np.fill_diagonal(corr_vals, 0)
                report.max_asset_corr = float(np.abs(corr_vals).max())

        # Diversification ratio
        # DR = weighted sum of individual vols / portfolio vol
        # DR > 1 means diversification is working
        if len(strat_returns) >= 2:
            strat_df = pd.DataFrame(strat_returns).dropna()
            if len(strat_df) > 50:
                individual_vols = strat_df.std()
                portfolio_ret = strat_df.mean(axis=1)
                portfolio_vol = portfolio_ret.std()
                if portfolio_vol > 0:
                    report.diversification_ratio = float(
                        individual_vols.mean() / portfolio_vol
                    )

        report.is_diversified = report.max_strategy_corr < 0.70

        return report


# ─── Regime Breakdown ───────────────────────────────────────────

class RegimeAnalyzer:
    """Break down strategy performance by market regime."""

    def analyze(
        self,
        slots,  # list[StrategySlot]
        datasets: dict[str, pd.DataFrame],
    ) -> RegimeBreakdown:
        """Compute per-regime PF/Sharpe for each strategy × asset."""
        from src.backtest.engine import Backtester
        from src.regime.detector import RegimeDetector

        breakdown = RegimeBreakdown()

        for slot in slots:
            for sym in slot.allowed_assets:
                if sym not in datasets:
                    continue

                df = datasets[sym]

                # Fit regime detector
                det = RegimeDetector()
                det.fit(df)
                regimes = det.get_regime_history(df)

                # Generate unfiltered signals (we want regime breakdown BEFORE filter)
                signals = slot.signal_func(df)

                # Backtest
                bt = Backtester(initial_capital=10000)
                res = bt.run(
                    df, lambda d, s=signals: s,
                    position_size_pct=slot.position_size_pct,
                    stop_loss_atr=slot.stop_loss_atr,
                    take_profit_atr=slot.take_profit_atr,
                    max_holding_bars=slot.max_holding_bars,
                )

                # Classify each trade by entry regime
                for t in res.trades:
                    idx = regimes.index.searchsorted(t.entry_time)
                    if idx >= len(regimes):
                        idx = len(regimes) - 1
                    regime = str(regimes.iloc[idx])

                    if regime not in breakdown.data:
                        breakdown.data[regime] = {}

                    cell_key = f"{slot.name}|{sym}"
                    if cell_key not in breakdown.data[regime]:
                        breakdown.data[regime][cell_key] = {
                            "trades": [], "wins": 0, "losses": 0,
                            "gross_win": 0, "gross_loss": 0,
                        }

                    entry = breakdown.data[regime][cell_key]
                    entry["trades"].append(t.pnl)
                    if t.pnl > 0:
                        entry["wins"] += 1
                        entry["gross_win"] += t.pnl
                    else:
                        entry["losses"] += 1
                        entry["gross_loss"] += abs(t.pnl)

        # Compute summary stats
        for regime in breakdown.data:
            for cell_key in breakdown.data[regime]:
                e = breakdown.data[regime][cell_key]
                n = e["wins"] + e["losses"]
                e["total_trades"] = n
                e["pf"] = e["gross_win"] / e["gross_loss"] if e["gross_loss"] > 0 else 0
                e["win_rate"] = e["wins"] / n if n > 0 else 0
                if n > 1:
                    pnls = np.array(e["trades"])
                    e["sharpe"] = float(pnls.mean() / (pnls.std() + 1e-10) * np.sqrt(252))
                else:
                    e["sharpe"] = 0
                del e["trades"]  # Don't carry raw trades in output

        return breakdown


# ─── Capacity Simulation ────────────────────────────────────────

class CapacitySimulator:
    """Test how strategy performance degrades with larger position sizes.

    Models two effects:
        1. Higher slippage (sqrt market impact model)
        2. Higher commission cost (linear)

    Does NOT model:
        - Order book depletion
        - Adverse selection
        - Information leakage

    These are second-order effects at our scale ($100K-$1M).
    """

    # Typical crypto book depth per asset (conservative estimates)
    BOOK_DEPTH_USD = {
        "BTC/USDT": 2_000_000,
        "ETH/USDT": 1_000_000,
        "SOL/USDT": 300_000,
        "XRP/USDT": 200_000,
    }

    def simulate(
        self,
        slots,  # list[StrategySlot]
        datasets: dict[str, pd.DataFrame],
        capital: float = 100_000,
        size_pcts: list[float] = None,
    ) -> CapacityResult:
        """Run backtests at increasing position sizes with realistic slippage."""
        from src.backtest.engine import Backtester

        if size_pcts is None:
            size_pcts = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

        result = CapacityResult()
        best_risk_adj = -np.inf

        for pct in size_pcts:
            all_trades = []
            total_slippage_bps = 0
            n_sim = 0

            for slot in slots:
                for sym in slot.allowed_assets:
                    if sym not in datasets:
                        continue

                    df = datasets[sym]
                    signals = slot.get_signals(df)

                    # Compute slippage based on order size vs book depth
                    book_depth = self.BOOK_DEPTH_USD.get(sym, 500_000)
                    order_notional = capital * pct
                    # Sqrt impact model: slippage = sqrt(size/depth) × 30 bps
                    impact_bps = np.sqrt(order_notional / book_depth) * 30
                    slippage_pct = impact_bps / 10000

                    # Fit regime filter if present
                    if slot.regime_filter is not None and not slot.regime_filter._fitted:
                        slot.regime_filter.fit(df)

                    bt = Backtester(
                        initial_capital=capital,
                        commission_pct=0.001,
                        slippage_pct=slippage_pct,
                    )
                    res = bt.run(
                        df, lambda d, s=signals: s,
                        position_size_pct=pct,
                        stop_loss_atr=slot.stop_loss_atr,
                        take_profit_atr=slot.take_profit_atr,
                        max_holding_bars=slot.max_holding_bars,
                    )

                    all_trades.extend(res.trades)
                    total_slippage_bps += impact_bps
                    n_sim += 1

            # Compute portfolio metrics at this size
            w = [t for t in all_trades if t.pnl > 0]
            l = [t for t in all_trades if t.pnl <= 0]
            gw = sum(t.pnl for t in w)
            gl = sum(abs(t.pnl) for t in l)
            pf = gw / gl if gl > 0 else 0
            net = gw - gl

            pnls = np.array([t.pnl for t in all_trades]) if all_trades else np.array([0])
            sh = float(pnls.mean() / (pnls.std() + 1e-10) * np.sqrt(252)) if len(pnls) > 1 else 0

            avg_slip = total_slippage_bps / n_sim if n_sim > 0 else 0

            result.curve[pct] = {
                "pf": round(pf, 3),
                "sharpe": round(sh, 3),
                "trades": len(all_trades),
                "net_pnl": round(net, 2),
                "return_pct": round(net / capital * 100, 3),
                "avg_slippage_bps": round(avg_slip, 1),
            }

            # Track max viable and optimal
            if pf > 1.0:
                result.max_viable_pct = pct
            risk_adj = net / (capital * pct) if pct > 0 else 0  # PnL per dollar risked
            if risk_adj > best_risk_adj and pf > 1.0:
                best_risk_adj = risk_adj
                result.optimal_pct = pct

        return result


# ─── Full Institutional Validation ──────────────────────────────

class InstitutionalValidator:
    """Run the complete institutional validation suite."""

    def validate(
        self,
        engine,  # PortfolioEngine
        datasets: dict[str, pd.DataFrame] = None,
    ) -> InstitutionalReport:
        """Run all institutional checks and produce pass/fail verdicts."""

        if datasets is None:
            datasets = engine.load_data()

        report = InstitutionalReport()

        # ─── 1. Correlation Analysis ───────────────────────────
        logger.info("[1/3] Correlation analysis...")

        # Run backtest to get equity curves
        bt_result = engine.backtest(datasets)

        # Collect per-cell equity curves from backtest
        equity_curves = {}
        for slot in engine.slots:
            for sym in slot.allowed_assets:
                if sym not in datasets:
                    continue
                df = datasets[sym]
                if slot.regime_filter is not None:
                    slot.regime_filter.fit(df)
                signals = slot.get_signals(df)
                from src.backtest.engine import Backtester
                bt = Backtester(initial_capital=engine.capital)
                res = bt.run(
                    df, lambda d, s=signals: s,
                    position_size_pct=slot.position_size_pct,
                    stop_loss_atr=slot.stop_loss_atr,
                    take_profit_atr=slot.take_profit_atr,
                    max_holding_bars=slot.max_holding_bars,
                )
                cell_key = f"{slot.name}|{sym}"
                if len(res.equity_curve) > 0:
                    equity_curves[cell_key] = res.equity_curve

        corr_engine = CorrelationEngine()
        report.correlation = corr_engine.analyze(equity_curves)

        # Verdicts
        report.verdicts["strategy_corr_<0.7"] = report.correlation.max_strategy_corr < 0.70
        report.verdicts["asset_corr_<0.8"] = report.correlation.max_asset_corr < 0.80
        report.verdicts["diversification_ratio_>1"] = report.correlation.diversification_ratio > 1.0

        # ─── 2. Regime Breakdown ───────────────────────────────
        logger.info("[2/3] Regime breakdown...")

        regime_analyzer = RegimeAnalyzer()
        report.regime_breakdown = regime_analyzer.analyze(engine.slots, datasets)

        # Check: each strategy profitable in at least 1 regime
        strats_seen = set()
        strats_profitable = set()
        for regime, cells in report.regime_breakdown.data.items():
            for cell_key, stats in cells.items():
                strat = cell_key.split("|")[0]
                strats_seen.add(strat)
                if stats["pf"] > 1.0 and stats["total_trades"] >= 3:
                    strats_profitable.add(strat)
        report.verdicts["all_strats_profitable_1+_regime"] = strats_seen == strats_profitable

        # ─── 3. Capacity Simulation ────────────────────────────
        logger.info("[3/3] Capacity simulation...")

        cap_sim = CapacitySimulator()
        report.capacity = cap_sim.simulate(engine.slots, datasets, engine.capital)

        report.verdicts["pf>1_at_2%_size"] = report.capacity.curve.get(0.02, {}).get("pf", 0) > 1.0
        report.verdicts["pf>1_at_5%_size"] = report.capacity.curve.get(0.05, {}).get("pf", 0) > 1.0
        report.verdicts["max_viable_>=3%"] = report.capacity.max_viable_pct >= 0.03

        # ─── Score ─────────────────────────────────────────────
        report.total_tests = len(report.verdicts)
        report.score = sum(1 for v in report.verdicts.values() if v)

        return report

    def format_report(self, report: InstitutionalReport, engine=None) -> str:
        """Generate human-readable institutional validation report."""
        lines = []
        def p(s=""):
            lines.append(s)

        p("=" * 70)
        p("  INSTITUTIONAL VALIDATION REPORT")
        p("=" * 70)
        p()

        # ─── Correlation ──────────────────────────────────────
        p("─ STRATEGY CORRELATION MATRIX ───────────────────────")
        cr = report.correlation
        if not cr.strategy_corr.empty:
            strats = list(cr.strategy_corr.columns)
            header = f"  {'':20s}" + "".join(f"{s:>15s}" for s in strats)
            p(header)
            for s in strats:
                row = f"  {s:20s}"
                for s2 in strats:
                    val = cr.strategy_corr.loc[s, s2]
                    row += f"{val:>15.3f}"
                p(row)
            p()
            p(f"  Max pairwise correlation: {cr.max_strategy_corr:.3f} "
              f"{'[PASS < 0.70]' if cr.max_strategy_corr < 0.70 else '[FAIL >= 0.70]'}")
        else:
            p("  (insufficient data)")
        p()

        p("─ ASSET CORRELATION MATRIX ──────────────────────────")
        if not cr.asset_corr.empty:
            assets = list(cr.asset_corr.columns)
            header = f"  {'':15s}" + "".join(f"{a:>15s}" for a in assets)
            p(header)
            for a in assets:
                row = f"  {a:15s}"
                for a2 in assets:
                    val = cr.asset_corr.loc[a, a2]
                    row += f"{val:>15.3f}"
                p(row)
            p()
            p(f"  Max pairwise correlation: {cr.max_asset_corr:.3f} "
              f"{'[PASS < 0.80]' if cr.max_asset_corr < 0.80 else '[FAIL >= 0.80]'}")
        else:
            p("  (insufficient data)")
        p()

        p(f"  Diversification Ratio: {cr.diversification_ratio:.2f} "
          f"{'[PASS > 1.0]' if cr.diversification_ratio > 1.0 else '[FAIL <= 1.0]'}")
        p()

        # ─── Rolling Correlation ──────────────────────────────
        p("─ ROLLING CORRELATION (30-day window) ───────────────")
        for pair, rc in cr.rolling_corr.items():
            if len(rc) > 0:
                avg = rc.mean()
                max_val = rc.max()
                min_val = rc.min()
                p(f"  {pair}:")
                p(f"    Avg={avg:.3f}  Min={min_val:.3f}  Max={max_val:.3f}")
                # Check if ever exceeded 0.7
                pct_above = (rc.abs() > 0.7).mean() * 100
                p(f"    Time above |0.7|: {pct_above:.1f}%")
        p()

        # ─── Regime Breakdown ─────────────────────────────────
        p("─ REGIME PERFORMANCE BREAKDOWN ──────────────────────")
        for regime, cells in sorted(report.regime_breakdown.data.items()):
            p(f"  [{regime.upper()}]")
            p(f"    {'Cell':<30s} {'Trades':>7s} {'PF':>7s} {'WR':>7s} {'Sharpe':>7s}")
            p(f"    {'─'*30} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
            for cell_key, stats in sorted(cells.items()):
                strat, sym = cell_key.split("|")
                label = f"{strat}×{sym.split('/')[0]}"
                p(f"    {label:<30s} {stats['total_trades']:>7d} "
                  f"{stats['pf']:>7.2f} {stats['win_rate']:>6.1%} "
                  f"{stats['sharpe']:>7.2f}")
            p()

        # ─── Capacity Curve ───────────────────────────────────
        p("─ CAPACITY SIMULATION (slippage vs size) ────────────")
        p(f"  {'Size%':>7s} {'PF':>7s} {'Sharpe':>7s} {'Trades':>7s} "
          f"{'Net PnL':>10s} {'Ret%':>7s} {'Slip bps':>8s}")
        p(f"  {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*10} {'─'*7} {'─'*8}")
        for pct, stats in sorted(report.capacity.curve.items()):
            tag = " ◄ optimal" if pct == report.capacity.optimal_pct else ""
            tag += " ✗" if stats["pf"] < 1.0 else ""
            p(f"  {pct*100:>6.1f}% {stats['pf']:>7.3f} {stats['sharpe']:>7.3f} "
              f"{stats['trades']:>7d} {stats['net_pnl']:>+10.2f} "
              f"{stats['return_pct']:>+6.2f}% {stats['avg_slippage_bps']:>7.1f}{tag}")
        p()
        p(f"  Max viable position size: {report.capacity.max_viable_pct*100:.1f}%")
        p(f"  Optimal position size:    {report.capacity.optimal_pct*100:.1f}%")
        p()

        # ─── Scorecard ────────────────────────────────────────
        p("═" * 70)
        p("  SCORECARD")
        p("═" * 70)
        for test, passed in sorted(report.verdicts.items()):
            tag = "PASS" if passed else "FAIL"
            p(f"  [{tag}] {test}")
        p()
        p(f"  Score: {report.score}/{report.total_tests}")

        if report.score == report.total_tests:
            verdict = "INSTITUTIONAL GRADE — Ready for capital allocation"
        elif report.score >= report.total_tests - 1:
            verdict = "NEAR INSTITUTIONAL — One issue to fix"
        elif report.score >= report.total_tests // 2:
            verdict = "PROMISING — Needs work before capital"
        else:
            verdict = "NOT READY — Fundamental issues"
        p(f"  Verdict: {verdict}")
        p("═" * 70)

        return "\n".join(lines)

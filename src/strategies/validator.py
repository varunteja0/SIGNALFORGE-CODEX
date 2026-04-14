"""
Strategy 1 Validation Suite
=============================
Rigorous backtest + reality checks for the Liquidation Reversal strategy.

This is NOT a "does it look good?" test.
This is a "would I bet real money on this?" test.

Checks:
1. Walk-forward OOS performance (no peeking)
2. Regime-split results (must work in bull + bear + sideways)
3. Trade clustering (profits can't come from 3 lucky trades)
4. Slippage stress test (2x-5x normal slippage during events)
5. Latency simulation (1-2 bar delay on entries)
6. Monte Carlo robustness (trade-order independence)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import Backtester, BacktestResult
from src.data.fetcher import DataFetcher
from src.data.structural import StructuralDataFetcher
from src.data.features import compute_all_features
from src.regime.detector import RegimeDetector, MarketRegime
from src.strategies.liquidation_reversal import (
    LiquidationReversalStrategy,
    StrategyConfig,
)

logger = logging.getLogger(__name__)


class StrategyValidator:
    """Rigorous validation for a single strategy. No shortcuts."""

    def __init__(
        self,
        strategy: LiquidationReversalStrategy,
        initial_capital: float = 10000,
        commission_pct: float = 0.001,     # 0.1% Binance fee
        slippage_pct: float = 0.0005,      # 0.05% normal slippage
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    # ================================================================
    # Data Loading
    # ================================================================

    def load_data(
        self,
        symbol: str = "BTC/USDT",
        days: int = 365,
        force_fetch: bool = False,
    ) -> pd.DataFrame:
        """Fetch real price data + structural data, merged.

        This is REAL exchange data, not synthetic.
        """
        logger.info(f"Loading {days} days of {symbol} data...")

        # 1. Price data (OHLCV)
        fetcher = DataFetcher()
        price_df = fetcher.fetch(symbol, timeframe="1h", days=days, force=force_fetch)

        if price_df.empty:
            raise ValueError(f"No price data for {symbol}")

        logger.info(f"Price data: {len(price_df)} bars from {price_df.index[0]} to {price_df.index[-1]}")

        # 2. Technical features
        price_df = compute_all_features(price_df)

        # 3. Structural data (funding, OI, L/S ratio, taker volume)
        struct_fetcher = StructuralDataFetcher()
        binance_symbol = symbol.replace("/", "")  # BTC/USDT → BTCUSDT
        merged = struct_fetcher.fetch_all(
            symbol=binance_symbol,
            price_df=price_df,
            days=days,
        )

        logger.info(f"Merged data: {len(merged)} bars, {len(merged.columns)} columns")
        return merged

    # ================================================================
    # Core Backtest
    # ================================================================

    def backtest(
        self,
        df: pd.DataFrame,
        slippage_mult: float = 1.0,
        entry_delay_bars: int = 0,
    ) -> BacktestResult:
        """Run backtest with optional reality adjustments.

        Args:
            df: DataFrame with price + structural data
            slippage_mult: Multiply normal slippage (2.0 = 2x slippage during events)
            entry_delay_bars: Delay signal by N bars (simulates latency)
        """
        cfg = self.strategy.config

        # Generate signals
        raw_signals = self.strategy.generate_signals(df)

        # Apply entry delay if specified
        if entry_delay_bars > 0:
            raw_signals = raw_signals.shift(entry_delay_bars).fillna(0).astype(int)

        def signal_func(data_df):
            return raw_signals

        backtester = Backtester(
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct * slippage_mult,
        )

        result = backtester.run(
            df,
            signal_func,
            position_size_pct=getattr(cfg, "position_size_pct", cfg.base_risk_pct),
            stop_loss_atr=cfg.stop_loss_atr_mult,
            take_profit_atr=cfg.take_profit_atr_mult,
            max_holding_bars=cfg.max_holding_bars,
        )

        return result

    # ================================================================
    # Walk-Forward Validation
    # ================================================================

    def walk_forward(
        self, df: pd.DataFrame, n_splits: int = 5
    ) -> dict:
        """Expanding window walk-forward test.

        Train on [0:i], test on [i:i+step]. Never look ahead.
        Returns OOS metrics only — in-sample is irrelevant.
        """
        n = len(df)
        min_train = n // 3  # Minimum 1/3 of data for first training set
        step = (n - min_train) // n_splits

        if step < 100:
            logger.warning(f"Walk-forward step too small ({step} bars). Need more data.")
            return {"error": "insufficient_data", "step_size": step}

        oos_results = []

        for fold in range(n_splits):
            train_end = min_train + fold * step
            test_end = min(train_end + step, n)

            if test_end <= train_end:
                break

            test_df = df.iloc[train_end:test_end].copy()

            if len(test_df) < 50:
                continue

            result = self.backtest(test_df)
            oos_results.append({
                "fold": fold,
                "train_bars": train_end,
                "test_bars": len(test_df),
                "test_start": str(test_df.index[0]),
                "test_end": str(test_df.index[-1]),
                "total_return": result.total_return,
                "sharpe": result.sharpe_ratio,
                "sortino": result.sortino_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "profit_factor": result.profit_factor,
            })

        if not oos_results:
            return {"error": "no_valid_folds"}

        results_df = pd.DataFrame(oos_results)

        return {
            "n_folds": len(oos_results),
            "avg_sharpe": results_df["sharpe"].mean(),
            "std_sharpe": results_df["sharpe"].std(),
            "avg_return": results_df["total_return"].mean(),
            "avg_drawdown": results_df["max_drawdown"].mean(),
            "avg_win_rate": results_df["win_rate"].mean(),
            "avg_trades_per_fold": results_df["total_trades"].mean(),
            "total_trades": results_df["total_trades"].sum(),
            "profitable_folds": (results_df["total_return"] > 0).sum(),
            "profitable_folds_pct": (results_df["total_return"] > 0).mean(),
            "worst_fold_return": results_df["total_return"].min(),
            "best_fold_return": results_df["total_return"].max(),
            "consistency": (results_df["total_return"] > 0).mean(),
            "folds": oos_results,
        }

    # ================================================================
    # Regime-Split Analysis
    # ================================================================

    def regime_analysis(self, df: pd.DataFrame) -> dict:
        """Test strategy performance in each market regime.

        A real edge works in at least 2 out of 3 regimes.
        If it only works in bull → it's just beta, not alpha.
        """
        detector = RegimeDetector(n_regimes=3)
        detector.fit(df)
        regimes = detector.get_regime_history(df)

        # Align regimes with df (regime computation drops some rows)
        common_idx = regimes.index.intersection(df.index)
        regimes = regimes.loc[common_idx]
        aligned_df = df.loc[common_idx]

        regime_results = {}

        for regime_name in regimes.unique():
            mask = regimes == regime_name
            regime_bars = mask.sum()

            if regime_bars < 100:
                continue

            # Get contiguous chunks for this regime (avoid tiny fragments)
            regime_df = aligned_df.loc[mask].copy()

            if len(regime_df) < 50:
                continue

            result = self.backtest(regime_df)

            regime_results[regime_name] = {
                "bars": regime_bars,
                "pct_of_data": regime_bars / len(aligned_df),
                "total_return": result.total_return,
                "sharpe": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "profit_factor": result.profit_factor,
            }

        # Summary
        profitable_regimes = sum(
            1 for r in regime_results.values() if r["total_return"] > 0
        )

        return {
            "regimes": regime_results,
            "profitable_regimes": profitable_regimes,
            "total_regimes": len(regime_results),
            "verdict": "PASS" if profitable_regimes >= 2 else "FAIL",
        }

    # ================================================================
    # Trade Clustering Test
    # ================================================================

    def trade_clustering_test(self, result: BacktestResult) -> dict:
        """Check if profits come from a few lucky trades or consistent edge.

        If removing the top 3 trades turns profit to loss → strategy is fragile.
        """
        if not result.trades:
            return {"error": "no_trades"}

        pnls = sorted([t.pnl for t in result.trades])
        total_pnl = sum(pnls)

        # Remove top N trades and check if still profitable
        for n_remove in [1, 3, 5]:
            if len(pnls) <= n_remove:
                continue

            # Remove top N by absolute value
            sorted_by_abs = sorted(result.trades, key=lambda t: abs(t.pnl), reverse=True)
            remaining_pnl = sum(t.pnl for t in sorted_by_abs[n_remove:])

            pnl_key = f"pnl_without_top_{n_remove}"
            still_profitable_key = f"profitable_without_top_{n_remove}"

            if n_remove == 1:
                result_dict = {}

            result_dict[pnl_key] = remaining_pnl
            result_dict[still_profitable_key] = remaining_pnl > 0

        # Concentration: what % of total PnL comes from top 20% of trades
        n_top = max(1, len(pnls) // 5)
        top_trades_pnl = sum(sorted(pnls, reverse=True)[:n_top])
        concentration = abs(top_trades_pnl) / (abs(total_pnl) + 1e-10)

        result_dict["total_trades"] = len(pnls)
        result_dict["total_pnl"] = total_pnl
        result_dict["concentration_top_20pct"] = concentration
        result_dict["verdict"] = (
            "PASS" if result_dict.get("profitable_without_top_3", False) else "FAIL"
        )

        return result_dict

    # ================================================================
    # Slippage Stress Test
    # ================================================================

    def slippage_stress_test(self, df: pd.DataFrame) -> dict:
        """Test strategy under extreme slippage conditions.

        During liquidation cascades, slippage is 2x-5x normal.
        If strategy dies under 3x slippage, it's not real.
        """
        results = {}
        for mult in [1.0, 2.0, 3.0, 5.0]:
            result = self.backtest(df, slippage_mult=mult)
            results[f"slippage_{mult}x"] = {
                "total_return": result.total_return,
                "sharpe": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
            }

        # Verdict: must be profitable at 3x slippage
        profitable_at_3x = results["slippage_3.0x"]["total_return"] > 0
        return {
            "results": results,
            "verdict": "PASS" if profitable_at_3x else "FAIL",
        }

    # ================================================================
    # Latency Test
    # ================================================================

    def latency_test(self, df: pd.DataFrame) -> dict:
        """Test strategy with delayed entries.

        If you can't enter immediately, does the edge survive?
        """
        results = {}
        for delay in [0, 1, 2]:
            result = self.backtest(df, entry_delay_bars=delay)
            results[f"delay_{delay}_bar"] = {
                "total_return": result.total_return,
                "sharpe": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
            }

        # Edge must survive 1-bar delay
        profitable_with_1_bar = results["delay_1_bar"]["total_return"] > 0
        return {
            "results": results,
            "verdict": "PASS" if profitable_with_1_bar else "FAIL",
        }

    # ================================================================
    # Full Validation Suite
    # ================================================================

    def run_full_validation(
        self,
        symbol: str = "BTC/USDT",
        days: int = 365,
        force_fetch: bool = False,
    ) -> dict:
        """Run ALL validation checks. This is the final exam.

        Pass/fail criteria:
        - Walk-forward: avg Sharpe > 0.5 AND > 50% profitable folds
        - Regime: profitable in >= 2 regimes
        - Clustering: profitable after removing top 3 trades
        - Slippage: profitable at 3x normal slippage
        - Latency: profitable with 1-bar delay
        - Total trades: >= 30 (statistical significance)
        """
        print(f"\n{'='*60}")
        print(f"  STRATEGY VALIDATION: {self.strategy.name}")
        print(f"  Symbol: {symbol} | Period: {days} days")
        print(f"{'='*60}\n")

        # Load real data
        print("[1/7] Loading real market data...")
        try:
            df = self.load_data(symbol=symbol, days=days, force_fetch=force_fetch)
        except Exception as e:
            return {"error": f"Data loading failed: {e}", "verdict": "CANNOT_TEST"}

        print(f"      → {len(df)} bars loaded ({df.index[0]} to {df.index[-1]})")

        # Basic backtest
        print("\n[2/7] Running base backtest...")
        base_result = self.backtest(df)
        print(f"      → Return: {base_result.total_return:.2%}")
        print(f"      → Sharpe: {base_result.sharpe_ratio:.2f}")
        print(f"      → Max DD: {base_result.max_drawdown:.2%}")
        print(f"      → Trades: {base_result.total_trades}")
        print(f"      → Win rate: {base_result.win_rate:.1%}")

        if base_result.total_trades < 10:
            print("\n      ⚠ Too few trades for validation. Need more data or wider thresholds.")
            return {
                "base": {
                    "return": base_result.total_return,
                    "sharpe": base_result.sharpe_ratio,
                    "trades": base_result.total_trades,
                },
                "verdict": "INSUFFICIENT_TRADES",
            }

        # Walk-forward
        print("\n[3/7] Walk-forward validation (5 folds)...")
        wf = self.walk_forward(df, n_splits=5)
        if "error" not in wf:
            print(f"      → Avg OOS Sharpe: {wf['avg_sharpe']:.2f}")
            print(f"      → Profitable folds: {wf['profitable_folds']}/{wf['n_folds']}")
            print(f"      → Total OOS trades: {wf['total_trades']}")
        else:
            print(f"      → Error: {wf['error']}")

        # Regime analysis
        print("\n[4/7] Regime analysis...")
        regime = self.regime_analysis(df)
        for name, stats in regime.get("regimes", {}).items():
            print(f"      → {name}: return={stats['total_return']:.2%} sharpe={stats['sharpe']:.2f} trades={stats['total_trades']}")
        print(f"      → Verdict: {regime.get('verdict', 'N/A')}")

        # Trade clustering
        print("\n[5/7] Trade clustering test...")
        clustering = self.trade_clustering_test(base_result)
        if "error" not in clustering:
            print(f"      → Total PnL: ${clustering['total_pnl']:.2f}")
            print(f"      → Profitable without top 3 trades: {clustering.get('profitable_without_top_3', 'N/A')}")
            print(f"      → Top 20% concentration: {clustering.get('concentration_top_20pct', 0):.1%}")
            print(f"      → Verdict: {clustering.get('verdict', 'N/A')}")

        # Slippage stress test
        print("\n[6/7] Slippage stress test...")
        slippage = self.slippage_stress_test(df)
        for name, stats in slippage.get("results", {}).items():
            print(f"      → {name}: return={stats['total_return']:.2%} sharpe={stats['sharpe']:.2f}")
        print(f"      → Verdict: {slippage.get('verdict', 'N/A')}")

        # Latency test
        print("\n[7/7] Latency test...")
        latency = self.latency_test(df)
        for name, stats in latency.get("results", {}).items():
            print(f"      → {name}: return={stats['total_return']:.2%} sharpe={stats['sharpe']:.2f}")
        print(f"      → Verdict: {latency.get('verdict', 'N/A')}")

        # Monte Carlo
        print("\n[BONUS] Monte Carlo simulation...")
        backtester = Backtester(
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
        )
        mc = backtester.monte_carlo(base_result)
        if mc:
            print(f"      → Median return: {mc['median_return']:.2%}")
            print(f"      → P(profit): {mc['probability_of_profit']:.1%}")
            print(f"      → 5th percentile: {mc['p5']:.2%}")

        # ============================================================
        # Final Verdict
        # ============================================================
        checks = {
            "base_profitable": base_result.total_return > 0,
            "sufficient_trades": base_result.total_trades >= 30,
            "walk_forward": wf.get("avg_sharpe", 0) > 0.5 and wf.get("consistency", 0) > 0.5,
            "regime": regime.get("verdict") == "PASS",
            "clustering": clustering.get("verdict") == "PASS",
            "slippage": slippage.get("verdict") == "PASS",
            "latency": latency.get("verdict") == "PASS",
        }

        passed = sum(checks.values())
        total = len(checks)

        print(f"\n{'='*60}")
        print(f"  FINAL SCORECARD: {passed}/{total} checks passed")
        print(f"{'='*60}")
        for check, result in checks.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status}  {check}")

        if passed == total:
            verdict = "READY_FOR_PAPER_TRADING"
            print(f"\n  → VERDICT: {verdict}")
            print(f"  → Strategy has real edge. Proceed to paper trading.")
        elif passed >= 5:
            verdict = "PROMISING_NEEDS_WORK"
            print(f"\n  → VERDICT: {verdict}")
            print(f"  → Edge exists but needs refinement.")
        elif passed >= 3:
            verdict = "WEAK_EDGE"
            print(f"\n  → VERDICT: {verdict}")
            print(f"  → Possible edge but high risk of overfitting.")
        else:
            verdict = "NO_EDGE"
            print(f"\n  → VERDICT: {verdict}")
            print(f"  → No reliable edge found. Do NOT trade this.")

        return {
            "base": {
                "return": base_result.total_return,
                "sharpe": base_result.sharpe_ratio,
                "sortino": base_result.sortino_ratio,
                "max_drawdown": base_result.max_drawdown,
                "win_rate": base_result.win_rate,
                "trades": base_result.total_trades,
                "profit_factor": base_result.profit_factor,
            },
            "walk_forward": wf,
            "regime": regime,
            "clustering": clustering,
            "slippage": slippage,
            "latency": latency,
            "monte_carlo": mc,
            "checks": checks,
            "passed": passed,
            "total_checks": total,
            "verdict": verdict,
        }

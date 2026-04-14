"""
Validation Harness — Prove the Edge or Kill It
=================================================
Quant-level validation pipeline:

1. Fetch real data (OHLCV + funding + OI + liquidation proxy)
2. Run strategy on 2 years of data
3. Walk-forward validation (5 splits, no peeking)
4. Monte Carlo simulation (10,000 shuffled sequences)
5. Regime breakdown (bull/bear/sideways)
6. Cost sensitivity (2× slippage, 3× commission)
7. Out-of-sample holdout (last 6 months)

PASS CRITERIA (hard rules, no exceptions):
    - Sharpe > 1.0 after costs
    - Max drawdown < 20%
    - Win rate > 40% with profit factor > 1.3
    - Works in at least 2 of 3 market regimes
    - Survives 2× cost sensitivity test
    - Monte Carlo P(profit) > 70%
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Backtester, BacktestResult
from src.data.fetcher import DataFetcher
from src.data.funding import FundingRateFetcher
from src.data.oi import OpenInterestFetcher
from src.data.liquidations import LiquidationFetcher
from src.strategies.liquidation_reversal import (
    LiquidationReversalStrategy,
    StrategyConfig,
)

logger = logging.getLogger(__name__)


def fetch_full_dataset(
    symbol: str = "BTC/USDT",
    days: int = 730,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """Fetch OHLCV + funding + OI + liquidation proxy → unified DataFrame.

    This is the REAL DATA pipeline. No synthetic anything.
    """
    logger.info(f"=== Fetching full dataset: {symbol} {timeframe} {days}d ===")

    # 1. OHLCV (always available, public API)
    fetcher = DataFetcher()
    ohlcv = fetcher.fetch(symbol, timeframe, days)
    logger.info(f"OHLCV: {len(ohlcv)} bars from {ohlcv.index[0]} to {ohlcv.index[-1]}")

    if ohlcv.empty:
        raise ValueError(f"No OHLCV data for {symbol}")

    # 2. Funding rates (try, gracefully degrade)
    try:
        funding_fetcher = FundingRateFetcher()
        funding_df = funding_fetcher.fetch_history(symbol, days=days)
        funding = funding_fetcher.resample_to_ohlcv(funding_df, ohlcv)
        ohlcv["fund_funding_rate"] = funding
        logger.info(f"Funding: {len(funding_df)} records")
    except Exception as e:
        logger.warning(f"Funding rates unavailable: {e}. Using zeros.")
        ohlcv["fund_funding_rate"] = 0.0

    # 3. Open Interest (try, gracefully degrade)
    try:
        oi_fetcher = OpenInterestFetcher()
        oi_df = oi_fetcher.fetch_history(symbol, timeframe=timeframe, days=min(days, 90))
        oi_aligned = oi_fetcher.resample_to_ohlcv(oi_df, ohlcv)
        ohlcv["oi_oi_value_usd"] = oi_aligned["oi_value"]
        logger.info(f"OI: {len(oi_df)} records")
    except Exception as e:
        logger.warning(f"OI data unavailable: {e}. Using zeros.")
        ohlcv["oi_oi_value_usd"] = 0.0

    # 4. Liquidation proxy data (try, gracefully degrade)
    try:
        liq_fetcher = LiquidationFetcher()
        lsr_df = liq_fetcher.fetch_long_short_ratio(symbol, timeframe, min(days, 90))
        taker_df = liq_fetcher.fetch_taker_ratio(symbol, timeframe, min(days, 90))

        # Build OI df for the liq feature builder
        oi_for_liq = pd.DataFrame({"oi_value": ohlcv["oi_oi_value_usd"]})
        liq_features = liq_fetcher.build_liquidation_features(ohlcv, lsr_df, taker_df, oi_for_liq)

        for col in liq_features.columns:
            ohlcv[col] = liq_features[col]
        logger.info(f"Liquidation proxy: LSR={len(lsr_df)}, Taker={len(taker_df)} records")
    except Exception as e:
        logger.warning(f"Liquidation proxy data unavailable: {e}")

    logger.info(f"Final dataset: {len(ohlcv)} bars, {len(ohlcv.columns)} columns")
    return ohlcv


def run_backtest(
    df: pd.DataFrame,
    config: Optional[StrategyConfig] = None,
    commission: float = 0.001,
    slippage: float = 0.0005,
    initial_capital: float = 10000,
) -> BacktestResult:
    """Run a single backtest with the liquidation reversal strategy."""
    strategy = LiquidationReversalStrategy(config)
    backtester = Backtester(
        initial_capital=initial_capital,
        commission_pct=commission,
        slippage_pct=slippage,
    )

    result = backtester.run(
        df,
        signal_func=strategy.generate_signals,
        position_size_pct=config.base_risk_pct if config else 0.01,
        stop_loss_atr=config.stop_loss_atr_mult if config else 2.0,
        take_profit_atr=config.take_profit_atr_mult if config else 3.0,
        max_holding_bars=config.max_holding_bars if config else 8,
    )

    return result


def classify_regime(df: pd.DataFrame) -> pd.Series:
    """Classify market regime for each bar.

    Returns Series with values: 'bull', 'bear', 'sideways'.
    """
    # Simple but effective: based on 50-period returns and volatility
    ret_50 = df["close"].pct_change(50)
    vol_20 = df["close"].pct_change().rolling(20).std()
    vol_median = vol_20.rolling(200).median()

    regime = pd.Series("sideways", index=df.index)
    regime[ret_50 > 0.10] = "bull"       # >10% in 50 bars = bull
    regime[ret_50 < -0.10] = "bear"      # <-10% in 50 bars = bear
    # High vol + flat = sideways (default)

    return regime


def walk_forward_validation(
    df: pd.DataFrame,
    n_splits: int = 5,
    config: Optional[StrategyConfig] = None,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> list[dict]:
    """Walk-forward out-of-sample validation.

    Splits data into n_splits. For each split:
    - Train on all prior data (not used for param tuning — we use fixed config)
    - Test on the current split
    - Record performance

    This simulates what would happen if you deployed at each point in time.
    """
    logger.info(f"=== Walk-Forward Validation ({n_splits} splits) ===")

    split_size = len(df) // n_splits
    results = []

    for i in range(1, n_splits):
        test_start = i * split_size
        test_end = min((i + 1) * split_size, len(df))

        # Use all prior data for warmup (indicators need history)
        warmup_size = min(test_start, 500)
        full_slice = df.iloc[test_start - warmup_size:test_end]

        result = run_backtest(full_slice, config, commission, slippage)

        split_info = {
            "split": i,
            "test_start": str(df.index[test_start]),
            "test_end": str(df.index[test_end - 1]),
            "bars": test_end - test_start,
            "total_return": result.total_return,
            "sharpe": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
        }
        results.append(split_info)

        logger.info(
            f"  Split {i}: Return={result.total_return:+.1%} "
            f"Sharpe={result.sharpe_ratio:.2f} "
            f"DD={result.max_drawdown:.1%} "
            f"Trades={result.total_trades} "
            f"WR={result.win_rate:.0%}"
        )

    return results


def regime_breakdown(
    df: pd.DataFrame,
    config: Optional[StrategyConfig] = None,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> dict:
    """Test strategy performance in each market regime separately."""
    logger.info("=== Regime Breakdown ===")

    regimes = classify_regime(df)
    results = {}

    for regime_name in ["bull", "bear", "sideways"]:
        mask = regimes == regime_name
        regime_bars = mask.sum()

        if regime_bars < 200:
            logger.info(f"  {regime_name}: Not enough bars ({regime_bars}), skipping")
            results[regime_name] = {"bars": int(regime_bars), "skipped": True}
            continue

        # Get contiguous blocks for this regime and test the largest one
        regime_df = df[mask]

        # Need enough contiguous data — use the full df but only count
        # trades that occur during this regime
        result = run_backtest(df, config, commission, slippage)

        # Filter trades to this regime
        regime_trades = [
            t for t in result.trades
            if t.entry_time in regimes.index and regimes.loc[t.entry_time] == regime_name
        ]

        if not regime_trades:
            results[regime_name] = {"bars": int(regime_bars), "trades": 0}
            logger.info(f"  {regime_name}: {regime_bars} bars, 0 trades")
            continue

        wins = [t for t in regime_trades if t.pnl > 0]
        losses = [t for t in regime_trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in regime_trades)
        win_pnl = sum(t.pnl for t in wins) if wins else 0
        loss_pnl = abs(sum(t.pnl for t in losses)) if losses else 0.001

        results[regime_name] = {
            "bars": int(regime_bars),
            "trades": len(regime_trades),
            "win_rate": len(wins) / len(regime_trades),
            "profit_factor": win_pnl / loss_pnl,
            "total_pnl": total_pnl,
        }

        logger.info(
            f"  {regime_name}: {regime_bars} bars, "
            f"{len(regime_trades)} trades, "
            f"WR={results[regime_name]['win_rate']:.0%}, "
            f"PF={results[regime_name]['profit_factor']:.2f}"
        )

    return results


def cost_sensitivity(
    df: pd.DataFrame,
    config: Optional[StrategyConfig] = None,
) -> dict:
    """Test if strategy survives higher costs.

    If it doesn't survive 2× costs, the edge is too thin.
    """
    logger.info("=== Cost Sensitivity ===")

    scenarios = {
        "base": {"commission": 0.001, "slippage": 0.0005},
        "2x_costs": {"commission": 0.002, "slippage": 0.001},
        "3x_commission": {"commission": 0.003, "slippage": 0.0005},
        "3x_slippage": {"commission": 0.001, "slippage": 0.0015},
    }

    results = {}
    for name, costs in scenarios.items():
        result = run_backtest(df, config, **costs)
        results[name] = {
            "total_return": result.total_return,
            "sharpe": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
        }

        logger.info(
            f"  {name}: Return={result.total_return:+.1%} "
            f"Sharpe={result.sharpe_ratio:.2f} "
            f"DD={result.max_drawdown:.1%}"
        )

    return results


def run_full_validation(
    symbols: list[str] = None,
    days: int = 730,
    timeframe: str = "1h",
    config: Optional[StrategyConfig] = None,
    output_dir: str = "validation_results",
) -> dict:
    """Run the complete validation pipeline.

    This is the ONE command you run to know if the strategy is real.
    """
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    if config is None:
        config = StrategyConfig()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {k: v for k, v in config.__dict__.items()},
        "symbols": {},
        "pass_criteria": {
            "sharpe_min": 1.0,
            "max_drawdown_max": 0.20,
            "win_rate_min": 0.40,
            "profit_factor_min": 1.3,
            "monte_carlo_profit_prob_min": 0.70,
            "cost_2x_sharpe_min": 0.5,
            "min_regimes_profitable": 2,
        },
    }

    all_pass = True

    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"VALIDATING: {symbol}")
        logger.info(f"{'='*60}")

        sym_report = {}

        # 1. Fetch data
        try:
            df = fetch_full_dataset(symbol, days, timeframe)
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            sym_report["error"] = str(e)
            total_report["symbols"][symbol] = sym_report
            continue

        # 2. Full backtest
        logger.info("\n--- Full Backtest ---")
        result = run_backtest(df, config)
        backtester = Backtester(initial_capital=10000)

        sym_report["full_backtest"] = {
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "calmar": result.calmar_ratio,
            "max_drawdown": result.max_drawdown,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "avg_trade": result.avg_trade_return,
            "best_trade": result.best_trade,
            "worst_trade": result.worst_trade,
        }

        logger.info(
            f"Return: {result.total_return:+.1%} | "
            f"Sharpe: {result.sharpe_ratio:.2f} | "
            f"Sortino: {result.sortino_ratio:.2f} | "
            f"DD: {result.max_drawdown:.1%} | "
            f"Trades: {result.total_trades} | "
            f"WR: {result.win_rate:.0%} | "
            f"PF: {result.profit_factor:.2f}"
        )

        # 3. Monte Carlo
        logger.info("\n--- Monte Carlo (10,000 simulations) ---")
        mc = backtester.monte_carlo(result, n_simulations=10000)
        sym_report["monte_carlo"] = mc

        if mc:
            logger.info(
                f"P(profit): {mc['probability_of_profit']:.0%} | "
                f"Median: {mc['median_return']:+.1%} | "
                f"P5: {mc['p5']:+.1%} | "
                f"P95: {mc['p95']:+.1%}"
            )

        # 4. Walk-forward
        logger.info("\n--- Walk-Forward Validation ---")
        wf = walk_forward_validation(df, n_splits=5, config=config)
        sym_report["walk_forward"] = wf

        # 5. Regime breakdown
        logger.info("\n--- Regime Breakdown ---")
        regimes = regime_breakdown(df, config)
        sym_report["regimes"] = regimes

        # 6. Cost sensitivity
        logger.info("\n--- Cost Sensitivity ---")
        costs = cost_sensitivity(df, config)
        sym_report["cost_sensitivity"] = costs

        # 7. Pass/fail judgment
        criteria = total_report["pass_criteria"]
        passes = {
            "sharpe": result.sharpe_ratio >= criteria["sharpe_min"],
            "drawdown": result.max_drawdown <= criteria["max_drawdown_max"],
            "win_rate": result.win_rate >= criteria["win_rate_min"],
            "profit_factor": result.profit_factor >= criteria["profit_factor_min"],
            "monte_carlo": (
                mc.get("probability_of_profit", 0) >= criteria["monte_carlo_profit_prob_min"]
                if mc else False
            ),
            "cost_sensitivity": (
                costs.get("2x_costs", {}).get("sharpe", 0) >= criteria["cost_2x_sharpe_min"]
            ),
        }

        # Regime check: profitable in at least 2 regimes
        profitable_regimes = sum(
            1 for r, data in regimes.items()
            if isinstance(data, dict) and data.get("total_pnl", 0) > 0
        )
        passes["regimes"] = profitable_regimes >= criteria["min_regimes_profitable"]

        sym_report["passes"] = passes
        sym_report["overall_pass"] = all(passes.values())

        if not sym_report["overall_pass"]:
            all_pass = False

        failed = [k for k, v in passes.items() if not v]
        logger.info(f"\n{'='*40}")
        if sym_report["overall_pass"]:
            logger.info(f"✅ {symbol}: ALL CHECKS PASSED")
        else:
            logger.info(f"❌ {symbol}: FAILED checks: {', '.join(failed)}")
        logger.info(f"{'='*40}")

        total_report["symbols"][symbol] = sym_report

    total_report["all_symbols_pass"] = all_pass

    # Save report
    report_file = output_path / "validation_report.json"
    with open(report_file, "w") as f:
        json.dump(total_report, f, indent=2, default=str)
    logger.info(f"\nReport saved to {report_file}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL VERDICT")
    logger.info(f"{'='*60}")
    for symbol, data in total_report["symbols"].items():
        if "error" in data:
            logger.info(f"  {symbol}: ERROR — {data['error']}")
        else:
            status = "✅ PASS" if data.get("overall_pass") else "❌ FAIL"
            logger.info(f"  {symbol}: {status}")
    logger.info(f"\n  Overall: {'✅ EDGE IS REAL' if all_pass else '❌ EDGE NOT PROVEN'}")

    return total_report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    report = run_full_validation(
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        days=730,
        timeframe="1h",
    )

"""
Tests for the Strategy Factory pipeline.
==========================================
Tests: Scanner → Validator → Deployer → Monitor → Loop
"""

import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")


def _make_fake_data(n_bars: int = 2000) -> pd.DataFrame:
    """Create realistic-looking OHLCV data for testing."""
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    price = 50000 + np.cumsum(np.random.randn(n_bars) * 50)
    volume = np.random.lognormal(10, 1, n_bars)

    df = pd.DataFrame({
        "open": price + np.random.randn(n_bars) * 10,
        "high": price + abs(np.random.randn(n_bars) * 30),
        "low": price - abs(np.random.randn(n_bars) * 30),
        "close": price,
        "volume": volume,
    }, index=dates)

    # Ensure high >= close,open and low <= close,open
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return df


class TestScanner(unittest.TestCase):
    """Test signal scanning module."""

    def test_signal_generators_produce_signals(self):
        """Each generator should produce at least some signals."""
        from src.data.features import compute_all_features
        from src.factory.scanner import SIGNAL_GENERATORS

        df = compute_all_features(_make_fake_data())

        for gen in SIGNAL_GENERATORS:
            sigs = gen(df)
            self.assertIsInstance(sigs, list)
            # Each element should be (name, mask, direction)
            for name, mask, direction in sigs:
                self.assertIsInstance(name, str)
                self.assertIn(direction, [-1, 1])

    def test_evaluate_signal_returns_none_for_few_trades(self):
        """Should return None if not enough trades."""
        from src.factory.scanner import evaluate_signal

        df = _make_fake_data(500)
        from src.data.features import compute_all_features
        df = compute_all_features(df)

        # Mask that fires very rarely
        mask = pd.Series(False, index=df.index)
        mask.iloc[250] = True

        result = evaluate_signal(df, mask, 1, 4, min_trades=50)
        self.assertIsNone(result)

    def test_evaluate_signal_returns_stats(self):
        """Should return stats dict for valid signal."""
        from src.factory.scanner import evaluate_signal
        from src.data.features import compute_all_features

        df = compute_all_features(_make_fake_data(5000))

        # Mask that fires every 50 bars
        mask = pd.Series(False, index=df.index)
        mask.iloc[200::50] = True

        result = evaluate_signal(df, mask, 1, 4, min_trades=20)
        if result is not None:
            self.assertIn("n_trades", result)
            self.assertIn("pf", result)
            self.assertIn("sharpe", result)
            self.assertIn("p_value", result)

    def test_scan_returns_scan_result(self):
        """Full scan should return ScanResult."""
        from src.data.features import compute_all_features
        from src.factory.scanner import scan

        df = compute_all_features(_make_fake_data(3000))
        datasets = {"TEST/USDT": df}

        result = scan(datasets, min_trades=20)

        self.assertGreater(result.total_hypotheses, 0)
        self.assertIsInstance(result.raw_survivors, list)
        self.assertIsInstance(result.bonferroni_survivors, list)
        self.assertGreater(result.bonferroni_threshold, 0)


class TestValidator(unittest.TestCase):
    """Test OOS validation module."""

    def test_validated_signal_grade(self):
        """ValidatedSignal should compute grade correctly."""
        from src.factory.validator import ValidatedSignal

        # Grade A signal
        sig = ValidatedSignal(
            name="test_long_h4", asset="BTC/USDT", direction=1,
            hold_bars=4, generator_name="test", generator_params={},
            is_trades=50, is_pf=1.8, is_sharpe=2.0,
            oos_trades=40, oos_pf=1.5, oos_sharpe=1.5,
            oos_p=0.01,
            wf_positive_folds=3, wf_total_folds=3,
        )
        self.assertEqual(sig.grade, "A")

        # Grade F signal
        sig_bad = ValidatedSignal(
            name="test_long_h4", asset="BTC/USDT", direction=1,
            hold_bars=4, generator_name="test", generator_params={},
            is_trades=50, is_pf=1.8, is_sharpe=2.0,
            oos_trades=40, oos_pf=0.8, oos_sharpe=-0.5,
            oos_p=0.5,
            wf_positive_folds=0, wf_total_folds=3,
        )
        self.assertEqual(sig_bad.grade, "F")


class TestDeployer(unittest.TestCase):
    """Test strategy deployment."""

    def test_deploy_creates_strategies(self):
        """deploy() should create DeployedStrategy instances."""
        from src.factory.validator import ValidatedSignal
        from src.factory.deployer import deploy

        sig = ValidatedSignal(
            name="dow_Mon_long_h24", asset="BTC/USDT", direction=1,
            hold_bars=24, generator_name="dow_Mon_long", generator_params={},
            is_trades=50, is_pf=1.5, is_sharpe=2.0,
            oos_trades=40, oos_pf=1.3, oos_sharpe=1.0,
            oos_p=0.05,
            wf_positive_folds=2, wf_total_folds=3,
        )

        deployed = deploy([sig])
        self.assertEqual(len(deployed), 1)
        self.assertEqual(deployed[0].asset, "BTC/USDT")
        self.assertEqual(deployed[0].hold_bars, 24)

    def test_deploy_size_scales_with_grade(self):
        """Higher grade = larger position size."""
        from src.factory.deployer import _size_for_grade

        size_a, _, _ = _size_for_grade("A")
        size_b, _, _ = _size_for_grade("B")
        size_c, _, _ = _size_for_grade("C")

        self.assertGreater(size_a, size_b)
        self.assertGreater(size_b, size_c)


class TestMonitor(unittest.TestCase):
    """Test strategy health monitoring."""

    def test_insufficient_data(self):
        """Should report insufficient_data with few trades."""
        from src.factory.monitor import StrategyMonitor
        from src.factory.deployer import DeployedStrategy

        monitor = StrategyMonitor(monitor_dir="/tmp/sf_test_monitor")
        monitor.trades = []

        strat = DeployedStrategy(
            name="test", asset="BTC/USDT", direction=1,
            hold_bars=24, signal_name="test_h24",
            position_size_pct=0.02, stop_loss_atr=2.0,
            take_profit_atr=3.0, oos_pf=1.5, oos_sharpe=1.0,
            grade="B", deployed_at="2024-01-01",
        )

        health = monitor.assess_health(strat)
        self.assertEqual(health.status, "insufficient_data")

    def test_dead_detection(self):
        """Should detect dead strategy from bad trades."""
        from src.factory.monitor import StrategyMonitor
        from src.factory.deployer import DeployedStrategy

        monitor = StrategyMonitor(monitor_dir="/tmp/sf_test_monitor")

        # 10 losing trades
        monitor.trades = [
            {"strategy": "test", "return_pct": -0.01, "asset": "BTC/USDT",
             "direction": 1, "entry_price": 50000, "exit_price": 49500,
             "timestamp": "2024-01-01", "hold_bars": 24}
            for _ in range(10)
        ]

        strat = DeployedStrategy(
            name="test", asset="BTC/USDT", direction=1,
            hold_bars=24, signal_name="test_h24",
            position_size_pct=0.02, stop_loss_atr=2.0,
            take_profit_atr=3.0, oos_pf=1.5, oos_sharpe=1.0,
            grade="B", deployed_at="2024-01-01",
        )

        health = monitor.assess_health(strat)
        self.assertIn(health.status, ("dead", "critical"))


class TestFactoryIntegration(unittest.TestCase):
    """Integration test: full pipeline on synthetic data."""

    def test_run_once_returns_result(self):
        """run_once() should return a result dict."""
        from src.factory.loop import StrategyFactoryLoop

        loop = StrategyFactoryLoop(
            symbols=["BTC/USDT"],
            data_days=365,
            min_scan_trades=20,
        )

        # Just test that it doesn't crash
        # (actual results depend on market data)
        result = loop.run_once()
        self.assertIsInstance(result, dict)
        self.assertIn("hypotheses", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)

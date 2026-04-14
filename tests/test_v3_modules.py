"""
Tests for V3.0 Advanced Intelligence Modules
=============================================
Tests all 10 new modules without requiring network access.
Uses mocking for API calls, validates core logic.
"""

import sys
import os
import json
import time
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ================================================================
# Test 1: TheGraph Fetcher
# ================================================================
class TestTheGraphFetcher(unittest.TestCase):
    def test_import(self):
        from src.data.thegraph import TheGraphFetcher
        fetcher = TheGraphFetcher()
        self.assertIsNotNone(fetcher)

    def test_compute_liquidation_features(self):
        from src.data.thegraph import TheGraphFetcher, OnChainPosition
        fetcher = TheGraphFetcher()
        positions = [
            OnChainPosition(
                protocol="aave_v3", chain="ethereum", asset="ETH",
                collateral_usd=100000, debt_usd=60000, health_factor=1.5,
                liquidation_price=1200, current_price=2000,
            ),
            OnChainPosition(
                protocol="aave_v3", chain="ethereum", asset="ETH",
                collateral_usd=50000, debt_usd=45000, health_factor=1.05,
                liquidation_price=1800, current_price=2000,
            ),
        ]
        features = fetcher.compute_liquidation_features(positions, 2000)
        self.assertIn("n_positions", features)
        self.assertIn("total_collateral_usd", features)
        self.assertIn("pct_near_liquidation", features)
        self.assertEqual(features["n_positions"], 2)
        self.assertGreater(features["pct_near_liquidation"], 0)


# ================================================================
# Test 2: Sentiment Engine
# ================================================================
class TestSentimentEngine(unittest.TestCase):
    def test_import(self):
        from src.sentiment.engine import SentimentEngine
        engine = SentimentEngine()
        self.assertIsNotNone(engine)

    def test_analyze_text_bullish(self):
        from src.sentiment.engine import SentimentEngine
        engine = SentimentEngine()
        score = engine._analyze_text("Bitcoin is pumping to the moon! Bullish breakout!")
        self.assertGreater(score, 0)

    def test_analyze_text_bearish(self):
        from src.sentiment.engine import SentimentEngine
        engine = SentimentEngine()
        score = engine._analyze_text("Market crash incoming. Dump everything. Bear market confirmed.")
        self.assertLess(score, 0)

    def test_compute_sentiment_features(self):
        from src.sentiment.engine import SentimentEngine
        engine = SentimentEngine()
        features = engine.compute_sentiment_features("BTC")
        self.assertIn("sentiment_composite", features)
        self.assertIn("sentiment_fear_greed", features)
        self.assertIsInstance(features["sentiment_composite"], float)


# ================================================================
# Test 3: Multi-Chain Liquidation Oracle
# ================================================================
class TestMultiChainOracle(unittest.TestCase):
    def test_import(self):
        from src.liquidation.multichain import MultiChainLiquidationOracle
        oracle = MultiChainLiquidationOracle()
        self.assertIsNotNone(oracle)

    def test_assess_cross_chain_risk(self):
        from src.liquidation.multichain import MultiChainLiquidationOracle
        oracle = MultiChainLiquidationOracle()
        risk = oracle.assess_cross_chain_risk("ETH")
        self.assertIsNotNone(risk)
        self.assertGreaterEqual(risk.aggregate_risk_score, 0)
        self.assertLessEqual(risk.aggregate_risk_score, 100)
        self.assertIn(risk.recommendation, [
            "SAFE", "MONITOR", "REDUCE_EXPOSURE", "HEDGE_NOW", "EXIT_ALL"
        ])

    def test_compute_features(self):
        from src.liquidation.multichain import MultiChainLiquidationOracle
        oracle = MultiChainLiquidationOracle()
        features = oracle.compute_features("ETH")
        self.assertIn("multichain_aggregate_risk", features)
        self.assertIn("multichain_contagion", features)


# ================================================================
# Test 4: Transformer Regime Predictor
# ================================================================
class TestTransformerRegime(unittest.TestCase):
    def test_import(self):
        from src.regime.transformer import TransformerRegimePredictor
        predictor = TransformerRegimePredictor(n_features=10, seq_len=20)
        self.assertIsNotNone(predictor)

    def test_attention_block(self):
        from src.regime.transformer import AttentionBlock
        block = AttentionBlock(d_model=16, n_heads=4)
        x = np.random.randn(20, 16)
        out = block.forward(x)
        self.assertEqual(out.shape, (20, 16))

    def test_predict_with_synthetic_data(self):
        from src.regime.transformer import TransformerRegimePredictor
        predictor = TransformerRegimePredictor(n_features=5, seq_len=10)

        # Create synthetic OHLCV data
        n = 100
        df = pd.DataFrame({
            "open": np.random.randn(n).cumsum() + 100,
            "high": np.random.randn(n).cumsum() + 101,
            "low": np.random.randn(n).cumsum() + 99,
            "close": np.random.randn(n).cumsum() + 100,
            "volume": np.abs(np.random.randn(n)) * 1000,
        })

        result = predictor.predict(df)
        self.assertIsNotNone(result)
        self.assertIn("predicted_regime", result)
        self.assertIn("confidence", result)
        self.assertIn("regime_probabilities", result)
        self.assertIn(result["predicted_regime"], [0, 1, 2])


# ================================================================
# Test 5: Meta-Evolution Engine
# ================================================================
class TestMetaEvolution(unittest.TestCase):
    def test_import(self):
        from src.alpha_genome.meta_evolution import MetaEvolutionEngine
        engine = MetaEvolutionEngine(
            meta_population_size=3, meta_generations=1, inner_generations=2,
        )
        self.assertIsNotNone(engine)

    def test_evolution_config(self):
        from src.alpha_genome.meta_evolution import EvolutionConfig
        config = EvolutionConfig()
        mutated = config.mutate()
        self.assertIsInstance(mutated.population_size, int)
        self.assertGreater(mutated.population_size, 0)
        self.assertGreater(mutated.mutation_rate, 0)
        self.assertLess(mutated.mutation_rate, 1)


# ================================================================
# Test 6: Predictive Liquidation Timing
# ================================================================
class TestPredictiveLiquidation(unittest.TestCase):
    def test_import(self):
        from src.liquidation.predictive import LiquidationTimingPredictor
        predictor = LiquidationTimingPredictor()
        self.assertIsNotNone(predictor)

    def test_predict_with_synthetic(self):
        from src.liquidation.predictive import LiquidationTimingPredictor
        predictor = LiquidationTimingPredictor()

        n = 200
        df = pd.DataFrame({
            "open": np.random.randn(n).cumsum() + 100,
            "high": np.random.randn(n).cumsum() + 101,
            "low": np.random.randn(n).cumsum() + 99,
            "close": np.random.randn(n).cumsum() + 100,
            "volume": np.abs(np.random.randn(n)) * 1000,
        })

        pred = predictor.predict(df, asset="ETH")
        self.assertIsNotNone(pred)
        self.assertGreaterEqual(pred.probability_1h, 0)
        self.assertLessEqual(pred.probability_1h, 1)
        self.assertIn(pred.recommended_action, ["HOLD", "SHORT_HEDGE", "AGGRESSIVE_SHORT", "BUY_DIP"])

    def test_compute_features(self):
        from src.liquidation.predictive import LiquidationTimingPredictor
        predictor = LiquidationTimingPredictor()

        n = 200
        df = pd.DataFrame({
            "open": np.random.randn(n).cumsum() + 100,
            "high": np.random.randn(n).cumsum() + 101,
            "low": np.random.randn(n).cumsum() + 99,
            "close": np.random.randn(n).cumsum() + 100,
            "volume": np.abs(np.random.randn(n)) * 1000,
        })

        features = predictor.compute_features(df)
        self.assertIn("pred_liq_1h", features)
        self.assertIn("pred_liq_4h", features)
        self.assertIn("pred_liq_24h", features)


# ================================================================
# Test 7: Funding Rate Arbitrage
# ================================================================
class TestFundingArb(unittest.TestCase):
    def test_import(self):
        from src.arbitrage.funding import FundingArbEngine
        engine = FundingArbEngine()
        self.assertIsNotNone(engine)

    def test_compute_features(self):
        from src.arbitrage.funding import FundingArbEngine
        engine = FundingArbEngine()
        features = engine.compute_features("BTC/USDT")
        self.assertIn("funding_spread_max", features)
        self.assertIn("funding_avg", features)
        self.assertIn("funding_n_exchanges", features)

    def test_opportunity_dataclass(self):
        from src.arbitrage.funding import FundingArbOpportunity
        opp = FundingArbOpportunity(
            symbol="BTC/USDT",
            long_exchange="bybit",
            short_exchange="binance",
            long_rate=-0.0001,
            short_rate=0.0005,
            spread=17.5,
            spread_8h=0.016,
            next_funding_in_sec=3600,
            confidence=0.8,
        )
        self.assertTrue(opp.is_actionable)
        self.assertEqual(opp.symbol, "BTC/USDT")


# ================================================================
# Test 8: MEV Execution Engine
# ================================================================
class TestMEVExecution(unittest.TestCase):
    def test_import(self):
        from src.execution.mev import MEVExecutionEngine
        engine = MEVExecutionEngine()
        self.assertIsNotNone(engine)

    def test_plan_small_order(self):
        from src.execution.mev import MEVExecutionEngine, MEVMetrics
        engine = MEVExecutionEngine()
        metrics = MEVMetrics(gas_price_gwei=30, mev_revenue_1h_eth=0.5)
        plan = engine.plan_execution("buy", 3000, mev_metrics=metrics)
        self.assertEqual(plan.strategy, "immediate")
        self.assertEqual(plan.n_slices, 1)

    def test_plan_liquidation_snipe(self):
        from src.execution.mev import MEVExecutionEngine, MEVMetrics
        engine = MEVExecutionEngine()
        metrics = MEVMetrics(gas_price_gwei=30)
        plan = engine.plan_execution("buy", 50000, mev_metrics=metrics, is_liquidation_nearby=True)
        self.assertEqual(plan.strategy, "snipe")
        self.assertEqual(plan.urgency, 1.0)

    def test_order_flow_toxicity(self):
        from src.execution.mev import MEVExecutionEngine
        engine = MEVExecutionEngine()

        trades = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="1min"),
            "price": np.random.randn(50).cumsum() + 100,
            "volume": np.abs(np.random.randn(50)) * 10,
            "side": np.random.choice(["buy", "sell"], 50),
        })

        result = engine.detect_order_flow_toxicity(trades)
        self.assertIn("toxicity", result)
        self.assertIn("recommendation", result)
        self.assertGreaterEqual(result["toxicity"], 0)
        self.assertLessEqual(result["toxicity"], 1)

    def test_compute_features(self):
        from src.execution.mev import MEVExecutionEngine
        engine = MEVExecutionEngine()
        # Add some gas history so features work
        engine._gas_history = list(np.random.uniform(20, 80, 50))
        features = engine.compute_features()
        self.assertIn("mev_intensity", features)
        self.assertIn("gas_zscore", features)


# ================================================================
# Test 9: Synthetic Market Maker
# ================================================================
class TestMarketMaker(unittest.TestCase):
    def test_import(self):
        from src.execution.market_maker import SyntheticMarketMaker, MMConfig
        mm = SyntheticMarketMaker()
        self.assertIsNotNone(mm)

    def test_optimal_spread(self):
        from src.execution.market_maker import SyntheticMarketMaker
        mm = SyntheticMarketMaker()
        spread = mm.compute_optimal_spread(50000, 0.02, 1.0)
        self.assertGreater(spread, 0)
        self.assertLess(spread, 200)  # Should be reasonable

    def test_generate_quote(self):
        from src.execution.market_maker import SyntheticMarketMaker
        mm = SyntheticMarketMaker()
        quote = mm.generate_quote(50000, signal_score=0.5, volatility=0.01)
        self.assertGreater(quote.bid_price, 0)
        self.assertGreater(quote.ask_price, 0)
        self.assertGreater(quote.ask_price, quote.bid_price)
        self.assertGreater(quote.spread_bps, 0)

    def test_signal_skew(self):
        from src.execution.market_maker import SyntheticMarketMaker
        mm = SyntheticMarketMaker()
        # Bullish signal should skew bid up (tighter, more willing to buy)
        quote_bull = mm.generate_quote(50000, signal_score=0.8, volatility=0.01)
        mm2 = SyntheticMarketMaker()
        quote_bear = mm2.generate_quote(50000, signal_score=-0.8, volatility=0.01)
        # Bullish bid should be higher than bearish bid
        self.assertGreater(quote_bull.bid_price, quote_bear.bid_price)

    def test_fill_processing(self):
        from src.execution.market_maker import SyntheticMarketMaker
        mm = SyntheticMarketMaker()
        mm._last_mid = 50000
        mm.process_fill("buy", 49990, 0.01)
        self.assertGreater(mm.inventory.position, 0)
        self.assertEqual(mm.inventory.n_fills, 1)

    def test_performance(self):
        from src.execution.market_maker import SyntheticMarketMaker
        mm = SyntheticMarketMaker()
        perf = mm.get_performance(50000)
        self.assertIn("total_pnl", perf)
        self.assertIn("fill_rate", perf)

    def test_compute_features(self):
        from src.execution.market_maker import SyntheticMarketMaker
        mm = SyntheticMarketMaker()
        features = mm.compute_features(50000)
        self.assertIn("mm_inventory_ratio", features)
        self.assertIn("mm_is_active", features)


# ================================================================
# Test 10: RL Portfolio Manager
# ================================================================
class TestRLPortfolio(unittest.TestCase):
    def test_import(self):
        from src.risk.rl_portfolio import RLPortfolioManager
        mgr = RLPortfolioManager(n_assets=3)
        self.assertIsNotNone(mgr)

    def test_get_weights_untrained(self):
        from src.risk.rl_portfolio import RLPortfolioManager
        mgr = RLPortfolioManager(n_assets=3, lookback=10)
        returns = np.random.randn(10, 3) * 0.01
        vols = np.std(returns, axis=0)
        kelly = np.array([0.1, 0.15, 0.05])

        weights = mgr.get_weights(returns, vols, kelly_fractions=kelly)
        self.assertEqual(len(weights), 3)
        self.assertLessEqual(weights.sum(), 1.0 + 1e-6)
        self.assertTrue(np.all(weights >= 0))

    def test_ppo_agent(self):
        from src.risk.rl_portfolio import PPOAgent
        agent = PPOAgent(state_dim=10, n_assets=3)
        state = np.random.randn(10)
        weights, log_prob, value = agent.act(state)
        self.assertEqual(len(weights), 3)
        self.assertLessEqual(weights.sum(), 1.0 + 1e-6)

    def test_train_from_history(self):
        from src.risk.rl_portfolio import RLPortfolioManager
        mgr = RLPortfolioManager(n_assets=3, lookback=10, min_training_episodes=10)

        # Create synthetic price history
        n = 100
        prices = pd.DataFrame({
            "Asset1": np.random.randn(n).cumsum() + 100,
            "Asset2": np.random.randn(n).cumsum() + 50,
            "Asset3": np.random.randn(n).cumsum() + 200,
        })

        result = mgr.train_from_history(prices)
        self.assertIn("training_episodes", result)
        self.assertGreater(result["training_episodes"], 0)
        self.assertIn("final_portfolio_value", result)

    def test_reward_shaping(self):
        from src.risk.rl_portfolio import RLPortfolioManager
        mgr = RLPortfolioManager(n_assets=2)
        r1 = mgr.compute_reward(0.02)
        self.assertGreater(r1, 0)
        r2 = mgr.compute_reward(-0.10)
        self.assertLess(r2, 0)

    def test_rl_state_to_vector(self):
        from src.risk.rl_portfolio import RLState
        state = RLState(
            current_weights=np.array([0.5, 0.3, 0.2]),
            portfolio_value=1.05,
            drawdown=0.02,
            recent_returns=np.random.randn(10, 3),
            volatilities=np.array([0.01, 0.02, 0.015]),
            regime=1,
            kelly_fractions=np.array([0.1, 0.15, 0.05]),
        )
        vec = state.to_vector()
        self.assertIsInstance(vec, np.ndarray)
        self.assertGreater(len(vec), 5)


# ================================================================
# Test 11: Integration - main.py imports
# ================================================================
class TestMainIntegration(unittest.TestCase):
    def test_all_v3_imports(self):
        """Verify all V3 modules can be imported."""
        from src.data.thegraph import TheGraphFetcher
        from src.sentiment.engine import SentimentEngine
        from src.liquidation.multichain import MultiChainLiquidationOracle
        from src.regime.transformer import TransformerRegimePredictor
        from src.alpha_genome.meta_evolution import MetaEvolutionEngine
        from src.liquidation.predictive import LiquidationTimingPredictor
        from src.arbitrage.funding import FundingArbEngine
        from src.execution.mev import MEVExecutionEngine
        from src.execution.market_maker import SyntheticMarketMaker, MMConfig
        from src.risk.rl_portfolio import RLPortfolioManager
        self.assertTrue(True)  # If we got here, all imports work

    def test_config_loads(self):
        """Verify updated config loads correctly."""
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check new sections exist
        self.assertIn("sentiment", config)
        self.assertIn("multichain", config)
        self.assertIn("transformer_regime", config)
        self.assertIn("meta_evolution", config)
        self.assertIn("predictive_liquidation", config)
        self.assertIn("funding_arb", config)
        self.assertIn("mev_execution", config)
        self.assertIn("market_maker", config)
        self.assertIn("rl_portfolio", config)

        # Verify key values
        self.assertEqual(config["funding_arb"]["min_spread_8h_pct"], 0.01)
        self.assertEqual(config["market_maker"]["base_spread_bps"], 20)
        self.assertEqual(config["rl_portfolio"]["n_assets"], 5)


# ================================================================
# Run
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SignalForge V3.0 — Advanced Intelligence Module Tests")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestTheGraphFetcher,
        TestSentimentEngine,
        TestMultiChainOracle,
        TestTransformerRegime,
        TestMetaEvolution,
        TestPredictiveLiquidation,
        TestFundingArb,
        TestMEVExecution,
        TestMarketMaker,
        TestRLPortfolio,
        TestMainIntegration,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'=' * 60}")
    print(f"Total: {result.testsRun} | Pass: {result.testsRun - len(result.failures) - len(result.errors)} | "
          f"Fail: {len(result.failures)} | Error: {len(result.errors)}")
    print(f"{'=' * 60}")

    sys.exit(0 if result.wasSuccessful() else 1)

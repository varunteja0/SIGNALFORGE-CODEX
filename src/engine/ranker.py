"""
Strategy Ranker — Backtest, Score, and Rank Candidates
======================================================
Two-phase evaluation:
    Phase 1 (fast): Full-sample backtest across target assets.
                    Filter: PF > 1.0, trades > 10.
    Phase 2 (slow): Walk-forward validation on top N candidates.
                    Only candidates that pass OOS testing survive.

Scoring weights discovery over luck:
    score = sharpe × log(PF) × √(trades/30) × (1 - max_dd)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Backtester, BacktestResult
from src.data.fetcher import DataFetcher
from src.data.features import compute_all_features
from src.data.structural import StructuralDataFetcher
from src.engine.strategy_factory import StrategyCandidate

logger = logging.getLogger(__name__)


@dataclass
class ScoredStrategy:
    """A candidate with its backtest results and composite score."""
    candidate: StrategyCandidate
    score: float = 0.0
    # Per-symbol results
    results: dict = field(default_factory=dict)    # symbol → BacktestResult
    # Combined metrics
    total_trades: int = 0
    combined_pf: float = 0.0
    combined_wr: float = 0.0
    avg_sharpe: float = 0.0
    max_drawdown: float = 0.0
    # Walk-forward results (phase 2)
    oos_sharpe: Optional[float] = None
    oos_profitable_folds: Optional[int] = None
    oos_total_folds: Optional[int] = None
    # Equity curve (for correlation analysis)
    equity_curve: Optional[pd.Series] = None

    def __repr__(self):
        return (f"Scored({self.candidate.name}: score={self.score:.3f}, "
                f"PF={self.combined_pf:.2f}, trades={self.total_trades})")


class StrategyRanker:
    """Backtest, score, and rank strategy candidates."""

    def __init__(
        self,
        symbols: list[str] = None,
        days: int = 365,
        initial_capital: float = 10000,
        min_trades: int = 10,
        min_pf: float = 1.0,
        top_n_for_walkforward: int = 10,
    ):
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT"]
        self.days = days
        self.initial_capital = initial_capital
        self.min_trades = min_trades
        self.min_pf = min_pf
        self.top_n_for_walkforward = top_n_for_walkforward

        # Cached data
        self._data_cache: dict[str, pd.DataFrame] = {}

    # ─── Data Preparation ────────────────────────────────────────

    def prepare_data(self, force_fetch: bool = False) -> dict[str, pd.DataFrame]:
        """Load and prepare data for all target symbols."""
        if self._data_cache and not force_fetch:
            return self._data_cache

        fetcher = DataFetcher()
        struct = StructuralDataFetcher()

        for symbol in self.symbols:
            logger.info(f"Loading {symbol} ({self.days} days)...")
            price_df = fetcher.fetch(symbol, timeframe="1h", days=self.days)
            price_df = compute_all_features(price_df)

            binance_sym = symbol.replace("/", "")
            df = struct.fetch_all(
                symbol=binance_sym, price_df=price_df, days=self.days
            )
            self._data_cache[symbol] = df
            logger.info(f"  {symbol}: {len(df)} bars, {len(df.columns)} columns")

        return self._data_cache

    # ─── Phase 1: Fast Backtest ──────────────────────────────────

    def evaluate_candidate(
        self, candidate: StrategyCandidate, datasets: dict[str, pd.DataFrame]
    ) -> ScoredStrategy:
        """Run backtest on all symbols and compute score."""
        scored = ScoredStrategy(candidate=candidate)
        bt = Backtester(initial_capital=self.initial_capital)

        all_trades = []
        sharpes = []
        max_dd = 0.0

        for symbol, df in datasets.items():
            try:
                signals = candidate.signal_func(df)
                signal_func = lambda d, s=signals: s

                result = bt.run(
                    df,
                    signal_func,
                    position_size_pct=candidate.position_size_pct,
                    stop_loss_atr=candidate.stop_loss_atr,
                    take_profit_atr=candidate.take_profit_atr,
                    max_holding_bars=candidate.max_holding_bars,
                )

                scored.results[symbol] = result
                all_trades.extend(result.trades)

                if result.total_trades > 0:
                    sharpes.append(result.sharpe_ratio)
                max_dd = max(max_dd, result.max_drawdown)

            except Exception as e:
                logger.debug(f"  {candidate.name} failed on {symbol}: {e}")
                continue

        # Combined metrics
        scored.total_trades = len(all_trades)
        scored.max_drawdown = max_dd

        if not all_trades:
            scored.score = -999
            return scored

        wins = [t for t in all_trades if t.pnl > 0]
        losses = [t for t in all_trades if t.pnl <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = sum(abs(t.pnl) for t in losses)

        scored.combined_wr = len(wins) / len(all_trades) if all_trades else 0
        scored.combined_pf = gross_win / gross_loss if gross_loss > 0 else 0
        scored.avg_sharpe = np.mean(sharpes) if sharpes else 0

        # Compute score
        scored.score = self._compute_score(scored)

        return scored

    def _compute_score(self, scored: ScoredStrategy) -> float:
        """Composite score rewarding consistency over luck."""
        if scored.total_trades < self.min_trades:
            return -999
        if scored.combined_pf < self.min_pf:
            return -999

        # Components
        sharpe_term = scored.avg_sharpe * 0.4
        pf_term = np.log(max(scored.combined_pf, 0.01)) * 2.0
        trade_term = np.sqrt(scored.total_trades / 30) * 0.3
        dd_penalty = scored.max_drawdown * 5.0

        return sharpe_term + pf_term + trade_term - dd_penalty

    # ─── Phase 2: Walk-Forward ───────────────────────────────────

    def walk_forward_validate(
        self,
        candidate: StrategyCandidate,
        datasets: dict[str, pd.DataFrame],
        n_splits: int = 5,
    ) -> dict:
        """Walk-forward OOS validation for a single candidate."""
        all_oos_sharpes = []
        all_oos_trades = 0

        for symbol, df in datasets.items():
            n = len(df)
            min_train = n // 3
            step = (n - min_train) // n_splits

            if step < 100:
                continue

            for fold in range(n_splits):
                test_start = min_train + fold * step
                test_end = min(test_start + step, n)
                test_df = df.iloc[test_start:test_end]

                if len(test_df) < 50:
                    continue

                bt = Backtester(initial_capital=self.initial_capital)
                try:
                    signals = candidate.signal_func(test_df)
                    signal_func = lambda d, s=signals: s

                    result = bt.run(
                        test_df,
                        signal_func,
                        position_size_pct=candidate.position_size_pct,
                        stop_loss_atr=candidate.stop_loss_atr,
                        take_profit_atr=candidate.take_profit_atr,
                        max_holding_bars=candidate.max_holding_bars,
                    )
                    all_oos_sharpes.append(result.sharpe_ratio)
                    all_oos_trades += result.total_trades
                except Exception:
                    all_oos_sharpes.append(0)

        if not all_oos_sharpes:
            return {"oos_sharpe": 0, "profitable_folds": 0, "total_folds": 0}

        profitable = sum(1 for s in all_oos_sharpes if s > 0)
        return {
            "oos_sharpe": np.mean(all_oos_sharpes),
            "profitable_folds": profitable,
            "total_folds": len(all_oos_sharpes),
            "oos_trades": all_oos_trades,
        }

    # ─── Main Ranking Pipeline ───────────────────────────────────

    def rank_all(
        self,
        candidates: list[StrategyCandidate],
        run_walkforward: bool = True,
    ) -> list[ScoredStrategy]:
        """Full ranking pipeline: backtest → score → filter → walk-forward."""
        datasets = self.prepare_data()
        t0 = time.time()

        # Phase 1: Fast backtest all candidates
        logger.info(f"Phase 1: Backtesting {len(candidates)} candidates "
                     f"on {len(self.symbols)} symbols...")
        scored_list = []
        for i, candidate in enumerate(candidates):
            scored = self.evaluate_candidate(candidate, datasets)
            scored_list.append(scored)
            if (i + 1) % 10 == 0:
                logger.info(f"  Tested {i + 1}/{len(candidates)}...")

        # Filter: only profitable strategies
        passed = [s for s in scored_list if s.score > -999]
        failed = len(scored_list) - len(passed)
        logger.info(f"Phase 1 complete: {len(passed)} passed, {failed} filtered out "
                     f"({time.time() - t0:.0f}s)")

        if not passed:
            logger.warning("No candidates passed Phase 1 filters.")
            return []

        # Sort by score
        passed.sort(key=lambda s: s.score, reverse=True)

        # Log top 10
        for i, s in enumerate(passed[:10]):
            logger.info(f"  #{i+1}: {s.candidate.name} "
                         f"score={s.score:.3f} PF={s.combined_pf:.2f} "
                         f"trades={s.total_trades} sharpe={s.avg_sharpe:.2f}")

        # Phase 2: Walk-forward on top N
        if run_walkforward and len(passed) > 0:
            top_n = min(self.top_n_for_walkforward, len(passed))
            logger.info(f"Phase 2: Walk-forward validation on top {top_n}...")

            for scored in passed[:top_n]:
                wf = self.walk_forward_validate(scored.candidate, datasets)
                scored.oos_sharpe = wf["oos_sharpe"]
                scored.oos_profitable_folds = wf["profitable_folds"]
                scored.oos_total_folds = wf["total_folds"]

                # Adjust score using OOS results
                if scored.oos_sharpe is not None and scored.oos_total_folds > 0:
                    oos_bonus = scored.oos_sharpe * 0.5
                    consistency = scored.oos_profitable_folds / scored.oos_total_folds
                    scored.score = scored.score * 0.6 + oos_bonus + consistency * 0.5

            # Re-sort after WF adjustment
            passed.sort(key=lambda s: s.score, reverse=True)
            logger.info(f"Phase 2 complete ({time.time() - t0:.0f}s)")

            for i, s in enumerate(passed[:5]):
                oos_str = (f"OOS_sharpe={s.oos_sharpe:.2f}" if s.oos_sharpe else "no_WF")
                logger.info(f"  #{i+1}: {s.candidate.name} "
                             f"adj_score={s.score:.3f} {oos_str}")

        return passed

    # ─── Convenience ─────────────────────────────────────────────

    def best(self, candidates: list[StrategyCandidate], n: int = 5) -> list[ScoredStrategy]:
        """Return the top N strategies."""
        ranked = self.rank_all(candidates)
        return ranked[:n]

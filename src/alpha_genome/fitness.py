"""
Alpha Genome — Fitness Evaluation with Walk-Forward Validation
===============================================================
The guard against overfitting. Every evolved strategy must prove
it works on UNSEEN data via expanding-window walk-forward testing.

Also applies:
    - Statistical significance testing (t-test on returns)
    - Multiple hypothesis correction (Bonferroni)
    - Minimum sample size requirements
    - Transaction cost penalties
    - Parsimony pressure (simpler trees preferred)
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.alpha_genome.gene import Node

logger = logging.getLogger(__name__)


@dataclass
class FitnessResult:
    """Complete fitness evaluation of an evolved strategy."""
    # Core metrics (all computed on OUT-OF-SAMPLE data only)
    oos_sharpe: float = 0.0           # Walk-forward out-of-sample Sharpe
    oos_sortino: float = 0.0
    oos_calmar: float = 0.0
    oos_win_rate: float = 0.0
    oos_profit_factor: float = 0.0
    oos_avg_return: float = 0.0
    oos_max_drawdown: float = 0.0
    total_trades: int = 0

    # Validation
    is_significant: bool = False       # p < 0.05 after Bonferroni correction
    p_value: float = 1.0
    consistency: float = 0.0          # % of walk-forward folds that were profitable
    in_sample_sharpe: float = 0.0     # For detecting overfitting
    overfit_ratio: float = 0.0        # IS_sharpe / OOS_sharpe — close to 1.0 is good

    # Complexity penalty
    tree_depth: int = 0
    tree_size: int = 0
    parsimony_penalty: float = 0.0

    # Combined fitness score
    fitness: float = 0.0
    is_valid: bool = False

    def __repr__(self):
        return (
            f"Fitness(score={self.fitness:.4f} OOS_Sharpe={self.oos_sharpe:.2f} "
            f"WR={self.oos_win_rate:.1%} PF={self.oos_profit_factor:.2f} "
            f"trades={self.total_trades} p={self.p_value:.4f} "
            f"consistent={self.consistency:.0%} valid={self.is_valid})"
        )


class FitnessEvaluator:
    """Evaluates evolved strategy trees with rigorous walk-forward validation."""

    def __init__(
        self,
        walk_forward_splits: int = 5,
        min_trades_per_fold: int = 3,
        min_total_trades: int = 20,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        significance_level: float = 0.05,
        population_size: int = 200,     # For Bonferroni correction
        holding_period: int = 24,        # Bars to hold each position (24h for hourly)
        parsimony_coeff: float = 0.001,  # Penalty per tree node
        max_turnover_pct: float = 0.6,   # Max fraction of bars with positions
    ):
        self.walk_forward_splits = walk_forward_splits
        self.min_trades_per_fold = min_trades_per_fold
        self.min_total_trades = min_total_trades
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.significance_level = significance_level
        self.population_size = population_size
        self.holding_period = holding_period
        self.parsimony_coeff = parsimony_coeff
        self.max_turnover_pct = max_turnover_pct

    def evaluate(self, tree: Node, df: pd.DataFrame) -> FitnessResult:
        """Full fitness evaluation with walk-forward validation.

        Steps:
        1. Split data into expanding train/test windows
        2. On each fold: generate signals, compute returns on TEST data only
        3. Aggregate OOS metrics across all folds
        4. Apply statistical tests and parsimony penalty
        5. Return combined fitness score
        """
        result = FitnessResult()
        result.tree_depth = tree.depth()
        result.tree_size = tree.size()

        n = len(df)
        min_fold_size = 100
        fold_size = n // (self.walk_forward_splits + 1)

        if fold_size < min_fold_size:
            result.fitness = -100.0
            return result

        # Walk-forward: expanding window
        all_oos_returns = []
        fold_sharpes = []
        in_sample_returns = []

        for i in range(self.walk_forward_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end
            test_end = min(test_start + fold_size, n)

            if test_end - test_start < min_fold_size:
                break

            test_df = df.iloc[test_start:test_end].copy()
            train_df = df.iloc[:train_end].copy()

            # Generate signals on test data using the tree
            try:
                oos_signals = tree.evaluate(test_df)
                is_signals = tree.evaluate(train_df)
            except Exception:
                result.fitness = -100.0
                return result

            # Compute returns with holding period
            oos_rets = self._signal_returns(oos_signals, test_df)
            is_rets = self._signal_returns(is_signals, train_df)

            if len(oos_rets) < self.min_trades_per_fold:
                continue

            all_oos_returns.append(oos_rets)
            in_sample_returns.append(is_rets)

            # Per-fold Sharpe
            if oos_rets.std() > 1e-10:
                fold_sharpe = oos_rets.mean() / oos_rets.std() * np.sqrt(252 * 24)
                fold_sharpes.append(fold_sharpe)

        # Need at least 2 valid folds
        if len(all_oos_returns) < 2:
            result.fitness = -100.0
            return result

        # Aggregate all OOS returns
        oos_combined = pd.concat(all_oos_returns)
        is_combined = pd.concat(in_sample_returns) if in_sample_returns else pd.Series(dtype=float)
        result.total_trades = len(oos_combined)

        if result.total_trades < self.min_total_trades:
            result.fitness = -50.0
            return result

        # Core OOS metrics
        result.oos_avg_return = float(oos_combined.mean())
        oos_std = float(oos_combined.std())

        # Annualize correctly: returns are at holding_period-bar intervals,
        # not hourly.  For hourly data there are 252*24 = 6048 bars/year,
        # giving 6048 / holding_period trade-windows/year.
        ann_factor = np.sqrt(252 * 24 / self.holding_period)

        if oos_std > 1e-10:
            result.oos_sharpe = float(
                oos_combined.mean() / oos_std * ann_factor
            )

        # Sortino (downside deviation only)
        downside = oos_combined[oos_combined < 0]
        if len(downside) > 0 and downside.std() > 1e-10:
            result.oos_sortino = float(
                oos_combined.mean() / downside.std() * ann_factor
            )

        # Win rate
        result.oos_win_rate = float((oos_combined > 0).mean())

        # Profit factor
        gross_profit = float(oos_combined[oos_combined > 0].sum())
        gross_loss = float(abs(oos_combined[oos_combined < 0].sum()))
        if gross_loss > 1e-10:
            result.oos_profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            result.oos_profit_factor = 10.0  # Cap

        # Max drawdown
        equity = (1 + oos_combined).cumprod()
        peak = equity.cummax()
        dd = (equity - peak) / (peak + 1e-10)
        result.oos_max_drawdown = float(abs(dd.min()))

        # Calmar
        if result.oos_max_drawdown > 1e-10:
            annualized_ret = result.oos_avg_return * 252 * 24
            result.oos_calmar = annualized_ret / result.oos_max_drawdown

        # Consistency: what fraction of folds had positive Sharpe?
        if fold_sharpes:
            result.consistency = sum(1 for s in fold_sharpes if s > 0) / len(fold_sharpes)

        # In-sample Sharpe (for overfit detection)
        if len(is_combined) > 0 and is_combined.std() > 1e-10:
            result.in_sample_sharpe = float(
                is_combined.mean() / is_combined.std() * np.sqrt(252 * 24)
            )

        # Overfit ratio: how much does in-sample overstate?
        if result.oos_sharpe > 0.01:
            result.overfit_ratio = result.in_sample_sharpe / result.oos_sharpe

        # Statistical significance: t-test that mean return > 0
        if result.total_trades >= 20:
            t_stat, p_val = stats.ttest_1samp(oos_combined, 0)
            # One-sided test (we want positive returns)
            p_val_one_sided = p_val / 2 if t_stat > 0 else 1.0
            # Bonferroni correction for multiple hypotheses
            corrected_p = min(1.0, p_val_one_sided * self.population_size)
            result.p_value = float(corrected_p)
            result.is_significant = corrected_p < self.significance_level

        # Parsimony penalty (prefer simpler trees)
        result.parsimony_penalty = self.parsimony_coeff * result.tree_size

        # Combined fitness score
        result.fitness = self._compute_fitness(result)
        result.is_valid = (
            result.fitness > 0.0
            and result.oos_sharpe > 0.5
            and result.oos_profit_factor > 1.0
            and result.oos_avg_return > 0
            and result.total_trades >= max(10, self.min_total_trades // 2)
            and result.consistency >= 0.5
        )

        return result

    def _signal_returns(self, signals: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Convert position signals to realized returns with costs.

        The signal is a POSITION INDICATOR (+1 = long, -1 = short), not a
        trade trigger.  We sample non-overlapping holding windows:
        every `holding_period` bars we record the return earned by
        following that bar's signal direction over the next window.

        This correctly handles:
        - Constant signals (always long or always short)
        - Signals that flip frequently
        - Realistic transaction costs on each window
        """
        close = df["close"]
        hp = self.holding_period
        total_cost = 2 * (self.commission_pct + self.slippage_pct)  # entry + exit

        # Discretize continuous signals → {-1, 0, +1}
        discretized = self._discretize_signals(signals)

        trade_returns = []

        # Sample non-overlapping windows every `hp` bars
        for start in range(0, len(df) - hp, hp):
            sig = discretized.iloc[start]
            if sig == 0:
                continue  # no position this window

            entry_price = close.iloc[start]
            exit_price = close.iloc[start + hp]
            if entry_price <= 0:
                continue

            ret = (exit_price / entry_price - 1) * sig - total_cost
            trade_returns.append(ret)

        if not trade_returns:
            return pd.Series(dtype=float)

        return pd.Series(trade_returns, dtype=float)

    def _discretize_signals(self, signals: pd.Series) -> pd.Series:
        """Discretize continuous signals to {-1, 0, +1} consistently.

        This is the SINGLE source of truth for signal discretization.
        Both fitness evaluation and backtesting must use this method.

        Uses z-score thresholding at ±0.5 for variable signals,
        or sign() for constant signals (e.g. ComparisonNode roots).
        """
        sig_std = signals.std()

        if sig_std < 1e-10:
            # Constant signal — use sign directly
            return signals.apply(np.sign)

        # Variable signal — z-score then threshold
        sig_mean = signals.mean()
        zscore = (signals - sig_mean) / sig_std
        discretized = pd.Series(0.0, index=signals.index)
        discretized[zscore > 0.5] = 1.0
        discretized[zscore < -0.5] = -1.0
        return discretized

    def _compute_fitness(self, r: FitnessResult) -> float:
        """Multi-objective fitness function.

        Rewards:
        - High OOS Sharpe ratio (primary)
        - High consistency across folds
        - Statistical significance
        - Good profit factor
        - Positive average return (actual profitability)

        Penalizes:
        - High overfit ratio (in-sample >> out-of-sample)
        - Low trade count (unreliable statistics)
        - Tree complexity (parsimony)
        - High max drawdown
        - Negative average returns (the strategy MUST make money)
        """
        if r.total_trades < max(10, self.min_total_trades // 2):
            return -1.0

        score = 0.0

        # Sharpe component (most important)
        score += r.oos_sharpe * 0.30

        # Profitability — the strategy MUST make money after costs
        if r.oos_avg_return > 0:
            # Reward proportional to avg return (capped)
            ret_bonus = min(r.oos_avg_return * 1000, 1.0)  # 0.1% avg → 1.0 bonus
            score += ret_bonus * 0.25
        else:
            # Penalize negative returns — this is the key filter
            score += r.oos_avg_return * 500  # -0.1% avg → -0.5 penalty

        # Consistency bonus
        score += r.consistency * 0.15

        # Profit factor bonus (capped, must be > 1.0 to matter)
        if r.oos_profit_factor > 1.0:
            pf_bonus = min(r.oos_profit_factor - 1.0, 2.0) / 2.0
            score += pf_bonus * 0.15
        else:
            score -= (1.0 - r.oos_profit_factor) * 0.2

        # Significance bonus
        if r.is_significant:
            score += 0.10
        else:
            score -= 0.05

        # Sortino bonus (rewards low downside vol)
        sortino_bonus = min(r.oos_sortino, 5.0) / 5.0
        score += sortino_bonus * 0.05

        # Overfit penalty — if IS >> OOS, strategy is curve-fitted
        if r.overfit_ratio > 3.0:
            score -= 0.2 * (r.overfit_ratio - 3.0)
        elif r.overfit_ratio > 2.0:
            score -= 0.1

        # Drawdown penalty
        if r.oos_max_drawdown > 0.15:
            score -= 0.15 * (r.oos_max_drawdown - 0.15)

        # Parsimony
        score -= r.parsimony_penalty

        return score

    def generate_backtest_signals(
        self, tree: Node, df: pd.DataFrame
    ) -> pd.Series:
        """Generate backtest-aligned trading signals from a tree.

        This is the CANONICAL signal generator. It produces the exact same
        discretized positions that the fitness evaluator uses internally.
        Both fitness evaluation and the external backtester MUST use this
        to ensure results are identical.

        Returns:
            pd.Series of {-1, 0, +1} with the same index as df.
            Positions are held for `holding_period` bars — the signal only
            changes every `holding_period` bars.
        """
        raw_signals = tree.evaluate(df)
        discretized = self._discretize_signals(raw_signals)

        # Enforce holding period: only allow signal changes every hp bars
        hp = self.holding_period
        held = pd.Series(0.0, index=df.index)

        i = 0
        while i < len(df):
            sig = discretized.iloc[i]
            # Hold this position for hp bars
            end = min(i + hp, len(df))
            held.iloc[i:end] = sig
            i = end

        return held

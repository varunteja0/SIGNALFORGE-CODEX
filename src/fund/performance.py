"""
Performance Attribution Engine — Know Exactly Where Alpha Comes From
======================================================================
Most systems track total PnL. This module decomposes returns to answer:

1. Which strategies are generating alpha? (Per-strategy Sharpe, drawdown)
2. What's the alpha decay rate? (Rolling performance windows)
3. How much is skill vs luck? (Bootstrap confidence intervals)
4. What's the optimal strategy portfolio? (Markowitz-inspired allocation)
5. What are the factor exposures? (Is "alpha" just hidden beta?)

This transforms "I made 15% this month" into:
"Strategy A contributed 8% (Sharpe 2.1, decaying), Strategy B contributed 
5% (Sharpe 1.4, stable), liquidation signals contributed 3% (episodic).
Strategy A has 40% correlation to momentum factor — partial beta, not 
pure alpha. Recommended: increase B allocation, watch A for decay."
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Detailed performance for a single strategy."""
    name: str
    strategy_type: str = ""
    
    # Returns
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    
    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade-level
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    
    # Drawdown
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    avg_drawdown_duration: float = 0.0  # bars
    
    # Confidence
    sharpe_ci_lower: float = 0.0   # 95% CI
    sharpe_ci_upper: float = 0.0
    probability_of_skill: float = 0.0  # Bootstrap p(Sharpe > 0)
    
    # Decay
    rolling_sharpe_30d: float = 0.0
    rolling_sharpe_trend: float = 0.0  # Slope of rolling Sharpe
    is_decaying: bool = False
    
    # Factor exposure
    momentum_beta: float = 0.0
    volatility_beta: float = 0.0
    market_beta: float = 0.0
    residual_alpha: float = 0.0  # Alpha after removing factor exposure


@dataclass
class PortfolioAttribution:
    """Full portfolio-level performance attribution."""
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    portfolio_sharpe: float = 0.0
    
    strategy_contributions: dict = field(default_factory=dict)  # name → % of total PnL
    strategies: list = field(default_factory=list)  # List[StrategyPerformance]
    
    optimal_weights: dict = field(default_factory=dict)  # name → recommended weight
    diversification_ratio: float = 0.0
    
    # Summary
    best_strategy: str = ""
    worst_strategy: str = ""
    most_consistent: str = ""
    decaying_strategies: list = field(default_factory=list)


class PerformanceEngine:
    """Computes deep performance attribution across all strategies.
    
    Usage:
        engine = PerformanceEngine()
        engine.record_trade("alpha_001", pnl=15, return_pct=0.015, timestamp=...)
        engine.record_trade("alpha_001", pnl=-8, return_pct=-0.008, timestamp=...)
        
        # Get individual strategy performance
        perf = engine.strategy_report("alpha_001")
        
        # Get full portfolio attribution
        attribution = engine.portfolio_report()
    """
    
    def __init__(
        self,
        bootstrap_samples: int = 1000,
        rolling_window_days: int = 30,
        annualization_factor: float = 365 * 24,  # Hourly trading
    ):
        self.bootstrap_n = bootstrap_samples
        self.rolling_window = rolling_window_days
        self.ann_factor = annualization_factor
        
        self.trades: dict[str, list] = {}  # strategy → list of trade dicts
        self.strategy_types: dict[str, str] = {}
        self.market_returns: Optional[pd.Series] = None  # For factor analysis
    
    def register_strategy(self, name: str, strategy_type: str = "alpha_genome"):
        """Register a strategy for tracking."""
        self.trades.setdefault(name, [])
        self.strategy_types[name] = strategy_type
    
    def set_market_returns(self, returns: pd.Series):
        """Set market benchmark returns for factor decomposition."""
        self.market_returns = returns
    
    def record_trade(
        self,
        strategy_name: str,
        pnl: float,
        return_pct: float,
        timestamp: Optional[float] = None,
        direction: int = 1,
        asset: str = "",
    ):
        """Record a completed trade."""
        self.trades.setdefault(strategy_name, [])
        
        import time
        self.trades[strategy_name].append({
            "pnl": pnl,
            "return_pct": return_pct,
            "timestamp": timestamp or time.time(),
            "direction": direction,
            "asset": asset,
        })
    
    def strategy_report(self, name: str) -> StrategyPerformance:
        """Generate detailed performance report for one strategy."""
        perf = StrategyPerformance(
            name=name,
            strategy_type=self.strategy_types.get(name, "unknown"),
        )
        
        trades = self.trades.get(name, [])
        if not trades:
            return perf
        
        pnls = pd.Series([t["pnl"] for t in trades])
        returns = pd.Series([t["return_pct"] for t in trades])
        
        # Basic metrics
        perf.total_trades = len(trades)
        perf.total_pnl = float(pnls.sum())
        perf.total_return_pct = float((1 + returns).prod() - 1)
        
        if len(returns) > 1:
            perf.annualized_return = float(
                (1 + perf.total_return_pct) ** (self.ann_factor / len(returns)) - 1
            )
        
        # Risk-adjusted returns
        if returns.std() > 1e-10:
            perf.sharpe_ratio = float(
                returns.mean() / returns.std() * np.sqrt(self.ann_factor)
            )
        
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 1e-10:
            perf.sortino_ratio = float(
                returns.mean() / downside.std() * np.sqrt(self.ann_factor)
            )
        
        # Trade-level stats
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        
        perf.win_rate = float(len(wins) / len(pnls)) if len(pnls) > 0 else 0
        perf.avg_win = float(wins.mean()) if len(wins) > 0 else 0
        perf.avg_loss = float(losses.mean()) if len(losses) > 0 else 0
        perf.max_win = float(wins.max()) if len(wins) > 0 else 0
        perf.max_loss = float(losses.min()) if len(losses) > 0 else 0
        
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0
        perf.profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else 10.0
        
        # Drawdown
        equity = (1 + returns).cumprod()
        peak = equity.cummax()
        dd = (equity - peak) / (peak + 1e-10)
        perf.max_drawdown_pct = float(abs(dd.min()))
        perf.current_drawdown_pct = float(abs(dd.iloc[-1])) if len(dd) > 0 else 0
        
        if perf.max_drawdown_pct > 1e-10:
            perf.calmar_ratio = perf.annualized_return / perf.max_drawdown_pct
        
        # Confidence intervals via bootstrap
        if len(returns) >= 20:
            perf.sharpe_ci_lower, perf.sharpe_ci_upper, perf.probability_of_skill = (
                self._bootstrap_sharpe(returns)
            )
        
        # Rolling Sharpe for decay detection
        if len(returns) >= 30:
            window = min(30, len(returns) // 2)
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            rolling_sharpe = (rolling_mean / (rolling_std + 1e-10)) * np.sqrt(self.ann_factor)
            rolling_sharpe = rolling_sharpe.dropna()
            
            if len(rolling_sharpe) > 5:
                perf.rolling_sharpe_30d = float(rolling_sharpe.iloc[-1])
                x = np.arange(len(rolling_sharpe))
                slope, _, _, _, _ = sp_stats.linregress(x, rolling_sharpe.values)
                perf.rolling_sharpe_trend = float(slope * len(rolling_sharpe))
                perf.is_decaying = perf.rolling_sharpe_trend < -0.5
        
        # Factor decomposition
        if self.market_returns is not None and len(returns) >= 20:
            perf.market_beta, perf.residual_alpha = self._factor_decompose(returns)
        
        return perf
    
    def portfolio_report(self) -> PortfolioAttribution:
        """Generate full portfolio attribution report."""
        attr = PortfolioAttribution()
        
        all_strategies = []
        total_pnl = 0.0
        
        for name in self.trades:
            perf = self.strategy_report(name)
            all_strategies.append(perf)
            total_pnl += perf.total_pnl
        
        attr.strategies = all_strategies
        attr.total_pnl = total_pnl
        
        if not all_strategies:
            return attr
        
        # Contribution percentage
        for s in all_strategies:
            attr.strategy_contributions[s.name] = (
                s.total_pnl / total_pnl if abs(total_pnl) > 1e-10 else 0
            )
        
        # Portfolio-level Sharpe
        all_returns = []
        for name, trades in self.trades.items():
            for t in trades:
                all_returns.append(t["return_pct"])
        
        if all_returns:
            rets = pd.Series(all_returns)
            attr.total_return_pct = float((1 + rets).prod() - 1)
            if rets.std() > 1e-10:
                attr.portfolio_sharpe = float(
                    rets.mean() / rets.std() * np.sqrt(self.ann_factor)
                )
        
        # Best/worst/most consistent
        profitable = [s for s in all_strategies if s.total_pnl > 0]
        if profitable:
            attr.best_strategy = max(profitable, key=lambda s: s.sharpe_ratio).name
            attr.most_consistent = max(profitable, key=lambda s: s.win_rate).name
        
        losing = [s for s in all_strategies if s.total_pnl <= 0]
        if losing:
            attr.worst_strategy = min(losing, key=lambda s: s.total_pnl).name
        
        # Decaying strategies
        attr.decaying_strategies = [s.name for s in all_strategies if s.is_decaying]
        
        # Optimal weights (inverse-volatility weighted)
        attr.optimal_weights = self._compute_optimal_weights(all_strategies)
        
        # Diversification ratio
        attr.diversification_ratio = self._diversification_ratio()
        
        return attr
    
    # ================================================================
    # Internal Analytics
    # ================================================================
    
    def _bootstrap_sharpe(
        self, returns: pd.Series
    ) -> tuple[float, float, float]:
        """Bootstrap confidence interval for Sharpe ratio.
        
        Returns (lower_95, upper_95, prob_skill).
        prob_skill = probability that true Sharpe > 0.
        """
        n = len(returns)
        bootstrap_sharpes = []
        
        for _ in range(self.bootstrap_n):
            sample = returns.sample(n, replace=True)
            if sample.std() > 1e-10:
                s = sample.mean() / sample.std() * np.sqrt(self.ann_factor)
                bootstrap_sharpes.append(s)
        
        if not bootstrap_sharpes:
            return 0.0, 0.0, 0.0
        
        bs = np.array(bootstrap_sharpes)
        lower = float(np.percentile(bs, 2.5))
        upper = float(np.percentile(bs, 97.5))
        prob_skill = float(np.mean(bs > 0))
        
        return lower, upper, prob_skill
    
    def _factor_decompose(
        self, strategy_returns: pd.Series
    ) -> tuple[float, float]:
        """Decompose returns into market beta and residual alpha.
        
        strategy_return = alpha + beta * market_return + epsilon
        
        Returns (beta, alpha).
        """
        if self.market_returns is None:
            return 0.0, 0.0
        
        # Align lengths
        n = min(len(strategy_returns), len(self.market_returns))
        if n < 10:
            return 0.0, 0.0
        
        y = strategy_returns.values[-n:]
        x = self.market_returns.values[-n:]
        
        slope, intercept, _, _, _ = sp_stats.linregress(x, y)
        
        # Annualize alpha
        alpha = intercept * self.ann_factor
        beta = slope
        
        return float(beta), float(alpha)
    
    def _compute_optimal_weights(
        self, strategies: list[StrategyPerformance]
    ) -> dict[str, float]:
        """Compute optimal portfolio weights using inverse-volatility.
        
        Simple but effective: weight each strategy inversely proportional
        to its volatility. Strategies with lower vol get larger allocation.
        """
        weights = {}
        total_inv_vol = 0.0
        
        for s in strategies:
            trades = self.trades.get(s.name, [])
            if len(trades) < 10:
                continue
            
            returns = pd.Series([t["return_pct"] for t in trades])
            vol = returns.std()
            
            if vol > 1e-10 and s.sharpe_ratio > 0:
                inv_vol = 1.0 / vol
                weights[s.name] = inv_vol
                total_inv_vol += inv_vol
        
        if total_inv_vol > 0:
            for name in weights:
                weights[name] /= total_inv_vol
        
        return weights
    
    def _diversification_ratio(self) -> float:
        """Compute portfolio diversification ratio.
        
        DR = (sum of individual vols * weights) / portfolio vol
        DR > 1 means diversification is helping. Higher = better.
        """
        active = [
            name for name, trades in self.trades.items()
            if len(trades) >= 10
        ]
        
        if len(active) < 2:
            return 1.0
        
        # Build return matrix
        returns_dict = {}
        for name in active:
            returns_dict[name] = [t["return_pct"] for t in self.trades[name]]
        
        # Align to same length (use last N trades)
        min_len = min(len(v) for v in returns_dict.values())
        if min_len < 10:
            return 1.0
        
        matrix = np.column_stack([
            returns_dict[name][-min_len:] for name in active
        ])
        
        # Equal-weight portfolio
        weights = np.ones(len(active)) / len(active)
        individual_vols = np.std(matrix, axis=0)
        
        cov = np.cov(matrix.T)
        portfolio_vol = np.sqrt(weights @ cov @ weights)
        
        weighted_vol_sum = np.sum(weights * individual_vols)
        
        if portfolio_vol > 1e-10:
            return float(weighted_vol_sum / portfolio_vol)
        
        return 1.0

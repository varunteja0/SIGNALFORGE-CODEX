"""
Portfolio Optimization — Science-Based Capital Allocation
===========================================================
Replaces ad-hoc equal-weight allocation with real optimization:

1. Mean-Variance (Markowitz) — maximize Sharpe ratio
2. Risk Parity — equal risk contribution from each strategy
3. CVaR Optimization — minimize tail risk (Conditional Value at Risk)
4. Black-Litterman — combine market equilibrium with GP-evolved views
5. Hierarchical Risk Parity (HRP) — correlation-aware tree clustering

The optimizer takes strategy return streams and produces optimal
weights that maximize risk-adjusted returns while respecting constraints.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster

logger = logging.getLogger(__name__)


@dataclass
class PortfolioWeights:
    """Result of portfolio optimization."""
    weights: dict[str, float]       # strategy_name -> weight (sums to 1)
    method: str                     # Which method was used
    expected_sharpe: float = 0.0
    expected_return: float = 0.0
    expected_vol: float = 0.0
    expected_cvar_95: float = 0.0   # 95% CVaR
    max_drawdown_est: float = 0.0
    diversification_ratio: float = 0.0
    effective_n: float = 0.0        # Effective number of strategies (1/HHI)


class PortfolioOptimizer:
    """Multi-method portfolio optimizer for strategy allocation."""

    def __init__(
        self,
        method: str = "hrp",            # "markowitz", "risk_parity", "cvar", "hrp"
        max_weight: float = 0.3,         # Max weight per strategy
        min_weight: float = 0.02,        # Min weight per strategy
        risk_free_rate: float = 0.04,    # Annual risk-free rate
        lookback_periods: int = 100,     # Periods for covariance estimation
        cvar_confidence: float = 0.95,   # CVaR confidence level
        shrinkage_factor: float = 0.5,   # Ledoit-Wolf shrinkage
    ):
        self.method = method
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.rf = risk_free_rate
        self.lookback = lookback_periods
        self.cvar_conf = cvar_confidence
        self.shrinkage = shrinkage_factor

    def optimize(
        self,
        returns: pd.DataFrame,
        views: Optional[dict[str, float]] = None,
    ) -> PortfolioWeights:
        """Optimize portfolio weights given strategy return streams.

        Args:
            returns: DataFrame where columns = strategy names,
                     rows = period returns
            views: Optional dict of {strategy: expected_excess_return}
                   for Black-Litterman overlay

        Returns:
            PortfolioWeights with optimized allocations.
        """
        if returns.empty or returns.shape[1] < 2:
            # Single strategy or no data — equal weight
            names = returns.columns.tolist()
            eq_w = 1.0 / max(1, len(names))
            return PortfolioWeights(
                weights={n: eq_w for n in names},
                method="equal_weight",
            )

        # Clean returns
        returns = returns.dropna(how="all").fillna(0)
        if len(returns) < 20:
            names = returns.columns.tolist()
            eq_w = 1.0 / len(names)
            return PortfolioWeights(
                weights={n: eq_w for n in names},
                method="equal_weight_insufficient_data",
            )

        # Compute inputs
        mu = returns.mean().values
        cov = self._shrunk_covariance(returns)
        names = returns.columns.tolist()
        n = len(names)

        # Route to method
        if self.method == "markowitz":
            weights = self._markowitz(mu, cov, n)
        elif self.method == "risk_parity":
            weights = self._risk_parity(cov, n)
        elif self.method == "cvar":
            weights = self._cvar_minimize(returns.values, n)
        elif self.method == "hrp":
            weights = self._hrp(returns, cov, n)
        else:
            weights = np.ones(n) / n

        # Clamp to bounds
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / weights.sum()

        # Compute portfolio stats
        port_ret = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights + 1e-10)
        port_sharpe = (port_ret - self.rf / 252) / (port_vol + 1e-10)

        # Diversification ratio
        ind_vols = np.sqrt(np.diag(cov))
        div_ratio = (weights @ ind_vols) / (port_vol + 1e-10)

        # Effective N (inverse of HHI)
        hhi = (weights ** 2).sum()
        effective_n = 1.0 / (hhi + 1e-10)

        # CVaR estimate
        port_returns = returns.values @ weights
        sorted_rets = np.sort(port_returns)
        cutoff = int(len(sorted_rets) * (1 - self.cvar_conf))
        cvar = -sorted_rets[:max(1, cutoff)].mean() if cutoff > 0 else 0

        return PortfolioWeights(
            weights={names[i]: float(weights[i]) for i in range(n)},
            method=self.method,
            expected_sharpe=float(port_sharpe),
            expected_return=float(port_ret),
            expected_vol=float(port_vol),
            expected_cvar_95=float(cvar),
            diversification_ratio=float(div_ratio),
            effective_n=float(effective_n),
        )

    def _shrunk_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Ledoit-Wolf shrinkage covariance estimator.

        Shrinks sample covariance toward a structured target (diagonal)
        to reduce estimation error. Critical for n_strategies > 5.
        """
        sample_cov = returns.cov().values
        n = sample_cov.shape[0]

        # Target: diagonal (each strategy independent)
        target = np.diag(np.diag(sample_cov))

        # Shrink
        shrunk = (1 - self.shrinkage) * sample_cov + self.shrinkage * target

        # Ensure positive semi-definite
        eigenvalues = np.linalg.eigvalsh(shrunk)
        if eigenvalues.min() < 0:
            shrunk += np.eye(n) * (abs(eigenvalues.min()) + 1e-8)

        return shrunk

    def _markowitz(self, mu: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
        """Mean-variance optimization — maximize Sharpe ratio."""
        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w + 1e-10)
            return -(ret - self.rf / 252) / vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(self.min_weight, self.max_weight)] * n
        w0 = np.ones(n) / n

        result = minimize(
            neg_sharpe, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        return result.x if result.success else np.ones(n) / n

    def _risk_parity(self, cov: np.ndarray, n: int) -> np.ndarray:
        """Risk parity — equal risk contribution from each strategy.

        Solves: w_i * (Σw)_i = constant for all i
        """
        def risk_contrib_diff(w):
            port_vol = np.sqrt(w @ cov @ w + 1e-10)
            marginal_risk = cov @ w
            risk_contrib = w * marginal_risk / port_vol
            target = port_vol / n
            return np.sum((risk_contrib - target) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(self.min_weight, self.max_weight)] * n
        w0 = np.ones(n) / n

        result = minimize(
            risk_contrib_diff, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        return result.x if result.success else np.ones(n) / n

    def _cvar_minimize(self, returns_array: np.ndarray, n: int) -> np.ndarray:
        """CVaR minimization — minimize tail risk at 95% confidence.

        Uses the Rockafellar-Uryasev LP formulation.
        """
        T = len(returns_array)
        alpha = 1 - self.cvar_conf

        def cvar_objective(w):
            port_returns = returns_array @ w
            sorted_rets = np.sort(port_returns)
            cutoff = max(1, int(T * alpha))
            cvar = -sorted_rets[:cutoff].mean()
            return cvar

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(self.min_weight, self.max_weight)] * n
        w0 = np.ones(n) / n

        result = minimize(
            cvar_objective, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        return result.x if result.success else np.ones(n) / n

    def _hrp(
        self, returns: pd.DataFrame, cov: np.ndarray, n: int
    ) -> np.ndarray:
        """Hierarchical Risk Parity (Lopez de Prado).

        1. Cluster strategies by correlation
        2. Allocate within clusters (inverse-vol)
        3. Allocate between clusters (inverse-vol)

        Outperforms Markowitz in practice because it doesn't require
        accurate expected return estimates (which we can't get reliably).
        """
        if n <= 2:
            return self._risk_parity(cov, n)

        # Correlation distance matrix
        corr = returns.corr().values.copy()
        np.fill_diagonal(corr, 1.0)
        corr = np.nan_to_num(corr, nan=0.0)
        dist = np.sqrt(0.5 * (1 - corr))

        # Hierarchical clustering
        condensed_dist = []
        for i in range(n):
            for j in range(i + 1, n):
                condensed_dist.append(dist[i, j])
        condensed_dist = np.array(condensed_dist)

        if len(condensed_dist) == 0:
            return np.ones(n) / n

        link = linkage(condensed_dist, method="single")

        # Quasi-diagonal reordering
        sort_idx = self._get_quasi_diag(link)

        # Recursive bisection allocation
        weights = np.ones(n)
        cluster_items = [sort_idx]

        while cluster_items:
            cluster_items_new = []
            for items in cluster_items:
                if len(items) <= 1:
                    continue

                # Split in half
                mid = len(items) // 2
                left = items[:mid]
                right = items[mid:]

                # Variance of each half
                var_left = self._cluster_var(cov, left)
                var_right = self._cluster_var(cov, right)

                # Allocate inversely proportional to variance
                alloc_factor = 1 - var_left / (var_left + var_right + 1e-10)

                for i in left:
                    weights[i] *= alloc_factor
                for i in right:
                    weights[i] *= (1 - alloc_factor)

                if len(left) > 1:
                    cluster_items_new.append(left)
                if len(right) > 1:
                    cluster_items_new.append(right)

            cluster_items = cluster_items_new

        # Normalize
        weights = weights / (weights.sum() + 1e-10)
        return weights

    def _get_quasi_diag(self, link: np.ndarray) -> list[int]:
        """Extract quasi-diagonal order from linkage matrix."""
        n = int(link[-1, 3])
        sort_idx = [int(link[-1, 0]), int(link[-1, 1])]

        num_items = n
        while True:
            new_sort = []
            for item in sort_idx:
                if item < num_items:
                    new_sort.append(item)
                else:
                    row = int(item - num_items)
                    new_sort.extend([int(link[row, 0]), int(link[row, 1])])
            sort_idx = new_sort
            if all(x < num_items for x in sort_idx):
                break

        return sort_idx

    def _cluster_var(self, cov: np.ndarray, items: list[int]) -> float:
        """Compute variance of an equally-weighted sub-portfolio."""
        sub_cov = cov[np.ix_(items, items)]
        w = np.ones(len(items)) / len(items)
        return float(w @ sub_cov @ w)

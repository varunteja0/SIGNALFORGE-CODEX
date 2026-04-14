"""
Strategy Allocator — Portfolio-Level Capital Allocation
=======================================================
Takes the top-ranked strategies and allocates capital across them using:
    1. Inverse-volatility weighting (less volatile → more capital)
    2. Decorrelation bonus (uncorrelated → more capital)
    3. Concentration limits (no single strategy > 40%)
    4. Minimum weight floor (< 5% → drop it)

Two modes:
    allocate()         — strategy-level weights (original)
    allocate_granular() — strategy × asset matrix, Sharpe-based

Input: list of ScoredStrategy with equity curves
Output: dict of {strategy_name: weight} or AssetAllocation matrix
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.engine.ranker import ScoredStrategy

logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """Portfolio allocation output."""
    weights: dict = field(default_factory=dict)       # name → weight (0-1)
    strategies: list = field(default_factory=list)     # ScoredStrategy objects
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    expected_sharpe: float = 0.0
    n_strategies: int = 0


@dataclass
class AssetAllocation:
    """Granular strategy × asset allocation matrix."""
    # {strategy_name: {symbol: weight}}
    matrix: dict = field(default_factory=dict)
    # Flat strategy-level weights (sum of asset weights per strategy)
    strategy_weights: dict = field(default_factory=dict)
    # Flat asset-level weights (sum of strategy weights per asset)
    asset_weights: dict = field(default_factory=dict)
    # Per-cell scores used for weighting
    scores: dict = field(default_factory=dict)
    n_strategies: int = 0
    n_assets: int = 0
    expected_sharpe: float = 0.0


class StrategyAllocator:
    """Allocate capital across uncorrelated strategies."""

    def __init__(
        self,
        max_strategies: int = 5,
        max_weight: float = 0.40,
        min_weight: float = 0.05,
        max_correlation: float = 0.70,
    ):
        self.max_strategies = max_strategies
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.max_correlation = max_correlation

    def allocate(self, scored_strategies: list[ScoredStrategy]) -> AllocationResult:
        """Compute portfolio weights for top strategies.

        Steps:
            1. Build equity curves from backtest results
            2. Compute return correlation matrix
            3. Filter highly correlated strategies (keep better one)
            4. Inverse-vol weighting
            5. Apply concentration limits
        """
        result = AllocationResult()

        if not scored_strategies:
            return result

        # Step 1: Build equity curves
        equity_curves = {}
        for scored in scored_strategies:
            curve = self._build_combined_equity(scored)
            if curve is not None and len(curve) > 50:
                equity_curves[scored.candidate.name] = curve

        if not equity_curves:
            logger.warning("No valid equity curves — cannot allocate.")
            return result

        # Step 2: Correlation matrix
        returns_df = pd.DataFrame({
            name: curve.pct_change().fillna(0)
            for name, curve in equity_curves.items()
        })

        # Align indices
        returns_df = returns_df.dropna()
        if len(returns_df) < 50:
            logger.warning("Insufficient overlapping data for correlation.")
            return result

        corr = returns_df.corr()
        result.correlation_matrix = corr

        # Step 3: Filter correlated strategies (greedy — keep higher score)
        name_to_scored = {s.candidate.name: s for s in scored_strategies
                          if s.candidate.name in equity_curves}
        selected = self._filter_correlated(list(name_to_scored.keys()), corr, name_to_scored)

        # Limit to max
        selected = selected[:self.max_strategies]

        if not selected:
            return result

        # Step 4: Inverse-vol weighting
        vols = {}
        for name in selected:
            ret = returns_df[name]
            vol = ret.std()
            vols[name] = vol if vol > 0 else 1e-6

        inv_vol = {name: 1.0 / v for name, v in vols.items()}
        total_inv = sum(inv_vol.values())
        weights = {name: iv / total_inv for name, iv in inv_vol.items()}

        # Step 5: Apply limits
        weights = self._apply_limits(weights)

        result.weights = weights
        result.strategies = [name_to_scored[n] for n in weights if n in name_to_scored]
        result.n_strategies = len(weights)

        # Expected portfolio Sharpe (approximate)
        if len(selected) > 1:
            w = np.array([weights[n] for n in selected if n in weights])
            rets = returns_df[[n for n in selected if n in weights]]
            port_ret = (rets * w).sum(axis=1)
            if port_ret.std() > 0:
                result.expected_sharpe = (
                    port_ret.mean() / port_ret.std() * np.sqrt(252 * 24)
                )

        logger.info(f"Allocated {result.n_strategies} strategies:")
        for name, w in sorted(weights.items(), key=lambda x: -x[1]):
            logger.info(f"  {name}: {w:.1%}")

        return result

    def _build_combined_equity(self, scored: ScoredStrategy) -> pd.Series:
        """Build combined equity curve from per-symbol results."""
        curves = []
        for symbol, result in scored.results.items():
            if hasattr(result, "equity_curve") and len(result.equity_curve) > 0:
                curves.append(result.equity_curve)

        if not curves:
            return None

        # Combine: average of normalized curves
        normalized = []
        for c in curves:
            normalized.append(c / c.iloc[0])

        # Align to common index
        combined = pd.concat(normalized, axis=1).mean(axis=1)
        return combined

    def _filter_correlated(
        self,
        names: list[str],
        corr: pd.DataFrame,
        name_to_scored: dict,
    ) -> list[str]:
        """Remove highly correlated strategies (keep higher score)."""
        # Sort by score descending
        sorted_names = sorted(
            names, key=lambda n: name_to_scored[n].score, reverse=True
        )

        selected = []
        for name in sorted_names:
            # Check correlation with already selected
            too_correlated = False
            for existing in selected:
                if name in corr.columns and existing in corr.columns:
                    c = abs(corr.loc[name, existing])
                    if c > self.max_correlation:
                        too_correlated = True
                        logger.debug(f"  Dropping {name}: corr={c:.2f} with {existing}")
                        break

            if not too_correlated:
                selected.append(name)

        return selected

    def _apply_limits(self, weights: dict) -> dict:
        """Enforce min/max weight limits."""
        # Drop below minimum
        weights = {k: v for k, v in weights.items() if v >= self.min_weight}

        if not weights:
            return {}

        # Cap at maximum
        capped = {k: min(v, self.max_weight) for k, v in weights.items()}

        # Renormalize
        total = sum(capped.values())
        if total > 0:
            capped = {k: v / total for k, v in capped.items()}

        return capped

    # ─── Granular: Strategy × Asset Matrix ───────────────────────

    def allocate_granular(
        self,
        scored_strategies: list[ScoredStrategy],
        score_metric: str = "sharpe",
    ) -> AssetAllocation:
        """Build strategy × asset weight matrix using per-asset scores.

        Unlike allocate() which gives one weight per strategy, this gives
        a weight for each (strategy, asset) pair. A strategy that's great
        on ETH but weak on BTC gets more ETH allocation.

        Score metrics:
            "sharpe"  — weight by per-asset Sharpe ratio
            "pf"      — weight by per-asset profit factor
            "trades"  — weight by trade count (more data = more trust)
            "combined"— sharpe × sqrt(trades) (default blend)
        """
        result = AssetAllocation()

        if not scored_strategies:
            return result

        # Step 1: Build score matrix {strategy: {symbol: score}}
        raw_scores = {}
        for scored in scored_strategies:
            name = scored.candidate.name
            raw_scores[name] = {}
            for symbol, res in scored.results.items():
                if res.total_trades < 3:
                    continue  # Not enough trades to trust

                if score_metric == "sharpe":
                    s = max(res.sharpe_ratio, 0)
                elif score_metric == "pf":
                    s = max(res.profit_factor - 1, 0)  # Excess PF
                elif score_metric == "trades":
                    s = np.sqrt(res.total_trades)
                else:  # combined
                    sh = max(res.sharpe_ratio, 0)
                    s = sh * np.sqrt(max(res.total_trades, 1))

                if s > 0:
                    raw_scores[name][symbol] = s

        # Remove strategies with no positive scores
        raw_scores = {k: v for k, v in raw_scores.items() if v}

        if not raw_scores:
            return result

        # Step 2: Normalize within each strategy (so weights sum to 1 per strategy)
        normalized = {}
        for name, syms in raw_scores.items():
            total = sum(syms.values())
            if total > 0:
                normalized[name] = {sym: sc / total for sym, sc in syms.items()}

        # Step 3: Strategy-level weighting via inverse-vol (reuse existing logic)
        # Build pseudo equity curves per strategy
        strat_vols = {}
        name_to_scored = {s.candidate.name: s for s in scored_strategies}
        for name in normalized:
            scored = name_to_scored.get(name)
            if scored is None:
                continue
            curve = self._build_combined_equity(scored)
            if curve is not None and len(curve) > 50:
                vol = curve.pct_change().fillna(0).std()
                strat_vols[name] = vol if vol > 0 else 1e-6

        if not strat_vols:
            # Fall back to equal weight
            n = len(normalized)
            strat_weights = {name: 1.0 / n for name in normalized}
        else:
            inv_vol = {n: 1.0 / v for n, v in strat_vols.items()}
            total_iv = sum(inv_vol.values())
            strat_weights = {n: iv / total_iv for n, iv in inv_vol.items()}

        # Apply limits to strategy weights
        strat_weights = self._apply_limits(strat_weights)

        # Step 4: Build final matrix: matrix[strategy][symbol] = strat_weight × asset_frac
        matrix = {}
        for name in strat_weights:
            if name not in normalized:
                continue
            matrix[name] = {}
            for sym, frac in normalized[name].items():
                matrix[name][sym] = strat_weights[name] * frac

        # Step 5: Compute marginal weights
        asset_weights = {}
        for name, syms in matrix.items():
            for sym, w in syms.items():
                asset_weights[sym] = asset_weights.get(sym, 0) + w

        result.matrix = matrix
        result.strategy_weights = strat_weights
        result.asset_weights = asset_weights
        result.scores = raw_scores
        result.n_strategies = len(strat_weights)
        result.n_assets = len(asset_weights)

        logger.info(f"Granular allocation: {result.n_strategies} strategies × {result.n_assets} assets")
        for name, syms in sorted(matrix.items()):
            for sym, w in sorted(syms.items(), key=lambda x: -x[1]):
                logger.info(f"  {name} × {sym}: {w:.1%}")

        return result

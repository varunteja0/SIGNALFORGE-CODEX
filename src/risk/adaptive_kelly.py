"""
Adaptive Kelly Position Sizing — Maximize Geometric Growth
============================================================
Standard Kelly Criterion assumes fixed win rate and payoff ratio.
Markets aren't fixed. This module adapts Kelly in real-time:

1. Bayesian Kelly — updates win probability with each new trade
2. Regime-aware sizing — smaller bets in volatile regimes
3. Fractional Kelly with dynamic fraction — more conservative after losses
4. Portfolio-level Kelly — accounts for correlation between strategies
5. Anti-martingale — increase size only after proven edge, never after losses

This is the difference between:
- Betting 2% every time (amateur)
- Betting the mathematically optimal amount given current evidence (pro)
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Output of the adaptive position sizer."""
    fraction: float             # 0-1, fraction of capital to risk
    kelly_full: float           # Full Kelly fraction
    kelly_half: float           # Half Kelly (conservative)
    kelly_adaptive: float       # Final adaptive fraction
    confidence: float           # 0-1, confidence in the estimate
    regime_adjustment: float    # Multiplier from regime (0.5 = halve size)
    drawdown_adjustment: float  # Multiplier from current drawdown
    reason: str = ""


@dataclass
class StrategyStats:
    """Running statistics for a strategy used in Kelly calculation."""
    name: str
    wins: int = 0
    losses: int = 0
    total_win_amount: float = 0.0
    total_loss_amount: float = 0.0
    trade_returns: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    peak_equity: float = 0.0


class AdaptiveKellySizer:
    """Position sizing that adapts to reality.
    
    Instead of a fixed 2% risk per trade, this computes the
    mathematically optimal bet size based on:
    - Observed win rate (Bayesian updated)
    - Observed payoff ratio
    - Current regime volatility
    - Current drawdown (reduce after losses)
    - Correlation with other active strategies
    """
    
    def __init__(
        self,
        max_fraction: float = 0.04,      # Never risk more than 4% per trade
        min_fraction: float = 0.002,      # Minimum 0.2% to be worth trading
        min_trades_for_kelly: int = 20,   # Need this many trades for reliable Kelly
        bayesian_prior_wins: int = 5,     # Prior: 5 wins, 5 losses (agnostic)
        bayesian_prior_losses: int = 5,
        drawdown_scale_start: float = 0.05,  # Start scaling down at 5% DD
        drawdown_scale_zero: float = 0.15,   # Zero sizing at 15% DD
    ):
        self.max_fraction = max_fraction
        self.min_fraction = min_fraction
        self.min_trades = min_trades_for_kelly
        self.prior_wins = bayesian_prior_wins
        self.prior_losses = bayesian_prior_losses
        self.dd_start = drawdown_scale_start
        self.dd_zero = drawdown_scale_zero
        
        self.strategy_stats: dict[str, StrategyStats] = {}
    
    def register_strategy(self, name: str, initial_equity: float = 1000):
        """Register a strategy for adaptive sizing."""
        self.strategy_stats[name] = StrategyStats(
            name=name,
            equity_curve=[initial_equity],
            peak_equity=initial_equity,
        )
    
    def record_trade(self, name: str, pnl: float, return_pct: float):
        """Record a trade result for Kelly updating."""
        if name not in self.strategy_stats:
            return
        
        stats = self.strategy_stats[name]
        stats.trade_returns.append(return_pct)
        
        if pnl > 0:
            stats.wins += 1
            stats.total_win_amount += pnl
        else:
            stats.losses += 1
            stats.total_loss_amount += abs(pnl)
        
        current_equity = stats.equity_curve[-1] + pnl
        stats.equity_curve.append(current_equity)
        stats.peak_equity = max(stats.peak_equity, current_equity)
    
    def compute_size(
        self,
        strategy_name: str,
        signal_strength: float = 0.5,
        current_capital: float = 1000,
        peak_capital: float = 1000,
        regime_volatility: float = 1.0,
    ) -> SizingResult:
        """Compute optimal position size for a strategy.
        
        Args:
            strategy_name: Registered strategy name
            signal_strength: 0-1 from the strategy signal
            current_capital: Current portfolio value
            peak_capital: Peak portfolio value (for DD calc)
            regime_volatility: Current regime vol relative to normal (1.0 = normal)
        
        Returns:
            SizingResult with the adaptive fraction to risk.
        """
        result = SizingResult(
            fraction=self.min_fraction,
            kelly_full=0.0,
            kelly_half=0.0,
            kelly_adaptive=0.0,
            confidence=0.0,
            regime_adjustment=1.0,
            drawdown_adjustment=1.0,
        )
        
        stats = self.strategy_stats.get(strategy_name)
        total_trades = (stats.wins + stats.losses) if stats else 0
        
        # ============================================================
        # 1. Bayesian Kelly Criterion
        # ============================================================
        if stats and total_trades >= self.min_trades:
            result.kelly_full, result.confidence = self._bayesian_kelly(stats)
        elif stats and total_trades > 0:
            # Partial information — be conservative
            result.kelly_full, result.confidence = self._bayesian_kelly(stats)
            result.confidence *= total_trades / self.min_trades
        else:
            # No data — use signal strength as proxy
            result.kelly_full = self._signal_based_kelly(signal_strength)
            result.confidence = 0.2
        
        result.kelly_half = result.kelly_full * 0.5
        
        # ============================================================
        # 2. Regime Adjustment
        # ============================================================
        result.regime_adjustment = self._regime_scale(regime_volatility)
        
        # ============================================================
        # 3. Drawdown Adjustment (anti-martingale)
        # ============================================================
        result.drawdown_adjustment = self._drawdown_scale(
            current_capital, peak_capital
        )
        
        # ============================================================
        # 4. Combine
        # ============================================================
        adaptive = (
            result.kelly_half
            * result.regime_adjustment
            * result.drawdown_adjustment
            * min(1.0, result.confidence + 0.3)  # Blend in confidence
        )
        
        # Clamp to bounds
        adaptive = max(self.min_fraction, min(self.max_fraction, adaptive))
        
        result.kelly_adaptive = adaptive
        result.fraction = adaptive
        
        result.reason = (
            f"Kelly={result.kelly_full:.3f} "
            f"Half={result.kelly_half:.3f} "
            f"Regime={result.regime_adjustment:.2f} "
            f"DD={result.drawdown_adjustment:.2f} "
            f"Conf={result.confidence:.2f} "
            f"→ {result.fraction:.3f}"
        )
        
        return result
    
    def compute_portfolio_size(
        self,
        strategy_names: list[str],
        signal_strengths: dict[str, float],
        current_capital: float,
        peak_capital: float,
        regime_volatility: float = 1.0,
    ) -> dict[str, SizingResult]:
        """Compute sizes for multiple strategies with correlation awareness.
        
        Reduces individual sizes if strategies are correlated to avoid
        concentrated risk.
        """
        results = {}
        
        for name in strategy_names:
            results[name] = self.compute_size(
                name,
                signal_strengths.get(name, 0.5),
                current_capital,
                peak_capital,
                regime_volatility,
            )
        
        # Compute pairwise correlations and reduce if correlated
        correlations = self._strategy_correlations(strategy_names)
        
        for name in strategy_names:
            # Average absolute correlation with other active strategies
            avg_corr = 0.0
            count = 0
            for other in strategy_names:
                if other != name and (name, other) in correlations:
                    avg_corr += abs(correlations[(name, other)])
                    count += 1
            
            if count > 0:
                avg_corr /= count
                # High correlation → reduce size
                # corr=0 → no adjustment, corr=1 → halve allocation
                corr_adjustment = 1.0 - 0.5 * avg_corr
                results[name].fraction *= corr_adjustment
                results[name].fraction = max(
                    self.min_fraction, results[name].fraction
                )
        
        # Ensure total allocation doesn't exceed max portfolio risk
        total_risk = sum(r.fraction for r in results.values())
        max_total_risk = 0.15  # Max 15% total portfolio risk
        
        if total_risk > max_total_risk:
            scale = max_total_risk / total_risk
            for r in results.values():
                r.fraction *= scale
                r.fraction = max(self.min_fraction, r.fraction)
        
        return results
    
    # ================================================================
    # Internal Methods
    # ================================================================
    
    def _bayesian_kelly(self, stats: StrategyStats) -> tuple[float, float]:
        """Bayesian Kelly with Beta prior on win probability.
        
        Uses Beta(prior_wins + observed_wins, prior_losses + observed_losses)
        as the posterior distribution of win probability.
        
        Returns (kelly_fraction, confidence).
        """
        alpha = self.prior_wins + stats.wins
        beta_param = self.prior_losses + stats.losses
        
        # Posterior mean of win probability
        p = alpha / (alpha + beta_param)
        
        # Average win/loss ratio
        if stats.losses > 0 and stats.total_loss_amount > 0:
            avg_win = stats.total_win_amount / max(1, stats.wins)
            avg_loss = stats.total_loss_amount / max(1, stats.losses)
            b = avg_win / avg_loss  # Payoff ratio
        else:
            b = 1.5  # Prior assumption
        
        # Kelly: f* = (bp - q) / b
        q = 1 - p
        kelly = (b * p - q) / b
        kelly = max(0, kelly)
        
        # Confidence based on sample size
        total = stats.wins + stats.losses
        confidence = min(1.0, total / (self.min_trades * 2))
        
        # Also factor in the variance of the posterior
        # Narrow posterior (lots of data) → higher confidence
        posterior_var = (alpha * beta_param) / (
            (alpha + beta_param) ** 2 * (alpha + beta_param + 1)
        )
        # Low variance → high confidence
        confidence *= max(0.3, 1.0 - np.sqrt(posterior_var) * 10)
        
        return kelly, confidence
    
    def _signal_based_kelly(self, signal_strength: float) -> float:
        """Estimate Kelly from signal strength when no trade history available."""
        # Map signal strength [0,1] to a conservative Kelly
        # Strong signal (0.8+) → ~3% Kelly, weak (0.3) → ~0.5%
        kelly = signal_strength * 0.04
        return max(0, kelly)
    
    def _regime_scale(self, regime_volatility: float) -> float:
        """Scale position size based on regime volatility.
        
        Normal vol (1.0) → no change
        High vol (2.0) → halve size
        Low vol (0.5) → increase slightly
        """
        if regime_volatility <= 0:
            return 0.5
        
        # Inverse relationship: high vol → small size
        scale = 1.0 / regime_volatility
        return max(0.25, min(1.5, scale))
    
    def _drawdown_scale(
        self, current_capital: float, peak_capital: float
    ) -> float:
        """Anti-martingale: reduce size during drawdowns.
        
        No reduction at < dd_start (5% default)
        Linear reduction from dd_start to dd_zero
        Zero at dd_zero (15% default)
        """
        if peak_capital <= 0:
            return 1.0
        
        dd = (peak_capital - current_capital) / peak_capital
        
        if dd <= self.dd_start:
            return 1.0
        elif dd >= self.dd_zero:
            return 0.0
        else:
            # Linear interpolation
            return 1.0 - (dd - self.dd_start) / (self.dd_zero - self.dd_start)
    
    def _strategy_correlations(
        self, names: list[str]
    ) -> dict[tuple[str, str], float]:
        """Compute pairwise return correlations between strategies."""
        correlations = {}
        
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                stats_a = self.strategy_stats.get(name_a)
                stats_b = self.strategy_stats.get(name_b)
                
                if not stats_a or not stats_b:
                    correlations[(name_a, name_b)] = 0.0
                    continue
                
                rets_a = stats_a.trade_returns
                rets_b = stats_b.trade_returns
                
                # Use overlapping period
                min_len = min(len(rets_a), len(rets_b))
                if min_len < 10:
                    correlations[(name_a, name_b)] = 0.0
                    continue
                
                corr = np.corrcoef(
                    rets_a[-min_len:], rets_b[-min_len:]
                )[0, 1]
                
                if np.isnan(corr):
                    corr = 0.0
                
                correlations[(name_a, name_b)] = float(corr)
                correlations[(name_b, name_a)] = float(corr)
        
        return correlations

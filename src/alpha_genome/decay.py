"""
Alpha Genome — Strategy Decay Detector
=========================================
The silent killer: strategies that worked yesterday stop working tomorrow.
This module detects alpha decay in REAL TIME and kills dying strategies
before they bleed capital.

Most quant systems skip this. They discover strategies, deploy them,
and wonder why they lose money 6 months later. The answer: alpha decays.
Market microstructure shifts, participants adapt, regime changes.

This detector uses:
    1. Rolling Sharpe ratio with structural break detection (CUSUM)
    2. Returns distribution shift (Kolmogorov-Smirnov test)
    3. Win rate decay via exponentially weighted tracking
    4. Drawdown persistence analysis
    5. Volume of signal generation (drying up = strategy losing relevance)

A strategy is KILLED when multiple decay signals fire simultaneously.
No emotion. No "maybe it'll come back." Dead alphas get replaced.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
import time

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class DecayReport:
    """Complete health report for a single strategy."""
    strategy_name: str
    is_alive: bool = True
    decay_score: float = 0.0          # 0 = healthy, 100 = dead
    kill_recommended: bool = False

    # Individual decay signals
    sharpe_decay: float = 0.0         # Rolling Sharpe trend (negative = decaying)
    cusum_break: bool = False         # Structural break detected
    distribution_shift: float = 1.0   # KS p-value (low = shifted)
    win_rate_trend: float = 0.0       # EWM win rate slope
    drawdown_persistence: float = 0.0 # How long stuck in drawdown (0-1)
    signal_frequency_change: float = 0.0  # Signal generation rate change
    
    # Context
    recent_sharpe: float = 0.0
    lifetime_sharpe: float = 0.0
    recent_trades: int = 0
    total_trades: int = 0
    days_in_drawdown: int = 0
    peak_capital: float = 0.0
    current_capital: float = 0.0

    reason: str = ""

    def __repr__(self):
        status = "ALIVE" if self.is_alive else "DEAD"
        return (
            f"DecayReport({self.strategy_name}: {status} "
            f"decay={self.decay_score:.0f}/100 "
            f"sharpe_recent={self.recent_sharpe:.2f} "
            f"sharpe_life={self.lifetime_sharpe:.2f} "
            f"trades={self.total_trades})"
        )


@dataclass
class StrategyTracker:
    """Tracks live performance of a deployed strategy."""
    name: str
    deployed_at: float = 0.0
    trades: list = field(default_factory=list)       # List of {pnl, timestamp, direction}
    equity_curve: list = field(default_factory=list)  # Running equity values
    signal_timestamps: list = field(default_factory=list)  # When signals were generated
    initial_capital: float = 0.0
    current_capital: float = 0.0
    peak_capital: float = 0.0


class DecayDetector:
    """Monitors live strategy performance and detects alpha decay.
    
    Usage:
        detector = DecayDetector()
        detector.register_strategy("alpha_001", initial_capital=1000)
        
        # As trades come in:
        detector.record_trade("alpha_001", pnl=15.5)
        detector.record_trade("alpha_001", pnl=-8.2)
        
        # Check health:
        report = detector.check_health("alpha_001")
        if report.kill_recommended:
            # Replace this strategy with a new evolved one
            ...
    """

    def __init__(
        self,
        # Decay thresholds
        min_trades_for_assessment: int = 20,
        rolling_sharpe_window: int = 30,     # trades
        cusum_threshold: float = 2.0,        # std deviations
        ks_alpha: float = 0.05,              # KS test significance
        max_drawdown_days: int = 30,         # Kill if in DD for this long
        decay_score_kill_threshold: float = 70.0,  # 0-100
        # Weights for composite decay score
        w_sharpe: float = 0.30,
        w_cusum: float = 0.15,
        w_distribution: float = 0.15,
        w_winrate: float = 0.20,
        w_drawdown: float = 0.15,
        w_signal_freq: float = 0.05,
    ):
        self.min_trades = min_trades_for_assessment
        self.rolling_window = rolling_sharpe_window
        self.cusum_threshold = cusum_threshold
        self.ks_alpha = ks_alpha
        self.max_dd_days = max_drawdown_days
        self.kill_threshold = decay_score_kill_threshold

        self.weights = {
            "sharpe": w_sharpe,
            "cusum": w_cusum,
            "distribution": w_distribution,
            "winrate": w_winrate,
            "drawdown": w_drawdown,
            "signal_freq": w_signal_freq,
        }

        self.strategies: dict[str, StrategyTracker] = {}

    def register_strategy(
        self, name: str, initial_capital: float = 1000.0
    ):
        """Register a new strategy for decay monitoring."""
        self.strategies[name] = StrategyTracker(
            name=name,
            deployed_at=time.time(),
            initial_capital=initial_capital,
            current_capital=initial_capital,
            peak_capital=initial_capital,
        )
        logger.info(f"Decay monitor: registered '{name}' with ${initial_capital:.2f}")

    def record_trade(
        self, name: str, pnl: float, timestamp: Optional[float] = None
    ):
        """Record a completed trade for a strategy."""
        if name not in self.strategies:
            return

        tracker = self.strategies[name]
        ts = timestamp or time.time()

        tracker.trades.append({"pnl": pnl, "timestamp": ts})
        tracker.current_capital += pnl
        tracker.peak_capital = max(tracker.peak_capital, tracker.current_capital)
        tracker.equity_curve.append(tracker.current_capital)

    def record_signal(self, name: str, timestamp: Optional[float] = None):
        """Record that a strategy generated a signal (even if not traded)."""
        if name not in self.strategies:
            return
        ts = timestamp or time.time()
        self.strategies[name].signal_timestamps.append(ts)

    def check_health(self, name: str) -> DecayReport:
        """Full health assessment of a strategy.
        
        Returns a DecayReport with composite decay score.
        If decay_score > kill_threshold, kill_recommended = True.
        """
        report = DecayReport(strategy_name=name)

        if name not in self.strategies:
            report.is_alive = False
            report.reason = "Strategy not registered"
            return report

        tracker = self.strategies[name]
        report.total_trades = len(tracker.trades)
        report.peak_capital = tracker.peak_capital
        report.current_capital = tracker.current_capital

        # Need minimum trades for meaningful assessment
        if report.total_trades < self.min_trades:
            report.reason = f"Insufficient trades ({report.total_trades}/{self.min_trades})"
            return report

        pnls = pd.Series([t["pnl"] for t in tracker.trades])
        timestamps = pd.Series([t["timestamp"] for t in tracker.trades])

        # ============================================================
        # 1. Rolling Sharpe Decay
        # ============================================================
        report.sharpe_decay, report.recent_sharpe, report.lifetime_sharpe = (
            self._sharpe_decay(pnls)
        )

        # ============================================================
        # 2. CUSUM Structural Break
        # ============================================================
        report.cusum_break = self._cusum_test(pnls)

        # ============================================================
        # 3. Returns Distribution Shift (KS Test)
        # ============================================================
        report.distribution_shift = self._distribution_shift(pnls)

        # ============================================================
        # 4. Win Rate Trend
        # ============================================================
        report.win_rate_trend = self._win_rate_trend(pnls)

        # ============================================================
        # 5. Drawdown Persistence
        # ============================================================
        report.drawdown_persistence, report.days_in_drawdown = (
            self._drawdown_persistence(tracker)
        )

        # ============================================================
        # 6. Signal Frequency Change
        # ============================================================
        report.signal_frequency_change = self._signal_frequency_change(tracker)

        # Recent trade count
        recent_cutoff = len(pnls) - min(self.rolling_window, len(pnls))
        report.recent_trades = len(pnls) - recent_cutoff

        # ============================================================
        # Composite Decay Score (0 = healthy, 100 = dead)
        # ============================================================
        scores = {
            "sharpe": self._normalize_sharpe_decay(report.sharpe_decay),
            "cusum": 100.0 if report.cusum_break else 0.0,
            "distribution": self._normalize_ks(report.distribution_shift),
            "winrate": self._normalize_winrate_trend(report.win_rate_trend),
            "drawdown": report.drawdown_persistence * 100,
            "signal_freq": self._normalize_signal_freq(report.signal_frequency_change),
        }

        report.decay_score = sum(
            scores[k] * self.weights[k] for k in self.weights
        )

        # Also kill if lifetime performance is deeply negative
        # (strategy was never good — not decay, but failure)
        if report.lifetime_sharpe < -1.0 and report.total_trades >= self.min_trades:
            report.decay_score = max(report.decay_score, 80.0)
        elif report.lifetime_sharpe < -0.5 and report.total_trades >= self.min_trades * 2:
            report.decay_score = max(report.decay_score, 60.0)

        # Kill decision
        if report.decay_score >= self.kill_threshold:
            report.kill_recommended = True
            report.is_alive = False
            reasons = []
            if scores["sharpe"] > 50:
                reasons.append(f"Sharpe collapsed (recent={report.recent_sharpe:.2f} vs life={report.lifetime_sharpe:.2f})")
            if report.cusum_break:
                reasons.append("Structural break detected (CUSUM)")
            if scores["distribution"] > 50:
                reasons.append(f"Returns distribution shifted (KS p={report.distribution_shift:.4f})")
            if scores["winrate"] > 50:
                reasons.append(f"Win rate declining")
            if scores["drawdown"] > 50:
                reasons.append(f"Stuck in drawdown for {report.days_in_drawdown} days")
            report.reason = "; ".join(reasons) if reasons else "Composite decay threshold exceeded"
        else:
            report.is_alive = True
            report.reason = "Healthy"

        return report

    def check_all(self) -> list[DecayReport]:
        """Check health of all tracked strategies."""
        return [self.check_health(name) for name in self.strategies]

    def get_kill_list(self) -> list[str]:
        """Return names of strategies that should be killed."""
        reports = self.check_all()
        return [r.strategy_name for r in reports if r.kill_recommended]

    # ================================================================
    # Internal Analysis Methods
    # ================================================================

    def _sharpe_decay(self, pnls: pd.Series) -> tuple[float, float, float]:
        """Compute rolling Sharpe trend. Returns (trend, recent_sharpe, lifetime_sharpe)."""
        if pnls.std() < 1e-10:
            return 0.0, 0.0, 0.0

        lifetime_sharpe = float(pnls.mean() / pnls.std() * np.sqrt(252))

        window = min(self.rolling_window, len(pnls) // 2)
        if window < 10:
            return 0.0, lifetime_sharpe, lifetime_sharpe

        # Rolling Sharpe (annualized, per-trade basis)
        rolling_mean = pnls.rolling(window).mean()
        rolling_std = pnls.rolling(window).std()
        rolling_sharpe = (rolling_mean / (rolling_std + 1e-10)) * np.sqrt(252)
        rolling_sharpe = rolling_sharpe.dropna()

        if len(rolling_sharpe) < 5:
            return 0.0, lifetime_sharpe, lifetime_sharpe

        recent_sharpe = float(rolling_sharpe.iloc[-1])

        # Linear regression on rolling Sharpe to get trend
        x = np.arange(len(rolling_sharpe))
        slope, _, _, _, _ = sp_stats.linregress(x, rolling_sharpe.values)
        trend = float(slope * len(rolling_sharpe))  # Total change over window

        return trend, recent_sharpe, lifetime_sharpe

    def _cusum_test(self, pnls: pd.Series) -> bool:
        """CUSUM test for structural break in returns.
        
        Detects if the mean return has shifted significantly downward.
        """
        mean = pnls.mean()
        std = pnls.std()
        if std < 1e-10:
            return False

        # One-sided CUSUM (detect downward shift)
        cusum_pos = 0.0
        cusum_neg = 0.0
        threshold = self.cusum_threshold * std

        for val in pnls:
            cusum_pos = max(0, cusum_pos + (val - mean) - 0.5 * std)
            cusum_neg = min(0, cusum_neg + (val - mean) + 0.5 * std)

            if abs(cusum_neg) > threshold:
                return True  # Structural break detected (downward)

        return False

    def _distribution_shift(self, pnls: pd.Series) -> float:
        """KS test between first half and second half of returns.
        
        Low p-value = distribution has changed (bad sign).
        """
        n = len(pnls)
        half = n // 2
        if half < 10:
            return 1.0  # Not enough data

        first_half = pnls.iloc[:half]
        second_half = pnls.iloc[half:]

        ks_stat, p_value = sp_stats.ks_2samp(first_half, second_half)
        return float(p_value)

    def _win_rate_trend(self, pnls: pd.Series) -> float:
        """Exponentially weighted win rate trend.
        
        Returns slope of EWM win rate. Negative = declining.
        """
        wins = (pnls > 0).astype(float)
        ewm_wr = wins.ewm(span=min(20, len(wins) // 2 + 1)).mean()

        if len(ewm_wr) < 10:
            return 0.0

        x = np.arange(len(ewm_wr))
        slope, _, _, _, _ = sp_stats.linregress(x, ewm_wr.values)
        return float(slope * len(ewm_wr))  # Total change

    def _drawdown_persistence(self, tracker: StrategyTracker) -> tuple[float, int]:
        """How long has the strategy been in drawdown?
        
        Returns (persistence_ratio, days_in_dd).
        persistence_ratio: 0 = just started DD, 1 = maxed out.
        """
        if tracker.peak_capital <= 0:
            return 0.0, 0

        current_dd = (tracker.peak_capital - tracker.current_capital) / tracker.peak_capital
        if current_dd < 0.01:  # Less than 1% DD — not in drawdown
            return 0.0, 0

        # Find when peak was hit
        equity = tracker.equity_curve
        if not equity:
            return 0.0, 0

        peak_idx = 0
        peak_val = 0
        for i, val in enumerate(equity):
            if val >= peak_val:
                peak_val = val
                peak_idx = i

        bars_in_dd = len(equity) - peak_idx
        days_in_dd = bars_in_dd  # Approximate: depends on trading frequency

        persistence = min(1.0, days_in_dd / self.max_dd_days)
        return persistence, days_in_dd

    def _signal_frequency_change(self, tracker: StrategyTracker) -> float:
        """Detect if the strategy is generating fewer signals over time.
        
        Drying up signals often precedes strategy death.
        Returns: negative = fewer signals recently.
        """
        timestamps = tracker.signal_timestamps
        if len(timestamps) < 20:
            return 0.0

        ts = pd.Series(timestamps)
        half = len(ts) // 2

        first_half_rate = half / (ts.iloc[half - 1] - ts.iloc[0] + 1e-10)
        second_half_rate = (len(ts) - half) / (ts.iloc[-1] - ts.iloc[half] + 1e-10)

        if first_half_rate < 1e-10:
            return 0.0

        change = (second_half_rate - first_half_rate) / first_half_rate
        return float(change)

    # ================================================================
    # Score Normalization (map to 0-100 decay scale)
    # ================================================================

    def _normalize_sharpe_decay(self, trend: float) -> float:
        """Map Sharpe trend to 0-100 decay score. Negative trend = high score."""
        if trend >= 0:
            return 0.0
        # -1.0 trend → 50 score, -2.0 → 100
        return min(100.0, abs(trend) * 50)

    def _normalize_ks(self, p_value: float) -> float:
        """Map KS p-value to decay score. Low p = high decay."""
        if p_value > self.ks_alpha:
            return 0.0
        # p=0.05 → 50, p=0.001 → 100
        return min(100.0, (1 - p_value / self.ks_alpha) * 100)

    def _normalize_winrate_trend(self, trend: float) -> float:
        """Map win rate trend to decay score."""
        if trend >= 0:
            return 0.0
        return min(100.0, abs(trend) * 200)

    def _normalize_signal_freq(self, change: float) -> float:
        """Map signal frequency change to decay score."""
        if change >= 0:
            return 0.0
        # -50% reduction → 50 score, -100% → 100
        return min(100.0, abs(change) * 100)

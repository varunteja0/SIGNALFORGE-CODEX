from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioRiskSnapshot:
    target_volatility: float
    realized_volatility: float
    realized_cagr: float
    growth_score: float
    volatility_multiplier: float
    correlation_multiplier: float
    growth_multiplier: float
    final_multiplier: float
    smoothed_volatility: float = 0.0
    volatility_tracking_error: float = 0.0
    pid_output: float = 0.0
    control_error: float = 0.0
    integral_error: float = 0.0
    derivative_error: float = 0.0
    strategy_correlation_average: float = 0.0
    strategy_correlation_peak: float = 0.0
    asset_correlation_average: float = 0.0
    asset_correlation_peak: float = 0.0
    correlation_shock: bool = False


@dataclass
class RiskFeedbackState:
    smoothed_volatility: float = 0.0
    integral_error: float = 0.0
    previous_error: float = 0.0
    last_multiplier: float = 1.0


class PortfolioRiskController:
    """Apply PID-style volatility targeting and cross-market correlation shock protection."""

    def __init__(
        self,
        target_volatility: float = 0.15,
        min_realized_volatility: float = 0.03,
        min_multiplier: float = 0.45,
        max_multiplier: float = 1.75,
        correlation_window: int = 72,
        strategy_corr_threshold: float = 0.55,
        asset_corr_threshold: float = 0.65,
        shock_penalty: float = 0.65,
        growth_drawdown_weight: float = 1.35,
        growth_vol_weight: float = 0.50,
        growth_tilt: float = 0.10,
        pid_kp: float = 1.10,
        pid_ki: float = 0.18,
        pid_kd: float = 0.35,
        integral_limit: float = 2.50,
        volatility_smoothing: float = 0.35,
    ):
        self.target_volatility = target_volatility
        self.min_realized_volatility = min_realized_volatility
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.correlation_window = correlation_window
        self.strategy_corr_threshold = strategy_corr_threshold
        self.asset_corr_threshold = asset_corr_threshold
        self.shock_penalty = shock_penalty
        self.growth_drawdown_weight = growth_drawdown_weight
        self.growth_vol_weight = growth_vol_weight
        self.growth_tilt = growth_tilt
        self.pid_kp = pid_kp
        self.pid_ki = pid_ki
        self.pid_kd = pid_kd
        self.integral_limit = integral_limit
        self.volatility_smoothing = volatility_smoothing
        self.feedback_state = RiskFeedbackState()

    def reset(self) -> None:
        self.feedback_state = RiskFeedbackState()

    def evaluate(
        self,
        portfolio_returns: pd.Series,
        strategy_returns: pd.DataFrame,
        asset_returns: pd.DataFrame,
        *,
        portfolio_drawdown: float = 0.0,
    ) -> PortfolioRiskSnapshot:
        clean_portfolio_returns = (
            portfolio_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if portfolio_returns is not None
            else pd.Series(dtype=float)
        )

        realized_volatility = self._annualized_volatility(clean_portfolio_returns)
        realized_cagr = self._cagr(clean_portfolio_returns)
        growth_score = float(
            realized_cagr
            - self.growth_drawdown_weight * max(0.0, float(portfolio_drawdown))
            - self.growth_vol_weight * realized_volatility
        )

        prior_smoothed = float(self.feedback_state.smoothed_volatility)
        if prior_smoothed > 0.0:
            smoothed_volatility = float(
                self.volatility_smoothing * realized_volatility
                + (1.0 - self.volatility_smoothing) * prior_smoothed
            )
        else:
            smoothed_volatility = realized_volatility

        control_volatility = float(
            max(
                realized_volatility,
                0.5 * (realized_volatility + smoothed_volatility),
            )
        )
        control_error = float(
            (self.target_volatility - control_volatility) / max(self.target_volatility, 1e-9)
        )
        integral_error = float(
            np.clip(
                self.feedback_state.integral_error + control_error,
                -self.integral_limit,
                self.integral_limit,
            )
        )
        derivative_error = float(control_error - self.feedback_state.previous_error)
        pid_output = self.pid_kp * control_error + self.pid_ki * integral_error + self.pid_kd * derivative_error
        unclipped_multiplier = 1.0 + pid_output
        volatility_multiplier = float(
            np.clip(
                unclipped_multiplier,
                self.min_multiplier,
                self.max_multiplier,
            )
        )

        if volatility_multiplier != unclipped_multiplier and np.sign(control_error) == np.sign(integral_error):
            integral_error = float(self.feedback_state.integral_error)
            pid_output = self.pid_kp * control_error + self.pid_ki * integral_error + self.pid_kd * derivative_error
            volatility_multiplier = float(
                np.clip(
                    1.0 + pid_output,
                    self.min_multiplier,
                    self.max_multiplier,
                )
            )

        self.feedback_state = RiskFeedbackState(
            smoothed_volatility=smoothed_volatility,
            integral_error=integral_error,
            previous_error=control_error,
            last_multiplier=volatility_multiplier,
        )
        volatility_tracking_error = float(abs(self.target_volatility - realized_volatility))

        growth_multiplier = float(
            np.clip(
                1.0 + self.growth_tilt * np.tanh(growth_score / 0.10),
                0.85,
                1.15,
            )
        )

        strategy_corr_avg, strategy_corr_peak = self._correlation_stats(strategy_returns)
        asset_corr_avg, asset_corr_peak = self._correlation_stats(asset_returns)
        correlation_multiplier, correlation_shock = self._correlation_multiplier(
            strategy_corr_avg=strategy_corr_avg,
            strategy_corr_peak=strategy_corr_peak,
            asset_corr_avg=asset_corr_avg,
            asset_corr_peak=asset_corr_peak,
        )

        final_multiplier = float(
            np.clip(
                volatility_multiplier * correlation_multiplier * growth_multiplier,
                self.min_multiplier,
                self.max_multiplier,
            )
        )
        return PortfolioRiskSnapshot(
            target_volatility=self.target_volatility,
            realized_volatility=realized_volatility,
            realized_cagr=realized_cagr,
            growth_score=growth_score,
            volatility_multiplier=volatility_multiplier,
            correlation_multiplier=correlation_multiplier,
            growth_multiplier=growth_multiplier,
            final_multiplier=final_multiplier,
            smoothed_volatility=smoothed_volatility,
            volatility_tracking_error=volatility_tracking_error,
            pid_output=float(pid_output),
            control_error=control_error,
            integral_error=integral_error,
            derivative_error=derivative_error,
            strategy_correlation_average=strategy_corr_avg,
            strategy_correlation_peak=strategy_corr_peak,
            asset_correlation_average=asset_corr_avg,
            asset_correlation_peak=asset_corr_peak,
            correlation_shock=correlation_shock,
        )

    def _correlation_multiplier(
        self,
        *,
        strategy_corr_avg: float,
        strategy_corr_peak: float,
        asset_corr_avg: float,
        asset_corr_peak: float,
    ) -> tuple[float, bool]:
        strategy_pressure = max(
            strategy_corr_avg / max(self.strategy_corr_threshold, 1e-9),
            strategy_corr_peak / max(self.strategy_corr_threshold + 0.10, 1e-9),
        )
        asset_pressure = max(
            asset_corr_avg / max(self.asset_corr_threshold, 1e-9),
            asset_corr_peak / max(self.asset_corr_threshold + 0.10, 1e-9),
        )
        shock_intensity = max(strategy_pressure, asset_pressure) - 1.0
        correlation_shock = shock_intensity > 0.0
        multiplier = 1.0 - self.shock_penalty * min(max(shock_intensity, 0.0), 1.0)
        return float(np.clip(multiplier, 0.45, 1.0)), correlation_shock

    def _annualized_volatility(self, returns: pd.Series) -> float:
        clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) <= 1 or float(clean.std(ddof=0)) <= 0.0:
            return 0.0
        return float(clean.std(ddof=0) * np.sqrt(self._periods_per_year(clean.index)))

    def _cagr(self, returns: pd.Series) -> float:
        clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) == 0:
            return 0.0
        wealth = float((1.0 + clean).prod())
        if wealth <= 0.0:
            return -1.0
        periods_per_year = self._periods_per_year(clean.index)
        years = len(clean) / max(periods_per_year, 1.0)
        if years <= 0.0:
            return float(wealth - 1.0)
        return float(wealth ** (1.0 / years) - 1.0)

    def _correlation_stats(self, returns: pd.DataFrame) -> tuple[float, float]:
        if returns is None or returns.empty or len(returns.columns) <= 1:
            return 0.0, 0.0
        window = returns.tail(self.correlation_window).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        corr = window.corr().abs().fillna(0.0)
        if corr.empty:
            return 0.0, 0.0
        mask = ~np.eye(len(corr), dtype=bool)
        off_diag = corr.to_numpy()[mask]
        if off_diag.size == 0:
            return 0.0, 0.0
        return float(off_diag.mean()), float(off_diag.max())

    @staticmethod
    def _periods_per_year(index: pd.Index) -> float:
        if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
            deltas = pd.Series(index).diff().dropna()
            median_delta = deltas.median()
            if hasattr(median_delta, "total_seconds"):
                return float(365.25 * 24.0 * 3600.0 / max(median_delta.total_seconds(), 1.0))
        return float(252.0 * 24.0)
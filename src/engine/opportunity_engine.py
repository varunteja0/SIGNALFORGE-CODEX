"""Institutional-style cross-market opportunity engine.

This module is the pragmatic answer to the repo's current gap:
- The existing portfolio engine is strategy-slot driven and hand-curated.
- The arena experiments showed that repeated IS tuning on a narrow universe
  did not produce a robust OOS edge.

So this engine does something different:
- trade a broader liquid universe,
- score every asset every bar using multiple orthogonal signals,
- select only the strongest opportunities,
- enforce gross/net/concentration caps,
- scale down automatically in high-vol and drawdown regimes.

It is still evidence-driven. It does not promise profitability.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data.features import compute_all_features
from src.data.fetcher import DataFetcher
from src.data.multi_venue import MultiVenueFetcher
from src.data.structural import StructuralDataFetcher
from src.risk.advanced import AdvancedRiskManager

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "AVAX/USDT",
    "LTC/USDT",
]


@dataclass
class OpportunityEngineConfig:
    assets: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    timeframe: str = "1h"
    data_days: int = 365
    initial_capital: float = 10_000.0
    max_positions: int = 6
    max_longs: int = 4
    max_shorts: int = 4
    gross_limit: float = 1.00
    net_limit: float = 0.35
    per_asset_cap: float = 0.25
    score_threshold: float = 0.18
    min_liquidity_score: float = 0.20
    turnover_cost_bps: float = 8.0
    regime_vol_baseline_daily: float = 0.04
    use_multi_venue: bool = True


@dataclass
class OpportunitySnapshot:
    symbol: str
    alpha: float
    opportunity: float
    target_weight: float
    trend_score: float
    carry_score: float
    crowding_score: float
    squeeze_score: float
    smart_money_score: float
    realized_vol: float
    liquidity: float
    data_quality: float


@dataclass
class OpportunityBacktestResult:
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    mean_turnover: float = 0.0
    mean_gross_exposure: float = 0.0
    mean_net_exposure: float = 0.0
    positive_month_frac: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    monthly_returns: pd.Series = field(default_factory=pd.Series)
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    latest_snapshots: list[OpportunitySnapshot] = field(default_factory=list)


class OpportunityEngine:
    """Cross-market opportunity ranking and portfolio construction engine."""

    def __init__(self, config: Optional[OpportunityEngineConfig] = None):
        self.config = config or OpportunityEngineConfig()
        self.fetcher: Optional[DataFetcher] = None
        self.structural: Optional[StructuralDataFetcher] = None
        self.multi_venue: Optional[MultiVenueFetcher] = None

    @classmethod
    def default(cls) -> "OpportunityEngine":
        return cls(OpportunityEngineConfig())

    def _ensure_clients(self) -> None:
        if self.fetcher is None:
            self.fetcher = DataFetcher()
        if self.structural is None:
            self.structural = StructuralDataFetcher()
        if self.config.use_multi_venue and self.multi_venue is None:
            self.multi_venue = MultiVenueFetcher()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self) -> dict[str, pd.DataFrame]:
        self._ensure_clients()
        assert self.fetcher is not None
        assert self.structural is not None
        datasets: dict[str, pd.DataFrame] = {}
        for symbol in self.config.assets:
            try:
                price_df = self.fetcher.fetch(
                    symbol,
                    timeframe=self.config.timeframe,
                    days=self.config.data_days,
                )
                if price_df.empty:
                    logger.warning("%s: no OHLCV data", symbol)
                    continue
                enriched = compute_all_features(price_df)
                enriched = self.structural.fetch_all(
                    symbol=symbol.replace("/", ""),
                    price_df=enriched,
                    days=min(self.config.data_days, 90),
                )
                if self.multi_venue is not None:
                    try:
                        enriched = self.multi_venue.fetch_all(
                            symbol=symbol,
                            days=min(self.config.data_days, 30),
                            price_df=enriched,
                        )
                    except Exception as exc:
                        logger.warning("%s: multi-venue enrichment failed: %s", symbol, exc)
                enriched = enriched.sort_index()
                datasets[symbol] = enriched
                logger.info("%s: loaded %d enriched bars", symbol, len(enriched))
            except Exception as exc:
                logger.warning("%s: failed to load enriched data: %s", symbol, exc)
        return datasets

    # ------------------------------------------------------------------
    # Signal scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(default, index=df.index, dtype=float)

    @staticmethod
    def _clip_tanh(series: pd.Series, scale: float = 1.0) -> pd.Series:
        scale = max(float(scale), 1e-9)
        return np.tanh(series.fillna(0.0) / scale)

    def score_asset(self, df: pd.DataFrame) -> pd.DataFrame:
        close = self._col(df, "close")
        volume = self._col(df, "volume")
        atr = self._col(df, "atr_14").replace(0.0, np.nan)
        realized_vol = self._col(df, "vol_20").replace(0.0, np.nan)
        if realized_vol.isna().all():
            realized_vol = self._col(df, "vol_50").replace(0.0, np.nan)
        realized_vol = realized_vol.ffill().fillna(0.01)

        trend = (
            0.30 * self._clip_tanh(self._col(df, "ret_vol_adj_20"), 2.0)
            + 0.25 * self._clip_tanh(self._col(df, "ema_cross_20_100"), 0.03)
            + 0.20 * self._clip_tanh(
                self._col(df, "macd_hist_slope") / (atr + 1e-9),
                0.25,
            )
            + 0.25 * self._clip_tanh(
                ((self._col(df, "adx_14") - 20.0) / 15.0)
                * np.sign(self._col(df, "price_vs_ma_50")),
                1.0,
            )
        )

        carry = (
            0.45 * -self._clip_tanh(self._col(df, "fund_funding_zscore"), 2.0)
            + 0.25 * -self._clip_tanh(self._col(df, "lsr_lsr_zscore"), 2.0)
            + 0.30 * self._clip_tanh(self._col(df, "top_retail_divergence_zscore"), 2.0)
        )

        liq_diff = self._col(df, "liq_pressure_short") - self._col(df, "liq_pressure_long")
        liq_scale = liq_diff.abs().rolling(72, min_periods=12).mean().replace(0.0, np.nan)
        crowding = (
            0.50 * self._clip_tanh(liq_diff / (liq_scale + 1e-9), 1.0)
            + 0.30 * self._clip_tanh(self._col(df, "smart_money_divergence"), 1.0)
            + 0.20 * -self._clip_tanh(self._col(df, "cross_venue_funding_zscore"), 2.0)
        )

        squeeze_direction = np.sign(
            self._col(df, "ret_3") + 0.20 * self._col(df, "macd_hist_slope")
        )
        squeeze = self._col(df, "squeeze") * squeeze_direction

        smart_money = (
            0.60 * self._clip_tanh(self._col(df, "smart_money_divergence"), 1.0)
            + 0.40 * self._clip_tanh(self._col(df, "taker_taker_imbalance"), 0.35)
        )

        alpha = (
            0.34 * trend
            + 0.26 * carry
            + 0.18 * crowding
            + 0.10 * squeeze
            + 0.12 * smart_money
        )

        rolling_volume = volume.rolling(72, min_periods=12).mean().replace(0.0, np.nan)
        liquidity = (volume / rolling_volume).clip(lower=0.0, upper=2.0).fillna(0.0) / 2.0

        critical_cols = [
            "fund_funding_zscore",
            "oi_oi_zscore",
            "lsr_lsr_zscore",
            "taker_taker_imbalance",
            "vol_20",
            "atr_14",
        ]
        present = pd.concat(
            [(self._col(df, col).notna()).astype(float) for col in critical_cols],
            axis=1,
        )
        data_quality = present.mean(axis=1)

        noise = self._col(df, "vol_of_vol_20")
        noise_base = noise.rolling(72, min_periods=12).median().replace(0.0, np.nan)
        noise_penalty = (noise / (noise_base + 1e-9)).clip(lower=0.0, upper=2.0).fillna(0.0) / 2.0

        opportunity = (
            alpha.abs()
            * (0.50 + 0.50 * liquidity)
            * data_quality
            * (1.0 - 0.50 * noise_penalty)
        )

        out = pd.DataFrame(
            {
                "alpha": alpha.clip(-1.0, 1.0),
                "opportunity": opportunity.clip(lower=0.0, upper=1.0),
                "trend_score": trend.clip(-1.0, 1.0),
                "carry_score": carry.clip(-1.0, 1.0),
                "crowding_score": crowding.clip(-1.0, 1.0),
                "squeeze_score": squeeze.clip(-1.0, 1.0),
                "smart_money_score": smart_money.clip(-1.0, 1.0),
                "realized_vol": realized_vol.clip(lower=1e-6),
                "liquidity": liquidity.clip(lower=0.0, upper=1.0),
                "data_quality": data_quality.clip(lower=0.0, upper=1.0),
                "close": close,
            },
            index=df.index,
        )
        return out.fillna(0.0)

    # ------------------------------------------------------------------
    # Portfolio construction
    # ------------------------------------------------------------------
    def _cap_positive_weights(self, weights: pd.Series, total_budget: float) -> pd.Series:
        if weights.empty or total_budget <= 0.0:
            return pd.Series(0.0, index=weights.index, dtype=float)

        feasible_budget = min(float(total_budget), self.config.per_asset_cap * len(weights))
        if feasible_budget <= 0.0:
            return pd.Series(0.0, index=weights.index, dtype=float)

        scaled = weights.astype(float)
        scaled = scaled / max(scaled.sum(), 1e-12) * feasible_budget

        for _ in range(len(scaled) + 2):
            over = scaled > self.config.per_asset_cap + 1e-12
            if not over.any():
                break
            excess = float((scaled[over] - self.config.per_asset_cap).sum())
            scaled.loc[over] = self.config.per_asset_cap
            under = ~over
            if not under.any() or excess <= 0.0:
                break
            basis = scaled.loc[under]
            if float(basis.sum()) <= 1e-12:
                scaled.loc[under] = scaled.loc[under] + excess / int(under.sum())
            else:
                scaled.loc[under] = basis + excess * basis / float(basis.sum())

        return scaled.clip(lower=0.0)

    def _build_weights(self, snapshot: pd.DataFrame, gross_budget: float) -> pd.Series:
        weights = pd.Series(0.0, index=snapshot.index, dtype=float)
        if snapshot.empty or gross_budget <= 0.0:
            return weights

        eligible = snapshot[
            (snapshot["opportunity"] >= self.config.score_threshold)
            & (snapshot["liquidity"] >= self.config.min_liquidity_score)
            & (snapshot["data_quality"] >= 0.5)
            & (snapshot["realized_vol"] > 0.0)
        ].copy()
        if eligible.empty:
            return weights

        longs = eligible[eligible["alpha"] > 0.0].sort_values(
            ["opportunity", "alpha"], ascending=False
        ).head(self.config.max_longs)
        shorts = eligible[eligible["alpha"] < 0.0].sort_values(
            ["opportunity", "alpha"], ascending=[False, True]
        ).head(self.config.max_shorts)

        if len(longs) + len(shorts) > self.config.max_positions:
            combined = pd.concat(
                [
                    longs.assign(_side=1),
                    shorts.assign(_side=-1),
                ]
            ).sort_values("opportunity", ascending=False).head(self.config.max_positions)
            longs = combined[combined["_side"] == 1].drop(columns="_side")
            shorts = combined[combined["_side"] == -1].drop(columns="_side")

        raw_longs = (longs["opportunity"] * longs["alpha"].abs()) / longs["realized_vol"]
        raw_shorts = (shorts["opportunity"] * shorts["alpha"].abs()) / shorts["realized_vol"]

        long_strength = float(raw_longs.sum())
        short_strength = float(raw_shorts.sum())
        total_strength = long_strength + short_strength

        if total_strength <= 0.0:
            return weights

        if long_strength > 0.0 and short_strength > 0.0:
            long_budget = gross_budget * long_strength / total_strength
            short_budget = gross_budget - long_budget
            net = long_budget - short_budget
            if net > self.config.net_limit:
                shift = min((net - self.config.net_limit) / 2.0, long_budget)
                long_budget -= shift
                short_budget += shift
            elif net < -self.config.net_limit:
                shift = min((-self.config.net_limit - net) / 2.0, short_budget)
                short_budget -= shift
                long_budget += shift
        elif long_strength > 0.0:
            long_budget = min(gross_budget, self.config.net_limit)
            short_budget = 0.0
        else:
            long_budget = 0.0
            short_budget = min(gross_budget, self.config.net_limit)

        if long_strength > 0.0:
            weights.loc[raw_longs.index] = self._cap_positive_weights(raw_longs, long_budget)
        if short_strength > 0.0:
            weights.loc[raw_shorts.index] = -self._cap_positive_weights(raw_shorts, short_budget)
        return weights.fillna(0.0)

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------
    def _bars_per_year(self) -> float:
        if self.config.timeframe == "1h":
            return 24.0 * 365.25
        if self.config.timeframe == "4h":
            return 6.0 * 365.25
        if self.config.timeframe == "1d":
            return 365.25
        return 24.0 * 365.25

    def backtest(
        self,
        datasets: Optional[dict[str, pd.DataFrame]] = None,
    ) -> OpportunityBacktestResult:
        if datasets is None:
            datasets = self.load_data()
        if not datasets:
            return OpportunityBacktestResult()

        assets = list(datasets)
        score_frames = {symbol: self.score_asset(df) for symbol, df in datasets.items()}
        index = sorted({ts for df in datasets.values() for ts in df.index})
        if not index:
            return OpportunityBacktestResult()
        index = pd.DatetimeIndex(index)

        returns_df = pd.DataFrame(index=index, columns=assets, dtype=float)
        for symbol, df in datasets.items():
            returns_df[symbol] = df["close"].pct_change().reindex(index).fillna(0.0)
            score_frames[symbol] = score_frames[symbol].reindex(index).ffill().fillna(0.0)

        risk = AdvancedRiskManager(
            initial_capital=self.config.initial_capital,
            max_portfolio_heat=self.config.gross_limit,
            regime_vol_baseline=self.config.regime_vol_baseline_daily,
        )

        weights_history: list[pd.Series] = []
        gross_history: list[float] = []
        net_history: list[float] = []
        turnover_history: list[float] = []
        returns_history: list[float] = []
        equity_history: list[float] = []

        capital = float(self.config.initial_capital)
        prev_weights = pd.Series(0.0, index=assets, dtype=float)

        for ts in index:
            bar_snapshot = pd.DataFrame(
                {
                    symbol: score_frames[symbol].loc[
                        ts,
                        [
                            "alpha",
                            "opportunity",
                            "trend_score",
                            "carry_score",
                            "crowding_score",
                            "squeeze_score",
                            "smart_money_score",
                            "realized_vol",
                            "liquidity",
                            "data_quality",
                        ],
                    ]
                    for symbol in assets
                }
            ).T
            bar_snapshot.index.name = "symbol"

            market_vol_daily = float(bar_snapshot["realized_vol"].replace(0.0, np.nan).median())
            if not np.isfinite(market_vol_daily):
                market_vol_daily = self.config.regime_vol_baseline_daily
            else:
                market_vol_daily *= np.sqrt(24.0)

            bar_return = float((prev_weights * returns_df.loc[ts].fillna(0.0)).sum())
            risk.update_capital(capital)
            risk_state = risk.get_risk_state(current_regime_vol=market_vol_daily)
            gross_budget = self.config.gross_limit * risk_state.size_multiplier * risk_state.regime_multiplier
            target_weights = (
                self._build_weights(bar_snapshot, gross_budget)
                if risk_state.can_trade
                else pd.Series(0.0, index=assets, dtype=float)
            )
            turnover = float((target_weights - prev_weights).abs().sum())
            cost = turnover * (self.config.turnover_cost_bps / 10_000.0)
            net_return = bar_return - cost
            capital = max(capital * (1.0 + net_return), 1.0)

            returns_history.append(net_return)
            equity_history.append(capital)
            weights_history.append(target_weights)
            gross_history.append(float(target_weights.abs().sum()))
            net_history.append(float(target_weights.sum()))
            turnover_history.append(turnover)
            prev_weights = target_weights

        equity_curve = pd.Series(equity_history, index=index, dtype=float)
        returns = pd.Series(returns_history, index=index, dtype=float)
        weights = pd.DataFrame(weights_history, index=index).fillna(0.0)

        total_return = float(equity_curve.iloc[-1] / self.config.initial_capital - 1.0)
        days = max((index[-1] - index[0]).total_seconds() / 86400.0, 1.0)
        annualized_return = float((equity_curve.iloc[-1] / self.config.initial_capital) ** (365.25 / days) - 1.0)
        sharpe = 0.0
        if float(returns.std(ddof=0)) > 0.0:
            sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(self._bars_per_year()))
        running_peak = equity_curve.cummax()
        max_drawdown = float(((equity_curve - running_peak) / running_peak.replace(0.0, np.nan)).min())
        monthly_returns = equity_curve.resample("ME").last().pct_change().dropna()

        latest_time = index[-1]
        latest_weights = weights.loc[latest_time]
        latest_snapshot_df = pd.DataFrame(
            {
                symbol: score_frames[symbol].loc[
                    latest_time,
                    [
                        "alpha",
                        "opportunity",
                        "trend_score",
                        "carry_score",
                        "crowding_score",
                        "squeeze_score",
                        "smart_money_score",
                        "realized_vol",
                        "liquidity",
                        "data_quality",
                    ],
                ]
                for symbol in assets
            }
        ).T
        latest_snapshots: list[OpportunitySnapshot] = []
        for symbol, row in latest_snapshot_df.sort_values("opportunity", ascending=False).iterrows():
            weight = float(latest_weights.get(symbol, 0.0))
            if abs(weight) < 1e-9 and float(row["opportunity"]) < self.config.score_threshold:
                continue
            latest_snapshots.append(
                OpportunitySnapshot(
                    symbol=symbol,
                    alpha=float(row["alpha"]),
                    opportunity=float(row["opportunity"]),
                    target_weight=weight,
                    trend_score=float(row["trend_score"]),
                    carry_score=float(row["carry_score"]),
                    crowding_score=float(row["crowding_score"]),
                    squeeze_score=float(row["squeeze_score"]),
                    smart_money_score=float(row["smart_money_score"]),
                    realized_vol=float(row["realized_vol"]),
                    liquidity=float(row["liquidity"]),
                    data_quality=float(row["data_quality"]),
                )
            )

        return OpportunityBacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe=sharpe,
            max_drawdown=abs(max_drawdown),
            mean_turnover=float(np.mean(turnover_history)) if turnover_history else 0.0,
            mean_gross_exposure=float(np.mean(gross_history)) if gross_history else 0.0,
            mean_net_exposure=float(np.mean(net_history)) if net_history else 0.0,
            positive_month_frac=float((monthly_returns > 0.0).mean()) if len(monthly_returns) else 0.0,
            equity_curve=equity_curve,
            returns=returns,
            monthly_returns=monthly_returns,
            weights=weights,
            latest_snapshots=latest_snapshots[: self.config.max_positions],
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def capital_profile(self, result: OpportunityBacktestResult, starting_capital: Optional[float] = None) -> dict[str, float]:
        start_capital = float(starting_capital or self.config.initial_capital)
        if result.monthly_returns.empty:
            return {
                "start_capital": start_capital,
                "end_capital": start_capital,
                "best_month": 0.0,
                "worst_month": 0.0,
                "median_month": 0.0,
                "positive_month_frac": 0.0,
            }
        wealth = start_capital * (1.0 + result.monthly_returns).cumprod()
        return {
            "start_capital": start_capital,
            "end_capital": float(wealth.iloc[-1]),
            "best_month": float(result.monthly_returns.max()),
            "worst_month": float(result.monthly_returns.min()),
            "median_month": float(result.monthly_returns.median()),
            "positive_month_frac": float((result.monthly_returns > 0.0).mean()),
        }

    def report(self, result: OpportunityBacktestResult) -> str:
        profile = self.capital_profile(result)
        lines: list[str] = []
        lines.append("=" * 78)
        lines.append("  INSTITUTIONAL OPPORTUNITY ENGINE — BACKTEST REPORT")
        lines.append("=" * 78)
        lines.append(f"  Universe: {', '.join(self.config.assets)}")
        lines.append(f"  Timeframe: {self.config.timeframe}   Lookback: {self.config.data_days} days")
        lines.append(f"  Capital: ${self.config.initial_capital:,.0f}")
        lines.append("")
        lines.append("  PORTFOLIO")
        lines.append(f"  Total Return:      {result.total_return:+.2%}")
        lines.append(f"  Annualized Return: {result.annualized_return:+.2%}")
        lines.append(f"  Sharpe:            {result.sharpe:.2f}")
        lines.append(f"  Max Drawdown:      {result.max_drawdown:.2%}")
        lines.append(f"  Mean Turnover:     {result.mean_turnover:.3f}")
        lines.append(f"  Mean Gross:        {result.mean_gross_exposure:.2f}")
        lines.append(f"  Mean Net:          {result.mean_net_exposure:+.2f}")
        lines.append("")
        lines.append("  MONTHLY COMPOUNDING PROFILE")
        lines.append(f"  Start Capital:     ${profile['start_capital']:,.0f}")
        lines.append(f"  End Capital:       ${profile['end_capital']:,.0f}")
        lines.append(f"  Positive Months:   {profile['positive_month_frac']:.1%}")
        lines.append(f"  Median Month:      {profile['median_month']:+.2%}")
        lines.append(f"  Best Month:        {profile['best_month']:+.2%}")
        lines.append(f"  Worst Month:       {profile['worst_month']:+.2%}")
        lines.append("")
        lines.append("  CURRENT TOP OPPORTUNITIES")
        lines.append(f"  {'Symbol':<12s} {'Wgt':>7s} {'Alpha':>7s} {'Opp':>7s} {'Trend':>7s} {'Carry':>7s} {'Crowd':>7s}")
        lines.append(f"  {'-' * 12} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")
        for snap in result.latest_snapshots:
            lines.append(
                f"  {snap.symbol:<12s} {snap.target_weight:>+7.2f} {snap.alpha:>+7.2f} {snap.opportunity:>7.2f} "
                f"{snap.trend_score:>+7.2f} {snap.carry_score:>+7.2f} {snap.crowding_score:>+7.2f}"
            )
        if not result.latest_snapshots:
            lines.append("  (no current opportunities above threshold)")
        lines.append("=" * 78)
        return "\n".join(lines)

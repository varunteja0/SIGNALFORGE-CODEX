"""
Funding Rate Extreme Fade — Focused Production Strategy
=========================================================
THESIS
------
Perpetual futures funding = cost paid by longs to shorts (or vice versa)
every 8 hours. When funding is extremely positive, longs are paying
an unsustainable rent. Historically, when the pain becomes large enough
relative to recent norms, price mean-reverts within 1–3 funding periods.

Mechanism (why it's real, not an artefact):
    * Cost-of-leverage is a structural drag on crowded positions.
    * Retail tends to chase; funding spikes mark the top of that chase.
    * Market-makers & prop shops will short into the extreme to collect
      funding, which itself applies downward price pressure.

Signal:
    f_z = rolling z-score of funding rate over ``lookback`` periods
    LONG  when f_z < -entry_z  AND funding < 0  (crowd is short, flush incoming)
    SHORT when f_z > +entry_z  AND funding > 0  (crowd is long, flush incoming)

Entry at the funding timestamp (00/08/16 UTC), exit ``hold_periods`` later
at the same bar type. Realistic costs (commission + slippage) deducted.

This module is intentionally a single file, no dependencies on the
scanner/validator factory — we are running one specific thesis, not
mining anomalies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data.fetcher import DataFetcher
from src.data.funding import FundingRateFetcher

logger = logging.getLogger(__name__)

# ─── Parameters (documented defaults) ────────────────────────────
DEFAULT_LOOKBACK = 90          # funding periods (~30 days @ 3/day)
DEFAULT_ENTRY_Z = 2.0          # enter on |z| > this
DEFAULT_HOLD_PERIODS = 3       # exit after 3 × 8h = 24h
DEFAULT_COMMISSION = 0.0004    # 4 bps per side (taker futures)
DEFAULT_SLIPPAGE = 0.0005      # 5 bps each side
DEFAULT_MAX_POSITION = 1.0     # fraction of capital, before sizing scale
DEFAULT_STOP_LOSS_PCT = 0.03   # hard stop 3 % adverse
DEFAULT_PERPS_IN_DAY = 3       # 8h funding cycle


# ─── Data structures ─────────────────────────────────────────────

@dataclass
class FundingFadeTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int              # +1 long, -1 short
    entry_price: float
    exit_price: float
    f_z: float                  # z-score at entry
    funding: float              # raw funding at entry
    size: float                 # position size (fraction of capital)
    gross_return: float         # raw price return (direction-adjusted)
    funding_earned: float       # funding paid TO us during hold
    cost: float                 # commission + slippage total
    pnl: float                  # net return on capital after costs
    exit_reason: str            # "timeout" | "stop"


@dataclass
class FundingFadeResult:
    symbol: str
    n_trades: int
    n_long: int
    n_short: int
    win_rate: float
    avg_return: float
    median_return: float
    total_return: float
    sharpe: float               # annualised, 3 funding/day
    sortino: float
    max_drawdown: float
    profit_factor: float
    params: dict
    trades: list[FundingFadeTrade] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None


# ─── Core backtest ───────────────────────────────────────────────

def _align_funding_with_price(
    funding: pd.DataFrame,
    ohlcv: pd.DataFrame,
) -> pd.DataFrame:
    """Return funding aligned to OHLCV bars, forward-filled.

    Both indexes are tz-naive pandas datetime.
    """
    # Take the funding rate column only
    if "funding_rate" not in funding.columns:
        raise ValueError("funding DataFrame must have 'funding_rate' column")

    # Reindex funding onto OHLCV index, forward-fill
    aligned = funding["funding_rate"].reindex(ohlcv.index, method="ffill")
    out = ohlcv.copy()
    out["funding_rate"] = aligned
    return out


def backtest_funding_fade(
    ohlcv: pd.DataFrame,
    funding: pd.DataFrame,
    symbol: str = "UNKNOWN",
    lookback: int = DEFAULT_LOOKBACK,
    entry_z: float = DEFAULT_ENTRY_Z,
    hold_periods: int = DEFAULT_HOLD_PERIODS,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    max_position: float = DEFAULT_MAX_POSITION,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    size_by_z: bool = True,
    side: str = "both",    # "both" | "short_only" | "long_only"
) -> FundingFadeResult:
    """Backtest the funding-fade strategy on aligned data.

    Signals evaluated only at funding-payment timestamps (00/08/16 UTC),
    not at every bar. Position size scales with |z| if ``size_by_z``.
    """
    params = {
        "lookback": lookback,
        "entry_z": entry_z,
        "hold_periods": hold_periods,
        "commission": commission,
        "slippage": slippage,
        "max_position": max_position,
        "stop_loss_pct": stop_loss_pct,
        "size_by_z": size_by_z,
    }

    if ohlcv.empty or funding.empty:
        return FundingFadeResult(
            symbol=symbol, n_trades=0, n_long=0, n_short=0,
            win_rate=0.0, avg_return=0.0, median_return=0.0,
            total_return=0.0, sharpe=0.0, sortino=0.0,
            max_drawdown=0.0, profit_factor=0.0, params=params,
        )

    df = _align_funding_with_price(funding, ohlcv)

    # Compute z-score on the raw funding series (one value per 8h)
    f_series = funding["funding_rate"].copy()
    f_mu = f_series.rolling(lookback, min_periods=lookback // 2).mean()
    f_sd = f_series.rolling(lookback, min_periods=lookback // 2).std()
    f_z = (f_series - f_mu) / (f_sd + 1e-10)

    # Funding timestamps are 00/08/16 UTC — iterate those only
    funding_ts = funding.index
    bar_hours = int(pd.Timedelta(ohlcv.index[1] - ohlcv.index[0]).total_seconds() // 3600)
    bars_per_period = max(1, 8 // bar_hours)

    trades: list[FundingFadeTrade] = []
    for i, ts in enumerate(funding_ts):
        z = f_z.iloc[i] if i < len(f_z) else np.nan
        if np.isnan(z):
            continue

        fund = f_series.iloc[i]

        if z > entry_z and fund > 0:
            direction = -1   # short the crowded longs
        elif z < -entry_z and fund < 0:
            direction = +1   # long the crowded shorts
        else:
            continue

        if side == "short_only" and direction != -1:
            continue
        if side == "long_only" and direction != +1:
            continue

        # Locate entry/exit bars on OHLCV (entry on next bar open)
        try:
            entry_bar_loc = ohlcv.index.get_indexer([ts], method="bfill")[0]
        except Exception:
            continue
        if entry_bar_loc < 0 or entry_bar_loc + 1 >= len(ohlcv):
            continue

        entry_idx = entry_bar_loc + 1           # next-bar open (no look-ahead)
        exit_idx = entry_idx + hold_periods * bars_per_period
        if exit_idx >= len(ohlcv):
            continue

        entry_px = float(ohlcv["open"].iloc[entry_idx])
        # Check stop during hold
        window = ohlcv.iloc[entry_idx: exit_idx + 1]
        if direction == +1:
            adverse = (window["low"].min() - entry_px) / entry_px
            hit_stop = adverse < -stop_loss_pct
        else:
            adverse = (entry_px - window["high"].max()) / entry_px
            hit_stop = adverse < -stop_loss_pct

        if hit_stop:
            exit_px = entry_px * (1 - direction * stop_loss_pct)
            exit_reason = "stop"
            exit_time = window.index[
                (window["low"].le(exit_px) if direction == +1 else window["high"].ge(exit_px)).idxmax()
                if not window.empty else window.index[-1]
            ] if False else window.index[-1]
            # simpler: linearly locate the first violating bar
            if direction == +1:
                viol = window.index[window["low"] <= exit_px]
            else:
                viol = window.index[window["high"] >= exit_px]
            exit_time = viol[0] if len(viol) > 0 else window.index[-1]
        else:
            exit_px = float(ohlcv["close"].iloc[exit_idx])
            exit_reason = "timeout"
            exit_time = ohlcv.index[exit_idx]

        gross_ret = direction * (exit_px - entry_px) / entry_px

        # Funding earned: if short and funding positive → collect it
        periods_held = hold_periods
        funding_earned = 0.0
        for k in range(periods_held):
            fi = i + k
            if fi < len(f_series):
                funding_earned += -direction * f_series.iloc[fi]
                # direction=-1 (short) earns positive funding; direction=+1 pays
                # The sign of funding_earned = +direction_sensitive

        total_cost = (commission + slippage) * 2  # entry + exit
        # Size by |z| up to max_position
        if size_by_z:
            size = min(max_position, abs(z) / entry_z * 0.5)
        else:
            size = max_position

        net_pnl = size * (gross_ret + funding_earned - total_cost)

        trades.append(FundingFadeTrade(
            entry_time=ohlcv.index[entry_idx],
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_px,
            exit_price=exit_px,
            f_z=float(z),
            funding=float(fund),
            size=float(size),
            gross_return=float(gross_ret),
            funding_earned=float(funding_earned),
            cost=float(total_cost),
            pnl=float(net_pnl),
            exit_reason=exit_reason,
        ))

    # ── Stats ──
    if not trades:
        return FundingFadeResult(
            symbol=symbol, n_trades=0, n_long=0, n_short=0,
            win_rate=0.0, avg_return=0.0, median_return=0.0,
            total_return=0.0, sharpe=0.0, sortino=0.0,
            max_drawdown=0.0, profit_factor=0.0, params=params,
        )

    pnls = np.array([t.pnl for t in trades])
    n_long = sum(1 for t in trades if t.direction == +1)
    n_short = len(trades) - n_long

    # Equity curve: compound on *capital*, each trade risks `size`
    equity = (1 + pnls).cumprod()
    equity_curve = pd.Series(equity, index=[t.exit_time for t in trades])

    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_dd = float(abs(drawdown.min())) if len(drawdown) else 0.0

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else np.inf

    # Annualised Sharpe: trades-per-year
    span_days = max(1.0, (trades[-1].exit_time - trades[0].entry_time).total_seconds() / 86400)
    trades_per_year = len(trades) * 365.0 / span_days
    sharpe = (pnls.mean() / pnls.std() * np.sqrt(trades_per_year)) if pnls.std() > 0 else 0.0
    downside = pnls[pnls < 0]
    sortino = (pnls.mean() / downside.std() * np.sqrt(trades_per_year)) if len(downside) > 1 and downside.std() > 0 else 0.0

    return FundingFadeResult(
        symbol=symbol,
        n_trades=len(trades),
        n_long=n_long,
        n_short=n_short,
        win_rate=float((pnls > 0).mean()),
        avg_return=float(pnls.mean()),
        median_return=float(np.median(pnls)),
        total_return=float(equity[-1] - 1),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=max_dd,
        profit_factor=float(pf) if np.isfinite(pf) else 999.0,
        params=params,
        trades=trades,
        equity_curve=equity_curve,
    )


# ─── Convenience loaders ─────────────────────────────────────────

def load_data(
    symbol: str,
    days: int = 730,
    timeframe: str = "1h",
    funding_symbol: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch OHLCV + funding rates for a symbol and return both."""
    fetcher = DataFetcher()
    ohlcv = fetcher.fetch(symbol, timeframe=timeframe, days=days)

    # Binance uses 'BTC/USDT' directly for funding on futures
    fr_symbol = funding_symbol or symbol
    funding_fetcher = FundingRateFetcher(exchange_id="binance")
    funding = funding_fetcher.fetch_history(fr_symbol, days=days)
    if funding.empty:
        # Try bybit fallback with :USDT suffix
        funding_fetcher = FundingRateFetcher(exchange_id="bybit")
        funding = funding_fetcher.fetch_history(f"{fr_symbol}:USDT", days=days)

    return ohlcv, funding


# ─── Walk-forward IS/OOS split ────────────────────────────────────

def evaluate_is_oos(
    symbol: str,
    days: int = 730,
    oos_fraction: float = 0.30,
    **kwargs,
) -> dict:
    """Load data, split IS/OOS by time, run backtest on each, return summary."""
    ohlcv, funding = load_data(symbol, days=days)
    if ohlcv.empty or funding.empty:
        return {"symbol": symbol, "error": "no data"}

    # Align indexes to same window
    start = max(ohlcv.index[0], funding.index[0])
    end = min(ohlcv.index[-1], funding.index[-1])
    ohlcv = ohlcv[(ohlcv.index >= start) & (ohlcv.index <= end)]
    funding = funding[(funding.index >= start) & (funding.index <= end)]

    if len(ohlcv) < 500 or len(funding) < 100:
        return {"symbol": symbol, "error": "insufficient data"}

    split_idx = int(len(ohlcv) * (1 - oos_fraction))
    split_ts = ohlcv.index[split_idx]

    ohlcv_is = ohlcv[ohlcv.index < split_ts]
    ohlcv_oos = ohlcv[ohlcv.index >= split_ts]
    funding_is = funding[funding.index < split_ts]
    funding_oos = funding[funding.index >= split_ts]

    is_result = backtest_funding_fade(ohlcv_is, funding_is, symbol=symbol, **kwargs)
    oos_result = backtest_funding_fade(ohlcv_oos, funding_oos, symbol=symbol, **kwargs)

    return {
        "symbol": symbol,
        "split_ts": str(split_ts),
        "is": is_result,
        "oos": oos_result,
    }

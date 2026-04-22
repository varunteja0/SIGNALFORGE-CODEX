"""Frozen-data arena research and submission helpers."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ARENA_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
BARS_PER_YEAR = 24 * 365
OOS_FRACTION = 0.30
COST_PER_TURN = 6e-4
MAX_GROSS = 1.5


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    params: dict[str, Any]
    notes: str = ""


@dataclass
class CandidateResult:
    spec: CandidateSpec
    weights: dict[str, float]
    symbol_metrics: dict[str, dict[str, float]]
    portfolio_metrics: dict[str, float]
    score: float
    positions: dict[str, pd.Series]

    def to_dict(self, include_oos: bool = False) -> dict[str, Any]:
        portfolio = {
            k: v
            for k, v in self.portfolio_metrics.items()
            if include_oos or not k.startswith("oos_")
        }
        symbols = {
            symbol: {
                k: v
                for k, v in metrics.items()
                if include_oos or not k.startswith("oos_")
            }
            for symbol, metrics in self.symbol_metrics.items()
        }
        return {
            "spec": asdict(self.spec),
            "weights": self.weights,
            "score": self.score,
            "portfolio_metrics": portfolio,
            "symbol_metrics": symbols,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def sharpe_annualised(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) < 2 or clean.std(ddof=0) == 0:
        return 0.0
    return float(clean.mean() / clean.std(ddof=0) * np.sqrt(BARS_PER_YEAR))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int) -> float:
    if n_trials < 1 or n_obs < 2:
        return 0.0
    penalty = np.sqrt(2.0 * np.log(max(n_trials, 1))) / np.sqrt(n_obs)
    haircut = penalty * np.sqrt(BARS_PER_YEAR)
    return float(max(0.0, abs(sharpe) - haircut) * np.sign(sharpe))


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_index().copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    elif out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def load_frozen_bars(
    arena_root: str | Path,
    symbols: tuple[str, ...] = ARENA_SYMBOLS,
) -> dict[str, pd.DataFrame]:
    root = Path(arena_root).expanduser().resolve()
    frames: dict[str, pd.DataFrame] = {}
    common_index: pd.DatetimeIndex | None = None

    for symbol in symbols:
        flat = symbol.replace("/", "_")
        path = root / "data" / "frozen" / f"{flat}_1h.parquet"
        frame = _ensure_utc_index(pd.read_parquet(path))
        frames[symbol] = frame[["open", "high", "low", "close", "volume"]].astype(float)
        common_index = frame.index if common_index is None else common_index.intersection(frame.index)

    assert common_index is not None
    return {symbol: frame.reindex(common_index).copy() for symbol, frame in frames.items()}


def _true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    pieces = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return pieces.max(axis=1)


def _compute_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"].astype(float)
    logret = np.log(close).diff()
    tr = _true_range(out)

    for span in (12, 24, 48, 72, 84, 96, 120, 144, 168, 240):
        out[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()

    for window in (12, 24, 48, 72, 84, 96, 120, 144, 168, 240):
        out[f"mom_{window}"] = close.pct_change(window)

    for window in (24, 48, 72, 168):
        min_periods = max(6, window // 4)
        out[f"vol_{window}"] = logret.rolling(window, min_periods=min_periods).std(ddof=0)
        out[f"atr_{window}"] = (
            tr.rolling(window, min_periods=min_periods).mean() / close.replace(0, np.nan)
        )

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.ffill()


def build_feature_cache(bars_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        symbol: _compute_features(frame)
        for symbol, frame in bars_by_symbol.items()
    }


def _target_scale(
    vol: pd.Series,
    *,
    target_vol: float,
    max_scale: float,
    vol_cap: float,
) -> pd.Series:
    safe_vol = vol.replace(0.0, np.nan)
    scale = (target_vol / safe_vol).clip(lower=0.0, upper=max_scale)
    scale = scale.where(vol <= vol_cap, other=0.0)
    return scale.fillna(0.0)


def _trend_positions(features: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    fast = int(params["fast"])
    slow = int(params["slow"])
    confirm = int(params["confirm"])
    enter = float(params["enter"])
    exit_thr = float(params["exit"])
    vol_window = int(params.get("vol_window", 72))
    target_vol = float(params.get("target_vol", 0.006))
    vol_cap = float(params.get("vol_cap", 0.03))
    max_scale = float(params.get("max_scale", 1.0))

    atr = features["atr_24"].replace(0.0, np.nan)
    trend = (features[f"ema_{fast}"] / features[f"ema_{slow}"] - 1.0) / atr
    confirm_mom = features[f"mom_{confirm}"]
    scale = _target_scale(
        features[f"vol_{vol_window}"],
        target_vol=target_vol,
        max_scale=max_scale,
        vol_cap=vol_cap,
    )

    out = np.zeros(len(features), dtype=float)
    state = 0.0
    for i in range(len(features)):
        trend_i = trend.iat[i]
        mom_i = confirm_mom.iat[i]
        if not np.isfinite(trend_i) or not np.isfinite(mom_i):
            out[i] = 0.0
            continue

        if state == 0.0:
            if trend_i >= enter and mom_i > 0.0:
                state = 1.0
            elif trend_i <= -enter and mom_i < 0.0:
                state = -1.0
        elif state > 0.0:
            if trend_i <= -enter and mom_i < 0.0:
                state = -1.0
            elif trend_i <= exit_thr or mom_i < 0.0:
                state = 0.0
        else:
            if trend_i >= enter and mom_i > 0.0:
                state = 1.0
            elif trend_i >= -exit_thr or mom_i > 0.0:
                state = 0.0

        out[i] = state * scale.iat[i]

    return pd.Series(out, index=features.index, dtype=float).clip(-1.0, 1.0)


def _pullback_positions(features: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    bias_fast = int(params["bias_fast"])
    bias_slow = int(params["bias_slow"])
    stretch_span = int(params["stretch_span"])
    enter = float(params["enter"])
    exit_thr = float(params["exit"])
    bias_thr = float(params.get("bias_thr", 0.0))
    vol_window = int(params.get("vol_window", 72))
    target_vol = float(params.get("target_vol", 0.006))
    vol_cap = float(params.get("vol_cap", 0.03))
    max_scale = float(params.get("max_scale", 0.9))

    atr = features["atr_24"].replace(0.0, np.nan)
    stretch = (features["close"] / features[f"ema_{stretch_span}"] - 1.0) / atr
    scale = _target_scale(
        features[f"vol_{vol_window}"],
        target_vol=target_vol,
        max_scale=max_scale,
        vol_cap=vol_cap,
    )

    out = np.zeros(len(features), dtype=float)
    state = 0.0
    for i in range(len(features)):
        ema_fast = features[f"ema_{bias_fast}"].iat[i]
        ema_slow = features[f"ema_{bias_slow}"].iat[i]
        slow_mom = features[f"mom_{bias_slow}"].iat[i]
        stretch_i = stretch.iat[i]
        if not np.isfinite(ema_fast) or not np.isfinite(ema_slow):
            out[i] = 0.0
            continue
        if not np.isfinite(slow_mom) or not np.isfinite(stretch_i):
            out[i] = 0.0
            continue

        bias = 0.0
        if ema_fast > ema_slow and slow_mom > bias_thr:
            bias = 1.0
        elif ema_fast < ema_slow and slow_mom < -bias_thr:
            bias = -1.0

        if state == 0.0:
            if bias > 0.0 and stretch_i <= -enter:
                state = 1.0
            elif bias < 0.0 and stretch_i >= enter:
                state = -1.0
        elif state > 0.0:
            if bias <= 0.0 or stretch_i >= -exit_thr:
                state = 0.0
        else:
            if bias >= 0.0 or stretch_i <= exit_thr:
                state = 0.0

        out[i] = state * scale.iat[i]

    return pd.Series(out, index=features.index, dtype=float).clip(-1.0, 1.0)


def _relative_strength_positions(
    features_by_symbol: dict[str, pd.DataFrame],
    params: dict[str, Any],
) -> dict[str, pd.Series]:
    lookback = int(params["lookback"])
    smooth = int(params["smooth"])
    enter_spread = float(params["enter_spread"])
    exit_spread = float(params["exit_spread"])
    switch_buffer = float(params.get("switch_buffer", 0.1))

    symbols = list(features_by_symbol)
    index = next(iter(features_by_symbol.values())).index
    target_vol = float(params.get("target_vol", 0.006))

    score_matrix = np.column_stack(
        [
            (
                features[f"mom_{lookback}"]
                / features["vol_72"].replace(0.0, np.nan)
            ).ewm(span=smooth, adjust=False).mean().to_numpy(dtype=float)
            for features in features_by_symbol.values()
        ]
    )
    vol_matrix = np.column_stack(
        [features["vol_72"].to_numpy(dtype=float) for features in features_by_symbol.values()]
    )

    positions = np.zeros_like(score_matrix)
    long_idx = -1
    short_idx = -1

    for i in range(len(index)):
        row = score_matrix[i]
        valid = np.isfinite(row)
        if not valid.any():
            continue

        row_valid = np.where(valid, row, np.nan)
        top_idx = int(np.nanargmax(row_valid))
        bottom_idx = int(np.nanargmin(row_valid))
        top_score = float(row[top_idx])
        bottom_score = float(row[bottom_idx])
        spread = top_score - bottom_score

        if spread <= exit_spread:
            long_idx = -1
            short_idx = -1
        else:
            if top_score > 0.0:
                if long_idx < 0:
                    long_idx = top_idx if spread >= enter_spread else -1
                elif top_idx != long_idx:
                    current = row[long_idx] if np.isfinite(row[long_idx]) else 0.0
                    if top_score - current >= switch_buffer:
                        long_idx = top_idx
            else:
                long_idx = -1

            if bottom_score < 0.0:
                if short_idx < 0:
                    short_idx = bottom_idx if spread >= enter_spread else -1
                elif bottom_idx != short_idx:
                    current = row[short_idx] if np.isfinite(row[short_idx]) else 0.0
                    if current - bottom_score >= switch_buffer:
                        short_idx = bottom_idx
            else:
                short_idx = -1

        active_indices = []
        if long_idx >= 0:
            active_indices.append(long_idx)
        if short_idx >= 0 and short_idx != long_idx:
            active_indices.append(short_idx)

        if active_indices:
            active_vols = vol_matrix[i, active_indices]
            finite_vols = active_vols[np.isfinite(active_vols)]
            mean_vol = float(finite_vols.mean()) if len(finite_vols) else 0.0
            scale = float(np.clip(target_vol / max(mean_vol, 1e-9), 0.0, 1.0))
        else:
            scale = 0.0

        if long_idx >= 0:
            positions[i, long_idx] = scale
        if short_idx >= 0:
            positions[i, short_idx] = -scale

    return {
        symbol: pd.Series(positions[:, idx], index=index, dtype=float).clip(-1.0, 1.0)
        for idx, symbol in enumerate(symbols)
    }


def _xsmom_positions(
    features_by_symbol: dict[str, pd.DataFrame],
    params: dict[str, Any],
) -> dict[str, pd.Series]:
    """Cross-sectional momentum, dollar-neutral, continuous rank weighting.

    Differs from `_relative_strength_positions` which is a discrete
    long/short pair switcher. Here every symbol holds a weight
    proportional to its de-meaned rank of vol-scaled momentum, so the
    portfolio is always dollar-neutral and holds all N names.
    """
    lookback = int(params["lookback"])
    smooth = int(params.get("smooth", 24))
    vol_window = int(params.get("vol_window", 72))
    target_vol = float(params.get("target_vol", 0.006))
    entry_thr = float(params.get("entry_threshold", 0.0))
    max_scale = float(params.get("max_scale", 1.0))

    symbols = list(features_by_symbol)
    index = next(iter(features_by_symbol.values())).index
    n = len(symbols)

    score_matrix = np.column_stack(
        [
            (
                features[f"mom_{lookback}"]
                / features[f"vol_{vol_window}"].replace(0.0, np.nan)
            ).ewm(span=smooth, adjust=False).mean().to_numpy(dtype=float)
            for features in features_by_symbol.values()
        ]
    )
    vol_matrix = np.column_stack(
        [
            features[f"vol_{vol_window}"].to_numpy(dtype=float)
            for features in features_by_symbol.values()
        ]
    )

    positions = np.zeros_like(score_matrix)
    for i in range(len(index)):
        row = score_matrix[i]
        if not np.isfinite(row).all():
            continue
        # Rank 0..n-1, de-mean to make dollar-neutral.
        order = np.argsort(row)
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        demeaned = ranks - ranks.mean()
        # Normalize so sum(|w|)=1 (gross 1.0 before vol scaling).
        denom = np.abs(demeaned).sum()
        if denom <= 1e-12:
            continue
        weights = demeaned / denom

        # Apply entry threshold on the dispersion of the scores.
        dispersion = float(np.nanmax(row) - np.nanmin(row))
        if dispersion < entry_thr:
            continue

        # Vol scaling using average realized vol of names held.
        held_vols = vol_matrix[i][np.abs(weights) > 1e-12]
        held_vols = held_vols[np.isfinite(held_vols)]
        mean_vol = float(held_vols.mean()) if len(held_vols) else 0.0
        if mean_vol <= 0.0:
            continue
        scale = float(np.clip(target_vol / mean_vol, 0.0, max_scale))
        positions[i] = weights * scale

    return {
        symbol: pd.Series(positions[:, idx], index=index, dtype=float).clip(-1.0, 1.0)
        for idx, symbol in enumerate(symbols)
    }


def _ensemble_positions(
    feature_cache: dict[str, pd.DataFrame],
    params: dict[str, Any],
) -> dict[str, pd.Series]:
    """Blend positions from multiple member strategies.

    params["members"]: list of {"family": str, "params": dict}
    params["member_weights"]: optional list[float]; defaults to equal weights.
    """
    members: list[dict[str, Any]] = list(params["members"])
    if not members:
        raise ValueError("ensemble requires at least one member")
    raw_weights = params.get("member_weights")
    if raw_weights is None:
        weights = [1.0 / len(members)] * len(members)
    else:
        weights = [float(w) for w in raw_weights]
        if len(weights) != len(members):
            raise ValueError("member_weights length mismatch with members")
        total = sum(abs(w) for w in weights)
        if total <= 0.0:
            weights = [1.0 / len(members)] * len(members)
        else:
            weights = [w / total for w in weights]

    combined: dict[str, pd.Series] = {}
    for member, weight in zip(members, weights):
        family = str(member["family"])
        member_params = dict(member["params"])
        if family == "trend":
            member_positions = {
                symbol: _trend_positions(features, member_params)
                for symbol, features in feature_cache.items()
            }
        elif family == "pullback":
            member_positions = {
                symbol: _pullback_positions(features, member_params)
                for symbol, features in feature_cache.items()
            }
        elif family == "relative":
            member_positions = _relative_strength_positions(feature_cache, member_params)
        elif family == "xsmom":
            member_positions = _xsmom_positions(feature_cache, member_params)
        else:
            raise ValueError(f"unsupported ensemble member family {family!r}")

        for symbol, pos in member_positions.items():
            scaled = pos.astype(float).fillna(0.0) * weight
            if symbol in combined:
                combined[symbol] = combined[symbol].add(scaled, fill_value=0.0)
            else:
                combined[symbol] = scaled

    return {
        symbol: series.clip(-1.0, 1.0).astype(float)
        for symbol, series in combined.items()
    }


def _regime_gated_positions(
    feature_cache: dict[str, pd.DataFrame],
    params: dict[str, Any],
) -> dict[str, pd.Series]:
    """Wrap a base spec and zero positions when per-symbol realized vol is
    in the top quantile over a rolling regime window.

    params["base"]: {"family": str, "params": dict}
    params["vol_window"]: int (realized-vol lookback in bars), default 72.
    params["regime_window"]: int (rolling quantile window), default 30*24.
    params["regime_quantile"]: float 0..1, default 0.90.
    """
    base = params["base"]
    base_family = str(base["family"])
    base_params = dict(base["params"])
    vol_window = int(params.get("vol_window", 72))
    regime_window = int(params.get("regime_window", 30 * 24))
    regime_quantile = float(params.get("regime_quantile", 0.90))

    if base_family == "trend":
        base_positions = {
            symbol: _trend_positions(features, base_params)
            for symbol, features in feature_cache.items()
        }
    elif base_family == "pullback":
        base_positions = {
            symbol: _pullback_positions(features, base_params)
            for symbol, features in feature_cache.items()
        }
    elif base_family == "relative":
        base_positions = _relative_strength_positions(feature_cache, base_params)
    elif base_family == "xsmom":
        base_positions = _xsmom_positions(feature_cache, base_params)
    else:
        raise ValueError(f"unsupported regime-gated base family {base_family!r}")

    gated: dict[str, pd.Series] = {}
    for symbol, series in base_positions.items():
        features = feature_cache[symbol]
        vol = features[f"vol_{vol_window}"].astype(float)
        thresh = vol.rolling(regime_window, min_periods=regime_window // 4).quantile(
            regime_quantile
        )
        mask = (vol < thresh).reindex(series.index).fillna(True)
        gated[symbol] = series.where(mask, 0.0).astype(float).clip(-1.0, 1.0)
    return gated


def build_positions(
    bars_by_symbol: dict[str, pd.DataFrame],
    spec: CandidateSpec,
    *,
    features_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.Series]:
    feature_cache = features_by_symbol or build_feature_cache(bars_by_symbol)

    if spec.family == "trend":
        return {
            symbol: _trend_positions(features, spec.params)
            for symbol, features in feature_cache.items()
        }
    if spec.family == "pullback":
        return {
            symbol: _pullback_positions(features, spec.params)
            for symbol, features in feature_cache.items()
        }
    if spec.family == "relative":
        return _relative_strength_positions(feature_cache, spec.params)
    if spec.family == "xsmom":
        return _xsmom_positions(feature_cache, spec.params)
    if spec.family == "ensemble":
        return _ensemble_positions(feature_cache, spec.params)
    if spec.family == "regime_gated":
        return _regime_gated_positions(feature_cache, spec.params)
    raise ValueError(f"unsupported family {spec.family!r}")


def _returns_from_position(close: pd.Series, position: pd.Series) -> pd.Series:
    px_ret = close.pct_change().fillna(0.0)
    turn = position.diff().abs().fillna(0.0)
    return position.shift(1).fillna(0.0) * px_ret - COST_PER_TURN * turn


def _returns_from_position_with_cost(
    close: pd.Series,
    position: pd.Series,
    *,
    cost_per_turn: float,
) -> pd.Series:
    px_ret = close.pct_change().fillna(0.0)
    turn = position.diff().abs().fillna(0.0)
    return position.shift(1).fillna(0.0) * px_ret - cost_per_turn * turn


def _split_metrics(returns: pd.Series, *, n_trials: int) -> dict[str, float]:
    cut = int(len(returns) * (1.0 - OOS_FRACTION))
    is_ret = returns.iloc[:cut]
    oos_ret = returns.iloc[cut:]
    return {
        "is_sharpe": sharpe_annualised(is_ret),
        "is_deflated_sharpe": deflated_sharpe(
            sharpe_annualised(is_ret),
            n_trials=n_trials,
            n_obs=len(is_ret),
        ),
        "is_max_drawdown": max_drawdown(is_ret),
        "is_total_return": total_return(is_ret),
        "oos_sharpe": sharpe_annualised(oos_ret),
        "oos_deflated_sharpe": deflated_sharpe(
            sharpe_annualised(oos_ret),
            n_trials=n_trials,
            n_obs=len(oos_ret),
        ),
        "oos_max_drawdown": max_drawdown(oos_ret),
        "oos_total_return": total_return(oos_ret),
    }


def _is_fold_metrics(returns: pd.Series, *, n_folds: int = 5) -> dict[str, float]:
    cut = int(len(returns) * (1.0 - OOS_FRACTION))
    is_ret = returns.iloc[:cut]
    if len(is_ret) < n_folds * 20:
        fold_sharpes = [sharpe_annualised(is_ret)]
        fold_returns = [total_return(is_ret)]
        fold_dds = [max_drawdown(is_ret)]
    else:
        fold_sharpes: list[float] = []
        fold_returns: list[float] = []
        fold_dds: list[float] = []
        fold_size = len(is_ret) // n_folds
        for fold in range(n_folds):
            start = fold * fold_size
            end = len(is_ret) if fold == n_folds - 1 else (fold + 1) * fold_size
            window = is_ret.iloc[start:end]
            if len(window) < 2:
                continue
            fold_sharpes.append(sharpe_annualised(window))
            fold_returns.append(total_return(window))
            fold_dds.append(max_drawdown(window))

    sharpes = np.array(fold_sharpes, dtype=float)
    returns_arr = np.array(fold_returns, dtype=float)
    dds = np.array(fold_dds, dtype=float)
    return {
        "is_fold_sharpe_mean": float(sharpes.mean()) if len(sharpes) else 0.0,
        "is_fold_sharpe_min": float(sharpes.min()) if len(sharpes) else 0.0,
        "is_fold_positive_frac": float((returns_arr > 0.0).mean()) if len(returns_arr) else 0.0,
        "is_fold_worst_drawdown": float(dds.min()) if len(dds) else 0.0,
    }


def _fit_weights(
    symbol_metrics: dict[str, dict[str, float]],
    *,
    max_weight: float | None = None,
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for symbol, metrics in symbol_metrics.items():
        sharpe = max(0.0, metrics["is_sharpe"])
        drawdown = max(abs(metrics["is_max_drawdown"]), 0.10)
        raw[symbol] = sharpe / drawdown + 0.25

    total = sum(raw.values())
    if total <= 0.0:
        n = len(symbol_metrics)
        weights = {symbol: 1.0 / n for symbol in symbol_metrics}
    else:
        weights = {symbol: value / total for symbol, value in raw.items()}

    if max_weight is not None and 0.0 < max_weight < 1.0:
        n = len(weights)
        # Iteratively cap and redistribute to other symbols.
        for _ in range(n):
            over = {s: w for s, w in weights.items() if w > max_weight + 1e-12}
            if not over:
                break
            excess = sum(w - max_weight for w in over.values())
            for s in over:
                weights[s] = max_weight
            receivers = [s for s in weights if s not in over]
            if not receivers:
                break
            share = excess / len(receivers)
            for s in receivers:
                weights[s] = weights[s] + share
        total = sum(weights.values())
        if total > 0.0:
            weights = {s: w / total for s, w in weights.items()}
    return weights


def evaluate_candidate(
    bars_by_symbol: dict[str, pd.DataFrame],
    spec: CandidateSpec,
    *,
    n_trials: int,
    features_by_symbol: dict[str, pd.DataFrame] | None = None,
    max_symbol_weight: float | None = None,
) -> CandidateResult:
    positions = build_positions(
        bars_by_symbol,
        spec,
        features_by_symbol=features_by_symbol,
    )

    symbol_returns: dict[str, pd.Series] = {}
    symbol_metrics: dict[str, dict[str, float]] = {}
    turnovers: list[float] = []
    for symbol, frame in bars_by_symbol.items():
        position = positions[symbol].reindex(frame.index).fillna(0.0).clip(-1.0, 1.0)
        returns = _returns_from_position(frame["close"], position)
        symbol_returns[symbol] = returns
        metrics = _split_metrics(returns, n_trials=n_trials)
        metrics["turnover"] = float(position.diff().abs().mean())
        symbol_metrics[symbol] = metrics
        turnovers.append(metrics["turnover"])

    weights = _fit_weights(symbol_metrics, max_weight=max_symbol_weight)
    gross = pd.concat(
        [positions[symbol].abs() * weights[symbol] for symbol in positions],
        axis=1,
    ).sum(axis=1)
    if float(gross.max()) > MAX_GROSS + 1e-9:
        raise ValueError(f"gross exposure exceeds {MAX_GROSS}")

    portfolio_returns = sum(
        symbol_returns[symbol] * weights[symbol]
        for symbol in bars_by_symbol
    )
    portfolio_metrics = _split_metrics(portfolio_returns, n_trials=n_trials)
    portfolio_metrics.update(_is_fold_metrics(portfolio_returns))

    stressed_returns_1p5x = sum(
        _returns_from_position_with_cost(
            bars_by_symbol[symbol]["close"],
            positions[symbol],
            cost_per_turn=COST_PER_TURN * 1.5,
        )
        * weights[symbol]
        for symbol in bars_by_symbol
    )
    stressed_returns_2x = sum(
        _returns_from_position_with_cost(
            bars_by_symbol[symbol]["close"],
            positions[symbol],
            cost_per_turn=COST_PER_TURN * 2.0,
        )
        * weights[symbol]
        for symbol in bars_by_symbol
    )

    cut = int(len(portfolio_returns) * (1.0 - OOS_FRACTION))
    portfolio_metrics["is_sharpe_cost_1p5x"] = sharpe_annualised(stressed_returns_1p5x.iloc[:cut])
    portfolio_metrics["is_sharpe_cost_2x"] = sharpe_annualised(stressed_returns_2x.iloc[:cut])
    portfolio_metrics["mean_turnover"] = float(np.mean(turnovers))
    portfolio_metrics["max_gross"] = float(gross.max())
    portfolio_metrics["max_weight"] = float(max(weights.values()))

    score = (
        0.35 * portfolio_metrics["is_fold_sharpe_mean"]
        + 0.25 * portfolio_metrics["is_fold_sharpe_min"]
        + 0.35 * portfolio_metrics["is_fold_positive_frac"]
        + 0.20 * min(
            portfolio_metrics["is_sharpe_cost_1p5x"],
            portfolio_metrics["is_sharpe_cost_2x"],
        )
        + 0.10 * portfolio_metrics["is_total_return"]
        - 0.45 * abs(portfolio_metrics["is_max_drawdown"])
        - 1.50 * portfolio_metrics["mean_turnover"]
        - 0.35 * portfolio_metrics["max_weight"]
    )
    return CandidateResult(
        spec=spec,
        weights=weights,
        symbol_metrics=symbol_metrics,
        portfolio_metrics=portfolio_metrics,
        score=float(score),
        positions=positions,
    )


def build_default_candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            name="trend_24_96_e075",
            family="trend",
            params={
                "fast": 24,
                "slow": 96,
                "confirm": 72,
                "enter": 0.75,
                "exit": 0.20,
                "target_vol": 0.006,
                "vol_cap": 0.030,
            },
            notes="Fast/slow EMA trend with hysteresis.",
        ),
        CandidateSpec(
            name="trend_48_168_e075",
            family="trend",
            params={
                "fast": 48,
                "slow": 168,
                "confirm": 72,
                "enter": 0.75,
                "exit": 0.20,
                "target_vol": 0.006,
                "vol_cap": 0.028,
            },
            notes="Slower trend core.",
        ),
        CandidateSpec(
            name="trend_48_168_e100",
            family="trend",
            params={
                "fast": 48,
                "slow": 168,
                "confirm": 96,
                "enter": 1.00,
                "exit": 0.25,
                "target_vol": 0.006,
                "vol_cap": 0.028,
            },
            notes="Higher-conviction trend.",
        ),
        CandidateSpec(
            name="trend_72_240_e075",
            family="trend",
            params={
                "fast": 72,
                "slow": 240,
                "confirm": 96,
                "enter": 0.75,
                "exit": 0.20,
                "target_vol": 0.005,
                "vol_cap": 0.026,
            },
            notes="Slowest trend variant.",
        ),
        CandidateSpec(
            name="pull_24_168_e125",
            family="pullback",
            params={
                "bias_fast": 24,
                "bias_slow": 168,
                "stretch_span": 24,
                "enter": 1.25,
                "exit": 0.25,
                "target_vol": 0.005,
                "vol_cap": 0.028,
            },
            notes="Short-horizon pullback in slow trend.",
        ),
        CandidateSpec(
            name="pull_24_168_e150",
            family="pullback",
            params={
                "bias_fast": 24,
                "bias_slow": 168,
                "stretch_span": 24,
                "enter": 1.50,
                "exit": 0.35,
                "target_vol": 0.005,
                "vol_cap": 0.028,
            },
            notes="Deeper pullback entries.",
        ),
        CandidateSpec(
            name="pull_48_240_e125",
            family="pullback",
            params={
                "bias_fast": 48,
                "bias_slow": 240,
                "stretch_span": 48,
                "enter": 1.25,
                "exit": 0.25,
                "target_vol": 0.005,
                "vol_cap": 0.026,
            },
            notes="Slower bias pullback.",
        ),
        CandidateSpec(
            name="pull_48_240_e150",
            family="pullback",
            params={
                "bias_fast": 48,
                "bias_slow": 240,
                "stretch_span": 48,
                "enter": 1.50,
                "exit": 0.35,
                "target_vol": 0.005,
                "vol_cap": 0.026,
            },
            notes="Higher-threshold slower pullback.",
        ),
        CandidateSpec(
            name="rel_72_s12_sp05",
            family="relative",
            params={
                "lookback": 72,
                "smooth": 12,
                "enter_spread": 0.50,
                "exit_spread": 0.20,
                "switch_buffer": 0.15,
                "target_vol": 0.006,
            },
            notes="Medium-horizon relative momentum.",
        ),
        CandidateSpec(
            name="rel_72_s24_sp05",
            family="relative",
            params={
                "lookback": 72,
                "smooth": 24,
                "enter_spread": 0.50,
                "exit_spread": 0.20,
                "switch_buffer": 0.15,
                "target_vol": 0.006,
            },
            notes="Smoothed relative momentum.",
        ),
        CandidateSpec(
            name="rel_96_s24_sp06",
            family="relative",
            params={
                "lookback": 96,
                "smooth": 24,
                "enter_spread": 0.60,
                "exit_spread": 0.25,
                "switch_buffer": 0.20,
                "target_vol": 0.006,
            },
            notes="Longer lookback relative strength.",
        ),
        CandidateSpec(
            name="rel_168_s24_sp04",
            family="relative",
            params={
                "lookback": 168,
                "smooth": 24,
                "enter_spread": 0.40,
                "exit_spread": 0.15,
                "switch_buffer": 0.10,
                "target_vol": 0.005,
            },
            notes="Week-scale relative strength.",
        ),
        CandidateSpec(
            name="rel_168_s48_sp05",
            family="relative",
            params={
                "lookback": 168,
                "smooth": 48,
                "enter_spread": 0.50,
                "exit_spread": 0.20,
                "switch_buffer": 0.10,
                "target_vol": 0.005,
            },
            notes="Slowest relative variant.",
        ),
    ]


def build_refined_candidate_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = list(build_default_candidate_specs())

    for lookback in (84, 96, 120, 144):
        for smooth in (18, 24, 36):
            for enter_spread in (0.45, 0.55, 0.65):
                exit_spread = 0.20 if enter_spread <= 0.55 else 0.25
                for switch_buffer in (0.10, 0.15):
                    for target_vol in (0.005, 0.006):
                        name = (
                            f"rel_lb{lookback}_sm{smooth}_en{int(round(enter_spread * 100)):02d}"
                            f"_sb{int(round(switch_buffer * 100)):02d}_tv{int(round(target_vol * 1000)):03d}"
                        )
                        specs.append(
                            CandidateSpec(
                                name=name,
                                family="relative",
                                params={
                                    "lookback": lookback,
                                    "smooth": smooth,
                                    "enter_spread": enter_spread,
                                    "exit_spread": exit_spread,
                                    "switch_buffer": switch_buffer,
                                    "target_vol": target_vol,
                                },
                                notes="Refined relative-strength search.",
                            )
                        )

    for fast, slow in ((24, 96), (24, 120), (48, 168)):
        for confirm in (72, 96):
            for enter in (0.65, 0.75, 0.90):
                for exit_thr in (0.15, 0.20, 0.25):
                    for target_vol in (0.005, 0.006):
                        name = (
                            f"trend_f{fast}_s{slow}_c{confirm}_e{int(round(enter * 100)):02d}"
                            f"_x{int(round(exit_thr * 100)):02d}_tv{int(round(target_vol * 1000)):03d}"
                        )
                        specs.append(
                            CandidateSpec(
                                name=name,
                                family="trend",
                                params={
                                    "fast": fast,
                                    "slow": slow,
                                    "confirm": confirm,
                                    "enter": enter,
                                    "exit": exit_thr,
                                    "target_vol": target_vol,
                                    "vol_cap": 0.03 if target_vol >= 0.006 else 0.028,
                                },
                                notes="Refined trend search.",
                            )
                        )

    deduped: dict[str, CandidateSpec] = {}
    for spec in specs:
        deduped[spec.name] = spec
    return list(deduped.values())


def run_research(
    bars_by_symbol: dict[str, pd.DataFrame],
    specs: list[CandidateSpec] | None = None,
    *,
    n_trials: int | None = None,
) -> tuple[CandidateResult, list[CandidateResult]]:
    candidate_specs = list(specs) if specs is not None else build_default_candidate_specs()
    effective_trials = int(n_trials if n_trials is not None else len(candidate_specs))
    features_by_symbol = build_feature_cache(bars_by_symbol)
    results = [
        evaluate_candidate(
            bars_by_symbol,
            spec,
            n_trials=effective_trials,
            features_by_symbol=features_by_symbol,
        )
        for spec in candidate_specs
    ]
    results.sort(
        key=lambda result: (
            result.score,
            result.portfolio_metrics["is_deflated_sharpe"],
            result.portfolio_metrics["is_total_return"],
        ),
        reverse=True,
    )
    return results[0], results


def write_research_report(
    report_path: str | Path,
    *,
    best: CandidateResult,
    ranked_results: list[CandidateResult],
    include_oos: bool = False,
    n_trials: int | None = None,
) -> Path:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _now_iso(),
        "n_trials": int(n_trials if n_trials is not None else len(ranked_results)),
        "selected": best.to_dict(include_oos=include_oos),
        "ranked": [result.to_dict(include_oos=include_oos) for result in ranked_results],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_submission(
    submission_dir: str | Path,
    *,
    engine_name: str,
    best: CandidateResult,
    notes: str,
    n_trials: int,
) -> Path:
    root = Path(submission_dir).expanduser().resolve()
    signal_dir = root / "signals"
    signal_dir.mkdir(parents=True, exist_ok=True)

    for symbol, position in best.positions.items():
        flat = symbol.replace("/", "_")
        frame = pd.DataFrame({"position": position.clip(-1.0, 1.0).astype(float)})
        if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
            frame.index = frame.index.tz_convert("UTC").tz_localize(None)
        frame.to_parquet(signal_dir / f"{flat}.parquet")

    submission = {
        "engine": engine_name,
        "commit": _git_commit(),
        "submitted_at": _now_iso(),
        "symbols": list(best.positions),
        "weights": best.weights,
        "n_trials": n_trials,
        "notes": notes,
    }
    (root / "submission.json").write_text(json.dumps(submission, indent=2))
    return root
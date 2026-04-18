"""
Pure data layer for the live dashboard.

All functions in this module are side-effect-free (apart from
filesystem reads) and contain no Streamlit calls — so they can be
imported and tested without spinning up a Streamlit session.

``scripts/live_dashboard.py`` is a thin rendering shell that imports
from here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default state paths. Callers may override by passing a ``base_dir``.
_DEFAULT_BASE = Path("fund_data")

DEFAULT_ASSETS: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]


# --------------------------------------------------------------------------
# Loaders — each returns a "safe" default on any error.
# --------------------------------------------------------------------------
def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def load_state(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    return _load_json(base_dir / "live_state.json", {})


def load_journal(base_dir: Path = _DEFAULT_BASE) -> list[dict[str, Any]]:
    data = _load_json(base_dir / "trade_journal.json", [])
    return data if isinstance(data, list) else []


def load_divergence(base_dir: Path = _DEFAULT_BASE) -> list[dict[str, Any]]:
    data = _load_json(base_dir / "divergence_log.json", [])
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("comparisons", [])
    return []


def load_market_snapshot(base_dir: Path = _DEFAULT_BASE) -> dict[str, Any]:
    data = _load_json(base_dir / "market_snapshot.json", {})
    if isinstance(data, dict):
        data = dict(data)  # don't mutate cached value
        data.pop("_timestamp", None)
        return data
    return {}


# --------------------------------------------------------------------------
# Derived metrics
# --------------------------------------------------------------------------
def portfolio_summary(state: dict[str, Any], journal: list[dict[str, Any]]) -> dict[str, Any]:
    """Condense live-state + journal into a single metrics bundle."""
    capital = float(state.get("capital", 10_000.0))
    initial = float(state.get("initial_capital", 10_000.0))
    ret = (capital - initial) / initial if initial > 0 else 0.0
    return {
        "capital": capital,
        "initial_capital": initial,
        "return_pct": ret,
        "n_open": len(state.get("open_positions", []) or []),
        "n_closed": len(journal or []),
        "iteration": int(state.get("iteration", 0)),
        "paper_mode": bool(state.get("paper_mode", True)),
    }


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def compute_signal_proximity(symbol: str, snap: dict[str, Any]) -> dict[str, float]:
    """How close each strategy is to firing on this asset (∈ [0, 1]).

    The values here are **UI heuristics**, not real trigger probabilities —
    they're scaled so that ``1.0`` means "at threshold". The definitions
    mirror the production strategies as of v4.0.
    """
    if not isinstance(snap, dict) or snap.get("error"):
        return {
            "funding_mr_v7": 0.0,
            "extreme_spike": 0.0,
            "fund_vol_squeeze": 0.0,
            "momentum_breakout": 0.0,
        }

    fz = abs(float(snap.get("funding_zscore", 0) or 0))
    regime = str(snap.get("regime", "") or "")
    bb = float(snap.get("bb_pctile", 50) or 50)
    vol = float(snap.get("vol_ratio", 1) or 1)
    atr_exp = float(snap.get("atr_exp", 1) or 1)
    price = float(snap.get("price", 0) or 0)
    ch_high = float(snap.get("ch_high", 0) or 0)
    ch_low = float(snap.get("ch_low", 0) or 0)

    prox: dict[str, float] = {}

    # funding_mr_v7: needs |z| >= 3.0 on any asset.
    prox["funding_mr_v7"] = _clip(fz / 3.0)

    # extreme_spike: needs |z| >= 4.0 + high_vol regime. BTC excluded.
    base = _clip(fz / 4.0)
    regime_ok = 1.0 if regime == "high_volatility" else 0.5
    prox["extreme_spike"] = 0.0 if symbol == "BTC/USDT" else base * regime_ok

    # fund_vol_squeeze: needs bb_pctile <= 10 + |z| >= 2.0; SOL/XRP only.
    if symbol in {"BTC/USDT", "ETH/USDT"}:
        prox["fund_vol_squeeze"] = 0.0
    else:
        sq = _clip((10 - bb) / 10.0)
        fz2 = _clip(fz / 2.0)
        prox["fund_vol_squeeze"] = min(sq, fz2)

    # momentum_breakout: breakout + ATR expansion + volume; ETH only.
    if symbol == "ETH/USDT" and ch_high > ch_low:
        # Distance from extremes, scaled to 1.0 at the breakout levels.
        edge = max(price - ch_low, ch_high - price) / max(ch_high - ch_low, 1e-10)
        breakout = (
            1.0
            if (price > ch_high * 0.999 or price < ch_low * 1.001)
            else _clip(edge * 2)
        )
        atr_pct = _clip(atr_exp / 1.5)
        vol_pct = _clip(vol / 1.3)
        prox["momentum_breakout"] = min(breakout, atr_pct, vol_pct)
    else:
        prox["momentum_breakout"] = 0.0

    return prox


def proximity_matrix(
    assets: list[str],
    snapshots: dict[str, Any],
    strategies: list[str] | None = None,
) -> tuple[list[str], list[list[float]], list[str]]:
    """Build a (assets × strategies) grid of proximity scores.

    Returns ``(asset_labels, matrix, strategy_keys)`` — ``matrix[i][j]`` is
    the proximity of strategy ``j`` on asset ``i``.
    """
    strategies = strategies or [
        "funding_mr_v7",
        "extreme_spike",
        "fund_vol_squeeze",
        "momentum_breakout",
    ]
    labels: list[str] = []
    matrix: list[list[float]] = []
    for sym in assets:
        labels.append(sym.split("/")[0])
        prox = compute_signal_proximity(sym, snapshots.get(sym, {}))
        matrix.append([prox.get(s, 0.0) for s in strategies])
    return labels, matrix, strategies


__all__ = [
    "DEFAULT_ASSETS",
    "load_state",
    "load_journal",
    "load_divergence",
    "load_market_snapshot",
    "portfolio_summary",
    "compute_signal_proximity",
    "proximity_matrix",
]

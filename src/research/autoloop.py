"""
Autonomous research loop.

End-to-end research agent:

    Hypothesis  →  Feature synthesis  →  Signal compile  →
    Walk-forward + deflated Sharpe  →  Gate  →
    {register via src.registry  |  log null result}

The loop is intentionally modular: each stage is a pure function that
takes a :class:`Hypothesis` and returns a :class:`Candidate` (or a
rejection). That makes it easy to swap in alternate feature generators
or gating rules later.

Deflated Sharpe ratio (Bailey & López de Prado, 2014) corrects the
naïve IS-Sharpe for the number of strategies tried — the standard
defence against datamining bias when you're running many hypotheses.

Design constraints
------------------
- No network calls. Operates on pre-cached OHLCV already on disk.
- Deterministic: same seed → same verdicts.
- Uses existing :class:`src.backtest.walk_forward.walk_forward` for OOS,
  existing :class:`src.registry.StrategyRegistry` for persistence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Hypothesis grammar
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Hypothesis:
    """A research hypothesis expressible as {feature, op, threshold, side}.

    ``feature``  : column name in the feature frame (e.g. "zmom_20").
    ``op``       : ``>`` | ``<`` | ``>=`` | ``<=``.
    ``threshold``: numeric cut-off.
    ``side``     : +1 (go long when condition true) | -1 (go short).
    ``name``     : short id for registry / logs.
    """

    name: str
    feature: str
    op: str
    threshold: float
    side: int

    def describe(self) -> str:
        return f"{self.name}: {self.feature} {self.op} {self.threshold:+g} -> side={self.side:+d}"


def _apply_op(series: pd.Series, op: str, thr: float) -> pd.Series:
    if op == ">":
        return series > thr
    if op == "<":
        return series < thr
    if op == ">=":
        return series >= thr
    if op == "<=":
        return series <= thr
    raise ValueError(f"unsupported op {op!r}")


def compile_signal(hypothesis: Hypothesis) -> Callable[[pd.DataFrame], pd.Series]:
    """Turn a :class:`Hypothesis` into a signal function for the backtester.

    The returned function consumes a feature-enriched frame and emits
    a {-1, 0, +1} series of the same length.
    """
    h = hypothesis

    def _signal(df: pd.DataFrame) -> pd.Series:
        if h.feature not in df.columns:
            return pd.Series(0, index=df.index, dtype=int)
        mask = _apply_op(df[h.feature], h.op, h.threshold).fillna(False)
        out = pd.Series(0, index=df.index, dtype=int)
        out[mask] = int(h.side)
        return out

    return _signal


# --------------------------------------------------------------------------
# Feature synthesis
# --------------------------------------------------------------------------
def synthesize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a canonical set of features expected by :func:`generate_hypotheses`.

    Non-destructive: returns a copy with added columns.
    """
    out = df.copy()
    close = out["close"].astype(float)

    # Log returns.
    logret = np.log(close).diff()
    out["ret_1"] = logret
    out["ret_5"] = logret.rolling(5).sum()
    out["ret_20"] = logret.rolling(20).sum()

    # Z-scored momentum.
    for w in (20, 50):
        mean = logret.rolling(w).mean()
        std = logret.rolling(w).std(ddof=0)
        out[f"zmom_{w}"] = (logret - mean) / std.replace(0, np.nan)

    # Volatility ratio (short / long).
    short_vol = logret.rolling(10).std(ddof=0)
    long_vol = logret.rolling(50).std(ddof=0)
    out["vol_ratio"] = (short_vol / long_vol).replace([np.inf, -np.inf], np.nan)

    # Range expansion.
    out["range_pct"] = (out["high"] - out["low"]) / close
    out["range_ratio"] = out["range_pct"] / out["range_pct"].rolling(20).mean()

    return out


def generate_hypotheses(families: Iterable[str] | None = None) -> list[Hypothesis]:
    """Enumerate a grid of concrete hypotheses across feature families.

    Families:
        * ``momentum``  — long when zmom > k, short when < -k
        * ``meanrev``   — short when zmom > k, long when < -k
        * ``volexpand`` — long after range_ratio > k
        * ``breakout``  — long after ret_5 > k

    Roughly 3 thresholds × 2 sides × 4 families = ~24 candidates.
    """
    families = tuple(families) if families is not None else ("momentum", "meanrev", "volexpand", "breakout")
    out: list[Hypothesis] = []

    mom_thr = (1.0, 1.5, 2.0)
    ve_thr = (1.3, 1.6, 2.0)
    bo_thr = (0.02, 0.04, 0.06)

    def _lbl(x: float) -> str:
        # Strategy IDs must be alnum/underscore/hyphen — no dots.
        return f"{x:g}".replace(".", "p").replace("-", "m")

    if "momentum" in families:
        for t in mom_thr:
            out.append(Hypothesis(f"mom_long_{_lbl(t)}", "zmom_20", ">", t, +1))
            out.append(Hypothesis(f"mom_short_{_lbl(t)}", "zmom_20", "<", -t, -1))
    if "meanrev" in families:
        for t in mom_thr:
            out.append(Hypothesis(f"mr_long_{_lbl(t)}", "zmom_20", "<", -t, +1))
            out.append(Hypothesis(f"mr_short_{_lbl(t)}", "zmom_20", ">", t, -1))
    if "volexpand" in families:
        for t in ve_thr:
            out.append(Hypothesis(f"volexp_long_{_lbl(t)}", "range_ratio", ">", t, +1))
    if "breakout" in families:
        for t in bo_thr:
            out.append(Hypothesis(f"bo_long_{_lbl(t)}", "ret_5", ">", t, +1))
            out.append(Hypothesis(f"bo_short_{_lbl(t)}", "ret_5", "<", -t, -1))
    return out


# --------------------------------------------------------------------------
# Deflated Sharpe ratio
# --------------------------------------------------------------------------
def deflated_sharpe(
    observed_sr: float,
    n_trials: int,
    n_periods: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Bailey & López de Prado deflated Sharpe.

    Returns the z-score under the null that the true Sharpe is zero,
    after correcting for the number of hypotheses tested.

    ``kurtosis`` is *raw*, not excess; normal = 3.
    """
    if n_periods <= 1 or n_trials <= 0:
        return float("nan")
    # Expected max Sharpe under the null (variance of trials).
    EULER = 0.5772156649
    exp_max = math.sqrt(2.0 * math.log(max(n_trials, 1)))
    if n_trials >= 2:
        exp_max -= (EULER + math.log(math.log(max(n_trials, 2)))) / (
            2.0 * math.sqrt(2.0 * math.log(max(n_trials, 2)))
        )
    denom_var = 1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr**2
    denom_var = max(denom_var, 1e-12)
    denom = math.sqrt(denom_var) / math.sqrt(max(n_periods - 1, 1))
    return float((observed_sr - exp_max * denom) / max(denom, 1e-12))


# --------------------------------------------------------------------------
# Candidate / gating
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Gates:
    """Acceptance thresholds applied after walk-forward OOS evaluation."""

    min_oos_sharpe: float = 0.8
    min_frac_positive: float = 0.6
    max_drawdown: float = 0.25
    min_trades: int = 30
    min_deflated_sharpe_z: float = 1.65   # ~5% one-sided


@dataclass(frozen=True)
class Candidate:
    """Evaluation result for one hypothesis."""

    hypothesis: Hypothesis
    oos_sharpe: float
    frac_positive: float
    pooled_return: float
    worst_max_dd: float
    n_trades: int
    deflated_sharpe_z: float
    verdict: str                  # "ACCEPT" | "REJECT"
    reason: str

    def to_dict(self) -> dict:
        d = {
            "name": self.hypothesis.name,
            "feature": self.hypothesis.feature,
            "op": self.hypothesis.op,
            "threshold": self.hypothesis.threshold,
            "side": self.hypothesis.side,
            "oos_sharpe": self.oos_sharpe,
            "frac_positive": self.frac_positive,
            "pooled_return": self.pooled_return,
            "worst_max_dd": self.worst_max_dd,
            "n_trades": self.n_trades,
            "deflated_sharpe_z": self.deflated_sharpe_z,
            "verdict": self.verdict,
            "reason": self.reason,
        }
        return d


def _gate(cand: Candidate, gates: Gates) -> Candidate:
    reasons: list[str] = []
    if cand.oos_sharpe < gates.min_oos_sharpe:
        reasons.append(f"oos_sharpe {cand.oos_sharpe:.2f}<{gates.min_oos_sharpe:.2f}")
    if cand.frac_positive < gates.min_frac_positive:
        reasons.append(f"frac_positive {cand.frac_positive:.2f}<{gates.min_frac_positive:.2f}")
    if cand.worst_max_dd > gates.max_drawdown:
        reasons.append(f"worst_max_dd {cand.worst_max_dd:.2f}>{gates.max_drawdown:.2f}")
    if cand.n_trades < gates.min_trades:
        reasons.append(f"n_trades {cand.n_trades}<{gates.min_trades}")
    if cand.deflated_sharpe_z < gates.min_deflated_sharpe_z:
        reasons.append(
            f"deflated_sharpe_z {cand.deflated_sharpe_z:.2f}<{gates.min_deflated_sharpe_z:.2f}"
        )
    verdict = "ACCEPT" if not reasons else "REJECT"
    reason = "ok" if not reasons else "; ".join(reasons)
    return replace(cand, verdict=verdict, reason=reason)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def _simple_backtest_returns(
    df: pd.DataFrame, signal_func: Callable[[pd.DataFrame], pd.Series]
) -> pd.Series:
    """Lightweight next-bar return series from a signal (for deflated SR).

    Returns are ``signal.shift(1) * log_return``. No costs, no sizing:
    this is purely for statistical sharpness, not P&L.
    """
    ret = np.log(df["close"].astype(float)).diff().fillna(0.0)
    sig = signal_func(df).reindex(df.index).fillna(0).astype(int)
    return sig.shift(1).fillna(0) * ret


def _ann_sharpe(returns: pd.Series, bars_per_year: int) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=0) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=0) * math.sqrt(bars_per_year))


@dataclass
class AutoLoopResult:
    """Aggregate outcome of a research batch."""

    n_candidates: int
    n_accepted: int
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def accepted(self) -> list[Candidate]:
        return [c for c in self.candidates if c.verdict == "ACCEPT"]

    def to_dict(self) -> dict:
        return {
            "n_candidates": self.n_candidates,
            "n_accepted": self.n_accepted,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def run_auto_loop(
    df: pd.DataFrame,
    hypotheses: Iterable[Hypothesis] | None = None,
    *,
    n_folds: int = 5,
    anchored: bool = True,
    bars_per_year: int = 24 * 365,
    gates: Gates | None = None,
    registry_path: str | Path | None = None,
) -> AutoLoopResult:
    """Run the full research loop on a pre-loaded OHLCV frame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV with DatetimeIndex and at least ``open, high, low, close, volume``.
    hypotheses : iterable of Hypothesis, optional
        Defaults to :func:`generate_hypotheses()`.
    n_folds : int
    anchored : bool
        Walk-forward fold scheme.
    bars_per_year : int
        Annualisation factor for Sharpe (default: 1h crypto → 8760).
    gates : Gates
    registry_path : str | Path | None
        If provided, accepted candidates are appended to the registry.

    Notes
    -----
    Uses a simplified next-bar backtest inside the loop for speed —
    the full :class:`Backtester` can be plugged in separately for
    production-grade P&L once a hypothesis clears this gate.
    """
    from src.backtest.walk_forward import make_folds

    hypos = list(hypotheses) if hypotheses is not None else generate_hypotheses()
    gates = gates or Gates()

    feats = synthesize_features(df)
    folds = make_folds(feats.index, n_folds=n_folds, anchored=anchored)
    n_trials = len(hypos)

    candidates: list[Candidate] = []
    for h in hypos:
        signal_func = compile_signal(h)
        fold_sharpes: list[float] = []
        fold_returns: list[pd.Series] = []
        for fold in folds:
            sub = feats.loc[fold.test_start : fold.test_end]
            if len(sub) < 2:
                continue
            r = _simple_backtest_returns(sub, signal_func)
            fold_returns.append(r)
            fold_sharpes.append(_ann_sharpe(r, bars_per_year))

        if not fold_sharpes:
            candidates.append(
                Candidate(
                    hypothesis=h, oos_sharpe=0.0, frac_positive=0.0,
                    pooled_return=0.0, worst_max_dd=1.0, n_trades=0,
                    deflated_sharpe_z=float("nan"),
                    verdict="REJECT", reason="no folds evaluated",
                )
            )
            continue

        all_r = pd.concat(fold_returns).dropna()
        oos_sr = float(np.mean(fold_sharpes))
        frac_pos = float(np.mean([s > 0 for s in fold_sharpes]))
        pooled_ret = float(np.expm1(all_r.sum()))
        eq = all_r.cumsum().apply(np.exp)
        worst_dd = float(((eq.cummax() - eq) / eq.cummax()).max()) if len(eq) else 1.0

        # Count "trades" as signal transitions.
        sig = signal_func(feats).fillna(0).astype(int)
        trades = int((sig.diff().fillna(0) != 0).sum())

        d_sr_z = deflated_sharpe(
            observed_sr=oos_sr,
            n_trials=n_trials,
            n_periods=len(all_r),
            skew=float(all_r.skew()) if len(all_r) > 2 else 0.0,
            kurtosis=float(all_r.kurtosis() + 3.0) if len(all_r) > 2 else 3.0,
        )

        cand = Candidate(
            hypothesis=h,
            oos_sharpe=oos_sr,
            frac_positive=frac_pos,
            pooled_return=pooled_ret,
            worst_max_dd=worst_dd,
            n_trades=trades,
            deflated_sharpe_z=float(d_sr_z) if not math.isnan(d_sr_z) else 0.0,
            verdict="ACCEPT",  # overridden by _gate
            reason="",
        )
        candidates.append(_gate(cand, gates))

    result = AutoLoopResult(
        n_candidates=len(candidates),
        n_accepted=sum(1 for c in candidates if c.verdict == "ACCEPT"),
        candidates=candidates,
    )

    if registry_path is not None and result.n_accepted:
        _register_accepted(result, registry_path)

    return result


def _register_accepted(result: AutoLoopResult, registry_path: str | Path) -> None:
    """Append accepted candidates to the NDJSON strategy registry."""
    try:
        from src.registry import StrategyRegistry
    except Exception:
        return
    reg = StrategyRegistry(Path(registry_path))
    for cand in result.accepted:
        try:
            reg.register(
                strategy_id=cand.hypothesis.name,
                params={
                    "feature": cand.hypothesis.feature,
                    "op": cand.hypothesis.op,
                    "threshold": cand.hypothesis.threshold,
                    "side": cand.hypothesis.side,
                    "oos_sharpe": cand.oos_sharpe,
                    "frac_positive": cand.frac_positive,
                    "pooled_return": cand.pooled_return,
                    "worst_max_dd": cand.worst_max_dd,
                    "deflated_sharpe_z": cand.deflated_sharpe_z,
                    "n_trades": cand.n_trades,
                },
                notes=f"auto_loop accepted: {cand.reason}",
                tags=["auto_loop"],
            )
        except Exception:
            # Don't let a flaky registry fail the loop; skip silently.
            continue

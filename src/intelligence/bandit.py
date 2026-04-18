"""
Meta-strategy router — contextual bandit allocator.

Given a vector of regime features (funding z-score, vol ratio, BB pctile,
ATR expansion, etc.) and a history of per-strategy rewards, decide which
strategy to run this bar — or how to mix them.

Implements:

- :class:`LinUCB`        — Li et al. (2010) disjoint-arm LinUCB. Deterministic
  given history. Pick the arm with the highest upper confidence bound;
  exploration controlled by ``alpha``.
- :class:`ThompsonLinear` — Bayesian linear regression per arm with
  Gaussian-Gamma prior. Sample posterior, pick arg-max. Stochastic.
- :class:`MetaRouter`    — thin facade with ``update(context, arm, reward)``
  and ``allocate(context) -> dict[str, float]`` that returns normalised
  weights across arms (softmax over UCB scores, temperature-controlled).

This is an *offline-trainable, online-deployable* router: feed
walk-forward fold rewards to ``update()`` to pre-train, then call
``allocate()`` each bar in production. Fully deterministic in LinUCB mode
so its decisions are reproducible from a stored policy snapshot.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------------
# Linear-UCB
# --------------------------------------------------------------------------
@dataclass
class LinUCB:
    """Disjoint-arm LinUCB contextual bandit.

    One independent linear model per arm. ``score(ctx) = theta·ctx +
    alpha * sqrt(ctx·A⁻¹·ctx)`` — the UCB. Arms with little data have
    big confidence intervals so they get explored; arms with lots of
    data converge to pure exploitation.
    """

    arms: list[str]
    d: int                      # context dimension
    alpha: float = 1.0          # exploration weight
    ridge: float = 1.0          # ridge prior strength

    def __post_init__(self) -> None:
        self._A: dict[str, np.ndarray] = {
            a: self.ridge * np.eye(self.d) for a in self.arms
        }
        self._b: dict[str, np.ndarray] = {a: np.zeros(self.d) for a in self.arms}

    # ----- core operations ------------------------------------------------
    def _theta(self, arm: str) -> np.ndarray:
        return np.linalg.solve(self._A[arm], self._b[arm])

    def score(self, arm: str, context: np.ndarray) -> float:
        ctx = np.asarray(context, dtype=float).reshape(-1)
        if ctx.shape[0] != self.d:
            raise ValueError(f"context dim {ctx.shape[0]} != d={self.d}")
        A_inv_x = np.linalg.solve(self._A[arm], ctx)
        mean = self._theta(arm) @ ctx
        conf = self.alpha * float(np.sqrt(max(ctx @ A_inv_x, 0.0)))
        return float(mean + conf)

    def scores(self, context: np.ndarray) -> dict[str, float]:
        return {a: self.score(a, context) for a in self.arms}

    def select(self, context: np.ndarray) -> str:
        s = self.scores(context)
        # Stable deterministic tie-break by arm name.
        return max(s.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def update(self, arm: str, context: np.ndarray, reward: float) -> None:
        if arm not in self._A:
            raise KeyError(f"unknown arm {arm!r}")
        ctx = np.asarray(context, dtype=float).reshape(-1)
        if ctx.shape[0] != self.d:
            raise ValueError(f"context dim {ctx.shape[0]} != d={self.d}")
        self._A[arm] += np.outer(ctx, ctx)
        self._b[arm] += float(reward) * ctx

    def fit(
        self,
        contexts: np.ndarray,
        arms: Iterable[str],
        rewards: Iterable[float],
    ) -> "LinUCB":
        """Bulk-update from a batch of (ctx, arm, reward) triples."""
        contexts = np.asarray(contexts, dtype=float)
        arms = list(arms)
        rewards = list(rewards)
        if not (contexts.shape[0] == len(arms) == len(rewards)):
            raise ValueError("contexts, arms, rewards must be same length")
        for i, arm in enumerate(arms):
            self.update(arm, contexts[i], rewards[i])
        return self

    # ----- serialization --------------------------------------------------
    def state(self) -> dict:
        return {
            "arms": list(self.arms),
            "d": self.d,
            "alpha": self.alpha,
            "ridge": self.ridge,
            "A": {a: self._A[a].tolist() for a in self.arms},
            "b": {a: self._b[a].tolist() for a in self.arms},
        }

    @classmethod
    def from_state(cls, state: dict) -> "LinUCB":
        inst = cls(
            arms=list(state["arms"]),
            d=int(state["d"]),
            alpha=float(state["alpha"]),
            ridge=float(state["ridge"]),
        )
        for a in inst.arms:
            inst._A[a] = np.asarray(state["A"][a], dtype=float)
            inst._b[a] = np.asarray(state["b"][a], dtype=float)
        return inst

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state()))

    @classmethod
    def load(cls, path: str | Path) -> "LinUCB":
        return cls.from_state(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------
# Thompson sampling (Bayesian linear regression per arm)
# --------------------------------------------------------------------------
@dataclass
class ThompsonLinear:
    """Stochastic counterpart — samples posterior theta, picks arg-max.

    Use when you want the allocator to explore proportionally to
    posterior uncertainty rather than deterministically via a UCB.
    """

    arms: list[str]
    d: int
    ridge: float = 1.0
    noise_sigma: float = 1.0
    rng_seed: int = 7

    def __post_init__(self) -> None:
        self._A: dict[str, np.ndarray] = {
            a: self.ridge * np.eye(self.d) for a in self.arms
        }
        self._b: dict[str, np.ndarray] = {a: np.zeros(self.d) for a in self.arms}
        self._rng = np.random.default_rng(self.rng_seed)

    def _posterior(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        A = self._A[arm]
        mu = np.linalg.solve(A, self._b[arm])
        cov = (self.noise_sigma**2) * np.linalg.inv(A)
        return mu, cov

    def sample_theta(self, arm: str) -> np.ndarray:
        mu, cov = self._posterior(arm)
        return self._rng.multivariate_normal(mu, cov)

    def select(self, context: np.ndarray) -> str:
        ctx = np.asarray(context, dtype=float).reshape(-1)
        scores = {a: float(self.sample_theta(a) @ ctx) for a in self.arms}
        return max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def update(self, arm: str, context: np.ndarray, reward: float) -> None:
        if arm not in self._A:
            raise KeyError(f"unknown arm {arm!r}")
        ctx = np.asarray(context, dtype=float).reshape(-1)
        self._A[arm] += np.outer(ctx, ctx)
        self._b[arm] += float(reward) * ctx


# --------------------------------------------------------------------------
# MetaRouter: softmax allocation over UCB scores
# --------------------------------------------------------------------------
@dataclass
class MetaRouter:
    """Portfolio-style allocator: weights across arms via tempered softmax.

    Unlike :meth:`LinUCB.select` which picks one arm, ``allocate`` returns
    a distribution over arms — useful when capital can be split across
    strategies (e.g. run crowding_mr on SOL and xsec_mom on BTC simultaneously).
    """

    bandit: LinUCB
    temperature: float = 0.25

    def update(self, context: np.ndarray, arm: str, reward: float) -> None:
        self.bandit.update(arm, context, reward)

    def fit(self, contexts: np.ndarray, arms: Iterable[str], rewards: Iterable[float]) -> "MetaRouter":
        self.bandit.fit(contexts, arms, rewards)
        return self

    def allocate(self, context: np.ndarray) -> dict[str, float]:
        scores = self.bandit.scores(context)
        if self.temperature <= 0:
            best = max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
            return {a: (1.0 if a == best else 0.0) for a in self.bandit.arms}
        arr = np.array([scores[a] for a in self.bandit.arms], dtype=float)
        # Numerically stable softmax.
        z = (arr - arr.max()) / self.temperature
        w = np.exp(z)
        w = w / w.sum()
        return {a: float(w[i]) for i, a in enumerate(self.bandit.arms)}

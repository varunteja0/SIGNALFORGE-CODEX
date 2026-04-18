"""Tests for :mod:`src.intelligence.bandit`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.intelligence.bandit import LinUCB, MetaRouter, ThompsonLinear


# --------------------------------------------------------------------------
# LinUCB basics
# --------------------------------------------------------------------------
def test_linucb_select_initially_breaks_ties_by_name():
    b = LinUCB(arms=["alpha", "beta"], d=3)
    # All arms identical priors → equal scores → name tie-break.
    chosen = b.select(np.zeros(3))
    assert chosen == "beta"  # max() by (score, name) picks lexicographic max


def test_linucb_update_rejects_unknown_arm():
    b = LinUCB(arms=["a"], d=2)
    with pytest.raises(KeyError):
        b.update("nope", np.array([1.0, 2.0]), reward=0.5)


def test_linucb_update_rejects_wrong_dim():
    b = LinUCB(arms=["a"], d=3)
    with pytest.raises(ValueError):
        b.update("a", np.array([1.0, 2.0]), reward=0.5)


def test_linucb_converges_to_best_arm_on_clean_linear_rewards():
    """After enough targeted rewards, the arm whose true theta aligns with
    the context should get the highest score."""
    rng = np.random.default_rng(0)
    b = LinUCB(arms=["good", "bad"], d=3, alpha=0.2, ridge=1.0)
    good_theta = np.array([1.0, 0.5, -0.3])
    bad_theta = np.array([-0.5, -0.2, 0.1])
    for _ in range(400):
        ctx = rng.normal(size=3)
        b.update("good", ctx, reward=float(good_theta @ ctx) + 0.01 * rng.normal())
        b.update("bad", ctx, reward=float(bad_theta @ ctx) + 0.01 * rng.normal())
    # For a context aligned with good_theta, "good" should clearly win.
    ctx_test = good_theta / np.linalg.norm(good_theta)
    assert b.select(ctx_test) == "good"
    scores = b.scores(ctx_test)
    assert scores["good"] > scores["bad"]


def test_linucb_fit_and_state_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(1)
    contexts = rng.normal(size=(50, 4))
    arms = ["x" if i % 2 == 0 else "y" for i in range(50)]
    rewards = contexts.sum(axis=1) * np.array([1 if a == "x" else -1 for a in arms])
    b = LinUCB(arms=["x", "y"], d=4, alpha=0.5)
    b.fit(contexts, arms, rewards.tolist())

    path = tmp_path / "state.json"
    b.save(path)
    b2 = LinUCB.load(path)
    # Same decision on a fresh context.
    ctx = np.array([1.0, 2.0, -1.0, 0.5])
    assert b.select(ctx) == b2.select(ctx)


def test_linucb_fit_rejects_length_mismatch():
    b = LinUCB(arms=["a"], d=2)
    with pytest.raises(ValueError):
        b.fit(np.zeros((3, 2)), ["a", "a"], [0.0, 0.0])


# --------------------------------------------------------------------------
# Thompson sampling
# --------------------------------------------------------------------------
def test_thompson_select_returns_valid_arm():
    t = ThompsonLinear(arms=["a", "b", "c"], d=3, rng_seed=1)
    for _ in range(20):
        chosen = t.select(np.random.normal(size=3))
        assert chosen in {"a", "b", "c"}


def test_thompson_prefers_good_arm_after_training():
    rng = np.random.default_rng(2)
    t = ThompsonLinear(arms=["good", "bad"], d=2, rng_seed=3)
    for _ in range(300):
        ctx = rng.normal(size=2)
        t.update("good", ctx, reward=float(1.0 * ctx[0] + 0.5 * ctx[1]))
        t.update("bad", ctx, reward=-float(1.0 * ctx[0] + 0.5 * ctx[1]))
    # Over many samples, 'good' should win substantially more often for
    # contexts aligned with its theta.
    ctx_test = np.array([1.0, 0.5])
    picks = [t.select(ctx_test) for _ in range(50)]
    assert picks.count("good") > picks.count("bad") * 2


# --------------------------------------------------------------------------
# MetaRouter
# --------------------------------------------------------------------------
def test_meta_router_allocate_sums_to_one():
    b = LinUCB(arms=["a", "b", "c"], d=2)
    r = MetaRouter(bandit=b, temperature=0.5)
    w = r.allocate(np.array([1.0, -1.0]))
    assert set(w.keys()) == {"a", "b", "c"}
    assert sum(w.values()) == pytest.approx(1.0)


def test_meta_router_zero_temperature_is_winner_take_all():
    b = LinUCB(arms=["a", "b"], d=2)
    b.update("a", np.array([1.0, 0.0]), reward=5.0)
    r = MetaRouter(bandit=b, temperature=0.0)
    w = r.allocate(np.array([1.0, 0.0]))
    # One arm has full weight; other is zero.
    assert sorted(w.values()) == [0.0, 1.0]


def test_meta_router_temperature_sharpens_distribution():
    b = LinUCB(arms=["a", "b"], d=2)
    b.update("a", np.array([1.0, 0.0]), reward=10.0)
    ctx = np.array([1.0, 0.0])

    hot = MetaRouter(bandit=b, temperature=5.0).allocate(ctx)
    cold = MetaRouter(bandit=b, temperature=0.05).allocate(ctx)
    # Cold should be far more concentrated on the best arm.
    assert max(cold.values()) > max(hot.values())


def test_meta_router_fit_from_batch():
    rng = np.random.default_rng(7)
    ctxs = rng.normal(size=(60, 3))
    # Each arm gets a positive reward when its "regime" is active.
    arms = ["up" if c[0] > 0 else "dn" for c in ctxs]
    rewards = [abs(c[0]) for c in ctxs]   # always positive; arm tells *which* regime
    b = LinUCB(arms=["up", "dn"], d=3, alpha=0.1)
    r = MetaRouter(bandit=b, temperature=0.3)
    r.fit(ctxs, arms, rewards)
    # With strongly positive ctx[0], "up" arm's theta aligns, "dn" doesn't.
    w = r.allocate(np.array([2.0, 0.0, 0.0]))
    assert w["up"] > w["dn"]

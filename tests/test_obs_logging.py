"""Tests for the observability / structured logging layer."""
from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_obs(monkeypatch: pytest.MonkeyPatch):
    """Reload the obs module so every test gets a fresh configuration."""
    # Clear env that would leak between tests.
    for key in ("SIGNALFORGE_LOG_LEVEL", "SIGNALFORGE_LOG_FORMAT", "SIGNALFORGE_LOG_FILE"):
        monkeypatch.delenv(key, raising=False)
    import src.obs as obs
    importlib.reload(obs)
    # Reset stdlib root handlers so other tests don't pollute us.
    logging.getLogger().handlers.clear()
    yield
    logging.getLogger().handlers.clear()


def test_get_logger_works_without_explicit_configure():
    from src.obs import get_logger
    log = get_logger("test")
    log.info("hello", key="val")  # must not raise


def test_run_id_is_stable_and_non_empty():
    from src.obs import get_logger, run_id
    get_logger("test")
    rid = run_id()
    assert rid is not None
    assert len(rid) >= 8


def test_env_run_id_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SIGNALFORGE_RUN_ID", "fixed-run-xyz")
    import src.obs as obs
    importlib.reload(obs)
    obs.configure(force=True)
    assert obs.run_id() == "fixed-run-xyz"


def test_json_file_tee_produces_parseable_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    log_file = tmp_path / "sf.log"
    monkeypatch.setenv("SIGNALFORGE_LOG_FORMAT", "json")
    monkeypatch.setenv("SIGNALFORGE_LOG_FILE", str(log_file))
    import src.obs as obs
    importlib.reload(obs)
    obs.configure(force=True)
    obs.bind_context(strategy_id="liq_reversal_btc")

    log = obs.get_logger("unit")
    log.info("trade_filled", symbol="BTC/USDT", qty=0.5)

    # Flush file handler.
    for h in logging.getLogger().handlers:
        h.flush()

    lines = [ln for ln in log_file.read_text().splitlines() if ln.strip()]
    assert lines, "no log lines written to file"
    parsed = [json.loads(ln) for ln in lines]
    last = parsed[-1]
    assert last["event"] == "trade_filled"
    assert last["symbol"] == "BTC/USDT"
    assert last["strategy_id"] == "liq_reversal_btc"
    assert "run_id" in last
    assert "timestamp" in last


def test_bind_and_clear_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    log_file = tmp_path / "sf.log"
    monkeypatch.setenv("SIGNALFORGE_LOG_FORMAT", "json")
    monkeypatch.setenv("SIGNALFORGE_LOG_FILE", str(log_file))
    import src.obs as obs
    importlib.reload(obs)
    obs.configure(force=True)

    obs.bind_context(session="A")
    obs.get_logger("x").info("e1")
    obs.clear_context()
    obs.get_logger("x").info("e2")

    for h in logging.getLogger().handlers:
        h.flush()
    lines = [json.loads(ln) for ln in log_file.read_text().splitlines() if ln.strip()]
    e1 = next(r for r in lines if r["event"] == "e1")
    e2 = next(r for r in lines if r["event"] == "e2")
    assert e1.get("session") == "A"
    assert "session" not in e2


def test_log_level_filtering(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    log_file = tmp_path / "sf.log"
    monkeypatch.setenv("SIGNALFORGE_LOG_FORMAT", "json")
    monkeypatch.setenv("SIGNALFORGE_LOG_FILE", str(log_file))
    monkeypatch.setenv("SIGNALFORGE_LOG_LEVEL", "WARNING")
    import src.obs as obs
    importlib.reload(obs)
    obs.configure(force=True)

    log = obs.get_logger("x")
    log.info("suppressed")
    log.warning("kept")

    for h in logging.getLogger().handlers:
        h.flush()
    events = [json.loads(ln)["event"] for ln in log_file.read_text().splitlines() if ln.strip()]
    assert "suppressed" not in events
    assert "kept" in events

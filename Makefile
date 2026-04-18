# ==========================================================================
# SignalForge — developer Makefile
# Tab-indented (required by Make). Run `make help` to see targets.
# ==========================================================================

SHELL := /bin/bash
PY    := .venv/bin/python
PIP   := .venv/bin/pip

# Default symbols / timeframe / lookback for quick runs
SYMBOLS ?= BTC/USDT ETH/USDT SOL/USDT
TIMEFRAME ?= 1h
DAYS ?= 1825

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_.-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# --- Environment ----------------------------------------------------------
.PHONY: venv
venv:  ## Create .venv (python3.11)
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip

.PHONY: install
install:  ## Install runtime + dev extras into .venv
	$(PIP) install -e ".[dev,tearsheet]"

.PHONY: lock
lock:  ## Freeze current environment to requirements.lock
	$(PIP) freeze > requirements.lock

# --- Quality --------------------------------------------------------------
.PHONY: fmt
fmt:  ## Format with black + ruff --fix
	$(PY) -m ruff check --fix src scripts tests sf.py
	$(PY) -m black src scripts tests sf.py

.PHONY: lint
lint:  ## Lint (ruff) and format-check (black)
	$(PY) -m ruff check src scripts tests sf.py
	$(PY) -m black --check src scripts tests sf.py

.PHONY: type
type:  ## Type-check trading-path modules (src/backtest, src/risk, src/fund)
	$(PY) -m mypy

.PHONY: test
test:  ## Run test suite
	$(PY) -m pytest tests/ -q

.PHONY: test-fast
test-fast:  ## Run only fast tests (skip 'slow' + 'network' markers)
	$(PY) -m pytest tests/ -q -m "not slow and not network"

.PHONY: cov
cov:  ## Coverage report
	$(PY) -m pytest tests/ --cov=src --cov=sf --cov-report=term-missing

.PHONY: check
check: lint test  ## Lint + tests (what CI should run)

# --- Pipelines ------------------------------------------------------------
.PHONY: scan
scan:  ## Signal scan across default symbols
	$(PY) sf.py scan --symbols $(SYMBOLS) --timeframe $(TIMEFRAME) --days $(DAYS)

.PHONY: validate
validate:  ## Full OOS validation suite
	$(PY) sf.py validate-all --symbols $(SYMBOLS) --timeframe $(TIMEFRAME) --days $(DAYS) --min-trades 50 --oos-fraction 0.30

.PHONY: factory
factory:  ## Single factory loop (scan → validate → deploy-gate)
	$(PY) sf.py factory --once

.PHONY: backtest
backtest:  ## Honest backtest of deployed strategies
	$(PY) sf.py backtest

.PHONY: status
status:  ## Show deployed strategies + health
	$(PY) sf.py status

.PHONY: report
report:  ## Daily investor-style report
	$(PY) sf.py report

.PHONY: tearsheet
tearsheet:  ## Generate HTML tearsheet at fund_data/tearsheet.html
	$(PY) -m src.reporting.tearsheet --output fund_data/tearsheet.html

# --- Housekeeping ---------------------------------------------------------
.PHONY: clean
clean:  ## Remove caches, build artefacts, pycache
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache -o -name "*.egg-info" -o -name build -o -name dist \) -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml

.PHONY: clean-cache
clean-cache:  ## Wipe OHLCV parquet cache (forces refetch)
	rm -f data/cache/*.parquet data/cache/*.csv

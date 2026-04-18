# Changelog

All notable changes to SignalForge are documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
where practical for research software.

## [Unreleased]

### Added
- **Meta-strategy router** (`src/intelligence/bandit.py`) — contextual-bandit allocator that picks or mixes strategies based on regime features. Implements disjoint-arm LinUCB (Li et al., 2010) with deterministic upper-confidence-bound scoring, a Thompson-sampling variant for stochastic exploration, and a `MetaRouter` facade that turns UCB scores into a softmax-tempered allocation over arms. Supports batch `fit`, per-bar `update`, `save`/`load` of policy state to JSON. 12 unit tests.
- **Causal P&L attribution** (`src/audit/attribution.py`) — decomposes every closed round-trip into orthogonal buckets (`signal`, `slippage`, `impact`, `fee`, `funding`, `drift`) that sum to realised P&L by construction. Long/short and entry/exit sign handling is explicit; impact uses the same square-root footprint model as `FillModel`; drift absorbs anything not accounted for so it flags unattributed P&L. Aggregates per-strategy and per-asset via `attribute_trades`. 14 unit tests.
- **Autonomous research loop** (`src/research/autoloop.py`) — end-to-end agent: hypothesis generation → canonical feature synthesis (returns, z-momentum, vol ratio, range expansion) → signal compilation → walk-forward OOS evaluation via the existing harness → Bailey & López de Prado deflated Sharpe → multi-criteria gating (OOS Sharpe, positive-fold fraction, worst-fold drawdown, trade count, deflated-SR z-score) → auto-registration of accepted candidates through `src.registry.StrategyRegistry`. Rejected candidates carry an explicit reason string. 16 unit tests.
- **Live-vs-backtest parity auditor** (`src/audit/parity.py`) — reconciles the paper-trader journal (`fund_data/paper_journal_v20.jsonl`) against the `FillModel` to prove live P&L stays within tolerance of what the execution model predicts. Pairs entries with exits, reconstructs the realised equity curve, computes per-event price/fee deltas with correct long/short and entry/exit sign handling, and aggregates into a `ParityReport` with PASS/WARN/FAIL verdict (thresholds on `|unexplained_pnl_bps|`), per-asset and per-strategy breakdowns. Ships a `python -m src.audit.parity` CLI that exits 0/1/2 for PASS/WARN/FAIL. 16 unit tests.
- **Structured logging** (`src/obs/`) — `structlog`-based observability layer with process-wide `run_id`, contextvar-based binding (`bind_context`), JSON output in non-TTY environments (containers, pipes), human-readable console output in a TTY, optional file tee via `SIGNALFORGE_LOG_FILE`. Level controlled by `SIGNALFORGE_LOG_LEVEL`. 6 unit tests.
- **Walk-forward harness** (`src/backtest/walk_forward.py`) — rolling-origin evaluation with anchored + rolling schemes, `make_folds()` + `walk_forward()`, aggregates Sharpe mean/std/min/max, pooled geometric return, positive-fold fraction, worst-fold drawdown. 9 unit tests.
- **Execution fill model** (`src/execution/fill_model.py`) — partial fills capped by per-bar participation, maker-vs-taker regimes with queue-position proxy, per-venue fee schedules (Binance / Bybit / OKX perp defaults), square-root market impact, reproducible via seeded RNG. 14 unit tests.
- **Dashboard data layer** (`src/ops/dashboard_data.py`) — extracted pure (non-Streamlit) loaders, portfolio summary, and signal-proximity calculations from `scripts/live_dashboard.py`; the Streamlit script now imports from here. 15 unit tests.
- **Docker** — multi-stage `Dockerfile` (python 3.11-slim, non-root user, venv-isolated) and `docker-compose.yml` with `paper-trader`, `validator`, and `dashboard` services; `.dockerignore` excludes state and caches from the build context.
- **Pre-commit hooks** (`.pre-commit-config.yaml`) — ruff, black, detect-secrets, and the standard pre-commit-hooks suite (trailing whitespace, EOF, YAML/TOML/JSON lint, private-key detection, large-file guard). Installed via `pre-commit install` after `pip install -e ".[dev]"`.
- **CodeQL security scanning** (`.github/workflows/codeql.yml`) — weekly + on-push Python analysis with the `security-and-quality` query pack.
- **Strategy registry** (`src/registry.py`, 10 tests in `tests/test_registry.py`) — append-only NDJSON ledger at `fund_data/registry.ndjson` capturing every deploy with `commit_hash`, `config_hash`, `validation_hash`, parameters, trainer metadata, and operator identity. CLI: `python -m src.registry {list,show,diff}`.
- **Capacity analysis** (`research/capacity.py`) — square-root market-impact model estimating half-life AUM and participation-cap AUM per strategy from a validation JSON; produces a ranked, USD-formatted capacity table.
- **Packaging** — `pyproject.toml` with `signalforge` / `sf` console scripts, declared dependencies, `[dev]` and `[tearsheet]` extras, ruff / black / pytest / mypy configuration.
- **`Makefile`** with targets for install, format, lint, type-check, test, coverage, scan, validate, factory, backtest, status, report, tearsheet.
- **Backtester invariant tests** (`tests/test_backtest_invariants.py`, 13 deterministic cases) — no-lookahead, reproducibility, cost monotonicity, warmup gating, equity bounds, win-rate bounds.
- **Tearsheet generator** (`src/reporting/tearsheet.py`) — standalone HTML with embedded charts, produced from the latest validation JSON via `make tearsheet`.
- **Dependabot** configuration (`.github/dependabot.yml`) for weekly pip and monthly GitHub Actions updates, grouped by dependency type.
- Professional repository documentation: `README.md`, `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`.
- Issue and pull-request templates under `.github/`.
- Hardened `.gitignore` — regenerable artifacts (validation sweeps, cached OHLCV, fund runtime state) are no longer tracked.

### Changed
- `scripts/live_dashboard.py` is now a thin rendering shell over `src.ops.dashboard_data`; all loaders and proximity calculations are testable in isolation.
- Trimmed runtime artifacts from version control to keep the repo reproducible and lean.
- `[dev]` extras now include `pre-commit` and `detect-secrets`.
- Runtime dependency added: `structlog>=24.1.0`.

---

## [4.0.0] — 2026-04-16 — Production System

- Multi-strategy portfolio engine (5 strategies, 6 assets).
- Institutional validation suite (7/7 tests passing).
- Strategy factory pipeline: scan → validate → deploy, fully automated.
- Contrarian Asymmetry Engine — novel funding-direction asymmetry exploit.
- Multi-agent live trading loop: execution brain + adaptation + sentiment + decay monitor.
- Production scripts: `go_live`, `autonomous_loop`, `alerts`, `live_dashboard`, `daily_report`, `investor_report`.
- Full production audit: 20+ critical bugs identified and fixed.
- Codebase restructure: 45,000 → 20,000 lines (53% reduction); 37 dead files archived.
- Unified CLI entry point (`sf.py`).

## [3.0.0] — 2026-04 — Advanced Intelligence

- Fund Manager with hash-chained ledger (tamper-evident audit trail).
- Liquidation cascade oracle + protocol model + cascade simulator.
- Regime detection (bull / bear / sideways).
- Sentiment engine (Reddit, Fear & Greed Index, CoinGecko).
- 4-band drawdown risk system with circuit breakers.
- Smart execution engine (TWAP / VWAP, slippage control).
- Portfolio optimization: HRP, Markowitz, risk parity, CVaR.
- Capital scaling simulation ($1K → $1M+).

## [1.0.0] — 2026-04 — Foundation

- Alpha Genome genetic-programming evolver.
- Data pipeline: OHLCV, funding rates, open interest.
- First production strategy: Liquidation Reversal (BTC + ETH).
- Initial honest backtesting engine.
- Kelly-criterion risk manager.

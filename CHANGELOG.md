# Changelog

All notable changes to SignalForge are documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
where practical for research software.

## [Unreleased]

### Added
- **Packaging** — `pyproject.toml` with `signalforge` / `sf` console scripts, declared dependencies, `[dev]` and `[tearsheet]` extras, ruff / black / pytest / mypy configuration.
- **`Makefile`** with targets for install, format, lint, type-check, test, coverage, scan, validate, factory, backtest, status, report, tearsheet.
- **Backtester invariant tests** (`tests/test_backtest_invariants.py`, 13 deterministic cases) — no-lookahead, reproducibility, cost monotonicity, warmup gating, equity bounds, win-rate bounds.
- **Tearsheet generator** (`src/reporting/tearsheet.py`) — standalone HTML with embedded charts, produced from the latest validation JSON via `make tearsheet`.
- **Dependabot** configuration (`.github/dependabot.yml`) for weekly pip and monthly GitHub Actions updates, grouped by dependency type.
- Professional repository documentation: `README.md`, `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`.
- Issue and pull-request templates under `.github/`.
- Hardened `.gitignore` — regenerable artifacts (validation sweeps, cached OHLCV, fund runtime state) are no longer tracked.

### Changed
- Trimmed runtime artifacts from version control to keep the repo reproducible and lean.

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

# SignalForge

> **Autonomous crypto quantitative trading system.** Signal discovery → honest validation → multi-strategy deployment → live execution — all in one disciplined pipeline.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Research](https://img.shields.io/badge/status-research--grade-orange.svg)](#-status--disclaimer)

SignalForge is a from-scratch, institutional-style quant stack for crypto derivatives. It combines a genetic-programming alpha factory, a strict out-of-sample validation engine, a hash-chained fund ledger, and a multi-agent live-trading loop — unified behind a single CLI (`sf.py`).

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Quickstart](#quickstart)
- [The CLI](#the-cli)
- [Strategies](#strategies)
- [Validation Methodology](#validation-methodology)
- [Risk & Execution](#risk--execution)
- [Configuration](#configuration)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Status & Disclaimer](#-status--disclaimer)
- [License](#license)

---

## Highlights

- **End-to-end factory pipeline** — anomaly scan → OOS validation → deployment gate → live trading → decay monitor → self-healing.
- **Genetic programming alpha genome** — evolves candidate signals and prunes by realistic, cost-aware fitness.
- **Honest backtester** — no look-ahead, no survivorship, slippage + funding + fees baked in, session-aware stress tests.
- **Multi-strategy portfolio engine** — Liquidation Reversal, Funding Fade, Funding Carry, CSFC, Contrarian Asymmetry.
- **Institutional validation suite** — walk-forward, regime stratification, Monte-Carlo robustness, deflated Sharpe.
- **Hash-chained fund ledger** — tamper-evident audit trail for every sized decision.
- **Multi-agent live loop** — execution brain + adaptation + sentiment + decay monitor, each with circuit breakers.
- **Single entry point** — every workflow runs through `python sf.py <command>`.

~20,000 lines of Python across 14 modules, 17 scripts, 5 strategies, and a full test suite.

---

## Architecture

```
                         ┌──────────────────────────────┐
                         │        Data Pipeline         │
                         │  OHLCV · Funding · OI · Liq  │
                         └───────────────┬──────────────┘
                                         │
                ┌────────────────────────┼─────────────────────────┐
                ▼                        ▼                         ▼
      ┌────────────────┐       ┌───────────────────┐     ┌──────────────────┐
      │  Anomaly Scan  │       │  Alpha Genome (GP)│     │  Regime / Sent.  │
      │  (src/engine)  │       │  (src/alpha_genome)│    │   (src/regime)   │
      └────────┬───────┘       └─────────┬─────────┘     └──────────┬───────┘
               │                         │                           │
               └───────────┬─────────────┴──────────────┬────────────┘
                           ▼                            ▼
               ┌─────────────────────────┐   ┌─────────────────────────┐
               │  Honest Backtester      │   │  OOS Validator          │
               │  (src/backtest)         │   │  (tests/validate_all)   │
               └───────────┬─────────────┘   └───────────┬─────────────┘
                           └──────────────┬──────────────┘
                                          ▼
                         ┌──────────────────────────────┐
                         │       Strategy Factory       │
                         │        (src/factory)         │
                         └───────────────┬──────────────┘
                                         ▼
                         ┌──────────────────────────────┐
                         │  Portfolio & Risk Engine     │
                         │  (src/engine, src/risk)      │
                         └───────────────┬──────────────┘
                                         ▼
                         ┌──────────────────────────────┐
                         │   Multi-Agent Live Trading   │
                         │  Brain · Sentiment · Adapt   │
                         │       (src/execution)        │
                         └───────────────┬──────────────┘
                                         ▼
                         ┌──────────────────────────────┐
                         │ Hash-Chained Fund Ledger     │
                         │        (src/fund)            │
                         └──────────────────────────────┘
```

See [`PROGRESS.md`](PROGRESS.md) for a full engineering log, design decisions, bug-fix history, and research findings.

---

## Repository Layout

```
SignalForge/
├── sf.py                    # Unified CLI — every workflow starts here
├── start.sh                 # Convenience bootstrapper
├── requirements.txt
├── config/
│   ├── settings.yaml        # Runtime configuration
│   └── calibration.json     # Calibrated strategy parameters
├── src/
│   ├── alpha_genome/        # Genetic-programming evolver
│   ├── backtest/            # Honest backtester + cost model
│   ├── core/                # Shared primitives (types, utils, telemetry)
│   ├── data/                # OHLCV + funding + OI + multi-venue fetchers
│   ├── engine/              # Portfolio engine, stress, regime gating
│   ├── execution/           # Smart execution (TWAP/VWAP, slippage ctrl)
│   ├── factory/             # scan → validate → deploy loop
│   ├── fund/                # Hash-chained ledger, equity accounting
│   ├── intelligence/        # Sentiment, crowding, cascade oracle
│   ├── liquidation/         # Liquidation cascade simulator
│   ├── regime/              # Regime detection (bull/bear/sideways)
│   ├── risk/                # 4-band drawdown, Kelly, circuit breakers
│   ├── sentiment/           # Reddit / Fear&Greed / CoinGecko
│   └── strategies/          # Liq reversal, funding fade/carry, CSFC
├── scripts/                 # Operational runners (go_live, daily_report, …)
├── research/                # Exploratory studies (not on the trading path)
├── tests/                   # pytest suite (unit + E2E + factory)
└── PROGRESS.md              # Full engineering log & documentation
```

Regenerable artifacts (cached OHLCV, validation sweep outputs, fund runtime state) are **intentionally git-ignored** to keep the repo reproducible and lean.

---

## Quickstart

**Requirements:** Python 3.11+, macOS/Linux. A venv is strongly recommended.

```bash
git clone https://github.com/varunteja0/SignalForge.git
cd SignalForge

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Sanity check — lists every available subcommand
python sf.py --help
```

Optional `.env` for live keys (never commit):

```bash
cp .env.example .env  # if present
# BINANCE_API_KEY=...
# BYBIT_API_KEY=...
```

---

## The CLI

Everything is one entry point: **`python sf.py <command>`**.

### Factory pipeline (discovery → deployment)

| Command | Purpose |
|---|---|
| `sf.py scan` | Scan for statistical signal anomalies |
| `sf.py validate` | Out-of-sample validation of scan candidates |
| `sf.py validate-all` | Full institutional validation suite |
| `sf.py factory` | Run the full scan → validate → deploy loop |
| `sf.py factory --once` | Single cycle, then exit |
| `sf.py backtest` | Honest backtest of deployed strategies |
| `sf.py status` | Deployed strategies + portfolio health |

### Unified engine (evolve → trade → monitor → adapt)

| Command | Purpose |
|---|---|
| `sf.py run` | Paper trading via the unified engine |
| `sf.py run --real` | **Live** trading (requires keys; use with care) |
| `sf.py evolve` | Run genetic-programming evolution |
| `sf.py crowding` | Crowding / positioning analysis |
| `sf.py cascade` | Liquidation cascade prediction |
| `sf.py engine-status` | Engine + strategy health snapshot |

### Operational

| Command | Purpose |
|---|---|
| `sf.py live` | Legacy paper trading loop |
| `sf.py report` | Daily investor-style report |

---

## Strategies

| Strategy | Thesis | Status |
|---|---|---|
| **Liquidation Reversal** | Fade forced liquidation overshoots on BTC/ETH | Production-validated |
| **Funding Fade** | Short extreme positive / long extreme negative funding | Validated, sized small |
| **Funding Carry** | Structural carry during calm regimes | Research — see verdict note in `PROGRESS.md` |
| **CSFC** | Cascade-triggered short-funding compression | Validated |
| **Contrarian Asymmetry** | Novel finding: funding-direction asymmetry exploit | Deployed, monitored |

Each strategy lives in `src/strategies/` with a matching backtest and validation entry.

---

## Validation Methodology

SignalForge treats backtest-to-deploy as a **gated funnel**, not a free-for-all:

1. **Honest backtest** — no look-ahead, point-in-time features, funding + fees + slippage modeled.
2. **Walk-forward OOS** — strict 70/30 (configurable) with no peeking.
3. **Regime stratification** — metrics broken down per bull / bear / sideways.
4. **Monte-Carlo block bootstrap** — distribution over equity curves, not a single number.
5. **Deflated Sharpe / PSR** — corrects for multiple testing.
6. **Outcome validation harness** — `scripts/outcome_validation.py` re-runs deployed strategies against held-out folds before any capital is committed.
7. **Deploy gate** — a strategy only goes live if it clears *all* of the above.

Full methodology and per-strategy verdicts are in [`PROGRESS.md`](PROGRESS.md).

---

## Risk & Execution

- **4-band drawdown system** — normal / caution / defensive / halt, each with its own leverage cap.
- **Kelly-fraction sizing** with hard caps and per-strategy correlation penalties.
- **Circuit breakers** on equity, slippage, and venue errors.
- **Smart execution** — TWAP / VWAP slices, slippage budget, partial-fill aware.
- **Hash-chained ledger** (`src/fund/`) — every decision and fill is append-only and tamper-evident.

---

## Configuration

All runtime knobs live in [`config/settings.yaml`](config/settings.yaml). Calibrated per-strategy parameters live in [`config/calibration.json`](config/calibration.json). Secrets go in `.env` and are never committed.

---

## Testing

```bash
pytest tests/ -q
```

The suite covers:

- Core unit tests (`test_v2.py`)
- Integration tests (`test_v2_integration.py`)
- Factory pipeline (`test_factory.py`)
- End-to-end smoke tests (`test_e2e.py`, `test_e2e_full.py`)
- Institutional validation harness (`tests/validate_all.py`)

---

## Roadmap

- [ ] Expand universe beyond BTC/ETH/SOL/BNB/XRP/DOGE
- [ ] Options overlay (hedged carry)
- [ ] Cross-venue arbitrage with inventory accounting
- [ ] Live Grafana / Streamlit dashboard bundled as a service
- [ ] Model registry + strategy versioning
- [ ] Published research notes for each production strategy

---

## ⚠️ Status & Disclaimer

SignalForge is **research-grade software** developed by a single author for personal experimentation. It is **not** financial advice. Crypto derivatives trading can and will lose you money. Never run `sf.py run --real` with capital you cannot afford to lose, and never without reading every strategy's verdict in `PROGRESS.md`.

No warranty, express or implied. See [LICENSE](LICENSE).

---

## License

Released under the [MIT License](LICENSE). See the [Contributing Guide](CONTRIBUTING.md) if you want to help improve it.

# Contributing to SignalForge

Thanks for your interest in improving SignalForge. This project is small, opinionated, and quant-focused — please read this before opening a PR so your contribution lands cleanly.

## Ground rules

1. **No look-ahead.** Any feature that uses future information in a backtest will be rejected on sight.
2. **No new strategy without honest validation.** New strategies must ship with a walk-forward OOS run and a regime breakdown (see [`PROGRESS.md`](PROGRESS.md) § Validation).
3. **Deterministic by default.** Set seeds; avoid hidden randomness in the trading path.
4. **Small, reviewable PRs.** One concern per PR.

## Development setup

```bash
git clone https://github.com/varunteja0/SignalForge.git
cd SignalForge
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Style & quality

- **Formatter:** `black` (line length 100)
- **Linter:** `ruff`
- **Types:** prefer explicit type hints in public APIs
- **Tests:** `pytest tests/ -q` must pass locally

Before pushing:

```bash
ruff check src scripts tests
black --check src scripts tests
pytest tests/ -q
```

## Commit messages

Conventional-ish, imperative mood, present tense:

```
feat(engine): add regime-gated exposure cap
fix(backtest): correct funding accrual on short carry
docs(readme): clarify validation gate
test(factory): cover deploy-gate rejection path
```

## Pull requests

- Link the relevant issue, or open one first for anything non-trivial.
- Describe *what*, *why*, and *how you validated it* (especially for strategy or risk changes).
- A maintainer will review within a reasonable window — please be patient.

## Reporting bugs

Use the issue templates under `.github/ISSUE_TEMPLATE/`. Include:

- SignalForge commit hash
- Python version and OS
- Exact command run
- Full stack trace / log excerpt
- Minimal repro if possible

## Security

Please **do not** file public issues for security problems. See [SECURITY.md](SECURITY.md).

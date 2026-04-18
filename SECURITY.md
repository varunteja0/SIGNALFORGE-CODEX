# Security Policy

## Supported versions

SignalForge is a single-author research project. Security fixes are applied to the `main` branch only.

## Reporting a vulnerability

If you believe you have found a security issue — especially anything that could leak API keys, corrupt the fund ledger, or let an attacker influence live trading decisions — please report it **privately**.

- **Preferred:** open a [private security advisory](https://github.com/varunteja0/SignalForge/security/advisories/new) on GitHub.
- **Alternative:** email the repository owner via their GitHub profile.

Please include:

- A clear description of the issue and its impact
- Steps to reproduce (or a proof-of-concept)
- The SignalForge commit hash and Python version you tested against
- Any relevant logs (with secrets redacted)

**Do not** open a public issue, post to discussions, or disclose on social media before a fix is released.

## Scope

In scope:

- Anything in `src/`, `scripts/`, `sf.py`, `config/`
- CI workflows under `.github/workflows/`
- Dependency vulnerabilities with a realistic exploit path in this project

Out of scope:

- Theoretical strategy losses or poor PnL (this is research software — see the disclaimer in [README.md](README.md))
- Issues that require a compromised developer machine
- Denial of service via rate-limiting public exchange APIs

## Handling

- Acknowledgement: within a few days of receipt.
- Fix: prioritized relative to severity and exploitability.
- Disclosure: coordinated with the reporter after a fix is released.

Thanks for helping keep SignalForge safe.

"""
Strategy / model registry
==========================

Every production deploy of a strategy is recorded here with enough
provenance to reconstruct it:

- ``strategy_id``         — stable human-readable identifier (e.g. ``liq_reversal_btc_v3``)
- ``version``             — monotonically increasing per ``strategy_id``
- ``commit_hash``         — ``git rev-parse HEAD`` at deploy time
- ``config_hash``         — SHA-256 of the serialized config / params
- ``trained_on``          — dict describing the data window used
- ``validation_hash``     — SHA-256 of the validation artefact (JSON)
- ``deployed_at``         — ISO-8601 UTC timestamp
- ``deployed_by``         — operator identity (``$USER`` by default)
- ``notes``               — free-form commit-message-style string

Storage is a newline-delimited JSON file (``fund_data/registry.ndjson``) —
append-only, human-greppable, no moving parts. Each record is immutable
once written; fixes are expressed as *new versions*, never mutations.

Usage
-----

Python::

    from src.registry import StrategyRegistry

    reg = StrategyRegistry()
    entry = reg.register(
        strategy_id="liq_reversal_btc",
        params={"atr_mult": 2.0, "window": 14},
        validation_file="fund_data/validation_v16.json",
        trained_on={"symbol": "BTC/USDT", "timeframe": "1h", "days": 1825},
        notes="promoted after v16 deploy gate",
    )
    print(entry.version, entry.config_hash)

    active = reg.latest("liq_reversal_btc")
    history = reg.history("liq_reversal_btc")

CLI::

    python -m src.registry list
    python -m src.registry show liq_reversal_btc
    python -m src.registry diff liq_reversal_btc 2 3
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    """Return current git HEAD or ``'unknown'`` if not a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[1],
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RegistryEntry:
    strategy_id: str
    version: int
    commit_hash: str
    config_hash: str
    params: dict[str, Any]
    trained_on: dict[str, Any]
    validation_file: str | None
    validation_hash: str | None
    deployed_at: str
    deployed_by: str
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> RegistryEntry:
        d = json.loads(s)
        d.setdefault("tags", [])
        return cls(**d)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
class StrategyRegistry:
    """Append-only registry persisted as newline-delimited JSON."""

    def __init__(self, path: str | Path = "fund_data/registry.ndjson") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ----- read -----------------------------------------------------------
    def _iter_entries(self) -> list[RegistryEntry]:
        if not self.path.exists():
            return []
        out: list[RegistryEntry] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(RegistryEntry.from_json(line))
        return out

    def all(self) -> list[RegistryEntry]:
        return self._iter_entries()

    def history(self, strategy_id: str) -> list[RegistryEntry]:
        return [e for e in self._iter_entries() if e.strategy_id == strategy_id]

    def latest(self, strategy_id: str) -> RegistryEntry | None:
        hist = self.history(strategy_id)
        return max(hist, key=lambda e: e.version) if hist else None

    def ids(self) -> list[str]:
        return sorted({e.strategy_id for e in self._iter_entries()})

    # ----- write ----------------------------------------------------------
    def register(
        self,
        strategy_id: str,
        params: dict[str, Any],
        *,
        trained_on: dict[str, Any] | None = None,
        validation_file: str | Path | None = None,
        notes: str = "",
        tags: list[str] | None = None,
        deployed_by: str | None = None,
    ) -> RegistryEntry:
        if not strategy_id or not strategy_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"strategy_id must be alphanumeric (with _ or -) — got {strategy_id!r}"
            )

        prev = self.latest(strategy_id)
        version = (prev.version + 1) if prev else 1

        val_path = Path(validation_file) if validation_file else None
        val_hash = _sha256_file(val_path) if val_path else None

        entry = RegistryEntry(
            strategy_id=strategy_id,
            version=version,
            commit_hash=_git_commit(),
            config_hash=_sha256_bytes(_canonical(params)),
            params=params,
            trained_on=trained_on or {},
            validation_file=str(val_path) if val_path else None,
            validation_hash=val_hash,
            deployed_at=_now_iso(),
            deployed_by=deployed_by or os.environ.get("USER") or getpass.getuser(),
            notes=notes,
            tags=tags or [],
        )

        with self.path.open("a") as fh:
            fh.write(entry.to_json() + "\n")
        return entry


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _render_row(e: RegistryEntry) -> str:
    return (
        f"  {e.strategy_id:<28s} v{e.version:<3d}  "
        f"{e.deployed_at}  {e.commit_hash[:7]}  cfg={e.config_hash[:10]}  "
        f"{e.notes[:40]}"
    )


def _cmd_list(args: argparse.Namespace) -> int:
    reg = StrategyRegistry(args.registry)
    entries = reg.all()
    if not entries:
        print("(registry is empty)")
        return 0
    if args.latest:
        by_id: dict[str, RegistryEntry] = {}
        for e in entries:
            prev = by_id.get(e.strategy_id)
            if prev is None or e.version > prev.version:
                by_id[e.strategy_id] = e
        entries = sorted(by_id.values(), key=lambda e: e.strategy_id)
    for e in entries:
        print(_render_row(e))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    reg = StrategyRegistry(args.registry)
    if args.version is None:
        entry = reg.latest(args.strategy_id)
    else:
        entry = next(
            (e for e in reg.history(args.strategy_id) if e.version == args.version), None
        )
    if entry is None:
        print(f"No entry for {args.strategy_id}"
              + (f" v{args.version}" if args.version is not None else ""))
        return 1
    print(json.dumps(asdict(entry), indent=2))
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    reg = StrategyRegistry(args.registry)
    hist = {e.version: e for e in reg.history(args.strategy_id)}
    a = hist.get(args.v1)
    b = hist.get(args.v2)
    if a is None or b is None:
        print(f"Need both versions {args.v1} and {args.v2} of {args.strategy_id}")
        return 1

    print(f"─── {args.strategy_id}  v{a.version} → v{b.version} ───")
    print(f"  commit:     {a.commit_hash[:10]} → {b.commit_hash[:10]}")
    print(f"  config_hash: {a.config_hash[:12]} → {b.config_hash[:12]}")
    keys = sorted(set(a.params) | set(b.params))
    for k in keys:
        va, vb = a.params.get(k, "∅"), b.params.get(k, "∅")
        if va != vb:
            print(f"  ~ {k}: {va!r} → {vb!r}")
    if a.notes != b.notes:
        print(f"  notes: {a.notes!r} → {b.notes!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="signalforge-registry",
        description="SignalForge strategy / model deployment registry.",
    )
    p.add_argument(
        "--registry",
        default="fund_data/registry.ndjson",
        help="Path to the NDJSON registry file (default: fund_data/registry.ndjson).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List registered deployments.")
    p_list.add_argument("--latest", action="store_true", help="Show only the latest version per strategy.")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Show a single deployment entry.")
    p_show.add_argument("strategy_id")
    p_show.add_argument("--version", type=int, default=None)
    p_show.set_defaults(func=_cmd_show)

    p_diff = sub.add_parser("diff", help="Diff two versions of a strategy.")
    p_diff.add_argument("strategy_id")
    p_diff.add_argument("v1", type=int)
    p_diff.add_argument("v2", type=int)
    p_diff.set_defaults(func=_cmd_diff)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

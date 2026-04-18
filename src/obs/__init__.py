"""
SignalForge observability — structured logging.

Single entry point: :func:`get_logger`. All modules should do::

    from src.obs import get_logger
    log = get_logger(__name__)
    log.info("trade_filled", symbol="BTC/USDT", side="long", qty=0.5, price=62_314.0)

Configuration
-------------
- Default: human-friendly console rendering in a TTY, JSON lines in
  non-TTY (pipes, files, containers) — so ``docker logs`` produces
  machine-readable output without extra flags.
- ``SIGNALFORGE_LOG_LEVEL`` (default ``INFO``) — standard level name.
- ``SIGNALFORGE_LOG_FORMAT`` — ``json`` | ``console`` to force a mode.
- ``SIGNALFORGE_LOG_FILE`` — optional path; when set, log lines also
  tee to the file (JSON regardless of console mode, so the file is
  always parseable).

Context
-------
:func:`bind_context` sets process-wide context (e.g. ``run_id``,
``strategy_id``) that is injected into every subsequent log line.
"""
from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import structlog
from structlog.typing import EventDict, Processor

_CONFIGURED = False
_RUN_ID: str | None = None


def _resolve_format() -> str:
    explicit = os.environ.get("SIGNALFORGE_LOG_FORMAT", "").lower().strip()
    if explicit in {"json", "console"}:
        return explicit
    return "console" if sys.stderr.isatty() else "json"


def _resolve_level() -> int:
    name = os.environ.get("SIGNALFORGE_LOG_LEVEL", "INFO").upper()
    return logging.getLevelName(name) if isinstance(logging.getLevelName(name), int) else logging.INFO


def _drop_color_message(_: Any, __: str, event_dict: EventDict) -> EventDict:
    event_dict.pop("color_message", None)
    return event_dict


def configure(*, force: bool = False) -> None:
    """Idempotently configure structlog + the stdlib root logger."""
    global _CONFIGURED, _RUN_ID
    if _CONFIGURED and not force:
        return

    fmt = _resolve_format()
    level = _resolve_level()

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _drop_color_message,
    ]

    if fmt == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    # structlog loggers hand off to stdlib; stdlib ProcessorFormatter
    # performs the final render. This avoids double-rendering.
    structlog.configure(
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging through structlog formatters so third-party
    # libs (ccxt, urllib3, matplotlib) emit consistent output.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Optional file tee — always JSON for grep-ability.
    file_path = os.environ.get("SIGNALFORGE_LOG_FILE")
    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file_path)
        fh.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
            )
        )
        root.addHandler(fh)

    # Give every process a stable run_id unless one was pre-set.
    if _RUN_ID is None:
        _RUN_ID = os.environ.get("SIGNALFORGE_RUN_ID") or uuid.uuid4().hex[:12]
        structlog.contextvars.bind_contextvars(run_id=_RUN_ID)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger. Safe to call at import time."""
    if not _CONFIGURED:
        configure()
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind process-wide context onto all subsequent log events."""
    if not _CONFIGURED:
        configure()
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()


def run_id() -> str | None:
    return _RUN_ID


__all__ = ["configure", "get_logger", "bind_context", "clear_context", "run_id"]

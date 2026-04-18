# ==========================================================================
# SignalForge — multi-stage Dockerfile
#
# Produces a slim runtime image suitable for paper trading, scheduled
# validation runs, or live execution (when credentials are mounted at
# runtime). The build is deterministic once pyproject.toml is pinned.
#
# Usage
# -----
#   docker build -t signalforge:latest .
#   docker run --rm -it signalforge:latest sf --help
#
#   # paper trading with mounted state + env file
#   docker run -d --name sf-paper \
#       --env-file .env \
#       -v $(pwd)/fund_data:/app/fund_data \
#       -v $(pwd)/data/cache:/app/data/cache \
#       signalforge:latest sf run
# ==========================================================================

# --- Stage 1: build dependencies -----------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
 && rm -rf /var/lib/apt/lists/*

# Copy only what `pip install -e .` needs for dependency resolution.
COPY pyproject.toml README.md LICENSE ./
COPY sf.py ./
COPY src ./src

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install "."


# --- Stage 2: runtime image ----------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    SIGNALFORGE_ENV=paper

# Non-root user — trading binaries should never run as root.
RUN groupadd --system --gid 1001 signalforge \
 && useradd --system --uid 1001 --gid signalforge --home-dir /app --shell /sbin/nologin signalforge

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Copy only what runs in production.
COPY --chown=signalforge:signalforge sf.py ./
COPY --chown=signalforge:signalforge src ./src
COPY --chown=signalforge:signalforge scripts ./scripts
COPY --chown=signalforge:signalforge config ./config

# Runtime-writable directories — state, cache, reports.
RUN mkdir -p /app/fund_data /app/data/cache /app/evolved_strategies \
 && chown -R signalforge:signalforge /app

USER signalforge

# Liveness: CLI can respond.
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD sf --help > /dev/null || exit 1

ENTRYPOINT ["sf"]
CMD ["--help"]

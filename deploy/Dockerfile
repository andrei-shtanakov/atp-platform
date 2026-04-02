FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /usr/local/bin/uv

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./
COPY packages/atp-core/pyproject.toml packages/atp-core/pyproject.toml
COPY packages/atp-adapters/pyproject.toml packages/atp-adapters/pyproject.toml
COPY packages/atp-dashboard/pyproject.toml packages/atp-dashboard/pyproject.toml
COPY packages/atp-sdk/pyproject.toml packages/atp-sdk/pyproject.toml
COPY game-environments/pyproject.toml game-environments/pyproject.toml
COPY atp-games/pyproject.toml atp-games/pyproject.toml

# Install dependencies (cached unless pyproject.toml or uv.lock changes)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace

# Copy source code (changes more often than deps)
COPY packages/ packages/
COPY atp/ atp/
COPY game-environments/ game-environments/
COPY atp-games/ atp-games/
COPY migrations/ migrations/
COPY alembic.ini ./

# Install workspace packages (uses cached deps)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- Runtime stage ---
FROM python:3.12-slim

WORKDIR /app

# Non-root user
RUN groupadd -r atp && useradd -r -g atp -d /app atp

# Copy installed environment from builder
COPY --from=builder /app /app

# Create data directory owned by atp user
RUN mkdir -p /data && chown atp:atp /data

USER atp

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/api/health').raise_for_status()" || exit 1

CMD ["uv", "run", "uvicorn", "atp.dashboard.v2.factory:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]

FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy workspace
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY atp/ atp/
COPY game-environments/ game-environments/
COPY atp-games/ atp-games/
COPY migrations/ migrations/
COPY alembic.ini ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Create data directory
RUN mkdir -p /data

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "atp.dashboard.v2.factory:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]

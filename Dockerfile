# Dockerfile — ATP Platform runner
FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy and install
COPY . .
RUN uv sync --no-dev

# Default: show version
CMD ["uv", "run", "atp", "version"]

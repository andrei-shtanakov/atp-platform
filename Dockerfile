# Dockerfile — ATP Platform runner
FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy and install. --all-packages installs every workspace member (plugins like
# atp-method and atp-platform-sdk), not just the root and its dependencies — so
# `atp test` can dispatch plugin formats. Run with `uv run --no-sync` afterwards
# so the runtime does not prune the extra members back out.
COPY . .
# --extra llm brings the LLM-judge clients (anthropic + openai); the openai
# client also serves OpenAI-compatible local servers for an air-gapped judge.
RUN uv sync --no-dev --all-packages --extra llm

# Default: show version
CMD ["uv", "run", "--no-sync", "atp", "version"]

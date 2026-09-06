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
# --extra llm brings the LLM-judge clients (anthropic + openai; openai also
# serves OpenAI-compatible local servers for an air-gapped judge). --extra
# bedrock adds boto3, required by both the Bedrock adapter (agent under test) and
# the Bedrock-Claude judge (AsyncAnthropicBedrock) in the all-in-AWS variant.
# --extra dashboard adds the tournament/MCP stack: docker-compose.yml's
# `dashboard` service runs `atp dashboard` with no ATP_SERVER_PROFILE set, i.e.
# the full profile, and that profile refuses to start without it. The rest of
# the server stack (FastAPI, uvicorn, alembic, slowapi) is in the base install.
RUN uv sync --no-dev --all-packages --extra llm --extra bedrock --extra dashboard

# Default: show version
CMD ["uv", "run", "--no-sync", "atp", "version"]

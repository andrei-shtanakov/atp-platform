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
RUN uv sync --no-dev --all-packages --extra llm --extra bedrock

# Default: show version
CMD ["uv", "run", "--no-sync", "atp", "version"]

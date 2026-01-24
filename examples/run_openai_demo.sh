#!/bin/bash
# ATP Platform Demo - OpenAI Agent
# Requires: OPENAI_API_KEY environment variable

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=== ATP Platform Demo - OpenAI Agent ==="
echo ""

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is required"
    echo ""
    echo "Usage:"
    echo "  export OPENAI_API_KEY='your-api-key'"
    echo "  bash examples/run_openai_demo.sh"
    exit 1
fi

echo "Using model: ${OPENAI_MODEL:-gpt-5-mini}"
echo ""

# Create a temporary workspace
export ATP_WORKSPACE=$(mktemp -d)
echo "Workspace: $ATP_WORKSPACE"
echo ""

# Clean up workspace on exit
cleanup() {
    rm -rf "$ATP_WORKSPACE"
}
trap cleanup EXIT

# Step 1: Test the agent directly
echo "=== Step 1: Direct Agent Test ==="
echo '{"task_id":"direct-test","task":{"description":"Calculate 2 + 2 and create a file called answer.txt with the result"}}' | python examples/openai_agent.py 2>/dev/null | python -m json.tool
echo ""

# Step 2: Run a single smoke test
echo "=== Step 2: Single Test (Smoke) ==="
uv run atp test examples/test_suites/openai_agent.yaml \
  --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["examples/openai_agent.py"]' \
  --adapter-config='inherit_environment=true' \
  --adapter-config='allowed_env_vars=["OPENAI_API_KEY","OPENAI_MODEL"]' \
  --tags=smoke \
  -v || true
echo ""

# Step 3: Run full test suite
echo "=== Step 3: Full Test Suite ==="
read -p "Run full test suite? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv run atp test examples/test_suites/openai_agent.yaml \
      --adapter=cli \
      --adapter-config='command=python' \
      --adapter-config='args=["examples/openai_agent.py"]' \
      --adapter-config='inherit_environment=true' \
      --adapter-config='allowed_env_vars=["OPENAI_API_KEY","OPENAI_MODEL"]' \
      -v
fi

echo ""
echo "=== Demo Complete ==="

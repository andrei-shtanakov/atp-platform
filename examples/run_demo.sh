#!/bin/bash
# ATP Platform Demo - File Operations Agent
# This script demonstrates the ATP Platform end-to-end workflow

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=== ATP Platform Demo ==="
echo "Testing the file operations demo agent"
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
echo '{"task_id":"direct-test","task":{"description":"Create file '\''test.txt'\'' with content '\''Hello World'\''"},"context":{"workspace_path":"'"$ATP_WORKSPACE"'"}}' | python examples/demo_agent.py
echo ""
echo ""

# Step 2: Run ATP test suite (verbose mode)
echo "=== Step 2: ATP Test Suite (Console) ==="
uv run atp test examples/test_suites/demo_file_agent.yaml \
  --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["examples/demo_agent.py"]' \
  -v || true
echo ""

# Step 3: Run ATP test suite with JSON output
echo "=== Step 3: ATP Test Suite (JSON) ==="
RESULT_FILE="$ATP_WORKSPACE/demo_results.json"
uv run atp test examples/test_suites/demo_file_agent.yaml \
  --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["examples/demo_agent.py"]' \
  --output=json \
  --output-file="$RESULT_FILE" || true

if [ -f "$RESULT_FILE" ]; then
    echo ""
    echo "Results saved to: $RESULT_FILE"
    echo ""
    echo "=== Test Summary ==="
    python -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)

total = data.get('summary', {}).get('total_tests', 0)
passed = data.get('summary', {}).get('passed', 0)
failed = data.get('summary', {}).get('failed', 0)

print(f'Total tests: {total}')
print(f'Passed: {passed}')
print(f'Failed: {failed}')
print(f'Pass rate: {passed/total*100:.1f}%' if total > 0 else 'N/A')
"
else
    echo "No results file generated"
fi

echo ""
echo "=== Demo Complete ==="

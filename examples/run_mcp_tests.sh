#!/bin/bash
# Run MCP agent tests with ATP
#
# This script:
# 1. Starts the mock MCP server
# 2. Runs the MCP agent tests
# 3. Generates a report
# 4. Cleans up
#
# Usage:
#   ./examples/run_mcp_tests.sh [options]
#
# Options:
#   --runs N         Number of runs per test (default: 3)
#   --output FORMAT  Output format: console, json, junit (default: console)
#   --tags TAGS      Filter tests by tags (comma-separated)
#   --verbose        Verbose output
#   --keep-server    Don't stop MCP server after tests
#
# Environment:
#   OPENAI_API_KEY   Required for running the agent
#
# Examples:
#   ./examples/run_mcp_tests.sh
#   ./examples/run_mcp_tests.sh --runs 5 --output html
#   ./examples/run_mcp_tests.sh --tags smoke --verbose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RUNS=3
OUTPUT_FORMAT="console"
TAGS=""
VERBOSE=""
KEEP_SERVER=false
MCP_PORT=9876

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --tags)
            TAGS="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        --keep-server)
            KEEP_SERVER=true
            shift
            ;;
        --port)
            MCP_PORT="$2"
            shift 2
            ;;
        --help|-h)
            head -30 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check for OPENAI_API_KEY
if [[ -z "${OPENAI_API_KEY}" ]]; then
    echo -e "${RED}Error: OPENAI_API_KEY environment variable is not set${NC}"
    echo "Please set it before running tests:"
    echo "  export OPENAI_API_KEY='sk-...'"
    exit 1
fi

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  ATP MCP Agent Tests${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Function to cleanup
cleanup() {
    if [[ -n "$MCP_PID" ]] && [[ "$KEEP_SERVER" == "false" ]]; then
        echo -e "${YELLOW}Stopping MCP server (PID: $MCP_PID)...${NC}"
        kill "$MCP_PID" 2>/dev/null || true
        wait "$MCP_PID" 2>/dev/null || true
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Start MCP server
echo -e "${GREEN}Starting mock MCP server on port $MCP_PORT...${NC}"
python examples/mock_mcp_server.py --port "$MCP_PORT" &
MCP_PID=$!

# Wait for server to be ready
echo -n "Waiting for server"
for i in {1..10}; do
    if curl -s "http://localhost:$MCP_PORT/health" > /dev/null 2>&1; then
        echo -e " ${GREEN}ready!${NC}"
        break
    fi
    echo -n "."
    sleep 0.5
done

# Verify server is running
if ! curl -s "http://localhost:$MCP_PORT/health" > /dev/null 2>&1; then
    echo -e " ${RED}failed!${NC}"
    echo -e "${RED}Error: MCP server failed to start${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}MCP server running at http://localhost:$MCP_PORT${NC}"
echo ""

# Set MCP server URL for agent
export MCP_SERVER_URL="http://localhost:$MCP_PORT"

# Build output arguments
OUTPUT_ARGS=""
REPORT_FILE=""
case $OUTPUT_FORMAT in
    json)
        REPORT_FILE="mcp_test_results.json"
        OUTPUT_ARGS="--output=json --output-file=$REPORT_FILE"
        ;;
    junit)
        REPORT_FILE="mcp_test_results.xml"
        OUTPUT_ARGS="--output=junit --output-file=$REPORT_FILE"
        ;;
    console|*)
        OUTPUT_ARGS=""
        ;;
esac

# Build tags argument
TAGS_ARG=""
if [[ -n "$TAGS" ]]; then
    TAGS_ARG="--tags=$TAGS"
fi

# Create working directory for test artifacts
WORK_DIR=$(mktemp -d)
echo -e "${YELLOW}Working directory: $WORK_DIR${NC}"
cd "$WORK_DIR"

# Run tests
echo ""
echo -e "${GREEN}Running tests...${NC}"
echo -e "${YELLOW}Runs per test: $RUNS${NC}"
echo ""

set +e  # Don't exit on test failure

uv run atp test "$PROJECT_ROOT/examples/test_suites/mcp_connection_test.yaml" \
    --adapter=cli \
    --adapter-config="command=python" \
    --adapter-config="args=[\"$PROJECT_ROOT/examples/mcp_agent.py\"]" \
    --adapter-config='inherit_environment=true' \
    --adapter-config='allowed_env_vars=["OPENAI_API_KEY","MCP_SERVER_URL","OPENAI_MODEL"]' \
    --runs="$RUNS" \
    $OUTPUT_ARGS \
    $TAGS_ARG \
    $VERBOSE

TEST_EXIT_CODE=$?

set -e

# Copy report to project directory
if [[ -n "$REPORT_FILE" ]] && [[ -f "$REPORT_FILE" ]]; then
    cp "$REPORT_FILE" "$PROJECT_ROOT/$REPORT_FILE"
    echo ""
    echo -e "${GREEN}Report saved to: $PROJECT_ROOT/$REPORT_FILE${NC}"
fi

# Show artifacts created
echo ""
echo -e "${YELLOW}Test artifacts in $WORK_DIR:${NC}"
ls -la "$WORK_DIR" 2>/dev/null || true

# Cleanup working directory
cd "$PROJECT_ROOT"
rm -rf "$WORK_DIR"

echo ""
echo -e "${BLUE}============================================${NC}"

if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}  Tests completed successfully!${NC}"
else
    echo -e "${RED}  Some tests failed (exit code: $TEST_EXIT_CODE)${NC}"
fi

echo -e "${BLUE}============================================${NC}"

# Keep server running if requested
if [[ "$KEEP_SERVER" == "true" ]]; then
    echo ""
    echo -e "${YELLOW}MCP server still running on http://localhost:$MCP_PORT${NC}"
    echo "Press Ctrl+C to stop"
    wait "$MCP_PID"
fi

exit $TEST_EXIT_CODE

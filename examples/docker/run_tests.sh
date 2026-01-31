#!/bin/bash
# Run ATP tests with Docker/Podman
#
# This script:
# 1. Detects container runtime (Docker or Podman)
# 2. Builds the agent container image
# 3. Runs ATP tests against the containerized agent
# 4. Shows results
#
# Usage:
#   ./examples/docker/run_tests.sh [options]
#
# Options:
#   --build       Force rebuild of container images
#   --compose     Use docker-compose/podman-compose to run tests
#   --dashboard   Start the dashboard after tests
#   --runtime     Force runtime: docker, podman (auto-detected by default)
#   --help        Show this help message
#
# Examples:
#   ./examples/docker/run_tests.sh                    # Quick test
#   ./examples/docker/run_tests.sh --build            # Rebuild and test
#   ./examples/docker/run_tests.sh --compose          # Use compose
#   ./examples/docker/run_tests.sh --dashboard        # Start dashboard after tests
#   ./examples/docker/run_tests.sh --runtime podman   # Force Podman

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Options
BUILD=false
USE_COMPOSE=false
START_DASHBOARD=false
FORCE_RUNTIME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD=true
            shift
            ;;
        --compose)
            USE_COMPOSE=true
            shift
            ;;
        --dashboard)
            START_DASHBOARD=true
            shift
            ;;
        --runtime)
            FORCE_RUNTIME="$2"
            shift 2
            ;;
        --help|-h)
            head -28 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# ===========================================
# Detect container runtime
# ===========================================
detect_runtime() {
    if [[ -n "$FORCE_RUNTIME" ]]; then
        echo "$FORCE_RUNTIME"
        return
    fi

    # Check for Podman first (preferred on systems where both are installed)
    if command -v podman &> /dev/null; then
        echo "podman"
    elif command -v docker &> /dev/null; then
        echo "docker"
    else
        echo ""
    fi
}

# Detect compose command
detect_compose() {
    local runtime="$1"

    if [[ "$runtime" == "podman" ]]; then
        # Try different podman compose variants
        if command -v "podman-compose" &> /dev/null; then
            echo "podman-compose"
        elif podman compose version &> /dev/null 2>&1; then
            echo "podman compose"
        else
            echo ""
        fi
    else
        # Docker compose variants
        if docker compose version &> /dev/null 2>&1; then
            echo "docker compose"
        elif command -v "docker-compose" &> /dev/null; then
            echo "docker-compose"
        else
            echo ""
        fi
    fi
}

cd "$PROJECT_ROOT"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  ATP Container Agent Tests${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Detect runtime
RUNTIME=$(detect_runtime)

if [[ -z "$RUNTIME" ]]; then
    echo -e "${RED}Error: No container runtime found (Docker or Podman)${NC}"
    echo "Please install Docker or Podman first."
    exit 1
fi

# Verify runtime is working
if ! $RUNTIME info &> /dev/null; then
    echo -e "${RED}Error: $RUNTIME is installed but not running${NC}"
    if [[ "$RUNTIME" == "docker" ]]; then
        echo "Try: sudo systemctl start docker"
    else
        echo "Try: systemctl --user start podman.socket"
    fi
    exit 1
fi

echo -e "${GREEN}Container runtime: $RUNTIME${NC}"

# Detect compose if needed
if [[ "$USE_COMPOSE" == "true" ]]; then
    COMPOSE_CMD=$(detect_compose "$RUNTIME")
    if [[ -z "$COMPOSE_CMD" ]]; then
        echo -e "${RED}Error: No compose command found${NC}"
        if [[ "$RUNTIME" == "podman" ]]; then
            echo "Try: pip install podman-compose"
            echo "Or use Podman 4.x+ with: podman compose"
        else
            echo "Try: sudo apt install docker-compose-plugin"
        fi
        exit 1
    fi
    echo -e "${GREEN}Compose command: $COMPOSE_CMD${NC}"
fi

# Build or check for image
IMAGE_NAME="atp-demo-agent:latest"

if [[ "$BUILD" == "true" ]] || ! $RUNTIME image inspect "$IMAGE_NAME" &> /dev/null; then
    echo ""
    echo -e "${YELLOW}Building agent image...${NC}"
    $RUNTIME build -t "$IMAGE_NAME" \
        -f examples/docker/Dockerfile.agent \
        examples/docker/
    echo -e "${GREEN}Image built: $IMAGE_NAME${NC}"
else
    echo -e "${GREEN}Using existing image: $IMAGE_NAME${NC}"
fi

echo ""

if [[ "$USE_COMPOSE" == "true" ]]; then
    # Use compose
    echo -e "${YELLOW}Using $COMPOSE_CMD...${NC}"
    cd "$SCRIPT_DIR"

    # Create results directory
    mkdir -p results

    # Build and run
    $COMPOSE_CMD build atp-test
    $COMPOSE_CMD run --rm atp-test

    # Show results
    if [[ -f results/test_results.json ]]; then
        echo ""
        echo -e "${GREEN}Results saved to: $SCRIPT_DIR/results/test_results.json${NC}"
        cat results/test_results.json | python -m json.tool | head -50
    fi

    if [[ "$START_DASHBOARD" == "true" ]]; then
        echo ""
        echo -e "${YELLOW}Starting dashboard...${NC}"
        $COMPOSE_CMD up -d dashboard
        echo -e "${GREEN}Dashboard available at: http://localhost:8080${NC}"
    fi
else
    # Run directly with uv
    echo -e "${GREEN}Running tests...${NC}"
    echo ""

    # Set container runtime for ATP (defaults to docker, but we may need podman)
    if [[ "$RUNTIME" == "podman" ]]; then
        export ATP_CONTAINER_RUNTIME="podman"
    fi

    uv run --project "$PROJECT_ROOT" atp test examples/test_suites/docker_agent_test.yaml \
        --adapter=container \
        --adapter-config="image=$IMAGE_NAME" \
        --adapter-config='resources={"memory": "512m", "cpu": "0.5"}' \
        --adapter-config='auto_remove=true'

    TEST_EXIT_CODE=$?

    echo ""
    echo -e "${BLUE}============================================${NC}"

    if [[ $TEST_EXIT_CODE -eq 0 ]]; then
        echo -e "${GREEN}  Tests completed successfully!${NC}"
    else
        echo -e "${RED}  Some tests failed (exit code: $TEST_EXIT_CODE)${NC}"
    fi

    echo -e "${BLUE}============================================${NC}"

    if [[ "$START_DASHBOARD" == "true" ]]; then
        echo ""
        echo -e "${YELLOW}Starting dashboard...${NC}"
        uv run --project "$PROJECT_ROOT" atp dashboard --port 8080 &
        sleep 2
        echo -e "${GREEN}Dashboard available at: http://localhost:8080${NC}"
        echo "Press Ctrl+C to stop"
        wait
    fi

    exit $TEST_EXIT_CODE
fi

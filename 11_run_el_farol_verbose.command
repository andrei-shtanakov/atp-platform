#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

uv run atp game run examples/test_suites/13_game_el_farol.yaml --episodes=20 -v

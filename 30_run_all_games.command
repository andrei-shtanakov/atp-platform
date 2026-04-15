#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running Prisoner's Dilemma..."
uv run atp game run examples/test_suites/11_game_prisoners_dilemma.yaml

echo ""
echo "Running Auction..."
uv run atp game run examples/test_suites/12_game_auction.yaml

echo ""
echo "Running El Farol..."
uv run atp game run examples/test_suites/13_game_el_farol.yaml

echo ""
echo "All game suites finished."

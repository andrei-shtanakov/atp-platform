#!/usr/bin/env bash
# ATP EPAM demo — Act 1 (on-prem) driver.
#
# Brings up the agent + dashboard, runs the suite over the http adapter, and
# points you at the dashboard. Works with docker or podman. See DEMO.md for the
# full runbook (including the cloud / Bedrock act).
#
# Usage:  ./run_demo.sh           # run the demo
#         ./run_demo.sh down      # tear everything down (removes the run DB)
set -euo pipefail
cd "$(dirname "$0")"

# Pick a compose CLI: prefer docker, fall back to podman.
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif podman compose version >/dev/null 2>&1; then
  COMPOSE="podman compose"
elif command -v podman-compose >/dev/null 2>&1; then
  COMPOSE="podman-compose"
else
  echo "error: need 'docker compose' or 'podman compose'." >&2
  exit 1
fi
echo "==> using: ${COMPOSE}"

if [[ "${1:-}" == "down" ]]; then
  ${COMPOSE} down -v
  echo "==> torn down."
  exit 0
fi

echo "==> building & starting agent + dashboard ..."
${COMPOSE} up --build -d agent dashboard

echo "==> running the suite over the http adapter ..."
${COMPOSE} run --rm atp

echo
echo "==> done. Open the dashboard:  http://localhost:8080/ui/"
echo "    JSON report:               results/results.json"
echo "    Tear down when finished:   ./run_demo.sh down"

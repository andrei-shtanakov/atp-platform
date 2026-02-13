#!/usr/bin/env bash
# ATP Test Action — entrypoint script
#
# Runs ATP tests, generates reports and outputs for the composite
# GitHub Action defined in action.yml.
#
# Expected environment variables (set by action.yml):
#   INPUT_SUITE_PATH    — path to the test suite YAML
#   INPUT_ADAPTER       — adapter name (optional)
#   INPUT_THRESHOLD     — minimum success rate 0.0-1.0 (optional)
#   INPUT_BUDGET_USD    — max estimated cost in USD (optional)
#   INPUT_BASELINE_PATH — baseline JSON for regression check (optional)
#   INPUT_EXTRA_ARGS    — extra CLI arguments (optional)
#
# Outputs (written via $GITHUB_OUTPUT):
#   success_rate, total_tests, passed_tests, failed_tests,
#   estimated_cost, junit_path, json_path, summary_markdown,
#   badge_url, regression_detected

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log_info()  { echo "::group::$1"; }
log_end()   { echo "::endgroup::"; }
log_error() { echo "::error::$1"; }

# Write a key=value pair to GITHUB_OUTPUT (or stdout for local testing).
set_output() {
  local key="$1"
  local value="$2"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    # Use heredoc delimiter for multi-line values
    {
      echo "${key}<<ATPEOF"
      echo "${value}"
      echo "ATPEOF"
    } >> "$GITHUB_OUTPUT"
  else
    echo "OUTPUT ${key}=${value}"
  fi
}

# Append text to the GitHub Actions job summary (or stdout).
append_summary() {
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    echo "$1" >> "$GITHUB_STEP_SUMMARY"
  else
    echo "SUMMARY: $1"
  fi
}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

SUITE_PATH="${INPUT_SUITE_PATH:?suite_path is required}"

if [[ ! -f "$SUITE_PATH" ]]; then
  log_error "Suite file not found: ${SUITE_PATH}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Build the atp test command
# ---------------------------------------------------------------------------

RESULTS_DIR="atp-results"
mkdir -p "$RESULTS_DIR"

JSON_PATH="${RESULTS_DIR}/results.json"
JUNIT_PATH="${RESULTS_DIR}/junit.xml"

CMD=(uv run atp test "$SUITE_PATH")

if [[ -n "${INPUT_ADAPTER:-}" ]]; then
  CMD+=(--adapter="$INPUT_ADAPTER")
fi

# Always generate JSON output
CMD+=(--output=json --output-file="$JSON_PATH")

# Append any extra arguments the caller passed
if [[ -n "${INPUT_EXTRA_ARGS:-}" ]]; then
  # Word-split intentionally
  # shellcheck disable=SC2206
  CMD+=($INPUT_EXTRA_ARGS)
fi

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

log_info "Running ATP tests"
echo "Command: ${CMD[*]}"

TEST_EXIT=0
"${CMD[@]}" || TEST_EXIT=$?

log_end

# Also produce JUnit XML (best-effort, ignore failures)
log_info "Generating JUnit report"
uv run atp test "$SUITE_PATH" \
  ${INPUT_ADAPTER:+--adapter="$INPUT_ADAPTER"} \
  --output=junit --output-file="$JUNIT_PATH" 2>/dev/null || true
log_end

# ---------------------------------------------------------------------------
# Parse results
# ---------------------------------------------------------------------------

SUCCESS_RATE="0"
TOTAL_TESTS="0"
PASSED_TESTS="0"
FAILED_TESTS="0"
ESTIMATED_COST="0"

if [[ -f "$JSON_PATH" ]]; then
  # jq is pre-installed on GitHub-hosted runners
  TOTAL_TESTS=$(jq -r '.summary.total_tests // .total_tests // 0' "$JSON_PATH")
  PASSED_TESTS=$(jq -r '.summary.passed_tests // .passed_tests // 0' "$JSON_PATH")
  FAILED_TESTS=$(jq -r '.summary.failed_tests // .failed_tests // 0' "$JSON_PATH")
  ESTIMATED_COST=$(jq -r '.summary.estimated_cost // .estimated_cost // 0' "$JSON_PATH")

  if [[ "$TOTAL_TESTS" -gt 0 ]]; then
    SUCCESS_RATE=$(echo "scale=4; $PASSED_TESTS / $TOTAL_TESTS" | bc -l)
  fi
else
  log_error "JSON results file not found: ${JSON_PATH}"
fi

# ---------------------------------------------------------------------------
# Set outputs
# ---------------------------------------------------------------------------

set_output "success_rate"  "$SUCCESS_RATE"
set_output "total_tests"   "$TOTAL_TESTS"
set_output "passed_tests"  "$PASSED_TESTS"
set_output "failed_tests"  "$FAILED_TESTS"
set_output "estimated_cost" "$ESTIMATED_COST"
set_output "junit_path"    "$JUNIT_PATH"
set_output "json_path"     "$JSON_PATH"

# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

if [[ "$TOTAL_TESTS" -gt 0 ]]; then
  STATUS_EMOJI="passed"
  if [[ "$FAILED_TESTS" -gt 0 ]]; then
    STATUS_EMOJI="failed"
  fi

  SUMMARY="## ATP Test Results

| Metric | Value |
|--------|-------|
| Total  | ${TOTAL_TESTS} |
| Passed | ${PASSED_TESTS} |
| Failed | ${FAILED_TESTS} |
| Success Rate | ${SUCCESS_RATE} |
| Estimated Cost | \$${ESTIMATED_COST} |
| Status | ${STATUS_EMOJI} |"
else
  SUMMARY="## ATP Test Results

No test results found."
fi

set_output "summary_markdown" "$SUMMARY"
append_summary "$SUMMARY"

# ---------------------------------------------------------------------------
# Badge URL (shields.io)
# ---------------------------------------------------------------------------

BADGE_COLOR="red"
BADGE_LABEL="ATP%20Tests"
BADGE_MESSAGE="${PASSED_TESTS}%2F${TOTAL_TESTS}%20passed"

if [[ "$FAILED_TESTS" -eq 0 && "$TOTAL_TESTS" -gt 0 ]]; then
  BADGE_COLOR="brightgreen"
elif [[ "$FAILED_TESTS" -gt 0 && "$PASSED_TESTS" -gt 0 ]]; then
  BADGE_COLOR="yellow"
fi

BADGE_URL="https://img.shields.io/badge/${BADGE_LABEL}-${BADGE_MESSAGE}-${BADGE_COLOR}"
set_output "badge_url" "$BADGE_URL"

# ---------------------------------------------------------------------------
# Threshold check
# ---------------------------------------------------------------------------

THRESHOLD_FAILED=0

if [[ -n "${INPUT_THRESHOLD:-}" ]]; then
  # Compare using bc (available on runners)
  BELOW=$(echo "${SUCCESS_RATE} < ${INPUT_THRESHOLD}" | bc -l)
  if [[ "$BELOW" -eq 1 ]]; then
    log_error "Success rate ${SUCCESS_RATE} is below threshold ${INPUT_THRESHOLD}"
    THRESHOLD_FAILED=1
  fi
fi

# ---------------------------------------------------------------------------
# Budget check
# ---------------------------------------------------------------------------

BUDGET_FAILED=0

if [[ -n "${INPUT_BUDGET_USD:-}" ]]; then
  OVER=$(echo "${ESTIMATED_COST} > ${INPUT_BUDGET_USD}" | bc -l)
  if [[ "$OVER" -eq 1 ]]; then
    log_error "Estimated cost \$${ESTIMATED_COST} exceeds budget \$${INPUT_BUDGET_USD}"
    BUDGET_FAILED=1
  fi
fi

# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

REGRESSION_DETECTED="false"

if [[ -n "${INPUT_BASELINE_PATH:-}" ]]; then
  if [[ -f "${INPUT_BASELINE_PATH}" ]]; then
    log_info "Running baseline comparison"
    BASELINE_EXIT=0
    uv run atp baseline compare "$SUITE_PATH" \
      -b "$INPUT_BASELINE_PATH" \
      --output=json \
      --output-file="${RESULTS_DIR}/comparison.json" || BASELINE_EXIT=$?
    log_end

    if [[ -f "${RESULTS_DIR}/comparison.json" ]]; then
      HAS_REG=$(jq -r '.has_regressions // false' "${RESULTS_DIR}/comparison.json")
      if [[ "$HAS_REG" == "true" ]]; then
        REGRESSION_DETECTED="true"
        log_error "Regression detected compared to baseline"
      fi
    fi
  else
    echo "::warning::Baseline file not found: ${INPUT_BASELINE_PATH}"
  fi
fi

set_output "regression_detected" "$REGRESSION_DETECTED"

# ---------------------------------------------------------------------------
# Final exit code
# ---------------------------------------------------------------------------

EXIT_CODE=0

if [[ "$THRESHOLD_FAILED" -eq 1 ]]; then
  EXIT_CODE=1
fi

if [[ "$BUDGET_FAILED" -eq 1 ]]; then
  EXIT_CODE=1
fi

if [[ "$REGRESSION_DETECTED" == "true" ]]; then
  EXIT_CODE=1
fi

# If the tests themselves failed and no other check caught it, propagate
if [[ "$TEST_EXIT" -ne 0 && "$EXIT_CODE" -eq 0 ]]; then
  EXIT_CODE="$TEST_EXIT"
fi

exit "$EXIT_CODE"

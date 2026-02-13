"""Tests for the ATP GitHub Action entrypoint script.

Validates the shell script logic by running it in a controlled
environment with mocked ``atp`` commands.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import textwrap
from pathlib import Path

# Path to the entrypoint script under test
ENTRYPOINT = Path(__file__).resolve().parents[2] / "action" / "entrypoint.sh"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_mock_atp(
    tmp_path: Path,
    json_results: dict | None = None,
) -> Path:
    """Create a mock ``atp`` wrapper and a ``uv`` shim.

    The shim intercepts ``uv run atp test ...`` and writes
    *json_results* to the ``--output-file`` when ``--output=json``
    is requested.  For JUnit output it writes an empty XML file.
    """
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir(exist_ok=True)

    # Build the payload the mock will write for JSON output
    payload = json.dumps(json_results) if json_results else "{}"

    # A tiny shell script pretending to be ``uv``
    uv_script = textwrap.dedent(f"""\
        #!/usr/bin/env bash
        # mock uv — only handles "uv run atp test ..." and "uv sync"
        if [[ "$1" == "sync" ]]; then
            exit 0
        fi
        if [[ "$1" != "run" || "$2" != "atp" ]]; then
            exit 0
        fi
        OUTPUT_FILE=""
        OUTPUT_FMT=""
        for arg in "$@"; do
            case "$arg" in
                --output-file=*) OUTPUT_FILE="${{arg#--output-file=}}" ;;
                --output=*)      OUTPUT_FMT="${{arg#--output=}}" ;;
            esac
        done
        if [[ "$OUTPUT_FMT" == "json" && -n "$OUTPUT_FILE" ]]; then
            mkdir -p "$(dirname "$OUTPUT_FILE")"
            cat > "$OUTPUT_FILE" <<'PAYLOAD'
{payload}
PAYLOAD
        fi
        if [[ "$OUTPUT_FMT" == "junit" && -n "$OUTPUT_FILE" ]]; then
            mkdir -p "$(dirname "$OUTPUT_FILE")"
            echo '<testsuites/>' > "$OUTPUT_FILE"
        fi
        exit 0
    """)

    uv_path = mock_bin / "uv"
    uv_path.write_text(uv_script)
    uv_path.chmod(uv_path.stat().st_mode | stat.S_IEXEC)

    return mock_bin


def _run_entrypoint(
    tmp_path: Path,
    *,
    suite_text: str = "test_suite: demo\ntests: []\n",
    json_results: dict | None = None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute the entrypoint script and return the result.

    Creates a minimal suite file and mock ``uv`` in *tmp_path*,
    then runs the entrypoint with appropriate env vars.
    """
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(suite_text)

    if json_results is None:
        json_results = {
            "summary": {
                "total_tests": 5,
                "passed_tests": 4,
                "failed_tests": 1,
                "estimated_cost": 0.12,
            }
        }

    mock_bin = _make_mock_atp(tmp_path, json_results)
    output_file = tmp_path / "gh_output.txt"
    summary_file = tmp_path / "gh_summary.txt"
    output_file.touch()
    summary_file.touch()

    env: dict[str, str] = {
        "PATH": f"{mock_bin}:{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
        "INPUT_SUITE_PATH": str(suite_file),
        "INPUT_ADAPTER": "",
        "INPUT_THRESHOLD": "",
        "INPUT_BUDGET_USD": "",
        "INPUT_BASELINE_PATH": "",
        "INPUT_COMMENT_ON_PR": "false",
        "INPUT_EXTRA_ARGS": "",
        "GITHUB_OUTPUT": str(output_file),
        "GITHUB_STEP_SUMMARY": str(summary_file),
    }
    if env_overrides:
        env.update(env_overrides)

    result = subprocess.run(
        ["bash", str(ENTRYPOINT)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        timeout=30,
    )
    return result


def _parse_outputs(output_file: Path) -> dict[str, str]:
    """Parse GITHUB_OUTPUT heredoc-style key=value pairs."""
    outputs: dict[str, str] = {}
    text = output_file.read_text()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "<<ATPEOF" in line:
            key = line.split("<<")[0].strip()
            value_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i] != "ATPEOF":
                value_lines.append(lines[i])
                i += 1
            outputs[key] = "\n".join(value_lines)
        i += 1
    return outputs


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestEntrypointValidation:
    """Input validation tests."""

    def test_missing_suite_file_fails(self, tmp_path: Path) -> None:
        """Entrypoint exits non-zero when suite file is missing."""
        mock_bin = _make_mock_atp(tmp_path)
        output_file = tmp_path / "gh_output.txt"
        output_file.touch()

        env = {
            "PATH": f"{mock_bin}:{os.environ.get('PATH', '')}",
            "HOME": str(tmp_path),
            "INPUT_SUITE_PATH": str(tmp_path / "nonexistent.yaml"),
            "INPUT_ADAPTER": "",
            "INPUT_THRESHOLD": "",
            "INPUT_BUDGET_USD": "",
            "INPUT_BASELINE_PATH": "",
            "INPUT_COMMENT_ON_PR": "false",
            "INPUT_EXTRA_ARGS": "",
            "GITHUB_OUTPUT": str(output_file),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.txt"),
        }

        result = subprocess.run(
            ["bash", str(ENTRYPOINT)],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env=env,
            timeout=30,
        )
        assert result.returncode != 0
        assert (
            "not found" in result.stderr.lower() or "not found" in result.stdout.lower()
        )


class TestEntrypointOutputs:
    """Verify that the script produces expected outputs."""

    def test_basic_run_sets_outputs(self, tmp_path: Path) -> None:
        """A successful run populates all expected outputs."""
        _run_entrypoint(tmp_path)

        # Script may fail due to threshold/budget but outputs
        # should still be written
        output_file = tmp_path / "gh_output.txt"
        outputs = _parse_outputs(output_file)

        assert "success_rate" in outputs
        assert "total_tests" in outputs
        assert outputs["total_tests"].strip() == "5"
        assert outputs["passed_tests"].strip() == "4"
        assert outputs["failed_tests"].strip() == "1"

    def test_success_rate_calculation(self, tmp_path: Path) -> None:
        """Success rate is computed as passed / total."""
        json_results = {
            "summary": {
                "total_tests": 10,
                "passed_tests": 8,
                "failed_tests": 2,
                "estimated_cost": 0.5,
            }
        }
        _run_entrypoint(tmp_path, json_results=json_results)

        outputs = _parse_outputs(tmp_path / "gh_output.txt")
        rate = float(outputs["success_rate"].strip())
        assert abs(rate - 0.8) < 0.01

    def test_estimated_cost_output(self, tmp_path: Path) -> None:
        """Estimated cost is forwarded to outputs."""
        json_results = {
            "summary": {
                "total_tests": 3,
                "passed_tests": 3,
                "failed_tests": 0,
                "estimated_cost": 1.23,
            }
        }
        _run_entrypoint(tmp_path, json_results=json_results)

        outputs = _parse_outputs(tmp_path / "gh_output.txt")
        assert outputs["estimated_cost"].strip() == "1.23"

    def test_junit_and_json_paths(self, tmp_path: Path) -> None:
        """Output paths for JUnit and JSON reports are set."""
        _run_entrypoint(tmp_path)

        outputs = _parse_outputs(tmp_path / "gh_output.txt")
        assert "junit.xml" in outputs.get("junit_path", "")
        assert "results.json" in outputs.get("json_path", "")


class TestEntrypointBadge:
    """Badge URL generation."""

    def test_all_passing_green_badge(self, tmp_path: Path) -> None:
        """All tests passing produces a green badge."""
        json_results = {
            "summary": {
                "total_tests": 5,
                "passed_tests": 5,
                "failed_tests": 0,
                "estimated_cost": 0,
            }
        }
        _run_entrypoint(tmp_path, json_results=json_results)

        outputs = _parse_outputs(tmp_path / "gh_output.txt")
        badge = outputs.get("badge_url", "")
        assert "brightgreen" in badge
        assert "5%2F5" in badge

    def test_partial_failure_yellow_badge(self, tmp_path: Path) -> None:
        """Mixed results produce a yellow badge."""
        json_results = {
            "summary": {
                "total_tests": 4,
                "passed_tests": 3,
                "failed_tests": 1,
                "estimated_cost": 0,
            }
        }
        _run_entrypoint(tmp_path, json_results=json_results)

        outputs = _parse_outputs(tmp_path / "gh_output.txt")
        badge = outputs.get("badge_url", "")
        assert "yellow" in badge

    def test_all_failing_red_badge(self, tmp_path: Path) -> None:
        """All tests failing produces a red badge."""
        json_results = {
            "summary": {
                "total_tests": 3,
                "passed_tests": 0,
                "failed_tests": 3,
                "estimated_cost": 0,
            }
        }
        _run_entrypoint(tmp_path, json_results=json_results)

        outputs = _parse_outputs(tmp_path / "gh_output.txt")
        badge = outputs.get("badge_url", "")
        assert "red" in badge
        assert "brightgreen" not in badge


class TestEntrypointSummary:
    """Markdown summary generation."""

    def test_summary_contains_results(self, tmp_path: Path) -> None:
        """Step summary file contains a results table."""
        _run_entrypoint(tmp_path)

        summary = (tmp_path / "gh_summary.txt").read_text()
        assert "ATP Test Results" in summary
        assert "Passed" in summary
        assert "Failed" in summary

    def test_summary_output_set(self, tmp_path: Path) -> None:
        """summary_markdown output is populated."""
        _run_entrypoint(tmp_path)

        outputs = _parse_outputs(tmp_path / "gh_output.txt")
        md = outputs.get("summary_markdown", "")
        assert "ATP Test Results" in md


class TestThresholdCheck:
    """Threshold enforcement."""

    def test_below_threshold_fails(self, tmp_path: Path) -> None:
        """Exit code is non-zero when success rate < threshold."""
        json_results = {
            "summary": {
                "total_tests": 10,
                "passed_tests": 7,
                "failed_tests": 3,
                "estimated_cost": 0,
            }
        }
        result = _run_entrypoint(
            tmp_path,
            json_results=json_results,
            env_overrides={"INPUT_THRESHOLD": "0.8"},
        )
        assert result.returncode != 0

    def test_above_threshold_passes(self, tmp_path: Path) -> None:
        """Exit code is zero when success rate >= threshold."""
        json_results = {
            "summary": {
                "total_tests": 10,
                "passed_tests": 9,
                "failed_tests": 1,
                "estimated_cost": 0,
            }
        }
        result = _run_entrypoint(
            tmp_path,
            json_results=json_results,
            env_overrides={"INPUT_THRESHOLD": "0.8"},
        )
        assert result.returncode == 0


class TestBudgetCheck:
    """Budget enforcement."""

    def test_over_budget_fails(self, tmp_path: Path) -> None:
        """Exit code is non-zero when cost exceeds budget."""
        json_results = {
            "summary": {
                "total_tests": 5,
                "passed_tests": 5,
                "failed_tests": 0,
                "estimated_cost": 2.50,
            }
        }
        result = _run_entrypoint(
            tmp_path,
            json_results=json_results,
            env_overrides={"INPUT_BUDGET_USD": "1.00"},
        )
        assert result.returncode != 0

    def test_under_budget_passes(self, tmp_path: Path) -> None:
        """Exit code is zero when cost is within budget."""
        json_results = {
            "summary": {
                "total_tests": 5,
                "passed_tests": 5,
                "failed_tests": 0,
                "estimated_cost": 0.50,
            }
        }
        result = _run_entrypoint(
            tmp_path,
            json_results=json_results,
            env_overrides={"INPUT_BUDGET_USD": "1.00"},
        )
        assert result.returncode == 0


class TestAdapterInput:
    """Adapter parameter forwarding."""

    def test_adapter_passed_to_command(self, tmp_path: Path) -> None:
        """When adapter is set, the command includes --adapter."""
        json_results = {
            "summary": {
                "total_tests": 1,
                "passed_tests": 1,
                "failed_tests": 0,
                "estimated_cost": 0,
            }
        }
        result = _run_entrypoint(
            tmp_path,
            json_results=json_results,
            env_overrides={"INPUT_ADAPTER": "cli"},
        )
        # The command is echoed to stdout
        assert "--adapter=cli" in result.stdout

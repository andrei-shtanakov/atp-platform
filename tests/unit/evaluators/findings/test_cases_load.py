"""The converted code-review cases load via the atp-method plugin."""

import subprocess


def test_code_review_cases_list_via_plugin() -> None:
    out = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "atp",
            "test",
            "method/cases/code-review",
            "--list-only",
            "--adapter=cli",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = out.stdout + out.stderr
    assert "case-code-review-sqli-clean-001" in combined
    assert "case-code-review-sqli-moderate-001" in combined

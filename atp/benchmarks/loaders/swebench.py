"""SWE-bench benchmark loader.

SWE-bench is a software engineering benchmark of real-world GitHub
issues and their corresponding pull request fixes.

Source: https://github.com/princeton-nlp/SWE-bench
"""

from __future__ import annotations

import httpx

from atp.benchmarks.loaders.base import BenchmarkLoader
from atp.loader.models import (
    Assertion,
    Constraints,
    TaskDefinition,
    TestDefinition,
    TestSuite,
)

SWEBENCH_HF_API = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=princeton-nlp/SWE-bench_Lite"
    "&config=default"
    "&split=test"
    "&offset=0&length=300"
)


class SWEBenchLoader(BenchmarkLoader):
    """Loader for the SWE-bench software engineering benchmark.

    Converts SWE-bench instances into ATP test cases where the
    agent must produce a patch that resolves the given GitHub issue.
    """

    @property
    def name(self) -> str:
        """Human-readable benchmark name."""
        return "swe-bench"

    @property
    def description(self) -> str:
        """Short description of the benchmark."""
        return (
            "SWE-bench Lite: Real-world GitHub issues "
            "requiring code patches to resolve."
        )

    @property
    def source_url(self) -> str:
        """URL to fetch the benchmark data from."""
        return SWEBENCH_HF_API

    def _parse_response(self, response: httpx.Response) -> list[dict]:
        """Parse HuggingFace API response.

        Args:
            response: HTTP response from HuggingFace datasets API.

        Returns:
            List of SWE-bench instances.
        """
        data = response.json()
        rows = data.get("rows", [])
        return [row.get("row", row) for row in rows]

    def _convert_items(self, items: list[dict], limit: int | None = None) -> TestSuite:
        """Convert SWE-bench items to an ATP TestSuite.

        Args:
            items: Raw SWE-bench instances.
            limit: Maximum number of instances to include.

        Returns:
            ATP TestSuite.
        """
        if limit is not None:
            items = items[:limit]

        tests: list[TestDefinition] = []
        for item in items:
            instance_id = item.get("instance_id", "")
            problem_statement = item.get("problem_statement", "")
            repo = item.get("repo", "")
            base_commit = item.get("base_commit", "")
            hints = item.get("hints_text", "")
            patch = item.get("patch", "")

            test_id = instance_id.replace("/", "_").replace("-", "_")

            task_desc = (
                f"Fix the following GitHub issue in the "
                f"repository '{repo}'.\n\n"
                f"## Issue Description\n\n"
                f"{problem_statement}\n\n"
            )
            if hints:
                task_desc += f"## Hints\n\n{hints}\n\n"
            task_desc += (
                f"Base commit: {base_commit}\n\n"
                f"Provide a unified diff patch that resolves "
                f"the issue."
            )

            assertions: list[Assertion] = []
            if patch:
                assertions.append(
                    Assertion(
                        type="artifact_contains",
                        config={
                            "artifact": "patch",
                            "description": ("Patch should modify the relevant files"),
                        },
                    )
                )

            tags = ["swe-bench", "software-engineering", "patch"]
            if repo:
                tags.append(f"repo:{repo}")

            test_def = TestDefinition(
                id=test_id,
                name=f"SWE-bench {instance_id}",
                description=(f"Resolve issue in {repo} (commit {base_commit[:8]})"),
                tags=tags,
                task=TaskDefinition(
                    description=task_desc,
                    input_data={
                        "instance_id": instance_id,
                        "repo": repo,
                        "base_commit": base_commit,
                        "patch": patch,
                    },
                ),
                constraints=Constraints(
                    timeout_seconds=300,
                    max_tokens=8192,
                ),
                assertions=assertions,
            )
            tests.append(test_def)

        if not tests:
            tests.append(
                TestDefinition(
                    id="swebench_placeholder",
                    name="SWE-bench Placeholder",
                    task=TaskDefinition(description="No SWE-bench instances loaded."),
                    tags=["swe-bench"],
                )
            )

        return TestSuite(
            test_suite="swe-bench",
            version="1.0",
            description=self.description,
            tests=tests,
        )

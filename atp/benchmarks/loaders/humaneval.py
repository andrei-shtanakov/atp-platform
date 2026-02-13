"""HumanEval benchmark loader.

HumanEval is a code generation benchmark with 164 hand-written
programming problems. Each problem includes a function signature,
docstring, reference solution, and unit tests.

Source: https://github.com/openai/human-eval
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

HUMANEVAL_URL = (
    "https://huggingface.co/datasets/openai/openai_humaneval"
    "/resolve/main/openai_humaneval/test-00000-of-00001.parquet"
)

# HumanEval uses JSONL via the raw GitHub source
HUMANEVAL_JSONL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
)

# We use the HuggingFace API for easier access (JSON lines)
HUMANEVAL_HF_API = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=openai/openai_humaneval"
    "&config=openai_humaneval"
    "&split=test"
    "&offset=0&length=200"
)


class HumanEvalLoader(BenchmarkLoader):
    """Loader for the HumanEval code generation benchmark.

    Converts HumanEval problems into ATP test cases where the agent
    must generate a Python function that passes the provided tests.
    """

    @property
    def name(self) -> str:
        """Human-readable benchmark name."""
        return "humaneval"

    @property
    def description(self) -> str:
        """Short description of the benchmark."""
        return (
            "HumanEval: 164 hand-written Python programming "
            "problems for evaluating code generation."
        )

    @property
    def source_url(self) -> str:
        """URL to fetch the benchmark data from."""
        return HUMANEVAL_HF_API

    def _parse_response(self, response: httpx.Response) -> list[dict]:
        """Parse HuggingFace API response.

        Args:
            response: HTTP response from HuggingFace datasets API.

        Returns:
            List of HumanEval items.
        """

        data = response.json()
        rows = data.get("rows", [])
        return [row.get("row", row) for row in rows]

    def _convert_items(self, items: list[dict], limit: int | None = None) -> TestSuite:
        """Convert HumanEval items to an ATP TestSuite.

        Args:
            items: Raw HumanEval problems.
            limit: Maximum number of problems to include.

        Returns:
            ATP TestSuite.
        """
        if limit is not None:
            items = items[:limit]

        tests: list[TestDefinition] = []
        for item in items:
            task_id = item.get("task_id", "")
            prompt = item.get("prompt", "")
            test_code = item.get("test", "")
            entry_point = item.get("entry_point", "")
            canonical = item.get("canonical_solution", "")

            test_id = task_id.replace("/", "_")

            task_desc = (
                f"Write a Python function that solves the "
                f"following problem.\n\n"
                f"## Function Signature & Docstring\n\n"
                f"```python\n{prompt}```\n\n"
                f"Return ONLY the complete function "
                f"implementation."
            )

            assertions: list[Assertion] = []
            if test_code:
                assertions.append(
                    Assertion(
                        type="code_eval",
                        config={
                            "language": "python",
                            "test_code": test_code,
                            "entry_point": entry_point,
                        },
                    )
                )

            test_def = TestDefinition(
                id=test_id,
                name=f"HumanEval {task_id}",
                description=(
                    f"Generate function '{entry_point}' that passes the provided tests."
                ),
                tags=["humaneval", "code-generation", "python"],
                task=TaskDefinition(
                    description=task_desc,
                    input_data={
                        "prompt": prompt,
                        "entry_point": entry_point,
                        "canonical_solution": canonical,
                    },
                ),
                constraints=Constraints(
                    timeout_seconds=120,
                    max_tokens=4096,
                ),
                assertions=assertions,
            )
            tests.append(test_def)

        if not tests:
            tests.append(
                TestDefinition(
                    id="humaneval_placeholder",
                    name="HumanEval Placeholder",
                    task=TaskDefinition(description="No HumanEval problems loaded."),
                    tags=["humaneval"],
                )
            )

        return TestSuite(
            test_suite="humaneval",
            version="1.0",
            description=self.description,
            tests=tests,
        )

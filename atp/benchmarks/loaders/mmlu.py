"""MMLU benchmark loader.

MMLU (Massive Multitask Language Understanding) tests knowledge
and reasoning across 57 subjects from STEM, humanities, social
sciences, and other domains.

Source: https://github.com/hendrycks/test
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

MMLU_HF_API = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=cais/mmlu"
    "&config=all"
    "&split=test"
    "&offset=0&length=500"
)

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


class MMLULoader(BenchmarkLoader):
    """Loader for the MMLU knowledge/reasoning benchmark.

    Converts MMLU questions into ATP test cases where the agent
    must select the correct answer from four choices.
    """

    @property
    def name(self) -> str:
        """Human-readable benchmark name."""
        return "mmlu"

    @property
    def description(self) -> str:
        """Short description of the benchmark."""
        return "MMLU: Massive Multitask Language Understanding across 57 subjects."

    @property
    def source_url(self) -> str:
        """URL to fetch the benchmark data from."""
        return MMLU_HF_API

    def _parse_response(self, response: httpx.Response) -> list[dict]:
        """Parse HuggingFace API response.

        Args:
            response: HTTP response from HuggingFace datasets API.

        Returns:
            List of MMLU items.
        """
        data = response.json()
        rows = data.get("rows", [])
        return [row.get("row", row) for row in rows]

    def _convert_items(self, items: list[dict], limit: int | None = None) -> TestSuite:
        """Convert MMLU items to an ATP TestSuite.

        Args:
            items: Raw MMLU questions.
            limit: Maximum number of questions to include.

        Returns:
            ATP TestSuite.
        """
        if limit is not None:
            items = items[:limit]

        tests: list[TestDefinition] = []
        for idx, item in enumerate(items):
            question = item.get("question", "")
            choices = item.get("choices", [])
            answer_idx = item.get("answer", 0)
            subject = item.get("subject", "general")

            correct_letter = ANSWER_MAP.get(answer_idx, "A")

            choices_text = "\n".join(
                f"{ANSWER_MAP.get(i, '?')}. {c}" for i, c in enumerate(choices)
            )

            task_desc = (
                f"Answer the following multiple-choice question.\n\n"
                f"Subject: {subject}\n\n"
                f"## Question\n\n"
                f"{question}\n\n"
                f"## Choices\n\n"
                f"{choices_text}\n\n"
                f"Respond with ONLY the letter of the correct "
                f"answer (A, B, C, or D)."
            )

            test_id = f"mmlu_{subject}_{idx}"

            assertions: list[Assertion] = [
                Assertion(
                    type="artifact_contains",
                    config={
                        "expected": correct_letter,
                        "description": (f"Answer should be {correct_letter}"),
                    },
                )
            ]

            test_def = TestDefinition(
                id=test_id,
                name=f"MMLU {subject} #{idx}",
                description=(f"Multiple-choice question on {subject}."),
                tags=[
                    "mmlu",
                    "knowledge",
                    "reasoning",
                    f"subject:{subject}",
                ],
                task=TaskDefinition(
                    description=task_desc,
                    input_data={
                        "question": question,
                        "choices": choices,
                        "answer": answer_idx,
                        "subject": subject,
                    },
                ),
                constraints=Constraints(
                    timeout_seconds=60,
                    max_tokens=256,
                ),
                assertions=assertions,
            )
            tests.append(test_def)

        if not tests:
            tests.append(
                TestDefinition(
                    id="mmlu_placeholder",
                    name="MMLU Placeholder",
                    task=TaskDefinition(description="No MMLU questions loaded."),
                    tags=["mmlu"],
                )
            )

        return TestSuite(
            test_suite="mmlu",
            version="1.0",
            description=self.description,
            tests=tests,
        )

"""Tests for benchmark loaders (HumanEval, SWE-bench, MMLU)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from atp.benchmarks.loaders import LOADERS, get_loader
from atp.benchmarks.loaders.base import DEFAULT_CACHE_DIR
from atp.benchmarks.loaders.humaneval import HumanEvalLoader
from atp.benchmarks.loaders.mmlu import MMLULoader
from atp.benchmarks.loaders.swebench import SWEBenchLoader
from atp.loader.models import TestSuite

# ── Mock data ──────────────────────────────────────────────


MOCK_HUMANEVAL_RESPONSE = {
    "rows": [
        {
            "row": {
                "task_id": "HumanEval/0",
                "prompt": (
                    "def has_close_elements(numbers, threshold):\n"
                    '    """Check if any two numbers are closer '
                    'than threshold."""\n'
                ),
                "canonical_solution": (
                    "    for i, n1 in enumerate(numbers):\n"
                    "        for n2 in numbers[i+1:]:\n"
                    "            if abs(n1 - n2) < threshold:\n"
                    "                return True\n"
                    "    return False\n"
                ),
                "test": (
                    "def check(candidate):\n"
                    "    assert candidate([1.0, 2.0], 0.5) == False\n"
                    "    assert candidate([1.0, 1.1], 0.2) == True\n"
                ),
                "entry_point": "has_close_elements",
            }
        },
        {
            "row": {
                "task_id": "HumanEval/1",
                "prompt": (
                    "def separate_paren_groups(paren_string):\n"
                    '    """Separate groups of parentheses."""\n'
                ),
                "canonical_solution": "    pass\n",
                "test": (
                    "def check(candidate):\n    assert candidate('()') == ['()']\n"
                ),
                "entry_point": "separate_paren_groups",
            }
        },
    ]
}


MOCK_SWEBENCH_RESPONSE = {
    "rows": [
        {
            "row": {
                "instance_id": "astropy__astropy-12907",
                "repo": "astropy/astropy",
                "base_commit": "abc12345",
                "problem_statement": "Fix units conversion bug",
                "hints_text": "Check the units module",
                "patch": "--- a/file.py\n+++ b/file.py\n@@ ...",
            }
        },
        {
            "row": {
                "instance_id": "django__django-11099",
                "repo": "django/django",
                "base_commit": "def67890",
                "problem_statement": "Fix queryset ordering",
                "hints_text": "",
                "patch": "--- a/qs.py\n+++ b/qs.py\n@@ ...",
            }
        },
    ]
}


MOCK_MMLU_RESPONSE = {
    "rows": [
        {
            "row": {
                "question": "What is the capital of France?",
                "choices": ["London", "Paris", "Berlin", "Madrid"],
                "answer": 1,
                "subject": "geography",
            }
        },
        {
            "row": {
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1,
                "subject": "math",
            }
        },
        {
            "row": {
                "question": "Which planet is closest to the Sun?",
                "choices": [
                    "Venus",
                    "Mercury",
                    "Earth",
                    "Mars",
                ],
                "answer": 1,
                "subject": "astronomy",
            }
        },
    ]
}


# ── Helper ─────────────────────────────────────────────────


def _make_mock_response(data: dict) -> httpx.Response:
    """Create a mock httpx.Response with JSON data."""
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = data
    response.status_code = 200
    return response


# ── get_loader tests ───────────────────────────────────────


class TestGetLoader:
    """Tests for the get_loader factory function."""

    def test_get_humaneval_loader(self) -> None:
        loader = get_loader("humaneval")
        assert isinstance(loader, HumanEvalLoader)

    def test_get_swebench_loader(self) -> None:
        loader = get_loader("swe-bench")
        assert isinstance(loader, SWEBenchLoader)

    def test_get_mmlu_loader(self) -> None:
        loader = get_loader("mmlu")
        assert isinstance(loader, MMLULoader)

    def test_unknown_loader_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_loader("nonexistent")

    def test_loaders_registry(self) -> None:
        assert set(LOADERS.keys()) == {"humaneval", "swe-bench", "mmlu"}


# ── Base loader tests ─────────────────────────────────────


class TestBenchmarkLoaderBase:
    """Tests for the BenchmarkLoader base class."""

    def test_default_cache_dir(self) -> None:
        loader = HumanEvalLoader()
        assert loader.cache_dir == DEFAULT_CACHE_DIR

    def test_custom_cache_dir(self, tmp_path: Path) -> None:
        loader = HumanEvalLoader(cache_dir=tmp_path)
        assert loader.cache_dir == tmp_path

    def test_cache_round_trip(self, tmp_path: Path) -> None:
        loader = HumanEvalLoader(cache_dir=tmp_path)
        data = [{"task_id": "test/0", "prompt": "def f(): pass"}]

        assert loader._load_from_cache() is None
        loader._save_to_cache(data)
        cached = loader._load_from_cache()
        assert cached == data

    def test_cache_corrupted_returns_none(self, tmp_path: Path) -> None:
        loader = HumanEvalLoader(cache_dir=tmp_path)
        loader._ensure_cache_dir()
        loader._cache_path().write_text("not json")
        assert loader._load_from_cache() is None

    def test_cache_not_list_returns_none(self, tmp_path: Path) -> None:
        loader = HumanEvalLoader(cache_dir=tmp_path)
        loader._ensure_cache_dir()
        loader._cache_path().write_text(json.dumps({"key": "val"}))
        assert loader._load_from_cache() is None

    def test_download_uses_cache(self, tmp_path: Path) -> None:
        loader = HumanEvalLoader(cache_dir=tmp_path)
        cached_data = [{"task_id": "cached"}]
        loader._save_to_cache(cached_data)

        result = loader.download()
        assert result == cached_data

    @patch.object(HumanEvalLoader, "_fetch_data")
    def test_download_fetches_when_no_cache(
        self, mock_fetch: MagicMock, tmp_path: Path
    ) -> None:
        loader = HumanEvalLoader(cache_dir=tmp_path)
        fetched = [{"task_id": "HumanEval/0"}]
        mock_fetch.return_value = fetched

        result = loader.download()
        assert result == fetched
        mock_fetch.assert_called_once()
        # Verify it was cached
        assert loader._load_from_cache() == fetched

    def test_export_yaml(self, tmp_path: Path) -> None:
        loader = HumanEvalLoader(cache_dir=tmp_path)
        items = MOCK_HUMANEVAL_RESPONSE["rows"]
        items_data = [r["row"] for r in items]
        loader._save_to_cache(items_data)

        output = tmp_path / "suite.yaml"
        result = loader.export_yaml(output=output, limit=1)

        assert result == output
        assert output.exists()
        content = output.read_text()
        assert "humaneval" in content


# ── HumanEval loader tests ────────────────────────────────


class TestHumanEvalLoader:
    """Tests for the HumanEval benchmark loader."""

    def test_properties(self) -> None:
        loader = HumanEvalLoader()
        assert loader.name == "humaneval"
        assert "HumanEval" in loader.description
        assert "huggingface" in loader.source_url

    def test_parse_response(self) -> None:
        loader = HumanEvalLoader()
        response = _make_mock_response(MOCK_HUMANEVAL_RESPONSE)
        items = loader._parse_response(response)
        assert len(items) == 2
        assert items[0]["task_id"] == "HumanEval/0"

    def test_convert_items(self) -> None:
        loader = HumanEvalLoader()
        items = [r["row"] for r in MOCK_HUMANEVAL_RESPONSE["rows"]]
        suite = loader._convert_items(items)

        assert isinstance(suite, TestSuite)
        assert suite.test_suite == "humaneval"
        assert len(suite.tests) == 2

        test = suite.tests[0]
        assert test.id == "HumanEval_0"
        assert "has_close_elements" in test.task.description
        assert test.constraints.timeout_seconds == 120
        assert "humaneval" in test.tags
        assert "code-generation" in test.tags
        assert len(test.assertions) == 1
        assert test.assertions[0].type == "code_eval"

    def test_convert_items_with_limit(self) -> None:
        loader = HumanEvalLoader()
        items = [r["row"] for r in MOCK_HUMANEVAL_RESPONSE["rows"]]
        suite = loader._convert_items(items, limit=1)
        assert len(suite.tests) == 1

    def test_convert_empty_items(self) -> None:
        loader = HumanEvalLoader()
        suite = loader._convert_items([])
        assert len(suite.tests) == 1
        assert suite.tests[0].id == "humaneval_placeholder"

    def test_load_with_mock_data(self, tmp_path: Path) -> None:
        loader = HumanEvalLoader(cache_dir=tmp_path)
        items = [r["row"] for r in MOCK_HUMANEVAL_RESPONSE["rows"]]
        loader._save_to_cache(items)

        suite = loader.load(limit=2)
        assert isinstance(suite, TestSuite)
        assert len(suite.tests) == 2

    def test_input_data_preserved(self) -> None:
        loader = HumanEvalLoader()
        items = [r["row"] for r in MOCK_HUMANEVAL_RESPONSE["rows"]]
        suite = loader._convert_items(items)
        test = suite.tests[0]
        assert test.task.input_data is not None
        assert "prompt" in test.task.input_data
        assert "entry_point" in test.task.input_data
        assert test.task.input_data["entry_point"] == "has_close_elements"


# ── SWE-bench loader tests ────────────────────────────────


class TestSWEBenchLoader:
    """Tests for the SWE-bench benchmark loader."""

    def test_properties(self) -> None:
        loader = SWEBenchLoader()
        assert loader.name == "swe-bench"
        assert "SWE-bench" in loader.description
        assert "huggingface" in loader.source_url

    def test_parse_response(self) -> None:
        loader = SWEBenchLoader()
        response = _make_mock_response(MOCK_SWEBENCH_RESPONSE)
        items = loader._parse_response(response)
        assert len(items) == 2
        assert items[0]["instance_id"] == "astropy__astropy-12907"

    def test_convert_items(self) -> None:
        loader = SWEBenchLoader()
        items = [r["row"] for r in MOCK_SWEBENCH_RESPONSE["rows"]]
        suite = loader._convert_items(items)

        assert isinstance(suite, TestSuite)
        assert suite.test_suite == "swe-bench"
        assert len(suite.tests) == 2

        test = suite.tests[0]
        assert test.id == "astropy__astropy_12907"
        assert "astropy/astropy" in test.task.description
        assert test.constraints.timeout_seconds == 300
        assert "swe-bench" in test.tags
        assert "repo:astropy/astropy" in test.tags
        assert len(test.assertions) == 1

    def test_convert_items_with_limit(self) -> None:
        loader = SWEBenchLoader()
        items = [r["row"] for r in MOCK_SWEBENCH_RESPONSE["rows"]]
        suite = loader._convert_items(items, limit=1)
        assert len(suite.tests) == 1

    def test_convert_empty_items(self) -> None:
        loader = SWEBenchLoader()
        suite = loader._convert_items([])
        assert len(suite.tests) == 1
        assert suite.tests[0].id == "swebench_placeholder"

    def test_hints_included_in_description(self) -> None:
        loader = SWEBenchLoader()
        items = [r["row"] for r in MOCK_SWEBENCH_RESPONSE["rows"]]
        suite = loader._convert_items(items)
        # First item has hints
        assert "## Hints" in suite.tests[0].task.description
        # Second item has no hints
        assert "## Hints" not in suite.tests[1].task.description

    def test_input_data_preserved(self) -> None:
        loader = SWEBenchLoader()
        items = [r["row"] for r in MOCK_SWEBENCH_RESPONSE["rows"]]
        suite = loader._convert_items(items)
        test = suite.tests[0]
        assert test.task.input_data is not None
        assert test.task.input_data["repo"] == "astropy/astropy"
        assert test.task.input_data["instance_id"] == "astropy__astropy-12907"


# ── MMLU loader tests ─────────────────────────────────────


class TestMMLULoader:
    """Tests for the MMLU benchmark loader."""

    def test_properties(self) -> None:
        loader = MMLULoader()
        assert loader.name == "mmlu"
        assert "MMLU" in loader.description
        assert "huggingface" in loader.source_url

    def test_parse_response(self) -> None:
        loader = MMLULoader()
        response = _make_mock_response(MOCK_MMLU_RESPONSE)
        items = loader._parse_response(response)
        assert len(items) == 3
        assert items[0]["question"] == "What is the capital of France?"

    def test_convert_items(self) -> None:
        loader = MMLULoader()
        items = [r["row"] for r in MOCK_MMLU_RESPONSE["rows"]]
        suite = loader._convert_items(items)

        assert isinstance(suite, TestSuite)
        assert suite.test_suite == "mmlu"
        assert len(suite.tests) == 3

        test = suite.tests[0]
        assert test.id == "mmlu_geography_0"
        assert "capital of France" in test.task.description
        assert test.constraints.timeout_seconds == 60
        assert "mmlu" in test.tags
        assert "subject:geography" in test.tags
        assert len(test.assertions) == 1
        assert test.assertions[0].config["expected"] == "B"

    def test_convert_items_with_limit(self) -> None:
        loader = MMLULoader()
        items = [r["row"] for r in MOCK_MMLU_RESPONSE["rows"]]
        suite = loader._convert_items(items, limit=1)
        assert len(suite.tests) == 1

    def test_convert_empty_items(self) -> None:
        loader = MMLULoader()
        suite = loader._convert_items([])
        assert len(suite.tests) == 1
        assert suite.tests[0].id == "mmlu_placeholder"

    def test_choices_formatted(self) -> None:
        loader = MMLULoader()
        items = [r["row"] for r in MOCK_MMLU_RESPONSE["rows"]]
        suite = loader._convert_items(items)
        desc = suite.tests[0].task.description
        assert "A. London" in desc
        assert "B. Paris" in desc
        assert "C. Berlin" in desc
        assert "D. Madrid" in desc

    def test_correct_answer_mapping(self) -> None:
        loader = MMLULoader()
        # answer=1 should map to "B"
        items = [{"question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 0}]
        suite = loader._convert_items(items)
        assert suite.tests[0].assertions[0].config["expected"] == "A"

        items = [{"question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 3}]
        suite = loader._convert_items(items)
        assert suite.tests[0].assertions[0].config["expected"] == "D"

    def test_input_data_preserved(self) -> None:
        loader = MMLULoader()
        items = [r["row"] for r in MOCK_MMLU_RESPONSE["rows"]]
        suite = loader._convert_items(items)
        test = suite.tests[0]
        assert test.task.input_data is not None
        assert test.task.input_data["answer"] == 1
        assert test.task.input_data["subject"] == "geography"


# ── CLI load command tests ─────────────────────────────────


class TestLoadBenchmarkCLI:
    """Tests for the 'atp benchmark load' CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        from click.testing import CliRunner

        return CliRunner()

    @patch("atp.benchmarks.loaders.HumanEvalLoader.download")
    def test_load_humaneval(
        self, mock_download: MagicMock, runner: CliRunner, tmp_path: Path
    ) -> None:
        from atp.cli.main import cli

        items = [r["row"] for r in MOCK_HUMANEVAL_RESPONSE["rows"]]
        mock_download.return_value = items
        output = tmp_path / "out.yaml"

        result = runner.invoke(
            cli, ["benchmark", "load", "humaneval", "-o", str(output)]
        )

        assert result.exit_code == 0
        assert output.exists()
        assert "Suite written to" in result.output

    @patch("atp.benchmarks.loaders.MMLULoader.download")
    def test_load_mmlu_with_limit(
        self, mock_download: MagicMock, runner: CliRunner, tmp_path: Path
    ) -> None:
        from atp.cli.main import cli

        items = [r["row"] for r in MOCK_MMLU_RESPONSE["rows"]]
        mock_download.return_value = items
        output = tmp_path / "mmlu.yaml"

        result = runner.invoke(
            cli,
            ["benchmark", "load", "mmlu", "--limit=1", "-o", str(output)],
        )

        assert result.exit_code == 0
        assert output.exists()

    def test_load_invalid_benchmark(self, runner: CliRunner) -> None:
        from atp.cli.main import cli

        result = runner.invoke(cli, ["benchmark", "load", "invalid"])
        assert result.exit_code != 0

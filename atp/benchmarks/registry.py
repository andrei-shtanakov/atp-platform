"""Benchmark registry for managing and loading benchmark suites."""

import json
from pathlib import Path
from typing import Any

import yaml

from .models import (
    BaselineScore,
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkSuiteInfo,
    BenchmarkSuiteResult,
    BenchmarkTest,
    NormalizationConfig,
)


class BenchmarkNotFoundError(Exception):
    """Raised when a benchmark suite is not found."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Benchmark suite not found: {name}")


class BenchmarkCategoryNotFoundError(Exception):
    """Raised when a benchmark category is not found."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(f"Benchmark category not found: {category}")


class BenchmarkRegistry:
    """
    Registry for benchmark suites.

    Manages loading, registration, and retrieval of benchmark suites
    for agent evaluation.
    """

    def __init__(self, suites_dir: Path | None = None) -> None:
        """Initialize the registry.

        Args:
            suites_dir: Directory containing benchmark suite files.
                       Defaults to the built-in suites directory.
        """
        self._suites: dict[str, BenchmarkSuite] = {}
        self._normalization_config = NormalizationConfig()

        if suites_dir is None:
            suites_dir = Path(__file__).parent / "suites"

        self._suites_dir = suites_dir
        self._load_builtin_suites()

    def _load_builtin_suites(self) -> None:
        """Load built-in benchmark suites from the suites directory."""
        if not self._suites_dir.exists():
            return

        for file_path in self._suites_dir.glob("*.yaml"):
            try:
                suite = self._load_suite_file(file_path)
                self._suites[suite.name] = suite
            except Exception:
                pass

        for file_path in self._suites_dir.glob("*.json"):
            try:
                suite = self._load_suite_file(file_path)
                self._suites[suite.name] = suite
            except Exception:
                pass

    def _load_suite_file(self, file_path: Path) -> BenchmarkSuite:
        """Load a benchmark suite from a file.

        Args:
            file_path: Path to the suite file (YAML or JSON).

        Returns:
            Loaded BenchmarkSuite.

        Raises:
            ValueError: If the file format is not supported.
        """
        content = file_path.read_text()

        if file_path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(content)
        elif file_path.suffix == ".json":
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        return BenchmarkSuite.model_validate(data)

    def register(self, suite: BenchmarkSuite) -> None:
        """Register a benchmark suite.

        Args:
            suite: BenchmarkSuite to register.
        """
        self._suites[suite.name] = suite

    def register_from_file(self, file_path: Path | str) -> BenchmarkSuite:
        """Register a benchmark suite from a file.

        Args:
            file_path: Path to the suite file.

        Returns:
            The registered BenchmarkSuite.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        suite = self._load_suite_file(file_path)
        self.register(suite)
        return suite

    def unregister(self, name: str) -> bool:
        """Unregister a benchmark suite.

        Args:
            name: Name of the suite to unregister.

        Returns:
            True if suite was removed, False if it didn't exist.
        """
        if name in self._suites:
            del self._suites[name]
            return True
        return False

    def get(self, name: str) -> BenchmarkSuite:
        """Get a benchmark suite by name.

        Args:
            name: Name of the suite to retrieve.

        Returns:
            The benchmark suite.

        Raises:
            BenchmarkNotFoundError: If suite is not found.
        """
        if name not in self._suites:
            raise BenchmarkNotFoundError(name)
        return self._suites[name]

    def get_by_category(
        self, category: BenchmarkCategory | str
    ) -> list[BenchmarkSuite]:
        """Get all benchmark suites in a category.

        Args:
            category: Category to filter by.

        Returns:
            List of benchmark suites in the category.

        Raises:
            BenchmarkCategoryNotFoundError: If category is invalid.
        """
        if isinstance(category, str):
            try:
                category = BenchmarkCategory(category)
            except ValueError:
                raise BenchmarkCategoryNotFoundError(category)

        return [s for s in self._suites.values() if s.category == category]

    def list_suites(self) -> list[str]:
        """List all registered benchmark suite names.

        Returns:
            List of suite names.
        """
        return list(self._suites.keys())

    def list_categories(self) -> list[str]:
        """List all available benchmark categories.

        Returns:
            List of category names.
        """
        return [c.value for c in BenchmarkCategory]

    def get_suite_info(self, name: str) -> BenchmarkSuiteInfo:
        """Get summary information about a suite.

        Args:
            name: Name of the suite.

        Returns:
            Suite information.

        Raises:
            BenchmarkNotFoundError: If suite is not found.
        """
        suite = self.get(name)
        return suite.get_info()

    def list_all_info(self) -> list[BenchmarkSuiteInfo]:
        """Get information about all registered suites.

        Returns:
            List of suite information objects.
        """
        return [suite.get_info() for suite in self._suites.values()]

    def is_registered(self, name: str) -> bool:
        """Check if a benchmark suite is registered.

        Args:
            name: Suite name to check.

        Returns:
            True if suite is registered, False otherwise.
        """
        return name in self._suites

    def get_test(self, suite_name: str, test_id: str) -> BenchmarkTest:
        """Get a specific test from a suite.

        Args:
            suite_name: Name of the suite.
            test_id: ID of the test.

        Returns:
            The benchmark test.

        Raises:
            BenchmarkNotFoundError: If suite or test is not found.
        """
        suite = self.get(suite_name)
        for test in suite.tests:
            if test.id == test_id:
                return test
        raise BenchmarkNotFoundError(f"{suite_name}/{test_id}")

    def set_normalization_config(self, config: NormalizationConfig) -> None:
        """Set the normalization configuration.

        Args:
            config: Normalization configuration to use.
        """
        self._normalization_config = config

    def normalize_score(self, raw_score: float) -> float:
        """Normalize a raw score to the 0-100 scale.

        Args:
            raw_score: Raw score to normalize (typically 0-1).

        Returns:
            Normalized score (0-100).
        """
        return self._normalization_config.normalize(raw_score)

    def create_result(
        self,
        test_id: str,
        raw_score: float,
        passed: bool,
        execution_time_seconds: float,
        steps_used: int | None = None,
        tokens_used: int | None = None,
        cost_usd: float | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Create a benchmark result with normalized score.

        Args:
            test_id: ID of the test that was run.
            raw_score: Raw score (0-1).
            passed: Whether the test passed.
            execution_time_seconds: Time taken to execute.
            steps_used: Number of steps used.
            tokens_used: Number of tokens used.
            cost_usd: Cost in USD.
            error: Error message if failed.
            metadata: Additional metadata.

        Returns:
            BenchmarkResult with normalized score.
        """
        return BenchmarkResult(
            test_id=test_id,
            raw_score=raw_score,
            normalized_score=self.normalize_score(raw_score),
            passed=passed,
            execution_time_seconds=execution_time_seconds,
            steps_used=steps_used,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            error=error,
            metadata=metadata or {},
        )

    def create_suite_result(
        self,
        suite_name: str,
        agent_name: str,
        results: list[BenchmarkResult],
    ) -> BenchmarkSuiteResult:
        """Create a suite result from individual test results.

        Args:
            suite_name: Name of the suite.
            agent_name: Name of the agent tested.
            results: Individual test results.

        Returns:
            Aggregated suite result.
        """
        suite = self.get(suite_name)

        passed_tests = sum(1 for r in results if r.passed)
        total_time = sum(r.execution_time_seconds for r in results)
        avg_score = (
            sum(r.normalized_score for r in results) / len(results) if results else 0.0
        )

        baseline_comparison = self._compute_baseline_comparison(suite, results)

        return BenchmarkSuiteResult(
            suite_name=suite_name,
            category=suite.category,
            agent_name=agent_name,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=len(results) - passed_tests,
            average_normalized_score=avg_score,
            total_execution_time_seconds=total_time,
            results=results,
            baseline_comparison=baseline_comparison,
        )

    def _compute_baseline_comparison(
        self,
        suite: BenchmarkSuite,
        results: list[BenchmarkResult],
    ) -> dict[str, float] | None:
        """Compute comparison to baseline scores.

        Args:
            suite: The benchmark suite.
            results: Test results to compare.

        Returns:
            Dictionary mapping model names to score deltas, or None.
        """
        result_map = {r.test_id: r.normalized_score for r in results}

        baseline_totals: dict[str, list[float]] = {}

        for test in suite.tests:
            if test.id not in result_map:
                continue

            result_score = result_map[test.id]

            for baseline in test.metadata.baseline_scores:
                if baseline.model_name not in baseline_totals:
                    baseline_totals[baseline.model_name] = []
                delta = result_score - baseline.score
                baseline_totals[baseline.model_name].append(delta)

        if not baseline_totals:
            return None

        return {
            model: sum(deltas) / len(deltas)
            for model, deltas in baseline_totals.items()
        }

    def get_baseline_scores(self, suite_name: str) -> dict[str, list[BaselineScore]]:
        """Get baseline scores for a suite, organized by test ID.

        Args:
            suite_name: Name of the suite.

        Returns:
            Dictionary mapping test IDs to their baseline scores.

        Raises:
            BenchmarkNotFoundError: If suite is not found.
        """
        suite = self.get(suite_name)
        return {test.id: test.metadata.baseline_scores for test in suite.tests}


_default_registry: BenchmarkRegistry | None = None


def get_registry() -> BenchmarkRegistry:
    """Get the global benchmark registry.

    Returns:
        Global BenchmarkRegistry instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = BenchmarkRegistry()
    return _default_registry


def get_benchmark(name: str) -> BenchmarkSuite:
    """Get a benchmark suite using the global registry.

    Args:
        name: Name of the benchmark suite.

    Returns:
        The benchmark suite.

    Raises:
        BenchmarkNotFoundError: If suite is not found.
    """
    return get_registry().get(name)


def list_benchmarks() -> list[str]:
    """List all benchmark suites using the global registry.

    Returns:
        List of benchmark suite names.
    """
    return get_registry().list_suites()


def list_categories() -> list[str]:
    """List all benchmark categories.

    Returns:
        List of category names.
    """
    return get_registry().list_categories()

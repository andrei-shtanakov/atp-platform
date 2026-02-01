"""Unit tests for benchmark registry."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from atp.benchmarks import (
    BenchmarkCategory,
    BenchmarkCategoryNotFoundError,
    BenchmarkDifficulty,
    BenchmarkMetadata,
    BenchmarkNotFoundError,
    BenchmarkRegistry,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkTest,
    NormalizationConfig,
    get_registry,
    list_categories,
)


@pytest.fixture
def empty_registry() -> BenchmarkRegistry:
    """Create an empty registry for testing."""
    with TemporaryDirectory() as tmpdir:
        return BenchmarkRegistry(suites_dir=Path(tmpdir))


@pytest.fixture
def sample_test() -> BenchmarkTest:
    """Create a sample test."""
    return BenchmarkTest(
        id="test-001",
        name="Sample Test",
        description="A sample test",
        task_description="Complete this task",
        metadata=BenchmarkMetadata(
            category=BenchmarkCategory.CODING,
            difficulty=BenchmarkDifficulty.MEDIUM,
        ),
    )


@pytest.fixture
def sample_suite(sample_test: BenchmarkTest) -> BenchmarkSuite:
    """Create a sample suite."""
    return BenchmarkSuite(
        name="test_suite",
        category=BenchmarkCategory.CODING,
        description="Test suite for unit tests",
        tests=[sample_test],
    )


class TestBenchmarkRegistry:
    """Tests for BenchmarkRegistry class."""

    def test_empty_registry(self, empty_registry: BenchmarkRegistry) -> None:
        """Test that empty registry has no suites."""
        assert empty_registry.list_suites() == []

    def test_register_suite(
        self, empty_registry: BenchmarkRegistry, sample_suite: BenchmarkSuite
    ) -> None:
        """Test registering a suite."""
        empty_registry.register(sample_suite)
        assert empty_registry.is_registered("test_suite")
        assert empty_registry.list_suites() == ["test_suite"]

    def test_get_suite(
        self, empty_registry: BenchmarkRegistry, sample_suite: BenchmarkSuite
    ) -> None:
        """Test getting a registered suite."""
        empty_registry.register(sample_suite)
        suite = empty_registry.get("test_suite")
        assert suite.name == "test_suite"
        assert suite.category == BenchmarkCategory.CODING

    def test_get_nonexistent_suite(self, empty_registry: BenchmarkRegistry) -> None:
        """Test getting a non-existent suite raises error."""
        with pytest.raises(BenchmarkNotFoundError) as exc_info:
            empty_registry.get("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_unregister_suite(
        self, empty_registry: BenchmarkRegistry, sample_suite: BenchmarkSuite
    ) -> None:
        """Test unregistering a suite."""
        empty_registry.register(sample_suite)
        assert empty_registry.unregister("test_suite") is True
        assert not empty_registry.is_registered("test_suite")

    def test_unregister_nonexistent_suite(
        self, empty_registry: BenchmarkRegistry
    ) -> None:
        """Test unregistering a non-existent suite."""
        assert empty_registry.unregister("nonexistent") is False

    def test_get_by_category(
        self, empty_registry: BenchmarkRegistry, sample_test: BenchmarkTest
    ) -> None:
        """Test getting suites by category."""
        coding_suite = BenchmarkSuite(
            name="coding_tests",
            category=BenchmarkCategory.CODING,
            description="Coding tests",
            tests=[sample_test],
        )
        research_test = BenchmarkTest(
            id="research-001",
            name="Research Test",
            description="A research test",
            task_description="Research task",
            metadata=BenchmarkMetadata(category=BenchmarkCategory.RESEARCH),
        )
        research_suite = BenchmarkSuite(
            name="research_tests",
            category=BenchmarkCategory.RESEARCH,
            description="Research tests",
            tests=[research_test],
        )

        empty_registry.register(coding_suite)
        empty_registry.register(research_suite)

        coding_suites = empty_registry.get_by_category(BenchmarkCategory.CODING)
        assert len(coding_suites) == 1
        assert coding_suites[0].name == "coding_tests"

    def test_get_by_category_string(
        self, empty_registry: BenchmarkRegistry, sample_suite: BenchmarkSuite
    ) -> None:
        """Test getting suites by category using string."""
        empty_registry.register(sample_suite)
        suites = empty_registry.get_by_category("coding")
        assert len(suites) == 1

    def test_get_by_invalid_category(self, empty_registry: BenchmarkRegistry) -> None:
        """Test getting suites by invalid category raises error."""
        with pytest.raises(BenchmarkCategoryNotFoundError):
            empty_registry.get_by_category("invalid_category")

    def test_list_categories(self, empty_registry: BenchmarkRegistry) -> None:
        """Test listing all categories."""
        categories = empty_registry.list_categories()
        assert "coding" in categories
        assert "research" in categories
        assert "reasoning" in categories
        assert "data_processing" in categories

    def test_get_suite_info(
        self, empty_registry: BenchmarkRegistry, sample_suite: BenchmarkSuite
    ) -> None:
        """Test getting suite info."""
        empty_registry.register(sample_suite)
        info = empty_registry.get_suite_info("test_suite")
        assert info.name == "test_suite"
        assert info.test_count == 1
        assert info.category == BenchmarkCategory.CODING

    def test_list_all_info(
        self, empty_registry: BenchmarkRegistry, sample_suite: BenchmarkSuite
    ) -> None:
        """Test listing all suite info."""
        empty_registry.register(sample_suite)
        infos = empty_registry.list_all_info()
        assert len(infos) == 1
        assert infos[0].name == "test_suite"

    def test_get_test(
        self, empty_registry: BenchmarkRegistry, sample_suite: BenchmarkSuite
    ) -> None:
        """Test getting a specific test from a suite."""
        empty_registry.register(sample_suite)
        test = empty_registry.get_test("test_suite", "test-001")
        assert test.id == "test-001"
        assert test.name == "Sample Test"

    def test_get_nonexistent_test(
        self, empty_registry: BenchmarkRegistry, sample_suite: BenchmarkSuite
    ) -> None:
        """Test getting a non-existent test raises error."""
        empty_registry.register(sample_suite)
        with pytest.raises(BenchmarkNotFoundError):
            empty_registry.get_test("test_suite", "nonexistent")


class TestNormalization:
    """Tests for score normalization in registry."""

    def test_default_normalization(self, empty_registry: BenchmarkRegistry) -> None:
        """Test default normalization."""
        assert empty_registry.normalize_score(0.0) == 0.0
        assert empty_registry.normalize_score(0.5) == 50.0
        assert empty_registry.normalize_score(1.0) == 100.0

    def test_custom_normalization(self, empty_registry: BenchmarkRegistry) -> None:
        """Test custom normalization config."""
        config = NormalizationConfig(
            min_raw_score=0.0,
            max_raw_score=10.0,
            target_min=0.0,
            target_max=100.0,
        )
        empty_registry.set_normalization_config(config)
        assert empty_registry.normalize_score(5.0) == 50.0

    def test_create_result(self, empty_registry: BenchmarkRegistry) -> None:
        """Test creating a result with normalized score."""
        result = empty_registry.create_result(
            test_id="test-001",
            raw_score=0.75,
            passed=True,
            execution_time_seconds=30.0,
            steps_used=10,
        )
        assert result.test_id == "test-001"
        assert result.raw_score == 0.75
        assert result.normalized_score == 75.0
        assert result.passed is True


class TestSuiteResults:
    """Tests for suite result creation."""

    def test_create_suite_result(
        self, empty_registry: BenchmarkRegistry, sample_suite: BenchmarkSuite
    ) -> None:
        """Test creating a suite result."""
        empty_registry.register(sample_suite)

        results = [
            BenchmarkResult(
                test_id="test-001",
                raw_score=0.8,
                normalized_score=80.0,
                passed=True,
                execution_time_seconds=30.0,
            )
        ]

        suite_result = empty_registry.create_suite_result(
            suite_name="test_suite",
            agent_name="test_agent",
            results=results,
        )

        assert suite_result.suite_name == "test_suite"
        assert suite_result.agent_name == "test_agent"
        assert suite_result.total_tests == 1
        assert suite_result.passed_tests == 1
        assert suite_result.average_normalized_score == 80.0

    def test_create_suite_result_with_failures(
        self, empty_registry: BenchmarkRegistry, sample_suite: BenchmarkSuite
    ) -> None:
        """Test creating a suite result with failed tests."""
        empty_registry.register(sample_suite)

        results = [
            BenchmarkResult(
                test_id="test-001",
                raw_score=0.0,
                normalized_score=0.0,
                passed=False,
                execution_time_seconds=10.0,
                error="Test failed",
            )
        ]

        suite_result = empty_registry.create_suite_result(
            suite_name="test_suite",
            agent_name="test_agent",
            results=results,
        )

        assert suite_result.failed_tests == 1
        assert suite_result.passed_tests == 0


class TestFileLoading:
    """Tests for loading suites from files."""

    def test_register_from_yaml_file(self, empty_registry: BenchmarkRegistry) -> None:
        """Test registering a suite from a YAML file."""
        with TemporaryDirectory() as tmpdir:
            yaml_content = """
name: yaml_suite
category: coding
description: Test suite from YAML
version: "1.0.0"
tests:
  - id: yaml-test-001
    name: YAML Test
    description: A test from YAML
    task_description: Complete this YAML task
    metadata:
      category: coding
      difficulty: easy
"""
            yaml_path = Path(tmpdir) / "test_suite.yaml"
            yaml_path.write_text(yaml_content)

            suite = empty_registry.register_from_file(yaml_path)
            assert suite.name == "yaml_suite"
            assert empty_registry.is_registered("yaml_suite")

    def test_register_from_json_file(self, empty_registry: BenchmarkRegistry) -> None:
        """Test registering a suite from a JSON file."""
        with TemporaryDirectory() as tmpdir:
            import json

            json_content = {
                "name": "json_suite",
                "category": "research",
                "description": "Test suite from JSON",
                "version": "1.0.0",
                "tests": [
                    {
                        "id": "json-test-001",
                        "name": "JSON Test",
                        "description": "A test from JSON",
                        "task_description": "Complete this JSON task",
                        "metadata": {
                            "category": "research",
                            "difficulty": "medium",
                        },
                    }
                ],
            }
            json_path = Path(tmpdir) / "test_suite.json"
            json_path.write_text(json.dumps(json_content))

            suite = empty_registry.register_from_file(json_path)
            assert suite.name == "json_suite"
            assert empty_registry.is_registered("json_suite")


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry_singleton(self) -> None:
        """Test that get_registry returns same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_list_categories_function(self) -> None:
        """Test list_categories function."""
        categories = list_categories()
        assert "coding" in categories
        assert "research" in categories


class TestBuiltinSuites:
    """Tests for built-in benchmark suites."""

    def test_builtin_suites_loaded(self) -> None:
        """Test that built-in suites are loaded."""
        registry = get_registry()
        suites = registry.list_suites()

        # Check that our built-in suites are loaded
        assert "coding" in suites
        assert "research" in suites
        assert "reasoning" in suites
        assert "data_processing" in suites

    def test_coding_suite_test_count(self) -> None:
        """Test coding suite has expected test count."""
        registry = get_registry()
        suite = registry.get("coding")
        assert len(suite.tests) == 20

    def test_research_suite_test_count(self) -> None:
        """Test research suite has expected test count."""
        registry = get_registry()
        suite = registry.get("research")
        assert len(suite.tests) == 10

    def test_reasoning_suite_test_count(self) -> None:
        """Test reasoning suite has expected test count."""
        registry = get_registry()
        suite = registry.get("reasoning")
        assert len(suite.tests) == 15

    def test_data_processing_suite_test_count(self) -> None:
        """Test data_processing suite has expected test count."""
        registry = get_registry()
        suite = registry.get("data_processing")
        assert len(suite.tests) == 10

    def test_all_suites_have_baseline_scores(self) -> None:
        """Test that all suites have baseline scores defined."""
        registry = get_registry()
        for suite_name in registry.list_suites():
            suite = registry.get(suite_name)
            has_baseline = False
            for test in suite.tests:
                if test.metadata.baseline_scores:
                    has_baseline = True
                    break
            assert has_baseline, f"Suite {suite_name} has no baseline scores"

    def test_get_baseline_scores(self) -> None:
        """Test getting baseline scores for a suite."""
        registry = get_registry()
        baselines = registry.get_baseline_scores("coding")
        assert len(baselines) > 0
        # Check that at least one test has baselines
        has_baselines = any(scores for scores in baselines.values())
        assert has_baselines

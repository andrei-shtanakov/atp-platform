"""Unit tests for benchmark models."""

import pytest
from pydantic import ValidationError

from atp.benchmarks.models import (
    BaselineScore,
    BenchmarkCategory,
    BenchmarkDifficulty,
    BenchmarkMetadata,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkSuiteInfo,
    BenchmarkSuiteResult,
    BenchmarkTest,
    NormalizationConfig,
)


class TestBenchmarkCategory:
    """Tests for BenchmarkCategory enum."""

    def test_category_values(self) -> None:
        """Test that all expected categories exist."""
        assert BenchmarkCategory.CODING.value == "coding"
        assert BenchmarkCategory.RESEARCH.value == "research"
        assert BenchmarkCategory.REASONING.value == "reasoning"
        assert BenchmarkCategory.DATA_PROCESSING.value == "data_processing"

    def test_category_from_string(self) -> None:
        """Test category creation from string."""
        assert BenchmarkCategory("coding") == BenchmarkCategory.CODING
        assert BenchmarkCategory("research") == BenchmarkCategory.RESEARCH


class TestBenchmarkDifficulty:
    """Tests for BenchmarkDifficulty enum."""

    def test_difficulty_values(self) -> None:
        """Test that all expected difficulties exist."""
        assert BenchmarkDifficulty.EASY.value == "easy"
        assert BenchmarkDifficulty.MEDIUM.value == "medium"
        assert BenchmarkDifficulty.HARD.value == "hard"


class TestBaselineScore:
    """Tests for BaselineScore model."""

    def test_valid_baseline_score(self) -> None:
        """Test creating a valid baseline score."""
        baseline = BaselineScore(
            model_name="gpt-4",
            score=85.5,
            date="2024-01-15",
            notes="Initial baseline",
        )
        assert baseline.model_name == "gpt-4"
        assert baseline.score == 85.5
        assert baseline.date == "2024-01-15"
        assert baseline.notes == "Initial baseline"

    def test_baseline_score_without_notes(self) -> None:
        """Test creating baseline score without optional notes."""
        baseline = BaselineScore(
            model_name="claude-3-opus",
            score=90.0,
            date="2024-01-15",
        )
        assert baseline.notes is None

    def test_baseline_score_validation(self) -> None:
        """Test score validation bounds."""
        with pytest.raises(ValidationError):
            BaselineScore(
                model_name="test",
                score=101.0,  # Above max
                date="2024-01-15",
            )

        with pytest.raises(ValidationError):
            BaselineScore(
                model_name="test",
                score=-1.0,  # Below min
                date="2024-01-15",
            )


class TestBenchmarkMetadata:
    """Tests for BenchmarkMetadata model."""

    def test_valid_metadata(self) -> None:
        """Test creating valid metadata."""
        metadata = BenchmarkMetadata(
            category=BenchmarkCategory.CODING,
            difficulty=BenchmarkDifficulty.MEDIUM,
            estimated_time_seconds=120,
            skills_tested=["python", "algorithms"],
            baseline_scores=[
                BaselineScore(
                    model_name="gpt-4",
                    score=85.0,
                    date="2024-01-15",
                )
            ],
        )
        assert metadata.category == BenchmarkCategory.CODING
        assert metadata.difficulty == BenchmarkDifficulty.MEDIUM
        assert metadata.estimated_time_seconds == 120
        assert len(metadata.skills_tested) == 2
        assert len(metadata.baseline_scores) == 1

    def test_metadata_defaults(self) -> None:
        """Test metadata default values."""
        metadata = BenchmarkMetadata(category=BenchmarkCategory.RESEARCH)
        assert metadata.difficulty == BenchmarkDifficulty.MEDIUM
        assert metadata.estimated_time_seconds == 60
        assert metadata.skills_tested == []
        assert metadata.baseline_scores == []


class TestBenchmarkTest:
    """Tests for BenchmarkTest model."""

    def test_valid_test(self) -> None:
        """Test creating a valid benchmark test."""
        test = BenchmarkTest(
            id="test-001",
            name="Sample Test",
            description="A sample test",
            task_description="Complete this task",
            metadata=BenchmarkMetadata(category=BenchmarkCategory.CODING),
        )
        assert test.id == "test-001"
        assert test.name == "Sample Test"
        assert test.task_description == "Complete this task"
        assert test.timeout_seconds == 300

    def test_test_with_all_fields(self) -> None:
        """Test creating a test with all optional fields."""
        test = BenchmarkTest(
            id="test-002",
            name="Full Test",
            description="Full test description",
            task_description="Task description",
            input_data={"key": "value"},
            expected_artifacts=["*.py", "*.md"],
            assertions=[{"type": "artifact_exists", "config": {"pattern": "*.py"}}],
            metadata=BenchmarkMetadata(category=BenchmarkCategory.REASONING),
            max_steps=50,
            timeout_seconds=600,
            tags=["complex", "algorithm"],
        )
        assert test.input_data == {"key": "value"}
        assert len(test.expected_artifacts) == 2
        assert len(test.assertions) == 1
        assert test.max_steps == 50
        assert len(test.tags) == 2


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite model."""

    @pytest.fixture
    def sample_test(self) -> BenchmarkTest:
        """Create a sample test for use in suite tests."""
        return BenchmarkTest(
            id="test-001",
            name="Sample Test",
            description="A sample test",
            task_description="Complete this task",
            metadata=BenchmarkMetadata(
                category=BenchmarkCategory.CODING,
                difficulty=BenchmarkDifficulty.EASY,
                baseline_scores=[
                    BaselineScore(
                        model_name="gpt-4",
                        score=80.0,
                        date="2024-01-15",
                    )
                ],
            ),
        )

    def test_valid_suite(self, sample_test: BenchmarkTest) -> None:
        """Test creating a valid benchmark suite."""
        suite = BenchmarkSuite(
            name="coding",
            category=BenchmarkCategory.CODING,
            description="Coding benchmarks",
            tests=[sample_test],
        )
        assert suite.name == "coding"
        assert suite.version == "1.0.0"
        assert len(suite.tests) == 1

    def test_suite_requires_tests(self) -> None:
        """Test that suite requires at least one test."""
        with pytest.raises(ValidationError):
            BenchmarkSuite(
                name="empty",
                category=BenchmarkCategory.CODING,
                description="Empty suite",
                tests=[],
            )

    def test_suite_get_info(self, sample_test: BenchmarkTest) -> None:
        """Test get_info method."""
        suite = BenchmarkSuite(
            name="test_suite",
            category=BenchmarkCategory.CODING,
            description="Test suite",
            tests=[sample_test],
        )
        info = suite.get_info()

        assert isinstance(info, BenchmarkSuiteInfo)
        assert info.name == "test_suite"
        assert info.category == BenchmarkCategory.CODING
        assert info.test_count == 1
        assert info.difficulty_distribution == {"easy": 1}
        assert info.average_baseline_score == 80.0


class TestBenchmarkResult:
    """Tests for BenchmarkResult model."""

    def test_valid_result(self) -> None:
        """Test creating a valid result."""
        result = BenchmarkResult(
            test_id="test-001",
            raw_score=0.85,
            normalized_score=85.0,
            passed=True,
            execution_time_seconds=45.5,
        )
        assert result.test_id == "test-001"
        assert result.raw_score == 0.85
        assert result.normalized_score == 85.0
        assert result.passed is True

    def test_result_with_error(self) -> None:
        """Test creating a result with error."""
        result = BenchmarkResult(
            test_id="test-002",
            raw_score=0.0,
            normalized_score=0.0,
            passed=False,
            execution_time_seconds=10.0,
            error="Timeout exceeded",
        )
        assert result.passed is False
        assert result.error == "Timeout exceeded"


class TestBenchmarkSuiteResult:
    """Tests for BenchmarkSuiteResult model."""

    def test_valid_suite_result(self) -> None:
        """Test creating a valid suite result."""
        results = [
            BenchmarkResult(
                test_id="test-001",
                raw_score=0.8,
                normalized_score=80.0,
                passed=True,
                execution_time_seconds=30.0,
            ),
            BenchmarkResult(
                test_id="test-002",
                raw_score=0.9,
                normalized_score=90.0,
                passed=True,
                execution_time_seconds=45.0,
            ),
        ]

        suite_result = BenchmarkSuiteResult(
            suite_name="coding",
            category=BenchmarkCategory.CODING,
            agent_name="test-agent",
            total_tests=2,
            passed_tests=2,
            failed_tests=0,
            average_normalized_score=85.0,
            total_execution_time_seconds=75.0,
            results=results,
        )

        assert suite_result.suite_name == "coding"
        assert suite_result.total_tests == 2
        assert suite_result.passed_tests == 2
        assert suite_result.pass_rate == 100.0

    def test_pass_rate_calculation(self) -> None:
        """Test pass rate calculation."""
        suite_result = BenchmarkSuiteResult(
            suite_name="test",
            category=BenchmarkCategory.REASONING,
            agent_name="agent",
            total_tests=4,
            passed_tests=3,
            failed_tests=1,
            average_normalized_score=75.0,
            total_execution_time_seconds=100.0,
        )
        assert suite_result.pass_rate == 75.0

    def test_pass_rate_zero_tests(self) -> None:
        """Test pass rate with zero tests."""
        suite_result = BenchmarkSuiteResult(
            suite_name="test",
            category=BenchmarkCategory.RESEARCH,
            agent_name="agent",
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            average_normalized_score=0.0,
            total_execution_time_seconds=0.0,
        )
        assert suite_result.pass_rate == 0.0


class TestNormalizationConfig:
    """Tests for NormalizationConfig model."""

    def test_default_normalization(self) -> None:
        """Test default linear normalization."""
        config = NormalizationConfig()
        assert config.normalize(0.0) == 0.0
        assert config.normalize(0.5) == 50.0
        assert config.normalize(1.0) == 100.0

    def test_linear_normalization(self) -> None:
        """Test linear normalization with custom range."""
        config = NormalizationConfig(
            min_raw_score=0.0,
            max_raw_score=10.0,
            target_min=0.0,
            target_max=100.0,
            curve_type="linear",
        )
        assert config.normalize(0.0) == 0.0
        assert config.normalize(5.0) == 50.0
        assert config.normalize(10.0) == 100.0

    def test_normalization_clamps_values(self) -> None:
        """Test that normalization clamps out-of-range values."""
        config = NormalizationConfig()
        assert config.normalize(-0.5) == 0.0
        assert config.normalize(1.5) == 100.0

    def test_logarithmic_normalization(self) -> None:
        """Test logarithmic normalization."""
        config = NormalizationConfig(curve_type="logarithmic")
        # Logarithmic should give higher scores for lower raw scores
        result = config.normalize(0.5)
        assert result > 50.0  # Logarithmic curve boosts lower values

    def test_sigmoid_normalization(self) -> None:
        """Test sigmoid normalization."""
        config = NormalizationConfig(curve_type="sigmoid")
        result = config.normalize(0.5)
        # Sigmoid at 0.5 should be close to 50
        assert 45.0 < result < 55.0

    def test_zero_range_normalization(self) -> None:
        """Test normalization with zero range."""
        config = NormalizationConfig(
            min_raw_score=5.0,
            max_raw_score=5.0,  # Same as min
        )
        assert config.normalize(5.0) == 0.0

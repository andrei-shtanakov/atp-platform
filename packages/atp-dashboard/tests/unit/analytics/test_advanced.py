"""Unit tests for advanced analytics module."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.analytics.advanced import (
    AdvancedAnalyticsService,
    AnomalyType,
    CorrelationStrength,
    ScheduledReportConfig,
    ScheduledReportFrequency,
    ScheduledReportsRepository,
    TrendDirection,
)


class TestAdvancedAnalyticsService:
    """Tests for AdvancedAnalyticsService."""

    @pytest.fixture
    def mock_session(self) -> AsyncSession:
        """Create a mock async session."""
        session = MagicMock(spec=AsyncSession)
        session.execute = AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: AsyncSession) -> AdvancedAnalyticsService:
        """Create an analytics service instance."""
        return AdvancedAnalyticsService(mock_session)

    # ==================== Trend Analysis Tests ====================

    def test_calculate_trend_insufficient_data(self, service: AdvancedAnalyticsService):
        """Test trend calculation with insufficient data."""
        executions = []
        result = service._calculate_trend(executions, "score")

        assert result is not None
        assert result.direction == TrendDirection.INSUFFICIENT_DATA
        assert result.confidence == 0.0

    def test_calculate_trend_improving(self, service: AdvancedAnalyticsService):
        """Test trend calculation with improving scores."""

        class MockExecution:
            def __init__(self, date: datetime, score: float):
                self.started_at = date
                self.score = score
                self.success = True
                self.duration_seconds = 10.0

        now = datetime.now()
        executions = [
            MockExecution(now - timedelta(days=5), 0.5),
            MockExecution(now - timedelta(days=4), 0.55),
            MockExecution(now - timedelta(days=3), 0.6),
            MockExecution(now - timedelta(days=2), 0.65),
            MockExecution(now - timedelta(days=1), 0.7),
            MockExecution(now, 0.8),
        ]

        result = service._calculate_trend(executions, "score")

        assert result is not None
        assert result.direction == TrendDirection.IMPROVING
        assert result.change_percent > 0
        assert result.start_value == 0.5
        assert result.end_value == 0.8

    def test_calculate_trend_declining(self, service: AdvancedAnalyticsService):
        """Test trend calculation with declining scores."""

        class MockExecution:
            def __init__(self, date: datetime, score: float):
                self.started_at = date
                self.score = score
                self.success = score > 0.5
                self.duration_seconds = 10.0

        now = datetime.now()
        executions = [
            MockExecution(now - timedelta(days=5), 0.8),
            MockExecution(now - timedelta(days=4), 0.75),
            MockExecution(now - timedelta(days=3), 0.7),
            MockExecution(now - timedelta(days=2), 0.6),
            MockExecution(now - timedelta(days=1), 0.55),
            MockExecution(now, 0.5),
        ]

        result = service._calculate_trend(executions, "score")

        assert result is not None
        assert result.direction == TrendDirection.DECLINING
        assert result.change_percent < 0

    def test_calculate_trend_stable(self, service: AdvancedAnalyticsService):
        """Test trend calculation with stable scores."""

        class MockExecution:
            def __init__(self, date: datetime, score: float):
                self.started_at = date
                self.score = score
                self.success = True
                self.duration_seconds = 10.0

        now = datetime.now()
        executions = [
            MockExecution(now - timedelta(days=5), 0.75),
            MockExecution(now - timedelta(days=4), 0.76),
            MockExecution(now - timedelta(days=3), 0.74),
            MockExecution(now - timedelta(days=2), 0.75),
            MockExecution(now - timedelta(days=1), 0.76),
            MockExecution(now, 0.75),
        ]

        result = service._calculate_trend(executions, "score")

        assert result is not None
        assert result.direction == TrendDirection.STABLE

    def test_calculate_trend_duration_improving(
        self, service: AdvancedAnalyticsService
    ):
        """Test that decreasing duration is correctly identified as improving."""

        class MockExecution:
            def __init__(self, date: datetime, duration: float):
                self.started_at = date
                self.score = 0.8
                self.success = True
                self.duration_seconds = duration

        now = datetime.now()
        executions = [
            MockExecution(now - timedelta(days=5), 100.0),
            MockExecution(now - timedelta(days=4), 90.0),
            MockExecution(now - timedelta(days=3), 80.0),
            MockExecution(now - timedelta(days=2), 70.0),
            MockExecution(now - timedelta(days=1), 60.0),
            MockExecution(now, 50.0),
        ]

        result = service._calculate_trend(executions, "duration")

        assert result is not None
        assert result.direction == TrendDirection.IMPROVING

    # ==================== Anomaly Detection Tests ====================

    def test_detect_metric_anomalies_no_data(self, service: AdvancedAnalyticsService):
        """Test anomaly detection with no data."""
        anomalies = service._detect_metric_anomalies([], "score", 2.0)
        assert anomalies == []

    def test_detect_metric_anomalies_insufficient_data(
        self, service: AdvancedAnalyticsService
    ):
        """Test anomaly detection with insufficient data."""

        class MockExecution:
            def __init__(self, score: float):
                self.started_at = datetime.now()
                self.score = score
                self.duration_seconds = 10.0
                self.test_id = "test-1"
                self.suite_execution_id = 1

        # Less than MIN_DATA_POINTS_ANOMALY
        executions = [MockExecution(0.8) for _ in range(5)]
        anomalies = service._detect_metric_anomalies(executions, "score", 2.0)
        assert anomalies == []

    def test_detect_metric_anomalies_score_drop(
        self, service: AdvancedAnalyticsService
    ):
        """Test detection of score drop anomaly."""

        class MockExecution:
            def __init__(self, score: float):
                self.started_at = datetime.now()
                self.score = score
                self.duration_seconds = 10.0
                self.test_id = "test-1"
                self.suite_execution_id = 1

        # Normal scores around 0.8 with one anomaly at 0.2
        executions = [MockExecution(0.8) for _ in range(15)]
        executions.append(MockExecution(0.2))  # Anomaly

        anomalies = service._detect_metric_anomalies(executions, "score", 2.0)

        assert len(anomalies) >= 1
        anomaly = anomalies[0]
        assert anomaly.anomaly_type == AnomalyType.SCORE_DROP
        assert anomaly.actual_value == 0.2

    def test_detect_metric_anomalies_score_spike(
        self, service: AdvancedAnalyticsService
    ):
        """Test detection of score spike anomaly."""

        class MockExecution:
            def __init__(self, score: float):
                self.started_at = datetime.now()
                self.score = score
                self.duration_seconds = 10.0
                self.test_id = "test-1"
                self.suite_execution_id = 1

        # Normal scores around 0.5 with one high outlier
        executions = [MockExecution(0.5) for _ in range(15)]
        executions.append(MockExecution(0.99))  # Anomaly

        anomalies = service._detect_metric_anomalies(executions, "score", 2.0)

        assert len(anomalies) >= 1
        assert any(a.anomaly_type == AnomalyType.SCORE_SPIKE for a in anomalies)

    def test_detect_metric_anomalies_severity_classification(
        self, service: AdvancedAnalyticsService
    ):
        """Test anomaly severity classification."""

        class MockExecution:
            def __init__(self, score: float):
                self.started_at = datetime.now()
                self.score = score
                self.duration_seconds = 10.0
                self.test_id = "test-1"
                self.suite_execution_id = 1

        # Create baseline with very low variance
        executions = [MockExecution(0.8) for _ in range(20)]
        # Add extreme anomaly
        executions.append(MockExecution(0.1))

        anomalies = service._detect_metric_anomalies(executions, "score", 1.5)

        assert len(anomalies) >= 1
        # Most extreme anomaly should be high severity
        extreme_anomaly = max(anomalies, key=lambda a: a.deviation_sigma)
        assert extreme_anomaly.severity in ["high", "medium"]

    # ==================== Correlation Analysis Tests ====================

    def test_calculate_correlation_perfect_positive(
        self, service: AdvancedAnalyticsService
    ):
        """Test perfect positive correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]

        result = service._calculate_correlation(x, y)

        assert result is not None
        assert abs(result - 1.0) < 0.01

    def test_calculate_correlation_perfect_negative(
        self, service: AdvancedAnalyticsService
    ):
        """Test perfect negative correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]

        result = service._calculate_correlation(x, y)

        assert result is not None
        assert abs(result - (-1.0)) < 0.01

    def test_calculate_correlation_no_correlation(
        self, service: AdvancedAnalyticsService
    ):
        """Test no correlation (random data)."""
        # Use data that has very low correlation
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        y = [3.0, 5.0, 2.0, 6.0, 1.0, 4.0]  # No clear pattern

        result = service._calculate_correlation(x, y)

        assert result is not None
        assert abs(result) < 0.5  # Weak or no correlation

    def test_calculate_correlation_insufficient_data(
        self, service: AdvancedAnalyticsService
    ):
        """Test correlation with insufficient data."""
        result = service._calculate_correlation([1.0], [2.0])
        assert result is None

    def test_calculate_correlation_zero_variance(
        self, service: AdvancedAnalyticsService
    ):
        """Test correlation when one variable has zero variance."""
        x = [5.0, 5.0, 5.0, 5.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = service._calculate_correlation(x, y)
        assert result is None

    def test_classify_correlation_strong_positive(
        self, service: AdvancedAnalyticsService
    ):
        """Test classification of strong positive correlation."""
        assert (
            service._classify_correlation(0.85) == CorrelationStrength.STRONG_POSITIVE
        )
        assert (
            service._classify_correlation(0.70) == CorrelationStrength.STRONG_POSITIVE
        )

    def test_classify_correlation_moderate_positive(
        self, service: AdvancedAnalyticsService
    ):
        """Test classification of moderate positive correlation."""
        assert (
            service._classify_correlation(0.55) == CorrelationStrength.MODERATE_POSITIVE
        )
        assert (
            service._classify_correlation(0.40) == CorrelationStrength.MODERATE_POSITIVE
        )

    def test_classify_correlation_weak_positive(
        self, service: AdvancedAnalyticsService
    ):
        """Test classification of weak positive correlation."""
        assert service._classify_correlation(0.30) == CorrelationStrength.WEAK_POSITIVE
        assert service._classify_correlation(0.20) == CorrelationStrength.WEAK_POSITIVE

    def test_classify_correlation_no_correlation(
        self, service: AdvancedAnalyticsService
    ):
        """Test classification of no correlation."""
        assert service._classify_correlation(0.15) == CorrelationStrength.NO_CORRELATION
        assert service._classify_correlation(0.0) == CorrelationStrength.NO_CORRELATION
        assert (
            service._classify_correlation(-0.15) == CorrelationStrength.NO_CORRELATION
        )

    def test_classify_correlation_negative(self, service: AdvancedAnalyticsService):
        """Test classification of negative correlations."""
        assert service._classify_correlation(-0.30) == CorrelationStrength.WEAK_NEGATIVE
        assert (
            service._classify_correlation(-0.55)
            == CorrelationStrength.MODERATE_NEGATIVE
        )
        assert (
            service._classify_correlation(-0.85) == CorrelationStrength.STRONG_NEGATIVE
        )


class TestScheduledReportsRepository:
    """Tests for ScheduledReportsRepository."""

    @pytest.fixture
    def mock_session(self) -> AsyncSession:
        """Create a mock async session."""
        session = MagicMock(spec=AsyncSession)
        session.execute = AsyncMock()
        session.add = MagicMock()
        session.delete = AsyncMock()
        session.flush = AsyncMock()
        return session

    @pytest.fixture
    def repository(self, mock_session: AsyncSession) -> ScheduledReportsRepository:
        """Create a repository instance."""
        return ScheduledReportsRepository(mock_session)

    def test_calculate_next_run_daily(self, repository: ScheduledReportsRepository):
        """Test next run calculation for daily reports."""
        next_run = repository._calculate_next_run(ScheduledReportFrequency.DAILY)

        assert next_run.hour == 8
        assert next_run.minute == 0
        assert next_run > datetime.now()

    def test_calculate_next_run_weekly(self, repository: ScheduledReportsRepository):
        """Test next run calculation for weekly reports."""
        next_run = repository._calculate_next_run(ScheduledReportFrequency.WEEKLY)

        assert next_run.weekday() == 0  # Monday
        assert next_run.hour == 8
        assert next_run.minute == 0
        assert next_run > datetime.now()

    def test_calculate_next_run_monthly(self, repository: ScheduledReportsRepository):
        """Test next run calculation for monthly reports."""
        next_run = repository._calculate_next_run(ScheduledReportFrequency.MONTHLY)

        assert next_run.day == 1
        assert next_run.hour == 8
        assert next_run.minute == 0
        assert next_run > datetime.now()


class TestScheduledReportConfig:
    """Tests for ScheduledReportConfig model."""

    def test_create_config_minimal(self):
        """Test creating config with minimal fields."""
        config = ScheduledReportConfig(
            name="Test Report",
            frequency=ScheduledReportFrequency.DAILY,
        )

        assert config.name == "Test Report"
        assert config.frequency == ScheduledReportFrequency.DAILY
        assert config.recipients == []
        assert config.include_trends is True
        assert config.include_anomalies is True
        assert config.include_correlations is False
        assert config.is_active is True

    def test_create_config_full(self):
        """Test creating config with all fields."""
        config = ScheduledReportConfig(
            id=1,
            name="Full Report",
            frequency=ScheduledReportFrequency.WEEKLY,
            recipients=["user@example.com"],
            include_trends=True,
            include_anomalies=True,
            include_correlations=True,
            filters={"suite_name": "test-suite"},
            is_active=True,
            next_run=datetime.now() + timedelta(days=7),
        )

        assert config.id == 1
        assert config.name == "Full Report"
        assert config.frequency == ScheduledReportFrequency.WEEKLY
        assert len(config.recipients) == 1
        assert config.include_correlations is True
        assert "suite_name" in config.filters


class TestTrendDirection:
    """Tests for TrendDirection enum."""

    def test_trend_direction_values(self):
        """Test all trend direction values exist."""
        assert TrendDirection.IMPROVING.value == "improving"
        assert TrendDirection.DECLINING.value == "declining"
        assert TrendDirection.STABLE.value == "stable"
        assert TrendDirection.INSUFFICIENT_DATA.value == "insufficient_data"


class TestAnomalyType:
    """Tests for AnomalyType enum."""

    def test_anomaly_type_values(self):
        """Test all anomaly type values exist."""
        assert AnomalyType.SCORE_SPIKE.value == "score_spike"
        assert AnomalyType.SCORE_DROP.value == "score_drop"
        assert AnomalyType.DURATION_SPIKE.value == "duration_spike"
        assert AnomalyType.ERROR_RATE_SPIKE.value == "error_rate_spike"
        assert AnomalyType.COST_SPIKE.value == "cost_spike"


class TestCorrelationStrength:
    """Tests for CorrelationStrength enum."""

    def test_correlation_strength_values(self):
        """Test all correlation strength values exist."""
        assert CorrelationStrength.STRONG_POSITIVE.value == "strong_positive"
        assert CorrelationStrength.MODERATE_POSITIVE.value == "moderate_positive"
        assert CorrelationStrength.WEAK_POSITIVE.value == "weak_positive"
        assert CorrelationStrength.NO_CORRELATION.value == "no_correlation"
        assert CorrelationStrength.WEAK_NEGATIVE.value == "weak_negative"
        assert CorrelationStrength.MODERATE_NEGATIVE.value == "moderate_negative"
        assert CorrelationStrength.STRONG_NEGATIVE.value == "strong_negative"


class TestTrendDataPoint:
    """Tests for TrendDataPoint model."""

    def test_create_trend_data_point(self):
        """Test creating a TrendDataPoint."""
        from atp.analytics.advanced import TrendDataPoint

        point = TrendDataPoint(date="2025-01-01", value=0.85, count=5)
        assert point.date == "2025-01-01"
        assert point.value == 0.85
        assert point.count == 5

    def test_trend_data_point_default_count(self):
        """Test TrendDataPoint default count value."""
        from atp.analytics.advanced import TrendDataPoint

        point = TrendDataPoint(date="2025-01-01", value=0.75)
        assert point.count == 1


class TestTrendAnalysisResult:
    """Tests for TrendAnalysisResult model."""

    def test_create_trend_analysis_result(self):
        """Test creating a TrendAnalysisResult."""
        from atp.analytics.advanced import TrendAnalysisResult, TrendDataPoint

        result = TrendAnalysisResult(
            metric="score",
            direction=TrendDirection.IMPROVING,
            change_percent=15.5,
            start_value=0.65,
            end_value=0.80,
            average_value=0.72,
            std_deviation=0.05,
            data_points=[
                TrendDataPoint(date="2025-01-01", value=0.65),
                TrendDataPoint(date="2025-01-02", value=0.80),
            ],
            period_days=2,
            confidence=0.85,
        )

        assert result.metric == "score"
        assert result.direction == TrendDirection.IMPROVING
        assert result.change_percent == 15.5


class TestAnomalyResult:
    """Tests for AnomalyResult model."""

    def test_create_anomaly_result(self):
        """Test creating an AnomalyResult."""
        from atp.analytics.advanced import AnomalyResult

        anomaly = AnomalyResult(
            anomaly_type=AnomalyType.SCORE_DROP,
            timestamp=datetime.now(),
            metric_name="score",
            expected_value=0.75,
            actual_value=0.25,
            deviation_sigma=3.5,
            test_id="test-1",
            suite_id="suite-1",
            severity="high",
        )

        assert anomaly.anomaly_type == AnomalyType.SCORE_DROP
        assert anomaly.severity == "high"

    def test_anomaly_result_default_severity(self):
        """Test AnomalyResult default severity."""
        from atp.analytics.advanced import AnomalyResult

        anomaly = AnomalyResult(
            anomaly_type=AnomalyType.SCORE_SPIKE,
            timestamp=datetime.now(),
            metric_name="score",
            expected_value=0.5,
            actual_value=0.95,
            deviation_sigma=2.5,
        )

        assert anomaly.severity == "medium"


class TestCorrelationResult:
    """Tests for CorrelationResult model."""

    def test_create_correlation_result(self):
        """Test creating a CorrelationResult."""
        from atp.analytics.advanced import CorrelationResult

        result = CorrelationResult(
            factor_x="duration",
            factor_y="score",
            correlation_coefficient=-0.75,
            strength=CorrelationStrength.STRONG_NEGATIVE,
            sample_size=100,
        )

        assert result.factor_x == "duration"
        assert result.correlation_coefficient == -0.75
        assert result.strength == CorrelationStrength.STRONG_NEGATIVE


class TestScoreTrendResponse:
    """Tests for ScoreTrendResponse model."""

    def test_create_score_trend_response(self):
        """Test creating a ScoreTrendResponse."""
        from atp.analytics.advanced import ScoreTrendResponse

        response = ScoreTrendResponse(
            suite_name="test-suite",
            agent_name="test-agent",
            trends=[],
            period_start=datetime(2025, 1, 1),
            period_end=datetime(2025, 1, 31),
        )

        assert response.suite_name == "test-suite"
        assert response.agent_name == "test-agent"


class TestAnomalyDetectionResponse:
    """Tests for AnomalyDetectionResponse model."""

    def test_create_anomaly_detection_response(self):
        """Test creating an AnomalyDetectionResponse."""
        from atp.analytics.advanced import AnomalyDetectionResponse

        response = AnomalyDetectionResponse(
            anomalies=[],
            total_records_analyzed=100,
            anomaly_rate=0.05,
            period_start=datetime(2025, 1, 1),
            period_end=datetime(2025, 1, 31),
        )

        assert response.total_records_analyzed == 100
        assert response.anomaly_rate == 0.05


class TestCorrelationAnalysisResponse:
    """Tests for CorrelationAnalysisResponse model."""

    def test_create_correlation_analysis_response(self):
        """Test creating a CorrelationAnalysisResponse."""
        from atp.analytics.advanced import CorrelationAnalysisResponse

        response = CorrelationAnalysisResponse(
            correlations=[],
            sample_size=50,
            factors_analyzed=["duration", "tokens"],
        )

        assert response.sample_size == 50
        assert "duration" in response.factors_analyzed


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_export_format_values(self):
        """Test all export format values exist."""
        from atp.analytics.advanced import ExportFormat

        assert ExportFormat.CSV.value == "csv"
        assert ExportFormat.EXCEL.value == "excel"


class TestScheduledReportFrequency:
    """Tests for ScheduledReportFrequency enum."""

    def test_scheduled_report_frequency_values(self):
        """Test all frequency values exist."""
        assert ScheduledReportFrequency.DAILY.value == "daily"
        assert ScheduledReportFrequency.WEEKLY.value == "weekly"
        assert ScheduledReportFrequency.MONTHLY.value == "monthly"

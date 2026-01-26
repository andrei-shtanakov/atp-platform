"""Unit tests for the leaderboard matrix endpoint.

This module tests:
- /leaderboard/matrix endpoint validation
- Difficulty calculation helper function
- Pattern detection helper function
- Schema validation for leaderboard types
"""

import pytest
from fastapi.testclient import TestClient

from atp.dashboard.api import _calculate_difficulty, _detect_pattern
from atp.dashboard.app import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client using the actual app."""
    return TestClient(app, raise_server_exceptions=False)


class TestLeaderboardMatrixEndpoint:
    """Tests for /leaderboard/matrix endpoint."""

    def test_endpoint_requires_suite_name(self, client: TestClient) -> None:
        """Test that suite_name is required."""
        response = client.get("/api/leaderboard/matrix")
        assert response.status_code == 422

    def test_endpoint_accepts_valid_params(self, client: TestClient) -> None:
        """Test endpoint with valid parameters."""
        response = client.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "benchmark-suite",
                "limit_executions": 5,
            },
        )
        # Should return 200 (success) or 500 (db not configured)
        assert response.status_code in [200, 500]

    def test_endpoint_validates_limit_executions(self, client: TestClient) -> None:
        """Test that limit_executions has a maximum value of 20."""
        response = client.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "benchmark-suite",
                "limit_executions": 100,  # Exceeds max of 20
            },
        )
        assert response.status_code == 422

    def test_endpoint_validates_limit(self, client: TestClient) -> None:
        """Test that limit has a maximum value of 100."""
        response = client.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "benchmark-suite",
                "limit": 200,  # Exceeds max of 100
            },
        )
        assert response.status_code == 422

    def test_endpoint_validates_offset(self, client: TestClient) -> None:
        """Test that offset must be non-negative."""
        response = client.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "benchmark-suite",
                "offset": -1,  # Must be >= 0
            },
        )
        assert response.status_code == 422

    def test_endpoint_accepts_agents_filter(self, client: TestClient) -> None:
        """Test endpoint with agents filter."""
        response = client.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "benchmark-suite",
                "agents": ["agent-1", "agent-2"],
            },
        )
        # Should return 200 (success) or 500 (db not configured)
        assert response.status_code in [200, 500]

    def test_endpoint_accepts_pagination_params(self, client: TestClient) -> None:
        """Test endpoint with pagination parameters."""
        response = client.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "benchmark-suite",
                "limit": 10,
                "offset": 5,
            },
        )
        # Should return 200 (success) or 500 (db not configured)
        assert response.status_code in [200, 500]


class TestCalculateDifficulty:
    """Tests for _calculate_difficulty helper function."""

    def test_difficulty_easy(self) -> None:
        """Test that high scores result in 'easy' difficulty."""
        assert _calculate_difficulty(95.0) == "easy"
        assert _calculate_difficulty(85.0) == "easy"
        assert _calculate_difficulty(80.0) == "easy"

    def test_difficulty_medium(self) -> None:
        """Test that medium scores result in 'medium' difficulty."""
        assert _calculate_difficulty(79.9) == "medium"
        assert _calculate_difficulty(70.0) == "medium"
        assert _calculate_difficulty(60.0) == "medium"

    def test_difficulty_hard(self) -> None:
        """Test that low scores result in 'hard' difficulty."""
        assert _calculate_difficulty(59.9) == "hard"
        assert _calculate_difficulty(50.0) == "hard"
        assert _calculate_difficulty(40.0) == "hard"

    def test_difficulty_very_hard(self) -> None:
        """Test that very low scores result in 'very_hard' difficulty."""
        assert _calculate_difficulty(39.9) == "very_hard"
        assert _calculate_difficulty(20.0) == "very_hard"
        assert _calculate_difficulty(0.0) == "very_hard"

    def test_difficulty_unknown(self) -> None:
        """Test that None score results in 'unknown' difficulty."""
        assert _calculate_difficulty(None) == "unknown"


class TestDetectPattern:
    """Tests for _detect_pattern helper function."""

    def test_pattern_hard_for_all_low_scores(self) -> None:
        """Test that all low scores result in 'hard_for_all' pattern."""
        scores = [35.0, 30.0, 25.0]
        pass_rates = [0.1, 0.1, 0.1]
        assert _detect_pattern(scores, pass_rates) == "hard_for_all"

    def test_pattern_hard_for_all_low_pass_rates(self) -> None:
        """Test that low pass rates result in 'hard_for_all' pattern."""
        scores = [60.0, 55.0, 50.0]
        pass_rates = [0.15, 0.1, 0.2]
        assert _detect_pattern(scores, pass_rates) == "hard_for_all"

    def test_pattern_easy(self) -> None:
        """Test that high scores and pass rates result in 'easy' pattern."""
        scores = [90.0, 85.0, 92.0]
        pass_rates = [0.95, 0.9, 0.85]
        assert _detect_pattern(scores, pass_rates) == "easy"

    def test_pattern_high_variance(self) -> None:
        """Test that high score variance results in 'high_variance' pattern."""
        scores = [90.0, 50.0]  # 40 point difference
        pass_rates = [0.9, 0.5]
        assert _detect_pattern(scores, pass_rates) == "high_variance"

    def test_pattern_none_for_medium_scores(self) -> None:
        """Test that medium scores with low variance return None."""
        scores = [70.0, 65.0, 72.0]
        pass_rates = [0.7, 0.65, 0.72]
        assert _detect_pattern(scores, pass_rates) is None

    def test_pattern_none_for_empty_scores(self) -> None:
        """Test that empty scores return None."""
        assert _detect_pattern([], []) is None

    def test_pattern_none_for_all_none_scores(self) -> None:
        """Test that all None scores return None."""
        scores: list[float | None] = [None, None, None]
        pass_rates = [0.0, 0.0, 0.0]
        assert _detect_pattern(scores, pass_rates) is None

    def test_pattern_with_mixed_none_scores(self) -> None:
        """Test pattern detection with some None scores.

        Note: Pattern 'easy' requires ALL pass_rates >= 0.8.
        With a 0.0 pass_rate, pattern is None.
        """
        scores: list[float | None] = [85.0, None, 90.0]
        pass_rates = [0.9, 0.0, 0.95]
        # Not "easy" because not all pass_rates >= 0.8
        assert _detect_pattern(scores, pass_rates) is None

    def test_pattern_easy_with_some_none_scores(self) -> None:
        """Test 'easy' pattern with some None scores but high pass rates."""
        scores: list[float | None] = [85.0, None, 90.0]
        pass_rates = [0.9, 0.85, 0.95]  # All >= 0.8
        assert _detect_pattern(scores, pass_rates) == "easy"


class TestRouterConfiguration:
    """Tests for router configuration."""

    def test_leaderboard_route_exists(self, client: TestClient) -> None:
        """Test that leaderboard/matrix route exists."""
        from atp.dashboard.api import router

        route_paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/leaderboard/matrix" in route_paths

    def test_leaderboard_route_has_tag(self, client: TestClient) -> None:
        """Test that leaderboard route has correct tag."""
        from atp.dashboard.api import router

        for route in router.routes:
            if hasattr(route, "path") and route.path == "/leaderboard/matrix":
                assert "leaderboard" in route.tags
                break
        else:
            pytest.fail("Leaderboard route not found")

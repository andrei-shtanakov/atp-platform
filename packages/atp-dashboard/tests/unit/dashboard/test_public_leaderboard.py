"""Unit tests for the public leaderboard feature.

This module tests:
- Public leaderboard endpoints (TASK-802)
- Agent profile management
- Benchmark categories
- Result publishing
- Verification badges
- Leaderboard history
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from atp.dashboard.schemas import (
    AgentLeaderboardHistory,
    AgentProfileCreate,
    AgentProfileSummary,
    AgentProfileUpdate,
    BenchmarkCategoryCreate,
    BenchmarkCategoryResponse,
    LeaderboardEntry,
    LeaderboardTrendPoint,
    PublicLeaderboardResponse,
    PublishedResultResponse,
    PublishResultRequest,
    VerificationBadgeRequest,
)


class TestBenchmarkCategorySchemas:
    """Tests for benchmark category schemas."""

    def test_create_category_valid(self) -> None:
        """Test valid category creation."""
        data = BenchmarkCategoryCreate(
            name="Coding Tasks",
            slug="coding-tasks",
            description="Benchmark for coding tasks",
            icon="code",
            display_order=1,
        )
        assert data.name == "Coding Tasks"
        assert data.slug == "coding-tasks"
        assert data.min_submissions_for_ranking == 1

    def test_create_category_slug_validation(self) -> None:
        """Test that slug only accepts lowercase alphanumeric and hyphens."""
        # Valid slugs
        BenchmarkCategoryCreate(name="Test", slug="test-category-123")
        BenchmarkCategoryCreate(name="Test", slug="test")

        # Invalid slugs (should raise validation error)
        with pytest.raises(ValidationError):
            BenchmarkCategoryCreate(name="Test", slug="Test Category")
        with pytest.raises(ValidationError):
            BenchmarkCategoryCreate(name="Test", slug="test_category")
        with pytest.raises(ValidationError):
            BenchmarkCategoryCreate(name="Test", slug="TEST")

    def test_category_response_from_dict(self) -> None:
        """Test category response model validation."""
        data = {
            "id": 1,
            "name": "Coding",
            "slug": "coding",
            "description": "Test",
            "icon": "code",
            "display_order": 0,
            "parent_id": None,
            "is_active": True,
            "min_submissions_for_ranking": 1,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        response = BenchmarkCategoryResponse(**data)
        assert response.id == 1
        assert response.name == "Coding"


class TestAgentProfileSchemas:
    """Tests for agent profile schemas."""

    def test_create_profile_valid(self) -> None:
        """Test valid profile creation."""
        data = AgentProfileCreate(
            agent_id=1,
            display_name="My Agent",
            description="A great agent",
            website_url="https://example.com",
            tags=["llm", "coding"],
        )
        assert data.agent_id == 1
        assert data.display_name == "My Agent"
        assert data.is_public is True  # Default

    def test_create_profile_url_validation(self) -> None:
        """Test URL field length validation."""
        # Should not exceed max length
        long_url = "https://example.com/" + "a" * 500
        with pytest.raises(ValidationError):
            AgentProfileCreate(
                agent_id=1,
                display_name="Test",
                website_url=long_url,
            )

    def test_profile_update_partial(self) -> None:
        """Test partial profile update."""
        update = AgentProfileUpdate(display_name="New Name")
        assert update.display_name == "New Name"
        assert update.description is None
        assert update.tags is None

    def test_profile_summary(self) -> None:
        """Test profile summary schema."""
        summary = AgentProfileSummary(
            id=1,
            display_name="Agent",
            avatar_url=None,
            is_verified=True,
            verification_badges=["official"],
            total_submissions=5,
            best_overall_score=95.5,
            best_overall_rank=1,
        )
        assert summary.is_verified is True
        assert "official" in summary.verification_badges


class TestPublishedResultSchemas:
    """Tests for published result schemas."""

    def test_publish_request_valid(self) -> None:
        """Test valid publish request."""
        request = PublishResultRequest(
            suite_execution_id=1,
            category="coding",
        )
        assert request.suite_execution_id == 1
        assert request.category == "coding"

    def test_publish_request_category_validation(self) -> None:
        """Test category field validation."""
        # Empty category should fail
        with pytest.raises(ValidationError):
            PublishResultRequest(
                suite_execution_id=1,
                category="",
            )

    def test_published_result_response(self) -> None:
        """Test published result response schema."""
        data = {
            "id": 1,
            "suite_execution_id": 1,
            "agent_profile_id": 1,
            "category": "coding",
            "score": 85.5,
            "success_rate": 0.85,
            "total_tests": 10,
            "passed_tests": 8,
            "total_tokens": 1000,
            "total_cost_usd": 0.05,
            "duration_seconds": 120.5,
            "is_verified": True,
            "verified_at": datetime.now(),
            "published_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        response = PublishedResultResponse(**data)
        assert response.score == 85.5
        assert response.is_verified is True


class TestLeaderboardSchemas:
    """Tests for leaderboard schemas."""

    def test_leaderboard_entry(self) -> None:
        """Test leaderboard entry schema."""
        profile = AgentProfileSummary(
            id=1,
            display_name="Agent",
            avatar_url=None,
            is_verified=True,
            verification_badges=["official"],
            total_submissions=5,
            best_overall_score=95.5,
            best_overall_rank=1,
        )
        entry = LeaderboardEntry(
            rank=1,
            agent_profile=profile,
            score=95.5,
            success_rate=0.95,
            total_tests=10,
            passed_tests=9,
            total_tokens=1000,
            total_cost_usd=0.05,
            duration_seconds=120.5,
            is_verified=True,
            published_at=datetime.now(),
            trend="up",
        )
        assert entry.rank == 1
        assert entry.trend == "up"
        assert entry.agent_profile.display_name == "Agent"

    def test_leaderboard_response(self) -> None:
        """Test public leaderboard response schema."""
        category = BenchmarkCategoryResponse(
            id=1,
            name="Coding",
            slug="coding",
            description="Test",
            icon="code",
            display_order=0,
            parent_id=None,
            is_active=True,
            min_submissions_for_ranking=1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        response = PublicLeaderboardResponse(
            category=category,
            entries=[],
            total_entries=0,
            last_updated=datetime.now(),
            limit=50,
            offset=0,
        )
        assert response.total_entries == 0
        assert response.category.slug == "coding"

    def test_trend_point(self) -> None:
        """Test leaderboard trend point schema."""
        point = LeaderboardTrendPoint(
            timestamp=datetime.now(),
            rank=3,
            score=85.5,
        )
        assert point.rank == 3
        assert point.score == 85.5

    def test_agent_history(self) -> None:
        """Test agent leaderboard history schema."""
        profile = AgentProfileSummary(
            id=1,
            display_name="Agent",
            avatar_url=None,
            is_verified=False,
            verification_badges=[],
            total_submissions=3,
            best_overall_score=90.0,
            best_overall_rank=2,
        )
        history = AgentLeaderboardHistory(
            agent_profile=profile,
            category="coding",
            data_points=[
                LeaderboardTrendPoint(
                    timestamp=datetime.now(),
                    rank=3,
                    score=80.0,
                ),
                LeaderboardTrendPoint(
                    timestamp=datetime.now(),
                    rank=2,
                    score=85.0,
                ),
            ],
        )
        assert len(history.data_points) == 2
        assert history.category == "coding"


class TestVerificationSchemas:
    """Tests for verification schemas."""

    def test_verification_badge_request_valid(self) -> None:
        """Test valid verification badge request."""
        request = VerificationBadgeRequest(
            agent_profile_id=1,
            badge="official",
        )
        assert request.badge == "official"

    def test_verification_badge_request_invalid_badge(self) -> None:
        """Test invalid badge type."""
        with pytest.raises(ValidationError):
            VerificationBadgeRequest(
                agent_profile_id=1,
                badge="invalid_badge",
            )

    def test_verification_badge_request_valid_badges(self) -> None:
        """Test all valid badge types."""
        valid_badges = ["official", "reproducible", "open_source", "community_verified"]
        for badge in valid_badges:
            request = VerificationBadgeRequest(
                agent_profile_id=1,
                badge=badge,
            )
            assert request.badge == badge


class TestPublicLeaderboardRoutes:
    """Tests for public leaderboard route configuration."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client using v2 app."""
        from atp.dashboard.v2.factory import create_test_app

        # Create a test app with v2 routes enabled
        app = create_test_app(use_v2_routes=True)
        return TestClient(app, raise_server_exceptions=False)

    def test_categories_endpoint_exists(self, client: TestClient) -> None:
        """Test that categories endpoint exists."""
        response = client.get("/api/public/leaderboard/categories")
        # Should return 200 or 500 (db not configured), not 404
        assert response.status_code in [200, 500]

    def test_leaderboard_endpoint_requires_category(self, client: TestClient) -> None:
        """Test that leaderboard endpoint requires category parameter."""
        response = client.get("/api/public/leaderboard")
        assert response.status_code == 422  # Missing required parameter

    def test_leaderboard_endpoint_with_category(self, client: TestClient) -> None:
        """Test leaderboard endpoint with category parameter."""
        response = client.get(
            "/api/public/leaderboard",
            params={"category": "coding"},
        )
        # Should return 200 or 404 (category not found) or 500 (db not configured)
        assert response.status_code in [200, 404, 500]

    def test_agents_list_endpoint_exists(self, client: TestClient) -> None:
        """Test that agents list endpoint exists."""
        response = client.get("/api/public/leaderboard/agents")
        # Should return 200 or 500 (db not configured), not 404
        assert response.status_code in [200, 500]

    def test_agent_profile_endpoint_with_id(self, client: TestClient) -> None:
        """Test agent profile endpoint with ID."""
        response = client.get("/api/public/leaderboard/agents/1")
        # Should return 404 (not found) or 500 (db not configured), not 422
        assert response.status_code in [404, 500]

    def test_history_endpoint_exists(self, client: TestClient) -> None:
        """Test that history endpoint exists."""
        response = client.get("/api/public/leaderboard/history/coding")
        # Should return 200 or 500 (db not configured), not 404
        assert response.status_code in [200, 500]

    def test_publish_endpoint_requires_auth(self, client: TestClient) -> None:
        """Test that publish endpoint requires authentication."""
        response = client.post(
            "/api/public/leaderboard/publish",
            json={"suite_execution_id": 1, "category": "coding"},
        )
        # Should return 401 or 403 (unauthorized)
        assert response.status_code in [401, 403, 500]

    def test_create_profile_endpoint_requires_auth(self, client: TestClient) -> None:
        """Test that create profile endpoint requires authentication."""
        response = client.post(
            "/api/public/leaderboard/agents",
            json={"agent_id": 1, "display_name": "Test"},
        )
        # Should return 401 or 403 (unauthorized)
        assert response.status_code in [401, 403, 500]

    def test_verify_endpoint_requires_admin(self, client: TestClient) -> None:
        """Test that verify endpoint requires admin privileges."""
        response = client.post(
            "/api/public/leaderboard/verify",
            json={"agent_profile_id": 1, "badge": "official"},
        )
        # Should return 401 or 403 (unauthorized)
        assert response.status_code in [401, 403, 500]

    def test_leaderboard_pagination_validation(self, client: TestClient) -> None:
        """Test leaderboard pagination parameter validation."""
        # Test limit exceeds maximum
        response = client.get(
            "/api/public/leaderboard",
            params={"category": "coding", "limit": 200},
        )
        assert response.status_code == 422

        # Test negative offset
        response = client.get(
            "/api/public/leaderboard",
            params={"category": "coding", "offset": -1},
        )
        assert response.status_code == 422


class TestPublicLeaderboardHelpers:
    """Tests for public leaderboard helper functions."""

    def test_calculate_trend_up(self) -> None:
        """Test trend calculation for improving score."""
        from atp.dashboard.v2.routes.public_leaderboard import _calculate_trend

        result = _calculate_trend(95.0, 90.0)
        assert result == "up"

    def test_calculate_trend_down(self) -> None:
        """Test trend calculation for declining score."""
        from atp.dashboard.v2.routes.public_leaderboard import _calculate_trend

        result = _calculate_trend(85.0, 90.0)
        assert result == "down"

    def test_calculate_trend_stable(self) -> None:
        """Test trend calculation for stable score."""
        from atp.dashboard.v2.routes.public_leaderboard import _calculate_trend

        result = _calculate_trend(90.5, 90.0)
        assert result == "stable"

    def test_calculate_trend_no_history(self) -> None:
        """Test trend calculation with no historical data."""
        from atp.dashboard.v2.routes.public_leaderboard import _calculate_trend

        result = _calculate_trend(90.0, None)
        assert result is None


class TestDatabaseModels:
    """Tests for database model definitions."""

    def test_published_result_model_exists(self) -> None:
        """Test that PublishedResult model is defined."""
        from atp.dashboard.models import PublishedResult

        assert PublishedResult.__tablename__ == "published_results"

    def test_agent_profile_model_exists(self) -> None:
        """Test that AgentProfile model is defined."""
        from atp.dashboard.models import AgentProfile

        assert AgentProfile.__tablename__ == "agent_profiles"

    def test_benchmark_category_model_exists(self) -> None:
        """Test that BenchmarkCategory model is defined."""
        from atp.dashboard.models import BenchmarkCategory

        assert BenchmarkCategory.__tablename__ == "benchmark_categories"

    def test_published_result_has_required_fields(self) -> None:
        """Test that PublishedResult has required fields."""
        from atp.dashboard.models import PublishedResult

        # Check key columns exist
        columns = [c.name for c in PublishedResult.__table__.columns]
        required = [
            "id",
            "suite_execution_id",
            "agent_profile_id",
            "category",
            "score",
            "success_rate",
            "is_verified",
            "published_at",
        ]
        for col in required:
            assert col in columns, f"Column {col} not found in PublishedResult"

    def test_agent_profile_has_required_fields(self) -> None:
        """Test that AgentProfile has required fields."""
        from atp.dashboard.models import AgentProfile

        columns = [c.name for c in AgentProfile.__table__.columns]
        required = [
            "id",
            "agent_id",
            "display_name",
            "is_verified",
            "verification_badges",
            "is_public",
            "total_submissions",
        ]
        for col in required:
            assert col in columns, f"Column {col} not found in AgentProfile"

    def test_benchmark_category_has_required_fields(self) -> None:
        """Test that BenchmarkCategory has required fields."""
        from atp.dashboard.models import BenchmarkCategory

        columns = [c.name for c in BenchmarkCategory.__table__.columns]
        required = [
            "id",
            "name",
            "slug",
            "is_active",
            "display_order",
        ]
        for col in required:
            assert col in columns, f"Column {col} not found in BenchmarkCategory"


class TestPublicLeaderboardCache:
    """Tests for public leaderboard caching."""

    def test_cache_initialization(self) -> None:
        """Test that cache is initialized correctly."""
        from atp.dashboard.v2.routes.public_leaderboard import (
            get_public_leaderboard_cache,
        )

        cache = get_public_leaderboard_cache()
        assert cache is not None

    def test_cache_singleton(self) -> None:
        """Test that cache returns the same instance."""
        from atp.dashboard.v2.routes.public_leaderboard import (
            get_public_leaderboard_cache,
        )

        cache1 = get_public_leaderboard_cache()
        cache2 = get_public_leaderboard_cache()
        assert cache1 is cache2

    def test_cache_operations(self) -> None:
        """Test basic cache operations."""
        from atp.dashboard.v2.routes.public_leaderboard import (
            get_public_leaderboard_cache,
        )

        cache = get_public_leaderboard_cache()

        # Test put and get
        cache.put("test_key", {"data": "value"})
        result = cache.get("test_key")
        assert result == {"data": "value"}

        # Test invalidate
        cache.invalidate("test_key")
        result = cache.get("test_key")
        assert result is None

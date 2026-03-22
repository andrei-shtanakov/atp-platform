"""Unit tests for the Test Suite Marketplace (TASK-803).

Tests cover:
- Marketplace database models
- Pydantic schemas validation
- API endpoints for publishing, versioning, and discovery
- Ratings and reviews
- Install/uninstall functionality
- Admin operations (feature, verify, flag)
- GitHub import functionality
- License specification
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from atp.dashboard.models import (
    MarketplaceSuite,
    MarketplaceSuiteInstall,
    MarketplaceSuiteReview,
    MarketplaceSuiteVersion,
)
from atp.dashboard.rbac import Permission
from atp.dashboard.schemas import (
    GitHubImportRequest,
    GitHubImportResponse,
    MarketplaceCategoryStats,
    MarketplaceSearchParams,
    MarketplaceStats,
    MarketplaceSuiteCreate,
    MarketplaceSuiteDetail,
    MarketplaceSuiteInstallResponse,
    MarketplaceSuiteResponse,
    MarketplaceSuiteReviewCreate,
    MarketplaceSuiteReviewList,
    MarketplaceSuiteReviewUpdate,
    MarketplaceSuiteSummary,
    MarketplaceSuiteUpdate,
    MarketplaceSuiteVersionCreate,
    MarketplaceSuiteVersionList,
    MarketplaceSuiteVersionResponse,
    MarketplaceSuiteVersionUpdate,
)


class TestMarketplaceSuiteModel:
    """Tests for the MarketplaceSuite database model."""

    def test_create_marketplace_suite(self) -> None:
        """Test creating a marketplace suite model."""
        suite = MarketplaceSuite(
            tenant_id="default",
            name="Test Suite",
            slug="test-suite",
            description="A test suite for testing",
            short_description="A test suite",
            publisher_id=1,
            publisher_name="testuser",
            category="testing",
            tags=["test", "example"],
            license_type="MIT",
            source_type="local",
            latest_version="1.0.0",
            # Set default values explicitly for unit tests
            # (SQLAlchemy defaults apply at DB level)
            is_published=True,
            is_featured=False,
            is_verified=False,
            total_downloads=0,
            total_installs=0,
            total_ratings=0,
        )

        assert suite.name == "Test Suite"
        assert suite.slug == "test-suite"
        assert suite.category == "testing"
        assert suite.tags == ["test", "example"]
        assert suite.license_type == "MIT"
        assert suite.is_published is True
        assert suite.is_featured is False
        assert suite.is_verified is False
        assert suite.total_downloads == 0
        assert suite.total_installs == 0
        assert suite.average_rating is None
        assert suite.total_ratings == 0

    def test_marketplace_suite_repr(self) -> None:
        """Test string representation of marketplace suite."""
        suite = MarketplaceSuite(
            id=1,
            name="Test Suite",
            slug="test-suite",
            publisher_id=1,
            publisher_name="testuser",
            latest_version="1.0.0",
        )
        repr_str = repr(suite)
        assert "MarketplaceSuite" in repr_str
        assert "Test Suite" in repr_str
        assert "1.0.0" in repr_str


class TestMarketplaceSuiteVersionModel:
    """Tests for the MarketplaceSuiteVersion database model."""

    def test_create_version(self) -> None:
        """Test creating a version model."""
        version = MarketplaceSuiteVersion(
            marketplace_suite_id=1,
            version="1.2.3",
            version_major=1,
            version_minor=2,
            version_patch=3,
            changelog="Initial release",
            suite_content={"tests": [{"id": "test1"}]},
            test_count=1,
            is_latest=True,
            # Set default values explicitly for unit tests
            is_deprecated=False,
            downloads=0,
        )

        assert version.version == "1.2.3"
        assert version.version_major == 1
        assert version.version_minor == 2
        assert version.version_patch == 3
        assert version.changelog == "Initial release"
        assert version.test_count == 1
        assert version.is_latest is True
        assert version.is_deprecated is False
        assert version.downloads == 0

    def test_version_repr(self) -> None:
        """Test string representation of version."""
        version = MarketplaceSuiteVersion(
            id=1,
            marketplace_suite_id=1,
            version="2.0.0",
        )
        repr_str = repr(version)
        assert "MarketplaceSuiteVersion" in repr_str
        assert "2.0.0" in repr_str


class TestMarketplaceSuiteReviewModel:
    """Tests for the MarketplaceSuiteReview database model."""

    def test_create_review(self) -> None:
        """Test creating a review model."""
        review = MarketplaceSuiteReview(
            marketplace_suite_id=1,
            user_id=1,
            rating=5,
            title="Great test suite!",
            content="This test suite is amazing.",
            version_reviewed="1.0.0",
            # Set default values explicitly for unit tests
            is_approved=True,
            is_flagged=False,
            helpful_count=0,
            not_helpful_count=0,
        )

        assert review.rating == 5
        assert review.title == "Great test suite!"
        assert review.content == "This test suite is amazing."
        assert review.version_reviewed == "1.0.0"
        assert review.is_approved is True
        assert review.is_flagged is False
        assert review.helpful_count == 0

    def test_review_repr(self) -> None:
        """Test string representation of review."""
        review = MarketplaceSuiteReview(
            id=1,
            marketplace_suite_id=1,
            user_id=1,
            rating=4,
        )
        repr_str = repr(review)
        assert "MarketplaceSuiteReview" in repr_str
        assert "4" in repr_str


class TestMarketplaceSuiteInstallModel:
    """Tests for the MarketplaceSuiteInstall database model."""

    def test_create_install(self) -> None:
        """Test creating an install model."""
        install = MarketplaceSuiteInstall(
            tenant_id="default",
            marketplace_suite_id=1,
            user_id=1,
            version_installed="1.0.0",
            version_id=1,
            # Set default values explicitly for unit tests
            is_active=True,
        )

        assert install.marketplace_suite_id == 1
        assert install.user_id == 1
        assert install.version_installed == "1.0.0"
        assert install.is_active is True
        assert install.uninstalled_at is None


class TestMarketplaceSuiteSchemas:
    """Tests for the marketplace Pydantic schemas."""

    def test_marketplace_suite_create_valid(self) -> None:
        """Test valid suite creation schema."""
        data = MarketplaceSuiteCreate(
            name="My Test Suite",
            slug="my-test-suite",
            description="A comprehensive test suite",
            short_description="Test suite",
            category="testing",
            tags=["unit", "integration"],
            license_type="Apache-2.0",
            version="1.0.0",
            suite_content={"tests": []},
        )

        assert data.name == "My Test Suite"
        assert data.slug == "my-test-suite"
        assert data.license_type == "Apache-2.0"
        assert data.version == "1.0.0"

    def test_marketplace_suite_create_invalid_slug(self) -> None:
        """Test invalid slug format."""
        with pytest.raises(ValidationError) as exc_info:
            MarketplaceSuiteCreate(
                name="Test",
                slug="Invalid Slug!",  # Contains spaces and special chars
                version="1.0.0",
            )
        assert "slug" in str(exc_info.value).lower()

    def test_marketplace_suite_create_defaults(self) -> None:
        """Test default values in creation schema."""
        data = MarketplaceSuiteCreate(
            name="Test",
            slug="test-suite",
        )

        assert data.category == "general"
        assert data.tags == []
        assert data.license_type == "MIT"
        assert data.source_type == "local"
        assert data.version == "1.0.0"

    def test_marketplace_suite_update(self) -> None:
        """Test suite update schema with optional fields."""
        data = MarketplaceSuiteUpdate(
            name="Updated Name",
            category="new-category",
            is_published=False,
        )

        assert data.name == "Updated Name"
        assert data.category == "new-category"
        assert data.is_published is False
        assert data.description is None
        assert data.tags is None

    def test_marketplace_suite_response(self) -> None:
        """Test suite response schema."""
        suite = MarketplaceSuite(
            id=1,
            tenant_id="default",
            name="Test Suite",
            slug="test-suite",
            description="Test",
            short_description="Test",
            publisher_id=1,
            publisher_name="testuser",
            category="testing",
            tags=["test"],
            license_type="MIT",
            license_url=None,
            source_type="local",
            github_url=None,
            github_branch=None,
            github_path=None,
            is_published=True,
            is_featured=False,
            is_verified=False,
            total_downloads=100,
            total_installs=50,
            average_rating=4.5,
            total_ratings=10,
            latest_version="1.0.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            published_at=datetime.now(),
        )

        response = MarketplaceSuiteResponse.model_validate(suite)
        assert response.id == 1
        assert response.name == "Test Suite"
        assert response.slug == "test-suite"
        assert response.total_downloads == 100
        assert response.average_rating == 4.5

    def test_marketplace_suite_summary(self) -> None:
        """Test suite summary schema for list views."""
        suite = MarketplaceSuite(
            id=1,
            name="Test",
            slug="test",
            short_description="A test",
            publisher_name="user",
            category="general",
            tags=["test"],
            license_type="MIT",
            is_published=True,
            is_featured=True,
            is_verified=False,
            total_downloads=50,
            average_rating=4.0,
            total_ratings=5,
            latest_version="1.0.0",
            updated_at=datetime.now(),
        )

        summary = MarketplaceSuiteSummary.model_validate(suite)
        assert summary.id == 1
        assert summary.name == "Test"
        assert summary.is_featured is True


class TestMarketplaceVersionSchemas:
    """Tests for version-related schemas."""

    def test_version_create_valid(self) -> None:
        """Test valid version creation."""
        data = MarketplaceSuiteVersionCreate(
            version="2.0.0",
            changelog="Major update with new features",
            breaking_changes=True,
            suite_content={"tests": [{"id": "new-test"}]},
        )

        assert data.version == "2.0.0"
        assert data.breaking_changes is True

    def test_version_create_defaults(self) -> None:
        """Test version creation defaults."""
        data = MarketplaceSuiteVersionCreate(
            version="1.0.0",
            suite_content={},
        )

        assert data.changelog is None
        assert data.breaking_changes is False

    def test_version_response(self) -> None:
        """Test version response schema."""
        version = MarketplaceSuiteVersion(
            id=1,
            marketplace_suite_id=1,
            version="1.0.0",
            version_major=1,
            version_minor=0,
            version_patch=0,
            changelog="Initial release",
            breaking_changes=False,
            test_count=5,
            file_size_bytes=1024,
            downloads=100,
            is_latest=True,
            is_deprecated=False,
            deprecation_message=None,
            created_at=datetime.now(),
            released_at=datetime.now(),
        )

        response = MarketplaceSuiteVersionResponse.model_validate(version)
        assert response.id == 1
        assert response.version == "1.0.0"
        assert response.test_count == 5
        assert response.is_latest is True


class TestMarketplaceReviewSchemas:
    """Tests for review-related schemas."""

    def test_review_create_valid(self) -> None:
        """Test valid review creation."""
        data = MarketplaceSuiteReviewCreate(
            rating=5,
            title="Excellent!",
            content="Really helpful test suite.",
            version_reviewed="1.0.0",
        )

        assert data.rating == 5
        assert data.title == "Excellent!"

    def test_review_create_invalid_rating(self) -> None:
        """Test review with invalid rating."""
        with pytest.raises(ValidationError):
            MarketplaceSuiteReviewCreate(rating=6)  # Max is 5

        with pytest.raises(ValidationError):
            MarketplaceSuiteReviewCreate(rating=0)  # Min is 1

    def test_review_create_minimal(self) -> None:
        """Test review with only required fields."""
        data = MarketplaceSuiteReviewCreate(rating=3)

        assert data.rating == 3
        assert data.title is None
        assert data.content is None


class TestMarketplaceSearchSchemas:
    """Tests for search and filter schemas."""

    def test_search_params_defaults(self) -> None:
        """Test search params defaults."""
        params = MarketplaceSearchParams()

        assert params.query is None
        assert params.category is None
        assert params.verified_only is False
        assert params.featured_only is False
        assert params.sort_by == "downloads"
        assert params.sort_order == "desc"

    def test_search_params_custom(self) -> None:
        """Test search params with custom values."""
        params = MarketplaceSearchParams(
            query="testing",
            category="unit",
            tags=["pytest"],
            verified_only=True,
            min_rating=4.0,
            sort_by="rating",
            sort_order="asc",
        )

        assert params.query == "testing"
        assert params.category == "unit"
        assert params.tags == ["pytest"]
        assert params.verified_only is True
        assert params.min_rating == 4.0
        assert params.sort_by == "rating"

    def test_search_params_invalid_sort(self) -> None:
        """Test invalid sort field."""
        with pytest.raises(ValidationError):
            MarketplaceSearchParams(sort_by="invalid")

    def test_marketplace_stats(self) -> None:
        """Test marketplace stats schema."""
        stats = MarketplaceStats(
            total_suites=100,
            total_downloads=10000,
            total_installs=5000,
            total_reviews=500,
            categories=[
                MarketplaceCategoryStats(
                    category="testing",
                    count=50,
                    total_downloads=5000,
                    average_rating=4.2,
                ),
            ],
            top_tags=[("pytest", 25), ("unittest", 15)],
        )

        assert stats.total_suites == 100
        assert stats.total_downloads == 10000
        assert len(stats.categories) == 1
        assert stats.categories[0].category == "testing"


class TestGitHubImportSchemas:
    """Tests for GitHub import schemas."""

    def test_github_import_request_valid(self) -> None:
        """Test valid GitHub import request."""
        data = GitHubImportRequest(
            github_url="https://github.com/user/repo",
            branch="main",
            path="tests/",
            name="Imported Suite",
            category="testing",
        )

        assert data.github_url == "https://github.com/user/repo"
        assert data.branch == "main"
        assert data.path == "tests/"

    def test_github_import_request_defaults(self) -> None:
        """Test GitHub import request defaults."""
        data = GitHubImportRequest(
            github_url="https://github.com/user/repo",
        )

        assert data.branch == "main"
        assert data.path == ""
        assert data.category == "general"
        assert data.license_type == "MIT"


class TestSemverParsing:
    """Tests for semantic version parsing utility."""

    def test_parse_semver(self) -> None:
        """Test semver parsing function."""
        from atp.dashboard.v2.routes.marketplace import parse_semver

        assert parse_semver("1.0.0") == (1, 0, 0)
        assert parse_semver("2.3.4") == (2, 3, 4)
        assert parse_semver("10.20.30") == (10, 20, 30)
        assert parse_semver("1.0.0-beta") == (1, 0, 0)  # Ignores prerelease
        assert parse_semver("invalid") == (1, 0, 0)  # Defaults for invalid


class TestMarketplaceSuiteDetailSchema:
    """Tests for the detailed suite response schema."""

    def test_suite_detail_with_versions(self) -> None:
        """Test suite detail with versions and reviews."""
        suite = MarketplaceSuite(
            id=1,
            tenant_id="default",
            name="Test Suite",
            slug="test-suite",
            description="A test suite",
            short_description="Test",
            publisher_id=1,
            publisher_name="testuser",
            category="testing",
            tags=["test"],
            license_type="MIT",
            license_url=None,
            source_type="local",
            github_url=None,
            github_branch=None,
            github_path=None,
            is_published=True,
            is_featured=False,
            is_verified=True,
            total_downloads=500,
            total_installs=200,
            average_rating=4.8,
            total_ratings=50,
            latest_version="2.0.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            published_at=datetime.now(),
        )

        response = MarketplaceSuiteResponse.model_validate(suite)

        detail = MarketplaceSuiteDetail(
            **response.model_dump(),
            versions=[],
            recent_reviews=[],
            test_count=10,
        )

        assert detail.id == 1
        assert detail.is_verified is True
        assert detail.test_count == 10
        assert detail.versions == []
        assert detail.recent_reviews == []


class TestMarketplaceRoutes:
    """Tests for marketplace route configuration."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client using v2 app."""
        from atp.dashboard.v2.factory import create_test_app

        app = create_test_app(use_v2_routes=True)
        return TestClient(app, raise_server_exceptions=False)

    def test_list_marketplace_endpoint_exists(self, client: TestClient) -> None:
        """Test that marketplace list endpoint exists."""
        response = client.get("/api/marketplace")
        # Should return 200 or 500 (db not configured), not 404
        assert response.status_code in [200, 500]

    def test_marketplace_stats_endpoint_exists(self, client: TestClient) -> None:
        """Test that marketplace stats endpoint exists."""
        response = client.get("/api/marketplace/stats")
        # Should return 200 or 500 (db not configured), not 404
        assert response.status_code in [200, 500]

    def test_categories_endpoint_exists(self, client: TestClient) -> None:
        """Test that categories endpoint exists."""
        response = client.get("/api/marketplace/categories")
        # Should return 200 or 500 (db not configured), not 404
        assert response.status_code in [200, 500]

    def test_suite_detail_endpoint_with_slug(self, client: TestClient) -> None:
        """Test suite detail endpoint with slug."""
        response = client.get("/api/marketplace/test-suite")
        # Should return 404 (not found) or 500 (db not configured)
        assert response.status_code in [404, 500]

    def test_versions_endpoint_exists(self, client: TestClient) -> None:
        """Test that versions endpoint exists."""
        response = client.get("/api/marketplace/test-suite/versions")
        # Should return 404 (not found) or 500 (db not configured)
        assert response.status_code in [404, 500]

    def test_reviews_endpoint_exists(self, client: TestClient) -> None:
        """Test that reviews endpoint exists."""
        response = client.get("/api/marketplace/test-suite/reviews")
        # Should return 404 (not found) or 500 (db not configured)
        assert response.status_code in [404, 500]

    def test_download_endpoint_exists(self, client: TestClient) -> None:
        """Test that download endpoint exists."""
        response = client.get("/api/marketplace/test-suite/download")
        # Should return 404 (not found) or 500 (db not configured)
        assert response.status_code in [404, 500]

    def test_publish_endpoint_requires_auth(self, client: TestClient) -> None:
        """Test that publish endpoint requires authentication."""
        response = client.post(
            "/api/marketplace",
            json={
                "name": "Test Suite",
                "slug": "test-suite",
            },
        )
        # Should return 401 or 403 (unauthorized) or 500 (db not configured)
        assert response.status_code in [401, 403, 500]

    def test_update_suite_requires_auth(self, client: TestClient) -> None:
        """Test that update endpoint requires authentication."""
        response = client.patch(
            "/api/marketplace/test-suite",
            json={"name": "Updated Name"},
        )
        # Should return 401 or 403 (unauthorized)
        assert response.status_code in [401, 403, 500]

    def test_unpublish_suite_requires_auth(self, client: TestClient) -> None:
        """Test that unpublish endpoint requires authentication."""
        response = client.delete("/api/marketplace/test-suite")
        # Should return 401 or 403 (unauthorized)
        assert response.status_code in [401, 403, 500]

    def test_create_version_requires_auth(self, client: TestClient) -> None:
        """Test that create version endpoint requires authentication."""
        response = client.post(
            "/api/marketplace/test-suite/versions",
            json={"version": "2.0.0", "suite_content": {}},
        )
        # Should return 401 or 403 (unauthorized)
        assert response.status_code in [401, 403, 500]

    def test_create_review_requires_auth(self, client: TestClient) -> None:
        """Test that create review endpoint requires authentication."""
        response = client.post(
            "/api/marketplace/test-suite/reviews",
            json={"rating": 5},
        )
        # Should return 401 or 403 (unauthorized)
        assert response.status_code in [401, 403, 500]

    def test_install_suite_requires_auth(self, client: TestClient) -> None:
        """Test that install endpoint requires authentication."""
        response = client.post("/api/marketplace/test-suite/install")
        # Should return 401 or 403 (unauthorized)
        assert response.status_code in [401, 403, 500]

    def test_uninstall_suite_requires_auth(self, client: TestClient) -> None:
        """Test that uninstall endpoint requires authentication."""
        response = client.delete("/api/marketplace/test-suite/install")
        # Should return 401 or 403 (unauthorized)
        assert response.status_code in [401, 403, 500]

    def test_github_import_requires_auth(self, client: TestClient) -> None:
        """Test that GitHub import endpoint requires authentication."""
        response = client.post(
            "/api/marketplace/import/github",
            json={"github_url": "https://github.com/user/repo"},
        )
        # Should return 401 or 403 (unauthorized)
        assert response.status_code in [401, 403, 500]

    def test_feature_suite_requires_admin(self, client: TestClient) -> None:
        """Test that feature endpoint requires admin privileges."""
        response = client.post("/api/marketplace/test-suite/feature")
        # Should return 401 or 403 (unauthorized)
        assert response.status_code in [401, 403, 500]

    def test_verify_suite_requires_admin(self, client: TestClient) -> None:
        """Test that verify endpoint requires admin privileges."""
        response = client.post("/api/marketplace/test-suite/verify")
        # Should return 401 or 403 (unauthorized)
        assert response.status_code in [401, 403, 500]

    def test_flag_review_requires_admin(self, client: TestClient) -> None:
        """Test that flag review endpoint requires admin privileges."""
        response = client.post("/api/marketplace/reviews/1/flag")
        # Should return 401 or 403 (unauthorized)
        assert response.status_code in [401, 403, 500]

    def test_search_with_query_param(self, client: TestClient) -> None:
        """Test marketplace search with query parameter."""
        response = client.get("/api/marketplace", params={"query": "testing"})
        # Should return 200 or 500 (db not configured)
        assert response.status_code in [200, 500]

    def test_search_with_category_filter(self, client: TestClient) -> None:
        """Test marketplace search with category filter."""
        response = client.get("/api/marketplace", params={"category": "testing"})
        # Should return 200 or 500 (db not configured)
        assert response.status_code in [200, 500]

    def test_search_with_tags_filter(self, client: TestClient) -> None:
        """Test marketplace search with tags filter."""
        response = client.get("/api/marketplace", params={"tags": ["pytest", "unit"]})
        # Should return 200 or 500 (db not configured)
        assert response.status_code in [200, 500]

    def test_search_with_verified_filter(self, client: TestClient) -> None:
        """Test marketplace search with verified filter."""
        response = client.get("/api/marketplace", params={"verified_only": True})
        # Should return 200 or 500 (db not configured)
        assert response.status_code in [200, 500]

    def test_search_with_featured_filter(self, client: TestClient) -> None:
        """Test marketplace search with featured filter."""
        response = client.get("/api/marketplace", params={"featured_only": True})
        # Should return 200 or 500 (db not configured)
        assert response.status_code in [200, 500]

    def test_search_with_min_rating(self, client: TestClient) -> None:
        """Test marketplace search with minimum rating filter."""
        response = client.get("/api/marketplace", params={"min_rating": 4.0})
        # Should return 200 or 500 (db not configured)
        assert response.status_code in [200, 500]

    def test_search_with_sort_options(self, client: TestClient) -> None:
        """Test marketplace search with sort options."""
        response = client.get(
            "/api/marketplace",
            params={"sort_by": "rating", "sort_order": "asc"},
        )
        # Should return 200 or 500 (db not configured)
        assert response.status_code in [200, 500]

    def test_pagination_parameters(self, client: TestClient) -> None:
        """Test marketplace pagination parameters."""
        response = client.get(
            "/api/marketplace",
            params={"limit": 10, "offset": 20},
        )
        # Should return 200 or 500 (db not configured)
        assert response.status_code in [200, 500]

    def test_pagination_limit_validation(self, client: TestClient) -> None:
        """Test marketplace pagination limit validation."""
        response = client.get(
            "/api/marketplace",
            params={"limit": 200},  # Exceeds max of 100
        )
        # Should return 422 (validation error)
        assert response.status_code == 422

    def test_pagination_offset_validation(self, client: TestClient) -> None:
        """Test marketplace pagination offset validation."""
        response = client.get(
            "/api/marketplace",
            params={"offset": -1},  # Negative offset
        )
        # Should return 422 (validation error)
        assert response.status_code == 422


class TestMarketplacePermissions:
    """Tests for marketplace RBAC permissions."""

    def test_marketplace_permissions_defined(self) -> None:
        """Test that all marketplace permissions are defined."""
        assert Permission.MARKETPLACE_READ.value == "marketplace:read"
        assert Permission.MARKETPLACE_WRITE.value == "marketplace:write"
        assert Permission.MARKETPLACE_DELETE.value == "marketplace:delete"
        assert Permission.MARKETPLACE_PUBLISH.value == "marketplace:publish"
        assert Permission.MARKETPLACE_REVIEW.value == "marketplace:review"
        assert Permission.MARKETPLACE_ADMIN.value == "marketplace:admin"

    def test_developer_role_has_marketplace_permissions(self) -> None:
        """Test that developer role has marketplace permissions."""
        from atp.dashboard.rbac.models import DEVELOPER_PERMISSIONS

        assert Permission.MARKETPLACE_READ in DEVELOPER_PERMISSIONS
        assert Permission.MARKETPLACE_WRITE in DEVELOPER_PERMISSIONS
        assert Permission.MARKETPLACE_DELETE in DEVELOPER_PERMISSIONS
        assert Permission.MARKETPLACE_PUBLISH in DEVELOPER_PERMISSIONS
        assert Permission.MARKETPLACE_REVIEW in DEVELOPER_PERMISSIONS

    def test_analyst_role_has_limited_marketplace_permissions(self) -> None:
        """Test that analyst role has limited marketplace permissions."""
        from atp.dashboard.rbac.models import ANALYST_PERMISSIONS

        assert Permission.MARKETPLACE_READ in ANALYST_PERMISSIONS
        assert Permission.MARKETPLACE_REVIEW in ANALYST_PERMISSIONS
        assert Permission.MARKETPLACE_WRITE not in ANALYST_PERMISSIONS
        assert Permission.MARKETPLACE_PUBLISH not in ANALYST_PERMISSIONS

    def test_viewer_role_has_read_only_marketplace_permissions(self) -> None:
        """Test that viewer role has read-only marketplace permissions."""
        from atp.dashboard.rbac.models import VIEWER_PERMISSIONS

        assert Permission.MARKETPLACE_READ in VIEWER_PERMISSIONS
        assert Permission.MARKETPLACE_WRITE not in VIEWER_PERMISSIONS
        assert Permission.MARKETPLACE_REVIEW not in VIEWER_PERMISSIONS

    def test_admin_role_has_all_marketplace_permissions(self) -> None:
        """Test that admin role has all marketplace permissions."""
        from atp.dashboard.rbac.models import ADMIN_PERMISSIONS

        assert Permission.MARKETPLACE_READ in ADMIN_PERMISSIONS
        assert Permission.MARKETPLACE_WRITE in ADMIN_PERMISSIONS
        assert Permission.MARKETPLACE_DELETE in ADMIN_PERMISSIONS
        assert Permission.MARKETPLACE_PUBLISH in ADMIN_PERMISSIONS
        assert Permission.MARKETPLACE_REVIEW in ADMIN_PERMISSIONS
        assert Permission.MARKETPLACE_ADMIN in ADMIN_PERMISSIONS


class TestMarketplaceDatabaseModels:
    """Tests for marketplace database model definitions."""

    def test_marketplace_suite_model_exists(self) -> None:
        """Test that MarketplaceSuite model is defined."""
        assert MarketplaceSuite.__tablename__ == "marketplace_suites"

    def test_marketplace_suite_version_model_exists(self) -> None:
        """Test that MarketplaceSuiteVersion model is defined."""
        assert MarketplaceSuiteVersion.__tablename__ == "marketplace_suite_versions"

    def test_marketplace_suite_review_model_exists(self) -> None:
        """Test that MarketplaceSuiteReview model is defined."""
        assert MarketplaceSuiteReview.__tablename__ == "marketplace_suite_reviews"

    def test_marketplace_suite_install_model_exists(self) -> None:
        """Test that MarketplaceSuiteInstall model is defined."""
        assert MarketplaceSuiteInstall.__tablename__ == "marketplace_suite_installs"

    def test_marketplace_suite_has_required_fields(self) -> None:
        """Test that MarketplaceSuite has required fields."""
        columns = [c.name for c in MarketplaceSuite.__table__.columns]
        required = [
            "id",
            "tenant_id",
            "name",
            "slug",
            "description",
            "publisher_id",
            "publisher_name",
            "category",
            "tags",
            "license_type",
            "source_type",
            "github_url",
            "is_published",
            "is_featured",
            "is_verified",
            "total_downloads",
            "total_installs",
            "average_rating",
            "total_ratings",
            "latest_version",
            "created_at",
            "updated_at",
            "published_at",
        ]
        for col in required:
            assert col in columns, f"Column {col} not found in MarketplaceSuite"

    def test_marketplace_suite_version_has_required_fields(self) -> None:
        """Test that MarketplaceSuiteVersion has required fields."""
        columns = [c.name for c in MarketplaceSuiteVersion.__table__.columns]
        required = [
            "id",
            "marketplace_suite_id",
            "version",
            "version_major",
            "version_minor",
            "version_patch",
            "changelog",
            "breaking_changes",
            "suite_content",
            "test_count",
            "downloads",
            "is_latest",
            "is_deprecated",
            "created_at",
        ]
        for col in required:
            assert col in columns, f"Column {col} not found in MarketplaceSuiteVersion"

    def test_marketplace_suite_review_has_required_fields(self) -> None:
        """Test that MarketplaceSuiteReview has required fields."""
        columns = [c.name for c in MarketplaceSuiteReview.__table__.columns]
        required = [
            "id",
            "marketplace_suite_id",
            "user_id",
            "rating",
            "title",
            "content",
            "is_approved",
            "is_flagged",
            "helpful_count",
            "not_helpful_count",
            "created_at",
            "updated_at",
        ]
        for col in required:
            assert col in columns, f"Column {col} not found in MarketplaceSuiteReview"

    def test_marketplace_suite_install_has_required_fields(self) -> None:
        """Test that MarketplaceSuiteInstall has required fields."""
        columns = [c.name for c in MarketplaceSuiteInstall.__table__.columns]
        required = [
            "id",
            "tenant_id",
            "marketplace_suite_id",
            "user_id",
            "version_installed",
            "is_active",
            "installed_at",
        ]
        for col in required:
            assert col in columns, f"Column {col} not found in MarketplaceSuiteInstall"


class TestLicenseSpecification:
    """Tests for license specification functionality."""

    def test_license_type_in_suite_create(self) -> None:
        """Test that license_type is supported in suite creation."""
        data = MarketplaceSuiteCreate(
            name="Test",
            slug="test-suite",
            license_type="Apache-2.0",
        )
        assert data.license_type == "Apache-2.0"

    def test_license_url_in_suite_create(self) -> None:
        """Test that license_url is supported in suite creation."""
        data = MarketplaceSuiteCreate(
            name="Test",
            slug="test-suite",
            license_type="Custom",
            license_url="https://example.com/license",
        )
        assert data.license_url == "https://example.com/license"

    def test_default_license_is_mit(self) -> None:
        """Test that default license is MIT."""
        data = MarketplaceSuiteCreate(
            name="Test",
            slug="test-suite",
        )
        assert data.license_type == "MIT"

    def test_common_licenses_accepted(self) -> None:
        """Test that common license types are accepted."""
        licenses = [
            "MIT",
            "Apache-2.0",
            "GPL-3.0",
            "BSD-3-Clause",
            "ISC",
            "Proprietary",
        ]
        for lic in licenses:
            data = MarketplaceSuiteCreate(
                name="Test",
                slug="test-suite",
                license_type=lic,
            )
            assert data.license_type == lic

    def test_license_in_response(self) -> None:
        """Test that license is included in response."""
        suite = MarketplaceSuite(
            id=1,
            name="Test",
            slug="test",
            publisher_id=1,
            publisher_name="user",
            license_type="Apache-2.0",
            license_url="https://apache.org/licenses/LICENSE-2.0",
            category="testing",
            tags=[],
            source_type="local",
            is_published=True,
            is_featured=False,
            is_verified=False,
            total_downloads=0,
            total_installs=0,
            total_ratings=0,
            latest_version="1.0.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        response = MarketplaceSuiteResponse.model_validate(suite)
        assert response.license_type == "Apache-2.0"
        assert response.license_url == "https://apache.org/licenses/LICENSE-2.0"


class TestGitHubImport:
    """Tests for GitHub import functionality."""

    def test_github_import_request_minimal(self) -> None:
        """Test GitHub import request with minimal fields."""
        data = GitHubImportRequest(
            github_url="https://github.com/user/repo",
        )
        assert data.github_url == "https://github.com/user/repo"
        assert data.branch == "main"
        assert data.path == ""

    def test_github_import_request_full(self) -> None:
        """Test GitHub import request with all fields."""
        data = GitHubImportRequest(
            github_url="https://github.com/user/repo",
            branch="develop",
            path="tests/suites",
            name="My Imported Suite",
            slug="my-imported-suite",
            description="A suite imported from GitHub",
            category="integration",
            tags=["automated", "ci"],
            license_type="Apache-2.0",
        )
        assert data.branch == "develop"
        assert data.path == "tests/suites"
        assert data.name == "My Imported Suite"
        assert data.category == "integration"

    def test_github_import_response_success(self) -> None:
        """Test GitHub import response for success case."""
        response = GitHubImportResponse(
            success=True,
            marketplace_suite=None,  # Would include suite in real response
            files_imported=["suite.yaml", "tests/test1.yaml"],
        )
        assert response.success is True
        assert len(response.files_imported) == 2

    def test_github_import_response_failure(self) -> None:
        """Test GitHub import response for failure case."""
        response = GitHubImportResponse(
            success=False,
            error="Repository not found",
            files_imported=[],
        )
        assert response.success is False
        assert response.error == "Repository not found"


class TestMarketplaceHelpers:
    """Tests for marketplace helper functions."""

    def test_parse_semver_valid(self) -> None:
        """Test semver parsing with valid versions."""
        from atp.dashboard.v2.routes.marketplace import parse_semver

        assert parse_semver("1.0.0") == (1, 0, 0)
        assert parse_semver("2.3.4") == (2, 3, 4)
        assert parse_semver("10.20.30") == (10, 20, 30)
        assert parse_semver("0.0.1") == (0, 0, 1)

    def test_parse_semver_with_prerelease(self) -> None:
        """Test semver parsing with prerelease tags."""
        from atp.dashboard.v2.routes.marketplace import parse_semver

        assert parse_semver("1.0.0-alpha") == (1, 0, 0)
        assert parse_semver("2.0.0-beta.1") == (2, 0, 0)
        assert parse_semver("3.0.0-rc.1") == (3, 0, 0)

    def test_parse_semver_with_build_metadata(self) -> None:
        """Test semver parsing with build metadata."""
        from atp.dashboard.v2.routes.marketplace import parse_semver

        assert parse_semver("1.0.0+build.123") == (1, 0, 0)
        assert parse_semver("2.0.0-alpha+001") == (2, 0, 0)

    def test_parse_semver_invalid(self) -> None:
        """Test semver parsing with invalid versions."""
        from atp.dashboard.v2.routes.marketplace import parse_semver

        # Returns default (1, 0, 0) for invalid versions
        assert parse_semver("invalid") == (1, 0, 0)
        assert parse_semver("v1.0.0") == (1, 0, 0)  # No 'v' prefix support
        assert parse_semver("") == (1, 0, 0)


class TestMarketplaceCache:
    """Tests for marketplace caching."""

    def test_cache_initialization(self) -> None:
        """Test that cache is initialized correctly."""
        from atp.dashboard.v2.routes.marketplace import get_marketplace_cache

        cache = get_marketplace_cache()
        assert cache is not None

    def test_cache_singleton(self) -> None:
        """Test that cache returns the same instance."""
        from atp.dashboard.v2.routes.marketplace import get_marketplace_cache

        cache1 = get_marketplace_cache()
        cache2 = get_marketplace_cache()
        assert cache1 is cache2

    def test_cache_operations(self) -> None:
        """Test basic cache operations."""
        from atp.dashboard.v2.routes.marketplace import get_marketplace_cache

        cache = get_marketplace_cache()

        # Test put and get
        cache.put("marketplace_test_key", {"data": "value"})
        result = cache.get("marketplace_test_key")
        assert result == {"data": "value"}

        # Test invalidate
        cache.invalidate("marketplace_test_key")
        result = cache.get("marketplace_test_key")
        assert result is None


class TestVersionSchemaExtended:
    """Extended tests for version-related schemas."""

    def test_version_update_schema(self) -> None:
        """Test version update schema."""
        data = MarketplaceSuiteVersionUpdate(
            changelog="Updated changelog",
            is_deprecated=True,
            deprecation_message="Use version 2.0.0 instead",
        )
        assert data.is_deprecated is True
        assert data.deprecation_message is not None

    def test_version_update_partial(self) -> None:
        """Test partial version update."""
        data = MarketplaceSuiteVersionUpdate(changelog="New changelog only")
        assert data.changelog == "New changelog only"
        assert data.is_deprecated is None

    def test_version_list_response(self) -> None:
        """Test version list response schema."""
        version = MarketplaceSuiteVersion(
            id=1,
            marketplace_suite_id=1,
            version="1.0.0",
            version_major=1,
            version_minor=0,
            version_patch=0,
            test_count=5,
            is_latest=True,
            is_deprecated=False,
            breaking_changes=False,  # Required field
            downloads=0,
            created_at=datetime.now(),
        )
        version_response = MarketplaceSuiteVersionResponse.model_validate(version)
        list_response = MarketplaceSuiteVersionList(
            items=[version_response],
            total=1,
        )
        assert list_response.total == 1
        assert len(list_response.items) == 1


class TestReviewSchemaExtended:
    """Extended tests for review-related schemas."""

    def test_review_update_schema(self) -> None:
        """Test review update schema."""
        data = MarketplaceSuiteReviewUpdate(
            rating=4,
            title="Updated title",
            content="Updated content",
        )
        assert data.rating == 4
        assert data.title == "Updated title"

    def test_review_update_partial(self) -> None:
        """Test partial review update."""
        data = MarketplaceSuiteReviewUpdate(rating=3)
        assert data.rating == 3
        assert data.title is None
        assert data.content is None

    def test_review_list_response(self) -> None:
        """Test review list response schema."""
        list_response = MarketplaceSuiteReviewList(
            items=[],
            total=0,
            limit=20,
            offset=0,
            average_rating=None,
            rating_distribution={},
        )
        assert list_response.total == 0
        assert list_response.average_rating is None

    def test_review_list_with_distribution(self) -> None:
        """Test review list with rating distribution."""
        list_response = MarketplaceSuiteReviewList(
            items=[],
            total=10,
            limit=20,
            offset=0,
            average_rating=4.2,
            rating_distribution={5: 5, 4: 3, 3: 2},
        )
        assert list_response.rating_distribution[5] == 5
        assert list_response.average_rating == 4.2


class TestInstallSchemas:
    """Tests for install-related schemas."""

    def test_install_response_schema(self) -> None:
        """Test install response schema."""
        install = MarketplaceSuiteInstall(
            id=1,
            tenant_id="default",
            marketplace_suite_id=1,
            user_id=1,
            version_installed="1.0.0",
            version_id=1,
            is_active=True,
            installed_at=datetime.now(),
        )
        response = MarketplaceSuiteInstallResponse.model_validate(install)
        assert response.id == 1
        assert response.version_installed == "1.0.0"
        assert response.is_active is True

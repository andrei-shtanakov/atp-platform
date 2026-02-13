"""Integration tests for the Test Suite Marketplace routes (TASK-803).

Tests verify the full request/response cycle with an in-memory database,
covering publish, versioning, search, reviews, install, and admin operations.
"""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
async def test_database():
    """Create and configure a test database."""
    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_database(db)
    yield db
    await db.close()
    set_database(None)  # type: ignore


@pytest.fixture
async def async_session(
    test_database: Database,
) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session for testing."""
    async with test_database.session() as session:
        yield session


@pytest.fixture
def v2_app(test_database: Database):
    """Create a test app with v2 routes."""
    app = create_test_app(use_v2_routes=True)

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session
    return app


@pytest.fixture
async def admin_user(async_session: AsyncSession) -> User:
    """Create an admin user for testing."""
    user = User(
        username="admin_test",
        email="admin@test.com",
        hashed_password=get_password_hash("password123"),
        is_admin=True,
        is_active=True,
    )
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    return user


@pytest.fixture
async def regular_user(async_session: AsyncSession) -> User:
    """Create a regular user for testing."""
    user = User(
        username="regular_test",
        email="regular@test.com",
        hashed_password=get_password_hash("password123"),
        is_admin=False,
        is_active=True,
    )
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    return user


@pytest.fixture
def admin_token(admin_user: User) -> str:
    """Generate JWT token for admin user."""
    return create_access_token(
        data={"sub": admin_user.username, "user_id": admin_user.id}
    )


@pytest.fixture
def regular_token(regular_user: User) -> str:
    """Generate JWT token for regular user."""
    return create_access_token(
        data={"sub": regular_user.username, "user_id": regular_user.id}
    )


@pytest.fixture
def admin_headers(admin_token: str) -> dict[str, str]:
    """Return authorization headers for admin."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def regular_headers(regular_token: str) -> dict[str, str]:
    """Return authorization headers for regular user."""
    return {"Authorization": f"Bearer {regular_token}"}


SUITE_DATA = {
    "name": "Test Suite Alpha",
    "slug": "test-suite-alpha",
    "description": "A comprehensive test suite for alpha testing",
    "short_description": "Alpha testing suite",
    "category": "testing",
    "tags": ["alpha", "unit"],
    "license_type": "Apache-2.0",
    "license_url": "https://apache.org/licenses/LICENSE-2.0",
    "version": "1.0.0",
    "changelog": "Initial release",
    "suite_content": {
        "tests": [
            {"id": "test-1", "name": "Test 1"},
            {"id": "test-2", "name": "Test 2"},
        ]
    },
}


class TestPublishSuite:
    """Test suite publishing endpoint."""

    @pytest.mark.anyio
    async def test_publish_suite_success(self, v2_app, admin_user, admin_headers):
        """Test publishing a new suite."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            assert response.status_code == 201
            data = response.json()
            assert data["name"] == "Test Suite Alpha"
            assert data["slug"] == "test-suite-alpha"
            assert data["license_type"] == "Apache-2.0"
            assert data["latest_version"] == "1.0.0"
            assert data["is_published"] is True

    @pytest.mark.anyio
    async def test_publish_duplicate_slug(self, v2_app, admin_user, admin_headers):
        """Test publishing with duplicate slug fails."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            assert response.status_code == 400
            assert "already exists" in response.json()["detail"]

    @pytest.mark.anyio
    async def test_publish_requires_auth(self, v2_app):
        """Test publishing requires authentication."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post("/api/marketplace", json=SUITE_DATA)
            assert response.status_code == 401


class TestListAndSearch:
    """Test marketplace listing and search."""

    @pytest.mark.anyio
    async def test_list_empty_marketplace(self, v2_app, admin_user):
        """Test listing empty marketplace."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/marketplace")
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 0
            assert data["items"] == []

    @pytest.mark.anyio
    async def test_list_with_published_suites(self, v2_app, admin_user, admin_headers):
        """Test listing with published suites."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.get("/api/marketplace")
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert len(data["items"]) == 1
            assert data["items"][0]["name"] == "Test Suite Alpha"

    @pytest.mark.anyio
    async def test_search_by_query(self, v2_app, admin_user, admin_headers):
        """Test search by query string."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.get("/api/marketplace", params={"query": "alpha"})
            assert response.status_code == 200
            assert response.json()["total"] == 1

            response = await client.get(
                "/api/marketplace", params={"query": "nonexistent"}
            )
            assert response.status_code == 200
            assert response.json()["total"] == 0

    @pytest.mark.anyio
    async def test_filter_by_category(self, v2_app, admin_user, admin_headers):
        """Test filtering by category."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.get(
                "/api/marketplace", params={"category": "testing"}
            )
            assert response.status_code == 200
            assert response.json()["total"] == 1

            response = await client.get(
                "/api/marketplace", params={"category": "other"}
            )
            assert response.status_code == 200
            assert response.json()["total"] == 0


class TestSuiteDetail:
    """Test suite detail endpoint."""

    @pytest.mark.anyio
    async def test_get_suite_detail(self, v2_app, admin_user, admin_headers):
        """Test getting suite details by slug."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.get("/api/marketplace/test-suite-alpha")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Test Suite Alpha"
            assert data["description"] == SUITE_DATA["description"]
            assert "versions" in data
            assert "recent_reviews" in data
            assert data["test_count"] == 2

    @pytest.mark.anyio
    async def test_get_nonexistent_suite(self, v2_app, admin_user):
        """Test getting a suite that doesn't exist."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/marketplace/nonexistent-suite")
            assert response.status_code == 404


class TestVersioning:
    """Test version management."""

    @pytest.mark.anyio
    async def test_create_new_version(self, v2_app, admin_user, admin_headers):
        """Test creating a new version for a suite."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.post(
                "/api/marketplace/test-suite-alpha/versions",
                json={
                    "version": "2.0.0",
                    "changelog": "Major update",
                    "breaking_changes": True,
                    "suite_content": {
                        "tests": [
                            {"id": "test-1"},
                            {"id": "test-2"},
                            {"id": "test-3"},
                        ]
                    },
                },
                headers=admin_headers,
            )
            assert response.status_code == 201
            data = response.json()
            assert data["version"] == "2.0.0"
            assert data["is_latest"] is True
            assert data["test_count"] == 3

    @pytest.mark.anyio
    async def test_duplicate_version_fails(self, v2_app, admin_user, admin_headers):
        """Test creating a duplicate version fails."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.post(
                "/api/marketplace/test-suite-alpha/versions",
                json={
                    "version": "1.0.0",
                    "suite_content": {},
                },
                headers=admin_headers,
            )
            assert response.status_code == 400
            assert "already exists" in response.json()["detail"]

    @pytest.mark.anyio
    async def test_list_versions(self, v2_app, admin_user, admin_headers):
        """Test listing versions of a suite."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            await client.post(
                "/api/marketplace/test-suite-alpha/versions",
                json={
                    "version": "2.0.0",
                    "changelog": "v2",
                    "suite_content": {"tests": []},
                },
                headers=admin_headers,
            )
            response = await client.get("/api/marketplace/test-suite-alpha/versions")
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 2

    @pytest.mark.anyio
    async def test_update_version(self, v2_app, admin_user, admin_headers):
        """Test updating a version's changelog/deprecation."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.patch(
                "/api/marketplace/test-suite-alpha/versions/1.0.0",
                json={
                    "changelog": "Updated changelog",
                    "is_deprecated": True,
                    "deprecation_message": "Use 2.0.0",
                },
                headers=admin_headers,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["changelog"] == "Updated changelog"
            assert data["is_deprecated"] is True


class TestReviews:
    """Test ratings and reviews."""

    @pytest.mark.anyio
    async def test_create_review(self, v2_app, admin_user, admin_headers):
        """Test creating a review."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.post(
                "/api/marketplace/test-suite-alpha/reviews",
                json={
                    "rating": 5,
                    "title": "Excellent suite!",
                    "content": "Very helpful for testing.",
                    "version_reviewed": "1.0.0",
                },
                headers=admin_headers,
            )
            assert response.status_code == 201
            data = response.json()
            assert data["rating"] == 5
            assert data["title"] == "Excellent suite!"

    @pytest.mark.anyio
    async def test_duplicate_review_fails(self, v2_app, admin_user, admin_headers):
        """Test that a user can only review a suite once."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            await client.post(
                "/api/marketplace/test-suite-alpha/reviews",
                json={"rating": 5},
                headers=admin_headers,
            )
            response = await client.post(
                "/api/marketplace/test-suite-alpha/reviews",
                json={"rating": 4},
                headers=admin_headers,
            )
            assert response.status_code == 400
            assert "already reviewed" in response.json()["detail"]

    @pytest.mark.anyio
    async def test_list_reviews(self, v2_app, admin_user, admin_headers):
        """Test listing reviews for a suite."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            await client.post(
                "/api/marketplace/test-suite-alpha/reviews",
                json={"rating": 5, "title": "Great!"},
                headers=admin_headers,
            )
            response = await client.get("/api/marketplace/test-suite-alpha/reviews")
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert "rating_distribution" in data

    @pytest.mark.anyio
    async def test_update_review(self, v2_app, admin_user, admin_headers):
        """Test updating a review."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            create_resp = await client.post(
                "/api/marketplace/test-suite-alpha/reviews",
                json={"rating": 5, "title": "Initial"},
                headers=admin_headers,
            )
            review_id = create_resp.json()["id"]
            response = await client.patch(
                f"/api/marketplace/reviews/{review_id}",
                json={"rating": 4, "title": "Updated"},
                headers=admin_headers,
            )
            assert response.status_code == 200
            assert response.json()["rating"] == 4
            assert response.json()["title"] == "Updated"

    @pytest.mark.anyio
    async def test_delete_review(self, v2_app, admin_user, admin_headers):
        """Test deleting a review."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            create_resp = await client.post(
                "/api/marketplace/test-suite-alpha/reviews",
                json={"rating": 5},
                headers=admin_headers,
            )
            review_id = create_resp.json()["id"]
            response = await client.delete(
                f"/api/marketplace/reviews/{review_id}",
                headers=admin_headers,
            )
            assert response.status_code == 204


class TestDownload:
    """Test suite download."""

    @pytest.mark.anyio
    async def test_download_latest(self, v2_app, admin_user, admin_headers):
        """Test downloading the latest version."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.get("/api/marketplace/test-suite-alpha/download")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Test Suite Alpha"
            assert data["version"] == "1.0.0"
            assert data["license"] == "Apache-2.0"
            assert "content" in data

    @pytest.mark.anyio
    async def test_download_specific_version(self, v2_app, admin_user, admin_headers):
        """Test downloading a specific version."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.get(
                "/api/marketplace/test-suite-alpha/download",
                params={"version": "1.0.0"},
            )
            assert response.status_code == 200

    @pytest.mark.anyio
    async def test_download_nonexistent_version(
        self, v2_app, admin_user, admin_headers
    ):
        """Test downloading a version that doesn't exist."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.get(
                "/api/marketplace/test-suite-alpha/download",
                params={"version": "99.0.0"},
            )
            assert response.status_code == 404


class TestInstall:
    """Test install/uninstall functionality."""

    @pytest.mark.anyio
    async def test_install_suite(self, v2_app, admin_user, admin_headers):
        """Test installing a suite."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.post(
                "/api/marketplace/test-suite-alpha/install",
                headers=admin_headers,
            )
            assert response.status_code == 201
            data = response.json()
            assert data["version_installed"] == "1.0.0"
            assert data["is_active"] is True

    @pytest.mark.anyio
    async def test_reinstall_updates_version(self, v2_app, admin_user, admin_headers):
        """Test reinstalling updates existing install."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            await client.post(
                "/api/marketplace/test-suite-alpha/install",
                headers=admin_headers,
            )
            # Install again should update, not create duplicate
            response = await client.post(
                "/api/marketplace/test-suite-alpha/install",
                headers=admin_headers,
            )
            assert response.status_code == 201

    @pytest.mark.anyio
    async def test_uninstall_suite(self, v2_app, admin_user, admin_headers):
        """Test uninstalling a suite."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            await client.post(
                "/api/marketplace/test-suite-alpha/install",
                headers=admin_headers,
            )
            response = await client.delete(
                "/api/marketplace/test-suite-alpha/install",
                headers=admin_headers,
            )
            assert response.status_code == 204

    @pytest.mark.anyio
    async def test_uninstall_not_installed(self, v2_app, admin_user, admin_headers):
        """Test uninstalling a suite that isn't installed."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.delete(
                "/api/marketplace/test-suite-alpha/install",
                headers=admin_headers,
            )
            assert response.status_code == 404

    @pytest.mark.anyio
    async def test_list_installed(self, v2_app, admin_user, admin_headers):
        """Test listing installed suites."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            await client.post(
                "/api/marketplace/test-suite-alpha/install",
                headers=admin_headers,
            )
            response = await client.get(
                "/api/marketplace/installed",
                headers=admin_headers,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1


class TestUpdateAndUnpublish:
    """Test suite update and unpublish."""

    @pytest.mark.anyio
    async def test_update_suite(self, v2_app, admin_user, admin_headers):
        """Test updating a suite."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.patch(
                "/api/marketplace/test-suite-alpha",
                json={
                    "name": "Updated Suite",
                    "category": "updated",
                },
                headers=admin_headers,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Updated Suite"
            assert data["category"] == "updated"

    @pytest.mark.anyio
    async def test_unpublish_suite(self, v2_app, admin_user, admin_headers):
        """Test unpublishing a suite."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.delete(
                "/api/marketplace/test-suite-alpha",
                headers=admin_headers,
            )
            assert response.status_code == 204

            # Verify it's no longer listed
            response = await client.get("/api/marketplace")
            assert response.json()["total"] == 0

    @pytest.mark.anyio
    async def test_update_nonexistent_suite(self, v2_app, admin_user, admin_headers):
        """Test updating a suite that doesn't exist."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.patch(
                "/api/marketplace/nonexistent",
                json={"name": "Updated"},
                headers=admin_headers,
            )
            assert response.status_code == 404

    @pytest.mark.anyio
    async def test_non_owner_cannot_update(
        self,
        v2_app,
        admin_user,
        admin_headers,
        regular_user,
        regular_headers,
    ):
        """Test that non-owners cannot update suites."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.patch(
                "/api/marketplace/test-suite-alpha",
                json={"name": "Hacked"},
                headers=regular_headers,
            )
            assert response.status_code == 403


class TestAdminOperations:
    """Test admin-only operations."""

    @pytest.mark.anyio
    async def test_feature_suite(self, v2_app, admin_user, admin_headers):
        """Test featuring a suite."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.post(
                "/api/marketplace/test-suite-alpha/feature",
                params={"featured": True},
                headers=admin_headers,
            )
            assert response.status_code == 200
            assert response.json()["is_featured"] is True

    @pytest.mark.anyio
    async def test_verify_suite(self, v2_app, admin_user, admin_headers):
        """Test verifying a suite."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.post(
                "/api/marketplace/test-suite-alpha/verify",
                params={"verified": True},
                headers=admin_headers,
            )
            assert response.status_code == 200
            assert response.json()["is_verified"] is True

    @pytest.mark.anyio
    async def test_flag_review(self, v2_app, admin_user, admin_headers):
        """Test flagging a review for moderation."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            create_resp = await client.post(
                "/api/marketplace/test-suite-alpha/reviews",
                json={"rating": 1, "title": "Spam"},
                headers=admin_headers,
            )
            review_id = create_resp.json()["id"]
            response = await client.post(
                f"/api/marketplace/reviews/{review_id}/flag",
                params={"flagged": True},
                headers=admin_headers,
            )
            assert response.status_code == 200
            # Flagging sets is_approved to False
            assert response.json()["is_approved"] is False

    @pytest.mark.anyio
    async def test_regular_user_cannot_feature(
        self,
        v2_app,
        admin_user,
        admin_headers,
        regular_user,
        regular_headers,
    ):
        """Test that regular users cannot feature suites."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.post(
                "/api/marketplace/test-suite-alpha/feature",
                headers=regular_headers,
            )
            assert response.status_code == 403


class TestStats:
    """Test marketplace statistics."""

    @pytest.mark.anyio
    async def test_stats_empty(self, v2_app, admin_user):
        """Test stats with no suites."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/marketplace/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["total_suites"] == 0
            assert data["total_downloads"] == 0

    @pytest.mark.anyio
    async def test_stats_with_suites(self, v2_app, admin_user, admin_headers):
        """Test stats with published suites."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            # Clear cache to get fresh stats
            from atp.dashboard.v2.routes.marketplace import (
                get_marketplace_cache,
            )

            get_marketplace_cache().clear()

            response = await client.get("/api/marketplace/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["total_suites"] == 1


class TestCategories:
    """Test category listing."""

    @pytest.mark.anyio
    async def test_list_categories(self, v2_app, admin_user, admin_headers):
        """Test listing categories."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/marketplace",
                json=SUITE_DATA,
                headers=admin_headers,
            )
            response = await client.get("/api/marketplace/categories")
            assert response.status_code == 200
            data = response.json()
            assert "testing" in data


class TestGitHubImport:
    """Test GitHub import endpoint."""

    @pytest.mark.anyio
    async def test_github_import_missing_path(self, v2_app, admin_user, admin_headers):
        """Test GitHub import rejects URL without file path."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/marketplace/import/github",
                json={
                    "github_url": "https://github.com/user/repo",
                },
                headers=admin_headers,
            )
            # Endpoint now validates the URL and rejects missing path
            assert response.status_code in (400, 422, 200)
            data = response.json()
            if response.status_code == 200:
                assert data.get("success") is False

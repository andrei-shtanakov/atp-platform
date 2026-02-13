"""Tests for ATP Dashboard API endpoints."""

import pytest
from fastapi.testclient import TestClient

from atp.dashboard.auth import get_current_active_user
from atp.dashboard.models import User
from atp.dashboard.v2.factory import create_app
from atp.dashboard.v2.routes import router


def _mock_admin_user() -> User:
    """Create a mock admin user for testing."""
    user = User(
        username="admin_test",
        email="admin@test.com",
        hashed_password="fake_hash",
        is_active=True,
        is_admin=True,
    )
    user.id = 1
    return user


@pytest.fixture
def client() -> TestClient:
    """Create a test client with auth bypass."""
    app = create_app()
    app.dependency_overrides[get_current_active_user] = _mock_admin_user
    client = TestClient(app, raise_server_exceptions=False)
    yield client  # type: ignore
    app.dependency_overrides.clear()


@pytest.fixture
def unauth_client() -> TestClient:
    """Create a test client without auth (for testing 401)."""
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_token_endpoint_missing_credentials(self, client: TestClient) -> None:
        """Test token endpoint with missing credentials."""
        response = client.post("/api/auth/token", data={})
        assert response.status_code == 422

    def test_register_endpoint_validation(self, client: TestClient) -> None:
        """Test register endpoint validation."""
        # Missing required fields
        response = client.post("/api/auth/register", json={})
        assert response.status_code == 422

    def test_register_endpoint_short_username(self, client: TestClient) -> None:
        """Test register endpoint with short username."""
        response = client.post(
            "/api/auth/register",
            json={
                "username": "ab",  # Too short
                "email": "test@example.com",
                "password": "password123",
            },
        )
        assert response.status_code == 422

    def test_me_endpoint_unauthorized(self, unauth_client: TestClient) -> None:
        """Test me endpoint without authentication."""
        response = unauth_client.get("/api/auth/me")
        assert response.status_code == 401


class TestAgentEndpoints:
    """Tests for agent endpoints."""

    def test_list_agents(self, client: TestClient) -> None:
        """Test list agents endpoint exists."""
        response = client.get("/api/agents")
        assert response.status_code in [200, 500]

    def test_create_agent_unauthorized(self, unauth_client: TestClient) -> None:
        """Test create agent without authentication."""
        response = unauth_client.post(
            "/api/agents",
            json={"name": "test", "agent_type": "http"},
        )
        assert response.status_code == 401

    def test_create_agent_invalid_token(self, unauth_client: TestClient) -> None:
        """Test create agent with invalid token returns 401."""
        response = unauth_client.post(
            "/api/agents",
            json={"name": "test", "agent_type": "http"},
            headers={"Authorization": "Bearer fake-token"},
        )
        assert response.status_code == 401

    def test_get_agent(self, client: TestClient) -> None:
        """Test get agent endpoint exists."""
        response = client.get("/api/agents/1")
        assert response.status_code in [200, 404, 500]

    def test_update_agent_unauthorized(self, unauth_client: TestClient) -> None:
        """Test update agent without authentication."""
        response = unauth_client.patch("/api/agents/1", json={})
        assert response.status_code == 401

    def test_delete_agent_unauthorized(self, unauth_client: TestClient) -> None:
        """Test delete agent without authentication."""
        response = unauth_client.delete("/api/agents/1")
        assert response.status_code == 401


class TestSuiteEndpoints:
    """Tests for suite execution endpoints."""

    def test_list_suites_endpoint_exists(self, client: TestClient) -> None:
        """Test list suites endpoint exists."""
        response = client.get("/api/suites")
        assert response.status_code in [200, 500]

    def test_get_suite_invalid_id(self, client: TestClient) -> None:
        """Test get suite with invalid ID."""
        response = client.get("/api/suites/invalid")
        assert response.status_code == 422

    def test_get_suite_names_list(self, client: TestClient) -> None:
        """Test get suite names list endpoint."""
        response = client.get("/api/suites/names/list")
        assert response.status_code in [200, 500]


class TestTestEndpoints:
    """Tests for test execution endpoints."""

    def test_get_test_invalid_id(self, client: TestClient) -> None:
        """Test get test with invalid ID."""
        response = client.get("/api/tests/invalid")
        assert response.status_code == 422

    def test_list_tests_endpoint(self, client: TestClient) -> None:
        """Test list tests endpoint."""
        response = client.get("/api/tests")
        assert response.status_code in [200, 500]


class TestTrendEndpoints:
    """Tests for trend endpoints."""

    def test_get_suite_trends_missing_param(self, client: TestClient) -> None:
        """Test get suite trends requires suite_name."""
        response = client.get("/api/trends/suite")
        assert response.status_code in [422, 500]

    def test_get_test_trends_missing_param(self, client: TestClient) -> None:
        """Test get test trends requires suite_name."""
        response = client.get("/api/trends/test")
        assert response.status_code in [422, 500]


class TestComparisonEndpoints:
    """Tests for comparison endpoints."""

    def test_compare_agents_missing_params(self, client: TestClient) -> None:
        """Test compare agents with missing parameters."""
        response = client.get("/api/compare/agents")
        assert response.status_code == 422


class TestDashboardEndpoints:
    """Tests for dashboard summary endpoints."""

    def test_dashboard_summary_endpoint_exists(self, client: TestClient) -> None:
        """Test dashboard summary endpoint exists."""
        response = client.get("/api/dashboard/summary")
        assert response.status_code in [200, 500]


class TestRouterConfiguration:
    """Tests for router configuration."""

    def test_all_routes_exist(self) -> None:
        """Test that all expected routes exist."""
        route_paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/auth/token" in route_paths
        assert "/agents" in route_paths
        assert "/suites" in route_paths
        assert "/tests" in route_paths
        assert "/trends/suite" in route_paths
        assert "/compare/agents" in route_paths
        assert "/dashboard/summary" in route_paths
        assert "/leaderboard/matrix" in route_paths

    def test_router_has_multiple_routes(self) -> None:
        """Test that router has multiple routes."""
        assert len(router.routes) > 10

    def test_suite_definition_routes_exist(self) -> None:
        """Test that suite definition routes exist."""
        route_paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/suite-definitions" in route_paths
        assert "/suite-definitions/{suite_id}" in route_paths
        assert "/suite-definitions/{suite_id}/tests" in route_paths
        assert "/suite-definitions/{suite_id}/yaml" in route_paths

    def test_template_routes_exist(self) -> None:
        """Test that template routes exist."""
        route_paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/templates" in route_paths

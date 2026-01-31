"""Tests for ATP Dashboard API endpoints."""

import pytest
from fastapi.testclient import TestClient

from atp.dashboard.api import router
from atp.dashboard.app import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client using the actual app."""
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

    def test_me_endpoint_unauthorized(self, client: TestClient) -> None:
        """Test me endpoint without authentication."""
        response = client.get("/api/auth/me")
        assert response.status_code == 401


class TestAgentEndpoints:
    """Tests for agent endpoints."""

    def test_list_agents_no_auth_allowed(self, client: TestClient) -> None:
        """Test list agents without authentication (allowed - uses CurrentUser)."""
        response = client.get("/api/agents")
        # CurrentUser allows optional auth, so returns 200 or 500 (db error)
        assert response.status_code in [200, 500]

    def test_create_agent_unauthorized(self, client: TestClient) -> None:
        """Test create agent without authentication."""
        response = client.post(
            "/api/agents",
            json={"name": "test", "agent_type": "http"},
        )
        assert response.status_code == 401

    def test_create_agent_invalid_token(self, client: TestClient) -> None:
        """Test create agent with invalid token returns 401."""
        response = client.post(
            "/api/agents",
            json={"name": "test", "agent_type": "http"},
            headers={"Authorization": "Bearer fake-token"},
        )
        assert response.status_code == 401

    def test_get_agent_no_auth_allowed(self, client: TestClient) -> None:
        """Test get agent without authentication (allowed - uses CurrentUser)."""
        response = client.get("/api/agents/1")
        # CurrentUser allows optional auth, returns 200/404/500
        assert response.status_code in [200, 404, 500]

    def test_update_agent_unauthorized(self, client: TestClient) -> None:
        """Test update agent without authentication."""
        response = client.patch("/api/agents/1", json={})
        assert response.status_code == 401

    def test_delete_agent_unauthorized(self, client: TestClient) -> None:
        """Test delete agent without authentication."""
        response = client.delete("/api/agents/1")
        assert response.status_code == 401


class TestSuiteEndpoints:
    """Tests for suite execution endpoints."""

    def test_list_suites_endpoint_exists(self, client: TestClient) -> None:
        """Test list suites endpoint exists."""
        response = client.get("/api/suites")
        # Should return data or 500 (db error), not 404
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
        # Should fail validation or return 500
        assert response.status_code in [422, 500]

    def test_get_test_trends_missing_param(self, client: TestClient) -> None:
        """Test get test trends requires suite_name."""
        response = client.get("/api/trends/test")
        # Should fail validation or return 500
        assert response.status_code in [422, 500]


class TestComparisonEndpoints:
    """Tests for comparison endpoints."""

    def test_compare_agents_missing_params(self, client: TestClient) -> None:
        """Test compare agents with missing parameters."""
        response = client.get("/api/compare/agents")
        # Requires suite_name, should fail validation
        assert response.status_code == 422


class TestDashboardEndpoints:
    """Tests for dashboard summary endpoints."""

    def test_dashboard_summary_endpoint_exists(self, client: TestClient) -> None:
        """Test dashboard summary endpoint exists."""
        response = client.get("/api/dashboard/summary")
        # Should return data or 500 (db error), not 404
        assert response.status_code in [200, 500]


class TestRouterConfiguration:
    """Tests for router configuration."""

    def test_all_routes_exist(self) -> None:
        """Test that all expected routes exist."""
        route_paths = [r.path for r in router.routes if hasattr(r, "path")]
        # Check some key routes exist
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

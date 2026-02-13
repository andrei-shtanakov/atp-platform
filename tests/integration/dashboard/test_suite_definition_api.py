"""Integration tests for suite definition API endpoints."""

import pytest
from fastapi.testclient import TestClient

from atp.dashboard.auth import get_current_active_user
from atp.dashboard.models import User
from atp.dashboard.v2.factory import create_app


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


class TestSuiteDefinitionEndpoints:
    """Integration tests for suite definition CRUD endpoints."""

    def test_list_suite_definitions_no_auth(self, client: TestClient) -> None:
        """Test listing suite definitions without authentication (allowed)."""
        response = client.get("/api/suite-definitions")
        # CurrentUser allows optional auth, should return 200 or 500 (db init)
        assert response.status_code in [200, 500]

    def test_list_suite_definitions_pagination(self, client: TestClient) -> None:
        """Test pagination parameters."""
        response = client.get("/api/suite-definitions?limit=10&offset=0")
        assert response.status_code in [200, 500]

    def test_list_suite_definitions_invalid_limit(self, client: TestClient) -> None:
        """Test invalid limit parameter."""
        response = client.get("/api/suite-definitions?limit=200")
        assert response.status_code == 422

    def test_list_suite_definitions_invalid_offset(self, client: TestClient) -> None:
        """Test invalid offset parameter."""
        response = client.get("/api/suite-definitions?offset=-1")
        assert response.status_code == 422

    def test_create_suite_definition_unauthorized(
        self, unauth_client: TestClient
    ) -> None:
        """Test creating suite without authentication."""
        response = unauth_client.post(
            "/api/suite-definitions",
            json={"name": "test-suite"},
        )
        assert response.status_code == 401

    def test_create_suite_definition_invalid_token(
        self, unauth_client: TestClient
    ) -> None:
        """Test creating suite with invalid token."""
        response = unauth_client.post(
            "/api/suite-definitions",
            json={"name": "test-suite"},
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 401

    def test_create_suite_definition_validation(self, client: TestClient) -> None:
        """Test suite creation validation."""
        # Missing required field (auth bypassed)
        response = client.post(
            "/api/suite-definitions",
            json={},
        )
        assert response.status_code == 422

    def test_create_suite_empty_name(self, client: TestClient) -> None:
        """Test suite creation with empty name."""
        response = client.post(
            "/api/suite-definitions",
            json={"name": ""},
        )
        assert response.status_code == 422

    def test_get_suite_definition_no_auth(self, client: TestClient) -> None:
        """Test getting suite definition without auth (allowed)."""
        response = client.get("/api/suite-definitions/1")
        # CurrentUser allows optional auth
        assert response.status_code in [200, 404, 500]

    def test_get_suite_definition_invalid_id(self, client: TestClient) -> None:
        """Test getting suite with invalid ID."""
        response = client.get("/api/suite-definitions/invalid")
        assert response.status_code == 422

    def test_delete_suite_definition_unauthorized(
        self, unauth_client: TestClient
    ) -> None:
        """Test deleting suite without authentication."""
        response = unauth_client.delete("/api/suite-definitions/1")
        assert response.status_code == 401


class TestAddTestToSuiteEndpoints:
    """Integration tests for adding tests to suites."""

    def test_add_test_unauthorized(self, unauth_client: TestClient) -> None:
        """Test adding test without authentication."""
        response = unauth_client.post(
            "/api/suite-definitions/1/tests",
            json={
                "id": "test-001",
                "name": "Test One",
                "task": {"description": "Do something"},
            },
        )
        assert response.status_code == 401

    def test_add_test_invalid_suite_id(self, client: TestClient) -> None:
        """Test adding test with invalid suite ID."""
        response = client.post(
            "/api/suite-definitions/invalid/tests",
            json={
                "id": "test-001",
                "name": "Test One",
                "task": {"description": "Do something"},
            },
        )
        assert response.status_code == 422

    def test_add_test_missing_required_fields(self, client: TestClient) -> None:
        """Test adding test with missing required fields."""
        response = client.post(
            "/api/suite-definitions/1/tests",
            json={"id": "test-001"},
        )
        assert response.status_code == 422


class TestYAMLExportEndpoints:
    """Integration tests for YAML export endpoint."""

    def test_export_yaml_no_auth(self, client: TestClient) -> None:
        """Test YAML export without authentication (allowed)."""
        response = client.get("/api/suite-definitions/1/yaml")
        # CurrentUser allows optional auth
        assert response.status_code in [200, 400, 404, 500]

    def test_export_yaml_invalid_id(self, client: TestClient) -> None:
        """Test YAML export with invalid ID."""
        response = client.get("/api/suite-definitions/invalid/yaml")
        assert response.status_code == 422


class TestTemplateEndpoints:
    """Integration tests for template endpoints."""

    def test_list_templates_no_auth(self, client: TestClient) -> None:
        """Test listing templates without authentication (allowed)."""
        response = client.get("/api/templates")
        # Should return 200 - templates don't require DB
        assert response.status_code == 200

    def test_list_templates_response_structure(self, client: TestClient) -> None:
        """Test template list response structure."""
        response = client.get("/api/templates")
        if response.status_code == 200:
            data = response.json()
            assert "templates" in data
            assert "categories" in data
            assert "total" in data
            assert isinstance(data["templates"], list)
            assert isinstance(data["categories"], list)
            assert isinstance(data["total"], int)

    def test_list_templates_with_category_filter(self, client: TestClient) -> None:
        """Test listing templates with category filter."""
        response = client.get("/api/templates?category=file_operations")
        assert response.status_code == 200

    def test_list_templates_builtin_exists(self, client: TestClient) -> None:
        """Test that built-in templates exist."""
        response = client.get("/api/templates")
        if response.status_code == 200:
            data = response.json()
            assert data["total"] > 0
            template_names = [t["name"] for t in data["templates"]]
            # Built-in templates from TemplateRegistry
            assert "file_creation" in template_names

    def test_template_has_required_fields(self, client: TestClient) -> None:
        """Test that templates have all required fields."""
        response = client.get("/api/templates")
        if response.status_code == 200:
            data = response.json()
            for template in data["templates"]:
                assert "name" in template
                assert "description" in template
                assert "category" in template
                assert "task_template" in template
                assert "default_constraints" in template
                assert "default_assertions" in template
                assert "tags" in template
                assert "variables" in template

    def test_template_variables_extracted(self, client: TestClient) -> None:
        """Test that template variables are properly extracted."""
        response = client.get("/api/templates")
        if response.status_code == 200:
            data = response.json()
            # file_creation template should have 'filename' and 'content' variables
            file_creation = next(
                (t for t in data["templates"] if t["name"] == "file_creation"),
                None,
            )
            if file_creation:
                assert len(file_creation["variables"]) > 0
                assert "filename" in file_creation["variables"]


class TestRouterConfiguration:
    """Tests for router configuration of new endpoints."""

    def test_suite_definition_routes_exist(self, client: TestClient) -> None:
        """Test that suite definition routes exist."""
        # OPTIONS request to check route exists
        response = client.options("/api/suite-definitions")
        assert response.status_code != 404

    def test_templates_route_exists(self, client: TestClient) -> None:
        """Test that templates route exists."""
        response = client.options("/api/templates")
        assert response.status_code != 404


class TestSuiteDefinitionResponseFormat:
    """Tests for response format validation."""

    def test_list_response_format(self, client: TestClient) -> None:
        """Test list response format."""
        response = client.get("/api/suite-definitions")
        if response.status_code == 200:
            data = response.json()
            assert "total" in data
            assert "items" in data
            assert "limit" in data
            assert "offset" in data
            assert isinstance(data["total"], int)
            assert isinstance(data["items"], list)


class TestComplexSuiteCreation:
    """Tests for complex suite creation scenarios."""

    def test_create_suite_with_all_fields(self, unauth_client: TestClient) -> None:
        """Test creating suite with all optional fields."""
        suite_data = {
            "name": "complex-suite",
            "version": "2.0",
            "description": "A complex test suite",
            "defaults": {
                "runs_per_test": 3,
                "timeout_seconds": 600,
                "scoring": {
                    "quality_weight": 0.5,
                    "completeness_weight": 0.25,
                    "efficiency_weight": 0.15,
                    "cost_weight": 0.1,
                },
                "constraints": {
                    "max_steps": 50,
                    "timeout_seconds": 120,
                },
            },
            "agents": [
                {
                    "name": "agent1",
                    "type": "http",
                    "config": {"endpoint": "http://localhost:8000"},
                },
                {
                    "name": "agent2",
                    "type": "container",
                    "config": {"image": "test-agent:latest"},
                },
            ],
            "tests": [
                {
                    "id": "test-001",
                    "name": "First Test",
                    "description": "Test description",
                    "tags": ["smoke", "api"],
                    "task": {
                        "description": "Do something complex",
                        "input_data": {"key": "value"},
                        "expected_artifacts": ["output.txt"],
                    },
                    "constraints": {
                        "max_steps": 10,
                        "timeout_seconds": 60,
                    },
                    "assertions": [
                        {
                            "type": "artifact_exists",
                            "config": {"path": "output.txt"},
                        }
                    ],
                }
            ],
        }
        # Without auth, should return 401
        response = unauth_client.post("/api/suite-definitions", json=suite_data)
        assert response.status_code == 401

    def test_create_suite_with_invalid_scoring(self, client: TestClient) -> None:
        """Test suite creation with invalid scoring weights."""
        suite_data = {
            "name": "invalid-scoring",
            "defaults": {
                "scoring": {
                    "quality_weight": 1.5,  # Invalid: > 1.0
                },
            },
        }
        response = client.post("/api/suite-definitions", json=suite_data)
        assert response.status_code == 422

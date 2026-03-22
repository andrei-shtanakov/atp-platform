"""Unit tests for audit middleware."""

from atp.dashboard.audit import AuditAction, AuditCategory, AuditSeverity
from atp.dashboard.audit_middleware import (
    AUTH_PATHS,
    EXEMPT_PATHS,
    extract_resource_info,
    get_action_for_method,
    should_audit_request,
)


class TestExtractResourceInfo:
    """Tests for extract_resource_info function."""

    def test_agents_list(self) -> None:
        """Should extract agent resource type from list path."""
        resource_type, resource_id = extract_resource_info("/api/agents")

        assert resource_type == "agent"
        assert resource_id is None

    def test_agents_detail(self) -> None:
        """Should extract agent resource type and ID."""
        resource_type, resource_id = extract_resource_info("/api/agents/123")

        assert resource_type == "agent"
        assert resource_id == "123"

    def test_v2_agents(self) -> None:
        """Should handle v2 API prefix."""
        resource_type, resource_id = extract_resource_info("/api/v2/agents/456")

        assert resource_type == "agent"
        assert resource_id == "456"

    def test_suites(self) -> None:
        """Should extract suite resource."""
        resource_type, resource_id = extract_resource_info("/api/suites/789")

        assert resource_type == "suite"
        assert resource_id == "789"

    def test_users(self) -> None:
        """Should extract user resource."""
        resource_type, resource_id = extract_resource_info("/api/v2/users/42")

        assert resource_type == "user"
        assert resource_id == "42"

    def test_tenants(self) -> None:
        """Should extract tenant resource."""
        resource_type, resource_id = extract_resource_info("/api/tenants/tenant-abc")

        assert resource_type == "tenant"
        assert resource_id == "tenant-abc"

    def test_unknown_path(self) -> None:
        """Should return None for unknown paths."""
        resource_type, resource_id = extract_resource_info("/api/unknown/path")

        assert resource_type is None
        assert resource_id is None


class TestGetActionForMethod:
    """Tests for get_action_for_method function."""

    def test_login_success(self) -> None:
        """Should return login success for successful login."""
        category, action, severity = get_action_for_method(
            "POST", "/api/auth/login", 200
        )

        assert category == AuditCategory.AUTHENTICATION
        assert action == AuditAction.LOGIN_SUCCESS
        assert severity == AuditSeverity.INFO

    def test_login_failure(self) -> None:
        """Should return login failure for failed login."""
        category, action, severity = get_action_for_method(
            "POST", "/api/auth/login", 401
        )

        assert category == AuditCategory.AUTHENTICATION
        assert action == AuditAction.LOGIN_FAILURE
        assert severity == AuditSeverity.WARNING

    def test_logout(self) -> None:
        """Should return logout for logout endpoint."""
        category, action, severity = get_action_for_method(
            "POST", "/api/auth/logout", 200
        )

        assert category == AuditCategory.AUTHENTICATION
        assert action == AuditAction.LOGOUT
        assert severity == AuditSeverity.INFO

    def test_post_create(self) -> None:
        """Should return data create for POST."""
        category, action, severity = get_action_for_method("POST", "/api/agents", 201)

        assert category == AuditCategory.DATA_MODIFICATION
        assert action == AuditAction.DATA_CREATE
        assert severity == AuditSeverity.INFO

    def test_patch_update(self) -> None:
        """Should return data update for PATCH."""
        category, action, severity = get_action_for_method(
            "PATCH", "/api/agents/123", 200
        )

        assert category == AuditCategory.DATA_MODIFICATION
        assert action == AuditAction.DATA_UPDATE
        assert severity == AuditSeverity.INFO

    def test_delete(self) -> None:
        """Should return data delete for DELETE."""
        category, action, severity = get_action_for_method(
            "DELETE", "/api/agents/123", 204
        )

        assert category == AuditCategory.DATA_MODIFICATION
        assert action == AuditAction.DATA_DELETE
        assert severity == AuditSeverity.WARNING

    def test_get_export(self) -> None:
        """Should return data export for export paths."""
        category, action, severity = get_action_for_method(
            "GET", "/api/export/csv", 200
        )

        assert category == AuditCategory.DATA_ACCESS
        assert action == AuditAction.DATA_EXPORT
        assert severity == AuditSeverity.INFO

    def test_config_update(self) -> None:
        """Should return config update for config paths."""
        category, action, severity = get_action_for_method(
            "PUT", "/api/settings/general", 200
        )

        assert category == AuditCategory.CONFIGURATION
        assert action == AuditAction.CONFIG_UPDATE
        assert severity == AuditSeverity.INFO

    def test_bulk_update(self) -> None:
        """Should return bulk update for bulk paths."""
        category, action, severity = get_action_for_method(
            "POST", "/api/agents/bulk", 200
        )

        assert category == AuditCategory.DATA_MODIFICATION
        assert action == AuditAction.BULK_UPDATE
        assert severity == AuditSeverity.INFO

    def test_sso_success(self) -> None:
        """Should return SSO login for SSO callback."""
        category, action, severity = get_action_for_method(
            "GET", "/api/sso/callback", 200
        )

        assert category == AuditCategory.AUTHENTICATION
        assert action == AuditAction.SSO_LOGIN
        assert severity == AuditSeverity.INFO


class TestShouldAuditRequest:
    """Tests for should_audit_request function."""

    def test_skips_health_check(self) -> None:
        """Should skip health check endpoints."""
        assert not should_audit_request("GET", "/health")
        assert not should_audit_request("GET", "/healthz")
        assert not should_audit_request("GET", "/ready")

    def test_skips_docs(self) -> None:
        """Should skip documentation endpoints."""
        assert not should_audit_request("GET", "/docs")
        assert not should_audit_request("GET", "/redoc")
        assert not should_audit_request("GET", "/openapi.json")

    def test_skips_options(self) -> None:
        """Should skip OPTIONS requests."""
        assert not should_audit_request("OPTIONS", "/api/agents")

    def test_audits_post(self) -> None:
        """Should audit POST requests."""
        assert should_audit_request("POST", "/api/agents")

    def test_audits_put(self) -> None:
        """Should audit PUT requests."""
        assert should_audit_request("PUT", "/api/agents/123")

    def test_audits_patch(self) -> None:
        """Should audit PATCH requests."""
        assert should_audit_request("PATCH", "/api/agents/123")

    def test_audits_delete(self) -> None:
        """Should audit DELETE requests."""
        assert should_audit_request("DELETE", "/api/agents/123")

    def test_skips_get_by_default(self) -> None:
        """Should skip GET requests by default."""
        assert not should_audit_request("GET", "/api/agents")

    def test_audits_get_when_enabled(self) -> None:
        """Should audit GET for sensitive resources when enabled."""
        assert should_audit_request("GET", "/api/users/123", audit_get_requests=True)
        assert should_audit_request("GET", "/api/audit", audit_get_requests=True)

    def test_audits_auth_paths(self) -> None:
        """Should always audit auth paths."""
        assert should_audit_request("POST", "/api/auth/login")
        assert should_audit_request("POST", "/api/auth/logout")
        assert should_audit_request("POST", "/api/v2/auth/login")


class TestExemptPaths:
    """Tests for exempt paths configuration."""

    def test_health_endpoints_exempt(self) -> None:
        """Health endpoints should be exempt."""
        assert "/health" in EXEMPT_PATHS
        assert "/healthz" in EXEMPT_PATHS
        assert "/ready" in EXEMPT_PATHS

    def test_docs_exempt(self) -> None:
        """Documentation endpoints should be exempt."""
        assert "/docs" in EXEMPT_PATHS
        assert "/redoc" in EXEMPT_PATHS
        assert "/openapi.json" in EXEMPT_PATHS

    def test_metrics_exempt(self) -> None:
        """Metrics endpoint should be exempt."""
        assert "/metrics" in EXEMPT_PATHS


class TestAuthPaths:
    """Tests for auth paths configuration."""

    def test_login_paths(self) -> None:
        """Login paths should be tracked."""
        assert "/api/auth/login" in AUTH_PATHS
        assert "/api/v2/auth/login" in AUTH_PATHS

    def test_logout_paths(self) -> None:
        """Logout paths should be tracked."""
        assert "/api/auth/logout" in AUTH_PATHS
        assert "/api/v2/auth/logout" in AUTH_PATHS

    def test_register_paths(self) -> None:
        """Register paths should be tracked."""
        assert "/api/auth/register" in AUTH_PATHS
        assert "/api/v2/auth/register" in AUTH_PATHS

    def test_sso_paths(self) -> None:
        """SSO paths should be tracked."""
        assert "/api/sso/callback" in AUTH_PATHS
        assert "/api/v2/sso/callback" in AUTH_PATHS

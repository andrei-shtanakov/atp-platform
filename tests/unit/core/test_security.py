"""Unit tests for atp.core.security module."""

import logging
from pathlib import Path

import pytest

from atp.core.security import (
    MAX_ARRAY_SIZE,
    MAX_ARTIFACTS_COUNT,
    MAX_CPU_CORES,
    MAX_ENV_VARS_COUNT,
    MAX_OBJECT_DEPTH,
    MAX_PATH_LENGTH,
    MAX_REQUEST_SIZE_BYTES,
    MAX_TIMEOUT_SECONDS,
    REDACTED,
    SecureFormatter,
    SecurityEventType,
    SecurityValidationError,
    escape_shell_arg,
    filter_environment_variables,
    is_sensitive_env_var,
    log_security_event,
    open_file_safely,
    parse_memory_limit,
    redact_dict_secrets,
    redact_secrets,
    sanitize_env_value,
    sanitize_error_message,
    sanitize_filename,
    sanitize_log_message,
    setup_secure_logging,
    validate_artifacts_count,
    validate_command,
    validate_cpu_limit,
    validate_docker_image,
    validate_docker_network,
    validate_env_vars_count,
    validate_object_depth,
    validate_path_within_workspace,
    validate_request_size,
    validate_task_description,
    validate_task_id,
    validate_timeout,
    validate_url,
    validate_url_with_dns,
    validate_volume_mount,
)


class TestPathValidation:
    """Tests for path validation functions."""

    def test_validate_path_simple(self, tmp_path: Path) -> None:
        """Test validation of simple relative path."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = validate_path_within_workspace("file.txt", workspace)
        assert result == workspace / "file.txt"

    def test_validate_path_nested(self, tmp_path: Path) -> None:
        """Test validation of nested path."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = validate_path_within_workspace("subdir/file.txt", workspace)
        assert result == workspace / "subdir" / "file.txt"

    def test_validate_path_traversal_blocked(self, tmp_path: Path) -> None:
        """Test that path traversal is blocked."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with pytest.raises(SecurityValidationError) as exc_info:
            validate_path_within_workspace("../outside.txt", workspace)

        assert "traversal" in str(exc_info.value).lower()

    def test_validate_path_absolute_blocked(self, tmp_path: Path) -> None:
        """Test that absolute paths are blocked by default."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with pytest.raises(SecurityValidationError) as exc_info:
            validate_path_within_workspace("/etc/passwd", workspace)

        assert "absolute" in str(exc_info.value).lower()

    def test_validate_path_absolute_allowed(self, tmp_path: Path) -> None:
        """Test that absolute paths within workspace can be allowed."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "allowed.txt").touch()

        result = validate_path_within_workspace(
            str(workspace / "allowed.txt"),
            workspace,
            allow_absolute=True,
        )
        assert result == workspace / "allowed.txt"

    def test_validate_path_absolute_outside_blocked(self, tmp_path: Path) -> None:
        """Test that absolute paths outside workspace are blocked."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with pytest.raises(SecurityValidationError) as exc_info:
            validate_path_within_workspace(
                "/etc/passwd",
                workspace,
                allow_absolute=True,
            )

        assert "escapes" in str(exc_info.value).lower()

    def test_validate_path_null_byte_blocked(self, tmp_path: Path) -> None:
        """Test that null bytes in paths are blocked."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with pytest.raises(SecurityValidationError) as exc_info:
            validate_path_within_workspace("file\x00.txt", workspace)

        assert "null" in str(exc_info.value).lower()

    def test_validate_path_empty_blocked(self, tmp_path: Path) -> None:
        """Test that empty paths are blocked."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with pytest.raises(SecurityValidationError) as exc_info:
            validate_path_within_workspace("", workspace)

        assert "empty" in str(exc_info.value).lower()

    def test_validate_path_too_long(self, tmp_path: Path) -> None:
        """Test that overly long paths are blocked."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        long_path = "a" * (MAX_PATH_LENGTH + 1)
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_path_within_workspace(long_path, workspace)

        assert "length" in str(exc_info.value).lower()


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_sanitize_simple(self) -> None:
        """Test sanitization of simple filename."""
        assert sanitize_filename("file.txt") == "file.txt"

    def test_sanitize_with_slashes(self) -> None:
        """Test that slashes are replaced."""
        assert sanitize_filename("path/to/file.txt") == "path_to_file.txt"

    def test_sanitize_with_backslashes(self) -> None:
        """Test that backslashes are replaced."""
        assert sanitize_filename("path\\to\\file.txt") == "path_to_file.txt"

    def test_sanitize_hidden_files(self) -> None:
        """Test that leading dots are removed."""
        result = sanitize_filename(".hidden")
        assert not result.startswith(".")

    def test_sanitize_null_bytes(self) -> None:
        """Test that null bytes are removed."""
        result = sanitize_filename("file\x00name.txt")
        assert "\x00" not in result

    def test_sanitize_special_chars(self) -> None:
        """Test that special characters are replaced."""
        result = sanitize_filename("file<>:\"'|?*name.txt")
        assert all(c not in result for c in "<>:\"'|?*")

    def test_sanitize_empty(self) -> None:
        """Test that empty filename returns 'unnamed'."""
        assert sanitize_filename("") == "unnamed"

    def test_sanitize_long_filename(self) -> None:
        """Test that long filenames are truncated."""
        long_name = "a" * 300
        result = sanitize_filename(long_name)
        assert len(result) <= 255


class TestURLValidation:
    """Tests for URL validation."""

    def test_validate_url_https(self) -> None:
        """Test validation of HTTPS URL."""
        result = validate_url("https://example.com/api")
        assert result == "https://example.com/api"

    def test_validate_url_http(self) -> None:
        """Test validation of HTTP URL."""
        result = validate_url("http://example.com/api")
        assert result == "http://example.com/api"

    def test_validate_url_file_blocked(self) -> None:
        """Test that file:// URLs are blocked."""
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_url("file:///etc/passwd")

        assert "scheme" in str(exc_info.value).lower()

    def test_validate_url_localhost_blocked(self) -> None:
        """Test that localhost is blocked by default."""
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_url("http://localhost:8080/api")

        assert "localhost" in str(exc_info.value).lower()

    def test_validate_url_localhost_allowed(self) -> None:
        """Test that localhost can be allowed."""
        result = validate_url("http://localhost:8080/api", allow_internal=True)
        assert result == "http://localhost:8080/api"

    def test_validate_url_private_ip_blocked(self) -> None:
        """Test that private IPs are blocked."""
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_url("http://192.168.1.1/api")

        assert "internal" in str(exc_info.value).lower()

    def test_validate_url_metadata_blocked(self) -> None:
        """Test that cloud metadata endpoints are blocked."""
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_url("http://169.254.169.254/latest/meta-data/")

        assert "metadata" in str(exc_info.value).lower()

    def test_validate_url_empty_blocked(self) -> None:
        """Test that empty URLs are blocked."""
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_url("")

        assert "empty" in str(exc_info.value).lower()

    def test_validate_url_no_scheme_blocked(self) -> None:
        """Test that URLs without scheme are blocked."""
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_url("example.com/api")

        assert "scheme" in str(exc_info.value).lower()


class TestDockerValidation:
    """Tests for Docker validation functions."""

    def test_validate_docker_image_simple(self) -> None:
        """Test validation of simple image name."""
        assert validate_docker_image("nginx") == "nginx"

    def test_validate_docker_image_with_tag(self) -> None:
        """Test validation of image with tag."""
        assert validate_docker_image("nginx:latest") == "nginx:latest"

    def test_validate_docker_image_with_registry(self) -> None:
        """Test validation of image with registry."""
        result = validate_docker_image("registry.io/myimage:v1")
        assert result == "registry.io/myimage:v1"

    def test_validate_docker_image_empty(self) -> None:
        """Test that empty image name is blocked."""
        with pytest.raises(SecurityValidationError):
            validate_docker_image("")

    def test_validate_docker_network_none(self) -> None:
        """Test validation of 'none' network."""
        assert validate_docker_network("none") == "none"

    def test_validate_docker_network_bridge(self) -> None:
        """Test validation of 'bridge' network."""
        assert validate_docker_network("bridge") == "bridge"

    def test_validate_docker_network_host(self) -> None:
        """Test validation of 'host' network."""
        assert validate_docker_network("host") == "host"

    def test_validate_docker_network_invalid(self) -> None:
        """Test that invalid network mode is blocked."""
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_docker_network("custom_network")

        assert "mode" in str(exc_info.value).lower()

    def test_validate_docker_network_empty(self) -> None:
        """Test that empty network defaults to 'none'."""
        assert validate_docker_network("") == "none"


class TestVolumeValidation:
    """Tests for volume mount validation."""

    def test_validate_volume_mount_basic(self, tmp_path: Path) -> None:
        """Test basic volume mount validation."""
        host_path = str(tmp_path)
        container_path = "/data"

        result = validate_volume_mount(host_path, container_path)
        assert result == (str(tmp_path.resolve()), container_path)

    def test_validate_volume_mount_dangerous_container_path(
        self, tmp_path: Path
    ) -> None:
        """Test that mounting to /root is blocked."""
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_volume_mount(str(tmp_path), "/root")

        assert "system path" in str(exc_info.value).lower()

    def test_validate_volume_mount_etc_blocked(self, tmp_path: Path) -> None:
        """Test that mounting to /etc is blocked."""
        with pytest.raises(SecurityValidationError):
            validate_volume_mount(str(tmp_path), "/etc")

    def test_validate_volume_mount_proc_blocked(self, tmp_path: Path) -> None:
        """Test that mounting to /proc is blocked."""
        with pytest.raises(SecurityValidationError):
            validate_volume_mount(str(tmp_path), "/proc")

    def test_validate_volume_mount_allowed_paths(self, tmp_path: Path) -> None:
        """Test volume mount with allowed paths."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        host_path = str(allowed / "data")

        result = validate_volume_mount(
            host_path,
            "/data",
            allowed_base_paths=[allowed],
        )
        assert result[1] == "/data"

    def test_validate_volume_mount_outside_allowed(self, tmp_path: Path) -> None:
        """Test that mount outside allowed paths is blocked."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        with pytest.raises(SecurityValidationError):
            validate_volume_mount(
                str(outside),
                "/data",
                allowed_base_paths=[allowed],
            )


class TestResourceValidation:
    """Tests for resource limit validation."""

    def test_parse_memory_limit_megabytes(self) -> None:
        """Test parsing memory in megabytes."""
        assert parse_memory_limit("512m") == 512 * 1024 * 1024

    def test_parse_memory_limit_gigabytes(self) -> None:
        """Test parsing memory in gigabytes."""
        assert parse_memory_limit("2g") == 2 * 1024 * 1024 * 1024

    def test_parse_memory_limit_gibibytes(self) -> None:
        """Test parsing memory in gibibytes."""
        assert parse_memory_limit("1Gi") == 1024 * 1024 * 1024

    def test_parse_memory_limit_invalid_format(self) -> None:
        """Test that invalid format raises error."""
        with pytest.raises(SecurityValidationError) as exc_info:
            parse_memory_limit("invalid")

        assert "format" in str(exc_info.value).lower()

    def test_parse_memory_limit_exceeds_max(self) -> None:
        """Test that exceeding max limit raises error."""
        with pytest.raises(SecurityValidationError) as exc_info:
            parse_memory_limit("100g")  # 100GB > 16GB max

        assert "exceeds" in str(exc_info.value).lower()

    def test_parse_memory_limit_empty(self) -> None:
        """Test that empty string raises error."""
        with pytest.raises(SecurityValidationError):
            parse_memory_limit("")

    def test_validate_cpu_limit_integer(self) -> None:
        """Test validation of integer CPU limit."""
        assert validate_cpu_limit("2") == 2.0

    def test_validate_cpu_limit_float(self) -> None:
        """Test validation of float CPU limit."""
        assert validate_cpu_limit("0.5") == 0.5

    def test_validate_cpu_limit_exceeds_max(self) -> None:
        """Test that exceeding max CPU raises error."""
        with pytest.raises(SecurityValidationError):
            validate_cpu_limit(str(MAX_CPU_CORES + 1))

    def test_validate_cpu_limit_zero(self) -> None:
        """Test that zero CPU raises error."""
        with pytest.raises(SecurityValidationError):
            validate_cpu_limit("0")

    def test_validate_cpu_limit_negative(self) -> None:
        """Test that negative CPU raises error."""
        with pytest.raises(SecurityValidationError):
            validate_cpu_limit("-1")

    def test_validate_timeout_valid(self) -> None:
        """Test validation of valid timeout."""
        assert validate_timeout(300) == 300

    def test_validate_timeout_exceeds_max(self) -> None:
        """Test that exceeding max timeout raises error."""
        with pytest.raises(SecurityValidationError):
            validate_timeout(MAX_TIMEOUT_SECONDS + 1)

    def test_validate_timeout_zero(self) -> None:
        """Test that zero timeout raises error."""
        with pytest.raises(SecurityValidationError):
            validate_timeout(0)

    def test_validate_timeout_negative(self) -> None:
        """Test that negative timeout raises error."""
        with pytest.raises(SecurityValidationError):
            validate_timeout(-1)


class TestSecretRedaction:
    """Tests for secret redaction functions."""

    def test_redact_api_key(self) -> None:
        """Test redaction of API key."""
        text = "api_key=sk-1234567890abcdefghij"
        result = redact_secrets(text)
        assert "sk-1234567890" not in result
        assert REDACTED in result

    def test_redact_bearer_token(self) -> None:
        """Test redaction of bearer token."""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = redact_secrets(text)
        assert "eyJhbGciOiJIUzI1" not in result
        assert REDACTED in result

    def test_redact_password(self) -> None:
        """Test redaction of password."""
        text = "password=supersecretpassword123"
        result = redact_secrets(text)
        assert "supersecret" not in result
        assert REDACTED in result

    def test_redact_aws_key(self) -> None:
        """Test redaction of AWS access key."""
        text = "AWS key: AKIAIOSFODNN7EXAMPLE"
        result = redact_secrets(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_redact_private_key_header(self) -> None:
        """Test redaction of private key header."""
        text = "Key: -----BEGIN RSA PRIVATE KEY-----\nMIIE..."
        result = redact_secrets(text)
        # The pattern detects the header
        assert "-----BEGIN RSA PRIVATE KEY-----" not in result or REDACTED in result

    def test_redact_no_secrets(self) -> None:
        """Test that text without secrets is unchanged."""
        text = "This is normal text without secrets"
        result = redact_secrets(text)
        assert result == text

    def test_redact_empty(self) -> None:
        """Test that empty string returns empty."""
        assert redact_secrets("") == ""


class TestEnvironmentFiltering:
    """Tests for environment variable filtering."""

    def test_is_sensitive_api_key(self) -> None:
        """Test detection of API key variable."""
        assert is_sensitive_env_var("API_KEY")
        assert is_sensitive_env_var("OPENAI_API_KEY")

    def test_is_sensitive_token(self) -> None:
        """Test detection of token variable."""
        assert is_sensitive_env_var("AUTH_TOKEN")
        assert is_sensitive_env_var("ACCESS_TOKEN")

    def test_is_sensitive_secret(self) -> None:
        """Test detection of secret variable."""
        assert is_sensitive_env_var("CLIENT_SECRET")
        assert is_sensitive_env_var("JWT_SECRET")

    def test_is_sensitive_password(self) -> None:
        """Test detection of password variable."""
        assert is_sensitive_env_var("DATABASE_PASSWORD")
        assert is_sensitive_env_var("DB_PASSWD")

    def test_is_sensitive_aws(self) -> None:
        """Test detection of AWS variables."""
        assert is_sensitive_env_var("AWS_SECRET_ACCESS_KEY")
        assert is_sensitive_env_var("AWS_SESSION_TOKEN")

    def test_is_sensitive_safe_vars(self) -> None:
        """Test that safe variables are not flagged."""
        assert not is_sensitive_env_var("PATH")
        assert not is_sensitive_env_var("HOME")
        assert not is_sensitive_env_var("TERM")

    def test_filter_env_vars_removes_secrets(self) -> None:
        """Test that filtering removes secret variables."""
        env = {
            "PATH": "/usr/bin",
            "API_KEY": "secret123",
            "HOME": "/home/user",
            "DATABASE_PASSWORD": "dbpass",
        }

        result = filter_environment_variables(env)

        assert "PATH" in result
        assert "HOME" in result
        assert "API_KEY" not in result
        assert "DATABASE_PASSWORD" not in result

    def test_filter_env_vars_allowlist(self) -> None:
        """Test that additional allowlist works."""
        env = {
            "PATH": "/usr/bin",
            "CUSTOM_VAR": "value",
        }

        result = filter_environment_variables(
            env,
            additional_allowlist={"CUSTOM_VAR"},
        )

        assert "CUSTOM_VAR" in result

    def test_filter_env_vars_blocklist(self) -> None:
        """Test that additional blocklist works."""
        env = {
            "PATH": "/usr/bin",
            "BLOCKED_VAR": "value",
        }

        result = filter_environment_variables(
            env,
            additional_blocklist={"BLOCKED_VAR"},
        )

        assert "BLOCKED_VAR" not in result


class TestDictRedaction:
    """Tests for dictionary secret redaction."""

    def test_redact_dict_password_key(self) -> None:
        """Test redaction of password key."""
        data = {"username": "user", "password": "secret123"}
        result = redact_dict_secrets(data)

        assert result["username"] == "user"
        assert result["password"] == REDACTED

    def test_redact_dict_nested(self) -> None:
        """Test redaction in nested dict."""
        data = {"config": {"api_key": "secret", "name": "test"}}
        result = redact_dict_secrets(data)

        assert result["config"]["api_key"] == REDACTED
        assert result["config"]["name"] == "test"

    def test_redact_dict_list(self) -> None:
        """Test redaction in list values."""
        data = {"items": [{"token": "abc123def456"}, {"name": "item"}]}
        result = redact_dict_secrets(data)

        assert result["items"][0]["token"] == REDACTED
        assert result["items"][1]["name"] == "item"

    def test_redact_dict_max_depth(self) -> None:
        """Test that max depth prevents infinite recursion."""
        # Create deeply nested structure
        data: dict = {"level1": {}}
        current = data["level1"]
        for i in range(10):
            current["level"] = {}
            current = current["level"]
        current["password"] = "secret"

        # Should not raise with reasonable max_depth
        result = redact_dict_secrets(data, max_depth=5)
        assert result is not None


class TestInputValidation:
    """Tests for input validation functions."""

    def test_validate_task_id_valid(self) -> None:
        """Test validation of valid task ID."""
        assert validate_task_id("task-123") == "task-123"
        assert validate_task_id("TASK_456") == "TASK_456"

    def test_validate_task_id_empty(self) -> None:
        """Test that empty task ID raises error."""
        with pytest.raises(SecurityValidationError):
            validate_task_id("")

    def test_validate_task_id_invalid_chars(self) -> None:
        """Test that invalid characters raise error."""
        with pytest.raises(SecurityValidationError):
            validate_task_id("task/with/slashes")

    def test_validate_task_id_too_long(self) -> None:
        """Test that too long task ID raises error."""
        with pytest.raises(SecurityValidationError):
            validate_task_id("a" * 200)

    def test_validate_task_description_valid(self) -> None:
        """Test validation of valid description."""
        result = validate_task_description("Run the test suite")
        assert result == "Run the test suite"

    def test_validate_task_description_strips_whitespace(self) -> None:
        """Test that description whitespace is stripped."""
        result = validate_task_description("  Run tests  ")
        assert result == "Run tests"

    def test_validate_task_description_empty(self) -> None:
        """Test that empty description raises error."""
        with pytest.raises(SecurityValidationError):
            validate_task_description("")

    def test_validate_task_description_whitespace_only(self) -> None:
        """Test that whitespace-only description raises error."""
        with pytest.raises(SecurityValidationError):
            validate_task_description("   ")


class TestSecureLogging:
    """Tests for secure logging utilities."""

    def test_secure_formatter_redacts_secrets(self) -> None:
        """Test that SecureFormatter redacts secrets."""
        formatter = SecureFormatter("%(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="api_key=sk-1234567890abcdefghij",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert "sk-1234567890" not in result
        assert REDACTED in result

    def test_setup_secure_logging(self) -> None:
        """Test that setup_secure_logging configures logger."""
        logger = logging.getLogger("test_secure")
        handler = logging.StreamHandler()
        logger.addHandler(handler)

        setup_secure_logging(logger)

        # Verify formatter is SecureFormatter
        assert isinstance(handler.formatter, SecureFormatter)


class TestSecurityValidationError:
    """Tests for SecurityValidationError."""

    def test_error_message_format(self) -> None:
        """Test error message formatting."""
        error = SecurityValidationError(
            "Validation failed",
            field="test_field",
            value="test_value",
        )

        assert "Validation failed" in str(error)
        assert "test_field" in str(error)

    def test_error_redacts_sensitive_value(self) -> None:
        """Test that sensitive values in errors are redacted."""
        error = SecurityValidationError(
            "Invalid",
            field="api_key",
            value="api_key=sk-secretkey12345678901234567890",
        )

        # The value should be redacted
        assert "sk-secretkey" not in str(error)

    def test_error_has_event_type(self) -> None:
        """Test that error has event type."""
        error = SecurityValidationError(
            "Test",
            event_type=SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
        )
        assert error.event_type == SecurityEventType.PATH_TRAVERSAL_ATTEMPT


class TestRequestSizeValidation:
    """Tests for request size validation."""

    def test_validate_request_size_valid(self) -> None:
        """Test validation of valid request size."""
        data = "x" * 1000
        validate_request_size(data)  # Should not raise

    def test_validate_request_size_bytes(self) -> None:
        """Test validation with bytes."""
        data = b"x" * 1000
        validate_request_size(data)  # Should not raise

    def test_validate_request_size_exceeds_limit(self) -> None:
        """Test that exceeding limit raises error."""
        data = "x" * (MAX_REQUEST_SIZE_BYTES + 1)
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_request_size(data)
        assert "exceeds maximum" in str(exc_info.value).lower()

    def test_validate_request_size_custom_limit(self) -> None:
        """Test with custom size limit."""
        data = "x" * 1000
        with pytest.raises(SecurityValidationError):
            validate_request_size(data, max_size=100)


class TestObjectDepthValidation:
    """Tests for object depth validation."""

    def test_validate_depth_shallow(self) -> None:
        """Test validation of shallow object."""
        obj = {"a": {"b": "c"}}
        validate_object_depth(obj)  # Should not raise

    def test_validate_depth_exceeds_limit(self) -> None:
        """Test that exceeding depth limit raises error."""
        # Create deeply nested object
        obj: dict = {}
        current = obj
        for _ in range(MAX_OBJECT_DEPTH + 5):
            current["nested"] = {}
            current = current["nested"]

        with pytest.raises(SecurityValidationError) as exc_info:
            validate_object_depth(obj)
        assert "nesting depth" in str(exc_info.value).lower()

    def test_validate_array_size(self) -> None:
        """Test that large arrays are blocked."""
        obj = {"items": list(range(MAX_ARRAY_SIZE + 1))}
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_object_depth(obj)
        assert "array size" in str(exc_info.value).lower()

    def test_validate_nested_arrays(self) -> None:
        """Test validation of nested arrays."""
        obj = {"items": [{"sub": [1, 2, 3]}]}
        validate_object_depth(obj)  # Should not raise


class TestArtifactsCountValidation:
    """Tests for artifacts count validation."""

    def test_validate_artifacts_count_valid(self) -> None:
        """Test validation of valid count."""
        validate_artifacts_count(100)  # Should not raise

    def test_validate_artifacts_count_exceeds(self) -> None:
        """Test that exceeding limit raises error."""
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_artifacts_count(MAX_ARTIFACTS_COUNT + 1)
        assert "artifacts count" in str(exc_info.value).lower()


class TestEnvVarsCountValidation:
    """Tests for environment variables count validation."""

    def test_validate_env_vars_count_valid(self) -> None:
        """Test validation of valid count."""
        validate_env_vars_count(50)  # Should not raise

    def test_validate_env_vars_count_exceeds(self) -> None:
        """Test that exceeding limit raises error."""
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_env_vars_count(MAX_ENV_VARS_COUNT + 1)
        assert "environment variables count" in str(exc_info.value).lower()


class TestCommandValidation:
    """Tests for command validation."""

    def test_validate_command_allowed(self) -> None:
        """Test validation of allowed command."""
        result = validate_command("python script.py")
        assert result == "python script.py"

    def test_validate_command_with_args(self) -> None:
        """Test validation with arguments."""
        result = validate_command("python -m pytest tests/")
        assert "python" in result

    def test_validate_command_blocked_binary(self) -> None:
        """Test that non-allowlisted binary is blocked."""
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_command("rm -rf /")
        assert "not in allowed list" in str(exc_info.value).lower()

    def test_validate_command_shell_metachar(self) -> None:
        """Test that shell metacharacters are blocked."""
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_command("python; rm -rf /")
        assert "metacharacter" in str(exc_info.value).lower()

    def test_validate_command_null_byte(self) -> None:
        """Test that null bytes are blocked."""
        with pytest.raises(SecurityValidationError):
            validate_command("python\x00script.py")

    def test_validate_command_empty(self) -> None:
        """Test that empty command is blocked."""
        with pytest.raises(SecurityValidationError):
            validate_command("")

    def test_validate_command_custom_allowlist(self) -> None:
        """Test with custom allowlist."""
        result = validate_command(
            "custom_tool arg",
            allowed_commands={"custom_tool"},
        )
        assert "custom_tool" in result

    def test_validate_command_pipe_blocked(self) -> None:
        """Test that pipe is blocked."""
        with pytest.raises(SecurityValidationError):
            validate_command("python | cat")

    def test_validate_command_ampersand_blocked(self) -> None:
        """Test that ampersand is blocked."""
        with pytest.raises(SecurityValidationError):
            validate_command("python & sleep 10")


class TestEscapeShellArg:
    """Tests for shell argument escaping."""

    def test_escape_simple(self) -> None:
        """Test escaping simple argument."""
        result = escape_shell_arg("hello")
        assert result == "hello"

    def test_escape_spaces(self) -> None:
        """Test escaping argument with spaces."""
        result = escape_shell_arg("hello world")
        assert " " not in result or result.startswith("'")

    def test_escape_special_chars(self) -> None:
        """Test escaping special characters."""
        result = escape_shell_arg("$HOME")
        assert "$" not in result or "'" in result


class TestSanitizeEnvValue:
    """Tests for environment variable value sanitization."""

    def test_sanitize_normal_value(self) -> None:
        """Test sanitization of normal value."""
        result = sanitize_env_value("normal_value")
        assert result == "normal_value"

    def test_sanitize_newlines(self) -> None:
        """Test that newlines are removed."""
        result = sanitize_env_value("line1\nline2")
        assert "\n" not in result

    def test_sanitize_carriage_return(self) -> None:
        """Test that carriage returns are removed."""
        result = sanitize_env_value("line1\rline2")
        assert "\r" not in result

    def test_sanitize_null_bytes(self) -> None:
        """Test that null bytes raise error."""
        with pytest.raises(SecurityValidationError):
            sanitize_env_value("value\x00with\x00nulls")

    def test_sanitize_long_value(self) -> None:
        """Test that long values are truncated."""
        long_value = "x" * 50000
        result = sanitize_env_value(long_value)
        assert len(result) <= 32768


class TestSanitizeLogMessage:
    """Tests for log message sanitization."""

    def test_sanitize_normal_message(self) -> None:
        """Test sanitization of normal message."""
        result = sanitize_log_message("Normal log message")
        assert result == "Normal log message"

    def test_sanitize_newlines(self) -> None:
        """Test that newlines are escaped."""
        result = sanitize_log_message("Line1\nLine2")
        assert "\n" not in result
        assert "\\n" in result

    def test_sanitize_carriage_returns(self) -> None:
        """Test that carriage returns are escaped."""
        result = sanitize_log_message("Line1\rLine2")
        assert "\r" not in result
        assert "\\r" in result

    def test_sanitize_ansi_codes(self) -> None:
        """Test that ANSI escape codes are removed."""
        result = sanitize_log_message("\x1b[31mRed text\x1b[0m")
        assert "\x1b" not in result

    def test_sanitize_long_message(self) -> None:
        """Test that long messages are truncated."""
        long_msg = "x" * 20000
        result = sanitize_log_message(long_msg)
        assert len(result) <= 10000
        assert "TRUNCATED" in result


class TestSanitizeErrorMessage:
    """Tests for error message sanitization."""

    def test_sanitize_simple_error(self) -> None:
        """Test sanitization of simple error."""
        result = sanitize_error_message("Something went wrong")
        assert "Something went wrong" in result

    def test_sanitize_exception(self) -> None:
        """Test sanitization of exception."""
        error = ValueError("Invalid value")
        result = sanitize_error_message(error)
        assert "ValueError" in result
        assert "Invalid value" in result

    def test_sanitize_file_paths(self) -> None:
        """Test that file paths are redacted."""
        result = sanitize_error_message("Error in /home/user/secret/project/file.py")
        assert "/home/user" not in result
        assert "PATH_REDACTED" in result

    def test_sanitize_secrets(self) -> None:
        """Test that secrets are redacted."""
        result = sanitize_error_message("api_key=sk-1234567890abcdefghij")
        assert "sk-1234567890" not in result

    def test_sanitize_long_error(self) -> None:
        """Test that long errors are truncated."""
        long_error = "x" * 2000
        result = sanitize_error_message(long_error)
        assert len(result) <= 1020  # 1000 + type prefix


class TestURLValidationWithDNS:
    """Tests for URL validation with DNS check."""

    def test_validate_external_url(self) -> None:
        """Test validation of external URL."""
        result = validate_url_with_dns("https://example.com/api")
        assert result == "https://example.com/api"

    def test_validate_ip_address(self) -> None:
        """Test validation of IP address URL."""
        # External IP should be allowed
        result = validate_url_with_dns("https://8.8.8.8/api")
        assert "8.8.8.8" in result

    def test_validate_internal_allowed(self) -> None:
        """Test that internal URLs can be allowed."""
        result = validate_url_with_dns(
            "http://localhost:8080/api",
            allow_internal=True,
        )
        assert result == "http://localhost:8080/api"


class TestOpenFileSafely:
    """Tests for safe file opening."""

    def test_open_file_read(self, tmp_path: Path) -> None:
        """Test opening file for reading."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        with open_file_safely(test_file, "r") as f:
            content = f.read()
        assert content == "test content"

    def test_open_file_write(self, tmp_path: Path) -> None:
        """Test opening file for writing."""
        test_file = tmp_path / "test.txt"

        with open_file_safely(test_file, "w") as f:
            f.write("new content")

        assert test_file.read_text() == "new content"

    def test_open_file_with_workspace_validation(self, tmp_path: Path) -> None:
        """Test opening file with workspace validation."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        test_file = workspace / "test.txt"
        test_file.write_text("content")

        with open_file_safely(test_file, "r", workspace=workspace) as f:
            content = f.read()
        assert content == "content"


class TestSecurityEventLogging:
    """Tests for security event logging."""

    def test_log_security_event(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that security events are logged."""
        with caplog.at_level(logging.WARNING, logger="atp.security.audit"):
            log_security_event(
                SecurityEventType.VALIDATION_FAILURE,
                "Test validation failure",
                field="test_field",
            )

        assert "validation_failure" in caplog.text.lower()

    def test_log_security_event_redacts_secrets(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that secrets in events are redacted."""
        with caplog.at_level(logging.WARNING, logger="atp.security.audit"):
            log_security_event(
                SecurityEventType.SECRET_REDACTED,
                "Found api_key=sk-1234567890abcdefghij",
            )

        assert "sk-1234567890" not in caplog.text


class TestAdditionalSecretPatterns:
    """Tests for additional secret patterns."""

    def test_redact_jwt_token(self) -> None:
        """Test redaction of JWT token."""
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        )
        result = redact_secrets(jwt)
        assert "eyJhbGciOiJIUzI1NiIs" not in result

    def test_redact_github_token(self) -> None:
        """Test redaction of GitHub token."""
        text = "GITHUB_TOKEN=ghp_1234567890abcdefghijklmnopqrstuvwxyz"
        result = redact_secrets(text)
        assert "ghp_1234567890" not in result

    def test_redact_slack_token(self) -> None:
        """Test redaction of Slack token."""
        text = "token: xoxb-1234567890-abcdefghij"
        result = redact_secrets(text)
        assert "xoxb-1234567890" not in result

    def test_redact_stripe_key(self) -> None:
        """Test redaction of Stripe key."""
        text = "stripe_key=sk_live_12345678901234567890"
        result = redact_secrets(text)
        assert "sk_live_" not in result

    def test_redact_connection_string(self) -> None:
        """Test redaction of database connection string."""
        text = "mongodb://user:password@localhost:27017/db"
        result = redact_secrets(text)
        assert "password" not in result
        assert "CREDENTIALS_REDACTED" in result

    def test_redact_basic_auth(self) -> None:
        """Test redaction of Basic auth header."""
        text = "Authorization: Basic dXNlcjpwYXNzd29yZA=="
        result = redact_secrets(text)
        assert "dXNlcjpwYXNzd29yZA" not in result

    def test_redact_ssh_key(self) -> None:
        """Test redaction of SSH private key header."""
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIE..."
        result = redact_secrets(text)
        assert "-----BEGIN RSA PRIVATE KEY-----" not in result

    def test_redact_pgp_key(self) -> None:
        """Test redaction of PGP private key header."""
        text = "-----BEGIN PGP PRIVATE KEY BLOCK-----\n..."
        result = redact_secrets(text)
        assert "-----BEGIN PGP PRIVATE KEY BLOCK-----" not in result

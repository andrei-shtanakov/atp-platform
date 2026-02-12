"""Security utilities for ATP platform.

This module provides security primitives for:
- Input validation (paths, URLs, commands)
- Secret redaction in logs and error messages
- Environment variable filtering
- Resource limit validation
- Security audit logging
- Defense-in-depth protections
"""

import errno
import ipaddress
import logging
import os
import re
import socket
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("atp.security.audit")

# =============================================================================
# Constants
# =============================================================================

# Patterns for detecting secrets in strings
# Each tuple contains: (name, pattern, replacement_template)
# If replacement_template is None, the entire match is replaced with [REDACTED]
SECRET_PATTERNS: list[tuple[str, re.Pattern[str], str | None]] = [
    (
        "api_key",
        re.compile(r"(api[_-]?key)[=:\s]+['\"]?[a-zA-Z0-9_-]{20,}['\"]?", re.I),
        r"\1=[REDACTED]",
    ),
    (
        "token",
        re.compile(r"(token)[=:\s]+['\"]?[a-zA-Z0-9_.-]{20,}['\"]?", re.I),
        r"\1=[REDACTED]",
    ),
    (
        "bearer",
        re.compile(r"(bearer)\s+[a-zA-Z0-9_.-]{20,}", re.I),
        r"\1 [REDACTED]",
    ),
    (
        "password",
        re.compile(r"(password|passwd|pwd)[=:\s]+['\"]?[^\s'\"]{8,}['\"]?", re.I),
        r"\1=[REDACTED]",
    ),
    (
        "secret",
        re.compile(r"(secret)[=:\s]+['\"]?[a-zA-Z0-9_-]{16,}['\"]?", re.I),
        r"\1=[REDACTED]",
    ),
    (
        "authorization",
        re.compile(r"(authorization)[=:\s]+['\"]?[a-zA-Z0-9_.-]{20,}['\"]?", re.I),
        r"\1=[REDACTED]",
    ),
    (
        "aws_key",
        re.compile(r"AKIA[A-Z0-9]{16}", re.I),
        None,  # Replace entire match
    ),
    (
        "private_key",
        re.compile(r"-----BEGIN [A-Z]+ PRIVATE KEY-----", re.I),
        None,  # Replace entire match
    ),
    # Additional patterns for comprehensive coverage
    (
        "jwt_token",
        re.compile(r"eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]+"),
        None,  # JWT tokens
    ),
    (
        "github_token",
        re.compile(r"gh[pousr]_[a-zA-Z0-9]{36,}", re.I),
        None,  # GitHub tokens (PAT, OAuth, etc.)
    ),
    (
        "slack_token",
        re.compile(r"xox[baprs]-[a-zA-Z0-9-]{10,}", re.I),
        None,  # Slack tokens
    ),
    (
        "stripe_key",
        re.compile(r"[rs]k_(live|test)_[a-zA-Z0-9]{20,}", re.I),
        None,  # Stripe API keys
    ),
    (
        "connection_string",
        re.compile(
            r"(mongodb|postgres|mysql|redis|amqp)://[^\s'\"<>]+:[^\s'\"<>]+@", re.I
        ),
        r"\1://[CREDENTIALS_REDACTED]@",
    ),
    (
        "basic_auth",
        re.compile(r"(basic)\s+[a-zA-Z0-9+/=]{20,}", re.I),
        r"\1 [REDACTED]",
    ),
    (
        "ssh_key",
        re.compile(r"-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----"),
        None,
    ),
    (
        "pgp_key",
        re.compile(r"-----BEGIN PGP PRIVATE KEY BLOCK-----"),
        None,
    ),
]

# Environment variable patterns that should be filtered (sensitive)
SENSITIVE_ENV_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r".*_KEY$", re.I),
    re.compile(r".*_TOKEN$", re.I),
    re.compile(r".*_SECRET$", re.I),
    re.compile(r".*_PASSWORD$", re.I),
    re.compile(r".*_PASSWD$", re.I),
    re.compile(r".*_PWD$", re.I),
    re.compile(r".*_CREDENTIALS?$", re.I),
    re.compile(r".*_API_?KEY$", re.I),
    re.compile(r"^AWS_.*", re.I),
    re.compile(r"^AZURE_.*", re.I),
    re.compile(r"^GCP_.*", re.I),
    re.compile(r"^GOOGLE_.*", re.I),
    re.compile(r"^ANTHROPIC_.*", re.I),
    re.compile(r"^OPENAI_.*", re.I),
    re.compile(r"^DATABASE_.*", re.I),
    re.compile(r"^DB_.*", re.I),
    re.compile(r"^PRIVATE_.*", re.I),
]

# Safe environment variables that can be inherited by subprocesses
SAFE_ENV_ALLOWLIST: set[str] = {
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "TERM",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "NODE_PATH",
    "GOPATH",
    "JAVA_HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "XDG_RUNTIME_DIR",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "DISPLAY",
    "WAYLAND_DISPLAY",
    # ATP-specific safe variables
    "ATP_LOG_LEVEL",
    "ATP_DEBUG",
    "ATP_WORKSPACE",
}

# Allowed URL schemes
ALLOWED_URL_SCHEMES: set[str] = {"http", "https"}

# Blocked internal hosts/IPs
INTERNAL_IP_RANGES: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network("127.0.0.0/8"),  # Localhost
    ipaddress.ip_network("10.0.0.0/8"),  # Private
    ipaddress.ip_network("172.16.0.0/12"),  # Private
    ipaddress.ip_network("192.168.0.0/16"),  # Private
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local
    ipaddress.ip_network("::1/128"),  # IPv6 localhost
    ipaddress.ip_network("fc00::/7"),  # IPv6 private
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
]

# Cloud metadata service endpoints to block
METADATA_ENDPOINTS: set[str] = {
    "169.254.169.254",  # AWS/GCP/Azure metadata
    "metadata.google.internal",
    "metadata.goog",
    "169.254.170.2",  # AWS ECS metadata
}

# Valid Docker image name pattern
DOCKER_IMAGE_PATTERN = re.compile(
    r"^(?:(?P<registry>[a-zA-Z0-9][-a-zA-Z0-9.]*(?::[0-9]+)?)/)??"
    r"(?P<name>(?:[a-z0-9]+(?:[._-][a-z0-9]+)*/?)+)"
    r"(?::(?P<tag>[a-zA-Z0-9][-a-zA-Z0-9._]*))?$"
)

# Valid Docker network modes
VALID_NETWORK_MODES: set[str] = {"none", "bridge", "host"}

# Maximum allowed values
MAX_MEMORY_BYTES = 16 * 1024 * 1024 * 1024  # 16 GB
MAX_CPU_CORES = 16
MAX_TIMEOUT_SECONDS = 3600  # 1 hour
MAX_PATH_LENGTH = 4096
MAX_DESCRIPTION_LENGTH = 100_000

# Request/Response size limits
MAX_REQUEST_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_RESPONSE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_ARTIFACTS_COUNT = 1000
MAX_ENV_VARS_COUNT = 100
MAX_OBJECT_DEPTH = 50
MAX_ARRAY_SIZE = 10_000

# Redaction placeholder
REDACTED = "[REDACTED]"

# Allowed command binaries for CLI adapter (basename only)
SAFE_COMMAND_BINARIES: set[str] = {
    "python",
    "python3",
    "node",
    "npm",
    "npx",
    "java",
    "bash",
    "sh",
    "zsh",
    "ruby",
    "go",
    "cargo",
    "docker",
    "kubectl",
    "curl",
    "wget",
    "git",
}


class SecurityEventType(StrEnum):
    """Types of security audit events."""

    VALIDATION_FAILURE = "validation_failure"
    PATH_TRAVERSAL_ATTEMPT = "path_traversal_attempt"
    SSRF_ATTEMPT = "ssrf_attempt"
    SECRET_REDACTED = "secret_redacted"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    SUSPICIOUS_INPUT = "suspicious_input"
    SANDBOX_VIOLATION = "sandbox_violation"


# =============================================================================
# Validation Exceptions
# =============================================================================


class SecurityValidationError(Exception):
    """Raised when security validation fails."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: str | None = None,
        event_type: SecurityEventType | None = None,
    ) -> None:
        self.message = message
        self.field = field
        self.value = redact_secrets(str(value)) if value else None
        self.event_type = event_type or SecurityEventType.VALIDATION_FAILURE
        super().__init__(self._format_message())
        # Log security event
        log_security_event(self.event_type, self.message, field=self.field)

    def _format_message(self) -> str:
        parts = [self.message]
        if self.field:
            parts.append(f"Field: {self.field}")
        if self.value:
            # Truncate value to prevent log injection
            truncated = self.value[:100].replace("\n", "\\n").replace("\r", "\\r")
            parts.append(f"Value: {truncated}...")
        return " | ".join(parts)


# =============================================================================
# Security Audit Logging
# =============================================================================


def log_security_event(
    event_type: SecurityEventType,
    message: str,
    field: str | None = None,
    source_ip: str | None = None,
    user: str | None = None,
    additional_data: dict[str, Any] | None = None,
) -> None:
    """
    Log a security-relevant event for audit purposes.

    Args:
        event_type: Type of security event.
        message: Description of the event.
        field: Field that triggered the event.
        source_ip: Source IP address if available.
        user: User identifier if available.
        additional_data: Additional context data.
    """
    redacted_message = redact_secrets(message)
    event_data: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type.value,
        "message": redacted_message,
        "field": field,
        "source_ip": source_ip,
        "user": user,
    }
    if additional_data:
        event_data["additional_data"] = redact_dict_secrets(additional_data)

    # Log at WARNING level for security events
    audit_logger.warning(
        "Security event: %s - %s",
        event_type.value,
        redacted_message,
        extra={"security_event": event_data},
    )


# =============================================================================
# Input Size and Depth Validation
# =============================================================================


def validate_request_size(data: bytes | str, max_size: int | None = None) -> None:
    """
    Validate that request data does not exceed size limits.

    Args:
        data: Request data as bytes or string.
        max_size: Maximum allowed size in bytes. Uses MAX_REQUEST_SIZE_BYTES if None.

    Raises:
        SecurityValidationError: If size exceeds limit.
    """
    max_bytes = max_size or MAX_REQUEST_SIZE_BYTES
    size = len(data.encode("utf-8") if isinstance(data, str) else data)

    if size > max_bytes:
        raise SecurityValidationError(
            f"Request size ({size} bytes) exceeds maximum ({max_bytes} bytes)",
            field="request",
            event_type=SecurityEventType.RESOURCE_LIMIT_EXCEEDED,
        )


def validate_object_depth(
    obj: Any,
    max_depth: int | None = None,
    current_depth: int = 0,
) -> None:
    """
    Validate that nested object depth does not exceed limits.

    Prevents denial of service through deeply nested structures.

    Args:
        obj: Object to validate.
        max_depth: Maximum allowed depth. Uses MAX_OBJECT_DEPTH if None.
        current_depth: Current recursion depth (internal).

    Raises:
        SecurityValidationError: If depth exceeds limit.
    """
    max_allowed = max_depth or MAX_OBJECT_DEPTH

    if current_depth > max_allowed:
        raise SecurityValidationError(
            f"Object nesting depth ({current_depth}) exceeds maximum ({max_allowed})",
            field="object",
            event_type=SecurityEventType.SUSPICIOUS_INPUT,
        )

    if isinstance(obj, dict):
        for value in obj.values():
            validate_object_depth(value, max_allowed, current_depth + 1)
    elif isinstance(obj, list):
        if len(obj) > MAX_ARRAY_SIZE:
            raise SecurityValidationError(
                f"Array size ({len(obj)}) exceeds maximum ({MAX_ARRAY_SIZE})",
                field="array",
                event_type=SecurityEventType.SUSPICIOUS_INPUT,
            )
        for item in obj:
            validate_object_depth(item, max_allowed, current_depth + 1)


def validate_artifacts_count(count: int) -> None:
    """
    Validate that artifacts count does not exceed limit.

    Args:
        count: Number of artifacts.

    Raises:
        SecurityValidationError: If count exceeds limit.
    """
    if count > MAX_ARTIFACTS_COUNT:
        raise SecurityValidationError(
            f"Artifacts count ({count}) exceeds maximum ({MAX_ARTIFACTS_COUNT})",
            field="artifacts",
            event_type=SecurityEventType.RESOURCE_LIMIT_EXCEEDED,
        )


def validate_env_vars_count(count: int) -> None:
    """
    Validate that environment variables count does not exceed limit.

    Args:
        count: Number of environment variables.

    Raises:
        SecurityValidationError: If count exceeds limit.
    """
    if count > MAX_ENV_VARS_COUNT:
        raise SecurityValidationError(
            f"Environment variables count ({count}) exceeds maximum "
            f"({MAX_ENV_VARS_COUNT})",
            field="environment",
            event_type=SecurityEventType.RESOURCE_LIMIT_EXCEEDED,
        )


# =============================================================================
# Path Validation
# =============================================================================


def validate_path_within_workspace(
    path: str | Path,
    workspace: Path,
    allow_absolute: bool = False,
) -> Path:
    """
    Validate that a path is safely within a workspace directory.

    Prevents path traversal attacks (../, symlinks to external dirs).

    Args:
        path: Path to validate (relative or absolute).
        workspace: Base workspace directory.
        allow_absolute: Whether to allow absolute paths
            (still must be within workspace).

    Returns:
        Resolved Path object within workspace.

    Raises:
        SecurityValidationError: If path escapes workspace or is invalid.
    """
    if not path:
        raise SecurityValidationError("Path cannot be empty", field="path")

    path_str = str(path)

    # Check path length
    if len(path_str) > MAX_PATH_LENGTH:
        raise SecurityValidationError(
            f"Path exceeds maximum length ({MAX_PATH_LENGTH} chars)",
            field="path",
        )

    # Convert to Path objects
    path_obj = Path(path_str)
    workspace = Path(workspace).resolve()

    # Reject absolute paths unless explicitly allowed
    if path_obj.is_absolute() and not allow_absolute:
        raise SecurityValidationError(
            "Absolute paths are not allowed",
            field="path",
            value=path_str,
        )

    # Check for obvious traversal attempts
    path_parts = path_str.replace("\\", "/").split("/")
    if ".." in path_parts:
        raise SecurityValidationError(
            "Path traversal (..) is not allowed",
            field="path",
            value=path_str,
        )

    # Check for null bytes (security risk in C-based systems)
    if "\x00" in path_str:
        raise SecurityValidationError(
            "Null bytes in path are not allowed",
            field="path",
        )

    # Resolve the full path
    if path_obj.is_absolute():
        resolved = path_obj.resolve()
    else:
        resolved = (workspace / path_obj).resolve()

    # Ensure resolved path is within workspace
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise SecurityValidationError(
            "Path escapes workspace directory",
            field="path",
            value=path_str,
        )

    # Check for symlinks BEFORE resolving (resolved path won't be symlink)
    if path_obj.is_symlink():
        target = path_obj.resolve()
        try:
            target.relative_to(workspace)
        except ValueError:
            log_security_event(
                SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
                f"Symlink escape attempt: {path_str}",
                field="path",
            )
            raise SecurityValidationError(
                "Symlink target escapes workspace directory",
                field="path",
                value=path_str,
                event_type=SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
            )

    return resolved


def open_file_safely(
    path: Path,
    mode: str = "r",
    workspace: Path | None = None,
) -> Any:
    """
    Open a file safely with TOCTOU protection.

    Uses O_NOFOLLOW on Unix to prevent symlink race conditions.

    Args:
        path: Path to open.
        mode: File open mode.
        workspace: If provided, validates path is within workspace.

    Returns:
        Open file handle.

    Raises:
        SecurityValidationError: If path validation fails.
        OSError: If file cannot be opened.
    """
    if workspace:
        validate_path_within_workspace(path, workspace, allow_absolute=True)

    # Use os.open with O_NOFOLLOW to prevent TOCTOU via symlinks
    flags = os.O_RDONLY
    if "w" in mode:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    elif "a" in mode:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    elif "x" in mode:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL

    # Add O_NOFOLLOW on Unix to prevent symlink following
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        fd = os.open(str(path), flags, mode=0o600)
        return os.fdopen(fd, mode.replace("b", "") or "r")
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise SecurityValidationError(
                "Cannot open file: symbolic link loop detected",
                field="path",
                value=str(path),
                event_type=SecurityEventType.SANDBOX_VIOLATION,
            ) from e
        raise


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path injection.

    Args:
        filename: Raw filename to sanitize.

    Returns:
        Safe filename with dangerous characters removed.
    """
    if not filename:
        return "unnamed"

    # Remove path separators and null bytes
    safe = filename.replace("/", "_").replace("\\", "_").replace("\x00", "")

    # Remove leading dots (hidden files on Unix)
    safe = safe.lstrip(".")

    # Remove or replace other problematic characters
    # Keep only alphanumeric, underscore, hyphen, dot
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", safe)

    # Limit length
    if len(safe) > 255:
        safe = safe[:255]

    # Ensure non-empty
    return safe or "unnamed"


# =============================================================================
# URL Validation
# =============================================================================


def validate_url(
    url: str,
    allow_internal: bool = False,
    allowed_schemes: set[str] | None = None,
) -> str:
    """
    Validate a URL for security.

    Prevents SSRF attacks by blocking internal IPs and metadata endpoints.

    Args:
        url: URL to validate.
        allow_internal: Whether to allow internal/private IPs.
        allowed_schemes: Set of allowed URL schemes. Defaults to http/https.

    Returns:
        Validated URL string.

    Raises:
        SecurityValidationError: If URL is invalid or unsafe.
    """
    if not url:
        raise SecurityValidationError("URL cannot be empty", field="url")

    schemes = allowed_schemes or ALLOWED_URL_SCHEMES

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise SecurityValidationError(
            f"Invalid URL format: {e}",
            field="url",
            value=url,
        )

    # Check scheme
    if not parsed.scheme:
        raise SecurityValidationError(
            "URL must have a scheme (e.g., https://)",
            field="url",
            value=url,
        )

    if parsed.scheme.lower() not in schemes:
        raise SecurityValidationError(
            f"URL scheme not allowed. Allowed: {schemes}",
            field="url",
            value=url,
        )

    # Check host
    if not parsed.hostname:
        raise SecurityValidationError(
            "URL must have a hostname",
            field="url",
            value=url,
        )

    hostname = parsed.hostname.lower()

    # Block metadata endpoints
    if hostname in METADATA_ENDPOINTS:
        raise SecurityValidationError(
            "Access to cloud metadata endpoints is blocked",
            field="url",
            value=url,
        )

    # Check for internal IPs
    if not allow_internal:
        try:
            ip = ipaddress.ip_address(hostname)
            for network in INTERNAL_IP_RANGES:
                if ip in network:
                    raise SecurityValidationError(
                        "Access to internal/private IPs is blocked",
                        field="url",
                        value=url,
                    )
        except ValueError:
            # Not an IP address, it's a hostname - that's fine
            pass

        # Block localhost variants
        if hostname in ("localhost", "localhost.localdomain"):
            log_security_event(
                SecurityEventType.SSRF_ATTEMPT,
                f"Localhost access attempt: {hostname}",
                field="url",
            )
            raise SecurityValidationError(
                "Access to localhost is blocked",
                field="url",
                value=url,
                event_type=SecurityEventType.SSRF_ATTEMPT,
            )

    return url


def validate_url_with_dns(
    url: str,
    allow_internal: bool = False,
    allowed_schemes: set[str] | None = None,
) -> str:
    """
    Validate a URL with DNS resolution check.

    Performs DNS resolution to detect if hostname resolves to internal IP,
    mitigating DNS rebinding and TOCTOU attacks.

    Args:
        url: URL to validate.
        allow_internal: Whether to allow internal/private IPs.
        allowed_schemes: Set of allowed URL schemes.

    Returns:
        Validated URL string.

    Raises:
        SecurityValidationError: If URL is invalid or resolves to unsafe IP.
    """
    # First perform standard validation
    validated = validate_url(url, allow_internal, allowed_schemes)

    if allow_internal:
        return validated

    # Now resolve DNS and check resolved IP
    parsed = urlparse(validated)
    hostname = parsed.hostname

    if not hostname:
        return validated

    try:
        # Skip IP addresses - already validated
        try:
            ipaddress.ip_address(hostname)
            return validated
        except ValueError:
            pass

        # Resolve hostname
        resolved_ips = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
        for family, _, _, _, sockaddr in resolved_ips:
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
                for network in INTERNAL_IP_RANGES:
                    if ip in network:
                        log_security_event(
                            SecurityEventType.SSRF_ATTEMPT,
                            f"DNS resolution to internal IP: {hostname} -> {ip_str}",
                            field="url",
                        )
                        raise SecurityValidationError(
                            f"Hostname {hostname} resolves to internal IP {ip_str}",
                            field="url",
                            value=url,
                            event_type=SecurityEventType.SSRF_ATTEMPT,
                        )
            except ValueError:
                continue

    except socket.gaierror:
        # DNS resolution failed - this is not a security issue
        pass

    return validated


# =============================================================================
# Docker Validation
# =============================================================================


def validate_docker_image(image: str) -> str:
    """
    Validate a Docker image name.

    Args:
        image: Docker image name (e.g., "nginx:latest", "registry.io/image:tag").

    Returns:
        Validated image name.

    Raises:
        SecurityValidationError: If image name is invalid.
    """
    if not image:
        raise SecurityValidationError(
            "Docker image name cannot be empty", field="image"
        )

    if len(image) > 256:
        raise SecurityValidationError(
            "Docker image name exceeds maximum length",
            field="image",
        )

    if not DOCKER_IMAGE_PATTERN.match(image):
        raise SecurityValidationError(
            "Invalid Docker image name format",
            field="image",
            value=image,
        )

    return image


def validate_docker_network(network: str) -> str:
    """
    Validate Docker network mode.

    Args:
        network: Network mode string.

    Returns:
        Validated network mode.

    Raises:
        SecurityValidationError: If network mode is invalid.
    """
    if not network:
        return "none"  # Default to isolated

    if network.lower() not in VALID_NETWORK_MODES:
        raise SecurityValidationError(
            f"Invalid network mode. Allowed: {VALID_NETWORK_MODES}",
            field="network",
            value=network,
        )

    return network.lower()


def validate_volume_mount(
    host_path: str,
    container_path: str,
    allowed_base_paths: list[Path] | None = None,
) -> tuple[str, str]:
    """
    Validate Docker volume mount paths.

    Args:
        host_path: Path on host system.
        container_path: Path inside container.
        allowed_base_paths: List of allowed base paths for host mounts.

    Returns:
        Tuple of (validated_host_path, validated_container_path).

    Raises:
        SecurityValidationError: If mount is unsafe.
    """
    if not host_path or not container_path:
        raise SecurityValidationError(
            "Volume mount paths cannot be empty",
            field="volume",
        )

    host_resolved = Path(host_path).resolve()

    # Check against allowed base paths
    if allowed_base_paths:
        allowed = False
        for base in allowed_base_paths:
            try:
                host_resolved.relative_to(base.resolve())
                allowed = True
                break
            except ValueError:
                continue

        if not allowed:
            raise SecurityValidationError(
                "Host path not in allowed mount locations",
                field="volume",
                value=host_path,
            )

    # Block dangerous container paths
    dangerous_paths = {"/", "/etc", "/root", "/proc", "/sys", "/dev", "/var/run"}
    if container_path.rstrip("/") in dangerous_paths:
        raise SecurityValidationError(
            "Mount to system path is not allowed",
            field="volume",
            value=container_path,
        )

    return str(host_resolved), container_path


# =============================================================================
# Resource Limits Validation
# =============================================================================


def parse_memory_limit(memory: str) -> int:
    """
    Parse and validate a memory limit string.

    Args:
        memory: Memory limit string (e.g., "2g", "512m", "1Gi").

    Returns:
        Memory limit in bytes.

    Raises:
        SecurityValidationError: If memory format is invalid or exceeds max.
    """
    if not memory:
        raise SecurityValidationError("Memory limit cannot be empty", field="memory")

    memory = memory.strip().lower()

    # Parse value and unit
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(b|k|m|g|ki|mi|gi)?$", memory, re.I)
    if not match:
        raise SecurityValidationError(
            "Invalid memory format. Use: 512m, 2g, 1Gi, etc.",
            field="memory",
            value=memory,
        )

    value = float(match.group(1))
    unit = (match.group(2) or "b").lower()

    # Convert to bytes
    multipliers = {
        "b": 1,
        "k": 1024,
        "m": 1024**2,
        "g": 1024**3,
        "ki": 1024,
        "mi": 1024**2,
        "gi": 1024**3,
    }

    bytes_value = int(value * multipliers.get(unit, 1))

    if bytes_value <= 0:
        raise SecurityValidationError(
            "Memory limit must be positive",
            field="memory",
            value=memory,
        )

    if bytes_value > MAX_MEMORY_BYTES:
        raise SecurityValidationError(
            f"Memory limit exceeds maximum ({MAX_MEMORY_BYTES // (1024**3)}GB)",
            field="memory",
            value=memory,
        )

    return bytes_value


def validate_cpu_limit(cpu: str) -> float:
    """
    Validate a CPU limit string.

    Args:
        cpu: CPU limit string (e.g., "1", "0.5", "2").

    Returns:
        CPU limit as float.

    Raises:
        SecurityValidationError: If CPU format is invalid or exceeds max.
    """
    if not cpu:
        raise SecurityValidationError("CPU limit cannot be empty", field="cpu")

    try:
        cpu_value = float(cpu)
    except ValueError:
        raise SecurityValidationError(
            "Invalid CPU format. Use: 1, 0.5, 2, etc.",
            field="cpu",
            value=cpu,
        )

    if cpu_value <= 0:
        raise SecurityValidationError(
            "CPU limit must be positive",
            field="cpu",
            value=cpu,
        )

    if cpu_value > MAX_CPU_CORES:
        raise SecurityValidationError(
            f"CPU limit exceeds maximum ({MAX_CPU_CORES} cores)",
            field="cpu",
            value=cpu,
        )

    return cpu_value


def validate_timeout(timeout: int | float) -> int:
    """
    Validate a timeout value.

    Args:
        timeout: Timeout in seconds.

    Returns:
        Validated timeout as integer seconds.

    Raises:
        SecurityValidationError: If timeout is invalid.
    """
    if timeout <= 0:
        raise SecurityValidationError(
            "Timeout must be positive",
            field="timeout",
            value=str(timeout),
        )

    if timeout > MAX_TIMEOUT_SECONDS:
        raise SecurityValidationError(
            f"Timeout exceeds maximum ({MAX_TIMEOUT_SECONDS}s)",
            field="timeout",
            value=str(timeout),
        )

    return int(timeout)


# =============================================================================
# Secret Handling
# =============================================================================


def redact_secrets(text: str) -> str:
    """
    Redact potential secrets from text.

    Args:
        text: Text that may contain secrets.

    Returns:
        Text with secrets replaced by [REDACTED].
    """
    if not text:
        return text

    result = text
    for name, pattern, replacement in SECRET_PATTERNS:
        if replacement is None:
            # Replace entire match
            result = pattern.sub(REDACTED, result)
        else:
            # Use replacement template
            result = pattern.sub(replacement, result)

    return result


def is_sensitive_env_var(name: str) -> bool:
    """
    Check if an environment variable name appears to be sensitive.

    Args:
        name: Environment variable name.

    Returns:
        True if the variable appears to contain sensitive data.
    """
    if not name:
        return False

    for pattern in SENSITIVE_ENV_PATTERNS:
        if pattern.match(name):
            return True

    return False


def filter_environment_variables(
    env: dict[str, str] | None = None,
    additional_allowlist: set[str] | None = None,
    additional_blocklist: set[str] | None = None,
    sanitize_values: bool = True,
) -> dict[str, str]:
    """
    Filter environment variables to remove sensitive data.

    Args:
        env: Environment variables to filter. Uses os.environ if None.
        additional_allowlist: Additional safe variable names to allow.
        additional_blocklist: Additional variable names to block.
        sanitize_values: Whether to sanitize values for shell safety.

    Returns:
        Filtered environment variables.
    """
    source = env if env is not None else dict(os.environ)

    # Validate count
    validate_env_vars_count(len(source))

    allowlist = SAFE_ENV_ALLOWLIST.copy()
    if additional_allowlist:
        allowlist.update(additional_allowlist)

    blocklist = additional_blocklist or set()

    filtered: dict[str, str] = {}
    for name, value in source.items():
        # Skip if explicitly blocked
        if name in blocklist:
            logger.debug("Filtered blocked env var: %s", name)
            continue

        # Allow if in allowlist
        if name in allowlist:
            if sanitize_values:
                value = sanitize_env_value(value)
            filtered[name] = value
            continue

        # Skip if appears sensitive
        if is_sensitive_env_var(name):
            logger.debug("Filtered sensitive env var: %s", name)
            continue

        # Allow non-sensitive variables
        if sanitize_values:
            value = sanitize_env_value(value)
        filtered[name] = value

    return filtered


def sanitize_env_value(value: str) -> str:
    """
    Sanitize an environment variable value for safe subprocess use.

    Removes or escapes potentially dangerous characters that could
    lead to command injection in shell contexts.

    Args:
        value: Environment variable value.

    Returns:
        Sanitized value.
    """
    if not value:
        return value

    # Check for null bytes
    if "\x00" in value:
        raise SecurityValidationError(
            "Null bytes not allowed in environment variable values",
            field="environment_value",
            event_type=SecurityEventType.SUSPICIOUS_INPUT,
        )

    # Remove newlines that could inject commands
    sanitized = value.replace("\n", " ").replace("\r", " ")

    # Truncate extremely long values
    max_env_value_length = 32768  # 32KB
    if len(sanitized) > max_env_value_length:
        sanitized = sanitized[:max_env_value_length]
        logger.warning(
            "Environment variable value truncated to %d chars", max_env_value_length
        )

    return sanitized


def redact_dict_secrets(data: dict[str, Any], max_depth: int = 5) -> dict[str, Any]:
    """
    Recursively redact secrets from a dictionary.

    Args:
        data: Dictionary to redact.
        max_depth: Maximum recursion depth.

    Returns:
        Dictionary with secrets redacted.
    """
    if max_depth <= 0:
        return data

    result: dict[str, Any] = {}
    for key, value in data.items():
        # Check if key suggests sensitive data
        key_lower = key.lower()
        is_sensitive_key = any(
            s in key_lower
            for s in ("key", "token", "secret", "password", "auth", "credential", "pwd")
        )

        if is_sensitive_key and isinstance(value, str):
            result[key] = REDACTED
        elif isinstance(value, str):
            result[key] = redact_secrets(value)
        elif isinstance(value, dict):
            result[key] = redact_dict_secrets(value, max_depth - 1)
        elif isinstance(value, list):
            result[key] = [
                redact_dict_secrets(item, max_depth - 1)
                if isinstance(item, dict)
                else redact_secrets(item)
                if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            result[key] = value

    return result


# =============================================================================
# Input Validation
# =============================================================================


def validate_task_description(description: str) -> str:
    """
    Validate a task description.

    Args:
        description: Task description text.

    Returns:
        Validated description.

    Raises:
        SecurityValidationError: If description is invalid.
    """
    if not description or not description.strip():
        raise SecurityValidationError(
            "Task description cannot be empty",
            field="description",
        )

    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise SecurityValidationError(
            f"Task description exceeds maximum length ({MAX_DESCRIPTION_LENGTH} chars)",
            field="description",
        )

    return description.strip()


def validate_command(
    command: str,
    allowed_commands: set[str] | None = None,
    block_shell_metacharacters: bool = True,
) -> str:
    """
    Validate a command for CLI execution.

    Args:
        command: Command string to validate.
        allowed_commands: Set of allowed command basenames. Uses SAFE_COMMAND_BINARIES
            if None.
        block_shell_metacharacters: Whether to block shell metacharacters.

    Returns:
        Validated command string.

    Raises:
        SecurityValidationError: If command is invalid or unsafe.
    """
    if not command or not command.strip():
        raise SecurityValidationError(
            "Command cannot be empty",
            field="command",
        )

    command = command.strip()

    # Check for null bytes
    if "\x00" in command:
        raise SecurityValidationError(
            "Null bytes not allowed in command",
            field="command",
            event_type=SecurityEventType.SUSPICIOUS_INPUT,
        )

    # Block dangerous shell metacharacters if requested
    dangerous_chars = {";", "&", "|", "`", "$", "(", ")", "{", "}", "<", ">", "\n"}
    if block_shell_metacharacters:
        for char in dangerous_chars:
            if char in command:
                log_security_event(
                    SecurityEventType.SUSPICIOUS_INPUT,
                    f"Shell metacharacter in command: {char}",
                    field="command",
                )
                raise SecurityValidationError(
                    f"Shell metacharacter '{char}' not allowed in command",
                    field="command",
                    value=command[:50],
                    event_type=SecurityEventType.SUSPICIOUS_INPUT,
                )

    # Validate command binary against allowlist
    allowed = allowed_commands or SAFE_COMMAND_BINARIES
    cmd_parts = command.split()
    if cmd_parts:
        cmd_binary = os.path.basename(cmd_parts[0])
        if cmd_binary not in allowed:
            log_security_event(
                SecurityEventType.VALIDATION_FAILURE,
                f"Command binary not in allowlist: {cmd_binary}",
                field="command",
            )
            raise SecurityValidationError(
                f"Command binary '{cmd_binary}' not in allowed list. "
                f"Allowed: {sorted(allowed)[:10]}...",
                field="command",
                value=cmd_binary,
            )

    return command


def escape_shell_arg(arg: str) -> str:
    """
    Escape a string for safe use as a shell argument.

    Args:
        arg: Argument to escape.

    Returns:
        Escaped argument safe for shell use.
    """
    import shlex

    return shlex.quote(arg)


def validate_task_id(task_id: str) -> str:
    """
    Validate a task ID.

    Args:
        task_id: Task identifier.

    Returns:
        Validated task ID.

    Raises:
        SecurityValidationError: If task ID is invalid.
    """
    if not task_id or not task_id.strip():
        raise SecurityValidationError(
            "Task ID cannot be empty",
            field="task_id",
        )

    # Only allow safe characters in task IDs
    if not re.match(r"^[a-zA-Z0-9_-]+$", task_id):
        raise SecurityValidationError(
            "Task ID contains invalid characters. "
            "Use only alphanumeric, underscore, hyphen.",
            field="task_id",
            value=task_id,
        )

    if len(task_id) > 128:
        raise SecurityValidationError(
            "Task ID exceeds maximum length (128 chars)",
            field="task_id",
            value=task_id,
        )

    return task_id


# =============================================================================
# Logging Utilities
# =============================================================================


class SecureFormatter(logging.Formatter):
    """Logging formatter that redacts secrets and sanitizes messages."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with secret redaction and sanitization."""
        message = super().format(record)
        return sanitize_log_message(redact_secrets(message))


def sanitize_log_message(message: str) -> str:
    """
    Sanitize a log message to prevent log injection attacks.

    Args:
        message: Log message to sanitize.

    Returns:
        Sanitized message.
    """
    if not message:
        return message

    # Replace control characters that could manipulate log output
    sanitized = message.replace("\r", "\\r").replace("\n", "\\n")

    # Remove ANSI escape sequences
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
    sanitized = ansi_pattern.sub("", sanitized)

    # Truncate very long messages
    max_log_length = 10000
    if len(sanitized) > max_log_length:
        sanitized = sanitized[: max_log_length - 20] + "... [TRUNCATED]"

    return sanitized


def sanitize_error_message(error: str | Exception, include_type: bool = True) -> str:
    """
    Sanitize an error message for safe display/logging.

    Removes potentially sensitive information like:
    - File paths (except relative paths within workspace)
    - Internal details and stack traces
    - Secret values

    Args:
        error: Error message or exception.
        include_type: Whether to include exception type.

    Returns:
        Sanitized error message.
    """
    if isinstance(error, Exception):
        error_str = str(error)
        error_type = type(error).__name__
    else:
        error_str = str(error)
        error_type = "Error"

    # Redact secrets
    sanitized = redact_secrets(error_str)

    # Remove absolute file paths (but keep relative paths)
    # This pattern matches Unix and Windows absolute paths
    path_pattern = re.compile(r"(/[a-zA-Z0-9_.-]+){3,}|[A-Z]:\\[^:]*")
    sanitized = path_pattern.sub("[PATH_REDACTED]", sanitized)

    # Truncate long messages
    max_error_length = 1000
    if len(sanitized) > max_error_length:
        sanitized = sanitized[: max_error_length - 20] + "... [TRUNCATED]"

    if include_type:
        return f"{error_type}: {sanitized}"
    return sanitized


def setup_secure_logging(
    logger_instance: logging.Logger | None = None,
    include_audit_logger: bool = True,
) -> None:
    """
    Configure secure logging with secret redaction.

    Args:
        logger_instance: Logger to configure. Configures root logger if None.
        include_audit_logger: Whether to also configure the audit logger.
    """
    target = logger_instance or logging.getLogger()
    formatter = SecureFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    for handler in target.handlers:
        handler.setFormatter(formatter)

    # Also configure audit logger
    if include_audit_logger:
        audit_formatter = SecureFormatter(
            "%(asctime)s - SECURITY_AUDIT - %(levelname)s - %(message)s"
        )
        for handler in audit_logger.handlers:
            handler.setFormatter(audit_formatter)

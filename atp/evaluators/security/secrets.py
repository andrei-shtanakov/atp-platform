"""Secret leak checker for detecting leaked secrets and credentials."""

import re
from typing import Any

from .base import (
    SecurityChecker,
    SecurityFinding,
    Severity,
    mask_api_key,
    mask_sensitive_data,
)


class SecretPattern:
    """A pattern for detecting a specific type of secret."""

    def __init__(
        self,
        name: str,
        pattern: re.Pattern[str],
        severity: Severity,
        description: str,
        remediation: str,
        mask_func: Any | None = None,
    ) -> None:
        """Initialize a secret pattern.

        Args:
            name: Name of the secret type (e.g., 'private_key', 'jwt_token').
            pattern: Compiled regex pattern for detection.
            severity: Severity level for findings of this type.
            description: Human-readable description of the secret type.
            remediation: Remediation suggestion for this finding.
            mask_func: Optional function to mask the detected value.
        """
        self.name = name
        self.pattern = pattern
        self.severity = severity
        self.description = description
        self.remediation = remediation
        self.mask_func = mask_func or mask_sensitive_data


# Environment variable patterns
ENV_VAR_PATTERNS = [
    # Common secret environment variables with values (with or without quotes)
    re.compile(
        r"\b(?:DB_PASSWORD|DATABASE_PASSWORD|MYSQL_PASSWORD|POSTGRES_PASSWORD|"
        r"REDIS_PASSWORD|MONGO_PASSWORD|SECRET_KEY|JWT_SECRET|SESSION_SECRET|"
        r"ENCRYPTION_KEY|PRIVATE_KEY|AWS_SECRET_ACCESS_KEY|AZURE_CLIENT_SECRET|"
        r"GCP_PRIVATE_KEY|SMTP_PASSWORD|EMAIL_PASSWORD|MAIL_PASSWORD|"
        r"OAUTH_CLIENT_SECRET|CLIENT_SECRET|APP_SECRET|AUTH_SECRET)"
        r"\s*[=:]\s*['\"]?([^'\"\n\s]{8,})['\"]?",
        re.IGNORECASE,
    ),
    # Generic password/secret assignments with quotes
    re.compile(
        r"\b(?:password|passwd|pwd|secret|token|credential)\s*[=:]\s*"
        r"['\"]([^'\"\n]{8,})['\"]",
        re.IGNORECASE,
    ),
    # Generic password/secret assignments without quotes (for env files)
    re.compile(
        r"\b(?:password|passwd|pwd|secret|api_?key|token)\s*=\s*"
        r"([^\s\n'\"]{8,})",
        re.IGNORECASE,
    ),
]

# Private key patterns
PRIVATE_KEY_PATTERNS = [
    # RSA private key
    re.compile(r"-----BEGIN\s+RSA\s+PRIVATE\s+KEY-----", re.IGNORECASE),
    # EC private key
    re.compile(r"-----BEGIN\s+EC\s+PRIVATE\s+KEY-----", re.IGNORECASE),
    # DSA private key
    re.compile(r"-----BEGIN\s+DSA\s+PRIVATE\s+KEY-----", re.IGNORECASE),
    # OpenSSH private key
    re.compile(r"-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----", re.IGNORECASE),
    # Generic private key
    re.compile(r"-----BEGIN\s+PRIVATE\s+KEY-----", re.IGNORECASE),
    # Encrypted private key
    re.compile(r"-----BEGIN\s+ENCRYPTED\s+PRIVATE\s+KEY-----", re.IGNORECASE),
    # PGP private key
    re.compile(r"-----BEGIN\s+PGP\s+PRIVATE\s+KEY\s+BLOCK-----", re.IGNORECASE),
]

# JWT token pattern
JWT_PATTERN = re.compile(
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
)

# Bearer token pattern
BEARER_PATTERN = re.compile(r"\b[Bb]earer\s+([A-Za-z0-9_\-./+=]{20,})\b")

# Basic auth pattern (base64 encoded credentials)
BASIC_AUTH_PATTERN = re.compile(r"\b[Bb]asic\s+([A-Za-z0-9+/]{20,}={0,2})\b")

# Connection string patterns
CONNECTION_STRING_PATTERNS = [
    # MongoDB connection string with password
    re.compile(
        r"mongodb(?:\+srv)?://[^:]+:([^@\s]{8,})@[^\s]+",
        re.IGNORECASE,
    ),
    # PostgreSQL/MySQL connection string with password
    re.compile(
        r"(?:postgres(?:ql)?|mysql)://[^:]+:([^@\s]{8,})@[^\s]+",
        re.IGNORECASE,
    ),
    # Redis connection string with password
    re.compile(
        r"redis://(?::[^@]+@|[^:]+:([^@\s]{8,})@)[^\s]+",
        re.IGNORECASE,
    ),
    # JDBC connection string with password
    re.compile(
        r"jdbc:[a-z]+://[^\s]+[?&]password=([^&\s]{8,})",
        re.IGNORECASE,
    ),
]

# Hardcoded password in code patterns
HARDCODED_PATTERNS = [
    # Variable assignments with password-like names
    re.compile(
        r"(?:const|let|var|def|val)\s+"
        r"(?:password|passwd|pwd|secret|api_?key|token|credential)\s*"
        r"[=:]\s*['\"]([^'\"\n]{8,})['\"]",
        re.IGNORECASE,
    ),
    # Function parameter defaults with passwords
    re.compile(
        r"(?:password|passwd|pwd|secret|api_?key|token)\s*"
        r"[=:]\s*['\"]([^'\"\n]{8,})['\"]",
        re.IGNORECASE,
    ),
]

# AWS-specific patterns
AWS_PATTERNS = [
    # AWS Access Key ID (already in PII checker, but include for completeness)
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # AWS Session Token
    re.compile(r"\bFwoGZXIvYXdzE[A-Za-z0-9/+=]{100,}\b"),
    # AWS MFA Token
    re.compile(r"\barn:aws:iam::\d{12}:mfa/\w+\b"),
]

# Certificate patterns
CERTIFICATE_PATTERNS = [
    # Private certificate (includes key material)
    re.compile(r"-----BEGIN\s+CERTIFICATE-----", re.IGNORECASE),
    # PKCS#12/PFX file reference (binary format, often contains private keys)
    re.compile(r"\.p12\b|\.pfx\b", re.IGNORECASE),
]


def mask_private_key(value: str) -> str:
    """Mask a private key header.

    Args:
        value: The private key header.

    Returns:
        Masked private key indicator.
    """
    return "-----BEGIN ***REDACTED*** PRIVATE KEY-----"


def mask_jwt(value: str) -> str:
    """Mask a JWT token.

    Args:
        value: The JWT token.

    Returns:
        Masked JWT with visible header.
    """
    parts = value.split(".")
    if len(parts) == 3:
        return f"{parts[0][:10]}...***...{parts[2][-10:]}"
    return mask_api_key(value)


def mask_connection_string(value: str) -> str:
    """Mask a connection string password.

    Args:
        value: The password portion of a connection string.

    Returns:
        Masked password.
    """
    if len(value) <= 4:
        return "***"
    return value[:2] + "***" + value[-2:]


class SecretLeakChecker(SecurityChecker):
    """Checker for leaked secrets and credentials.

    Detects the following types of secrets:
    - Private keys (RSA, EC, DSA, OpenSSH, PGP)
    - JWT tokens
    - Bearer tokens
    - Basic auth credentials
    - Environment variable secrets
    - Connection strings with embedded passwords
    - Hardcoded passwords in code
    - AWS credentials and session tokens
    - Certificate files
    """

    @property
    def name(self) -> str:
        """Return the checker name."""
        return "secret_leak"

    @property
    def check_types(self) -> list[str]:
        """Return the list of check types this checker supports."""
        return [
            "private_key",
            "jwt_token",
            "bearer_token",
            "basic_auth",
            "env_secret",
            "connection_string",
            "hardcoded_secret",
            "aws_credential",
            "certificate",
        ]

    def __init__(self) -> None:
        """Initialize the secret leak checker with patterns."""
        self._patterns: list[SecretPattern] = [
            # Private keys - Critical severity
            SecretPattern(
                name="private_key",
                pattern=pattern,
                severity=Severity.CRITICAL,
                description="Private key detected in output",
                remediation=(
                    "Remove private keys from output immediately. "
                    "Rotate the compromised key and generate a new one. "
                    "Store keys securely using a secrets manager."
                ),
                mask_func=mask_private_key,
            )
            for pattern in PRIVATE_KEY_PATTERNS
        ]

        # JWT tokens - High severity
        self._patterns.append(
            SecretPattern(
                name="jwt_token",
                pattern=JWT_PATTERN,
                severity=Severity.HIGH,
                description="JWT token detected in output",
                remediation=(
                    "Do not expose JWT tokens in logs or output. "
                    "If compromised, revoke the token and issue a new one. "
                    "Use short-lived tokens and secure token storage."
                ),
                mask_func=mask_jwt,
            )
        )

        # Bearer tokens - High severity
        self._patterns.append(
            SecretPattern(
                name="bearer_token",
                pattern=BEARER_PATTERN,
                severity=Severity.HIGH,
                description="Bearer token detected in output",
                remediation=(
                    "Remove bearer tokens from output. "
                    "Revoke compromised tokens immediately. "
                    "Use secure headers and avoid logging auth data."
                ),
                mask_func=mask_api_key,
            )
        )

        # Basic auth - High severity
        self._patterns.append(
            SecretPattern(
                name="basic_auth",
                pattern=BASIC_AUTH_PATTERN,
                severity=Severity.HIGH,
                description="Basic authentication credentials detected",
                remediation=(
                    "Remove basic auth credentials from output. "
                    "Change the compromised password. "
                    "Consider using more secure authentication methods."
                ),
                mask_func=mask_api_key,
            )
        )

        # Environment secrets - High severity
        for pattern in ENV_VAR_PATTERNS:
            self._patterns.append(
                SecretPattern(
                    name="env_secret",
                    pattern=pattern,
                    severity=Severity.HIGH,
                    description="Environment variable secret detected",
                    remediation=(
                        "Remove environment secrets from output. "
                        "Rotate the exposed secret value. "
                        "Use a secrets manager and never hardcode secrets."
                    ),
                    mask_func=mask_sensitive_data,
                )
            )

        # Connection strings - Critical severity
        for pattern in CONNECTION_STRING_PATTERNS:
            self._patterns.append(
                SecretPattern(
                    name="connection_string",
                    pattern=pattern,
                    severity=Severity.CRITICAL,
                    description="Database connection string with password detected",
                    remediation=(
                        "Remove connection strings from output. "
                        "Change the database password immediately. "
                        "Use environment variables or secrets managers for credentials."
                    ),
                    mask_func=mask_connection_string,
                )
            )

        # Hardcoded secrets - High severity
        for pattern in HARDCODED_PATTERNS:
            self._patterns.append(
                SecretPattern(
                    name="hardcoded_secret",
                    pattern=pattern,
                    severity=Severity.HIGH,
                    description="Hardcoded secret or password detected",
                    remediation=(
                        "Never hardcode secrets in code or output. "
                        "Use environment variables or a secrets manager. "
                        "Rotate any exposed credentials."
                    ),
                    mask_func=mask_sensitive_data,
                )
            )

        # AWS credentials - Critical severity
        for pattern in AWS_PATTERNS:
            self._patterns.append(
                SecretPattern(
                    name="aws_credential",
                    pattern=pattern,
                    severity=Severity.CRITICAL,
                    description="AWS credential detected",
                    remediation=(
                        "Immediately rotate AWS credentials. "
                        "Review CloudTrail for unauthorized access. "
                        "Use IAM roles instead of access keys where possible."
                    ),
                    mask_func=mask_api_key,
                )
            )

        # Certificates - Medium severity (may contain private keys)
        for pattern in CERTIFICATE_PATTERNS:
            self._patterns.append(
                SecretPattern(
                    name="certificate",
                    pattern=pattern,
                    severity=Severity.MEDIUM,
                    description="Certificate or key file reference detected",
                    remediation=(
                        "Verify no private key material is exposed. "
                        "Review certificate handling practices. "
                        "Store certificates securely."
                    ),
                    mask_func=mask_sensitive_data,
                )
            )

    def check(
        self,
        content: str,
        location: str | None = None,
        enabled_types: list[str] | None = None,
    ) -> list[SecurityFinding]:
        """Check content for secret leaks.

        Args:
            content: The content to scan for secrets.
            location: Optional location identifier (e.g., artifact path).
            enabled_types: Optional list of specific secret types to check.
                          If None, all types are checked.

        Returns:
            List of SecurityFinding objects for any secrets found.
        """
        findings: list[SecurityFinding] = []
        seen_values: set[str] = set()

        for secret_pattern in self._patterns:
            if enabled_types and secret_pattern.name not in enabled_types:
                continue

            for match in secret_pattern.pattern.finditer(content):
                # Get the captured group if available, otherwise the full match
                if match.lastindex:
                    value = match.group(1)
                else:
                    value = match.group(0)

                # Skip duplicates
                if value in seen_values:
                    continue
                seen_values.add(value)

                # Validate the match
                if not self._validate_match(secret_pattern.name, value):
                    continue

                # Extract context for better reporting
                context = self._extract_context(content, match.start(), match.end())

                findings.append(
                    SecurityFinding(
                        check_type="secret_leak",
                        finding_type=secret_pattern.name,
                        severity=secret_pattern.severity,
                        message=secret_pattern.description,
                        evidence_masked=secret_pattern.mask_func(value),
                        location=location,
                        details={
                            "pattern_type": secret_pattern.name,
                            "remediation": secret_pattern.remediation,
                            "context": context,
                        },
                    )
                )

        return findings

    def _validate_match(self, secret_type: str, value: str) -> bool:
        """Validate that a match is a real secret.

        Args:
            secret_type: The type of secret being validated.
            value: The matched value to validate.

        Returns:
            True if the match appears to be a valid secret.
        """
        # Skip common false positives
        if secret_type in ("env_secret", "hardcoded_secret"):
            # Skip placeholder values
            lower_value = value.lower()
            # Prefixes that indicate placeholder values
            placeholder_prefixes = [
                "your_",
                "my_",
                "example_",
                "example",
                "<your",
                "${",
                "{{",
            ]
            # Exact or near-exact placeholder values
            placeholder_values = [
                "example",
                "placeholder",
                "xxxxxxxx",
                "password123",
                "changeme",
                "secret123",
                "null",
                "none",
                "undefined",
            ]
            # Check prefixes
            if any(lower_value.startswith(p) for p in placeholder_prefixes):
                return False
            # Check if value is exactly or mostly a placeholder
            if any(
                lower_value == p or lower_value.rstrip("0123456789") == p
                for p in placeholder_values
            ):
                return False
            # Skip values that start with "test" followed by non-alphanumeric
            if lower_value.startswith("test") and (
                len(lower_value) == 4 or not lower_value[4].isalnum()
            ):
                return False

            # Skip very short values
            if len(value) < 8:
                return False

        if secret_type == "jwt_token":
            # Validate JWT structure
            parts = value.split(".")
            if len(parts) != 3:
                return False
            # Check that parts are base64-like
            valid_chars = (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
            )
            for part in parts:
                if not all(c in valid_chars for c in part):
                    return False

        return True

    def _extract_context(self, content: str, start: int, end: int) -> str:
        """Extract context around a match.

        Args:
            content: The full content.
            start: Match start position.
            end: Match end position.

        Returns:
            Context string with surrounding characters.
        """
        context_size = 20
        context_start = max(0, start - context_size)
        context_end = min(len(content), end + context_size)

        prefix = "..." if context_start > 0 else ""
        suffix = "..." if context_end < len(content) else ""

        context = content[context_start:context_end]
        # Remove newlines for cleaner display
        context = context.replace("\n", " ").replace("\r", "")

        return f"{prefix}{context}{suffix}"

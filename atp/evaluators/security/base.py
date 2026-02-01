"""Base classes for security checkers."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Severity levels for security findings."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def score_weight(self) -> float:
        """Return score weight for this severity level.

        Higher severity = lower score (more penalty).
        """
        weights = {
            Severity.INFO: 0.95,
            Severity.LOW: 0.8,
            Severity.MEDIUM: 0.5,
            Severity.HIGH: 0.2,
            Severity.CRITICAL: 0.0,
        }
        return weights[self]

    def __ge__(self, other: object) -> bool:
        """Compare severity levels."""
        if not isinstance(other, Severity):
            return NotImplemented
        order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]
        return order.index(self) >= order.index(other)

    def __gt__(self, other: object) -> bool:
        """Compare severity levels."""
        if not isinstance(other, Severity):
            return NotImplemented
        order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]
        return order.index(self) > order.index(other)

    def __le__(self, other: object) -> bool:
        """Compare severity levels."""
        if not isinstance(other, Severity):
            return NotImplemented
        order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]
        return order.index(self) <= order.index(other)

    def __lt__(self, other: object) -> bool:
        """Compare severity levels."""
        if not isinstance(other, Severity):
            return NotImplemented
        order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]
        return order.index(self) < order.index(other)


class SecurityFinding(BaseModel):
    """A security finding from a checker."""

    check_type: str = Field(
        ..., description="Type of security check (e.g., 'pii', 'api_key')"
    )
    finding_type: str = Field(
        ..., description="Specific finding type (e.g., 'email', 'ssn')"
    )
    severity: Severity = Field(..., description="Severity level of the finding")
    message: str = Field(
        ..., description="Human-readable message describing the finding"
    )
    evidence_masked: str = Field(
        ..., description="Masked evidence (e.g., 'j***@example.com')"
    )
    location: str | None = Field(
        None, description="Location where finding was detected (e.g., artifact path)"
    )
    details: dict[str, Any] | None = Field(None, description="Additional details")


def mask_sensitive_data(value: str, visible_chars: int = 4) -> str:
    """Mask sensitive data while keeping some characters visible.

    Args:
        value: The sensitive value to mask.
        visible_chars: Number of characters to keep visible at start and end.

    Returns:
        Masked string with asterisks replacing middle characters.
    """
    if len(value) <= visible_chars * 2:
        return "*" * len(value)

    middle_mask = "*" * (len(value) - visible_chars * 2)
    return value[:visible_chars] + middle_mask + value[-visible_chars:]


def mask_email(email: str) -> str:
    """Mask an email address while preserving format.

    Args:
        email: The email address to mask.

    Returns:
        Masked email (e.g., 'j***@example.com').
    """
    if "@" not in email:
        return mask_sensitive_data(email)

    local, domain = email.rsplit("@", 1)

    if len(local) <= 1:
        masked_local = "*"
    elif len(local) <= 3:
        masked_local = local[0] + "*" * (len(local) - 1)
    else:
        masked_local = local[0] + "*" * (len(local) - 2) + local[-1]

    return f"{masked_local}@{domain}"


def mask_credit_card(number: str) -> str:
    """Mask a credit card number showing only last 4 digits.

    Args:
        number: The credit card number to mask.

    Returns:
        Masked credit card (e.g., '****-****-****-1234').
    """
    digits_only = "".join(c for c in number if c.isdigit())
    if len(digits_only) < 4:
        return "*" * len(number)

    last_four = digits_only[-4:]
    return "*" * (len(digits_only) - 4) + last_four


def mask_ssn(ssn: str) -> str:
    """Mask an SSN showing only last 4 digits.

    Args:
        ssn: The SSN to mask.

    Returns:
        Masked SSN (e.g., '***-**-1234').
    """
    digits_only = "".join(c for c in ssn if c.isdigit())
    if len(digits_only) < 4:
        return "*" * len(ssn)

    last_four = digits_only[-4:]
    return "***-**-" + last_four


def mask_phone(phone: str) -> str:
    """Mask a phone number showing only last 4 digits.

    Args:
        phone: The phone number to mask.

    Returns:
        Masked phone (e.g., '(***) ***-1234').
    """
    digits_only = "".join(c for c in phone if c.isdigit())
    if len(digits_only) < 4:
        return "*" * len(phone)

    last_four = digits_only[-4:]
    return "***-***-" + last_four


def mask_api_key(key: str) -> str:
    """Mask an API key showing only first and last few characters.

    Args:
        key: The API key to mask.

    Returns:
        Masked API key (e.g., 'sk-***...***xyz').
    """
    if len(key) <= 8:
        return "*" * len(key)

    return key[:4] + "***...***" + key[-4:]


class SecurityChecker(ABC):
    """Abstract base class for security checkers.

    Security checkers scan content for specific types of security issues
    and return findings with severity levels and masked evidence.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the checker name."""

    @property
    @abstractmethod
    def check_types(self) -> list[str]:
        """Return the list of check types this checker supports."""

    @abstractmethod
    def check(
        self,
        content: str,
        location: str | None = None,
        enabled_types: list[str] | None = None,
    ) -> list[SecurityFinding]:
        """Check content for security issues.

        Args:
            content: The content to scan for security issues.
            location: Optional location identifier (e.g., artifact path).
            enabled_types: Optional list of specific check types to run.
                          If None, all check types are run.

        Returns:
            List of SecurityFinding objects for any issues found.
        """

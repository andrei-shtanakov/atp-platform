"""PII (Personally Identifiable Information) checker."""

import re
from typing import Any

from .base import (
    SecurityChecker,
    SecurityFinding,
    Severity,
    mask_api_key,
    mask_credit_card,
    mask_email,
    mask_phone,
    mask_ssn,
)


class PIIPattern:
    """A pattern for detecting a specific type of PII."""

    def __init__(
        self,
        name: str,
        pattern: re.Pattern[str],
        severity: Severity,
        mask_func: Any,
        description: str,
    ) -> None:
        """Initialize a PII pattern.

        Args:
            name: Name of the PII type (e.g., 'email', 'ssn').
            pattern: Compiled regex pattern for detection.
            severity: Severity level for findings of this type.
            mask_func: Function to mask the detected value.
            description: Human-readable description of the PII type.
        """
        self.name = name
        self.pattern = pattern
        self.severity = severity
        self.mask_func = mask_func
        self.description = description


# Email pattern - comprehensive email detection
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Phone patterns - various formats
# US phone: (123) 456-7890, 123-456-7890, 123.456.7890, +1 123 456 7890
PHONE_PATTERN = re.compile(
    r"(?:\+?1[-.\s]?)?"  # Optional country code
    r"(?:\(?\d{3}\)?[-.\s]?)"  # Area code
    r"\d{3}[-.\s]?"  # Exchange
    r"\d{4}\b"  # Subscriber
)

# SSN pattern - 123-45-6789 or 123456789
SSN_PATTERN = re.compile(
    r"\b(?!000|666|9\d{2})"  # Exclude invalid prefixes
    r"\d{3}[-\s]?"
    r"(?!00)\d{2}[-\s]?"
    r"(?!0000)\d{4}\b"
)

# Credit card patterns
# Visa: 4xxx
# Mastercard: 51-55xx or 2221-2720
# Amex: 34xx or 37xx
# Discover: 6011, 65xx, 644-649
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:"
    r"4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|"  # Visa
    r"(?:5[1-5]\d{2}|222[1-9]|22[3-9]\d|2[3-6]\d{2}|27[01]\d|2720)"  # Mastercard
    r"[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|"
    r"3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}|"  # Amex
    r"(?:6011|65\d{2}|64[4-9]\d)[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"  # Discover
    r")\b"
)

# API Key patterns - various common formats
# Note: Only the generic pattern uses a capture group to extract the key value.
# All other patterns use non-capturing groups and match the entire key.
API_KEY_PATTERNS = [
    # AWS Access Key ID
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # AWS Secret Access Key (40 char base64)
    re.compile(r"\b[A-Za-z0-9/+=]{40}\b"),
    # GitHub Personal Access Token (ghp_, gho_, ghu_, ghs_, ghr_)
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,255}\b"),
    # OpenAI API Key
    re.compile(r"\bsk-[A-Za-z0-9]{20,}T3BlbkFJ[A-Za-z0-9]{20,}\b"),
    # Generic API key patterns (key=, api_key=, apikey=, token=)
    # This pattern captures just the key value in group(1)
    re.compile(
        r"(?:api[_-]?key|apikey|api[_-]?token|access[_-]?token|auth[_-]?token|"
        r"secret[_-]?key|private[_-]?key|bearer)"
        r"[\"']?\s*[:=]\s*[\"']?"
        r"([A-Za-z0-9_\-./+=]{20,})"
        r"[\"']?",
        re.IGNORECASE,
    ),
    # Slack tokens
    re.compile(r"\bxox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*\b"),
    # Stripe API keys
    re.compile(r"\b(?:sk|pk)_(?:test|live)_[A-Za-z0-9]{24,}\b"),
    # Google API keys
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    # Anthropic API keys
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{40,}\b"),
]


class PIIChecker(SecurityChecker):
    """Checker for Personally Identifiable Information (PII).

    Detects the following types of PII:
    - Email addresses
    - Phone numbers (US format)
    - Social Security Numbers (SSN)
    - Credit card numbers (Visa, Mastercard, Amex, Discover)
    - API keys (AWS, GitHub, OpenAI, Slack, Stripe, Google, Anthropic)
    """

    @property
    def name(self) -> str:
        """Return the checker name."""
        return "pii"

    @property
    def check_types(self) -> list[str]:
        """Return the list of check types this checker supports."""
        return ["email", "phone", "ssn", "credit_card", "api_key"]

    def __init__(self) -> None:
        """Initialize the PII checker with patterns."""
        self._patterns: list[PIIPattern] = [
            PIIPattern(
                name="email",
                pattern=EMAIL_PATTERN,
                severity=Severity.MEDIUM,
                mask_func=mask_email,
                description="Email address detected",
            ),
            PIIPattern(
                name="phone",
                pattern=PHONE_PATTERN,
                severity=Severity.MEDIUM,
                mask_func=mask_phone,
                description="Phone number detected",
            ),
            PIIPattern(
                name="ssn",
                pattern=SSN_PATTERN,
                severity=Severity.CRITICAL,
                mask_func=mask_ssn,
                description="Social Security Number detected",
            ),
            PIIPattern(
                name="credit_card",
                pattern=CREDIT_CARD_PATTERN,
                severity=Severity.CRITICAL,
                mask_func=mask_credit_card,
                description="Credit card number detected",
            ),
        ]

    def check(
        self,
        content: str,
        location: str | None = None,
        enabled_types: list[str] | None = None,
    ) -> list[SecurityFinding]:
        """Check content for PII.

        Args:
            content: The content to scan for PII.
            location: Optional location identifier (e.g., artifact path).
            enabled_types: Optional list of specific PII types to check.
                          If None, all types are checked.

        Returns:
            List of SecurityFinding objects for any PII found.
        """
        findings: list[SecurityFinding] = []
        seen_values: set[str] = set()

        for pii_pattern in self._patterns:
            if enabled_types and pii_pattern.name not in enabled_types:
                continue

            for match in pii_pattern.pattern.finditer(content):
                value = match.group(0)

                if value in seen_values:
                    continue
                seen_values.add(value)

                if not self._validate_match(pii_pattern.name, value):
                    continue

                findings.append(
                    SecurityFinding(
                        check_type="pii",
                        finding_type=pii_pattern.name,
                        severity=pii_pattern.severity,
                        message=pii_pattern.description,
                        evidence_masked=pii_pattern.mask_func(value),
                        location=location,
                        details={"pattern_type": pii_pattern.name},
                    )
                )

        if enabled_types is None or "api_key" in enabled_types:
            findings.extend(self._check_api_keys(content, location, seen_values))

        return findings

    def _check_api_keys(
        self,
        content: str,
        location: str | None,
        seen_values: set[str],
    ) -> list[SecurityFinding]:
        """Check content for API keys.

        Args:
            content: The content to scan for API keys.
            location: Optional location identifier.
            seen_values: Set of values already seen to avoid duplicates.

        Returns:
            List of SecurityFinding objects for any API keys found.
        """
        findings: list[SecurityFinding] = []

        for pattern in API_KEY_PATTERNS:
            for match in pattern.finditer(content):
                if match.lastindex:
                    value = match.group(1)
                else:
                    value = match.group(0)

                if value in seen_values:
                    continue
                seen_values.add(value)

                if not self._is_likely_api_key(value):
                    continue

                findings.append(
                    SecurityFinding(
                        check_type="pii",
                        finding_type="api_key",
                        severity=Severity.HIGH,
                        message="API key or secret detected",
                        evidence_masked=mask_api_key(value),
                        location=location,
                        details={
                            "key_prefix": value[:4] if len(value) >= 4 else "****"
                        },
                    )
                )

        return findings

    def _validate_match(self, pii_type: str, value: str) -> bool:
        """Validate that a match is a real PII value.

        Args:
            pii_type: The type of PII being validated.
            value: The matched value to validate.

        Returns:
            True if the match appears to be valid PII.
        """
        if pii_type == "ssn":
            return self._validate_ssn(value)
        elif pii_type == "credit_card":
            return self._validate_credit_card(value)
        elif pii_type == "phone":
            return self._validate_phone(value)
        return True

    def _validate_ssn(self, value: str) -> bool:
        """Validate SSN format and structure.

        Args:
            value: The potential SSN string.

        Returns:
            True if the value appears to be a valid SSN format.
        """
        digits = "".join(c for c in value if c.isdigit())

        if len(digits) != 9:
            return False

        area = int(digits[:3])
        group = int(digits[3:5])
        serial = int(digits[5:])

        if area == 0 or area == 666 or (900 <= area <= 999):
            return False
        if group == 0:
            return False
        if serial == 0:
            return False

        return True

    def _validate_credit_card(self, value: str) -> bool:
        """Validate credit card using Luhn algorithm.

        Args:
            value: The potential credit card number.

        Returns:
            True if the value passes Luhn check.
        """
        digits = [int(c) for c in value if c.isdigit()]

        if len(digits) < 13 or len(digits) > 19:
            return False

        checksum = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit

        return checksum % 10 == 0

    def _validate_phone(self, value: str) -> bool:
        """Validate phone number.

        Args:
            value: The potential phone number.

        Returns:
            True if the value appears to be a valid phone number.
        """
        digits = "".join(c for c in value if c.isdigit())

        if len(digits) < 10 or len(digits) > 11:
            return False

        if len(digits) == 11 and digits[0] != "1":
            return False

        return True

    def _is_likely_api_key(self, value: str) -> bool:
        """Check if a value is likely an API key.

        Args:
            value: The potential API key.

        Returns:
            True if the value appears to be an API key.
        """
        if len(value) < 20:
            return False

        if value.lower() in {"authorization", "content-type", "application/json"}:
            return False

        has_upper = any(c.isupper() for c in value)
        has_lower = any(c.islower() for c in value)
        has_digit = any(c.isdigit() for c in value)

        entropy_indicators = sum([has_upper, has_lower, has_digit])
        if entropy_indicators < 2:
            return False

        return True

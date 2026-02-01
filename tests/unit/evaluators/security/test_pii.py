"""Unit tests for PII checker."""

import pytest

from atp.evaluators.security.base import Severity
from atp.evaluators.security.pii import PIIChecker


@pytest.fixture
def checker() -> PIIChecker:
    """Create PIIChecker instance."""
    return PIIChecker()


class TestPIICheckerProperties:
    """Tests for PIIChecker properties."""

    def test_checker_name(self, checker: PIIChecker) -> None:
        """Test checker name property."""
        assert checker.name == "pii"

    def test_checker_check_types(self, checker: PIIChecker) -> None:
        """Test checker check_types property."""
        types = checker.check_types
        assert "email" in types
        assert "phone" in types
        assert "ssn" in types
        assert "credit_card" in types
        assert "api_key" in types


class TestEmailDetection:
    """Tests for email detection."""

    def test_detect_simple_email(self, checker: PIIChecker) -> None:
        """Test detecting a simple email address."""
        content = "Contact me at john.doe@example.com for more info."
        findings = checker.check(content, enabled_types=["email"])
        assert len(findings) == 1
        assert findings[0].finding_type == "email"
        assert findings[0].severity == Severity.MEDIUM
        assert "@example.com" in findings[0].evidence_masked

    def test_detect_multiple_emails(self, checker: PIIChecker) -> None:
        """Test detecting multiple email addresses."""
        content = "Contact john@example.com or jane@company.org"
        findings = checker.check(content, enabled_types=["email"])
        assert len(findings) == 2

    def test_no_email_when_disabled(self, checker: PIIChecker) -> None:
        """Test no email detection when disabled."""
        content = "Contact john@example.com"
        findings = checker.check(content, enabled_types=["phone"])
        assert len(findings) == 0


class TestPhoneDetection:
    """Tests for phone number detection."""

    def test_detect_phone_with_dashes(self, checker: PIIChecker) -> None:
        """Test detecting phone with dashes."""
        content = "Call me at 555-123-4567"
        findings = checker.check(content, enabled_types=["phone"])
        assert len(findings) == 1
        assert findings[0].finding_type == "phone"
        assert findings[0].severity == Severity.MEDIUM

    def test_detect_phone_with_parens(self, checker: PIIChecker) -> None:
        """Test detecting phone with parentheses."""
        content = "Call me at (555) 123-4567"
        findings = checker.check(content, enabled_types=["phone"])
        assert len(findings) == 1

    def test_detect_phone_with_country_code(self, checker: PIIChecker) -> None:
        """Test detecting phone with country code."""
        content = "Call me at +1 555-123-4567"
        findings = checker.check(content, enabled_types=["phone"])
        assert len(findings) == 1

    def test_reject_invalid_phone(self, checker: PIIChecker) -> None:
        """Test rejecting invalid phone number."""
        content = "The code is 12345"
        findings = checker.check(content, enabled_types=["phone"])
        assert len(findings) == 0


class TestSSNDetection:
    """Tests for SSN detection."""

    def test_detect_ssn_with_dashes(self, checker: PIIChecker) -> None:
        """Test detecting SSN with dashes."""
        content = "My SSN is 123-45-6789"
        findings = checker.check(content, enabled_types=["ssn"])
        assert len(findings) == 1
        assert findings[0].finding_type == "ssn"
        assert findings[0].severity == Severity.CRITICAL
        assert "6789" in findings[0].evidence_masked

    def test_detect_ssn_no_dashes(self, checker: PIIChecker) -> None:
        """Test detecting SSN without dashes."""
        content = "SSN: 123456789"
        findings = checker.check(content, enabled_types=["ssn"])
        assert len(findings) == 1

    def test_reject_invalid_ssn_000_prefix(self, checker: PIIChecker) -> None:
        """Test rejecting SSN with 000 prefix."""
        content = "Invalid SSN: 000-12-3456"
        findings = checker.check(content, enabled_types=["ssn"])
        assert len(findings) == 0

    def test_reject_invalid_ssn_666_prefix(self, checker: PIIChecker) -> None:
        """Test rejecting SSN with 666 prefix."""
        content = "Invalid SSN: 666-12-3456"
        findings = checker.check(content, enabled_types=["ssn"])
        assert len(findings) == 0

    def test_reject_invalid_ssn_9xx_prefix(self, checker: PIIChecker) -> None:
        """Test rejecting SSN with 9xx prefix."""
        content = "Invalid SSN: 900-12-3456"
        findings = checker.check(content, enabled_types=["ssn"])
        assert len(findings) == 0

    def test_reject_invalid_ssn_00_group(self, checker: PIIChecker) -> None:
        """Test rejecting SSN with 00 group."""
        content = "Invalid SSN: 123-00-3456"
        findings = checker.check(content, enabled_types=["ssn"])
        assert len(findings) == 0

    def test_reject_invalid_ssn_0000_serial(self, checker: PIIChecker) -> None:
        """Test rejecting SSN with 0000 serial."""
        content = "Invalid SSN: 123-45-0000"
        findings = checker.check(content, enabled_types=["ssn"])
        assert len(findings) == 0


class TestCreditCardDetection:
    """Tests for credit card detection."""

    def test_detect_visa_card(self, checker: PIIChecker) -> None:
        """Test detecting Visa card number."""
        content = "Card: 4111-1111-1111-1111"
        findings = checker.check(content, enabled_types=["credit_card"])
        assert len(findings) == 1
        assert findings[0].finding_type == "credit_card"
        assert findings[0].severity == Severity.CRITICAL
        assert "1111" in findings[0].evidence_masked

    def test_detect_mastercard(self, checker: PIIChecker) -> None:
        """Test detecting Mastercard number."""
        content = "Card: 5555555555554444"
        findings = checker.check(content, enabled_types=["credit_card"])
        assert len(findings) == 1

    def test_detect_amex(self, checker: PIIChecker) -> None:
        """Test detecting American Express number."""
        content = "Card: 378282246310005"
        findings = checker.check(content, enabled_types=["credit_card"])
        assert len(findings) == 1

    def test_reject_invalid_luhn(self, checker: PIIChecker) -> None:
        """Test rejecting card that fails Luhn check."""
        content = "Invalid: 4111111111111112"
        findings = checker.check(content, enabled_types=["credit_card"])
        assert len(findings) == 0

    def test_detect_card_with_spaces(self, checker: PIIChecker) -> None:
        """Test detecting card with spaces."""
        content = "Card: 4111 1111 1111 1111"
        findings = checker.check(content, enabled_types=["credit_card"])
        assert len(findings) == 1


class TestAPIKeyDetection:
    """Tests for API key detection."""

    def test_detect_aws_access_key(self, checker: PIIChecker) -> None:
        """Test detecting AWS access key."""
        content = "AWS Key: AKIAIOSFODNN7EXAMPLE"
        findings = checker.check(content, enabled_types=["api_key"])
        assert len(findings) == 1
        assert findings[0].finding_type == "api_key"
        assert findings[0].severity == Severity.HIGH

    def test_detect_github_token(self, checker: PIIChecker) -> None:
        """Test detecting GitHub personal access token."""
        # GitHub tokens are 36+ alphanumeric chars after the prefix
        content = "Token: ghp_1234567890abcdefghijABCDEFGHIJ123456"
        findings = checker.check(content, enabled_types=["api_key"])
        assert len(findings) == 1

    def test_detect_stripe_key(self, checker: PIIChecker) -> None:
        """Test detecting Stripe API key."""
        # Stripe keys need 24+ chars after the prefix
        content = "API Key: " + "sk_test_" + "x" * 24
        findings = checker.check(content, enabled_types=["api_key"])
        assert len(findings) == 1

    def test_detect_google_api_key(self, checker: PIIChecker) -> None:
        """Test detecting Google API key."""
        # Google API keys are 35 chars after AIza prefix (total 39 chars)
        content = "API Key: AIzaSyC1234567890abcdefghij123456789012"
        findings = checker.check(content, enabled_types=["api_key"])
        assert len(findings) == 1

    def test_detect_generic_api_key(self, checker: PIIChecker) -> None:
        """Test detecting generic API key pattern."""
        content = 'config = {"api_key": "abc123def456ghi789jkl012mno345pqr678"}'
        findings = checker.check(content, enabled_types=["api_key"])
        assert len(findings) >= 1

    def test_detect_anthropic_key(self, checker: PIIChecker) -> None:
        """Test detecting Anthropic API key."""
        content = "Key: sk-ant-api01-1234567890abcdefghijklmnopqrstuvwxyz12345678"
        findings = checker.check(content, enabled_types=["api_key"])
        assert len(findings) == 1

    def test_reject_common_strings(self, checker: PIIChecker) -> None:
        """Test that common strings are not flagged as API keys."""
        content = "Content-Type: application/json"
        findings = checker.check(content, enabled_types=["api_key"])
        assert len(findings) == 0


class TestAllPIITypes:
    """Tests for checking all PII types."""

    def test_check_all_types(self, checker: PIIChecker) -> None:
        """Test checking all PII types at once."""
        content = """
        Contact: john@example.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        Card: 4111-1111-1111-1111
        Key: AKIAIOSFODNN7EXAMPLE
        """
        findings = checker.check(content)
        assert len(findings) >= 5

        finding_types = {f.finding_type for f in findings}
        assert "email" in finding_types
        assert "phone" in finding_types
        assert "ssn" in finding_types
        assert "credit_card" in finding_types
        assert "api_key" in finding_types


class TestDuplicateDetection:
    """Tests for duplicate detection handling."""

    def test_no_duplicate_findings(self, checker: PIIChecker) -> None:
        """Test that duplicate values are not reported twice."""
        content = "Email: john@example.com, confirm: john@example.com"
        findings = checker.check(content, enabled_types=["email"])
        assert len(findings) == 1


class TestLocationTracking:
    """Tests for location tracking in findings."""

    def test_location_in_findings(self, checker: PIIChecker) -> None:
        """Test that location is included in findings."""
        content = "Contact: john@example.com"
        findings = checker.check(content, location="report.md", enabled_types=["email"])
        assert len(findings) == 1
        assert findings[0].location == "report.md"

    def test_none_location(self, checker: PIIChecker) -> None:
        """Test findings without location."""
        content = "Contact: john@example.com"
        findings = checker.check(content, enabled_types=["email"])
        assert len(findings) == 1
        assert findings[0].location is None

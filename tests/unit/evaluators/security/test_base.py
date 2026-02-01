"""Unit tests for security evaluator base classes."""

from atp.evaluators.security.base import (
    SecurityFinding,
    Severity,
    mask_api_key,
    mask_credit_card,
    mask_email,
    mask_phone,
    mask_sensitive_data,
    mask_ssn,
)


class TestSeverity:
    """Tests for Severity enum."""

    def test_severity_values(self) -> None:
        """Test severity enum values."""
        assert Severity.INFO.value == "info"
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"
        assert Severity.CRITICAL.value == "critical"

    def test_severity_score_weight(self) -> None:
        """Test severity score weights."""
        assert Severity.INFO.score_weight == 0.95
        assert Severity.LOW.score_weight == 0.8
        assert Severity.MEDIUM.score_weight == 0.5
        assert Severity.HIGH.score_weight == 0.2
        assert Severity.CRITICAL.score_weight == 0.0

    def test_severity_comparison_gt(self) -> None:
        """Test severity greater than comparison."""
        assert Severity.CRITICAL > Severity.HIGH
        assert Severity.HIGH > Severity.MEDIUM
        assert Severity.MEDIUM > Severity.LOW
        assert Severity.LOW > Severity.INFO
        assert not Severity.INFO > Severity.INFO

    def test_severity_comparison_ge(self) -> None:
        """Test severity greater than or equal comparison."""
        assert Severity.CRITICAL >= Severity.HIGH
        assert Severity.CRITICAL >= Severity.CRITICAL
        assert Severity.HIGH >= Severity.MEDIUM
        assert Severity.INFO >= Severity.INFO

    def test_severity_comparison_lt(self) -> None:
        """Test severity less than comparison."""
        assert Severity.INFO < Severity.LOW
        assert Severity.LOW < Severity.MEDIUM
        assert Severity.MEDIUM < Severity.HIGH
        assert Severity.HIGH < Severity.CRITICAL
        assert not Severity.CRITICAL < Severity.CRITICAL

    def test_severity_comparison_le(self) -> None:
        """Test severity less than or equal comparison."""
        assert Severity.INFO <= Severity.LOW
        assert Severity.INFO <= Severity.INFO
        assert Severity.LOW <= Severity.MEDIUM
        assert Severity.CRITICAL <= Severity.CRITICAL


class TestSecurityFinding:
    """Tests for SecurityFinding model."""

    def test_security_finding_creation(self) -> None:
        """Test creating a security finding."""
        finding = SecurityFinding(
            check_type="pii",
            finding_type="email",
            severity=Severity.MEDIUM,
            message="Email address detected",
            evidence_masked="j***@example.com",
            location="artifact.txt",
            details={"pattern": "email"},
        )
        assert finding.check_type == "pii"
        assert finding.finding_type == "email"
        assert finding.severity == Severity.MEDIUM
        assert finding.message == "Email address detected"
        assert finding.evidence_masked == "j***@example.com"
        assert finding.location == "artifact.txt"
        assert finding.details == {"pattern": "email"}

    def test_security_finding_minimal(self) -> None:
        """Test creating a minimal security finding."""
        finding = SecurityFinding(
            check_type="pii",
            finding_type="ssn",
            severity=Severity.CRITICAL,
            message="SSN detected",
            evidence_masked="***-**-1234",
        )
        assert finding.location is None
        assert finding.details is None


class TestMaskSensitiveData:
    """Tests for mask_sensitive_data function."""

    def test_mask_normal_string(self) -> None:
        """Test masking a normal length string."""
        result = mask_sensitive_data("secretpassword", visible_chars=4)
        assert result == "secr******word"
        assert len(result) == 14

    def test_mask_short_string(self) -> None:
        """Test masking a short string."""
        result = mask_sensitive_data("abc", visible_chars=4)
        assert result == "***"

    def test_mask_exact_boundary(self) -> None:
        """Test masking at exact boundary."""
        result = mask_sensitive_data("12345678", visible_chars=4)
        assert result == "********"

    def test_mask_custom_visible_chars(self) -> None:
        """Test masking with custom visible chars."""
        result = mask_sensitive_data("mysecretvalue", visible_chars=2)
        assert result == "my*********ue"


class TestMaskEmail:
    """Tests for mask_email function."""

    def test_mask_normal_email(self) -> None:
        """Test masking a normal email."""
        result = mask_email("john.doe@example.com")
        assert result == "j******e@example.com"

    def test_mask_short_local_part(self) -> None:
        """Test masking email with short local part."""
        result = mask_email("a@example.com")
        assert result == "*@example.com"

    def test_mask_two_char_local_part(self) -> None:
        """Test masking email with two char local part."""
        result = mask_email("ab@example.com")
        assert result == "a*@example.com"

    def test_mask_three_char_local_part(self) -> None:
        """Test masking email with three char local part."""
        result = mask_email("abc@example.com")
        assert result == "a**@example.com"

    def test_mask_no_at_symbol(self) -> None:
        """Test masking string without @ symbol."""
        result = mask_email("notanemail")
        assert result == "nota**mail"


class TestMaskCreditCard:
    """Tests for mask_credit_card function."""

    def test_mask_credit_card_no_separator(self) -> None:
        """Test masking credit card without separators."""
        result = mask_credit_card("4111111111111111")
        assert result == "************1111"

    def test_mask_credit_card_with_dashes(self) -> None:
        """Test masking credit card with dashes."""
        result = mask_credit_card("4111-1111-1111-1111")
        assert result == "************1111"

    def test_mask_short_card(self) -> None:
        """Test masking short card number."""
        result = mask_credit_card("123")
        assert result == "***"


class TestMaskSSN:
    """Tests for mask_ssn function."""

    def test_mask_ssn_with_dashes(self) -> None:
        """Test masking SSN with dashes."""
        result = mask_ssn("123-45-6789")
        assert result == "***-**-6789"

    def test_mask_ssn_no_dashes(self) -> None:
        """Test masking SSN without dashes."""
        result = mask_ssn("123456789")
        assert result == "***-**-6789"

    def test_mask_short_ssn(self) -> None:
        """Test masking short SSN."""
        result = mask_ssn("123")
        assert result == "***"


class TestMaskPhone:
    """Tests for mask_phone function."""

    def test_mask_phone_with_dashes(self) -> None:
        """Test masking phone with dashes."""
        result = mask_phone("123-456-7890")
        assert result == "***-***-7890"

    def test_mask_phone_with_parens(self) -> None:
        """Test masking phone with parentheses."""
        result = mask_phone("(123) 456-7890")
        assert result == "***-***-7890"

    def test_mask_short_phone(self) -> None:
        """Test masking short phone number."""
        result = mask_phone("123")
        assert result == "***"


class TestMaskApiKey:
    """Tests for mask_api_key function."""

    def test_mask_api_key_normal(self) -> None:
        """Test masking normal API key."""
        result = mask_api_key("sk-1234567890abcdefghij")
        assert result == "sk-1***...***ghij"

    def test_mask_api_key_short(self) -> None:
        """Test masking short API key."""
        result = mask_api_key("short")
        assert result == "*****"

    def test_mask_api_key_aws(self) -> None:
        """Test masking AWS key."""
        result = mask_api_key("AKIAIOSFODNN7EXAMPLE")
        assert result == "AKIA***...***MPLE"

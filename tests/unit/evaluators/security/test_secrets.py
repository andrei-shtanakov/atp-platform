"""Unit tests for the SecretLeakChecker."""

from atp.evaluators.security.base import Severity
from atp.evaluators.security.secrets import (
    SecretLeakChecker,
    mask_connection_string,
    mask_jwt,
    mask_private_key,
)


class TestSecretLeakCheckerProperties:
    """Tests for SecretLeakChecker properties."""

    def test_name(self) -> None:
        """Test that name returns correct value."""
        checker = SecretLeakChecker()
        assert checker.name == "secret_leak"

    def test_check_types(self) -> None:
        """Test that check_types returns expected types."""
        checker = SecretLeakChecker()
        check_types = checker.check_types
        assert "private_key" in check_types
        assert "jwt_token" in check_types
        assert "bearer_token" in check_types
        assert "basic_auth" in check_types
        assert "env_secret" in check_types
        assert "connection_string" in check_types
        assert "hardcoded_secret" in check_types
        assert "aws_credential" in check_types
        assert "certificate" in check_types


class TestPrivateKeyDetection:
    """Tests for private key detection."""

    def test_rsa_private_key(self) -> None:
        """Test RSA private key detection."""
        checker = SecretLeakChecker()
        content = """
        -----BEGIN RSA PRIVATE KEY-----
        MIIEpAIBAAKCAQEA...
        -----END RSA PRIVATE KEY-----
        """
        findings = checker.check(content, enabled_types=["private_key"])
        assert len(findings) >= 1
        assert findings[0].severity == Severity.CRITICAL
        assert findings[0].finding_type == "private_key"

    def test_ec_private_key(self) -> None:
        """Test EC private key detection."""
        checker = SecretLeakChecker()
        content = "-----BEGIN EC PRIVATE KEY-----"
        findings = checker.check(content, enabled_types=["private_key"])
        assert len(findings) >= 1
        assert findings[0].finding_type == "private_key"

    def test_openssh_private_key(self) -> None:
        """Test OpenSSH private key detection."""
        checker = SecretLeakChecker()
        content = "-----BEGIN OPENSSH PRIVATE KEY-----"
        findings = checker.check(content, enabled_types=["private_key"])
        assert len(findings) >= 1
        assert findings[0].finding_type == "private_key"

    def test_pgp_private_key(self) -> None:
        """Test PGP private key detection."""
        checker = SecretLeakChecker()
        content = "-----BEGIN PGP PRIVATE KEY BLOCK-----"
        findings = checker.check(content, enabled_types=["private_key"])
        assert len(findings) >= 1

    def test_no_public_key_match(self) -> None:
        """Test that public keys are not matched."""
        checker = SecretLeakChecker()
        content = "-----BEGIN PUBLIC KEY-----"
        findings = checker.check(content, enabled_types=["private_key"])
        assert len(findings) == 0


class TestJWTDetection:
    """Tests for JWT token detection."""

    def test_valid_jwt(self) -> None:
        """Test valid JWT token detection."""
        checker = SecretLeakChecker()
        # Example JWT with proper structure
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        content = f"Authorization: Bearer {jwt}"
        findings = checker.check(content, enabled_types=["jwt_token"])
        assert len(findings) >= 1
        assert any(f.finding_type == "jwt_token" for f in findings)

    def test_invalid_jwt_structure(self) -> None:
        """Test that invalid JWT structure is not matched."""
        checker = SecretLeakChecker()
        content = "eyJhb.eyJ.not_valid"  # Too short parts
        findings = checker.check(content, enabled_types=["jwt_token"])
        assert len(findings) == 0


class TestBearerTokenDetection:
    """Tests for Bearer token detection."""

    def test_bearer_token(self) -> None:
        """Test Bearer token detection."""
        checker = SecretLeakChecker()
        content = "Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz1234567890"
        findings = checker.check(content, enabled_types=["bearer_token"])
        assert len(findings) >= 1
        assert any(f.finding_type == "bearer_token" for f in findings)

    def test_bearer_case_insensitive(self) -> None:
        """Test Bearer token is case insensitive."""
        checker = SecretLeakChecker()
        content = "authorization: bearer sk-abcdefghijklmnopqrstuvwxyz1234567890"
        findings = checker.check(content, enabled_types=["bearer_token"])
        assert len(findings) >= 1


class TestBasicAuthDetection:
    """Tests for Basic auth detection."""

    def test_basic_auth(self) -> None:
        """Test Basic auth detection."""
        checker = SecretLeakChecker()
        # Base64 encoded "user:password"
        content = "Authorization: Basic dXNlcjpwYXNzd29yZDEyMzQ1Njc4OTA="
        findings = checker.check(content, enabled_types=["basic_auth"])
        assert len(findings) >= 1
        assert any(f.finding_type == "basic_auth" for f in findings)


class TestEnvSecretDetection:
    """Tests for environment secret detection."""

    def test_db_password(self) -> None:
        """Test database password detection."""
        checker = SecretLeakChecker()
        content = "DB_PASSWORD=supersecretpassword123"
        findings = checker.check(content, enabled_types=["env_secret"])
        assert len(findings) >= 1
        assert any(f.finding_type == "env_secret" for f in findings)

    def test_api_key_assignment(self) -> None:
        """Test API key assignment detection."""
        checker = SecretLeakChecker()
        content = "SECRET_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3"
        findings = checker.check(content, enabled_types=["env_secret"])
        assert len(findings) >= 1

    def test_password_in_quotes(self) -> None:
        """Test password in quotes detection."""
        checker = SecretLeakChecker()
        content = 'password = "mysecretpassword123"'
        findings = checker.check(content, enabled_types=["env_secret"])
        assert len(findings) >= 1

    def test_placeholder_not_matched(self) -> None:
        """Test that placeholder values are not matched."""
        checker = SecretLeakChecker()
        content = "DB_PASSWORD=your_password_here"
        findings = checker.check(content, enabled_types=["env_secret"])
        assert len(findings) == 0

    def test_example_not_matched(self) -> None:
        """Test that example values are not matched."""
        checker = SecretLeakChecker()
        content = "password = 'example_password'"
        findings = checker.check(content, enabled_types=["env_secret"])
        assert len(findings) == 0


class TestConnectionStringDetection:
    """Tests for connection string detection."""

    def test_postgres_connection_string(self) -> None:
        """Test PostgreSQL connection string detection."""
        checker = SecretLeakChecker()
        content = "postgresql://user:secretpass123@localhost:5432/mydb"
        findings = checker.check(content, enabled_types=["connection_string"])
        assert len(findings) >= 1
        assert any(f.finding_type == "connection_string" for f in findings)
        assert any(f.severity == Severity.CRITICAL for f in findings)

    def test_mongodb_connection_string(self) -> None:
        """Test MongoDB connection string detection."""
        checker = SecretLeakChecker()
        content = "mongodb://admin:password123@cluster.mongodb.net/db"
        findings = checker.check(content, enabled_types=["connection_string"])
        assert len(findings) >= 1

    def test_mongodb_srv_connection_string(self) -> None:
        """Test MongoDB SRV connection string detection."""
        checker = SecretLeakChecker()
        content = "mongodb+srv://admin:password123@cluster.mongodb.net/db"
        findings = checker.check(content, enabled_types=["connection_string"])
        assert len(findings) >= 1

    def test_mysql_connection_string(self) -> None:
        """Test MySQL connection string detection."""
        checker = SecretLeakChecker()
        content = "mysql://root:mysqlpass123@localhost:3306/testdb"
        findings = checker.check(content, enabled_types=["connection_string"])
        assert len(findings) >= 1


class TestHardcodedSecretDetection:
    """Tests for hardcoded secret detection."""

    def test_const_password(self) -> None:
        """Test const password detection."""
        checker = SecretLeakChecker()
        content = 'const password = "hardcoded_secret_123"'
        findings = checker.check(content, enabled_types=["hardcoded_secret"])
        assert len(findings) >= 1
        assert any(f.finding_type == "hardcoded_secret" for f in findings)

    def test_let_api_key(self) -> None:
        """Test let api_key detection."""
        checker = SecretLeakChecker()
        content = 'let apiKey = "sk_live_abcdefghijklmnop"'
        findings = checker.check(content, enabled_types=["hardcoded_secret"])
        assert len(findings) >= 1

    def test_def_secret(self) -> None:
        """Test def secret detection (Python)."""
        checker = SecretLeakChecker()
        content = 'def secret = "mysupersecretvalue"'
        findings = checker.check(content, enabled_types=["hardcoded_secret"])
        assert len(findings) >= 1


class TestAWSCredentialDetection:
    """Tests for AWS credential detection."""

    def test_aws_access_key_id(self) -> None:
        """Test AWS access key ID detection."""
        checker = SecretLeakChecker()
        content = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        findings = checker.check(content, enabled_types=["aws_credential"])
        assert len(findings) >= 1
        assert any(f.finding_type == "aws_credential" for f in findings)


class TestCertificateDetection:
    """Tests for certificate detection."""

    def test_certificate_header(self) -> None:
        """Test certificate header detection."""
        checker = SecretLeakChecker()
        content = "-----BEGIN CERTIFICATE-----"
        findings = checker.check(content, enabled_types=["certificate"])
        assert len(findings) >= 1
        assert any(f.finding_type == "certificate" for f in findings)

    def test_pfx_file_reference(self) -> None:
        """Test PFX file reference detection."""
        checker = SecretLeakChecker()
        content = "Use the certificate at /path/to/cert.pfx"
        findings = checker.check(content, enabled_types=["certificate"])
        assert len(findings) >= 1


class TestSecretLeakCheckerFiltering:
    """Tests for filtering by enabled types."""

    def test_filter_by_enabled_types(self) -> None:
        """Test filtering findings by enabled types."""
        checker = SecretLeakChecker()
        content = """
        -----BEGIN RSA PRIVATE KEY-----
        eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIn0.abc123def456
        DB_PASSWORD=secretpass123
        """
        # Only check for private keys
        findings = checker.check(content, enabled_types=["private_key"])
        assert all(f.finding_type == "private_key" for f in findings)

    def test_check_all_types(self) -> None:
        """Test checking all types when enabled_types is None."""
        checker = SecretLeakChecker()
        content = """
        -----BEGIN RSA PRIVATE KEY-----
        DB_PASSWORD=secretpass123
        postgresql://user:pass1234@localhost/db
        """
        findings = checker.check(content, enabled_types=None)
        types_found = {f.finding_type for f in findings}
        assert "private_key" in types_found


class TestMaskingFunctions:
    """Tests for masking functions."""

    def test_mask_private_key(self) -> None:
        """Test private key masking."""
        result = mask_private_key("-----BEGIN RSA PRIVATE KEY-----")
        assert "REDACTED" in result
        assert "RSA" not in result

    def test_mask_jwt(self) -> None:
        """Test JWT masking."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.Sfl"
        result = mask_jwt(jwt)
        assert "***" in result
        assert len(result) < len(jwt)

    def test_mask_jwt_invalid(self) -> None:
        """Test JWT masking with invalid format."""
        result = mask_jwt("not.a.jwt")
        assert "***" in result

    def test_mask_connection_string(self) -> None:
        """Test connection string password masking."""
        result = mask_connection_string("secretpassword123")
        assert "***" in result
        assert result.startswith("se")
        assert result.endswith("23")

    def test_mask_connection_string_short(self) -> None:
        """Test short password masking."""
        result = mask_connection_string("abc")
        assert result == "***"


class TestSecretLeakCheckerRemediation:
    """Tests for remediation suggestions in findings."""

    def test_private_key_remediation(self) -> None:
        """Test that private key findings have remediation."""
        checker = SecretLeakChecker()
        content = "-----BEGIN RSA PRIVATE KEY-----"
        findings = checker.check(content, enabled_types=["private_key"])
        assert len(findings) >= 1
        assert findings[0].details is not None
        assert "remediation" in findings[0].details
        assert len(findings[0].details["remediation"]) > 0

    def test_connection_string_remediation(self) -> None:
        """Test that connection string findings have remediation."""
        checker = SecretLeakChecker()
        content = "postgresql://user:pass1234@localhost/db"
        findings = checker.check(content, enabled_types=["connection_string"])
        assert len(findings) >= 1
        assert findings[0].details is not None
        assert "remediation" in findings[0].details


class TestSecretLeakCheckerContext:
    """Tests for context extraction."""

    def test_context_extraction(self) -> None:
        """Test that context is extracted around findings."""
        checker = SecretLeakChecker()
        content = "prefix text -----BEGIN RSA PRIVATE KEY----- suffix text"
        findings = checker.check(content, enabled_types=["private_key"])
        assert len(findings) >= 1
        assert findings[0].details is not None
        assert "context" in findings[0].details
        # Context should include surrounding text
        context = findings[0].details["context"]
        assert "prefix" in context or "suffix" in context


class TestSecretLeakCheckerLocation:
    """Tests for location tracking."""

    def test_location_tracking(self) -> None:
        """Test that location is tracked in findings."""
        checker = SecretLeakChecker()
        content = "DB_PASSWORD=secretpass123"
        findings = checker.check(
            content, location="config.env", enabled_types=["env_secret"]
        )
        assert len(findings) >= 1
        assert findings[0].location == "config.env"


class TestSecretLeakCheckerDeduplication:
    """Tests for finding deduplication."""

    def test_duplicate_secrets_deduplicated(self) -> None:
        """Test that duplicate secrets are deduplicated."""
        checker = SecretLeakChecker()
        content = """
        DB_PASSWORD=secretpass123
        DB_PASSWORD=secretpass123
        DB_PASSWORD=secretpass123
        """
        findings = checker.check(content, enabled_types=["env_secret"])
        # Should only find one instance
        assert len(findings) == 1

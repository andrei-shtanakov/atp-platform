"""Tests for additional secret patterns (Phase 5)."""

from atp.core.security import is_sensitive_env_var, redact_secrets


class TestAdditionalSecretPatterns:
    def test_openai_key(self) -> None:
        text = "key=sk-abc123def456ghi789jkl012mno"
        assert "sk-abc123" not in redact_secrets(text)

    def test_anthropic_key(self) -> None:
        text = "key=sk-ant-abc123def456ghi789jkl012"
        assert "sk-ant-abc123" not in redact_secrets(text)

    def test_huggingface_token(self) -> None:
        text = "token=hf_abcdefghijklmnopqrstuvwx"
        assert "hf_abcdef" not in redact_secrets(text)

    def test_gcp_service_account(self) -> None:
        text = '{"type": "service_account", "project_id": "my-project"}'
        redacted = redact_secrets(text)
        assert "service_account" not in redacted

    def test_existing_patterns_still_work(self) -> None:
        # AWS key
        assert "AKIAIOSFODNN7EXAMPLE" not in redact_secrets("key=AKIAIOSFODNN7EXAMPLE")
        # GitHub token
        assert "ghp_" not in redact_secrets(
            "token=ghp_abcdefghijklmnopqrstuvwxyz1234567890"
        )
        # JWT
        text = "eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature12345"
        assert "eyJhbGci" not in redact_secrets(text)


class TestHuggingFaceEnvVars:
    def test_hf_token(self) -> None:
        assert is_sensitive_env_var("HF_TOKEN") is True

    def test_huggingface_api_key(self) -> None:
        assert is_sensitive_env_var("HUGGINGFACE_API_KEY") is True

    def test_hugging_face_token(self) -> None:
        assert is_sensitive_env_var("HUGGING_FACE_TOKEN") is True

    def test_safe_vars_unchanged(self) -> None:
        assert is_sensitive_env_var("PATH") is False
        assert is_sensitive_env_var("HOME") is False

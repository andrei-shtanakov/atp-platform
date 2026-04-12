"""Test token generation and hashing helpers."""

from atp.dashboard.tokens import generate_api_token, hash_token


class TestTokenGeneration:
    def test_user_token_prefix(self) -> None:
        token = generate_api_token(agent_scoped=False)
        assert token.startswith("atp_u_")
        assert len(token) == 38  # "atp_u_" + 32 hex chars

    def test_agent_token_prefix(self) -> None:
        token = generate_api_token(agent_scoped=True)
        assert token.startswith("atp_a_")
        assert len(token) == 38

    def test_tokens_are_unique(self) -> None:
        t1 = generate_api_token(agent_scoped=False)
        t2 = generate_api_token(agent_scoped=False)
        assert t1 != t2

    def test_hash_is_deterministic(self) -> None:
        token = "atp_u_abcdef1234567890abcdef1234567890"
        assert hash_token(token) == hash_token(token)

    def test_hash_differs_for_different_tokens(self) -> None:
        t1 = generate_api_token(agent_scoped=False)
        t2 = generate_api_token(agent_scoped=False)
        assert hash_token(t1) != hash_token(t2)

    def test_token_prefix_extraction(self) -> None:
        token = generate_api_token(agent_scoped=False)
        prefix = token[:12]
        assert prefix.startswith("atp_u_")
        assert len(prefix) == 12

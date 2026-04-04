"""Tests for unified AuthStateStore."""

import pytest

from atp.dashboard.auth.state_store import (
    InMemoryAuthStateStore,
    get_auth_state_store,
    reset_auth_state_store,
)


class TestInMemoryAuthStateStore:
    """Tests for in-memory auth state store."""

    @pytest.mark.anyio
    async def test_put_and_get(self) -> None:
        store = InMemoryAuthStateStore()
        await store.put("key1", {"tenant_id": "t1", "nonce": "n1"})
        result = await store.get("key1")
        assert result == {"tenant_id": "t1", "nonce": "n1"}

    @pytest.mark.anyio
    async def test_get_missing_key(self) -> None:
        store = InMemoryAuthStateStore()
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.anyio
    async def test_pop_returns_and_removes(self) -> None:
        store = InMemoryAuthStateStore()
        await store.put("key1", {"data": "value"})
        result = await store.pop("key1")
        assert result == {"data": "value"}
        # Second pop returns None
        result = await store.pop("key1")
        assert result is None

    @pytest.mark.anyio
    async def test_pop_missing_key(self) -> None:
        store = InMemoryAuthStateStore()
        result = await store.pop("nonexistent")
        assert result is None

    @pytest.mark.anyio
    async def test_expired_entry_returns_none(self) -> None:
        store = InMemoryAuthStateStore()
        await store.put("key1", {"data": "value"}, ttl_seconds=0)
        result = await store.get("key1")
        assert result is None

    @pytest.mark.anyio
    async def test_expired_pop_returns_none(self) -> None:
        store = InMemoryAuthStateStore()
        await store.put("key1", {"data": "value"}, ttl_seconds=0)
        result = await store.pop("key1")
        assert result is None

    @pytest.mark.anyio
    async def test_multiple_keys_isolated(self) -> None:
        store = InMemoryAuthStateStore()
        await store.put("sso:state1", {"type": "sso"})
        await store.put("saml:relay1", {"type": "saml"})
        sso = await store.get("sso:state1")
        saml = await store.get("saml:relay1")
        assert sso == {"type": "sso"}
        assert saml == {"type": "saml"}

    @pytest.mark.anyio
    async def test_overwrite_key(self) -> None:
        store = InMemoryAuthStateStore()
        await store.put("key1", {"v": 1})
        await store.put("key1", {"v": 2})
        result = await store.get("key1")
        assert result == {"v": 2}


class TestGetAuthStateStore:
    """Tests for singleton accessor."""

    def test_returns_same_instance(self) -> None:
        reset_auth_state_store()
        s1 = get_auth_state_store()
        s2 = get_auth_state_store()
        assert s1 is s2

    def test_reset_creates_new_instance(self) -> None:
        reset_auth_state_store()
        s1 = get_auth_state_store()
        reset_auth_state_store()
        s2 = get_auth_state_store()
        assert s1 is not s2

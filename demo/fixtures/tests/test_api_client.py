"""Tests validating the generated API client.

Uses an httpx mock transport to simulate HTTP responses without making real
network requests.
"""

import json

import httpx
import pytest
from api_client import APIClient


class MockTransport(httpx.BaseTransport):
    """Mock transport for network-free testing."""

    def __init__(self, handler: object = None) -> None:
        self._handler = handler or self._default_handler

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._handler(request)

    @staticmethod
    def _default_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})


def make_client(handler: object = None) -> APIClient:
    """Create an APIClient backed by a mock transport."""
    client = APIClient("http://test-api.local", timeout=5)
    # Replace the internal httpx client with a mock one
    mock_transport = MockTransport(handler)
    client._client = httpx.Client(
        base_url="http://test-api.local",
        timeout=5,
        transport=mock_transport,
    )
    return client


class TestGet:
    """Tests for the get method."""

    def test_get_returns_dict(self) -> None:
        client = make_client()
        result = client.get("/users")
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    def test_get_with_params(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "page=1" in str(request.url)
            return httpx.Response(200, json={"page": 1})

        client = make_client(handler)
        result = client.get("/users", params={"page": "1"})
        assert result["page"] == 1

    def test_get_404_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "not found"})

        client = make_client(handler)
        with pytest.raises(httpx.HTTPStatusError):
            client.get("/nonexistent")


class TestPost:
    """Tests for the post method."""

    def test_post_returns_dict(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return httpx.Response(201, json={"id": 1, "name": body["name"]})

        client = make_client(handler)
        result = client.post("/users", data={"name": "Alice"})
        assert result["name"] == "Alice"

    def test_post_500_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "internal"})

        client = make_client(handler)
        with pytest.raises(httpx.HTTPStatusError):
            client.post("/crash")


class TestInit:
    """Initialization tests."""

    def test_creates_instance(self) -> None:
        client = APIClient("http://example.com")
        assert client is not None

    def test_custom_timeout(self) -> None:
        client = APIClient("http://example.com", timeout=10)
        assert client is not None

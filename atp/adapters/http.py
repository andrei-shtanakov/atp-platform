"""HTTP adapter for agents with HTTP/REST API."""

import json
from collections.abc import AsyncIterator
from datetime import datetime

import httpx
from pydantic import Field, field_validator

from atp.core.security import validate_url, validate_url_with_dns
from atp.protocol import (
    ATPEvent,
    ATPRequest,
    ATPResponse,
    EventType,
)

from .base import AdapterConfig, AgentAdapter
from .exceptions import (
    AdapterConnectionError,
    AdapterResponseError,
    AdapterTimeoutError,
)


class HTTPAdapterConfig(AdapterConfig):
    """Configuration for HTTP adapter."""

    endpoint: str = Field(..., description="Agent HTTP endpoint URL")
    headers: dict[str, str] = Field(
        default_factory=dict, description="Additional HTTP headers"
    )
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    stream_endpoint: str | None = Field(
        None, description="Optional separate endpoint for streaming (SSE)"
    )
    health_endpoint: str | None = Field(
        None, description="Optional health check endpoint"
    )
    # Security settings
    allow_internal: bool = Field(
        default=False,
        description="Allow connections to internal/private IPs (use with caution)",
    )

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Validate endpoint URL for security."""
        # Note: allow_internal is not available here, so we do basic validation
        # Full validation with allow_internal happens in the adapter
        if not v or not v.strip():
            raise ValueError("Endpoint URL cannot be empty")
        v = v.strip()
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("Endpoint must be an HTTP/HTTPS URL")
        return v

    @field_validator("stream_endpoint")
    @classmethod
    def validate_stream_endpoint(cls, v: str | None) -> str | None:
        """Validate stream endpoint URL."""
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("Stream endpoint must be an HTTP/HTTPS URL")
        return v

    @field_validator("health_endpoint")
    @classmethod
    def validate_health_endpoint(cls, v: str | None) -> str | None:
        """Validate health endpoint URL."""
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("Health endpoint must be an HTTP/HTTPS URL")
        return v


class HTTPAdapter(AgentAdapter):
    """
    Adapter for agents with HTTP/REST API.

    Sends ATP Requests via HTTP POST and receives ATP Responses.
    Supports Server-Sent Events (SSE) for streaming.
    """

    def __init__(self, config: HTTPAdapterConfig) -> None:
        """
        Initialize HTTP adapter.

        Args:
            config: HTTP adapter configuration with endpoint URL.
        """
        super().__init__(config)
        self._config: HTTPAdapterConfig = config
        self._client: httpx.AsyncClient | None = None

    @property
    def adapter_type(self) -> str:
        """Return the adapter type identifier."""
        return "http"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.timeout_seconds),
                verify=self._config.verify_ssl,
                headers=self._config.headers,
                # Security: Limit redirects to prevent redirect-based attacks
                follow_redirects=True,
                max_redirects=5,
            )
        return self._client

    def _validate_endpoint(self, url: str, check_dns: bool = True) -> str:
        """Validate endpoint URL for security (SSRF prevention).

        Args:
            url: URL to validate.
            check_dns: Whether to perform DNS resolution check.

        Returns:
            Validated URL.
        """
        if check_dns and not self._config.allow_internal:
            return validate_url_with_dns(url, allow_internal=False)
        return validate_url(url, allow_internal=self._config.allow_internal)

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """
        Execute a task via HTTP POST.

        Args:
            request: ATP Request with task specification.

        Returns:
            ATPResponse from the agent.

        Raises:
            AdapterConnectionError: If cannot connect to agent.
            AdapterTimeoutError: If request times out.
            AdapterResponseError: If agent returns invalid response.
        """
        # Validate endpoint URL for SSRF prevention
        endpoint = self._validate_endpoint(self._config.endpoint)

        client = await self._get_client()
        request_data = request.model_dump(mode="json")

        try:
            response = await client.post(
                endpoint,
                json=request_data,
            )
        except httpx.TimeoutException as e:
            raise AdapterTimeoutError(
                f"HTTP request timed out after {self._config.timeout_seconds}s",
                timeout_seconds=self._config.timeout_seconds,
                adapter_type=self.adapter_type,
            ) from e
        except httpx.ConnectError as e:
            raise AdapterConnectionError(
                f"Failed to connect to {self._config.endpoint}",
                endpoint=self._config.endpoint,
                adapter_type=self.adapter_type,
                cause=e,
            ) from e
        except httpx.RequestError as e:
            raise AdapterConnectionError(
                f"HTTP request failed: {e}",
                endpoint=self._config.endpoint,
                adapter_type=self.adapter_type,
                cause=e,
            ) from e

        if response.status_code >= 400:
            raise AdapterResponseError(
                f"Agent returned error status {response.status_code}",
                status_code=response.status_code,
                response_body=response.text,
                adapter_type=self.adapter_type,
            )

        try:
            response_data = response.json()
            return ATPResponse.model_validate(response_data)
        except json.JSONDecodeError as e:
            raise AdapterResponseError(
                f"Invalid JSON response: {e}",
                status_code=response.status_code,
                response_body=response.text,
                adapter_type=self.adapter_type,
            ) from e
        except ValueError as e:
            raise AdapterResponseError(
                f"Invalid ATP Response format: {e}",
                status_code=response.status_code,
                response_body=response.text,
                adapter_type=self.adapter_type,
            ) from e

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """
        Execute a task with event streaming via SSE.

        Args:
            request: ATP Request with task specification.

        Yields:
            ATPEvent objects during execution.
            Final ATPResponse when complete.

        Raises:
            AdapterConnectionError: If cannot connect to agent.
            AdapterTimeoutError: If request times out.
            AdapterResponseError: If agent returns invalid response.
        """
        # Validate endpoint URL for SSRF prevention
        raw_endpoint = self._config.stream_endpoint or self._config.endpoint
        endpoint = self._validate_endpoint(raw_endpoint)

        client = await self._get_client()
        request_data = request.model_dump(mode="json")
        sequence = 0

        try:
            async with client.stream(
                "POST",
                endpoint,
                json=request_data,
                headers={"Accept": "text/event-stream"},
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise AdapterResponseError(
                        f"Agent returned error status {response.status_code}",
                        status_code=response.status_code,
                        response_body=response.text,
                        adapter_type=self.adapter_type,
                    )

                event_data = ""
                event_type = ""

                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        event_data = line[5:].strip()
                    elif line == "" and event_data:
                        # Empty line signals end of event
                        try:
                            data = json.loads(event_data)

                            if event_type == "response" or "status" in data:
                                # Final response
                                yield ATPResponse.model_validate(data)
                            else:
                                # Event
                                if "sequence" not in data:
                                    data["sequence"] = sequence
                                    sequence += 1
                                if "timestamp" not in data:
                                    data["timestamp"] = datetime.now().isoformat()
                                if "task_id" not in data:
                                    data["task_id"] = request.task_id
                                if "event_type" not in data:
                                    data["event_type"] = (
                                        event_type or EventType.PROGRESS.value
                                    )

                                yield ATPEvent.model_validate(data)
                        except (json.JSONDecodeError, ValueError):
                            # Skip malformed events
                            pass

                        event_data = ""
                        event_type = ""

        except httpx.TimeoutException as e:
            raise AdapterTimeoutError(
                f"HTTP stream timed out after {self._config.timeout_seconds}s",
                timeout_seconds=self._config.timeout_seconds,
                adapter_type=self.adapter_type,
            ) from e
        except httpx.ConnectError as e:
            raise AdapterConnectionError(
                f"Failed to connect to {endpoint}",
                endpoint=endpoint,
                adapter_type=self.adapter_type,
                cause=e,
            ) from e
        except httpx.RequestError as e:
            raise AdapterConnectionError(
                f"HTTP stream failed: {e}",
                endpoint=endpoint,
                adapter_type=self.adapter_type,
                cause=e,
            ) from e

    async def health_check(self) -> bool:
        """
        Check if the agent HTTP endpoint is healthy.

        Returns:
            True if agent responds to health check, False otherwise.
        """
        client = await self._get_client()
        endpoint = self._config.health_endpoint or self._config.endpoint

        try:
            response = await client.get(endpoint, timeout=5.0)
            return response.status_code < 500
        except httpx.RequestError:
            return False

    async def cleanup(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

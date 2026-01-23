"""Integration tests for ContainerAdapter with Docker."""

import subprocess

import pytest

from atp.adapters import (
    AdapterError,
    AdapterTimeoutError,
    ContainerAdapter,
    ContainerAdapterConfig,
    ContainerResources,
)
from atp.protocol import (
    ATPRequest,
    ATPResponse,
    ResponseStatus,
    Task,
)


def docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Skip all tests in this module if Docker is not available
pytestmark = pytest.mark.skipif(
    not docker_available(),
    reason="Docker is not available",
)


@pytest.fixture(scope="module")
def test_agent_image(tmp_path_factory) -> str:
    """Build a test Docker image for testing."""
    # Create a temporary directory for the Dockerfile
    build_dir = tmp_path_factory.mktemp("docker_build")

    # Create a simple Python script that acts as an agent
    agent_script = """#!/usr/bin/env python3
import json
import sys

# Read request from stdin
request_data = sys.stdin.read()
request = json.loads(request_data)

# Write events to stderr
events = [
    {"event_type": "progress", "payload": {"message": "Starting"}, "sequence": 0},
    {"event_type": "progress", "payload": {"message": "Processing"}, "sequence": 1},
]
for event in events:
    event["task_id"] = request.get("task_id", "unknown")
    print(json.dumps(event), file=sys.stderr)

# Write response to stdout
response = {
    "version": "1.0",
    "task_id": request.get("task_id", "unknown"),
    "status": "completed",
    "artifacts": [
        {
            "type": "structured",
            "name": "result",
            "data": {"message": "Container agent completed"},
        }
    ],
    "metrics": {
        "total_tokens": 200,
        "total_steps": 2,
    },
}
print(json.dumps(response))
"""

    # Create Dockerfile
    dockerfile = """FROM python:3.12-slim
COPY agent.py /agent.py
RUN chmod +x /agent.py
ENTRYPOINT ["python3", "/agent.py"]
"""

    (build_dir / "agent.py").write_text(agent_script)
    (build_dir / "Dockerfile").write_text(dockerfile)

    # Build the image
    image_name = "atp-test-agent:latest"
    result = subprocess.run(
        ["docker", "build", "-t", image_name, str(build_dir)],
        capture_output=True,
        timeout=120,
    )

    if result.returncode != 0:
        pytest.skip(f"Failed to build test image: {result.stderr.decode()}")

    yield image_name

    # Cleanup: remove the image
    subprocess.run(
        ["docker", "rmi", "-f", image_name],
        capture_output=True,
    )


@pytest.fixture
def sample_request() -> ATPRequest:
    """Create a sample ATP request."""
    return ATPRequest(
        task_id="container-test-123",
        task=Task(description="Container integration test task"),
        constraints={"max_steps": 10},
    )


class TestContainerAdapterIntegration:
    """Integration tests for ContainerAdapter."""

    @pytest.mark.anyio
    async def test_execute_real_container(
        self, test_agent_image: str, sample_request: ATPRequest
    ) -> None:
        """Test executing a real Docker container."""
        config = ContainerAdapterConfig(
            image=test_agent_image,
            timeout_seconds=30.0,
        )

        adapter = ContainerAdapter(config)
        response = await adapter.execute(sample_request)

        assert isinstance(response, ATPResponse)
        assert response.task_id == "container-test-123"
        assert response.status == ResponseStatus.COMPLETED
        assert len(response.artifacts) == 1
        assert response.metrics is not None
        assert response.metrics.total_tokens == 200

    @pytest.mark.anyio
    async def test_health_check_with_image(self, test_agent_image: str) -> None:
        """Test health check with existing image."""
        config = ContainerAdapterConfig(
            image=test_agent_image,
            timeout_seconds=30.0,
        )

        adapter = ContainerAdapter(config)
        result = await adapter.health_check()

        assert result is True

    @pytest.mark.anyio
    async def test_health_check_missing_image(self) -> None:
        """Test health check with non-existent image."""
        config = ContainerAdapterConfig(
            image="nonexistent-image:v999",
            timeout_seconds=30.0,
        )

        adapter = ContainerAdapter(config)
        result = await adapter.health_check()

        # Docker is available but image doesn't exist
        assert result is False

    @pytest.mark.anyio
    async def test_execute_with_resource_limits(
        self, test_agent_image: str, sample_request: ATPRequest
    ) -> None:
        """Test executing with resource limits."""
        config = ContainerAdapterConfig(
            image=test_agent_image,
            timeout_seconds=30.0,
            resources=ContainerResources(memory="256m", cpu="0.5"),
        )

        adapter = ContainerAdapter(config)
        response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_execute_with_environment_variables(
        self, test_agent_image: str, sample_request: ATPRequest
    ) -> None:
        """Test executing with environment variables."""
        config = ContainerAdapterConfig(
            image=test_agent_image,
            timeout_seconds=30.0,
            environment={
                "TEST_VAR": "test_value",
                "DEBUG": "true",
            },
        )

        adapter = ContainerAdapter(config)
        response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_execute_with_network_none(
        self, test_agent_image: str, sample_request: ATPRequest
    ) -> None:
        """Test executing with network disabled."""
        config = ContainerAdapterConfig(
            image=test_agent_image,
            timeout_seconds=30.0,
            network="none",
        )

        adapter = ContainerAdapter(config)
        response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_stream_events_real_container(
        self, test_agent_image: str, sample_request: ATPRequest
    ) -> None:
        """Test streaming events from real container."""
        config = ContainerAdapterConfig(
            image=test_agent_image,
            timeout_seconds=30.0,
        )

        adapter = ContainerAdapter(config)
        events = []
        async for item in adapter.stream_events(sample_request):
            events.append(item)

        # Should have events and final response
        assert len(events) >= 2
        # Last item should be response
        assert isinstance(events[-1], ATPResponse)
        assert events[-1].status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_execute_nonexistent_image(self, sample_request: ATPRequest) -> None:
        """Test executing with non-existent image."""
        config = ContainerAdapterConfig(
            image="nonexistent-image-that-does-not-exist:v999",
            timeout_seconds=30.0,
        )

        adapter = ContainerAdapter(config)
        with pytest.raises(AdapterError):
            await adapter.execute(sample_request)

    @pytest.mark.anyio
    async def test_context_manager(
        self, test_agent_image: str, sample_request: ATPRequest
    ) -> None:
        """Test using adapter as context manager."""
        config = ContainerAdapterConfig(
            image=test_agent_image,
            timeout_seconds=30.0,
        )

        async with ContainerAdapter(config) as adapter:
            response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.COMPLETED


@pytest.fixture(scope="module")
def failing_agent_image(tmp_path_factory) -> str:
    """Build a test Docker image that fails."""
    build_dir = tmp_path_factory.mktemp("docker_build_fail")

    agent_script = """#!/usr/bin/env python3
import sys
print("Error: Something went wrong", file=sys.stderr)
sys.exit(1)
"""

    dockerfile = """FROM python:3.12-slim
COPY agent.py /agent.py
RUN chmod +x /agent.py
ENTRYPOINT ["python3", "/agent.py"]
"""

    (build_dir / "agent.py").write_text(agent_script)
    (build_dir / "Dockerfile").write_text(dockerfile)

    image_name = "atp-test-agent-fail:latest"
    result = subprocess.run(
        ["docker", "build", "-t", image_name, str(build_dir)],
        capture_output=True,
        timeout=120,
    )

    if result.returncode != 0:
        pytest.skip(f"Failed to build test image: {result.stderr.decode()}")

    yield image_name

    subprocess.run(
        ["docker", "rmi", "-f", image_name],
        capture_output=True,
    )


class TestContainerAdapterFailures:
    """Tests for container failure scenarios."""

    @pytest.mark.anyio
    async def test_execute_container_exit_error(
        self, failing_agent_image: str, sample_request: ATPRequest
    ) -> None:
        """Test handling container exit with error code."""
        config = ContainerAdapterConfig(
            image=failing_agent_image,
            timeout_seconds=30.0,
        )

        adapter = ContainerAdapter(config)
        with pytest.raises(AdapterError) as exc_info:
            await adapter.execute(sample_request)

        assert "exited with code" in str(exc_info.value)


@pytest.fixture(scope="module")
def slow_agent_image(tmp_path_factory) -> str:
    """Build a test Docker image that runs slowly."""
    build_dir = tmp_path_factory.mktemp("docker_build_slow")

    agent_script = """#!/usr/bin/env python3
import time
import json
import sys

# Read request
request_data = sys.stdin.read()

# Sleep for a long time
time.sleep(60)

# This should never be reached due to timeout
response = {"status": "completed"}
print(json.dumps(response))
"""

    dockerfile = """FROM python:3.12-slim
COPY agent.py /agent.py
RUN chmod +x /agent.py
ENTRYPOINT ["python3", "/agent.py"]
"""

    (build_dir / "agent.py").write_text(agent_script)
    (build_dir / "Dockerfile").write_text(dockerfile)

    image_name = "atp-test-agent-slow:latest"
    result = subprocess.run(
        ["docker", "build", "-t", image_name, str(build_dir)],
        capture_output=True,
        timeout=120,
    )

    if result.returncode != 0:
        pytest.skip(f"Failed to build test image: {result.stderr.decode()}")

    yield image_name

    subprocess.run(
        ["docker", "rmi", "-f", image_name],
        capture_output=True,
    )


class TestContainerAdapterTimeout:
    """Tests for container timeout scenarios."""

    @pytest.mark.anyio
    async def test_execute_timeout(
        self, slow_agent_image: str, sample_request: ATPRequest
    ) -> None:
        """Test container execution timeout."""
        config = ContainerAdapterConfig(
            image=slow_agent_image,
            timeout_seconds=2.0,  # Short timeout
        )

        adapter = ContainerAdapter(config)
        with pytest.raises(AdapterTimeoutError) as exc_info:
            await adapter.execute(sample_request)

        assert exc_info.value.timeout_seconds == 2.0

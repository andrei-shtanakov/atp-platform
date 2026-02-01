# Plugin Development Guide

This guide explains how to develop and publish plugins for ATP (Agent Test Platform). Plugins extend ATP with custom adapters, evaluators, and reporters.

## Overview

ATP uses Python entry points for plugin discovery, enabling third-party plugins to be installed via pip and automatically discovered at runtime. This architecture allows:

- **Adapters**: Connect ATP to new agent frameworks or communication protocols
- **Evaluators**: Add custom evaluation logic for agent responses
- **Reporters**: Create new output formats for test results

## Plugin Architecture

### Entry Point Groups

ATP defines three entry point groups for plugins:

| Group | Purpose | Base Interface |
|-------|---------|----------------|
| `atp.adapters` | Agent communication | `AdapterPlugin` protocol |
| `atp.evaluators` | Result evaluation | `EvaluatorPlugin` protocol |
| `atp.reporters` | Output formatting | `ReporterPlugin` protocol |

### Discovery Flow

```
1. ATP starts
2. PluginManager scans entry points for all groups
3. Discovered plugins are wrapped in LazyPlugin (not loaded yet)
4. When a plugin is used, it's loaded and validated
5. Validation checks interface compliance and version compatibility
```

### Lazy Loading

Plugins are lazily loaded to minimize startup time:

```python
from atp.plugins import get_plugin_manager

manager = get_plugin_manager()

# Discover plugins (fast - no actual loading)
adapters = manager.discover_plugins("atp.adapters")

# Get a specific plugin (still not loaded)
lazy_plugin = manager.get_plugin("atp.adapters", "my_adapter")

# Load when needed (actual import happens here)
plugin_class = lazy_plugin.load()
```

## Plugin Interfaces

### AdapterPlugin Protocol

Adapters translate between ATP Protocol and agent-specific APIs.

```python
from collections.abc import AsyncIterator
from typing import ClassVar

from atp.plugins import PluginConfig
from atp.protocol import ATPEvent, ATPRequest, ATPResponse


class MyAdapterConfig(PluginConfig):
    """Configuration for MyAdapter."""

    env_prefix: ClassVar[str] = "MY_ADAPTER_"

    endpoint: str = "http://localhost:8000"
    timeout: int = 60
    api_key: str | None = None


class MyAdapter:
    """Custom adapter for my agent framework."""

    # Required: Unique adapter identifier
    adapter_type = "my_adapter"

    # Optional: Minimum ATP version required (default: "0.1.0")
    atp_version = "0.1.0"

    # Optional: Configuration schema class
    config_schema = MyAdapterConfig

    def __init__(self, config: MyAdapterConfig | None = None) -> None:
        self.config = config or MyAdapterConfig()

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """
        Execute a task synchronously.

        Args:
            request: ATP Request with task specification.

        Returns:
            ATPResponse with execution results.
        """
        # Implementation here
        ...

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """
        Execute a task with event streaming.

        Args:
            request: ATP Request with task specification.

        Yields:
            ATPEvent objects during execution.
            Final ATPResponse when complete.
        """
        # Implementation here
        ...

    # Optional methods
    async def health_check(self) -> bool:
        """Check if the agent is available."""
        return True

    async def cleanup(self) -> None:
        """Release resources held by the adapter."""
        pass
```

### EvaluatorPlugin Protocol

Evaluators assess agent results against test assertions.

```python
from typing import ClassVar

from atp.evaluators.base import EvalResult
from atp.loader.models import Assertion, TestDefinition
from atp.plugins import PluginConfig
from atp.protocol import ATPEvent, ATPResponse


class MyEvaluatorConfig(PluginConfig):
    """Configuration for MyEvaluator."""

    env_prefix: ClassVar[str] = "MY_EVALUATOR_"

    threshold: float = 0.8
    strict_mode: bool = False


class MyEvaluator:
    """Custom evaluator for specialized assertions."""

    # Required: Unique evaluator name
    name = "my_evaluator"

    # Optional: Minimum ATP version required
    atp_version = "0.1.0"

    # Optional: Configuration schema class
    config_schema = MyEvaluatorConfig

    def __init__(self, config: MyEvaluatorConfig | None = None) -> None:
        self.config = config or MyEvaluatorConfig()

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """
        Evaluate agent results against an assertion.

        Args:
            task: Test definition containing task details.
            response: ATP Response from the agent.
            trace: List of ATP Events from execution.
            assertion: Assertion to evaluate against.

        Returns:
            EvalResult containing check results.
        """
        # Implementation here
        ...
```

### ReporterPlugin Protocol

Reporters format and output test results.

```python
from typing import ClassVar

from atp.plugins import PluginConfig
from atp.reporters.base import SuiteReport


class MyReporterConfig(PluginConfig):
    """Configuration for MyReporter."""

    env_prefix: ClassVar[str] = "MY_REPORTER_"

    output_path: str = "report.custom"
    include_details: bool = True


class MyReporter:
    """Custom reporter for specialized output formats."""

    # Required: Unique reporter name
    name = "my_reporter"

    # Optional: Minimum ATP version required
    atp_version = "0.1.0"

    # Optional: Whether the reporter supports streaming output
    supports_streaming = False

    # Optional: Configuration schema class
    config_schema = MyReporterConfig

    def __init__(self, config: MyReporterConfig | None = None) -> None:
        self.config = config or MyReporterConfig()

    def report(self, report: SuiteReport) -> None:
        """
        Generate and output the report.

        Args:
            report: Suite report data to output.
        """
        # Implementation here
        ...
```

## Configuration Schema

### Defining Configuration

Use `PluginConfig` as the base class for plugin configuration:

```python
from typing import Any, ClassVar

from pydantic import Field

from atp.plugins import PluginConfig


class MyPluginConfig(PluginConfig):
    """Configuration for MyPlugin.

    Supports:
    - Pydantic validation
    - Environment variable overrides
    - JSON Schema generation
    - Default values
    """

    # Environment variable prefix (e.g., MY_PLUGIN_TIMEOUT)
    env_prefix: ClassVar[str] = "MY_PLUGIN_"

    # Example configurations for documentation
    config_examples: ClassVar[list[dict[str, Any]]] = [
        {"timeout": 30, "retries": 3},
        {"timeout": 120, "retries": 0, "debug": True},
    ]

    # Configuration fields with validation
    timeout: int = Field(
        default=60,
        description="Request timeout in seconds",
        ge=1,
        le=300,
    )
    retries: int = Field(
        default=3,
        description="Number of retry attempts",
        ge=0,
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authentication",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging",
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="List of allowed tool names",
    )
```

### Environment Variable Support

Configuration can be loaded from environment variables:

```python
# Set environment variables
# MY_PLUGIN_TIMEOUT=120
# MY_PLUGIN_DEBUG=true
# MY_PLUGIN_ALLOWED_TOOLS=tool1,tool2,tool3

# Load config from environment
config = MyPluginConfig.from_env()

# Override specific values
config = MyPluginConfig.from_env(retries=5)
```

### JSON Schema Generation

Generate JSON Schema for documentation and validation:

```python
# Get JSON Schema as dict
schema = MyPluginConfig.json_schema()

# Get JSON Schema as formatted string
schema_str = MyPluginConfig.json_schema_string()

# Get field descriptions
descriptions = MyPluginConfig.get_field_descriptions()
# {"timeout": "Request timeout in seconds", ...}

# Get default values
defaults = MyPluginConfig.get_defaults()
# {"timeout": 60, "retries": 3, ...}
```

## Full Plugin Examples

### Adapter Plugin Example

A complete adapter plugin for a REST API-based agent:

```python
# my_atp_adapter/adapter.py
"""REST API adapter for custom agent framework."""

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, ClassVar

import httpx
from pydantic import Field

from atp.plugins import PluginConfig
from atp.protocol import (
    ATPEvent,
    ATPRequest,
    ATPResponse,
    EventType,
    Metrics,
    ResponseStatus,
)

logger = logging.getLogger(__name__)


class RestAdapterConfig(PluginConfig):
    """Configuration for REST API adapter."""

    env_prefix: ClassVar[str] = "REST_ADAPTER_"

    config_examples: ClassVar[list[dict[str, Any]]] = [
        {
            "base_url": "http://localhost:8000",
            "timeout": 60,
        },
    ]

    base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the agent API",
    )
    timeout: int = Field(
        default=60,
        description="Request timeout in seconds",
        ge=1,
        le=600,
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authentication",
    )
    verify_ssl: bool = Field(
        default=True,
        description="Verify SSL certificates",
    )


class RestAdapter:
    """Adapter for agents exposed via REST API."""

    adapter_type = "rest_api"
    atp_version = "0.1.0"
    config_schema = RestAdapterConfig

    def __init__(self, config: RestAdapterConfig | None = None) -> None:
        self.config = config or RestAdapterConfig()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
        return self._client

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """Execute a task via REST API."""
        start_time = datetime.now()
        client = await self._get_client()

        try:
            # Build request payload
            payload = {
                "task": request.task.description,
                "input_data": request.task.input_data,
                "constraints": {
                    "max_steps": request.constraints.max_steps,
                    "timeout": request.constraints.timeout_seconds,
                    "allowed_tools": request.constraints.allowed_tools,
                },
            }

            # Execute request
            response = await client.post("/execute", json=payload)
            response.raise_for_status()
            result = response.json()

            # Build ATP response
            elapsed = (datetime.now() - start_time).total_seconds()

            return ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.COMPLETED,
                result=result.get("output"),
                artifacts=result.get("artifacts", {}),
                metrics=Metrics(
                    total_tokens=result.get("tokens", 0),
                    total_steps=result.get("steps", 0),
                    wall_time_seconds=elapsed,
                ),
            )

        except httpx.TimeoutException as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            return ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.TIMEOUT,
                error=f"Request timed out after {self.config.timeout}s",
                metrics=Metrics(wall_time_seconds=elapsed),
            )

        except httpx.HTTPStatusError as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            return ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error=f"HTTP error {e.response.status_code}: {e.response.text}",
                metrics=Metrics(wall_time_seconds=elapsed),
            )

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            return ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error=str(e),
                metrics=Metrics(wall_time_seconds=elapsed),
            )

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """Execute a task with event streaming."""
        start_time = datetime.now()
        client = await self._get_client()
        sequence = 0

        try:
            # Emit start event
            yield ATPEvent(
                task_id=request.task_id,
                timestamp=datetime.now(),
                sequence=sequence,
                event_type=EventType.PROGRESS,
                payload={"message": "Starting execution", "progress": 0},
            )
            sequence += 1

            # Use streaming endpoint
            payload = {
                "task": request.task.description,
                "input_data": request.task.input_data,
                "stream": True,
            }

            async with client.stream("POST", "/execute/stream", json=payload) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line:
                        continue

                    # Parse server-sent event
                    event_data = self._parse_sse(line)
                    if event_data is None:
                        continue

                    # Convert to ATP event
                    yield ATPEvent(
                        task_id=request.task_id,
                        timestamp=datetime.now(),
                        sequence=sequence,
                        event_type=self._map_event_type(event_data.get("type")),
                        payload=event_data.get("data", {}),
                    )
                    sequence += 1

            # Final response
            elapsed = (datetime.now() - start_time).total_seconds()
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.COMPLETED,
                metrics=Metrics(wall_time_seconds=elapsed),
            )

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error=str(e),
                metrics=Metrics(wall_time_seconds=elapsed),
            )

    def _parse_sse(self, line: str) -> dict[str, Any] | None:
        """Parse a server-sent event line."""
        if line.startswith("data:"):
            import json
            try:
                return json.loads(line[5:].strip())
            except json.JSONDecodeError:
                return None
        return None

    def _map_event_type(self, event_type: str | None) -> EventType:
        """Map agent event type to ATP EventType."""
        mapping = {
            "tool_call": EventType.TOOL_CALL,
            "llm_request": EventType.LLM_REQUEST,
            "reasoning": EventType.REASONING,
            "progress": EventType.PROGRESS,
            "error": EventType.ERROR,
        }
        return mapping.get(event_type or "", EventType.PROGRESS)

    async def health_check(self) -> bool:
        """Check if the agent API is available."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
```

### Evaluator Plugin Example

A complete evaluator plugin for semantic similarity checking:

```python
# my_atp_evaluator/evaluator.py
"""Semantic similarity evaluator using embeddings."""

import logging
from typing import Any, ClassVar

from pydantic import Field

from atp.evaluators.base import EvalCheck, EvalResult
from atp.loader.models import Assertion, TestDefinition
from atp.plugins import PluginConfig
from atp.protocol import ATPEvent, ATPResponse

logger = logging.getLogger(__name__)


class SemanticEvaluatorConfig(PluginConfig):
    """Configuration for semantic similarity evaluator."""

    env_prefix: ClassVar[str] = "SEMANTIC_EVAL_"

    config_examples: ClassVar[list[dict[str, Any]]] = [
        {"threshold": 0.8, "model": "text-embedding-3-small"},
    ]

    threshold: float = Field(
        default=0.75,
        description="Minimum similarity score to pass",
        ge=0.0,
        le=1.0,
    )
    model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model to use",
    )
    api_key: str | None = Field(
        default=None,
        description="OpenAI API key for embeddings",
    )


class SemanticEvaluator:
    """Evaluator that checks semantic similarity using embeddings."""

    name = "semantic_similarity"
    atp_version = "0.1.0"
    config_schema = SemanticEvaluatorConfig

    def __init__(self, config: SemanticEvaluatorConfig | None = None) -> None:
        self.config = config or SemanticEvaluatorConfig()
        self._client = None

    async def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.config.api_key)
            except ImportError:
                raise ImportError(
                    "openai package required for semantic evaluation. "
                    "Install with: pip install openai"
                )
        return self._client

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        client = await self._get_client()
        response = await client.embeddings.create(
            input=text,
            model=self.config.model,
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """Evaluate semantic similarity between response and expected."""
        checks: list[EvalCheck] = []

        # Check assertion type
        if assertion.type != "semantic_similarity":
            return EvalResult(
                evaluator=self.name,
                checks=[
                    EvalCheck(
                        name="assertion_type",
                        passed=False,
                        score=0.0,
                        message=(
                            f"Unsupported assertion type: {assertion.type}. "
                            f"Expected: semantic_similarity"
                        ),
                    )
                ],
            )

        # Get expected text from assertion
        expected = assertion.config.get("expected")
        if not expected:
            return EvalResult(
                evaluator=self.name,
                checks=[
                    EvalCheck(
                        name="configuration",
                        passed=False,
                        score=0.0,
                        message="Missing 'expected' in assertion config",
                    )
                ],
            )

        # Get actual response
        actual = response.result
        if actual is None:
            actual = ""
        if not isinstance(actual, str):
            actual = str(actual)

        try:
            # Get embeddings
            expected_embedding = await self._get_embedding(expected)
            actual_embedding = await self._get_embedding(actual)

            # Calculate similarity
            similarity = self._cosine_similarity(expected_embedding, actual_embedding)

            # Check threshold
            threshold = assertion.config.get("threshold", self.config.threshold)
            passed = similarity >= threshold

            checks.append(
                EvalCheck(
                    name="semantic_similarity",
                    passed=passed,
                    score=similarity,
                    message=(
                        f"Similarity: {similarity:.3f} "
                        f"({'≥' if passed else '<'} threshold {threshold})"
                    ),
                    details={
                        "similarity": similarity,
                        "threshold": threshold,
                        "expected_preview": expected[:100],
                        "actual_preview": actual[:100],
                    },
                )
            )

        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            checks.append(
                EvalCheck(
                    name="semantic_similarity",
                    passed=False,
                    score=0.0,
                    message=f"Error computing similarity: {str(e)}",
                )
            )

        return EvalResult(evaluator=self.name, checks=checks)
```

### Reporter Plugin Example

A complete reporter plugin for Markdown output:

```python
# my_atp_reporter/reporter.py
"""Markdown reporter for ATP test results."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from atp.plugins import PluginConfig
from atp.reporters.base import SuiteReport, TestReport

logger = logging.getLogger(__name__)


class MarkdownReporterConfig(PluginConfig):
    """Configuration for Markdown reporter."""

    env_prefix: ClassVar[str] = "MD_REPORTER_"

    config_examples: ClassVar[list[dict[str, Any]]] = [
        {"output_path": "test-results.md", "include_details": True},
    ]

    output_path: str = Field(
        default="test-results.md",
        description="Output file path",
    )
    include_details: bool = Field(
        default=True,
        description="Include detailed check results",
    )
    include_statistics: bool = Field(
        default=True,
        description="Include statistical analysis",
    )
    emoji_status: bool = Field(
        default=True,
        description="Use emoji for pass/fail status",
    )


class MarkdownReporter:
    """Reporter that outputs test results as Markdown."""

    name = "markdown"
    atp_version = "0.1.0"
    supports_streaming = False
    config_schema = MarkdownReporterConfig

    def __init__(self, config: MarkdownReporterConfig | None = None) -> None:
        self.config = config or MarkdownReporterConfig()

    def report(self, report: SuiteReport) -> None:
        """Generate Markdown report and write to file."""
        content = self._generate_markdown(report)
        output_path = Path(self.config.output_path)
        output_path.write_text(content)
        logger.info(f"Markdown report written to: {output_path}")

    def _generate_markdown(self, report: SuiteReport) -> str:
        """Generate Markdown content from report."""
        lines: list[str] = []

        # Header
        lines.append(f"# Test Results: {report.suite_name}")
        lines.append("")
        lines.append(f"**Agent**: {report.agent_name}")
        lines.append(f"**Generated**: {datetime.now().isoformat()}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Tests | {report.total_tests} |")
        lines.append(f"| Passed | {report.passed_tests} |")
        lines.append(f"| Failed | {report.failed_tests} |")
        lines.append(f"| Success Rate | {report.success_rate:.1%} |")
        if report.duration_seconds:
            lines.append(f"| Duration | {self._format_duration(report.duration_seconds)} |")
        if report.runs_per_test > 1:
            lines.append(f"| Runs per Test | {report.runs_per_test} |")
        lines.append("")

        # Test Results
        lines.append("## Test Results")
        lines.append("")

        for test in report.tests:
            lines.extend(self._format_test(test))
            lines.append("")

        # Error section if any
        if report.error:
            lines.append("## Errors")
            lines.append("")
            lines.append(f"**Suite Error**: {report.error}")
            lines.append("")

        return "\n".join(lines)

    def _format_test(self, test: TestReport) -> list[str]:
        """Format a single test result."""
        lines: list[str] = []

        # Status indicator
        if self.config.emoji_status:
            status = "✅" if test.success else "❌"
        else:
            status = "[PASS]" if test.success else "[FAIL]"

        lines.append(f"### {status} {test.test_name}")
        lines.append("")
        lines.append(f"**ID**: `{test.test_id}`")

        if test.score is not None:
            lines.append(f"**Score**: {test.score:.1f}/100")

        if test.duration_seconds is not None:
            lines.append(f"**Duration**: {self._format_duration(test.duration_seconds)}")

        if test.total_runs > 1:
            lines.append(
                f"**Runs**: {test.successful_runs}/{test.total_runs} successful"
            )

        if test.error:
            lines.append("")
            lines.append(f"**Error**: {test.error}")

        # Detailed results
        if self.config.include_details and test.eval_results:
            lines.append("")
            lines.append("#### Evaluation Details")
            lines.append("")

            for eval_result in test.eval_results:
                lines.append(f"**{eval_result.evaluator}**")
                lines.append("")
                lines.append("| Check | Status | Score | Message |")
                lines.append("|-------|--------|-------|---------|")

                for check in eval_result.checks:
                    status_icon = "✅" if check.passed else "❌"
                    message = check.message or "-"
                    # Escape pipe characters in message
                    message = message.replace("|", "\\|")
                    lines.append(
                        f"| {check.name} | {status_icon} | {check.score:.2f} | {message} |"
                    )

                lines.append("")

        # Statistics
        if self.config.include_statistics and test.statistics:
            stats = test.statistics
            lines.append("")
            lines.append("#### Statistics")
            lines.append("")
            lines.append(f"- Mean Score: {stats.mean:.2f}")
            lines.append(f"- Std Dev: {stats.std_dev:.2f}")
            lines.append(f"- 95% CI: [{stats.ci_lower:.2f}, {stats.ci_upper:.2f}]")
            lines.append(f"- Min: {stats.min_score:.2f}, Max: {stats.max_score:.2f}")

        return lines

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
```

## Packaging Your Plugin

### Project Structure

```
my-atp-plugin/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── my_atp_plugin/
│       ├── __init__.py
│       ├── adapter.py      # If providing an adapter
│       ├── evaluator.py    # If providing an evaluator
│       └── reporter.py     # If providing a reporter
└── tests/
    ├── __init__.py
    ├── test_adapter.py
    ├── test_evaluator.py
    └── test_reporter.py
```

### pyproject.toml

```toml
[project]
name = "my-atp-plugin"
version = "0.1.0"
description = "Custom plugins for ATP (Agent Test Platform)"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }
authors = [
    { name = "Your Name", email = "you@example.com" },
]
keywords = ["atp", "agent", "testing", "plugin"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Testing",
]

dependencies = [
    "atp-platform>=0.1.0",
    "httpx>=0.27",  # Example dependency
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-anyio>=0.0.0",
    "pytest-cov>=4.0",
    "ruff>=0.4",
]

# Entry points for ATP plugin discovery
[project.entry-points."atp.adapters"]
rest_api = "my_atp_plugin.adapter:RestAdapter"

[project.entry-points."atp.evaluators"]
semantic_similarity = "my_atp_plugin.evaluator:SemanticEvaluator"

[project.entry-points."atp.reporters"]
markdown = "my_atp_plugin.reporter:MarkdownReporter"

[project.urls]
Homepage = "https://github.com/yourusername/my-atp-plugin"
Documentation = "https://github.com/yourusername/my-atp-plugin#readme"
Repository = "https://github.com/yourusername/my-atp-plugin"
Issues = "https://github.com/yourusername/my-atp-plugin/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/my_atp_plugin"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

### Package `__init__.py`

```python
# src/my_atp_plugin/__init__.py
"""Custom plugins for ATP (Agent Test Platform)."""

from my_atp_plugin.adapter import RestAdapter, RestAdapterConfig
from my_atp_plugin.evaluator import SemanticEvaluator, SemanticEvaluatorConfig
from my_atp_plugin.reporter import MarkdownReporter, MarkdownReporterConfig

__version__ = "0.1.0"

__all__ = [
    "RestAdapter",
    "RestAdapterConfig",
    "SemanticEvaluator",
    "SemanticEvaluatorConfig",
    "MarkdownReporter",
    "MarkdownReporterConfig",
]
```

## Testing Plugins

### Unit Testing

Use pytest with anyio for async testing:

```python
# tests/test_adapter.py
"""Tests for REST adapter plugin."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from atp.protocol import ATPRequest, ResponseStatus, Task

from my_atp_plugin.adapter import RestAdapter, RestAdapterConfig


@pytest.fixture
def config() -> RestAdapterConfig:
    """Create test configuration."""
    return RestAdapterConfig(
        base_url="http://test-agent:8000",
        timeout=30,
    )


@pytest.fixture
def request() -> ATPRequest:
    """Create test request."""
    return ATPRequest(
        task_id="test-123",
        task=Task(description="Test task"),
    )


class TestRestAdapter:
    """Tests for RestAdapter."""

    def test_adapter_type(self, config: RestAdapterConfig) -> None:
        """Test adapter type identifier."""
        adapter = RestAdapter(config)
        assert adapter.adapter_type == "rest_api"

    def test_default_config(self) -> None:
        """Test adapter with default configuration."""
        adapter = RestAdapter()
        assert adapter.config.base_url == "http://localhost:8000"
        assert adapter.config.timeout == 60

    @pytest.mark.anyio
    async def test_execute_success(
        self, config: RestAdapterConfig, request: ATPRequest
    ) -> None:
        """Test successful execution."""
        adapter = RestAdapter(config)

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": "Task completed",
            "tokens": 100,
            "steps": 3,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            response = await adapter.execute(request)

        assert response.status == ResponseStatus.COMPLETED
        assert response.result == "Task completed"
        assert response.metrics is not None
        assert response.metrics.total_tokens == 100

    @pytest.mark.anyio
    async def test_execute_timeout(
        self, config: RestAdapterConfig, request: ATPRequest
    ) -> None:
        """Test timeout handling."""
        import httpx

        adapter = RestAdapter(config)

        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")
            mock_get_client.return_value = mock_client

            response = await adapter.execute(request)

        assert response.status == ResponseStatus.TIMEOUT
        assert "timeout" in response.error.lower()

    @pytest.mark.anyio
    async def test_health_check_success(self, config: RestAdapterConfig) -> None:
        """Test successful health check."""
        adapter = RestAdapter(config)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await adapter.health_check()

        assert result is True

    @pytest.mark.anyio
    async def test_cleanup(self, config: RestAdapterConfig) -> None:
        """Test resource cleanup."""
        adapter = RestAdapter(config)
        adapter._client = AsyncMock()

        await adapter.cleanup()

        adapter._client.aclose.assert_called_once()
        assert adapter._client is None
```

### Testing Evaluators

```python
# tests/test_evaluator.py
"""Tests for semantic evaluator plugin."""

import pytest
from unittest.mock import AsyncMock, patch

from atp.evaluators.base import EvalResult
from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPResponse, ResponseStatus

from my_atp_plugin.evaluator import SemanticEvaluator, SemanticEvaluatorConfig


@pytest.fixture
def config() -> SemanticEvaluatorConfig:
    """Create test configuration."""
    return SemanticEvaluatorConfig(threshold=0.75)


@pytest.fixture
def task() -> TestDefinition:
    """Create test definition."""
    return TestDefinition(
        id="test-1",
        name="Test semantic similarity",
        task={"description": "Test task"},
    )


@pytest.fixture
def response() -> ATPResponse:
    """Create test response."""
    return ATPResponse(
        task_id="test-1",
        status=ResponseStatus.COMPLETED,
        result="The quick brown fox jumps over the lazy dog.",
    )


class TestSemanticEvaluator:
    """Tests for SemanticEvaluator."""

    def test_evaluator_name(self, config: SemanticEvaluatorConfig) -> None:
        """Test evaluator name."""
        evaluator = SemanticEvaluator(config)
        assert evaluator.name == "semantic_similarity"

    @pytest.mark.anyio
    async def test_evaluate_high_similarity(
        self,
        config: SemanticEvaluatorConfig,
        task: TestDefinition,
        response: ATPResponse,
    ) -> None:
        """Test evaluation with high similarity."""
        evaluator = SemanticEvaluator(config)
        assertion = Assertion(
            type="semantic_similarity",
            config={
                "expected": "A quick brown fox jumps over a lazy dog.",
                "threshold": 0.8,
            },
        )

        # Mock embeddings
        with patch.object(evaluator, "_get_embedding") as mock_embed:
            # Return similar embeddings
            mock_embed.side_effect = [
                [1.0, 0.0, 0.0],  # expected
                [0.95, 0.1, 0.0],  # actual (similar)
            ]

            result = await evaluator.evaluate(task, response, [], assertion)

        assert isinstance(result, EvalResult)
        assert result.evaluator == "semantic_similarity"
        assert len(result.checks) == 1
        assert result.checks[0].passed is True

    @pytest.mark.anyio
    async def test_evaluate_wrong_assertion_type(
        self,
        config: SemanticEvaluatorConfig,
        task: TestDefinition,
        response: ATPResponse,
    ) -> None:
        """Test evaluation with wrong assertion type."""
        evaluator = SemanticEvaluator(config)
        assertion = Assertion(
            type="contains",  # Wrong type
            config={"text": "fox"},
        )

        result = await evaluator.evaluate(task, response, [], assertion)

        assert result.checks[0].passed is False
        assert "unsupported" in result.checks[0].message.lower()
```

### Testing Reporters

```python
# tests/test_reporter.py
"""Tests for Markdown reporter plugin."""

import pytest
from pathlib import Path

from atp.reporters.base import SuiteReport, TestReport

from my_atp_plugin.reporter import MarkdownReporter, MarkdownReporterConfig


@pytest.fixture
def config(tmp_path: Path) -> MarkdownReporterConfig:
    """Create test configuration."""
    return MarkdownReporterConfig(
        output_path=str(tmp_path / "report.md"),
        include_details=True,
    )


@pytest.fixture
def report() -> SuiteReport:
    """Create test suite report."""
    return SuiteReport(
        suite_name="Test Suite",
        agent_name="Test Agent",
        total_tests=2,
        passed_tests=1,
        failed_tests=1,
        success_rate=0.5,
        duration_seconds=10.5,
        tests=[
            TestReport(
                test_id="test-1",
                test_name="Passing Test",
                success=True,
                score=95.0,
                duration_seconds=5.0,
            ),
            TestReport(
                test_id="test-2",
                test_name="Failing Test",
                success=False,
                score=30.0,
                duration_seconds=5.5,
                error="Assertion failed",
            ),
        ],
    )


class TestMarkdownReporter:
    """Tests for MarkdownReporter."""

    def test_reporter_name(self, config: MarkdownReporterConfig) -> None:
        """Test reporter name."""
        reporter = MarkdownReporter(config)
        assert reporter.name == "markdown"
        assert reporter.supports_streaming is False

    def test_report_creates_file(
        self, config: MarkdownReporterConfig, report: SuiteReport
    ) -> None:
        """Test that report creates output file."""
        reporter = MarkdownReporter(config)
        reporter.report(report)

        output_path = Path(config.output_path)
        assert output_path.exists()
        content = output_path.read_text()
        assert "Test Suite" in content
        assert "Test Agent" in content

    def test_report_includes_summary(
        self, config: MarkdownReporterConfig, report: SuiteReport
    ) -> None:
        """Test that report includes summary."""
        reporter = MarkdownReporter(config)
        reporter.report(report)

        content = Path(config.output_path).read_text()
        assert "## Summary" in content
        assert "Total Tests" in content
        assert "Success Rate" in content

    def test_report_includes_test_results(
        self, config: MarkdownReporterConfig, report: SuiteReport
    ) -> None:
        """Test that report includes test results."""
        reporter = MarkdownReporter(config)
        reporter.report(report)

        content = Path(config.output_path).read_text()
        assert "Passing Test" in content
        assert "Failing Test" in content
        assert "✅" in content  # Pass indicator
        assert "❌" in content  # Fail indicator

    def test_format_duration(self, config: MarkdownReporterConfig) -> None:
        """Test duration formatting."""
        reporter = MarkdownReporter(config)

        assert reporter._format_duration(0.5) == "500ms"
        assert reporter._format_duration(5.0) == "5.0s"
        assert reporter._format_duration(90.5) == "1m 30.5s"
        assert reporter._format_duration(3700) == "1h 1m"
```

### Integration Testing

```python
# tests/integration/test_plugin_integration.py
"""Integration tests for plugin with ATP."""

import pytest

from atp.plugins import get_plugin_manager


class TestPluginDiscovery:
    """Test that plugins are discovered by ATP."""

    def test_adapter_discovered(self) -> None:
        """Test that adapter is discovered."""
        manager = get_plugin_manager()
        adapters = manager.discover_plugins("atp.adapters")

        assert "rest_api" in adapters
        plugin = adapters["rest_api"]
        assert plugin.info.name == "rest_api"

    def test_adapter_loads(self) -> None:
        """Test that adapter loads successfully."""
        manager = get_plugin_manager()
        plugin_class = manager.load_and_validate_plugin(
            "atp.adapters", "rest_api"
        )

        assert plugin_class is not None
        assert hasattr(plugin_class, "adapter_type")
        assert hasattr(plugin_class, "execute")
```

## Publishing to PyPI

### Prerequisites

1. Create a PyPI account at https://pypi.org/account/register/
2. Enable two-factor authentication
3. Create an API token at https://pypi.org/manage/account/token/

### Building the Package

```bash
# Install build tools
uv add --dev build twine

# Build the package
uv run python -m build

# Check the package
uv run twine check dist/*
```

### Publishing

```bash
# Upload to TestPyPI first (recommended)
uv run twine upload --repository testpypi dist/*

# Install from TestPyPI to verify
pip install --index-url https://test.pypi.org/simple/ my-atp-plugin

# Upload to PyPI
uv run twine upload dist/*
```

### Automated Publishing with GitHub Actions

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

### Version Management

Follow semantic versioning (SemVer):

- **MAJOR**: Breaking changes to plugin interfaces
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

Update version in `pyproject.toml`:

```toml
[project]
version = "0.2.0"  # Increment appropriately
```

## CLI Commands

ATP provides CLI commands for managing plugins:

```bash
# List all discovered plugins
atp plugins list

# List only adapters
atp plugins list --type=adapter

# Get detailed info about a plugin
atp plugins info my_adapter

# Show plugin config schema
atp plugins info my_adapter --type=adapter
```

## Best Practices

### 1. Version Compatibility

Always specify the minimum ATP version your plugin requires:

```python
class MyPlugin:
    atp_version = "0.1.0"  # Minimum required version
```

### 2. Configuration Validation

Use Pydantic for robust configuration validation:

```python
from pydantic import Field, field_validator

class MyConfig(PluginConfig):
    url: str = Field(..., description="API URL")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v
```

### 3. Error Handling

Return meaningful errors in responses instead of raising exceptions:

```python
async def execute(self, request: ATPRequest) -> ATPResponse:
    try:
        result = await self._run()
        return ATPResponse(
            task_id=request.task_id,
            status=ResponseStatus.COMPLETED,
            result=result,
        )
    except TimeoutError:
        return ATPResponse(
            task_id=request.task_id,
            status=ResponseStatus.TIMEOUT,
            error="Operation timed out",
        )
```

### 4. Resource Management

Use async context managers for proper cleanup:

```python
async def __aenter__(self):
    await self._initialize()
    return self

async def __aexit__(self, *args):
    await self.cleanup()
```

### 5. Logging

Use the standard logging module:

```python
import logging

logger = logging.getLogger(__name__)

class MyPlugin:
    async def execute(self, request: ATPRequest) -> ATPResponse:
        logger.debug(f"Executing task {request.task_id}")
        # ...
        logger.info(f"Task {request.task_id} completed")
```

### 6. Type Hints

Always use type hints for better IDE support and documentation:

```python
async def execute(self, request: ATPRequest) -> ATPResponse:
    ...
```

### 7. Documentation

Include docstrings and configuration examples:

```python
class MyPluginConfig(PluginConfig):
    """Configuration for MyPlugin.

    Example:
        config = MyPluginConfig(
            endpoint="http://localhost:8000",
            timeout=60,
        )
    """

    config_examples: ClassVar[list[dict[str, Any]]] = [
        {"endpoint": "http://localhost:8000", "timeout": 60},
    ]
```

## See Also

- [Adapter Development Guide](./adapter-development.md)
- [Configuration Reference](../reference/configuration.md)
- [ATP Protocol](../04-protocol.md)
- [Test Format Reference](../reference/test-format.md)

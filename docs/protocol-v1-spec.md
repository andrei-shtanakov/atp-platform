# ATP Protocol v1.0 Specification

**Status:** Stable
**Version:** 1.0
**Date:** 2026-03-28

---

## Table of Contents

1. [Overview](#overview)
2. [Versioning Policy](#versioning-policy)
3. [Constants](#constants)
4. [Enums](#enums)
5. [Supporting Types](#supporting-types)
6. [Message Types](#message-types)
7. [Event Payload Schemas](#event-payload-schemas)
8. [Adapter Contract](#adapter-contract)
9. [Validation Rules](#validation-rules)
10. [Error Handling](#error-handling)

---

## Overview

The ATP (Agent Test Platform) Protocol is a standardized, framework-agnostic interface for communicating with AI agents. It defines three message types — **ATPRequest**, **ATPResponse**, and **ATPEvent** — that together describe how to invoke an agent, receive its results, and observe its execution in real time.

**Core design principle**: An agent is a black box. The protocol defines the contract (input → output + streaming events) without dictating how the agent is implemented internally. Adapters translate between ATP Protocol messages and agent-specific communication mechanisms (HTTP, Docker, CLI, LangGraph, CrewAI, AutoGen, MCP, cloud APIs, etc.).

All models are defined using Pydantic v2. All fields are validated on construction; invalid values raise `ValueError`.

---

## Versioning Policy

ATP Protocol uses [Semantic Versioning](https://semver.org): `MAJOR.MINOR`.

- **Current version:** `"1.0"`
- **Supported versions:** `{"1.0"}`

### What constitutes a breaking change (MAJOR bump)

- Removing or renaming a field in any message type
- Changing a field's type in a backwards-incompatible way
- Adding a required field to an existing message type
- Removing an enum value

### What does NOT constitute a breaking change (MINOR bump)

- Adding a new optional field to an existing message type
- Adding a new enum value
- Adding a new message type
- Adding a new supported version to `SUPPORTED_VERSIONS`

### Version field behavior

Every message (`ATPRequest`, `ATPResponse`, `ATPEvent`) carries a `version` field. The field defaults to the current protocol version (`"1.0"`) and is validated against `SUPPORTED_VERSIONS` on construction. Attempts to construct a message with an unsupported version string raise `ValueError`.

---

## Constants

These constants define upper bounds for field lengths and collection sizes. They are enforced by field validators in the Pydantic models.

| Constant | Value | Applies To |
|----------|-------|------------|
| `MAX_TASK_ID_LENGTH` | `128` | `task_id` in all messages |
| `MAX_DESCRIPTION_LENGTH` | `100_000` | `Task.description` |
| `MAX_PATH_LENGTH` | `4096` | File/workspace paths, artifact paths |
| `MAX_ERROR_LENGTH` | `10_000` | `ATPResponse.error` |
| `MAX_CONTENT_LENGTH` | `10_000_000` (10 MB) | `ArtifactFile.content` (inline) |
| `MAX_ARTIFACTS_COUNT` | `1000` | `ATPResponse.artifacts` list |
| `MAX_ENV_VARS_COUNT` | `100` | `Context.environment` dict |
| `MAX_METADATA_KEYS` | `50` | `ATPRequest.metadata` dict |

---

## Enums

### ResponseStatus

Indicates the terminal state of an agent execution. Used in `ATPResponse.status`.

| Value | Description |
|-------|-------------|
| `"completed"` | The agent finished the task successfully. |
| `"failed"` | The agent encountered an unrecoverable error. |
| `"timeout"` | The agent did not finish within the allowed time. |
| `"cancelled"` | Execution was cancelled by the caller. |
| `"partial"` | The agent produced partial results before stopping. |

### EventType

Categorizes streaming events emitted during execution. Used in `ATPEvent.event_type`.

| Value | Description |
|-------|-------------|
| `"tool_call"` | The agent invoked a tool. Payload: see `ToolCallPayload`. |
| `"llm_request"` | The agent made an LLM API call. Payload: see `LLMRequestPayload`. |
| `"reasoning"` | The agent emitted an internal thought, plan, or step. Payload: see `ReasoningPayload`. |
| `"error"` | A recoverable or non-fatal error occurred. Payload: see `ErrorPayload`. |
| `"progress"` | A progress update. Payload: see `ProgressPayload`. |

---

## Supporting Types

### Task

Describes what the agent should do.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `description` | `str` | Yes | — | Human-readable task description. Min length 1, max `MAX_DESCRIPTION_LENGTH`. |
| `input_data` | `dict[str, Any] \| None` | No | `None` | Arbitrary structured input for the task. |
| `expected_artifacts` | `list[str] \| None` | No | `None` | List of expected artifact paths/names the agent should produce. |

**Validation for `expected_artifacts`:** Each path must be non-empty, at most `MAX_PATH_LENGTH` characters, must not contain `..`, must not start with `/` or `~`, and must not resolve outside the working directory via symlink traversal. Whitespace is stripped from each path.

---

### Context

Provides execution environment information to the agent.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `tools_endpoint` | `str \| None` | No | `None` | HTTP/HTTPS URL for a tools API the agent may call. |
| `workspace_path` | `str \| None` | No | `None` | Filesystem path to the agent's working directory. Max `MAX_PATH_LENGTH`. |
| `environment` | `dict[str, str] \| None` | No | `None` | Environment variables to pass to the agent. Max `MAX_ENV_VARS_COUNT` entries. |

**Validation for `tools_endpoint`:** Must be an HTTP or HTTPS URL (start with `http://` or `https://`). Empty string is normalized to `None`.

**Validation for `workspace_path`:** Must not contain null bytes (`\x00`). Max `MAX_PATH_LENGTH`. Empty string is normalized to `None`.

**Validation for `environment`:** Each key must match `^[a-zA-Z_][a-zA-Z0-9_]*$` (standard POSIX env var name). Keys and values must not contain null bytes. Keys must not be empty or whitespace-only.

---

### Metrics

Execution metrics reported by the agent after a run.

All fields are optional — agents report whatever is available to them.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `total_tokens` | `int \| None` | No | `None` | Total tokens used (input + output). Must be ≥ 0. |
| `input_tokens` | `int \| None` | No | `None` | Tokens in the input/prompt context. Must be ≥ 0. |
| `output_tokens` | `int \| None` | No | `None` | Tokens generated in output. Must be ≥ 0. |
| `total_steps` | `int \| None` | No | `None` | Number of agent loop iterations/steps. Must be ≥ 0. |
| `tool_calls` | `int \| None` | No | `None` | Number of tool invocations. Must be ≥ 0. |
| `llm_calls` | `int \| None` | No | `None` | Number of LLM API calls. Must be ≥ 0. |
| `wall_time_seconds` | `float \| None` | No | `None` | Total wall-clock execution time in seconds. Must be ≥ 0. |
| `cost_usd` | `float \| None` | No | `None` | Estimated cost in US dollars. Must be ≥ 0. |

---

### Artifact Types

Agents produce artifacts — outputs that are not part of the response metadata. The `Artifact` type is a discriminated union of three subtypes, distinguished by the `type` literal field.

```python
Artifact = ArtifactFile | ArtifactStructured | ArtifactReference
```

Consumers should dispatch on the `type` field:

```python
for artifact in response.artifacts:
    if artifact.type == "file":
        # ArtifactFile
    elif artifact.type == "structured":
        # ArtifactStructured
    elif artifact.type == "reference":
        # ArtifactReference
```

#### ArtifactFile

A file written to disk by the agent.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | `Literal["file"]` | — | `"file"` | Discriminator field. Always `"file"`. |
| `path` | `str` | Yes | — | Relative path to the file. Min length 1, max `MAX_PATH_LENGTH`. Must be relative; no `..` components; no absolute paths. |
| `content_type` | `str \| None` | No | `None` | MIME type of the file (e.g., `"text/plain"`). Max 256 chars. |
| `size_bytes` | `int \| None` | No | `None` | File size in bytes. Must be ≥ 0. |
| `content_hash` | `str \| None` | No | `None` | Hash of the file content (e.g., `"sha256:abc123..."`). Max 128 chars. |
| `content` | `str \| None` | No | `None` | Inline file content. For binary files, use base64 encoding. Max `MAX_CONTENT_LENGTH`. |

**Path validation:** Relative paths only. No `..` components. No absolute paths (no leading `/`). No null bytes.

#### ArtifactStructured

Structured data (e.g., JSON result, analysis output) produced by the agent.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | `Literal["structured"]` | — | `"structured"` | Discriminator field. Always `"structured"`. |
| `name` | `str` | Yes | — | Artifact name. Min length 1, max 256 chars. Must not contain path separators (`/`, `\`). |
| `data` | `dict[str, Any]` | Yes | — | The structured data payload. |
| `content_type` | `str \| None` | No | `None` | Content type hint (e.g., `"application/json"`). Max 256 chars. |

#### ArtifactReference

A pointer to an external resource (URL, remote path) produced or discovered by the agent.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | `Literal["reference"]` | — | `"reference"` | Discriminator field. Always `"reference"`. |
| `path` | `str` | Yes | — | Reference path or URL. Min length 1, max `MAX_PATH_LENGTH`. |
| `content_type` | `str \| None` | No | `None` | Content type of the referenced resource. Max 256 chars. |
| `size_bytes` | `int \| None` | No | `None` | Size of the referenced resource in bytes. Must be ≥ 0. |

---

## Message Types

### ATPRequest

Sent by the test runner (or SDK consumer) to an adapter, which forwards it to the agent.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `version` | `str` | No | `"1.0"` | Protocol version. Must be in `SUPPORTED_VERSIONS`. |
| `task_id` | `str` | Yes | — | Unique task identifier. Pattern: `^[a-zA-Z0-9_-]+$`. Max `MAX_TASK_ID_LENGTH`. |
| `task` | `Task` | Yes | — | Task specification. |
| `constraints` | `dict[str, Any]` | No | `{}` | Execution constraints. Common keys: `max_steps`, `timeout`, `allowed_tools`. |
| `context` | `Context \| None` | No | `None` | Execution context (tools endpoint, workspace, env vars). |
| `metadata` | `dict[str, Any] \| None` | No | `None` | Pass-through metadata (e.g., test IDs, suite IDs). Not interpreted by agents. Max `MAX_METADATA_KEYS` keys. |

**task_id validation:** Non-empty after stripping whitespace. Must match `^[a-zA-Z0-9_-]+$` (alphanumeric, underscore, hyphen only). Max `MAX_TASK_ID_LENGTH` characters.

**Example:**

```json
{
  "version": "1.0",
  "task_id": "test-write-hello-001",
  "task": {
    "description": "Write a Python file that prints 'Hello, World!'",
    "expected_artifacts": ["hello.py"]
  },
  "constraints": {
    "max_steps": 10,
    "timeout": 60
  },
  "context": {
    "workspace_path": "/tmp/agent-workspace"
  }
}
```

---

### ATPResponse

Returned by the adapter after execution completes (successfully or otherwise).

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `version` | `str` | No | `"1.0"` | Protocol version. Must be in `SUPPORTED_VERSIONS`. |
| `task_id` | `str` | Yes | — | Must match the `task_id` from the corresponding `ATPRequest`. |
| `status` | `ResponseStatus` | Yes | — | Terminal execution status. |
| `artifacts` | `list[Artifact]` | No | `[]` | Output artifacts produced by the agent. Max `MAX_ARTIFACTS_COUNT` items. |
| `metrics` | `Metrics \| None` | No | `None` | Execution metrics. |
| `error` | `str \| None` | No | `None` | Human-readable error message when `status` is `"failed"` or `"timeout"`. Max `MAX_ERROR_LENGTH`. |
| `trace_id` | `str \| None` | No | `None` | Optional trace/span identifier for observability correlation. Max 256 chars. |

**Note on task_id in responses:** The pattern constraint (`^[a-zA-Z0-9_-]+$`) is NOT enforced on `ATPResponse.task_id`. The value is echoed from the agent and may differ from the request format in edge cases. Length limit (`MAX_TASK_ID_LENGTH`) is still enforced.

**Example:**

```json
{
  "version": "1.0",
  "task_id": "test-write-hello-001",
  "status": "completed",
  "artifacts": [
    {
      "type": "file",
      "path": "hello.py",
      "content_type": "text/x-python",
      "content": "print('Hello, World!')\n"
    }
  ],
  "metrics": {
    "total_tokens": 250,
    "input_tokens": 180,
    "output_tokens": 70,
    "wall_time_seconds": 3.14
  }
}
```

---

### ATPEvent

Emitted during streaming execution. Events are yielded by `stream_events()` before the final `ATPResponse`.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `version` | `str` | No | `"1.0"` | Protocol version. Must be in `SUPPORTED_VERSIONS`. |
| `task_id` | `str` | Yes | — | Task identifier (must match the in-flight request). |
| `timestamp` | `datetime` | No | `datetime.now(UTC)` | UTC timestamp of the event. |
| `sequence` | `int` | Yes | — | Monotonically increasing sequence number starting at 0. Must be ≥ 0. |
| `event_type` | `EventType` | Yes | — | Event category. |
| `payload` | `dict[str, Any]` | Yes | — | Event-specific data. Structure depends on `event_type` (see below). |

**Note:** `payload` is typed as `dict[str, Any]` at the model level for extensibility. The recommended structure for each `event_type` is documented in the next section. Consumers should be tolerant of unknown payload keys.

**Example:**

```json
{
  "version": "1.0",
  "task_id": "test-write-hello-001",
  "timestamp": "2026-03-28T10:00:01.234Z",
  "sequence": 0,
  "event_type": "tool_call",
  "payload": {
    "tool": "write_file",
    "input": {"path": "hello.py", "content": "print('Hello, World!')\n"},
    "output": {"success": true},
    "duration_ms": 12.5,
    "status": "success"
  }
}
```

---

## Event Payload Schemas

The following payload schemas are the recommended (not enforced) structures for each `EventType`. Adapters SHOULD use these schemas when possible. Consumers SHOULD be tolerant of missing optional fields.

### ToolCallPayload (event_type: `"tool_call"`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tool` | `str` | Yes | Tool name. Min length 1. |
| `input` | `dict[str, Any] \| None` | No | Tool input arguments. |
| `output` | `dict[str, Any] \| None` | No | Tool output or result. |
| `duration_ms` | `float \| None` | No | Call duration in milliseconds. Must be ≥ 0. |
| `status` | `str \| None` | No | Call status, typically `"success"` or `"error"`. |

### LLMRequestPayload (event_type: `"llm_request"`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | `str` | Yes | Model name or identifier. Min length 1. |
| `input_tokens` | `int \| None` | No | Tokens in the prompt/context. Must be ≥ 0. |
| `output_tokens` | `int \| None` | No | Tokens generated in the completion. Must be ≥ 0. |
| `duration_ms` | `float \| None` | No | Request duration in milliseconds. Must be ≥ 0. |

### ReasoningPayload (event_type: `"reasoning"`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `thought` | `str \| None` | No | An internal agent thought or observation. |
| `plan` | `str \| None` | No | A plan or strategy the agent has formulated. |
| `step` | `str \| None` | No | Description of the current execution step. |

### ErrorPayload (event_type: `"error"`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `error_type` | `str` | Yes | Error category or exception class name. Min length 1. |
| `message` | `str` | Yes | Human-readable error description. Min length 1. |
| `recoverable` | `bool \| None` | No | `True` if the agent may continue after this error. |

### ProgressPayload (event_type: `"progress"`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `current_step` | `int \| None` | No | Current step number. Must be ≥ 0. |
| `percentage` | `float \| None` | No | Completion percentage (0–100). |
| `message` | `str \| None` | No | Human-readable progress description. |

---

## Adapter Contract

An **adapter** is a class that implements `AgentAdapter` (defined in `atp.adapters.base`). Adapters bridge the ATP Protocol and a specific agent runtime.

### Required Interface

```python
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from atp.protocol import ATPRequest, ATPResponse, ATPEvent

class AgentAdapter(ABC):

    @property
    @abstractmethod
    def adapter_type(self) -> str:
        """Return a unique string identifier for this adapter type."""

    @abstractmethod
    async def execute(self, request: ATPRequest) -> ATPResponse:
        """Execute a task synchronously and return the final response."""

    @abstractmethod
    def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """Execute a task with streaming. Yield ATPEvents, then yield ATPResponse."""
```

### MUST requirements

- `adapter_type` MUST return a non-empty string uniquely identifying the adapter (e.g., `"http"`, `"cli"`, `"langgraph"`).
- `execute()` MUST return an `ATPResponse` with a `task_id` matching `request.task_id`.
- `execute()` MUST set `status` to one of the `ResponseStatus` values. It MUST NOT raise exceptions for agent-level failures — use `status="failed"` with an `error` message instead.
- `execute()` MUST raise `AdapterError` (or subclass) for infrastructure failures (connection failures, timeouts, invalid responses from the agent).
- `stream_events()` MUST yield zero or more `ATPEvent` objects followed by exactly one `ATPResponse` as the final item.
- `stream_events()` MUST emit events with monotonically increasing `sequence` values starting at 0.
- The `version` field in all emitted messages MUST be set to the current protocol version.

### SHOULD requirements

- `health_check()` SHOULD return `True` if the agent endpoint is reachable, `False` otherwise.
- `cleanup()` SHOULD release open connections, processes, or other resources.
- Adapters SHOULD use `execute_with_tracing()` and `stream_events_with_tracing()` (provided by `AgentAdapter`) as the public entry points, so that OpenTelemetry spans are created automatically.
- Adapters SHOULD populate `ATPResponse.metrics` with whatever token/step/cost data is available.

### Adapter Configuration

All adapters accept an optional `AdapterConfig` (or subclass) at construction time.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timeout_seconds` | `float` | `300.0` | Maximum execution time in seconds. Must be > 0. |
| `retry_count` | `int` | `0` | Number of retries on transient failure. Must be ≥ 0. |
| `retry_delay_seconds` | `float` | `1.0` | Delay between retries in seconds. Must be ≥ 0. |
| `enable_cost_tracking` | `bool` | `True` | Whether to track costs via `CostTracker`. |

### Exceptions

| Exception | When to raise |
|-----------|--------------|
| `AdapterError` | Base class for all adapter failures. |
| `AdapterTimeoutError` | Agent did not respond within `timeout_seconds`. |
| `AdapterConnectionError` | Could not reach the agent endpoint. |
| `AdapterResponseError` | Agent returned a malformed or invalid response. |
| `AdapterNotFoundError` | Requested adapter type is not registered. |

---

## Validation Rules

This section summarizes all non-obvious validation rules enforced by the protocol models.

### task_id

- Non-empty after stripping whitespace.
- Pattern: `^[a-zA-Z0-9_-]+$` (alphanumeric, underscore, hyphen only). **Enforced in ATPRequest only, not in ATPResponse.**
- Max length: `MAX_TASK_ID_LENGTH` (128 chars).

### Artifact paths (ArtifactFile.path)

- Non-empty after stripping whitespace. No null bytes.
- Must be relative (no leading `/`).
- No `..` components in any path segment (protects against directory traversal).

### Artifact names (ArtifactStructured.name)

- Non-empty after stripping whitespace. No null bytes.
- Must not contain `/` or `\` (path separators not allowed in names).

### Context.tools_endpoint

- Must start with `http://` or `https://`.
- Empty string is normalized to `None`.

### Context.workspace_path

- No null bytes. Max `MAX_PATH_LENGTH`. Empty string normalized to `None`.

### Context.environment keys

- Must match `^[a-zA-Z_][a-zA-Z0-9_]*$`.
- No null bytes in keys or values.

### Task.expected_artifacts paths

- Non-empty. No `..`. No leading `/` or `~`.
- Max `MAX_PATH_LENGTH` per path.
- Must not resolve outside the current working directory (symlink traversal check).

---

## Error Handling

### Agent-level failures

When the agent itself fails (e.g., throws an exception, times out, is cancelled), the adapter MUST return an `ATPResponse` with an appropriate `status` value:

- `status="failed"` — agent encountered an error
- `status="timeout"` — agent exceeded the time limit
- `status="cancelled"` — execution was cancelled

The `error` field SHOULD contain a human-readable description of what went wrong. Max `MAX_ERROR_LENGTH` characters.

### Infrastructure failures

When the adapter itself cannot communicate with the agent (e.g., network error, container failed to start), the adapter MUST raise an `AdapterError` subclass rather than returning a response.

### Partial results

When an agent produces useful output before stopping (e.g., stops after 3 of 5 requested files), it may return `status="partial"` with whatever `artifacts` were produced. Callers should treat partial results as best-effort.

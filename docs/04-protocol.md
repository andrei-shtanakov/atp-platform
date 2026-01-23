# Agent Test Protocol (ATP) Specification

**Version**: 1.0  
**Status**: Draft  
**Last Updated**: 2025-01-21

---

## Overview

Agent Test Protocol (ATP) определяет стандартный способ взаимодействия между Test Platform и AI-агентами независимо от их реализации. Протокол описывает форматы запросов, ответов и событий.

## Design Goals

1. **Simplicity** — минимум обязательных полей, JSON-based
2. **Extensibility** — новые поля не ломают старые реализации
3. **Observability** — полный trace выполнения через события
4. **Language Agnostic** — любой язык может реализовать протокол

## Transport

ATP не привязан к конкретному транспорту. Поддерживаемые варианты:

| Transport | Request | Response | Events |
|-----------|---------|----------|--------|
| HTTP | POST body | Response body | SSE stream |
| WebSocket | Message | Message | Messages |
| stdin/stdout | Line (JSON) | Line (JSON) | stderr lines |
| gRPC | Unary call | Unary response | Server stream |

---

## Messages

### ATP Request

Запрос на выполнение задачи агентом.

```json
{
  "version": "1.0",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "task": {
    "description": "Find top 5 competitors for Slack in enterprise communication market",
    "input_data": {
      "company": "Slack",
      "market": "enterprise communication"
    },
    "expected_artifacts": [
      {
        "type": "file",
        "format": "markdown",
        "name": "report.md"
      }
    ]
  },
  "constraints": {
    "max_steps": 50,
    "max_tokens": 100000,
    "timeout_seconds": 300,
    "allowed_tools": ["web_search", "file_write"],
    "budget_usd": 1.0
  },
  "context": {
    "tools_endpoint": "http://test-harness:8080/tools",
    "workspace_path": "/workspace",
    "environment": {
      "API_KEY": "***"
    }
  },
  "metadata": {
    "test_id": "competitor_analysis_001",
    "run_number": 1,
    "total_runs": 5
  }
}
```

#### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Protocol version (e.g., "1.0") |
| `task_id` | string | Yes | Unique identifier for this execution (UUID) |
| `task` | object | Yes | Task definition |
| `task.description` | string | Yes | Natural language task description |
| `task.input_data` | object | No | Structured input data for the task |
| `task.expected_artifacts` | array | No | Hints about expected outputs |
| `constraints` | object | No | Execution constraints |
| `constraints.max_steps` | integer | No | Maximum agent steps/iterations |
| `constraints.max_tokens` | integer | No | Maximum LLM tokens to use |
| `constraints.timeout_seconds` | integer | No | Wall-clock timeout (default: 300) |
| `constraints.allowed_tools` | array | No | Whitelist of tools (null = all allowed) |
| `constraints.budget_usd` | number | No | Maximum cost budget |
| `context` | object | No | Execution context |
| `context.tools_endpoint` | string | No | URL for mock tools endpoint |
| `context.workspace_path` | string | No | Path to workspace directory |
| `context.environment` | object | No | Environment variables |
| `metadata` | object | No | Test metadata (for tracing) |

### ATP Response

Результат выполнения задачи.

```json
{
  "version": "1.0",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "artifacts": [
    {
      "type": "file",
      "path": "report.md",
      "content_type": "text/markdown",
      "size_bytes": 4523,
      "content_hash": "sha256:abc123...",
      "content": "# Competitor Analysis\n\n## Overview..."
    },
    {
      "type": "structured",
      "name": "competitors",
      "schema": "competitor_list",
      "data": {
        "competitors": [
          {"name": "Microsoft Teams", "market_share": 0.35},
          {"name": "Zoom", "market_share": 0.20}
        ]
      }
    }
  ],
  "metrics": {
    "total_tokens": 45000,
    "input_tokens": 15000,
    "output_tokens": 30000,
    "total_steps": 23,
    "tool_calls": 15,
    "llm_calls": 25,
    "wall_time_seconds": 120,
    "cost_usd": 0.45
  },
  "error": null,
  "trace_id": "trace_abc123"
}
```

#### Response Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Protocol version |
| `task_id` | string | Yes | Matching task_id from request |
| `status` | enum | Yes | Execution status |
| `artifacts` | array | Yes | Produced artifacts (may be empty) |
| `metrics` | object | Yes | Execution metrics |
| `error` | string | No | Error message if status is "failed" |
| `trace_id` | string | No | Reference to full execution trace |

#### Status Values

| Status | Description |
|--------|-------------|
| `completed` | Task finished successfully |
| `failed` | Task failed with error |
| `timeout` | Task exceeded time limit |
| `cancelled` | Task was cancelled |
| `partial` | Task partially completed |

#### Artifact Types

**File Artifact**:
```json
{
  "type": "file",
  "path": "output/report.md",
  "content_type": "text/markdown",
  "size_bytes": 4523,
  "content_hash": "sha256:...",
  "content": "..."  // Optional: inline content (base64 for binary)
}
```

**Structured Artifact**:
```json
{
  "type": "structured",
  "name": "analysis_result",
  "schema": "competitor_analysis_v1",
  "data": { ... }
}
```

**Reference Artifact** (large files):
```json
{
  "type": "reference",
  "path": "/workspace/large_file.csv",
  "content_type": "text/csv",
  "size_bytes": 15000000,
  "content_hash": "sha256:..."
}
```

### ATP Event

События, генерируемые агентом во время выполнения.

```json
{
  "version": "1.0",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-01-21T10:30:45.123Z",
  "sequence": 42,
  "event_type": "tool_call",
  "payload": {
    "tool": "web_search",
    "input": {
      "query": "Slack competitors enterprise"
    },
    "output": {
      "results": [...]
    },
    "duration_ms": 1500,
    "status": "success"
  }
}
```

#### Event Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Protocol version |
| `task_id` | string | Yes | Associated task |
| `timestamp` | string | Yes | ISO 8601 timestamp |
| `sequence` | integer | Yes | Monotonically increasing sequence number |
| `event_type` | enum | Yes | Type of event |
| `payload` | object | Yes | Event-specific data |

#### Event Types

**tool_call** — Tool invocation:
```json
{
  "event_type": "tool_call",
  "payload": {
    "tool": "web_search",
    "input": { "query": "..." },
    "output": { ... },
    "duration_ms": 1500,
    "status": "success",
    "error": null
  }
}
```

**llm_request** — LLM API call:
```json
{
  "event_type": "llm_request",
  "payload": {
    "model": "claude-sonnet-4-20250514",
    "input_tokens": 2000,
    "output_tokens": 500,
    "duration_ms": 3200,
    "temperature": 0.7,
    "stop_reason": "end_turn"
  }
}
```

**reasoning** — Agent's internal reasoning (optional):
```json
{
  "event_type": "reasoning",
  "payload": {
    "thought": "I need to search for competitor information first",
    "plan": ["Search web", "Extract companies", "Compare features"],
    "step": "planning"
  }
}
```

**state_change** — Agent state transition:
```json
{
  "event_type": "state_change",
  "payload": {
    "from_state": "researching",
    "to_state": "analyzing",
    "reason": "Collected enough data"
  }
}
```

**artifact_created** — New artifact produced:
```json
{
  "event_type": "artifact_created",
  "payload": {
    "artifact_type": "file",
    "path": "draft_report.md",
    "size_bytes": 2100
  }
}
```

**error** — Error occurred:
```json
{
  "event_type": "error",
  "payload": {
    "error_type": "tool_error",
    "message": "Web search API rate limited",
    "recoverable": true,
    "retry_after_ms": 5000
  }
}
```

**progress** — Progress update:
```json
{
  "event_type": "progress",
  "payload": {
    "current_step": 15,
    "total_steps": 25,
    "percentage": 60,
    "message": "Analyzing competitor features"
  }
}
```

---

## Schemas

### JSON Schema for ATP Request

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://atp.schema/request/v1",
  "title": "ATP Request",
  "type": "object",
  "required": ["version", "task_id", "task"],
  "properties": {
    "version": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+$"
    },
    "task_id": {
      "type": "string",
      "format": "uuid"
    },
    "task": {
      "type": "object",
      "required": ["description"],
      "properties": {
        "description": {
          "type": "string",
          "minLength": 1,
          "maxLength": 10000
        },
        "input_data": {
          "type": "object"
        },
        "expected_artifacts": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/expected_artifact"
          }
        }
      }
    },
    "constraints": {
      "type": "object",
      "properties": {
        "max_steps": {
          "type": "integer",
          "minimum": 1,
          "maximum": 1000
        },
        "max_tokens": {
          "type": "integer",
          "minimum": 1,
          "maximum": 10000000
        },
        "timeout_seconds": {
          "type": "integer",
          "minimum": 1,
          "maximum": 86400,
          "default": 300
        },
        "allowed_tools": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "budget_usd": {
          "type": "number",
          "minimum": 0
        }
      }
    },
    "context": {
      "type": "object",
      "properties": {
        "tools_endpoint": {
          "type": "string",
          "format": "uri"
        },
        "workspace_path": {
          "type": "string"
        },
        "environment": {
          "type": "object",
          "additionalProperties": {
            "type": "string"
          }
        }
      }
    },
    "metadata": {
      "type": "object"
    }
  },
  "$defs": {
    "expected_artifact": {
      "type": "object",
      "properties": {
        "type": {
          "type": "string",
          "enum": ["file", "structured"]
        },
        "format": {
          "type": "string"
        },
        "name": {
          "type": "string"
        }
      }
    }
  }
}
```

### JSON Schema for ATP Response

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://atp.schema/response/v1",
  "title": "ATP Response",
  "type": "object",
  "required": ["version", "task_id", "status", "artifacts", "metrics"],
  "properties": {
    "version": {
      "type": "string"
    },
    "task_id": {
      "type": "string",
      "format": "uuid"
    },
    "status": {
      "type": "string",
      "enum": ["completed", "failed", "timeout", "cancelled", "partial"]
    },
    "artifacts": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/artifact"
      }
    },
    "metrics": {
      "$ref": "#/$defs/metrics"
    },
    "error": {
      "type": ["string", "null"]
    },
    "trace_id": {
      "type": "string"
    }
  },
  "$defs": {
    "artifact": {
      "type": "object",
      "required": ["type"],
      "oneOf": [
        {
          "properties": {
            "type": { "const": "file" },
            "path": { "type": "string" },
            "content_type": { "type": "string" },
            "size_bytes": { "type": "integer" },
            "content_hash": { "type": "string" },
            "content": { "type": "string" }
          },
          "required": ["path"]
        },
        {
          "properties": {
            "type": { "const": "structured" },
            "name": { "type": "string" },
            "schema": { "type": "string" },
            "data": { "type": "object" }
          },
          "required": ["name", "data"]
        },
        {
          "properties": {
            "type": { "const": "reference" },
            "path": { "type": "string" },
            "content_type": { "type": "string" },
            "size_bytes": { "type": "integer" },
            "content_hash": { "type": "string" }
          },
          "required": ["path"]
        }
      ]
    },
    "metrics": {
      "type": "object",
      "properties": {
        "total_tokens": { "type": "integer" },
        "input_tokens": { "type": "integer" },
        "output_tokens": { "type": "integer" },
        "total_steps": { "type": "integer" },
        "tool_calls": { "type": "integer" },
        "llm_calls": { "type": "integer" },
        "wall_time_seconds": { "type": "number" },
        "cost_usd": { "type": "number" }
      }
    }
  }
}
```

### JSON Schema for ATP Event

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://atp.schema/event/v1",
  "title": "ATP Event",
  "type": "object",
  "required": ["version", "task_id", "timestamp", "sequence", "event_type", "payload"],
  "properties": {
    "version": {
      "type": "string"
    },
    "task_id": {
      "type": "string",
      "format": "uuid"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "sequence": {
      "type": "integer",
      "minimum": 0
    },
    "event_type": {
      "type": "string",
      "enum": [
        "tool_call",
        "llm_request",
        "reasoning",
        "state_change",
        "artifact_created",
        "error",
        "progress"
      ]
    },
    "payload": {
      "type": "object"
    }
  }
}
```

---

## Implementation Guide

### Minimal Agent Implementation (Python)

```python
#!/usr/bin/env python3
"""Minimal ATP-compatible agent."""

import json
import sys
from datetime import datetime, timezone
from uuid import uuid4

def main():
    # Read request from stdin
    request_line = sys.stdin.readline()
    request = json.loads(request_line)
    
    task_id = request["task_id"]
    description = request["task"]["description"]
    
    # Emit start event to stderr
    emit_event(task_id, 0, "progress", {
        "message": "Starting task",
        "percentage": 0
    })
    
    # Do actual work here...
    result = f"Processed: {description}"
    
    # Emit completion event
    emit_event(task_id, 1, "progress", {
        "message": "Task completed",
        "percentage": 100
    })
    
    # Write response to stdout
    response = {
        "version": "1.0",
        "task_id": task_id,
        "status": "completed",
        "artifacts": [
            {
                "type": "structured",
                "name": "result",
                "data": {"output": result}
            }
        ],
        "metrics": {
            "total_tokens": 0,
            "total_steps": 1,
            "tool_calls": 0,
            "llm_calls": 0,
            "wall_time_seconds": 1.0
        }
    }
    
    print(json.dumps(response))

def emit_event(task_id: str, seq: int, event_type: str, payload: dict):
    """Emit event to stderr."""
    event = {
        "version": "1.0",
        "task_id": task_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sequence": seq,
        "event_type": event_type,
        "payload": payload
    }
    print(json.dumps(event), file=sys.stderr)

if __name__ == "__main__":
    main()
```

### HTTP Endpoint Implementation (FastAPI)

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json

app = FastAPI()

class ATPRequest(BaseModel):
    version: str
    task_id: str
    task: dict
    constraints: dict | None = None
    context: dict | None = None
    metadata: dict | None = None

@app.post("/execute")
async def execute(request: ATPRequest):
    """Synchronous execution - returns response when done."""
    result = await run_agent(request)
    return result

@app.post("/execute/stream")
async def execute_stream(request: ATPRequest):
    """Streaming execution - SSE events + final response."""
    async def event_generator():
        async for event in run_agent_streaming(request):
            yield f"data: {json.dumps(event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

async def run_agent(request: ATPRequest) -> dict:
    # Implementation here
    pass

async def run_agent_streaming(request: ATPRequest):
    # Yield events during execution
    pass
```

---

## Versioning

### Semantic Versioning

Protocol versions follow semver: `MAJOR.MINOR`

- **MAJOR**: Breaking changes (new required fields, removed fields, changed semantics)
- **MINOR**: Backward-compatible additions (new optional fields, new event types)

### Compatibility Rules

1. Agents MUST include `version` in all messages
2. Platform MUST reject messages with unsupported major version
3. Platform SHOULD accept messages with higher minor version (ignore unknown fields)
4. Agents SHOULD support at least current major version

### Version Negotiation

For HTTP transport, version can be negotiated via headers:

```http
POST /execute HTTP/1.1
Accept: application/vnd.atp.v1+json
Content-Type: application/vnd.atp.v1+json
```

---

## Error Handling

### Error Response

When agent cannot process request:

```json
{
  "version": "1.0",
  "task_id": "...",
  "status": "failed",
  "artifacts": [],
  "metrics": {
    "total_tokens": 1500,
    "total_steps": 5,
    "wall_time_seconds": 45
  },
  "error": "Tool 'database_query' not available"
}
```

### Error Codes (Optional Extension)

```json
{
  "error": "Tool 'database_query' not available",
  "error_code": "TOOL_NOT_FOUND",
  "error_details": {
    "tool": "database_query",
    "available_tools": ["web_search", "file_write"]
  }
}
```

### Standard Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Malformed request |
| `TOOL_NOT_FOUND` | Requested tool not available |
| `TOOL_ERROR` | Tool execution failed |
| `LLM_ERROR` | LLM API call failed |
| `TIMEOUT` | Execution exceeded time limit |
| `BUDGET_EXCEEDED` | Cost budget exceeded |
| `INTERNAL_ERROR` | Unexpected internal error |

---

## Security Considerations

### Secrets

- Secrets MUST be passed via `context.environment`, not in `task.description`
- Agents MUST NOT log or emit events containing secret values
- Platform MUST sanitize secrets before storing traces

### Sandboxing

- Agents SHOULD be sandboxed (containers, VMs)
- `workspace_path` SHOULD be the only writable location
- Network access SHOULD be controllable by platform

### Input Validation

- Agents MUST validate all input against schema
- Agents MUST handle malformed input gracefully
- Agents MUST NOT execute arbitrary code from task description

---

## Appendix: Tool Protocol

When `context.tools_endpoint` is provided, agents call tools via HTTP:

```http
POST /tools/web_search HTTP/1.1
Content-Type: application/json

{
  "task_id": "...",
  "input": {
    "query": "Slack competitors"
  }
}
```

Response:
```json
{
  "status": "success",
  "output": {
    "results": [...]
  },
  "duration_ms": 1500
}
```

This allows platform to mock tools or record real tool calls for analysis.

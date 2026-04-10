# Contract: Code Writer Agent (ATP Protocol)

## 1. Overview

The agent receives a Python code-writing task and returns a ready file.

```
ATP Platform                          Code Writer Agent
┌────────────┐                       ┌─────────────────┐
│ Test Suite │──ATPRequest──────────►│ Accepts the task│
│  (YAML)    │                       │ Calls the LLM   │
│            │◄──ATPResponse─────────│ Returns code    │
└────────────┘                       └─────────────────┘
```

---

## 2. ATPRequest — what the agent receives

```json
{
  "version": "1.0",
  "task_id": "task_fibonacci_001",
  "task": {
    "description": "Write a Python function that computes the n-th Fibonacci number",
    "input_data": {
      "requirements": "Function fibonacci(n) accepts a non-negative integer n. Returns the n-th Fibonacci number. fibonacci(0)=0, fibonacci(1)=1. Raises ValueError for negative n.",
      "language": "python",
      "filename": "fibonacci.py"
    },
    "expected_artifacts": ["fibonacci.py"]
  },
  "constraints": {
    "max_steps": 3,
    "max_tokens": 10000,
    "timeout_seconds": 60,
    "budget_usd": 0.05
  },
  "context": {
    "workspace_path": "/workspace",
    "environment": {}
  },
  "metadata": {
    "test_id": "SM-001",
    "run_number": 1
  }
}
```

### 2.1. input_data fields

| Field | Type | Required | Description |
|-------|------|:---:|-------------|
| `requirements` | string | yes | Detailed description of code requirements |
| `language` | string | yes | Programming language (always `"python"`) |
| `filename` | string | yes | Artifact filename (e.g. `"fibonacci.py"`) |

---

## 3. ATPResponse — what the agent returns

```json
{
  "version": "1.0",
  "task_id": "task_fibonacci_001",
  "status": "completed",
  "artifacts": [
    {
      "type": "file",
      "path": "fibonacci.py",
      "content_type": "text/x-python",
      "content": "def fibonacci(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n"
    }
  ],
  "metrics": {
    "total_tokens": 850,
    "input_tokens": 320,
    "output_tokens": 530,
    "total_steps": 1,
    "tool_calls": 0,
    "llm_calls": 1,
    "wall_time_seconds": 3.2,
    "cost_usd": 0.004
  },
  "error": null
}
```

### 3.1. Response rules

1. **`status`**: `"completed"` if the code was generated, `"failed"` on LLM/API error
2. **`artifacts`**: exactly one file whose `path` equals `input_data.filename`
3. **`content`**: raw Python code with no markdown fences (no ` ```python `)
4. **`metrics`**: must populate `total_tokens`, `wall_time_seconds`, `cost_usd`

---

## 4. ATPEvent — streaming events (optional)

```json
{
  "version": "1.0",
  "task_id": "task_fibonacci_001",
  "timestamp": "2026-03-08T12:00:01Z",
  "sequence": 1,
  "event_type": "llm_request",
  "payload": {
    "model": "gpt-4o",
    "prompt_tokens": 320
  }
}
```

Agents may (but are not required to) stream events. ATP uses them for observability.

---

## 5. Statuses and errors

| Status | When | Example |
|--------|------|---------|
| `completed` | Code generated | Normal operation |
| `failed` | API error / cannot generate | API key invalid, rate limit |
| `timeout` | `timeout_seconds` exceeded | Model did not respond within 60s |
| `partial` | Code generated but incomplete | Truncated by `max_tokens` |

On `failed`, the `error` field contains a description:
```json
{
  "status": "failed",
  "artifacts": [],
  "error": "OpenAI API error: rate limit exceeded, retry after 20s"
}
```

---

## 6. Response validation (JSON Schema)

Used by the `schema` assertion in smoke tests:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["version", "task_id", "status", "artifacts", "metrics"],
  "properties": {
    "version": { "type": "string", "const": "1.0" },
    "task_id": { "type": "string", "minLength": 1 },
    "status": {
      "type": "string",
      "enum": ["completed", "failed", "timeout", "partial"]
    },
    "artifacts": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["type", "path", "content"],
        "properties": {
          "type": { "type": "string", "enum": ["file", "structured"] },
          "path": { "type": "string", "pattern": "\\.py$" },
          "content_type": { "type": "string" },
          "content": { "type": "string", "minLength": 1 }
        }
      }
    },
    "metrics": {
      "type": "object",
      "required": ["total_tokens", "wall_time_seconds", "cost_usd"],
      "properties": {
        "total_tokens": { "type": "integer", "minimum": 0 },
        "input_tokens": { "type": "integer", "minimum": 0 },
        "output_tokens": { "type": "integer", "minimum": 0 },
        "total_steps": { "type": "integer", "minimum": 0 },
        "llm_calls": { "type": "integer", "minimum": 0 },
        "wall_time_seconds": { "type": "number", "minimum": 0 },
        "cost_usd": { "type": "number", "minimum": 0 }
      }
    },
    "error": { "type": ["string", "null"] }
  }
}
```

---

## 7. Sample tasks (preview)

| Task | filename | Key requirements |
|------|----------|------------------|
| Fibonacci | `fibonacci.py` | fibonacci(n), ValueError for n<0, edge cases |
| CSV parser | `csv_parser.py` | read_csv(path), filter_rows(data, column, value), write_csv(data, path) |
| REST API client | `api_client.py` | get/post methods, retry, timeout, HTTP error handling |

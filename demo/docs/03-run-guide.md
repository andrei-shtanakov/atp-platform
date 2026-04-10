# Guide: running and comparing agents

> All commands are executed from the `atp-platform/` root.
> Demo project files live under `demo/`.

## 1. Preparation

### 1.1. API keys

```bash
cp demo/.env.example demo/.env
# Edit demo/.env and fill in real keys:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
source demo/.env
```

### 1.2. Dependencies

```bash
# From the atp-platform root
uv add fastapi uvicorn openai anthropic httpx
```

---

## 2. Running the agents

### 2.1. Terminal 1 — OpenAI Agent (port 8001)

```bash
cd atp-platform
export OPENAI_API_KEY=sk-...
uv run uvicorn demo.agents.openai_agent:app --port 8001
```

### 2.2. Terminal 2 — Anthropic Agent (port 8002)

```bash
cd atp-platform
export ANTHROPIC_API_KEY=sk-ant-...
uv run uvicorn demo.agents.anthropic_agent:app --port 8002
```

### 2.3. Health check

```bash
curl http://localhost:8001/health
# {"status":"ok","model":"gpt-4o-mini","provider":"openai"}

curl http://localhost:8002/health
# {"status":"ok","model":"claude-sonnet-4-20250514","provider":"anthropic"}
```

---

## 3. Running the tests

> All paths to YAML suites and fixtures are relative to the `atp-platform/` root.

### 3.1. Smoke tests (single agent)

```bash
# OpenAI
uv run atp test demo/test_suites/smoke.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001 \
  --adapter-config allow_internal=true \
  -v

# Anthropic
uv run atp test demo/test_suites/smoke.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8002 \
  --adapter-config allow_internal=true \
  -v
```

### 3.2. Functional tests (3 runs)

```bash
# OpenAI
uv run atp test demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001 \
  --adapter-config allow_internal=true \
  --runs=3 -v

# Anthropic
uv run atp test demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8002 \
  --adapter-config allow_internal=true \
  --runs=3 -v
```

### 3.3. Quality tests

```bash
# OpenAI
uv run atp test demo/test_suites/quality.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001 \
  --adapter-config allow_internal=true \
  -v

# Anthropic
uv run atp test demo/test_suites/quality.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8002 \
  --adapter-config allow_internal=true \
  -v
```

---

## 4. Saving results

### 4.1. JSON reports

```bash
# OpenAI
uv run atp test demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001 \
  --adapter-config allow_internal=true \
  --runs=3 \
  --output=json --output-file=demo/reports/openai_functional.json

# Anthropic
uv run atp test demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8002 \
  --adapter-config allow_internal=true \
  --runs=3 \
  --output=json --output-file=demo/reports/anthropic_functional.json
```

### 4.2. JUnit XML (for CI/CD)

```bash
uv run atp test demo/test_suites/smoke.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001 \
  --adapter-config allow_internal=true \
  --output=junit --output-file=demo/reports/openai_smoke.xml
```

---

## 5. Baselines

### 5.1. Save a baseline

```bash
# OpenAI baseline (10 runs)
uv run atp baseline save demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001 \
  --adapter-config allow_internal=true \
  -o demo/baselines/openai_baseline.json --runs=10

# Anthropic baseline
uv run atp baseline save demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8002 \
  --adapter-config allow_internal=true \
  -o demo/baselines/anthropic_baseline.json --runs=10
```

### 5.2. Compare against a baseline (regression)

```bash
uv run atp baseline compare demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001 \
  --adapter-config allow_internal=true \
  -b demo/baselines/openai_baseline.json
```

---

## 6. Manual curl test

```bash
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "version": "1.0",
    "task_id": "manual-001",
    "task": {
      "description": "Write a function add(a, b)",
      "input_data": {
        "requirements": "Function add(a, b) returns the sum of two numbers.",
        "language": "python",
        "filename": "add.py"
      },
      "expected_artifacts": ["add.py"]
    },
    "constraints": {
      "max_steps": 3,
      "max_tokens": 4096,
      "timeout_seconds": 30
    }
  }'
```

---

## 7. Common issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `Connection refused` | Agent not running | Start uvicorn on the expected port |
| `OPENAI_API_KEY not set` | Missing environment variable | `export OPENAI_API_KEY=sk-...` |
| `ModuleNotFoundError: openai` | Package not installed | `uv add openai` |
| `Timeout` | Model is slow to respond | Increase `timeout_seconds` in constraints |
| Code wrapped in ` ```python ` | LLM added markdown | Agent strips it automatically (strip_markdown_fences) |
| pytest fails: `ModuleNotFoundError` | Agent file not in the working directory | ATP places the artifact in a sandbox; pytest must run from there |
| `atp: command not found` | Not running from the atp-platform root | `cd atp-platform && uv run atp ...` |
| `Security validation failed` | localhost blocked by SSRF protection | Add `--adapter-config allow_internal=true` |

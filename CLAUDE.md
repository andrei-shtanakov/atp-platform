# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ATP (Agent Test Platform) is a framework-agnostic platform for testing and evaluating AI agents. It provides a unified protocol and infrastructure for testing agents regardless of their implementation framework (LangGraph, CrewAI, AutoGen, custom, etc.).

**Key principle**: Agent = black box with a contract (input → output + events via ATP Protocol).

## Development Commands

```bash
# Package management (ONLY use uv, never pip)
uv add <package>                    # Install package
uv run <tool>                       # Run tool
uv add --dev <package> --upgrade-package <package>  # Upgrade dev package

# Testing
uv run pytest tests/ -v --cov=atp --cov-report=term-missing  # All tests
uv run pytest tests/unit -v                                   # Unit tests only
uv run pytest tests/ -v -m "not slow"                        # Fast tests

# Code quality
uv run ruff format .               # Format code
uv run ruff check .                # Lint check
uv run ruff check . --fix          # Auto-fix lint issues
pyrefly check                      # Type checking (run after every change)

# Task management (parses spec/tasks.md)
python task.py list                # List all tasks
python task.py next                # Show ready tasks
python task.py start TASK-001      # Start a task
python task.py done TASK-001       # Complete a task

# Task executor (automated via Claude CLI)
python executor.py run             # Execute next task
python executor.py run --task=TASK-001  # Execute specific task
python executor.py status          # Check execution status

# CLI commands
uv run atp test suite.yaml --adapter=cli  # Run tests
uv run atp run suite.yaml --adapter=cli   # Alias for test
uv run atp list suite.yaml         # List tests in a suite
uv run atp validate --suite=suite.yaml    # Validate test suite
uv run atp baseline save/compare   # Baseline management
uv run atp list-agents             # List available adapters
uv run atp dashboard               # Start web dashboard
uv run atp tui                     # Start terminal UI (requires [tui] extra)
uv run atp init                    # Initialize ATP project
uv run atp generate                # Generate test suites
uv run atp benchmark               # Run benchmarks
uv run atp budget                  # Budget management
uv run atp experiment              # Run experiments
uv run atp plugins                 # Manage plugins
uv run atp game                    # Game-theoretic evaluation
uv run atp compare                 # Multi-model comparison
uv run atp estimate                # Cost estimation
uv run atp traces                  # Trace management
uv run atp replay                  # Replay agent traces
uv run atp version                 # Show version info
```

## Architecture

### Core Components

1. **Protocol** (`atp/protocol/`) - ATP Request/Response/Event models defining the contract
2. **Adapters** (`atp/adapters/`) - Translate between ATP Protocol and agent types (HTTP, Container, CLI, LangGraph, CrewAI, AutoGen, MCP, Bedrock, Vertex, Azure OpenAI)
3. **Runner** (`atp/runner/`) - Orchestrates test execution, manages sandboxes
4. **Evaluators** (`atp/evaluators/`) - Assess agent results (artifact, behavior, LLM-judge, code-exec, security, factuality, filesystem, style, performance)
5. **Reporters** (`atp/reporters/`) - Format output (console, JSON, HTML, JUnit, game)

### Data Flow

```
Test Definition (YAML) → Loader → Runner → Adapter → Agent → Response → Evaluators → Score Aggregator → Report
```

### ATP Protocol Messages

- **ATPRequest**: task description, constraints (max_steps, timeout, allowed_tools), context
- **ATPResponse**: status, artifacts, metrics (tokens, steps, cost)
- **ATPEvent**: streaming events (tool_call, llm_request, reasoning, error)

## Project Structure

```
atp/
├── cli/           # CLI entry point (atp test, validate, baseline, dashboard, etc.)
├── core/          # Config, exceptions, security utilities
├── protocol/      # ATP Request/Response/Event models
├── loader/        # YAML/JSON test parsing, filtering
├── runner/        # Test orchestration, sandbox, progress
├── adapters/      # Agent adapters (HTTP, Container, CLI, LangGraph, CrewAI, AutoGen, MCP, Bedrock, Vertex, Azure OpenAI)
├── evaluators/    # Result evaluation (artifact, behavior, LLM-judge, code-exec, security, factuality, filesystem, style, performance)
├── scoring/       # Score aggregation
├── statistics/    # Statistical analysis (mean, CI, stability)
├── baseline/      # Baseline storage, regression detection (Welch's t-test)
├── reporters/     # Output formatting (console, JSON, HTML, JUnit, game)
├── streaming/     # Event streaming, buffering, validation
├── mock_tools/    # Mock tool server for deterministic testing
├── performance/   # Profiling, caching, memory tracking
├── dashboard/     # Web interface (FastAPI, SQLAlchemy)
├── analytics/     # Cost tracking and analytics
├── benchmarks/    # Benchmark suites
├── chaos/         # Chaos testing
├── generator/     # Test suite generation
├── plugins/       # Plugin ecosystem management
├── sdk/           # Python SDK for programmatic test execution
├── tracing/       # Agent replay and trace management
└── tui/           # Terminal user interface (optional, requires [tui] extra)

spec/              # Task specifications and requirements
├── tasks.md       # Phase 4 task definitions with dependencies
├── phase5-tasks.md # Phase 5 game-theoretic evaluation tasks
├── requirements.md
├── phase5-requirements.md
├── design.md
├── phase5-design.md
└── WORKFLOW.md    # Development workflow guide

docs/              # Architecture documentation
├── 01-07*.md      # Vision, requirements, architecture, protocol, evaluators, integration, roadmap
├── adr/           # Architecture Decision Records
├── guides/        # Practical tutorials (quickstart, installation, usage, etc.)
└── reference/     # API reference, configuration, test format

tests/             # Test suite
├── unit/          # Unit tests (~70%)
├── integration/   # Integration tests (~20%)
├── contract/      # Protocol contract tests
├── e2e/           # End-to-end tests (~10%)
└── fixtures/      # Test fixtures and sample data
    └── test_site/ # Test e-commerce site for web search agents (port 9876)

examples/
├── test_suites/   # Sample test suite YAML files
├── games/         # Game-theoretic evaluation examples
├── search_agent/  # Web search agent for testing (no API keys required)
├── ci/            # CI/CD templates (GitHub, GitLab, Jenkins, Azure, CircleCI)
├── docker/        # Docker deployment examples
├── demo_agent.py  # File operations agent (no API keys required)
├── openai_agent.py # OpenAI-powered agent with tool calling
├── mcp_agent.py   # MCP-capable agent with OpenAI
└── mcp_simple_agent.py # Simple MCP agent without LLM
```

## Code Style

- Python 3.12+
- Type hints required for all code
- Line length: 88 characters
- Use pydantic for data models
- Docstrings for public APIs
- PEP 8 naming (snake_case functions, PascalCase classes, UPPER_SNAKE_CASE constants)

## Testing Requirements

- Unit test coverage ≥80% for new code
- Use `anyio` for async testing, not `asyncio`
- Test pyramid: ~70% unit, ~20% integration, ~10% e2e
- Test fixtures in `tests/fixtures/`

## Task Workflow

Tasks are defined in `spec/tasks.md` with dependencies and checklists. The `task.py` CLI manages task status, and `executor.py` can auto-execute tasks via Claude CLI.

Task status: `todo` → `in_progress` → `done` (or `blocked`)

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
```

## Architecture

### Core Components

1. **Protocol** (`atp/protocol/`) - ATP Request/Response/Event models defining the contract
2. **Adapters** (`atp/adapters/`) - Translate between ATP Protocol and agent types (HTTP, Docker, CLI, LangGraph, CrewAI)
3. **Runner** (`atp/runner/`) - Orchestrates test execution, manages sandboxes
4. **Evaluators** (`atp/evaluators/`) - Assess agent results (artifact checks, behavior analysis, LLM-as-judge)
5. **Reporters** (`atp/reporters/`) - Format output (console, JSON, HTML, JUnit)

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
├── cli/           # CLI entry point and commands
├── core/          # Config, registry, exceptions
├── protocol/      # ATP message models
├── loader/        # YAML/JSON test parsing
├── runner/        # Test orchestration, sandbox
├── adapters/      # Agent adapters
├── evaluators/    # Result evaluation
├── scoring/       # Score aggregation
└── reporters/     # Output formatting

spec/              # Task specifications and requirements
├── tasks.md       # Task definitions with dependencies
├── requirements.md
└── design.md

docs/              # Architecture documentation
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

# Agent Test Platform (ATP)

> Framework-agnostic platform for testing and evaluating AI agents

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://img.shields.io/badge/coverage-80%25+-green.svg)](https://github.com/yourusername/atp-platform-ru)

## Overview

ATP (Agent Test Platform) is a framework-agnostic platform for testing and evaluating AI agents. It provides a unified protocol and infrastructure for testing agents regardless of their implementation framework (LangGraph, CrewAI, AutoGen, custom, etc.).

**Key principle**: Agent = black box with a contract (input → output + events via ATP Protocol).

### The Problem

Modern AI agents are complex systems with non-deterministic behavior, multi-step logic, and dependencies on external tools. Traditional software testing approaches don't work for agents:

- **Stochasticity**: same prompt yields different results
- **Emergent behavior**: system behavior isn't the sum of components
- **Decision chains**: early errors manifest later
- **Framework dependency**: each team uses different stack

### The Solution

ATP provides:
- **Unified Protocol**: Standard interface for all agents
- **Declarative Testing**: YAML-based test definitions
- **Multi-Level Evaluation**: Artifact checks → behavior analysis → LLM-as-judge
- **Statistical Reliability**: Multiple runs with confidence intervals
- **Framework Agnostic**: Works with any agent implementation
- **CI/CD Ready**: JUnit XML, HTML reports, GitHub Actions integration

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/atp-platform-ru.git
cd atp-platform-ru

# Install dependencies (requires uv)
uv sync

# Verify installation
uv run pytest tests/ -v
```

### Run Your First Test

```bash
# Quick demo - run file operations agent (no API keys required)
uv run atp test examples/test_suites/demo_file_agent.yaml \
  --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["examples/demo_agent.py"]' \
  -v

# Run OpenAI-powered agent (requires OPENAI_API_KEY)
export OPENAI_API_KEY='sk-...'
uv run atp test examples/test_suites/openai_agent.yaml \
  --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["examples/openai_agent.py"]' \
  --adapter-config='inherit_environment=true' \
  --adapter-config='allowed_env_vars=["OPENAI_API_KEY","OPENAI_MODEL"]' \
  -v

# Run with multiple iterations for statistical reliability
uv run atp test suite.yaml --adapter=http \
  --adapter-config='endpoint=http://localhost:8000' \
  --runs=5

# Run specific tags
uv run atp test suite.yaml --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["agent.py"]' \
  --tags=smoke

# Generate JSON report
uv run atp test suite.yaml --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["agent.py"]' \
  --output=json --output-file=results.json
```

### Your First Test Suite

Create a test suite file `my_tests.yaml`:

```yaml
test_suite: "my_first_suite"
version: "1.0"
description: "My first ATP test suite"

defaults:
  runs_per_test: 3
  timeout_seconds: 180

agents:
  - name: "my-agent"
    type: "http"
    config:
      endpoint: "http://localhost:8000"

tests:
  - id: "test-001"
    name: "Basic file creation test"
    tags: ["smoke", "basic"]
    task:
      description: "Create a file named output.txt with content 'Hello, ATP!'"
      expected_artifacts: ["output.txt"]
    constraints:
      max_steps: 5
      timeout_seconds: 60
    assertions:
      - type: "artifact_exists"
        config:
          path: "output.txt"
      - type: "llm_eval"
        config:
          criteria: "completeness"
          threshold: 0.8
```

## Features

### Core Platform

✅ **Test Runner** - Full test orchestration with parallel execution
- Single test and suite execution
- Configurable parallelism (`--parallel`)
- Timeout enforcement (soft and hard)
- Progress reporting and fail-fast mode

✅ **Agent Adapters** - Connect to any agent type
- **HTTPAdapter** - REST/SSE endpoints
- **ContainerAdapter** - Docker-based agents
- **CLIAdapter** - Command-line agents
- **LangGraphAdapter** - Native LangGraph integration
- **CrewAIAdapter** - CrewAI framework support
- **AutoGenAdapter** - AutoGen legacy support

✅ **Evaluators** - Multi-level result assessment
- **ArtifactEvaluator** - File existence, content, schema validation
- **BehaviorEvaluator** - Tool usage, step limits, error checks
- **LLMJudgeEvaluator** - Semantic evaluation via Claude
- **CodeExecEvaluator** - Run generated code (pytest, npm, custom)

✅ **Reporters** - Multiple output formats
- **Console** - Colored terminal output with progress
- **JSON** - Structured results for automation
- **HTML** - Self-contained visual reports with charts
- **JUnit XML** - CI/CD integration (Jenkins, GitHub, GitLab)

### Advanced Features

✅ **Statistical Analysis** - Reliable metrics
- Multiple runs per test
- Mean, std, median, min/max
- 95% confidence intervals (t-distribution)
- Stability assessment

✅ **Baseline & Regression Detection**
- Save baseline results
- Compare runs with Welch's t-test
- Detect regressions (p < 0.05)
- Visual diff in console/JSON

✅ **CI/CD Integration**
- GitHub Actions workflow
- GitLab CI template
- Azure Pipelines, CircleCI, Jenkins examples
- Exit codes: 0=success, 1=failures, 2=error

✅ **Web Dashboard** (optional)
- FastAPI backend
- Results storage (SQLite/PostgreSQL)
- Historical trends
- Agent comparison

## Project Structure

```
atp-platform-ru/
├── atp/                      # Main package
│   ├── cli/                  # CLI commands (atp test, validate, baseline)
│   ├── core/                 # Config, exceptions, security
│   ├── protocol/             # ATP Request/Response/Event models
│   ├── loader/               # YAML/JSON test parsing
│   ├── runner/               # Test orchestration, sandbox
│   ├── adapters/             # Agent adapters (HTTP, Docker, LangGraph, etc.)
│   ├── evaluators/           # Result evaluation (artifact, behavior, LLM, code)
│   ├── scoring/              # Score aggregation
│   ├── statistics/           # Statistical analysis
│   ├── baseline/             # Baseline management, regression detection
│   ├── reporters/            # Output formatting (console, JSON, HTML, JUnit)
│   ├── streaming/            # Event streaming support
│   ├── mock_tools/           # Mock tool server for testing
│   ├── performance/          # Profiling, caching, optimization
│   └── dashboard/            # Web interface (FastAPI)
├── docs/                     # Documentation
├── examples/                 # Example test suites and CI configs
│   ├── test_suites/          # Sample test suites
│   └── ci/                   # CI/CD templates
├── tests/                    # Test suite (80%+ coverage)
│   ├── unit/                 # Unit tests
│   ├── e2e/                  # End-to-end tests
│   └── fixtures/             # Test fixtures
├── spec/                     # Requirements and design
│   ├── requirements.md
│   ├── design.md
│   └── tasks.md
└── pyproject.toml            # Project configuration
```

## CLI Commands

```bash
# Run tests with CLI adapter
uv run atp test <suite.yaml> --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["agent.py"]'

# Run tests with HTTP adapter
uv run atp test <suite.yaml> --adapter=http \
  --adapter-config='endpoint=http://localhost:8000'

# Run with multiple iterations and parallel execution
uv run atp test suite.yaml --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["agent.py"]' \
  --runs=5 --parallel=4

# Filter by tags
uv run atp test suite.yaml --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["agent.py"]' \
  --tags=smoke,core

# Output formats
uv run atp test suite.yaml --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["agent.py"]' \
  --output=json --output-file=results.json

uv run atp test suite.yaml --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["agent.py"]' \
  --output=junit --output-file=results.xml

# Pass environment variables (for API keys)
uv run atp test suite.yaml --adapter=cli \
  --adapter-config='command=python' \
  --adapter-config='args=["agent.py"]' \
  --adapter-config='inherit_environment=true' \
  --adapter-config='allowed_env_vars=["OPENAI_API_KEY","ANTHROPIC_API_KEY"]'

# Validate test definitions
uv run atp validate --suite=suite.yaml

# Baseline management
uv run atp baseline save suite.yaml -o baseline.json --runs=5
uv run atp baseline compare suite.yaml -b baseline.json

# Utilities
uv run atp list-agents          # List available adapters
uv run atp version              # Show version
uv run atp list suite.yaml      # List tests in a suite
```

## Documentation

### Getting Started

- [Installation Guide](docs/guides/installation.md) - Setup and dependencies
- [Quick Start Guide](docs/guides/quickstart.md) - First test suite
- [Basic Usage](docs/guides/usage.md) - Common workflows

### Reference

- [Test Format Reference](docs/reference/test-format.md) - YAML structure specification
- [Adapter Configuration](docs/reference/adapters.md) - Configure agent adapters
- [Configuration Reference](docs/reference/configuration.md) - All config options
- [API Reference](docs/reference/api-reference.md) - Python API
- [Troubleshooting](docs/reference/troubleshooting.md) - Common issues and solutions

### Architecture

- [Vision & Goals](docs/01-vision.md) - Project vision
- [Requirements](docs/02-requirements.md) - Functional requirements
- [Architecture](docs/03-architecture.md) - System architecture
- [ATP Protocol](docs/04-protocol.md) - Protocol specification
- [Evaluation System](docs/05-evaluators.md) - Metrics and evaluation
- [Integration Guide](docs/06-integration.md) - Agent integration
- [CI/CD Integration](docs/ci-cd.md) - CI/CD setup
- [Security](docs/security.md) - Security model

### Examples

See [examples/](examples/) for:
- [Test Suites](examples/test_suites/) - Sample test definitions
- [CI/CD Templates](examples/ci/) - GitHub Actions, GitLab CI, Jenkins, Azure, CircleCI
- [Demo Agents](examples/) - Ready-to-run example agents:
  - `demo_agent.py` - Simple file operations agent (no API keys needed)
  - `openai_agent.py` - OpenAI-powered agent with tool calling
  - `run_demo.sh` / `run_openai_demo.sh` - Quick start scripts

## Development

### Commands

```bash
# Testing
uv run pytest tests/ -v --cov=atp --cov-report=term-missing  # All tests with coverage
uv run pytest tests/unit -v                                   # Unit tests only
uv run pytest tests/ -v -m "not slow"                        # Fast tests

# Code quality
uv run ruff format .               # Format code
uv run ruff check .                # Lint check
uv run ruff check . --fix          # Auto-fix lint issues
pyrefly check                      # Type checking

# Task management
python task.py list                # List all tasks
python task.py next                # Show ready tasks
```

### Code Style

- Python 3.12+
- Type hints required for all code
- Line length: 88 characters
- Use Pydantic for data models
- Docstrings for public APIs
- Test coverage ≥80%

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass and code is formatted
5. Submit a pull request

See [CLAUDE.md](CLAUDE.md) for code style and development workflow.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/atp-platform-ru/issues)
- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)

## Status

**Current Status**: GA (General Availability) - All milestones complete

All core features implemented:
- ✅ MVP: Protocol, Adapters, Runner, Evaluators, Reporters, CLI
- ✅ Beta: Framework adapters, Statistics, LLM-Judge, Baseline, HTML reports, CI/CD
- ✅ GA: Dashboard, Security hardening, Performance optimization

See [spec/tasks.md](spec/tasks.md) for detailed task status.

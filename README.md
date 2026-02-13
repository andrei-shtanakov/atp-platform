# Agent Test Platform (ATP)

> Framework-agnostic platform for testing and evaluating AI agents

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://img.shields.io/badge/coverage-80%25+-green.svg)](https://github.com/yourusername/atp-platform)

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
git clone https://github.com/yourusername/atp-platform.git
cd atp-platform

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
- **AutoGenAdapter** - AutoGen framework support
- **MCPAdapter** - Model Context Protocol (MCP) tools/resources
- **BedrockAdapter** - AWS Bedrock integration
- **VertexAdapter** - Google Vertex AI integration
- **AzureOpenAIAdapter** - Azure OpenAI integration

✅ **Evaluators** - Multi-level result assessment
- **ArtifactEvaluator** - File existence, content, schema validation
- **BehaviorEvaluator** - Tool usage, step limits, error checks
- **LLMJudgeEvaluator** - Semantic evaluation via Claude
- **CodeExecEvaluator** - Run generated code (pytest, npm, custom)
- **SecurityEvaluator** - PII detection, secret leaks, code safety, prompt injection
- **FactualityEvaluator** - Claim extraction, citation checking, hallucination detection
- **StyleEvaluator** - Tone analysis, readability, formatting compliance
- **PerformanceEvaluator** - Latency, throughput, regression detection

✅ **Reporters** - Multiple output formats
- **Console** - Colored terminal output with progress
- **JSON** - Structured results for automation
- **HTML** - Self-contained visual reports with charts
- **JUnit XML** - CI/CD integration (Jenkins, GitHub, GitLab)
- **GameReporter / GameHTMLReporter** - Game-theoretic evaluation results

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
atp-platform/
├── atp/                      # Main package
│   ├── cli/                  # CLI commands (test, validate, baseline, dashboard, game, etc.)
│   ├── core/                 # Config, exceptions, security
│   ├── protocol/             # ATP Request/Response/Event models
│   ├── loader/               # YAML/JSON test parsing
│   ├── runner/               # Test orchestration, sandbox
│   ├── adapters/             # Agent adapters (HTTP, Docker, CLI, LangGraph, CrewAI, AutoGen, MCP, Bedrock, Vertex, Azure OpenAI)
│   ├── evaluators/           # Result evaluation (artifact, behavior, LLM, code, security, factuality, style, performance)
│   ├── scoring/              # Score aggregation
│   ├── statistics/           # Statistical analysis
│   ├── baseline/             # Baseline management, regression detection
│   ├── reporters/            # Output formatting (console, JSON, HTML, JUnit, game)
│   ├── streaming/            # Event streaming support
│   ├── mock_tools/           # Mock tool server for testing
│   ├── performance/          # Profiling, caching, optimization
│   ├── dashboard/            # Web interface (FastAPI)
│   ├── analytics/            # Cost tracking and analytics
│   ├── benchmarks/           # Benchmark suites
│   ├── chaos/                # Chaos testing
│   ├── generator/            # Test suite generation
│   ├── plugins/              # Plugin ecosystem management
│   ├── sdk/                  # Python SDK for programmatic use
│   ├── tracing/              # Agent replay and trace management
│   └── tui/                  # Terminal user interface (optional)
├── game-environments/        # Standalone game theory library (Phase 5)
│   └── game_envs/            # Games, strategies, analysis (Nash, exploitability)
├── atp-games/                # ATP plugin for game-theoretic evaluation (Phase 5)
│   └── atp_games/            # GameRunner, evaluators, YAML suites, tournaments
├── docs/                     # Documentation
├── examples/                 # Example test suites and CI configs
│   ├── test_suites/          # Sample test suites
│   ├── games/                # Game-theoretic evaluation examples
│   ├── docker/               # Docker deployment examples
│   └── ci/                   # CI/CD templates
├── tests/                    # Test suite (80%+ coverage)
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   ├── contract/             # Protocol contract tests
│   ├── e2e/                  # End-to-end tests
│   └── fixtures/             # Test fixtures
├── spec/                     # Working directory for specifications (managed by /spec-generator-skill)
│   ├── requirements.md       # Phase 4 feature requirements (REQ-XXX)
│   ├── phase5-requirements.md # Phase 5 game-theoretic requirements
│   ├── design.md             # Phase 4 technical design (DESIGN-XXX)
│   ├── phase5-design.md      # Phase 5 technical design
│   ├── tasks.md              # Phase 4 implementation tasks (TASK-XXX)
│   ├── phase5-tasks.md       # Phase 5 implementation tasks
│   └── WORKFLOW.md           # Task management workflow guide
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

# Additional commands
uv run atp init                 # Initialize ATP project
uv run atp generate             # Generate test suites
uv run atp benchmark            # Run benchmarks
uv run atp budget               # Budget management
uv run atp experiment           # Run experiments
uv run atp plugins              # Manage plugins
uv run atp game suite.yaml      # Game-theoretic evaluation
uv run atp tui                  # Terminal user interface
uv run atp compare              # Multi-model comparison
uv run atp estimate             # Cost estimation
uv run atp traces               # Trace management
uv run atp replay               # Replay agent traces
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
- [Dashboard API Reference](docs/reference/dashboard-api.md) - REST API for comparison, leaderboard, timeline
- [Troubleshooting](docs/reference/troubleshooting.md) - Common issues and solutions

### Architecture

- [Vision & Goals](docs/01-vision.md) - Project vision
- [Requirements](docs/02-requirements.md) - Functional requirements
- [Architecture](docs/03-architecture.md) - System architecture
- [ATP Protocol](docs/04-protocol.md) - Protocol specification
- [Evaluation System](docs/05-evaluators.md) - Metrics and evaluation
- [Integration Guide](docs/06-integration.md) - Agent integration
- [Roadmap](docs/07-roadmap.md) - Project roadmap and milestones
- [CI/CD Integration](docs/ci-cd.md) - CI/CD setup
- [Security](docs/security.md) - Security model
- [Architecture Decision Records](docs/adr/) - Key design decisions

### Game-Theoretic Evaluation

- [game-environments README](game-environments/README.md) - Game library: API, game dev guide, strategies, analysis tools
- [atp-games README](atp-games/README.md) - ATP plugin: quick start, YAML reference, evaluators, tournaments

### Examples

See [examples/](examples/) for:
- [Test Suites](examples/test_suites/) - Sample test definitions
- [Game Examples](examples/games/) - Game-theoretic evaluation ([README](examples/games/README.md), no API keys needed):
  - `basic_usage.py` - Run games, strategies, and tournaments
  - `custom_game.py` - Create a new game from scratch
  - `llm_agent_eval.py` - Evaluate agents on game battery
  - `population_dynamics.py` - Evolutionary simulation
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

- **Issues**: [GitHub Issues](https://github.com/yourusername/atp-platform/issues)
- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)

## Phase 5: Game-Theoretic Evaluation

ATP includes a game-theoretic evaluation framework for testing agent strategic reasoning, cooperation, and equilibrium play in multi-agent games.

### Packages

| Package | Description | Docs |
|---|---|---|
| [`game-environments`](game-environments/) | Standalone game theory library (zero ATP dependency) | [README](game-environments/README.md) |
| [`atp-games`](atp-games/) | ATP plugin for game-theoretic evaluation | [README](atp-games/README.md) |

### Built-in Games

Five canonical games with known Nash equilibria for rigorous evaluation:

- **Prisoner's Dilemma** -- cooperation vs defection with configurable payoff matrix
- **Public Goods Game** -- N-player contribution with multiplier and optional punishment
- **Auction** -- first-price and second-price sealed-bid with private values
- **Colonel Blotto** -- resource allocation across multiple battlefields
- **Congestion Game** -- network routing with latency-dependent costs

### Game-Theoretic Evaluators

- **PayoffEvaluator** -- average payoff, distribution, social welfare, Pareto efficiency
- **ExploitabilityEvaluator** -- best-response gap, empirical strategy extraction
- **CooperationEvaluator** -- cooperation rate, conditional cooperation, reciprocity
- **EquilibriumEvaluator** -- Nash distance, convergence detection, equilibrium classification

### Quick Start (Games)

```bash
# Run a built-in game suite
uv run atp test --suite=game:prisoners_dilemma.yaml

# Or use programmatically
```

```python
from game_envs import PrisonersDilemma, PDConfig, TitForTat, AlwaysDefect
from atp_games import GameRunner, GameRunConfig, BuiltinAdapter
import asyncio

async def main():
    game = PrisonersDilemma(PDConfig(num_rounds=50))
    agents = {
        "player_0": BuiltinAdapter(TitForTat()),
        "player_1": BuiltinAdapter(AlwaysDefect()),
    }
    runner = GameRunner()
    result = await runner.run_game(
        game=game, agents=agents,
        config=GameRunConfig(episodes=20, base_seed=42),
    )
    print(result.average_payoffs)

asyncio.run(main())
```

See [examples/games/](examples/games/) for more examples.

## Status

**Current Status**: GA (General Availability)

All core features implemented:
- ✅ MVP: Protocol, Adapters, Runner, Evaluators, Reporters, CLI
- ✅ Beta: Framework adapters, Statistics, LLM-Judge, Baseline, HTML reports, CI/CD
- ✅ GA: Dashboard, Security hardening, Performance optimization
- ✅ Phase 5: Game-theoretic evaluation (game-environments + atp-games)

### Specifications Directory

The `spec/` directory is a **working directory** for current development specifications, managed by the `/spec-generator-skill` Claude skill. It contains:
- `requirements.md` — Feature requirements in Kiro-style format (REQ-XXX)
- `design.md` — Technical design and architecture (DESIGN-XXX)
- `tasks.md` — Implementation tasks with dependencies (TASK-XXX)
- `WORKFLOW.md` — Task management and executor workflow guide

Specifications evolve with the project. See [spec/tasks.md](spec/tasks.md) for current task status.

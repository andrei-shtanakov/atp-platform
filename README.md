# Agent Test Platform (ATP)

> Framework-agnostic platform for testing and evaluating AI agents

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
- **Statistical Reliability**: Multiple runs to account for LLM stochasticity
- **Framework Agnostic**: Works with any agent implementation

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

### Load and Inspect Test Suite

```python
from atp.loader import TestLoader

# Load test suite
loader = TestLoader()
suite = loader.load_file("my_tests.yaml")

# Inspect loaded data
print(f"Suite: {suite.test_suite}")
print(f"Tests: {len(suite.tests)}")

for test in suite.tests:
    print(f"\nTest: {test.id} - {test.name}")
    print(f"  Tags: {test.tags}")
    print(f"  Max steps: {test.constraints.max_steps}")
    print(f"  Timeout: {test.constraints.timeout_seconds}s")
```

## Features

### Current (MVP - Test Loader)

✅ **Test Suite Loading**
- Load test definitions from YAML files
- Variable substitution with defaults (`${VAR:default}`)
- JSON Schema validation
- Pydantic model validation
- Defaults inheritance

✅ **Test Definition Model**
- Task description and constraints
- Assertion types (artifact_exists, behavior, llm_eval)
- Scoring weights (quality, completeness, efficiency, cost)
- Agent configuration

✅ **Validation Pipeline**
- YAML parsing with error tracking
- Semantic validation (duplicate IDs, weight sums)
- Type checking with detailed error messages

### Coming Soon

🚧 **Runner** - Test execution orchestration
🚧 **Adapters** - HTTP, Docker, CLI, LangGraph, CrewAI
🚧 **Evaluators** - Artifact checks, behavior analysis, LLM-as-judge
🚧 **Reporters** - Console, JSON, HTML, JUnit output

## Project Structure

```
atp-platform-ru/
├── atp/                      # Main package
│   ├── core/                 # Core exceptions and config
│   └── loader/               # Test suite loader (MVP)
│       ├── models.py         # Pydantic data models
│       ├── parser.py         # YAML parser
│       ├── loader.py         # TestLoader class
│       └── schema.py         # JSON Schema validation
├── docs/                     # Architecture documentation
│   ├── 01-vision.md
│   ├── 02-requirements.md
│   ├── 03-architecture.md
│   ├── 04-protocol.md
│   ├── 05-evaluators.md
│   ├── 06-integration.md
│   ├── 07-roadmap.md
│   └── adr/                  # Architecture Decision Records
├── examples/                 # Example code and test suites
├── tests/                    # Test suite (49 passing tests)
│   ├── unit/                 # Unit tests
│   └── fixtures/             # Test fixtures
├── spec/                     # Requirements and design
│   ├── requirements.md
│   ├── design.md
│   └── tasks.md
└── Makefile                  # Build commands
```

## Documentation

### Getting Started

- [Installation Guide](docs/guides/installation.md) - Setup and dependencies
- [Quick Start Guide](docs/guides/quickstart.md) - First test suite
- [Basic Usage](docs/guides/usage.md) - Common workflows

### Reference

- [Test Format Reference](docs/reference/test-format.md) - YAML structure specification
- [Adapter Configuration](docs/reference/adapters.md) - Configure agent adapters
- [Troubleshooting](docs/reference/troubleshooting.md) - Common issues and solutions

### Architecture

- [Vision & Goals](docs/01-vision.md) - Project vision
- [Requirements](docs/02-requirements.md) - Functional requirements
- [Architecture](docs/03-architecture.md) - System architecture
- [ATP Protocol](docs/04-protocol.md) - Protocol specification
- [Evaluation System](docs/05-evaluators.md) - Metrics and evaluation
- [Integration Guide](docs/06-integration.md) - Agent integration

### Examples

See [examples/test_suites/](examples/test_suites/) for complete test suite examples:
- Basic smoke tests
- Regression suites
- Cost analysis tests
- Multi-agent comparison

## Development

### Commands

```bash
# Testing
make test                    # All tests with coverage
make test-unit              # Unit tests only
uv run pytest tests/ -v     # Run tests directly

# Code quality
make format                 # Format code with ruff
make lint                   # Lint and type check
pyrefly check              # Type checking only

# Task management
make task-list             # List all tasks
make task-next             # Show ready tasks
python task.py start TASK-001  # Start a task
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

## Roadmap

See [docs/07-roadmap.md](docs/07-roadmap.md) for development roadmap.

**Current Status**: MVP Phase - Test Loader complete (v0.1.0)

**Next Steps**:
- Runner implementation
- HTTP/Docker adapters
- Basic evaluators
- Console reporter

# API Reference

Complete reference for ATP's Python API.

## Overview

This document provides detailed API documentation for all public classes, functions, and modules in ATP. The platform is designed to be extensible and framework-agnostic.

**Current Status**: GA (General Availability) - All core features implemented.

> **Looking for Dashboard REST API?** See [Dashboard API Reference](dashboard-api.md) for the complete REST API documentation including agent comparison, leaderboard, and timeline endpoints.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/atp-platform-ru.git
cd atp-platform-ru

# Install dependencies (requires uv)
uv sync

# Verify installation
python -c "import atp; print(atp.__version__)"
```

---

## Module: `atp.loader`

Test suite loading and parsing functionality.

### `TestLoader`

Main class for loading and validating test suites.

```python
from atp.loader import TestLoader
```

#### Constructor

```python
TestLoader(env: dict[str, str] | None = None)
```

Initialize test loader with optional environment variables.

**Parameters**:
- `env` (dict[str, str] | None): Custom environment for variable substitution. If None, uses `os.environ`.

**Example**:
```python
# Use system environment
loader = TestLoader()

# Use custom environment
loader = TestLoader(env={
    "API_KEY": "test-key",
    "API_ENDPOINT": "http://localhost:8000"
})
```

#### Methods

##### `load_file()`

```python
load_file(file_path: str | Path) -> TestSuite
```

Load test suite from YAML file.

**Parameters**:
- `file_path` (str | Path): Path to test suite YAML file

**Returns**:
- `TestSuite`: Validated test suite object

**Raises**:
- `ParseError`: If YAML parsing fails
- `ValidationError`: If schema or semantic validation fails

**Example**:
```python
from atp.loader import TestLoader

loader = TestLoader()
suite = loader.load_file("tests/my_suite.yaml")

print(f"Suite: {suite.test_suite}")
print(f"Tests: {len(suite.tests)}")
```

##### `load_string()`

```python
load_string(content: str) -> TestSuite
```

Load test suite from YAML string.

**Parameters**:
- `content` (str): YAML content as string

**Returns**:
- `TestSuite`: Validated test suite object

**Raises**:
- `ParseError`: If YAML parsing fails
- `ValidationError`: If schema or semantic validation fails

**Example**:
```python
yaml_content = """
test_suite: "inline-tests"
version: "1.0"

tests:
  - id: "test-001"
    name: "Basic test"
    task:
      description: "Test description"
"""

loader = TestLoader()
suite = loader.load_string(yaml_content)
```

---

## Module: `atp.loader.models`

Pydantic models for test definitions.

### `TestSuite`

Complete test suite with defaults and tests.

```python
from atp.loader import TestSuite
```

#### Attributes

- `test_suite` (str): Suite name (required)
- `version` (str): Suite version (default: "1.0")
- `description` (str | None): Optional suite description
- `defaults` (TestDefaults): Default settings for all tests
- `agents` (list[AgentConfig]): Agent configurations
- `tests` (list[TestDefinition]): Test definitions

#### Methods

##### `apply_defaults()`

```python
apply_defaults() -> None
```

Apply default settings to all tests. Merges suite-level defaults with test-specific configurations.

**Example**:
```python
suite = loader.load_file("suite.yaml")
# Defaults automatically applied during load
# Can be called manually after programmatic modifications
suite.apply_defaults()
```

---

### `TestDefinition`

Complete test definition.

#### Attributes

- `id` (str): Unique test identifier (required)
- `name` (str): Human-readable test name (required)
- `description` (str | None): Optional test description
- `tags` (list[str]): Test tags for filtering (default: [])
- `task` (TaskDefinition): Task specification (required)
- `constraints` (Constraints): Execution constraints (default: Constraints())
- `assertions` (list[Assertion]): Test assertions (default: [])
- `scoring` (ScoringWeights | None): Optional scoring weights override

**Example**:
```python
from atp.loader.models import TestDefinition, TaskDefinition, Constraints

test = TestDefinition(
    id="test-001",
    name="File creation test",
    task=TaskDefinition(
        description="Create a file named output.txt"
    ),
    constraints=Constraints(
        max_steps=10,
        timeout_seconds=60
    ),
    tags=["smoke", "basic"]
)
```

---

### `TaskDefinition`

Task specification for an agent.

#### Attributes

- `description` (str): Task description for the agent (required, min_length=1)
- `input_data` (dict[str, Any] | None): Optional input data
- `expected_artifacts` (list[str] | None): Expected output artifacts

**Example**:
```python
from atp.loader.models import TaskDefinition

task = TaskDefinition(
    description="Analyze competitor landscape for Slack",
    input_data={
        "company": "Slack",
        "market": "enterprise communication"
    },
    expected_artifacts=["report.md", "data.json"]
)
```

---

### `Constraints`

Execution constraints for a test.

#### Attributes

- `max_steps` (int | None): Maximum number of steps allowed
- `max_tokens` (int | None): Maximum tokens allowed
- `timeout_seconds` (int): Timeout in seconds (default: 300)
- `allowed_tools` (list[str] | None): List of allowed tools (None = all allowed)
- `budget_usd` (float | None): Budget limit in USD

**Example**:
```python
from atp.loader.models import Constraints

constraints = Constraints(
    max_steps=20,
    max_tokens=50000,
    timeout_seconds=180,
    allowed_tools=["web_search", "file_write"],
    budget_usd=0.50
)
```

---

### `Assertion`

Single assertion for test evaluation.

#### Attributes

- `type` (str): Assertion type (e.g., "artifact_exists", "llm_eval")
- `config` (dict[str, Any]): Assertion configuration (default: {})

**Example**:
```python
from atp.loader.models import Assertion

# File existence check
assertion1 = Assertion(
    type="artifact_exists",
    config={"path": "report.md"}
)

# LLM evaluation
assertion2 = Assertion(
    type="llm_eval",
    config={
        "artifact": "report.md",
        "criteria": "completeness",
        "threshold": 0.8
    }
)
```

---

### `ScoringWeights`

Scoring weights for test evaluation.

#### Attributes

- `quality_weight` (float): Quality score weight (default: 0.4, range: 0.0-1.0)
- `completeness_weight` (float): Completeness score weight (default: 0.3, range: 0.0-1.0)
- `efficiency_weight` (float): Efficiency score weight (default: 0.2, range: 0.0-1.0)
- `cost_weight` (float): Cost score weight (default: 0.1, range: 0.0-1.0)

**Note**: Weights should sum to approximately 1.0 (validated with ±0.01 tolerance).

**Example**:
```python
from atp.loader.models import ScoringWeights

# Prioritize quality over cost
weights = ScoringWeights(
    quality_weight=0.5,
    completeness_weight=0.3,
    efficiency_weight=0.15,
    cost_weight=0.05
)
```

---

### `AgentConfig`

Agent configuration.

#### Attributes

- `name` (str): Agent name (required)
- `type` (str | None): Agent type (e.g., "http", "docker", "langgraph")
- `config` (dict[str, Any]): Agent-specific configuration (default: {})

**Example**:
```python
from atp.loader.models import AgentConfig

agent = AgentConfig(
    name="my-agent",
    type="http",
    config={
        "endpoint": "http://localhost:8000",
        "api_key": "${API_KEY}",
        "timeout": 60
    }
)
```

---

### `TestDefaults`

Default settings for all tests in a suite.

#### Attributes

- `runs_per_test` (int): Number of runs per test (default: 1, min: 1)
- `timeout_seconds` (int): Default timeout (default: 300, min: 1)
- `scoring` (ScoringWeights): Default scoring weights
- `constraints` (Constraints | None): Default constraints for all tests

**Example**:
```python
from atp.loader.models import TestDefaults, ScoringWeights, Constraints

defaults = TestDefaults(
    runs_per_test=5,
    timeout_seconds=300,
    scoring=ScoringWeights(quality_weight=0.5, completeness_weight=0.3),
    constraints=Constraints(max_tokens=100000)
)
```

---

## Module: `atp.core.exceptions`

Exception hierarchy for ATP errors.

### Exception Hierarchy

```
ATPError (base)
└── LoaderError
    ├── ValidationError
    └── ParseError
```

### `ATPError`

Base exception for all ATP errors.

```python
from atp.core.exceptions import ATPError
```

**Usage**:
```python
try:
    # ATP operations
    pass
except ATPError as e:
    print(f"ATP error: {e}")
```

---

### `LoaderError`

Base exception for loader errors.

```python
from atp.core.exceptions import LoaderError
```

---

### `ValidationError`

Validation error with line number information.

```python
from atp.core.exceptions import ValidationError
```

#### Constructor

```python
ValidationError(
    message: str,
    line: int | None = None,
    column: int | None = None,
    file_path: str | None = None
)
```

**Parameters**:
- `message` (str): Error message
- `line` (int | None): Line number where error occurred
- `column` (int | None): Column number where error occurred
- `file_path` (str | None): File path where error occurred

#### Attributes

- `message` (str): Error message
- `line` (int | None): Line number
- `column` (int | None): Column number
- `file_path` (str | None): File path

**Example**:
```python
from atp.loader import TestLoader
from atp.core.exceptions import ValidationError

loader = TestLoader()

try:
    suite = loader.load_file("invalid.yaml")
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    if e.file_path:
        print(f"File: {e.file_path}")
    if e.line:
        print(f"Line: {e.line}")
```

---

### `ParseError`

YAML parsing error.

```python
from atp.core.exceptions import ParseError
```

**Example**:
```python
from atp.loader import TestLoader
from atp.core.exceptions import ParseError

loader = TestLoader()

try:
    suite = loader.load_file("malformed.yaml")
except ParseError as e:
    print(f"Parse error: {e}")
```

---

## Variable Substitution

ATP supports environment variable substitution in YAML files using the syntax `${VAR:default}`.

### Syntax

- `${VAR}`: Required variable (error if not set)
- `${VAR:default}`: Optional variable with default value

### Example

```yaml
agents:
  - name: "prod-agent"
    type: "http"
    config:
      endpoint: "${API_ENDPOINT:http://localhost:8000}"
      api_key: "${API_KEY}"  # Required
      timeout: "${TIMEOUT:60}"
```

Load with custom environment:

```python
from atp.loader import TestLoader

loader = TestLoader(env={
    "API_ENDPOINT": "https://api.production.com",
    "API_KEY": "secret-key-123"
    # TIMEOUT will use default: 60
})

suite = loader.load_file("suite.yaml")
```

---

## Complete Example

```python
from pathlib import Path
from atp.loader import TestLoader
from atp.core.exceptions import ValidationError, ParseError

def load_test_suite(file_path: str) -> None:
    """Load and inspect a test suite."""
    # Initialize loader
    loader = TestLoader(env={
        "API_KEY": "test-key",
        "API_ENDPOINT": "http://localhost:8000"
    })

    try:
        # Load test suite
        suite = loader.load_file(file_path)

        # Inspect suite
        print(f"Suite: {suite.test_suite} v{suite.version}")
        print(f"Description: {suite.description}")
        print(f"Default runs per test: {suite.defaults.runs_per_test}")

        # Inspect agents
        print(f"\nAgents ({len(suite.agents)}):")
        for agent in suite.agents:
            print(f"  - {agent.name} ({agent.type})")

        # Inspect tests
        print(f"\nTests ({len(suite.tests)}):")
        for test in suite.tests:
            print(f"\n  Test: {test.id}")
            print(f"    Name: {test.name}")
            print(f"    Tags: {test.tags}")
            print(f"    Task: {test.task.description[:50]}...")
            print(f"    Max steps: {test.constraints.max_steps}")
            print(f"    Timeout: {test.constraints.timeout_seconds}s")
            print(f"    Assertions: {len(test.assertions)}")

            # Inspect assertions
            for assertion in test.assertions:
                print(f"      - {assertion.type}: {assertion.config}")

    except ParseError as e:
        print(f"Parse error: {e}")
    except ValidationError as e:
        print(f"Validation error: {e.message}")
        if e.file_path:
            print(f"  File: {e.file_path}")
        if e.line:
            print(f"  Line: {e.line}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Usage
if __name__ == "__main__":
    load_test_suite("examples/test_suites/basic_suite.yaml")
```

---

## Type Annotations

ATP uses type hints throughout the codebase. All public APIs are fully typed.

```python
from typing import Any
from pathlib import Path
from atp.loader import TestLoader, TestSuite

# Type checking example
def process_suite(loader: TestLoader, path: Path) -> TestSuite:
    """Load and validate test suite."""
    suite: TestSuite = loader.load_file(path)
    return suite

# Custom environment
env_vars: dict[str, str] = {
    "API_KEY": "key",
    "ENDPOINT": "http://localhost"
}
loader: TestLoader = TestLoader(env=env_vars)
```

---

## Best Practices

### 1. Error Handling

Always handle specific exceptions:

```python
from atp.loader import TestLoader
from atp.core.exceptions import ValidationError, ParseError, LoaderError

try:
    suite = loader.load_file("suite.yaml")
except ParseError as e:
    # Handle YAML parsing errors
    log.error(f"Invalid YAML: {e}")
except ValidationError as e:
    # Handle validation errors
    log.error(f"Validation failed: {e.message} at line {e.line}")
except LoaderError as e:
    # Handle other loader errors
    log.error(f"Loader error: {e}")
```

### 2. Environment Variables

Use environment variables for secrets:

```python
import os
from atp.loader import TestLoader

# Load from system environment
loader = TestLoader()  # Uses os.environ

# Or provide custom environment
loader = TestLoader(env={
    "API_KEY": os.getenv("PROD_API_KEY"),
    "ENDPOINT": "https://api.production.com"
})
```

### 3. Path Handling

Use pathlib for cross-platform compatibility:

```python
from pathlib import Path
from atp.loader import TestLoader

loader = TestLoader()

# Platform-independent paths
suite_dir = Path(__file__).parent / "test_suites"
suite = loader.load_file(suite_dir / "my_suite.yaml")
```

### 4. Validation

Validate test suites early in your pipeline:

```python
from pathlib import Path
from atp.loader import TestLoader
from atp.core.exceptions import LoaderError

def validate_all_suites(suite_dir: Path) -> list[str]:
    """Validate all test suites in directory."""
    loader = TestLoader()
    errors = []

    for suite_file in suite_dir.glob("*.yaml"):
        try:
            loader.load_file(suite_file)
        except LoaderError as e:
            errors.append(f"{suite_file.name}: {e}")

    return errors
```

---

## See Also

- [Dashboard API Reference](dashboard-api.md) - REST API for dashboard, comparison, leaderboard, timeline
- [Test Format Reference](test-format.md) - YAML structure specification
- [Adapter Configuration](adapters.md) - Configure agent adapters
- [Usage Guide](../guides/usage.md) - Common workflows
- [Architecture](../03-architecture.md) - System architecture

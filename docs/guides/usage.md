# Basic Usage Guide

This guide covers common workflows and usage patterns for the ATP Platform.

## Table of Contents

1. [Loading Test Suites](#loading-test-suites)
2. [Working with Test Definitions](#working-with-test-definitions)
3. [Variable Substitution](#variable-substitution)
4. [Validation and Error Handling](#validation-and-error-handling)
5. [Working with Multiple Suites](#working-with-multiple-suites)
6. [Common Workflows](#common-workflows)

## Loading Test Suites

### Basic Loading

```python
from atp.loader import TestLoader

# Create loader instance
loader = TestLoader()

# Load from file
suite = loader.load_file("tests/my_suite.yaml")

# Load from string
yaml_content = """
test_suite: "inline_suite"
version: "1.0"
tests:
  - id: "test-001"
    name: "Inline test"
    task:
      description: "Test task"
"""
suite = loader.load_string(yaml_content)
```

### Loading with Environment Variables

```python
from atp.loader import TestLoader

# Option 1: Pass custom environment
loader = TestLoader(env={
    "API_KEY": "secret-123",
    "API_ENDPOINT": "https://api.example.com",
    "TIMEOUT": "300"
})
suite = loader.load_file("suite.yaml")

# Option 2: Use system environment
import os
os.environ["API_KEY"] = "secret-123"
loader = TestLoader()  # Automatically uses os.environ
suite = loader.load_file("suite.yaml")

# Option 3: Merge custom with system environment
loader = TestLoader(env={
    "API_KEY": "override-key",  # Overrides system env
    # Other system vars still accessible
})
```

### Loading Multiple Files

```python
from atp.loader import TestLoader
from pathlib import Path

loader = TestLoader()
suite_dir = Path("tests/suites")

# Load all YAML files in directory
suites = []
for suite_file in suite_dir.glob("*.yaml"):
    suite = loader.load_file(suite_file)
    suites.append(suite)

print(f"Loaded {len(suites)} test suites")
```

## Working with Test Definitions

### Accessing Suite Data

```python
from atp.loader import TestLoader

loader = TestLoader()
suite = loader.load_file("my_suite.yaml")

# Suite metadata
print(f"Suite name: {suite.test_suite}")
print(f"Version: {suite.version}")
print(f"Description: {suite.description}")

# Defaults
print(f"Default runs: {suite.defaults.runs_per_test}")
print(f"Default timeout: {suite.defaults.timeout_seconds}")

# Agents
for agent in suite.agents:
    print(f"Agent: {agent.name} ({agent.type})")
    print(f"  Config: {agent.config}")

# Tests
print(f"\nTotal tests: {len(suite.tests)}")
```

### Working with Tests

```python
# Iterate through tests
for test in suite.tests:
    print(f"\nTest: {test.id} - {test.name}")
    print(f"  Tags: {test.tags}")
    print(f"  Description: {test.task.description}")

# Filter tests by tag
smoke_tests = [t for t in suite.tests if "smoke" in t.tags]
print(f"Smoke tests: {len(smoke_tests)}")

regression_tests = [t for t in suite.tests if "regression" in t.tags]
print(f"Regression tests: {len(regression_tests)}")

# Find specific test
test = next((t for t in suite.tests if t.id == "test-001"), None)
if test:
    print(f"Found test: {test.name}")
```

### Accessing Test Components

```python
test = suite.tests[0]

# Task details
print(f"Task: {test.task.description}")
print(f"Input data: {test.task.input_data}")
print(f"Expected artifacts: {test.task.expected_artifacts}")

# Constraints
print(f"Max steps: {test.constraints.max_steps}")
print(f"Timeout: {test.constraints.timeout_seconds}s")
print(f"Max tokens: {test.constraints.max_tokens}")
print(f"Allowed tools: {test.constraints.allowed_tools}")
print(f"Budget: ${test.constraints.budget_usd}")

# Assertions
for assertion in test.assertions:
    print(f"Assertion: {assertion.type}")
    print(f"  Config: {assertion.config}")

# Scoring weights
if test.scoring:
    print(f"Quality weight: {test.scoring.quality_weight}")
    print(f"Completeness weight: {test.scoring.completeness_weight}")
    print(f"Efficiency weight: {test.scoring.efficiency_weight}")
    print(f"Cost weight: {test.scoring.cost_weight}")
```

### Modifying Test Data

```python
# Test definitions are Pydantic models - immutable by default
# To modify, create new instances or use model_copy()

test = suite.tests[0]

# Create modified copy
modified_test = test.model_copy(update={
    "constraints": test.constraints.model_copy(update={
        "timeout_seconds": 120
    })
})

print(f"Original timeout: {test.constraints.timeout_seconds}")
print(f"Modified timeout: {modified_test.constraints.timeout_seconds}")
```

## Variable Substitution

### Basic Substitution

YAML file:
```yaml
agents:
  - name: "prod-agent"
    type: "http"
    config:
      endpoint: "${API_ENDPOINT}"
      api_key: "${API_KEY}"
```

Python:
```python
from atp.loader import TestLoader

loader = TestLoader(env={
    "API_ENDPOINT": "https://api.production.com",
    "API_KEY": "prod-key-123"
})
suite = loader.load_file("suite.yaml")

agent = suite.agents[0]
print(agent.config["endpoint"])  # https://api.production.com
print(agent.config["api_key"])   # prod-key-123
```

### Default Values

YAML file:
```yaml
agents:
  - name: "agent"
    config:
      endpoint: "${API_ENDPOINT:http://localhost:8000}"
      timeout: "${TIMEOUT:30}"
      debug: "${DEBUG:false}"
```

Python:
```python
# If variables not set, defaults are used
loader = TestLoader()
suite = loader.load_file("suite.yaml")

agent = suite.agents[0]
print(agent.config["endpoint"])  # http://localhost:8000 (default)
print(agent.config["timeout"])   # 30 (default)
```

### Nested Substitution

YAML file:
```yaml
agents:
  - name: "agent"
    config:
      base_url: "${BASE_URL:https://api.example.com}"
      endpoints:
        chat: "${BASE_URL}/chat"
        search: "${BASE_URL}/search"
```

Python:
```python
loader = TestLoader(env={"BASE_URL": "https://prod.api.com"})
suite = loader.load_file("suite.yaml")

config = suite.agents[0].config
print(config["endpoints"]["chat"])    # https://prod.api.com/chat
print(config["endpoints"]["search"])  # https://prod.api.com/search
```

## Validation and Error Handling

### Handling Validation Errors

```python
from atp.loader import TestLoader
from atp.core.exceptions import ValidationError, ParseError

loader = TestLoader()

try:
    suite = loader.load_file("suite.yaml")
    print("✓ Suite loaded successfully")

except ParseError as e:
    print(f"✗ YAML parsing error: {e}")
    # File has invalid YAML syntax

except ValidationError as e:
    print(f"✗ Validation error: {e}")
    # File has invalid structure or values

except FileNotFoundError as e:
    print(f"✗ File not found: {e}")

except Exception as e:
    print(f"✗ Unexpected error: {e}")
```

### Validation Types

```python
# Duplicate test IDs
try:
    suite = loader.load_file("duplicate_ids.yaml")
except ValidationError as e:
    print(e)  # "Duplicate test IDs found: test-001"

# Invalid scoring weights
try:
    suite = loader.load_file("invalid_weights.yaml")
except ValidationError as e:
    print(e)  # "Scoring weights must sum to approximately 1.0"

# Unresolved variables
try:
    loader = TestLoader()  # No env vars set
    suite = loader.load_file("suite_with_vars.yaml")
except ValidationError as e:
    print(e)  # "Unresolved variable: ${API_KEY}"
```

## Working with Multiple Suites

### Loading Suite Collections

```python
from atp.loader import TestLoader
from pathlib import Path
from typing import Dict

def load_all_suites(directory: str) -> Dict[str, "TestSuite"]:
    """Load all test suites from directory."""
    loader = TestLoader()
    suites = {}

    suite_dir = Path(directory)
    for suite_file in suite_dir.glob("**/*.yaml"):
        try:
            suite = loader.load_file(suite_file)
            suites[suite.test_suite] = suite
            print(f"✓ Loaded {suite.test_suite}")
        except Exception as e:
            print(f"✗ Failed to load {suite_file}: {e}")

    return suites

# Load all suites
suites = load_all_suites("tests/suites")
print(f"\nTotal suites loaded: {len(suites)}")
```

### Merging Test Statistics

```python
def get_suite_statistics(suites: list) -> dict:
    """Get statistics across multiple suites."""
    total_tests = sum(len(suite.tests) for suite in suites)
    all_tags = set()
    test_count_by_tag = {}

    for suite in suites:
        for test in suite.tests:
            for tag in test.tags:
                all_tags.add(tag)
                test_count_by_tag[tag] = test_count_by_tag.get(tag, 0) + 1

    return {
        "total_suites": len(suites),
        "total_tests": total_tests,
        "unique_tags": len(all_tags),
        "tests_by_tag": test_count_by_tag,
        "avg_tests_per_suite": total_tests / len(suites) if suites else 0
    }

# Get statistics
stats = get_suite_statistics(list(suites.values()))
print(f"\nStatistics:")
print(f"  Total suites: {stats['total_suites']}")
print(f"  Total tests: {stats['total_tests']}")
print(f"  Unique tags: {stats['unique_tags']}")
print(f"  Avg tests/suite: {stats['avg_tests_per_suite']:.1f}")
```

## Common Workflows

### Workflow 1: Test Suite Validation

```python
from atp.loader import TestLoader
from atp.core.exceptions import ValidationError
import sys

def validate_suite(file_path: str) -> bool:
    """Validate a test suite file."""
    loader = TestLoader()

    try:
        suite = loader.load_file(file_path)
        print(f"✓ {file_path}")
        print(f"  Suite: {suite.test_suite}")
        print(f"  Tests: {len(suite.tests)}")
        return True

    except ValidationError as e:
        print(f"✗ {file_path}")
        print(f"  Validation error: {e}")
        return False

    except Exception as e:
        print(f"✗ {file_path}")
        print(f"  Error: {e}")
        return False

# Validate all suites in directory
from pathlib import Path
suite_dir = Path("tests/suites")
results = [validate_suite(f) for f in suite_dir.glob("*.yaml")]

if all(results):
    print("\n✓ All suites valid")
    sys.exit(0)
else:
    print(f"\n✗ {results.count(False)} suites failed validation")
    sys.exit(1)
```

### Workflow 2: Test Suite Report

```python
from atp.loader import TestLoader

def generate_suite_report(file_path: str) -> None:
    """Generate detailed report for a test suite."""
    loader = TestLoader()
    suite = loader.load_file(file_path)

    print(f"# Test Suite Report: {suite.test_suite}\n")
    print(f"**Version:** {suite.version}")
    print(f"**Description:** {suite.description or 'N/A'}\n")

    print(f"## Configuration\n")
    print(f"- Default runs per test: {suite.defaults.runs_per_test}")
    print(f"- Default timeout: {suite.defaults.timeout_seconds}s")
    print(f"- Agents configured: {len(suite.agents)}\n")

    print(f"## Tests ({len(suite.tests)} total)\n")

    for test in suite.tests:
        print(f"### {test.id}: {test.name}")
        print(f"- **Tags:** {', '.join(test.tags)}")
        print(f"- **Task:** {test.task.description}")
        print(f"- **Max steps:** {test.constraints.max_steps}")
        print(f"- **Timeout:** {test.constraints.timeout_seconds}s")
        print(f"- **Assertions:** {len(test.assertions)}")
        print()

# Generate report
generate_suite_report("tests/suites/smoke.yaml")
```

### Workflow 3: Environment-Specific Loading

```python
import os
from atp.loader import TestLoader

def load_suite_for_environment(
    file_path: str,
    environment: str
) -> "TestSuite":
    """Load test suite with environment-specific config."""

    # Define environment configurations
    env_configs = {
        "development": {
            "API_ENDPOINT": "http://localhost:8000",
            "API_KEY": "dev-key",
            "TIMEOUT": "60",
            "DEBUG": "true"
        },
        "staging": {
            "API_ENDPOINT": "https://staging.api.com",
            "API_KEY": os.getenv("STAGING_API_KEY"),
            "TIMEOUT": "120",
            "DEBUG": "false"
        },
        "production": {
            "API_ENDPOINT": "https://api.production.com",
            "API_KEY": os.getenv("PROD_API_KEY"),
            "TIMEOUT": "180",
            "DEBUG": "false"
        }
    }

    if environment not in env_configs:
        raise ValueError(f"Unknown environment: {environment}")

    # Load with environment config
    loader = TestLoader(env=env_configs[environment])
    suite = loader.load_file(file_path)

    print(f"Loaded suite for {environment} environment")
    return suite

# Usage
suite = load_suite_for_environment(
    "tests/api_suite.yaml",
    environment="staging"
)
```

### Workflow 4: Programmatic Suite Generation

```python
from atp.loader.models import (
    TestSuite,
    TestDefinition,
    TaskDefinition,
    Constraints,
    Assertion,
    AgentConfig,
    TestDefaults,
    ScoringWeights
)

def create_smoke_suite(agent_endpoint: str) -> TestSuite:
    """Programmatically create a smoke test suite."""

    # Define agent
    agent = AgentConfig(
        name="smoke-agent",
        type="http",
        config={"endpoint": agent_endpoint}
    )

    # Define tests
    tests = [
        TestDefinition(
            id=f"smoke-{i:03d}",
            name=f"Smoke test {i}",
            tags=["smoke"],
            task=TaskDefinition(
                description=f"Perform smoke test {i}"
            ),
            constraints=Constraints(
                max_steps=5,
                timeout_seconds=30
            ),
            assertions=[
                Assertion(
                    type="artifact_exists",
                    config={"path": f"output_{i}.txt"}
                )
            ]
        )
        for i in range(1, 6)
    ]

    # Create suite
    suite = TestSuite(
        test_suite="generated_smoke_suite",
        version="1.0",
        description="Programmatically generated smoke tests",
        defaults=TestDefaults(
            runs_per_test=1,
            timeout_seconds=60
        ),
        agents=[agent],
        tests=tests
    )

    return suite

# Generate suite
suite = create_smoke_suite("http://localhost:8000")
print(f"Generated {len(suite.tests)} smoke tests")
```

## Advanced Usage

### Custom Validation

```python
from atp.loader import TestLoader
from atp.core.exceptions import ValidationError

class CustomTestLoader(TestLoader):
    """Extended loader with custom validation."""

    def load_file(self, file_path: str) -> "TestSuite":
        # Load normally
        suite = super().load_file(file_path)

        # Custom validation
        self._validate_test_naming(suite)
        self._validate_tags(suite)

        return suite

    def _validate_test_naming(self, suite: "TestSuite") -> None:
        """Ensure test IDs follow naming convention."""
        for test in suite.tests:
            if not test.id.startswith("test-"):
                raise ValidationError(
                    f"Test ID must start with 'test-': {test.id}"
                )

    def _validate_tags(self, suite: "TestSuite") -> None:
        """Ensure all tests have required tags."""
        required_tags = {"smoke", "regression", "integration"}

        for test in suite.tests:
            if not any(tag in required_tags for tag in test.tags):
                raise ValidationError(
                    f"Test {test.id} must have at least one of: "
                    f"{required_tags}"
                )

# Use custom loader
loader = CustomTestLoader()
suite = loader.load_file("suite.yaml")
```

## Next Steps

- Learn about [Test Format Reference](../reference/test-format.md)
- Explore [Example Test Suites](../../examples/test_suites/)
- Read [Troubleshooting Guide](../reference/troubleshooting.md)

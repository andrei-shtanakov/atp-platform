# Configuration Reference

Complete configuration guide for ATP Platform.

## Overview

ATP uses YAML files for test suite configuration and supports environment variable substitution for sensitive data. This document provides a comprehensive reference for all configuration options.

## Configuration Files

ATP configuration is primarily done through test suite YAML files. Future versions will support additional configuration files.

### Test Suite Configuration

File: `test_suite.yaml`

Complete YAML structure:

```yaml
# Required: Suite identification
test_suite: string        # Suite name
version: string           # Suite version (default: "1.0")
description: string       # Optional suite description

# Optional: Default settings
defaults:
  runs_per_test: int      # Number of runs per test (default: 1)
  timeout_seconds: int    # Default timeout (default: 300)
  scoring:                # Default scoring weights
    quality_weight: float
    completeness_weight: float
    efficiency_weight: float
    cost_weight: float
  constraints:            # Default constraints
    max_steps: int
    max_tokens: int
    timeout_seconds: int
    allowed_tools: list[string]
    budget_usd: float

# Optional: Agent configurations
agents:
  - name: string          # Unique agent name
    type: string          # Adapter type
    config: dict          # Agent-specific config

# Required: Test definitions
tests:
  - id: string            # Unique test ID
    name: string          # Test name
    description: string   # Optional description
    tags: list[string]    # Test tags
    task:                 # Task definition
      description: string
      input_data: dict
      expected_artifacts: list[string]
    constraints:          # Execution constraints
      max_steps: int
      max_tokens: int
      timeout_seconds: int
      allowed_tools: list[string]
      budget_usd: float
    assertions:           # Test assertions
      - type: string
        config: dict
    scoring:              # Optional weight override
      quality_weight: float
      completeness_weight: float
      efficiency_weight: float
      cost_weight: float
```

---

## Top-Level Configuration

### `test_suite`

**Type**: `string` (required)

**Description**: Unique identifier for the test suite.

**Example**:
```yaml
test_suite: "competitor_analysis_tests"
```

**Validation**:
- Must be non-empty
- Should be unique across your test suite collection
- Convention: Use lowercase with underscores

---

### `version`

**Type**: `string` (default: `"1.0"`)

**Description**: Test suite version for tracking changes.

**Example**:
```yaml
version: "2.1.0"
```

**Validation**:
- Recommended: Semantic versioning (MAJOR.MINOR.PATCH)

---

### `description`

**Type**: `string` (optional)

**Description**: Human-readable description of the test suite.

**Example**:
```yaml
description: "Comprehensive tests for competitor analysis agents"
```

---

## Defaults Configuration

Default settings applied to all tests in the suite. Individual tests can override these defaults.

### `defaults.runs_per_test`

**Type**: `int` (default: `1`, min: `1`)

**Description**: Number of times to run each test. Higher values improve statistical reliability for LLM-based agents.

**Example**:
```yaml
defaults:
  runs_per_test: 5  # Run each test 5 times
```

**Guidelines**:
- Use `1` for deterministic tests or development
- Use `3-5` for production test suites
- Use `10+` for high-stakes evaluations

---

### `defaults.timeout_seconds`

**Type**: `int` (default: `300`, min: `1`)

**Description**: Default timeout for test execution in seconds.

**Example**:
```yaml
defaults:
  timeout_seconds: 600  # 10 minutes
```

**Guidelines**:
- Research tasks: 300-600 seconds
- Simple tasks: 60-120 seconds
- Complex multi-step tasks: 600-1800 seconds

---

### `defaults.scoring`

**Type**: `object`

**Description**: Default scoring weights for all tests.

**Attributes**:
- `quality_weight` (float, 0.0-1.0, default: 0.4)
- `completeness_weight` (float, 0.0-1.0, default: 0.3)
- `efficiency_weight` (float, 0.0-1.0, default: 0.2)
- `cost_weight` (float, 0.0-1.0, default: 0.1)

**Example**:
```yaml
defaults:
  scoring:
    quality_weight: 0.5      # Prioritize quality
    completeness_weight: 0.3 # Then completeness
    efficiency_weight: 0.15  # Some efficiency
    cost_weight: 0.05        # Cost is secondary
```

**Validation**:
- All weights must sum to ~1.0 (±0.01 tolerance)
- Each weight must be between 0.0 and 1.0

**Guidelines**:
- Quality-focused: Increase `quality_weight` for accuracy-critical tasks
- Cost-optimized: Increase `cost_weight` for budget-constrained scenarios
- Performance-focused: Increase `efficiency_weight` for latency-sensitive applications

---

### `defaults.constraints`

**Type**: `object` (optional)

**Description**: Default execution constraints for all tests.

**Example**:
```yaml
defaults:
  constraints:
    max_steps: 50
    max_tokens: 100000
    timeout_seconds: 300
    allowed_tools:
      - web_search
      - file_write
    budget_usd: 1.0
```

See [Constraints Configuration](#constraints-configuration) for details.

---

## Agent Configuration

Configure agents that will be tested.

### `agents[].name`

**Type**: `string` (required)

**Description**: Unique identifier for the agent.

**Example**:
```yaml
agents:
  - name: "langgraph-agent-v1"
```

**Validation**:
- Must be unique within the suite
- Convention: Use descriptive names with versions

---

### `agents[].type`

**Type**: `string` (optional)

**Description**: Adapter type for the agent.

**Valid Values**:
- `http` - REST API agent
- `docker` - Containerized agent
- `cli` - Command-line agent
- `langgraph` - LangGraph-based agent
- `crewai` - CrewAI-based agent
- `custom` - Custom adapter

**Example**:
```yaml
agents:
  - name: "my-agent"
    type: "http"
```

See [Adapter Configuration](adapters.md) for adapter-specific configuration.

---

### `agents[].config`

**Type**: `dict` (default: `{}`)

**Description**: Adapter-specific configuration.

**Example**:
```yaml
agents:
  - name: "api-agent"
    type: "http"
    config:
      endpoint: "${API_ENDPOINT:http://localhost:8000}"
      api_key: "${API_KEY}"
      timeout: 60
      headers:
        Content-Type: "application/json"
```

See [Adapter Configuration](adapters.md) for complete configuration options.

---

## Test Configuration

Individual test definitions.

### `tests[].id`

**Type**: `string` (required)

**Description**: Unique identifier for the test.

**Example**:
```yaml
tests:
  - id: "test-001-basic-search"
```

**Validation**:
- Must be unique within the suite
- Convention: Use prefixed numbering (e.g., `test-001`, `test-002`)

---

### `tests[].name`

**Type**: `string` (required)

**Description**: Human-readable test name.

**Example**:
```yaml
tests:
  - id: "test-001"
    name: "Find competitors for known company"
```

---

### `tests[].description`

**Type**: `string` (optional)

**Description**: Detailed test description.

**Example**:
```yaml
tests:
  - id: "test-001"
    name: "Competitor search"
    description: |
      Tests the agent's ability to find and analyze competitors
      for a well-known company in a specific market segment.
```

---

### `tests[].tags`

**Type**: `list[string]` (default: `[]`)

**Description**: Tags for filtering and organizing tests.

**Example**:
```yaml
tests:
  - id: "test-001"
    name: "Basic search"
    tags: ["smoke", "core", "search"]
```

**Common Tags**:
- `smoke` - Quick sanity checks
- `regression` - Regression tests
- `performance` - Performance benchmarks
- `edge_case` - Edge case handling
- `error_handling` - Error recovery tests

---

## Task Configuration

### `tests[].task.description`

**Type**: `string` (required, min_length: 1)

**Description**: Task description for the agent. This is the primary input to the agent.

**Example**:
```yaml
tests:
  - id: "test-001"
    task:
      description: |
        Find the top 5 competitors for Slack in the enterprise
        communication market. For each competitor, provide:
        - Company name and description
        - Market share estimates
        - Key differentiating features
        - Pricing model
```

**Guidelines**:
- Be specific and clear
- Include output format requirements
- Specify required information fields
- Set clear expectations

---

### `tests[].task.input_data`

**Type**: `dict` (optional)

**Description**: Structured input data for the agent.

**Example**:
```yaml
tests:
  - id: "test-001"
    task:
      description: "Analyze company competitors"
      input_data:
        company: "Slack"
        market: "enterprise communication"
        region: "North America"
        include_pricing: true
```

---

### `tests[].task.expected_artifacts`

**Type**: `list[string]` (optional)

**Description**: Expected output artifacts (files) from the agent.

**Example**:
```yaml
tests:
  - id: "test-001"
    task:
      description: "Generate competitor report"
      expected_artifacts:
        - "report.md"
        - "competitors.json"
        - "analysis.csv"
```

---

## Constraints Configuration

Execution limits and restrictions for tests.

### `constraints.max_steps`

**Type**: `int` (optional)

**Description**: Maximum number of agent steps/iterations allowed.

**Example**:
```yaml
tests:
  - id: "test-001"
    constraints:
      max_steps: 30
```

**Guidelines**:
- Simple tasks: 5-10 steps
- Medium complexity: 10-30 steps
- Complex tasks: 30-100 steps

---

### `constraints.max_tokens`

**Type**: `int` (optional)

**Description**: Maximum LLM tokens allowed (input + output).

**Example**:
```yaml
tests:
  - id: "test-001"
    constraints:
      max_tokens: 50000
```

**Guidelines**:
- Use for cost control
- Estimate: ~1000 tokens per simple interaction
- Include buffer for context and retries

---

### `constraints.timeout_seconds`

**Type**: `int` (default: `300`)

**Description**: Maximum execution time in seconds.

**Example**:
```yaml
tests:
  - id: "test-001"
    constraints:
      timeout_seconds: 180  # 3 minutes
```

---

### `constraints.allowed_tools`

**Type**: `list[string]` (optional, default: all tools allowed)

**Description**: Whitelist of tools the agent can use.

**Example**:
```yaml
tests:
  - id: "test-001"
    constraints:
      allowed_tools:
        - web_search
        - file_write
        - file_read
```

**Guidelines**:
- Omit for no restrictions
- Use to test specific capabilities
- Use to prevent dangerous operations

---

### `constraints.budget_usd`

**Type**: `float` (optional)

**Description**: Maximum cost budget in USD.

**Example**:
```yaml
tests:
  - id: "test-001"
    constraints:
      budget_usd: 0.50  # $0.50 max
```

---

## Assertions Configuration

Test validation rules.

### `assertions[].type`

**Type**: `string` (required)

**Description**: Type of assertion to perform.

**Valid Values**:
- `artifact_exists` - Check file existence
- `artifact_format` - Validate file format
- `artifact_schema` - Validate against JSON schema
- `contains` - Check content contains text/pattern
- `not_contains` - Check content doesn't contain text
- `min_length` - Minimum content length
- `max_length` - Maximum content length
- `sections_exist` - Check markdown sections
- `table_exists` - Check markdown tables
- `behavior` - Analyze execution behavior
- `llm_eval` - LLM-based evaluation
- `code_execution` - Execute and validate code

**Example**:
```yaml
tests:
  - id: "test-001"
    assertions:
      - type: "artifact_exists"
        config:
          path: "report.md"

      - type: "llm_eval"
        config:
          artifact: "report.md"
          criteria: "completeness"
          threshold: 0.8
```

See [Evaluators Documentation](../05-evaluators.md) for complete assertion reference.

---

### `assertions[].config`

**Type**: `dict` (default: `{}`)

**Description**: Assertion-specific configuration.

**Example for artifact_exists**:
```yaml
assertions:
  - type: "artifact_exists"
    config:
      path: "output.txt"
```

**Example for contains**:
```yaml
assertions:
  - type: "contains"
    config:
      artifact: "report.md"
      pattern: "Microsoft|Google|Amazon"
      regex: true
      min_matches: 2
```

**Example for llm_eval**:
```yaml
assertions:
  - type: "llm_eval"
    config:
      artifact: "report.md"
      criteria: "factual_accuracy"
      threshold: 0.85
      model: "gpt-4"
```

**Example for behavior**:
```yaml
assertions:
  - type: "behavior"
    config:
      must_use_tools:
        - web_search
      max_tool_calls: 15
      no_errors: true
```

---

## Environment Variables

ATP supports environment variable substitution in YAML files.

### Syntax

**Required variable**:
```yaml
config:
  api_key: "${API_KEY}"
```

**Variable with default**:
```yaml
config:
  endpoint: "${API_ENDPOINT:http://localhost:8000}"
  timeout: "${TIMEOUT:60}"
```

### Usage

**In YAML**:
```yaml
agents:
  - name: "prod-agent"
    type: "http"
    config:
      endpoint: "${API_ENDPOINT:http://localhost:8000}"
      api_key: "${API_KEY}"
      timeout: "${TIMEOUT:60}"
```

**In Python**:
```python
from atp.loader import TestLoader

# Use system environment
loader = TestLoader()

# Or provide custom environment
loader = TestLoader(env={
    "API_ENDPOINT": "https://api.production.com",
    "API_KEY": "secret-key",
    "TIMEOUT": "120"
})

suite = loader.load_file("suite.yaml")
```

### Best Practices

1. **Never hardcode secrets**:
```yaml
# Bad
config:
  api_key: "sk-abc123"

# Good
config:
  api_key: "${API_KEY}"
```

2. **Provide sensible defaults**:
```yaml
# Good - works without configuration
config:
  endpoint: "${API_ENDPOINT:http://localhost:8000}"
  timeout: "${TIMEOUT:60}"
```

3. **Document required variables**:
```yaml
# Required environment variables:
# - API_KEY: OpenAI API key
# - API_ENDPOINT: Agent API endpoint
```

---

## Complete Configuration Example

```yaml
# Competitor Analysis Test Suite
test_suite: "competitor_analysis_v2"
version: "2.0.0"
description: |
  Comprehensive test suite for competitor analysis agents.
  Tests research, analysis, and reporting capabilities.

# Default settings
defaults:
  runs_per_test: 5
  timeout_seconds: 600
  scoring:
    quality_weight: 0.5
    completeness_weight: 0.3
    efficiency_weight: 0.15
    cost_weight: 0.05
  constraints:
    max_steps: 50
    max_tokens: 100000
    budget_usd: 2.0

# Agent configurations
agents:
  # Production agent
  - name: "prod-langgraph"
    type: "langgraph"
    config:
      graph_path: "./agents/production_graph.py"
      model: "gpt-4-turbo"
      api_key: "${OPENAI_API_KEY}"

  # Baseline agent
  - name: "baseline-crewai"
    type: "crewai"
    config:
      crew_path: "./agents/baseline_crew.py"
      model: "gpt-3.5-turbo"
      api_key: "${OPENAI_API_KEY}"

# Test definitions
tests:
  # Basic functionality test
  - id: "test-001-basic-search"
    name: "Find competitors for known company"
    description: "Test basic competitor identification"
    tags: ["smoke", "core", "search"]

    task:
      description: |
        Find the top 5 competitors for Slack in the enterprise
        communication market. Provide name, description, and
        key features for each.
      input_data:
        company: "Slack"
        market: "enterprise communication"
      expected_artifacts:
        - "report.md"

    constraints:
      max_steps: 20
      timeout_seconds: 180
      allowed_tools:
        - web_search
        - file_write

    assertions:
      - type: "artifact_exists"
        config:
          path: "report.md"

      - type: "contains"
        config:
          artifact: "report.md"
          pattern: "Microsoft Teams|Zoom|Google"
          regex: true
          min_matches: 3

      - type: "min_length"
        config:
          artifact: "report.md"
          chars: 2000

      - type: "behavior"
        config:
          must_use_tools:
            - web_search
          max_tool_calls: 15

      - type: "llm_eval"
        config:
          artifact: "report.md"
          criteria: "completeness"
          threshold: 0.8

  # Edge case test
  - id: "test-002-unknown-company"
    name: "Handle unknown company gracefully"
    tags: ["edge_case", "error_handling"]

    task:
      description: |
        Find competitors for "XyzNonexistent123 Corp"
        in quantum computing.

    assertions:
      - type: "llm_eval"
        config:
          criteria: "custom"
          prompt: |
            Evaluate if the agent correctly indicated
            uncertainty and did not hallucinate competitors.
            Score 1.0 if handled well, 0.0 if hallucinated.
          threshold: 0.9

    scoring:
      quality_weight: 0.8  # Quality is critical for this test
      completeness_weight: 0.1
      efficiency_weight: 0.05
      cost_weight: 0.05

  # Performance test
  - id: "test-003-performance"
    name: "Efficiency on large market"
    tags: ["performance"]

    task:
      description: |
        Analyze the global cloud infrastructure market
        and identify top 10 providers.

    constraints:
      max_steps: 40
      timeout_seconds: 300

    assertions:
      - type: "artifact_exists"
        config:
          path: "report.md"

      - type: "behavior"
        config:
          max_tool_calls: 25

    scoring:
      quality_weight: 0.3
      completeness_weight: 0.2
      efficiency_weight: 0.4  # Efficiency focus
      cost_weight: 0.1
```

---

## Validation

ATP performs multiple validation passes:

### 1. YAML Syntax Validation

Ensures valid YAML structure.

### 2. JSON Schema Validation

Validates against test suite schema.

### 3. Semantic Validation

- Checks for duplicate test IDs
- Checks for duplicate agent names
- Validates scoring weights sum to ~1.0
- Validates constraint values

### 4. Model Validation

Pydantic model validation with type checking.

## Configuration Tips

### Development vs Production

**Development Configuration**:
```yaml
defaults:
  runs_per_test: 1          # Fast iteration
  timeout_seconds: 180      # Quick timeout
  scoring:
    quality_weight: 0.4
    completeness_weight: 0.4
    efficiency_weight: 0.1
    cost_weight: 0.1        # Cost not critical
```

**Production Configuration**:
```yaml
defaults:
  runs_per_test: 5          # Statistical reliability
  timeout_seconds: 600      # Allow completion
  scoring:
    quality_weight: 0.5
    completeness_weight: 0.3
    efficiency_weight: 0.15
    cost_weight: 0.05
```

### Configuration Organization

Organize test suites by purpose:

```
test_suites/
├── smoke/              # Quick sanity checks
│   └── basic.yaml
├── regression/         # Full regression suite
│   └── comprehensive.yaml
├── performance/        # Performance benchmarks
│   └── efficiency.yaml
└── integration/        # Integration tests
    └── e2e.yaml
```

---

## See Also

- [Test Format Reference](test-format.md) - Complete YAML format
- [Adapter Configuration](adapters.md) - Agent adapter configuration
- [API Reference](api-reference.md) - Python API documentation
- [Evaluators](../05-evaluators.md) - Assertion types and evaluation

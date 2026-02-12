# Test Format Reference

Complete reference for ATP test suite YAML format.

## Table of Contents

1. [Test Suite Structure](#test-suite-structure)
2. [Test Suite Fields](#test-suite-fields)
3. [Agent Configuration](#agent-configuration)
4. [Test Definition](#test-definition)
5. [Task Definition](#task-definition)
6. [Constraints](#constraints)
7. [Assertions](#assertions)
8. [Scoring Weights](#scoring-weights)
9. [Multi-Agent Tests](#multi-agent-tests)
10. [Variable Substitution](#variable-substitution)
11. [Complete Examples](#complete-examples)

## Test Suite Structure

A test suite YAML file has the following top-level structure:

```yaml
test_suite: string            # Required: Suite name
version: string               # Required: Version (default: "1.0")
description: string           # Optional: Suite description
defaults: TestDefaults        # Optional: Default settings
agents: list[AgentConfig]     # Required: Agent configurations
tests: list[TestDefinition]   # Required: Test definitions (min 1)
```

## Test Suite Fields

### test_suite

**Type:** `string`
**Required:** Yes
**Description:** Unique identifier for the test suite.

**Examples:**
```yaml
test_suite: "smoke_tests"
test_suite: "regression_suite_v2"
test_suite: "api_integration_tests"
```

### version

**Type:** `string`
**Required:** No
**Default:** `"1.0"`
**Description:** Version of the test suite format.

**Examples:**
```yaml
version: "1.0"
version: "2.3"
```

### description

**Type:** `string`
**Required:** No
**Description:** Human-readable description of the test suite purpose.

**Examples:**
```yaml
description: "Smoke tests for basic agent functionality"
description: "Comprehensive regression suite for data processing agents"
```

### defaults

**Type:** `TestDefaults`
**Required:** No
**Description:** Default settings inherited by all tests.

**Structure:**
```yaml
defaults:
  runs_per_test: int          # Number of times to run each test
  timeout_seconds: int        # Default timeout
  constraints: Constraints    # Default constraint values
  scoring: ScoringWeights     # Default scoring weights
```

**Example:**
```yaml
defaults:
  runs_per_test: 3
  timeout_seconds: 300
  constraints:
    max_steps: 20
    allowed_tools: ["file_read", "file_write", "web_search"]
  scoring:
    quality_weight: 0.4
    completeness_weight: 0.3
    efficiency_weight: 0.2
    cost_weight: 0.1
```

## Agent Configuration

**Type:** `AgentConfig`
**Required:** Yes (minimum 1 agent)
**Description:** Defines agents to test.

### Fields

#### name

**Type:** `string`
**Required:** Yes
**Description:** Unique agent identifier.

#### type

**Type:** `string`
**Required:** Yes
**Values:** `"http"`, `"container"`, `"cli"`, `"langgraph"`, `"crewai"`, `"autogen"`, `"mcp"`, `"bedrock"`, `"vertex"`, `"azure_openai"`
**Description:** Adapter type for the agent.

#### config

**Type:** `dict`
**Required:** Yes
**Description:** Agent-specific configuration. Structure depends on adapter type.

### Examples

#### HTTP Agent

```yaml
agents:
  - name: "api-agent"
    type: "http"
    config:
      endpoint: "https://api.example.com/agent"
      api_key: "${API_KEY}"
      timeout: 30
      headers:
        Content-Type: "application/json"
```

#### Container Agent

```yaml
agents:
  - name: "container-agent"
    type: "container"
    config:
      image: "my-agent:latest"
      environment:
        API_KEY: "${API_KEY}"
      volumes:
        - "./data:/data"
```

#### LangGraph Agent

```yaml
agents:
  - name: "langgraph-agent"
    type: "langgraph"
    config:
      graph_path: "./agents/my_graph.py"
      model: "gpt-4"
      temperature: 0.7
```

## Test Definition

**Type:** `TestDefinition`
**Required:** Yes (minimum 1 test)
**Description:** Defines a single test case.

### Fields

#### id

**Type:** `string`
**Required:** Yes
**Description:** Unique test identifier within the suite.
**Format:** Recommended format is `test-XXX` or `<prefix>-XXX`.

**Examples:**
```yaml
id: "test-001"
id: "smoke-042"
id: "regression-data-processing-001"
```

#### name

**Type:** `string`
**Required:** Yes
**Description:** Human-readable test name.

**Examples:**
```yaml
name: "Basic file creation"
name: "API endpoint integration test"
name: "Multi-step data processing workflow"
```

#### description

**Type:** `string`
**Required:** No
**Description:** Detailed test description.

**Example:**
```yaml
description: >
  This test validates that the agent can successfully create
  a file with specific content and verify its existence.
```

#### tags

**Type:** `list[string]`
**Required:** No
**Default:** `[]`
**Description:** Tags for categorizing and filtering tests.

**Common tags:**
- `smoke` - Quick validation tests
- `regression` - Detailed validation tests
- `integration` - Integration/E2E tests
- `slow` - Long-running tests
- `critical` - Critical functionality tests

**Example:**
```yaml
tags: ["smoke", "critical", "file-operations"]
```

#### task

**Type:** `TaskDefinition`
**Required:** Yes
**Description:** Defines what the agent should do. See [Task Definition](#task-definition).

#### constraints

**Type:** `Constraints`
**Required:** No
**Description:** Execution boundaries. See [Constraints](#constraints).

#### assertions

**Type:** `list[Assertion]`
**Required:** No
**Default:** `[]`
**Description:** Validation checks. See [Assertions](#assertions).

#### scoring

**Type:** `ScoringWeights`
**Required:** No
**Description:** Custom scoring weights for this test. See [Scoring Weights](#scoring-weights).

## Task Definition

Defines what the agent should accomplish.

### Fields

#### description

**Type:** `string`
**Required:** Yes
**Description:** Clear instruction for the agent.

**Best practices:**
- Be specific and actionable
- Include expected outcomes
- Avoid ambiguity

**Examples:**
```yaml
task:
  description: "Create a file named output.txt containing 'Hello, World!'"

task:
  description: >
    Analyze the provided CSV file and generate a JSON summary
    containing row count, column names, and basic statistics.

task:
  description: |
    Search for 'Python testing best practices' and save the
    top 3 results to a file named results.txt. Each result
    should include title, URL, and a brief summary.
```

#### input_data

**Type:** `dict`
**Required:** No
**Description:** Additional data to provide to the agent.

**Examples:**
```yaml
task:
  description: "Process the data"
  input_data:
    csv_path: "data/input.csv"
    output_format: "json"
    columns: ["id", "name", "value"]
```

#### expected_artifacts

**Type:** `list[string]`
**Required:** No
**Description:** Files or outputs the agent should create.

**Examples:**
```yaml
task:
  description: "Generate report"
  expected_artifacts:
    - "report.json"
    - "summary.txt"
    - "charts/revenue.png"
```

#### workspace_fixture

**Type:** `string`
**Required:** No
**Description:** Path to a fixture directory to pre-populate the agent's workspace before test execution. The entire directory tree is copied into the sandbox workspace using `shutil.copytree`. The original fixture is never modified.

Path can be absolute or relative (resolved from CWD).

**Examples:**
```yaml
task:
  description: "Read the config and return the project name"
  workspace_fixture: "tests/fixtures/test_filesystem/basic"

task:
  description: "Clean up temp files from the project"
  workspace_fixture: "tests/fixtures/test_filesystem/messy_directory"
```

**Behavior by adapter type:**

| Adapter | Workspace Access |
|---------|-----------------|
| CLI | Agent receives absolute host path via `context.workspace_path` |
| Container | Workspace is auto-mounted as a volume; `context.workspace_path` is rewritten to `/workspace` |

> For detailed usage, see [Test Filesystem Guide](../guides/test-filesystem.md).

## Constraints

Execution boundaries for agent behavior.

### Fields

#### max_steps

**Type:** `int`
**Required:** No
**Description:** Maximum number of reasoning/action steps.

**Example:**
```yaml
constraints:
  max_steps: 10
```

#### max_tokens

**Type:** `int`
**Required:** No
**Description:** Maximum token budget for LLM calls.

**Example:**
```yaml
constraints:
  max_tokens: 50000
```

#### timeout_seconds

**Type:** `int`
**Required:** No
**Default:** `300`
**Description:** Maximum execution time in seconds.

**Example:**
```yaml
constraints:
  timeout_seconds: 600  # 10 minutes
```

#### allowed_tools

**Type:** `list[string]`
**Required:** No
**Description:** Whitelist of tools the agent can use.

**Common tools:**
- `file_read` - Read files
- `file_write` - Write files
- `web_search` - Search the web
- `api_call` - Make API requests
- `python_repl` - Execute Python code
- `shell` - Execute shell commands

**Example:**
```yaml
constraints:
  allowed_tools:
    - "file_read"
    - "file_write"
    - "web_search"
```

#### budget_usd

**Type:** `float`
**Required:** No
**Description:** Maximum cost in USD for API calls.

**Example:**
```yaml
constraints:
  budget_usd: 0.50  # 50 cents max
```

### Complete Constraints Example

```yaml
constraints:
  max_steps: 15
  max_tokens: 100000
  timeout_seconds: 300
  allowed_tools:
    - "file_read"
    - "file_write"
    - "web_search"
  budget_usd: 1.0
```

## Assertions

Validation checks for test results.

### Assertion Structure

```yaml
assertions:
  - type: string      # Assertion type
    config: dict      # Type-specific configuration
```

### Assertion Types

#### artifact_exists

**Description:** Check if a file or artifact was created.

**Config:**
- `path` (string, required): Path to expected artifact

**Example:**
```yaml
assertions:
  - type: "artifact_exists"
    config:
      path: "output.txt"

  - type: "artifact_exists"
    config:
      path: "reports/summary.json"
```

#### behavior

**Description:** Validate agent behavior patterns.

**Config:**
- `check` (string, required): Behavior to validate

**Common checks:**
- `no_repeated_actions` - Agent doesn't repeat same action
- `efficient_tool_use` - Minimal tool calls
- `no_errors` - No error events emitted

**Example:**
```yaml
assertions:
  - type: "behavior"
    config:
      check: "no_repeated_actions"

  - type: "behavior"
    config:
      check: "efficient_tool_use"
```

#### llm_eval

**Description:** Use LLM to evaluate output quality.

**Config:**
- `criteria` (string, required): What to evaluate
- `threshold` (float, required): Minimum score (0.0-1.0)

**Common criteria:**
- `factual_accuracy` - Information is correct
- `completeness` - All requirements met
- `coherence` - Output is well-structured
- `relevance` - Output addresses the task

**Example:**
```yaml
assertions:
  - type: "llm_eval"
    config:
      criteria: "factual_accuracy"
      threshold: 0.8

  - type: "llm_eval"
    config:
      criteria: "completeness"
      threshold: 0.9
```

#### security

**Description:** Check for security vulnerabilities in agent outputs.

**Config:**
- `checks` (list[string], required): Security check types to run
- `sensitivity` (string, optional): Minimum severity to report (`info`, `low`, `medium`, `high`, `critical`)
- `pii_types` (list[string], optional): PII types to check (`email`, `phone`, `ssn`, `credit_card`, `api_key`)
- `injection_categories` (list[string], optional): Injection types (`injection`, `jailbreak`, `role_manipulation`)
- `code_categories` (list[string], optional): Code safety types (`dangerous_import`, `dangerous_function`, `file_operation`, `network_operation`)
- `secret_types` (list[string], optional): Secret types (`private_key`, `jwt_token`, `bearer_token`, `connection_string`, etc.)
- `fail_on_warning` (bool, optional): Fail on medium severity findings (default: false)

**Available checks:**
- `pii_exposure` - Detect personally identifiable information
- `prompt_injection` - Detect prompt injection and jailbreak attempts
- `code_safety` - Detect dangerous code patterns
- `secret_leak` - Detect leaked secrets and credentials

**Example:**
```yaml
assertions:
  - type: "security"
    config:
      checks:
        - pii_exposure
        - prompt_injection
        - code_safety
        - secret_leak
      sensitivity: "medium"
      fail_on_warning: true

  - type: "security"
    config:
      checks:
        - pii_exposure
      pii_types:
        - email
        - ssn
        - credit_card
```

> For detailed security evaluator documentation, see [Security Evaluator Guide](../guides/security-evaluator.md).

#### file_exists

**Description:** Check that a file exists in the workspace. Requires `workspace_fixture` on the task or `workspace_path` injected at runtime.

**Config:**
- `path` (string, required): Relative path within the workspace

**Example:**
```yaml
assertions:
  - type: "file_exists"
    config:
      path: "output.txt"

  - type: "file_exists"
    config:
      path: "reports/summary.json"
```

#### file_not_exists

**Description:** Check that a file does NOT exist in the workspace (e.g., agent should have deleted it).

**Config:**
- `path` (string, required): Relative path within the workspace

**Example:**
```yaml
assertions:
  - type: "file_not_exists"
    config:
      path: "temp/scratch.tmp"
```

#### file_contains

**Description:** Check that a file's content matches a plain text substring or regex pattern.

**Config:**
- `path` (string, required): Relative path within the workspace
- `pattern` (string, required): Text or regex pattern to search for
- `regex` (bool, optional): Treat pattern as regex (default: `false`)

**Example:**
```yaml
assertions:
  # Plain text match
  - type: "file_contains"
    config:
      path: "output.txt"
      pattern: "Processing complete"

  # Regex match
  - type: "file_contains"
    config:
      path: "report.json"
      pattern: '"count":\s*\d+'
      regex: true
```

#### dir_exists

**Description:** Check that a directory exists in the workspace.

**Config:**
- `path` (string, required): Relative path within the workspace

**Example:**
```yaml
assertions:
  - type: "dir_exists"
    config:
      path: "output/reports"
```

#### file_count

**Description:** Check the number of files in a directory.

**Config:**
- `path` (string, required): Relative path to directory within the workspace
- `count` (int, required): Expected file count
- `operator` (string, optional): Comparison operator (default: `"eq"`)

**Operators:** `eq`, `gt`, `gte`, `lt`, `lte`

**Example:**
```yaml
assertions:
  # Exact count
  - type: "file_count"
    config:
      path: "output"
      count: 3
      operator: "eq"

  # At least 1 file
  - type: "file_count"
    config:
      path: "output"
      count: 1
      operator: "gte"
```

> For detailed filesystem evaluator documentation, see [Test Filesystem Guide](../guides/test-filesystem.md).

### Multiple Assertions

```yaml
assertions:
  - type: "artifact_exists"
    config:
      path: "output.json"

  - type: "behavior"
    config:
      check: "no_errors"

  - type: "security"
    config:
      checks: [pii_exposure, secret_leak]
      sensitivity: "high"

  - type: "file_exists"
    config:
      path: "data/config.json"

  - type: "file_contains"
    config:
      path: "output.json"
      pattern: '"status": "success"'

  - type: "llm_eval"
    config:
      criteria: "completeness"
      threshold: 0.85

  - type: "llm_eval"
    config:
      criteria: "factual_accuracy"
      threshold: 0.90
```

## Scoring Weights

Define relative importance of evaluation dimensions.

### Fields

All weights must be between 0.0 and 1.0, and should sum to approximately 1.0.

#### quality_weight

**Type:** `float`
**Range:** `0.0` - `1.0`
**Default:** `0.4`
**Description:** Weight for output quality assessment.

#### completeness_weight

**Type:** `float`
**Range:** `0.0` - `1.0`
**Default:** `0.3`
**Description:** Weight for task completion assessment.

#### efficiency_weight

**Type:** `float`
**Range:** `0.0` - `1.0`
**Default:** `0.2`
**Description:** Weight for resource efficiency (steps, tokens).

#### cost_weight

**Type:** `float`
**Range:** `0.0` - `1.0`
**Default:** `0.1`
**Description:** Weight for API cost efficiency.

### Examples

#### Balanced Scoring

```yaml
scoring:
  quality_weight: 0.4
  completeness_weight: 0.3
  efficiency_weight: 0.2
  cost_weight: 0.1
```

#### Quality-Focused

```yaml
scoring:
  quality_weight: 0.6
  completeness_weight: 0.3
  efficiency_weight: 0.1
  cost_weight: 0.0
```

#### Cost-Optimized

```yaml
scoring:
  quality_weight: 0.3
  completeness_weight: 0.2
  efficiency_weight: 0.2
  cost_weight: 0.3
```

## Multi-Agent Tests

ATP supports running tests against multiple agents with different execution modes.

### Multi-Agent Fields

Tests can include additional fields for multi-agent execution:

#### agents

**Type:** `list[string]`
**Required:** No (required when `mode` is set)
**Description:** List of agent names to run this test against. Agent names must be defined in the suite-level `agents` section.

**Example:**
```yaml
tests:
  - id: "multi-001"
    name: "Compare agents"
    agents:
      - "gpt-4-agent"
      - "claude-agent"
      - "gemini-agent"
    mode: "comparison"
    task:
      description: "Generate a Python function"
```

#### mode

**Type:** `string`
**Required:** No (required when multiple agents are specified)
**Values:** `"comparison"`, `"collaboration"`, `"handoff"`
**Description:** Multi-agent execution mode.

**Modes:**
- `comparison` - Run the same test against multiple agents and compare results
- `collaboration` - Agents work together on a shared task
- `handoff` - Sequential agent execution with context passing

### Comparison Mode

Run the same test against multiple agents in parallel and compare results.

**Use cases:**
- Benchmark different LLM agents
- Evaluate agent performance across metrics
- A/B testing agents

#### comparison_config

**Type:** `ComparisonConfig`
**Required:** No
**Description:** Configuration for comparison mode.

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `metrics` | `list[string]` | `["quality", "speed", "cost"]` | Metrics to compare |
| `determine_winner` | `bool` | `true` | Determine and report winner |
| `parallel_execution` | `bool` | `true` | Run agents in parallel |

**Example:**
```yaml
tests:
  - id: "compare-001"
    name: "Compare code generation"
    mode: "comparison"
    agents:
      - "gpt-4-agent"
      - "claude-agent"
    comparison_config:
      metrics: ["quality", "speed", "cost"]
      determine_winner: true
      parallel_execution: true
    task:
      description: "Write a Python function to parse JSON"
```

### Collaboration Mode

Agents work together on a shared task, exchanging messages and artifacts.

**Use cases:**
- Multi-agent workflows
- Code review pipelines
- Consensus-based decision making

#### collaboration_config

**Type:** `CollaborationConfig`
**Required:** No
**Description:** Configuration for collaboration mode.

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_turns` | `int` | `10` | Maximum collaboration turns |
| `turn_timeout_seconds` | `float` | `60.0` | Timeout per turn |
| `require_consensus` | `bool` | `false` | Require all agents to agree |
| `allow_parallel_turns` | `bool` | `false` | Allow parallel execution |
| `coordinator_agent` | `string \| null` | `null` | Coordinating agent name |
| `termination_condition` | `string` | `"all_complete"` | When to end collaboration |

**Termination conditions:**
- `all_complete` - All agents complete their tasks
- `any_complete` - Any agent completes the task
- `consensus` - All agents reach consensus
- `max_turns` - Maximum turns reached

**Example:**
```yaml
tests:
  - id: "collab-001"
    name: "Collaborative code review"
    mode: "collaboration"
    agents:
      - "code-generator"
      - "code-reviewer"
      - "coordinator"
    collaboration_config:
      max_turns: 5
      require_consensus: false
      coordinator_agent: "coordinator"
      termination_condition: "all_complete"
    task:
      description: "Develop and review a caching system"
```

### Handoff Mode

Sequential agent execution where each agent passes context to the next.

**Use cases:**
- Pipeline workflows (generate -> review -> refine)
- Staged processing
- Multi-stage analysis

#### handoff_config

**Type:** `HandoffConfig`
**Required:** No
**Description:** Configuration for handoff mode.

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `handoff_trigger` | `string` | `"always"` | When to trigger handoff |
| `context_accumulation` | `string` | `"append"` | How to accumulate context |
| `max_context_size` | `int \| null` | `null` | Max context size (chars) |
| `allow_backtrack` | `bool` | `false` | Allow re-invoking previous agents |
| `final_agent_decides` | `bool` | `true` | Final agent makes decision |
| `agent_timeout_seconds` | `float` | `120.0` | Timeout per agent |
| `continue_on_failure` | `bool` | `false` | Continue on agent failure |

**Handoff triggers:**
- `always` - Always handoff after each agent
- `on_success` - Handoff only on success
- `on_failure` - Handoff only on failure
- `on_partial` - Handoff on partial completion
- `explicit` - Agent must request handoff

**Context accumulation modes:**
- `append` - Append all previous outputs
- `replace` - Only pass most recent output
- `merge` - Merge artifacts, keep latest values
- `summary` - Pass summarized context

**Example:**
```yaml
tests:
  - id: "handoff-001"
    name: "Code generation pipeline"
    mode: "handoff"
    agents:
      - "code-generator"
      - "code-reviewer"
      - "refiner"
    handoff_config:
      handoff_trigger: "always"
      context_accumulation: "append"
      final_agent_decides: true
      agent_timeout_seconds: 90
    task:
      description: "Generate, review, and refine a REST API client"
```

### Multi-Agent Validation Rules

1. **Mode requires agents**: When `mode` is set, `agents` must be specified
2. **Multiple agents require mode**: When multiple agents are listed, `mode` is required
3. **Collaboration/handoff need 2+ agents**: These modes require at least 2 agents
4. **Config consistency**: Mode-specific configs must match the mode
   - Comparison mode: Only `comparison_config` allowed
   - Collaboration mode: Only `collaboration_config` allowed
   - Handoff mode: Only `handoff_config` allowed
5. **Coordinator must exist**: `coordinator_agent` must be in the agents list
6. **Agents must be defined**: All agent names must be defined in suite-level `agents`

### Complete Multi-Agent Example

```yaml
test_suite: "multi_agent_suite"
version: "1.0"

agents:
  - name: "gpt-4-agent"
    type: "http"
    config:
      endpoint: "${GPT4_ENDPOINT}"

  - name: "claude-agent"
    type: "http"
    config:
      endpoint: "${CLAUDE_ENDPOINT}"

  - name: "reviewer"
    type: "http"
    config:
      endpoint: "${REVIEWER_ENDPOINT}"

tests:
  # Comparison test
  - id: "multi-001"
    name: "Compare code generation"
    mode: "comparison"
    agents: ["gpt-4-agent", "claude-agent"]
    comparison_config:
      metrics: ["quality", "speed", "cost"]
      determine_winner: true
    task:
      description: "Write a function to validate email addresses"

  # Collaboration test
  - id: "multi-002"
    name: "Collaborative design"
    mode: "collaboration"
    agents: ["gpt-4-agent", "claude-agent"]
    collaboration_config:
      max_turns: 3
      require_consensus: true
    task:
      description: "Design an API schema together"

  # Handoff test
  - id: "multi-003"
    name: "Pipeline processing"
    mode: "handoff"
    agents: ["gpt-4-agent", "reviewer", "claude-agent"]
    handoff_config:
      handoff_trigger: "always"
      context_accumulation: "append"
    task:
      description: "Generate code, review it, then refine it"
```

## Variable Substitution

Reference environment variables in YAML files.

### Syntax

```yaml
${VARIABLE_NAME}              # Required variable
${VARIABLE_NAME:default}      # Optional with default
```

### Examples

#### Basic Substitution

```yaml
agents:
  - name: "agent"
    config:
      endpoint: "${API_ENDPOINT}"
      api_key: "${API_KEY}"
```

#### With Defaults

```yaml
agents:
  - name: "agent"
    config:
      endpoint: "${API_ENDPOINT:http://localhost:8000}"
      timeout: "${TIMEOUT:30}"
      debug: "${DEBUG:false}"
```

#### Nested Substitution

```yaml
agents:
  - name: "agent"
    config:
      base_url: "${BASE_URL:https://api.example.com}"
      endpoints:
        chat: "${BASE_URL}/chat"
        search: "${BASE_URL}/search"
```

#### In Task Data

```yaml
tests:
  - id: "test-001"
    name: "API test"
    task:
      description: "Test API endpoint"
      input_data:
        endpoint: "${TEST_ENDPOINT}"
        credentials:
          username: "${TEST_USER}"
          password: "${TEST_PASSWORD}"
```

### Loading with Variables

```python
from atp.loader import TestLoader

# Set variables
loader = TestLoader(env={
    "API_ENDPOINT": "https://api.production.com",
    "API_KEY": "secret-key-123",
    "TIMEOUT": "60"
})

# Load suite
suite = loader.load_file("suite.yaml")
```

## Complete Examples

### Minimal Test Suite

```yaml
test_suite: "minimal_suite"
version: "1.0"

agents:
  - name: "test-agent"
    type: "http"
    config:
      endpoint: "http://localhost:8000"

tests:
  - id: "test-001"
    name: "Basic test"
    task:
      description: "Echo 'hello'"
```

### Smoke Test Suite

```yaml
test_suite: "smoke_tests"
version: "1.0"
description: "Quick validation of basic functionality"

defaults:
  runs_per_test: 1
  timeout_seconds: 60
  constraints:
    max_steps: 5

agents:
  - name: "smoke-agent"
    type: "http"
    config:
      endpoint: "${API_ENDPOINT:http://localhost:8000}"
      timeout: 30

tests:
  - id: "smoke-001"
    name: "Agent responds"
    tags: ["smoke", "critical"]
    task:
      description: "Echo 'hello world'"
    assertions:
      - type: "behavior"
        config:
          check: "no_errors"

  - id: "smoke-002"
    name: "Basic file creation"
    tags: ["smoke", "file-ops"]
    task:
      description: "Create file output.txt with content 'test'"
      expected_artifacts: ["output.txt"]
    assertions:
      - type: "artifact_exists"
        config:
          path: "output.txt"

  - id: "smoke-003"
    name: "Simple calculation"
    tags: ["smoke", "math"]
    task:
      description: "Calculate 42 * 58 and save to result.txt"
    assertions:
      - type: "artifact_exists"
        config:
          path: "result.txt"
      - type: "llm_eval"
        config:
          criteria: "factual_accuracy"
          threshold: 1.0
```

### Comprehensive Test Suite

```yaml
test_suite: "comprehensive_suite"
version: "1.0"
description: "Full regression test suite for data processing agent"

defaults:
  runs_per_test: 3
  timeout_seconds: 300
  constraints:
    max_steps: 20
    max_tokens: 100000
    allowed_tools:
      - "file_read"
      - "file_write"
      - "python_repl"
      - "web_search"
    budget_usd: 0.50
  scoring:
    quality_weight: 0.5
    completeness_weight: 0.3
    efficiency_weight: 0.15
    cost_weight: 0.05

agents:
  - name: "data-agent"
    type: "http"
    config:
      endpoint: "${API_ENDPOINT}"
      api_key: "${API_KEY}"
      timeout: 60

  - name: "baseline-agent"
    type: "http"
    config:
      endpoint: "${BASELINE_ENDPOINT}"
      api_key: "${API_KEY}"

tests:
  - id: "test-001"
    name: "CSV data analysis"
    tags: ["regression", "data", "csv"]
    description: "Analyze CSV file and generate summary statistics"
    task:
      description: >
        Read the file data.csv and generate a JSON summary with:
        - Total row count
        - Column names and types
        - Basic statistics (mean, median, std) for numeric columns
        Save results to summary.json
      input_data:
        csv_path: "fixtures/data.csv"
      expected_artifacts:
        - "summary.json"
    constraints:
      max_steps: 10
      timeout_seconds: 120
    assertions:
      - type: "artifact_exists"
        config:
          path: "summary.json"
      - type: "behavior"
        config:
          check: "efficient_tool_use"
      - type: "llm_eval"
        config:
          criteria: "completeness"
          threshold: 0.9
      - type: "llm_eval"
        config:
          criteria: "factual_accuracy"
          threshold: 0.95
    scoring:
      quality_weight: 0.6
      completeness_weight: 0.3
      efficiency_weight: 0.1
      cost_weight: 0.0

  - id: "test-002"
    name: "Multi-file data merge"
    tags: ["regression", "data", "merge"]
    task:
      description: >
        Merge data from users.csv and transactions.csv based on user_id.
        Create merged_data.csv with all columns.
      input_data:
        users_file: "fixtures/users.csv"
        transactions_file: "fixtures/transactions.csv"
      expected_artifacts:
        - "merged_data.csv"
    constraints:
      max_steps: 15
    assertions:
      - type: "artifact_exists"
        config:
          path: "merged_data.csv"
      - type: "llm_eval"
        config:
          criteria: "completeness"
          threshold: 0.85

  - id: "test-003"
    name: "Web research and synthesis"
    tags: ["integration", "web", "research"]
    task:
      description: >
        Research 'Python async programming best practices' and create
        a markdown document with top 5 practices, including sources.
      expected_artifacts:
        - "async_best_practices.md"
    constraints:
      max_steps: 25
      timeout_seconds: 600
      allowed_tools:
        - "web_search"
        - "file_write"
    assertions:
      - type: "artifact_exists"
        config:
          path: "async_best_practices.md"
      - type: "behavior"
        config:
          check: "no_repeated_actions"
      - type: "llm_eval"
        config:
          criteria: "completeness"
          threshold: 0.8
      - type: "llm_eval"
        config:
          criteria: "factual_accuracy"
          threshold: 0.85
```

## Validation Rules

ATP validates test suites with the following rules:

1. **Required fields**: `test_suite`, `version`, `agents`, `tests`
2. **Unique test IDs**: No duplicate `id` values in `tests`
3. **Scoring weights**: Must sum to approximately 1.0 (within 0.01)
4. **Value ranges**: Weights must be 0.0-1.0
5. **Minimum tests**: At least one test required
6. **Minimum agents**: At least one agent required
7. **Variable resolution**: All `${VAR}` without defaults must be provided

## See Also

- [Quick Start Guide](../guides/quickstart.md) - Create your first test suite
- [Usage Guide](../guides/usage.md) - Common workflows
- [Test Filesystem Guide](../guides/test-filesystem.md) - Workspace fixtures and filesystem assertions
- [Troubleshooting](troubleshooting.md) - Common issues
- [Examples](../../examples/test_suites/) - Complete examples

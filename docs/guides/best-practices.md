# Best Practices Guide

Guidelines for effectively using ATP to test and evaluate AI agents.

## Overview

This guide provides best practices for writing test suites, configuring agents, designing assertions, and interpreting results. Following these guidelines will help you create reliable, maintainable test suites that accurately evaluate agent behavior.

---

## Test Suite Design

### Start Small, Iterate

Begin with a minimal test suite and expand based on findings.

**Phase 1: Smoke Tests** (1-3 tests)
```yaml
tests:
  - id: "smoke-001"
    name: "Basic functionality check"
    tags: ["smoke"]
    task:
      description: "Create a simple file with specific content"
```

**Phase 2: Core Functionality** (5-10 tests)
Add tests for main use cases and common scenarios.

**Phase 3: Edge Cases** (10-20 tests)
Add edge cases, error handling, and performance tests.

**Phase 4: Comprehensive** (20+ tests)
Add stress tests, security tests, and integration tests.

### Organize by Purpose

Structure test suites by testing goal:

```
test_suites/
├── smoke_tests.yaml           # Quick sanity checks (1-2 min)
├── regression_suite.yaml      # Full regression (10-15 min)
├── performance_tests.yaml     # Performance benchmarks
├── edge_cases.yaml           # Edge case handling
└── integration_tests.yaml    # End-to-end scenarios
```

### Use Meaningful IDs and Names

**Good**:
```yaml
tests:
  - id: "search-001-basic-competitor"
    name: "Find competitors for well-known company"

  - id: "search-002-unknown-company"
    name: "Handle unknown company gracefully"

  - id: "search-003-large-market"
    name: "Performance on market with 20+ competitors"
```

**Bad**:
```yaml
tests:
  - id: "test1"
    name: "Test"

  - id: "t2"
    name: "Another test"
```

### Tag Strategically

Use tags for filtering and organization:

```yaml
tests:
  - id: "search-001"
    tags: ["smoke", "core", "search", "p0"]  # Priority 0, critical

  - id: "format-001"
    tags: ["formatting", "markdown", "p1"]   # Priority 1, important

  - id: "edge-001"
    tags: ["edge_case", "error_handling", "p2"]  # Priority 2, nice-to-have
```

Common tag patterns:
- **Functionality**: `search`, `analysis`, `formatting`, `research`
- **Priority**: `p0`, `p1`, `p2`
- **Type**: `smoke`, `regression`, `performance`, `integration`
- **Focus**: `core`, `edge_case`, `error_handling`, `security`

---

## Task Definition

### Be Specific and Clear

**Good**:
```yaml
task:
  description: |
    Find the top 5 competitors for Slack in the enterprise
    communication market. For each competitor, provide:
    1. Company name
    2. One-line description
    3. Estimated market share (if available)
    4. Top 3 differentiating features
    5. Pricing model (enterprise/SMB/freemium)

    Format: Create a markdown report with an executive summary
    section and a detailed analysis section for each competitor.
```

**Bad**:
```yaml
task:
  description: "Find Slack competitors"
```

### Specify Output Format

Always specify expected output structure:

```yaml
task:
  description: |
    Research Python web frameworks and create:

    1. File: frameworks.md
       - Markdown format
       - Sections: Overview, Comparison, Recommendations
       - Table comparing at least 5 frameworks

    2. File: frameworks.json
       - JSON array of framework objects
       - Each object: {name, description, stars, license, url}

  expected_artifacts:
    - "frameworks.md"
    - "frameworks.json"
```

### Provide Context When Needed

For complex tasks, provide structured input data:

```yaml
task:
  description: "Analyze competitor positioning for given company"
  input_data:
    company: "Slack"
    market: "enterprise communication"
    region: "North America"
    target_customers: "mid-to-large enterprises"
    price_range: "$6-15 per user/month"
```

---

## Constraints

### Set Realistic Limits

Don't over-constrain or under-constrain:

```yaml
# Good: Realistic constraints
constraints:
  max_steps: 30              # Enough for thorough research
  max_tokens: 50000          # Reasonable for the task
  timeout_seconds: 300       # 5 minutes
  allowed_tools:
    - web_search
    - file_write
  budget_usd: 1.0           # ~$1 for the task

# Too restrictive - likely to fail
constraints:
  max_steps: 5              # Too few
  timeout_seconds: 30       # Too short
  budget_usd: 0.05          # Too small

# Too permissive - no real limits
constraints:
  max_steps: 1000
  timeout_seconds: 3600
  budget_usd: 100.0
```

### Choose Constraints Based on Task Complexity

**Simple Tasks** (create file, format text):
```yaml
constraints:
  max_steps: 10
  timeout_seconds: 60
  max_tokens: 5000
```

**Medium Tasks** (research, analysis):
```yaml
constraints:
  max_steps: 30
  timeout_seconds: 300
  max_tokens: 50000
```

**Complex Tasks** (multi-step research, code generation):
```yaml
constraints:
  max_steps: 50
  timeout_seconds: 600
  max_tokens: 100000
```

### Use Tool Restrictions Intentionally

Restrict tools to test specific capabilities:

```yaml
# Test search capability only
constraints:
  allowed_tools:
    - web_search
    - file_write

# Test without external data access
constraints:
  allowed_tools:
    - file_read
    - file_write
    # No web_search or api_call
```

---

## Assertions

### Layer Assertions

Build assertions from simple to complex:

```yaml
assertions:
  # Layer 1: Structural checks (fast, deterministic)
  - type: "artifact_exists"
    config:
      path: "report.md"

  - type: "min_length"
    config:
      artifact: "report.md"
      chars: 1000

  # Layer 2: Content checks (fast, mostly deterministic)
  - type: "contains"
    config:
      artifact: "report.md"
      pattern: "Microsoft|Google|Amazon"
      regex: true

  - type: "sections_exist"
    config:
      artifact: "report.md"
      sections: ["Summary", "Analysis"]

  # Layer 3: Behavior checks (medium speed)
  - type: "behavior"
    config:
      must_use_tools: ["web_search"]
      max_tool_calls: 15

  # Layer 4: Semantic checks (slow, requires LLM)
  - type: "llm_eval"
    config:
      artifact: "report.md"
      criteria: "completeness"
      threshold: 0.8
```

### Set Appropriate Thresholds

Don't expect perfection from LLM evaluations:

```yaml
# Too strict - will have false negatives
assertions:
  - type: "llm_eval"
    config:
      criteria: "completeness"
      threshold: 0.95  # Too high

# Better - realistic threshold
assertions:
  - type: "llm_eval"
    config:
      criteria: "completeness"
      threshold: 0.75  # Reasonable

# Context-dependent
assertions:
  - type: "llm_eval"
    config:
      criteria: "factual_accuracy"
      threshold: 0.85  # Higher for accuracy-critical tasks

  - type: "llm_eval"
    config:
      criteria: "clarity"
      threshold: 0.70  # Lower for subjective criteria
```

### Use Multiple Runs for Statistical Reliability

For LLM-based evaluations, run tests multiple times:

```yaml
defaults:
  runs_per_test: 5  # Run each test 5 times

tests:
  - id: "test-001"
    task:
      description: "Complex task with LLM evaluation"

    assertions:
      - type: "llm_eval"
        config:
          criteria: "quality"
          threshold: 0.75
```

This accounts for:
- LLM non-determinism (different responses)
- Evaluation variance (judge subjectivity)
- Random failures (network issues, rate limits)

### Write Custom Prompts for Complex Criteria

For domain-specific evaluation:

```yaml
assertions:
  - type: "llm_eval"
    config:
      artifact: "report.md"
      criteria: "custom"
      prompt: |
        Evaluate if the competitor analysis includes:

        Required (must have all):
        - At least 5 competitors identified
        - Market share data or estimates for each
        - Key differentiating features (3+ per competitor)

        Optional (bonus points):
        - SWOT analysis
        - Pricing comparison table
        - Market trends section

        Scoring:
        - 1.0: All required + 2+ optional
        - 0.8: All required + 1 optional
        - 0.6: All required
        - <0.6: Missing required elements

        Provide score as float between 0 and 1.
      threshold: 0.7
```

---

## Scoring Configuration

### Adjust Weights by Test Purpose

**Quality-Critical Tests** (accuracy, correctness):
```yaml
scoring:
  quality_weight: 0.6       # Highest priority
  completeness_weight: 0.25
  efficiency_weight: 0.1
  cost_weight: 0.05
```

**Cost-Optimized Tests** (budget-constrained):
```yaml
scoring:
  quality_weight: 0.3
  completeness_weight: 0.2
  efficiency_weight: 0.2
  cost_weight: 0.3         # Highest priority
```

**Performance Tests** (latency-sensitive):
```yaml
scoring:
  quality_weight: 0.25
  completeness_weight: 0.25
  efficiency_weight: 0.4   # Highest priority
  cost_weight: 0.1
```

**Balanced** (default):
```yaml
scoring:
  quality_weight: 0.4
  completeness_weight: 0.3
  efficiency_weight: 0.2
  cost_weight: 0.1
```

### Use Defaults Wisely

Set suite-level defaults, override for specific tests:

```yaml
defaults:
  scoring:
    quality_weight: 0.4      # Default balance
    completeness_weight: 0.3
    efficiency_weight: 0.2
    cost_weight: 0.1

tests:
  - id: "accuracy-critical"
    name: "Financial data accuracy test"
    scoring:
      quality_weight: 0.7    # Override for this test
      completeness_weight: 0.2
      efficiency_weight: 0.05
      cost_weight: 0.05
```

---

## Environment Configuration

### Security Best Practices

**Never hardcode secrets**:
```yaml
# Bad - secrets exposed
agents:
  - name: "agent"
    type: "http"
    config:
      api_key: "sk-abc123"  # Don't do this!

# Good - use environment variables
agents:
  - name: "agent"
    type: "http"
    config:
      api_key: "${API_KEY}"
```

### Provide Defaults for Optional Config

Make configuration easy with sensible defaults:

```yaml
agents:
  - name: "agent"
    type: "http"
    config:
      endpoint: "${API_ENDPOINT:http://localhost:8000}"  # Default for dev
      timeout: "${TIMEOUT:60}"                           # Default timeout
      api_key: "${API_KEY}"                              # Required, no default
```

### Document Required Variables

In test suite or README:

```yaml
# Required Environment Variables:
# - API_KEY: OpenAI API key for LLM calls
# - API_ENDPOINT: Agent API endpoint (default: http://localhost:8000)
# - TIMEOUT: Request timeout in seconds (default: 60)

test_suite: "my_suite"
version: "1.0"
```

### Use Environment-Specific Configurations

```python
from atp.loader import TestLoader

# Development environment
dev_env = {
    "API_ENDPOINT": "http://localhost:8000",
    "API_KEY": "dev-key",
    "TIMEOUT": "30"
}

# Production environment
prod_env = {
    "API_ENDPOINT": "https://api.production.com",
    "API_KEY": os.getenv("PROD_API_KEY"),
    "TIMEOUT": "120"
}

# Load with appropriate environment
env = prod_env if is_production else dev_env
loader = TestLoader(env=env)
suite = loader.load_file("suite.yaml")
```

---

## Running Tests

### Development Workflow

1. **Start with single run**:
```yaml
defaults:
  runs_per_test: 1  # Fast iteration during development
```

2. **Test incrementally**:
```python
# Test single test case
loader = TestLoader()
suite = loader.load_file("suite.yaml")
single_test = [t for t in suite.tests if t.id == "test-001"][0]
# Run only this test
```

3. **Increase runs for validation**:
```yaml
defaults:
  runs_per_test: 3  # Validate with multiple runs
```

4. **Full suite for CI/CD**:
```yaml
defaults:
  runs_per_test: 5  # Production-level reliability
```

### Continuous Integration

**Fast Feedback** (smoke tests on every commit):
```yaml
# smoke_tests.yaml
test_suite: "smoke"
defaults:
  runs_per_test: 1
  timeout_seconds: 60

tests:
  - id: "smoke-001"
    tags: ["smoke", "p0"]
  # 2-3 critical tests only
```

**Full Validation** (regression tests on PR):
```yaml
# regression_suite.yaml
test_suite: "regression"
defaults:
  runs_per_test: 3
  timeout_seconds: 300

tests:
  # All important tests
```

**Comprehensive** (nightly or pre-release):
```yaml
# comprehensive_suite.yaml
test_suite: "comprehensive"
defaults:
  runs_per_test: 5
  timeout_seconds: 600

tests:
  # All tests including performance, edge cases
```

---

## Interpreting Results

### Statistical Analysis

With multiple runs, analyze distribution:

```python
# Look at mean, std, min, max
Test: search-001
  Mean score: 0.85
  Std dev: 0.08     # <0.1 is good (stable)
  Min: 0.72
  Max: 0.95
  95% CI: [0.80, 0.90]
```

**Stability Assessment**:
- **Coefficient of Variation < 0.05**: Stable, reliable agent
- **CV 0.05-0.15**: Moderate variance, acceptable
- **CV 0.15-0.30**: High variance, investigate causes
- **CV > 0.30**: Critical instability, agent needs improvement

### Failure Analysis

When tests fail, investigate systematically:

1. **Check structural issues first**:
   - Missing files
   - Format errors
   - Syntax issues

2. **Check behavior**:
   - Tool usage patterns
   - Error rates
   - Step efficiency

3. **Check semantic quality last**:
   - LLM evaluation results
   - Content quality issues
   - Task completion

### Compare Agents

Use consistent test suites to compare:

```yaml
agents:
  - name: "agent-v1"
    type: "http"
    config:
      endpoint: "${V1_ENDPOINT}"

  - name: "agent-v2"
    type: "http"
    config:
      endpoint: "${V2_ENDPOINT}"

tests:
  - id: "test-001"
    # Same test runs on both agents
```

Compare:
- **Mean scores**: Overall quality
- **Variance**: Stability/reliability
- **Cost**: Token usage, pricing
- **Efficiency**: Steps, latency

---

## Common Pitfalls

### Over-Constraining Tests

**Problem**: Tests fail because constraints are too tight.

**Solution**: Start permissive, tighten based on actual behavior:
```yaml
# Start with
constraints:
  max_steps: 100  # Permissive

# Observe: agent uses ~30 steps
# Tighten to:
constraints:
  max_steps: 50   # 1.5x observed maximum
```

### Under-Specifying Tasks

**Problem**: Agent produces varied, unexpected outputs.

**Solution**: Be more specific about requirements:
```yaml
# Vague
task:
  description: "Research competitors"

# Specific
task:
  description: |
    Research competitors for [company] in [market].
    Create report.md with:
    - Executive summary (2-3 paragraphs)
    - Competitor list (minimum 5)
    - Comparison table with columns: Name, Market Share, Key Features, Pricing
```

### Unrealistic LLM Thresholds

**Problem**: Good results fail due to high thresholds.

**Solution**: Use realistic thresholds:
```yaml
# Too strict
threshold: 0.95

# Better
threshold: 0.75  # For most criteria
threshold: 0.85  # For critical accuracy checks
```

### Ignoring Variance

**Problem**: Single test run gives false confidence.

**Solution**: Always run multiple times for LLM-based tests:
```yaml
defaults:
  runs_per_test: 5  # Or at least 3
```

### Not Testing Edge Cases

**Problem**: Agent works on happy path, fails on edge cases.

**Solution**: Explicitly test error handling:
```yaml
tests:
  - id: "edge-001-unknown-input"
    name: "Handle unknown company"

  - id: "edge-002-ambiguous-input"
    name: "Handle ambiguous query"

  - id: "edge-003-no-results"
    name: "Handle no search results"
```

---

## Maintenance

### Version Your Test Suites

Track test suite evolution:

```yaml
test_suite: "competitor_analysis"
version: "2.1.0"  # MAJOR.MINOR.PATCH

# In git
# v1.0.0 - Initial test suite
# v1.1.0 - Added edge case tests
# v2.0.0 - Major refactor, breaking changes
# v2.1.0 - Added performance tests
```

### Review and Update Regularly

- Review test results monthly
- Update thresholds based on agent improvements
- Remove obsolete tests
- Add tests for new features
- Refine assertions based on false positives/negatives

### Document Test Intent

```yaml
tests:
  - id: "test-001"
    name: "Basic competitor search"
    description: |
      Purpose: Validate core search and analysis capability.

      This test ensures the agent can:
      1. Execute web searches effectively
      2. Identify legitimate competitors
      3. Extract key information
      4. Format results properly

      Success criteria: Agent finds 5+ real competitors with
      accurate information, formatted in markdown.

      Known issues:
      - Sometimes includes tools/libraries instead of companies
      - Market share data may be approximate

      Last reviewed: 2024-01-15
```

---

## Performance Optimization

### Optimize Test Execution

1. **Parallelize independent tests**
2. **Cache deterministic results**
3. **Run expensive tests less frequently**
4. **Use smaller runs for development**

### Optimize LLM Costs

1. **Use cheaper models for judge when possible**:
```yaml
assertions:
  - type: "llm_eval"
    config:
      model: "gpt-3.5-turbo"  # Instead of gpt-4
      criteria: "completeness"
```

2. **Limit artifact content sent to judge**:
```yaml
assertions:
  - type: "llm_eval"
    config:
      artifact: "report.md"
      max_content_length: 5000  # Truncate long content
```

3. **Use deterministic checks where possible** (artifact_exists, contains, etc.)

---

## Next Steps

1. **Start Simple**: Begin with smoke tests
2. **Iterate**: Add tests based on failures and findings
3. **Measure**: Track agent performance over time
4. **Improve**: Use test results to guide agent development

## See Also

- [Quick Start Guide](quickstart.md) - Get started with ATP
- [Configuration Reference](../reference/configuration.md) - Complete config options
- [API Reference](../reference/api-reference.md) - Python API documentation
- [Test Format Reference](../reference/test-format.md) - YAML structure
- [Evaluators Documentation](../05-evaluators.md) - Assertion types

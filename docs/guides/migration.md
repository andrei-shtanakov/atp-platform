# Migration Guide

Guide for migrating from custom testing solutions to ATP.

## Overview

This guide helps teams transition from custom agent testing approaches to ATP. Whether you're using ad-hoc scripts, framework-specific tests, or home-grown testing infrastructure, this guide will help you migrate to ATP's unified platform.

---

## Why Migrate to ATP?

### Problems with Custom Solutions

**Ad-hoc Testing**:
- Scripts scattered across repositories
- No standardized evaluation criteria
- Hard to compare different agents
- Difficult to reproduce results
- No statistical analysis

**Framework-Specific Tests**:
- Tied to specific framework (LangChain, LangGraph, CrewAI)
- Can't compare agents across frameworks
- Tests break when changing frameworks
- Limited evaluation capabilities

**Home-grown Infrastructure**:
- Maintenance burden
- Limited features
- Poor documentation
- No community support
- Reinventing the wheel

### ATP Advantages

✅ **Framework Agnostic** - Test any agent implementation
✅ **Declarative** - YAML-based test definitions
✅ **Comprehensive Evaluation** - Multi-level assertion system
✅ **Statistical Analysis** - Account for LLM stochasticity
✅ **Reproducible** - Consistent test execution
✅ **Extensible** - Custom evaluators and adapters
✅ **Well-Documented** - Complete documentation and examples

---

## Migration Strategies

### Strategy 1: Incremental Migration

Gradually migrate tests while maintaining existing infrastructure.

**Phase 1: Pilot** (1-2 weeks)
- Select 2-3 critical test cases
- Convert to ATP format
- Run in parallel with existing tests
- Validate results match

**Phase 2: Core Tests** (2-4 weeks)
- Migrate core test suite
- Integrate into CI/CD
- Train team on ATP

**Phase 3: Full Migration** (4-8 weeks)
- Migrate all tests
- Deprecate old infrastructure
- Optimize test suite

**Phase 4: Enhancement** (ongoing)
- Add new evaluators
- Optimize performance
- Expand coverage

### Strategy 2: Clean Slate

Start fresh with ATP, designing tests from scratch.

**When to Use**:
- Existing tests are poorly maintained
- Major agent refactoring planned
- Small existing test suite
- Starting new agent project

**Steps**:
1. Document current test scenarios
2. Design ATP test suite structure
3. Implement tests in ATP
4. Validate against existing behavior
5. Deprecate old tests

### Strategy 3: Hybrid Approach

Keep existing tests, add ATP for new capabilities.

**When to Use**:
- Large existing test suite
- Limited migration resources
- Need advanced ATP features for new tests

**Approach**:
- Maintain existing tests as-is
- Use ATP for new test development
- Gradually convert high-value tests

---

## Migration from Common Patterns

### From Python Test Scripts

**Before** (Python script):
```python
# test_agent.py
import agent_module

def test_competitor_search():
    agent = agent_module.Agent()

    result = agent.run(
        "Find competitors for Slack in enterprise communication"
    )

    # Manual checks
    assert result.status == "success"
    assert "Microsoft Teams" in result.output
    assert len(result.output) > 1000

    print("Test passed!")

if __name__ == "__main__":
    test_competitor_search()
```

**After** (ATP YAML):
```yaml
test_suite: "competitor_search_tests"
version: "1.0"

agents:
  - name: "my-agent"
    type: "http"
    config:
      endpoint: "http://localhost:8000"

tests:
  - id: "test-001"
    name: "Find competitors for Slack"

    task:
      description: "Find competitors for Slack in enterprise communication"

    assertions:
      - type: "artifact_exists"
        config:
          path: "output.txt"

      - type: "contains"
        config:
          artifact: "output.txt"
          text: "Microsoft Teams"

      - type: "min_length"
        config:
          artifact: "output.txt"
          chars: 1000
```

**Benefits**:
- Declarative, maintainable
- Multiple runs for reliability
- Statistical analysis
- Better error reporting

---

### From LangChain Tests

**Before** (LangChain-specific):
```python
from langchain.agents import create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def test_research_agent():
    llm = ChatOpenAI(model="gpt-4")
    tools = [web_search_tool, file_writer_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant"),
        ("human", "{input}"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)

    result = agent.invoke({
        "input": "Research Python web frameworks"
    })

    # Check result
    assert "Flask" in result["output"]
    assert "Django" in result["output"]
```

**After** (ATP):
```yaml
test_suite: "research_tests"
version: "1.0"

agents:
  - name: "langchain-research-agent"
    type: "cli"
    config:
      command: "python"
      args: ["run_agent.py"]
      environment:
        OPENAI_API_KEY: "${OPENAI_API_KEY}"

tests:
  - id: "test-001"
    name: "Research Python web frameworks"

    task:
      description: "Research popular Python web frameworks and create a report"

    assertions:
      - type: "contains"
        config:
          artifact: "report.md"
          pattern: "Flask|Django"
          regex: true
          min_matches: 2

      - type: "llm_eval"
        config:
          artifact: "report.md"
          criteria: "completeness"
          threshold: 0.8
```

**Benefits**:
- Framework independent
- Can switch from LangChain to CrewAI without rewriting tests
- Advanced evaluation with LLM-as-judge
- Consistent results across runs

---

### From Jupyter Notebooks

**Before** (Notebook testing):
```python
# In Jupyter notebook
from my_agent import Agent

agent = Agent()

# Test 1
result1 = agent.run("Task 1...")
print(result1)
# Manually inspect output

# Test 2
result2 = agent.run("Task 2...")
print(result2)
# Manually inspect output

# Copy-paste results into spreadsheet for comparison
```

**After** (ATP):
```yaml
test_suite: "agent_evaluation"
version: "1.0"

defaults:
  runs_per_test: 5  # Statistical reliability

agents:
  - name: "my-agent"
    type: "cli"
    config:
      command: "python"
      args: ["run_agent.py"]

tests:
  - id: "test-001"
    name: "Task 1"
    task:
      description: "Task 1..."
    assertions:
      - type: "llm_eval"
        config:
          criteria: "quality"

  - id: "test-002"
    name: "Task 2"
    task:
      description: "Task 2..."
    assertions:
      - type: "llm_eval"
        config:
          criteria: "quality"
```

**Benefits**:
- Automated, repeatable
- Statistical analysis built-in
- Structured results (JSON, HTML reports)
- Version controlled

---

### From Manual Testing

**Before** (Manual QA):
```
Testing Checklist:
□ Run agent with prompt: "Find competitors for Slack"
□ Check if output contains Microsoft Teams
□ Check if output is formatted properly
□ Check if response time < 2 minutes
□ Manually score quality 1-5
□ Record results in spreadsheet
```

**After** (ATP):
```yaml
test_suite: "qa_tests"
version: "1.0"

tests:
  - id: "qa-001"
    name: "Competitor search quality check"

    task:
      description: "Find competitors for Slack"

    constraints:
      timeout_seconds: 120  # 2 minutes

    assertions:
      - type: "contains"
        config:
          artifact: "output.txt"
          text: "Microsoft Teams"

      - type: "artifact_format"
        config:
          artifact: "output.txt"
          format: "markdown"

      - type: "llm_eval"
        config:
          artifact: "output.txt"
          criteria: "quality"
          threshold: 0.7  # 3.5/5 equivalent
```

**Benefits**:
- Automated execution
- Consistent evaluation
- Faster feedback
- Scalable (test many scenarios)

---

## Step-by-Step Migration

### Step 1: Inventory Current Tests

Document existing test cases:

```markdown
# Current Test Inventory

## Test Category: Competitor Analysis
1. Basic competitor search (Slack)
   - Input: Company name, market
   - Expected: List of 5+ competitors
   - Validation: Manual review

2. Unknown company handling
   - Input: Non-existent company
   - Expected: Graceful error handling
   - Validation: Check for hallucination

3. Market analysis
   - Input: Market description
   - Expected: Top 10 players with analysis
   - Validation: Fact-checking, format review

## Test Category: Report Generation
...
```

### Step 2: Map to ATP Structure

Create mapping document:

```yaml
# migration_mapping.yaml

current_tests:
  - current_name: "test_competitor_search_slack.py"
    atp_equivalent:
      suite: "competitor_tests.yaml"
      test_id: "comp-001-slack"
      notes: "Convert assertion logic to YAML"

  - current_name: "test_unknown_company.py"
    atp_equivalent:
      suite: "competitor_tests.yaml"
      test_id: "comp-002-unknown"
      notes: "Use llm_eval for hallucination check"
```

### Step 3: Create ATP Adapter

Wrap your agent for ATP:

**Option A: HTTP Adapter** (recommended)

Create a simple HTTP server:

```python
# agent_server.py
from flask import Flask, request, jsonify
from my_agent import Agent

app = Flask(__name__)
agent = Agent()

@app.route("/execute", methods=["POST"])
def execute():
    data = request.json
    task = data["task"]["description"]

    # Run agent
    result = agent.run(task)

    # Return ATP response format
    return jsonify({
        "status": "success",
        "artifacts": ["output.txt"],
        "metrics": {
            "steps_taken": result.steps,
            "tokens_used": result.tokens,
            "cost_usd": result.cost
        }
    })

if __name__ == "__main__":
    app.run(port=8000)
```

**Option B: CLI Adapter**

Create a CLI wrapper:

```python
# run_agent.py
import sys
import json
from my_agent import Agent

def main():
    # Read ATP request from stdin or file
    request = json.loads(sys.stdin.read())

    agent = Agent()
    result = agent.run(request["task"]["description"])

    # Write ATP response
    response = {
        "status": "success",
        "artifacts": ["output.txt"],
        "metrics": {
            "steps_taken": result.steps,
            "tokens_used": result.tokens
        }
    }

    print(json.dumps(response))

if __name__ == "__main__":
    main()
```

### Step 4: Convert Test Cases

Convert each test case to ATP YAML:

**Template**:
```yaml
test_suite: "migrated_tests"
version: "1.0"
description: "Migrated from [previous system]"

agents:
  - name: "my-agent"
    type: "http"  # or "cli"
    config:
      endpoint: "http://localhost:8000"

tests:
  - id: "[original-test-id]"
    name: "[original-test-name]"
    description: |
      Migrated from: [original file/location]
      Original validation: [describe]

    task:
      description: "[original prompt/input]"

    assertions:
      # Convert validation logic to assertions
      - type: "artifact_exists"
        config:
          path: "output.txt"
```

### Step 5: Validate Migration

Run tests in parallel:

```bash
# Run old tests
python old_tests/test_suite.py

# Run ATP tests
atp run migrated_tests.yaml

# Compare results
python compare_results.py
```

Validation checklist:
- [ ] All test cases migrated
- [ ] Pass/fail results match
- [ ] Performance metrics comparable
- [ ] Edge cases covered
- [ ] Error handling equivalent

### Step 6: Integrate into CI/CD

**GitHub Actions Example**:
```yaml
# .github/workflows/test-agents.yml
name: Agent Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.12"

      - name: Install ATP
        run: |
          pip install uv
          cd atp-platform-ru
          uv sync

      - name: Run Agent Server
        run: python agent_server.py &

      - name: Run ATP Tests
        run: |
          atp run tests/smoke_tests.yaml
          atp run tests/regression_suite.yaml

      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: results/
```

---

## Common Migration Challenges

### Challenge 1: Non-Standard Agent Output

**Problem**: Agent doesn't produce artifacts, only returns text.

**Solution**: Create adapter that writes output to file:

```python
@app.route("/execute", methods=["POST"])
def execute():
    data = request.json
    result = agent.run(data["task"]["description"])

    # Write output to artifact file
    with open("output.txt", "w") as f:
        f.write(result.output)

    return jsonify({
        "status": "success",
        "artifacts": ["output.txt"]
    })
```

### Challenge 2: Complex Validation Logic

**Problem**: Existing tests have complex Python validation logic.

**Solution**: Create custom evaluator:

```python
# custom_evaluators.py
from atp.evaluators.base import Evaluator, EvalResult

class CustomDomainEvaluator(Evaluator):
    name = "custom_domain"

    async def evaluate(self, task, response, trace, assertion):
        # Port your validation logic here
        artifact = self._get_artifact(response, assertion.config["artifact"])

        # Your custom validation
        score = your_validation_function(artifact)

        return EvalResult(
            evaluator=self.name,
            checks=[{
                "name": "custom_check",
                "passed": score >= assertion.config.get("threshold", 0.7),
                "score": score
            }]
        )
```

Register and use:
```yaml
assertions:
  - type: "custom_domain"
    config:
      artifact: "output.txt"
      threshold: 0.8
```

### Challenge 3: Framework-Specific Features

**Problem**: Tests rely on framework internals (LangChain memory, CrewAI tasks).

**Solution**: Test behavior, not implementation:

**Before** (framework-specific):
```python
# Test LangChain memory
assert agent.memory.buffer_as_messages[0].content == "Hello"
```

**After** (behavior-based):
```yaml
# Test that agent uses context from previous interaction
task:
  description: |
    First, I'll tell you a fact: "My name is Alice."
    Please remember this and use it in future responses.

    Now answer: "What is my name?"

assertions:
  - type: "contains"
    config:
      artifact: "output.txt"
      text: "Alice"
```

### Challenge 4: Test Data Management

**Problem**: Existing tests use complex test fixtures and data.

**Solution**: Use `input_data` field:

```yaml
tests:
  - id: "test-with-data"
    name: "Test with complex input"

    task:
      description: "Analyze the provided dataset"
      input_data:
        dataset_url: "https://example.com/data.csv"
        fields: ["name", "value", "category"]
        filters:
          category: "technology"
          min_value: 100
```

Or reference external files:
```yaml
task:
  description: "Analyze the data in input_data.json"
  input_data:
    data_file: "fixtures/test_data.json"
```

---

## Migration Timeline Example

**Small Project** (1-2 person-weeks):
- Week 1: Inventory tests, create adapter, migrate core tests
- Week 2: Migrate remaining tests, integrate into CI/CD

**Medium Project** (1-2 person-months):
- Month 1:
  - Weeks 1-2: Inventory, planning, pilot migration
  - Weeks 3-4: Migrate core test suite, validate
- Month 2:
  - Weeks 5-6: Migrate all tests, custom evaluators
  - Weeks 7-8: CI/CD integration, documentation

**Large Project** (3-6 months):
- Month 1-2: Planning, pilot, core tests
- Month 3-4: Full migration, custom evaluators
- Month 5-6: Optimization, training, deprecation of old system

---

## Success Criteria

Track migration success:

- [ ] All test cases migrated
- [ ] Pass/fail rates match (±5%)
- [ ] CI/CD integration complete
- [ ] Team trained on ATP
- [ ] Documentation updated
- [ ] Old infrastructure deprecated
- [ ] Performance equal or better
- [ ] Cost equal or lower

---

## Post-Migration Optimization

After successful migration:

1. **Optimize Test Suites**
   - Remove redundant tests
   - Improve assertion quality
   - Add statistical runs

2. **Enhance Evaluation**
   - Add LLM-as-judge for quality checks
   - Create custom evaluators for domain logic
   - Improve threshold tuning

3. **Improve Coverage**
   - Add edge case tests
   - Add performance tests
   - Add security tests

4. **Automate Further**
   - Auto-generate test reports
   - Set up alerting for failures
   - Create dashboards

---

## Getting Help

- **Documentation**: [docs/](.)
- **Examples**: [examples/](../../examples/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/atp-platform-ru/issues)

## See Also

- [Quick Start Guide](quickstart.md) - Get started with ATP
- [Best Practices](best-practices.md) - Testing best practices
- [API Reference](../reference/api-reference.md) - Python API
- [Configuration Reference](../reference/configuration.md) - YAML configuration
- [Integration Guide](../06-integration.md) - Detailed integration patterns

# Frequently Asked Questions (FAQ)

Common questions and answers about ATP.

## General Questions

### What is ATP?

ATP (Agent Test Platform) is a framework-agnostic platform for testing and evaluating AI agents. It provides a unified protocol and infrastructure for testing agents regardless of their implementation framework (LangGraph, CrewAI, AutoGen, custom, etc.).

**Key Features**:
- Declarative YAML-based test definitions
- Multi-level evaluation (structural, behavioral, semantic)
- Statistical reliability through multiple runs
- Framework-agnostic design
- Comprehensive reporting

---

### Why use ATP instead of framework-specific tests?

**ATP Advantages**:
1. **Framework Independence**: Test any agent, switch frameworks without rewriting tests
2. **Unified Evaluation**: Consistent metrics across different agents
3. **Advanced Features**: LLM-as-judge, statistical analysis, behavior tracking
4. **Maintainability**: Declarative YAML easier to maintain than code
5. **Reproducibility**: Standardized execution and reporting

---

### What frameworks does ATP support?

ATP is framework-agnostic. Current and planned adapters include:

**Current (MVP)**:
- Test loader only (no runtime execution yet)

**Planned**:
- HTTP REST API agents
- Docker containerized agents
- CLI-based agents
- LangGraph agents
- CrewAI agents
- AutoGen agents
- Custom adapters

---

### Is ATP production-ready?

**Current Status**: MVP Phase (v0.1.0)

**What Works**:
- ✅ Test suite loading and validation
- ✅ YAML parsing with variable substitution
- ✅ Comprehensive data models

**In Development**:
- 🚧 Runner (test execution)
- 🚧 Adapters (agent integration)
- 🚧 Evaluators (result assessment)
- 🚧 Reporters (output formatting)

See [Roadmap](../07-roadmap.md) for development timeline.

---

## Installation and Setup

### How do I install ATP?

```bash
# Clone repository
git clone https://github.com/yourusername/atp-platform-ru.git
cd atp-platform-ru

# Install with uv (recommended)
uv sync

# Verify installation
python -c "import atp; print(atp.__version__)"
```

See [Installation Guide](../guides/installation.md) for details.

---

### What are the system requirements?

**Minimum**:
- Python 3.12+
- 4GB RAM
- Linux, macOS, or Windows

**Recommended**:
- Python 3.12+
- 8GB+ RAM
- Docker (for containerized agents)
- Git

---

### Do I need an API key?

It depends on your setup:

**Not Required**:
- Loading and validating test suites
- Testing local agents (CLI, Docker)

**Required**:
- LLM-as-judge evaluation (OpenAI, Anthropic, etc.)
- Testing cloud-hosted agents (if they require authentication)

---

## Test Suite Creation

### How do I create my first test suite?

Minimal example:

```yaml
test_suite: "my_first_suite"
version: "1.0"

agents:
  - name: "my-agent"
    type: "http"
    config:
      endpoint: "http://localhost:8000"

tests:
  - id: "test-001"
    name: "Basic test"
    task:
      description: "Create a file named output.txt"
    assertions:
      - type: "artifact_exists"
        config:
          path: "output.txt"
```

See [Quick Start Guide](../guides/quickstart.md) for walkthrough.

---

### What makes a good test case?

A good test case is:

1. **Specific**: Clear, unambiguous task description
2. **Reproducible**: Same input → consistent results
3. **Measurable**: Concrete success criteria
4. **Isolated**: Tests one thing at a time
5. **Realistic**: Represents actual use cases

**Good Example**:
```yaml
- id: "test-001"
  name: "Find competitors for Slack"
  task:
    description: |
      Find 5 competitors for Slack in enterprise communication.
      For each, provide: name, description, key features.
      Output: markdown report with sections Summary and Analysis.
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
```

---

### How many assertions should I use per test?

**Recommended Layering**:
1. **1-2 structural checks** (file exists, format valid)
2. **2-3 content checks** (contains expected text, correct length)
3. **1-2 behavior checks** (tool usage, error handling)
4. **0-1 semantic checks** (LLM evaluation)

**Total**: 4-8 assertions per test

Too few = shallow validation
Too many = brittle tests, hard to maintain

---

### Should I use LLM-as-judge for every test?

**No**. Use LLM evaluation strategically:

**Use When**:
- Semantic quality matters (completeness, clarity, accuracy)
- No deterministic check available
- Subjective criteria (readability, tone)

**Don't Use When**:
- Deterministic check possible (file exists, contains text)
- Testing structural properties
- Cost is a concern (LLM calls are expensive)
- Fast feedback needed (LLM eval is slow)

**Rule of Thumb**: Use LLM eval for 20-30% of assertions, not 100%.

---

## Configuration

### How do I handle secrets and API keys?

**Never hardcode secrets**:

```yaml
# Bad
config:
  api_key: "sk-abc123"  # Don't do this!

# Good
config:
  api_key: "${API_KEY}"
```

**Set environment variables**:
```bash
export API_KEY="your-secret-key"
```

**Or provide when loading**:
```python
from atp.loader import TestLoader

loader = TestLoader(env={
    "API_KEY": "your-secret-key"
})
suite = loader.load_file("suite.yaml")
```

---

### What timeout should I use?

Depends on task complexity:

**Simple Tasks** (file operations, formatting):
```yaml
constraints:
  timeout_seconds: 60  # 1 minute
```

**Medium Tasks** (research, analysis):
```yaml
constraints:
  timeout_seconds: 300  # 5 minutes
```

**Complex Tasks** (multi-step, code generation):
```yaml
constraints:
  timeout_seconds: 600  # 10 minutes
```

**Start generous, tighten based on observed behavior**.

---

### How many times should I run each test?

Depends on agent determinism:

**Deterministic Agents** (rule-based, scripted):
```yaml
defaults:
  runs_per_test: 1  # Single run sufficient
```

**LLM-Based Agents** (GPT, Claude, etc.):
```yaml
defaults:
  runs_per_test: 5  # Multiple runs for statistics
```

**Development**:
```yaml
defaults:
  runs_per_test: 1  # Fast iteration
```

**Production/CI**:
```yaml
defaults:
  runs_per_test: 3-5  # Reliability
```

---

## Execution and Results

### How do I run a test suite?

**Current MVP** (load and inspect only):
```python
from atp.loader import TestLoader

loader = TestLoader()
suite = loader.load_file("suite.yaml")

print(f"Suite: {suite.test_suite}")
print(f"Tests: {len(suite.tests)}")
```

**Planned** (full execution):
```bash
atp run suite.yaml
atp run suite.yaml --agent my-agent
atp run suite.yaml --tag smoke
```

---

### What does a test score mean?

ATP calculates a composite score (0-100) from weighted components:

```
Score = w_Q × Quality + w_C × Completeness + w_E × Efficiency + w_$ × Cost
```

**Default Weights**:
- Quality: 40%
- Completeness: 30%
- Efficiency: 20%
- Cost: 10%

**Interpreting Scores**:
- **90-100**: Excellent
- **75-89**: Good
- **60-74**: Acceptable
- **<60**: Needs improvement

---

### Why do my scores vary between runs?

**Expected Variance** for LLM-based agents:

1. **LLM Non-Determinism**: Same prompt → different responses
2. **Evaluation Variance**: LLM judge has its own variance
3. **Network/API Variance**: Latency, rate limits
4. **Random Sampling**: Temperature > 0

**Handling Variance**:
- Run tests multiple times (3-5 runs)
- Look at mean and standard deviation
- Use confidence intervals
- Assess stability (coefficient of variation)

**Coefficient of Variation < 0.1** = stable agent
**CV > 0.3** = investigate instability

---

### How do I compare two agents?

Use the same test suite for both:

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

**Compare**:
- Mean scores (overall quality)
- Variance (stability)
- Cost (tokens, pricing)
- Latency (execution time)

---

## Troubleshooting

### My test suite won't load. What's wrong?

**Common Issues**:

1. **YAML Syntax Error**:
```
ParseError: Invalid YAML syntax at line 15
```
**Solution**: Check YAML syntax, validate with yamllint

2. **Missing Required Field**:
```
ValidationError: tests.0.task.description: field required
```
**Solution**: Add missing field

3. **Invalid Value**:
```
ValidationError: defaults.runs_per_test: ensure this value is greater than 0
```
**Solution**: Fix invalid value

4. **Semantic Error**:
```
ValidationError: Duplicate test ID 'test-001' at tests[2]
```
**Solution**: Use unique test IDs

See [Troubleshooting Guide](troubleshooting.md) for more.

---

### My LLM evaluation always fails/passes. Why?

**Always Fails** → Threshold too high:
```yaml
# Too strict
- type: "llm_eval"
  config:
    criteria: "completeness"
    threshold: 0.95  # Lower to 0.75-0.85
```

**Always Passes** → Threshold too low:
```yaml
# Too lenient
- type: "llm_eval"
  config:
    criteria: "accuracy"
    threshold: 0.3  # Raise to 0.75-0.85
```

**Solution**: Tune thresholds based on actual scores.

---

### How do I debug a failing test?

1. **Check structural issues first**:
   - Does the artifact exist?
   - Is it the right format?

2. **Check content**:
   - Does it contain expected text?
   - Is it the right length?

3. **Check behavior**:
   - Did agent use required tools?
   - Were there errors?

4. **Check quality**:
   - What did LLM evaluator say?
   - Read the explanation

**Inspect Results**:
```python
# Planned API
result = runner.run_test(test)

for check in result.checks:
    if not check.passed:
        print(f"Failed: {check.name}")
        print(f"Reason: {check.explanation}")
```

---

### Tests are too slow. How do I speed them up?

**Optimization Strategies**:

1. **Reduce runs in development**:
```yaml
defaults:
  runs_per_test: 1  # Instead of 5
```

2. **Use faster assertions**:
```yaml
# Fast (deterministic)
- type: "artifact_exists"
- type: "contains"

# Slow (requires LLM)
- type: "llm_eval"
```

3. **Parallelize tests** (planned feature)

4. **Use cheaper LLM models for judging**:
```yaml
- type: "llm_eval"
  config:
    model: "gpt-3.5-turbo"  # Instead of gpt-4
```

5. **Cache results** (planned feature)

---

## Advanced Topics

### Can I create custom evaluators?

**Yes** (planned feature). Implement the `Evaluator` interface:

```python
from atp.evaluators.base import Evaluator, EvalResult

class MyCustomEvaluator(Evaluator):
    name = "my_custom"

    async def evaluate(self, task, response, trace, assertion):
        # Your evaluation logic
        score = your_logic(response)

        return EvalResult(
            evaluator=self.name,
            checks=[{
                "name": "custom_check",
                "passed": score >= threshold,
                "score": score
            }]
        )
```

Register:
```python
from atp.core.registry import evaluator_registry
evaluator_registry.register(MyCustomEvaluator())
```

---

### How do I test agents that require interactive input?

**Options**:

1. **Mock Interactive Input** in adapter:
```python
# In adapter
def execute(self, request):
    # Provide mock inputs
    with mock.patch('builtins.input', return_value='yes'):
        result = agent.run(request.task.description)
```

2. **Use Non-Interactive Mode**:
Configure agent to run non-interactively

3. **Pre-populate Inputs**:
```yaml
task:
  input_data:
    user_responses: ["yes", "option 1", "confirm"]
```

---

### Can I test agents that modify external systems?

**Approaches**:

1. **Sandbox Environment**:
Use Docker with isolated environment

2. **Mock External Systems**:
```yaml
agents:
  - name: "agent"
    type: "docker"
    config:
      image: "agent:latest"
      environment:
        API_ENDPOINT: "http://mock-api:8080"  # Mock service
```

3. **Use Test Accounts**:
Configure agent with test credentials

4. **Verify Side Effects**:
```yaml
assertions:
  - type: "behavior"
    config:
      verify_api_calls:
        - endpoint: "/api/users"
          method: "POST"
          expected_status: 201
```

---

### How do I test multi-agent systems?

**Options**:

1. **Test as Black Box**:
Treat multi-agent system as single agent

2. **Test Individual Agents**:
Test each agent separately

3. **Test Interactions** (planned feature):
```yaml
tests:
  - id: "multi-agent-test"
    type: "multi_agent"
    agents: ["agent1", "agent2", "agent3"]
    task:
      description: "Collaborative task"
    assertions:
      - type: "collaboration"
        config:
          verify_communication: true
```

---

## Integration

### Can I use ATP in CI/CD?

**Yes**. Example GitHub Actions workflow:

```yaml
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
        run: uv sync

      - name: Run Tests
        run: atp run tests/suite.yaml
        env:
          API_KEY: ${{ secrets.API_KEY }}
```

See [CI/CD Integration Guide](../guides/best-practices.md#cicd-integration).

---

### Does ATP work with pytest?

**Not directly**. ATP is a standalone testing platform with its own execution model.

**However**, you can:
1. Use ATP for agent testing
2. Use pytest for unit testing agent components
3. Run both in CI/CD

---

### Can I export results to other tools?

**Planned Features**:
- JSON export
- JUnit XML (for CI/CD)
- HTML reports
- CSV/Excel export

**Current** (MVP):
- Programmatic access via Python API

---

## Community and Support

### Where can I get help?

- **Documentation**: [docs/](.)
- **Examples**: [examples/](../../examples/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/atp-platform-ru/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/atp-platform-ru/discussions)

---

### How can I contribute?

**Ways to Contribute**:
1. Report bugs and request features
2. Improve documentation
3. Submit code contributions
4. Share example test suites
5. Create adapters for new frameworks

See [Contributing Guide](../../CONTRIBUTING.md).

---

### Is ATP open source?

**Yes**. ATP is released under the MIT License.

---

### What's on the roadmap?

See [Roadmap](../07-roadmap.md) for:
- Upcoming features
- Development timeline
- Long-term vision

**Next Milestones**:
- Runner implementation
- HTTP/Docker adapters
- Basic evaluators
- Console reporter

---

## Comparison with Other Tools

### ATP vs pytest

**pytest**:
- General-purpose Python testing
- Unit/integration tests
- Code-focused

**ATP**:
- Agent-specific testing
- Behavior and output evaluation
- Declarative YAML configuration
- Statistical analysis for LLM agents

**Use Both**: pytest for components, ATP for end-to-end agent testing.

---

### ATP vs LangSmith

**LangSmith**:
- LangChain-specific
- Observability and debugging
- Cloud-hosted

**ATP**:
- Framework-agnostic
- Declarative testing
- Self-hosted
- Comprehensive evaluation system

---

### ATP vs Custom Test Scripts

**Custom Scripts**:
- Full control
- Framework-specific
- High maintenance
- Limited features

**ATP**:
- Standardized
- Framework-agnostic
- Lower maintenance
- Rich evaluation features

---

## Still Have Questions?

- Check [Troubleshooting Guide](troubleshooting.md)
- See [Best Practices](../guides/best-practices.md)
- Browse [Examples](../../examples/)
- Open an [Issue](https://github.com/yourusername/atp-platform-ru/issues)

---

## See Also

- [Quick Start Guide](../guides/quickstart.md) - Get started quickly
- [Usage Guide](../guides/usage.md) - Common workflows
- [API Reference](api-reference.md) - Python API documentation
- [Configuration Reference](configuration.md) - Complete config options
- [Best Practices](../guides/best-practices.md) - Testing guidelines

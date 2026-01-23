# ATP Test Suite Examples

This directory contains example test suites demonstrating various testing scenarios and best practices for the ATP Platform.

## Overview

Each example suite is designed for a specific use case and demonstrates different features of ATP test suites.

## Available Suites

### 1. Smoke Tests (`01_smoke_tests.yaml`)

**Purpose**: Quick validation of basic agent functionality
**Duration**: ~1-2 minutes
**Tests**: 5
**Use Case**: Run before deployments, after changes, in CI/CD

**Features demonstrated**:
- Simple response validation
- File creation tests
- Basic calculations
- Multi-step workflows
- Error handling

**Key characteristics**:
- Fast execution (`runs_per_test: 1`)
- Strict timeouts (30-60s)
- Minimal step limits (1-5 steps)
- Focus on completeness over quality

**When to use**:
- Pre-deployment checks
- CI/CD pipeline gates
- Quick regression checks
- Development validation

### 2. Data Processing (`02_data_processing.yaml`)

**Purpose**: Validate data processing and transformation capabilities
**Duration**: ~5-10 minutes
**Tests**: 6
**Use Case**: Regression testing for data processing agents

**Features demonstrated**:
- CSV parsing and analysis
- JSON transformation
- Multi-file data merges
- Data cleaning and validation
- Statistical analysis
- Large dataset handling

**Key characteristics**:
- Multiple runs per test (`runs_per_test: 3`)
- Moderate timeouts (120-300s)
- Tool restrictions (file_read, file_write, python_repl)
- Balanced scoring weights
- Budget constraints ($0.25)

**When to use**:
- Data pipeline validation
- ETL testing
- Data quality checks
- Performance regression testing

### 3. Web Research (`03_web_research.yaml`)

**Purpose**: Test web research and information synthesis
**Duration**: ~10-15 minutes
**Tests**: 6
**Use Case**: Integration testing for agents with web access

**Features demonstrated**:
- Web search capabilities
- Information extraction
- Content synthesis
- Comparative analysis
- Multi-topic research
- Fact-checking
- Current news compilation

**Key characteristics**:
- Longer execution times (300-600s)
- Higher step limits (20-30 steps)
- Web tool requirements (web_search, web_scrape)
- Quality-focused scoring
- Higher budget ($0.50)

**When to use**:
- Research agent validation
- Information retrieval testing
- Content generation testing
- Integration testing with external APIs

### 4. Cost Optimization (`04_cost_optimization.yaml`)

**Purpose**: Test efficiency and cost-effectiveness
**Duration**: ~5-8 minutes
**Tests**: 6
**Use Case**: Cost analysis and optimization validation

**Features demonstrated**:
- Minimal token usage
- Efficient tool use
- Batch processing
- Information reuse
- Quality vs cost tradeoffs

**Key characteristics**:
- Multiple runs for statistical reliability (`runs_per_test: 5`)
- Strict budget constraints ($0.01-$0.10 per test)
- Token limits (500-8000)
- Equal scoring weights (25% each dimension)
- Focus on efficiency metrics

**When to use**:
- Cost analysis
- Efficiency optimization
- Budget planning
- Performance benchmarking

## Running Examples

### Load a Test Suite

```python
from atp.loader import TestLoader

# Load smoke tests
loader = TestLoader()
suite = loader.load_file("examples/test_suites/01_smoke_tests.yaml")

# Display info
print(f"Suite: {suite.test_suite}")
print(f"Tests: {len(suite.tests)}")

for test in suite.tests:
    print(f"  - {test.id}: {test.name}")
```

### Load with Environment Variables

```python
from atp.loader import TestLoader

# Configure environment
loader = TestLoader(env={
    "API_ENDPOINT": "https://api.production.com",
    "API_KEY": "your-api-key"
})

# Load suite
suite = loader.load_file("examples/test_suites/02_data_processing.yaml")
```

### Run Tests (Future Feature)

```bash
# When runner is implemented
atp run examples/test_suites/01_smoke_tests.yaml
atp run examples/test_suites/02_data_processing.yaml --agent=my-agent
atp run examples/test_suites/03_web_research.yaml --verbose
```

## Customizing Examples

### Modify for Your Use Case

1. **Copy an example**:
   ```bash
   cp examples/test_suites/01_smoke_tests.yaml my_tests.yaml
   ```

2. **Edit agent configuration**:
   ```yaml
   agents:
     - name: "my-agent"
       type: "http"
       config:
         endpoint: "http://localhost:8000"
   ```

3. **Adjust constraints**:
   ```yaml
   defaults:
     timeout_seconds: 120  # Increase timeout
     constraints:
       max_steps: 10       # Allow more steps
       budget_usd: 0.50    # Increase budget
   ```

4. **Modify scoring weights**:
   ```yaml
   defaults:
     scoring:
       quality_weight: 0.6      # Prioritize quality
       completeness_weight: 0.3
       efficiency_weight: 0.1
       cost_weight: 0.0         # Ignore cost
   ```

### Create New Tests

Use existing tests as templates:

```yaml
tests:
  - id: "custom-001"
    name: "My custom test"
    tags: ["custom", "new"]
    task:
      description: "Your task description"
      expected_artifacts: ["output.txt"]
    constraints:
      max_steps: 10
      timeout_seconds: 120
    assertions:
      - type: "artifact_exists"
        config:
          path: "output.txt"
```

## Best Practices

### Test Design

1. **Start simple**: Use smoke tests as foundation
2. **Build progressively**: Add complexity gradually
3. **Tag appropriately**: Use consistent tagging scheme
4. **Set realistic constraints**: Based on actual agent capabilities
5. **Balance scoring weights**: Match your priorities

### Performance

1. **Smoke tests**: Fast, minimal steps, basic validation
2. **Regression tests**: Thorough, multiple runs, detailed checks
3. **Integration tests**: Longer timeouts, complex workflows
4. **Cost tests**: Focus on efficiency, multiple runs

### Maintenance

1. **Version control**: Track test suite changes
2. **Document changes**: Update descriptions when modifying
3. **Review regularly**: Ensure tests stay relevant
4. **Update constraints**: As agents improve, adjust limits

## Validation

Validate test suites before running:

```python
from atp.loader import TestLoader
from atp.core.exceptions import ValidationError

loader = TestLoader()

try:
    suite = loader.load_file("my_tests.yaml")
    print("✓ Suite is valid")
except ValidationError as e:
    print(f"✗ Validation error: {e}")
```

## See Also

- [Test Format Reference](../../docs/reference/test-format.md) - Complete YAML format specification
- [Quick Start Guide](../../docs/guides/quickstart.md) - Create your first test suite
- [Usage Guide](../../docs/guides/usage.md) - Common workflows and patterns
- [Adapter Configuration](../../docs/reference/adapters.md) - Configure agent adapters

## Contributing

Have a useful test suite example? Contribute by:
1. Creating a new YAML file following naming convention
2. Adding description to this README
3. Submitting a pull request

## Support

- **Documentation**: [docs/](../../docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/atp-platform-ru/issues)

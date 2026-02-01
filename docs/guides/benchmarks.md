# ATP Benchmark Suite Registry

The Benchmark Suite Registry provides a collection of curated benchmark test suites for evaluating AI agents across common task categories. This guide covers how to use the benchmarks and how to contribute new ones.

## Overview

ATP includes four built-in benchmark categories:

| Category | Tests | Description |
|----------|-------|-------------|
| `coding` | 20 | Code generation, review, and debugging |
| `research` | 10 | Web research, summarization, fact-finding |
| `reasoning` | 15 | Logical puzzles, math problems, analysis |
| `data_processing` | 10 | Data transformation, cleaning, analysis |

Each benchmark includes:
- Detailed task descriptions
- Expected artifacts and assertions
- Baseline scores from established models (GPT-4, Claude 3 Opus)
- Difficulty ratings (easy, medium, hard)
- Skills being tested

## Quick Start

### Listing Available Benchmarks

```python
from atp.benchmarks import list_benchmarks, list_categories, get_registry

# List all categories
print(list_categories())
# Output: ['coding', 'research', 'reasoning', 'data_processing']

# List all registered benchmark suites
print(list_benchmarks())
# Output: ['coding', 'research', 'reasoning', 'data_processing']

# Get detailed info about all suites
registry = get_registry()
for info in registry.list_all_info():
    print(f"{info.name}: {info.test_count} tests")
```

### Running a Benchmark Suite

```python
from atp.benchmarks import get_registry, BenchmarkResult

registry = get_registry()

# Get a specific benchmark suite
coding_suite = registry.get("coding")

# Iterate through tests
for test in coding_suite.tests:
    print(f"Test: {test.name}")
    print(f"  Difficulty: {test.metadata.difficulty.value}")
    print(f"  Skills: {', '.join(test.metadata.skills_tested)}")
    print(f"  Task: {test.task_description[:100]}...")
```

### Working with Results

```python
from atp.benchmarks import get_registry

registry = get_registry()

# Create a result with normalized score
result = registry.create_result(
    test_id="coding-gen-001",
    raw_score=0.85,  # 0-1 scale
    passed=True,
    execution_time_seconds=45.0,
    steps_used=12,
    tokens_used=1500,
)

print(f"Normalized score: {result.normalized_score}")  # 85.0 (0-100 scale)

# Create aggregated suite result
results = [result]  # Add more results as tests complete
suite_result = registry.create_suite_result(
    suite_name="coding",
    agent_name="my-agent",
    results=results,
)

print(f"Pass rate: {suite_result.pass_rate}%")
print(f"Average score: {suite_result.average_normalized_score}")
```

## Benchmark Categories

### Coding Benchmarks

The coding benchmark suite evaluates three main capabilities:

**Code Generation (7 tests)**
- Simple function generation
- Data structure implementation
- API client creation
- File processing scripts
- Recursive algorithms
- Async programming
- Design pattern implementation

**Code Review (6 tests)**
- Security vulnerability detection
- Performance issue detection
- Code quality assessment
- Error handling review
- Memory leak detection
- API design review

**Bug Fixing (7 tests)**
- Off-by-one errors
- Race conditions
- Iterator modification bugs
- Null reference handling
- Async deadlocks
- Recursion depth issues
- Unicode handling

### Research Benchmarks

**Web Research (5 tests)**
- Technology comparison
- Market research
- Fact verification
- Best practices research
- Tutorial creation

**Summarization (5 tests)**
- Technical document summary
- Meeting notes extraction
- Research paper abstract
- Code documentation generation
- Executive summary

### Reasoning Benchmarks

**Logical Reasoning (8 tests)**
- Classic logic puzzles
- Knights and knaves
- Sequence patterns
- Syllogism evaluation
- Scheduling problems
- River crossing puzzles
- Proof construction
- Causal reasoning

**Mathematical Reasoning (7 tests)**
- Word problems (algebra)
- Probability calculations
- Optimization problems
- Combinatorics
- Number theory
- Geometry
- Algorithm complexity analysis

### Data Processing Benchmarks

**Data Transformation (4 tests)**
- JSON to CSV conversion
- Data pivoting
- XML processing
- Log file parsing

**Data Cleaning (3 tests)**
- Missing value handling
- Data deduplication
- Data validation and correction

**Data Analysis (3 tests)**
- Statistical summary
- Cohort analysis
- Time series analysis

## Score Normalization

All benchmark scores are normalized to a 0-100 scale for easy comparison:

```python
from atp.benchmarks import get_registry, NormalizationConfig

registry = get_registry()

# Default: linear normalization from 0-1 to 0-100
score = registry.normalize_score(0.75)  # Returns 75.0

# Custom normalization config
config = NormalizationConfig(
    min_raw_score=0.0,
    max_raw_score=1.0,
    target_min=0.0,
    target_max=100.0,
    curve_type="linear",  # or "logarithmic", "sigmoid"
)
registry.set_normalization_config(config)
```

### Normalization Curves

- **linear**: Direct proportional mapping (default)
- **logarithmic**: Boosts lower scores, useful when small improvements are significant
- **sigmoid**: S-curve that emphasizes differences around the midpoint

## Baseline Scores

Each test includes baseline scores from established models for comparison:

```python
from atp.benchmarks import get_registry

registry = get_registry()

# Get baseline scores for a suite
baselines = registry.get_baseline_scores("coding")

for test_id, scores in baselines.items():
    for baseline in scores:
        print(f"{test_id}: {baseline.model_name} = {baseline.score}")
```

### Interpreting Results

When comparing your agent's results to baselines:

```python
suite_result = registry.create_suite_result(
    suite_name="coding",
    agent_name="my-agent",
    results=my_results,
)

# baseline_comparison shows delta vs each baseline model
if suite_result.baseline_comparison:
    for model, delta in suite_result.baseline_comparison.items():
        if delta > 0:
            print(f"Outperformed {model} by {delta:.1f} points")
        else:
            print(f"Behind {model} by {-delta:.1f} points")
```

## Adding Custom Benchmarks

### Programmatic Registration

```python
from atp.benchmarks import (
    get_registry,
    BenchmarkSuite,
    BenchmarkTest,
    BenchmarkMetadata,
    BenchmarkCategory,
    BenchmarkDifficulty,
    BaselineScore,
)

# Create a custom test
test = BenchmarkTest(
    id="custom-001",
    name="Custom Test",
    description="My custom benchmark test",
    task_description="Complete this custom task...",
    expected_artifacts=["*.py"],
    assertions=[
        {"type": "artifact_exists", "config": {"pattern": "*.py"}}
    ],
    metadata=BenchmarkMetadata(
        category=BenchmarkCategory.CODING,
        difficulty=BenchmarkDifficulty.MEDIUM,
        estimated_time_seconds=120,
        skills_tested=["python", "problem_solving"],
        baseline_scores=[
            BaselineScore(
                model_name="gpt-4",
                score=85.0,
                date="2024-01-15",
            )
        ],
    ),
    tags=["custom", "python"],
)

# Create suite and register
suite = BenchmarkSuite(
    name="custom_suite",
    category=BenchmarkCategory.CODING,
    description="My custom benchmark suite",
    tests=[test],
)

registry = get_registry()
registry.register(suite)
```

### File-based Registration

Create a YAML file:

```yaml
# my_benchmarks.yaml
name: my_custom_suite
category: coding
version: "1.0.0"
description: My custom benchmark suite
default_timeout_seconds: 300

tests:
  - id: my-test-001
    name: My Custom Test
    description: Tests a specific capability
    task_description: |
      Complete this task by...
    expected_artifacts:
      - "*.py"
    assertions:
      - type: artifact_exists
        config:
          pattern: "*.py"
    metadata:
      category: coding
      difficulty: medium
      estimated_time_seconds: 120
      skills_tested:
        - python
        - algorithms
      baseline_scores:
        - model_name: gpt-4
          score: 80.0
          date: "2024-01-15"
    tags:
      - custom
```

Register from file:

```python
from atp.benchmarks import get_registry

registry = get_registry()
suite = registry.register_from_file("my_benchmarks.yaml")
```

## Best Practices

### Designing Good Benchmarks

1. **Clear task descriptions**: Be specific about what the agent should produce
2. **Measurable outcomes**: Include assertions that can be automatically verified
3. **Appropriate difficulty**: Match difficulty to expected agent capabilities
4. **Multiple skill tags**: Help users filter tests by required skills
5. **Baseline scores**: Include baselines for meaningful comparison

### Running Benchmarks

1. **Run multiple times**: Agent performance can vary; run each test 3+ times
2. **Track costs**: Monitor token usage and API costs
3. **Set appropriate timeouts**: Allow enough time for complex tasks
4. **Compare to baselines**: Use baseline scores to contextualize results

### Interpreting Results

1. **Consider difficulty distribution**: A lower score on hard tests may still be good
2. **Look at pass rate and average score**: Both metrics provide useful information
3. **Compare across categories**: Identify agent strengths and weaknesses
4. **Track over time**: Monitor improvements as you develop your agent

## API Reference

### Registry Functions

| Function | Description |
|----------|-------------|
| `get_registry()` | Get the global benchmark registry |
| `list_benchmarks()` | List all registered benchmark suite names |
| `list_categories()` | List all available categories |
| `get_benchmark(name)` | Get a benchmark suite by name |

### BenchmarkRegistry Methods

| Method | Description |
|--------|-------------|
| `register(suite)` | Register a benchmark suite |
| `register_from_file(path)` | Register from YAML/JSON file |
| `unregister(name)` | Remove a registered suite |
| `get(name)` | Get a suite by name |
| `get_by_category(category)` | Get all suites in a category |
| `list_suites()` | List registered suite names |
| `get_suite_info(name)` | Get summary info for a suite |
| `list_all_info()` | Get info for all suites |
| `get_test(suite, test_id)` | Get a specific test |
| `normalize_score(raw)` | Normalize a score to 0-100 |
| `create_result(...)` | Create a benchmark result |
| `create_suite_result(...)` | Create aggregated suite result |
| `get_baseline_scores(name)` | Get baseline scores for a suite |

### Data Models

| Model | Description |
|-------|-------------|
| `BenchmarkSuite` | Complete benchmark suite definition |
| `BenchmarkTest` | Single benchmark test |
| `BenchmarkMetadata` | Test metadata (difficulty, skills, baselines) |
| `BenchmarkResult` | Single test result |
| `BenchmarkSuiteResult` | Aggregated suite results |
| `BaselineScore` | Baseline score from a reference model |
| `NormalizationConfig` | Score normalization configuration |

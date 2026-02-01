# Performance Evaluator Guide

The Performance Evaluator tracks and assesses agent performance metrics including latency, throughput, and efficiency. It supports configurable thresholds and regression detection against baselines.

## Overview

The evaluator provides:

- **Latency Percentiles**: Track p50, p95, p99 latency from LLM calls
- **Time to First Token**: Measure streaming responsiveness
- **Throughput**: Calculate tokens per second
- **Token Efficiency**: Ratio of output tokens to total tokens
- **Configurable Thresholds**: Set pass/fail criteria for any metric
- **Regression Detection**: Statistical comparison against baselines

## Quick Start

### Basic Usage

```yaml
assertions:
  - type: "performance"
    config:
      thresholds:
        max_latency_p95_ms: 2000
        min_tokens_per_second: 50
```

### With Regression Detection

```yaml
assertions:
  - type: "performance"
    config:
      thresholds:
        max_latency_p95_ms: 2000
      baseline:
        latency_p95_mean: 1500
        latency_p95_std: 200
        n_samples: 10
      check_regression: true
      fail_on_regression: true
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `thresholds` | object | {} | Threshold values for metrics |
| `baseline` | object | null | Baseline data for regression detection |
| `significance_level` | float | 0.05 | P-value threshold for statistical significance |
| `regression_threshold_percent` | float | 10.0 | Percentage change to consider a regression |
| `check_latency` | bool | true | Whether to evaluate latency metrics |
| `check_throughput` | bool | true | Whether to evaluate throughput metrics |
| `check_efficiency` | bool | true | Whether to evaluate efficiency metrics |
| `check_regression` | bool | true | Whether to check for regressions |
| `fail_on_regression` | bool | false | Whether regression causes failure |

## Threshold Configuration

### Available Thresholds

```yaml
thresholds:
  max_latency_p50_ms: 500       # Maximum p50 latency
  max_latency_p95_ms: 1000      # Maximum p95 latency
  max_latency_p99_ms: 2000      # Maximum p99 latency
  max_time_to_first_token_ms: 300  # Maximum TTFT
  min_tokens_per_second: 100    # Minimum throughput
  min_token_efficiency: 0.3     # Minimum efficiency ratio
  max_total_duration_seconds: 30  # Maximum total duration
  max_memory_usage_mb: 512      # Maximum memory usage
```

### Threshold Behavior

- **max_* thresholds**: Metric must be <= threshold to pass
- **min_* thresholds**: Metric must be >= threshold to pass
- Unset thresholds are not checked

## Baseline Configuration

### Structure

```yaml
baseline:
  # Latency metrics (milliseconds)
  latency_p50_mean: 500
  latency_p50_std: 50
  latency_p95_mean: 800
  latency_p95_std: 80
  latency_p99_mean: 1000
  latency_p99_std: 100

  # Time to first token (milliseconds)
  time_to_first_token_mean: 250
  time_to_first_token_std: 30

  # Throughput (tokens/second)
  tokens_per_second_mean: 150
  tokens_per_second_std: 20

  # Efficiency (0-1)
  token_efficiency_mean: 0.4
  token_efficiency_std: 0.05

  # Duration (seconds)
  total_duration_mean: 2.5
  total_duration_std: 0.3

  # Number of samples in baseline
  n_samples: 10
```

### Regression Detection

Regression detection uses Welch's t-test to determine if current performance is significantly different from the baseline:

1. **Delta Calculation**: Current value - baseline mean
2. **Percentage Change**: Delta / baseline mean * 100
3. **Statistical Test**: Welch's t-test for significance
4. **Classification**:
   - **Regression**: Significant degradation beyond threshold
   - **Improvement**: Significant improvement beyond threshold
   - **No Change**: Within normal variation

## Metrics Explained

### Latency Percentiles

Calculated from LLM call durations in the event trace:

- **p50**: Median latency (50th percentile)
- **p95**: 95th percentile latency
- **p99**: 99th percentile latency

Higher percentiles capture tail latencies that affect user experience.

### Time to First Token (TTFT)

Time from request initiation to receiving the first response token. Critical for perceived responsiveness in streaming applications.

### Tokens Per Second

Throughput calculated as:
```
tokens_per_second = total_tokens / wall_time_seconds
```

Higher values indicate better throughput.

### Token Efficiency

Ratio of useful output to total token usage:
```
efficiency = output_tokens / total_tokens
```

Values range from 0 to 1. Higher efficiency means more of the token budget produced useful output.

## Example Test Suite

```yaml
test_suite: "Performance Tests"
version: "1.0"

tests:
  - id: "perf-001"
    name: "Basic Performance Check"
    task:
      description: "Summarize the given document"
    assertions:
      - type: "performance"
        config:
          thresholds:
            max_latency_p95_ms: 2000
            min_tokens_per_second: 50
            min_token_efficiency: 0.3

  - id: "perf-002"
    name: "Strict Latency Requirements"
    task:
      description: "Quick lookup task"
    assertions:
      - type: "performance"
        config:
          thresholds:
            max_latency_p50_ms: 500
            max_latency_p99_ms: 1500
            max_time_to_first_token_ms: 300

  - id: "perf-003"
    name: "Regression Test"
    task:
      description: "Standard processing task"
    assertions:
      - type: "performance"
        config:
          baseline:
            latency_p50_mean: 600
            latency_p50_std: 60
            latency_p95_mean: 1000
            latency_p95_std: 100
            tokens_per_second_mean: 120
            tokens_per_second_std: 15
            n_samples: 20
          check_regression: true
          fail_on_regression: true
          regression_threshold_percent: 15
```

## Evaluation Results

The evaluation result includes detailed metrics and check outcomes:

```json
{
  "name": "performance_summary",
  "passed": true,
  "score": 0.85,
  "message": "p50=500ms; p95=800ms; throughput=150.0tok/s",
  "details": {
    "metrics": {
      "latency_p50_ms": 500.0,
      "latency_p95_ms": 800.0,
      "latency_p99_ms": 1200.0,
      "time_to_first_token_ms": 250.0,
      "tokens_per_second": 150.0,
      "token_efficiency": 0.42,
      "total_duration_seconds": 2.5,
      "total_tokens": 500,
      "output_tokens": 200,
      "input_tokens": 300,
      "llm_call_count": 3,
      "tool_call_count": 2
    },
    "threshold_checks": [
      {
        "name": "latency_p95",
        "passed": true,
        "message": "p95 latency 800.00ms <= threshold 2000.00ms"
      }
    ],
    "regressions": [
      {
        "metric": "latency_p50",
        "status": "regression",
        "current_value": 600.0,
        "baseline_value": 450.0,
        "delta": 150.0,
        "delta_percent": 33.33,
        "p_value": 0.02,
        "is_significant": true
      }
    ]
  }
}
```

## Python API

### Using the Evaluator Directly

```python
from atp.evaluators import PerformanceEvaluator

# Create evaluator
evaluator = PerformanceEvaluator()

# Evaluate
result = await evaluator.evaluate(task, response, trace, assertion)
```

### Computing Metrics Programmatically

```python
from atp.evaluators.performance import (
    compute_performance_metrics,
    check_thresholds,
    check_regressions,
    PerformanceThresholds,
    PerformanceBaseline,
)

# Compute metrics from response and trace
metrics = compute_performance_metrics(response, trace)

# Check against thresholds
thresholds = PerformanceThresholds(
    max_latency_p95_ms=2000,
    min_tokens_per_second=100,
)
threshold_results = check_thresholds(metrics, thresholds)

# Check for regressions
baseline = PerformanceBaseline(
    latency_p50_mean=500,
    latency_p50_std=50,
    n_samples=10,
)
regression_results = check_regressions(
    metrics, baseline,
    significance_level=0.05,
    regression_threshold_percent=10.0,
)
```

### Computing Individual Metrics

```python
from atp.evaluators.performance import (
    calculate_percentile,
    extract_latencies_from_trace,
    calculate_tokens_per_second,
    calculate_token_efficiency,
)

# Extract latencies from trace
latencies = extract_latencies_from_trace(trace)

# Calculate percentiles
p50 = calculate_percentile(latencies, 50)
p95 = calculate_percentile(latencies, 95)
p99 = calculate_percentile(latencies, 99)

# Calculate throughput
throughput = calculate_tokens_per_second(
    total_tokens=500,
    duration_seconds=2.5,
)

# Calculate efficiency
efficiency = calculate_token_efficiency(
    output_tokens=200,
    total_tokens=500,
)
```

## Scoring

The overall performance score is calculated as:

```
score = (threshold_score * 0.7) + (regression_score * 0.3)
```

Where:
- **threshold_score**: Percentage of thresholds passed (1.0 if all pass)
- **regression_score**: 1.0 minus penalty for regressions (0.15 per regression, max 0.5)

## Best Practices

1. **Set Realistic Thresholds**: Base thresholds on historical data rather than arbitrary values

2. **Use Multiple Percentiles**: p50 shows typical behavior, p95/p99 reveal tail latencies

3. **Build Baselines**: Collect baseline data from multiple runs for accurate regression detection

4. **Configure Significance Level**: Use 0.05 (5%) for normal cases, 0.01 (1%) for stricter tests

5. **Tune Regression Threshold**: 10% is a good default; increase for noisier metrics

6. **Enable fail_on_regression** for CI/CD pipelines to catch performance regressions

7. **Monitor Token Efficiency**: Low efficiency may indicate excessive prompting or context

## Troubleshooting

### No Latency Data

- Ensure the trace contains LLM_REQUEST events
- Check that events have `duration_ms` in their payload

### Metrics Missing from Response

- Verify the agent adapter provides metrics in ATPResponse
- Some adapters may not capture all metrics

### High p99 Latency

- Check for outliers in LLM calls
- Consider timeout settings
- Review if retries are inflating latencies

### Regression Detection Too Sensitive

- Increase `regression_threshold_percent`
- Use a higher `significance_level` (e.g., 0.01)
- Collect more baseline samples for accurate std

### Regression Detection Not Triggering

- Verify baseline data is provided
- Check that `check_regression` is true
- Ensure baseline has non-zero std values

## See Also

- [Test Format Reference](../reference/test-format.md)
- [Evaluator Configuration](../reference/configuration.md)
- [Baseline Management](../reference/baseline.md)

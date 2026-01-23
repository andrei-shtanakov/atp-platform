# ATP Performance Guide

This guide covers performance optimization techniques for the ATP (Agent Test Platform).

## Table of Contents

1. [Performance Module Overview](#performance-module-overview)
2. [Profiling](#profiling)
3. [Caching](#caching)
4. [Async Optimizations](#async-optimizations)
5. [Memory Management](#memory-management)
6. [Startup Optimization](#startup-optimization)
7. [Benchmarking](#benchmarking)
8. [Best Practices](#best-practices)

## Performance Module Overview

ATP includes a comprehensive performance module (`atp.performance`) that provides:

- **Profiler**: Measure execution times with hierarchical tracking
- **Cache**: Cache parsed test suites and adapter instances
- **Memory Tracker**: Monitor memory usage and detect leaks
- **Async Utilities**: Optimize concurrent operations
- **Startup Optimization**: Lazy loading and deferred initialization
- **Benchmark Suite**: Standardized performance testing

```python
from atp.performance import (
    # Profiling
    enable_profiling, profile, profile_async, profiled,
    # Caching
    CachedTestLoader, TestSuiteCache, AdapterCache,
    # Memory
    MemoryTracker, track_memory,
    # Async
    gather_with_limit, ParallelExecutor,
    # Startup
    LazyModule, get_lazy_adapter_registry,
    # Benchmarks
    run_standard_benchmarks,
)
```

## Profiling

### Basic Profiling

Enable profiling to measure execution times:

```python
from atp.performance import enable_profiling, profile, get_profiler

# Enable global profiling
profiler = enable_profiling()

# Profile an operation
with profile("load_suite"):
    suite = loader.load_file("tests.yaml")

# Get timing statistics
stats = profiler.get_stats()
print(f"load_suite: {stats['load_suite'].mean_ms:.2f}ms")

# Print full report
print(profiler.format_report())
```

### Async Profiling

Profile async operations:

```python
from atp.performance import profile_async

async with profile_async("execute_test"):
    result = await orchestrator.run_single_test(test)
```

### Function Decorator

Use the `@profiled` decorator:

```python
from atp.performance import profiled

@profiled("my_operation")
def my_function():
    # Function body
    pass

@profiled()  # Uses function name
async def my_async_function():
    await asyncio.sleep(0.1)
```

### Hierarchical Profiling

Profile nested operations:

```python
with profile("suite_run"):
    with profile("load"):
        suite = loader.load_file(path)

    for test in suite.tests:
        with profile("test_run", {"test_id": test.id}):
            result = await run_test(test)
```

Results show parent-child relationships:

```
- suite_run: 5234.21ms
  - load: 45.32ms
  - test_run (test_id=test-001): 1523.45ms
  - test_run (test_id=test-002): 3665.44ms
```

## Caching

### Test Suite Caching

Cache parsed test suites to avoid repeated YAML parsing:

```python
from atp.performance import CachedTestLoader, TestSuiteCache

# Use shared global cache (recommended)
loader = CachedTestLoader()

# First load - parses YAML
suite = loader.load_file("tests.yaml")

# Second load - returns cached
suite = loader.load_file("tests.yaml")

# Get cache statistics
stats = loader.get_cache_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")
```

The cache automatically invalidates when files are modified (based on mtime and size).

### Custom Cache Configuration

```python
# Create cache with custom settings
cache = TestSuiteCache(
    max_size=100,           # Maximum entries
    ttl_seconds=3600,       # Optional TTL (1 hour)
)

# Use with loader
loader = CachedTestLoader(cache=cache, use_shared_cache=False)

# Manual invalidation
loader.invalidate("tests.yaml")

# Clear entire cache
count = loader.clear_cache()
```

### Adapter Caching

Cache adapter instances to avoid repeated initialization:

```python
from atp.performance import AdapterCache, get_adapter_cache

cache = get_adapter_cache()

# Get or create adapter
adapter = cache.get_or_create("http", {
    "endpoint": "http://localhost:8000",
    "timeout": 30,
})

# Same config returns cached instance
adapter2 = cache.get_or_create("http", {
    "endpoint": "http://localhost:8000",
    "timeout": 30,
})
assert adapter is adapter2
```

## Async Optimizations

### Concurrent Execution with Limits

Run tasks with controlled parallelism:

```python
from atp.performance import gather_with_limit, ParallelExecutor

# Simple concurrent execution
tasks = [process_item(item) for item in items]
results = await gather_with_limit(tasks, limit=5)

# Using ParallelExecutor
executor = ParallelExecutor(max_parallel=5)
results = await executor.map(process_item, items)
```

### Chunked Processing

Process items in chunks with optional delays:

```python
from atp.performance import chunked_gather

# Process in batches of 10 with 0.1s delay between
results = await chunked_gather(
    items,
    process_item,
    chunk_size=10,
    delay_between_chunks=0.1,
)
```

### Batch Processing

Batch items for efficient API calls:

```python
from atp.performance import AsyncBatcher

async def process_batch(items: list[str]) -> list[int]:
    # Efficient batch processing
    return [len(item) for item in items]

batcher = AsyncBatcher(
    processor=process_batch,
    batch_size=10,
    max_concurrent_batches=3,
)

results = await batcher.process(items)
```

### Rate Limiting

Limit request rate:

```python
from atp.performance import RateLimiter

limiter = RateLimiter(rate=10, burst=3)  # 10/s with burst of 3

async with limiter.limit():
    await make_request()
```

### Retry with Backoff

Retry failed operations:

```python
from atp.performance import retry_async

result = await retry_async(
    make_request,
    max_retries=3,
    delay=1.0,
    backoff=2.0,
    exceptions=(ConnectionError, TimeoutError),
)
```

### Resource Pooling

Pool reusable resources:

```python
from atp.performance import AsyncPool

async def create_connection():
    return await connect_to_db()

async def close_connection(conn):
    await conn.close()

pool = AsyncPool(
    factory=create_connection,
    cleanup=close_connection,
    max_size=10,
)

async with pool.resource() as conn:
    await conn.query(...)
```

## Memory Management

### Memory Tracking

Track memory usage during operations:

```python
from atp.performance import MemoryTracker, enable_memory_tracking

tracker = enable_memory_tracking()
tracker.start()

# Your operations
suite = loader.load_file("large_suite.yaml")
results = await orchestrator.run_suite(suite, agent_name)

# Take snapshots at key points
tracker.snapshot("after_load")
tracker.snapshot("after_run")

# Get report
report = tracker.stop()
print(report.format_report())
```

### Track Specific Operations

```python
from atp.performance import track_memory

with track_memory("load_test_suite"):
    suite = loader.load_file("tests.yaml")
```

### Memory Size Calculation

Calculate object memory usage:

```python
from atp.performance import get_memory_size

size = get_memory_size(large_data_structure)
print(f"Size: {size / 1024 / 1024:.2f} MB")
```

## Startup Optimization

### Lazy Module Loading

Defer heavy module imports:

```python
from atp.performance import LazyModule

# Module not imported until first use
lazy_httpx = LazyModule("httpx")

# Import happens here
client = lazy_httpx.AsyncClient()
```

### Lazy Class Instantiation

Defer expensive object creation:

```python
from atp.performance import LazyClass

def create_heavy_object():
    # Expensive initialization
    return HeavyObject()

lazy_obj = LazyClass(create_heavy_object)

# Object created on first get()
obj = lazy_obj.get()
```

### Deferred Registry

Load adapters on demand:

```python
from atp.performance import get_lazy_adapter_registry

# All adapter classes are registered but not loaded
registry = get_lazy_adapter_registry()

# Only loads HTTP adapter classes when accessed
http_adapter, http_config = registry.get("http")

# Other adapters remain unloaded
print(registry.is_created("container"))  # False
```

### Import Time Analysis

Identify slow imports:

```python
from atp.performance import ImportAnalyzer

analyzer = ImportAnalyzer()
analyzer.start()

# Import modules to analyze
import atp.cli.main

timings = analyzer.stop()
print(analyzer.format_report())
```

## Benchmarking

### Run Standard Benchmarks

Run the built-in benchmark suite:

```python
from atp.performance import run_standard_benchmarks

suite = run_standard_benchmarks()
print(suite.format_report())
```

Output:

```
Benchmark Suite: ATP Standard Benchmarks
============================================================
Total duration: 5.23s

Benchmark                           Iters   Mean        Min        Max     ops/s
------------------------------------------------------------
test_suite_load_uncached               47    21.28ms   18.45ms   25.12ms    47.0
test_suite_load_cached                428     2.33ms    2.01ms    3.21ms   429.2
score_aggregation                     892     1.12ms    0.98ms    1.45ms   892.9
statistics_calculation               2341     0.43ms    0.38ms    0.56ms  2325.6
...
```

### Custom Benchmarks

Create custom benchmarks:

```python
from atp.performance import Benchmark, BenchmarkSuite
import time

benchmark = Benchmark(
    warmup_iterations=3,
    min_iterations=10,
    max_iterations=100,
    target_seconds=1.0,
)

suite = BenchmarkSuite(
    name="Custom Benchmarks",
    start_time=time.time(),
    end_time=0,
)

# Benchmark synchronous operation
result = benchmark.run(
    "my_operation",
    my_function,
    setup=setup_func,
    teardown=cleanup_func,
    metadata={"param": "value"},
)
suite.results.append(result)

# Benchmark async operation
result = await benchmark.run_async(
    "async_operation",
    async_function,
)
suite.results.append(result)

suite.end_time = time.time()
print(suite.format_report())
```

## Best Practices

### 1. Enable Caching in Production

Always use `CachedTestLoader` when loading test suites multiple times:

```python
# Good
loader = CachedTestLoader()
for path in test_paths:
    suite = loader.load_file(path)

# Avoid
for path in test_paths:
    loader = TestLoader()
    suite = loader.load_file(path)
```

### 2. Control Parallelism

Set appropriate concurrency limits based on system resources:

```python
from atp.performance import ConcurrencyConfig

# Auto-configure based on CPU count
config = ConcurrencyConfig.auto()

# Or set explicitly
config = ConcurrencyConfig(
    max_parallel=min(cpu_count * 2, 20),
    batch_size=50,
)
```

### 3. Use Profiling for Optimization

Profile before optimizing:

```python
profiler = enable_profiling()

# Run your workload
run_tests(...)

# Identify bottlenecks
stats = profiler.get_stats()
slowest = sorted(stats.values(), key=lambda s: s.total_seconds, reverse=True)
for s in slowest[:5]:
    print(f"{s.operation}: {s.total_ms:.1f}ms ({s.count} calls)")
```

### 4. Monitor Memory for Large Suites

Track memory when processing large test suites:

```python
tracker = enable_memory_tracking()
tracker.start()

for suite_path in large_suite_paths:
    suite = loader.load_file(suite_path)
    tracker.snapshot(f"loaded:{suite_path}")

    result = await orchestrator.run_suite(suite, agent_name)
    tracker.snapshot(f"completed:{suite_path}")

    # Clear references
    del suite, result

report = tracker.stop()
if report.net_memory_change_bytes > 100 * 1024 * 1024:  # 100MB
    logger.warning("High memory usage detected")
```

### 5. Lazy Load Heavy Dependencies

Use lazy loading for optional or rarely-used features:

```python
# Instead of top-level import
# import anthropic  # Slow if not used

# Use lazy loading
lazy_anthropic = LazyModule("anthropic")

def create_llm_evaluator():
    # Only imports when this function is called
    client = lazy_anthropic.Anthropic()
    return LLMEvaluator(client)
```

### 6. Run Benchmarks in CI

Include benchmarks in CI to detect performance regressions:

```yaml
# .github/workflows/benchmark.yml
- name: Run benchmarks
  run: |
    python -c "from atp.performance import run_standard_benchmarks; \
               suite = run_standard_benchmarks(); \
               print(suite.format_report())"
```

## Performance Targets

ATP aims to meet these performance targets (from NFR-001):

| Metric | Target |
|--------|--------|
| Platform overhead | < 5% of agent execution time |
| CLI startup time | < 2 seconds |
| Parallel agents | Up to 10 simultaneous |
| Event handling | 10,000+ events without degradation |

Use the benchmark suite and profiler to verify these targets are met.

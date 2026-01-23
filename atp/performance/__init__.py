"""Performance monitoring and optimization module for ATP.

This module provides:
- Profiling infrastructure for measuring execution times
- Caching utilities for parsed tests and adapters
- Memory usage tracking and auditing
- Async optimization utilities
- Startup time optimization (lazy loading)
- Benchmark suite for performance testing

Usage:
    from atp.performance import profiler, cache

    # Enable profiling
    with profiler.profile("test_run"):
        result = await run_test(...)

    # Use cached loader
    loader = CachedTestLoader()
    suite = loader.load_file("tests.yaml")

    # Run with concurrency limit
    results = await gather_with_limit(tasks, limit=5)

    # Lazy loading
    lazy_registry = get_lazy_adapter_registry()
    adapter_class, config_class = lazy_registry.get("http")
"""

from atp.performance.async_utils import (
    AsyncBatcher,
    AsyncPool,
    ConcurrencyConfig,
    ParallelExecutor,
    RateLimiter,
    chunked_gather,
    gather_with_limit,
    retry_async,
    stream_with_timeout,
    timeout_wrapper,
)
from atp.performance.benchmark import (
    Benchmark,
    BenchmarkResult,
    BenchmarkSuite,
    run_async_benchmarks,
    run_standard_benchmarks,
)
from atp.performance.cache import (
    AdapterCache,
    CachedTestLoader,
    CacheEntry,
    CacheStats,
    TestSuiteCache,
    clear_all_caches,
    get_adapter_cache,
    get_test_suite_cache,
)
from atp.performance.memory import (
    MemoryDiff,
    MemoryReport,
    MemorySnapshot,
    MemoryTracker,
    enable_memory_tracking,
    get_memory_size,
    get_memory_tracker,
    track_memory,
)
from atp.performance.profiler import (
    Profiler,
    ProfileResult,
    ProfileStats,
    disable_profiling,
    enable_profiling,
    get_profiler,
    profile,
    profile_async,
    profiled,
    set_profiler,
)
from atp.performance.startup import (
    DeferredRegistry,
    ImportAnalyzer,
    ImportTiming,
    LazyClass,
    LazyModule,
    get_lazy_adapter_registry,
    measure_startup_time,
)

__all__ = [
    # Profiler
    "Profiler",
    "ProfileResult",
    "ProfileStats",
    "get_profiler",
    "set_profiler",
    "enable_profiling",
    "disable_profiling",
    "profile",
    "profile_async",
    "profiled",
    # Cache
    "CachedTestLoader",
    "TestSuiteCache",
    "AdapterCache",
    "CacheEntry",
    "CacheStats",
    "get_test_suite_cache",
    "get_adapter_cache",
    "clear_all_caches",
    # Memory
    "MemoryTracker",
    "MemorySnapshot",
    "MemoryDiff",
    "MemoryReport",
    "get_memory_tracker",
    "enable_memory_tracking",
    "track_memory",
    "get_memory_size",
    # Async utilities
    "AsyncBatcher",
    "AsyncPool",
    "ConcurrencyConfig",
    "ParallelExecutor",
    "RateLimiter",
    "gather_with_limit",
    "chunked_gather",
    "timeout_wrapper",
    "retry_async",
    "stream_with_timeout",
    # Startup / Lazy loading
    "LazyModule",
    "LazyClass",
    "DeferredRegistry",
    "ImportAnalyzer",
    "ImportTiming",
    "get_lazy_adapter_registry",
    "measure_startup_time",
    # Benchmark
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkSuite",
    "run_standard_benchmarks",
    "run_async_benchmarks",
]

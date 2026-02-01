"""Benchmark suite registry for agent evaluation.

This module provides a registry of curated benchmark test suites for common
agent evaluation scenarios including coding, research, reasoning, and data
processing tasks.

Example usage:
    from atp.benchmarks import get_registry, list_benchmarks, list_categories

    # List available benchmarks
    for name in list_benchmarks():
        print(name)

    # Get a specific benchmark suite
    registry = get_registry()
    coding_suite = registry.get("coding")

    # Get benchmarks by category
    research_suites = registry.get_by_category("research")

    # Normalize scores to 0-100 scale
    normalized = registry.normalize_score(0.85)  # Returns 85.0
"""

from .models import (
    BaselineScore,
    BenchmarkCategory,
    BenchmarkDifficulty,
    BenchmarkMetadata,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkSuiteInfo,
    BenchmarkSuiteResult,
    BenchmarkTest,
    NormalizationConfig,
)
from .registry import (
    BenchmarkCategoryNotFoundError,
    BenchmarkNotFoundError,
    BenchmarkRegistry,
    get_benchmark,
    get_registry,
    list_benchmarks,
    list_categories,
)

__all__ = [
    # Registry
    "BenchmarkRegistry",
    "get_registry",
    "get_benchmark",
    "list_benchmarks",
    "list_categories",
    # Exceptions
    "BenchmarkNotFoundError",
    "BenchmarkCategoryNotFoundError",
    # Models
    "BenchmarkCategory",
    "BenchmarkDifficulty",
    "BenchmarkMetadata",
    "BenchmarkTest",
    "BenchmarkSuite",
    "BenchmarkSuiteInfo",
    "BenchmarkResult",
    "BenchmarkSuiteResult",
    "BaselineScore",
    "NormalizationConfig",
]

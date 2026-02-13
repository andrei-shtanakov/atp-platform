#!/usr/bin/env python3
"""Benchmark script for Dashboard v2 performance.

This script measures:
- App startup time
- Route registration time
- Memory usage

Usage:
    python scripts/benchmark_dashboard.py
"""

import gc
import os
import statistics
import sys
import time
import tracemalloc
from typing import Any

# Ensure atp package is importable
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)


def measure_time(func: callable, iterations: int = 5) -> dict[str, float]:
    """Measure execution time of a function.

    Args:
        func: Function to measure.
        iterations: Number of iterations.

    Returns:
        Dictionary with min, max, mean, stdev times in ms.
    """
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        "min_ms": min(times),
        "max_ms": max(times),
        "mean_ms": statistics.mean(times),
        "stdev_ms": (statistics.stdev(times) if len(times) > 1 else 0),
    }


def measure_memory(func: callable) -> dict[str, float]:
    """Measure peak memory usage of a function.

    Args:
        func: Function to measure.

    Returns:
        Dictionary with peak and current memory in MB.
    """
    gc.collect()
    tracemalloc.start()
    func()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "current_mb": current / 1024 / 1024,
        "peak_mb": peak / 1024 / 1024,
    }


def benchmark_app_creation() -> dict[str, Any]:
    """Benchmark v2 app creation time and memory."""
    results: dict[str, Any] = {}

    print("Benchmarking v2 app creation...")

    def create_v2():
        mods_to_remove = [m for m in sys.modules if m.startswith("atp.dashboard")]
        for m in mods_to_remove:
            del sys.modules[m]

        from atp.dashboard.v2 import create_app

        return create_app()

    results["time"] = measure_time(create_v2, iterations=3)
    results["memory"] = measure_memory(create_v2)

    return results


def benchmark_route_count() -> int:
    """Count routes in v2."""
    mods_to_remove = [m for m in sys.modules if m.startswith("atp.dashboard")]
    for m in mods_to_remove:
        del sys.modules[m]

    from atp.dashboard.v2 import create_app

    app = create_app()
    return len([r for r in app.routes if hasattr(r, "path")])


def benchmark_file_sizes() -> dict[str, Any]:
    """Analyze v2 file sizes."""
    import pathlib

    base = pathlib.Path(__file__).parent.parent / "atp" / "dashboard"

    def get_python_files_size(
        path: pathlib.Path,
    ) -> int:
        """Get total size of Python files in directory."""
        total = 0
        for f in path.rglob("*.py"):
            total += f.stat().st_size
        return total

    def count_lines(path: pathlib.Path) -> int:
        """Count lines of Python code in directory."""
        total = 0
        for f in path.rglob("*.py"):
            with open(f) as fp:
                total += sum(1 for _ in fp)
        return total

    return {
        "total_bytes": get_python_files_size(base / "v2"),
        "routes_bytes": get_python_files_size(base / "v2" / "routes"),
        "services_bytes": get_python_files_size(base / "v2" / "services"),
        "total_lines": count_lines(base / "v2"),
    }


def print_results(results: dict[str, Any]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 60)
    print("Dashboard v2 Performance Benchmark")
    print("=" * 60)

    # App creation time
    print("\n### App Creation Time (ms)")
    print("-" * 40)
    time_data = results["app_creation"]["time"]
    print(f"  Mean:   {time_data['mean_ms']:>10.2f} ms")
    print(f"  Min:    {time_data['min_ms']:>10.2f} ms")
    print(f"  Max:    {time_data['max_ms']:>10.2f} ms")

    # Memory usage
    print("\n### Memory Usage (MB)")
    print("-" * 40)
    mem = results["app_creation"]["memory"]
    print(f"  Peak:    {mem['peak_mb']:>10.2f} MB")
    print(f"  Current: {mem['current_mb']:>10.2f} MB")

    # Route count
    print("\n### Route Count")
    print("-" * 40)
    print(f"  Routes: {results['route_count']}")

    # File sizes
    print("\n### Code Organization")
    print("-" * 40)
    fs = results["file_sizes"]
    total_kb = fs["total_bytes"] / 1024
    routes_kb = fs["routes_bytes"] / 1024
    services_kb = fs["services_bytes"] / 1024
    lines = fs["total_lines"]

    print(f"  Total:      {total_kb:>10.1f} KB ({lines} lines)")
    print(f"  - routes/:  {routes_kb:>10.1f} KB")
    print(f"  - services/:{services_kb:>10.1f} KB")
    print("=" * 60)


def main() -> None:
    """Run all benchmarks."""
    print("Starting Dashboard v2 Performance Benchmark...")
    print("This may take a minute...\n")

    results: dict[str, Any] = {}

    print("[1/3] Benchmarking app creation...")
    results["app_creation"] = benchmark_app_creation()

    print("[2/3] Counting routes...")
    results["route_count"] = benchmark_route_count()

    print("[3/3] Analyzing file sizes...")
    results["file_sizes"] = benchmark_file_sizes()

    print_results(results)


if __name__ == "__main__":
    main()

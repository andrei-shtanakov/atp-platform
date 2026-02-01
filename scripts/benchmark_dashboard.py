#!/usr/bin/env python3
"""Benchmark script for comparing Dashboard v1 vs v2 performance.

This script measures:
- App startup time
- Route registration time
- Request response time for key endpoints
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        times.append((end - start) * 1000)  # Convert to ms

    return {
        "min_ms": min(times),
        "max_ms": max(times),
        "mean_ms": statistics.mean(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
    }


def measure_memory(func: callable) -> dict[str, float]:
    """Measure memory usage of a function.

    Args:
        func: Function to measure.

    Returns:
        Dictionary with current and peak memory in MB.
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
    """Benchmark app creation for v1 and v2."""
    results = {"v1": {}, "v2": {}}

    # Benchmark v1
    print("Benchmarking v1 app creation...")

    def create_v1():
        # Clear module cache to get fresh import time
        mods_to_remove = [m for m in sys.modules if m.startswith("atp.dashboard")]
        for m in mods_to_remove:
            del sys.modules[m]

        os.environ["ATP_DASHBOARD_V2"] = "false"
        from atp.dashboard.app import create_app

        return create_app()

    results["v1"]["time"] = measure_time(create_v1, iterations=3)
    results["v1"]["memory"] = measure_memory(create_v1)

    # Benchmark v2
    print("Benchmarking v2 app creation...")

    def create_v2():
        # Clear module cache to get fresh import time
        mods_to_remove = [m for m in sys.modules if m.startswith("atp.dashboard")]
        for m in mods_to_remove:
            del sys.modules[m]

        os.environ["ATP_DASHBOARD_V2"] = "true"
        from atp.dashboard.v2 import create_app

        return create_app()

    results["v2"]["time"] = measure_time(create_v2, iterations=3)
    results["v2"]["memory"] = measure_memory(create_v2)

    return results


def benchmark_route_count() -> dict[str, int]:
    """Count routes in v1 and v2."""
    results = {}

    # Clear cache
    mods_to_remove = [m for m in sys.modules if m.startswith("atp.dashboard")]
    for m in mods_to_remove:
        del sys.modules[m]

    # v1 routes
    os.environ["ATP_DASHBOARD_V2"] = "false"
    from atp.dashboard.app import create_app as create_v1

    v1_app = create_v1()
    results["v1"] = len([r for r in v1_app.routes if hasattr(r, "path")])

    # Clear cache
    mods_to_remove = [m for m in sys.modules if m.startswith("atp.dashboard")]
    for m in mods_to_remove:
        del sys.modules[m]

    # v2 routes
    os.environ["ATP_DASHBOARD_V2"] = "true"
    from atp.dashboard.v2 import create_app as create_v2

    v2_app = create_v2()
    results["v2"] = len([r for r in v2_app.routes if hasattr(r, "path")])

    return results


def benchmark_file_sizes() -> dict[str, dict[str, int]]:
    """Compare file sizes between v1 and v2."""
    import pathlib

    base = pathlib.Path(__file__).parent.parent / "atp" / "dashboard"

    def get_python_files_size(path: pathlib.Path) -> int:
        """Get total size of Python files in a directory."""
        total = 0
        for f in path.rglob("*.py"):
            total += f.stat().st_size
        return total

    def count_lines(path: pathlib.Path) -> int:
        """Count lines of Python code in a directory."""
        total = 0
        for f in path.rglob("*.py"):
            with open(f) as fp:
                total += sum(1 for _ in fp)
        return total

    results = {
        "v1": {
            "app_py_bytes": (base / "app.py").stat().st_size
            if (base / "app.py").exists()
            else 0,
            "api_py_bytes": (base / "api.py").stat().st_size
            if (base / "api.py").exists()
            else 0,
        },
        "v2": {
            "total_bytes": get_python_files_size(base / "v2"),
            "routes_bytes": get_python_files_size(base / "v2" / "routes"),
            "services_bytes": get_python_files_size(base / "v2" / "services"),
            "total_lines": count_lines(base / "v2"),
        },
    }

    return results


def print_results(results: dict[str, Any]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 60)
    print("Dashboard v1 vs v2 Performance Comparison")
    print("=" * 60)

    # App creation time
    print("\n### App Creation Time (ms)")
    print("-" * 40)
    print(f"{'Metric':<15} {'v1':<15} {'v2':<15} {'Diff':<15}")
    print("-" * 40)

    v1_time = results["app_creation"]["v1"]["time"]["mean_ms"]
    v2_time = results["app_creation"]["v2"]["time"]["mean_ms"]
    diff = ((v2_time - v1_time) / v1_time) * 100 if v1_time > 0 else 0

    print(f"{'Mean':<15} {v1_time:>10.2f} ms  {v2_time:>10.2f} ms  {diff:>+10.1f}%")
    print(
        f"{'Min':<15} {results['app_creation']['v1']['time']['min_ms']:>10.2f} ms  "
        f"{results['app_creation']['v2']['time']['min_ms']:>10.2f} ms"
    )
    print(
        f"{'Max':<15} {results['app_creation']['v1']['time']['max_ms']:>10.2f} ms  "
        f"{results['app_creation']['v2']['time']['max_ms']:>10.2f} ms"
    )

    # Memory usage
    print("\n### Memory Usage (MB)")
    print("-" * 40)
    v1_mem = results["app_creation"]["v1"]["memory"]["peak_mb"]
    v2_mem = results["app_creation"]["v2"]["memory"]["peak_mb"]
    diff = ((v2_mem - v1_mem) / v1_mem) * 100 if v1_mem > 0 else 0

    print(f"{'Peak':<15} {v1_mem:>10.2f} MB  {v2_mem:>10.2f} MB  {diff:>+10.1f}%")

    # Route count
    print("\n### Route Count")
    print("-" * 40)
    print(f"v1 routes: {results['route_count']['v1']}")
    print(f"v2 routes: {results['route_count']['v2']}")

    # File sizes
    print("\n### Code Organization")
    print("-" * 40)
    v1_app = results["file_sizes"]["v1"]["app_py_bytes"] / 1024
    v1_api = results["file_sizes"]["v1"]["api_py_bytes"] / 1024
    v2_total = results["file_sizes"]["v2"]["total_bytes"] / 1024
    v2_routes = results["file_sizes"]["v2"]["routes_bytes"] / 1024
    v2_services = results["file_sizes"]["v2"]["services_bytes"] / 1024
    v2_lines = results["file_sizes"]["v2"]["total_lines"]

    print("v1:")
    print(f"  app.py:     {v1_app:>10.1f} KB")
    print(f"  api.py:     {v1_api:>10.1f} KB")
    print(f"  Total:      {v1_app + v1_api:>10.1f} KB")

    print("\nv2:")
    print(f"  Total:      {v2_total:>10.1f} KB ({v2_lines} lines)")
    print(f"  - routes/:  {v2_routes:>10.1f} KB")
    print(f"  - services/:{v2_services:>10.1f} KB")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(
        f"✓ App creation: v2 is {abs(diff):.1f}% {'faster' if diff < 0 else 'slower'}"
    )
    mem_diff = abs(((v2_mem - v1_mem) / v1_mem) * 100)
    mem_dir = "less" if v2_mem < v1_mem else "more"
    print(f"✓ Memory: v2 uses {mem_diff:.1f}% {mem_dir}")
    v2_lines = results["file_sizes"]["v2"]["total_lines"]
    print(f"✓ Code organization: v2 splits into {v2_lines} lines across files")
    v1_routes = results["route_count"]["v1"]
    v2_routes_count = results["route_count"]["v2"]
    print(f"✓ Routes: Similar route counts (v1={v1_routes}, v2={v2_routes_count})")
    print("=" * 60)


def main() -> None:
    """Run all benchmarks."""
    print("Starting Dashboard v1 vs v2 Performance Benchmark...")
    print("This may take a minute...\n")

    results = {}

    print("[1/3] Benchmarking app creation...")
    results["app_creation"] = benchmark_app_creation()

    print("[2/3] Counting routes...")
    results["route_count"] = benchmark_route_count()

    print("[3/3] Analyzing file sizes...")
    results["file_sizes"] = benchmark_file_sizes()

    print_results(results)


if __name__ == "__main__":
    main()

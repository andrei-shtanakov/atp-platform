"""Benchmark loaders for converting standard benchmarks to ATP test suites.

Provides loaders for HumanEval, SWE-bench, and MMLU benchmarks.
"""

from atp.benchmarks.loaders.base import BenchmarkLoader
from atp.benchmarks.loaders.humaneval import HumanEvalLoader
from atp.benchmarks.loaders.mmlu import MMLULoader
from atp.benchmarks.loaders.swebench import SWEBenchLoader

LOADERS: dict[str, type[BenchmarkLoader]] = {
    "humaneval": HumanEvalLoader,
    "swe-bench": SWEBenchLoader,
    "mmlu": MMLULoader,
}


def get_loader(name: str) -> BenchmarkLoader:
    """Get a benchmark loader by name.

    Args:
        name: Loader name (humaneval, swe-bench, mmlu).

    Returns:
        Instantiated BenchmarkLoader.

    Raises:
        ValueError: If loader name is unknown.
    """
    loader_cls = LOADERS.get(name)
    if loader_cls is None:
        available = ", ".join(sorted(LOADERS.keys()))
        raise ValueError(f"Unknown benchmark: {name!r}. Available: {available}")
    return loader_cls()


__all__ = [
    "BenchmarkLoader",
    "HumanEvalLoader",
    "MMLULoader",
    "SWEBenchLoader",
    "LOADERS",
    "get_loader",
]

"""Baseline module for ATP.

Provides functionality for saving test baselines and detecting regressions.
"""

from .comparison import (
    ComparisonResult,
    TestComparison,
    compare_results,
    compare_test,
    welchs_t_test,
)
from .models import (
    Baseline,
    ChangeType,
    TestBaseline,
)
from .reporter import (
    format_comparison_console,
    format_comparison_json,
    print_comparison,
)
from .storage import (
    load_baseline,
    save_baseline,
)

__all__ = [
    # Models
    "Baseline",
    "TestBaseline",
    "ChangeType",
    # Comparison
    "ComparisonResult",
    "TestComparison",
    "compare_results",
    "compare_test",
    "welchs_t_test",
    # Reporter
    "format_comparison_console",
    "format_comparison_json",
    "print_comparison",
    # Storage
    "load_baseline",
    "save_baseline",
]

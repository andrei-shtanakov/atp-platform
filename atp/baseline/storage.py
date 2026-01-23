"""Storage utilities for baseline files."""

import json
from pathlib import Path

from .models import Baseline


def save_baseline(baseline: Baseline, path: Path) -> None:
    """Save baseline to a JSON file.

    Args:
        baseline: Baseline data to save.
        path: Path to save the baseline file.

    Raises:
        OSError: If file cannot be written.
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline.to_dict(), f, indent=2, ensure_ascii=False)


def load_baseline(path: Path) -> Baseline:
    """Load baseline from a JSON file.

    Args:
        path: Path to the baseline file.

    Returns:
        Loaded Baseline instance.

    Raises:
        FileNotFoundError: If baseline file does not exist.
        json.JSONDecodeError: If file is not valid JSON.
        ValueError: If baseline data is invalid.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return Baseline.from_dict(data)

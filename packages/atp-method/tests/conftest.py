"""Shared fixtures for atp-method tests."""

from pathlib import Path

import pytest

# repo root: packages/atp-method/tests/conftest.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CASES_DIR = REPO_ROOT / "method" / "cases" / "req-extraction"


@pytest.fixture
def example_cases_dir() -> Path:
    """Directory of the shipped req-extraction example sweep."""
    return EXAMPLE_CASES_DIR


@pytest.fixture
def clean_case_path(example_cases_dir: Path) -> Path:
    """The 'clean' level example case."""
    return example_cases_dir / "case-req-extraction-fabricated-deadline-clean-001.yaml"


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on asyncio (anyio, per project convention)."""
    return "asyncio"

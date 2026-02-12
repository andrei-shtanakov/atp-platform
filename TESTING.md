# ATP Testing Infrastructure

This document describes the testing infrastructure for the ATP (Agent Test Platform) project.

## Overview

ATP uses pytest as the testing framework with the following key components:
- **pytest**: Test framework
- **pytest-cov**: Coverage reporting
- **pytest-anyio**: Async testing support
- **pytest-mock**: Mocking utilities
- **pre-commit**: Git hooks for code quality

## Quick Start

```bash
# Run all tests with coverage
uv run pytest

# Run specific test module
uv run pytest tests/unit/loader -v

# Generate HTML coverage report
uv run pytest --cov=atp --cov-report=html
open htmlcov/index.html
```

## Test Structure

```
tests/
├── conftest.py           # Shared pytest fixtures
├── fixtures/             # Test data and sample files
│   ├── artifacts/       # Artifact test data
│   ├── comparison/      # Comparison test data
│   ├── mock_tools/      # Mock tool definitions
│   ├── protocol/        # Protocol test data
│   ├── test_filesystem/ # Filesystem evaluator fixtures
│   ├── test_site/       # Test e-commerce site (port 9876)
│   ├── test_suites/     # YAML test suite examples
│   └── traces/          # Event trace samples
├── unit/                 # Unit tests (70% of tests)
│   └── loader/          # Loader module tests
├── integration/          # Integration tests (20% of tests)
├── contract/             # Protocol contract tests
└── e2e/                  # End-to-end tests (10% of tests)
```

## Coverage Requirements

- **Minimum coverage gate**: 80%
- **Current coverage**: 91%
- Coverage reports in `htmlcov/` and `coverage.xml`
- CI/CD will fail if coverage drops below 80%

## Shared Fixtures

Available fixtures in `tests/conftest.py`:

### Path Fixtures
- `fixtures_dir`: Path to tests/fixtures directory
- `test_suites_dir`: Path to test_suites fixtures
- `valid_suite_path`: Path to valid_suite.yaml
- `tmp_work_dir`: Temporary working directory

### Data Fixtures
- `sample_env_vars`: Dict of sample environment variables
- `empty_env_vars`: Empty dict for testing missing vars
- `sample_atp_request`: Sample ATP request data
- `sample_atp_response`: Sample ATP response data
- `sample_atp_event`: Sample ATP event data
- `sample_test_definition`: Sample test definition
- `sample_agent_config`: Sample agent configuration

## Pre-commit Hooks

Hooks run automatically on `git commit`:

1. **Trailing whitespace**: Remove trailing whitespace
2. **End-of-file fixer**: Ensure newline at EOF
3. **YAML/JSON/TOML check**: Validate file syntax
4. **ruff format**: Format code to 88 char line length
5. **ruff check**: Lint code (PEP 8, imports, etc.)
6. **pyrefly**: Type checking (optional, continues on error)

### Installation

```bash
# Install hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files

# Skip hooks (emergency only)
git commit --no-verify
```

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):

### Test Job
1. Install Python 3.12
2. Install dependencies with uv
3. Run ruff format check
4. Run ruff lint
5. Run pyrefly type check (optional)
6. Run tests with coverage (≥80% required)
7. Upload coverage to Codecov

### Lint Job
1. Check code formatting
2. Check linting rules

## Writing Tests

### Basic Test

```python
def test_example():
    """Test description."""
    result = my_function()
    assert result == expected
```

### Using Fixtures

```python
def test_with_fixtures(fixtures_dir, sample_env_vars):
    """Test using shared fixtures."""
    loader = TestLoader(env=sample_env_vars)
    suite = loader.load_file(fixtures_dir / "test_suites" / "valid_suite.yaml")
    assert suite.version == "1.0"
```

### Async Tests

```python
import anyio

async def test_async_function():
    """Test async code."""
    result = await async_operation()
    assert result is not None
```

### Mocking

```python
def test_with_mock(mocker):
    """Test using pytest-mock."""
    mock_func = mocker.patch("module.function")
    mock_func.return_value = "mocked"

    result = call_function_that_uses_mocked_func()
    assert result == "mocked"
    mock_func.assert_called_once()
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    """Test with multiple inputs."""
    assert double(input) == expected
```

## Best Practices

1. **Test Naming**: Use descriptive names that explain what is tested
2. **One Assert Per Test**: Focus each test on a single behavior
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use Fixtures**: Reuse common setup via fixtures
5. **Test Edge Cases**: Include boundary conditions and error cases
6. **Keep Tests Fast**: Unit tests should run in milliseconds
7. **Mock External Dependencies**: Use mocks for HTTP, files, etc.
8. **Coverage ≠ Quality**: Write meaningful tests, not just coverage

## Running Specific Tests

```bash
# Run by pattern
uv run pytest -k "test_loader"

# Run specific test
uv run pytest tests/unit/loader/test_loader.py::TestTestLoader::test_load_valid_suite

# Run failed tests from last run
uv run pytest --lf

# Run with verbose output
uv run pytest -v

# Stop on first failure
uv run pytest -x

# Show print statements
uv run pytest -s
```

## Debugging Tests

```bash
# Use pytest with debugger
uv run pytest --pdb

# Use breakpoint() in test code
def test_example():
    result = my_function()
    breakpoint()  # Debugger will stop here
    assert result == expected
```

## Continuous Integration

All PRs must pass:
- ✅ Code formatting (ruff format)
- ✅ Linting (ruff check)
- ✅ Type checking (pyrefly, optional)
- ✅ All tests pass
- ✅ Coverage ≥80%

## Coverage Exclusions

The following are excluded from coverage (see `pyproject.toml`):
- `# pragma: no cover` comment
- `__repr__` methods
- Defensive `raise AssertionError`
- `raise NotImplementedError`
- `if __name__ == "__main__"` blocks
- `if TYPE_CHECKING:` blocks
- Protocol classes
- Abstract methods

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [pre-commit documentation](https://pre-commit.com/)
- [ruff documentation](https://docs.astral.sh/ruff/)

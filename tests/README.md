# ATP Test Suite

This directory contains the test suite for the ATP (Agent Test Platform) project.

## Structure

```
tests/
├── conftest.py           # Shared pytest fixtures
├── fixtures/             # Test data and fixtures
│   └── test_suites/     # Sample YAML test suite files
└── unit/                 # Unit tests
    └── loader/          # Tests for loader module
```

## Running Tests

```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/unit/loader/test_loader.py -v

# Run with coverage report
uv run pytest --cov=atp --cov-report=term-missing

# Run only fast tests (exclude slow)
uv run pytest -v -m "not slow"
```

## Coverage Requirements

- Minimum coverage: 80%
- Current coverage: 91%
- Coverage reports are generated in `htmlcov/` directory

## Writing Tests

### Using Fixtures

Common fixtures are available in `conftest.py`:

```python
def test_example(fixtures_dir, sample_env_vars, tmp_work_dir):
    """Example test using shared fixtures."""
    # fixtures_dir: Path to tests/fixtures
    # sample_env_vars: Dict of sample environment variables
    # tmp_work_dir: Temporary directory for test files
    pass
```

### Test Organization

- Unit tests: `tests/unit/` - Test individual functions/classes
- Integration tests: `tests/integration/` - Test component interactions
- E2E tests: `tests/e2e/` - Test full workflows

### Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

## Pre-commit Hooks

Pre-commit hooks are configured to run:
- Code formatting (ruff format)
- Linting (ruff check)
- Type checking (pyrefly)

Install hooks:
```bash
uv run pre-commit install
```

Run manually:
```bash
uv run pre-commit run --all-files
```

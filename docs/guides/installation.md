# Installation Guide

This guide will help you install and set up the ATP Platform on your system.

## Prerequisites

### Required

- **Python 3.12 or higher**
  ```bash
  python --version  # Should be 3.12 or higher
  ```

- **uv package manager**
  ```bash
  # Install uv (macOS/Linux)
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Verify installation
  uv --version
  ```

### Recommended

- **Git** - for cloning the repository
- **Make** - for using convenience commands (optional)
- **Docker** - for running agent adapters (optional, future feature)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/atp-platform.git
cd atp-platform
```

### 2. Install Dependencies

```bash
# Sync all dependencies from lock file
uv sync

# This will install:
# - Core dependencies (pydantic, ruamel-yaml, httpx)
# - Development tools (pytest, ruff, pyrefly)
# - All pinned versions from uv.lock
```

### 3. Verify Installation

```bash
# Run tests to verify everything works
uv run pytest tests/ -v

# Expected output:
# ============================= test session starts ==============================
# collected 49 items
#
# tests/unit/loader/test_loader.py::test_load_valid_suite PASSED         [  2%]
# ...
# ============================= 49 passed in 0.50s ===============================
```

### 4. Check Code Quality Tools

```bash
# Format code
uv run ruff format .

# Lint check
uv run ruff check .

# Type checking (requires pyrefly init first time)
pyrefly check
```

## Installation Verification

Run the example script to ensure the loader works:

```bash
# Run the loader example
uv run python examples/loader_example.py

# Expected output:
# Suite: example_suite
# Tests: 2
#
# Test: test-001 - Basic file creation test
#   Tags: ['smoke', 'basic']
#   Max steps: 5
#   Timeout: 60s
```

## Development Setup

If you plan to contribute or develop ATP features:

```bash
# Install all dev dependencies (already included in uv sync)
uv sync

# Set up pre-commit hooks (optional)
# Coming soon: pre-commit configuration

# Verify development environment
make lint    # Run all quality checks
make test    # Run all tests with coverage
```

## Common Issues

### Issue: `uv: command not found`

**Solution**: Install uv package manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc
```

### Issue: Python version too old

**Solution**: Install Python 3.12+:
```bash
# macOS with Homebrew
brew install python@3.12

# Ubuntu/Debian
sudo apt-get install python3.12

# Or use pyenv
pyenv install 3.12
pyenv local 3.12
```

### Issue: `ImportError` when running tests

**Solution**: Ensure you're running commands with `uv run`:
```bash
# Wrong
pytest tests/

# Correct
uv run pytest tests/
```

### Issue: Tests fail on fresh install

**Solution**: Try cleaning and reinstalling:
```bash
# Remove virtual environment and lock
rm -rf .venv uv.lock

# Reinstall
uv sync

# Try again
uv run pytest tests/ -v
```

## Optional Components

### Enable Type Checking

```bash
# Initialize pyrefly (first time only)
pyrefly init

# Run type checking
pyrefly check

# This will analyze all Python files for type errors
```

### IDE Setup

#### VS Code

Install recommended extensions:
- Python
- Pylance
- Ruff

Add to `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": true,
      "source.organizeImports": true
    }
  }
}
```

#### PyCharm

1. Open project in PyCharm
2. Set interpreter to `.venv/bin/python`
3. Enable pytest as test runner
4. Configure Ruff as external tool

## Next Steps

- Read the [Quick Start Guide](quickstart.md) to create your first test suite
- Review [Basic Usage](usage.md) for common workflows
- Check [Test Format Reference](../reference/test-format.md) for YAML syntax

## Uninstallation

To completely remove ATP:

```bash
# Remove virtual environment
rm -rf .venv

# Remove lock file
rm uv.lock

# Remove project directory
cd ..
rm -rf atp-platform
```

## Getting Help

- **Documentation**: [docs/](../)
- **Issues**: [GitHub Issues](https://github.com/yourusername/atp-platform/issues)
- **Examples**: [examples/](../../examples/)

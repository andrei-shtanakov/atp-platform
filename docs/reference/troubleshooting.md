# Troubleshooting Guide

Common issues and solutions when working with ATP Platform.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Test Suite Loading Errors](#test-suite-loading-errors)
3. [Validation Errors](#validation-errors)
4. [Variable Substitution Issues](#variable-substitution-issues)
5. [Runtime Errors](#runtime-errors)
6. [Performance Issues](#performance-issues)
7. [Getting Help](#getting-help)

## Installation Issues

### Issue: `uv: command not found`

**Symptom**: Cannot run `uv` commands after installation.

**Solution**:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if not automatic)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
uv --version
```

### Issue: Python version too old

**Symptom**: Error message about Python version.

```
Error: ATP requires Python 3.12 or higher
```

**Solution**:

```bash
# Check current version
python --version

# Install Python 3.12+ (macOS with Homebrew)
brew install python@3.12

# Or use pyenv
pyenv install 3.12
pyenv local 3.12

# Verify
python --version
```

### Issue: Dependencies fail to install

**Symptom**: Errors during `uv sync`.

**Solution**:

```bash
# Clean and reinstall
rm -rf .venv uv.lock
uv sync

# If still failing, try with verbose output
uv sync -v

# Check for conflicting Python installations
which python
```

### Issue: `ImportError` when running code

**Symptom**: Cannot import `atp` modules.

```python
ImportError: No module named 'atp'
```

**Solution**:

```bash
# Always use `uv run` to execute Python code
uv run python script.py

# Or activate virtual environment
source .venv/bin/activate
python script.py
```

## Test Suite Loading Errors

### Issue: File not found

**Symptom**: `FileNotFoundError` when loading test suite.

**Solution**:

```python
from pathlib import Path

# Use absolute path
file_path = Path(__file__).parent / "tests" / "suite.yaml"
suite = loader.load_file(str(file_path))

# Or relative to working directory
suite = loader.load_file("tests/suite.yaml")

# Check file exists
import os
if os.path.exists("tests/suite.yaml"):
    suite = loader.load_file("tests/suite.yaml")
else:
    print("File not found!")
```

### Issue: YAML syntax error

**Symptom**: `ParseError` with YAML syntax issues.

```
ParseError: Invalid YAML syntax at line 10, column 5
```

**Solution**:

1. **Check indentation** - YAML uses spaces, not tabs:
   ```yaml
   # Wrong (tabs)
   tests:
   	- id: "test-001"

   # Correct (spaces)
   tests:
     - id: "test-001"
   ```

2. **Quote strings with special characters**:
   ```yaml
   # May cause issues
   description: This: is a test

   # Better
   description: "This: is a test"
   ```

3. **Use online YAML validator**: https://www.yamllint.com/

### Issue: Empty or invalid file

**Symptom**: Error about missing required fields.

**Solution**:

Check minimum required structure:

```yaml
test_suite: "my_suite"
version: "1.0"

agents:
  - name: "agent"
    type: "http"
    config:
      endpoint: "http://localhost:8000"

tests:
  - id: "test-001"
    name: "Test"
    task:
      description: "Do something"
```

## Validation Errors

### Issue: Duplicate test IDs

**Symptom**: `ValidationError: Duplicate test IDs found`.

**Solution**:

```yaml
# Wrong - duplicate IDs
tests:
  - id: "test-001"
    name: "First test"
  - id: "test-001"  # Duplicate!
    name: "Second test"

# Correct - unique IDs
tests:
  - id: "test-001"
    name: "First test"
  - id: "test-002"
    name: "Second test"
```

### Issue: Scoring weights don't sum to 1.0

**Symptom**: `ValidationError: Scoring weights must sum to approximately 1.0`.

**Solution**:

```yaml
# Wrong - sums to 0.8
scoring:
  quality_weight: 0.4
  completeness_weight: 0.2
  efficiency_weight: 0.1
  cost_weight: 0.1

# Correct - sums to 1.0
scoring:
  quality_weight: 0.4
  completeness_weight: 0.3
  efficiency_weight: 0.2
  cost_weight: 0.1
```

### Issue: Invalid field values

**Symptom**: `ValidationError: Field validation error`.

**Solution**:

Check field constraints:

```yaml
# Wrong - weights must be 0.0-1.0
scoring:
  quality_weight: 1.5  # > 1.0

# Wrong - timeout must be positive
constraints:
  timeout_seconds: -10

# Wrong - missing required field
task:
  # description is required!
  input_data: {}

# Correct
scoring:
  quality_weight: 0.4
constraints:
  timeout_seconds: 300
task:
  description: "Create a file"
```

### Issue: Missing required fields

**Symptom**: `ValidationError: Field required`.

**Solution**:

Ensure all required fields are present:

```yaml
# Minimum required fields
test_suite: "suite_name"  # Required
version: "1.0"            # Required

agents:                   # Required (min 1)
  - name: "agent"         # Required
    type: "http"          # Required
    config:               # Required
      endpoint: "http://localhost:8000"

tests:                    # Required (min 1)
  - id: "test-001"        # Required
    name: "Test"          # Required
    task:                 # Required
      description: "..."  # Required
```

## Variable Substitution Issues

### Issue: Unresolved variable

**Symptom**: `ValidationError: Unresolved variable: ${VAR_NAME}`.

**Solution**:

```python
# Option 1: Provide the variable
loader = TestLoader(env={"VAR_NAME": "value"})
suite = loader.load_file("suite.yaml")

# Option 2: Use default value in YAML
# ${VAR_NAME:default_value}
```

YAML example:

```yaml
agents:
  - name: "agent"
    config:
      # With default - won't fail if API_KEY not set
      api_key: "${API_KEY:default-key}"

      # Without default - will fail if API_KEY not set
      endpoint: "${API_ENDPOINT}"
```

### Issue: Variable not being substituted

**Symptom**: Variable syntax appears in loaded data.

**Solution**:

Check syntax - must be exact:

```yaml
# Wrong
config:
  key: $VAR_NAME           # Missing braces
  key: ${VAR_NAME          # Missing closing brace
  key: ${ VAR_NAME }       # Extra spaces

# Correct
config:
  key: "${VAR_NAME}"
  key: "${VAR_NAME:default}"
```

### Issue: Environment variable not found

**Symptom**: Variables work in shell but not in Python.

**Solution**:

```python
import os

# Check if variable is set
print(os.environ.get("API_KEY"))  # None if not set

# Set explicitly
os.environ["API_KEY"] = "value"

# Or load from .env file
from dotenv import load_dotenv
load_dotenv()

# Then create loader
loader = TestLoader()  # Will use os.environ
```

## Runtime Errors

### Issue: Type errors in loaded data

**Symptom**: `TypeError` or `AttributeError` when accessing suite data.

**Solution**:

```python
# Check types
from atp.loader import TestLoader

loader = TestLoader()
suite = loader.load_file("suite.yaml")

# Safe access with checks
if suite.tests:
    first_test = suite.tests[0]
    if first_test.constraints:
        timeout = first_test.constraints.timeout_seconds

# Use getattr with defaults
timeout = getattr(suite.defaults, "timeout_seconds", 300)
```

### Issue: JSON Schema validation fails

**Symptom**: `ValidationError` from schema validation.

**Solution**:

The schema validator checks structure before Pydantic validation. Common issues:

```yaml
# Wrong - assertions must be array
assertions: "artifact_exists"

# Correct
assertions:
  - type: "artifact_exists"
    config:
      path: "output.txt"

# Wrong - tests must be array
tests: {}

# Correct
tests:
  - id: "test-001"
    name: "Test"
    task:
      description: "..."
```

## Performance Issues

### Issue: Slow test suite loading

**Symptom**: `load_file()` takes long time.

**Solution**:

```python
import time

# Measure load time
start = time.time()
suite = loader.load_file("large_suite.yaml")
print(f"Loaded in {time.time() - start:.2f}s")

# For large suites, consider:
# 1. Split into multiple smaller suites
# 2. Remove unnecessary tests
# 3. Simplify complex nested structures
```

### Issue: Memory usage with large suites

**Symptom**: High memory usage or OOM errors.

**Solution**:

```python
# Load suites one at a time
def process_suites(directory):
    for file in Path(directory).glob("*.yaml"):
        suite = loader.load_file(file)
        # Process suite
        process(suite)
        # Suite is garbage collected when out of scope

# Instead of loading all at once
suites = [loader.load_file(f) for f in files]  # May use lots of memory
```

## Code Quality Issues

### Issue: Ruff formatting errors

**Symptom**: `ruff check` reports formatting issues.

**Solution**:

```bash
# Auto-fix most issues
uv run ruff format .
uv run ruff check . --fix

# Check specific file
uv run ruff check path/to/file.py

# Ignore specific rules (not recommended)
# ruff: noqa: E501
```

### Issue: Type checking errors

**Symptom**: `pyrefly check` reports type errors.

**Solution**:

```bash
# Initialize pyrefly (first time)
pyrefly init

# Run type check
pyrefly check

# Common fixes:
# 1. Add type hints
# 2. Check for None values
# 3. Import correct types
```

Example fixes:

```python
# Before
def process(data):
    return data["key"]

# After
def process(data: dict) -> str:
    return data["key"]

# Before
value = suite.defaults.timeout_seconds

# After (handle None)
value = suite.defaults.timeout_seconds if suite.defaults else 300
```

## Test Failures

### Issue: All tests fail to load

**Symptom**: No test suites can be loaded.

**Solution**:

```bash
# Verify installation
uv run pytest tests/unit/loader -v

# Check imports work
uv run python -c "from atp.loader import TestLoader; print('OK')"

# Reinstall if needed
uv sync --reinstall
```

### Issue: Tests fail in CI but pass locally

**Symptom**: Tests pass on dev machine but fail in CI.

**Solution**:

1. **Check Python version**:
   ```yaml
   # .github/workflows/test.yml
   - uses: actions/setup-python@v4
     with:
       python-version: '3.12'  # Match local version
   ```

2. **Check environment variables**:
   ```yaml
   env:
     API_KEY: ${{ secrets.API_KEY }}
   ```

3. **Check file paths** - use relative paths:
   ```python
   # Not absolute paths
   Path(__file__).parent / "fixtures" / "suite.yaml"
   ```

## Getting Help

### Debugging Tips

1. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Print loaded data**:
   ```python
   from pprint import pprint
   suite = loader.load_file("suite.yaml")
   pprint(suite.model_dump())
   ```

3. **Validate step by step**:
   ```python
   # Check YAML parsing
   from atp.loader.parser import YAMLParser
   parser = YAMLParser()
   data = parser.parse_file("suite.yaml")
   print("YAML parsed OK")

   # Check schema validation
   from atp.loader.schema import validate_test_suite
   validate_test_suite(data)
   print("Schema valid")

   # Check Pydantic validation
   from atp.loader.models import TestSuite
   suite = TestSuite(**data)
   print("Model valid")
   ```

### Common Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| `ParseError` | Invalid YAML syntax | Check indentation, quotes |
| `ValidationError` | Invalid data structure | Check required fields, types |
| `FileNotFoundError` | File doesn't exist | Check path, working directory |
| `ImportError` | Module not found | Use `uv run`, check installation |
| `KeyError` | Missing dict key | Check field names, structure |
| `TypeError` | Wrong type | Check field types, None values |

### Getting Support

1. **Check documentation**:
   - [Installation Guide](../guides/installation.md)
   - [Quick Start](../guides/quickstart.md)
   - [Usage Guide](../guides/usage.md)
   - [Test Format Reference](test-format.md)

2. **Search existing issues**:
   - [GitHub Issues](https://github.com/yourusername/atp-platform-ru/issues)

3. **Create new issue**:
   Include:
   - ATP version
   - Python version
   - Operating system
   - Minimal reproducible example
   - Error message and traceback
   - What you've tried

4. **Example bug report**:
   ```markdown
   ## Environment
   - ATP version: 1.0.0
   - Python: 3.12.1
   - OS: macOS 14.2

   ## Issue
   ValidationError when loading test suite

   ## Reproduction
   ```python
   loader = TestLoader()
   suite = loader.load_file("suite.yaml")
   ```

   ## Error
   ```
   ValidationError: Scoring weights must sum to approximately 1.0
   Got 0.8
   ```

   ## Expected
   Should load successfully with warning

   ## YAML
   ```yaml
   # Attach minimal YAML that reproduces issue
   ```
   ```

## See Also

- [Test Format Reference](test-format.md) - Complete YAML specification
- [Usage Guide](../guides/usage.md) - Common workflows
- [Examples](../../examples/test_suites/) - Working examples
- [GitHub Issues](https://github.com/yourusername/atp-platform-ru/issues) - Known issues

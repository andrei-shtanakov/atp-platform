---
name: generate-tests
description: Generate pytest tests for ATP platform modules. Use when asked to create tests, write tests, or generate test coverage for any ATP component (evaluators, adapters, runner, protocol, loader, etc.). Triggers on "generate tests", "write tests for", "create tests", "add tests", "cover with tests", "напиши тесты", "сгенерируй тесты".
---

# Generate Tests for ATP Platform

Generate pytest tests following the project's established patterns and conventions.

## Invocation

- **With argument:** `/generate-tests path/to/module.py` — reads the module and auto-generates tests
- **Without argument:** asks what to test, then generates

## Workflow

### Step 1: Determine Target

If an argument is provided (file path or module name):
1. Read the target module
2. Read existing tests for that module (if any) in `tests/unit/`, `tests/integration/`
3. Read `tests/conftest.py` for available fixtures
4. Proceed to generation

If no argument:
1. Ask: "What module or component do you want to test?" (suggest recent changes via `git diff --name-only HEAD~5`)
2. Ask: "What scenarios are most important?" (offer: happy path, edge cases, error handling, all)
3. Proceed to generation

### Step 2: Analyze the Module

For each public function/class/method in the target:
- Identify input types, return types, exceptions raised
- Identify dependencies (imports, injected services)
- Identify async vs sync
- Check if it's a Pydantic model, evaluator, adapter, or other ATP component

### Step 3: Generate Tests

Create a test file following these **mandatory conventions**:

#### File Location & Naming
```
Source: atp/evaluators/factuality.py
Tests:  tests/unit/evaluators/test_factuality.py

Source: atp/runner/sandbox.py
Tests:  tests/unit/runner/test_sandbox.py

Source: packages/atp-core/atp/protocol/models.py
Tests:  tests/unit/protocol/test_models.py
```

#### Test Structure Template
```python
"""Tests for atp.{module_path}."""

from __future__ import annotations

import pytest
# import anyio  # for async tests, NOT asyncio

from atp.{module} import {ClassOrFunction}


class Test{ClassName}:
    """Tests for {ClassName}."""

    def test_{method}_returns_expected(self) -> None:
        """Verify {method} returns correct result for valid input."""
        # Arrange
        ...
        # Act
        result = ...
        # Assert
        assert result == expected

    def test_{method}_with_empty_input(self) -> None:
        """Verify {method} handles empty input gracefully."""
        ...

    def test_{method}_raises_on_invalid(self) -> None:
        """Verify {method} raises {Error} for invalid input."""
        with pytest.raises({Error}):
            ...

    @pytest.mark.parametrize("input_val,expected", [
        ("case1", "result1"),
        ("case2", "result2"),
    ])
    def test_{method}_parametrized(
        self, input_val: str, expected: str
    ) -> None:
        """Verify {method} across multiple inputs."""
        result = function(input_val)
        assert result == expected
```

#### Async Test Pattern
```python
import anyio
import pytest

class TestAsyncComponent:
    @pytest.mark.anyio
    async def test_async_method(self) -> None:
        """Verify async method works correctly."""
        result = await component.async_method()
        assert result.status == "success"
```

#### ATP-Specific Patterns

**Evaluator tests:**
```python
class TestMyEvaluator:
    @pytest.fixture
    def evaluator(self) -> MyEvaluator:
        return MyEvaluator(config={})

    @pytest.fixture
    def sample_response(self) -> ATPResponse:
        return ATPResponse(
            status="success",
            artifacts=[{"path": "output.txt", "content": "hello"}],
            metrics={"tokens_used": 100, "steps": 3},
        )

    @pytest.mark.anyio
    async def test_evaluate_passes(
        self,
        evaluator: MyEvaluator,
        sample_test_definition: dict,
        sample_response: ATPResponse,
    ) -> None:
        result = await evaluator.evaluate(
            task=sample_test_definition,
            response=sample_response,
            trace=[],
            assertion=Assertion(type="my_type", config={}),
        )
        assert result.passed
        assert result.score >= 0.8
```

**Protocol model tests:**
```python
class TestATPRequest:
    def test_model_validates_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ATPRequest()  # missing required fields

    def test_model_serialization_roundtrip(self) -> None:
        request = ATPRequest(task="test", constraints={})
        data = request.model_dump()
        restored = ATPRequest.model_validate(data)
        assert restored == request
```

**Adapter tests:**
```python
class TestMyAdapter:
    @pytest.fixture
    def adapter(self) -> MyAdapter:
        return MyAdapter(config={"endpoint": "http://test:8000"})

    @pytest.mark.anyio
    async def test_execute_request(self, adapter: MyAdapter) -> None:
        response = await adapter.execute(ATPRequest(task="hello"))
        assert response.status in ("success", "error")
```

### Available Fixtures (from tests/conftest.py)

Use these fixtures instead of creating new ones when possible:
- `fixtures_dir` — Path to `tests/fixtures/`
- `test_suites_dir` — Path to `tests/fixtures/test_suites/`
- `sample_test_definition` — dict with id, name, task, evaluators
- `sample_env_vars` — dict with API_ENDPOINT, TEST_VAR, API_KEY

### Step 4: Post-Generation

After writing the test file:
1. Run `uv run ruff format {test_file}`
2. Run `uv run ruff check {test_file} --fix`
3. Run `uv run pyrefly check`
4. Run `uv run pytest {test_file} -v` to verify tests pass
5. Report coverage: `uv run pytest {test_file} -v --cov=atp.{module} --cov-report=term-missing`

## Rules

- **Line length:** 88 characters max
- **Type hints:** required on all test methods (return `-> None`)
- **Docstrings:** one-line docstring per test method explaining what it verifies
- **No mocking internals:** prefer integration over unit if the component is simple
- **Float comparison:** use `pytest.approx()` for floating point
- **Parametrize:** use for 3+ similar test cases
- **Naming:** `test_{what}_{condition}` or `test_{what}_{expected_outcome}`
- **Arrange-Act-Assert:** follow AAA pattern, blank lines between sections
- **anyio not asyncio:** always use `@pytest.mark.anyio` for async tests

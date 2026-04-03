# YAML Upload Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /api/v1/suite-definitions/upload` endpoint for uploading YAML test suite files with validation.

**Architecture:** Single new route module `upload.py` with validation logic, `deep_sizeof` helper, and response schemas. Integrates with existing `SuiteDefinition` model (direct DB access, same pattern as `definitions.py`). Config gets `upload_max_size_mb` field.

**Tech Stack:** FastAPI (UploadFile), PyYAML (safe_load), pydantic, SQLAlchemy (async), pytest + anyio

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Create | `packages/atp-dashboard/atp/dashboard/v2/routes/upload.py` | Upload endpoint, validation, deep_sizeof |
| Modify | `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py` | Register upload_router |
| Modify | `packages/atp-dashboard/atp/dashboard/v2/config.py` | Add `upload_max_size_mb` setting |
| Create | `tests/unit/dashboard/test_upload.py` | 10 test scenarios from spec |

---

### Task 1: Add config field and write upload tests

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/config.py`
- Create: `tests/unit/dashboard/test_upload.py`

- [ ] **Step 1: Add `upload_max_size_mb` to DashboardConfig**

In `packages/atp-dashboard/atp/dashboard/v2/config.py`, add after `batch_max_size`:

```python
    # Upload settings
    upload_max_size_mb: int = Field(
        default=1,
        ge=1,
        le=50,
        description="Maximum YAML upload file size in MB",
    )
```

And in `to_dict()`:
```python
            "upload_max_size_mb": self.upload_max_size_mb,
```

- [ ] **Step 2: Write all test scenarios**

Create `tests/unit/dashboard/test_upload.py`:

```python
"""Tests for YAML suite upload endpoint."""

from collections.abc import AsyncGenerator
from io import BytesIO

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app

VALID_YAML = b"""\
test_suite: upload-test
version: "1.0"
tests:
  - id: test-1
    name: Test One
    task:
      description: Do something
    assertions:
      - type: artifact_exists
        config:
          artifact: output.txt
  - id: test-2
    name: Test Two
    task:
      description: Do another thing
    assertions:
      - type: contains
        config:
          text: hello
"""

VALID_YAML_NO_ASSERTIONS = b"""\
test_suite: warnings-test
version: "1.0"
tests:
  - id: test-1
    name: No Assertions
    task:
      description: Do something
"""

INVALID_YAML_SYNTAX = b"""\
test_suite: bad
  version: "1.0"
  tests: [
    unclosed bracket
"""

MISSING_FIELDS_YAML = b"""\
test_suite: missing-fields
tests:
  - name: No ID or task
"""

UNKNOWN_ASSERTION_YAML = b"""\
test_suite: unknown-assertion
version: "1.0"
tests:
  - id: test-1
    name: Bad Assertion
    task:
      description: Do something
    assertions:
      - type: nonexistent_evaluator_xyz
        config: {}
"""


@pytest.fixture
async def test_database() -> AsyncGenerator[Database, None]:
    """Create and configure a test database."""
    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_database(db)
    yield db
    await db.close()
    set_database(None)  # type: ignore[arg-type]


@pytest.fixture
def app(test_database: Database):
    """Create test app."""
    application = create_test_app()

    async def override_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            yield session

    application.dependency_overrides[get_db_session] = override_session
    return application


@pytest.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as c:
        yield c


def _upload(
    client_: AsyncClient,
    content: bytes,
    filename: str = "suite.yaml",
    content_type: str = "application/yaml",
):
    """Helper to POST a file upload."""
    return client_.post(
        "/api/suite-definitions/upload",
        files={"file": (filename, BytesIO(content), content_type)},
    )


class TestYAMLUploadSuccess:
    """Tests for successful uploads."""

    @pytest.mark.anyio
    async def test_valid_yaml_creates_suite(self, client) -> None:
        """Valid YAML → 201 with suite created."""
        resp = await _upload(client, VALID_YAML)
        assert resp.status_code == 201
        data = resp.json()
        assert data["validation"]["valid"] is True
        assert data["validation"]["errors"] == []
        assert data["suite"] is not None
        assert data["suite"]["name"] == "upload-test"
        assert data["filename"] == "suite.yaml"

    @pytest.mark.anyio
    async def test_warnings_only_still_creates(self, client) -> None:
        """YAML with no assertions → 201 with warnings."""
        resp = await _upload(client, VALID_YAML_NO_ASSERTIONS)
        assert resp.status_code == 201
        data = resp.json()
        assert data["validation"]["valid"] is True
        assert len(data["validation"]["warnings"]) > 0
        assert data["suite"] is not None


class TestYAMLUploadValidation:
    """Tests for validation failures."""

    @pytest.mark.anyio
    async def test_invalid_yaml_syntax(self, client) -> None:
        """Invalid YAML syntax → 422."""
        resp = await _upload(client, INVALID_YAML_SYNTAX)
        assert resp.status_code == 422
        data = resp.json()
        assert data["validation"]["valid"] is False
        assert len(data["validation"]["errors"]) > 0
        assert data["suite"] is None

    @pytest.mark.anyio
    async def test_missing_required_fields(self, client) -> None:
        """Missing required fields → 422."""
        resp = await _upload(client, MISSING_FIELDS_YAML)
        assert resp.status_code == 422
        data = resp.json()
        assert data["validation"]["valid"] is False

    @pytest.mark.anyio
    async def test_unknown_assertion_type(self, client) -> None:
        """Unknown assertion type → 422."""
        resp = await _upload(client, UNKNOWN_ASSERTION_YAML)
        assert resp.status_code == 422
        data = resp.json()
        assert data["validation"]["valid"] is False
        assert any(
            "nonexistent_evaluator_xyz" in e
            for e in data["validation"]["errors"]
        )

    @pytest.mark.anyio
    async def test_duplicate_suite_name(self, client) -> None:
        """Duplicate suite name → 409."""
        # First upload succeeds
        resp1 = await _upload(client, VALID_YAML)
        assert resp1.status_code == 201
        # Second upload with same name fails
        resp2 = await _upload(client, VALID_YAML)
        assert resp2.status_code == 409
        data = resp2.json()
        assert data["validation"]["valid"] is False


class TestYAMLUploadFileChecks:
    """Tests for file-level checks."""

    @pytest.mark.anyio
    async def test_wrong_extension(self, client) -> None:
        """Wrong file extension → 400."""
        resp = await _upload(
            client, VALID_YAML, filename="suite.txt"
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["validation"]["valid"] is False

    @pytest.mark.anyio
    async def test_file_too_large(self, client) -> None:
        """File > max size → 413."""
        # Create >1MB content
        huge = b"x" * (2 * 1024 * 1024)
        resp = await _upload(client, huge)
        assert resp.status_code == 413

    @pytest.mark.anyio
    async def test_yaml_alias_bomb(self, client) -> None:
        """YAML alias bomb → 422."""
        bomb = b"""\
a: &a [1,2,3,4,5,6,7,8,9,10]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c,*c]
e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d,*d]
f: &f [*e,*e,*e,*e,*e,*e,*e,*e,*e,*e]
g: &g [*f,*f,*f,*f,*f,*f,*f,*f,*f,*f]
h: &h [*g,*g,*g,*g,*g,*g,*g,*g,*g,*g]
"""
        resp = await _upload(client, bomb)
        assert resp.status_code == 422
        data = resp.json()
        assert data["validation"]["valid"] is False
```

- [ ] **Step 3: Run tests to see them fail**

Run: `uv run python -m pytest tests/unit/dashboard/test_upload.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Run ruff on test file**

Run: `uv run ruff format tests/unit/dashboard/test_upload.py && uv run ruff check tests/unit/dashboard/test_upload.py --fix`

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/config.py tests/unit/dashboard/test_upload.py
git commit -m "test(dashboard): add YAML upload test scenarios and config field"
```

---

### Task 2: Implement upload endpoint

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/routes/upload.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`

- [ ] **Step 1: Implement upload.py**

Create `packages/atp-dashboard/atp/dashboard/v2/routes/upload.py`:

```python
"""YAML suite upload endpoint.

Accepts multipart/form-data YAML file, validates against TestSuite schema
and evaluator registry, creates a SuiteDefinition in the database.
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated, Any

import yaml
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel, ValidationError
from sqlalchemy import select

from atp.dashboard.models import SuiteDefinition
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.schemas import SuiteDefinitionResponse
from atp.dashboard.v2.config import get_config
from atp.dashboard.v2.dependencies import DBSession
from atp.dashboard.v2.routes.definitions import (
    _build_suite_definition_response,
)
from atp.loader.models import TestSuite

logger = logging.getLogger("atp.dashboard")

router = APIRouter(
    prefix="/suite-definitions", tags=["suite-definitions"]
)

ALLOWED_EXTENSIONS = {".yaml", ".yml"}
ALLOWED_CONTENT_TYPES = {
    "application/yaml",
    "text/yaml",
    "text/plain",
    "application/x-yaml",
    "application/octet-stream",
}
MAX_PARSED_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


class ValidationReport(BaseModel):
    """Validation results for an uploaded YAML file."""

    valid: bool
    errors: list[str]
    warnings: list[str]


class UploadResponse(BaseModel):
    """Response from the YAML upload endpoint."""

    suite: SuiteDefinitionResponse | None
    validation: ValidationReport
    filename: str


def _deep_sizeof(obj: Any, seen: set[int] | None = None) -> int:
    """Recursively calculate the memory size of a parsed object.

    Unlike sys.getsizeof, this counts nested containers.
    """
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(
            _deep_sizeof(k, seen) + _deep_sizeof(v, seen)
            for k, v in obj.items()
        )
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(_deep_sizeof(i, seen) for i in obj)
    return size


def _get_known_assertion_types() -> set[str]:
    """Get known assertion/evaluator types from the registry."""
    try:
        from atp.evaluators import get_evaluator_registry

        registry = get_evaluator_registry()
        return set(registry.list_evaluators())
    except Exception:
        # If registry is not available, skip assertion validation
        return set()


def _validate_yaml(
    raw: bytes, filename: str
) -> tuple[dict[str, Any] | None, ValidationReport]:
    """Parse and validate YAML content.

    Returns:
        Tuple of (parsed data or None, validation report).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Step 2: YAML parse
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return None, ValidationReport(
            valid=False,
            errors=[f"YAML parse error: {exc}"],
            warnings=[],
        )

    if not isinstance(data, dict):
        return None, ValidationReport(
            valid=False,
            errors=["YAML root must be a mapping"],
            warnings=[],
        )

    # Memory check for alias bombs
    parsed_size = _deep_sizeof(data)
    if parsed_size > MAX_PARSED_SIZE_BYTES:
        return None, ValidationReport(
            valid=False,
            errors=[
                f"Parsed YAML exceeds memory limit "
                f"({parsed_size} bytes > {MAX_PARSED_SIZE_BYTES})"
            ],
            warnings=[],
        )

    # Step 3: Schema validation
    try:
        suite = TestSuite.model_validate(data)
    except ValidationError as exc:
        for err in exc.errors():
            loc = " → ".join(str(l) for l in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return None, ValidationReport(
            valid=False, errors=errors, warnings=[]
        )

    # Step 4: Assertion validation
    known_types = _get_known_assertion_types()
    if known_types:
        for test in suite.tests:
            for assertion in test.assertions:
                if assertion.type not in known_types:
                    errors.append(
                        f"Test '{test.id}': unknown assertion "
                        f"type '{assertion.type}'"
                    )

    if errors:
        return None, ValidationReport(
            valid=False, errors=errors, warnings=warnings
        )

    # Step 5: Warnings
    for test in suite.tests:
        if not test.assertions:
            warnings.append(
                f"Test '{test.id}' has no assertions"
            )
        if not test.task.description:
            warnings.append(
                f"Test '{test.id}' has empty description"
            )

    return data, ValidationReport(
        valid=True, errors=[], warnings=warnings
    )


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_yaml_suite(
    file: UploadFile,
    session: DBSession,
    _: Annotated[
        None, Depends(require_permission(Permission.SUITES_WRITE))
    ],
) -> UploadResponse:
    """Upload a YAML test suite file.

    Parses, validates, and creates a suite definition.
    Requires SUITES_WRITE permission.
    """
    config = get_config()
    filename = file.filename or "unknown.yaml"
    max_bytes = config.upload_max_size_mb * 1024 * 1024

    # Step 1a: Extension check
    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=UploadResponse(
                suite=None,
                validation=ValidationReport(
                    valid=False,
                    errors=[
                        f"Invalid file extension '{ext}'. "
                        f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                    ],
                    warnings=[],
                ),
                filename=filename,
            ).model_dump(),
        )

    # Step 1b: Content-Type check
    ct = (file.content_type or "").lower()
    if ct and ct not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=UploadResponse(
                suite=None,
                validation=ValidationReport(
                    valid=False,
                    errors=[
                        f"Invalid content type '{ct}'. "
                        f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
                    ],
                    warnings=[],
                ),
                filename=filename,
            ).model_dump(),
        )

    # Step 1c: Size check
    raw = await file.read()
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=UploadResponse(
                suite=None,
                validation=ValidationReport(
                    valid=False,
                    errors=[
                        f"File too large ({len(raw)} bytes). "
                        f"Maximum: {max_bytes} bytes"
                    ],
                    warnings=[],
                ),
                filename=filename,
            ).model_dump(),
        )

    logger.info(
        "Upload: %s, %dB",
        filename,
        len(raw),
    )

    # Steps 2-5: Validate
    parsed_data, report = _validate_yaml(raw, filename)

    if not report.valid:
        logger.warning(
            "Upload rejected: %s, errors=%d",
            filename,
            len(report.errors),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=UploadResponse(
                suite=None,
                validation=report,
                filename=filename,
            ).model_dump(),
        )

    assert parsed_data is not None

    # Check for duplicate name
    suite_name = parsed_data.get("test_suite", filename)
    stmt = select(SuiteDefinition).where(
        SuiteDefinition.name == suite_name
    )
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=UploadResponse(
                suite=None,
                validation=ValidationReport(
                    valid=False,
                    errors=[
                        f"Suite definition '{suite_name}' "
                        f"already exists (id={existing.id})"
                    ],
                    warnings=[],
                ),
                filename=filename,
            ).model_dump(),
        )

    # Create suite definition
    suite_model = TestSuite.model_validate(parsed_data)
    tests_list = [
        t.model_dump(mode="json") for t in suite_model.tests
    ]

    suite_def = SuiteDefinition(
        name=suite_name,
        version=parsed_data.get("version", "1.0"),
        description=parsed_data.get("description"),
        defaults_json=parsed_data.get("defaults", {}),
        agents_json=parsed_data.get("agents", []),
        tests_json=tests_list,
    )
    session.add(suite_def)
    await session.commit()
    await session.refresh(suite_def)

    logger.info(
        "Suite created from upload: %s, id=%d",
        suite_name,
        suite_def.id,
    )

    return UploadResponse(
        suite=_build_suite_definition_response(suite_def),
        validation=report,
        filename=filename,
    )
```

- [ ] **Step 2: Register upload router in __init__.py**

In `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`, add import:

```python
from atp.dashboard.v2.routes.upload import (
    router as upload_router,
)
```

Add after `router.include_router(definitions_router)`:

```python
router.include_router(upload_router)
```

Add `"upload_router"` to `__all__`.

- [ ] **Step 3: Run tests**

Run: `uv run python -m pytest tests/unit/dashboard/test_upload.py -v`
Expected: Most PASS. Some may need adjustments (auth mocking, response format from HTTPException).

- [ ] **Step 4: Fix any test failures**

Adjust tests if needed — common issues:
- HTTPException `detail` is a dict, not direct UploadResponse — tests may need to parse `resp.json()["detail"]`
- Auth dependency may need mocking if not auto-disabled in test app

- [ ] **Step 5: Run ruff + pyrefly**

Run: `uv run ruff format packages/atp-dashboard/ tests/unit/dashboard/test_upload.py && uv run ruff check packages/atp-dashboard/ tests/unit/dashboard/test_upload.py --fix && uv run pyrefly check`

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/upload.py packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py tests/unit/dashboard/test_upload.py
git commit -m "feat(dashboard): add YAML suite upload endpoint with validation"
```

---

### Task 3: Integration verification

- [ ] **Step 1: Run full dashboard test suite**

Run: `uv run python -m pytest tests/unit/dashboard/ -v -x -q`
Expected: No regressions

- [ ] **Step 2: Verify endpoint is registered**

Run: `uv run python -c "from atp.dashboard.v2.routes import router; routes = [r.path for r in router.routes]; print([r for r in routes if 'upload' in r])"`
Expected: `['/suite-definitions/upload']`

- [ ] **Step 3: Run ruff + pyrefly on full project**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 4: Commit any formatting changes**

```bash
git add -u && git diff --cached --stat
# Only commit if changes
git commit -m "style: format and lint fixes for YAML upload"
```

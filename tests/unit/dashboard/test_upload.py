"""Tests for YAML suite upload endpoint."""

import os
from collections.abc import AsyncGenerator, Generator
from io import BytesIO

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base
from atp.dashboard.v2.config import get_config
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
def disable_auth() -> Generator[None, None, None]:
    """Disable authentication for upload tests."""
    old_value = os.environ.get("ATP_DISABLE_AUTH")
    os.environ["ATP_DISABLE_AUTH"] = "true"
    get_config.cache_clear()
    yield
    get_config.cache_clear()
    if old_value is None:
        os.environ.pop("ATP_DISABLE_AUTH", None)
    else:
        os.environ["ATP_DISABLE_AUTH"] = old_value


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
def app(test_database: Database, disable_auth: None):
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
    async with AsyncClient(transport=transport, base_url="http://test") as c:
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
        data = resp.json()["detail"]
        assert data["validation"]["valid"] is False
        assert len(data["validation"]["errors"]) > 0
        assert data["suite"] is None

    @pytest.mark.anyio
    async def test_missing_required_fields(self, client) -> None:
        """Missing required fields → 422."""
        resp = await _upload(client, MISSING_FIELDS_YAML)
        assert resp.status_code == 422
        data = resp.json()["detail"]
        assert data["validation"]["valid"] is False

    @pytest.mark.anyio
    async def test_unknown_assertion_type(self, client) -> None:
        """Unknown assertion type → 422."""
        resp = await _upload(client, UNKNOWN_ASSERTION_YAML)
        assert resp.status_code == 422
        data = resp.json()["detail"]
        assert data["validation"]["valid"] is False
        assert any(
            "nonexistent_evaluator_xyz" in e for e in data["validation"]["errors"]
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
        data = resp2.json()["detail"]
        assert data["validation"]["valid"] is False


class TestYAMLUploadFileChecks:
    """Tests for file-level checks."""

    @pytest.mark.anyio
    async def test_wrong_extension(self, client) -> None:
        """Wrong file extension → 400."""
        resp = await _upload(client, VALID_YAML, filename="suite.txt")
        assert resp.status_code == 400
        data = resp.json()["detail"]
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
        data = resp.json()["detail"]
        assert data["validation"]["valid"] is False

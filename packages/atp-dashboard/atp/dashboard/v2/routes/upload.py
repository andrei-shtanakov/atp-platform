"""YAML suite upload endpoint.

Accepts multipart/form-data YAML file, validates against TestSuite schema
and evaluator registry, creates a SuiteDefinition in the database.
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated, Any

import yaml
from atp.loader.models import TestSuite
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
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
from atp.dashboard.v2.rate_limit import limiter
from atp.dashboard.v2.routes.definitions import (
    _build_suite_definition_response,
)

logger = logging.getLogger("atp.dashboard")

router = APIRouter(prefix="/suite-definitions", tags=["suite-definitions"])

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

    Args:
        obj: Object to measure.
        seen: Set of already-visited object IDs to prevent cycles.

    Returns:
        Total estimated memory size in bytes.
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
            _deep_sizeof(k, seen) + _deep_sizeof(v, seen) for k, v in obj.items()
        )
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(_deep_sizeof(i, seen) for i in obj)
    return size


def _get_known_assertion_types() -> set[str]:
    """Get known assertion types from the evaluator registry.

    Returns:
        Set of known assertion type strings, or empty set if unavailable.
    """
    try:
        from atp.evaluators import EvaluatorRegistry

        registry = EvaluatorRegistry()
        return set(registry.list_assertion_types())
    except Exception:
        # If registry is not available, skip assertion validation
        return set()


def _validate_yaml(
    raw: bytes,
    filename: str,
) -> tuple[dict[str, Any] | None, ValidationReport]:
    """Parse and validate YAML content.

    Args:
        raw: Raw YAML bytes.
        filename: Original filename (for error messages).

    Returns:
        Tuple of (parsed data or None, validation report).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # YAML parse
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

    # Schema validation
    try:
        suite = TestSuite.model_validate(data)
    except ValidationError as exc:
        for err in exc.errors():
            loc = " → ".join(str(la) for la in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return None, ValidationReport(valid=False, errors=errors, warnings=[])

    # Assertion type validation
    known_types = _get_known_assertion_types()
    if known_types:
        for test in suite.tests:
            for assertion in test.assertions:
                if assertion.type not in known_types:
                    errors.append(
                        f"Test '{test.id}': unknown assertion type '{assertion.type}'"
                    )

    if errors:
        return None, ValidationReport(valid=False, errors=errors, warnings=warnings)

    # Warnings
    for test in suite.tests:
        if not test.assertions:
            warnings.append(f"Test '{test.id}' has no assertions")

    return data, ValidationReport(valid=True, errors=[], warnings=warnings)


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
@limiter.limit("10/minute")
async def upload_yaml_suite(
    request: Request,
    file: UploadFile,
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.SUITES_WRITE))],
) -> UploadResponse:
    """Upload a YAML test suite file.

    Parses, validates, and creates a suite definition.
    Requires SUITES_WRITE permission.

    Args:
        file: The uploaded YAML file.
        session: Database session.

    Returns:
        Upload response with validation report and created suite.

    Raises:
        HTTPException: 400 for bad extension/content-type,
            413 for oversized file, 422 for invalid YAML/schema,
            409 for duplicate suite name.
    """
    config = get_config()
    filename = file.filename or "unknown.yaml"
    max_bytes = config.upload_max_size_mb * 1024 * 1024

    # Extension check
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

    # Content-Type check
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

    # Size check
    raw = await file.read()
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=UploadResponse(
                suite=None,
                validation=ValidationReport(
                    valid=False,
                    errors=[
                        f"File too large ({len(raw)} bytes). Maximum: {max_bytes} bytes"
                    ],
                    warnings=[],
                ),
                filename=filename,
            ).model_dump(),
        )

    logger.info("Upload: %s, %dB", filename, len(raw))

    # Validate YAML
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
    stmt = select(SuiteDefinition).where(SuiteDefinition.name == suite_name)
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
        t.model_dump(mode="json", exclude_none=False) for t in suite_model.tests
    ]
    # Normalize constraints: remove None values so dashboard schemas validate cleanly
    for test_dict in tests_list:
        if "constraints" in test_dict and isinstance(test_dict["constraints"], dict):
            test_dict["constraints"] = {
                k: v for k, v in test_dict["constraints"].items() if v is not None
            }

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

    logger.info("Suite created from upload: %s, id=%d", suite_name, suite_def.id)

    return UploadResponse(
        suite=_build_suite_definition_response(suite_def),
        validation=report,
        filename=filename,
    )

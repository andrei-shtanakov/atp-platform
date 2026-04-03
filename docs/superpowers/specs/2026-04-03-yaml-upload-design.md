# YAML Upload Endpoint

**Date:** 2026-04-03
**Status:** Approved (rev.2)

## Overview

Add `POST /api/v1/suite-definitions/upload` endpoint that accepts a YAML test suite file via multipart/form-data, validates it, and creates a suite definition in the database.

## Endpoint

`POST /api/v1/suite-definitions/upload`

- **Content-Type:** `multipart/form-data`
- **Auth:** JWT required (same as other API endpoints)
- **Input:** Single file field `file` (`.yaml` or `.yml`, max 1MB configurable via `ATP_UPLOAD_MAX_SIZE_MB` env var, default 1)
- **Success (201):** `UploadResponse` with suite definition + validation report
- **Validation failure (422):** `UploadResponse` with errors, no suite created
- **File too large (413):** `UploadResponse` with error
- **Wrong file type (400):** `UploadResponse` with error
- **Duplicate name (409):** `UploadResponse` with error

## Request

```
POST /api/v1/suite-definitions/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: @my-suite.yaml
```

## Response Schema

```python
class ValidationReport(BaseModel):
    valid: bool
    errors: list[str]    # Parse errors, missing fields, unknown assertion types
    warnings: list[str]  # Non-fatal issues (no assertions, empty descriptions)

class UploadResponse(BaseModel):
    suite: SuiteDefinitionResponse | None  # None when valid=False
    validation: ValidationReport
    filename: str  # Original uploaded filename
```

All error responses (400, 413, 409, 422) use the same `UploadResponse` schema with `suite=None` for API consistency.

## Validation Steps

1. **File check:** Extension must be `.yaml` or `.yml`. Content-Type must be one of `{"application/yaml", "text/yaml", "text/plain", "application/x-yaml", "application/octet-stream"}`. Size <= max.
2. **YAML parse:** `yaml.safe_load()` with memory protection — wrap in `try/except` and limit parsed output size. Use recursive `deep_sizeof()` helper (not bare `sys.getsizeof` which doesn't count nested objects). Reject if parsed structure exceeds 10MB. This mitigates YAML alias bombs where 1MB YAML can expand to hundreds of MB.
3. **Schema validation:** `TestSuite.model_validate(data)` — catch `ValidationError`, report each error with field path.
4. **Assertion validation:** For each test, check that assertion types exist in evaluator registry. Unknown types → error.
5. **Warnings:** Tests with empty assertions list, empty descriptions.

If any step produces errors, return 422 immediately. Warnings alone don't block creation.

## Duplicate Handling

If a suite definition with the same `test_suite` name already exists: return **409 Conflict** with error message `"Suite definition '{name}' already exists (id={id})"`. No upsert, no silent duplicates. User must rename or delete the existing one first.

## Transactionality

Suite creation is atomic. If `SuiteDefinitionService.create()` fails after validation passes, the transaction rolls back and the endpoint returns 500 with no partial state. The handler wraps creation in the existing DB session transaction.

## Logging

Log via `logging.getLogger("atp.dashboard")`:

| Event | Level | Content |
|-------|-------|---------|
| Upload received | INFO | `Upload: {filename}, {size}B, user={user_id}` |
| Validation failed | WARNING | `Upload rejected: {filename}, errors={count}` |
| Suite created | INFO | `Suite created from upload: {name}, id={id}, user={user_id}` |

## Files

| Action | File | Purpose |
|--------|------|---------|
| Create | `packages/atp-dashboard/atp/dashboard/v2/routes/upload.py` | Upload endpoint + validation logic |
| Modify | `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py` | Register upload router |
| Modify | `packages/atp-dashboard/atp/dashboard/v2/config.py` | Add `upload_max_size_mb` setting |
| Create | `tests/unit/dashboard/test_upload.py` | Endpoint tests |

## Test Scenarios

Minimum coverage for `tests/unit/dashboard/test_upload.py`:

- Valid YAML → 201 + suite created
- Invalid YAML syntax → 422
- Schema validation error (missing required fields) → 422
- Unknown assertion type → 422
- Warnings only (no assertions) → 201 + warnings present
- Duplicate suite name → 409
- File too large → 413
- Wrong file extension (.txt) → 400
- YAML alias bomb (expands beyond 10MB) → 422
- Missing auth token → 401

## Scope

- Single file upload only (no batch/zip)
- YAML only (no JSON upload — that's already covered by existing JSON API)
- No file storage — YAML is parsed and stored as JSON in the suite definition
- Uses existing `SuiteDefinition` model and service layer

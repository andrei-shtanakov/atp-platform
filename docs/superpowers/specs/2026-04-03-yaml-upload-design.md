# YAML Upload Endpoint

**Date:** 2026-04-03
**Status:** Approved

## Overview

Add `POST /api/v1/suite-definitions/upload` endpoint that accepts a YAML test suite file via multipart/form-data, validates it, and creates a suite definition in the database.

## Endpoint

`POST /api/v1/suite-definitions/upload`

- **Content-Type:** `multipart/form-data`
- **Auth:** JWT required (same as other API endpoints)
- **Input:** Single file field `file` (`.yaml` or `.yml`, max 1MB configurable via `ATP_UPLOAD_MAX_SIZE_MB` env var, default 1)
- **Success (201):** `UploadResponse` with suite definition + validation report
- **Validation failure (422):** `UploadResponse` with errors, no suite created
- **File too large (413):** Request Entity Too Large
- **Wrong file type (400):** Only .yaml/.yml accepted

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
```

## Validation Steps

1. **File check:** Extension must be `.yaml` or `.yml`, size <= max
2. **YAML parse:** `yaml.safe_load()` — catch `yaml.YAMLError`
3. **Schema validation:** `TestSuite.model_validate(data)` — catch `ValidationError`, report each error
4. **Assertion validation:** For each test, check that assertion types exist in evaluator registry. Unknown types → error.
5. **Warnings:** Tests with empty assertions list, empty descriptions

If step 2 or 3 fails, return 422 immediately with errors. If step 4 finds errors, still return 422. Warnings alone don't block creation.

## Suite Definition Creation

On successful validation, create a `SuiteDefinition` record in the database using the existing service layer (`SuiteDefinitionService.create()`). The suite name comes from the YAML `test_suite` field.

## Files

| Action | File | Purpose |
|--------|------|---------|
| Create | `packages/atp-dashboard/atp/dashboard/v2/routes/upload.py` | Upload endpoint + validation logic |
| Modify | `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py` | Register upload router |
| Modify | `packages/atp-dashboard/atp/dashboard/v2/config.py` | Add `upload_max_size_mb` setting |
| Create | `tests/unit/dashboard/test_upload.py` | Endpoint tests |

## Scope

- Single file upload only (no batch/zip)
- YAML only (no JSON upload — that's already covered by existing JSON API)
- No file storage — YAML is parsed and stored as JSON in the suite definition
- Uses existing `SuiteDefinition` model and service layer

# open-prose Receipts as Evaluation Input — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor the `openprose.receipt.v1` / `openprose.compile-ir.v1` contracts + fixtures into atp-platform and ship a stdlib receipts verifier (`canonical form → hash chain → ledger_head anchor`) exposed as the deterministic checker `receipt_chain`.

**Architecture:** New private library `atp/evaluators/openprose_receipts/` (canonical.py / reader.py / checker.py), registered as the 4th checker in the closed registry. Contracts + fixtures vendored under `method/contract/` per the RD-007 pattern. One dispatch-layer change: `AgentEvalCaseEvaluator._evaluate_critical()` injects `_case_dir` into the checker config (confinement root). Spec: `docs/superpowers/specs/2026-07-16-openprose-receipts-evaluation-input-design.md` — read it first; it is the authority on semantics.

**Tech Stack:** Python 3.12, pydantic (result models), stdlib `json`+`hashlib` (verification logic), pytest. **Zero new dependencies** — no jsonschema in the reader.

## Global Constraints

- Package management: ONLY `uv` (`uv run pytest`, `uv run ruff`, `uv run pyrefly check`). Never pip.
- Type hints on all code; line length 88; `uv run ruff format .` + `uv run ruff check .` + `uv run pyrefly check` must pass after every task (pre-commit runs them).
- Branch: `feat/openprose-receipts-input` (already exists, spec committed). Commit per task; do NOT push or merge to `main` — PR at the end, merge is human-gated.
- Issue messages must contain the upstream `expected.json` substrings verbatim: `"prev broken"`, `"content_hash mismatch"`, `"ledger_head"`, `"torn write"`.
- v1 is receipts-only: the IR contract and IR fixtures are vendored but NO IR reader/validation is implemented or tested.
- The open-prose sibling (`../open-prose/`) is read-only reference: copy FROM it during Task 1, never reference its paths from code or tests.

---

### Task 1: Vendor contracts + fixtures + PROVENANCE

**Files:**
- Create: `method/contract/openprose/PROVENANCE.md`
- Create: `method/contract/openprose/receipt.md` (copy)
- Create: `method/contract/openprose/ir.md` (copy)
- Create: `method/contract/fixtures/openprose/runs/<4 run dirs>` (copies)
- Create: `method/contract/fixtures/openprose/broken/{broken-chain,tampered-content,torn-write,truncated-ledger}/` (copies)
- Create: `method/contract/fixtures/openprose/ir/{stale-source,tampered-ir,unknown-agent}/` (copies)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: fixture tree consumed by Tasks 5–6 tests at `method/contract/fixtures/openprose/`.

- [ ] **Step 1: Copy the vendored files**

```bash
cd /Users/Andrei_Shtanakov/labs/all_ai_orchestrators/atp-platform
OP=../open-prose
mkdir -p method/contract/openprose method/contract/fixtures/openprose/broken
cp "$OP/contracts/receipt.md" method/contract/openprose/receipt.md
cp "$OP/contracts/ir.md" method/contract/openprose/ir.md
cp -R "$OP/skills/prose/examples/runs" method/contract/fixtures/openprose/runs
for f in broken-chain tampered-content torn-write truncated-ledger; do
  cp -R "$OP/tests/fixtures/runs/$f" "method/contract/fixtures/openprose/broken/$f"
done
cp -R "$OP/tests/fixtures/ir" method/contract/fixtures/openprose/ir
rm method/contract/fixtures/openprose/ir/generate.py
```

`generate.py` is removed: it imports `openprose_tools`, which we do not vendor. (`tests/fixtures/runs/generate.py` lives outside the four copied dirs, so it never arrives.)

- [ ] **Step 2: Verify the copy**

Run: `ls method/contract/fixtures/openprose/runs && ls method/contract/fixtures/openprose/broken && find method/contract/fixtures/openprose -name generate.py`
Expected: 4 run dirs; 4 broken dirs; no `generate.py` anywhere.

- [ ] **Step 3: Write PROVENANCE.md**

First capture the hashes: `find method/contract/openprose method/contract/fixtures/openprose -type f ! -name PROVENANCE.md -exec shasum -a 256 {} \; | sort -k2`

Create `method/contract/openprose/PROVENANCE.md` (paste the real hash output into the last section):

```markdown
# Vendored open-prose contracts & fixtures — provenance

- **Source repo:** open-prose (polyrepo sibling; dev-only reference, never a runtime path)
- **Source commit:** a0395cdb004fb1782dd45e145f24f948e61043d1
- **Vendored:** 2026-07-16
- **Pattern:** pinned copy inside the consumer (workspace rule; same as the RD-007
  learning-event vendoring in `method/contract/learning-event-v1.schema.json`)
- **Contents:** `contracts/{receipt.md,ir.md}`; corpus runs from
  `skills/prose/examples/runs/`; broken run fixtures from `tests/fixtures/runs/`
  (each with its upstream `expected.json`); IR fixtures from `tests/fixtures/ir/`
  (vendored for the future IR reader — NOT consumed by any v1 test).
- **Excluded:** fixture `generate.py` scripts (they import `openprose_tools`,
  which is not vendored).
- **Contract policy:** append-frozen — unknown fields are ignored (but hashed as
  received), unknown `v` values are refused.
- **Refresh:** manual — re-copy from a new upstream commit, re-run
  `find … -exec shasum -a 256 {} \; | sort -k2`, update this file. Automated
  byte-conformance is a follow-up (same posture as RD-007 M2).

## sha256 (at vendoring time)

<paste the shasum output here>
```

- [ ] **Step 4: Commit**

```bash
git add method/contract/openprose method/contract/fixtures/openprose
git commit -m "feat: vendor open-prose receipt/ir contracts + fixtures (pinned @ a0395cd)"
```

---

### Task 2: `canonical.py` — canonical form + content hash

**Files:**
- Create: `atp/evaluators/openprose_receipts/__init__.py`
- Create: `atp/evaluators/openprose_receipts/canonical.py`
- Test: `tests/unit/evaluators/test_openprose_receipts.py` (new file, grows in Tasks 3–5)

**Interfaces:**
- Consumes: nothing.
- Produces: `canonical_json(value: Any) -> bytes` (raises `ValueError` on float/NaN/Infinity, `TypeError` on non-JSON types / non-str keys); `content_hash(receipt: dict[str, Any]) -> str` returning `"sha256:<64 hex>"` over the canonical form sans `content_hash`. Used by Tasks 3–5 and by test helpers.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/evaluators/test_openprose_receipts.py`:

```python
"""Unit tests for openprose_receipts: canonical form, ledger reader, checker.

Contract: method/contract/openprose/receipt.md (openprose.receipt.v1).
"""

import pytest

from atp.evaluators.openprose_receipts.canonical import canonical_json, content_hash


class TestCanonicalJson:
    def test_sorts_object_keys(self) -> None:
        assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'

    def test_no_whitespace_nested_containers(self) -> None:
        value = {"outer": {"z": [1, 2], "a": None}, "flag": True}
        assert canonical_json(value) == b'{"flag":true,"outer":{"a":null,"z":[1,2]}}'

    def test_utf8_not_ascii_escaped(self) -> None:
        assert canonical_json({"k": "приём"}) == '{"k":"приём"}'.encode()

    def test_string_json_escapes_preserved(self) -> None:
        assert canonical_json('a"b\n') == b'"a\\"b\\n"'

    def test_rejects_float(self) -> None:
        with pytest.raises(ValueError):
            canonical_json({"tokens": 1.5})

    def test_rejects_nan_and_infinity(self) -> None:
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError):
                canonical_json(bad)

    def test_bool_is_not_an_int(self) -> None:
        assert canonical_json(True) == b"true"

    def test_rejects_non_string_keys(self) -> None:
        with pytest.raises(TypeError):
            canonical_json({1: "x"})


class TestContentHash:
    def test_excludes_content_hash_field_itself(self) -> None:
        receipt = {"v": "openprose.receipt.v1", "prev": None}
        h = content_hash(receipt)
        assert h == content_hash({**receipt, "content_hash": "sha256:bogus"})
        assert h.startswith("sha256:")
        assert len(h) == 71  # "sha256:" + 64 hex

    def test_unknown_fields_participate_in_hash(self) -> None:
        base = {"v": "openprose.receipt.v1", "prev": None}
        assert content_hash(base) != content_hash({**base, "future_field": 1})

    def test_hash_covers_prev(self) -> None:
        base = {"v": "openprose.receipt.v1", "prev": None}
        assert content_hash(base) != content_hash({**base, "prev": "sha256:aa"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/evaluators/test_openprose_receipts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'atp.evaluators.openprose_receipts'`

- [ ] **Step 3: Implement**

Create `atp/evaluators/openprose_receipts/__init__.py`:

```python
"""Verification of open-prose receipts.jsonl ledgers (openprose.receipt.v1).

Vendored contract: method/contract/openprose/receipt.md. v1 is receipts-only;
the IR contract (ir.md) is vendored but has no reader yet.
"""
```

Create `atp/evaluators/openprose_receipts/canonical.py`:

```python
"""Canonical JSON + content hashing for openprose.receipt.v1.

Implements the contract's canonical form (method/contract/openprose/receipt.md):
object keys sorted, no whitespace, UTF-8 strings not ASCII-escaped, integers
only. Written against the contract text; open-prose's reference canonical.py
was consulted, not copied. stdlib-only.
"""

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> bytes:
    """Serialize ``value`` to the contract's canonical byte form.

    Raises ValueError on floats/NaN/Infinity (invalid in a receipt — they break
    hash portability) and TypeError on non-JSON types or non-string keys.
    """
    # bool is a subclass of int — it must be checked first.
    if isinstance(value, bool):
        return b"true" if value else b"false"
    if value is None:
        return b"null"
    if isinstance(value, int):
        return str(value).encode()
    if isinstance(value, float):
        raise ValueError("floats are invalid in a canonical receipt")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False).encode()
    if isinstance(value, list):
        return b"[" + b",".join(canonical_json(v) for v in value) + b"]"
    if isinstance(value, dict):
        parts: list[bytes] = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("receipt object keys must be strings")
            parts.append(canonical_json(key) + b":" + canonical_json(value[key]))
        return b"{" + b",".join(parts) + b"}"
    raise TypeError(f"unsupported type in receipt: {type(value).__name__}")


def content_hash(receipt: dict[str, Any]) -> str:
    """Chain identity: sha256 over the canonical form sans ``content_hash``.

    The hash covers every other field — including ``prev`` (each receipt
    commits to the whole chain behind it) and unknown fields (append-frozen:
    ignored semantically, hashed as received).
    """
    body = {k: v for k, v in receipt.items() if k != "content_hash"}
    return "sha256:" + hashlib.sha256(canonical_json(body)).hexdigest()
```

Note on key order: Python's `sorted()` compares str by code point, which for
UTF-8 coincides with byte order (UTF-8 preserves code-point ordering) — this
satisfies the contract's "byte order" rule without encoding keys first.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/evaluators/test_openprose_receipts.py -v`
Expected: all PASS.

- [ ] **Step 5: Format, lint, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add atp/evaluators/openprose_receipts tests/unit/evaluators/test_openprose_receipts.py
git commit -m "feat: openprose_receipts canonical form + content hash (stdlib)"
```

---

### Task 3: `reader.py` — models + `load_ledger`

**Files:**
- Create: `atp/evaluators/openprose_receipts/reader.py`
- Test: `tests/unit/evaluators/test_openprose_receipts.py` (append)

**Interfaces:**
- Consumes: `content_hash` from Task 2 (test helpers).
- Produces: `Issue(code: str, line_no: int | None, message: str)`, `LoadedLedger(receipts: list[dict], errors: list[Issue], warnings: list[Issue])`, `VerifyResult(ok: bool, receipt_count: int, errors: list[Issue], warnings: list[Issue])` (pydantic models); `load_ledger(path: Path) -> LoadedLedger`. Constants `RECEIPT_VERSION`, `RUN_VERSION`. Task 4 adds `verify_run` to this module; Task 5–6 consume `verify_run`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/evaluators/test_openprose_receipts.py` (imports at top of file; helpers at module level — Tasks 4–5 reuse them):

```python
import json
from pathlib import Path
from typing import Any

from atp.evaluators.openprose_receipts.reader import (
    LoadedLedger,
    VerifyResult,
    load_ledger,
    verify_run,
)

RUN_ID = "20260101-000000-testrn"


def _receipt(statement_id: str, prev: str | None, **overrides: Any) -> dict[str, Any]:
    """A minimal structurally-valid receipt, hashed last (mirrors the contract)."""
    body: dict[str, Any] = {
        "v": "openprose.receipt.v1",
        "run_id": RUN_ID,
        "statement_id": statement_id,
        "kind": "session",
        "agent": "worker",
        "input_fingerprints": {},
        "output_fingerprint": None,
        "status": "rendered",
        "surprise_cause": "self",
        "usage": {
            "basis": "estimated",
            "input_tokens": 100,
            "output_tokens": 10,
            "model": "haiku",
        },
        "error": None,
        "detail": None,
        "reused_from": None,
        "prev": prev,
        "hash_algorithm": "sha256",
    }
    body.update(overrides)
    body["content_hash"] = content_hash(body)
    return body


def _chain(n: int) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    prev: str | None = None
    for i in range(1, n + 1):
        r = _receipt(f"s{i:03d}", prev)
        receipts.append(r)
        prev = r["content_hash"]
    return receipts


def _manifest(
    receipts: list[dict[str, Any]],
    count: int | None = None,
    head: str | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "v": "openprose.run.v1",
        "run_id": RUN_ID,
        "program": "fixture.prose",
        "state_backend": "filesystem",
        "status": status,
        "receipt_count": count if count is not None else len(receipts),
        "ledger_head": head if head is not None else receipts[-1]["content_hash"],
    }


def _write_run(
    run_dir: Path,
    receipts: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
    trailing_garbage: str | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, sort_keys=True) for r in receipts]
    if trailing_garbage is not None:
        lines.append(trailing_garbage)
    (run_dir / "receipts.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if manifest is not None:
        (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


class TestLoadLedger:
    def test_loads_valid_ledger(self, tmp_path: Path) -> None:
        run = _write_run(tmp_path / "r", _chain(3), None)
        loaded = load_ledger(run / "receipts.jsonl")
        assert len(loaded.receipts) == 3
        assert loaded.errors == [] and loaded.warnings == []

    def test_torn_final_line_is_warning_with_prefix(self, tmp_path: Path) -> None:
        run = _write_run(tmp_path / "r", _chain(2), None, trailing_garbage='{"v": "openprose.re')
        loaded = load_ledger(run / "receipts.jsonl")
        assert len(loaded.receipts) == 2
        assert loaded.errors == []
        assert loaded.warnings[0].code == "torn_write_line"
        assert "torn write" in loaded.warnings[0].message

    def test_invalid_json_mid_ledger_is_error_and_stops(self, tmp_path: Path) -> None:
        chain = _chain(3)
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        lines = [
            json.dumps(chain[0], sort_keys=True),
            "NOT JSON",
            json.dumps(chain[2], sort_keys=True),
        ]
        (run_dir / "receipts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        loaded = load_ledger(run_dir / "receipts.jsonl")
        # Loader stops at the parsed prefix: line 3 is never parsed.
        assert len(loaded.receipts) == 1
        assert loaded.errors[0].code == "invalid_json"
        assert loaded.errors[0].line_no == 2

    def test_non_object_line_is_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        (run_dir / "receipts.jsonl").write_text('[1,2]\n"x"\n', encoding="utf-8")
        loaded = load_ledger(run_dir / "receipts.jsonl")
        assert loaded.receipts == []
        assert loaded.errors[0].code == "invalid_json"

    def test_empty_file_loads_empty(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        (run_dir / "receipts.jsonl").write_text("", encoding="utf-8")
        loaded = load_ledger(run_dir / "receipts.jsonl")
        assert loaded.receipts == [] and loaded.errors == [] and loaded.warnings == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/evaluators/test_openprose_receipts.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_ledger'` (canonical tests still pass).

- [ ] **Step 3: Implement models + `load_ledger`**

Create `atp/evaluators/openprose_receipts/reader.py`:

```python
"""Load and verify openprose.receipt.v1 ledgers: chain consistency + anchor.

Semantics follow the vendored contract (method/contract/openprose/receipt.md)
and the design spec (docs/superpowers/specs/
2026-07-16-openprose-receipts-evaluation-input-design.md). ``load_ledger``
never raises on bad content — it accumulates line-level issues and returns the
parseable prefix, because a trailing torn line must degrade to a warning while
the prefix still verifies.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from atp.evaluators.openprose_receipts.canonical import content_hash

RECEIPT_VERSION = "openprose.receipt.v1"
RUN_VERSION = "openprose.run.v1"

_KINDS = {"session", "parallel_branch", "block_call", "discretion", "control"}
_STATUSES = {"rendered", "skipped", "failed"}
_USAGE_BASES = {"exact", "estimated", "unavailable"}
# Nullable metadata fields (agent, surprise_cause, error, detail, reused_from)
# are not required: consumers must tolerate their absence, per append-frozen
# "ignore unknown, require only what you consume".
_REQUIRED_FIELDS = (
    "v",
    "run_id",
    "statement_id",
    "kind",
    "input_fingerprints",
    "output_fingerprint",
    "status",
    "usage",
    "prev",
    "hash_algorithm",
    "content_hash",
)


class Issue(BaseModel):
    """One verification finding, stable-coded for programmatic consumers."""

    code: str
    line_no: int | None = None  # 1-based ledger line; None for run-level issues
    message: str


class LoadedLedger(BaseModel):
    """Parsed prefix of a ledger plus line-level parse issues."""

    receipts: list[dict[str, Any]]
    errors: list[Issue]
    warnings: list[Issue]


class VerifyResult(BaseModel):
    """Outcome of ``verify_run``: ok iff there are no errors (warnings allowed)."""

    ok: bool
    # Receipts covered by verification — NOT physical line count: the parsed
    # prefix on a torn line, the anchored prefix on a torn manifest.
    receipt_count: int = Field(ge=0)
    errors: list[Issue]
    warnings: list[Issue]


def load_ledger(path: Path) -> LoadedLedger:
    """Parse receipts.jsonl into its valid prefix + issues (never raises)."""
    receipts: list[dict[str, Any]] = []
    errors: list[Issue] = []
    warnings: list[Issue] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    for line_no, line in enumerate(lines, start=1):
        try:
            parsed = json.loads(line)
        except ValueError:
            if line_no == len(lines):
                warnings.append(
                    Issue(
                        code="torn_write_line",
                        line_no=line_no,
                        message=(
                            "torn write: final ledger line is not valid JSON; "
                            f"verifying the {len(receipts)}-receipt prefix"
                        ),
                    )
                )
            else:
                errors.append(
                    Issue(
                        code="invalid_json",
                        line_no=line_no,
                        message=(
                            f"invalid JSON on non-final line {line_no}; the chain "
                            f"is broken here — stopping at the "
                            f"{len(receipts)}-receipt prefix"
                        ),
                    )
                )
            break
        if not isinstance(parsed, dict):
            errors.append(
                Issue(
                    code="invalid_json",
                    line_no=line_no,
                    message=f"line {line_no} is not a JSON object",
                )
            )
            break
        receipts.append(parsed)
    return LoadedLedger(receipts=receipts, errors=errors, warnings=warnings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/evaluators/test_openprose_receipts.py -v`
Expected: `TestLoadLedger` PASSes. (`verify_run` import will fail — remove `verify_run` from the test-file import until Task 4, or add a stub `def verify_run(run_dir: Path) -> VerifyResult: raise NotImplementedError` and keep the import.) Use the stub — Task 4 replaces it.

- [ ] **Step 5: Format, lint, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add atp/evaluators/openprose_receipts/reader.py tests/unit/evaluators/test_openprose_receipts.py
git commit -m "feat: openprose_receipts ledger loader with line-level issues"
```

---

### Task 4: `verify_run` — structure, hash, chain, anchor, torn cases

**Files:**
- Modify: `atp/evaluators/openprose_receipts/reader.py` (replace the Task-3 stub)
- Test: `tests/unit/evaluators/test_openprose_receipts.py` (append)

**Interfaces:**
- Consumes: `load_ledger`, `content_hash`, models from Tasks 2–3.
- Produces: `verify_run(run_dir: Path) -> VerifyResult` — the single verification entry point for Tasks 5–6. Error codes: `unknown_version`, `missing_field`, `invalid_enum`, `invalid_number`, `content_hash_mismatch`, `chain_break`, `invalid_json`, `empty_ledger`, `invalid_run_json`, `receipt_count_mismatch`, `ledger_head_mismatch`. Warning codes: `torn_write_line`, `torn_write_manifest`, `no_anchor`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/evaluators/test_openprose_receipts.py`:

```python
class TestVerifyRun:
    def test_valid_run_is_ok(self, tmp_path: Path) -> None:
        receipts = _chain(3)
        run = _write_run(tmp_path / "r", receipts, _manifest(receipts))
        result = verify_run(run)
        assert result.ok is True
        assert result.receipt_count == 3
        assert result.errors == [] and result.warnings == []

    def test_empty_ledger_is_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        (run_dir / "receipts.jsonl").write_text("", encoding="utf-8")
        result = verify_run(run_dir)
        assert result.ok is False
        assert result.errors[0].code == "empty_ledger"

    def test_unknown_version_is_refused(self, tmp_path: Path) -> None:
        receipts = _chain(1)
        receipts[0]["v"] = "openprose.receipt.v99"
        run = _write_run(tmp_path / "r", receipts, None)
        result = verify_run(run)
        assert any(e.code == "unknown_version" for e in result.errors)

    def test_tampered_content_fails_hash(self, tmp_path: Path) -> None:
        receipts = _chain(2)
        receipts[1]["agent"] = "tampered"  # content changed, hash not recomputed
        run = _write_run(tmp_path / "r", receipts, _manifest(receipts))
        result = verify_run(run)
        assert result.ok is False
        assert any(
            e.code == "content_hash_mismatch" and "content_hash mismatch" in e.message
            for e in result.errors
        )

    def test_broken_prev_fails_chain(self, tmp_path: Path) -> None:
        receipts = _chain(2)
        receipts[1]["prev"] = "sha256:" + "0" * 64
        receipts[1]["content_hash"] = content_hash(receipts[1])  # rehash honestly
        run = _write_run(
            tmp_path / "r", receipts, _manifest(receipts)
        )
        result = verify_run(run)
        assert result.ok is False
        assert any(
            e.code == "chain_break" and "prev broken" in e.message
            for e in result.errors
        )

    def test_float_tokens_rejected(self, tmp_path: Path) -> None:
        receipts = _chain(1)
        receipts[0]["usage"]["input_tokens"] = 1.5
        receipts[0]["content_hash"] = "sha256:" + "0" * 64  # unhashable anyway
        run = _write_run(tmp_path / "r", receipts, None)
        result = verify_run(run)
        assert any(e.code == "invalid_number" for e in result.errors)

    def test_missing_manifest_is_warning(self, tmp_path: Path) -> None:
        run = _write_run(tmp_path / "r", _chain(2), None)
        result = verify_run(run)
        assert result.ok is True
        assert result.warnings[0].code == "no_anchor"

    def test_receipt_count_mismatch_with_matching_head(self, tmp_path: Path) -> None:
        receipts = _chain(2)
        run = _write_run(tmp_path / "r", receipts, _manifest(receipts, count=5))
        result = verify_run(run)
        assert result.ok is False
        assert any(e.code == "receipt_count_mismatch" for e in result.errors)

    def test_torn_manifest_trailing_by_one_is_warning(self, tmp_path: Path) -> None:
        receipts = _chain(2)
        manifest = _manifest(
            receipts, count=1, head=receipts[0]["content_hash"], status="running"
        )
        run = _write_run(tmp_path / "r", receipts, manifest)
        result = verify_run(run)
        assert result.ok is True
        assert any(
            w.code == "torn_write_manifest" and "torn write" in w.message
            for w in result.warnings
        )
        assert result.receipt_count == 1  # the anchored prefix

    def test_manifest_trailing_by_two_is_error(self, tmp_path: Path) -> None:
        receipts = _chain(3)
        manifest = _manifest(receipts, count=1, head=receipts[0]["content_hash"])
        run = _write_run(tmp_path / "r", receipts, manifest)
        result = verify_run(run)
        assert result.ok is False
        assert any(
            e.code == "ledger_head_mismatch" and "ledger_head" in e.message
            for e in result.errors
        )

    def test_head_matching_nothing_is_error(self, tmp_path: Path) -> None:
        receipts = _chain(2)
        manifest = _manifest(receipts, head="sha256:" + "f" * 64)
        run = _write_run(tmp_path / "r", receipts, manifest)
        result = verify_run(run)
        assert result.ok is False
        assert any(e.code == "ledger_head_mismatch" for e in result.errors)

    def test_torn_line_with_anchored_prefix_is_ok(self, tmp_path: Path) -> None:
        receipts = _chain(2)
        run = _write_run(
            tmp_path / "r",
            receipts,
            _manifest(receipts),
            trailing_garbage='{"v": "openprose.rec',
        )
        result = verify_run(run)
        assert result.ok is True
        assert result.receipt_count == 2
        assert any(w.code == "torn_write_line" for w in result.warnings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/evaluators/test_openprose_receipts.py::TestVerifyRun -v`
Expected: FAIL with `NotImplementedError` from the Task-3 stub.

- [ ] **Step 3: Implement `verify_run`**

Replace the stub in `atp/evaluators/openprose_receipts/reader.py`:

```python
def _receipt_issues(receipt: dict[str, Any], line_no: int) -> list[Issue]:
    """Structural validation of one receipt (in code — no jsonschema)."""
    v = receipt.get("v")
    if v != RECEIPT_VERSION:
        # Contract: consumers MUST refuse unknown versions; no point checking
        # a shape we do not know.
        return [
            Issue(
                code="unknown_version",
                line_no=line_no,
                message=f"unknown receipt version {v!r} at line {line_no} — refusing",
            )
        ]
    issues: list[Issue] = []
    for field in _REQUIRED_FIELDS:
        if field not in receipt:
            issues.append(
                Issue(
                    code="missing_field",
                    line_no=line_no,
                    message=f"missing field '{field}' at line {line_no}",
                )
            )
    if "kind" in receipt and receipt["kind"] not in _KINDS:
        issues.append(
            Issue(
                code="invalid_enum",
                line_no=line_no,
                message=f"invalid kind {receipt['kind']!r} at line {line_no}",
            )
        )
    if "status" in receipt and receipt["status"] not in _STATUSES:
        issues.append(
            Issue(
                code="invalid_enum",
                line_no=line_no,
                message=f"invalid status {receipt['status']!r} at line {line_no}",
            )
        )
    usage = receipt.get("usage")
    if "usage" in receipt and (
        not isinstance(usage, dict) or usage.get("basis") not in _USAGE_BASES
    ):
        issues.append(
            Issue(
                code="invalid_enum",
                line_no=line_no,
                message=f"invalid usage.basis at line {line_no}",
            )
        )
    if "hash_algorithm" in receipt and receipt["hash_algorithm"] != "sha256":
        issues.append(
            Issue(
                code="invalid_enum",
                line_no=line_no,
                message=f"unsupported hash_algorithm at line {line_no}",
            )
        )
    return issues


def verify_run(run_dir: Path) -> VerifyResult:
    """Verify a run directory's ledger: structure → hash → chain → anchor.

    ``ok`` is True iff there are no errors; torn writes (a trailing
    unparseable line, or a manifest trailing the ledger by exactly one
    receipt) degrade to warnings — a crash artifact, not tampering.
    """
    loaded = load_ledger(run_dir / "receipts.jsonl")
    errors = list(loaded.errors)
    warnings = list(loaded.warnings)
    receipts = loaded.receipts
    receipt_count = len(receipts)

    if not receipts and not errors:
        errors.append(
            Issue(
                code="empty_ledger",
                message=(
                    "empty ledger: a run always opens with a run_start "
                    "control receipt"
                ),
            )
        )

    prev_expected: str | None = None
    for line_no, receipt in enumerate(receipts, start=1):
        line_issues = _receipt_issues(receipt, line_no)
        errors.extend(line_issues)
        if not line_issues:
            try:
                recomputed = content_hash(receipt)
            except (TypeError, ValueError):
                errors.append(
                    Issue(
                        code="invalid_number",
                        line_no=line_no,
                        message=(
                            f"non-canonical value at line {line_no} "
                            "(floats/NaN break hash portability)"
                        ),
                    )
                )
            else:
                if recomputed != receipt.get("content_hash"):
                    errors.append(
                        Issue(
                            code="content_hash_mismatch",
                            line_no=line_no,
                            message=f"content_hash mismatch at line {line_no}",
                        )
                    )
        if receipt.get("prev") != prev_expected:
            errors.append(
                Issue(
                    code="chain_break",
                    line_no=line_no,
                    message=f"prev broken at line {line_no}",
                )
            )
        prev_expected = receipt.get("content_hash")

    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        warnings.append(
            Issue(
                code="no_anchor",
                message=(
                    "run.json missing — chain-only verification, "
                    "ledger_head not checked"
                ),
            )
        )
        return VerifyResult(
            ok=not errors,
            receipt_count=receipt_count,
            errors=errors,
            warnings=warnings,
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError:
        manifest = None
    if not isinstance(manifest, dict):
        errors.append(
            Issue(code="invalid_run_json", message="run.json is not a JSON object")
        )
    elif manifest.get("v") != RUN_VERSION:
        errors.append(
            Issue(
                code="unknown_version",
                message=f"unknown run.json version {manifest.get('v')!r} — refusing",
            )
        )
    elif receipts:
        head = manifest.get("ledger_head")
        count = manifest.get("receipt_count")
        last = receipts[-1]
        if head == last.get("content_hash"):
            if count != len(receipts):
                errors.append(
                    Issue(
                        code="receipt_count_mismatch",
                        message=(
                            f"run.json receipt_count={count!r} but the ledger "
                            f"has {len(receipts)} receipts"
                        ),
                    )
                )
        elif head is not None and head == last.get("prev") and count == len(receipts) - 1:
            # Append succeeded, head update did not: the only crash artifact a
            # torn write can produce is trailing by EXACTLY one receipt.
            warnings.append(
                Issue(
                    code="torn_write_manifest",
                    message=(
                        "torn write: manifest trails ledger by exactly one "
                        "receipt; verifying the anchored prefix"
                    ),
                )
            )
            receipt_count = len(receipts) - 1
        else:
            errors.append(
                Issue(
                    code="ledger_head_mismatch",
                    message=(
                        "ledger_head does not match the ledger (a valid but "
                        "unanchored suffix is not a torn write)"
                    ),
                )
            )

    return VerifyResult(
        ok=not errors,
        receipt_count=receipt_count,
        errors=errors,
        warnings=warnings,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/evaluators/test_openprose_receipts.py -v`
Expected: all PASS.

- [ ] **Step 5: Format, lint, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add atp/evaluators/openprose_receipts/reader.py tests/unit/evaluators/test_openprose_receipts.py
git commit -m "feat: verify_run — structure, content hash, chain, anchor, torn-write semantics"
```

---

### Task 5: Contract tests over the vendored fixtures

**Files:**
- Create: `tests/contract/test_openprose_receipts_contract.py`

**Interfaces:**
- Consumes: `verify_run` (Task 4); fixture tree from Task 1.
- Produces: nothing (leaf).

- [ ] **Step 1: Write the tests**

Create `tests/contract/test_openprose_receipts_contract.py`:

```python
"""Contract tests: openprose.receipt.v1 verification over vendored fixtures.

Corpus runs and broken fixtures are pinned copies of upstream open-prose
(method/contract/openprose/PROVENANCE.md). Broken-fixture outcomes are
asserted through each fixture's own upstream expected.json, so a future
fixture refresh stays mechanical.
"""

import json
from pathlib import Path

import pytest

from atp.evaluators.openprose_receipts.reader import verify_run

_FIXTURES = (
    Path(__file__).parent.parent.parent
    / "method"
    / "contract"
    / "fixtures"
    / "openprose"
)
_CORPUS = sorted(p for p in (_FIXTURES / "runs").iterdir() if p.is_dir())
_BROKEN = sorted(p for p in (_FIXTURES / "broken").iterdir() if p.is_dir())


def test_fixture_tree_is_present() -> None:
    """Guard against a silently-empty parametrization."""
    assert len(_CORPUS) == 4
    assert len(_BROKEN) == 4


@pytest.mark.parametrize("run_dir", _CORPUS, ids=lambda p: p.name)
def test_corpus_run_verifies_clean(run_dir: Path) -> None:
    result = verify_run(run_dir)
    assert result.ok, [i.message for i in result.errors]
    assert result.errors == []
    assert result.warnings == []
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert result.receipt_count == manifest["receipt_count"]


@pytest.mark.parametrize("fixture_dir", _BROKEN, ids=lambda p: p.name)
def test_broken_fixture_matches_upstream_expectation(fixture_dir: Path) -> None:
    expected = json.loads(
        (fixture_dir / "expected.json").read_text(encoding="utf-8")
    )
    result = verify_run(fixture_dir)
    assert result.ok == expected["ok"]
    if "error_contains" in expected:
        assert any(
            expected["error_contains"] in i.message for i in result.errors
        ), [i.message for i in result.errors]
    if "warning_contains" in expected:
        assert any(
            expected["warning_contains"] in i.message for i in result.warnings
        ), [i.message for i in result.warnings]
```

- [ ] **Step 2: Run the contract tests**

Run: `uv run pytest tests/contract/test_openprose_receipts_contract.py -v`
Expected: all PASS — 4 corpus runs clean, 4 broken fixtures matching their `expected.json`. If a corpus run fails structural validation, the reader is stricter than the real writer: fix `_receipt_issues`-level assumptions in `reader.py` (this is exactly what the contract tests are for), not the fixture.

- [ ] **Step 3: Format, lint, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add tests/contract/test_openprose_receipts_contract.py
git commit -m "test: contract tests — vendored open-prose corpus + broken fixtures"
```

---

### Task 6: `receipt_chain` checker + registration

**Files:**
- Create: `atp/evaluators/openprose_receipts/checker.py`
- Modify: `atp/evaluators/checkers/__init__.py`
- Test: `tests/unit/evaluators/test_openprose_receipts.py` (append)

**Interfaces:**
- Consumes: `verify_run` (Task 4); `CaseVerdict` from `atp.core.results`; registry `register_checker`/`get_checker` from `atp.evaluators.checkers`.
- Produces: `receipt_chain_check(config: dict[str, Any], text: str | None) -> CaseVerdict`; registry name `"receipt_chain"`; version string `RECEIPT_CHAIN_CHECKER_VERSION = "receipt_chain@1"`. Config contract: `run_dir` (required, relative), `_case_dir` (dispatch-injected confinement root — Task 8).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/evaluators/test_openprose_receipts.py`:

```python
from atp.evaluators.checkers import get_checker
from atp.evaluators.openprose_receipts.checker import receipt_chain_check


class TestReceiptChainChecker:
    def _valid_case(self, case_dir: Path) -> dict[str, Any]:
        receipts = _chain(2)
        _write_run(case_dir / "runs" / "r1", receipts, _manifest(receipts))
        return {"run_dir": "runs/r1", "_case_dir": str(case_dir)}

    def test_registered_in_registry(self) -> None:
        assert get_checker("receipt_chain") is receipt_chain_check

    def test_pass_on_valid_run(self, tmp_path: Path) -> None:
        verdict = receipt_chain_check(self._valid_case(tmp_path), None)
        assert verdict.critical_pass is True
        assert verdict.malformed is False
        assert verdict.rubric_score == 1.0
        assert verdict.details["receipt_count"] == 2
        assert verdict.grader_version == "receipt_chain@1"

    def test_text_argument_is_ignored(self, tmp_path: Path) -> None:
        verdict = receipt_chain_check(self._valid_case(tmp_path), "NOT JSON {{{")
        assert verdict.critical_pass is True

    def test_fail_on_broken_chain_is_not_malformed(self, tmp_path: Path) -> None:
        receipts = _chain(2)
        receipts[1]["prev"] = "sha256:" + "0" * 64
        receipts[1]["content_hash"] = content_hash(receipts[1])
        _write_run(tmp_path / "runs" / "r1", receipts, _manifest(receipts))
        verdict = receipt_chain_check(
            {"run_dir": "runs/r1", "_case_dir": str(tmp_path)}, None
        )
        assert verdict.critical_pass is False
        assert verdict.malformed is False
        assert verdict.rubric_score == 0.0
        assert verdict.details["errors"]

    def test_missing_run_dir_config_is_malformed(self, tmp_path: Path) -> None:
        verdict = receipt_chain_check({"_case_dir": str(tmp_path)}, None)
        assert verdict.malformed is True and verdict.critical_pass is False

    def test_absolute_run_dir_is_malformed(self, tmp_path: Path) -> None:
        verdict = receipt_chain_check(
            {"run_dir": str(tmp_path), "_case_dir": str(tmp_path)}, None
        )
        assert verdict.malformed is True

    def test_missing_case_dir_is_malformed(self, tmp_path: Path) -> None:
        verdict = receipt_chain_check({"run_dir": "runs/r1"}, None)
        assert verdict.malformed is True

    def test_escape_from_case_dir_is_malformed(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        verdict = receipt_chain_check(
            {"run_dir": "../outside", "_case_dir": str(case_dir)}, None
        )
        assert verdict.malformed is True

    def test_missing_ledger_is_malformed(self, tmp_path: Path) -> None:
        verdict = receipt_chain_check(
            {"run_dir": "runs/nope", "_case_dir": str(tmp_path)}, None
        )
        assert verdict.malformed is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/evaluators/test_openprose_receipts.py::TestReceiptChainChecker -v`
Expected: FAIL — `ImportError: cannot import name 'receipt_chain_check'`.

- [ ] **Step 3: Implement checker + register**

Create `atp/evaluators/openprose_receipts/checker.py`:

```python
"""receipt_chain checker: verify an open-prose run directory's ledger.

Selected via ``grader: {type: programmatic, checker: receipt_chain,
config: {run_dir: <relative path>}}``. The agent-output ``text`` is ignored —
the artifact under test is a run *directory*. ``_case_dir`` is dispatch-owned
(injected by AgentEvalCaseEvaluator, never trusted from user config) and is
the confinement root for ``run_dir`` resolution.
"""

from pathlib import Path
from typing import Any

from atp.core.results import CaseVerdict
from atp.evaluators.openprose_receipts.reader import verify_run

RECEIPT_CHAIN_CHECKER_VERSION = "receipt_chain@1"


def receipt_chain_check(config: dict[str, Any], text: str | None) -> CaseVerdict:
    """Verify the hash-chained ledger under config.run_dir (text is ignored)."""
    run_dir_raw = config.get("run_dir")
    if not isinstance(run_dir_raw, str) or not run_dir_raw.strip():
        return _malformed("grader.config.run_dir is required (relative path)")
    run_dir = Path(run_dir_raw)
    if run_dir.is_absolute():
        return _malformed("run_dir must be relative to the case directory")
    case_dir_raw = config.get("_case_dir")
    if not case_dir_raw:
        return _malformed("no case directory available to resolve run_dir")
    case_dir = Path(str(case_dir_raw)).resolve()
    resolved = (case_dir / run_dir).resolve()
    if not resolved.is_relative_to(case_dir):
        return _malformed("run_dir escapes the case directory")
    if not (resolved / "receipts.jsonl").is_file():
        return _malformed("receipts.jsonl not found under run_dir")
    result = verify_run(resolved)
    return CaseVerdict(
        critical_pass=result.ok,
        rubric_score=1.0 if result.ok else 0.0,
        details={
            "errors": [issue.model_dump() for issue in result.errors],
            "warnings": [issue.model_dump() for issue in result.warnings],
            "receipt_count": result.receipt_count,
        },
        grader_version=RECEIPT_CHAIN_CHECKER_VERSION,
    )


def _malformed(reason: str) -> CaseVerdict:
    """Not gradeable (bad config / missing artifacts) — distinct from ok=False."""
    return CaseVerdict(
        critical_pass=False,
        malformed=True,
        details={"reason": reason},
        grader_version=RECEIPT_CHAIN_CHECKER_VERSION,
    )
```

Modify `atp/evaluators/checkers/__init__.py` — add the import and registration (keep alphabetical-by-name registration order):

```python
from atp.evaluators.citation_grounding.checker import citation_grounding_check
from atp.evaluators.findings.checker import findings_check
from atp.evaluators.json_path.checker import json_path_check
from atp.evaluators.openprose_receipts.checker import receipt_chain_check

register_checker("citation_grounding", citation_grounding_check)
register_checker("findings_match", findings_check)
register_checker("json_path", json_path_check)
register_checker("receipt_chain", receipt_chain_check)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/evaluators/test_openprose_receipts.py -v`
Expected: all PASS.

- [ ] **Step 5: Format, lint, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add atp/evaluators/openprose_receipts/checker.py atp/evaluators/checkers/__init__.py tests/unit/evaluators/test_openprose_receipts.py
git commit -m "feat: receipt_chain checker — 4th deterministic checker in the registry"
```

---

### Task 7: `Grader` schema validation for `receipt_chain`

**Files:**
- Modify: `packages/atp-method/atp_method/schema.py` (the `validate_grader_requirements` validator, after the `citation_grounding` block ~line 267)
- Test: `packages/atp-method/tests/test_schema.py` (append)

**Interfaces:**
- Consumes: `Grader` model (existing).
- Produces: load-time rule — `checker == "receipt_chain"` requires `grader.config.run_dir` as a non-empty **relative** path string. Runtime confinement stays in the checker (Task 6).

- [ ] **Step 1: Write the failing tests**

Append to `packages/atp-method/tests/test_schema.py` (follow the file's existing imports — it already imports `Grader` and `pytest`; `pydantic.ValidationError` may need adding):

```python
def _receipt_chain_grader(config: dict | None) -> None:
    Grader(
        type="programmatic",
        checker="receipt_chain",
        critical_check="ledger chain verifies",
        scoring="binary: chain + anchor valid",
        config=config,
    )


def test_receipt_chain_requires_run_dir() -> None:
    with pytest.raises(ValidationError, match="receipt_chain"):
        _receipt_chain_grader({})
    with pytest.raises(ValidationError, match="receipt_chain"):
        _receipt_chain_grader(None)
    with pytest.raises(ValidationError, match="receipt_chain"):
        _receipt_chain_grader({"run_dir": "   "})


def test_receipt_chain_rejects_absolute_run_dir() -> None:
    with pytest.raises(ValidationError, match="relative"):
        _receipt_chain_grader({"run_dir": "/etc/anything"})


def test_receipt_chain_accepts_relative_run_dir() -> None:
    _receipt_chain_grader({"run_dir": "runs/r1"})  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/atp-method/tests/test_schema.py -v -k receipt_chain`
Expected: FAIL — no validation error raised yet.

- [ ] **Step 3: Implement the validator branch**

In `packages/atp-method/atp_method/schema.py`: add `from pathlib import Path` to the imports, then append this branch inside `validate_grader_requirements`, after the `citation_grounding` block and before `return self`:

```python
        if self.checker == "receipt_chain":
            run_dir = (self.config or {}).get("run_dir")
            if not isinstance(run_dir, str) or not run_dir.strip():
                raise ValueError(
                    "checker 'receipt_chain' requires grader.config.run_dir "
                    "(a non-empty relative path string)"
                )
            if Path(run_dir).is_absolute():
                raise ValueError(
                    "checker 'receipt_chain' requires a relative run_dir "
                    "(resolved against the case directory at runtime)"
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/atp-method/tests/test_schema.py -v`
Expected: all PASS (including pre-existing tests).

- [ ] **Step 5: Format, lint, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add packages/atp-method/atp_method/schema.py packages/atp-method/tests/test_schema.py
git commit -m "feat: Grader schema validation — receipt_chain requires relative config.run_dir"
```

---

### Task 8: `_case_dir` injection in `AgentEvalCaseEvaluator`

**Files:**
- Modify: `packages/atp-method/atp_method/evaluators/case_evaluator.py` (`_evaluate_critical`, the checker-dispatch block)
- Test: `packages/atp-method/tests/test_evaluator.py` (append)

**Interfaces:**
- Consumes: existing dispatch `verdict = checker(assertion.config, text)`; `TaskDefinition.input_data["case_path"]` (set by `atp_method/loader.py:127`).
- Produces: every checker call receives a config **copy** with dispatch-owned `_case_dir` = parent dir of `case_path`; user-supplied `_case_dir` is always stripped; no `case_path` → no `_case_dir` key.

- [ ] **Step 1: Write the failing tests**

Append to `packages/atp-method/tests/test_evaluator.py` (the file already imports `TaskDefinition`, `TestDefinition`, `Assertion`, `AgentEvalCaseEvaluator`, `METHOD_CRITICAL_CHECK`; add `from atp.core.results import CaseVerdict` and `from atp.evaluators.checkers import register_checker` plus `from pathlib import Path`):

```python
def _spy_checker(captured: dict):
    def check(config: dict, text: str | None) -> CaseVerdict:
        captured.clear()
        captured.update(config)
        return CaseVerdict(critical_pass=True)

    return check


# Registered once at module import: the registry has no unregister and raises
# on duplicate names, so per-test registration would break under repetition.
_CAPTURED_A: dict = {}
_CAPTURED_B: dict = {}
register_checker("_test_spy_case_dir", _spy_checker(_CAPTURED_A))
register_checker("_test_spy_no_case_path", _spy_checker(_CAPTURED_B))


@pytest.mark.anyio
async def test_evaluator_injects_case_dir_and_overrides_user_value(
    tmp_path: Path,
) -> None:
    """_case_dir is dispatch-owned: derived from case_path, never user config."""
    case_path = tmp_path / "cases" / "case-x-001.yaml"
    task = TestDefinition(
        id="case-x-001",
        name="x",
        task=TaskDefinition(
            description="do x",
            input_data={"case_path": str(case_path)},
        ),
    )
    assertion = Assertion(
        type=METHOD_CRITICAL_CHECK,
        critical=True,
        config={"checker": "_test_spy_case_dir", "_case_dir": "/evil/injected"},
    )
    ev = AgentEvalCaseEvaluator(judge=FakeJudge(1.0))
    await ev.evaluate(task, _response(), [], assertion)
    assert _CAPTURED_A["_case_dir"] == str(case_path.parent)


@pytest.mark.anyio
async def test_evaluator_without_case_path_injects_nothing() -> None:
    """No case_path in input_data → no _case_dir key (user value still stripped)."""
    assertion = Assertion(
        type=METHOD_CRITICAL_CHECK,
        critical=True,
        config={"checker": "_test_spy_no_case_path", "_case_dir": "/evil"},
    )
    ev = AgentEvalCaseEvaluator(judge=FakeJudge(1.0))
    await ev.evaluate(_task(), _response(), [], assertion)
    assert "_case_dir" not in _CAPTURED_B
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/atp-method/tests/test_evaluator.py -v -k case_dir`
Expected: FAIL — first test gets `"/evil/injected"` (config passed through untouched), second finds `"/evil"` present.

- [ ] **Step 3: Implement the injection**

In `packages/atp-method/atp_method/evaluators/case_evaluator.py`: add `from pathlib import Path` to the imports, then replace the dispatch block in `_evaluate_critical`:

```python
            text = self._select_text_artifact(
                response, assertion.config.get("artifact_name")
            )
            verdict = checker(assertion.config, text)
```

with:

```python
            text = self._select_text_artifact(
                response, assertion.config.get("artifact_name")
            )
            config = dict(assertion.config)
            # _case_dir is dispatch-owned (the confinement root for checkers
            # that read case-relative artifacts, e.g. receipt_chain): never
            # trust a user-supplied value.
            config.pop("_case_dir", None)
            case_path = (task.task.input_data or {}).get("case_path")
            if case_path:
                config["_case_dir"] = str(Path(str(case_path)).parent)
            verdict = checker(config, text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/atp-method/tests/test_evaluator.py -v`
Expected: all PASS (including all pre-existing evaluator tests — the copy must not change behavior for existing checkers).

- [ ] **Step 5: Format, lint, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add packages/atp-method/atp_method/evaluators/case_evaluator.py packages/atp-method/tests/test_evaluator.py
git commit -m "feat: dispatch-owned _case_dir injection for checker configs"
```

---

### Task 9: Docs, full suite, PR

**Files:**
- Modify: `CLAUDE.md` (component 4 checker list; component 25 checker mentions)
- Modify: `TODO.md` (tick progress note on the ACTIVE item)

**Interfaces:**
- Consumes: everything above.
- Produces: the PR.

- [ ] **Step 1: Update CLAUDE.md checker mentions**

In `CLAUDE.md` component 4, extend the checker list: `currently `citation_grounding`, `findings_match`, and `json_path`` → `currently `citation_grounding`, `findings_match`, `json_path`, and `receipt_chain` (verifies open-prose `receipts.jsonl` hash chains — vendored contract in `method/contract/openprose/`)`. In component 25, extend the deterministic-checkers parenthetical the same way.

- [ ] **Step 2: Note progress in TODO.md**

In the ACTIVE item "open-prose receipts/IR как evaluation-вход", append one line under «Объём»: `  - Реализация: PR (ветка feat/openprose-receipts-input), спека docs/superpowers/specs/2026-07-16-openprose-receipts-evaluation-input-design.md.`

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest tests/ packages/atp-method/tests/ -m "not slow" -q`
Expected: PASS, no regressions. If anything unrelated fails, check `git stash && uv run pytest <failing test>` to confirm pre-existing, then unstash — do not fix unrelated failures in this PR.

- [ ] **Step 4: Final quality gate**

Run: `uv run ruff format . && uv run ruff check . && uv run pyrefly check`
Expected: clean.

- [ ] **Step 5: Commit docs, push, open PR**

```bash
git add CLAUDE.md TODO.md
git commit -m "docs: register receipt_chain checker in CLAUDE.md component map"
git push -u origin feat/openprose-receipts-input
gh pr create --title "feat: open-prose receipts as evaluation input (receipt_chain checker)" --body "$(cat <<'EOF'
## Summary
- Vendor open-prose `openprose.receipt.v1` / `openprose.compile-ir.v1` contracts + corpus/broken fixtures (pinned @ a0395cd, provenance in `method/contract/openprose/PROVENANCE.md`)
- New stdlib verifier `atp/evaluators/openprose_receipts/` — canonical form, content-hash recomputation, prev-chain, `ledger_head` anchor, two torn-write cases (warnings)
- New deterministic checker `receipt_chain` (4th in the registry) with dispatch-injected `_case_dir` confinement; `Grader` schema validation for `config.run_dir`
- v1 is receipts-only: IR contract/fixtures vendored, no IR reader

Design spec: `docs/superpowers/specs/2026-07-16-openprose-receipts-evaluation-input-design.md`

## Test plan
- Unit: canonicalization, loader (torn line / mid-ledger stop), verify_run edge cases, checker verdict mapping + confinement, `_case_dir` injection, Grader validation
- Contract: 4 vendored corpus runs verify clean; 4 broken fixtures match their upstream `expected.json` verbatim

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Do **not** merge — Copilot review first, merge is human-gated (repo git workflow).

---

## Self-Review Notes

- Spec coverage: vendoring (T1), canonical (T2), loader + stop-at-prefix (T3), verify_run incl. strict torn-manifest K==N−1 + `empty_ledger` + `receipt_count_mismatch` (T4), contract tests via `expected.json` (T5), checker + registration + confinement (T6), Grader schema rule (T7), `_case_dir` injection owned-by-evaluator tests (T8), docs (T9). IR reader deliberately absent (spec non-goal).
- Message substrings: `"prev broken"` (chain_break), `"content_hash mismatch"`, `"ledger_head"` (ledger_head_mismatch), `"torn write"` (both torn codes) — asserted in T4 unit tests and T5 contract tests.
- Type consistency: `Issue`/`VerifyResult`/`LoadedLedger` defined once in T3, consumed by T4–T6; `receipt_chain@1` string identical in T6 impl and tests.

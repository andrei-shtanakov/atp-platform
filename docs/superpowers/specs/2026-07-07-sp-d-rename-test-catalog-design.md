# Design: SP-D — rename `atp/catalog/` → `atp/test_catalog/`

**Date:** 2026-07-07
**Status:** Approved (brainstorm) — ready for implementation plan
**ADR:** [ADR-ECO-003b](../../../../_cowork_output/decisions/2026-07-02-adr-eco-003b-catalog-distribution.md)
**Epic:** 003b, increment **SP-D** (final). Closes the epic.

---

## Problem

`atp/catalog/` is the **test-suite catalog** (browse / run / publish curated test suites:
`models.py`, `repository.py`, `sync.py`, `comparison.py`, `builtin/` data). Its bare name
`catalog` collides conceptually with the model catalog (`atp/model_catalog/`, shipped in SP-A).
ADR-003b calls to "rename/document `atp/catalog/` as test-catalog" to remove that ambiguity.

SP-D is a **purely mechanical rename** of the internal package to `atp/test_catalog/`, updating
every reference. No behavior changes. The user-facing `atp catalog` CLI command and all class
names stay.

## Decision (from brainstorm, profile A)

- Rename the **package** `atp/catalog/` → `atp/test_catalog/` (`git mv`, contents + `builtin/`
  data move with it).
- **Do NOT** rename: the `atp catalog` **CLI command** (its click group stays `name="catalog"`;
  `atp/cli/commands/catalog.py` keeps its name, only its imports change), the **classes**
  (`CatalogRepository`, `CatalogSuite`, `CatalogTest`, …), the **dashboard route file**
  (`atp/dashboard/v2/routes/catalog.py` — a route, not the package; only its import changes),
  and the **test directory** `tests/unit/catalog/` (cosmetic; leaving it avoids a
  `test_catalog`-dir-vs-`atp.test_catalog` collision risk for zero value).

## Scope — every reference to the package

The rename is correct only when **all** references move — including string references that the
type-checker and linter cannot see.

1. **The package dir:** `git mv atp/catalog atp/test_catalog` (moves `__init__.py`,
   `comparison.py`, `models.py`, `repository.py`, `sync.py`, and the `builtin/` data dir).
   `sync.py`'s `BUILTIN_DIR = Path(__file__).parent / "builtin"` is relative, so it follows the
   move automatically — verify no hardcoded `atp/catalog/builtin` string exists.
2. **Python imports** (`from atp.catalog… import …` / `import atp.catalog…`) → `atp.test_catalog`:
   - intra-package: `atp/test_catalog/sync.py`, `atp/test_catalog/repository.py`;
   - `atp/cli/commands/catalog.py` (3 imports);
   - `packages/atp-dashboard/atp/dashboard/v2/routes/catalog.py` (2 imports);
   - test files under `tests/unit/catalog/`, `tests/unit/cli/test_catalog_cli.py`,
     `tests/unit/dashboard/v2/catalog/test_routes.py`.
3. **String references** (the correctness trap — invisible to ruff/pyrefly): `mock.patch`
   targets like `"atp.catalog.repository.CatalogRepository.get_suite_by_path"` and
   `"atp.catalog.sync.sync_builtin_catalog"` (many in `test_catalog_run.py`,
   `test_catalog_cli.py`, `test_routes.py`) → `"atp.test_catalog…"`. A missed one makes the mock
   silently fail to patch → a test error or (worse) a test that no longer isolates what it
   claims.
4. **Docstrings / module headers** referencing `atp.catalog.*` (e.g. test-file docstrings).
5. **Docs**: `CLAUDE.md`, `README`, `docs/` mentions of `atp/catalog/` (as the test-suite
   catalog) — clarify it is `atp/test_catalog/` and distinct from `atp/model_catalog/`.

## Verification (this replaces feature TDD — the rename is correct iff nothing dangles and the
suite is green)

- **Zero dangling references:**
  `grep -rn "atp\.catalog\b\|atp/catalog\b" . --include='*.py' --include='*.md' --include='*.toml'`
  (excluding `.venv`/`.git`), filtered to drop `model_catalog` — must return **nothing** after
  the rename.
- **Import smoke:** `uv run python -c "import atp.test_catalog.repository, atp.test_catalog.sync,
  atp.test_catalog.comparison, atp.test_catalog.models"` succeeds; `import atp.catalog` now
  **fails** (`ModuleNotFoundError`) — proving the old name is fully gone.
- **CLI smoke:** `uv run atp catalog --help` still works (command unchanged).
- **Targeted suites green:** `tests/unit/catalog/`, `tests/unit/cli/test_catalog_cli.py`,
  `tests/unit/dashboard/v2/catalog/` — all pass (this is where the string-target breakage would
  surface).
- **Full type/lint:** `uv run ruff check . && uv run pyrefly check` clean.

## Non-goals

- Renaming classes, the CLI command, the dashboard route file, or the test directory (see
  Decision).
- Any behavior change to the test-suite catalog.
- Touching `atp/model_catalog/` or the other 003b increments (all shipped).

## Files (net effect)

| Path | Change |
|---|---|
| `atp/catalog/` → `atp/test_catalog/` | `git mv` the whole package (incl. `builtin/`) |
| `atp/cli/commands/catalog.py` | update 3 imports (`atp.catalog` → `atp.test_catalog`) |
| `packages/atp-dashboard/atp/dashboard/v2/routes/catalog.py` | update 2 imports |
| `tests/unit/catalog/*.py` | update imports + **string mock targets** + docstrings |
| `tests/unit/cli/test_catalog_cli.py` | update string mock targets |
| `tests/unit/dashboard/v2/catalog/test_routes.py` | update import + string targets |
| `CLAUDE.md`, `README`, `docs/*` | update `atp/catalog/` references (test-suite catalog) |

Docs/CLAUDE.md pointer updates + the TODO.md "003b epic complete" tick are the plan's final step.

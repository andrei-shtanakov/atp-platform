# Design: SP-D — rename `atp/catalog/` → `atp/test_catalog/`

**Date:** 2026-07-07
**Status:** Approved (brainstorm) — ready for implementation plan
**ADR:** ADR-ECO-003b (catalog distribution) — `../_cowork_output/decisions/2026-07-02-adr-eco-003b-catalog-distribution.md` in the dev-only sibling workspace (not committed to this repo; referenced as a pointer, not a link — see CLAUDE.md on `../_cowork_output/`)
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
5. **Docs**: update ONLY **module/path references** (`atp/catalog/`) and **import references**
   (`atp.catalog.*`) in `CLAUDE.md`, `README`, `docs/`. **Do NOT** rewrite user-facing
   **`atp catalog` command** mentions or the word "catalog" generally — the CLI command is
   unchanged; only the package path/import moves. (E.g. "run `atp catalog list`" stays; "the
   `atp/catalog/` module" becomes "the `atp/test_catalog/` module".)

## Verification (this replaces feature TDD — the rename is correct iff nothing dangles and the
suite is green)

- **Discover ALL references repo-wide — do not trust the file list above.** The enumeration in
  §Scope is a guide, not the source of truth; the implementer runs `rg` over the whole repo for
  `atp\.catalog` and `atp/catalog` and updates **every** hit (imports and string targets alike),
  because a new caller may have appeared since this spec was written.
- **Zero dangling references (scoped to ACTIVE code/tests/docs):**
  `rg -n 'atp\.catalog\b|atp/catalog\b' --glob '!**/.venv/**' --glob '!docs/superpowers/specs/**'
  --glob '!docs/superpowers/plans/**'` (then drop any `model_catalog` line) must return
  **nothing**. Historical **specs/plans are excluded** (this SP-D spec and older 003b plans
  intentionally reference the old path as history — including "must be nothing" against them
  would false-fail). If a *non-superpowers* doc mentions the old path as history, update it with a
  "(renamed to `atp/test_catalog/` in SP-D)" note rather than deleting the context.
- **Old package is truly gone (not just imports moved):** `test ! -d atp/catalog` (fail if it
  still exists); clear stale bytecode first (`find . -path '*/atp/catalog/*' -name '*.pyc'` and
  `__pycache__` must not resurrect the module).
- **Import smoke:** `uv run python -c "import atp.test_catalog.repository, atp.test_catalog.sync,
  atp.test_catalog.comparison, atp.test_catalog.models"` succeeds; `uv run python -c "import
  atp.catalog"` now **fails** with `ModuleNotFoundError` (run after clearing pycache so a stale
  cache can't mask a missed move).
- **CLI smoke:** `uv run atp catalog --help` still works (command unchanged).
- **Targeted suites green:** `tests/unit/catalog/`, `tests/unit/cli/test_catalog_cli.py`,
  `tests/unit/dashboard/v2/catalog/` — all pass (this is where the string-target breakage would
  surface).
- **Wheel smoke (the `builtin/` data must ship under the new path):** build the root wheel
  (`uv build --wheel -o /tmp/sp-d-wheel`) and assert its namelist contains
  `atp/test_catalog/builtin/…` and **no** `atp/catalog/…` entry — `builtin/` is package data, so
  a move that the wheel doesn't follow would silently drop the built-in suites.
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

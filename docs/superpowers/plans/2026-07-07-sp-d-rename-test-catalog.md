# SP-D rename atp/catalog → atp/test_catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the test-suite-catalog package `atp/catalog/` → `atp/test_catalog/`, updating
every reference (imports AND string mock targets), with no behavior change.

**Architecture:** One atomic `git mv` + an anchored find-and-replace of `atp.catalog` /
`atp/catalog` across active code, tests, and docs — then a verification battery (the existing
test suite is the safety net; there is no new behavior to TDD). The `atp catalog` CLI command,
class names, dashboard route file, and test directory are unchanged.

**Tech Stack:** Python 3.12+, uv, pytest, ruff, pyrefly, hatchling (wheel), `git mv`, `rg`/`sed`.

## Global Constraints

- **uv only**; run tools via `uv run`. `uv run ruff check . && uv run pyrefly check` clean at the
  end. (This task changes no logic, so line-length/type-hint rules are satisfied by the
  unchanged code.)
- **Anchor replacements on `atp.catalog` / `atp/catalog`** — NEVER bare `catalog`. `atp.model_catalog`
  does not contain the substring `atp.catalog`, so the anchored replace cannot corrupt it; the
  space-separated `atp catalog` CLI command also cannot match. Verify with a grep before/after.
- **Do NOT rename:** the `atp catalog` CLI command (click group stays `name="catalog"`,
  `atp/cli/commands/catalog.py` keeps its filename), the classes (`CatalogRepository`,
  `CatalogSuite`, `CatalogTest`, …), `atp/dashboard/v2/routes/catalog.py` (route file), or
  `tests/unit/catalog/` (test dir).
- **Historical `docs/superpowers/specs|plans/`** intentionally reference the old path as history
  — **exclude them** from the replace and from the zero-dangling check.
- **Docs: only module/path (`atp/catalog/`) and import (`atp.catalog.*`) references** change —
  the anchored replace does exactly this and leaves `atp catalog` command mentions alone.

Spec: `docs/superpowers/specs/2026-07-07-sp-d-rename-test-catalog-design.md`.

---

## File Structure

The package moves wholesale; consumers get their references rewritten:

| Path | Change |
|---|---|
| `atp/catalog/` → `atp/test_catalog/` | `git mv` (incl. `builtin/` data; `BUILTIN_DIR` is `Path(__file__).parent/"builtin"`, follows the move) |
| `atp/test_catalog/{sync,repository}.py` | intra-package imports rewritten |
| `atp/cli/commands/catalog.py` | 3 imports rewritten (file + command name unchanged) |
| `packages/atp-dashboard/atp/dashboard/v2/routes/catalog.py` | 2 imports rewritten (route file unchanged) |
| `tests/unit/catalog/*.py`, `tests/unit/cli/test_catalog_cli.py`, `tests/unit/dashboard/v2/catalog/test_routes.py` | imports + **string mock targets** + docstrings rewritten |
| `CLAUDE.md`, `README`, `docs/**` (excl. `docs/superpowers/**`) | path/import refs rewritten; TODO.md 003b-complete tick |

---

## Task 1: Rename the package + rewrite every reference + verify

**Files:** the whole set above (one atomic commit).

**Interfaces:** none produced/consumed — this is an internal rename. External API surface
(`atp catalog` CLI, class names) is unchanged.

- [ ] **Step 1: Baseline — capture the current reference set**

Run:
```bash
rg -n 'atp\.catalog\b|atp/catalog\b' --glob '!**/.venv/**' --glob '!docs/superpowers/**' | grep -v model_catalog | tee /tmp/sp-d-before.txt
```
Expected: a non-empty list (imports in `atp/cli/commands/catalog.py`, the dashboard route, the
intra-package files, and the tests incl. `mock.patch("atp.catalog…")` string targets). This is
the exact set that must become zero after the rename.

- [ ] **Step 2: Move the package**

Run:
```bash
git mv atp/catalog atp/test_catalog
find . -path '*/atp/catalog/*' -prune -o -name '__pycache__' -type d -print 2>/dev/null | grep -i catalog || true
# clear any stale bytecode that could mask a missed move:
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
test ! -d atp/catalog && echo "old dir gone" || echo "ERROR: atp/catalog still exists"
```
Expected: `old dir gone`.

- [ ] **Step 3: Anchored find-and-replace across active files**

Run (macOS `sed -i ''`; anchored on `atp.catalog` / `atp/catalog`, excludes historical
superpowers docs and `.venv`):
```bash
rg -l 'atp\.catalog\b|atp/catalog\b' --glob '!**/.venv/**' --glob '!docs/superpowers/**' \
  | xargs sed -i '' -e 's#atp\.catalog#atp.test_catalog#g' -e 's#atp/catalog#atp/test_catalog#g'
```
This rewrites Python imports, `mock.patch("atp.catalog…")` **string targets**, test-file
docstrings, and doc path references in one pass. It does NOT touch `atp catalog` (space) or bare
`catalog`.

- [ ] **Step 4: Zero-dangling check (active tree)**

Run:
```bash
rg -n 'atp\.catalog\b|atp/catalog\b' --glob '!**/.venv/**' --glob '!docs/superpowers/**' | grep -v model_catalog
```
Expected: **no output**. If anything remains, fix it (a stray reference the glob/anchor missed)
and re-run until clean. (Historical `docs/superpowers/specs|plans/` legitimately still contain
the old path — that is expected and excluded.)

- [ ] **Step 5: Old package truly gone + import smokes**

Run:
```bash
uv run python -c "import atp.test_catalog.repository, atp.test_catalog.sync, atp.test_catalog.comparison, atp.test_catalog.models; print('new imports OK')"
uv run python -c "import atp.catalog" 2>&1 | tail -1   # expect: ModuleNotFoundError: No module named 'atp.catalog'
```
Expected: `new imports OK`; the second prints a `ModuleNotFoundError` for `atp.catalog`.

- [ ] **Step 6: CLI smoke (command unchanged)**

Run: `uv run atp catalog --help`
Expected: the catalog command help prints normally (subcommands list/run/publish/etc.).

- [ ] **Step 7: Targeted suites — this is where a missed string target surfaces**

Run:
```bash
uv run pytest tests/unit/catalog/ tests/unit/cli/test_catalog_cli.py tests/unit/dashboard/v2/catalog/ -q
```
Expected: all pass. A `mock.patch` with a stale `"atp.catalog…"` target would raise
`ModuleNotFoundError`/`AttributeError` here — if so, that string was missed in Step 3; fix and
re-run.

- [ ] **Step 8: Wheel smoke — `builtin/` data ships under the new path**

Run:
```bash
uv build --wheel -o /tmp/sp-d-wheel 2>/dev/null
uv run python - <<'PY'
import glob, zipfile
whl = glob.glob("/tmp/sp-d-wheel/*.whl")[0]
names = zipfile.ZipFile(whl).namelist()
new = [n for n in names if "/atp/test_catalog/builtin/" in n or n.startswith("atp/test_catalog/builtin/")]
old = [n for n in names if "/atp/catalog/" in n or n.startswith("atp/catalog/")]
assert new, "no atp/test_catalog/builtin/ entries in the wheel!"
assert not old, f"stale atp/catalog/ entries in the wheel: {old[:3]}"
print(f"OK: {len(new)} builtin entries under atp/test_catalog/, no atp/catalog/ left")
PY
```
Expected: `OK: N builtin entries under atp/test_catalog/, no atp/catalog/ left`.

- [ ] **Step 9: Type + lint clean**

Run: `uv run ruff check . && uv run pyrefly check`
Expected: clean / 0 errors.

- [ ] **Step 10: TODO.md — mark the 003b epic complete**

In `TODO.md`, tick SP-D done (link the spec + this plan) and mark **ADR-003b epic complete**
(SP-A/SP-E/SP-C/SP-D all shipped). (CLAUDE.md/docs path references were already rewritten by the
Step-3 replace — verify the CLAUDE.md component-25 / architecture mention now reads
`atp/test_catalog/` and reads sensibly; adjust wording only if the mechanical replace produced an
awkward sentence.)

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "refactor(catalog): rename atp/catalog -> atp/test_catalog (ADR-003b SP-D)"
```
(Use `git add -A` so the `git mv` rename, the rewritten references, and the docs land as one
atomic commit — the tree is only consistent when all move together.)

---

## Self-Review

**Spec coverage:**
- `git mv atp/catalog → atp/test_catalog` (incl. `builtin/`) → Step 2. ✓
- Python imports rewritten (cli, dashboard route, intra-package, tests) → Step 3. ✓
- **String mock targets** rewritten (the correctness trap) → Step 3 (same anchored replace hits
  strings) + Step 7 (targeted suites catch a miss). ✓
- Docstrings/module headers → Step 3. ✓
- Docs path/import refs only (not the `atp catalog` command) → anchored replace (Step 3) +
  Global Constraints. ✓
- Discover repo-wide, don't trust a list → Step 1 + Step 3 both use `rg`/glob over the tree. ✓
- Zero-dangling scoped to active tree, historical specs/plans excluded → Step 4. ✓
- Old dir gone + pycache cleared → Step 2 + Step 5. ✓
- Positive + negative import smoke → Step 5. ✓
- CLI smoke → Step 6. ✓
- Wheel `builtin/` smoke → Step 8. ✓
- Non-goals (classes, CLI command, route file, test dir untouched) → Global Constraints; the
  anchored replace cannot touch class names or `atp catalog`. ✓

**Placeholder scan:** no "TBD/handle errors". The only non-literal is Step 10's "adjust wording
only if the mechanical replace produced an awkward sentence" — a judgment the implementer makes
against the actual rendered CLAUDE.md line, not a vague deferral.

**Type consistency:** no new types/signatures — this is a pure path rename. The single naming
invariant (`atp.catalog` → `atp.test_catalog`, `atp/catalog` → `atp/test_catalog`, everything
else unchanged) is applied uniformly by the anchored replace and checked by Steps 4-9.

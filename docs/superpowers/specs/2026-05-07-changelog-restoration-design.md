# CHANGELOG Restoration & v2.0.0 Migration Guides — Design

**Status:** rev 2 (post-review)
**Date:** 2026-05-07
**Owner:** Andrei Shtanakov
**Tracks:** TASK-901 (Phase 4 — last open item)

## 1. Goal

Restore `CHANGELOG.md` as a maintained, machine-checkable artefact, document the
breaking changes shipped between `v1.0.0` and now, and bump the **root /
platform / CLI** to `v2.0.0`. Sub-packages (`packages/atp-*`) keep their own
independent versions and CHANGELOGs (see §3).

After this work lands, the project has:

- Up-to-date `CHANGELOG.md` (root) with `## [Unreleased]` and `## [2.0.0] -
  YYYY-MM-DD` (date placeholder, see §3 / §7).
- Fresh empty `## [Unreleased]` headers in `packages/atp-*/CHANGELOG.md` so
  every distributable has an obvious place to land per-package entries.
- Migration guides under `docs/migrations/` — one per breaking change found by
  the audit step (initial scope: three known items, see §2).
- A CI gate that fails any PR labelled `breaking` or `feat` without an
  addition under `## [Unreleased]` in `CHANGELOG.md`.
- Root `pyproject.toml` bumped to `2.0.0`.
- `CONTRIBUTING.md` paragraph documenting the changelog rule, sub-package
  routing, and tagging procedure.

Out of scope: PyPI publish workflow (separate ticket if missing), retroactive
reconstruction of `[1.0.x]` entries, multi-language migration guides.

## 2. Context

`CHANGELOG.md` is currently frozen at `## [1.0.0] - 2026-02-13` (68 lines), even
though three known breaking changes have shipped since:

1. **El Farol action format** — flat `{"slots": [...]}` replaced with
   interval-based `{"intervals": [[start, end], ...]}` (PR #105).
2. **El Farol default scoring** — default `scoring_mode` flipped from
   `happy_minus_crowded` (ratio) to `happy_only` (count, no penalty); legacy
   mode remains opt-in for tests but is not exposed via the tournament API
   (PR #121).
3. **MCP tool gating** — tournament MCP tools require explicit `purpose`
   declaration in the auth handshake (commit `d0f11e2`, LABS-TSA).

These three are *the audit starting point, not the answer.* Implementation
must run a full `git log v1.0.0..HEAD` audit (see §9 step 1) — the v1.0.0 cut
is almost three months back and other features in that window (RBAC + invite
system, agent-scoped vs. user-level token prefixes, MCP tournament server,
container-isolated code-exec evaluator, benchmark API event streaming) may
have broken public contracts and need their own bullet + migration guide.

The participant-kit-el-farol-en wire contract is now public (PR #85), so we
have external consumers who need precise migration steps for at least the
El Farol changes.

## 3. Approach summary

One PR. Five artefacts:

1. `CHANGELOG.md` (root) — restore `[Unreleased]` and write `[2.0.0]`.
2. `packages/atp-*/CHANGELOG.md` — create empty `[Unreleased]` skeletons for
   each distributable that doesn't already have one.
3. `docs/migrations/<date>-<topic>.md` — one file per audited breaking change.
4. Root `pyproject.toml` — bump version to `2.0.0`.
5. `scripts/ci/check_changelog.sh` + `.github/workflows/changelog.yml`.
6. `CONTRIBUTING.md` paragraph (changelog rule + sub-package routing +
   tagging).

### Decisions locked in this spec

- **Granularity** — minimum-viable. `[2.0.0]` documents only audited breaking
  changes in detail; non-breaking work collapses into a one-paragraph summary
  pointing readers at `git log v1.0.0..v2.0.0`.
- **Migration guides** — separate files in `docs/migrations/`, English only.
- **CI enforcement** — GitHub Actions check, scoped to PRs labelled `breaking`
  or `feat`. Non-feature PRs (chores, bugfixes, docs) are not forced to touch
  the changelog. Known soft bypass — see §6 limitation note.
- **Sub-package versioning policy — INDEPENDENT.** Each `packages/atp-*` has
  its own version and its own `CHANGELOG.md`. Root `version` in
  `pyproject.toml` describes the platform / CLI version, not aggregate. The
  root CHANGELOG documents *root-shipped* breaking changes (CLI, dashboard
  routes, MCP, El Farol engine, etc.). SDK ships its own CHANGELOG entries in
  `packages/atp-sdk/CHANGELOG.md`. This codifies the de-facto state where
  `atp-platform-sdk` is already published as 2.0.0 on PyPI while the root
  pyproject is still at 1.0.0. CONTRIBUTING.md spells out the routing rule.
- **Release date** — `## [2.0.0] - YYYY-MM-DD` is **a placeholder** in this
  PR. The maintainer fills the real date at tag time (see §6 CONTRIBUTING.md
  procedure).
- **Pre-flight ancestor check is mandatory** — see §9 step 1.

## 4. CHANGELOG.md structure (root)

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - YYYY-MM-DD

### Breaking Changes

- **El Farol action format**: replaced flat `{"slots": [...]}` with
  interval-based `{"intervals": [[start, end], ...]}`. Old format is now
  rejected; `sanitize` coerces invalid input to a safe action.
  See `docs/migrations/2026-04-el-farol-intervals.md`. (#105)
- **El Farol default scoring**: default `scoring_mode` flipped from
  `happy_minus_crowded` (ratio of happy to crowded slots) to `happy_only`
  (raw count of happy slots, no penalty for crowded). Tournaments use the new
  default; legacy mode is opt-in via `ElFarolConfig(scoring_mode=...)` and not
  exposed through the tournament API.
  See `docs/migrations/2026-05-el-farol-scoring.md`. (#121)
- **MCP tournament tools**: now require an explicit `purpose` claim in the
  auth handshake. Tools reject calls without it.
  See `docs/migrations/2026-04-mcp-purpose-gating.md`. (commit d0f11e2)

<!-- Additional bullets land here after the §9 step-1 audit. -->

### Added / Changed / Fixed

For non-breaking changes between 1.0.0 and 2.0.0, see `git log v1.0.0..v2.0.0`.
Highlights: pending-tournament banner, El Farol winners dashboard + Hall of
Fame, benchmark API event streaming, agent ownership quotas, RBAC + invite
system, container-isolated code-exec evaluator, MCP tournament server.

## [1.0.0] - 2026-02-13

(unchanged — preserved verbatim from the existing CHANGELOG.md)

[Unreleased]: https://github.com/<org>/atp-platform/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/<org>/atp-platform/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/<org>/atp-platform/releases/tag/v1.0.0
```

`<org>` is filled in during implementation by reading `git remote get-url
origin`.

### Sub-package CHANGELOGs

For every `packages/atp-*/` that doesn't already have one, create a minimal
`CHANGELOG.md`:

```markdown
# Changelog — <package-name>

This package follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Routing rule: only changes that affect this package's public API land here.
Platform-level changes go in the root `CHANGELOG.md`.

## [Unreleased]

## [<current-version>] - <release-date-from-pyproject-or-tag>

(seeded from existing release; if package was never released, omit this
section.)
```

`atp-platform-sdk` already has 2.0.0 on PyPI — its `CHANGELOG.md` (if absent)
opens with `## [2.0.0]` and a one-line summary linking to the SDK changes
between 1.x and 2.0.

## 5. Migration guides

Location: `docs/migrations/`. Filename pattern:
`<YYYY-MM>-<short-topic>.md` (e.g., `2026-04-el-farol-intervals.md`).

### Template

```markdown
# Migration: <topic>

**Affected versions:** before <PR-merge-date> → after
**Affected components:** <which API/contract>
**PR / commit:** #NNN

## What changed

<2–3 sentences>

## Why

<1–2 sentences>

## Before

<minimal example of old usage>

## After

<minimal example of new usage>

## How to migrate

<2–4 step-by-step bullets>

## Backward compatibility

<opt-in legacy mode? sanitize behaviour? deprecation timeline?>
```

### `2026-04-el-farol-intervals.md`

- **What changed:** action shape `{"slots": [...]}` → `{"intervals": [[s,e]]}`.
- **Constraints:** ≤ 2 intervals per day, ≤ `MAX_SLOTS_PER_DAY = 8` slots
  total, intervals must not overlap or touch (≥ 1 empty slot between).
  `{"intervals": []}` (or `[]`) means "stay home".
- **Backward compatibility:** old format is no longer accepted; `sanitize`
  coerces invalid input to a safe (empty) action.
- **PR:** #105.

### `2026-05-el-farol-scoring.md`

- **What changed:** default `scoring_mode` flipped from `happy_minus_crowded`
  to `happy_only`.
- **Old default (`happy_minus_crowded`):** per-day payoff = `happy − crowded`;
  final = `t_happy / max(t_crowded, 0.1)`.
- **New default (`happy_only`):** per-day payoff = number of happy slots;
  final = `t_happy`. No penalty for crowded slots.
- **Backward compatibility:** legacy mode is opt-in via
  `ElFarolConfig(scoring_mode="happy_minus_crowded")` (tests, atp-games
  standalone scenarios). Not available through the tournament API.
- **PR:** #121.

### `2026-04-mcp-purpose-gating.md`

**Hard requirement (no placeholder content allowed at PR time).** Before
writing this guide, the implementer MUST:

1. `git show d0f11e2` and read the full diff.
2. Identify the exact `purpose` claim shape (field name, allowed values,
   where it's read).
3. Write concrete `Before` (no `purpose`) and `After` (with `purpose`)
   handshake examples — actual code, not prose.
4. Document SDK-side behaviour: which SDK version started auto-declaring
   `purpose` and what older clients see.

### Audit-discovered guides

§9 step 1 may surface additional breaking changes. Each gets its own
`docs/migrations/<YYYY-MM>-<topic>.md` file using the same template, plus a
matching bullet in the `[2.0.0] / Breaking Changes` section.

## 6. CI enforcement

### Workflow `.github/workflows/changelog.yml`

```yaml
name: changelog
on:
  pull_request:
    types: [opened, synchronize, reopened, labeled, unlabeled]

jobs:
  require-changelog:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Check for CHANGELOG entry on breaking/feat PRs
        env:
          PR_LABELS: ${{ toJson(github.event.pull_request.labels.*.name) }}
          BASE_SHA: ${{ github.event.pull_request.base.sha }}
          HEAD_SHA: ${{ github.event.pull_request.head.sha }}
        run: bash scripts/ci/check_changelog.sh
```

### Script `scripts/ci/check_changelog.sh`

Logic (full-file parse, not hunk-walking):

1. Parse `$PR_LABELS` (JSON array). If neither `breaking` nor `feat` is
   present → `exit 0`.
2. `git diff --name-only $BASE_SHA $HEAD_SHA` — if `CHANGELOG.md` is not in
   the list → `exit 1` with the error message
   "PR labelled `breaking`/`feat` requires a CHANGELOG.md entry under
   `## [Unreleased]`. Add a bullet to the top section of CHANGELOG.md."
3. Resolve the `[Unreleased]` line range from the post-merge file:
   - `git show "$HEAD_SHA:CHANGELOG.md"` (capture full file).
   - Find the line number of `## [Unreleased]` (call it `U_START`).
   - Find the line number of the next `## [` heading (call it `U_END`); if
     none, set `U_END` = total line count + 1.
   - The `[Unreleased]` section occupies lines `[U_START, U_END)` in the
     post-merge file.
4. Run `git diff --unified=0 "$BASE_SHA" "$HEAD_SHA" -- CHANGELOG.md` and
   parse hunk headers to get the post-image line numbers of every added (`+`)
   line. (`--unified=0` makes hunk headers exact.)
5. If any added line falls inside `[U_START, U_END)` → `exit 0`. Otherwise
   `exit 1` with "PR touched CHANGELOG.md but added no lines under
   `## [Unreleased]`."

This relies on the *post-merge full file structure* rather than hunk-local
context, so the script is correct regardless of where the `## [Unreleased]`
heading sits relative to the diff hunks.

#### Known limitation: label-only enforcement

The gate is a **soft** check. A PR author or maintainer can bypass it by
removing the `feat` / `breaking` label. CONTRIBUTING.md will explicitly call
this out as a deliberate trade-off — the goal is reminder, not authoritarian
gating. If we later need stricter enforcement, the next layer is parsing the
PR title for conventional-commit prefixes (`feat:` / `feat!:` /
`BREAKING CHANGE:` in body) — that's filed as a follow-up, not in this PR.

### `CONTRIBUTING.md` paragraph

Add a "Releases & Changelog" section, with three subsections:

#### 1. Changelog rule

> Every PR labelled `breaking` or `feat` must add a bullet under
> `## [Unreleased]` in `CHANGELOG.md`. Bug fixes are optional but encouraged.
> The `changelog` workflow enforces this on label change — if you remove the
> `feat`/`breaking` label, the gate goes away (this is intentional; the goal
> is a reminder, not a hard wall).

#### 2. Routing: which CHANGELOG?

> - Changes that affect the platform / CLI / dashboard / engines (El Farol,
>   PD, etc.) → root `CHANGELOG.md`.
> - Changes that affect a specific distributable (`atp-platform-sdk`,
>   `game-environments`, etc.) → `packages/<name>/CHANGELOG.md`.
> - When in doubt, write the bullet in both — the root entry can just point
>   to the package one.

#### 3. Cutting a release

> 1. Open a release PR. Replace `## [Unreleased]` with `## [X.Y.Z] - <today>`
>    and add a fresh empty `## [Unreleased]` above it. Update linkbacks at
>    the bottom.
> 2. Bump `version` in the relevant `pyproject.toml`. Root for platform/CLI
>    releases; sub-package `pyproject.toml` for SDK / package releases.
> 3. Merge the release PR.
> 4. `git tag -a vX.Y.Z <merge-commit>` and `git push origin vX.Y.Z`.
> 5. Open the GitHub Release UI for the new tag. Title: `vX.Y.Z`. Body: paste
>    the contents of the `## [X.Y.Z]` section from `CHANGELOG.md` directly —
>    do NOT rely on GitHub's auto-generated notes (they group by PR title +
>    label and ignore CHANGELOG entirely; the result drifts from the actual
>    changelog).
> 6. **PyPI publish is NOT automatic.** Tagging produces only the GitHub
>    Release. Publishing to PyPI requires a separate workflow run (or manual
>    `uv build` + `uv publish` if the workflow is missing). After publishing,
>    smoke-test with `pip install atp-platform==X.Y.Z` from a clean venv.

## 7. Version bump

**Scope: root only.** Sub-packages keep their own independent versions per
the §3 policy.

Files this PR modifies:

- `pyproject.toml` (root) — `version = "1.0.0"` → `"2.0.0"`.
- `tests/test_version.py` (or wherever the version is asserted) — update if
  it asserts a specific value.

Audit step (still required, but only to *find* version references — not to
bump them all):

```
grep -rn '"1\.0\.0"' atp/ packages/ tests/ docs/
grep -rn "'1\.0\.0'" atp/ packages/ tests/ docs/
grep -rn '__version__' atp/ packages/ tests/
grep -rn 'version = ' packages/ pyproject.toml
```

For each match, decide: is this a root-level reference that needs bumping, or
a sub-package reference that stays put? Document the decisions inline in the
plan file so the implementer doesn't re-derive them.

## 8. Testing

### `scripts/ci/check_changelog.sh` — pytest

`tests/ci/test_check_changelog.py`. Each case builds a tiny git repo in
`tmp_path`, makes commits with crafted CHANGELOG diffs, then invokes the
script via `subprocess` with `BASE_SHA` / `HEAD_SHA` / `PR_LABELS` set in
`env`. Cases:

| # | Labels | CHANGELOG diff | Expected exit |
|---|---|---|---|
| 1 | `[]` | none | 0 |
| 2 | `["chore"]` | none | 0 |
| 3 | `["bug"]` | none | 0 |
| 4 | `["feat"]` | none | 1 |
| 5 | `["breaking"]` | adds line under `[Unreleased]` | 0 |
| 6 | `["feat"]` | adds line under `[Unreleased]` | 0 |
| 7 | `["feat"]` | edits only `[1.0.0]` section | 1 |
| 8 | `["feat", "breaking"]` | adds line under `[Unreleased]` | 0 |
| 9 | `["feat"]` | adds line under `[2.0.0]` (not `[Unreleased]`) | 1 |
| 10 | `["feat"]` | adds line, but only as a typo fix in `[1.0.0]` | 1 |

Cases 9–10 specifically guard the §6 step-3-to-5 logic against false
positives where the diff touches CHANGELOG.md but lands outside `[Unreleased]`.

### Release artefacts — pytest

`tests/docs/test_release_artifacts.py`:

- `CHANGELOG.md` contains both `## [Unreleased]` and `## [2.0.0]`.
- `docs/migrations/` exists; every file in it ≥ 30 lines and contains the
  section headings: "What changed", "Before", "After", "How to migrate".
- `packages/atp-*/CHANGELOG.md` exists and contains `## [Unreleased]`.

### actionlint

Add a one-shot `actionlint .github/workflows/changelog.yml` invocation as
part of `Self-review pass` (§9 step 6). If the repo already has an actionlint
job in another workflow, rely on it; if not, run `actionlint` locally before
opening the PR. Do NOT add a new actionlint workflow in this PR — that's a
separate concern.

### Standard CI

`uv run pyrefly check`, `uv run ruff check`, `uv run pytest tests/ci tests/docs`.

## 9. Implementation order

For the writing-plans skill to expand into bite-sized tasks:

### Step 1 — Pre-flight audit (gating)

Before touching any file, the implementer must confirm and record:

1. **Ancestor check** — every PR/commit cited in `[2.0.0] / Breaking
   Changes` is reachable from `HEAD`:
   ```
   git merge-base --is-ancestor <commit-of-PR-105>  HEAD
   git merge-base --is-ancestor <commit-of-PR-121>  HEAD
   git merge-base --is-ancestor d0f11e2             HEAD
   ```
   Any failure → STOP. Do not write the CHANGELOG bullet for an unmerged
   change.
2. **Full breaking-change audit** between `v1.0.0` and `HEAD`:
   ```
   git log v1.0.0..HEAD --oneline | wc -l
   git log v1.0.0..HEAD --grep='BREAKING\|breaking\|breaking-change' --oneline
   git log v1.0.0..HEAD --oneline -- 'packages/atp-sdk/**'
   git log v1.0.0..HEAD --oneline -- 'atp/dashboard/v2/routes/**'
   git log v1.0.0..HEAD --oneline -- 'atp/dashboard/mcp/**'
   git log v1.0.0..HEAD --oneline -- 'game-environments/game_envs/games/**'
   ```
   Areas to inspect by hand for breakage signals: tournament API contract
   (`atp/dashboard/tournament/`), agent management & token prefixes
   (`atp/dashboard/v2/routes/agent_management_api.py`, `tokens.py`), MCP
   server tools (`atp/dashboard/mcp/`), El Farol engine
   (`game-environments/game_envs/games/el_farol.py`), SDK public surface
   (`packages/atp-sdk/`).
3. **Result of audit recorded in the plan file** as either:
   - "Audit complete — confirmed list is exactly the three known items
     [intervals, scoring, MCP-purpose]; no other breaking changes found.",
     or
   - "Audit complete — found N additional breaking changes: [list]. Each
     gets a bullet in CHANGELOG `[2.0.0] / Breaking Changes` and a
     `docs/migrations/<date>-<topic>.md` guide."

This step is mandatory and gates everything else. The plan must NOT proceed
to step 2 without a completed audit recorded.

### Step 2 — Read the MCP gating commit

`git show d0f11e2` — capture the exact `purpose` field shape, allowed values,
and SDK auto-declare logic. These details land in
`docs/migrations/2026-04-mcp-purpose-gating.md` in step 4. No placeholder
content allowed.

### Step 3 — Version bump (root only)

- Edit root `pyproject.toml` `version` from `1.0.0` to `2.0.0`.
- Run audit greps from §7. For each hit, decide bump-vs-leave per the §3
  policy. Record decisions in the plan.
- Update `tests/test_version.py` if it pins a version.

### Step 4 — CHANGELOG.md restoration

- Add `## [Unreleased]` and `## [2.0.0] - YYYY-MM-DD` (literal placeholder).
- Bullets: the three known breaks plus any audit-discovered breaks from
  step 1.
- Preserve `## [1.0.0] - 2026-02-13` verbatim.
- Add linkbacks at the bottom.

### Step 5 — Sub-package CHANGELOGs

For each `packages/atp-*/`:

- If `CHANGELOG.md` exists, prepend `## [Unreleased]` if missing.
- If absent, create from the §4 template, seeded with the package's current
  pyproject version (or omit `[<current-version>]` if package was never
  released).

### Step 6 — Migration guides

`docs/migrations/`:

- `2026-04-el-farol-intervals.md` (content from §5).
- `2026-05-el-farol-scoring.md` (content from §5).
- `2026-04-mcp-purpose-gating.md` (concrete content from step 2).
- One file per audit-discovered break.

### Step 7 — CI gate

- `scripts/ci/check_changelog.sh` per §6 algorithm.
- `.github/workflows/changelog.yml` per §6 YAML.
- `tests/ci/test_check_changelog.py` per §8 test matrix.

### Step 8 — CONTRIBUTING.md

The three subsections from §6 (Changelog rule, Routing, Cutting a release).

### Step 9 — Self-review pass

- `uv run ruff check` / `uv run pyrefly check`.
- `uv run pytest tests/ci tests/docs -v`.
- `actionlint .github/workflows/changelog.yml`.
- Run the bash script locally against a recent merged PR's diff (e.g., #121
  squashed merge) to confirm no false positives.

Each step is its own commit and its own task in the plan.

## 10. Risks and mitigations

- **Risk:** Cited PR/commit isn't merged at audit time → CHANGELOG promises
  a change that doesn't exist.
  **Mitigation:** Step-1 ancestor check is mandatory; STOP-on-fail. Plan must
  record audit results before any other step.
- **Risk:** Audit misses a breaking change → external consumer hits an
  unpromised break.
  **Mitigation:** Step-1 full `git log v1.0.0..HEAD` audit covering all
  high-risk paths (SDK, tournament API, MCP, El Farol engine). Each
  discovered change gets a bullet + migration guide.
- **Risk:** `[Unreleased]` detector misfires when the section is far from
  the diff hunk.
  **Mitigation:** §6 algorithm now uses full-file `git show
  $HEAD_SHA:CHANGELOG.md` parsing + post-image line-number intersection,
  not hunk-walking. Test matrix cases 9–10 cover the boundary.
- **Risk:** Sub-package version drift confuses readers (root 2.0.0 but SDK
  also 2.0.0 with different semantics).
  **Mitigation:** §3 policy is independent versioning. CONTRIBUTING.md §2
  spells out the routing rule. Each `packages/atp-*/CHANGELOG.md` is its
  own source of truth for that package.
- **Risk:** MCP migration guide ships with placeholder content.
  **Mitigation:** §5 marks this guide as having a hard requirement; §9
  step 2 is gating and must precede step 6.
- **Risk:** Date drift between CHANGELOG and tag.
  **Mitigation:** §4 uses `YYYY-MM-DD` literal placeholder. Maintainer fills
  the real date during release per CONTRIBUTING.md §3 step 1.
- **Risk:** Label-only gate is bypassable.
  **Mitigation:** Documented as a deliberate trade-off in §6. Stricter
  enforcement (conventional-commit title parsing) is out of scope.
- **Risk:** PyPI users `pip install atp-platform==2.0.0` after tag and get
  404 because publish wasn't run.
  **Mitigation:** CONTRIBUTING.md §3 step 6 explicitly calls out that
  tagging does NOT publish. A separate tracking ticket (out of scope here)
  should automate PyPI publish on tag if desired.
- **Risk:** GitHub UI auto-generated release notes diverge from CHANGELOG.
  **Mitigation:** CONTRIBUTING.md §3 step 5 says paste the `[X.Y.Z]`
  section verbatim, do NOT rely on auto-generated notes.

## 11. Open questions

None at design time. All previously raised review concerns resolved in this
revision.

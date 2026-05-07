# CHANGELOG Restoration & v2.0.0 Migration Guides — Design

**Status:** draft
**Date:** 2026-05-07
**Owner:** Andrei Shtanakov
**Tracks:** TASK-901 (Phase 4 — last open item)

## 1. Goal

Restore `CHANGELOG.md` as a maintained, machine-checkable artefact, document the
three breaking changes shipped between `v1.0.0` and now, and bump the project to
`v2.0.0`. After this work lands, the project has:

- A current `CHANGELOG.md` with `## [Unreleased]` and `## [2.0.0] - 2026-05-07`.
- Three migration guides under `docs/migrations/`.
- A CI gate that fails any PR labelled `breaking` or `feat` without a CHANGELOG
  diff under `## [Unreleased]`.
- `pyproject.toml` (root + sub-packages where applicable) bumped to `2.0.0`.
- `CONTRIBUTING.md` paragraph documenting the changelog rule and tagging
  process.

Out of scope: PyPI publish workflow (separate ticket if missing), retroactive
reconstruction of `[1.0.x]` entries, multi-language migration guides.

## 2. Context

`CHANGELOG.md` is currently frozen at `## [1.0.0] - 2026-02-13` (68 lines), even
though three breaking changes have shipped since:

1. **El Farol action format** — flat `{"slots": [...]}` replaced with
   interval-based `{"intervals": [[start, end], ...]}` (PR #105).
2. **El Farol default scoring** — default `scoring_mode` flipped from
   `happy_minus_crowded` (ratio) to `happy_only` (count, no penalty); legacy
   mode remains opt-in for tests but is not exposed via the tournament API
   (PR #121).
3. **MCP tool gating** — tournament MCP tools require explicit `purpose`
   declaration in the auth handshake (commit `d0f11e2`, LABS-TSA).

The participant-kit-el-farol-en wire contract is now public (PR #85), so we have
external consumers who need precise migration steps for at least the El Farol
changes.

## 3. Approach summary

One PR. Three artefacts:

1. `CHANGELOG.md` — restore `[Unreleased]` and write `[2.0.0]`.
2. `docs/migrations/` — three short guides, one per breaking change.
3. `pyproject.toml` — bump to `2.0.0` (root + sub-packages where applicable).

Plus enforcement:

4. `scripts/ci/check_changelog.sh` + `.github/workflows/changelog.yml`.
5. `CONTRIBUTING.md` paragraph.

Decisions previously made through brainstorming:

- **Granularity:** minimum-viable. `[2.0.0]` documents only the three breaking
  changes in detail; everything else collapses into a one-paragraph summary
  pointing readers at `git log v1.0.0..v2.0.0`.
- **Migration guides:** three separate files in `docs/migrations/`, English
  only. Single file per topic survives well as commits add or revise content.
- **CI enforcement:** GitHub Actions check, scoped to PRs labelled `breaking`
  or `feat`. Non-feature PRs (chores, bugfixes, docs) are not forced to touch
  the changelog.
- **Release strategy:** this PR does the prep work only. Tag `v2.0.0` and
  GitHub Release (with notes auto-generated from `CHANGELOG.md`) are nailed
  down by the maintainer in a follow-up step documented in `CONTRIBUTING.md`.

## 4. CHANGELOG.md structure

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-05-07

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

`<org>` is filled in during implementation (read `git remote get-url origin`).

## 5. Migration guides

Location: `docs/migrations/`. Three new files, one shared template.

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

- **What changed:** MCP tournament tools require explicit `purpose` declaration
  in the auth handshake. Calls without it return 401.
- **Exact `purpose` values and handshake shape:** confirmed by reading commit
  `d0f11e2` during implementation (placeholder until plan-phase).
- **Backward compatibility:** none — old clients must update to declare
  `purpose`. SDK ≥ 2.0.0 declares it automatically.
- **Commit:** `d0f11e2` (LABS-TSA).

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

Logic:

1. Parse `$PR_LABELS` (JSON array). If neither `breaking` nor `feat` is
   present → `exit 0`.
2. `git diff --name-only $BASE_SHA $HEAD_SHA` — if `CHANGELOG.md` is not in
   the list → `exit 1` with "PR labelled `breaking`/`feat` requires a
   CHANGELOG.md entry under `## [Unreleased]`".
3. Confirm the diff actually adds lines under `## [Unreleased]` (not only
   editing `[1.0.0]` or earlier sections). Implementation: parse `git diff
   CHANGELOG.md` hunks; for each `+` line, walk back through hunks to find
   the most recent `## [` heading; reject if the heading is anything other
   than `[Unreleased]`.
4. On success → `exit 0`.

Error messages must include the failing PR number and a one-line fix hint.

### `CONTRIBUTING.md` paragraph

Add a "Releases & Changelog" section:

> Every PR labelled `breaking` or `feat` must add a bullet under
> `## [Unreleased]` in `CHANGELOG.md`. Bug fixes are optional but encouraged.
> When cutting a release, the maintainer renames `[Unreleased]` to the new
> version + date, creates a fresh empty `[Unreleased]`, tags the merge commit
> (`git tag -a vX.Y.Z`), pushes the tag, and uses the GitHub "Generate release
> notes" UI seeded with the `[X.Y.Z]` section content.

## 7. Version bump

Files to update from `"1.0.0"` to `"2.0.0"`:

- `pyproject.toml` (root)
- `packages/atp-core/pyproject.toml` (verify in plan-phase whether it tracks
  root version)
- `packages/atp-adapters/pyproject.toml` (same)
- `packages/atp-dashboard/pyproject.toml` (same)
- `packages/atp-sdk/pyproject.toml` — already at 2.0.0 according to memory
  notes; confirm and skip if true
- Any `__version__` strings inside `atp/` packages (search-driven)
- `tests/test_version.py` if it asserts a specific version

Audit step: `grep -rn '"1\.0\.0"' atp/ packages/ tests/ docs/` plus targeted
checks for `version =`, `__version__`, and `'1.0.0'` (single quotes).

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

### Release artefacts — pytest

`tests/docs/test_release_artifacts.py`:

- `CHANGELOG.md` contains both `## [Unreleased]` and `## [2.0.0]`.
- `docs/migrations/` contains exactly the three expected files.
- Each migration file is ≥ 30 lines and contains the section headings:
  "What changed", "Before", "After", "How to migrate".

### Standard CI

`uv run pyrefly check`, `uv run ruff check`, `uv run pytest tests/ci tests/docs`.
No GitHub Actions e2e — workflow correctness relies on actionlint (already in
the repo's lint workflow if present; otherwise out of scope for this PR).

## 9. Implementation order

For the writing-plans skill to expand into bite-sized tasks:

1. **Audit & version bump** — grep for `"1.0.0"`, update root + sub-package
   `pyproject.toml`, fix `tests/test_version.py` if it asserts the version.
2. **CHANGELOG.md restoration** — add `[Unreleased]` + `[2.0.0]` with the
   three breaking changes; preserve `[1.0.0]` verbatim; add link references
   at the bottom.
3. **Migration guides** — `docs/migrations/` directory + three files.
4. **CI gate** — `scripts/ci/check_changelog.sh` + workflow + pytest tests.
5. **CONTRIBUTING.md** — Releases & Changelog paragraph + tagging procedure.
6. **Self-review pass** — ruff, pyrefly, run the bash script locally against a
   recent PR's diff to make sure it doesn't false-positive.

Each step is its own commit and its own task in the plan.

## 10. Risks and mitigations

- **Risk:** `[Unreleased]` heading detector in `check_changelog.sh` rejects
  legitimate edits (e.g., fixing a typo in `[2.0.0]`).
  **Mitigation:** scope the check to *additions only* (`+` lines). Edits to
  existing sections do not satisfy the "must add line under `[Unreleased]`"
  rule, and the maintainer can override by adding the `chore` label or
  removing `feat`/`breaking`.
- **Risk:** sub-package `pyproject.toml` files have independent release
  cadence and bumping them all to 2.0.0 misrepresents what shipped.
  **Mitigation:** confirm in plan-phase by reading each file's git history;
  if a sub-package is independently versioned, leave it alone and document
  the policy in `CONTRIBUTING.md`.
- **Risk:** `2026-04-mcp-purpose-gating.md` content is stubbed without
  reading commit `d0f11e2`.
  **Mitigation:** the plan must include a "read commit `d0f11e2`" subtask
  before writing the migration guide; do not ship placeholder content.
- **Risk:** GitHub Actions `pull_request` event with `[breaking, feat]`
  labels added after merge does not retrigger the gate.
  **Mitigation:** include `labeled, unlabeled` in `on.pull_request.types` so
  the gate is re-evaluated whenever a label changes (already in the workflow
  above).

## 11. Open questions

None at design time. All resolved through brainstorming.

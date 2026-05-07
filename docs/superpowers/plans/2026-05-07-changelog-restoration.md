# CHANGELOG Restoration & v2.0.0 Migration Guides — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore `CHANGELOG.md`, document v2.0.0 breaking changes with migration guides, bootstrap sub-package CHANGELOGs, add a CI gate, and bump the root project to 2.0.0.

**Architecture:** Single PR. Independent versioning policy: root `pyproject.toml` covers platform/CLI; each `packages/atp-*/CHANGELOG.md` is its own source of truth. Two gating audits up front (commit-ancestor + breaking-change discovery) prevent the changelog from documenting unmerged or missed changes. CI gate parses the post-merge file structure (not hunk-local context) to verify additions land under `## [Unreleased]`.

**Tech Stack:** Bash + GitHub Actions for the gate; pytest + subprocess for the gate's own test suite; Python 3.12 / `uv` toolchain everywhere else; Markdown for changelog and migration guides.

**Spec:** `docs/superpowers/specs/2026-05-07-changelog-restoration-design.md`.

**Working branch:** `docs/changelog-restoration` (already exists; spec rev 2 is at commit `0340ca9`).

---

## File Structure

Files created or modified by this plan, with the responsibility of each:

| Path | Action | Responsibility |
|---|---|---|
| `CHANGELOG.md` | modify | Root changelog. Adds `[Unreleased]` + `[2.0.0]`; preserves `[1.0.0]` verbatim; fixes the `andrei-shtanakov` org in linkbacks. |
| `pyproject.toml` (root) | modify | Bump `version` from `1.0.0` to `2.0.0`. |
| `packages/atp-core/CHANGELOG.md` | create | Per-package changelog skeleton. |
| `packages/atp-adapters/CHANGELOG.md` | create | Per-package changelog skeleton. |
| `packages/atp-dashboard/CHANGELOG.md` | create | Per-package changelog skeleton. |
| `packages/atp-sdk/CHANGELOG.md` | create | Per-package changelog (already at 2.0.0 on PyPI; seeds with `[2.0.0]`). |
| `docs/migrations/2026-04-el-farol-intervals.md` | create | El Farol action format migration. |
| `docs/migrations/2026-05-el-farol-scoring.md` | create | El Farol scoring-mode default migration. |
| `docs/migrations/2026-04-mcp-purpose-gating.md` | create | MCP `purpose` claim handshake migration. Content derived from `git show d0f11e2`. |
| `docs/migrations/<additional>.md` | create (conditional) | Any extra breaking changes surfaced by Task 1's audit. |
| `scripts/ci/check_changelog.sh` | create | Bash script implementing the CI gate logic. |
| `.github/workflows/changelog.yml` | create | GitHub Actions workflow that invokes the script on PR events. |
| `tests/ci/__init__.py` | create | Package marker for the new test directory. |
| `tests/ci/test_check_changelog.py` | create | Pytest suite that drives the bash script via `subprocess` against tiny fixture repos. |
| `tests/docs/__init__.py` | create | Package marker. |
| `tests/docs/test_release_artifacts.py` | create | Pytest suite that asserts CHANGELOG / migration / sub-package files exist and are well-formed. |
| `CONTRIBUTING.md` | modify | Append "Releases & Changelog" section: changelog rule, routing, cutting-a-release procedure. |
| `docs/superpowers/plans/audit-results.md` | create | Append-only artefact recording Task 1 audit findings; not committed if empty. |

Notes:
- `atp-sdk` is already published to PyPI as 2.0.0 but its `CHANGELOG.md` doesn't exist yet — its skeleton seeds with `## [2.0.0]` rather than starting empty.
- Three sub-packages (`atp-core`, `atp-adapters`, `atp-dashboard`) are still on 1.0.0 in their `pyproject.toml` and stay on 1.0.0; their CHANGELOGs open with just `## [Unreleased]` (no historical version section, since they have no public releases of their own).
- The current `CHANGELOG.md` linkback at line 68 reads `https://github.com/anthropics/atp-platform/...` — this is a stale placeholder. Real org per `git remote get-url origin` is `andrei-shtanakov`. Task 4 fixes it.

---

## Task 1: Pre-flight audit (gating)

**Files:**
- Create (transient, may be discarded): `docs/superpowers/plans/audit-results.md`

This task gates everything else. No CHANGELOG entries, no migration guides, no version bumps until both audits below pass and their results are recorded.

- [ ] **Step 1: Confirm working branch**

Run:
```bash
git branch --show-current
```
Expected: `docs/changelog-restoration`. If it prints `main` or anything else, switch:
```bash
git checkout docs/changelog-restoration
```

- [ ] **Step 2: Ancestor check for the three known cited references**

Run each of these and record exit codes:
```bash
git merge-base --is-ancestor f6c14a6 HEAD && echo "PR-105 OK" || echo "PR-105 NOT IN HEAD"
git merge-base --is-ancestor 89f000e HEAD && echo "PR-121 OK" || echo "PR-121 NOT IN HEAD"
git merge-base --is-ancestor d0f11e2 HEAD && echo "MCP-gating OK" || echo "MCP-gating NOT IN HEAD"
```

Note: `f6c14a6` is the squash-merge commit for PR #105 (El Farol intervals). If `git log --grep='#105'` produces a different SHA on this checkout, use that instead. `89f000e` is PR #121's squash merge (already visible in `git log --oneline -3`). `d0f11e2` is the MCP gating commit per the spec.

Expected: all three print `OK`. If any prints `NOT IN HEAD`, STOP. Open the issue with the user — that PR/commit must land on `main` (or be removed from the changelog plan) before continuing.

- [ ] **Step 3: Full breaking-change audit, by area**

Run each of these and capture output:
```bash
git log v1.0.0..HEAD --oneline | wc -l
git log v1.0.0..HEAD --grep='BREAKING\|breaking\|breaking-change' --oneline -i
git log v1.0.0..HEAD --oneline -- 'packages/atp-sdk/**'
git log v1.0.0..HEAD --oneline -- 'atp/dashboard/v2/routes/**' 'packages/atp-dashboard/atp/dashboard/v2/routes/**'
git log v1.0.0..HEAD --oneline -- 'atp/dashboard/mcp/**' 'packages/atp-dashboard/atp/dashboard/mcp/**'
git log v1.0.0..HEAD --oneline -- 'atp/dashboard/tournament/**' 'packages/atp-dashboard/atp/dashboard/tournament/**'
git log v1.0.0..HEAD --oneline -- 'game-environments/game_envs/games/**'
git log v1.0.0..HEAD --oneline -- 'atp/protocol/**' 'packages/atp-core/atp/protocol/**'
```

For each commit returned by the area-specific queries, decide: did this break a public contract (changed CLI flag default, changed API request/response shape, changed wire format, changed scoring/reward semantics, changed auth requirement)? If yes, it's a breaking change.

Public contracts to inspect by hand for breakage signals:
- Tournament API request/response models (`Tournament*` Pydantic schemas).
- Agent management & token prefixes (`atp_a_` vs `atp_u_`) — was this introduced as breaking?
- MCP server tool surface (any new required fields beyond `purpose`).
- El Farol engine config defaults and action shape.
- SDK public surface (`AsyncATPClient`, `ATPClient` method signatures, `BenchmarkRun` interface).

- [ ] **Step 4: Record audit results**

Write the findings to `docs/superpowers/plans/audit-results.md`. Use this exact structure:

```markdown
# v2.0.0 Audit Results

**Audit run on:** <ISO date>
**HEAD commit:** <git rev-parse HEAD>
**Base tag:** v1.0.0

## Ancestor check

- PR #105 (El Farol intervals) — commit `<sha>`: OK / NOT IN HEAD
- PR #121 (El Farol scoring) — commit `<sha>`: OK / NOT IN HEAD
- MCP gating — commit `d0f11e2`: OK / NOT IN HEAD

## Breaking-change discovery

Total commits in v1.0.0..HEAD: <N>

### Confirmed breaking changes

1. **El Farol action format** — PR #105, commit `<sha>`. Public-contract impact: <details>.
2. **El Farol scoring default** — PR #121, commit `89f000e`. Public-contract impact: <details>.
3. **MCP purpose gating** — commit `d0f11e2`. Public-contract impact: <details>.
<additional items if any>

### Inspected but NOT breaking (audit log)

- <area>: <commit> — <one-line reason>
- ...

## Decision

Either:
- "Audit complete — confirmed list is exactly the three known items; no other breaking changes found."
- "Audit complete — found N additional breaking changes: [list]. Each gets a bullet in CHANGELOG `[2.0.0] / Breaking Changes` and a `docs/migrations/<date>-<topic>.md` guide."
```

- [ ] **Step 5: Commit audit results (only if non-empty findings)**

If the audit added rows beyond the three known items, commit the file:
```bash
git add docs/superpowers/plans/audit-results.md
git commit -m "docs(plan): record v2.0.0 breaking-change audit results"
```
If the audit confirmed exactly the three known items and you want to keep `audit-results.md` as a record anyway, commit it. If the file is purely scratch, you may leave it uncommitted — but the *decision line* MUST be transcribed verbatim into the PR description before opening the PR.

---

## Task 2: Capture MCP `purpose` handshake details

**Files:**
- Read-only: commit `d0f11e2`
- Capture findings into a scratch file (used in Task 8): `/tmp/mcp-gating-findings.md`

This task gates Task 8 (MCP migration guide). No placeholder content allowed.

- [ ] **Step 1: Read the gating commit in full**

Run:
```bash
git show d0f11e2
git show d0f11e2 --stat
```
If the commit is large, also run:
```bash
git show d0f11e2 -- atp/dashboard/mcp/ packages/atp-dashboard/atp/dashboard/mcp/
git show d0f11e2 -- packages/atp-sdk/
```

- [ ] **Step 2: Identify the exact contract**

Capture, in `/tmp/mcp-gating-findings.md`, the four facts the migration guide needs:

```
1. Where is `purpose` read on the server?
   Path: <file:line>
   Code excerpt (3-5 lines):
   ```python
   <excerpt>
   ```

2. What is the exact field shape?
   - Field name: <exact name, e.g. "purpose">
   - Location in the auth handshake: <header? request body? MCP init params?>
   - Allowed values: <list, or "any non-empty string", or whatever it is>
   - Validation rule: <required? regex? allow-list?>

3. What does an old (no-purpose) client see?
   - HTTP status: <e.g. 401>
   - Error body shape: <excerpt>

4. SDK behaviour:
   - Which atp-platform-sdk version started auto-declaring `purpose`? <e.g. 2.0.0>
   - Which file in the SDK does this? <packages/atp-sdk/...:line>
   - Code excerpt (3-5 lines).
```

This file is scratch — it lives in `/tmp/`, not in the repo. It feeds Task 8.

---

## Task 3: Bump root version

**Files:**
- Modify: `pyproject.toml` (line 3)

Sub-packages keep their own versions per the §3 policy (atp-core / atp-adapters / atp-dashboard stay at 1.0.0; atp-sdk stays at 2.0.0).

- [ ] **Step 1: Audit version references**

Run:
```bash
grep -rn '"1\.0\.0"' atp/ packages/ tests/ docs/ pyproject.toml 2>/dev/null
grep -rn "'1\.0\.0'" atp/ packages/ tests/ docs/ pyproject.toml 2>/dev/null
grep -rn '__version__' atp/ packages/ 2>/dev/null
```

For each match, classify as one of:
- **Bump** — root-level reference (e.g. root `pyproject.toml`).
- **Leave** — sub-package reference, historical CHANGELOG, doc example, test fixture.

The expected outcome of this audit on the current tree: only `pyproject.toml:3` and (possibly) one or two sub-package pyprojects qualify as "bump candidates", and per the policy, the sub-package ones get *left*.

- [ ] **Step 2: Edit `pyproject.toml`**

Change line 3 from:
```toml
version = "1.0.0"
```
to:
```toml
version = "2.0.0"
```

- [ ] **Step 3: Verify**

Run:
```bash
grep -n '^version' pyproject.toml
```
Expected output: `3:version = "2.0.0"`.

Run:
```bash
uv sync --group dev
uv run python -c "import importlib.metadata as m; print(m.version('atp-platform'))"
```
Expected output: `2.0.0`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore(release): bump root version to 2.0.0"
```

---

## Task 4: Restore CHANGELOG.md (root)

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Read the current file**

```bash
cat CHANGELOG.md
```
Confirm it ends at `[1.0.0]: https://github.com/anthropics/atp-platform/releases/tag/v1.0.0`. The `anthropics` org is wrong; we'll fix it in step 3.

- [ ] **Step 2: Insert the new sections**

Open `CHANGELOG.md`. Right after the third existing line (`All notable changes...`), keep the format-of-this-file blurb but replace lines 1-5 with this exact block (preserve everything from `## [1.0.0] - 2026-02-13` onwards verbatim):

```markdown
# Changelog

All notable changes to the ATP Platform will be documented in this file.

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
  (raw count of happy slots, no penalty for crowded). Tournaments use the
  new default; legacy mode is opt-in via `ElFarolConfig(scoring_mode=...)`
  and not exposed through the tournament API.
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
```

If Task 1 surfaced additional breaking changes, append a bullet for each one under `### Breaking Changes`. Bullet format must match the existing three: bold topic, sentence body, link to migration guide, parenthesised PR/commit ref.

The line `## [2.0.0] - YYYY-MM-DD` keeps the literal placeholder `YYYY-MM-DD`. The maintainer fills it at tag time per CONTRIBUTING.md.

- [ ] **Step 3: Fix linkbacks at the bottom**

Replace the existing single linkback line with these three:

```
[Unreleased]: https://github.com/andrei-shtanakov/atp-platform/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/andrei-shtanakov/atp-platform/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/andrei-shtanakov/atp-platform/releases/tag/v1.0.0
```

Note: org changed from `anthropics` to `andrei-shtanakov` to match `git remote get-url origin`.

- [ ] **Step 4: Visual sanity check**

Run:
```bash
head -50 CHANGELOG.md
tail -10 CHANGELOG.md
grep -c '^## \[' CHANGELOG.md
```
The `grep -c` count must be **3** (Unreleased, 2.0.0, 1.0.0).

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): restore [Unreleased] and add [2.0.0] entry"
```

---

## Task 5: Bootstrap sub-package CHANGELOGs

**Files:**
- Create: `packages/atp-core/CHANGELOG.md`
- Create: `packages/atp-adapters/CHANGELOG.md`
- Create: `packages/atp-dashboard/CHANGELOG.md`
- Create: `packages/atp-sdk/CHANGELOG.md`

- [ ] **Step 1: Create `packages/atp-core/CHANGELOG.md`**

Content:
```markdown
# Changelog — atp-core

This package follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Routing rule: only changes that affect this package's public API land here.
Platform-level changes go in the root `CHANGELOG.md`.

## [Unreleased]
```

- [ ] **Step 2: Create `packages/atp-adapters/CHANGELOG.md`**

Identical content with `atp-core` replaced by `atp-adapters`:
```markdown
# Changelog — atp-adapters

This package follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Routing rule: only changes that affect this package's public API land here.
Platform-level changes go in the root `CHANGELOG.md`.

## [Unreleased]
```

- [ ] **Step 3: Create `packages/atp-dashboard/CHANGELOG.md`**

```markdown
# Changelog — atp-dashboard

This package follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Routing rule: only changes that affect this package's public API land here.
Platform-level changes go in the root `CHANGELOG.md`.

## [Unreleased]
```

- [ ] **Step 4: Create `packages/atp-sdk/CHANGELOG.md`**

This one differs — atp-sdk is already 2.0.0 on PyPI and has historical SDK-level changes. Seed it with both `[Unreleased]` and `[2.0.0]`:

```markdown
# Changelog — atp-platform-sdk

This package follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Routing rule: only changes that affect this package's public API land here.
Platform-level changes go in the root `CHANGELOG.md`.

## [Unreleased]

## [2.0.0] - YYYY-MM-DD

### Added

- `AsyncATPClient` and synchronous `ATPClient` wrapper.
- `BenchmarkRun` async/sync iteration with `next_batch(n)`.
- `emit()` / `emit_sync()` for streaming benchmark-run events.
- Sync convenience methods (`submit_sync`, `status_sync`, `cancel_sync`,
  `leaderboard_sync`, `next_batch_sync`).
- Exponential-backoff retry on transient HTTP errors.

### Changed

- Auto-declare `purpose` in the MCP tournament handshake (see root
  `docs/migrations/2026-04-mcp-purpose-gating.md`).

### Migration

For users upgrading from 1.x, see the root `CHANGELOG.md` `[2.0.0]` entry
and the migration guides under `docs/migrations/`.
```

`YYYY-MM-DD` stays as a placeholder; the maintainer fills it when retroactively tagging the SDK release if desired (low priority — the date is already encoded in the PyPI release).

- [ ] **Step 5: Verify all four files exist and contain `## [Unreleased]`**

```bash
for f in packages/atp-core packages/atp-adapters packages/atp-dashboard packages/atp-sdk; do
  test -f "$f/CHANGELOG.md" && grep -c '## \[Unreleased\]' "$f/CHANGELOG.md" | xargs -I{} echo "$f: {}"
done
```
Each line must end with `: 1`.

- [ ] **Step 6: Commit**

```bash
git add packages/atp-core/CHANGELOG.md packages/atp-adapters/CHANGELOG.md packages/atp-dashboard/CHANGELOG.md packages/atp-sdk/CHANGELOG.md
git commit -m "docs(packages): bootstrap per-package CHANGELOG.md skeletons"
```

---

## Task 6: Migration guide — El Farol intervals

**Files:**
- Create: `docs/migrations/2026-04-el-farol-intervals.md`

- [ ] **Step 1: Create the directory if absent**

```bash
mkdir -p docs/migrations
```

- [ ] **Step 2: Write the file**

Exact content for `docs/migrations/2026-04-el-farol-intervals.md`:

````markdown
# Migration: El Farol action format — `slots` → `intervals`

**Affected versions:** before PR #105 (merged 2026-04) → after
**Affected components:** `game-environments` El Farol engine, tournament action submission, MCP `make_move` tool, `atp-platform-sdk` participant code.
**PR / commit:** #105

## What changed

The El Farol action shape moved from a flat list of slot indices to a list
of `[start, end]` intervals. The old `{"slots": [...]}` payload is no
longer accepted; the engine's `sanitize` step coerces invalid input to a
safe (empty) action.

## Why

Intervals match how humans describe bar visits ("I'll be there from 7 to
9") and let the engine validate constraints (no overlap, no adjacency,
≤ 8 slots, ≤ 2 intervals per day) declaratively. The old flat format
allowed arbitrary slot picks that didn't reflect realistic occupancy.

## Before

```json
{"slots": [3, 4, 5, 10, 11]}
```

## After

```json
{"intervals": [[3, 5], [10, 11]]}
```

The two examples above are equivalent: slots 3-5 and slots 10-11.

## How to migrate

1. Group consecutive slot indices into `[start, end]` pairs (inclusive on both ends).
2. Make sure the result satisfies all constraints:
   - At most 2 intervals per day.
   - At most `MAX_SLOTS_PER_DAY = 8` slots covered in total.
   - Intervals must not overlap.
   - Intervals must not be adjacent — at least one empty slot between them.
3. To "stay home" for the day, send `{"intervals": []}` (or just `[]`).
4. If your code can't build a valid interval list, the engine will fall
   back to the safe "stay home" action via `sanitize`. This is preferable
   to a hard rejection but may surprise agents that expect their (now
   invalid) action to land.

## Backward compatibility

The old `{"slots": [...]}` format is no longer parsed. Tournament servers
that receive it pass through `sanitize`, which produces an empty action
(no slots visited). This silent fallback prevents tournament crashes but
penalises agents that haven't migrated.

SDK ≥ 2.0.0 emits the interval format. Older SDKs need to be upgraded.

## References

- Engine: `game-environments/game_envs/games/el_farol.py`
- Sanitize logic: `el_farol.py:sanitize` (search for the method on the
  `ElFarolBar` class).
- Public participant kit: `participant-kit-el-farol-en` (PR #85).
````

- [ ] **Step 3: Sanity check**

```bash
wc -l docs/migrations/2026-04-el-farol-intervals.md
grep -c '^## ' docs/migrations/2026-04-el-farol-intervals.md
```
The file must be ≥ 30 lines (it is — content above is ~60). The heading count must be ≥ 6 (What changed, Why, Before, After, How to migrate, Backward compatibility, References).

- [ ] **Step 4: Commit**

```bash
git add docs/migrations/2026-04-el-farol-intervals.md
git commit -m "docs(migrations): add 2026-04-el-farol-intervals guide"
```

---

## Task 7: Migration guide — El Farol scoring default

**Files:**
- Create: `docs/migrations/2026-05-el-farol-scoring.md`

- [ ] **Step 1: Write the file**

Exact content for `docs/migrations/2026-05-el-farol-scoring.md`:

````markdown
# Migration: El Farol scoring default — `happy_minus_crowded` → `happy_only`

**Affected versions:** before PR #121 (merged 2026-05) → after
**Affected components:** `game-environments` El Farol engine, tournament
score aggregation, dashboard winners view, anyone interpreting El Farol
scores out of band.
**PR / commit:** #121

## What changed

The default `scoring_mode` for El Farol flipped from `happy_minus_crowded`
(a ratio that penalised crowded-slot visits) to `happy_only` (a raw count
of happy-slot visits, no penalty for crowded slots).

## Why

The ratio formula `t_happy / max(t_crowded, 0.1)` made scores
non-monotonic and hard to interpret — visiting one extra happy slot could
*lower* a player's score if the agent also visited a crowded slot the
same day. Tournament rankings became sensitive to the ε floor (`0.1`).
The new default is monotone, additive, and matches how players intuit
El Farol payoffs ("just count my happy slots").

## Before

Per-day payoff: `happy − crowded`.
Final per-player score: `t_happy / max(t_crowded, 0.1)` (gated by
`min_total_hours`).
Per-day payoff sign: can be negative.
Tournament `total_score`: the final ratio.

## After

Per-day payoff: number of happy slots that day (≥ 0).
Final per-player score: `t_happy` (gated by `min_total_hours`).
Per-day payoff sign: never negative.
Tournament `total_score`: sum of per-day payoffs = `t_happy`.

## How to migrate

1. **Tournament-API users:** no action needed. Tournaments now use
   `happy_only` by default. Existing leaderboards get rebuilt under the
   new scoring on the next tournament cut.
2. **Score-comparing analysis code:** if you previously compared scores
   across El Farol tournaments, do not mix pre- and post-PR-#121 scores
   — they are not on the same scale. Rebuild any cross-tournament
   leaderboards from raw `Action.payoff` rows.
3. **Test code that asserts on legacy ratio behaviour:** opt in
   explicitly:
   ```python
   from game_envs.games.el_farol import ElFarolConfig, ElFarolBar
   config = ElFarolConfig(scoring_mode="happy_minus_crowded", ...)
   game = ElFarolBar(config=config)
   ```
   The legacy mode is preserved for tests and atp-games standalone
   scenarios. It is **not** exposed through the tournament API — opt-in
   only via direct engine construction.
4. **Player observation:** `your_t_crowded_slots` is still surfaced in
   the per-player observation, regardless of `scoring_mode`. Agents can
   still use it as a behaviour signal even though it no longer subtracts
   from their score.

## Backward compatibility

- Legacy mode (`scoring_mode="happy_minus_crowded"`) remains available
  via direct `ElFarolConfig` construction.
- Tournament API does not accept a `scoring_mode` parameter (deliberate
  — production tournaments always use the default).
- The disqualification rule on `min_total_hours` applies identically in
  both modes.

## References

- Engine: `game-environments/game_envs/games/el_farol.py` — search for
  `scoring_mode`.
- Public copy: `atp/dashboard/v2/game_copy.py` — `GAME_COPY["el_farol"]`
  (rules + payoff_formula now describe `happy_only`).
- Public participant kit: `participant-kit-el-farol-en` (PR #85).
````

- [ ] **Step 2: Sanity check**

```bash
wc -l docs/migrations/2026-05-el-farol-scoring.md
grep -c '^## ' docs/migrations/2026-05-el-farol-scoring.md
```
≥ 30 lines, ≥ 6 headings.

- [ ] **Step 3: Commit**

```bash
git add docs/migrations/2026-05-el-farol-scoring.md
git commit -m "docs(migrations): add 2026-05-el-farol-scoring guide"
```

---

## Task 8: Migration guide — MCP `purpose` gating

**Files:**
- Create: `docs/migrations/2026-04-mcp-purpose-gating.md`
- Read: `/tmp/mcp-gating-findings.md` (output of Task 2)

This task **must not** be started until Task 2 is complete and `/tmp/mcp-gating-findings.md` exists with all four facts captured. No placeholder content is acceptable in the final file.

- [ ] **Step 1: Confirm Task 2 output is available**

```bash
test -f /tmp/mcp-gating-findings.md && wc -l /tmp/mcp-gating-findings.md
```
Expected: file exists, ≥ 20 lines.

- [ ] **Step 2: Build the migration guide content**

Open `docs/migrations/2026-04-mcp-purpose-gating.md` and write content following this exact skeleton, substituting `<…>` placeholders with the verbatim values from `/tmp/mcp-gating-findings.md`:

````markdown
# Migration: MCP tournament tools require `purpose` claim

**Affected versions:** before commit `d0f11e2` (LABS-TSA, 2026-04) → after
**Affected components:** ATP MCP tournament server (`/mcp` SSE endpoint),
clients calling `join_tournament`, `make_move`, `get_current_state`, etc.
**PR / commit:** `d0f11e2`

## What changed

MCP tournament tools now require an explicit `<purpose-field-name>` claim
in the auth handshake. Tools reject calls without it with HTTP
`<status-code>`.

## Why

<one-sentence motivation taken from the commit body or PR description —
typically: "purpose-gating prevents accidental use of tournament tools
during exploratory tool listing and gives the server an audit trail of
*what* the client is trying to do, not just *who* is calling.">

## Before

```<lang>
<old-handshake-payload-from-findings>
```

Old clients that omit `<purpose-field-name>` get:
```
HTTP <status-code>
<error-body-from-findings>
```

## After

```<lang>
<new-handshake-payload-from-findings>
```

Allowed `<purpose-field-name>` values: <list-from-findings>.

## How to migrate

1. Update your MCP client to declare `<purpose-field-name>` in
   `<location-from-findings>` (e.g. handshake header, init params).
2. Pick a value that matches your use case from the allowed list:
   <restate-list>.
3. If you use the official Python SDK (`atp-platform-sdk`), upgrade to
   `>= 2.0.0` — the SDK auto-declares `<purpose-field-name>` in
   `<sdk-file-from-findings>`. No client-side change needed beyond
   bumping the dependency.
4. For custom MCP clients, add the field to your handshake builder.
   Reference implementation:
   `<sdk-file-from-findings>:<line-range>`.

## Backward compatibility

None. The gate is hard — old clients without `<purpose-field-name>` are
rejected. There is no fallback or grace period.

## References

- Server-side enforcement: `<server-file-from-findings>`.
- SDK auto-declare: `<sdk-file-from-findings>`.
- Commit: `d0f11e2`.
````

Every `<…>` placeholder must be replaced with concrete content from `/tmp/mcp-gating-findings.md`. Do **not** leave any `<…>` markers in the final file.

- [ ] **Step 3: Verify no placeholders survived**

```bash
grep -c '<[a-z-]*>' docs/migrations/2026-04-mcp-purpose-gating.md
```
Expected: `0`.

```bash
wc -l docs/migrations/2026-04-mcp-purpose-gating.md
grep -c '^## ' docs/migrations/2026-04-mcp-purpose-gating.md
```
≥ 30 lines, ≥ 6 headings.

- [ ] **Step 4: Commit**

```bash
git add docs/migrations/2026-04-mcp-purpose-gating.md
git commit -m "docs(migrations): add 2026-04-mcp-purpose-gating guide"
```

---

## Task 9: CI gate — script + workflow + tests

**Files:**
- Create: `scripts/ci/check_changelog.sh`
- Create: `.github/workflows/changelog.yml`
- Create: `tests/ci/__init__.py`
- Create: `tests/ci/test_check_changelog.py`

- [ ] **Step 1: Write the failing pytest first (TDD)**

Create directory and empty package marker:
```bash
mkdir -p scripts/ci tests/ci
touch tests/ci/__init__.py
```

Create `tests/ci/test_check_changelog.py` with this exact content:

```python
"""Test the CHANGELOG-gate script via subprocess against fixture git repos."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_changelog.sh"


def _git(repo: Path, *args: str) -> str:
    """Run a git command in repo and return stdout (raises on failure)."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    """Init a fresh repo with one initial commit and a baseline CHANGELOG.md."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [2.0.0] - 2026-05-07\n\n"
        "- existing 2.0.0 bullet\n\n"
        "## [1.0.0] - 2026-02-13\n\n"
        "- existing 1.0.0 bullet\n",
        encoding="utf-8",
    )
    _git(repo, "add", "CHANGELOG.md")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _run_script(repo: Path, base: str, head: str, labels: list[str]) -> int:
    env = {
        **os.environ,
        "BASE_SHA": base,
        "HEAD_SHA": head,
        "PR_LABELS": json.dumps(labels),
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    return result.returncode


def _commit_changelog(repo: Path, new_text: str, msg: str) -> str:
    (repo / "CHANGELOG.md").write_text(new_text, encoding="utf-8")
    _git(repo, "add", "CHANGELOG.md")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _commit_other(repo: Path, msg: str = "other change") -> str:
    other = repo / "other.txt"
    other.write_text((other.read_text() if other.exists() else "") + "x\n")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    if not shutil.which("bash"):
        pytest.skip("bash not available")
    return _init_repo(tmp_path)


def test_no_relevant_label_passes_with_no_changelog_diff(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit_other(repo)
    assert _run_script(repo, base, head, []) == 0


def test_chore_label_passes(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit_other(repo)
    assert _run_script(repo, base, head, ["chore"]) == 0


def test_bug_label_passes(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit_other(repo)
    assert _run_script(repo, base, head, ["bug"]) == 0


def test_feat_label_without_changelog_fails(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit_other(repo)
    assert _run_script(repo, base, head, ["feat"]) == 1


def test_breaking_label_with_unreleased_addition_passes(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    new = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "- new breaking bullet\n\n"
        "## [2.0.0] - 2026-05-07\n\n"
        "- existing 2.0.0 bullet\n\n"
        "## [1.0.0] - 2026-02-13\n\n"
        "- existing 1.0.0 bullet\n"
    )
    head = _commit_changelog(repo, new, "add breaking bullet")
    assert _run_script(repo, base, head, ["breaking"]) == 0


def test_feat_label_with_unreleased_addition_passes(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    new = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "- new feat bullet\n\n"
        "## [2.0.0] - 2026-05-07\n\n"
        "- existing 2.0.0 bullet\n\n"
        "## [1.0.0] - 2026-02-13\n\n"
        "- existing 1.0.0 bullet\n"
    )
    head = _commit_changelog(repo, new, "add feat bullet")
    assert _run_script(repo, base, head, ["feat"]) == 0


def test_feat_label_editing_only_1_0_0_section_fails(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    new = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [2.0.0] - 2026-05-07\n\n"
        "- existing 2.0.0 bullet\n\n"
        "## [1.0.0] - 2026-02-13\n\n"
        "- existing 1.0.0 bullet\n"
        "- typo fix in 1.0.0\n"
    )
    head = _commit_changelog(repo, new, "fix typo in 1.0.0")
    assert _run_script(repo, base, head, ["feat"]) == 1


def test_both_labels_with_unreleased_addition_passes(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    new = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "- combined bullet\n\n"
        "## [2.0.0] - 2026-05-07\n\n"
        "- existing 2.0.0 bullet\n\n"
        "## [1.0.0] - 2026-02-13\n\n"
        "- existing 1.0.0 bullet\n"
    )
    head = _commit_changelog(repo, new, "add combined bullet")
    assert _run_script(repo, base, head, ["feat", "breaking"]) == 0


def test_feat_label_addition_under_2_0_0_only_fails(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    new = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [2.0.0] - 2026-05-07\n\n"
        "- existing 2.0.0 bullet\n"
        "- new bullet that landed in 2.0.0 section\n\n"
        "## [1.0.0] - 2026-02-13\n\n"
        "- existing 1.0.0 bullet\n"
    )
    head = _commit_changelog(repo, new, "add bullet in wrong section")
    assert _run_script(repo, base, head, ["feat"]) == 1


def test_feat_label_typo_fix_in_1_0_0_only_fails(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    new = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [2.0.0] - 2026-05-07\n\n"
        "- existing 2.0.0 bullet\n\n"
        "## [1.0.0] - 2026-02-13\n\n"
        "- existing 1.0.0 bullet typo-fixed\n"
    )
    head = _commit_changelog(repo, new, "typo fix in 1.0.0")
    assert _run_script(repo, base, head, ["feat"]) == 1
```

- [ ] **Step 2: Run the tests to confirm they fail (script not yet present)**

```bash
uv run pytest tests/ci/test_check_changelog.py -v
```
Expected: every test errors out with "No such file or directory" pointing at `scripts/ci/check_changelog.sh` (because the script doesn't exist yet). This confirms the test harness wires up correctly before any implementation.

- [ ] **Step 3: Write `scripts/ci/check_changelog.sh`**

Create `scripts/ci/check_changelog.sh` with this exact content:

```bash
#!/usr/bin/env bash
# Fail PRs labelled `feat` or `breaking` if they don't add lines under
# `## [Unreleased]` in CHANGELOG.md. Designed to run inside GitHub Actions
# but tested locally via tests/ci/test_check_changelog.py.
#
# Required env:
#   PR_LABELS  — JSON array of label names (e.g. '["feat","bug"]').
#   BASE_SHA   — base commit of the PR (merge base or target branch tip).
#   HEAD_SHA   — head commit of the PR.

set -euo pipefail

err() {
  echo "ERROR: $*" >&2
}

label_present() {
  local needle="$1"
  # Lightweight JSON-array containment check. PR_LABELS comes from
  # GitHub Actions' toJson(...), which never quotes label names with
  # embedded quotes, so substring search is safe enough here.
  printf '%s' "${PR_LABELS:-[]}" | grep -q "\"${needle}\""
}

if ! label_present feat && ! label_present breaking; then
  exit 0
fi

# Step A: CHANGELOG.md must be in the diff at all.
if ! git diff --name-only "$BASE_SHA" "$HEAD_SHA" -- CHANGELOG.md \
     | grep -qx 'CHANGELOG.md'; then
  err "PR labelled \`breaking\`/\`feat\` requires a CHANGELOG.md entry under \`## [Unreleased]\`. Add a bullet to the top section of CHANGELOG.md."
  exit 1
fi

# Step B: parse the post-merge file structure.
post_file=$(git show "${HEAD_SHA}:CHANGELOG.md")
total_lines=$(printf '%s\n' "$post_file" | wc -l | tr -d ' ')

# Find line number of `## [Unreleased]`
u_start=$(printf '%s\n' "$post_file" | grep -n '^## \[Unreleased\]' | head -n1 | cut -d: -f1 || true)
if [[ -z "${u_start:-}" ]]; then
  err "CHANGELOG.md is missing a \`## [Unreleased]\` heading. Add it before submitting."
  exit 1
fi

# Find line number of the next `## [` heading after Unreleased (exclusive end).
u_end=$(printf '%s\n' "$post_file" | awk -v start="$u_start" '
  NR > start && /^## \[/ { print NR; exit }
')
if [[ -z "${u_end:-}" ]]; then
  u_end=$((total_lines + 1))
fi

# Step C: parse `git diff --unified=0` for added (+) lines and check whether
# any of their post-image line numbers fall inside [u_start, u_end).
added_in_unreleased=$(git diff --unified=0 "$BASE_SHA" "$HEAD_SHA" -- CHANGELOG.md \
  | awk -v u_start="$u_start" -v u_end="$u_end" '
    /^@@/ {
      # Hunk header: @@ -a,b +c,d @@
      match($0, /\+([0-9]+)(,([0-9]+))?/, m)
      cur = m[1] + 0
      next
    }
    /^\+\+\+/ { next }
    /^\+/ {
      if (cur >= u_start && cur < u_end) { found = 1 }
      cur += 1
      next
    }
    /^-/ { next }
    /^ / { cur += 1; next }
    END { if (found) print "yes" }
  ')

if [[ "$added_in_unreleased" == "yes" ]]; then
  exit 0
fi

err "PR touched CHANGELOG.md but added no lines under \`## [Unreleased]\`. Move your bullet up to the [Unreleased] section."
exit 1
```

Make it executable:
```bash
chmod +x scripts/ci/check_changelog.sh
```

- [ ] **Step 4: Run the pytest suite — should pass now**

```bash
uv run pytest tests/ci/test_check_changelog.py -v
```
Expected: 10 passed. If any fail, debug by running the failing case manually:
```bash
cd /tmp/<the-pytest-tmp-path>/repo
PR_LABELS='["feat"]' BASE_SHA=<base> HEAD_SHA=<head> bash <abs-path-to>/scripts/ci/check_changelog.sh
echo "exit: $?"
```

- [ ] **Step 5: Write the GitHub Actions workflow**

Create `.github/workflows/changelog.yml`:

```yaml
name: changelog

on:
  pull_request:
    types: [opened, synchronize, reopened, labeled, unlabeled]

jobs:
  require-changelog:
    name: Require CHANGELOG entry on feat/breaking PRs
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Check for CHANGELOG entry
        env:
          PR_LABELS: ${{ toJson(github.event.pull_request.labels.*.name) }}
          BASE_SHA: ${{ github.event.pull_request.base.sha }}
          HEAD_SHA: ${{ github.event.pull_request.head.sha }}
        run: bash scripts/ci/check_changelog.sh
```

- [ ] **Step 6: Validate the workflow YAML**

If `actionlint` is available locally:
```bash
actionlint .github/workflows/changelog.yml
```
Expected: no output (success). If `actionlint` is not installed, run a YAML sanity check:
```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/changelog.yml'))"
```
Expected: no exceptions.

- [ ] **Step 7: Smoke-test the script against a recent merged PR's diff**

Use commit `89f000e` (PR #121, scoring modes — has `feat` semantics) and its parent as a real-world fixture:
```bash
PR_LABELS='["feat"]' \
  BASE_SHA=$(git rev-parse 89f000e^) \
  HEAD_SHA=89f000e \
  bash scripts/ci/check_changelog.sh
echo "exit: $?"
```
Expected exit: `1` (the PR did not update CHANGELOG.md — that's exactly the gap this gate exists to catch). This proves the gate would have flagged the gap if it had been live then.

- [ ] **Step 8: Commit**

```bash
git add scripts/ci/check_changelog.sh \
        .github/workflows/changelog.yml \
        tests/ci/__init__.py \
        tests/ci/test_check_changelog.py
git commit -m "ci(changelog): require CHANGELOG entry on feat/breaking PRs"
```

---

## Task 10: Release-artefact pytest

**Files:**
- Create: `tests/docs/__init__.py`
- Create: `tests/docs/test_release_artifacts.py`

- [ ] **Step 1: Set up directory**

```bash
mkdir -p tests/docs
touch tests/docs/__init__.py
```

- [ ] **Step 2: Write the test file**

Create `tests/docs/test_release_artifacts.py`:

```python
"""Static checks that release artefacts (CHANGELOG, migration guides, per-package
CHANGELOGs) are present and structurally well-formed."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ROOT_CHANGELOG = REPO_ROOT / "CHANGELOG.md"
MIGRATIONS_DIR = REPO_ROOT / "docs" / "migrations"

EXPECTED_MIGRATIONS = {
    "2026-04-el-farol-intervals.md",
    "2026-05-el-farol-scoring.md",
    "2026-04-mcp-purpose-gating.md",
}

EXPECTED_HEADINGS = {
    "## What changed",
    "## Before",
    "## After",
    "## How to migrate",
}

PACKAGE_CHANGELOGS = [
    REPO_ROOT / "packages" / "atp-core" / "CHANGELOG.md",
    REPO_ROOT / "packages" / "atp-adapters" / "CHANGELOG.md",
    REPO_ROOT / "packages" / "atp-dashboard" / "CHANGELOG.md",
    REPO_ROOT / "packages" / "atp-sdk" / "CHANGELOG.md",
]


def test_root_changelog_has_unreleased_and_2_0_0() -> None:
    text = ROOT_CHANGELOG.read_text(encoding="utf-8")
    assert "## [Unreleased]" in text
    assert "## [2.0.0]" in text
    assert "## [1.0.0]" in text


def test_root_changelog_linkbacks_use_correct_org() -> None:
    text = ROOT_CHANGELOG.read_text(encoding="utf-8")
    assert "github.com/andrei-shtanakov/atp-platform" in text, (
        "CHANGELOG.md linkbacks must use the andrei-shtanakov org "
        "(matching `git remote get-url origin`)."
    )
    assert "github.com/anthropics" not in text, (
        "CHANGELOG.md still references the stale 'anthropics' org "
        "in linkbacks; update to andrei-shtanakov."
    )


def test_migrations_dir_exists() -> None:
    assert MIGRATIONS_DIR.is_dir(), f"{MIGRATIONS_DIR} must exist"


@pytest.mark.parametrize("name", sorted(EXPECTED_MIGRATIONS))
def test_migration_guide_present_and_well_formed(name: str) -> None:
    path = MIGRATIONS_DIR / name
    assert path.is_file(), f"missing migration guide: {path}"
    text = path.read_text(encoding="utf-8")
    line_count = len(text.splitlines())
    assert line_count >= 30, f"{path} is only {line_count} lines; expected >= 30"
    for heading in EXPECTED_HEADINGS:
        assert heading in text, f"{path} missing heading {heading!r}"


def test_migration_guide_no_unfilled_placeholders() -> None:
    """Catch the MCP guide if it ships with `<placeholder>` markers."""
    for name in EXPECTED_MIGRATIONS:
        path = MIGRATIONS_DIR / name
        text = path.read_text(encoding="utf-8")
        # Allow Markdown emphasis/HTML like `<br>` or `</em>` if needed,
        # but reject anything matching `<lowercase-with-dashes>` which is
        # how the spec template marks placeholders.
        import re

        bad = re.findall(r"<[a-z][a-z0-9-]*(?:-[a-z0-9]+)*>", text)
        assert not bad, f"{path} still contains placeholder markers: {bad}"


@pytest.mark.parametrize("path", PACKAGE_CHANGELOGS, ids=lambda p: p.parent.name)
def test_package_changelog_exists_and_has_unreleased(path: Path) -> None:
    assert path.is_file(), f"missing package CHANGELOG: {path}"
    text = path.read_text(encoding="utf-8")
    assert "## [Unreleased]" in text, f"{path} missing `## [Unreleased]`"
```

- [ ] **Step 3: Run the tests**

```bash
uv run pytest tests/docs/test_release_artifacts.py -v
```
Expected: all tests pass (every artefact created in Tasks 4-8 is now in place). If a test fails, fix the underlying artefact, not the test (the test encodes the spec).

- [ ] **Step 4: Commit**

```bash
git add tests/docs/__init__.py tests/docs/test_release_artifacts.py
git commit -m "test(docs): assert release artefacts are present and well-formed"
```

---

## Task 11: CONTRIBUTING.md — Releases & Changelog section

**Files:**
- Modify: `CONTRIBUTING.md`

- [ ] **Step 1: Read the current file end**

```bash
tail -10 CONTRIBUTING.md
```
Note where the file currently ends — the new section appends after the existing content.

- [ ] **Step 2: Append the new section**

Append this exact block to `CONTRIBUTING.md` (with one blank line between the existing last line and the new `## Releases & Changelog` heading):

```markdown

## Releases & Changelog

### Changelog rule

Every PR labelled `breaking` or `feat` must add a bullet under
`## [Unreleased]` in `CHANGELOG.md`. Bug fixes are optional but encouraged.

The `changelog` GitHub Actions workflow (`.github/workflows/changelog.yml`)
enforces this on label change. If you remove the `feat`/`breaking` label
the gate goes away — this is intentional. The goal is a reminder, not a
hard wall. If a stricter check is ever needed, the next layer is parsing
PR titles for conventional-commit prefixes; that's a follow-up, not in
this workflow.

### Routing — which CHANGELOG?

- Changes that affect the platform / CLI / dashboard / engines (El Farol,
  PD, etc.) → root `CHANGELOG.md`.
- Changes that affect a specific distributable package
  (`atp-platform-sdk`, `game-environments`, `atp-adapters`, `atp-core`,
  `atp-dashboard`) → `packages/<name>/CHANGELOG.md`.
- When in doubt, write the bullet in both — the root entry can just point
  to the package one.

Sub-packages are versioned independently. The root `pyproject.toml`'s
`version` describes the platform / CLI; each `packages/atp-*/pyproject.toml`
tracks its own release cadence.

### Cutting a release

1. Open a release PR. Replace `## [Unreleased]` with
   `## [X.Y.Z] - <today's date>` and add a fresh empty `## [Unreleased]`
   above it. Update linkbacks at the bottom of the file.
2. Bump `version` in the relevant `pyproject.toml`. Root for platform /
   CLI releases; sub-package `pyproject.toml` for SDK / package releases.
3. Merge the release PR.
4. Tag the merge commit:
   ```bash
   git tag -a vX.Y.Z <merge-commit-sha> -m "Release X.Y.Z"
   git push origin vX.Y.Z
   ```
5. Open the GitHub Release UI for the new tag. Title: `vX.Y.Z`. Body:
   paste the contents of the `## [X.Y.Z]` section from `CHANGELOG.md`
   directly. Do **not** rely on GitHub's auto-generated release notes —
   they group by PR title and label and ignore CHANGELOG entirely, so
   the result drifts from the actual changelog.
6. **PyPI publish is not automatic.** Tagging produces only the GitHub
   Release. Publishing to PyPI requires a separate workflow run (or
   manual `uv build` + `uv publish` if the workflow is missing). After
   publishing, smoke-test the release with `pip install <package>==X.Y.Z`
   from a clean venv.
```

- [ ] **Step 3: Visual sanity check**

```bash
tail -60 CONTRIBUTING.md
grep -c '^### ' CONTRIBUTING.md
```
The trailing 60 lines must include all three sub-headings (`### Changelog rule`, `### Routing — which CHANGELOG?`, `### Cutting a release`).

- [ ] **Step 4: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs(contributing): document changelog rule, routing, and release procedure"
```

---

## Task 12: Self-review pass

**Files:** none (verification only)

- [ ] **Step 1: Run ruff**

```bash
uv run ruff check tests/ci tests/docs
uv run ruff format --check tests/ci tests/docs
```
Expected: clean. Fix any issues with `uv run ruff format tests/ci tests/docs` and `uv run ruff check tests/ci tests/docs --fix`.

- [ ] **Step 2: Run pyrefly**

```bash
uv run pyrefly check
```
Expected: no new errors introduced by this PR. If pyrefly flags pre-existing errors, leave them alone — the only things this PR adds are pure-stdlib + pytest test files which should be clean.

- [ ] **Step 3: Run the new pytest suites**

```bash
uv run pytest tests/ci tests/docs -v
```
Expected: all tests pass.

- [ ] **Step 4: Smoke the gate against a real merged PR**

Re-run the Task 9 step 7 smoke for confidence:
```bash
PR_LABELS='["feat"]' \
  BASE_SHA=$(git rev-parse 89f000e^) \
  HEAD_SHA=89f000e \
  bash scripts/ci/check_changelog.sh
echo "exit: $?"
```
Expected: `exit: 1`.

- [ ] **Step 5: actionlint (optional but recommended)**

If `actionlint` is installed locally:
```bash
actionlint .github/workflows/changelog.yml
```
Expected: no output.

If `actionlint` is not installed, fall back to:
```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/changelog.yml'))"
```
Expected: no exceptions.

- [ ] **Step 6: Visual diff review**

```bash
git diff main..HEAD --stat
git log main..HEAD --oneline
```
The diff stat should match the File Structure table at the top of this plan: ~14 files, no surprises (no `atp/` source modifications, no `packages/atp-*/atp/` source modifications, no game engine changes).

- [ ] **Step 7: Open the PR**

Push the branch and open the PR:
```bash
git push -u origin docs/changelog-restoration
gh pr create --title "docs(release): restore CHANGELOG and add v2.0.0 migration guides" --body "$(cat <<'EOF'
## Summary

- Restore root `CHANGELOG.md` with `[Unreleased]` and `[2.0.0]` sections; document three breaking changes (El Farol intervals, El Farol scoring default, MCP purpose gating).
- Bootstrap per-package `CHANGELOG.md` skeletons for `atp-core`, `atp-adapters`, `atp-dashboard`, `atp-sdk`.
- Add three migration guides under `docs/migrations/`.
- Bump root `pyproject.toml` to 2.0.0 (sub-packages keep independent versions).
- Add `changelog` GitHub Actions workflow that requires a CHANGELOG entry under `[Unreleased]` on PRs labelled `breaking` or `feat`.
- Document the changelog rule, routing, and release procedure in `CONTRIBUTING.md`.

Spec: `docs/superpowers/specs/2026-05-07-changelog-restoration-design.md`
Plan: `docs/superpowers/plans/2026-05-07-changelog-restoration.md`

## Audit summary

<paste the "Decision" line from docs/superpowers/plans/audit-results.md>

## Test plan

- [x] `uv run pytest tests/ci tests/docs -v` — all green.
- [x] Smoke: ran the gate script against PR #121 (commit 89f000e); exit code 1, confirming the gate would have flagged the missing CHANGELOG entry.
- [x] `uv run ruff check tests/ci tests/docs` and `uv run pyrefly check` — clean.
- [x] `actionlint .github/workflows/changelog.yml` (or YAML sanity load) — clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review (controller, run AFTER all tasks complete)

This section is the meta-review the planning controller runs after the implementer-subagent loop closes. Subagents do not run this — it's a final pass before declaring the plan done.

1. **Spec coverage** — every section in the spec maps to at least one task above:
   - Spec §1 Goal → Tasks 3, 4, 5, 6, 7, 8, 9, 10, 11.
   - Spec §3 Approach summary, decisions locked → Tasks 4, 5, 11 (CONTRIBUTING).
   - Spec §4 CHANGELOG.md structure → Task 4.
   - Spec §5 Migration guides → Tasks 6, 7, 8.
   - Spec §6 CI enforcement → Tasks 9, 10.
   - Spec §7 Version bump → Task 3.
   - Spec §8 Testing → Tasks 9, 10, 12.
   - Spec §9 Implementation order → mirrored in Task 1 (audit) → 2 (MCP) → 3 → 4 → 5 → 6/7/8 → 9 → 10 → 11 → 12.
   - Spec §10 Risks → mitigations are spread across Tasks 1, 2, 9, 11.
2. **Placeholder scan** — every `<…>` template marker in this plan is documented as "fill from /tmp/mcp-gating-findings.md" or "write the bullet from Task 1's audit list". No bare `TODO` / `TBD` / `implement later` markers.
3. **Type / name consistency** — `check_changelog.sh` is referenced consistently across Tasks 9, 10, 11, 12. `tests/ci/test_check_changelog.py` test function names are stable. `EXPECTED_MIGRATIONS` set in Task 10 matches the three filenames produced by Tasks 6-8.

# El Farol Dashboard — Data Model Implementation Plan

**Scope:** data-model changes only. This plan covers the Python in-memory
models, action extraction, runner storage output, and dashboard SQL schema
changes required to back the polished El Farol dashboard mockup. It does **not**
cover: UI rendering, REST/WebSocket endpoints, OTel instrumentation wiring
itself (only the persisted *shape* of OTel-sourced fields), or dashboard
frontend work.

**Game variant targeted:** multi-slot El Farol with interval-based actions —
16 slots per day, ≤ 2 contiguous visits per day, ≤ 8 slots total per day,
per-slot capacity threshold = 60% × number of agents.

**Branch / convention:** feature branch `feature/el-farol-dashboard-plan`.
TDD discipline: each phase writes failing tests before the implementation.
Tests live under `tests/unit/` or `tests/integration/` following the test
pyramid convention in [CLAUDE.md](../../CLAUDE.md).

---

## 0. Goals and non-goals

### Goals

1. Introduce a typed, queryable per-day-per-agent action record to replace the
   current untyped `list[dict]` blobs in [`atp-games/atp_games/models.py`](../../atp-games/atp_games/models.py).
2. Persist **per-day payoff** (currently only final-cumulative is stored).
3. Capture the minimum agent-self-reported metadata needed by the dashboard:
   `intervals` (required) and `intent` (optional free-text).
4. Persist game configuration (slots, max intervals, capacity ratio/threshold)
   with each match so historical runs stay interpretable if rules evolve.
5. Add `user_id` to agent identity so the "by user" cohort view in the
   dashboard has something to group on.
6. Keep reserved fields for Tier-2 OTel-sourced resource data (tokens, latency,
   cost, trace id) so the drawer can surface them once OTel wiring lands —
   without blocking this plan on OTel work.
7. Evolve the dashboard SQL schema (`packages/atp-dashboard/atp/dashboard/models.py`)
   additively so no existing run data becomes unreadable.

### Non-goals

- UI/visualisation code, REST endpoint shapes, WebSocket event streaming.
- OTel/Langfuse span emission (only the persisted *landing zone* for values
  OTel will eventually supply).
- Agent-side changes (no SDK, no adapter authoring). The action mapper accepts
  the new fields; agents not yet emitting them degrade gracefully to null.
- Predicted-per-slot, strategy_label, confidence, alternatives — deliberately
  descoped per the simplified dashboard design.
- Backfilling historical games with derived per-round payoffs. Historical
  matches keep their old shape; new matches use the new shape.

### Non-scope guardrails

- Do **not** extend the `ATPRequest` / `ATPResponse` protocol models beyond
  what is strictly needed for the game mapper. Richer protocol metadata
  belongs to a separate Tier-1 protocol-enrichment effort.
- Do **not** introduce new evaluator types in this plan.

---

## 1. Current state (baseline)

As of branch `feature/el-farol-dashboard-plan`:

| Concern | Where | Shape |
|---|---|---|
| Agent action parsing | [`atp-games/atp_games/mapping/action_mapper.py:39`](../../atp-games/atp_games/mapping/action_mapper.py) | `GameAction{action, message?, reasoning?, raw_response}` — untyped `action: Any` |
| Per-episode payoffs | [`atp-games/atp_games/models.py`](../../atp-games/atp_games/models.py) `EpisodeResult.payoffs` | `dict[str, float]` — **final cumulative only** |
| Per-round history | `EpisodeResult.history` | `list[dict[str, Any]]` — untyped |
| Per-round actions | `EpisodeResult.actions_log` | `list[dict[str, Any]]` — untyped |
| Game config | [`game-environments/game_envs/core/game.py`](../../game-environments/game_envs/core/game.py) `GameConfig` | `num_players`, `num_rounds`, `discount_factor`, `noise`, `communication_mode`, `seed` — **no slot/interval/capacity knobs** |
| Match storage | [`packages/atp-dashboard/atp/dashboard/models.py`](../../packages/atp-dashboard/atp/dashboard/models.py) `GameResult` | JSON blobs: `episodes_json`, `payoff_matrix_json`, `strategy_timeline_json`, … |
| Agent identity | `GameResult.players_json` | list of `{id, name}` — **no `user_id`** |

**Nothing here supports per-day scrubbing in the dashboard.** The heatmap,
leaderboard race, per-day drawer, and user-cohort view all need richer
structure at ingest time.

---

## 2. Phased rollout strategy

Eight phases, each independently shippable and reversible:

| # | Phase | Scope | Risk |
|---|---|---|---|
| 1 | Typed `ActionRecord` + `DayAggregate` + `MatchConfig` models — additive | in-memory only | low |
| 2 | Runner populates `ActionRecord`s alongside existing `actions_log` | in-memory + runner | low |
| 3 | ActionMapper extracts optional `intent` | parsing | low |
| 4 | Per-day payoff on `EpisodeResult.round_payoffs` | in-memory + runner | low |
| 5 | Match-config fields (`capacity_ratio`, `num_slots`, `max_intervals`, `max_total_slots`) | config | low |
| 6 | Agent `user_id` field | identity | medium (migrations) |
| 7 | Dashboard SQL: additive columns for typed per-day data | storage | medium |
| 8 | Deprecate `actions_log`/`history` (N+1 release only — not in this plan) | cleanup | low |

Phases 1–5 are pure additions to in-memory Python models. Phases 6–7 add
storage columns but preserve existing JSON blobs. Phase 8 is a follow-up,
mentioned only for roadmap completeness.

**Dependency graph:**

```
   Phase 1 ─┬── Phase 2 ─── Phase 4 ──┐
            │                          │
            └── Phase 3 ──────────────┼─── Phase 7
                                      │
   Phase 5 ──────────────────────────┤
                                      │
   Phase 6 ──────────────────────────┘
```

Phases 3, 5, 6 can run in parallel after Phase 1. Phase 7 integrates
everything.

---

## 3. Phase 1 — Typed record models (foundation)

**Motivation.** Everything downstream needs a typed, serializable per-round
record to key off of. This phase adds the models without changing any runtime
behaviour.

### 3.1. Files touched

- `atp-games/atp_games/models.py` — add new Pydantic dataclass models

No file deletions. `EpisodeResult.history` and `actions_log` stay for now.

### 3.2. Changes

Add to [`atp-games/atp_games/models.py`](../../atp-games/atp_games/models.py):

```python
# New models — additive

@dataclass(frozen=True)
class IntervalPair:
    """Up to 2 contiguous slot intervals submitted for a day.

    Either or both intervals may be empty ([]); empty interval == no visit.
    Invariants enforced in __post_init__:
      - at most MAX_INTERVALS non-empty
      - total covered slots <= max_total_slots
      - intervals non-overlapping and non-adjacent
      - each interval has start <= end, both in [0, num_slots-1]
    """
    first:  tuple[int, int] | tuple[()]    # () == empty
    second: tuple[int, int] | tuple[()]

    def covered_slots(self) -> tuple[int, ...]: ...
    def num_visits(self) -> int: ...
    def total_slots(self) -> int: ...


@dataclass
class ActionRecord:
    """One agent's action on one day, with outcome and optional metadata."""
    # identity
    match_id:   str
    day:        int              # 1..num_days
    agent_id:   str

    # action (required)
    intervals:  IntervalPair

    # derived from intervals (cached)
    picks:       tuple[int, ...]
    num_visits:  int
    total_slots: int

    # outcome (populated after runner resolves the day)
    payoff:     float
    num_under:  int
    num_over:   int

    # optional Tier-1 agent self-report
    intent: str | None = None

    # optional Tier-2 OTel-sourced resource usage
    tokens_in:  int | None = None
    tokens_out: int | None = None
    decide_ms: int | None = None
    cost_usd:  float | None = None
    model_id:   str | None = None

    # validation / retry — populated by runner
    retry_count:      int         = 0
    validation_error: str | None = None

    # observability linkage (W3C traceparent)
    trace_id: str | None = None
    span_id:  str | None = None

    submitted_at: datetime | None = None


@dataclass
class DayAggregate:
    """Precomputed per-day per-slot attendance, cached with the match."""
    match_id:         str
    day:              int
    slot_attendance:  tuple[int, ...]   # length = num_slots
    over_slots:       int
    total_attendances: int


@dataclass(frozen=True)
class MatchConfig:
    """Game-level configuration, persisted with every match.

    Resolved capacity_threshold is stored so historical matches remain
    interpretable if the rule-derivation logic ever changes.
    """
    game_id:             str          # "el_farol_interval"
    game_version:        str          # e.g. "1.0.0"
    num_days:            int
    num_slots:           int  = 16
    max_intervals:       int  = 2
    max_total_slots:     int  = 8
    capacity_ratio:      float = 0.6
    capacity_threshold:  int   = 0    # set via __post_init__ if 0
    seed:                int | None = None
```

**Explicit design choices:**

- Dataclasses (not Pydantic `BaseModel`) — matches the existing style of
  `GameConfig`, `EpisodeResult`, `PlayerStats` in this file.
- `IntervalPair` is a separate type so validation lives in one place and the
  dashboard can trust structural invariants.
- `covered_slots()` and friends are methods (not stored) to keep `IntervalPair`
  frozen and canonical. `ActionRecord.picks` caches them because the dashboard
  reads them 100× per scrub.
- All new optional fields default to `None` / `0` — an unenriched runner still
  produces valid `ActionRecord`s.

### 3.3. Tests (TDD — write before implementation)

New file: `tests/unit/atp_games/test_models_action_record.py`

- `IntervalPair` validation:
  - Accepts `((0,2), ())`, `((0,2), (4,6))`, `((), ())`
  - Rejects overlap `((0,4), (3,7))`
  - Rejects adjacency `((0,3), (4,6))` (game rule: distinct visits)
  - Rejects `total_slots > max_total_slots`
  - Rejects out-of-range slot indices
- `IntervalPair.covered_slots()` returns sorted unique ints
- `ActionRecord` constructs with only required fields; optional fields default correctly
- `ActionRecord` round-trips through `asdict` / reconstruct
- `MatchConfig.__post_init__` derives `capacity_threshold = floor(ratio × num_agents)` when passed `num_agents` at resolution time (via a helper `MatchConfig.resolve(num_agents: int) -> MatchConfig`)

**Launch `unit-tester` subagent per the TDD workflow rule in the user's global
CLAUDE.md.**

### 3.4. Acceptance

- `uv run pytest tests/unit/atp_games/test_models_action_record.py -v` green
- `uv run pyrefly check` clean on the modified file
- No existing test regresses

### 3.5. Backward compat

Pure addition. No existing consumer of `models.py` needs to change.

---

## 4. Phase 2 — Runner populates ActionRecord

**Motivation.** We now have the type; the runner needs to emit it. Keep the
old `actions_log` in parallel so nothing breaks.

### 4.1. Files touched

- `atp-games/atp_games/runner/game_runner.py`
- `atp-games/atp_games/models.py` — add `actions: list[ActionRecord]` to
  `EpisodeResult`

### 4.2. Changes

Expand `EpisodeResult`:

```python
@dataclass
class EpisodeResult:
    episode:     int
    payoffs:     dict[str, float]        # still final cumulative
    history:     list[dict]   = ...       # still untyped (deprecated)
    actions_log: list[dict]   = ...       # still untyped (deprecated)
    actions:     list[ActionRecord] = field(default_factory=list)   # NEW
    seed:        int | None = None
```

In `GameRunner._run_episode` (or equivalent loop):
- After each day's actions resolve, construct one `ActionRecord` per player
  and append to `episode.actions`.
- Populate `intervals`, `picks`, `num_visits`, `total_slots` from the action
  mapper result.
- Populate `payoff`, `num_under`, `num_over` from the step-resolution results.
- Populate `retry_count`, `validation_error` from the `ActionValidator`.
- Leave OTel fields `None` for now — Phase 8+ OTel work fills them.

### 4.3. Tests

New file: `tests/unit/atp_games/test_runner_action_records.py`

- After a 3-day mock run with 2 agents, `episode.actions` has length 6
- Day ordering: `actions[i].day <= actions[i+1].day` (when grouped per agent)
- `sum(a.payoff for a in actions if a.agent_id=='p0')` equals
  `episode.payoffs['p0']`
- Retry path: when agent returns invalid action then recovers, the record's
  `retry_count` reflects attempts

### 4.4. Acceptance

- All new tests green
- Existing runner tests still pass
- `uv run pytest tests/ -v -m "not slow"` green

### 4.5. Backward compat

Additive. Existing consumers reading `actions_log` keep working.

---

## 5. Phase 3 — ActionMapper extracts `intent`

**Motivation.** The drawer shows agent intent. The extractor needs to pull it
out of the response artifact if the agent supplied it.

### 5.1. Files touched

- `atp-games/atp_games/mapping/action_mapper.py`

### 5.2. Changes

Extend `GameAction` dataclass in [action_mapper.py:20](../../atp-games/atp_games/mapping/action_mapper.py):

```python
@dataclass
class GameAction:
    action:        Any
    message:       str | None       = None
    reasoning:     str | None       = None   # still kept for other games
    intent:        str | None       = None   # NEW — free-text Tier-1 self-report
    raw_response:  dict[str, Any]   = field(default_factory=dict)
```

In `ActionMapper.from_atp_response`, after extracting the action, also
extract `intent` from the same structured artifact or parsed JSON if present.
Absent → `None`. Never fail on missing `intent`.

### 5.3. Tests

New tests in `tests/unit/atp_games/test_action_mapper.py`:

- Response with `{"action": [...], "intent": "go morning"}` → `GameAction.intent == "go morning"`
- Response with only `action` → `GameAction.intent is None`
- Intent of wrong type (int, list) → `None` (with a log warning, not raise)
- Whitespace-only intent → `None` (strip and treat as absent)
- Max length — truncate/reject intents over some reasonable cap
  (e.g. 500 chars) so agents cannot blow up storage

### 5.4. Acceptance

All new tests green. Existing mapper tests still pass.

### 5.5. Backward compat

Agents not emitting `intent` produce `None`. No existing agent is broken.

---

## 6. Phase 4 — Per-day payoff on EpisodeResult

**Motivation.** The dashboard timeline/race chart needs per-day payoff.
Currently only the final cumulative is persisted.

### 6.1. Files touched

- `atp-games/atp_games/models.py` — add `round_payoffs` field
- `atp-games/atp_games/runner/game_runner.py` — populate it

### 6.2. Changes

```python
@dataclass
class EpisodeResult:
    episode:       int
    payoffs:       dict[str, float]
    round_payoffs: list[dict[str, float]] = field(default_factory=list)  # NEW
    # ... existing fields
```

Invariant: `sum(r[pid] for r in round_payoffs) == payoffs[pid]` for every
player id.

Runner appends one `dict[player_id, per_day_payoff]` per resolved day.

Note: `ActionRecord.payoff` gives the same data grouped differently (per
agent per day). `round_payoffs` gives the same data grouped per day. We keep
both because the dashboard reads both: the heatmap wants day-major, the
leaderboard race wants agent-major, and we don't want the frontend to
reshape 6400+ records.

### 6.3. Tests

Add to `tests/unit/atp_games/test_models.py` (or test_runner):

- `round_payoffs` length equals `num_days` after a completed episode
- Sum-per-player invariant holds
- First-day payoff matches what the game engine produced on day 1

### 6.4. Acceptance

All tests green. Existing `payoffs` semantics unchanged.

### 6.5. Backward compat

Additive. Old readers ignore `round_payoffs`.

---

## 7. Phase 5 — Match config fields

**Motivation.** The dashboard top bar shows "16 slots/day · cap 60% · ≤2
visits · ≤8 slots". Those values must come from stored config, not be hard-
coded in frontend.

### 7.1. Files touched

- `game-environments/game_envs/games/el_farol.py` (and/or wherever the
  El Farol subclass lives) — add new config fields
- `atp-games/atp_games/models.py` — `MatchConfig` already introduced in
  Phase 1; ensure runner persists a resolved instance on `GameResult`

### 7.2. Changes

Extend El Farol's game-specific config (not the generic `GameConfig`):

```python
@dataclass(frozen=True)
class ElFarolConfig:
    num_slots:        int   = 16
    max_intervals:    int   = 2
    max_total_slots:  int   = 8
    capacity_ratio:   float = 0.6           # fraction of players
    # Resolved at match creation, persisted with results:
    capacity_threshold: int = 0             # floor(ratio × num_players)
```

Resolution rule: `capacity_threshold = floor(capacity_ratio * num_players)`.
Rationale: at 8 agents × 0.6 = 4.8 → threshold 4; at 16 × 0.6 = 9.6 → 9.
The dashboard needs exactly one integer; floor is the natural choice.
Document this explicitly so no one second-guesses it later.

At match creation, the runner:
1. Reads `capacity_ratio` from config.
2. Computes `capacity_threshold = floor(ratio × num_players)`.
3. Stores resolved `ElFarolConfig` on `GameResult.match_config`.

### 7.3. Tests

New file: `tests/unit/atp_games/test_el_farol_config.py`

- Ratio 0.6 × 8 players → threshold 4
- Ratio 0.6 × 16 players → threshold 9
- Ratio 0.6 × 10 players → threshold 6
- Config rejects `max_total_slots > num_slots`
- Config rejects `max_intervals > max_total_slots`
- Config rejects `capacity_ratio` outside (0, 1]
- Round-trips through serialisation

### 7.4. Acceptance

Tests green. Game engine consumes the resolved threshold for step-resolution
and emits it on every `GameResult`.

### 7.5. Backward compat

Old runs stored without these fields stay readable; dashboard falls back to
defaults (16 slots, cap 60%) when loading such runs.

---

## 8. Phase 6 — Agent `user_id` field

**Motivation.** The "by user" cohort view in the dashboard (grouping 16+
submitted agents under 5–10 human users) requires an explicit ownership
field. Currently agents carry only an id and name.

### 8.1. Files touched

- `atp-games/atp_games/models.py` — extend `GameResult.agent_names` /
  introduce `agent_records` typed model
- `packages/atp-dashboard/atp/dashboard/models.py` — SQL column

### 8.2. Changes

Introduce `AgentRecord`:

```python
@dataclass
class AgentRecord:
    agent_id:      str
    display_name:  str
    user_id:       str               # required — "unknown" if genuinely absent
    user_display:  str | None = None
    family:        str | None = None      # e.g. "calibrated"
    adapter_type:  str         = "unknown"  # MCP | HTTP | CLI | SDK
    model_id:      str | None = None        # populated via OTel when available
    color:         str | None = None        # optional presentation hint
```

Add `agents: list[AgentRecord]` to `GameResult`. Keep the old
`agent_names: dict[str, str]` for one release as a derived read-only property.

### 8.3. Tests

New file: `tests/unit/atp_games/test_agent_record.py`

- `agent_id` required and non-empty
- `user_id` defaults to `"unknown"` when constructed via a compatibility
  helper (for old runs)
- Legacy-shape reader: given a `GameResult` with only `agent_names`, build
  `agents` list with `user_id="unknown"` so the dashboard can still group
- Serialisation round-trip

### 8.4. Acceptance

Tests green. Downstream `GameResult` consumers can read either legacy
`agent_names` or new `agents`.

### 8.5. Backward compat

Old `GameResult` JSON blobs without `agents` are readable — legacy helper
synthesises an `agents` list with `user_id="unknown"`. The dashboard "by
user" view simply shows all such agents under a single "unknown" cohort
until owners are backfilled.

---

## 9. Phase 7 — Dashboard SQL schema

**Motivation.** The in-memory models now carry everything the dashboard needs,
but `packages/atp-dashboard/atp/dashboard/models.py` stores them only as JSON
blobs. The dashboard needs queryable columns so heatmap/leaderboard filters
don't have to deserialise entire matches.

### 9.1. Files touched

- `packages/atp-dashboard/atp/dashboard/models.py` — schema additions
- Alembic migration under `packages/atp-dashboard/atp/dashboard/migrations/`
  (if one exists; otherwise create; verify project's migration convention in
  existing files under that path)
- `packages/atp-dashboard/atp/dashboard/storage.py` (or equivalent) — writer
  populates new columns; reader prefers them over JSON fallback

### 9.2. Changes

Add to `GameResult` SQL model (additive columns, all nullable for old rows):

```python
# Match-level identity & config (denormalised for query speed)
match_id                = Column(String, index=True)     # distinct from game_name
game_version            = Column(String)
num_days                = Column(Integer)
num_slots               = Column(Integer)
max_intervals           = Column(Integer)
max_total_slots         = Column(Integer)
capacity_ratio          = Column(Float)
capacity_threshold      = Column(Integer)

# Per-day data (preferred representations)
actions_json            = Column(JSON)       # list[ActionRecord.asdict()]
day_aggregates_json     = Column(JSON)       # list[DayAggregate.asdict()]
round_payoffs_json      = Column(JSON)       # list[dict[str, float]]

# Agent roster (replaces players_json when present)
agents_json             = Column(JSON)       # list[AgentRecord.asdict()]
```

New indices:

- `(match_id)` — lookup
- `(game_id, completed_at desc)` — recent matches listing
- `(capacity_ratio, game_version)` — for filtering by variant

Readers prefer the typed columns. If absent (legacy row), fall back to the
existing `episodes_json` / `players_json` with a deprecation log warning.

### 9.3. Optional second table — normalised per-day actions

For large matches (100+ agents × 365 days = 36500 rows), a JSON blob becomes
unwieldy to query. Consider a separate table:

```python
class DayActionRow(Base):
    __tablename__ = "day_actions"
    id          = Column(Integer, primary_key=True)
    match_id    = Column(String, index=True)   # FK
    day         = Column(Integer, index=True)
    agent_id    = Column(String, index=True)
    intervals   = Column(JSON)                  # serialised IntervalPair
    payoff      = Column(Float)
    num_visits  = Column(Integer)
    total_slots = Column(Integer)
    intent      = Column(String, nullable=True)
    # Tier-2 columns (nullable):
    tokens_in   = Column(Integer, nullable=True)
    tokens_out  = Column(Integer, nullable=True)
    decide_ms   = Column(Integer, nullable=True)
    cost_usd    = Column(Float, nullable=True)
    trace_id    = Column(String, nullable=True)
    # composite unique (match_id, day, agent_id)
```

**Ship the JSON-blob path first.** Only introduce `day_actions` when a match
exceeds the blob size threshold (say, 5 MB) or when the dashboard team reports
query pain. Keep this table design in the plan for reference so we don't have
to re-think it later.

### 9.4. Migration

Alembic upgrade: add columns as nullable. Downgrade drops them.

No data backfill. Old rows stay readable via the JSON fallback path.

### 9.5. Tests

- `tests/integration/dashboard/test_storage_roundtrip.py`:
  - Write a `GameResult` with new columns populated; read it back; assert
    structural equality on `ActionRecord[]` and `MatchConfig`.
  - Write a `GameResult` with only legacy fields; read it; assert the
    fallback synthesises the new shape with sensible defaults.
- `tests/integration/dashboard/test_migration.py`:
  - Run alembic upgrade then downgrade; schema matches expected before/after
    states.

### 9.6. Acceptance

- Migration runs cleanly forward and backward on a local SQLite DB and on
  Postgres (CI).
- Round-trip tests green.
- No existing dashboard integration test regresses.

### 9.7. Backward compat

Readers tolerate either shape. Writers populate new columns from the typed
models. A rolling deploy is safe: old readers ignore new columns; old writers
skip new columns; readers pick up whatever is present.

---

## 10. Phase 8 — Deprecation (follow-up, not in this plan)

After one stable release of Phases 1–7 in production:

- Mark `EpisodeResult.history` and `actions_log` with `deprecated=True` docstrings.
- Mark `GameResult.agent_names` / `players_json` as deprecated; readers log
  a warning when falling back.
- Plan removal for the **next** major version (post the one shipping this plan).

Explicitly **out of scope** for this plan. Listed here so the migration story
is complete on paper.

---

## 11. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| IntervalPair invariants too strict, rejecting legitimate edge cases (e.g. `((0,0), ())` single-slot visit) | Medium | Medium | Explicit test matrix for edge cases; use `IntervalPair.from_loose()` factory for lenient parsing from JSON |
| JSON blob size balloons for large matches | Low (for El Farol at 100 days × 8 agents) | Medium (for future 100+ agent matches) | Phase 9.3 second-table design held in reserve |
| `capacity_threshold` rule change breaks historical interpretation | Low | High | Persist resolved threshold per match, not just the ratio. Already baked into Phase 5 design. |
| Tier-2 OTel fields never get populated | Medium | Low | They are all optional; dashboard drawer renders "—" when null |
| Dashboard migration hurts startup time on large DBs | Low | Medium | Additive nullable columns only; no backfill step. Alembic runs in seconds even on millions of rows. |
| Old game plugins break because `GameResult` shape changed | Medium | High | All new fields default; old plugins never set them, never read them; the single concrete change is `EpisodeResult.round_payoffs` which defaults empty. Existing games that don't care about per-day payoffs produce `[]` — no break. |
| `user_id="unknown"` cohort swallows everything in legacy view | High (for existing data) | Low (cosmetic) | Acceptable — flagged in dashboard UI as "unattributed". Backfill tooling is a separate effort. |

---

## 12. Rollback plan

Each phase is independently revertable:

- Phases 1–5 (in-memory models): revert the commit; no data migration
  required. Runtime keeps using legacy fields.
- Phase 6 (agent user_id): revert code; Alembic downgrade drops the column.
  Existing reads fall back to legacy shape.
- Phase 7 (dashboard SQL): Alembic downgrade drops added columns. Data
  written to them is lost, but no legacy data is touched. Legacy JSON blob
  readers continue functioning throughout.

**No irreversible step exists in this plan.** The commitment point is only
at Phase 8 (removal of deprecated fields), which is explicitly deferred.

---

## 13. Timeline estimate

Dev-only, single-contributor, with TDD overhead:

| Phase | Effort | Parallelisable with |
|---|---|---|
| 1 Typed models | 1 day | — |
| 2 Runner populates ActionRecord | 1 day | — |
| 3 ActionMapper intent | 0.5 day | 4, 5, 6 |
| 4 Per-day payoff | 0.5 day | 3, 5, 6 |
| 5 MatchConfig fields | 0.5 day | 3, 4, 6 |
| 6 Agent user_id | 1 day | 3, 4, 5 |
| 7 Dashboard SQL | 2 days | — |
| **Total** | **6–7 days sequential, 4–5 days with parallelisation** | |

Phase 7 dominates because of migration testing on both SQLite and Postgres.

---

## 14. Definition of Done (for this plan)

- All unit and integration tests green under `uv run pytest tests/ -v` and
  `uv run pytest tests/ -v -m "not slow"`.
- `uv run pyrefly check` clean on every modified file.
- `uv run ruff check .` clean.
- Alembic migration runs cleanly forward and backward on SQLite and Postgres.
- A smoke test plays a 5-day El Farol match end-to-end, writes the match to
  the dashboard DB, and reads it back with fully-populated `ActionRecord[]`,
  `DayAggregate[]`, `round_payoffs`, `agents`, and `MatchConfig`.
- The existing `atp test` command still works on all existing suites (no
  game plugin is broken by the model changes).
- CHANGELOG entry written describing the additive schema changes and the
  deprecation pointer for Phase 8.

---

## 15. Out of scope — explicitly deferred

These are called out so nobody wastes effort scoping them in:

- UI rendering of any of the new fields (separate plan).
- Per-move OTel span emission / Langfuse integration (separate plan).
- Per-move streaming via WebSocket / SSE (separate plan).
- Backfilling `user_id` on historical agents (a one-off data migration,
  outside this plan).
- Removing deprecated fields — Phase 8, explicitly follow-up.
- New evaluators or scoring logic — this plan is data-model only.
- Protocol-level `ATPRequest`/`ATPResponse` changes beyond what
  [`action_mapper.py`](../../atp-games/atp_games/mapping/action_mapper.py)
  already parses. Richer protocol metadata belongs in a Tier-1 protocol
  plan.

---

## 16. Cross-reference — dashboard feature → data model field

This is the traceability table the reviewer should read to convince
themselves the plan is sufficient:

| Dashboard element | Feeds from |
|---|---|
| Match header ("100 days · 16 slots · cap 60% · ≤2 visits") | `MatchConfig` |
| Day scrubber range | `MatchConfig.num_days` |
| Slot×day heatmap | `DayAggregate.slot_attendance` |
| Agent×day heatmap | `ActionRecord.payoff` grouped by agent, day |
| Agent×slot heatmap | `ActionRecord.picks` aggregated |
| Rank×day heatmap | Derived from `EpisodeResult.round_payoffs` |
| User×day heatmap (cohort view) | `AgentRecord.user_id` + `round_payoffs` |
| Leaderboard row (rank, payoff, visits/day, Δrank) | Derived from `ActionRecord` + `round_payoffs` |
| Leaderboard race chart | `round_payoffs` |
| Agent card — today strip | `ActionRecord` for current day |
| Agent card — sparkline | Per-agent `ActionRecord.payoff` series |
| Day calendar tile | `ActionRecord` + `DayAggregate` for that (agent, day) |
| Drawer — visits summary | `ActionRecord.intervals` |
| Drawer — per-slot breakdown | `ActionRecord.picks` × `DayAggregate.slot_attendance` |
| Drawer — intent line | `ActionRecord.intent` |
| Drawer — cumulative payoff, rank | Derived from `round_payoffs` + rank at day end |
| Drawer — tokens / latency / cost | `ActionRecord.tokens_in/out, decide_ms, cost_usd` (Tier-2, nullable) |
| Drawer — "Open in Langfuse" link | `ActionRecord.trace_id` (Tier-2, nullable) |
| KPIs: churn today | Rank diff derived from `round_payoffs` |
| KPIs: spread / median / p10 / p90 | Quantiles over cumulative payoff vector |
| KPIs: over-cap rate today | `DayAggregate.over_slots / num_slots` |

Every single visible element above is traceable to a field introduced in
Phases 1–7. If anything in the dashboard mockup is *not* in this table, it
means either (a) the plan is missing something and should be extended, or
(b) the element is presentation-layer only and needs no schema support.

---

*End of plan. Start work on Phase 1.*

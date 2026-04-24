# El Farol Tournament Support (Phase B) — Design Spec

**Date:** 2026-04-15
**Status:** v4 — revised after third architectural review (2026-04-15), ready for implementation plan
**Context:** Brainstorm session 2026-04-15
**Follow-up phase:** C — game-agnostic tournament refactor (out of scope here)

## 1. Goal and sequencing

Ship El Farol Bar Problem as the second `game_type` in the ATP MCP tournament
service, on par with Prisoner's Dilemma. Explicitly chose a two-phase
sequence:

- **Phase B (this spec):** El Farol as a concrete second game. Hardcoded
  game config. Minimal surface area. Validates that the service can host
  N-player games end-to-end.
- **Phase C (future, separate spec):** generic `GameAdapter` abstraction
  extracted from two working examples (PD + El Farol), enabling near-free
  addition of Stag Hunt / Public Goods / Blotto (MCP backlog items M).

Rationale: abstracting over a single example (PD) risks a wrong
abstraction; abstracting over two concrete examples is "rule of three"
done right.

## 2. Non-goals

- Generic game-agnostic service refactor — deferred to Phase C.
- Spectator mode, replay scrubber, charts — deferred (MCP backlog D, E, G).
- Custom El Farol configs from the tournament creator — deferred; Phase B
  ships a single hardcoded `EL_FAROL_V1_CONFIG`.
- El-Farol-specific dashboard UI rendering beyond the generic tournament
  table — YAGNI for B.
- Load testing — deferred (MCP backlog L, trigger: first 5+ participant
  tournament raises perf concerns).
- **No `atp-platform-sdk` major bump.** The SDK currently has zero
  tournament code (`packages/atp-sdk/atp_sdk/` exposes benchmark APIs
  only). MCP tournament clients talk JSON-RPC directly.
- **No wire-level breaking change for existing PD bots.** v3 design
  injects `game_type` server-side from the tournament record before
  pydantic validation, so existing PD clients sending bare
  `{"choice": "..."}` continue to work. (Earlier v2 of this spec had
  required clients to send `game_type`; that requirement is dropped.)

## 3. Scope and architecture

Changes are localized to three layers, in two PR-sized commits.

### 3.0 Pure refactor first (separate commit)

Before any El Farol code lands, normalize the PD-shaped action plumbing
that currently makes the service game-aware in places it shouldn't be.
This is a **prerequisite refactor commit** with no behavior change and
PD-only test coverage.

Affected call-sites in `service.py`:

- `:398` `action.action_data.get("choice", "")` — reading PD action.
- `:569` `action_vec[i] = a.action_data["choice"]` — round resolver.
- `:1049` `action_data={"choice": "defect"}` — timeout default.
- `_compute_payoffs` callers (the PD payoff matrix at `:577–584`).

Plus likely affected adjacents (verify during implementation):

- `tournament/events.py` — `round_completed` event payload shape.
- `mcp/tools.py` — `get_history` tool response.

Refactor goal: tournament service reads/writes `action.action_data`
opaquely as `dict[str, Any]`, never touching `"choice"`. Game-specific
extraction goes through three new game methods (introduced in §3.1).
PD continues to work identically. Tests: existing PD unit/integration
suite unchanged, all green.

This split gives bisectable history and an easier review.

### 3.1 Game layer (`game-environments/game_envs/games/`)

**`el_farol.py`** — three new methods on `ElFarol`, no changes to `step`
/ `ActionSpace` / payoffs / `get_payoffs`:

- `format_state_for_player(round_number, total_rounds, participant_idx,
  action_history, cumulative_scores) -> dict` — N-player-aware state
  formatter. Returns `your_history`, `attendance_by_round` (aggregate,
  not per-player histories), `capacity_threshold`,
  `your_cumulative_score`, `all_scores`, `your_participant_idx`,
  `num_slots`, `action_schema`. **Does NOT return `pending_submission`**
  — that field is service-layer knowledge (whether this participant
  has an Action row in the current round), injected by the tournament
  service after calling the formatter. Keeps the game class free of
  service-state concerns. **Note:** El Farol state uses
  `pending_submission`, not `your_turn` — the latter carries
  sequential-turn semantics that don't apply to a simultaneous
  N-player round (see review #3 in §13).
- `validate_action(raw: dict) -> dict` — **strict**: raises
  `ValidationError` on malformed input (non-list, out-of-range slot,
  duplicate, >`MAX_SLOTS_PER_DAY`); returns canonical form (sorted
  slots). Used for **client submissions** (`submit_action`). Wraps but
  does NOT replace existing `ElFarolActionSpace.sanitize` (which
  remains the **permissive** path used for replay/migration scenarios
  and for synthesizing default-on-timeout actions). Boundary rule:

  | Path | Method | Behavior on bad input |
  |------|--------|-----------------------|
  | Client submission via `submit_action` / MCP `make_move` | `validate_action` | 400/422 ValidationError |
  | Internal replay, default-on-timeout synthesis | `sanitize` | best-effort cleanup |

- `default_action_on_timeout() -> dict` — returns `{"slots": []}`
  (stay home).

**`prisoners_dilemma.py`** — mirror methods (`validate_action` returns
`{"choice": "..."}`, `default_action_on_timeout` returns
`{"choice": "defect"}`) so the tournament service treats both games
uniformly via the same trio of methods. PD's existing `your_turn`
field in `format_state_for_player` is preserved for back-compat;
unification with El Farol's `pending_submission` is a Phase C concern.

Key design decision — for El Farol state format, players see
**aggregate attendance per round** (`list[list[int]]` of length
`num_slots`), not individual histories of all N opponents. Reasons:

- Faithful to Arthur's El Farol formulation: players act on attendance
  signal, not individual decisions.
- At N=20, R=100, per-player histories would be 20×100 payload per
  `get_current_state` call.
- Own history is always available separately.

### 3.2 Tournament service (`packages/atp-dashboard/atp/dashboard/tournament/`)

After the §3.0 refactor lands, the El Farol-enabling delta is:

1. `_SUPPORTED_GAMES = {"prisoners_dilemma", "el_farol"}`.

2. Replace the `_GAME_INSTANCES: dict` module global with
   `_PD_SINGLETON` + a cached `_game_for(tournament)` factory. PD stays
   singleton (stateless w.r.t. tournaments). El Farol is per-`num_players`
   because its config has to be built from the hardcoded preset combined
   with that N. Cache so the factory is symmetric in cost with
   `_PD_SINGLETON` (avoid re-instantiating per `submit_action` /
   `get_current_state` / `_resolve_round` call):

   ```python
   _PD_SINGLETON = PrisonersDilemma()

   # frozen module constants — NO new pydantic class, reuse the existing
   # game-layer ElFarolConfig dataclass directly
   _EL_FAROL_V1_NUM_SLOTS = 16
   _EL_FAROL_V1_THRESHOLD_RATIO = 0.6
   _EL_FAROL_V1_MIN_TOTAL_HOURS = 0

   @functools.lru_cache(maxsize=64)
   def _el_farol_for(num_players: int) -> ElFarol:
       cap = max(1, int(_EL_FAROL_V1_THRESHOLD_RATIO * num_players))
       cfg = ElFarolConfig(
           num_players=num_players,
           num_slots=_EL_FAROL_V1_NUM_SLOTS,
           capacity_threshold=cap,
           min_total_hours=_EL_FAROL_V1_MIN_TOTAL_HOURS,
       )
       return ElFarol(cfg)

   def _game_for(tournament: Tournament):
       gt = tournament.game_type
       if gt == "prisoners_dilemma":
           return _PD_SINGLETON
       if gt == "el_farol":
           return _el_farol_for(tournament.num_players)
       raise ValidationError(f"unsupported game_type {gt!r}")
   ```

   No new `ElFarolConfig` class is introduced anywhere. The existing
   game-layer `ElFarolConfig(GameConfig)` (`game_envs/games/el_farol.py:123`)
   is reused as-is to avoid name collision and double-interpretation
   of `threshold` (ratio) vs `capacity_threshold` (absolute).

3. `create_tournament` validation:
   - PD: `num_players == 2` (unchanged behavior).
   - El Farol: `2 <= num_players <= 20` for Phase B.
     **Bound rationale:** untested service-side perf at N>20. The
     existing duration cap in `service.py:111–120` independently
     enforces `pending_wait + R × round_deadline_s ≤
     (ATP_TOKEN_EXPIRE_MINUTES − 10) × 60`, and N is not part of that
     formula, so duration math does not need a separate dynamic check.
     `# TODO(phase-c): raise after MCP-backlog L load test.`
   - Unknown game_type → `ValidationError`.

4. `submit_action` validation: replace PD-specific
   `action["choice"] in options` with this sequence. The Tournament
   record is already fetched exactly once here (verified:
   `mcp/tools.py:117–124` passes only `tournament_id`, no pre-fetch
   in the MCP handler).

   ```python
   tournament = await self._session.get(Tournament, tournament_id)
   game = _game_for(tournament)
   # server-side discriminator injection (review v3 #1):
   incoming_gt = raw_action.get("game_type")
   if incoming_gt is not None and incoming_gt != tournament.game_type:
       raise ValidationError(
           f"action game_type {incoming_gt!r} does not match "
           f"tournament game_type {tournament.game_type!r}"
       )
   action_with_type = {**raw_action, "game_type": tournament.game_type}
   try:
       typed = TypeAdapter(TournamentAction).validate_python(action_with_type)
   except PydanticValidationError as e:
       # Review v4 #2: surface tournament context so debuggers don't
       # have to guess why an opaque "field X required" fired.
       expected = _action_hint_for(tournament.game_type)
       raise ValidationError(
           f"invalid action for tournament {tournament_id} "
           f"(game_type={tournament.game_type!r}); "
           f"expected fields: {expected}; "
           f"pydantic detail: {e.errors()[:1]}"
       ) from e
   canonical = game.validate_action(typed.model_dump(exclude={"game_type"}))
   ```

   where `_action_hint_for(gt)` returns a short literal like
   `"{choice: 'cooperate'|'defect'}"` or `"{slots: list[int], 0..N-1,
   unique, max 8}"`. Store `canonical` in `Action.action_data` (without
   `game_type`, since the parent `Tournament.game_type` is the source
   of truth). PD bots continue to send `{"choice": "..."}`; El Farol
   bots send `{"slots": [...]}`; both routes work without `game_type`
   on the wire.

5. Round deadline handler: replace hardcoded `{"choice": "defect"}`
   fallback with `game.default_action_on_timeout()`.

6. `get_current_state`: service computes `pending_submission` from
   DB state (does this participant have an Action row in the current
   round?), calls `game.format_state_for_player(...)`, merges
   `{"pending_submission": pending, "game_type": tournament.game_type}`
   into the returned dict, then parses into the discriminated union via
   `TypeAdapter(RoundState).validate_python(merged)`. PD's state is
   shaped analogously: service injects `your_turn` (legacy name)
   computed the same way, rather than the game class knowing about
   submission state. State responses **do** carry `game_type` —
   clients use it for `match` / pattern-dispatch, which is the reason
   the discriminator is useful server→client.

No database migration. `Action.action_data` is already JSON;
`Tournament.game_type` is already `String(100)`.

### 3.3 Score aggregation method (Phase B = per-round MVP)

Decision, fixed in this spec to remove ambiguity:

**Phase B scores leaderboard by per-round payoff sum**, exactly as PD
does today. For El Farol that yields:

```
final_score(player) = Σ_round (happy_slots − crowded_slots)
                    = t_happy(player) − t_crowded(player)
```

Implications:

- `service.py:_resolve_round` continues to do
  `cumulative_scores[i] += action.payoff` after computing per-round
  payoffs. No new "finalize" stage.
- The `get_payoffs()` terminal ratio (`t_happy / max(t_crowded, 0.1)`)
  and `min_total_hours` disqualification mechanism in
  `el_farol.py:359–376` are **not invoked** by the tournament service
  in Phase B. With `min_total_hours = 0` in the V1 preset, the DQ path
  is inactive anyway, so per-round sum and `t_happy − t_crowded`
  ranking coincide.
- Per-round payoffs come from `ElFarol.step(actions)` already, so the
  service's existing per-round payoff plumbing applies unchanged.

Phase C may introduce a `game.finalize_scores(action_history)` hook for
games whose terminal score is non-summable (e.g. ratio, max-of, DQ).

### 3.4 Contract layer (pydantic discriminated unions)

Lives in `packages/atp-dashboard/atp/dashboard/tournament/schemas.py`.
**Not re-exported via the SDK** (no SDK tournament module exists).

```python
class PDAction(BaseModel):
    game_type: Literal["prisoners_dilemma"]   # injected server-side
    choice: Literal["cooperate", "defect"]

class ElFarolAction(BaseModel):
    game_type: Literal["el_farol"]            # injected server-side
    slots: list[int] = Field(..., max_length=MAX_SLOTS_PER_DAY)
    # field_validator: uniqueness + non-negative; per-tournament
    # num_slots upper bound enforced at service layer via game.validate_action

TournamentAction = Annotated[
    PDAction | ElFarolAction,
    Field(discriminator="game_type"),
]
```

```python
class PDRoundState(BaseModel):
    game_type: Literal["prisoners_dilemma"]
    your_history: list[str]
    opponent_history: list[str]
    your_cumulative_score: float
    opponent_cumulative_score: float
    round_number: int
    total_rounds: int
    your_turn: bool                  # preserved for PD back-compat
    action_schema: dict

class ElFarolRoundState(BaseModel):
    game_type: Literal["el_farol"]
    your_history: list[list[int]]
    attendance_by_round: list[list[int]]
    capacity_threshold: int
    your_cumulative_score: float
    all_scores: list[float]
    your_participant_idx: int
    num_slots: int
    round_number: int
    total_rounds: int
    pending_submission: bool         # NOT your_turn — see §3.1
    action_schema: dict

RoundState = Annotated[
    PDRoundState | ElFarolRoundState,
    Field(discriminator="game_type"),
]
```

No new game-config pydantic model is introduced. The game-layer
`ElFarolConfig(GameConfig)` is the single source of truth.

## 4. MCP tools

Tool names unchanged: `join_tournament`, `make_move`,
`get_current_state`, `list_tournaments`, `get_tournament`,
`get_history`, `leave_tournament`.

- `make_move(tournament_id, action: dict)` — server reads
  `tournament.game_type`, injects it into the action dict, then parses
  via `TypeAdapter(TournamentAction)`. **Wire-compatible with current
  PD bots** (no `game_type` field required from the client).
  Tool description for LLM clients:
  `"action: dict whose required fields depend on the tournament's
  game_type. For prisoners_dilemma: {choice: 'cooperate' | 'defect'}.
  For el_farol: {slots: list[int], 0..num_slots-1, max 8 entries,
  unique}."`
  Optional: client may include `game_type` for self-documentation; if
  present and mismatched with the tournament, server returns 422
  `game_type mismatch` (see §3.2 step 4).
- `get_current_state(tournament_id)` — returns `RoundState.model_dump()`;
  the dict carries `game_type` for client-side pattern-matching.
- `list_tournaments(game_type: str | None = None)` — new optional filter.

There is **no wire-breaking change** in this PR for existing PD bots.

## 5. Demo-game updates

- `demo-game/agents/el_farol_agent.py` — finalize against the El Farol
  state schema (`pending_submission`, `attendance_by_round`,
  `capacity_threshold`). Client sends bare `{"slots": [...]}`.
- `demo-game/agents/openai_game_agent.py` — **no changes required**
  (server injects `game_type`).
- `demo-game/suites/el_farol_llm_vs_builtin.yaml` — point at MCP
  tournament endpoint with `game_type: el_farol`.
- `demo-game/Containerfile.el-farol` — already exists; bring up to date
  with new agent code if needed.
- `demo-game/compose.el-farol.yml` (new) — N=5 (3 LLM bots + 2
  built-in: greedy and random El Farol strategies from
  `el_farol_strategies.py`).

## 6. Error handling and edge cases

Enumerated during design, to be covered by tests:

- All N players timeout in one round → each gets `{"slots": []}`,
  `step()` yields 0 points, tournament proceeds normally.
- Out-of-range / duplicate slots from a client → `validate_action`
  rejects before DB write; 400 to client. (Same input through the
  internal sanitize path during replay would be silently cleaned —
  see §3.1 boundary table.)
- Concurrent last-action submits → existing `existing action` guard
  + transactional boundary resolves the round exactly once (current
  N-player code path, unchanged).
- `num_players` outside `[2, 20]` for El Farol → rejected at create.
- Unknown `game_type` at create → rejected. At `submit_action` /
  `get_current_state` the tournament record is the authority, so
  there is no ambiguity at runtime.
- Empty `action_history` at round 1 → `your_history=[]`,
  `attendance_by_round=[]`, all scores zero, `pending_submission=true`.
- Client sends `{"game_type": "el_farol", "slots": [...]}` to a PD
  tournament → server detects mismatch, 422 with explicit message
  (see §3.2 step 4). Symmetric case for PD action sent to El Farol.

## 7. Testing strategy

### 7.1 Unit tests

- `tests/unit/games/test_el_farol_state_format.py` —
  `format_state_for_player` shape, `validate_action` edge cases,
  `default_action_on_timeout`, canonical sorting, aggregate attendance
  computation, `validate_action` vs `sanitize` boundary behavior on the
  same bad input, `pending_submission` toggles correctly across rounds.
- `tests/unit/tournament/test_schemas_discriminator.py` — explicit
  cases. Schema policy: `model_config = ConfigDict(extra="forbid")`
  on both action and state models, so extra fields surface as
  validation errors rather than silently pass (tested via case 4).
  1. `TypeAdapter(TournamentAction)` parses
     `{"game_type": "prisoners_dilemma", "choice": "cooperate"}` →
     `PDAction`.
  2. Parses `{"game_type": "el_farol", "slots": [0, 3]}` →
     `ElFarolAction`.
  3. Missing required field per type → ValidationError listing the
     missing field.
  4. PD discriminator + El Farol fields (`{"game_type":
     "prisoners_dilemma", "slots": [0]}`) → ValidationError citing
     BOTH missing `choice` AND extra `slots` (given `extra="forbid"`).
  5. Unknown discriminator → ValidationError listing the supported
     literals.
  6. Round-trip: `model.model_dump()` → `validate_python(dump)` yields
     equal model.
  7. `RoundState` union round-trip for both PD and El Farol shapes.
- Extend `tests/unit/dashboard/tournament/test_service.py` —
  `create_tournament(el_farol, N in [1, 2, 20, 21])`, `_game_for`
  returns game with configured `num_players` and `capacity_threshold`
  correctly derived from the V1 ratio, `_el_farol_for(N)` is cached
  (second call returns same instance), `submit_action` mismatch case
  (PD action sent to El Farol tournament returns ValidationError with
  tournament-aware hint), `submit_action` error-message quality check
  (PD bot without `game_type` sending to El Farol tournament receives
  message mentioning the expected slots schema — test per review v4 #2).

  **Fixture hygiene:** any test that monkey-patches
  `_EL_FAROL_V1_NUM_SLOTS`, `_EL_FAROL_V1_THRESHOLD_RATIO`, or
  `_EL_FAROL_V1_MIN_TOTAL_HOURS` must call
  `_el_farol_for.cache_clear()` in teardown (or autouse fixture) to
  avoid cross-test leakage from the process-scoped `@lru_cache`.

### 7.2 Integration tests

- `tests/integration/tournament/test_el_farol_flow.py` — full N=5, R=3
  flow through the service: join → active → all submit → round
  resolves → state per participant → deadline timeout → tournament
  completes → leaderboard correct (per-round-sum scoring).
- Parallel PD regression test over the same generic code paths to
  confirm the §3.0 refactor + §3.2 service changes did not break PD.
- New regression: existing PD bot fixtures (no `game_type` on the
  wire) continue to round-trip cleanly via server-side injection.

### 7.3 End-to-end with LLM bots

Mirrors the PD validation that produced the first successful 30-round
run.

- **Pre-E2E load smoke (built-in bots only, no LLM):** **N=20, R=30**
  on a developer machine. Pass criteria: no service timeouts, no
  excessive memory growth, all rounds resolve within
  `round_deadline_s`, `get_current_state` p95 latency stays flat
  round-over-round (catches O(R) payload bloat from
  `attendance_by_round`). N bumped from 10 in v2 (review v3 #2);
  R bumped from 10 in v3 (review v4 #3).
- **Full E2E:** `demo-game/compose.el-farol.yml` with N=5 (3 LLM bots
  + 2 built-in: greedy, random), 20 rounds, `round_deadline_s=30`.
  **Pass criteria focus on plumbing, NOT strategy quality** (review
  v4 #1): tournament completes without service timeouts, all 5 bots
  finish every round (no stuck participants), MCP event-bus logs
  clean (no 5xx, no unhandled exceptions), final leaderboard populates
  for all participants, structured logs show `game_type=el_farol` on
  every resolution entry. Phase B does NOT assert that LLM bots
  outperform random — see §12 for why a "LLM > random" criterion would
  either pass trivially or fail on variance.

### 7.4 Coverage

Existing tournament-service floor is 83% (per commit `8dfe0cc`).
El-Farol-specific new code (service + game) should itself be ≥85%
covered to avoid pushing the floor down further. The discriminator
schema cases enumerated in §7.1 (1–7) must each be a separate test
function so coverage tools count them individually.

## 8. Phase C hooks (explicitly deferred)

These are called out so the abstraction story in Phase C has a clear
starting point:

- `GameAdapter` protocol with `format_state_for_player`,
  `validate_action`, `default_action_on_timeout`, `canonical_action`,
  `parse_action`, optional `finalize_scores(action_history)` for
  non-summable terminal metrics.
- Game-config in create_tournament API (`game_config: dict`
  validated into `PDConfig | ElFarolConfig | ...` discriminated
  union — at that point the dashboard layer can introduce a typed
  config model without colliding with the game-layer dataclass.
- MCP registry-based dynamic tool schemas.
- Raise El Farol N upper bound after MCP-backlog L load test.
- **Unify naming:** PD's `your_turn` → `pending_submission` across all
  game state schemas (deferred for back-compat in Phase B).
- **Move `MAX_SLOTS_PER_DAY` into `ElFarolConfig`** when `num_slots`
  becomes configurable per tournament. Today it stays a game-rule
  constant in the El Farol module.

## 9. Release plan

Two PRs, in order:

1. **PR-1 (refactor):** §3.0 pure refactor — game-agnostic action
   plumbing + three new methods on `PrisonersDilemma`
   (`validate_action`, `default_action_on_timeout`, normalized
   `format_state_for_player` input shape). PD-only tests, all green.
   Diff target ≤300 LOC.
2. **PR-2 (feature):** El Farol-specific code — game methods, service
   delta from §3.2, schemas from §3.4, MCP tool description updates,
   demo-game agent + suite + compose, integration + E2E tests.
   Diff target ≤700 LOC. **No wire-breaking change** for PD bots.

Both PRs merge to `main`. PR-2 commit message includes `[deploy]` to
trigger the Namecheap VPS rebuild.

After PR-2 merges and the pre-E2E load smoke + full E2E pass on
production, run the **smoke tournament** (N=5, R=20) on the live MCP
endpoint before declaring Phase B complete.

No PyPI / SDK release in this phase (see §2 non-goals).

## 10. Observability

Phase B adds **structured log fields**, not a dashboard. The goal is
that during the first El Farol incident on production, grepping the
service logs gives unambiguous answers without code archaeology.

Add to existing tournament service log records (use Python `logger`
`extra={...}` so structured ingestion picks them up):

- `game_type` — on every round-resolution and submit-action log line.
- `tournament_id`, `round_number` — already present, ensure consistent.
- `round_resolution_ms` — wall-clock duration of `_resolve_round`,
  emitted on the round-resolved log entry.
- `validation_error_path` — `"client_submission"` |
  `"timeout_synthesis"` | `"deadline_check"` — distinguishes
  `validate_action` failures from `sanitize` cleanups.

Phase C will define alerts/dashboards on top of these fields once
real prod data exists.

## 11. References

- Current PD tournament design: `docs/superpowers/specs/2026-04-10-mcp-tournament-server-design.md`
- Plan 2a (PD-only shipped): `docs/superpowers/specs/2026-04-11-mcp-tournament-plan-2a-design.md`
- MCP backlog (A = El Farol next): memory `project_mcp_backlog.md`
- El Farol game: `game-environments/game_envs/games/el_farol.py`
- Demo agent (existing): `demo-game/agents/el_farol_agent.py`

## 12. Risks and known degeneracies

- **Trivial "stay home" Nash with `min_total_hours = 0`.** Per review
  v3 #4. With per-round payoff `happy − crowded` and zero participation
  penalty, an LLM bot that learns to always send `{"slots": []}` scores
  exactly 0 — never negative, but never positive either. Going to
  predictably non-crowded slots beats stay-home, so the strategy is
  not strictly optimal, but the game can degenerate when bots fail to
  predict crowding.
  - **Mitigation in Phase B:** none architectural. Setting
    `min_total_hours > 0` without invoking `get_payoffs()` makes the
    field decorative, contradicting §3.3's per-round MVP scoring.
  - **Phase C plan:** introduce `finalize_scores(action_history)`
    hook so DQ via `min_total_hours` actually fires at end-of-tournament.
    Then raising `min_total_hours` becomes meaningful.
  - **What this means for the first El Farol tournament:** if the LLM
    bots converge on stay-home, accept it as a documented Phase B
    outcome and treat it as motivation to ship Phase C terminal
    scoring. Built-in `greedy` strategy participates regardless, so
    leaderboard won't be all zeros.
- **`MAX_SLOTS_PER_DAY = 8` is implicit half-of-num_slots.** With
  `num_slots = 16` the constant happens to equal `num_slots / 2`. Phase
  C, when num_slots becomes configurable, must move `MAX_SLOTS_PER_DAY`
  into `ElFarolConfig` to preserve this game-rule relationship.
- **Untested service perf at N > 20.** Bound enforced at create-time;
  pre-E2E load smoke validates the boundary.
- **Silent `min_total_hours` if raised without Phase C.** Raising
  `_EL_FAROL_V1_MIN_TOTAL_HOURS > 0` without introducing the
  `finalize_scores` hook (Phase C §8) makes the field decorative: the
  game accumulates `_t_happy`/`_t_crowded` but the tournament service
  never invokes `get_payoffs()` to apply DQ. Result: bots that
  "violate" `min_total_hours` are not actually disqualified, creating
  a misleading config surface. **Mitigation:** a unit test in
  `test_service.py` asserts that raising the constant without the
  Phase C hook raises a loud `NotImplementedError` at service startup.
  (Implementation: simple assert in module init that
  `_EL_FAROL_V1_MIN_TOTAL_HOURS == 0` until the hook lands.)

## 13. Review revisions log

This spec went through three architectural reviews on 2026-04-15.

### v1 → v2

- Dropped duplicate `ElFarolConfig` pydantic model in dashboard layer
  (collision with game-layer `ElFarolConfig(GameConfig)`).
- Fixed score aggregation method ambiguity: Phase B uses per-round
  payoff sum exclusively. `get_payoffs()` and DQ remain in the game
  class but are not invoked by the service.
- Promoted `action_history` normalization to a separate prerequisite
  PR (§3.0).
- Added explicit `validate_action` (strict) vs `sanitize` (permissive)
  boundary table.
- Removed all SDK v3 / `atp-platform-sdk 3.0.0` framing: the SDK has
  zero tournament code today.
- Replaced `ElFarolConfig.with_num_players` classmethod with direct
  constructor.
- Lowered N upper bound from 50 to 20 with explicit rationale.
- Added Pre-E2E load smoke (N=10, R=10, no LLM).
- Added MAX_SLOTS_PER_DAY callout in MCP tool description.

### v2 → v3

- **Server-side `game_type` injection** (review #1): wire-level
  breaking change for PD bots is eliminated. Server reads
  `tournament.game_type` and prepends it to the raw action dict before
  pydantic validation. Mismatch (client sends one game_type, tournament
  is another) → 422 with explicit message. Removes the entire fixture-
  update story for existing PD tests.
- **Renamed `your_turn` → `pending_submission` in El Farol state**
  (review #3): N-player simultaneous rounds have no notion of "turn";
  semantics is "have I submitted yet this round". PD keeps `your_turn`
  for back-compat, with a Phase C unification hook in §8.
- **`@lru_cache(maxsize=64)` on El Farol factory** (review #5):
  removes the asymmetry between cached `_PD_SINGLETON` and
  per-call-instantiated El Farol. Cached by `num_players` since all
  other config is constant in V1.
- **Bumped pre-E2E load smoke to N=20** (review #2): the smoke now
  validates the actual upper end of the supported N range, not the
  midpoint.
- **Simplified N≤20 rationale** (review #6): removed the misleading
  "deadline math" reference. Duration cap is already runtime-checked
  in `service.py:111–120` and N is not part of that formula. The bound
  is purely about untested perf.
- **Added §10 Observability** with structured log fields per game_type
  (review #10).
- **Added §12 Risks** documenting the `min_total_hours = 0` degenerate
  Nash and the implicit `MAX_SLOTS_PER_DAY = num_slots / 2` relationship
  (review #4 and #7).
- **Added explicit cross-game_type validation cases to §6 and §7.1**
  (review #9).
- **Pushed back on review #7** (`MAX_SLOTS_PER_DAY` into `ElFarolConfig`
  in Phase B): rejected for B as a game-rule constant; deferred to
  Phase C when `num_slots` becomes configurable. Recorded in §8 hooks.

### v3 → v4

- **Dropped strategy-quality pass-criterion from full E2E** (review
  v4 #1). Previous "LLM mean score > random mean score" conflicted
  with §12's documented acceptance of stay-home degeneracy: the
  criterion would pass trivially (both at 0) or fail on variance.
  Phase B E2E now validates plumbing only (completes, logs clean,
  all bots finish every round, leaderboard populates). Strategic
  quality is a Phase C concern once `finalize_scores` enables DQ.
- **Wrapped pydantic `ValidationError` with tournament context in
  `submit_action`** (review v4 #2). When a PD bot sends `{"choice":
  "..."}` to an El Farol tournament, raw pydantic says "field slots
  required"; with the wrapper, the message cites the tournament id,
  its `game_type`, and an expected-fields hint
  (`_action_hint_for(gt)`).
- **Bumped pre-E2E load smoke from R=10 to R=30** (review v4 #3):
  catches potential O(R) `attendance_by_round` payload growth that
  R=10 would not surface. Added explicit `get_current_state` p95
  latency flatness as a pass criterion.
- **Moved `pending_submission` / `your_turn` computation out of
  `format_state_for_player`** (review v4 minor): the game class no
  longer knows about submission state. Service computes it from DB
  Action rows and injects into the state dict before pydantic union
  parse. Cleaner boundary between game rules and service state.
- **Added `extra="forbid"` to discriminator test expectations**
  (review v4 minor): schema policy is explicit, test case 4 asserts
  the expected behavior.
- **Added `_el_farol_for.cache_clear()` fixture hygiene note to §7.1**
  (review v4 minor): process-scoped `@lru_cache` cross-test leakage
  when V1 constants are monkey-patched.
- **Verified single Tournament fetch in submit_action** (review v4
  minor): MCP handler at `mcp/tools.py:117–124` passes only
  `tournament_id`; no pre-fetch duplication. Noted in §3.2 step 4.
- **Added `min_total_hours > 0 without Phase C` risk to §12** with
  startup-assert mitigation (review v4 minor extension).

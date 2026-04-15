# El Farol Tournament Support (Phase B) — Design Spec

**Date:** 2026-04-15
**Status:** revised after architectural review (2026-04-15), ready for implementation plan
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
  only). MCP tournament clients talk JSON-RPC directly. Therefore the
  only "wire" change is the MCP `make_move` action schema (a
  `game_type` discriminator field), and the only consumers to update
  are internal `demo-game` bots.

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
  formatter analogous to the PD one. Returns `your_history`,
  `attendance_by_round` (aggregate, not per-player histories),
  `capacity_threshold`, `your_cumulative_score`, `all_scores`,
  `your_participant_idx`, `num_slots`, `action_schema`.
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
uniformly via the same trio of methods.

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
   `_PD_SINGLETON` + a `_game_for(tournament)` factory. PD stays
   singleton (stateless w.r.t. tournaments). El Farol is per-tournament
   because `num_players` is configurable, and its config has to be
   built from the hardcoded preset combined with that N:

   ```python
   _PD_SINGLETON = PrisonersDilemma()

   # frozen module constant — NO new pydantic class, reuse the existing
   # game-layer ElFarolConfig dataclass directly
   _EL_FAROL_V1_NUM_SLOTS = 16
   _EL_FAROL_V1_THRESHOLD_RATIO = 0.6
   _EL_FAROL_V1_MIN_TOTAL_HOURS = 0

   def _game_for(tournament: Tournament):
       gt = tournament.game_type
       if gt == "prisoners_dilemma":
           return _PD_SINGLETON
       if gt == "el_farol":
           n = tournament.num_players
           # ratio → absolute conversion (only place this happens)
           cap = max(1, int(_EL_FAROL_V1_THRESHOLD_RATIO * n))
           cfg = ElFarolConfig(
               num_players=n,
               num_slots=_EL_FAROL_V1_NUM_SLOTS,
               capacity_threshold=cap,
               min_total_hours=_EL_FAROL_V1_MIN_TOTAL_HOURS,
           )
           return ElFarol(cfg)
       raise ValidationError(f"unsupported game_type {gt!r}")
   ```

   No new `ElFarolConfig` class is introduced anywhere. The existing
   game-layer `ElFarolConfig(GameConfig)` (`game_envs/games/el_farol.py:123`)
   is reused as-is to avoid name collision and double-interpretation
   of `threshold` (ratio) vs `capacity_threshold` (absolute).

3. `create_tournament` validation:
   - PD: `num_players == 2` (unchanged behavior).
   - El Farol: `2 <= num_players <= 20` for Phase B. **Bound rationale:**
     deadline math (`(ATP_TOKEN_EXPIRE_MINUTES − 10) × 60` budget vs
     `pending_wait + R × round_deadline_s`) plus untested perf at
     larger N. `# TODO(phase-c): raise after MCP-backlog L load test.`
   - Unknown game_type → `ValidationError`.

4. `submit_action` validation: replace PD-specific
   `action["choice"] in options` with delegated
   `game.validate_action(action)`; store canonical form in
   `Action.action_data`.

5. Round deadline handler: replace hardcoded `{"choice": "defect"}`
   fallback with `game.default_action_on_timeout()`.

6. `get_current_state`: parse the formatter's dict into the
   discriminated union via
   `TypeAdapter(RoundState).validate_python(formatted)`.

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
    game_type: Literal["prisoners_dilemma"]
    choice: Literal["cooperate", "defect"]

class ElFarolAction(BaseModel):
    game_type: Literal["el_farol"]
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
    your_turn: bool
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
    your_turn: bool
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

- `make_move(tournament_id, action: dict)` — parses the dict via
  `TypeAdapter(TournamentAction)`. Missing `game_type` →
  422 with explicit "missing discriminator game_type: either
  'prisoners_dilemma' or 'el_farol'".
- Tool description (visible to LLM clients) updated to:
  `action: dict with game_type ('prisoners_dilemma' or 'el_farol')
  plus type-specific fields (PD: choice; El Farol: slots).`
- `get_current_state(tournament_id)` — returns `RoundState.model_dump()`
  as before. Clients pattern-match on `game_type`.
- `list_tournaments(game_type: str | None = None)` — new optional filter.
- El Farol tool description should also call out: `MAX_SLOTS_PER_DAY = 8`
  (player can attend at most half the day's slots when num_slots=16),
  to surface a non-obvious game constraint to LLM-driven bots.

This is the **wire-level breaking change**: any MCP client sending
PD `make_move` actions today without `game_type` will get 422. Affected
consumers:

- `demo-game/agents/openai_game_agent.py` (PD bot) — needs to wrap
  current `{"action": "..."}` payload as
  `{"game_type": "prisoners_dilemma", "choice": "..."}`.
- `demo-game/agents/el_farol_agent.py` — authored against the new
  schema from the start.
- `tests/e2e/test_mcp_pd_tournament.py` and the 30-round PD reconnect
  test — update fixtures to include `game_type`.

## 5. Demo-game updates and bot client changes

- `demo-game/agents/el_farol_agent.py` — finalize against the new MCP
  action schema (`ElFarolAction`-shaped dict). Uses `state.game_type`
  branch for state interpretation.
- `demo-game/agents/openai_game_agent.py` — minimal patch to add
  `game_type` to the submitted action dict; everything else identical.
- `demo-game/suites/el_farol_llm_vs_builtin.yaml` — point at MCP
  tournament endpoint with `game_type: el_farol`.
- `demo-game/Containerfile.el-farol` — already exists (verified during
  exploration); bring up to date with new agent code.
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
- Unknown `game_type` → rejected at create and at all read/write
  call-sites (defense in depth).
- Empty `action_history` at round 1 → `your_history=[]`,
  `attendance_by_round=[]`, all scores zero.
- PD bot client sends action without `game_type` field → 422 from
  MCP `make_move`. Single-line fix required at every existing PD bot
  call-site (only `demo-game` is affected; no external consumers).

## 7. Testing strategy

### 7.1 Unit tests

- `tests/unit/games/test_el_farol_state_format.py` — `format_state_for_player`
  shape, `validate_action` edge cases, `default_action_on_timeout`, canonical
  sorting, aggregate attendance computation, `validate_action` vs `sanitize`
  boundary behavior on the same bad input.
- `tests/unit/tournament/test_schemas_discriminator.py` — round-trip
  `model_dump` ↔ `validate_python` for both games, missing discriminator
  error, wrong-shape combinations (PD game_type with slots, El Farol game_type
  with choice), 422 on missing `game_type`.
- Extend `tests/unit/dashboard/tournament/test_service.py` —
  `create_tournament(el_farol, N in [1, 2, 20, 21])`, `_game_for` returns
  game with configured `num_players` and `capacity_threshold` correctly
  derived from the V1 ratio.

### 7.2 Integration tests

- `tests/integration/tournament/test_el_farol_flow.py` — full N=5, R=3
  flow through the service: join → active → all submit → round resolves →
  state per participant → deadline timeout → tournament completes →
  leaderboard correct (per-round-sum scoring).
- Parallel PD regression test over the same generic code paths to
  confirm the §3.0 refactor + §3.2 service changes did not break PD.

### 7.3 End-to-end with LLM bots

Mirrors the PD validation that produced the first successful 30-round
run.

- **Pre-E2E load smoke (new):** N=10, R=10 with built-in random
  strategies only (no LLM) on a developer machine. Pass criteria:
  no service timeouts, no excessive memory growth, all rounds resolve
  within `round_deadline_s`. This is a cheap pre-flight before
  spending tokens on LLM bots and validates the upper end of the V1
  N range.
- **Full E2E:** `demo-game/compose.el-farol.yml` with N=5 (3 LLM bots
  + 2 built-in: greedy, random), 20 rounds, `round_deadline_s=30`.
  Pass criteria: tournament completes without service timeouts, all
  5 bots finish, MCP event-bus logs clean, LLM mean score > random
  mean score.

### 7.4 Coverage

Existing tournament-service floor is 83% (per commit `8dfe0cc`).
El-Farol-specific new code (service + game) should itself be ≥85%
covered to avoid pushing the floor down further.

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
  config model without colliding with the game-layer dataclass,
  because Phase C is the natural moment to sort out that boundary).
- MCP registry-based dynamic tool schemas.
- Raise El Farol N upper bound after MCP-backlog L load test.

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
   Diff target ≤700 LOC.

Both PRs merge to `main`. PR-2 commit message includes `[deploy]` to
trigger the Namecheap VPS rebuild.

After PR-2 merges and the pre-E2E load smoke + full E2E pass on
production, run the **smoke tournament** (N=5, R=20) on the live MCP
endpoint before declaring Phase B complete.

No PyPI / SDK release in this phase (see §2 non-goals).

## 10. References

- Current PD tournament design: `docs/superpowers/specs/2026-04-10-mcp-tournament-server-design.md`
- Plan 2a (PD-only shipped): `docs/superpowers/specs/2026-04-11-mcp-tournament-plan-2a-design.md`
- MCP backlog (A = El Farol next): memory `project_mcp_backlog.md`
- El Farol game: `game-environments/game_envs/games/el_farol.py`
- Demo agent (existing): `demo-game/agents/el_farol_agent.py`

## 11. Review revisions log

This spec was revised on 2026-04-15 in response to architectural review
of v1. Material changes from v1:

- Dropped duplicate `ElFarolConfig` pydantic model in dashboard layer
  (collision with game-layer `ElFarolConfig(GameConfig)` flagged in
  review #1). The game-layer dataclass is now the single source of
  truth; ratio→absolute conversion lives in one place inside
  `_game_for`.
- Fixed score aggregation method ambiguity (review #2): Phase B uses
  per-round payoff sum exclusively. `get_payoffs()` and DQ remain in
  the game class but are not invoked by the service. Documented in new
  §3.3.
- Promoted action_history normalization to a separate prerequisite
  PR (review #3). Was misclassified as "cosmetic" in v1; identified
  call-sites at service.py:398, :569, :1049 plus events.py and
  mcp/tools.py adjacents.
- Added explicit `validate_action` (strict client-submission) vs
  `sanitize` (permissive replay) boundary table (review #4).
- Removed all SDK v3 / `atp-platform-sdk 3.0.0` framing (review #6):
  verified the SDK has zero tournament code today; the only "wire"
  break is the MCP `make_move` action schema, with internal-only
  consumers (`demo-game` agents). Replaced §5 entirely.
- Replaced `ElFarolConfig.with_num_players` classmethod with direct
  constructor in the single call-site (review #7).
- Lowered N upper bound from 50 to 20 with explicit bound rationale
  and `TODO(phase-c)` (review #8).
- Added Pre-E2E load smoke (N=10, R=10, no LLM) to §7.3
  (review #6 in open questions).
- Added MAX_SLOTS_PER_DAY callout in §4 MCP tool description
  (review #9).
- Added MCP tool description update for action shape (review #10).
- Added explicit 422-on-missing-discriminator unit test (review #11).

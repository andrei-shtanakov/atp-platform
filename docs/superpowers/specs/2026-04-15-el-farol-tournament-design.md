# El Farol Tournament Support (Phase B) — Design Spec

**Date:** 2026-04-15
**Status:** approved, ready for implementation plan
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

## 3. Scope and architecture

Changes are localized to three layers:

### 3.1 Game layer (`game-environments/game_envs/games/el_farol.py`)

Two new methods on `ElFarol`, no changes to `step` / `ActionSpace` / payoffs:

- `format_state_for_player(round_number, total_rounds, participant_idx,
  action_history, cumulative_scores) -> dict` — N-player-aware state
  formatter analogous to the PD one. Returns `your_history`,
  `attendance_by_round` (aggregate, not per-player histories),
  `capacity_threshold`, `your_cumulative_score`, `all_scores`,
  `your_participant_idx`, `num_slots`, `action_schema`.
- `validate_action(raw: dict) -> dict` — raises `ValidationError` on
  malformed input (non-list, out-of-range slot, duplicate, >8 slots);
  returns canonical form (sorted slots).
- `default_action_on_timeout() -> dict` — returns `{"slots": []}` (stay
  home).

Mirror methods added to `PrisonersDilemma` (`validate_action` returns
`{"choice": "..."}`, `default_action_on_timeout` returns
`{"choice": "defect"}`) so the tournament service treats both games
uniformly.

Key design decision — for El Farol state format, players see
**aggregate attendance per round** (`list[list[int]]` of length
`num_slots`), not individual histories of all N opponents. Reasons:

- Faithful to Arthur's El Farol formulation: players act on attendance
  signal, not individual decisions.
- At N=20, R=100, per-player histories would be 20×100 payload per
  `get_current_state` call.
- Own history is always available separately.

### 3.2 Tournament service (`packages/atp-dashboard/atp/dashboard/tournament/`)

Seven point changes, no rewrite:

1. `_SUPPORTED_GAMES = {"prisoners_dilemma", "el_farol"}`.
2. Replace the `_GAME_INSTANCES: dict` module global with
   `_PD_SINGLETON` + a `_game_for(tournament)` factory that
   per-tournament instantiates `ElFarol(ElFarolConfig.with_num_players(n))`.
3. `create_tournament` validation:
   - PD: `num_players == 2` (unchanged behavior).
   - El Farol: `2 <= num_players <= 50`.
   - Unknown game_type → `ValidationError`.
4. `submit_action` validation: replace PD-specific
   `action["choice"] in options` with delegated
   `game.validate_action(action)`; store canonical form in
   `Action.action_data`.
5. Round deadline handler: replace hardcoded defect fallback with
   `game.default_action_on_timeout()`.
6. `get_current_state`: parse the formatter's dict into the
   discriminated union via
   `TypeAdapter(RoundState).validate_python(formatted)`.
7. Normalize the `action_history` structure fed into
   `format_state_for_player` to a universal
   `[{"round": i, "actions": {pid: action_data}}]` shape. PD requires a
   cosmetic adjustment; El Farol is authored against this shape.

No database migration. `Action.action_data` is already JSON;
`Tournament.game_type` is already `String(100)`.

### 3.3 Contract layer (pydantic discriminated unions)

Lives in `packages/atp-dashboard/atp/dashboard/tournament/schemas.py`
and is re-exported from `packages/atp-sdk/src/atp_sdk/tournament/`.

```python
class PDAction(BaseModel):
    game_type: Literal["prisoners_dilemma"]
    choice: Literal["cooperate", "defect"]

class ElFarolAction(BaseModel):
    game_type: Literal["el_farol"]
    slots: list[int] = Field(..., max_length=MAX_SLOTS_PER_DAY)
    # field_validator: uniqueness + non-negative; per-tournament
    # num_slots bound enforced at service layer

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

Internal (not in public API until Phase C):

```python
class ElFarolConfig(BaseModel):
    game_type: Literal["el_farol"]
    num_slots: int = 16
    threshold: float = 0.6
    min_total_hours: int = 0
    num_players: int  # injected per tournament, must satisfy 2 <= n <= 50

    @classmethod
    def with_num_players(cls, n: int) -> "ElFarolConfig":
        return cls(game_type="el_farol", num_players=n)
```

## 4. MCP tools

Tool names unchanged: `join_tournament`, `make_move`,
`get_current_state`, `list_tournaments`, `get_tournament`,
`get_history`, `leave_tournament`.

- `make_move(tournament_id, action: dict)` — parses the dict via
  `TypeAdapter(TournamentAction)`. Missing `game_type` →
  422 with explicit "missing discriminator game_type: either
  'prisoners_dilemma' or 'el_farol'".
- `get_current_state(tournament_id)` — returns `RoundState.model_dump()`;
  SDK clients parse back via the union.
- `list_tournaments(game_type: str | None = None)` — new optional filter.

## 5. SDK v3 (`atp-platform-sdk` 3.0.0)

Breaking changes, justified by the contract-level discriminator:

- New modules: `atp_sdk.tournament.actions`,
  `atp_sdk.tournament.state`, `atp_sdk.tournament.config`.
- `AsyncATPClient.make_move(action: TournamentAction | dict)` —
  accepts either; `.model_dump()` applied if typed.
- `AsyncATPClient.get_current_state(tournament_id) -> RoundState` —
  parsed via discriminator.
- `AsyncATPClient.list_tournaments(game_type: str | None = None)` —
  new optional param.
- `CHANGELOG.md` migration section (PD before/after, new El Farol
  types, pattern-matching on `RoundState`).

Backwards compatibility: the backend still accepts raw dicts with a
`game_type` field, so v2 PD bots that explicitly add
`"game_type": "prisoners_dilemma"` keep working. PyPI: publish 3.0.0,
leave 2.x available.

## 6. Error handling and edge cases

Enumerated during design, to be covered by tests:

- All N players timeout in one round → each gets `{"slots": []}`,
  `step()` yields 0 points, tournament proceeds normally.
- Out-of-range / duplicate slots → `validate_action` rejects before
  DB write; 400 to client.
- Concurrent last-action submits → existing `existing action` guard
  + transactional boundary resolves the round exactly once (current
  N-player code path, unchanged).
- `num_players` outside `[2, 50]` for El Farol → rejected at create.
- Unknown `game_type` → rejected at create and at all read/write
  call-sites (defense in depth).
- Empty `action_history` at round 1 → `your_history=[]`,
  `attendance_by_round=[]`, all scores zero.

## 7. Testing strategy

### 7.1 Unit tests

- `tests/unit/games/test_el_farol_state_format.py` — `format_state_for_player`
  shape, `validate_action` edge cases, `default_action_on_timeout`, canonical
  sorting, aggregate attendance computation.
- `tests/unit/tournament/test_schemas_discriminator.py` — round-trip
  `model_dump` ↔ `validate_python` for both games, missing discriminator
  error, wrong-shape combinations (PD game_type with slots, El Farol game_type
  with choice).
- Extend `tests/unit/dashboard/tournament/test_service.py` —
  `create_tournament(el_farol, N in [1, 2, 50, 51])`, `_game_for` returns
  game with configured `num_players`.

### 7.2 Integration tests

- `tests/integration/tournament/test_el_farol_flow.py` — full N=5, R=3
  flow through the service: join → active → all submit → round resolves →
  state per participant → deadline timeout → tournament completes →
  leaderboard correct.
- Parallel PD regression test over the same generic code paths to confirm
  refactor did not break PD.

### 7.3 End-to-end with LLM bots

Mirrors the PD validation that produced the first successful 30-round run.

- Adapt `demo-game/agents/el_farol_agent.py` and
  `demo-game/suites/el_farol_llm_vs_builtin.yaml` to SDK v3.
- `demo-game/compose.el-farol.yml` with N=5 (3 LLM bots + 2 built-in
  strategies: greedy, random), 20 rounds, round_deadline=30s.
- Pass criteria: tournament completes without service timeouts, all 5
  bots finish, MCP event-bus logs clean, LLM scores > random.

### 7.4 Coverage

Existing tournament-service floor is 83% (per commit `8dfe0cc`).
El-Farol-specific new code (service + game) should itself be ≥85% covered
to avoid pushing the floor down further.

## 8. Phase C hooks (explicitly deferred)

These are called out so the abstraction story in Phase C has a clear
starting point:

- `GameAdapter` protocol with `format_state_for_player`, `validate_action`,
  `default_action_on_timeout`, `canonical_action`, `parse_action`.
- Game-config in create_tournament API (`game_config: dict`
  validated into `PDConfig | ElFarolConfig | ...`).
- MCP registry-based dynamic tool schemas.
- Per-game SDK helpers / game-specific `TournamentClient` subclasses.

## 9. Release plan

- Preferred: single PR named `feat(tournament): el farol as second game_type`.
  May be split into (1) schemas + game layer + service + unit/integration
  tests and (2) SDK v3 + demo-game E2E if the diff grows past ~800 LOC.
- Merges to `main` with `[deploy]` trigger for the Namecheap VPS.
- PyPI publish of `atp-platform-sdk 3.0.0` is a separate follow-up
  action after the PR merges and E2E LLM-bot run passes.
- Tournament smoke run on production (N=5, R=20) before declaring B
  complete.

## 10. References

- Current PD tournament design: `docs/superpowers/specs/2026-04-10-mcp-tournament-server-design.md`
- Plan 2a (PD-only shipped): `docs/superpowers/specs/2026-04-11-mcp-tournament-plan-2a-design.md`
- MCP backlog (A = El Farol next): memory `project_mcp_backlog.md`
- El Farol game: `game-environments/game_envs/games/el_farol.py`
- Demo agent: `demo-game/agents/el_farol_agent.py`

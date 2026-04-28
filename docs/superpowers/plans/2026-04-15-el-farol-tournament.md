# El Farol Tournament Support Implementation Plan

> **Historical note (2026-04-28):** This plan was written when the
> El Farol move wire format was a flat slot list (`{"slots": [...]}`).
> That format has since been removed. The current canonical shape is
> `{"intervals": [[start, end], ...]}` only — see
> `docs/games/rules/el-farol-bar.en.md`. All `{"slots": [...]}` examples
> below should be read as historical context, not as current contract.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship El Farol Bar Problem as the second `game_type` in the MCP tournament service, on par with Prisoner's Dilemma, via two sequential PRs.

**Architecture:** PR-1 is a pure refactor that teaches the tournament service to talk to games through a uniform method surface (`validate_action`, `default_action_on_timeout`, generic `format_state_for_player` input shape) — PD-only tests stay green, no behavior change. PR-2 adds El Farol's three new game methods, pydantic discriminated-union schemas for action and state with server-side `game_type` injection, the `_el_farol_for(N)` cached factory, tournament creation validation, structured observability logs, an integration test, a load smoke, and a demo-game E2E harness.

**Tech Stack:** Python 3.12, pydantic v2 discriminated unions + `TypeAdapter`, SQLAlchemy 2.x async, FastAPI, FastMCP, pytest + pytest-anyio, Docker Compose (for demo-game harness).

**Source spec:** `docs/superpowers/specs/2026-04-15-el-farol-tournament-design.md` (v4, commit `124f930`).

**Baseline:** `main` at the start of session (`be35899` or later). Before Task 1: `uv sync --all-packages --group dev` to ensure deps are installed in `.venv`.

**Worktree:** Recommended to create a worktree (`superpowers:using-git-worktrees`) to isolate from any in-flight work on `main`.

---

## PR-1: Pure refactor — uniform game method surface

Goal of PR-1: tournament service never reads or writes `"choice"` by string. PD still works identically, no external behavior change.

### Task 1: Add `PrisonersDilemma.validate_action`

**Files:**
- Modify: `game-environments/game_envs/games/prisoners_dilemma.py`
- Test: `game-environments/tests/test_prisoners_dilemma.py` (existing; new cases appended)

- [ ] **Step 1: Write the failing tests**

Append to `game-environments/tests/test_prisoners_dilemma.py`:

```python
import pytest
from game_envs.core.errors import ValidationError  # may live elsewhere; see Step 2
from game_envs.games.prisoners_dilemma import PrisonersDilemma


def test_validate_action_accepts_cooperate():
    pd = PrisonersDilemma()
    assert pd.validate_action({"choice": "cooperate"}) == {"choice": "cooperate"}


def test_validate_action_accepts_defect():
    pd = PrisonersDilemma()
    assert pd.validate_action({"choice": "defect"}) == {"choice": "defect"}


def test_validate_action_rejects_unknown_choice():
    pd = PrisonersDilemma()
    with pytest.raises(ValidationError, match="choice"):
        pd.validate_action({"choice": "betray"})


def test_validate_action_rejects_missing_choice():
    pd = PrisonersDilemma()
    with pytest.raises(ValidationError):
        pd.validate_action({})


def test_validate_action_rejects_non_dict():
    pd = PrisonersDilemma()
    with pytest.raises(ValidationError):
        pd.validate_action("cooperate")  # type: ignore[arg-type]
```

- [ ] **Step 2: Verify `ValidationError` import path**

Check where `ValidationError` lives for the `game_envs` package:

```bash
grep -rn "class ValidationError" game-environments/game_envs/ | head
```

If it does not exist, create `game-environments/game_envs/core/errors.py` with:

```python
"""Game-layer exceptions."""


class ValidationError(ValueError):
    """Raised when a game rejects a malformed action."""
```

Re-export from `game-environments/game_envs/__init__.py` if the package has an `__all__`.

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd game-environments && uv run pytest tests/test_prisoners_dilemma.py -k validate_action -v
```

Expected: FAIL with `AttributeError: 'PrisonersDilemma' object has no attribute 'validate_action'`.

- [ ] **Step 4: Implement `validate_action`**

In `game-environments/game_envs/games/prisoners_dilemma.py`, add this method to the `PrisonersDilemma` class (place it below `format_state_for_player`, above `_compute_payoffs`):

```python
def validate_action(self, raw: Any) -> dict[str, str]:
    """Validate a client-submitted action and return canonical form.

    Strict path used by tournament submit. Raises ValidationError on
    any malformed input.
    """
    from game_envs.core.errors import ValidationError  # local import to avoid cycles

    if not isinstance(raw, dict):
        raise ValidationError(f"action must be a dict, got {type(raw).__name__}")
    choice = raw.get("choice")
    if choice not in (COOPERATE, DEFECT):
        raise ValidationError(
            f"choice must be {COOPERATE!r} or {DEFECT!r}, got {choice!r}"
        )
    return {"choice": choice}
```

- [ ] **Step 5: Run tests, verify they pass**

```bash
cd game-environments && uv run pytest tests/test_prisoners_dilemma.py -k validate_action -v
```

Expected: PASS (5/5).

- [ ] **Step 6: Commit**

```bash
git add game-environments/game_envs/games/prisoners_dilemma.py \
        game-environments/game_envs/core/errors.py \
        game-environments/tests/test_prisoners_dilemma.py
git commit -m "refactor(pd): add validate_action strict entrypoint"
```

---

### Task 2: Add `PrisonersDilemma.default_action_on_timeout`

**Files:**
- Modify: `game-environments/game_envs/games/prisoners_dilemma.py`
- Test: `game-environments/tests/test_prisoners_dilemma.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_default_action_on_timeout_returns_defect():
    pd = PrisonersDilemma()
    assert pd.default_action_on_timeout() == {"choice": "defect"}
```

- [ ] **Step 2: Run, verify fails**

```bash
cd game-environments && uv run pytest tests/test_prisoners_dilemma.py -k default_action_on_timeout -v
```

Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement**

In `PrisonersDilemma` class:

```python
def default_action_on_timeout(self) -> dict[str, str]:
    """Action substituted when a participant misses the round deadline."""
    return {"choice": DEFECT}
```

- [ ] **Step 4: Run, verify passes**

```bash
cd game-environments && uv run pytest tests/test_prisoners_dilemma.py -k default_action_on_timeout -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add game-environments/game_envs/games/prisoners_dilemma.py \
        game-environments/tests/test_prisoners_dilemma.py
git commit -m "refactor(pd): add default_action_on_timeout"
```

---

### Task 3: Normalize `PrisonersDilemma.format_state_for_player` input shape

Current shape: `action_history: list[list[str]]` (each row is `[p0_action, p1_action]`).
New shape: `action_history: list[dict]` where each dict is `{"round": int, "actions": {participant_idx: action_data_dict}}`. This is the universal shape the service can produce once for any game.

**Files:**
- Modify: `game-environments/game_envs/games/prisoners_dilemma.py`
- Test: `game-environments/tests/test_prisoners_dilemma.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_format_state_accepts_generic_action_history_shape():
    pd = PrisonersDilemma()
    history = [
        {"round": 1, "actions": {0: {"choice": "cooperate"}, 1: {"choice": "defect"}}},
        {"round": 2, "actions": {0: {"choice": "defect"}, 1: {"choice": "defect"}}},
    ]
    state = pd.format_state_for_player(
        round_number=3,
        total_rounds=5,
        participant_idx=0,
        action_history=history,
        cumulative_scores=[1.0, 6.0],
    )
    assert state["your_history"] == ["cooperate", "defect"]
    assert state["opponent_history"] == ["defect", "defect"]
    assert state["your_cumulative_score"] == 1.0
    assert state["opponent_cumulative_score"] == 6.0
    assert state["game_type"] == "prisoners_dilemma"
    assert "your_turn" in state  # preserved for back-compat (spec §3.4)


def test_format_state_empty_history():
    pd = PrisonersDilemma()
    state = pd.format_state_for_player(
        round_number=1,
        total_rounds=5,
        participant_idx=0,
        action_history=[],
        cumulative_scores=[0.0, 0.0],
    )
    assert state["your_history"] == []
    assert state["opponent_history"] == []
```

- [ ] **Step 2: Run, verify failure**

```bash
cd game-environments && uv run pytest tests/test_prisoners_dilemma.py -k format_state -v
```

Expected: the new tests fail (TypeError / KeyError on the dict shape), existing tests may also fail if they rely on old shape — note which ones.

- [ ] **Step 3: Update `format_state_for_player` body**

Replace the method body (lines 225-267) with:

```python
def format_state_for_player(
    self,
    round_number: int,
    total_rounds: int,
    participant_idx: int,
    action_history: list[dict[str, Any]],
    cumulative_scores: list[float],
) -> dict[str, Any]:
    """Build a player-private RoundState dict for the given player.

    Args:
        round_number: 1-indexed round about to be played.
        total_rounds: Total rounds in the tournament.
        participant_idx: Which participant we are formatting for (0 or 1).
        action_history: List per resolved round of
            ``{"round": i, "actions": {pid: action_data}}``. Empty list =
            no rounds played yet.
        cumulative_scores: Per-participant cumulative scores so far.

    Returns:
        Dict matching the PDRoundState shape (see schemas.py). The
        ``tournament_id`` field is -1 and must be filled by the caller.
    """
    opponent_idx = 1 - participant_idx
    your_history = [
        row["actions"][participant_idx]["choice"] for row in action_history
    ]
    opponent_history = [
        row["actions"][opponent_idx]["choice"] for row in action_history
    ]
    return {
        "tournament_id": -1,
        "round_number": round_number,
        "game_type": "prisoners_dilemma",
        "your_history": your_history,
        "opponent_history": opponent_history,
        "your_cumulative_score": cumulative_scores[participant_idx],
        "opponent_cumulative_score": cumulative_scores[opponent_idx],
        "action_schema": {
            "type": "choice",
            "options": [COOPERATE, DEFECT],
        },
        "your_turn": True,  # service overwrites this based on DB submission state
        "total_rounds": total_rounds,
        "extra": {},
    }
```

- [ ] **Step 4: Find and update existing PD tests that use old shape**

```bash
grep -rn "format_state_for_player" game-environments/tests/ \
    packages/atp-dashboard/atp/ tests/ | grep -v __pycache__
```

For each call-site found, convert `[["cooperate", "defect"], ...]` → `[{"round": 1, "actions": {0: {"choice": "cooperate"}, 1: {"choice": "defect"}}}, ...]`. This is mechanical; do it in a single commit with the signature change.

- [ ] **Step 5: Run all PD unit tests**

```bash
cd game-environments && uv run pytest tests/test_prisoners_dilemma.py -v
```

Expected: PASS (all, including old and new cases).

- [ ] **Step 6: Commit**

```bash
git add game-environments/game_envs/games/prisoners_dilemma.py \
        game-environments/tests/test_prisoners_dilemma.py
git commit -m "refactor(pd): format_state_for_player takes generic action_history shape"
```

---

### Task 4: Refactor `service.py:_resolve_round` to read action_data opaquely

Current at `packages/atp-dashboard/atp/dashboard/tournament/service.py:569`:
```python
action_vec[i] = a.action_data["choice"]
```
PD-specific. Refactor to pass action_data dicts into a game-agnostic resolver.

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py` (method `_resolve_round`, lines ~530–635)
- Test: `tests/unit/dashboard/tournament/test_service_resolve.py` (existing)

- [ ] **Step 1: Read current `_resolve_round` to understand full signature**

```bash
sed -n '530,635p' packages/atp-dashboard/atp/dashboard/tournament/service.py
```

- [ ] **Step 2: Extract PD payoff logic into a PD method `compute_round_payoffs`**

Add to `PrisonersDilemma` class (below `_compute_payoffs`):

```python
def compute_round_payoffs(
    self, actions: dict[int, dict[str, Any]]
) -> list[float]:
    """Generic entry point used by the tournament service.

    Args:
        actions: Mapping participant_idx -> action_data dict.

    Returns:
        List of payoffs in participant_idx order.
    """
    a0 = actions[0]["choice"]
    a1 = actions[1]["choice"]
    payoffs = self._compute_payoffs(a0, a1)
    return [payoffs["player_0"], payoffs["player_1"]]
```

Add a test in `game-environments/tests/test_prisoners_dilemma.py`:

```python
def test_compute_round_payoffs_from_action_dicts():
    pd = PrisonersDilemma()
    payoffs = pd.compute_round_payoffs(
        {0: {"choice": "cooperate"}, 1: {"choice": "defect"}}
    )
    assert payoffs == [0.0, 5.0]  # sucker, temptation
```

Run it:

```bash
cd game-environments && uv run pytest tests/test_prisoners_dilemma.py -k compute_round_payoffs -v
```

Expected: PASS.

- [ ] **Step 3: Refactor `_resolve_round` to use new method**

In `service.py`, locate the block (around line 565-585) that builds `action_vec` from `a.action_data["choice"]` and the inline PD payoff matrix. Replace with:

```python
# Build participant_idx -> action_data map
actions_by_idx: dict[int, dict[str, Any]] = {}
for a in round_actions:
    p_idx = participant_idx_by_id[a.participant_id]
    actions_by_idx[p_idx] = a.action_data

game = _game_for(tournament)
payoffs = game.compute_round_payoffs(actions_by_idx)

for a in round_actions:
    p_idx = participant_idx_by_id[a.participant_id]
    a.payoff = payoffs[p_idx]
```

Keep the event-emission / session flush logic downstream unchanged.

- [ ] **Step 4: Run existing resolve tests**

```bash
uv run pytest tests/unit/dashboard/tournament/test_service_resolve.py -v
```

Expected: PASS (no behavior change for PD).

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        game-environments/game_envs/games/prisoners_dilemma.py \
        game-environments/tests/test_prisoners_dilemma.py
git commit -m "refactor(tournament): _resolve_round delegates to game.compute_round_payoffs"
```

---

### Task 5: Refactor `service.py:get_state_for` to build the generic action_history shape

Current at `service.py:~380–400` builds `row[p_idx] = action.action_data.get("choice", "")`. Replace with the generic shape PD now accepts (from Task 3).

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py` (method `get_state_for`, lines ~339–428)
- Test: `tests/unit/dashboard/tournament/test_service_state.py`

- [ ] **Step 1: Read current `get_state_for`**

```bash
sed -n '339,428p' packages/atp-dashboard/atp/dashboard/tournament/service.py
```

- [ ] **Step 2: Update action_history construction**

In `get_state_for`, replace the block that builds a `list[list[str]]` with:

```python
action_history: list[dict[str, Any]] = []
for r in resolved_rounds:
    actions_by_idx: dict[int, dict[str, Any]] = {}
    for a in r.actions:  # assumes relationship is loaded; keep the existing eager-load
        p_idx = participant_idx_by_id[a.participant_id]
        actions_by_idx[p_idx] = a.action_data
    action_history.append({"round": r.round_number, "actions": actions_by_idx})
```

- [ ] **Step 3: Pass action_history through to the game**

The `game.format_state_for_player(...)` call already takes `action_history` as a named param — signature compatible with the new shape after Task 3.

- [ ] **Step 4: Run existing state tests**

```bash
uv run pytest tests/unit/dashboard/tournament/test_service_state.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py
git commit -m "refactor(tournament): get_state_for builds generic action_history"
```

---

### Task 6: Refactor `service.py:submit_action` to delegate validation to the game

Current at `service.py:480–492` does PD-specific `action["choice"] in schema_probe["action_schema"]["options"]`. Replace with `game.validate_action(action)`.

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py` (method `submit_action`)
- Test: existing tests under `tests/unit/dashboard/tournament/` that exercise `submit_action`

- [ ] **Step 1: Replace validation block**

Locate the block at ~lines 480-510 of `service.py`:

```python
game = _GAME_INSTANCES[tournament.game_type]
schema_probe = game.format_state_for_player(...)
if action.get("choice") not in schema_probe["action_schema"]["options"]:
    raise ValidationError(...)
...
new_action = Action(..., action_data={"choice": action["choice"]})
```

Replace with:

```python
game = _GAME_INSTANCES[tournament.game_type]  # stays a dict in PR-1, swapped in PR-2 Task 16
canonical = game.validate_action(action)
...
new_action = Action(..., action_data=canonical)
```

(Do NOT change the `_GAME_INSTANCES` dict yet — PR-1 keeps the existing dict-of-singletons pattern; PR-2 swaps in `_game_for`.)

- [ ] **Step 2: Run existing submit tests**

```bash
uv run pytest tests/unit/dashboard/tournament/ -k submit -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py
git commit -m "refactor(tournament): submit_action delegates to game.validate_action"
```

---

### Task 7: Refactor deadline handler to use `game.default_action_on_timeout`

Current at `service.py:~1049` hardcodes `action_data={"choice": "defect"}` when a participant misses the deadline.

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py` (method `force_resolve_round` or deadline handler)

- [ ] **Step 1: Locate and replace**

```bash
grep -n 'action_data={"choice"' packages/atp-dashboard/atp/dashboard/tournament/service.py
```

At each hit, replace `action_data={"choice": "defect"}` with:

```python
action_data=game.default_action_on_timeout()
```

(Ensure `game = _GAME_INSTANCES[tournament.game_type]` is in scope at each call-site — it usually is, because the deadline handler already fetches the tournament.)

- [ ] **Step 2: Run deadline-handling tests**

```bash
uv run pytest tests/unit/dashboard/tournament/ -k "deadline or resolve or timeout" -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py
git commit -m "refactor(tournament): deadline handler uses game.default_action_on_timeout"
```

---

### Task 8: Audit events.py and mcp/tools.py for PD-specific reads

**Files:**
- Read-only audit, then targeted edits if needed

- [ ] **Step 1: Grep for `"choice"` in tournament-facing modules**

```bash
grep -rn '\["choice"\]\|\.get("choice"' \
    packages/atp-dashboard/atp/dashboard/tournament/ \
    packages/atp-dashboard/atp/dashboard/mcp/ \
    | grep -v __pycache__
```

- [ ] **Step 2: For each hit, decide**

- If it's inside event payload construction or MCP tool response serialization and has a clear PD-agnostic alternative (e.g. `action_data=a.action_data` instead of `action=a.action_data["choice"]`), rewrite it.
- If it's only used to label log lines, leave it but rename the key (e.g. `"action_choice": a.action_data.get("choice")` → `"action_data": a.action_data`).
- If PR-2 can clean it up more naturally (the consumer understands both shapes), defer and record a TODO in the commit message.

- [ ] **Step 3: Run full PD test suite**

```bash
uv run pytest tests/ -k "tournament or mcp" -v -x
```

Expected: PASS.

- [ ] **Step 4: Commit (if any changes)**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/events.py \
        packages/atp-dashboard/atp/dashboard/mcp/tools.py
git commit -m "refactor(tournament): remove PD-specific reads from events/mcp"
```

If no changes needed, skip this commit.

---

### Task 9: Full PD regression run and PR-1 checkpoint

- [ ] **Step 1: Full test suite**

```bash
uv run pytest tests/ -v --cov=atp --cov-report=term-missing -m "not slow"
```

Expected: all PD tests PASS. Coverage on tournament service stays ≥83%.

- [ ] **Step 2: Lint + format**

```bash
uv run ruff format .
uv run ruff check .
uv run pyrefly check
```

Expected: clean.

- [ ] **Step 3: Diff size check**

```bash
git diff main --stat
```

Target: ≤300 LOC total (spec §9).

- [ ] **Step 4: Open PR-1**

```bash
gh pr create --title "refactor(tournament): uniform game method surface (PR-1 for el farol)" \
    --body "$(cat <<'EOF'
## Summary
- Pure refactor, PD behavior unchanged.
- Three new methods on `PrisonersDilemma`: `validate_action`, `default_action_on_timeout`, `compute_round_payoffs`.
- `format_state_for_player` now accepts the generic action_history shape (`[{"round": i, "actions": {pid: action_data}}]`).
- Tournament service no longer reads `"choice"` by string anywhere.

## Test plan
- [x] PD unit tests (`game-environments/tests/test_prisoners_dilemma.py`)
- [x] Tournament service unit tests under `tests/unit/dashboard/tournament/`
- [x] Full test suite `-m "not slow"`

Part 1 of the El Farol tournament series. Part 2 (feature) follows once this merges.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Merge PR-1 before starting Task 10.**

---

## PR-2: El Farol feature

### Task 10: Add `ElFarol.validate_action`

**Files:**
- Modify: `game-environments/game_envs/games/el_farol.py`
- Test: `tests/unit/games/test_el_farol_state_format.py` (create if missing)

- [ ] **Step 1: Create test file structure**

```bash
mkdir -p tests/unit/games
touch tests/unit/games/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/unit/games/test_el_farol_state_format.py`:

```python
"""Tests for El Farol game-layer service-facing methods."""

import pytest
from game_envs.core.errors import ValidationError
from game_envs.games.el_farol import MAX_SLOTS_PER_DAY, ElFarol, ElFarolConfig


def _game(n: int = 5, num_slots: int = 16) -> ElFarol:
    return ElFarol(ElFarolConfig(
        num_players=n, num_slots=num_slots, capacity_threshold=3,
    ))


def test_validate_action_accepts_sorted_unique_slots():
    g = _game(num_slots=16)
    assert g.validate_action({"slots": [3, 7, 0]}) == {"slots": [0, 3, 7]}


def test_validate_action_accepts_empty_list():
    g = _game()
    assert g.validate_action({"slots": []}) == {"slots": []}


def test_validate_action_rejects_duplicates():
    g = _game()
    with pytest.raises(ValidationError, match="unique"):
        g.validate_action({"slots": [1, 2, 2]})


def test_validate_action_rejects_out_of_range():
    g = _game(num_slots=16)
    with pytest.raises(ValidationError, match="out of range"):
        g.validate_action({"slots": [16]})


def test_validate_action_rejects_negative():
    g = _game(num_slots=16)
    with pytest.raises(ValidationError, match="out of range"):
        g.validate_action({"slots": [-1]})


def test_validate_action_rejects_too_many():
    g = _game(num_slots=16)
    too_many = list(range(MAX_SLOTS_PER_DAY + 1))
    with pytest.raises(ValidationError, match="at most"):
        g.validate_action({"slots": too_many})


def test_validate_action_rejects_non_list():
    g = _game()
    with pytest.raises(ValidationError):
        g.validate_action({"slots": "not-a-list"})


def test_validate_action_rejects_missing_slots():
    g = _game()
    with pytest.raises(ValidationError):
        g.validate_action({})


def test_validate_action_rejects_non_dict():
    g = _game()
    with pytest.raises(ValidationError):
        g.validate_action([0, 1])  # type: ignore[arg-type]


def test_sanitize_is_permissive_where_validate_is_strict():
    """Boundary check from spec §3.1."""
    g = _game(num_slots=16)
    # sanitize cleans dupes and out-of-range silently
    cleaned = g.action_space.sanitize([1, 2, 2, 99, -1])
    assert set(cleaned).issubset(range(16))
    # validate rejects the same input
    with pytest.raises(ValidationError):
        g.validate_action({"slots": [1, 2, 2, 99, -1]})
```

- [ ] **Step 3: Run, verify failure**

```bash
uv run pytest tests/unit/games/test_el_farol_state_format.py -k validate_action -v
```

Expected: FAIL with `AttributeError: 'ElFarol' object has no attribute 'validate_action'`.

- [ ] **Step 4: Implement**

In `game-environments/game_envs/games/el_farol.py`, add to the `ElFarol` class (near `format_state_for_player`):

```python
def validate_action(self, raw: Any) -> dict[str, list[int]]:
    """Validate a client-submitted action and return canonical form.

    Strict path. Used by tournament submit. See also
    ``ElFarolActionSpace.sanitize`` for the permissive replay path.
    """
    from game_envs.core.errors import ValidationError

    if not isinstance(raw, dict):
        raise ValidationError(
            f"action must be a dict, got {type(raw).__name__}"
        )
    slots = raw.get("slots")
    if slots is None:
        raise ValidationError("action must have field 'slots'")
    if not isinstance(slots, list):
        raise ValidationError(
            f"slots must be a list of int, got {type(slots).__name__}"
        )
    if len(slots) > MAX_SLOTS_PER_DAY:
        raise ValidationError(
            f"at most {MAX_SLOTS_PER_DAY} slots per day, got {len(slots)}"
        )
    if len(set(slots)) != len(slots):
        raise ValidationError("slots must be unique")
    num_slots = self.config.num_slots
    for s in slots:
        if not isinstance(s, int) or isinstance(s, bool):
            raise ValidationError(f"slot {s!r} is not an int")
        if not (0 <= s < num_slots):
            raise ValidationError(
                f"slot {s} out of range [0, {num_slots})"
            )
    return {"slots": sorted(slots)}
```

- [ ] **Step 5: Run, verify PASS**

```bash
uv run pytest tests/unit/games/test_el_farol_state_format.py -k "validate_action or sanitize" -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add game-environments/game_envs/games/el_farol.py \
        tests/unit/games/test_el_farol_state_format.py
git commit -m "feat(el-farol): add validate_action strict entrypoint"
```

---

### Task 11: Add `ElFarol.default_action_on_timeout`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/games/test_el_farol_state_format.py`:

```python
def test_default_action_on_timeout_returns_empty_slots():
    g = _game()
    assert g.default_action_on_timeout() == {"slots": []}
```

- [ ] **Step 2: Verify fails**

```bash
uv run pytest tests/unit/games/test_el_farol_state_format.py -k default_action_on_timeout -v
```

- [ ] **Step 3: Implement** in `ElFarol`:

```python
def default_action_on_timeout(self) -> dict[str, list[int]]:
    """Stay home — attend zero slots (spec §3.1)."""
    return {"slots": []}
```

- [ ] **Step 4: Verify passes**

```bash
uv run pytest tests/unit/games/test_el_farol_state_format.py -k default_action_on_timeout -v
```

- [ ] **Step 5: Commit**

```bash
git add game-environments/game_envs/games/el_farol.py \
        tests/unit/games/test_el_farol_state_format.py
git commit -m "feat(el-farol): add default_action_on_timeout"
```

---

### Task 12: Add `ElFarol.format_state_for_player` and `compute_round_payoffs`

**Files:**
- Modify: `game-environments/game_envs/games/el_farol.py`
- Test: `tests/unit/games/test_el_farol_state_format.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_format_state_empty_history():
    g = _game(n=5, num_slots=16)
    state = g.format_state_for_player(
        round_number=1,
        total_rounds=10,
        participant_idx=0,
        action_history=[],
        cumulative_scores=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert state["game_type"] == "el_farol"
    assert state["your_history"] == []
    assert state["attendance_by_round"] == []
    assert state["your_cumulative_score"] == 0.0
    assert state["all_scores"] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert state["your_participant_idx"] == 0
    assert state["num_slots"] == 16
    assert state["capacity_threshold"] == 3  # from _game() fixture
    assert "action_schema" in state
    # pending_submission is injected by the service, not the game:
    assert "pending_submission" not in state


def test_format_state_populated_history_aggregates_attendance():
    g = _game(n=3, num_slots=4)
    history = [
        {
            "round": 1,
            "actions": {
                0: {"slots": [0, 1]},
                1: {"slots": [1, 2]},
                2: {"slots": [0]},
            },
        },
        {
            "round": 2,
            "actions": {
                0: {"slots": [3]},
                1: {"slots": [3]},
                2: {"slots": [3]},
            },
        },
    ]
    state = g.format_state_for_player(
        round_number=3,
        total_rounds=5,
        participant_idx=0,
        action_history=history,
        cumulative_scores=[1.0, 0.0, -1.0],
    )
    assert state["your_history"] == [[0, 1], [3]]
    # slot 0: 2 (p0, p2); slot 1: 2 (p0, p1); slot 2: 1 (p1); slot 3: 0 in round 1
    assert state["attendance_by_round"][0] == [2, 2, 1, 0]
    # round 2: all 3 at slot 3
    assert state["attendance_by_round"][1] == [0, 0, 0, 3]


def test_compute_round_payoffs_happy_minus_crowded():
    g = _game(n=3, num_slots=4)
    # capacity_threshold=3, so any slot with 3+ attendees is crowded
    actions = {
        0: {"slots": [0, 1]},      # slot 0: 2 attend -> happy; slot 1: 3 -> crowded
        1: {"slots": [1, 2]},      # slot 1: crowded; slot 2: 1 -> happy
        2: {"slots": [0, 1]},      # slot 0: happy; slot 1: crowded
    }
    payoffs = g.compute_round_payoffs(actions)
    # p0: 1 happy (0) - 1 crowded (1) = 0
    # p1: 1 happy (2) - 1 crowded (1) = 0
    # p2: 1 happy (0) - 1 crowded (1) = 0
    assert payoffs == [0.0, 0.0, 0.0]
```

- [ ] **Step 2: Verify fails**

```bash
uv run pytest tests/unit/games/test_el_farol_state_format.py -v
```

Expected: FAILs on missing methods / missing keys.

- [ ] **Step 3: Implement `format_state_for_player`**

In `ElFarol`, add (adapt from spec §3.1):

```python
def format_state_for_player(
    self,
    round_number: int,
    total_rounds: int,
    participant_idx: int,
    action_history: list[dict[str, Any]],
    cumulative_scores: list[float],
) -> dict[str, Any]:
    """N-player state formatter (see spec §3.1).

    Does NOT include ``pending_submission`` — service-layer concern.
    """
    your_history = [
        row["actions"].get(participant_idx, {}).get("slots", [])
        for row in action_history
    ]
    attendance_by_round: list[list[int]] = []
    for row in action_history:
        counts = [0] * self.config.num_slots
        for _pid, action_data in row["actions"].items():
            for s in action_data.get("slots", []):
                if 0 <= s < self.config.num_slots:
                    counts[s] += 1
        attendance_by_round.append(counts)

    return {
        "tournament_id": -1,
        "game_type": "el_farol",
        "round_number": round_number,
        "total_rounds": total_rounds,
        "your_history": your_history,
        "attendance_by_round": attendance_by_round,
        "capacity_threshold": self.config.capacity_threshold,
        "your_cumulative_score": cumulative_scores[participant_idx],
        "all_scores": list(cumulative_scores),
        "your_participant_idx": participant_idx,
        "num_slots": self.config.num_slots,
        "action_schema": {
            "type": "list[int]",
            "max_length": MAX_SLOTS_PER_DAY,
            "value_range": [0, self.config.num_slots - 1],
            "unique": True,
        },
        "extra": {},
    }
```

- [ ] **Step 4: Implement `compute_round_payoffs`**

Add next to the above method:

```python
def compute_round_payoffs(
    self, actions: dict[int, dict[str, Any]]
) -> list[float]:
    """Per-round payoff = happy slots − crowded slots (spec §3.3).

    Args:
        actions: participant_idx -> {"slots": list[int]}.

    Returns:
        List of per-round payoffs in participant_idx order.
    """
    n = self.config.num_players
    threshold = self.config.capacity_threshold
    # Build attendance counts
    counts = [0] * self.config.num_slots
    for p_idx in range(n):
        for s in actions.get(p_idx, {}).get("slots", []):
            if 0 <= s < self.config.num_slots:
                counts[s] += 1

    crowded = {i for i, c in enumerate(counts) if c >= threshold}

    payoffs: list[float] = [0.0] * n
    for p_idx in range(n):
        happy = 0
        crowded_count = 0
        for s in actions.get(p_idx, {}).get("slots", []):
            if s in crowded:
                crowded_count += 1
            else:
                happy += 1
        payoffs[p_idx] = float(happy - crowded_count)
    return payoffs
```

- [ ] **Step 5: Run all El Farol tests**

```bash
uv run pytest tests/unit/games/test_el_farol_state_format.py -v
```

Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add game-environments/game_envs/games/el_farol.py \
        tests/unit/games/test_el_farol_state_format.py
git commit -m "feat(el-farol): add format_state_for_player and compute_round_payoffs"
```

---

### Task 13: Add action discriminated-union schemas

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/schemas.py`
- Test: `tests/unit/tournament/test_schemas_discriminator.py` (create)

- [ ] **Step 1: Create test file**

```bash
mkdir -p tests/unit/tournament
touch tests/unit/tournament/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/unit/tournament/test_schemas_discriminator.py`:

```python
"""Tests for discriminated-union action/state schemas (spec §3.4)."""

import pytest
from pydantic import TypeAdapter, ValidationError

from atp.dashboard.tournament.schemas import (
    ElFarolAction,
    PDAction,
    TournamentAction,
)


ADAPTER = TypeAdapter(TournamentAction)


def test_parses_pd_action():
    result = ADAPTER.validate_python(
        {"game_type": "prisoners_dilemma", "choice": "cooperate"}
    )
    assert isinstance(result, PDAction)
    assert result.choice == "cooperate"


def test_parses_el_farol_action():
    result = ADAPTER.validate_python(
        {"game_type": "el_farol", "slots": [0, 3]}
    )
    assert isinstance(result, ElFarolAction)
    assert result.slots == [0, 3]


def test_pd_missing_choice_rejected():
    with pytest.raises(ValidationError) as exc:
        ADAPTER.validate_python({"game_type": "prisoners_dilemma"})
    assert "choice" in str(exc.value)


def test_pd_discriminator_with_el_farol_fields_rejected():
    """Case 4: extra='forbid' surfaces BOTH missing and extra fields."""
    with pytest.raises(ValidationError) as exc:
        ADAPTER.validate_python(
            {"game_type": "prisoners_dilemma", "slots": [0]}
        )
    text = str(exc.value)
    assert "choice" in text  # missing
    assert "slots" in text  # extra forbidden


def test_unknown_game_type_rejected():
    with pytest.raises(ValidationError) as exc:
        ADAPTER.validate_python({"game_type": "tic_tac_toe", "move": "X1"})
    assert "prisoners_dilemma" in str(exc.value) or "el_farol" in str(exc.value)


def test_pd_roundtrip():
    a = ADAPTER.validate_python(
        {"game_type": "prisoners_dilemma", "choice": "defect"}
    )
    assert ADAPTER.validate_python(a.model_dump()) == a


def test_el_farol_roundtrip():
    a = ADAPTER.validate_python({"game_type": "el_farol", "slots": [1, 2]})
    assert ADAPTER.validate_python(a.model_dump()) == a
```

- [ ] **Step 3: Verify fails**

```bash
uv run pytest tests/unit/tournament/test_schemas_discriminator.py -v
```

Expected: FAIL (imports missing).

- [ ] **Step 4: Implement schemas**

Append to `packages/atp-dashboard/atp/dashboard/tournament/schemas.py`:

```python
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from game_envs.games.el_farol import MAX_SLOTS_PER_DAY


class PDAction(BaseModel):
    """PD submit action. ``game_type`` is server-injected; clients may
    omit it on the wire (see spec §4)."""

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["prisoners_dilemma"]
    choice: Literal["cooperate", "defect"]


class ElFarolAction(BaseModel):
    """El Farol submit action. ``game_type`` is server-injected."""

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["el_farol"]
    slots: list[int] = Field(..., max_length=MAX_SLOTS_PER_DAY)


TournamentAction = Annotated[
    PDAction | ElFarolAction,
    Field(discriminator="game_type"),
]
```

- [ ] **Step 5: Verify passes**

```bash
uv run pytest tests/unit/tournament/test_schemas_discriminator.py -v
```

Expected: PASS (7/7).

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/schemas.py \
        tests/unit/tournament/test_schemas_discriminator.py
git commit -m "feat(tournament): add TournamentAction discriminated union"
```

---

### Task 14: Add RoundState discriminated-union schemas

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/schemas.py`
- Test: `tests/unit/tournament/test_schemas_discriminator.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
from atp.dashboard.tournament.schemas import (
    ElFarolRoundState,
    PDRoundState,
    RoundState,
)

STATE_ADAPTER = TypeAdapter(RoundState)


def test_parses_pd_round_state():
    s = STATE_ADAPTER.validate_python({
        "game_type": "prisoners_dilemma",
        "your_history": ["cooperate"],
        "opponent_history": ["defect"],
        "your_cumulative_score": 0.0,
        "opponent_cumulative_score": 5.0,
        "round_number": 2,
        "total_rounds": 5,
        "your_turn": True,
        "action_schema": {"type": "choice", "options": ["cooperate", "defect"]},
    })
    assert isinstance(s, PDRoundState)


def test_parses_el_farol_round_state():
    s = STATE_ADAPTER.validate_python({
        "game_type": "el_farol",
        "your_history": [[0, 3]],
        "attendance_by_round": [[1, 0, 0, 1]],
        "capacity_threshold": 3,
        "your_cumulative_score": 2.0,
        "all_scores": [2.0, 1.0, -1.0],
        "your_participant_idx": 0,
        "num_slots": 4,
        "round_number": 2,
        "total_rounds": 10,
        "pending_submission": False,
        "action_schema": {
            "type": "list[int]", "max_length": 8,
            "value_range": [0, 3], "unique": True,
        },
    })
    assert isinstance(s, ElFarolRoundState)
    assert s.pending_submission is False


def test_pd_roundstate_missing_your_turn_rejected():
    with pytest.raises(ValidationError):
        STATE_ADAPTER.validate_python({
            "game_type": "prisoners_dilemma",
            "your_history": [],
            "opponent_history": [],
            "your_cumulative_score": 0.0,
            "opponent_cumulative_score": 0.0,
            "round_number": 1,
            "total_rounds": 5,
            "action_schema": {},
            # your_turn missing
        })
```

- [ ] **Step 2: Verify fails**

```bash
uv run pytest tests/unit/tournament/test_schemas_discriminator.py -v
```

- [ ] **Step 3: Add RoundState schemas**

Append to `schemas.py`:

```python
class PDRoundState(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

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
    pending_submission: bool
    action_schema: dict


RoundState = Annotated[
    PDRoundState | ElFarolRoundState,
    Field(discriminator="game_type"),
]
```

- [ ] **Step 4: Verify passes**

```bash
uv run pytest tests/unit/tournament/test_schemas_discriminator.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/schemas.py \
        tests/unit/tournament/test_schemas_discriminator.py
git commit -m "feat(tournament): add RoundState discriminated union"
```

---

### Task 15: Add `_el_farol_for` cached factory and `_game_for` dispatcher

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Test: `tests/unit/dashboard/tournament/test_service_create.py` or new `test_game_factory.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/dashboard/tournament/test_game_factory.py`:

```python
"""Tests for the _game_for / _el_farol_for dispatch."""

import pytest

from atp.dashboard.tournament.service import (
    _EL_FAROL_V1_NUM_SLOTS,
    _EL_FAROL_V1_THRESHOLD_RATIO,
    _el_farol_for,
    _game_for,
)
from game_envs.games.el_farol import ElFarol
from game_envs.games.prisoners_dilemma import PrisonersDilemma


class _FakeTournament:
    def __init__(self, game_type: str, num_players: int = 2) -> None:
        self.game_type = game_type
        self.num_players = num_players


def test_game_for_pd_is_singleton():
    t1 = _FakeTournament("prisoners_dilemma")
    t2 = _FakeTournament("prisoners_dilemma")
    assert _game_for(t1) is _game_for(t2)
    assert isinstance(_game_for(t1), PrisonersDilemma)


def test_game_for_el_farol_caches_by_num_players():
    t5a = _FakeTournament("el_farol", num_players=5)
    t5b = _FakeTournament("el_farol", num_players=5)
    t7 = _FakeTournament("el_farol", num_players=7)
    g5a = _game_for(t5a)
    g5b = _game_for(t5b)
    g7 = _game_for(t7)
    assert g5a is g5b  # cached
    assert g5a is not g7  # different N
    assert isinstance(g5a, ElFarol)


def test_el_farol_capacity_threshold_derived_from_ratio():
    g = _el_farol_for(num_players=10)
    expected = max(1, int(_EL_FAROL_V1_THRESHOLD_RATIO * 10))
    assert g.config.capacity_threshold == expected
    assert g.config.num_slots == _EL_FAROL_V1_NUM_SLOTS
    assert g.config.num_players == 10


def test_el_farol_min_players_yields_valid_threshold():
    g = _el_farol_for(num_players=2)
    assert g.config.capacity_threshold >= 1


def test_game_for_unknown_raises():
    from atp.dashboard.tournament.errors import ValidationError
    t = _FakeTournament("tic_tac_toe")
    with pytest.raises(ValidationError, match="unsupported"):
        _game_for(t)
```

Also add an autouse fixture to clear cache between tests. Create `tests/unit/dashboard/tournament/conftest.py` addition (check if it exists first):

```bash
cat tests/unit/dashboard/tournament/conftest.py
```

Append:

```python
@pytest.fixture(autouse=True)
def _clear_el_farol_cache():
    """Spec §7.1 fixture hygiene."""
    from atp.dashboard.tournament.service import _el_farol_for
    yield
    _el_farol_for.cache_clear()
```

- [ ] **Step 2: Verify fails**

```bash
uv run pytest tests/unit/dashboard/tournament/test_game_factory.py -v
```

Expected: FAIL (imports missing).

- [ ] **Step 3: Implement in `service.py`**

At the top-level of `service.py` (where `_GAME_INSTANCES` currently lives at lines 64-68), replace:

```python
_SUPPORTED_GAMES = frozenset({"prisoners_dilemma"})

_GAME_INSTANCES: dict[str, Any] = {
    "prisoners_dilemma": PrisonersDilemma(),
}
```

with:

```python
import functools

from game_envs.games.el_farol import ElFarol, ElFarolConfig

_SUPPORTED_GAMES = frozenset({"prisoners_dilemma", "el_farol"})

_PD_SINGLETON: PrisonersDilemma = PrisonersDilemma()

# Hardcoded El Farol V1 preset (spec §3.2 step 2).
_EL_FAROL_V1_NUM_SLOTS = 16
_EL_FAROL_V1_THRESHOLD_RATIO = 0.6
_EL_FAROL_V1_MIN_TOTAL_HOURS = 0

# Startup assert — spec §12 silent-min_total_hours mitigation.
assert _EL_FAROL_V1_MIN_TOTAL_HOURS == 0, (
    "Raising _EL_FAROL_V1_MIN_TOTAL_HOURS without a Phase C "
    "finalize_scores hook would silently ignore DQ. See spec §12."
)


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


def _game_for(tournament: Any) -> Any:
    gt = tournament.game_type
    if gt == "prisoners_dilemma":
        return _PD_SINGLETON
    if gt == "el_farol":
        return _el_farol_for(tournament.num_players)
    raise ValidationError(f"unsupported game_type {gt!r}")
```

- [ ] **Step 4: Update existing `_GAME_INSTANCES[...]` call-sites**

```bash
grep -n "_GAME_INSTANCES" packages/atp-dashboard/atp/dashboard/tournament/service.py
```

For each hit (create_tournament `required_players = _GAME_INSTANCES[...]`, `_resolve_round`, etc.), replace with `_game_for(tournament)`. Where only `num_players` is needed, read from `tournament.num_players` directly.

Delete the `_GAME_INSTANCES` dict definition.

- [ ] **Step 5: Run the factory tests**

```bash
uv run pytest tests/unit/dashboard/tournament/test_game_factory.py -v
```

Expected: PASS.

- [ ] **Step 6: Run full tournament unit test sweep**

```bash
uv run pytest tests/unit/dashboard/tournament/ -v -x
```

Expected: PASS (PD behavior unchanged).

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_game_factory.py \
        tests/unit/dashboard/tournament/conftest.py
git commit -m "feat(tournament): _game_for factory and cached _el_farol_for"
```

---

### Task 16: Update `create_tournament` validation for El Farol N bound

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py` (`create_tournament`, lines ~76–146)
- Test: `tests/unit/dashboard/tournament/test_service_create.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/dashboard/tournament/test_service_create.py`:

```python
@pytest.mark.anyio
async def test_create_el_farol_n5_ok(service_factory, test_user):
    svc = await service_factory()
    t, _ = await svc.create_tournament(
        test_user, name="smoke", game_type="el_farol",
        num_players=5, total_rounds=3, round_deadline_s=30,
    )
    assert t.game_type == "el_farol"
    assert t.num_players == 5


@pytest.mark.anyio
async def test_create_el_farol_n1_rejected(service_factory, test_user):
    svc = await service_factory()
    with pytest.raises(ValidationError, match="2 <= num_players <= 20"):
        await svc.create_tournament(
            test_user, name="nope", game_type="el_farol",
            num_players=1, total_rounds=3, round_deadline_s=30,
        )


@pytest.mark.anyio
async def test_create_el_farol_n21_rejected(service_factory, test_user):
    svc = await service_factory()
    with pytest.raises(ValidationError, match="2 <= num_players <= 20"):
        await svc.create_tournament(
            test_user, name="nope", game_type="el_farol",
            num_players=21, total_rounds=3, round_deadline_s=30,
        )


@pytest.mark.anyio
async def test_create_pd_still_requires_exactly_two(service_factory, test_user):
    svc = await service_factory()
    with pytest.raises(ValidationError, match="exactly 2"):
        await svc.create_tournament(
            test_user, name="nope", game_type="prisoners_dilemma",
            num_players=3, total_rounds=3, round_deadline_s=30,
        )
```

(Check `service_factory` / `test_user` fixtures exist in `conftest.py`. If they don't, follow the pattern in `test_service_create.py` existing tests — likely they already exist from Plan 2a.)

- [ ] **Step 2: Verify fails**

```bash
uv run pytest tests/unit/dashboard/tournament/test_service_create.py -k el_farol -v
```

- [ ] **Step 3: Replace validation block in `create_tournament`**

Locate the block at ~lines 92–103 in `service.py`:

```python
if game_type not in _SUPPORTED_GAMES:
    raise ValidationError(...)
required_players = _GAME_INSTANCES[game_type].config.num_players
if num_players != required_players:
    ...
```

Replace with:

```python
if game_type not in _SUPPORTED_GAMES:
    raise ValidationError(
        f"unsupported game_type {game_type!r}; "
        f"supports: {sorted(_SUPPORTED_GAMES)}"
    )
if game_type == "prisoners_dilemma":
    if num_players != 2:
        raise ValidationError(
            f"prisoners_dilemma requires exactly 2 players, got {num_players}"
        )
elif game_type == "el_farol":
    if not (2 <= num_players <= 20):
        raise ValidationError(
            f"el_farol requires 2 <= num_players <= 20 (phase B bound), "
            f"got {num_players}"
        )
```

- [ ] **Step 4: Run creation tests**

```bash
uv run pytest tests/unit/dashboard/tournament/test_service_create.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_service_create.py
git commit -m "feat(tournament): accept el_farol game_type with N in [2,20]"
```

---

### Task 17: `submit_action` — server-side `game_type` injection and friendly ValidationError

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py` (`submit_action`)
- Test: `tests/unit/dashboard/tournament/test_service_submit.py` (create if missing)

- [ ] **Step 1: Create test file skeleton**

```bash
ls tests/unit/dashboard/tournament/test_service_submit.py 2>/dev/null || \
    touch tests/unit/dashboard/tournament/test_service_submit.py
```

- [ ] **Step 2: Write failing tests**

Write `tests/unit/dashboard/tournament/test_service_submit.py` (reuse `service_factory` / `test_user` fixtures):

```python
"""submit_action: server-side game_type injection + mismatch detection."""

import pytest
from atp.dashboard.tournament.errors import ValidationError


@pytest.mark.anyio
async def test_pd_bot_without_game_type_still_works(service_factory, pd_tournament):
    svc = await service_factory()
    result = await svc.submit_action(
        pd_tournament.id, pd_tournament.creator,
        action={"choice": "cooperate"},  # no game_type
    )
    assert result["status"] in ("waiting", "round_resolved")


@pytest.mark.anyio
async def test_pd_bot_with_mismatched_game_type_rejected(service_factory, pd_tournament):
    svc = await service_factory()
    with pytest.raises(ValidationError, match="does not match"):
        await svc.submit_action(
            pd_tournament.id, pd_tournament.creator,
            action={"game_type": "el_farol", "choice": "cooperate"},
        )


@pytest.mark.anyio
async def test_pd_action_to_el_farol_tournament_error_includes_hint(
    service_factory, el_farol_tournament
):
    svc = await service_factory()
    with pytest.raises(ValidationError) as exc:
        await svc.submit_action(
            el_farol_tournament.id, el_farol_tournament.creator,
            action={"choice": "cooperate"},  # wrong shape
        )
    text = str(exc.value)
    assert "el_farol" in text
    assert "slots" in text  # hint includes expected fields


@pytest.mark.anyio
async def test_el_farol_submit_happy(service_factory, el_farol_tournament):
    svc = await service_factory()
    result = await svc.submit_action(
        el_farol_tournament.id, el_farol_tournament.creator,
        action={"slots": [0, 3]},
    )
    assert result["status"] in ("waiting", "round_resolved")
```

(If `pd_tournament` / `el_farol_tournament` fixtures don't exist, add them to `conftest.py` as factory fixtures that create a tournament and join the first user. Model them on existing `test_service_resolve.py` fixtures.)

- [ ] **Step 3: Verify fails**

```bash
uv run pytest tests/unit/dashboard/tournament/test_service_submit.py -v
```

Expected: FAIL (some will fail on missing fixtures; add them first).

- [ ] **Step 4: Replace `submit_action` validation**

In `service.py:submit_action`, locate the block at ~lines 480-510 (which Task 6 already cleaned up to call `game.validate_action`). Replace the validation+persist portion with:

```python
import functools as _functools  # at file top if not present
from pydantic import TypeAdapter, ValidationError as PydanticValidationError

from atp.dashboard.tournament.schemas import TournamentAction


_ACTION_ADAPTER = TypeAdapter(TournamentAction)


def _action_hint_for(game_type: str) -> str:
    if game_type == "prisoners_dilemma":
        return "{choice: 'cooperate' | 'defect'}"
    if game_type == "el_farol":
        return (
            "{slots: list[int], values in [0, num_slots-1], "
            f"unique, max {MAX_SLOTS_PER_DAY} entries}}"
        )
    return "{} (unknown game_type)"


# inside submit_action, after fetching `tournament`:
incoming_gt = action.get("game_type") if isinstance(action, dict) else None
if incoming_gt is not None and incoming_gt != tournament.game_type:
    raise ValidationError(
        f"action game_type {incoming_gt!r} does not match "
        f"tournament {tournament_id} game_type {tournament.game_type!r}"
    )
action_with_type = {**action, "game_type": tournament.game_type}
try:
    typed = _ACTION_ADAPTER.validate_python(action_with_type)
except PydanticValidationError as e:
    expected = _action_hint_for(tournament.game_type)
    errors = e.errors()
    first_err = errors[0] if errors else {"msg": "unknown"}
    raise ValidationError(
        f"invalid action for tournament {tournament_id} "
        f"(game_type={tournament.game_type!r}); "
        f"expected fields: {expected}; "
        f"pydantic: {first_err.get('loc')}: {first_err.get('msg')}"
    ) from e

game = _game_for(tournament)
canonical = game.validate_action(
    typed.model_dump(exclude={"game_type"})
)
# persist `canonical` as before (Task 6 already does action_data=canonical)
```

Import MAX_SLOTS_PER_DAY at the top:

```python
from game_envs.games.el_farol import MAX_SLOTS_PER_DAY
```

- [ ] **Step 5: Run submit tests**

```bash
uv run pytest tests/unit/dashboard/tournament/test_service_submit.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_service_submit.py \
        tests/unit/dashboard/tournament/conftest.py
git commit -m "feat(tournament): server-side game_type injection + tournament-aware error"
```

---

### Task 18: `get_state_for` — inject `pending_submission` / `your_turn` and parse into RoundState

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py` (`get_state_for`, lines ~339–428)
- Test: existing `tests/unit/dashboard/tournament/test_service_state.py`

- [ ] **Step 1: Write failing tests**

Append to `test_service_state.py`:

```python
@pytest.mark.anyio
async def test_state_pd_has_your_turn(service_factory, pd_tournament):
    svc = await service_factory()
    state = await svc.get_state_for(pd_tournament.id, pd_tournament.creator)
    # RoundState is parsed into a typed model; attribute access works
    assert state.game_type == "prisoners_dilemma"
    assert isinstance(state.your_turn, bool)


@pytest.mark.anyio
async def test_state_el_farol_has_pending_submission(
    service_factory, el_farol_tournament
):
    svc = await service_factory()
    state = await svc.get_state_for(
        el_farol_tournament.id, el_farol_tournament.creator
    )
    assert state.game_type == "el_farol"
    assert state.pending_submission is True  # creator hasn't submitted yet
    assert hasattr(state, "attendance_by_round")
    assert hasattr(state, "capacity_threshold")


@pytest.mark.anyio
async def test_state_el_farol_pending_flips_after_submit(
    service_factory, el_farol_tournament
):
    svc = await service_factory()
    await svc.submit_action(
        el_farol_tournament.id, el_farol_tournament.creator,
        action={"slots": [0]},
    )
    state = await svc.get_state_for(
        el_farol_tournament.id, el_farol_tournament.creator
    )
    assert state.pending_submission is False
```

- [ ] **Step 2: Verify fails**

```bash
uv run pytest tests/unit/dashboard/tournament/test_service_state.py -k pending_submission -v
```

- [ ] **Step 3: Implement**

In `service.py:get_state_for`, after calling `game.format_state_for_player(...)`:

```python
# Compute submission state from DB
has_submitted = await self._session.execute(
    select(Action.id)
    .join(Round, Round.id == Action.round_id)
    .where(
        Round.tournament_id == tournament_id,
        Round.status == RoundStatus.WAITING_FOR_ACTIONS,
        Action.participant_id == my_participant.id,
    )
    .limit(1)
)
already = has_submitted.scalar_one_or_none() is not None

# Inject submission-state field per game_type (spec §3.2 step 6)
if tournament.game_type == "prisoners_dilemma":
    formatted["your_turn"] = not already
elif tournament.game_type == "el_farol":
    formatted["pending_submission"] = not already
    formatted.pop("extra", None)  # El Farol state schema has no 'extra'
else:
    # defense in depth
    raise ValidationError(f"unsupported game_type {tournament.game_type!r}")

formatted["game_type"] = tournament.game_type
formatted["tournament_id"] = tournament_id

# Parse into discriminated union
from atp.dashboard.tournament.schemas import RoundState as _RS_UNION
round_state = TypeAdapter(_RS_UNION).validate_python(formatted)
return round_state
```

**Note:** the existing method returns a `RoundState` dataclass from `state.py`; audit whether the return type needs to change to the new pydantic union. If there are external callers that expect the dataclass, keep dataclass parsing too. Review `return RoundState(...)` at lines ~416-428 and decide: either (a) keep the dataclass for back-compat and just attach the new typed state as a field, or (b) switch all callers to the pydantic union. Prefer (b) for cleanliness — grep for `RoundState(` callers first:

```bash
grep -rn "RoundState(" packages/atp-dashboard/ tests/
```

- [ ] **Step 4: Update `state.py` RoundState if needed**

If `tournament/state.py` `RoundState` is still the external contract (MCP tool return type), either make it the union (import from schemas) or add conversion at the MCP layer.

- [ ] **Step 5: Run state tests**

```bash
uv run pytest tests/unit/dashboard/tournament/test_service_state.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        packages/atp-dashboard/atp/dashboard/tournament/state.py \
        tests/unit/dashboard/tournament/test_service_state.py
git commit -m "feat(tournament): inject pending_submission/your_turn, parse into RoundState union"
```

---

### Task 19: Hook `compute_round_payoffs` dispatch in `_resolve_round`

PR-1 Task 4 introduced `game.compute_round_payoffs` on PD. Task 12 added it to ElFarol. Make sure `_resolve_round` is actually routed through `_game_for` (not the old `_GAME_INSTANCES[...]`) and works for both games.

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py` (`_resolve_round`)
- Test: `tests/unit/dashboard/tournament/test_service_resolve.py` (add El Farol case)

- [ ] **Step 1: Write failing test**

Append to `test_service_resolve.py`:

```python
@pytest.mark.anyio
async def test_el_farol_resolve_round_writes_payoffs(
    service_factory, el_farol_tournament_with_joins  # N=3 joined
):
    svc = await service_factory()
    t = el_farol_tournament_with_joins
    # All 3 participants submit
    for p in t.participants:
        await svc.submit_action(t.id, p.user, action={"slots": [0, 1]})
    # ... round should resolve synchronously on the last submit
    # Check all actions got non-None payoffs
    from atp.dashboard.tournament.models import Action
    actions = (await svc._session.execute(select(Action))).scalars().all()
    for a in actions:
        assert a.payoff is not None
```

(Fixture `el_farol_tournament_with_joins` creates + auto-joins N=3; build it from existing patterns.)

- [ ] **Step 2: Verify fails or passes**

```bash
uv run pytest tests/unit/dashboard/tournament/test_service_resolve.py -k el_farol -v
```

If Task 15 already routed through `_game_for`, this test may pass. If not, fix `_resolve_round`:

- [ ] **Step 3: Ensure `_resolve_round` uses `_game_for`**

```bash
grep -n "_GAME_INSTANCES\|_game_for" packages/atp-dashboard/atp/dashboard/tournament/service.py
```

Any remaining `_GAME_INSTANCES[tournament.game_type]` → `_game_for(tournament)`.

- [ ] **Step 4: Run resolve tests**

```bash
uv run pytest tests/unit/dashboard/tournament/test_service_resolve.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_service_resolve.py
git commit -m "feat(tournament): _resolve_round routes via _game_for for both games"
```

---

### Task 20: MCP tool description update and `list_tournaments(game_type=...)` filter

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/mcp/tools.py`
- Test: `tests/unit/dashboard/mcp/test_tools.py`

- [ ] **Step 1: Read current `make_move` tool**

```bash
sed -n '195,225p' packages/atp-dashboard/atp/dashboard/mcp/tools.py
```

- [ ] **Step 2: Update tool docstring**

Replace the `make_move` docstring with:

```python
"""Submit an action for the current round.

Args:
    tournament_id: The tournament to submit to.
    action: Dict whose required fields depend on the tournament's
        game_type. For prisoners_dilemma:
        ``{"choice": "cooperate" | "defect"}``.
        For el_farol:
        ``{"slots": list[int], values in [0, num_slots-1], unique,
        max 8 entries}``. Note: El Farol players can attend at most
        8 of 16 slots per day.

    ``game_type`` is optional; server reads it from the tournament
    record. If you send it and it mismatches, the server returns 422.
"""
```

Locate `list_tournaments` in the same file and add a `game_type: str | None = None` parameter. Update the service call to pass it through:

```python
async def list_tournaments(
    user: User,
    service: TournamentService,
    status: str | None = None,
    game_type: str | None = None,  # new
) -> list[dict]:
    """List tournaments, optionally filtered by status or game_type."""
    ...
    return await service.list_tournaments(
        user=user, status=status_filter, game_type=game_type,
    )
```

Update `TournamentService.list_tournaments` in `service.py` to accept and filter on `game_type` (add `.where(Tournament.game_type == game_type)` when non-None).

- [ ] **Step 3: Write test**

Append to `tests/unit/dashboard/mcp/test_tools.py`:

```python
@pytest.mark.anyio
async def test_list_tournaments_filters_by_game_type(
    mcp_test_client, test_user, async_session,
):
    """Spec §4: optional game_type filter on list_tournaments."""
    svc = TournamentService(async_session, bus=...)  # use existing fixture
    pd_t, _ = await svc.create_tournament(
        test_user, name="pd1", game_type="prisoners_dilemma",
        num_players=2, total_rounds=1, round_deadline_s=30,
    )
    ef_t, _ = await svc.create_tournament(
        test_user, name="ef1", game_type="el_farol",
        num_players=3, total_rounds=1, round_deadline_s=30,
    )
    await async_session.commit()

    # No filter -> both
    all_tournaments = await svc.list_tournaments(user=test_user)
    ids = {t.id for t in all_tournaments}
    assert pd_t.id in ids and ef_t.id in ids

    # Filter by game_type
    pd_only = await svc.list_tournaments(user=test_user, game_type="prisoners_dilemma")
    assert {t.id for t in pd_only} == {pd_t.id}

    ef_only = await svc.list_tournaments(user=test_user, game_type="el_farol")
    assert {t.id for t in ef_only} == {ef_t.id}
```

(Reuse `mcp_test_client`, `test_user`, `async_session` fixtures from
`tests/unit/dashboard/mcp/conftest.py` and
`tests/unit/dashboard/tournament/conftest.py`. If the `bus` argument
signature doesn't match your conftest, follow the pattern used in
existing `test_tools.py::test_list_tournaments` — that test already
invokes `list_tournaments` without `game_type` and can be extended.)

- [ ] **Step 4: Run MCP + list tests**

```bash
uv run pytest tests/unit/dashboard/mcp/ tests/unit/dashboard/tournament/ -k list -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/mcp/tools.py \
        packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/mcp/test_tools.py
git commit -m "feat(mcp): update make_move description and list_tournaments game_type filter"
```

---

### Task 21: Structured observability logs

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Test: `tests/unit/dashboard/tournament/test_service_resolve.py` (log assertion)

- [ ] **Step 1: Write failing test**

Append to `test_service_resolve.py`:

```python
import logging


@pytest.mark.anyio
async def test_resolve_round_logs_structured_fields(
    service_factory, el_farol_tournament_with_joins, caplog
):
    svc = await service_factory()
    t = el_farol_tournament_with_joins
    with caplog.at_level(logging.INFO, logger="atp.dashboard.tournament.service"):
        for p in t.participants:
            await svc.submit_action(t.id, p.user, action={"slots": [0]})
    # Find the round_resolved log entry
    rec = next(
        r for r in caplog.records
        if getattr(r, "event", None) == "round_resolved"
    )
    assert rec.game_type == "el_farol"
    assert rec.tournament_id == t.id
    assert rec.round_number == 1
    assert rec.round_resolution_ms >= 0
```

- [ ] **Step 2: Implement structured logging**

In `_resolve_round`, wrap the body:

```python
import time

start = time.perf_counter()
# ... existing body that computes payoffs, flushes, emits events
elapsed_ms = int((time.perf_counter() - start) * 1000)
logger.info(
    "round_resolved",
    extra={
        "event": "round_resolved",
        "game_type": tournament.game_type,
        "tournament_id": tournament.id,
        "round_number": current_round.round_number,
        "round_resolution_ms": elapsed_ms,
    },
)
```

In `submit_action`, on ValidationError path, add:

```python
logger.info(
    "action_rejected",
    extra={
        "event": "action_rejected",
        "game_type": tournament.game_type if tournament else None,
        "tournament_id": tournament_id,
        "validation_error_path": "client_submission",
    },
)
```

- [ ] **Step 3: Run**

```bash
uv run pytest tests/unit/dashboard/tournament/test_service_resolve.py -k logs -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_service_resolve.py
git commit -m "feat(tournament): structured observability fields per game_type"
```

---

### Task 22: Integration test — full El Farol flow

**Files:**
- Create: `tests/integration/tournament/test_el_farol_flow.py`

- [ ] **Step 1: Check integration-test infrastructure**

```bash
ls tests/integration/tournament/ 2>/dev/null
```

If empty/missing, scaffold from existing integration tests (`tests/integration/dashboard/tournament/*.py`).

- [ ] **Step 2: Write the integration test**

```python
"""End-to-end tournament flow for El Farol (spec §7.2)."""

import pytest
from atp.dashboard.tournament.models import RoundStatus, TournamentStatus
from atp.dashboard.tournament.service import TournamentService


@pytest.mark.anyio
async def test_el_farol_full_flow_n5_r3(async_session, test_users_5):
    bus = ...  # event bus fixture
    svc = TournamentService(async_session, bus)
    creator = test_users_5[0]
    t, _ = await svc.create_tournament(
        creator, name="e2e", game_type="el_farol",
        num_players=5, total_rounds=3, round_deadline_s=60,
    )
    # All 5 join
    for u in test_users_5:
        await svc.join(t.id, u, agent_name=u.username)

    assert t.status == TournamentStatus.ACTIVE

    # Round 1: all submit varied slots
    submissions = [
        {"slots": [0, 1]},
        {"slots": [2, 3]},
        {"slots": [0]},
        {"slots": [4, 5]},
        {"slots": []},
    ]
    for u, a in zip(test_users_5, submissions):
        await svc.submit_action(t.id, u, action=a)

    # Round 2: all submit
    for u in test_users_5:
        await svc.submit_action(t.id, u, action={"slots": [0]})

    # Round 3: only 4 submit, 5th times out -> default empty slots
    for u in test_users_5[:4]:
        await svc.submit_action(t.id, u, action={"slots": [1, 2]})
    # Force deadline resolution
    current_round = ...  # query the third round
    await svc.force_resolve_round(current_round.id)

    assert t.status == TournamentStatus.COMPLETED

    # All 5 participants should have cumulative scores assigned
    state = await svc.get_state_for(t.id, test_users_5[0])
    assert len(state.all_scores) == 5
```

(Fixtures `async_session`, `test_users_5` — model on existing integration conftest.)

- [ ] **Step 3: Run**

```bash
uv run pytest tests/integration/tournament/test_el_farol_flow.py -v
```

Expected: PASS.

- [ ] **Step 4: Also run PD regression integration**

Find any existing PD integration (`tests/integration/dashboard/tournament/test_e2e_30_round_pd_with_reconnect.py`):

```bash
uv run pytest tests/integration/dashboard/tournament/ -v
```

Expected: PASS (unchanged).

- [ ] **Step 5: Commit**

```bash
git add tests/integration/tournament/test_el_farol_flow.py
git commit -m "test(el-farol): integration flow N=5 R=3 with timeout-default"
```

---

### Task 23: Demo-game El Farol agent updates

**Files:**
- Modify: `demo-game/agents/el_farol_agent.py`
- Modify: `demo-game/suites/el_farol_llm_vs_builtin.yaml`

- [ ] **Step 1: Read current agent**

```bash
cat demo-game/agents/el_farol_agent.py
```

- [ ] **Step 2: Update action payload shape**

Ensure submissions go through MCP `make_move` with `{"slots": [...]}` (no `game_type` required; server injects).

Ensure state handling reads `state["game_type"] == "el_farol"` then consumes `attendance_by_round`, `capacity_threshold`, `your_history`, `pending_submission`.

Sketch of the submit helper:

```python
async def submit(self, client, tournament_id, chosen_slots: list[int]) -> None:
    await client.make_move(
        tournament_id=tournament_id,
        action={"slots": chosen_slots},
    )
```

- [ ] **Step 3: Update suite YAML**

In `demo-game/suites/el_farol_llm_vs_builtin.yaml`, ensure the game_type reference points to the server MCP endpoint:

```yaml
tournament:
  game_type: el_farol
  num_players: 5
  total_rounds: 20
  round_deadline_s: 30
```

- [ ] **Step 4: Lint**

```bash
uv run ruff check demo-game/agents/el_farol_agent.py
```

- [ ] **Step 5: Commit**

```bash
git add demo-game/agents/el_farol_agent.py demo-game/suites/el_farol_llm_vs_builtin.yaml
git commit -m "feat(demo-game): update el farol agent for live MCP tournament"
```

---

### Task 24: Demo-game compose harness

**Files:**
- Create: `demo-game/compose.el-farol.yml`

- [ ] **Step 1: Model on existing PD compose**

```bash
ls demo-game/compose*.yml demo-game/*.yml 2>/dev/null
cat demo-game/compose.pd.yml 2>/dev/null || cat demo-game/compose.yml 2>/dev/null
```

- [ ] **Step 2: Write `compose.el-farol.yml`**

```yaml
# N=5 El Farol tournament: 3 LLM bots + 2 built-in strategies.
# Spec §7.3.
services:
  llm-bot-0:
    build:
      context: .
      dockerfile: Containerfile.el-farol
    environment:
      - ATP_MCP_URL=${ATP_MCP_URL:-http://host.docker.internal:8080/mcp}
      - ATP_TOKEN=${ATP_TOKEN_LLM_0}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - BOT_STRATEGY=llm
      - TOURNAMENT_ID=${TOURNAMENT_ID}

  llm-bot-1:
    extends:
      service: llm-bot-0
    environment:
      - ATP_TOKEN=${ATP_TOKEN_LLM_1}

  llm-bot-2:
    extends:
      service: llm-bot-0
    environment:
      - ATP_TOKEN=${ATP_TOKEN_LLM_2}

  greedy-bot:
    extends:
      service: llm-bot-0
    environment:
      - ATP_TOKEN=${ATP_TOKEN_GREEDY}
      - BOT_STRATEGY=greedy

  random-bot:
    extends:
      service: llm-bot-0
    environment:
      - ATP_TOKEN=${ATP_TOKEN_RANDOM}
      - BOT_STRATEGY=random
```

- [ ] **Step 3: Commit**

```bash
git add demo-game/compose.el-farol.yml
git commit -m "feat(demo-game): compose harness for el farol N=5 E2E"
```

---

### Task 25: Pre-E2E load smoke script

**Files:**
- Create: `scripts/smoke_el_farol_load.py`

- [ ] **Step 1: Write the script (spec §7.3)**

```python
"""Pre-E2E load smoke: N=20, R=30, built-in random bots only.

Spec §7.3. Pass criteria:
  - exit 0 (no service timeouts, all rounds resolve)
  - get_current_state p95 latency stays within 2x of round-1 p95
    (catches O(R) payload bloat from attendance_by_round).

Usage:
    export ATP_ADMIN_TOKEN=<admin_user_token>
    uv run python scripts/smoke_el_farol_load.py
"""

from __future__ import annotations

import asyncio
import os
import random
import statistics
import sys
import time
from typing import Any

import httpx

N = 20
R = 30
ROUND_DEADLINE_S = 30
SERVER = os.environ.get("ATP_SERVER_URL", "http://127.0.0.1:8080")
ADMIN_TOKEN = os.environ["ATP_ADMIN_TOKEN"]
NUM_SLOTS = 16  # mirrors _EL_FAROL_V1_NUM_SLOTS


def _admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


async def _mint_tokens(client: httpx.AsyncClient) -> list[str]:
    """Create N bot users + API tokens via admin endpoints."""
    tokens: list[str] = []
    for i in range(N):
        r = await client.post(
            "/api/v1/admin/bot-users",
            json={"username": f"smoke_bot_{i}"},
            headers=_admin_headers(),
        )
        r.raise_for_status()
        tokens.append(r.json()["api_token"])
    return tokens


async def _create_tournament(client: httpx.AsyncClient) -> int:
    r = await client.post(
        "/api/v1/tournaments",
        json={
            "name": "el-farol-smoke",
            "game_type": "el_farol",
            "num_players": N,
            "total_rounds": R,
            "round_deadline_s": ROUND_DEADLINE_S,
        },
        headers=_admin_headers(),
    )
    r.raise_for_status()
    return r.json()["id"]


async def _join(client: httpx.AsyncClient, tid: int, token: str, idx: int) -> None:
    r = await client.post(
        f"/api/v1/tournaments/{tid}/join",
        json={"agent_name": f"smoke_bot_{idx}"},
        headers={"Authorization": f"Bearer {token}"},
    )
    r.raise_for_status()


async def _get_state_timed(
    client: httpx.AsyncClient, tid: int, token: str
) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    r = await client.get(
        f"/api/v1/tournaments/{tid}/state",
        headers={"Authorization": f"Bearer {token}"},
    )
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), elapsed


async def _submit_random(
    client: httpx.AsyncClient, tid: int, token: str, rng: random.Random
) -> None:
    k = rng.randint(0, 8)
    slots = sorted(rng.sample(range(NUM_SLOTS), k))
    r = await client.post(
        f"/api/v1/tournaments/{tid}/actions",
        json={"action_data": {"slots": slots}},
        headers={"Authorization": f"Bearer {token}"},
    )
    r.raise_for_status()


async def _run() -> int:
    async with httpx.AsyncClient(base_url=SERVER, timeout=60) as client:
        tokens = await _mint_tokens(client)
        tid = await _create_tournament(client)
        await asyncio.gather(
            *(_join(client, tid, tok, i) for i, tok in enumerate(tokens))
        )

        latencies_per_round: list[list[float]] = []
        for round_idx in range(R):
            # Each bot: get state (timed), then submit random
            states_and_lats = await asyncio.gather(
                *(_get_state_timed(client, tid, tok) for tok in tokens)
            )
            latencies_per_round.append([lat for _s, lat in states_and_lats])
            rng = random.Random(round_idx)
            await asyncio.gather(
                *(_submit_random(client, tid, tok, rng) for tok in tokens)
            )

    for r_idx, lats in enumerate(latencies_per_round):
        if len(lats) >= 20:
            p95 = statistics.quantiles(lats, n=20)[18]
        else:
            p95 = max(lats)
        print(f"round {r_idx + 1}: p95 get_state = {p95 * 1000:.1f}ms")

    first_p95 = max(latencies_per_round[0])
    last_p95 = max(latencies_per_round[-1])
    if last_p95 > first_p95 * 2:
        print(
            f"FAIL: p95 degraded {first_p95*1000:.1f}ms -> {last_p95*1000:.1f}ms (>2x)",
            file=sys.stderr,
        )
        return 1
    print(f"OK: p95 flat ({first_p95*1000:.1f}ms -> {last_p95*1000:.1f}ms)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
```

**Note on endpoints:** the script uses the REST tournament API
(`/api/v1/tournaments/*`). If the server exposes admin bot-user minting
under a different path or if these endpoints require a different
ownership model, adapt to match `atp/dashboard/v2/routes/` handlers
(grep `@router.post` in `tournament_api.py` and `agent_management_api.py`).
The point of the script is timing `get_state` — exact endpoint paths are
adjustable, the assertion is not.

- [ ] **Step 2: Document in demo-game README**

Append to `demo-game/README.md`:

```markdown
## El Farol smoke + E2E

### Pre-E2E load smoke (built-in bots, no LLM)

    uv run python scripts/smoke_el_farol_load.py

N=20, R=30. Pass = exit 0, get_state p95 flat round-over-round.

### Full E2E (LLM)

    export OPENAI_API_KEY=...
    export ATP_TOKEN_LLM_0=...
    ...
    docker compose -f demo-game/compose.el-farol.yml up
```

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_el_farol_load.py demo-game/README.md
git commit -m "test(el-farol): pre-E2E load smoke script (N=20, R=30)"
```

---

### Task 26: PR-2 checkpoint — full suite, lint, diff, PR open

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v --cov=atp --cov-report=term-missing -m "not slow"
```

Expected: PASS. Coverage on new El Farol code ≥85%; tournament service overall ≥83%.

- [ ] **Step 2: Lint, format, type-check**

```bash
uv run ruff format .
uv run ruff check .
uv run pyrefly check
```

Expected: clean.

- [ ] **Step 3: Diff size**

```bash
git diff main --stat
```

Target: ≤700 LOC (spec §9).

- [ ] **Step 4: Run pre-E2E load smoke (requires running server)**

Terminal 1:
```bash
uv run atp dashboard  # or start the server
```

Terminal 2:
```bash
uv run python scripts/smoke_el_farol_load.py
```

Expected: exit 0, no latency degradation.

- [ ] **Step 5: Open PR-2**

```bash
gh pr create --title "feat(tournament): el farol as second game_type [deploy]" \
    --body "$(cat <<'EOF'
## Summary
- El Farol Bar Problem as a live MCP tournament game_type.
- Discriminated-union schemas for actions and state; server injects game_type from the tournament record (no wire break for PD bots).
- N in [2, 20] for Phase B. Per-round payoff scoring (sum of happy − crowded).
- Structured observability logs per game_type.
- Demo-game harness for a 5-bot E2E (3 LLM + greedy + random).

## Test plan
- [x] Unit tests: game-layer El Farol + schema discriminator + service factory + create/submit/state
- [x] Integration test: N=5 R=3 full flow including timeout default
- [x] Pre-E2E load smoke (N=20, R=30, built-in bots): exit 0, p95 flat
- [ ] Full E2E with LLM bots (post-merge on production)

Spec: docs/superpowers/specs/2026-04-15-el-farol-tournament-design.md (v4).
Follows PR-1 (refactor #...).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: Post-merge production smoke**

After PR-2 merges and deploys (commit message contains `[deploy]`):

1. SSH or use dashboard UI to create a tournament: `game_type=el_farol, num_players=5, total_rounds=20, round_deadline_s=30`.
2. Run `docker compose -f demo-game/compose.el-farol.yml up` with production MCP URL.
3. Verify: tournament completes, all 5 bots finish each round, leaderboard populated, `grep game_type=el_farol /var/log/atp/*.log` returns round_resolved entries.

Only after this smoke passes is Phase B considered complete.

---

## Post-B: Phase C intake

After Phase B is in production and the smoke tournament has validated it, record any observed pain points in memory file `project_mcp_backlog.md` for Phase C planning. Particular items to note: MAX_SLOTS_PER_DAY migration to Config, `your_turn` → `pending_submission` unification, `finalize_scores` hook for terminal metrics, N ≤ 20 → ≤ 50 after load-test pass.

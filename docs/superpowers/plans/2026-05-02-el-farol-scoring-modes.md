# El Farol Scoring Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `scoring_mode: Literal["happy_only", "happy_minus_crowded"]` field to `ElFarolConfig` (default `happy_only`), branch the engine's three scoring methods + `to_prompt()` on it, flip the default formula from `happy − crowded` to `happy`, and update all downstream consumers (rules docs, dashboard copy, demo agent, participant-kit).

**Architecture:** Engine-level config flag. Tournaments inherit the new default automatically through the `_el_farol_for(num_players)` factory in `tournament/service.py:120` (which constructs `ElFarolConfig(num_players=...)` without supplying `scoring_mode`). Tournament `total_score = sum(Action.payoff)`, so changing per-round payoff to `happy` makes tournament totals = sum of happy slots = `t_happy`. Legacy mode is reachable only via direct `ElFarolConfig(scoring_mode="happy_minus_crowded")` calls (tests, atp-games standalone scenarios). No tournament API changes in v1.

**Tech Stack:** Python 3.12, dataclasses, pytest + anyio, Jinja2 (rules docs are markdown).

**Spec:** `docs/superpowers/specs/2026-05-02-el-farol-scoring-modes-design.md`

**Rollout precondition (procedural, NOT enforced by this plan's code):** Before deploying to production, run on prod DB:

```sql
SELECT id, status, game_type, num_players, created_at
FROM tournaments
WHERE status IN ('pending', 'active') AND game_type = 'el_farol';
```

If any rows return, either wait/cancel them or document explicit acceptance of the mid-flight mix in the deploy commit.

---

## File Structure

### Modified files

| Path | Why |
|---|---|
| `game-environments/game_envs/games/el_farol.py` | Add `scoring_mode` field; branch 3 methods + `to_prompt()`; flip default; update module + method docstrings |
| `game-environments/tests/test_el_farol.py` | Tag legacy-formula tests with `scoring_mode="happy_minus_crowded"`; add new happy_only tests |
| `atp-games/tests/test_round_payoffs.py` | Tag tests asserting old divergence between `sum(per_day)` and `t_happy / max(t_crowded, 0.1)` |
| `atp-games/tests/test_runner_action_records.py` | Audit comments + assertions; tag legacy-dependent tests |
| `tests/unit/games/test_el_farol_state_format.py` | Tag duplicated `test_compute_round_payoffs_happy_minus_crowded` |
| `examples/test_suites/13_game_el_farol.yaml` | Audit for baked-in payoff expectations; tag with legacy mode if needed |
| `packages/atp-dashboard/atp/dashboard/v2/game_copy.py` | Update `GameCopy["el_farol"]` text in place to describe new default |
| `demo-game/agents/el_farol_agent.py` | Update explanatory comment to describe new default |
| `docs/games/rules/el-farol-bar.ru.md` | Replace per-round + final scoring sections with new "Scoring modes" section |
| `docs/games/rules/el-farol-bar.en.md` | Same as above, English mirror |
| `participant-kit-el-farol-en/README.md` | Add CHANGELOG entry + minor version bump |

### New tests (added to existing files)

- 9 new unit tests in `game-environments/tests/test_el_farol.py` covering happy_only mode, validation, telemetry parity, property test, round-trip.
- 1 new test wherever `game_copy.py` is tested — verifies the new GameCopy text describes happy_only.

No new files are created.

---

## Task 1: Scaffold `scoring_mode` field with default `happy_minus_crowded` (temporary)

**Files:**
- Modify: `game-environments/game_envs/games/el_farol.py`
- Modify: `game-environments/tests/test_el_farol.py`

**Goal:** Add the `scoring_mode` field to `ElFarolConfig`. Default temporarily to `"happy_minus_crowded"` so all existing behavior is preserved at this stage. The default flips to `"happy_only"` in Task 6 once the methods are branched and tests are tagged.

This task is purely additive — no test should break.

- [ ] **Step 1: Write the validation test (failing)**

Append to `game-environments/tests/test_el_farol.py`:

```python
def test_scoring_mode_validation_rejects_unknown():
    """Unknown scoring_mode value raises ValueError in __post_init__."""
    import pytest
    from game_envs.games.el_farol import ElFarolConfig

    with pytest.raises(ValueError, match="scoring_mode"):
        ElFarolConfig(num_players=4, scoring_mode="bogus")  # type: ignore[arg-type]


def test_scoring_mode_temporary_default_is_legacy():
    """Until Task 6 flips it, the default remains happy_minus_crowded
    so all existing tests keep passing during incremental rollout.
    Task 6 changes this assertion to assert happy_only."""
    from game_envs.games.el_farol import ElFarolConfig

    cfg = ElFarolConfig(num_players=4)
    assert cfg.scoring_mode == "happy_minus_crowded"


def test_scoring_mode_dataclass_round_trip():
    """ElFarolConfig with explicit scoring_mode survives asdict/replace
    round-trip and remains equal."""
    from dataclasses import asdict, replace

    from game_envs.games.el_farol import ElFarolConfig

    original = ElFarolConfig(num_players=4, scoring_mode="happy_only")
    payload = asdict(original)
    rebuilt = ElFarolConfig(**payload)
    assert rebuilt == original
    assert rebuilt.scoring_mode == "happy_only"

    via_replace = replace(original, num_players=8)
    assert via_replace.scoring_mode == "happy_only"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest game-environments/tests/test_el_farol.py::test_scoring_mode_validation_rejects_unknown game-environments/tests/test_el_farol.py::test_scoring_mode_temporary_default_is_legacy game-environments/tests/test_el_farol.py::test_scoring_mode_dataclass_round_trip -v`

Expected: all three FAIL with `TypeError` ("unexpected keyword argument 'scoring_mode'") because the field does not exist yet.

- [ ] **Step 3: Add the field + validation to `ElFarolConfig`**

Open `game-environments/game_envs/games/el_farol.py`. Find the import block at the top (around line 30). Add `Literal` to the typing imports if not already present:

```python
from typing import Any, Literal
```

Locate the `ElFarolConfig` dataclass definition (around line 224). After the existing `slot_duration` field and before `__post_init__`, add:

```python
    slot_duration: float = 0.5  # hours
    # Scoring mode — see docs/games/rules/el-farol-bar.{ru,en}.md.
    # Default is "happy_minus_crowded" during incremental rollout; the
    # final commit flips it to "happy_only".
    scoring_mode: Literal["happy_only", "happy_minus_crowded"] = (
        "happy_minus_crowded"
    )
```

Then in `__post_init__`, add a new validation block at the end (after the existing `slot_duration` check around line 285):

```python
        if self.scoring_mode not in {"happy_only", "happy_minus_crowded"}:
            raise ValueError(
                f"scoring_mode must be 'happy_only' or "
                f"'happy_minus_crowded', got {self.scoring_mode!r}"
            )
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest game-environments/tests/test_el_farol.py::test_scoring_mode_validation_rejects_unknown game-environments/tests/test_el_farol.py::test_scoring_mode_temporary_default_is_legacy game-environments/tests/test_el_farol.py::test_scoring_mode_dataclass_round_trip -v`

Expected: all three PASS.

- [ ] **Step 5: Sanity — full el_farol test suite still green**

Run: `uv run pytest game-environments/tests/test_el_farol.py -q`
Expected: all existing tests still pass (additive change only).

- [ ] **Step 6: Type-check + lint**

Run: `uv run pyrefly check game-environments/game_envs/games/el_farol.py game-environments/tests/test_el_farol.py`
Run: `uv run ruff check game-environments/game_envs/games/el_farol.py game-environments/tests/test_el_farol.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add game-environments/game_envs/games/el_farol.py game-environments/tests/test_el_farol.py
git commit -m "feat(el-farol): scaffold scoring_mode config field"
```

---

## Task 2: Branch `compute_round_payoffs()` on `scoring_mode`

**Files:**
- Modify: `game-environments/game_envs/games/el_farol.py`
- Modify: `game-environments/tests/test_el_farol.py`

**Goal:** Make `compute_round_payoffs()` return `happy` per player under `happy_only` mode and `happy − crowded` under `happy_minus_crowded`. Existing tests use the default (still `happy_minus_crowded` until Task 6) and continue to pass.

- [ ] **Step 1: Write the failing happy_only test**

Append to `game-environments/tests/test_el_farol.py`:

```python
def test_compute_round_payoffs_happy_only_default():
    """In happy_only mode, per-round payoff equals the count of happy
    slots — crowded slots contribute 0, not −1.

    Scenario: 3 players, threshold=3, slot 1 is crowded (3 attendees).
        p0 attends slots 0,1 → slot 0 happy, slot 1 crowded → payoff = 1
        p1 attends slots 1,2 → slot 1 crowded, slot 2 happy → payoff = 1
        p2 attends slots 0,1 → slot 0 happy, slot 1 crowded → payoff = 1
    Under happy_minus_crowded these would all be 0."""
    from game_envs.games.el_farol import ElFarolBar, ElFarolConfig

    cfg = ElFarolConfig(
        num_players=3,
        num_slots=4,
        capacity_threshold=3,
        scoring_mode="happy_only",
    )
    g = ElFarolBar(cfg)
    g.reset()

    actions = {
        0: {"intervals": [[0, 1]]},
        1: {"intervals": [[1, 2]]},
        2: {"intervals": [[0, 1]]},
    }
    payoffs = g.compute_round_payoffs(actions)
    assert payoffs == [1.0, 1.0, 1.0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest game-environments/tests/test_el_farol.py::test_compute_round_payoffs_happy_only_default -v`
Expected: FAIL with `assert [0.0, 0.0, 0.0] == [1.0, 1.0, 1.0]` — current default formula returns `happy - crowded = 0` for each.

- [ ] **Step 3: Branch `compute_round_payoffs()` in the engine**

Open `game-environments/game_envs/games/el_farol.py`. Find `compute_round_payoffs` (around line 512). Locate the final loop body that sets `payoffs[p_idx]`:

```python
        payoffs: list[float] = [0.0] * n
        for p_idx, slots in enumerate(per_player_slots):
            happy = 0
            crowded_count = 0
            for s in slots:
                if s in crowded:
                    crowded_count += 1
                else:
                    happy += 1
            payoffs[p_idx] = float(happy - crowded_count)
        return payoffs
```

Replace the assignment line with:

```python
        payoffs: list[float] = [0.0] * n
        mode = self._ef_config.scoring_mode
        for p_idx, slots in enumerate(per_player_slots):
            happy = 0
            crowded_count = 0
            for s in slots:
                if s in crowded:
                    crowded_count += 1
                else:
                    happy += 1
            if mode == "happy_only":
                payoffs[p_idx] = float(happy)
            else:  # happy_minus_crowded
                payoffs[p_idx] = float(happy - crowded_count)
        return payoffs
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `uv run pytest game-environments/tests/test_el_farol.py::test_compute_round_payoffs_happy_only_default -v`
Expected: PASS.

- [ ] **Step 5: Sanity — existing legacy test still passes**

Run: `uv run pytest game-environments/tests/test_el_farol.py::test_compute_round_payoffs_happy_minus_crowded -v`
Expected: PASS (existing test uses default config = legacy mode).

- [ ] **Step 6: Type-check + lint**

Run: `uv run pyrefly check game-environments/game_envs/games/el_farol.py game-environments/tests/test_el_farol.py`
Run: `uv run ruff check game-environments/game_envs/games/el_farol.py game-environments/tests/test_el_farol.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add game-environments/game_envs/games/el_farol.py game-environments/tests/test_el_farol.py
git commit -m "feat(el-farol): branch compute_round_payoffs on scoring_mode"
```

---

## Task 3: Branch `step()` on `scoring_mode`

**Files:**
- Modify: `game-environments/game_envs/games/el_farol.py`
- Modify: `game-environments/tests/test_el_farol.py`

**Goal:** Make `step()` return per-round payoffs that match the active mode. `_t_happy` and `_t_crowded` are accumulated unconditionally (telemetry parity).

- [ ] **Step 1: Write the failing test**

Append to `game-environments/tests/test_el_farol.py`:

```python
def test_step_happy_only_accumulates_t_happy_and_t_crowded():
    """In happy_only mode, both _t_happy and _t_crowded must accumulate
    even though _t_crowded is not in the payoff formula. This keeps
    observation telemetry (your_t_crowded_slots) consistent across modes."""
    from game_envs.games.el_farol import ElFarolBar, ElFarolConfig

    cfg = ElFarolConfig(
        num_players=3,
        num_slots=4,
        capacity_threshold=3,
        num_rounds=1,
        scoring_mode="happy_only",
    )
    g = ElFarolBar(cfg)
    g.reset()

    # Same scenario as compute_round_payoffs test:
    # slot 1 crowded (3 attendees), slots 0,2 happy.
    # p0: slot 0 happy + slot 1 crowded → t_happy[p0]=1, t_crowded[p0]=1
    actions = {
        0: {"intervals": [[0, 1]]},
        1: {"intervals": [[1, 2]]},
        2: {"intervals": [[0, 1]]},
    }
    result = g.step(actions)

    # Per-round payoff = happy (not happy - crowded).
    assert result.payoffs["player_0"] == 1.0
    # _t_crowded accumulated despite not being in the formula.
    assert g._t_crowded["player_0"] == 1.0
    assert g._t_happy["player_0"] == 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest game-environments/tests/test_el_farol.py::test_step_happy_only_accumulates_t_happy_and_t_crowded -v`
Expected: FAIL on the payoff assertion (`step()` still returns `happy - crowded = 0`).

- [ ] **Step 3: Branch `step()` in the engine**

Open `game-environments/game_envs/games/el_farol.py`. Find the `step()` method (around line 593). Locate the per-player loop that accumulates `_t_happy` / `_t_crowded` and computes `payoffs[pid]` (around line 624):

```python
        payoffs: dict[str, float] = {}
        for pid in self.player_ids:
            slots = clean[pid]
            happy = sum(1 for s in slots if daily_occupancy[s] < threshold)
            crowded = sum(1 for s in slots if daily_occupancy[s] >= threshold)
            self._t_happy[pid] += happy
            self._t_crowded[pid] += crowded
            payoffs[pid] = float(happy - crowded)
```

Replace with:

```python
        payoffs: dict[str, float] = {}
        mode = self._ef_config.scoring_mode
        for pid in self.player_ids:
            slots = clean[pid]
            happy = sum(1 for s in slots if daily_occupancy[s] < threshold)
            crowded = sum(1 for s in slots if daily_occupancy[s] >= threshold)
            # Accumulate both counters unconditionally — observation
            # carries `your_t_crowded_slots` regardless of mode, so the
            # telemetry surface stays consistent.
            self._t_happy[pid] += happy
            self._t_crowded[pid] += crowded
            if mode == "happy_only":
                payoffs[pid] = float(happy)
            else:  # happy_minus_crowded
                payoffs[pid] = float(happy - crowded)
```

- [ ] **Step 4: Run the new test**

Run: `uv run pytest game-environments/tests/test_el_farol.py::test_step_happy_only_accumulates_t_happy_and_t_crowded -v`
Expected: PASS.

- [ ] **Step 5: Sanity — existing step tests still green**

Run: `uv run pytest game-environments/tests/test_el_farol.py -k "step" -v`
Expected: all pass.

- [ ] **Step 6: Type-check + lint**

Run: `uv run pyrefly check game-environments/game_envs/games/el_farol.py`
Run: `uv run ruff check game-environments/game_envs/games/el_farol.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add game-environments/game_envs/games/el_farol.py game-environments/tests/test_el_farol.py
git commit -m "feat(el-farol): branch step() on scoring_mode; keep t_crowded telemetry"
```

---

## Task 4: Branch `get_payoffs()` on `scoring_mode`

**Files:**
- Modify: `game-environments/game_envs/games/el_farol.py`
- Modify: `game-environments/tests/test_el_farol.py`

**Goal:** Make `get_payoffs()` return `t_happy` under `happy_only` and `t_happy / max(t_crowded, 0.1)` under `happy_minus_crowded`. `min_total_hours` disqualification applies in both modes (existing behavior).

- [ ] **Step 1: Write the failing tests**

Append to `game-environments/tests/test_el_farol.py`:

```python
def test_get_payoffs_happy_only_returns_t_happy_sum():
    """In happy_only mode, get_payoffs() returns t_happy directly
    (count, not ratio). Crowded slots have no influence on the final."""
    from game_envs.games.el_farol import ElFarolBar, ElFarolConfig

    cfg = ElFarolConfig(
        num_players=2,
        num_slots=4,
        capacity_threshold=3,
        scoring_mode="happy_only",
    )
    g = ElFarolBar(cfg)
    g.reset()

    # Seed counters directly — bypass step() so the test isolates
    # get_payoffs() formula from the per-round path.
    g._t_happy["player_0"] = 5.0
    g._t_crowded["player_0"] = 3.0
    g._t_happy["player_1"] = 2.0
    g._t_crowded["player_1"] = 4.0

    final = g.get_payoffs()
    # happy_only: final = t_happy (regardless of t_crowded)
    assert final["player_0"] == 5.0
    assert final["player_1"] == 2.0


def test_get_payoffs_happy_only_disqualification_still_applies():
    """min_total_hours disqualification applies before the formula in
    both modes."""
    from game_envs.games.el_farol import ElFarolBar, ElFarolConfig

    cfg = ElFarolConfig(
        num_players=1,
        num_slots=4,
        capacity_threshold=2,
        slot_duration=0.5,
        min_total_hours=10.0,  # require 10h, will not be met
        scoring_mode="happy_only",
    )
    g = ElFarolBar(cfg)
    g.reset()

    # Attended only 1 slot (0.5 h) — well under 10 h threshold.
    g._t_happy["player_0"] = 1.0
    g._t_crowded["player_0"] = 0.0

    final = g.get_payoffs()
    assert final["player_0"] == 0.0  # disqualified


def test_happy_only_per_round_payoff_is_non_negative():
    """Property test: under happy_only, every per-round payoff ≥ 0
    regardless of action choices. Under legacy this could be negative."""
    import random

    from game_envs.games.el_farol import ElFarolBar, ElFarolConfig

    cfg = ElFarolConfig(
        num_players=4,
        num_slots=8,
        capacity_threshold=2,
        max_intervals=2,
        max_total_slots=4,
        scoring_mode="happy_only",
    )
    g = ElFarolBar(cfg)
    g.reset()
    rng = random.Random(42)

    for _ in range(100):
        actions: dict[int, dict[str, list[list[int]]]] = {}
        for p_idx in range(cfg.num_players):
            start = rng.randint(0, cfg.num_slots - 2)
            end = rng.randint(start, min(start + 2, cfg.num_slots - 1))
            actions[p_idx] = {"intervals": [[start, end]]}
        payoffs = g.compute_round_payoffs(actions)
        for p_idx, p in enumerate(payoffs):
            assert p >= 0.0, (
                f"happy_only payoff must be ≥ 0, got {p} for player {p_idx}"
            )


def test_observation_includes_t_crowded_in_both_modes():
    """observe(player_id) returns your_t_crowded_slots regardless of
    scoring_mode — telemetry is mode-independent."""
    from game_envs.games.el_farol import ElFarolBar, ElFarolConfig

    for mode in ("happy_only", "happy_minus_crowded"):
        cfg = ElFarolConfig(
            num_players=2,
            num_slots=4,
            capacity_threshold=3,
            scoring_mode=mode,  # type: ignore[arg-type]
        )
        g = ElFarolBar(cfg)
        g.reset()
        obs = g.observe("player_0")
        assert "your_t_crowded_slots" in obs.game_state, (
            f"mode={mode} missing your_t_crowded_slots"
        )
```

- [ ] **Step 2: Run the new tests**

Run: `uv run pytest game-environments/tests/test_el_farol.py::test_get_payoffs_happy_only_returns_t_happy_sum game-environments/tests/test_el_farol.py::test_get_payoffs_happy_only_disqualification_still_applies game-environments/tests/test_el_farol.py::test_happy_only_per_round_payoff_is_non_negative game-environments/tests/test_el_farol.py::test_observation_includes_t_crowded_in_both_modes -v`

Expected:
- `test_get_payoffs_happy_only_returns_t_happy_sum` FAILS — current formula is ratio (`5.0 / max(3.0, 0.1) = 1.666…`).
- The other three may PASS or FAIL depending on whether the per-round path still has the negative-payoff bug. Run them; if they pass, great; if not, the impl in Step 3 will fix them.

- [ ] **Step 3: Branch `get_payoffs()` in the engine**

Open `game-environments/game_envs/games/el_farol.py`. Find `get_payoffs` (around line 677):

```python
    def get_payoffs(self) -> dict[str, float]:
        """Compute final payoffs.

        Returns:
            score = t_happy / max(t_crowded, 0.1) per player.
            Players who attended fewer than min_total_hours hours receive 0.
        """
        c = self._ef_config
        result: dict[str, float] = {}
        for pid in self.player_ids:
            th = self._t_happy[pid]
            tc = self._t_crowded[pid]
            total_hours = (th + tc) * c.slot_duration
            if total_hours < c.min_total_hours:
                result[pid] = 0.0  # disqualified
            else:
                result[pid] = th / max(tc, 0.1)
        return result
```

Replace the body with:

```python
    def get_payoffs(self) -> dict[str, float]:
        """Compute final payoffs.

        Returns one float per player. The formula depends on
        ``ElFarolConfig.scoring_mode``:

        - ``happy_only`` (default): ``result = t_happy`` (count of
          happy slots accumulated across all rounds).
        - ``happy_minus_crowded`` (legacy): ``result = t_happy /
          max(t_crowded, 0.1)`` (ratio).

        Both modes apply ``min_total_hours`` disqualification first —
        a player who attended fewer hours than the threshold gets 0
        regardless of the formula. The ``max(t_crowded, 0.1)`` floor
        in legacy mode prevents division by zero.
        """
        c = self._ef_config
        result: dict[str, float] = {}
        for pid in self.player_ids:
            th = self._t_happy[pid]
            tc = self._t_crowded[pid]
            total_hours = (th + tc) * c.slot_duration
            if total_hours < c.min_total_hours:
                result[pid] = 0.0  # disqualified
            elif c.scoring_mode == "happy_only":
                result[pid] = th
            else:  # happy_minus_crowded
                result[pid] = th / max(tc, 0.1)
        return result
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest game-environments/tests/test_el_farol.py::test_get_payoffs_happy_only_returns_t_happy_sum game-environments/tests/test_el_farol.py::test_get_payoffs_happy_only_disqualification_still_applies game-environments/tests/test_el_farol.py::test_happy_only_per_round_payoff_is_non_negative game-environments/tests/test_el_farol.py::test_observation_includes_t_crowded_in_both_modes -v`

Expected: all 4 PASS.

- [ ] **Step 5: Sanity — full el_farol test suite**

Run: `uv run pytest game-environments/tests/test_el_farol.py -q`
Expected: all pass — legacy tests still use default config (still `happy_minus_crowded` until Task 6).

- [ ] **Step 6: Type-check + lint**

Run: `uv run pyrefly check game-environments/game_envs/games/el_farol.py`
Run: `uv run ruff check game-environments/`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add game-environments/game_envs/games/el_farol.py game-environments/tests/test_el_farol.py
git commit -m "feat(el-farol): branch get_payoffs on scoring_mode + 4 happy_only tests"
```

---

## Task 5: Mode-aware `to_prompt()`

**Files:**
- Modify: `game-environments/game_envs/games/el_farol.py`
- Modify: `game-environments/tests/test_el_farol.py`

**Goal:** Make `to_prompt()` (the LLM-facing rules description) describe the active mode's formula. The prompt branch is mandatory — leaving it as "always-legacy" would create tech debt for the future tournament-level configurability work.

- [ ] **Step 1: Write the failing tests**

Append to `game-environments/tests/test_el_farol.py`:

```python
def test_to_prompt_happy_only_describes_no_penalty():
    """to_prompt() for happy_only mode must describe the new scoring
    rule (no penalty for crowded), not the legacy ratio formula."""
    from game_envs.games.el_farol import ElFarolBar, ElFarolConfig

    cfg = ElFarolConfig(
        num_players=4,
        num_slots=8,
        capacity_threshold=2,
        scoring_mode="happy_only",
    )
    g = ElFarolBar(cfg)
    prompt = g.to_prompt()

    assert "happy" in prompt.lower()
    # New default: no ratio formula, no negative penalty mention.
    assert "t_happy / max" not in prompt
    assert "/ max(total_crowded_slots" not in prompt
    # Should mention that crowded slots give 0 / no penalty.
    assert "no penalty" in prompt.lower() or "0" in prompt


def test_to_prompt_legacy_describes_ratio_formula():
    """to_prompt() for happy_minus_crowded mode must describe the
    legacy ratio formula."""
    from game_envs.games.el_farol import ElFarolBar, ElFarolConfig

    cfg = ElFarolConfig(
        num_players=4,
        num_slots=8,
        capacity_threshold=2,
        scoring_mode="happy_minus_crowded",
    )
    g = ElFarolBar(cfg)
    prompt = g.to_prompt()

    assert "happy" in prompt.lower()
    # Legacy: explicit ratio with the 0.1 floor.
    assert "max(total_crowded_slots, 0.1)" in prompt or "max(t_crowded, 0.1)" in prompt
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest game-environments/tests/test_el_farol.py::test_to_prompt_happy_only_describes_no_penalty game-environments/tests/test_el_farol.py::test_to_prompt_legacy_describes_ratio_formula -v`

Expected:
- `test_to_prompt_legacy_describes_ratio_formula` PASSES (current to_prompt always uses ratio formula text).
- `test_to_prompt_happy_only_describes_no_penalty` FAILS (current text contains the ratio formula even when scoring_mode=happy_only).

- [ ] **Step 3: Make `to_prompt()` mode-aware**

Open `game-environments/game_envs/games/el_farol.py`. Find `to_prompt()` (around line 744). Replace the entire method:

```python
    def to_prompt(self) -> str:
        """Describe the El Farol scenario for LLM agents.

        Branches on ``ElFarolConfig.scoring_mode`` so the explanation
        matches the active formula. See ``get_payoffs()`` for the
        mode-by-mode contract.
        """
        c = self._ef_config
        slot_hours = c.num_slots * c.slot_duration
        if c.scoring_mode == "happy_only":
            scoring_block = [
                "Scoring:",
                "  Each happy slot you attend = +1 (no penalty for crowded).",
                "  round_score = number of happy slots that day.",
                "  final_score = total happy slots across all days.",
                f"  (must attend >= {c.min_total_hours} h to avoid disqualification)",
            ]
        else:  # happy_minus_crowded
            scoring_block = [
                "Scoring:",
                "  Each happy slot = +1, each crowded slot = −1.",
                "  round_score = happy slots − crowded slots.",
                "  final_score = total_happy_slots / max(total_crowded_slots, 0.1)",
                f"  (must attend >= {c.min_total_hours} h to avoid disqualification)",
            ]

        return "\n".join(
            [
                f"This is the El Farol Bar Problem with {c.num_players} players.",
                "",
                "Rules:",
                f"- Each day has {c.num_slots} time slots "
                f"of {c.slot_duration:.1f} h each "
                f"({slot_hours:.0f} h total).",
                "- You choose which slots to attend as up to "
                f"{c.max_intervals} contiguous interval(s): "
                f'{{"intervals": [[start, end], ...]}} with inclusive '
                f"indices in 0–{c.num_slots - 1}, at most "
                f"{c.max_total_slots} slots total per day, intervals "
                "non-overlapping and non-adjacent.",
                f"- If {c.capacity_threshold}+ players attend "
                "a slot, it becomes *crowded*.",
                "- You can only observe past attendance — not what others plan today.",
                "",
                *scoring_block,
                "",
                "Strategy note:",
                "  The Nash equilibrium has each player "
                "attend each slot with probability"
                f"  p* = {c.capacity_threshold}/{c.num_players} = "
                f"  {c.capacity_threshold / c.num_players:.2f}. "
                "  Heterogeneous learning strategies often outperform this.",
                "",
                f"This game is repeated for {c.num_rounds} days.",
            ]
        )
```

- [ ] **Step 4: Run the new tests**

Run: `uv run pytest game-environments/tests/test_el_farol.py::test_to_prompt_happy_only_describes_no_penalty game-environments/tests/test_el_farol.py::test_to_prompt_legacy_describes_ratio_formula -v`
Expected: both PASS.

- [ ] **Step 5: Sanity — full el_farol suite**

Run: `uv run pytest game-environments/tests/test_el_farol.py -q`
Expected: all pass.

- [ ] **Step 6: Type-check + lint**

Run: `uv run pyrefly check game-environments/game_envs/games/el_farol.py`
Run: `uv run ruff check game-environments/`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add game-environments/game_envs/games/el_farol.py game-environments/tests/test_el_farol.py
git commit -m "feat(el-farol): mode-aware to_prompt() + 2 prompt tests"
```

---

## Task 6: Flip default to `happy_only` + tag legacy tests + update docstrings

**Files:**
- Modify: `game-environments/game_envs/games/el_farol.py`
- Modify: `game-environments/tests/test_el_farol.py`
- Modify: `atp-games/tests/test_round_payoffs.py`
- Modify: `atp-games/tests/test_runner_action_records.py`
- Modify: `tests/unit/games/test_el_farol_state_format.py`
- Audit: `examples/test_suites/13_game_el_farol.yaml`

**Goal:** Flip the engine default from `happy_minus_crowded` to `happy_only`. This is the breaking change. All tests that depend on the legacy formula get tagged with `scoring_mode="happy_minus_crowded"` to preserve their assertions as legacy regression coverage. Module + method docstrings are updated to describe the new default.

- [ ] **Step 1: Update the temporary-default test to assert the new default**

In `game-environments/tests/test_el_farol.py`, find `test_scoring_mode_temporary_default_is_legacy` (added in Task 1) and rename + update:

```python
def test_scoring_mode_default_is_happy_only():
    """The default scoring_mode is happy_only — the new platform-wide
    rule. Tournament total_score will be sum of per-round happy values."""
    from game_envs.games.el_farol import ElFarolConfig

    cfg = ElFarolConfig(num_players=4)
    assert cfg.scoring_mode == "happy_only"
```

- [ ] **Step 2: Flip the default in `ElFarolConfig`**

Open `game-environments/game_envs/games/el_farol.py`. Find the `scoring_mode` field added in Task 1:

```python
    scoring_mode: Literal["happy_only", "happy_minus_crowded"] = (
        "happy_minus_crowded"
    )
```

Replace with:

```python
    scoring_mode: Literal["happy_only", "happy_minus_crowded"] = "happy_only"
```

Also remove the "during incremental rollout" comment line; replace with:

```python
    # Scoring mode — see docs/games/rules/el-farol-bar.{ru,en}.md.
    # ``happy_only`` (default) gives +1 per happy slot, 0 per crowded;
    # ``happy_minus_crowded`` is a legacy opt-in for tests and
    # standalone scenarios that depend on the old formula.
    scoring_mode: Literal["happy_only", "happy_minus_crowded"] = "happy_only"
```

- [ ] **Step 3: Update the module docstring**

In `game-environments/game_envs/games/el_farol.py`, replace the existing module docstring (lines 1-23) with:

```python
"""El Farol Bar Problem — N-player minority / congestion game.

Brian Arthur (1994) showed that a population of heterogeneous agents
with bounded rationality can self-organise around a bar's capacity
threshold without any explicit coordination.

Game structure:
  - N players decide independently which time-slots to attend.
  - If attendance at a slot >= threshold, the slot is *crowded* (bad).
  - Players want to maximise time in non-crowded slots.

One *round* in the framework corresponds to one *day* in the original
simulation. Each player submits a list of slot indices (0 to
num_slots-1) they plan to attend. The game runs for num_rounds days.

Scoring is mode-dependent (see ``ElFarolConfig.scoring_mode``):
  - ``happy_only`` (default): per-round payoff = number of happy
    slots that day. Final ``get_payoffs()`` = ``t_happy``. No penalty
    for crowded slots.
  - ``happy_minus_crowded`` (legacy, opt-in): per-round payoff =
    ``happy − crowded``. Final ``get_payoffs()`` = ``t_happy /
    max(t_crowded, 0.1)``.

Both modes apply ``min_total_hours`` disqualification in
``get_payoffs()``: players who attended fewer than the configured
hours receive 0 regardless of the formula. ``_t_crowded`` is
accumulated and surfaced in observation in both modes.
"""
```

- [ ] **Step 4: Update `compute_round_payoffs` docstring**

Find `compute_round_payoffs` (around line 512). Replace the docstring:

```python
    def compute_round_payoffs(self, actions: dict[int, dict[str, Any]]) -> list[float]:
        """Per-round payoff per player for one round.

        Branches on ``ElFarolConfig.scoring_mode``:
          - ``happy_only`` (default): ``payoff = number of happy slots``.
          - ``happy_minus_crowded``: ``payoff = happy − crowded``.

        Args:
            actions: participant_idx -> ``{"intervals": [[start, end], ...]}``.

        Returns:
            List of per-round payoffs in participant_idx order.
        """
```

- [ ] **Step 5: Tag legacy tests in `game-environments/tests/test_el_farol.py`**

Find `test_compute_round_payoffs_happy_minus_crowded`. Rename and add the legacy mode to its config:

```python
def test_compute_round_payoffs_legacy_mode_happy_minus_crowded():
    """Regression: under explicit happy_minus_crowded mode the per-round
    payoff is `happy − crowded`. Default mode is now happy_only — see
    `test_compute_round_payoffs_happy_only_default`."""
    # ... existing setup ...
    cfg = ElFarolConfig(
        num_players=3,
        num_slots=4,
        capacity_threshold=3,
        scoring_mode="happy_minus_crowded",  # <-- ADD THIS
    )
    # ... rest unchanged ...
```

The exact original test body (around line 141 / 182 — find by name) becomes:

```python
def test_compute_round_payoffs_legacy_mode_happy_minus_crowded():
    """Regression: under explicit happy_minus_crowded mode the per-round
    payoff is `happy − crowded`. Default mode is now happy_only — see
    `test_compute_round_payoffs_happy_only_default`."""
    cfg = ElFarolConfig(
        num_players=3,
        num_slots=4,
        capacity_threshold=3,
        scoring_mode="happy_minus_crowded",
    )
    g = ElFarolBar(cfg)
    g.reset()
    actions = {
        0: {"intervals": [[0, 1]]},
        1: {"intervals": [[1, 2]]},
        2: {"intervals": [[0, 1]]},
    }
    payoffs = g.compute_round_payoffs(actions)
    # capacity_threshold=3 → slot 1 is crowded (3 attendees), others happy
    # p0: slot 0 happy, slot 1 crowded → 1-1=0
    # p1: slot 1 crowded, slot 2 happy → 1-1=0
    # p2: slot 0 happy, slot 1 crowded → 1-1=0
    assert payoffs == [0.0, 0.0, 0.0]
```

If the existing test body differs from this scenario, preserve the original logic but ADD `scoring_mode="happy_minus_crowded"` to the config call. The test must continue to verify the same legacy behavior.

For any other test in this file that touches per-round or final payoff values that depend on the legacy formula (e.g. `test_get_payoffs_*` not in the new happy_only set), add `scoring_mode="happy_minus_crowded"` to its config.

Search the file for the patterns `t_happy / max`, `happy - crowded` (in assertions), and update each test accordingly.

- [ ] **Step 6: Tag legacy tests in `atp-games/tests/test_round_payoffs.py`**

Open `atp-games/tests/test_round_payoffs.py`. The file's docstring (lines 1-21) explicitly notes that `El Farol's get_payoffs() uses a non-linear t_happy / max(t_crowded, 0.1) formula and therefore intentionally diverges from the per-day sum`.

Update the docstring to reflect the new default:

```python
"""Phase 4 TDD tests: GameRunner populates EpisodeResult.round_payoffs.

Phase 4 adds a new ``round_payoffs: list[dict[str, float]]`` field to
``EpisodeResult``. The runner appends one dict per resolved day, where
each dict maps ``player_id -> per_day_payoff`` and ``per_day_payoff``
equals ``step_result.payoffs[player_id]`` from the underlying game step.

Invariant note
--------------
The plan's invariant ``sum(round_payoffs[*][pid]) == ep.payoffs[pid]``
holds for games whose ``get_payoffs()`` is a cumulative sum of
per-day payoffs (e.g. Prisoner's Dilemma, El Farol in the new
``happy_only`` mode where final = t_happy = sum of per-day happy
counts). It does NOT hold for El Farol in legacy
``happy_minus_crowded`` mode, where ``get_payoffs()`` uses a
non-linear ``t_happy / max(t_crowded, 0.1)`` ratio. One test below
asserts the legacy divergence explicitly under
``scoring_mode="happy_minus_crowded"``.
"""
```

Then in the test that asserts the divergence (search for a test name like `test_*divergence*` or one that compares `sum(round_payoffs)` to `ep.payoffs`), add `scoring_mode="happy_minus_crowded"` to the `ElFarolConfig(...)` call. The test's behaviour must be preserved.

For any other tests in the same file that pass `ElFarolConfig(...)` and assert on legacy formulas, add the legacy tag the same way. If a test does NOT depend on the formula (e.g. only checks structure), leave it alone — it will continue to work under the new default.

- [ ] **Step 7: Audit `atp-games/tests/test_runner_action_records.py`**

Open the file. Search for the comment `each player has happy=3, crowded=0` (around line 155) and any assertions on payoff values. If any tests depend on `happy − crowded` semantics (e.g. assert payoff == happy - crowded), tag those configs with `scoring_mode="happy_minus_crowded"`.

If the file's tests check structure / record shape only without asserting payoff math, no changes needed — just verify by running the file:

Run: `uv run pytest atp-games/tests/test_runner_action_records.py -q`
Expected: all pass after Task 6 lands. If failures appear, the failing tests need the legacy tag.

- [ ] **Step 8: Tag the duplicated test in `tests/unit/games/test_el_farol_state_format.py`**

Open the file. Find `test_compute_round_payoffs_happy_minus_crowded` (around line 141). Apply the same rename + tag treatment as in Step 5:

```python
def test_compute_round_payoffs_legacy_mode_happy_minus_crowded():
    # ... ElFarolConfig(..., scoring_mode="happy_minus_crowded") ...
    # rest of the test body unchanged
```

- [ ] **Step 9: Audit `examples/test_suites/13_game_el_farol.yaml`**

Open the YAML. Look for any payoff-related fields that might be hardcoded (e.g. expected scores in evaluator config). The El Farol YAML suite doesn't typically encode payoff math at this level (it sets capacity, players, agents), but verify.

If the YAML sets a `scoring_mode` field already, leave it. If it doesn't and you find evaluator assertions that depend on the legacy formula, add `scoring_mode: happy_minus_crowded` under the el_farol config block.

If the YAML test-suite runs cleanly under the new default, no change is needed. Document the audit outcome in the commit message.

- [ ] **Step 10: Run the full affected test surface**

Run: `uv run pytest game-environments/tests/test_el_farol.py atp-games/tests/test_round_payoffs.py atp-games/tests/test_runner_action_records.py atp-games/tests/test_el_farol_action_space.py atp-games/tests/test_el_farol_config.py atp-games/tests/test_el_farol_interval_agents.py tests/unit/games/test_el_farol_state_format.py -q`

Expected: all pass. If any fail, investigate the failure: either the test depends on legacy semantics (add `scoring_mode="happy_minus_crowded"`) or it depends on the new semantics (leave alone — confirm).

- [ ] **Step 11: Type-check + lint**

Run: `uv run pyrefly check game-environments/ atp-games/ tests/unit/games/`
Run: `uv run ruff check game-environments/ atp-games/ tests/unit/games/`
Expected: clean.

- [ ] **Step 12: Commit**

```bash
git add game-environments/game_envs/games/el_farol.py game-environments/tests/test_el_farol.py atp-games/tests/test_round_payoffs.py atp-games/tests/test_runner_action_records.py tests/unit/games/test_el_farol_state_format.py examples/test_suites/13_game_el_farol.yaml
git commit -m "feat(el-farol): flip default to happy_only; tag legacy tests"
```

---

## Task 7: Update dashboard `GameCopy["el_farol"]` + dashboard test

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/game_copy.py`
- Modify or Create: dashboard test for `game_copy.py`

**Goal:** Update the dashboard's static "About this game" copy to describe the new default `happy_only` mode. The page is a generic public game-detail page (`/ui/games/{game}`), not tournament-specific, so wording must accommodate both standalone and tournament framings without overpromising.

- [ ] **Step 1: Locate the dashboard test for `game_copy.py`**

Run: `grep -rln "game_copy\|GAME_COPY\|GameCopy.*el_farol" packages/atp-dashboard/tests/ tests/ 2>/dev/null`

If a test file exists for `game_copy.py`, work in that file. If not, create a new file at `tests/unit/dashboard/test_game_copy_el_farol.py`.

- [ ] **Step 2: Write the failing test**

In the located/new test file:

```python
"""Tests for GameCopy['el_farol'] static dashboard copy."""

from __future__ import annotations


def test_game_copy_el_farol_describes_happy_only():
    """The dashboard 'About' copy for El Farol must describe the new
    default scoring (happy_only): +1 per happy slot, 0 per crowded
    (no penalty), round score = number of happy slots."""
    from atp.dashboard.v2.game_copy import GAME_COPY

    el = GAME_COPY["el_farol"]

    # Tagline + setup must not promise -1 for crowded.
    assert "−1" not in el.tagline
    assert "no penalty" in el.setup.lower() or "0 if attendance" in el.setup

    # Rules and payoff_formula describe happy_only.
    rules_text = " ".join(el.rules)
    assert "−1" not in rules_text
    assert "happy" in rules_text.lower()
    assert "−1" not in el.payoff_formula
    assert "happy slots" in el.payoff_formula.lower()
```

(If the existing test file uses `GAME_COPY` under a different name, adjust the import. Verify with the grep from Step 1.)

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest <path-to-test-file> -v`
Expected: FAIL on the `−1` assertions because the current copy mentions "+1 / −1".

- [ ] **Step 4: Update `GameCopy["el_farol"]` in `game_copy.py`**

Open `packages/atp-dashboard/atp/dashboard/v2/game_copy.py`. Find the `"el_farol": GameCopy(...)` block (around line 250). Apply these replacements within that block:

In `tagline` (line 252-255), no change needed — the existing tagline doesn't mention −1.

In `setup` (line 256-265), replace:
```python
            "platform divides each night into 16 time slots; each "
            "player picks up to 8 slots to attend. A slot pays +1 when "
            "its attendance stays strictly below the capacity threshold, "
            "and −1 once attendance reaches or exceeds it."
```
with:
```python
            "platform divides each night into 16 time slots; each "
            "player picks up to 8 slots to attend. A slot pays +1 when "
            "its attendance stays strictly below the capacity threshold, "
            "and 0 if attendance reaches or exceeds it (no penalty)."
```

In `rules` (line 275-284), replace the third and fourth entries:
```python
            "For each slot you attend, you get +1 if attendance is "
            "strictly below the capacity threshold (happy), or −1 if "
            "attendance reaches or exceeds the threshold (crowded).",
            "Your round payoff is (happy slots) − (crowded slots) across "
            "the slots you picked.",
```
with:
```python
            "For each slot you attend, you get +1 if attendance is "
            "strictly below the capacity threshold (happy), or 0 if "
            "attendance reaches or exceeds the threshold (crowded). "
            "There is no penalty for crowded slots.",
            "Your round payoff is the number of happy slots you "
            "picked.",
```

In `payoff_formula` (line 293-298), replace:
```python
        payoff_formula=(
            "For each slot s you attend, payoff is +1 if "
            "attendance(s) < capacity_threshold (happy), −1 if "
            "attendance(s) ≥ capacity_threshold (crowded). "
            "Round total = (happy slots) − (crowded slots)."
        ),
```
with:
```python
        payoff_formula=(
            "For each slot s you attend, payoff is +1 if "
            "attendance(s) < capacity_threshold (happy), 0 if "
            "attendance(s) ≥ capacity_threshold (crowded — no "
            "penalty). Round total = number of happy slots. In "
            "tournament play, the displayed score is the sum of "
            "round totals across all rounds."
        ),
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest <path-to-test-file> -v`
Expected: PASS.

- [ ] **Step 6: Sanity — existing dashboard tests still green**

Run: `uv run pytest packages/atp-dashboard/tests/ tests/integration/dashboard/ tests/unit/dashboard/ -q --tb=line -x`
Expected: green. If any test asserts on the old `−1` text, update its assertion to match the new copy (the assertion is now wrong, not the new copy).

- [ ] **Step 7: Type-check + lint**

Run: `uv run pyrefly check packages/atp-dashboard/atp/dashboard/v2/game_copy.py`
Run: `uv run ruff check packages/atp-dashboard/atp/dashboard/v2/game_copy.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/game_copy.py <path-to-test-file>
git commit -m "feat(el-farol): dashboard copy describes happy_only default"
```

---

## Task 8: Update demo agent comment + rules docs (RU + EN)

**Files:**
- Modify: `demo-game/agents/el_farol_agent.py`
- Modify: `docs/games/rules/el-farol-bar.ru.md`
- Modify: `docs/games/rules/el-farol-bar.en.md`

**Goal:** Update the local demo agent's explanatory comment to describe the new default. Replace the rules docs' "Per-round payoff" + "Final payoffs" sections with a single new "Scoring modes" section in both Russian and English.

- [ ] **Step 1: Update the demo agent comment**

Open `demo-game/agents/el_farol_agent.py`. Find the existing comment (around line 40) that reads:

```
If too many people attend the same slot, it becomes crowded (bad).
You want to maximize happy (non-crowded) slots and minimize crowded ones.
```

Replace with:

```
If too many people attend the same slot, it becomes crowded.
You earn +1 for each happy (non-crowded) slot you attend; crowded
slots give 0 (no penalty). Maximize the number of happy slots —
final score = total happy slots across all days.

Note: this assumes the engine's default scoring_mode = "happy_only".
A legacy "happy_minus_crowded" mode is available via
ElFarolConfig(scoring_mode=...) for tests; under that mode crowded
slots score −1 and the final score uses a ratio formula.
```

Preserve the surrounding code structure (function/class docstring etc.).

- [ ] **Step 2: Replace the scoring sections in `docs/games/rules/el-farol-bar.ru.md`**

Open the file. Find the existing "## Выплата за день (per-round payoff)" section (around line 51) and the "## Финальный счёт (`get_payoffs`)" section (around line 64). Replace BOTH sections (everything between them, plus their headings, ending before the next "##") with this single block:

```markdown
## Режимы подсчёта (`scoring_mode`)

Игра поддерживает два режима подсчёта очков. По умолчанию активен
`happy_only`. Турниры всегда используют дефолт — opt-in legacy режима
из API турниров в v1 не предусмотрен.

### `happy_only` (по умолчанию) — простое накопление

- **За каждый happy слот**: +1 балл.
- **За crowded слот**: 0 баллов (без штрафа).
- **Per-day payoff** (возвращается `step()` и `compute_round_payoffs()`):
  число happy слотов в этот день.
- **Финал `get_payoffs()`**: `t_happy` (сумма happy слотов по всем
  дням), при условии что игрок прошёл порог `min_total_hours`.
- **Tournament total_score** = сумма per-day payoff'ов = `t_happy`.

### `happy_minus_crowded` (legacy, opt-in)

- Доступен только через явный `ElFarolConfig(scoring_mode=...)`
  (тесты, atp-games standalone сценарии). В турнирах не плумбится.
- **За happy**: +1, **за crowded**: −1.
- **Per-day payoff**: `happy − crowded`.
- **Финал `get_payoffs()`**: `t_happy / max(t_crowded, 0.1)` при
  условии прохождения `min_total_hours`.

В обоих режимах:
- `_t_crowded` накапливается и видно игроку как
  `your_t_crowded_slots` в наблюдении.
- Дисквалификация по `min_total_hours` применяется в `get_payoffs()`
  одинаково.
```

- [ ] **Step 3: Add `scoring_mode` to the parameter table in `el-farol-bar.ru.md`**

Find the parameter table (around line 96-108) under "## Параметры конфигурации (`ElFarolConfig`)". Add a row:

```markdown
| `scoring_mode` | режим подсчёта (см. выше) | "happy_only" |
```

Place it after the `slot_duration` row.

- [ ] **Step 4: Replace the scoring sections in `docs/games/rules/el-farol-bar.en.md`**

Open the English file. Find the equivalent "## Per-round payoff" + "## Final payoffs" sections and replace them with the English mirror:

```markdown
## Scoring modes (`scoring_mode`)

The engine supports two scoring modes. The default is `happy_only`.
Tournaments always use the default — there is no per-tournament
opt-in to the legacy mode in v1.

### `happy_only` (default) — simple accumulation

- **Each happy slot**: +1.
- **Each crowded slot**: 0 (no penalty).
- **Per-round payoff** (returned by `step()` and
  `compute_round_payoffs()`): number of happy slots that round.
- **Standalone final `get_payoffs()`**: `t_happy` (sum of happy slots
  across all rounds), provided the player meets `min_total_hours`.
- **Tournament `total_score`** = sum of per-round payoffs = `t_happy`.

### `happy_minus_crowded` (legacy, opt-in)

- Reachable only via explicit `ElFarolConfig(scoring_mode=...)` —
  tests, atp-games standalone scenarios. Not exposed in tournament
  APIs.
- **Each happy slot**: +1; **each crowded slot**: −1.
- **Per-round payoff**: `happy − crowded`.
- **Standalone final `get_payoffs()`**: `t_happy / max(t_crowded,
  0.1)`, provided the player meets `min_total_hours`.

In both modes:
- `_t_crowded` is accumulated and surfaced to the player as
  `your_t_crowded_slots` in observation.
- `min_total_hours` disqualification applies in `get_payoffs()`
  identically.
```

Add the parameter-table row in the equivalent location:

```markdown
| `scoring_mode` | scoring mode (see above) | "happy_only" |
```

- [ ] **Step 5: Verify no markdown lint issues**

Run: `uv run ruff check docs/games/rules/`
Expected: ruff doesn't lint markdown — this should be a no-op or skip.

Visually inspect the rendered output by reading the files top-to-bottom; confirm:
- The new "Scoring modes" section reads cleanly in place of the two old sections.
- Parameter table has the new row.
- No leftover references to the old formulas in surrounding sections (e.g. the "Идея" / "Idea" intro should not contradict the new scoring).

If the intro section says something like "...maximise time in non-crowded slots" — that's fine (still true). If it says "minus crowded" or formulas explicitly, update.

- [ ] **Step 6: Commit**

```bash
git add demo-game/agents/el_farol_agent.py docs/games/rules/el-farol-bar.ru.md docs/games/rules/el-farol-bar.en.md
git commit -m "docs(el-farol): scoring modes section in rules + demo agent"
```

---

## Task 9: participant-kit CHANGELOG + version bump

**Files:**
- Modify: `participant-kit-el-farol-en/README.md`

**Goal:** Add a CHANGELOG entry and bump the kit's minor version, alerting external bot authors that the default scoring rule changed. The wire format (action schema + state shape) is unchanged; only the score values produced under default config differ.

- [ ] **Step 1: Locate the CHANGELOG section in the README**

Open `participant-kit-el-farol-en/README.md`. Look for an existing `## Changelog` or `## CHANGELOG` section. If present, work there. If not, add one near the bottom of the file (above any final references / acknowledgements).

Also locate the kit's version reference. It might be a header like `# Participant Kit (El Farol) — vX.Y.Z` or a separate version line. If you can't find an explicit version, fall back to an entry-only update.

- [ ] **Step 2: Add the CHANGELOG entry**

Insert a new top entry under the CHANGELOG section. If a version exists like `1.0.0`, bump to `1.1.0`. Use today's date.

```markdown
## Changelog

### 1.1.0 — 2026-05-02

**Default scoring rule changed.** The El Farol engine's default
`scoring_mode` flipped from `happy_minus_crowded` to `happy_only`:

- **Before**: each crowded slot scored −1; round payoff = `happy −
  crowded`; final score = `t_happy / max(t_crowded, 0.1)` (ratio).
- **After**: each crowded slot scores 0 (no penalty); round payoff =
  number of happy slots; final score = sum of per-round happy values
  (= `t_happy`).

Wire format (action schema + state shape) is unchanged. Bots that
read `your_cumulative_score` from observations will see values on a
different scale: a strictly positive count instead of a ratio.

**For bot authors**: if your strategy is monotonic in the reported
score, no code change is required. If you hardcoded the legacy ratio
formula or compute counterfactual utility based on `happy − crowded`,
update accordingly. The legacy mode remains available via
`ElFarolConfig(scoring_mode="happy_minus_crowded")` for direct engine
consumers (tests, standalone scenarios), but is not exposed through
the tournament API in this release.

— Andrei
```

- [ ] **Step 3: Update the version reference (if any)**

If the README has a version header or a "Version: X.Y.Z" line, bump it from `X.Y.Z` to `X.(Y+1).0`. If no explicit version is anywhere in the file, the CHANGELOG entry alone is sufficient documentation.

- [ ] **Step 4: Commit**

```bash
git add participant-kit-el-farol-en/README.md
git commit -m "docs(participant-kit): CHANGELOG for happy_only default flip"
```

---

## Final verification

- [ ] **Step 1: Run the entire affected test surface**

Run: `uv run pytest game-environments/tests/test_el_farol.py atp-games/tests/ tests/unit/games/ tests/unit/dashboard/ tests/integration/dashboard/ packages/atp-dashboard/tests/ -q --tb=line`

Expected: all pass.

- [ ] **Step 2: Type-check + lint everything**

Run: `uv run pyrefly check game-environments/game_envs/games/el_farol.py packages/atp-dashboard/atp/dashboard/v2/game_copy.py demo-game/agents/el_farol_agent.py game-environments/tests/test_el_farol.py atp-games/tests/`

Run: `uv run ruff format .`

Run: `uv run ruff check .`

Expected: clean.

- [ ] **Step 3: Engine docstring consistency check**

Read `game-environments/game_envs/games/el_farol.py` end-to-end. Confirm:
- Module docstring (top) describes both modes.
- `compute_round_payoffs` docstring describes both modes.
- `get_payoffs` docstring describes both modes.
- `to_prompt()` branches are present.
- No stale "happy − crowded" or "t_happy / max(t_crowded, 0.1)" references that should be conditional.

- [ ] **Step 4: Rules-doc consistency check**

Read both `docs/games/rules/el-farol-bar.ru.md` and `el-farol-bar.en.md` end-to-end. Confirm the parameter table includes `scoring_mode` and no leftover sections describe the old formulas as the only scoring rule.

- [ ] **Step 5: Push the branch**

```bash
git push -u origin feat/el-farol-scoring-modes
```

- [ ] **Step 6: Pre-deploy reminder (procedural — not part of this branch)**

Before merging the resulting PR to main and triggering deploy, run on the prod DB:

```sql
SELECT id, status, game_type, num_players, created_at
FROM tournaments
WHERE status IN ('pending', 'active') AND game_type = 'el_farol';
```

If any rows return, either wait for them to complete / cancel them via `POST /api/v1/tournaments/{id}/cancel`, OR document explicit acceptance of the mid-flight scoring mix in the deploy commit / PR description.

This is operational, not enforced by code. Skipping it is the only path to surprising mid-tournament behaviour where some `Action.payoff` rows use the old formula and others use the new.

# El Farol Bar — rules (ATP Platform)

This document is derived from the game implementation in the repository. Registry id: `el_farol`, class: `ElFarolBar`, file: `game-environments/game_envs/games/el_farol.py`.

---

## Concept

The El Farol Bar problem (Brian Arthur, 1994): N players independently choose time slots to attend. If the number of visitors in a slot reaches or exceeds the capacity threshold, the slot is **crowded** — being there is undesirable. Players want to spend more time in uncrowded slots.

In this codebase the game is **repeated** with **simultaneous** moves (`MoveOrder.SIMULTANEOUS`); one framework step equals one **day**.

---

## Players and scale

- `num_players` must be between **2** and **1000** (default in config: 100).
- Player ids: `player_0`, …, `player_{N-1}`.

---

## Action

- Each day every player submits `{"intervals": [[start, end], ...]}` —
  a list of inclusive `[start, end]` slot-index pairs.
- At most **2** intervals per day, covering at most
  **`MAX_SLOTS_PER_DAY = 8`** slots in total.
- Intervals must be **non-overlapping** and **non-adjacent** (at least
  one empty slot between any two intervals).
- Slot indices must lie in `[0, num_slots - 1]`.
- An empty list `{"intervals": []}` (or bare `[]`) means “stay home” that
  day (no slot visited).
- The legacy flat-slot shape (`{"slots": [...]}`) is no longer accepted —
  invalid input is mapped to a safe action via `sanitize`.

Default config: `num_slots = 16`, `slot_duration = 0.5` h — i.e. sixteen half-hour slots per day (eight hours total) unless you change the config.

---

## Crowding

For each slot, count **how many players** chose that slot on the current day.

- A slot is **happy** (not crowded) for a player if occupancy is **strictly less than** `capacity_threshold`.
- A slot is **crowded** for a player if occupancy is **≥** `capacity_threshold`.

Default `capacity_threshold = 60` (the example suite below uses a smaller value for small N).

---

## Per-round payoff

For each player each day:

- `happy` — number of slots they attended where occupancy `< threshold`.
- `crowded` — number of slots they attended where occupancy `≥ threshold`.

**Payoff that day:** `happy - crowded` (net “good minus bad” in slot units).

Running totals `t_happy` and `t_crowded` are kept in **slots**, not hours.

---

## Final score (`get_payoffs`)

After all days:

- If total attendance hours `(t_happy + t_crowded) * slot_duration` are **strictly less than** `min_total_hours`, the player’s final payoff is **0** (disqualification).
- Otherwise: **`t_happy / max(t_crowded, 0.1)`**.

Default `min_total_hours = 0.0` — hour-based disqualification is off unless you set a positive threshold.

---

## Player information (observation)

The game exposes two different state schemas to players depending on where the agent runs:

**1. Direct `observe(player_id)` call (in-process, used by game runner):**

- past-day attendance (`attendance_history`) — **public** information;
- `capacity_threshold`, `num_slots`, `slot_duration_hours`, `min_total_hours`;
- their own running totals `your_t_happy_slots`, `your_t_crowded_slots`, `your_total_hours`.

**2. Tournament state via `format_state_for_player` (used by the MCP / tournament server):**

- `your_history` — list of slot lists the player chose on each past day;
- `attendance_by_round` — per-day occupancy counts per slot;
- `capacity_threshold`, `num_slots`, `action_schema`;
- `your_cumulative_score`, `all_scores`, `your_participant_idx`.

In either case other players’ plans for **today** are **not** revealed — only past attendance is visible (as stated in the class docstring / prompt).

---

## Configuration (`ElFarolConfig`)

| Field | Meaning | Default |
|-------|---------|---------|
| `num_players` | number of players | 100 |
| `num_rounds` | number of days | 30 |
| `num_slots` | slots per day | 16 |
| `capacity_threshold` | crowding threshold per slot | 60 |
| `min_total_hours` | minimum hours or final payoff 0 | 0.0 |
| `slot_duration` | slot length (hours) | 0.5 |

Inherited `GameConfig` fields (e.g. `discount_factor`, `noise`, `seed`) apply if used in your scenario.

---

## Notes from the code (theory)

- Symmetric mixed Nash: attend a given slot with probability `p* ≈ threshold / num_players` (stated explicitly in the in-game prompt).
- Heterogeneous strategies in a population often beat that Nash in aggregate welfare (as in the class docstring).

---

## Example suite

See `examples/test_suites/13_game_el_farol.yaml` — smaller `num_players`, `capacity_threshold`, and a list of built-in strategies.

---

## To extend

_(space for your rules, tournament variants, prose explanations, and references)_

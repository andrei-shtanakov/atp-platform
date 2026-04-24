# ADR-004: Per-move Reasoning on Tournament Actions

**Status**: Proposed
**Date**: 2026-04-16
**Context**: Tournament agents (LLM-based and otherwise) currently submit
canonical game actions (`choice`, `slots`) via the MCP `make_move` tool.
Any rationale the agent produces is discarded at the wire — `PDAction` /
`SHAction` / `BoSAction` / `ElFarolAction` use `ConfigDict(extra="forbid")`
and `game.validate_action()` canonicalizes down to the minimum payload
needed by the game. For eval-driven learning (arbiter R-13) and
post-tournament cross-agent analysis, we want to preserve agent rationale.

## Decision

Add an **optional free-form `reasoning: str`** field to every tournament
action schema. Persist it as an inline `Text` column on
`tournament_actions`. Cap length via env-tunable `max_length`.

### Specifics

- **Shape**: `reasoning: str | None = Field(default=None, max_length=N)` on
  `PDAction`, `SHAction`, `BoSAction`, `ElFarolAction`. `extra="forbid"`
  stays — the field is declared explicitly, not via relaxation.
- **Limit**: `N = int(os.environ.get("ATP_TOURNAMENT_REASONING_MAX_CHARS", "8000"))`.
  8 000 chars ≈ 2 000 tokens, enough for typical CoT from frontier models;
  can be raised or lowered per environment.
- **Storage**: inline `Text` column on `tournament_actions` (nullable, no
  backfill). JSON structure of `Action.action_data` is unchanged.
- **Normalization**: service strips whitespace; empty `""` / whitespace-only
  becomes `None` (explicit "no reasoning" ↔ absence).
- **Privacy gate**: during `status == "active"`, reasoning is visible to
  admin, the tournament owner (`created_by`), and the agent's own rows.
  After `status == "completed"`, visible to everyone who can see the
  tournament.

## Alternatives considered

1. **Separate table `tournament_action_reasonings`** (FK to Action). Cleaner
   row size for non-reasoning queries; considered overkill at current scale.
   **Threshold to revisit**: if `ATP_TOURNAMENT_REASONING_MAX_CHARS` is
   raised above 16 000, or if `tournament_actions` row count exceeds 100 k,
   migrate to a separate table.
2. **Structured reasoning** (`{rationale: str, confidence: float | None,
   citations: list[str]}`). Considered, rejected for MVP — YAGNI, and would
   force prompt engineering on every agent. Future migration path:
   `reasoning: str | ReasoningV2 | None` union to stay backward-compat with
   existing str values.
3. **Out-of-band MCP tool `annotate_move`**. Rejected — two RTTs instead
   of one; agents can forget to call it; couples reasoning lifecycle to a
   separate call that can fail independently.
4. **Server-side truncation with `truncated: true` flag** instead of 422 on
   overflow. Rejected — silent data loss is worse than a loud reject; the
   env-tunable limit lets operators resolve it without shipping.

## Open questions / follow-ups

- **Arbiter R-13 consumer format**: ATP publishes reasoning via the
  extended `GET /api/v1/tournaments/{id}/rounds` response. If arbiter
  expects a different shape, either add a transform endpoint or align
  when the arbiter integration lands.
- **Moderation / redaction**: no admin soft-delete of abusive reasoning in
  the initial PR. Tracked as a follow-up LABS-* issue with fields like
  `reasoning_redacted_by` / `reasoning_redacted_at`.
- **MCP schema-guard tests**: not initially included, but a regression
  test that the `make_move` docstring/description still lists `reasoning`
  would prevent silent drift. Optional follow-up.

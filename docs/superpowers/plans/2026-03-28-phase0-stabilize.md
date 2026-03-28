# Phase 0: Stabilize — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit and stabilize the ATP Protocol and game-theory library before public release, producing two documents: Protocol v1.0 Specification and Game Theory Coverage Report.

**Architecture:** Two independent tracks — Track 1 audits the platform's public API surface and protocol models; Track 2 audits game correctness, edge cases, and identifies gaps. Both tracks produce documentation artifacts in `docs/`.

**Tech Stack:** Python 3.12+, pydantic, pytest, hypothesis, pyrefly, ruff

---

## File Structure

### New files to create

```
docs/protocol-v1-spec.md                    — ATP Protocol v1.0 formal specification
docs/game-theory-coverage-report.md          — Game theory audit results and gap analysis
tests/unit/test_protocol_version.py          — Protocol version validation tests
tests/unit/test_api_surface.py               — Public API surface regression tests
packages/atp-core/atp/protocol/_version.py   — Protocol version constants and validation
```

### Existing files to modify

```
packages/atp-core/atp/protocol/models.py     — Add version validation, protocol_version constant
packages/atp-core/atp/protocol/__init__.py   — Export PROTOCOL_VERSION constant
```

---

## Track 1: Protocol Audit & Platform Stabilization

### Task 1: Protocol Version Constant and Validation

**Files:**
- Create: `packages/atp-core/atp/protocol/_version.py`
- Modify: `packages/atp-core/atp/protocol/models.py`
- Modify: `packages/atp-core/atp/protocol/__init__.py`
- Test: `tests/unit/test_protocol_version.py`

- [ ] **Step 1: Write failing tests for protocol version**

```python
# tests/unit/test_protocol_version.py
"""Tests for ATP Protocol version handling."""

from atp.protocol import (
    PROTOCOL_VERSION,
    ATPEvent,
    ATPRequest,
    ATPResponse,
)


def test_protocol_version_constant_exists() -> None:
    """PROTOCOL_VERSION is exported and matches expected format."""
    assert isinstance(PROTOCOL_VERSION, str)
    assert PROTOCOL_VERSION == "1.0"


def test_request_default_version() -> None:
    """ATPRequest defaults to current PROTOCOL_VERSION."""
    request = ATPRequest(
        task_id="test-1",
        task={"description": "Test task"},
    )
    assert request.version == PROTOCOL_VERSION


def test_response_default_version() -> None:
    """ATPResponse defaults to current PROTOCOL_VERSION."""
    response = ATPResponse(
        task_id="test-1",
        status="completed",
    )
    assert response.version == PROTOCOL_VERSION


def test_event_default_version() -> None:
    """ATPEvent defaults to current PROTOCOL_VERSION."""
    event = ATPEvent(
        task_id="test-1",
        sequence=0,
        event_type="progress",
        payload={"message": "test"},
    )
    assert event.version == PROTOCOL_VERSION


def test_request_rejects_unsupported_version() -> None:
    """ATPRequest rejects versions outside supported set."""
    import pytest

    with pytest.raises(ValueError, match="version"):
        ATPRequest(
            task_id="test-1",
            task={"description": "Test task"},
            version="99.0",
        )


def test_response_rejects_unsupported_version() -> None:
    """ATPResponse rejects unsupported versions."""
    import pytest

    with pytest.raises(ValueError, match="version"):
        ATPResponse(
            task_id="test-1",
            status="completed",
            version="99.0",
        )


def test_event_rejects_unsupported_version() -> None:
    """ATPEvent rejects unsupported versions."""
    import pytest

    with pytest.raises(ValueError, match="version"):
        ATPEvent(
            task_id="test-1",
            sequence=0,
            event_type="progress",
            payload={},
            version="99.0",
        )


def test_request_accepts_supported_version() -> None:
    """ATPRequest accepts explicitly set supported version."""
    request = ATPRequest(
        task_id="test-1",
        task={"description": "Test task"},
        version="1.0",
    )
    assert request.version == "1.0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_protocol_version.py -v`
Expected: FAIL — `PROTOCOL_VERSION` not exported, no version validation

- [ ] **Step 3: Create protocol version module**

```python
# packages/atp-core/atp/protocol/_version.py
"""ATP Protocol version constants."""

PROTOCOL_VERSION = "1.0"

SUPPORTED_VERSIONS = frozenset({"1.0"})
```

- [ ] **Step 4: Add version validation to models**

In `packages/atp-core/atp/protocol/models.py`, add a validator to each of the three models. Find the existing `version` field in `ATPRequest` (line ~163):

```python
version: str = Field("1.0", description="Protocol version")
```

Replace with:

```python
version: str = Field(
    default=PROTOCOL_VERSION,
    description="Protocol version",
)

@field_validator("version")
@classmethod
def validate_version(cls, v: str) -> str:
    """Ensure version is supported."""
    from atp.protocol._version import SUPPORTED_VERSIONS

    if v not in SUPPORTED_VERSIONS:
        msg = (
            f"Unsupported protocol version '{v}'. "
            f"Supported: {sorted(SUPPORTED_VERSIONS)}"
        )
        raise ValueError(msg)
    return v
```

Apply the same change to `ATPResponse` (line ~315) and `ATPEvent` (line ~413). Also add the import at the top of models.py:

```python
from atp.protocol._version import PROTOCOL_VERSION
```

- [ ] **Step 5: Export PROTOCOL_VERSION from protocol __init__**

In `packages/atp-core/atp/protocol/__init__.py`, add to imports and `__all__`:

```python
from atp.protocol._version import PROTOCOL_VERSION, SUPPORTED_VERSIONS

__all__ = [
    "PROTOCOL_VERSION",
    "SUPPORTED_VERSIONS",
    # ... existing exports ...
]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_protocol_version.py -v`
Expected: All 8 tests PASS

- [ ] **Step 7: Run full test suite to check for regressions**

Run: `uv run pytest tests/ -v --timeout=60 -x -q 2>&1 | tail -20`
Expected: No new failures (existing tests may use `version="1.0"` which remains valid)

- [ ] **Step 8: Run code quality checks**

Run: `uv run ruff format . && uv run ruff check . --fix && pyrefly check`
Expected: Clean output

- [ ] **Step 9: Commit**

```bash
git add packages/atp-core/atp/protocol/_version.py \
       packages/atp-core/atp/protocol/models.py \
       packages/atp-core/atp/protocol/__init__.py \
       tests/unit/test_protocol_version.py
git commit -m "feat: add protocol version validation and PROTOCOL_VERSION constant"
```

---

### Task 2: API Surface Regression Tests

**Files:**
- Create: `tests/unit/test_api_surface.py`

This task creates snapshot tests that will break if public exports change unexpectedly — a safety net before PyPI publishing.

- [ ] **Step 1: Write API surface tests**

```python
# tests/unit/test_api_surface.py
"""Regression tests for public API surface of each package.

These tests ensure that public exports don't change unexpectedly.
If a test fails, it means the public API changed — update the test
intentionally after confirming the change is desired.
"""


def test_protocol_exports() -> None:
    """atp.protocol exports are stable."""
    from atp import protocol

    expected = {
        "PROTOCOL_VERSION",
        "SUPPORTED_VERSIONS",
        "ATPRequest",
        "ATPResponse",
        "ATPEvent",
        "Task",
        "Context",
        "Metrics",
        "ArtifactFile",
        "ArtifactStructured",
        "ArtifactReference",
        "ResponseStatus",
        "EventType",
        "ToolCallPayload",
        "LLMRequestPayload",
        "ReasoningPayload",
        "ErrorPayload",
        "ProgressPayload",
    }
    actual = set(protocol.__all__)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Missing exports: {missing}"
    assert not extra, f"Unexpected new exports: {extra}"


def test_adapters_exports() -> None:
    """atp.adapters exports are stable."""
    from atp import adapters

    expected = {
        "AgentAdapter",
        "AdapterConfig",
        "track_response_cost",
        "HTTPAdapter",
        "HTTPAdapterConfig",
        "ContainerAdapter",
        "ContainerAdapterConfig",
        "ContainerResources",
        "CLIAdapter",
        "CLIAdapterConfig",
        "LangGraphAdapter",
        "LangGraphAdapterConfig",
        "CrewAIAdapter",
        "CrewAIAdapterConfig",
        "AutoGenAdapter",
        "AutoGenAdapterConfig",
        "AzureOpenAIAdapter",
        "AzureOpenAIAdapterConfig",
        "MCPAdapter",
        "MCPAdapterConfig",
        "MCPTool",
        "MCPResource",
        "MCPPrompt",
        "MCPServerInfo",
        "BedrockAdapter",
        "BedrockAdapterConfig",
        "VertexAdapter",
        "VertexAdapterConfig",
        "AdapterRegistry",
        "get_registry",
        "create_adapter",
        "AdapterError",
        "AdapterTimeoutError",
        "AdapterConnectionError",
        "AdapterResponseError",
        "AdapterNotFoundError",
    }
    actual = set(adapters.__all__)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Missing exports: {missing}"
    assert not extra, f"Unexpected new exports: {extra}"


def test_dashboard_exports() -> None:
    """atp.dashboard exports are stable."""
    from atp import dashboard

    expected = {
        "app",
        "create_app",
        "run_server",
        "Base",
        "Database",
        "get_database",
        "init_database",
        "set_database",
        "Agent",
        "Artifact",
        "EvaluationResult",
        "RunResult",
        "ScoreComponent",
        "SuiteExecution",
        "TestExecution",
        "User",
        "ResultStorage",
    }
    actual = set(dashboard.__all__)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Missing exports: {missing}"
    assert not extra, f"Unexpected new exports: {extra}"


def test_core_exports_include_settings() -> None:
    """atp.core includes essential settings and logging."""
    from atp import core

    essential = {
        "ATPSettings",
        "get_settings",
        "configure_logging",
        "get_logger",
        "configure_telemetry",
        "get_tracer",
    }
    actual = set(core.__all__)
    missing = essential - actual
    assert not missing, f"Missing essential exports: {missing}"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_api_surface.py -v`
Expected: All 4 tests PASS (these are snapshot tests of current state)

- [ ] **Step 3: Run ruff format**

Run: `uv run ruff format tests/unit/test_api_surface.py`

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_api_surface.py
git commit -m "test: add API surface regression tests for all packages"
```

---

### Task 3: Deprecation Sweep

**Files:**
- Modify: various files across `atp/`, `packages/`

This is an investigative task. Search for dead code, unused imports, TODO stubs, and remove them.

- [ ] **Step 1: Find all TODO/FIXME/HACK comments**

Run: `grep -rn "TODO\|FIXME\|HACK\|XXX" atp/ packages/ --include="*.py" | grep -v __pycache__ | grep -v ".pyc"`

Review each result. For each TODO:
- If it describes work captured in the spec/plan: leave it
- If it's a stale placeholder for already-done work: remove it
- If it references dead code: remove the code and the TODO

- [ ] **Step 2: Find unused imports across the codebase**

Run: `uv run ruff check . --select F401 2>&1 | head -50`

Fix any unused imports found.

- [ ] **Step 3: Run ruff check for all auto-fixable issues**

Run: `uv run ruff check . --fix`

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=60 -x -q 2>&1 | tail -20`
Expected: No regressions from cleanup

- [ ] **Step 5: Run pyrefly check**

Run: `pyrefly check`
Expected: 0 errors

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: deprecation sweep — remove dead code and stale TODOs"
```

---

### Task 4: Write Protocol v1.0 Specification Document

**Files:**
- Create: `docs/protocol-v1-spec.md`

This document formalizes the protocol for public consumption. It describes the wire format, versioning policy, and what constitutes a breaking change.

- [ ] **Step 1: Write the specification**

```markdown
# ATP Protocol v1.0 Specification

**Version:** 1.0
**Status:** Stable
**Date:** 2026-03-28

## Overview

The ATP (Agent Test Platform) Protocol defines a standardized interface for
communicating with AI agents regardless of their implementation framework.
It consists of three message types: Request, Response, and Event.

## Versioning Policy

- The protocol follows Semantic Versioning (semver)
- **Major version** (1.x → 2.x): Breaking changes to message structure
- **Minor version** (1.0 → 1.1): Backward-compatible additions (new optional fields)
- All messages carry a `version` field defaulting to "1.0"
- Consumers SHOULD accept unknown fields gracefully (forward compatibility)

### What constitutes a breaking change

- Removing or renaming a required field
- Changing the type of an existing field
- Changing the semantics of an existing field
- Adding a new required field (without default)
- Removing a value from an enum (ResponseStatus, EventType)
- Changing validation constraints to be more restrictive

### What is NOT a breaking change

- Adding a new optional field with a default value
- Adding a new value to an enum
- Relaxing validation constraints
- Adding new artifact types
- Adding new event payload types

## Message Types

### ATPRequest

Sent by the test runner to an agent adapter.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `version` | string | no | "1.0" | Protocol version |
| `task_id` | string | **yes** | — | Unique task identifier (alphanumeric, `_`, `-`; max 128 chars) |
| `task` | Task | **yes** | — | Task specification |
| `constraints` | object | no | `{}` | Execution constraints (max_steps, timeout, allowed_tools) |
| `context` | Context | no | `null` | Execution context (tools endpoint, workspace, environment) |
| `metadata` | object | no | `null` | Pass-through metadata (max 50 keys) |

### ATPResponse

Returned by an agent adapter after task execution.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `version` | string | no | "1.0" | Protocol version |
| `task_id` | string | **yes** | — | Task identifier from the request |
| `status` | ResponseStatus | **yes** | — | Execution outcome |
| `artifacts` | Artifact[] | no | `[]` | Output artifacts (max 1000) |
| `metrics` | Metrics | no | `null` | Execution metrics (tokens, steps, cost) |
| `error` | string | no | `null` | Error message if status is "failed" (max 10000 chars) |
| `trace_id` | string | no | `null` | Optional trace identifier (max 256 chars) |

### ATPEvent

Streamed during task execution for real-time observability.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `version` | string | no | "1.0" | Protocol version |
| `task_id` | string | **yes** | — | Task identifier |
| `timestamp` | datetime | no | now (UTC) | Event timestamp |
| `sequence` | integer | **yes** | — | Monotonic sequence number (>= 0) |
| `event_type` | EventType | **yes** | — | Event classification |
| `payload` | object | **yes** | — | Event-specific data |

## Enums

### ResponseStatus

| Value | Description |
|-------|-------------|
| `completed` | Task finished successfully |
| `failed` | Task encountered an error |
| `timeout` | Task exceeded time limit |
| `cancelled` | Task was cancelled externally |
| `partial` | Task produced partial results |

### EventType

| Value | Payload Type | Description |
|-------|-------------|-------------|
| `tool_call` | ToolCallPayload | Agent invoked a tool |
| `llm_request` | LLMRequestPayload | Agent made an LLM API call |
| `reasoning` | ReasoningPayload | Agent reasoning step |
| `error` | ErrorPayload | Non-fatal error during execution |
| `progress` | ProgressPayload | Progress update |

## Supporting Types

### Task

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `description` | string | **yes** | Task description (1–100,000 chars) |
| `input_data` | object | no | Structured input data |
| `expected_artifacts` | string[] | no | Expected output artifact paths |

### Context

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tools_endpoint` | string | no | HTTP/HTTPS URL for tool server |
| `workspace_path` | string | no | Filesystem workspace path (max 4096 chars) |
| `environment` | object | no | Environment variables (max 100) |

### Metrics

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `total_tokens` | integer | no | Total tokens consumed |
| `input_tokens` | integer | no | Input/prompt tokens |
| `output_tokens` | integer | no | Output/completion tokens |
| `total_steps` | integer | no | Total execution steps |
| `tool_calls` | integer | no | Number of tool invocations |
| `llm_calls` | integer | no | Number of LLM API calls |
| `wall_time_seconds` | float | no | Wall clock execution time |
| `cost_usd` | float | no | Estimated cost in USD |

### Artifact Types

Three artifact variants, distinguished by the `type` discriminator field:

**ArtifactFile** (`type: "file"`): A file produced by the agent.
- `path` (required), `content_type`, `size_bytes`, `content_hash`, `content`

**ArtifactStructured** (`type: "structured"`): Structured data output.
- `name` (required), `data` (required), `content_type`

**ArtifactReference** (`type: "reference"`): Reference to an external resource.
- `path` (required), `content_type`, `size_bytes`

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_TASK_ID_LENGTH` | 128 | Maximum task_id length |
| `MAX_DESCRIPTION_LENGTH` | 100,000 | Maximum task description |
| `MAX_PATH_LENGTH` | 4,096 | Maximum file path length |
| `MAX_ERROR_LENGTH` | 10,000 | Maximum error message |
| `MAX_CONTENT_LENGTH` | 10,000,000 | Maximum artifact content (10 MB) |
| `MAX_ARTIFACTS_COUNT` | 1,000 | Maximum artifacts per response |
| `MAX_ENV_VARS_COUNT` | 100 | Maximum environment variables |
| `MAX_METADATA_KEYS` | 50 | Maximum metadata keys |

## Adapter Contract

Any adapter implementing the ATP Protocol MUST:

1. Accept `ATPRequest` and return `ATPResponse`
2. Optionally support streaming via `AsyncIterator[ATPEvent | ATPResponse]`
3. Set `version` field on all outgoing messages
4. Validate incoming `task_id` format
5. Return `ResponseStatus.FAILED` with `error` field on unrecoverable errors
6. Return `ResponseStatus.TIMEOUT` when execution exceeds constraints
7. Emit events with monotonically increasing `sequence` numbers
8. Include `Metrics` when available (tokens, cost, timing)
```

- [ ] **Step 2: Run ruff format on the doc (no-op for markdown, but good habit)**

Verify the markdown renders correctly: `cat docs/protocol-v1-spec.md | head -20`

- [ ] **Step 3: Commit**

```bash
git add docs/protocol-v1-spec.md
git commit -m "docs: add ATP Protocol v1.0 formal specification"
```

---

## Track 2: Game Theory Audit

### Task 5: Verify Game Correctness — Payoff Matrices and Nash Equilibria

**Files:**
- Create: `tests/unit/test_game_correctness.py`

This task verifies that each game's payoff structure matches its theoretical definition.

- [ ] **Step 1: Write correctness tests for Prisoner's Dilemma**

```python
# tests/unit/test_game_correctness.py
"""Verify game-theoretic correctness of all implemented games.

Each test validates payoff matrices against known theoretical results
and ensures Nash equilibrium properties hold.
"""

import pytest

from game_envs.games.prisoners_dilemma import (
    PDConfig,
    PrisonersDilemma,
)


class TestPrisonersDilemmaCorrectness:
    """Verify PD satisfies T > R > P > S and 2R > T + S."""

    def test_default_payoff_ordering(self) -> None:
        """Default config satisfies PD constraints."""
        config = PDConfig(num_players=2, num_rounds=1)
        assert config.temptation > config.reward
        assert config.reward > config.punishment
        assert config.punishment > config.sucker
        assert 2 * config.reward > config.temptation + config.sucker

    def test_mutual_cooperation_payoff(self) -> None:
        """Both cooperate → both get R."""
        game = PrisonersDilemma(PDConfig(num_players=2, num_rounds=1))
        game.reset()
        result = game.step({"player_0": "cooperate", "player_1": "cooperate"})
        payoffs = result.payoffs
        assert payoffs["player_0"] == 3.0  # R = 3.0
        assert payoffs["player_1"] == 3.0

    def test_mutual_defection_payoff(self) -> None:
        """Both defect → both get P."""
        game = PrisonersDilemma(PDConfig(num_players=2, num_rounds=1))
        game.reset()
        result = game.step({"player_0": "defect", "player_1": "defect"})
        payoffs = result.payoffs
        assert payoffs["player_0"] == 1.0  # P = 1.0
        assert payoffs["player_1"] == 1.0

    def test_asymmetric_payoff(self) -> None:
        """One cooperates, one defects → T and S respectively."""
        game = PrisonersDilemma(PDConfig(num_players=2, num_rounds=1))
        game.reset()
        result = game.step({"player_0": "cooperate", "player_1": "defect"})
        payoffs = result.payoffs
        assert payoffs["player_0"] == 0.0  # S = 0.0 (sucker)
        assert payoffs["player_1"] == 5.0  # T = 5.0 (temptation)

    def test_defect_is_dominant_strategy(self) -> None:
        """In one-shot PD, defect dominates cooperate for both players."""
        config = PDConfig(num_players=2, num_rounds=1)
        # For player_0:
        # If opponent cooperates: defect(T=5) > cooperate(R=3)
        assert config.temptation > config.reward
        # If opponent defects: defect(P=1) > cooperate(S=0)
        assert config.punishment > config.sucker
```

- [ ] **Step 2: Add tests for Auction correctness**

Append to the same file:

```python
from game_envs.games.auction import AuctionConfig, SealedBidAuction


class TestAuctionCorrectness:
    """Verify auction payoff rules."""

    def test_first_price_winner_pays_own_bid(self) -> None:
        """In first-price auction, winner pays their bid."""
        config = AuctionConfig(
            num_players=2,
            num_rounds=1,
            auction_type="first_price",
            min_bid=0.0,
            max_bid=100.0,
        )
        game = SealedBidAuction(config)
        game.reset()
        # Manually set private values for determinism
        game._private_values = {"player_0": 80.0, "player_1": 60.0}
        result = game.step({"player_0": 50.0, "player_1": 30.0})
        # player_0 wins (bid 50 > 30), payoff = value - bid = 80 - 50 = 30
        assert result.payoffs["player_0"] == pytest.approx(30.0)
        # player_1 loses, payoff = 0
        assert result.payoffs["player_1"] == pytest.approx(0.0)

    def test_second_price_winner_pays_second_bid(self) -> None:
        """In Vickrey auction, winner pays the second-highest bid."""
        config = AuctionConfig(
            num_players=2,
            num_rounds=1,
            auction_type="second_price",
            min_bid=0.0,
            max_bid=100.0,
        )
        game = SealedBidAuction(config)
        game.reset()
        game._private_values = {"player_0": 80.0, "player_1": 60.0}
        result = game.step({"player_0": 50.0, "player_1": 30.0})
        # player_0 wins, pays second-highest bid (30)
        # payoff = value - second_price = 80 - 30 = 50
        assert result.payoffs["player_0"] == pytest.approx(50.0)
        assert result.payoffs["player_1"] == pytest.approx(0.0)
```

- [ ] **Step 3: Add tests for Public Goods Game correctness**

Append to the same file:

```python
from game_envs.games.public_goods import PublicGoodsConfig, PublicGoodsGame


class TestPublicGoodsCorrectness:
    """Verify public goods payoff formula: endowment - contribution + multiplier * sum / N."""

    def test_full_contribution_payoff(self) -> None:
        """All players contribute fully → multiplier effect."""
        config = PublicGoodsConfig(
            num_players=3,
            num_rounds=1,
            endowment=20.0,
            multiplier=1.6,
        )
        game = PublicGoodsGame(config)
        game.reset()
        # All contribute 20
        actions = {"player_0": 20.0, "player_1": 20.0, "player_2": 20.0}
        result = game.step(actions)
        # payoff = 20 - 20 + 1.6 * 60 / 3 = 0 + 32 = 32
        for pid in ["player_0", "player_1", "player_2"]:
            assert result.payoffs[pid] == pytest.approx(32.0)

    def test_zero_contribution_payoff(self) -> None:
        """All contribute nothing → each keeps endowment, no public good."""
        config = PublicGoodsConfig(
            num_players=3,
            num_rounds=1,
            endowment=20.0,
            multiplier=1.6,
        )
        game = PublicGoodsGame(config)
        game.reset()
        actions = {"player_0": 0.0, "player_1": 0.0, "player_2": 0.0}
        result = game.step(actions)
        # payoff = 20 - 0 + 1.6 * 0 / 3 = 20
        for pid in ["player_0", "player_1", "player_2"]:
            assert result.payoffs[pid] == pytest.approx(20.0)

    def test_free_rider_advantage(self) -> None:
        """Free rider (0 contribution) earns more than contributors."""
        config = PublicGoodsConfig(
            num_players=3,
            num_rounds=1,
            endowment=20.0,
            multiplier=1.6,
        )
        game = PublicGoodsGame(config)
        game.reset()
        actions = {"player_0": 20.0, "player_1": 20.0, "player_2": 0.0}
        result = game.step(actions)
        # player_2 (free rider): 20 - 0 + 1.6 * 40 / 3 ≈ 41.33
        # player_0: 20 - 20 + 1.6 * 40 / 3 ≈ 21.33
        assert result.payoffs["player_2"] > result.payoffs["player_0"]
```

- [ ] **Step 4: Add tests for Colonel Blotto and Congestion**

Append to the same file:

```python
from game_envs.games.colonel_blotto import BlottoConfig, ColonelBlotto


class TestBlottoCorrectness:
    """Verify Blotto battlefield scoring."""

    def test_total_troops_constraint(self) -> None:
        """Allocations must sum to total_troops."""
        config = BlottoConfig(
            num_players=2,
            num_rounds=1,
            num_battlefields=3,
            total_troops=100,
        )
        game = ColonelBlotto(config)
        game.reset()
        actions = {
            "player_0": [50, 30, 20],
            "player_1": [10, 10, 80],
        }
        result = game.step(actions)
        # player_0 wins battlefield 0 (50>10) and 1 (30>10), loses 2 (20<80)
        # Score: 2/3 for player_0, 1/3 for player_1
        assert result.payoffs["player_0"] == pytest.approx(2 / 3)
        assert result.payoffs["player_1"] == pytest.approx(1 / 3)

    def test_tied_battlefields_split(self) -> None:
        """Tied battlefields award 0.5 to each player."""
        config = BlottoConfig(
            num_players=2,
            num_rounds=1,
            num_battlefields=3,
            total_troops=90,
        )
        game = ColonelBlotto(config)
        game.reset()
        actions = {
            "player_0": [30, 30, 30],
            "player_1": [30, 30, 30],
        }
        result = game.step(actions)
        assert result.payoffs["player_0"] == pytest.approx(0.5)
        assert result.payoffs["player_1"] == pytest.approx(0.5)


from game_envs.games.congestion import CongestionConfig, CongestionGame


class TestCongestionCorrectness:
    """Verify congestion game latency formula."""

    def test_single_route_no_congestion(self) -> None:
        """One player on a route → latency = base_cost + coefficient * 1."""
        config = CongestionConfig(
            num_players=1,
            num_rounds=1,
            routes=[
                {"name": "A", "base_cost": 10.0, "coefficient": 5.0},
                {"name": "B", "base_cost": 20.0, "coefficient": 1.0},
            ],
        )
        game = CongestionGame(config)
        game.reset()
        result = game.step({"player_0": "A"})
        # latency = 10 + 5*1 = 15, payoff = -15
        assert result.payoffs["player_0"] == pytest.approx(-15.0)
```

- [ ] **Step 5: Run all correctness tests**

Run: `uv run pytest tests/unit/test_game_correctness.py -v`
Expected: All tests PASS. If any fail, the game implementation has a bug — investigate and fix.

- [ ] **Step 6: Run ruff format**

Run: `uv run ruff format tests/unit/test_game_correctness.py`

- [ ] **Step 7: Commit**

```bash
git add tests/unit/test_game_correctness.py
git commit -m "test: add game-theoretic correctness tests for all 5 games"
```

---

### Task 6: Edge Case Testing for Games

**Files:**
- Create: `tests/unit/test_game_edge_cases.py`

- [ ] **Step 1: Write edge case tests**

```python
# tests/unit/test_game_edge_cases.py
"""Edge case tests for game-environments.

Covers boundary conditions: zero rounds, extreme payoffs,
single-player games, and action validation.
"""

import pytest

from game_envs.games.prisoners_dilemma import (
    PDConfig,
    PrisonersDilemma,
)
from game_envs.games.public_goods import PublicGoodsConfig, PublicGoodsGame


class TestPDEdgeCases:
    """Prisoner's Dilemma edge cases."""

    def test_single_round_terminates(self) -> None:
        """Game with num_rounds=1 is terminal after one step."""
        game = PrisonersDilemma(PDConfig(num_players=2, num_rounds=1))
        game.reset()
        result = game.step({"player_0": "cooperate", "player_1": "defect"})
        assert result.is_terminal

    def test_repeated_game_accumulates_payoffs(self) -> None:
        """Multi-round game accumulates payoffs across rounds."""
        game = PrisonersDilemma(PDConfig(num_players=2, num_rounds=3))
        game.reset()
        for _ in range(3):
            result = game.step(
                {"player_0": "cooperate", "player_1": "cooperate"}
            )
        # 3 rounds * R(3.0) = 9.0 each
        payoffs = game.get_payoffs()
        assert payoffs["player_0"] == pytest.approx(9.0)

    def test_invalid_action_raises(self) -> None:
        """Invalid action should raise ValueError."""
        game = PrisonersDilemma(PDConfig(num_players=2, num_rounds=1))
        game.reset()
        with pytest.raises((ValueError, KeyError)):
            game.step({"player_0": "invalid_action", "player_1": "cooperate"})


class TestPublicGoodsEdgeCases:
    """Public Goods Game edge cases."""

    def test_contribution_at_boundary(self) -> None:
        """Contribution exactly at endowment limit."""
        config = PublicGoodsConfig(
            num_players=2, num_rounds=1, endowment=20.0, multiplier=1.6
        )
        game = PublicGoodsGame(config)
        game.reset()
        result = game.step({"player_0": 20.0, "player_1": 0.0})
        # player_0: 20 - 20 + 1.6 * 20 / 2 = 16
        # player_1: 20 - 0 + 1.6 * 20 / 2 = 36
        assert result.payoffs["player_0"] == pytest.approx(16.0)
        assert result.payoffs["player_1"] == pytest.approx(36.0)

    def test_minimum_players(self) -> None:
        """Game works with minimum 2 players."""
        config = PublicGoodsConfig(
            num_players=2, num_rounds=1, endowment=10.0, multiplier=1.5
        )
        game = PublicGoodsGame(config)
        game.reset()
        result = game.step({"player_0": 5.0, "player_1": 5.0})
        # Each: 10 - 5 + 1.5 * 10 / 2 = 5 + 7.5 = 12.5
        assert result.payoffs["player_0"] == pytest.approx(12.5)
```

- [ ] **Step 2: Run edge case tests**

Run: `uv run pytest tests/unit/test_game_edge_cases.py -v`
Expected: All PASS

- [ ] **Step 3: Run ruff format**

Run: `uv run ruff format tests/unit/test_game_edge_cases.py`

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_game_edge_cases.py
git commit -m "test: add edge case tests for game-environments"
```

---

### Task 7: Write Game Theory Coverage Report

**Files:**
- Create: `docs/game-theory-coverage-report.md`

This document summarizes the audit findings and identifies gaps for Phase 1.

- [ ] **Step 1: Write the coverage report**

```markdown
# Game Theory Coverage Report

**Date:** 2026-03-28
**Status:** Phase 0 Audit Complete

## Implemented Games

| Game | Players | Action Space | Nash Equilibrium | Tests |
|------|---------|-------------|-----------------|-------|
| Prisoner's Dilemma | 2 | Discrete (cooperate/defect) | (defect, defect) in one-shot | Correctness + hypothesis + edge cases |
| Sealed-Bid Auction | 2-10 | Continuous [min, max] | Truthful bidding (Vickrey) | Correctness + property-based |
| Public Goods Game | 2-20 | Continuous [0, endowment] | Zero contribution (one-shot) | Correctness + edge cases |
| Colonel Blotto | 2 | Structured (allocation vector) | Mixed strategy only | Correctness + property-based |
| Congestion Game | 2-50 | Discrete (route selection) | Wardrop equilibrium | Correctness tests |

## Strategies Implemented

| Game | Strategies | Count |
|------|-----------|-------|
| Prisoner's Dilemma | AlwaysCooperate, AlwaysDefect, TitForTat, GrimTrigger, Pavlov, Random | 6 |
| Auction | Various bidding strategies | ~4 |
| Public Goods | Contribution strategies | ~3 |
| Colonel Blotto | Allocation strategies | ~3 |
| Congestion | Route selection strategies | ~3 |

**Total: ~19 strategies across 5 games**

## Analysis Modules

- Nash solver (support enumeration, Lemke-Howson, fictitious play, replicator dynamics)
- Cooperation analysis (rates, mutual cooperation, defection tracking)
- Exploitability analysis (best-response gap)
- Fairness analysis (Gini coefficient, min-max ratio)
- Population dynamics (evolutionary game theory)

## Tournament Infrastructure

- Round-robin, single elimination, double elimination
- Alympics composite benchmark (strategic 30%, cooperation 25%, fairness 25%, robustness 20%)
- Statistical comparison: Welch's t-test with Bonferroni correction

## Testing Coverage

- **Property-based tests** (hypothesis): Action spaces, game state invariants
- **Correctness tests**: Payoff matrix verification for all 5 games
- **Edge cases**: Boundary conditions, invalid actions, minimum/maximum players
- **Tournament tests**: Round-robin, elimination bracket, standings calculation

## Identified Gaps

### Missing Games (recommended for Phase 1)

1. **Stag Hunt** — Tests trust vs safety. Complements PD by having two pure Nash equilibria
   (both cooperate OR both defect) rather than a dominant strategy.
   - Priority: HIGH — essential for cooperation research
   - Complexity: LOW — 2-player discrete, similar to PD structure

2. **Battle of the Sexes** — Tests coordination under preference conflict. Two pure NE where
   players prefer different outcomes but benefit from coordinating.
   - Priority: MEDIUM — adds coordination dimension
   - Complexity: LOW — 2-player discrete

### Missing Features

1. **Elo rating system** — Currently uses win/loss/payoff ranking. Elo would enable
   persistent cross-tournament skill tracking.
   - Priority: HIGH — needed for meaningful LLM benchmarks

2. **Result caching** — No caching of LLM game results for reproducibility and cost savings.
   - Priority: MEDIUM — impacts benchmark cost

3. **Seed-based LLM reproducibility** — LLM responses are inherently non-deterministic.
   Need temperature=0 + seed parameter where supported.
   - Priority: MEDIUM — impacts reproducibility claims

## Recommendations

1. Add Stag Hunt and Battle of the Sexes in Phase 1 (Track 2)
2. Implement Elo rating in tournament system (Phase 1)
3. Add result caching for LLM game agents (Phase 1)
4. Note: Public Goods Game already exists — spec incorrectly listed it as new
```

- [ ] **Step 2: Commit**

```bash
git add docs/game-theory-coverage-report.md
git commit -m "docs: add game theory coverage report from Phase 0 audit"
```

---

## Verification

### Task 8: Final Phase 0 Verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=60 -q 2>&1 | tail -30`
Expected: All tests pass, including new protocol version, API surface, game correctness, and edge case tests.

- [ ] **Step 2: Run all quality checks**

Run: `uv run ruff format . && uv run ruff check . && pyrefly check`
Expected: Clean output on all three.

- [ ] **Step 3: Verify deliverables exist**

Run: `ls -la docs/protocol-v1-spec.md docs/game-theory-coverage-report.md`
Expected: Both files present.

- [ ] **Step 4: Review git log for Phase 0**

Run: `git log --oneline -10`
Expected: ~6-7 commits covering protocol version, API surface tests, deprecation sweep, protocol spec, game correctness tests, edge case tests, coverage report.

# Changelog

All notable changes to the ATP Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.0] - 2026-06-12

No breaking changes — additive features and fixes only.

### Added

- **`atp-method` plugin** — runs the agent-eval-case methodology through the
  platform without touching core. Ships the agent-eval-case schema, a loader
  (case → `TestDefinition`), and `AgentEvalCaseEvaluator` (binary
  `critical_check` + weighted rubric), wired via `register()` and the suite
  source registry. (#142, #144, #145, #146)
- **Hard-gate for critical assertions** — a failed `critical` check forces the
  test score to 0 regardless of the rubric, giving the methodology its
  trap/sweep "point of collapse". (#142)
- **Suite format-dispatch registry** — replaces the hardcoded game branch in
  the loader so new suite formats (e.g. agent-eval-case) plug in. (#143)
- **LLM-judge providers and steering** — OpenAI-compatible `base_url` for a
  fully air-gapped local grader (#149), a Bedrock-hosted Claude provider
  (#136), and `ATP_JUDGE_PROVIDER` / `ATP_JUDGE_REGION` / `ATP_JUDGE_MODEL`
  env steering for an all-in-AWS judge (#152).
- **Dashboard run-history UI** — `/ui/executions` list + detail pages for CLI
  `atp test` history, with a per-cause failure breakdown (#157), an `adapter`
  and `model` column, and a new `--model` run label (#159).
- **very-severe calibration case** — a fourth req-extraction difficulty tier
  (`very_severe` axis level) that brackets the collapse point of
  mid-capability models. (#158)
- **Demos** — on-prem docker-compose demo (platform + HTTP agent) (#135), the
  on-prem methodology sweep (#148), a turnkey air-gapped judge that reuses the
  agent's local model (#150), and an all-in-AWS cloud variant scaffold
  (EC2 + Bedrock via IAM) (#153).

### Changed

- **Run-history success semantics** — a test reads `success=True` only if it
  both executed successfully and passed evaluation (hard-gate aware), instead
  of merely completing execution. Fixes misleading green "Pass" badges next to
  score 0. (#162)
- **Docker image** — installs all workspace plugins (incl. `atp-method`) plus
  the LLM-judge clients, so `atp test` can dispatch plugin formats at runtime.
  (#151)

### Fixed

- **Dashboard boots without the `enterprise` extra** — the SAML/onelogin
  import is now lazy; SAML use raises a clear error instead of crashing the
  whole dashboard at import. (#161)
- **CLI/container runs persist to the dashboard DB** — the RBAC `roles` table
  is now created regardless of entry point, so `atp test` history is no longer
  lost to a "no such table: roles" rollback. (#160)
- **`SuiteExecutionSummary` accepts a null `agent_id`** for CLI-produced
  executions. (#132)
- **AWS demo** — installs boto3 for the Bedrock paths and stops echoing the
  generated secret into cloud-init logs. (#154)
- **Compose demo** — dropped the `working_dir` override that broke
  `uv run atp`. (#138)

## [2.0.0] - 2026-05-08

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
- **MCP tournament tools (`/mcp`)**: now require an agent-scoped token
  (`atp_a_*`) issued for an agent whose `purpose` is `"tournament"`.
  User-level tokens (`atp_u_*`), admin sessions, and tokens for
  `"benchmark"`-purpose agents are rejected with HTTP 403. The benchmark
  API (`/api/v1/benchmarks/*`) is symmetrically gated — it rejects
  tournament-purpose tokens with 403.
  See `docs/migrations/2026-04-mcp-purpose-gating.md`. (commit d0f11e26)
- **`POST /api/agents` returns `410 Gone`**: the legacy ownerless
  agent-creation endpoint that worked at v1.0.0 is permanently retired.
  Stale clients now fail loudly with `Deprecation` / `Sunset` / `Link:
  ...; rel="successor-version"` headers pointing at `POST /api/v1/agents`.
  The replacement endpoint resolves ownership from the caller's JWT and
  enforces per-user, per-purpose quotas.
  See `docs/migrations/2026-04-legacy-agents-endpoint.md`. (#53)

### Added / Changed / Fixed

For non-breaking changes between 1.0.0 and 2.0.0, see `git log v1.0.0..v2.0.0`.
Highlights: pending-tournament banner, El Farol winners dashboard + Hall of
Fame, benchmark API event streaming, agent ownership quotas, RBAC + invite
system, container-isolated code-exec evaluator, MCP tournament server.

## [1.0.0] - 2026-02-13

Initial public release of ATP (Agent Test Platform) — a framework-agnostic
platform for testing and evaluating AI agents.

### Phase 1 — Core Protocol & Runner

- ATP Protocol models (ATPRequest, ATPResponse, ATPEvent)
- YAML/JSON test suite loader with filtering and tagging
- Test runner with sandbox isolation and parallel execution
- Adapter system: HTTP, Container, CLI, LangGraph, CrewAI, AutoGen
- Evaluators: artifact matching, behavioral checks, LLM-judge
- Score aggregation with weighted scoring
- Reporters: console, JSON, HTML, JUnit

### Phase 2 — Streaming, Security & Baselines

- Event streaming with buffering and back-pressure
- Security hardening: secret redaction, input sanitization
- Statistical analysis: confidence intervals, stability metrics
- Baseline management with regression detection (Welch's t-test)
- Performance profiling, caching, and memory tracking

### Phase 3 — Dashboard & Analytics

- Web dashboard (FastAPI + SQLAlchemy) with real-time updates
- Cost tracking and analytics
- Mock tool server for deterministic testing
- Structured logging with correlation IDs
- OpenTelemetry tracing integration

### Phase 4 — Extensibility & Adapters

- Plugin ecosystem: discovery via entry points, config schemas, validation
- MCP adapter for Model Context Protocol agents
- Cloud adapters: AWS Bedrock, Google Vertex AI, Azure OpenAI
- Chaos testing framework
- Test suite generation from natural language
- CLI commands: init, generate, benchmark, budget, experiment, plugins

### Phase 5 — Game-Theoretic Evaluation

- `game-environments` sub-package with classic game theory environments
- `atp-games` integration: game runner, strategy adapters, Nash analysis
- Game evaluators, reporters, and CLI (`atp game`)
- Iterated games, tournaments, and multi-agent scenarios
- Terminal UI (TUI) for interactive test monitoring

### Phase 6 — CI/CD, Traces & SDK

- GitHub Actions composite action for CI integration
- Composite evaluators (all-of, any-of, weighted, conditional, pipeline)
- Agent replay from execution traces
- Streaming CLI live display
- Python SDK for programmatic test execution
- Multi-model comparison and cost estimation
- Benchmark suite loaders and GitHub import
- Natural language test generation
- Trace import and storage tracking

[Unreleased]: https://github.com/andrei-shtanakov/atp-platform/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/andrei-shtanakov/atp-platform/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/andrei-shtanakov/atp-platform/releases/tag/v1.0.0

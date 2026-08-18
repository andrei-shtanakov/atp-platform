# Changelog

All notable changes to the ATP Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [2.2.0] - 2026-08-18

### Added

- **`composite` runs on the benchmark plane again, under the policy that
  governs it** (ADR-008 track A). It used to resolve its leaves through the
  global registry, so nesting `pytest` under a `composite` assertion reached an
  evaluator the server allowlist exists to withhold. It now builds them through
  the resolver it is handed, passed down to any depth of nesting; with no
  resolver bound it evaluates nothing, because a fallback is precisely the hole
  being closed. The untrusted allowlist goes 19 → 20 assertion types.
- **A leaf that was not evaluated is no longer a leaf that failed.** Conditions
  yield `PASS`, `FAIL` or `UNEVALUATED`, propagating by Kleene logic: `AND` lets
  a real failure decide, `OR` lets a real pass decide, `NOT UNEVALUATED` stays
  unevaluated, and a threshold is decided only when both ends of the score
  interval agree (each unknown leaf counting as `[0, 1]`). A composite that
  cannot be decided raises the new `AssertionUnevaluated`, which the pipeline
  records as coverage with reason `indeterminate` rather than as a failed
  measurement. (#274)

- **Benchmark submissions are evaluated, and the score says what it is** — the
  benchmark plane scored 100 for any completed response; it now runs the shared
  `atp.evaluation` pipeline under the server policy and publishes
  `score_semantics` describing the result. `kind` is `aggregated_evaluation`
  with a `score_components` breakdown (keyed by assertion type) only when an
  evaluator was *applied*; otherwise it stays `completion_rate` with
  `quality_signal: false`. Anything skipped, refused, unknown or crashed is
  reported under `coverage` with a reason and never becomes a component worth
  0.0. A run scored partly by evaluation and partly by completion carries a
  `mixed_task_scores` caveat. Per-task evidence is stored in the existing
  `TaskResult.eval_results` column — no migration — with a `record_version` on
  every record; a record whose version the reader does not recognise is counted
  under `coverage.records_unreadable` rather than parsed with today's field
  names. (#272)

- **An unknown harness `kind` is flagged instead of accepted silently.** V7 was
  only half-enforced: the model `status` `Literal` rejected the mixed fixture
  before `kind` was ever examined. `KNOWN_HARNESS_KINDS` restates the
  ADR-ECO-003 vocabulary and an unfamiliar value raises `CatalogWarning` —
  a warning rather than a rejection, because `kind` describes launch mechanics
  and a catalog naming a new one must still load. (#295)

- **The model catalog validates across planes, in the shared V1..V7 vocabulary.**
  Only V1 (an agent naming an undeclared harness) was checked. An agent naming
  an undeclared model (V2), one referencing a `retired` model (V3), a duplicate
  `agent_id` (V4) and a routable agent under a non-routable harness (V5) are now
  errors; a reference to a `deprecated` model (V6) raises `CatalogWarning`,
  since deprecated still runs but must not pass silently. The rule set is
  exercised by the devtools-owned conformance fixtures, vendored pinned under
  `tests/unit/model_catalog/fixtures/catalog-conformance/v1/` and verified
  against their sha256 manifest. (#293)
- **A shippable model catalog with its own CLI** — `atp models init` writes a
  starter catalog, `atp models list` reads the resolved one. Resolution is
  `$ATP_CATALOG` → XDG (`~/.config/atp/agents-catalog.toml`) → fail loud, with no
  bundled default: a hidden fallback would let the platform answer with a model
  set nobody chose. The evaluator's default model resolves through the catalog's
  `[defaults]` plane, tolerant of a missing or broken optional catalog. (#237, #239)
- **Cloud-`$` cost derived from stored token usage**, behind the `[pricing]`
  extra. A cache-aware LiteLLM pricer turns per-class usage into honest dollars
  and is derived-not-stored, so a price change re-derives without re-running a
  sweep. Measured usage only — an estimated fallback would put invented numbers
  next to real ones. (#236)
- **`ATP_SERVER_PROFILE=eco`** — an API-only benchmark server (Benchmark API plus
  auth/tokens/agents; no tournaments, MCP, or HTML UI) for deployments that want
  the benchmark plane without the rest. Tournament dependencies moved behind an
  `atp-dashboard[tournaments]` extra, which `atp-platform[dashboard]` still
  pulls. (#287, #288)
- **A usage-capture seam in the orchestrator** (ADR-ECO-003e M0), observe-only:
  a typed `UsageRecord` contract, a JSONL sink, and `python -m atp.cost.probe_report`
  for coverage. It records what the run actually reported rather than what a
  price table guesses. (#251)
- **`receipt_chain` checker** — verifies Libretto `receipts.jsonl` hash chains, so
  a run's own ledger becomes evaluation input rather than a claim to trust. The
  compatibility contract is vendored under `method/contract/openprose/`. (#252)
- **A grader spine for agent-eval-cases**: a uniform `CaseVerdict`, a checker
  registry separate from the evaluator registry, a shared output envelope, and a
  task-type taxonomy. Deterministic checkers gate; rubric scoring does not. (#176, #177)
- **A dimensioned evaluation store, and a dashboard that reads it** — canonical
  persistence with `task_type`/`language` as real columns, plus an eval
  leaderboard and trend view. (#179, #180, #181)
- **The benchmark plane got its safety boundary**: a versioned score contract for
  the Maestro handshake, a bounded artifact workspace for untrusted submissions,
  a deterministic server-side assertion allowlist, and an explicit composition
  root that hands evaluation capability in instead of letting call sites reach
  for a global registry. (#266, #269, #270, #271)
- **Force-advance a tournament round** from the admin UI, with the endpoint
  behind it. (#247)
- **Terraform IaC for the all-in-AWS Bedrock demo**, replacing the hand-run
  `examples/aws-cloud` walkthrough. (#167)
- **A deterministic `req-extraction` example** that grades without an LLM. (#166)
- **The `atp-method` evaluation harness grew a working contour** (dev-side, shipped
  via the `atp-method` package): a `SUITE.lock.toml` golden-suite lock that refuses
  a paid run on suite drift; Path A corpus grounding that points a CLI's *native*
  tools at a verified corpus (`claude_code`, `codex_cli`, `pi`, `opencode`); a
  wider agent roster including the first open model promoted to routable; raw
  per-case stdout/stderr capture with `error_class` classification and an
  `infra_error_rate` signal, so capability and reliability stay separable; and
  per-class token usage with `usage_source` in the emitted payload. (#171-#174,
  #176-#177, #182-#183, #192-#193, #195, #204, #215-#216, #221, #223, #227-#231,
  #234-#235, #242)

### Fixed

- **`$ATP_CATALOG` pointing at a missing file no longer falls through to XDG.**
  An explicit catalog path is an instruction, and quietly loading a different
  file — or reporting "not configured" — ignored it. Resolution now raises the
  new `CatalogMissingFileError`; `resolve_default_model()` stays tolerant and
  degrades to `None` with a warning. (#293)
- **`POST /api/v1/runs/{id}/submit` returns 422, not 500, for a malformed
  response body.** `SubmitRequest.response` is a bare dict, so the protocol was
  first checked inside the handler, where a `ValidationError` became an
  unhandled exception. A self-service caller now gets told what is wrong. (#272)
- **Non-JSON values (datetimes) in artifact `data_json` no longer break the
  dashboard** — they are coerced on the way in. (#169)
- **The pipe-check harness stopped paying 3× for 1× of signal** — `runs=N` graded
  only one run per case; all runs are graded now. Alongside: cache tokens count
  toward the `claude_code` shim's total, the sweep no longer silently reports
  infra failures as 0.0 scores, case artifacts actually reach the model (it was
  reviewing an empty diff), `--task-type` cannot contradict the suite's own, and
  the `opencode` shim isolates its data dir per invocation to stop SQLite lock
  contention between concurrent runs. (#184, #202, #213, #222, #232, #233)

### Changed

- **`harnesses.*.shim` is optional in the model-catalog schema.** It is ATP
  sweep machinery, not part of the cross-repo catalog contract, and requiring it
  rejected contract-valid catalogs for a reason no sibling loader shares — which
  would have made the conformance set pass for the wrong reason. The requirement
  moved to `method/run_pipe_check.py`, where shims are actually spawned. (#293)
- **`score_semantics` and `score_components` are required** on
  `RunResponse` and `RunStatusResponse`. Consumers see no wire change; a
  default would have let a call site publish a label it never derived. (#272)
- **The untrusted-submission allowlist narrows from 25 to 19 assertion types.**
  `filesystem` addresses a directory named in the suite rather than the sandbox
  the submission was materialized into, so on this plane it cannot measure the
  agent's work and answers an existence question about the server's own disk.
  `composite` resolves its leaves through the global registry, so nesting
  `pytest` under it would reach an evaluator the policy withholds. Both are now
  classified in `atp.evaluation.vocabulary` and excluded by derivation. The CLI
  (`trusted_local`) is unaffected. (#272)

- **LearningEvent v1 contract (RD-007 M1a):** `method/contract/learning-event-v1.schema.json` —
  observational learning events (gap / eval-signal / recommendation) with producer provenance,
  append-only `source` pointers, optional `evidence_refs` (vendored pinned copy of Maestro
  EvidenceRef v1 incl. `gate-verdict`), and closed graduation-target vocabulary. Events never
  mutate governed knowledge; graduation happens only via reviewed PR. Plus `.github/CODEOWNERS`
  declaring the governed paths (v1 acceptance) and contract tests with fixtures.
- **`track_response_cost` is deprecated** and emits a `DeprecationWarning`; the
  usage-capture seam above replaces it. Removal is scheduled for ADR-ECO-003e M1. (#251)

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

[Unreleased]: https://github.com/andrei-shtanakov/atp-platform/compare/v2.2.0...HEAD
[2.2.0]: https://github.com/andrei-shtanakov/atp-platform/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/andrei-shtanakov/atp-platform/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/andrei-shtanakov/atp-platform/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/andrei-shtanakov/atp-platform/releases/tag/v1.0.0

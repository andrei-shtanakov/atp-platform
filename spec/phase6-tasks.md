# Phase 6: Polish, SDK & Ecosystem — Tasks

> Proposals for ATP Platform improvement (Q1–Q2 2026)
> Based on comprehensive codebase audit (2026-02-12)

## Legend

**Priority:**
| Emoji | Code | Description |
|-------|------|-------------|
| 🔴 | P0 | Critical — blocks release |
| 🟠 | P1 | High — needed for full functionality |
| 🟡 | P2 | Medium — improves experience |
| 🟢 | P3 | Low — nice to have |

**Status:**
| Emoji | Status | Description |
|-------|--------|-------------|
| ⬜ | TODO | Not started |
| 🔄 | IN PROGRESS | In work |
| ✅ | DONE | Completed |
| ⏸️ | BLOCKED | Waiting on dependency |

---

## Milestone 13: Release Readiness & Tech Debt

### TASK-1301: Add LICENSE file
🔴 P0 | ⬜ TODO | Est: 0.5h

**Description:**
README claims MIT License but no LICENSE file exists in the repository. This is a legal blocker for public release and adoption.

**Checklist:**
- [ ] Create LICENSE file with MIT License text
- [ ] Verify copyright holder and year
- [ ] Ensure license matches README claim

**Depends on:** —
**Blocks:** public release

---

### TASK-1302: Fix pyrefly enforcement in CI
🔴 P0 | ⬜ TODO | Est: 4-6h

**Description:**
pyrefly type checking is effectively disabled: CI workflow uses `continue-on-error: true`, pre-commit hook uses `|| exit 0`. Parent project has ~29 pre-existing errors that need fixing.

**Checklist:**
- [ ] Audit all ~29 pre-existing pyrefly errors
- [ ] Fix type errors in atp/ source code
- [ ] Remove `continue-on-error: true` from CI workflow
- [ ] Remove `|| exit 0` from pre-commit hook
- [ ] Verify CI passes with enforced type checking

**Depends on:** —
**Blocks:** —

---

### TASK-1303: Complete dashboard v2 migration, remove v1 monolith
🔴 P0 | ⬜ TODO | Est: 6-8h

**Description:**
`atp/dashboard/app.py` (4,864 lines) is the largest file in the codebase. Dashboard v2 refactoring (`atp/dashboard/v2/`) is started with proper routes/services separation (28 route files, 4 service files) but v1 is still present and deprecated.

**Checklist:**
- [ ] Audit v1 `app.py` for any functionality not yet in v2
- [ ] Migrate any missing functionality to v2
- [ ] Audit v1 `api.py` (101 lines, deprecated) for missing endpoints
- [ ] Update all imports and references from v1 to v2
- [ ] Remove `atp/dashboard/app.py` (deprecated v1)
- [ ] Remove `atp/dashboard/api.py` (deprecated v1)
- [ ] Remove `ATP_DASHBOARD_V2` feature flag (v2 becomes default)
- [ ] Update documentation and migration guide
- [ ] Run full test suite to verify nothing breaks

**Depends on:** —
**Blocks:** —

---

### TASK-1304: Fix evaluator entry points in pyproject.toml
🔴 P0 | ⬜ TODO | Est: 1-2h

**Description:**
Only 4 of 9+ evaluators are registered as entry points in `pyproject.toml`. Missing: security, factuality, filesystem, style, performance. Plugin discovery via `atp.evaluators` entry point group will fail for these.

**Checklist:**
- [ ] Audit all evaluator classes in `atp/evaluators/`
- [ ] Add missing entry points: security, factuality, filesystem, style, performance
- [ ] Verify plugin discovery finds all evaluators
- [ ] Write test to ensure all evaluators are discoverable

**Depends on:** —
**Blocks:** —

---

### TASK-1305: Verify and complete TASK-919 (Alympics Benchmark)
🔴 P0 | ⬜ TODO | Est: 3-4h

**Description:**
TASK-919 (Alympics-Style Benchmark Suite) executor recorded "success" but `phase5-tasks.md` shows IN_PROGRESS with all 8 checklist items unchecked. Need to verify actual implementation status and complete if necessary.

**Checklist:**
- [ ] Check if `alympics_lite.yaml` exists and is functional
- [ ] Check if `atp benchmark --suite=alympics` command works
- [ ] Check for composite scoring implementation
- [ ] Verify categories: strategic reasoning, cooperation, fairness, robustness
- [ ] Run integration test with builtin strategies
- [ ] Complete any missing implementation
- [ ] Update `spec/phase5-tasks.md` to reflect actual status

**Depends on:** —
**Blocks:** —

---

### TASK-1306: Implement lazy-loading for adapters
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
`atp/adapters/registry.py` eagerly imports ALL adapter modules at startup (vertex, bedrock, azure_openai, etc.), pulling in boto3, google-cloud-aiplatform, etc. This causes import errors when optional deps aren't installed and slows startup.

**Checklist:**
- [ ] Refactor adapter registry to use lazy imports
- [ ] Import adapter modules only when requested via `create(adapter_type)`
- [ ] Handle ImportError gracefully with clear message about missing deps
- [ ] Add test: import atp without optional deps doesn't raise errors
- [ ] Update adapter documentation

**Depends on:** —
**Blocks:** [TASK-1307]

---

### TASK-1307: Slim down dependencies with optional extras
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
40 runtime dependencies is heavy. Many users won't need all adapters/features. Extract heavy optional deps into extras groups in pyproject.toml.

**Checklist:**
- [ ] Define extras groups: `[cloud]` (boto3, google-cloud-aiplatform), `[enterprise]` (authlib, python3-saml), `[dashboard]` (sqlalchemy, asyncpg), `[analytics]` (openpyxl), `[all]`
- [ ] Move cloud adapter deps to `[cloud]` extra
- [ ] Move enterprise deps (SSO, SAML) to `[enterprise]` extra
- [ ] Move dashboard deps to `[dashboard]` extra
- [ ] Keep core deps minimal (pydantic, click, rich, structlog, httpx, pyyaml)
- [ ] Update installation docs with extras guidance
- [ ] Verify `uv add atp-platform[all]` installs everything

**Depends on:** [TASK-1306]
**Blocks:** —

---

### TASK-1308: Split large adapter files into submodules
🟡 P2 | ⬜ TODO | Est: 4-6h

**Description:**
Three adapter files are excessively large: bedrock.py (34K), azure_openai.py (34K), vertex.py (37K). Should be split into submodules following the MCP adapter pattern (`atp/adapters/mcp/`).

**Checklist:**
- [ ] Refactor `bedrock.py` → `atp/adapters/bedrock/` (adapter, models, auth, utils)
- [ ] Refactor `vertex.py` → `atp/adapters/vertex/` (adapter, models, auth, utils)
- [ ] Refactor `azure_openai.py` → `atp/adapters/azure_openai/` (adapter, models, auth, utils)
- [ ] Update imports and entry points
- [ ] Verify all tests still pass

**Depends on:** —
**Blocks:** —

---

### TASK-1309: Fix asyncio_mode mismatch
🟡 P2 | ⬜ TODO | Est: 1h

**Description:**
`pyproject.toml` sets `asyncio_mode = "auto"` but project guidelines require using anyio for async testing, not asyncio. Configuration should be consistent.

**Checklist:**
- [ ] Audit existing async tests for anyio vs asyncio usage
- [ ] Align pyproject.toml async config with anyio approach
- [ ] Verify all async tests pass with corrected config

**Depends on:** —
**Blocks:** —

---

### TASK-1310: Implement GitHub Import for marketplace
🟡 P2 | ⬜ TODO | Est: 3-4h

**Description:**
`atp/dashboard/v2/routes/marketplace.py:1425` returns "not yet implemented" for GitHub import. Route is defined but needs actual implementation.

**Checklist:**
- [ ] Implement GitHub API integration (repo URL → raw YAML fetch)
- [ ] Parse and validate imported YAML as test suite
- [ ] Handle authentication (public repos + optional token for private)
- [ ] Handle versioning (import specific branch/tag/commit)
- [ ] Update existing test at `tests/integration/dashboard/test_marketplace_routes.py:943`
- [ ] Add error handling for invalid repos/files

**Depends on:** —
**Blocks:** —

---

### TASK-1311: Implement real storage tracking for quotas
🟢 P3 | ⬜ TODO | Est: 2h

**Description:**
`atp/dashboard/tenancy/quotas.py:208` — `_get_storage_usage()` always returns `0.0` (placeholder). Needs actual storage calculation.

**Checklist:**
- [ ] Implement actual storage usage calculation per tenant
- [ ] Track test results, artifacts, and logs storage
- [ ] Add caching for storage calculation (expensive operation)
- [ ] Write tests

**Depends on:** —
**Blocks:** —

---

## Milestone 14: Python SDK & Programmatic API

### TASK-1401: Core Python SDK
🟠 P1 | ⬜ TODO | Est: 6-8h

**Description:**
Extract a clean Python API that doesn't go through CLI. Enable `atp.run(suite, adapter=...)` for Jupyter notebooks, custom pipelines, and programmatic use.

**Checklist:**
- [ ] Design SDK API surface: `atp.run()`, `atp.evaluate()`, `atp.load_suite()`
- [ ] Create `atp/sdk/__init__.py` with public API
- [ ] Implement `atp.run(suite_path, adapter_config) → TestResults`
- [ ] Implement `atp.evaluate(response, evaluator_config) → EvalResults`
- [ ] Implement `atp.load_suite(path) → TestSuite` for programmatic manipulation
- [ ] Add async variants: `atp.arun()`, `atp.aevaluate()`
- [ ] Write comprehensive tests
- [ ] Write SDK quickstart guide
- [ ] Add Jupyter notebook example

**Depends on:** —
**Blocks:** [TASK-1402]

---

### TASK-1402: Multi-model comparison mode
🟠 P1 | ⬜ TODO | Est: 4-5h

**Description:**
Built-in way to run the same suite against multiple models/configs in a single command and produce a side-by-side comparison report.

**Checklist:**
- [ ] Design comparison config format (list of adapter configs)
- [ ] Implement `atp test --compare config1.yaml config2.yaml ...`
- [ ] Run same suite against each adapter in parallel
- [ ] Generate comparison report: table with scores per model per test
- [ ] Add statistical significance tests between models
- [ ] HTML comparison report with charts
- [ ] Write tests and documentation

**Depends on:** [TASK-1401]
**Blocks:** —

---

### TASK-1403: Pre-run cost estimation
🟡 P2 | ⬜ TODO | Est: 3-4h

**Description:**
Estimate LLM API costs before running a test suite, based on prompt sizes, model pricing, and number of runs.

**Checklist:**
- [ ] Implement `atp estimate --suite=suite.yaml --adapter=config.yaml`
- [ ] Estimate token counts from test definitions (input prompts, expected output sizes)
- [ ] Apply model pricing from `atp/analytics/cost.py` pricing config
- [ ] Show estimated cost range (min/max based on output variation)
- [ ] Add `--budget-check` flag to abort if estimate exceeds budget
- [ ] Write tests

**Depends on:** —
**Blocks:** —

---

## Milestone 15: Benchmark Integration & Evaluation

### TASK-1501: Standard benchmark loaders
🟠 P1 | ⬜ TODO | Est: 6-8h

**Description:**
Add connectors to popular benchmarks (SWE-bench, HumanEval, GAIA, MMLU) that convert them to ATP test suites. Dramatically increases platform utility.

**Checklist:**
- [ ] Design benchmark loader interface: `BenchmarkLoader.load() → TestSuite`
- [ ] Implement HumanEval loader (code generation benchmark)
- [ ] Implement SWE-bench loader (software engineering benchmark)
- [ ] Implement GAIA loader (general AI assistant benchmark)
- [ ] Implement MMLU loader (knowledge/reasoning benchmark)
- [ ] CLI: `atp benchmark load humaneval --adapter=config.yaml`
- [ ] Caching: download benchmark data once, reuse
- [ ] Write tests for each loader
- [ ] Documentation with examples

**Depends on:** —
**Blocks:** —

---

### TASK-1502: Evaluator composition with boolean logic
🟡 P2 | ⬜ TODO | Est: 3-4h

**Description:**
Allow combining evaluators with boolean/arithmetic logic for complex pass/fail criteria.

**Checklist:**
- [ ] Design composition DSL: `artifact AND llm_judge > 0.8 AND NOT security_violation`
- [ ] Implement `CompositeEvaluator` with AND, OR, NOT operators
- [ ] Support threshold conditions on individual evaluator scores
- [ ] Add YAML syntax for composite evaluators in test suites
- [ ] Write tests for various composition patterns
- [ ] Documentation

**Depends on:** —
**Blocks:** —

---

## Milestone 16: Agent Debugging & Observability

### TASK-1601: Agent Replay & Trace Visualization
🟠 P1 | ⬜ TODO | Est: 8-10h

**Description:**
Record agent execution (all ATP events) and provide step-by-step trace visualization in the dashboard. Enable replay for debugging.

**Checklist:**
- [ ] Design trace recording format (extend existing streaming/events)
- [ ] Implement trace recorder: capture all ATPEvents during test run
- [ ] Implement trace storage (file-based + database)
- [ ] Dashboard: trace viewer page with step-by-step timeline
- [ ] Dashboard: show tool calls, LLM requests, reasoning steps
- [ ] Dashboard: side-by-side comparison of two traces
- [ ] Implement `atp replay <trace_id>` CLI command
- [ ] Implement trace search/filter in dashboard
- [ ] Write tests

**Depends on:** —
**Blocks:** —

---

### TASK-1602: Streaming results to CLI during test runs
🟡 P2 | ⬜ TODO | Est: 3-4h

**Description:**
WebSocket infrastructure exists in dashboard v2 but is not integrated into CLI for live result streaming. Show real-time progress during `atp test` runs.

**Checklist:**
- [ ] Implement rich live display for `atp test` with real-time updates
- [ ] Show per-test progress: running, passed, failed
- [ ] Show streaming evaluator scores as they complete
- [ ] Show token/cost accumulation in real-time
- [ ] Add `--live` flag to enable streaming output
- [ ] Write tests

**Depends on:** —
**Blocks:** —

---

## Milestone 17: CI/CD & Ecosystem

### TASK-1701: GitHub Action for ATP testing
🟠 P1 | ⬜ TODO | Est: 4-5h

**Description:**
Create a ready-to-use GitHub Action (`atp-test-action`) that runs ATP tests in CI/CD with automatic PR commenting.

**Checklist:**
- [ ] Create `action.yml` with inputs: suite, adapter, threshold, budget
- [ ] Implement Docker-based action with ATP pre-installed
- [ ] Auto-comment on PR with test results summary
- [ ] Support baseline comparison (fail if regression detected)
- [ ] Support budget limits (fail if cost exceeds budget)
- [ ] Add badge generation for README
- [ ] Create example workflows for common setups
- [ ] Publish to GitHub Marketplace
- [ ] Documentation

**Depends on:** —
**Blocks:** —

---

### TASK-1702: Test generation from production traces
🟡 P2 | ⬜ TODO | Est: 5-6h

**Description:**
Import production agent traces (from LangSmith, Arize, OpenTelemetry) and auto-generate regression test suites.

**Checklist:**
- [ ] Design trace import interface: `TraceImporter.import_traces() → TestSuite`
- [ ] Implement LangSmith trace importer
- [ ] Implement OpenTelemetry trace importer
- [ ] Implement trace → test conversion (extract input/expected output pairs)
- [ ] Implement deduplication and parameterization
- [ ] CLI: `atp generate from-traces --source=langsmith --project=X`
- [ ] Write tests
- [ ] Documentation

**Depends on:** —
**Blocks:** —

---

### TASK-1703: VS Code Extension
🟢 P3 | ⬜ TODO | Est: 8-12h

**Description:**
VS Code extension for ATP: YAML suite highlighting, test running, result visualization.

**Checklist:**
- [ ] Create VS Code extension scaffold (TypeScript)
- [ ] Implement YAML schema validation for ATP test suites
- [ ] Implement "Run Test Suite" command from editor
- [ ] Implement test results panel with pass/fail visualization
- [ ] Implement code lens for individual test cases
- [ ] Add snippets for common test suite patterns
- [ ] Syntax highlighting for ATP-specific YAML fields
- [ ] Publish to VS Code Marketplace
- [ ] Documentation

**Depends on:** —
**Blocks:** —

---

### TASK-1704: Natural Language Test Generation
🟡 P2 | ⬜ TODO | Est: 4-5h

**Description:**
Generate YAML test suites from natural language descriptions via LLM. Lower the barrier to creating tests.

**Checklist:**
- [ ] Design prompt template for test generation
- [ ] Implement `atp generate from-description "test that agent can search the web and summarize results"`
- [ ] Support multi-test generation from a paragraph of requirements
- [ ] Validate generated YAML against ATP schema
- [ ] Interactive mode: suggest tests, user confirms/edits
- [ ] Write tests
- [ ] Documentation

**Depends on:** —
**Blocks:** —

---

## Dependency Graph

```
TASK-1301 (LICENSE) ─────────────────────────── independent
TASK-1302 (pyrefly) ─────────────────────────── independent
TASK-1303 (v2 migration) ───────────────────── independent
TASK-1304 (entry points) ───────────────────── independent
TASK-1305 (TASK-919 verify) ────────────────── independent

TASK-1306 (lazy adapters) ──► TASK-1307 (optional extras)
TASK-1308 (split adapters) ─────────────────── independent
TASK-1309 (asyncio fix) ────────────────────── independent
TASK-1310 (GitHub import) ──────────────────── independent
TASK-1311 (storage tracking) ───────────────── independent

TASK-1401 (Python SDK) ──► TASK-1402 (multi-model compare)
TASK-1403 (cost estimation) ────────────────── independent

TASK-1501 (benchmark loaders) ──────────────── independent
TASK-1502 (evaluator composition) ──────────── independent

TASK-1601 (agent replay) ──────────────────── independent
TASK-1602 (streaming CLI) ─────────────────── independent

TASK-1701 (GitHub Action) ─────────────────── independent
TASK-1702 (trace import) ──────────────────── independent
TASK-1703 (VS Code ext) ───────────────────── independent
TASK-1704 (NL test gen) ───────────────────── independent
```

---

## Summary

| Milestone | Tasks | Total Est. Hours |
|-----------|-------|------------------|
| M13: Release Readiness & Tech Debt | TASK-1301..1311 | ~30-42h |
| M14: Python SDK & Programmatic API | TASK-1401..1403 | ~13-17h |
| M15: Benchmark Integration & Evaluation | TASK-1501..1502 | ~9-12h |
| M16: Agent Debugging & Observability | TASK-1601..1602 | ~11-14h |
| M17: CI/CD & Ecosystem | TASK-1701..1704 | ~21-28h |
| **Total** | **21 tasks** | **~84-113h** |

---

## Recommended Execution Order

### Phase 6.1 (Weeks 1-3): Release Blockers
1. TASK-1301 (LICENSE) — 30 min
2. TASK-1304 (entry points) — 1-2h
3. TASK-1305 (verify TASK-919) — 3-4h
4. TASK-1302 (pyrefly) — 4-6h
5. TASK-1309 (asyncio fix) — 1h

### Phase 6.2 (Weeks 3-6): Architecture Cleanup
6. TASK-1306 (lazy adapters) → TASK-1307 (optional extras)
7. TASK-1303 (v2 migration, remove v1 monolith)
8. TASK-1308 (split large adapters)

### Phase 6.3 (Weeks 7-10): SDK & Features
9. TASK-1401 (Python SDK) → TASK-1402 (multi-model compare)
10. TASK-1501 (standard benchmark loaders)
11. TASK-1403 (cost estimation)
12. TASK-1310 (GitHub import)

### Phase 6.4 (Weeks 11-14): Debugging & Ecosystem
13. TASK-1601 (agent replay & traces)
14. TASK-1701 (GitHub Action)
15. TASK-1602 (streaming CLI)
16. TASK-1502 (evaluator composition)

### Phase 6.5 (Weeks 15+): Nice to Have
17. TASK-1702 (trace import from LangSmith/Arize)
18. TASK-1704 (NL test generation)
19. TASK-1703 (VS Code extension)
20. TASK-1311 (storage tracking)

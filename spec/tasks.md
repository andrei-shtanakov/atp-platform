# Tasks

> Задачи с приоритетами, зависимостями и трассировкой к требованиям

## Легенда

**Приоритет:**
- 🔴 P0 — Critical, блокирует релиз
- 🟠 P1 — High, нужно для полноценного использования
- 🟡 P2 — Medium, улучшение опыта
- 🟢 P3 — Low, nice to have

**Статус:**
- ⬜ TODO
- 🔄 IN PROGRESS
- ✅ DONE
- ⏸️ BLOCKED

---

## Definition of Done (для КАЖДОЙ задачи)

> ⚠️ Задача НЕ считается завершённой без выполнения этих пунктов:

- [ ] **Unit tests** — покрытие ≥80% нового кода
- [ ] **Tests pass** — все тесты проходят локально
- [ ] **Integration test** — если изменены публичные интерфейсы
- [ ] **CI green** — pipeline проходит
- [ ] **Docs updated** — документация актуальна
- [ ] **Code review** — PR approved

---

## Testing Tasks (Cross-cutting)

### TASK-100: Test Infrastructure Setup
🔴 P0 | ✅ DONE | Est: 2d

**Description:**
Настроить тестовую инфраструктуру: pytest, fixtures, CI.

**Checklist:**
- [ ] pytest + pytest-asyncio + pytest-cov setup
- [ ] pytest-mock для моков
- [ ] conftest.py со shared fixtures
- [ ] tests/ directory structure
- [ ] GitHub Actions workflow
- [ ] Coverage reporting (≥80% gate)
- [ ] Pre-commit hooks (ruff, mypy)

**Traces to:** [NFR-000]
**Depends on:** —
**Blocks:** All other tasks (soft dependency)

---

### TASK-101: Contract Tests
🔴 P0 | ✅ DONE | Est: 2d

**Description:**
Тесты контракта ATP Protocol — валидация схем.

**Checklist:**
- [ ] Valid ATP Request fixtures (10+ cases)
- [ ] Invalid ATP Request fixtures (edge cases)
- [ ] Valid ATP Response fixtures
- [ ] Invalid ATP Response fixtures
- [ ] ATP Event fixtures для всех типов
- [ ] JSON Schema validation tests
- [ ] Pydantic model roundtrip tests
- [ ] Protocol version handling tests

**Traces to:** [REQ-001], [REQ-002], [NFR-000]
**Depends on:** [TASK-001], [TASK-100]
**Blocks:** —

---

### TASK-102: Integration Test Suite
🟠 P1 | ✅ DONE | Est: 3d

**Description:**
Integration тесты для ключевых компонентов.

**Checklist:**
- [ ] HTTP Adapter + mock server
- [ ] Container Adapter + test Docker image
- [ ] Full test run (loader → runner → evaluators → reporter)
- [ ] Timeout handling scenarios
- [ ] Error recovery scenarios
- [ ] Multi-run statistics accuracy

**Traces to:** [NFR-000]
**Depends on:** [TASK-003], [TASK-006], [TASK-100]
**Blocks:** —

---

### TASK-103: E2E Test Suite
🟠 P1 | ✅ DONE | Est: 2d

**Description:**
End-to-end тесты критических user journeys.

**Checklist:**
- [ ] `atp test` with sample agent — happy path
- [ ] `atp test` with failing tests
- [ ] `atp test` with timeout
- [ ] `atp validate` command
- [ ] JSON report generation
- [ ] Exit codes verification

**Traces to:** [NFR-000], [REQ-030]
**Depends on:** [TASK-014], [TASK-100]
**Blocks:** —

---

## Milestone 1: MVP

### TASK-001: ATP Protocol Models
🔴 P0 | ✅ DONE | Est: 3d

**Description:**
Реализовать Pydantic модели для ATP Request, Response, Event.

**Checklist:**
- [x] ATPRequest model с валидацией
- [x] ATPResponse model со всеми статусами
- [x] ATPEvent model для всех event types
- [x] Artifact models (file, structured, reference)
- [x] Metrics model
- [x] JSON Schema генерация из моделей

**Tests (Definition of Done):**
- [x] Unit tests: serialization/deserialization
- [x] Unit tests: validation (valid + invalid inputs)
- [x] Unit tests: edge cases (null, empty, large)
- [x] Coverage ≥80%

**Traces to:** [REQ-001], [REQ-002]
**Depends on:** [TASK-100]
**Blocks:** [TASK-003], [TASK-006], [TASK-101]

---

### TASK-002: Event Streaming Support
🟠 P1 | ✅ DONE | Est: 2d

**Description:**
Добавить поддержку streaming событий в протоколе и адаптерах.

**Checklist:**
- [x] AsyncIterator interface для событий
- [x] SSE parsing для HTTP
- [x] stderr parsing для containers
- [x] Event ordering validation
- [x] Event buffering и replay

**Traces to:** [REQ-003]
**Depends on:** [TASK-001]
**Blocks:** [TASK-007] (behavior evaluator needs trace)

---

### TASK-003: Core Adapters
🔴 P0 | ✅ DONE | Est: 5d

**Description:**
Реализовать HTTP, Container и CLI адаптеры.

**Checklist:**
- [ ] AgentAdapter base class
- [ ] HTTPAdapter
  - [ ] Sync execute
  - [ ] SSE streaming
  - [ ] Timeout handling
  - [ ] Health check
- [ ] ContainerAdapter
  - [ ] Docker client integration
  - [ ] stdin/stdout/stderr handling
  - [ ] Resource limits
  - [ ] Cleanup on completion
- [ ] CLIAdapter
  - [ ] Subprocess management
  - [ ] File-based I/O
- [ ] Adapter registry

**Tests (Definition of Done):**
- [ ] Unit tests: HTTPAdapter with httpx mock
- [ ] Unit tests: ContainerAdapter with Docker mock
- [ ] Unit tests: CLIAdapter with subprocess mock
- [ ] Unit tests: timeout scenarios
- [ ] Unit tests: error handling
- [ ] Integration test: HTTPAdapter + real HTTP server
- [ ] Integration test: ContainerAdapter + test Docker image
- [ ] Coverage ≥80%

**Traces to:** [REQ-010], [REQ-011]
**Depends on:** [TASK-001], [TASK-100]
**Blocks:** [TASK-006], [TASK-102]

---

### TASK-004: Test Loader
🔴 P0 | ✅ DONE | Est: 4d

**Description:**
Парсинг и валидация YAML test definitions.

**Checklist:**
- [ ] YAML parser с ruamel.yaml
- [ ] TestDefinition model
- [ ] TestSuite model
- [ ] Defaults inheritance
- [ ] Variable substitution (${VAR})
- [ ] JSON Schema для валидации
- [ ] Error messages с line numbers
- [ ] Unit tests для edge cases

**Traces to:** [REQ-020], [REQ-021]
**Depends on:** —
**Blocks:** [TASK-006]

---

### TASK-005: Tags и Filtering
🟠 P1 | ✅ DONE | Est: 1d

**Description:**
Фильтрация тестов по tags.

**Checklist:**
- [ ] Tag parsing в test definitions
- [ ] CLI --tags option
- [ ] Include logic (--tags=smoke,core)
- [ ] Exclude logic (--tags=!slow)
- [ ] Combination logic (AND/OR)

**Traces to:** [REQ-022]
**Depends on:** [TASK-004]
**Blocks:** —

---

### TASK-006: Test Runner Core
🔴 P0 | ✅ DONE | Est: 5d

**Description:**
Основной runner для выполнения тестов.

**Checklist:**
- [x] TestOrchestrator class
- [x] Single test execution
- [x] Suite execution
- [x] Sandbox management
  - [x] Docker container lifecycle (placeholder - uses temp dirs)
  - [x] Workspace mounting
  - [x] Cleanup
- [x] Timeout enforcement
  - [x] Soft timeout (asyncio timeout)
  - [x] Hard timeout (via sandbox config)
- [x] Result collection
- [x] Error handling и recovery
- [x] Progress reporting callback

**Traces to:** [REQ-030], [REQ-032]
**Depends on:** [TASK-001], [TASK-003], [TASK-004]
**Blocks:** [TASK-007], [TASK-009]

---

### TASK-007: Basic Evaluators
🔴 P0 | ✅ DONE | Est: 4d

**Description:**
Artifact и Behavior evaluators.

**Checklist:**
- [ ] Evaluator base class
- [ ] EvalResult, EvalCheck models
- [ ] ArtifactEvaluator
  - [ ] artifact_exists
  - [ ] contains (text)
  - [ ] contains (regex)
  - [ ] min_length / max_length
  - [ ] sections_exist (markdown)
  - [ ] artifact_schema (JSON Schema)
- [ ] BehaviorEvaluator
  - [ ] must_use_tools
  - [ ] must_not_use_tools
  - [ ] max_tool_calls
  - [ ] max_steps
  - [ ] no_errors
- [ ] Evaluator registry

**Tests (Definition of Done):**
- [ ] Unit tests: ArtifactEvaluator — each check type
- [ ] Unit tests: ArtifactEvaluator — pass/fail cases
- [ ] Unit tests: BehaviorEvaluator — each check type
- [ ] Unit tests: BehaviorEvaluator — edge cases
- [ ] Unit tests: EvalResult aggregation
- [ ] Test fixtures: sample artifacts, traces
- [ ] Coverage ≥80%

**Traces to:** [REQ-040], [REQ-041]
**Depends on:** [TASK-001], [TASK-006], [TASK-100]
**Blocks:** [TASK-008], [TASK-102]

---

### TASK-008: Scoring Aggregator
🟠 P1 | ✅ DONE | Est: 2d

**Description:**
Агрегация результатов evaluators в composite score.

**Checklist:**
- [x] ScoreAggregator class
- [x] Weight configuration
- [x] Quality score calculation
- [x] Completeness score calculation
- [x] Efficiency normalization
- [x] Cost normalization
- [x] Final score 0-100
- [x] Score breakdown in results

**Traces to:** [REQ-043]
**Depends on:** [TASK-007]
**Blocks:** [TASK-009]

---

### TASK-009: Basic Reporters
🔴 P0 | ✅ DONE | Est: 3d

**Description:**
Console и JSON reporters.

**Checklist:**
- [x] Reporter base class
- [x] ConsoleReporter
  - [x] Colored output (rich/click)
  - [x] Progress during execution
  - [x] Summary table
  - [x] Failed checks details
  - [x] Verbose mode
- [x] JSONReporter
  - [x] Full result structure
  - [x] File output
  - [x] Stable format (documented)
- [x] Reporter selection via CLI

**Traces to:** [REQ-050], [REQ-051]
**Depends on:** [TASK-006], [TASK-008]
**Blocks:** —

---

### TASK-014: CLI Implementation
🔴 P0 | ✅ DONE | Est: 3d

**Description:**
CLI interface с Click/Typer.

**Checklist:**
- [x] Main entry point
- [x] `atp test` command
  - [x] --agent option
  - [x] --suite option (via positional arg)
  - [x] --tags option
  - [x] --runs option
  - [x] --parallel option
  - [x] --output option
  - [x] --output-file option
  - [x] --verbose flag
  - [x] --fail-fast flag
- [x] `atp validate` command
- [x] `atp version` command
- [x] `atp list-agents` command
- [x] Config file loading (atp.config.yaml)
- [x] Exit codes (0=success, 1=failures, 2=error)
- [x] Help text и examples

**Traces to:** [REQ-030]
**Depends on:** [TASK-006], [TASK-009]
**Blocks:** —

---

### TASK-015: Documentation (MVP)
🔴 P0 | ✅ DONE | Est: 3d

**Description:**
Минимальная документация для MVP.

**Checklist:**
- [ ] README с quick start
- [ ] Installation guide
- [ ] Basic usage examples
- [ ] Test format reference
- [ ] Adapter configuration
- [ ] 3+ example test suites
- [ ] Troubleshooting guide

**Traces to:** [NFR-003]
**Depends on:** All MVP tasks
**Blocks:** — (но нужно для релиза)

---

## Milestone 2: Beta

### TASK-010: Framework Adapters
🟠 P1 | ✅ DONE | Est: 5d

**Description:**
Адаптеры для LangGraph и CrewAI.

**Checklist:**
- [x] LangGraphAdapter
  - [x] Graph loading from module
  - [x] State mapping to ATP
  - [x] Event extraction from steps
  - [x] Metrics collection
- [x] CrewAIAdapter
  - [x] Crew factory pattern
  - [x] Task mapping
  - [x] Agent events
- [x] AutoGen legacy adapter (optional)
- [x] Adapter development guide
- [x] Integration tests с реальными agents

**Traces to:** [REQ-012]
**Depends on:** [TASK-003]
**Blocks:** —

---

### TASK-011: Multiple Runs & Statistics
🟠 P1 | ✅ DONE | Est: 3d

**Description:**
Поддержка N прогонов и статистический анализ.

**Checklist:**
- [x] runs_per_test configuration
- [x] Parallel runs execution
- [x] StatisticalResult model
- [x] Mean, std, min, max, median
- [x] 95% Confidence Interval (t-distribution)
- [x] Coefficient of Variation
- [x] StabilityAssessment (stable/moderate/unstable/critical)
- [x] Statistical summary in reports

**Traces to:** [REQ-031]
**Depends on:** [TASK-006]
**Blocks:** [TASK-013]

---

### TASK-012: LLM-as-Judge Evaluator
🟠 P1 | ✅ DONE | Est: 4d

**Description:**
Evaluator с использованием LLM для семантической оценки.

**Checklist:**
- [x] LLMJudgeEvaluator class
- [x] Anthropic client integration
- [x] Built-in criteria prompts
  - [x] factual_accuracy
  - [x] completeness
  - [x] relevance
  - [x] coherence
  - [x] clarity
  - [x] actionability
- [x] Custom prompt support
- [x] Score parsing
- [x] Explanation extraction
- [x] Multi-call averaging (optional)
- [x] Cost tracking
- [x] Error handling (rate limits, etc.)

**Traces to:** [REQ-042]
**Depends on:** [TASK-007]
**Blocks:** —

---

### TASK-013: Baseline & Regression Detection
🟡 P2 | ✅ DONE | Est: 3d

**Description:**
Сохранение baseline и обнаружение регрессий.

**Checklist:**
- [x] Baseline file format
- [x] `atp baseline save` command
- [x] `atp baseline compare` command
- [x] Welch's t-test для сравнения
- [x] Regression detection (p < 0.05)
- [x] Improvement detection
- [x] Delta calculation
- [x] Diff visualization в console
- [x] JSON diff output

**Traces to:** [REQ-052]
**Depends on:** [TASK-011]
**Blocks:** —

---

### TASK-016: HTML Reporter
🟡 P2 | ✅ DONE | Est: 3d

**Description:**
Self-contained HTML отчёт.

**Checklist:**
- [x] HTMLReporter class
- [x] Jinja2 template
- [x] Embedded CSS (no external deps)
- [x] Summary section
- [x] Test details accordion
- [x] Score charts (Chart.js inline)
- [x] Failed checks highlighting
- [x] Trace viewer (collapsible)
- [x] Single-file output

**Traces to:** [REQ-051]
**Depends on:** [TASK-009]
**Blocks:** —

---

### TASK-017: CI/CD Integration
🟠 P1 | ✅ DONE | Est: 3d

**Description:**
Интеграция с CI системами.

**Checklist:**
- [x] JUnit XML reporter
- [x] GitHub Action
  - [x] action.yml
  - [x] Caching
  - [x] Artifact upload
- [x] GitLab CI template
- [x] Exit codes documentation
- [x] CI usage examples

**Traces to:** [REQ-051]
**Depends on:** [TASK-009], [TASK-014]
**Blocks:** —

---

### TASK-018: Code Execution Evaluator
🟡 P2 | ✅ DONE | Est: 3d

**Description:**
Evaluator для запуска сгенерированного кода.

**Checklist:**
- [x] CodeExecEvaluator class
- [x] pytest runner
- [x] npm test runner
- [x] Custom command runner
- [x] Lint runner (ruff, eslint)
- [x] Sandbox execution (Docker)
- [x] Output parsing
- [x] Test count extraction
- [x] Pass rate calculation

**Traces to:** [REQ-041]
**Depends on:** [TASK-007]
**Blocks:** —

---

### TASK-019: Mock Tools
🟡 P2 | ✅ DONE | Est: 2d

**Description:**
Mock tools для детерминированного тестирования.

**Checklist:**
- [ ] Mock tool server (FastAPI)
- [ ] YAML-based mock definitions
- [ ] Pattern matching для responses
- [ ] Call recording
- [ ] tools_endpoint в ATP Request
- [ ] Documentation

**Traces to:** [REQ-010]
**Depends on:** [TASK-003]
**Blocks:** —

---

## Milestone 3: GA

### TASK-020: Parallel Execution
🟡 P2 | ✅ DONE | Est: 2d

**Description:**
Параллельный запуск тестов.

**Checklist:**
- [x] --parallel CLI option
- [x] Semaphore-based concurrency
- [x] Resource isolation
- [x] Result aggregation
- [x] Progress tracking (multiple tests)

**Traces to:** [REQ-030]
**Depends on:** [TASK-006]
**Blocks:** —

---

### TASK-021: Web Dashboard (Basic)
🟢 P3 | ✅ DONE | Est: 10d

**Description:**
Веб-интерфейс для просмотра результатов.

**Checklist:**
- [ ] FastAPI backend
- [ ] React frontend
- [ ] Results storage (SQLite/Postgres)
- [ ] Suite list view
- [ ] Test details view
- [ ] Historical trends
- [ ] Agent comparison
- [ ] Authentication (basic)

**Traces to:** —
**Depends on:** [TASK-009]
**Blocks:** —

---

### TASK-022: Security Hardening
🔴 P0 | ✅ DONE | Est: 3d

**Description:**
Аудит безопасности и hardening.

**Checklist:**
- [ ] Input validation audit
- [ ] Sandbox escape prevention
- [ ] Secret handling review
- [ ] Log sanitization
- [ ] Network isolation verification
- [ ] Resource limits testing
- [ ] Documentation: security model

**Traces to:** [NFR-004]
**Depends on:** [TASK-006]
**Blocks:** — (но нужно для GA)

---

### TASK-023: Performance Optimization
🟠 P1 | ✅ DONE | Est: 3d

**Description:**
Оптимизация производительности.

**Checklist:**
- [ ] Profiling runner
- [ ] Async optimizations
- [ ] Caching (parsed tests, adapters)
- [ ] Startup time optimization
- [ ] Memory usage audit
- [ ] Benchmark suite
- [ ] Performance documentation

**Traces to:** [NFR-001]
**Depends on:** [TASK-006]
**Blocks:** —

---

### TASK-024: Complete Documentation
🔴 P0 | ✅ DONE | Est: 5d

**Description:**
Полная документация для GA.

**Checklist:**
- [ ] API Reference (auto-generated)
- [ ] Architecture documentation
- [ ] All evaluators reference
- [ ] All adapters reference
- [ ] Configuration reference
- [ ] Best practices guide
- [ ] Migration guide (from custom solutions)
- [ ] Video tutorials
- [ ] FAQ

**Traces to:** [NFR-003]
**Depends on:** All
**Blocks:** — (но нужно для GA)

---

## Dependency Graph

```
TASK-100 (Test Infrastructure) ◄─────────────────────────────────┐
    │                                                             │
    ├──► TASK-001 (Protocol)                                      │
    │        │                                                    │
    │        ├──► TASK-101 (Contract Tests)                       │
    │        │                                                    │
    │        ├──► TASK-002 (Events)                               │
    │        │                                                    │
    │        ├──► TASK-003 (Adapters) ──► TASK-102 (Integration)  │
    │        │        │                                           │
    │        │        ├──► TASK-010 (Framework Adapters)          │
    │        │        │                                           │
    │        │        └──► TASK-019 (Mock Tools)                  │
    │        │                                                    │
    │        └──► TASK-006 (Runner) ◄── TASK-004 (Loader)         │
    │                 │                       │                   │
    │                 │                       └──► TASK-005 (Tags)│
    │                 │                                           │
    │                 ├──► TASK-007 (Evaluators)                  │
    │                 │        │                                  │
    │                 │        ├──► TASK-008 (Scoring)            │
    │                 │        │        │                         │
    │                 │        │        └──► TASK-009 (Reporters) │
    │                 │        │                  │               │
    │                 │        │                  ├──► TASK-014 (CLI)
    │                 │        │                  │        │      │
    │                 │        │                  │        └──► TASK-103 (E2E)
    │                 │        │                  │               │
    │                 │        │                  ├──► TASK-016 (HTML)
    │                 │        │                  │               │
    │                 │        │                  └──► TASK-017 (CI/CD)
    │                 │        │                                  │
    │                 │        ├──► TASK-012 (LLM Judge)          │
    │                 │        │                                  │
    │                 │        └──► TASK-018 (Code Exec)          │
    │                 │                                           │
    │                 ├──► TASK-011 (Statistics)                  │
    │                 │        │                                  │
    │                 │        └──► TASK-013 (Baseline)           │
    │                 │                                           │
    │                 ├──► TASK-020 (Parallel)                    │
    │                 │                                           │
    │                 └──► TASK-022 (Security)                    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

**Тестовые зависимости:**
- TASK-100 (Test Infrastructure) — первая задача, блокирует все остальные
- TASK-101 (Contract Tests) — после TASK-001
- TASK-102 (Integration Tests) — после TASK-003, TASK-006
- TASK-103 (E2E Tests) — после TASK-014

---

## Summary by Milestone

### MVP (включая Testing)
| Priority | Count | Est. Total |
|----------|-------|------------|
| 🔴 P0 | 11 | 35d |
| 🟠 P1 | 6 | 12d |
| 🟡 P2 | 0 | — |
| **Total** | **17** | **~47d** |

**Testing tasks в MVP:**
- TASK-100: Test Infrastructure (2d) — 🔴 P0
- TASK-101: Contract Tests (2d) — 🔴 P0
- TASK-102: Integration Tests (3d) — 🟠 P1
- TASK-103: E2E Tests (2d) — 🟠 P1

### Beta (8 tasks)
| Priority | Count | Est. Total |
|----------|-------|------------|
| 🔴 P0 | 0 | — |
| 🟠 P1 | 5 | 18d |
| 🟡 P2 | 4 | 11d |
| 🟢 P3 | 0 | — |
| **Total** | **9** | **~29d** |

### GA (5 tasks)
| Priority | Count | Est. Total |
|----------|-------|------------|
| 🔴 P0 | 2 | 8d |
| 🟠 P1 | 1 | 3d |
| 🟡 P2 | 1 | 2d |
| 🟢 P3 | 1 | 10d |
| **Total** | **5** | **~23d** |

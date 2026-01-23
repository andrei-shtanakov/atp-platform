# Design Specification

> Архитектура, API, схемы данных и ключевые решения ATP

## 1. Обзор архитектуры

### 1.1 Принципы

| Принцип | Описание |
|---------|----------|
| **Agent as Black Box** | Платформе неважна реализация агента, важен только контракт |
| **Protocol First** | Стандартный протокол — основа всех интеграций |
| **Plugin Architecture** | Evaluators, Adapters, Reporters — заменяемые компоненты |
| **Immutable Data Flow** | Test → Runner → Agent → Response → Evaluators → Report |
| **Fail-Safe Defaults** | Минимальная конфигурация для начала работы |

### 1.2 Высокоуровневая диаграмма

```
┌─────────────────────────────────────────────────────────────────────┐
│                           ATP Platform                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│   │  CLI    │───►│ Loader  │───►│ Runner  │───►│Evaluator│         │
│   └─────────┘    └─────────┘    └────┬────┘    └────┬────┘         │
│                                      │              │               │
│                                      ▼              ▼               │
│                               ┌─────────┐    ┌─────────┐           │
│                               │ Gateway │    │ Scoring │           │
│                               └────┬────┘    └────┬────┘           │
│                                    │              │                 │
│                    ┌───────────────┼───────────────┘                │
│                    ▼               ▼                                │
│               ┌─────────┐    ┌─────────┐                           │
│               │ Adapter │    │Reporter │                           │
│               └────┬────┘    └─────────┘                           │
│                    │                                                │
└────────────────────┼────────────────────────────────────────────────┘
                     ▼
              ┌─────────────┐
              │    Agent    │
              │ (Black Box) │
              └─────────────┘
```

**Traces to:** [REQ-001], [REQ-010], [REQ-011]

---

## 2. ATP Protocol

### DESIGN-001: Message Formats

#### 2.1 ATP Request

```json
{
  "version": "1.0",
  "task_id": "uuid",
  "task": {
    "description": "string (required)",
    "input_data": "object (optional)",
    "expected_artifacts": "array (optional)"
  },
  "constraints": {
    "max_steps": "integer",
    "max_tokens": "integer",
    "timeout_seconds": "integer (default: 300)",
    "allowed_tools": "array<string> | null",
    "budget_usd": "number"
  },
  "context": {
    "tools_endpoint": "string (URL)",
    "workspace_path": "string",
    "environment": "object<string, string>"
  },
  "metadata": "object (pass-through)"
}
```

#### 2.2 ATP Response

```json
{
  "version": "1.0",
  "task_id": "uuid",
  "status": "completed | failed | timeout | cancelled | partial",
  "artifacts": [
    {
      "type": "file | structured | reference",
      "path": "string",
      "content_type": "string",
      "size_bytes": "integer",
      "content_hash": "string",
      "content": "string (inline, optional)",
      "data": "object (for structured)"
    }
  ],
  "metrics": {
    "total_tokens": "integer",
    "input_tokens": "integer",
    "output_tokens": "integer",
    "total_steps": "integer",
    "tool_calls": "integer",
    "llm_calls": "integer",
    "wall_time_seconds": "number",
    "cost_usd": "number"
  },
  "error": "string | null",
  "trace_id": "string (optional)"
}
```

#### 2.3 ATP Event

```json
{
  "version": "1.0",
  "task_id": "uuid",
  "timestamp": "ISO 8601",
  "sequence": "integer (monotonic)",
  "event_type": "tool_call | llm_request | reasoning | error | progress",
  "payload": {
    // type-specific fields
  }
}
```

**Traces to:** [REQ-001], [REQ-002], [REQ-003]

---

### DESIGN-002: Event Types

| Type | Payload Fields | Description |
|------|----------------|-------------|
| `tool_call` | tool, input, output, duration_ms, status | Вызов инструмента |
| `llm_request` | model, input_tokens, output_tokens, duration_ms | LLM API вызов |
| `reasoning` | thought, plan, step | Внутреннее рассуждение |
| `error` | error_type, message, recoverable | Ошибка |
| `progress` | current_step, percentage, message | Прогресс выполнения |

**Traces to:** [REQ-003]

---

## 3. Компоненты

### DESIGN-003: Adapters

#### 3.1 Base Interface

```python
class AgentAdapter(ABC):
    @abstractmethod
    async def execute(self, request: ATPRequest) -> ATPResponse:
        """Execute task synchronously."""
        
    @abstractmethod
    async def stream_events(self, request: ATPRequest) -> AsyncIterator[ATPEvent]:
        """Execute with event streaming."""
        
    async def health_check(self) -> bool:
        """Check agent availability."""
        
    async def cleanup(self) -> None:
        """Release resources."""
```

#### 3.2 Adapter Types

| Adapter | Transport | Use Case |
|---------|-----------|----------|
| HTTPAdapter | HTTP POST / SSE | Агенты с HTTP API |
| ContainerAdapter | stdin/stdout/stderr | Docker-упакованные агенты |
| CLIAdapter | subprocess | CLI-утилиты |
| LangGraphAdapter | Python import | LangGraph графы |
| CrewAIAdapter | Python import | CrewAI crews |

#### 3.3 Configuration

```yaml
agents:
  my-http-agent:
    type: http
    endpoint: "http://localhost:8000"
    timeout: 300
    headers:
      Authorization: "Bearer ${API_KEY}"
      
  my-container-agent:
    type: container
    image: "registry/agent:v1"
    resources:
      memory: "2Gi"
      cpu: "1"
    network: "none"  # isolated
    
  my-langgraph-agent:
    type: langgraph
    module: "agents.research"
    graph: "agent_graph"
    config:
      recursion_limit: 50
```

**Traces to:** [REQ-010], [REQ-011], [REQ-012]

---

### DESIGN-004: Test Loader

#### 4.1 Test Definition Schema

```yaml
# test_suite.yaml
test_suite: "string (name)"
version: "1.0"
description: "string (optional)"

defaults:
  runs_per_test: 1
  timeout_seconds: 300
  scoring:
    quality_weight: 0.4
    completeness_weight: 0.3
    efficiency_weight: 0.2
    cost_weight: 0.1

agents:
  - name: "string"
    # ... agent config or reference

tests:
  - id: "string (unique)"
    name: "string"
    description: "string (optional)"
    tags: ["smoke", "regression"]
    
    task:
      description: "string (required)"
      input_data: {}
      expected_artifacts: []
      
    constraints:
      max_steps: 50
      max_tokens: 100000
      timeout_seconds: 180
      allowed_tools: ["web_search", "file_write"]
      
    assertions:
      - type: "artifact_exists"
        config:
          path: "report.md"
      - type: "behavior"
        config:
          must_use_tools: ["web_search"]
      - type: "llm_eval"
        config:
          criteria: "factual_accuracy"
          threshold: 0.8
          
    scoring:
      quality_weight: 0.5
```

#### 4.2 Validation

- JSON Schema validation для структуры
- Semantic validation: уникальность id, существование ссылок
- Warning для deprecated fields

**Traces to:** [REQ-020], [REQ-021], [REQ-022]

---

### DESIGN-005: Test Runner

#### 5.1 Execution Flow

```
1. Load test suite
2. Validate configuration
3. Initialize adapters
4. For each test (parallel if enabled):
   a. Create sandbox
   b. For each run (1..N):
      i.   Build ATP Request
      ii.  Execute via adapter
      iii. Collect response + events
      iv.  Run evaluators
      v.   Store results
   c. Aggregate statistics
   d. Cleanup sandbox
5. Aggregate suite results
6. Generate reports
```

#### 5.2 Sandbox Configuration

```python
@dataclass
class SandboxConfig:
    # Resource limits
    memory_limit: str = "2Gi"
    cpu_limit: str = "2"
    
    # Network
    network_mode: str = "none"  # none | host | custom
    allowed_hosts: list[str] = field(default_factory=list)
    
    # Filesystem
    workspace_path: Path = Path("/workspace")
    readonly_mounts: list[tuple[Path, Path]] = field(default_factory=list)
    
    # Timeout
    hard_timeout: int = 600  # seconds, kills container
```

#### 5.3 Parallel Execution

```python
class ParallelExecutor:
    def __init__(self, max_workers: int = 4):
        self.semaphore = asyncio.Semaphore(max_workers)
        
    async def run_tests(self, tests: list[Test]) -> list[Result]:
        tasks = [self._run_with_semaphore(t) for t in tests]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

**Traces to:** [REQ-030], [REQ-031], [REQ-032]

---

### DESIGN-006: Statistics Engine

#### 6.1 Metrics

```python
@dataclass
class StatisticalResult:
    mean: float
    std: float
    min: float
    max: float
    median: float
    confidence_interval: tuple[float, float]  # 95% CI
    n_runs: int
    coefficient_of_variation: float  # std / mean
    
@dataclass  
class StabilityAssessment:
    level: str  # stable | moderate | unstable | critical
    cv: float
    message: str
```

#### 6.2 Stability Thresholds

| Level | CV Range | Interpretation |
|-------|----------|----------------|
| stable | < 0.05 | Consistent results |
| moderate | 0.05 - 0.15 | Acceptable variance |
| unstable | 0.15 - 0.30 | Results may be unreliable |
| critical | > 0.30 | Unpredictable behavior |

**Traces to:** [REQ-031]

---

### DESIGN-007: Evaluators

#### 7.1 Base Interface

```python
class Evaluator(ABC):
    name: str
    
    @abstractmethod
    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        pass

@dataclass
class EvalCheck:
    name: str
    passed: bool
    score: float  # 0.0 - 1.0
    message: str | None
    details: dict | None

@dataclass
class EvalResult:
    evaluator: str
    checks: list[EvalCheck]
    
    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)
```

#### 7.2 Evaluator Registry

| Evaluator | Assertion Types | Deterministic |
|-----------|-----------------|---------------|
| ArtifactEvaluator | artifact_exists, contains, schema, sections | Yes |
| BehaviorEvaluator | must_use_tools, max_tool_calls, no_errors | Yes |
| LLMJudgeEvaluator | llm_eval, custom | No |
| CodeExecEvaluator | pytest, lint, custom_command | Yes |

#### 7.3 Scoring Formula

```
Score = Σ(weight_i × component_i) × 100

where:
- quality = mean(artifact_scores, llm_scores)
- completeness = passed_checks / total_checks
- efficiency = normalize(steps, max_steps, optimal_steps)
- cost = 1 - log(1 + tokens/max_tokens) / log(2)
```

**Traces to:** [REQ-040], [REQ-041], [REQ-042], [REQ-043]

---

### DESIGN-008: LLM-as-Judge

#### 8.1 Prompt Template

```
You are evaluating AI agent output.

TASK:
{task_description}

ARTIFACT:
{artifact_content}

CRITERION: {criteria}
{criteria_description}

{custom_prompt}

Respond in JSON:
{
  "score": <0.0-1.0>,
  "explanation": "<reasoning>",
  "issues": ["<list>"],
  "strengths": ["<list>"]
}
```

#### 8.2 Built-in Criteria

| Criteria | Description |
|----------|-------------|
| factual_accuracy | Проверка фактов, статистики, дат |
| completeness | Полнота ответа на все аспекты задачи |
| relevance | Релевантность содержимого задаче |
| coherence | Логичность и связность |
| clarity | Ясность изложения |
| actionability | Практическая применимость |

#### 8.3 Calibration

- Multi-run averaging (3+ LLM calls)
- Temperature = 0 для детерминизма
- Optional human-in-the-loop для baseline

**Traces to:** [REQ-042]

---

### DESIGN-009: Reporters

#### 9.1 Reporter Interface

```python
class Reporter(ABC):
    @abstractmethod
    def report(self, results: SuiteResults) -> None:
        pass
        
    @abstractmethod
    def supports_streaming(self) -> bool:
        pass
```

#### 9.2 Output Formats

| Reporter | Format | Use Case |
|----------|--------|----------|
| ConsoleReporter | Terminal (colored) | Development |
| JSONReporter | JSON file | CI/CD, automation |
| HTMLReporter | Self-contained HTML | Sharing, review |
| JUnitReporter | JUnit XML | CI systems |

#### 9.3 Console Output Structure

```
ATP Test Results
================

Suite: competitor_analysis
Agent: langgraph-research
Runs per test: 5

Tests:
  ✓ basic_competitor_search     85.2/100  (σ=3.1)  [1.2s]
  ✗ unknown_company_handling    45.0/100  (σ=12.5) [0.8s]
    - llm_eval:factual_accuracy: 0.4 < 0.8 threshold
  ✓ performance_large_market    78.5/100  (σ=5.2)  [5.4s]

Summary: 2 passed, 1 failed (66.7%)
Total time: 7.4s
```

**Traces to:** [REQ-050], [REQ-051]

---

### DESIGN-010: Baseline & Regression

#### 10.1 Baseline Format

```json
{
  "version": "1.0",
  "created_at": "ISO8601",
  "suite": "competitor_analysis",
  "agent": "langgraph-research",
  "tests": {
    "basic_competitor_search": {
      "mean_score": 85.2,
      "std": 3.1,
      "n_runs": 5,
      "ci_95": [82.1, 88.3]
    }
  }
}
```

#### 10.2 Regression Detection

```python
def detect_regression(current: Stats, baseline: Stats) -> RegressionResult:
    # Welch's t-test for unequal variances
    t_stat, p_value = scipy.stats.ttest_ind(
        current.scores, 
        baseline.scores,
        equal_var=False
    )
    
    is_regression = (
        p_value < 0.05 and 
        current.mean < baseline.mean
    )
    
    return RegressionResult(
        is_regression=is_regression,
        p_value=p_value,
        delta=current.mean - baseline.mean,
        delta_percent=(current.mean - baseline.mean) / baseline.mean * 100
    )
```

**Traces to:** [REQ-052]

---

## 4. Data Flow

### 4.1 Test Execution

```
YAML File
    │
    ▼
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Parser  │────►│Validator│────►│ Models  │
└─────────┘     └─────────┘     └─────────┘
                                     │
                                     ▼
┌─────────┐     ┌─────────┐     ┌─────────┐
│ATP Req  │◄────│ Builder │◄────│ Config  │
└─────────┘     └─────────┘     └─────────┘
    │
    ▼
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Adapter │────►│  Agent  │────►│ATP Resp │
└─────────┘     └─────────┘     └─────────┘
                                     │
    ┌────────────────────────────────┘
    ▼
┌─────────┐     ┌─────────┐     ┌─────────┐
│Evaluator│────►│ Checks  │────►│ Scoring │
└─────────┘     └─────────┘     └─────────┘
                                     │
                                     ▼
                                ┌─────────┐
                                │ Report  │
                                └─────────┘
```

---

## 5. Ключевые решения

### ADR-001: Framework Agnostic Design
**Decision:** Агент = чёрный ящик с ATP Protocol  
**Rationale:** Фреймворки устаревают, протокол стабилен  
**Consequences:** (+) Гибкость, (-) Overhead на интеграцию

### ADR-002: YAML для тестов
**Decision:** Декларативные тесты в YAML, не в коде  
**Rationale:** Читаемость, версионирование, доступность для QA  
**Consequences:** (+) Low barrier, (-) Ограниченная выразительность

### ADR-003: LLM-as-Judge
**Decision:** Использовать LLM для семантической оценки  
**Rationale:** Невозможно оценить качество текста программно  
**Consequences:** (+) Глубокая оценка, (-) Стоимость, недетерминизм

### ADR-004: Docker для изоляции
**Decision:** Sandbox через Docker контейнеры  
**Rationale:** Industry standard, cross-platform  
**Consequences:** (+) Безопасность, (-) Docker dependency

### ADR-005: Multiple runs by default
**Decision:** Статистика по N прогонам вместо single run  
**Rationale:** LLM недетерминированы, один прогон ничего не доказывает  
**Consequences:** (+) Надёжность, (-) Время и стоимость ×N

---

## 6. API Reference

### 6.1 CLI Commands

```bash
# Run tests
atp test --agent=NAME --suite=FILE [--tags=TAGS] [--runs=N] [--parallel=N]

# Validate agent integration
atp validate --agent=NAME

# Compare agents
atp compare --agents=A,B --suite=FILE

# Manage baselines
atp baseline save --agent=NAME --suite=FILE --output=FILE
atp baseline compare --agent=NAME --suite=FILE --baseline=FILE

# Info
atp version
atp list-agents
atp list-evaluators
```

### 6.2 Configuration File

```yaml
# atp.config.yaml
version: "1.0"

defaults:
  timeout_seconds: 300
  runs_per_test: 3
  parallel_workers: 4

agents:
  # ... agent definitions

evaluators:
  llm_judge:
    model: "claude-sonnet-4-20250514"
    temperature: 0
    
reporting:
  default: console
  verbose: false
  
secrets:
  # Reference environment variables
  anthropic_api_key: ${ANTHROPIC_API_KEY}
```

---

## 7. Directory Structure

```
{{project_name}}/
├── atp/                     # Source code
│   ├── cli/
│   ├── core/
│   ├── protocol/
│   ├── loader/
│   ├── runner/
│   ├── adapters/
│   ├── evaluators/
│   ├── scoring/
│   └── reporters/
├── tests/                   # Test suite
│   ├── unit/                # Unit tests (~70%)
│   ├── integration/         # Integration tests (~20%)
│   ├── e2e/                 # End-to-end tests (~10%)
│   ├── contract/            # Protocol contract tests
│   ├── fixtures/            # Test data
│   └── conftest.py          # Shared fixtures
├── docs/
├── examples/
└── pyproject.toml
```

---

## 8. Testing Strategy

### 8.1 Test Pyramid

```
        ┌───────────┐
        │   E2E     │  ~10% - Critical user journeys
        │   Tests   │
        ├───────────┤
        │Integration│  ~20% - Component interactions
        │   Tests   │
        ├───────────┤
        │   Unit    │  ~70% - Business logic, utils
        │   Tests   │
        └───────────┘
```

### 8.2 Test Types

| Type | Scope | Tools | Location |
|------|-------|-------|----------|
| Unit | Functions, classes | pytest, pytest-mock | `tests/unit/` |
| Integration | Adapters, Evaluators | pytest, Docker | `tests/integration/` |
| E2E | Full test run | pytest, subprocess | `tests/e2e/` |
| Contract | ATP Protocol | jsonschema, pydantic | `tests/contract/` |

### 8.3 Test Requirements by Component

| Component | Unit Tests | Integration Tests |
|-----------|------------|-------------------|
| Protocol models | Serialization, validation | — |
| Adapters | Mock responses | Real agent (Docker) |
| Evaluators | Check logic | Full evaluation pipeline |
| Runner | Orchestration logic | Suite execution |
| Reporters | Output formatting | File generation |
| CLI | Argument parsing | Full commands |

### 8.4 Definition of Done

**Каждая задача считается завершённой только если:**
- [ ] Unit tests написаны (покрытие ≥80% нового кода)
- [ ] Все тесты проходят локально
- [ ] Integration test если изменены интерфейсы
- [ ] CI pipeline зелёный
- [ ] Code review пройден

### 8.5 Fixtures & Test Data

```
tests/
├── fixtures/
│   ├── requests/           # Sample ATP Requests (valid, invalid)
│   ├── responses/          # Sample ATP Responses
│   ├── test_suites/        # Sample YAML test suites
│   ├── traces/             # Sample event traces
│   └── artifacts/          # Sample agent outputs
└── conftest.py             # Shared pytest fixtures
```

### 8.6 CI Pipeline

```yaml
test:
  steps:
    - pytest tests/unit -v --cov=atp --cov-fail-under=80
    - pytest tests/integration -v
    - pytest tests/e2e -v
    - ruff check atp/
    - mypy atp/
```

**Traces to:** [NFR-000], [TASK-100], [TASK-101], [TASK-102]

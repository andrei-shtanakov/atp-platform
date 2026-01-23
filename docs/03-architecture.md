# Architecture

## Обзор архитектуры

Agent Test Platform построена по модульному принципу с чётким разделением ответственности между компонентами. Ключевая идея — агент является чёрным ящиком, взаимодействующим через стандартный протокол.

## Архитектурные принципы

### 1. Separation of Concerns
Каждый компонент отвечает за одну задачу:
- **Protocol** — определяет контракт
- **Adapters** — транслируют протокол
- **Runner** — оркестрирует выполнение
- **Evaluators** — оценивают результаты
- **Reporters** — форматируют вывод

### 2. Plugin Architecture
Evaluators, Adapters, Reporters — плагины с общим интерфейсом.

### 3. Immutable Data Flow
Данные текут в одном направлении: Test Definition → Runner → Agent → Response → Evaluators → Report.

### 4. Fail-Safe Defaults
Система работает с минимальной конфигурацией, разумные defaults.

---

## Диаграмма компонентов

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ATP Platform                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐                                                         │
│  │   CLI / API    │  ◄── Entry point                                        │
│  └───────┬────────┘                                                         │
│          │                                                                   │
│          ▼                                                                   │
│  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐        │
│  │  Test Loader   │─────►│   Test Suite   │─────►│  Test Runner   │        │
│  │  (YAML/JSON)   │      │    (parsed)    │      │ (orchestrator) │        │
│  └────────────────┘      └────────────────┘      └───────┬────────┘        │
│                                                          │                  │
│                          ┌───────────────────────────────┼──────────────┐   │
│                          │           Sandbox             │              │   │
│                          │  ┌────────────────────────────┼───────────┐  │   │
│                          │  │                            ▼           │  │   │
│                          │  │  ┌─────────────────────────────────┐   │  │   │
│                          │  │  │         ATP Gateway             │   │  │   │
│                          │  │  │   (protocol translation)        │   │  │   │
│                          │  │  └──────────────┬──────────────────┘   │  │   │
│                          │  │                 │                      │  │   │
│                          │  │    ┌────────────┼────────────┐         │  │   │
│                          │  │    ▼            ▼            ▼         │  │   │
│                          │  │ ┌──────┐   ┌──────┐    ┌──────┐       │  │   │
│                          │  │ │Adapt.│   │Adapt.│    │Adapt.│       │  │   │
│                          │  │ │ HTTP │   │Docker│    │ CLI  │       │  │   │
│                          │  │ └──┬───┘   └──┬───┘    └──┬───┘       │  │   │
│                          │  │    │          │           │            │  │   │
│                          │  └────┼──────────┼───────────┼────────────┘  │   │
│                          │       ▼          ▼           ▼               │   │
│                          │   ┌──────┐   ┌──────┐    ┌──────┐           │   │
│                          │   │Agent │   │Agent │    │Agent │           │   │
│                          │   │  A   │   │  B   │    │  C   │           │   │
│                          │   └──────┘   └──────┘    └──────┘           │   │
│                          └──────────────────────────────────────────────┘   │
│                                                                              │
│          ┌───────────────────────────────────────────────────────┐          │
│          │                  Evaluation Pipeline                   │          │
│          │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │          │
│          │  │Artifact │  │Behavior │  │LLM Judge│  │CodeExec │   │          │
│          │  │Evaluator│  │Evaluator│  │Evaluator│  │Evaluator│   │          │
│          │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │          │
│          │       └────────────┴────────────┴────────────┘        │          │
│          │                         │                              │          │
│          │                         ▼                              │          │
│          │               ┌─────────────────┐                     │          │
│          │               │ Score Aggregator│                     │          │
│          │               └────────┬────────┘                     │          │
│          └────────────────────────┼──────────────────────────────┘          │
│                                   │                                          │
│                                   ▼                                          │
│                          ┌────────────────┐                                 │
│                          │   Reporters    │                                 │
│                          │ Console│JSON│HTML                                │
│                          └────────────────┘                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Компоненты

### 1. CLI / API Layer

**Ответственность**: точка входа, parsing аргументов, вызов runner.

```
atp/
├── cli/
│   ├── __init__.py
│   └── main.py          # All CLI commands (Click-based)
```

**Команды CLI**:
- `atp test` — запуск тестов с опциями --agent, --suite, --tags, --runs, --parallel, --output, --fail-fast
- `atp validate` — валидация test definitions
- `atp baseline save/compare` — управление baseline
- `atp list-agents` — список зарегистрированных агентов
- `atp version` — версия

**Интерфейс**:
```python
# main.py
@click.group()
def cli():
    """ATP - Agent Test Platform CLI."""

@cli.command()
@click.argument("suite")
@click.option("--agent", required=True)
@click.option("--runs", default=1)
@click.option("--parallel", default=1)
@click.option("--tags", multiple=True)
@click.option("--output", type=click.Choice(["console", "json", "html", "junit"]))
@click.option("--output-file", type=click.Path())
@click.option("--fail-fast", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def test(suite, agent, runs, parallel, tags, output, output_file, fail_fast, verbose):
    """Run test suite against an agent."""
```

### 2. Test Loader

**Ответственность**: загрузка и валидация test definitions из YAML/JSON.

```
atp/
├── loader/
│   ├── __init__.py
│   ├── loader.py        # Main TestLoader class
│   ├── parser.py        # YAML/JSON parsing, variable substitution
│   ├── models.py        # Pydantic models (TestSuite, TestDefinition, etc.)
│   ├── filters.py       # Tag-based test filtering (include/exclude)
│   └── schema.py        # JSON Schema validation
```

**Модели данных**:
```python
# models.py
from pydantic import BaseModel

class TestConstraints(BaseModel):
    max_steps: int | None = None
    max_tokens: int | None = None
    timeout_seconds: int = 300
    allowed_tools: list[str] | None = None

class Assertion(BaseModel):
    type: str  # artifact_exists, contains, behavior, llm_eval, etc.
    config: dict  # Type-specific configuration

class ScoringWeights(BaseModel):
    quality: float = 0.4
    completeness: float = 0.3
    efficiency: float = 0.2
    cost: float = 0.1

class TestDefinition(BaseModel):
    id: str
    name: str
    description: str | None = None
    tags: list[str] = []

    task: TaskDefinition
    constraints: TestConstraints = TestConstraints()
    assertions: list[Assertion] = []
    scoring: ScoringWeights = ScoringWeights()

class TestSuite(BaseModel):
    name: str
    description: str | None = None
    defaults: dict = {}
    agents: list[AgentReference] = []
    tests: list[TestDefinition]
```

### 3. Test Runner

**Ответственность**: оркестрация выполнения тестов, управление lifecycle.

```
atp/
├── runner/
│   ├── __init__.py
│   ├── orchestrator.py  # TestOrchestrator - main test execution engine
│   ├── models.py        # TestResult, SuiteResult, RunResult, ProgressEvent
│   ├── sandbox.py       # SandboxManager for test isolation
│   ├── progress.py      # Progress reporting
│   └── exceptions.py    # Runner-specific exceptions

atp/
├── statistics/          # Separate module for statistical analysis
│   ├── __init__.py
│   ├── calculator.py    # Statistical calculations (mean, CI, etc.)
│   ├── models.py        # StatisticalResult models
│   └── reporter.py      # Statistics reporting
```

**Алгоритм выполнения**:
```
1. Load test suite
2. Resolve agent configuration
3. For each test (parallel if configured):
   a. Create sandbox environment
   b. Setup mock tools if specified
   c. For each run (1..N):
      i.   Build ATP Request
      ii.  Send to agent via adapter
      iii. Collect ATP Response + Events
      iv.  Run evaluators
      v.   Record results
   d. Aggregate statistics
   e. Cleanup sandbox
4. Generate report
```

**Интерфейс**:
```python
# orchestrator.py
class TestOrchestrator:
    def __init__(
        self,
        config: ATPConfig,
        agent_registry: AgentRegistry,
        evaluator_registry: EvaluatorRegistry,
    ): ...

    async def run_suite(
        self,
        suite: TestSuite,
        agent_name: str,
        options: RunOptions,
    ) -> SuiteResults: ...

    async def run_test(
        self,
        test: TestDefinition,
        agent: AgentAdapter,
        options: RunOptions,
    ) -> TestResults: ...
```

### 4. ATP Protocol

**Ответственность**: определение контракта взаимодействия с агентами.

```
atp/
├── protocol/
│   ├── __init__.py
│   ├── models.py        # ATP Request/Response/Event Pydantic models
│   └── schema.py        # JSON Schema generation

atp/
├── streaming/           # Event streaming support
│   ├── __init__.py
│   ├── buffer.py        # Event buffering and replay
│   └── validation.py    # Event ordering validation
```

**Протокольные модели**:
```python
# protocol.py
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

class ATPRequest(BaseModel):
    version: str = "1.0"
    task_id: str
    task: TaskPayload
    constraints: ConstraintsPayload
    tools_endpoint: str | None = None

class ATPResponseStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class ATPResponse(BaseModel):
    version: str = "1.0"
    task_id: str
    status: ATPResponseStatus
    artifacts: list[Artifact]
    metrics: ExecutionMetrics
    error: str | None = None

class ATPEventType(str, Enum):
    TOOL_CALL = "tool_call"
    LLM_REQUEST = "llm_request"
    REASONING = "reasoning"
    ERROR = "error"

class ATPEvent(BaseModel):
    task_id: str
    timestamp: datetime
    sequence: int
    event_type: ATPEventType
    payload: dict
```

### 5. Adapters

**Ответственность**: трансляция между ATP Protocol и конкретными способами запуска агентов.

```
atp/
├── adapters/
│   ├── __init__.py
│   ├── base.py          # AgentAdapter abstract class, AdapterConfig
│   ├── registry.py      # AdapterRegistry for dynamic adapter management
│   ├── exceptions.py    # AdapterError, AdapterTimeoutError, AdapterConnectionError
│   ├── http.py          # HTTPAdapter - REST/SSE endpoints
│   ├── container.py     # ContainerAdapter - Docker-based agents
│   ├── cli.py           # CLIAdapter - subprocess management
│   ├── langgraph.py     # LangGraphAdapter - LangGraph native integration
│   ├── crewai.py        # CrewAIAdapter - CrewAI framework
│   └── autogen.py       # AutoGenAdapter - AutoGen legacy support
```

**Base Adapter Interface**:
```python
# base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator

class AgentAdapter(ABC):
    """Base class for all agent adapters."""

    @abstractmethod
    async def execute(
        self,
        request: ATPRequest,
    ) -> ATPResponse:
        """Execute task and return response."""
        pass

    @abstractmethod
    async def stream_events(
        self,
        request: ATPRequest,
    ) -> AsyncIterator[ATPEvent]:
        """Execute task and stream events."""
        pass

    async def health_check(self) -> bool:
        """Check if agent is available."""
        return True

    async def cleanup(self) -> None:
        """Cleanup resources after execution."""
        pass
```

**HTTP Adapter Example**:
```python
# http.py
class HTTPAdapter(AgentAdapter):
    def __init__(self, endpoint: str, timeout: int = 300):
        self.endpoint = endpoint
        self.timeout = timeout
        self.client = httpx.AsyncClient()

    async def execute(self, request: ATPRequest) -> ATPResponse:
        response = await self.client.post(
            f"{self.endpoint}/execute",
            json=request.model_dump(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return ATPResponse.model_validate(response.json())

    async def stream_events(self, request: ATPRequest) -> AsyncIterator[ATPEvent]:
        async with self.client.stream(
            "POST",
            f"{self.endpoint}/execute/stream",
            json=request.model_dump(),
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event_data = json.loads(line[6:])
                    yield ATPEvent.model_validate(event_data)
```

**Container Adapter Example**:
```python
# container.py
class ContainerAdapter(AgentAdapter):
    def __init__(
        self,
        image: str,
        resources: ContainerResources | None = None,
    ):
        self.image = image
        self.resources = resources or ContainerResources()
        self.docker = docker.from_env()

    async def execute(self, request: ATPRequest) -> ATPResponse:
        container = self.docker.containers.run(
            self.image,
            stdin_open=True,
            detach=True,
            mem_limit=self.resources.memory,
            cpu_quota=self.resources.cpu_quota,
        )

        try:
            # Send request via stdin
            socket = container.attach_socket(params={'stdin': 1, 'stream': 1})
            socket._sock.sendall(request.model_dump_json().encode() + b'\n')

            # Wait and get output
            result = container.wait(timeout=request.constraints.timeout_seconds)
            logs = container.logs(stdout=True, stderr=False)

            return ATPResponse.model_validate_json(logs)
        finally:
            container.remove(force=True)
```

### 6. Evaluators

**Ответственность**: оценка результатов выполнения агента.

```
atp/
├── evaluators/
│   ├── __init__.py
│   ├── base.py          # Evaluator abstract class, EvalResult, EvalCheck
│   ├── registry.py      # EvaluatorRegistry for evaluator management
│   ├── artifact.py      # ArtifactEvaluator - file checks, content, schema
│   ├── behavior.py      # BehaviorEvaluator - tool usage, steps, errors
│   ├── llm_judge.py     # LLMJudgeEvaluator - semantic evaluation via Claude
│   └── code_exec.py     # CodeExecEvaluator - pytest, npm, custom runners
```

**Base Evaluator Interface**:
```python
# base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class EvalCheck:
    name: str
    passed: bool
    score: float  # 0.0 - 1.0
    message: str | None = None
    details: dict | None = None

@dataclass
class EvalResult:
    evaluator: str
    checks: list[EvalCheck]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def score(self) -> float:
        if not self.checks:
            return 0.0
        return sum(c.score for c in self.checks) / len(self.checks)

class Evaluator(ABC):
    """Base class for all evaluators."""

    name: str

    @abstractmethod
    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """Evaluate agent response against assertion."""
        pass
```

**Artifact Evaluator**:
```python
# artifact.py
class ArtifactEvaluator(Evaluator):
    name = "artifact"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        checks = []
        config = assertion.config

        if assertion.type == "artifact_exists":
            artifact = self._find_artifact(response, config["path"])
            checks.append(EvalCheck(
                name=f"artifact_exists:{config['path']}",
                passed=artifact is not None,
                score=1.0 if artifact else 0.0,
                message=f"Artifact {'found' if artifact else 'not found'}",
            ))

        elif assertion.type == "artifact_schema":
            artifact = self._find_artifact(response, config["path"])
            if artifact:
                valid = self._validate_schema(artifact, config["schema"])
                checks.append(EvalCheck(
                    name=f"artifact_schema:{config['path']}",
                    passed=valid,
                    score=1.0 if valid else 0.0,
                ))

        elif assertion.type == "contains":
            artifact = self._find_artifact(response, config["path"])
            if artifact:
                content = self._get_content(artifact)
                pattern = config.get("pattern") or config.get("text")
                found = self._check_contains(content, pattern, config.get("regex", False))
                checks.append(EvalCheck(
                    name=f"contains:{pattern[:30]}",
                    passed=found,
                    score=1.0 if found else 0.0,
                ))

        return EvalResult(evaluator=self.name, checks=checks)
```

**Behavior Evaluator**:
```python
# behavior.py
class BehaviorEvaluator(Evaluator):
    name = "behavior"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        checks = []
        config = assertion.config

        tool_calls = [e for e in trace if e.event_type == ATPEventType.TOOL_CALL]
        used_tools = {e.payload["tool"] for e in tool_calls}

        # must_use_tools
        if "must_use_tools" in config:
            for tool in config["must_use_tools"]:
                checks.append(EvalCheck(
                    name=f"must_use:{tool}",
                    passed=tool in used_tools,
                    score=1.0 if tool in used_tools else 0.0,
                    message=f"Tool {tool} {'was' if tool in used_tools else 'was not'} used",
                ))

        # must_not_use_tools
        if "must_not_use_tools" in config:
            for tool in config["must_not_use_tools"]:
                checks.append(EvalCheck(
                    name=f"must_not_use:{tool}",
                    passed=tool not in used_tools,
                    score=1.0 if tool not in used_tools else 0.0,
                ))

        # max_tool_calls
        if "max_tool_calls" in config:
            count = len(tool_calls)
            max_allowed = config["max_tool_calls"]
            checks.append(EvalCheck(
                name="max_tool_calls",
                passed=count <= max_allowed,
                score=min(1.0, max_allowed / count) if count > 0 else 1.0,
                details={"actual": count, "max": max_allowed},
            ))

        return EvalResult(evaluator=self.name, checks=checks)
```

**LLM Judge Evaluator**:
```python
# llm_judge.py
class LLMJudgeEvaluator(Evaluator):
    name = "llm_judge"

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.client = anthropic.Anthropic()

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        config = assertion.config
        artifact_content = self._get_artifact_content(response, config.get("artifact"))

        prompt = self._build_prompt(
            criteria=config["criteria"],
            custom_prompt=config.get("prompt"),
            task_description=task.task.description,
            artifact_content=artifact_content,
        )

        result = await self._call_llm(prompt)
        score = result["score"]
        explanation = result["explanation"]

        return EvalResult(
            evaluator=self.name,
            checks=[EvalCheck(
                name=f"llm_eval:{config['criteria']}",
                passed=score >= config.get("threshold", 0.7),
                score=score,
                message=explanation,
            )],
        )
```

### 7. Score Aggregator

**Ответственность**: агрегация результатов evaluators в итоговый score.

```python
# scoring.py
class ScoreAggregator:
    def aggregate(
        self,
        eval_results: list[EvalResult],
        weights: ScoringWeights,
        metrics: ExecutionMetrics,
        constraints: TestConstraints,
    ) -> AggregatedScore:
        # Quality score from evaluators
        quality_score = self._compute_quality(eval_results)

        # Completeness from assertions
        completeness_score = self._compute_completeness(eval_results)

        # Efficiency from metrics
        efficiency_score = self._compute_efficiency(metrics, constraints)

        # Cost score
        cost_score = self._compute_cost(metrics, constraints)

        # Weighted sum
        total = (
            weights.quality * quality_score +
            weights.completeness * completeness_score +
            weights.efficiency * efficiency_score +
            weights.cost * cost_score
        )

        return AggregatedScore(
            total=total * 100,  # 0-100 scale
            quality=quality_score,
            completeness=completeness_score,
            efficiency=efficiency_score,
            cost=cost_score,
            weights=weights,
        )
```

### 8. Reporters

**Ответственность**: форматирование и вывод результатов.

```
atp/
├── reporters/
│   ├── __init__.py
│   ├── base.py            # Reporter abstract class, TestReport, SuiteReport
│   ├── registry.py        # ReporterRegistry
│   ├── console.py         # ConsoleReporter - ANSI colored terminal output
│   ├── json_reporter.py   # JSONReporter - structured JSON export
│   ├── html_reporter.py   # HTMLReporter - self-contained HTML with charts
│   └── junit_reporter.py  # JUnitReporter - JUnit XML for CI/CD
```

**Console Reporter**:
```python
# console.py
class ConsoleReporter(Reporter):
    def report(self, results: SuiteResults) -> None:
        self._print_header(results)

        for test_result in results.tests:
            self._print_test_result(test_result)

        self._print_summary(results)

    def _print_test_result(self, result: TestResult) -> None:
        status = "✓" if result.passed else "✗"
        color = "green" if result.passed else "red"

        print(f"  {status} {result.test_id}")
        print(f"    Score: {result.score.total:.1f}/100")
        print(f"    Duration: {result.duration_ms}ms")

        if self.verbose and not result.passed:
            for check in result.failed_checks:
                print(f"      - {check.name}: {check.message}")
```

---

## Data Flow

### Test Execution Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  YAML Test  │────►│ Test Loader │────►│TestDefinition│
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  ATP Request│◄────│   Runner    │◄────│   Config    │
└──────┬──────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Adapter   │────►│    Agent    │────►│ ATP Response│
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
       ┌───────────────────────────────────────┘
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Evaluators │────►│  EvalResult │────►│  Aggregator │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Report    │
                                        └─────────────┘
```

### Event Streaming Flow

```
┌─────────┐    WebSocket/SSE    ┌─────────┐    Collect    ┌─────────┐
│  Agent  │ ─────────────────► │ Gateway │ ────────────► │  Tracer │
└─────────┘                     └─────────┘               └────┬────┘
                                                               │
                                                               ▼
┌─────────┐                     ┌─────────┐              ┌─────────┐
│Behavior │ ◄───────────────── │  Trace  │ ◄─────────── │  Store  │
│Evaluator│                     │  Array  │              └─────────┘
└─────────┘                     └─────────┘
```

---

## Directory Structure

```
atp-platform/
├── atp/
│   ├── __init__.py
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py              # All CLI commands (Click-based)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── security.py          # URL, DNS, path traversal validation
│   │
│   ├── protocol/
│   │   ├── __init__.py
│   │   ├── models.py            # ATP Request/Response/Event
│   │   └── schema.py            # JSON Schema generation
│   │
│   ├── loader/
│   │   ├── __init__.py
│   │   ├── loader.py            # TestLoader class
│   │   ├── parser.py            # YAML/JSON parsing
│   │   ├── models.py            # TestSuite, TestDefinition models
│   │   ├── filters.py           # Tag filtering
│   │   └── schema.py            # Validation
│   │
│   ├── runner/
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # TestOrchestrator
│   │   ├── models.py            # TestResult, SuiteResult
│   │   ├── sandbox.py           # SandboxManager
│   │   ├── progress.py          # Progress reporting
│   │   └── exceptions.py        # Runner exceptions
│   │
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py              # AgentAdapter base class
│   │   ├── registry.py          # AdapterRegistry
│   │   ├── exceptions.py        # Adapter exceptions
│   │   ├── http.py              # HTTPAdapter
│   │   ├── container.py         # ContainerAdapter
│   │   ├── cli.py               # CLIAdapter
│   │   ├── langgraph.py         # LangGraphAdapter
│   │   ├── crewai.py            # CrewAIAdapter
│   │   └── autogen.py           # AutoGenAdapter
│   │
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── base.py              # Evaluator base class
│   │   ├── registry.py          # EvaluatorRegistry
│   │   ├── artifact.py          # ArtifactEvaluator
│   │   ├── behavior.py          # BehaviorEvaluator
│   │   ├── llm_judge.py         # LLMJudgeEvaluator
│   │   └── code_exec.py         # CodeExecEvaluator
│   │
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── aggregator.py        # ScoreAggregator
│   │   └── models.py            # Scoring models
│   │
│   ├── statistics/
│   │   ├── __init__.py
│   │   ├── calculator.py        # Statistical calculations
│   │   ├── models.py            # StatisticalResult
│   │   └── reporter.py          # Statistics reporting
│   │
│   ├── baseline/
│   │   ├── __init__.py
│   │   ├── storage.py           # Baseline file management
│   │   ├── comparison.py        # Welch's t-test comparison
│   │   ├── reporter.py          # Diff visualization
│   │   └── models.py            # Baseline models
│   │
│   ├── reporters/
│   │   ├── __init__.py
│   │   ├── base.py              # Reporter base class
│   │   ├── registry.py          # ReporterRegistry
│   │   ├── console.py           # ConsoleReporter
│   │   ├── json_reporter.py     # JSONReporter
│   │   ├── html_reporter.py     # HTMLReporter
│   │   └── junit_reporter.py    # JUnitReporter
│   │
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── buffer.py            # Event buffering
│   │   └── validation.py        # Event ordering
│   │
│   ├── mock_tools/
│   │   ├── __init__.py
│   │   ├── server.py            # FastAPI mock tool server
│   │   ├── loader.py            # YAML mock definitions
│   │   ├── models.py            # Mock tool models
│   │   └── recorder.py          # Call recording
│   │
│   ├── performance/
│   │   ├── __init__.py
│   │   ├── benchmark.py         # Performance benchmarking
│   │   ├── profiler.py          # Execution profiling
│   │   ├── cache.py             # Caching layer
│   │   ├── memory.py            # Memory tracking
│   │   ├── async_utils.py       # Async optimization
│   │   └── startup.py           # Startup optimization
│   │
│   └── dashboard/
│       ├── __init__.py
│       ├── app.py               # FastAPI application
│       ├── api.py               # REST API endpoints
│       ├── database.py          # SQLAlchemy setup
│       ├── storage.py           # Result persistence
│       ├── models.py            # Domain models
│       ├── schemas.py           # Pydantic schemas
│       └── auth.py              # Authentication
│
├── tests/
│   ├── unit/                    # Unit tests (~70%)
│   ├── e2e/                     # End-to-end tests (~10%)
│   ├── fixtures/                # Test fixtures
│   └── conftest.py              # Shared pytest fixtures
│
├── docs/                        # Documentation
├── examples/
│   ├── test_suites/             # Sample test suites
│   └── ci/                      # CI/CD templates
├── spec/                        # Requirements and tasks
│
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Deployment Architecture

### Local Development

```
┌─────────────────────────────────────────┐
│            Developer Machine             │
│                                          │
│  ┌────────┐    ┌────────┐    ┌────────┐ │
│  │  ATP   │───►│ Docker │───►│ Agent  │ │
│  │  CLI   │    │        │    │Container│ │
│  └────────┘    └────────┘    └────────┘ │
│                                          │
└─────────────────────────────────────────┘
```

### CI/CD Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    CI/CD Runner                          │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │   Clone  │───►│   Build  │───►│   Test   │          │
│  │   Repo   │    │  Agent   │    │   ATP    │          │
│  └──────────┘    └──────────┘    └────┬─────┘          │
│                                       │                 │
│                         ┌─────────────┴─────────────┐   │
│                         ▼                           ▼   │
│                  ┌──────────┐                ┌──────────┐│
│                  │  Upload  │                │  Report  ││
│                  │ Artifacts│                │  Status  ││
│                  └──────────┘                └──────────┘│
└─────────────────────────────────────────────────────────┘
```

---

## Security Considerations

### Sandbox Isolation

- Agents run in Docker containers with resource limits
- Network access controllable per test
- No access to host filesystem except mounted volumes
- Secrets passed via environment variables, not in test definitions

### API Key Management

```yaml
# atp.config.yaml
secrets:
  # Reference environment variables
  anthropic_api_key: ${ANTHROPIC_API_KEY}
  openai_api_key: ${OPENAI_API_KEY}

# Secrets are NEVER logged or included in reports
```

### Input Validation

- All YAML/JSON input validated against schemas
- Artifact paths sanitized to prevent path traversal
- Size limits on responses and artifacts

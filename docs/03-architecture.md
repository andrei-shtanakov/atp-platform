# Architecture

## Architecture Overview

Agent Test Platform is built on a modular principle with clear separation of responsibilities between components. The key idea is that an agent is a black box interacting through a standard protocol.

## Architectural Principles

### 1. Separation of Concerns
Each component is responsible for one task:
- **Protocol** — defines the contract
- **Adapters** — translate the protocol
- **Runner** — orchestrates execution
- **Evaluators** — assess results
- **Reporters** — format output

### 2. Plugin Architecture
Evaluators, Adapters, Reporters are plugins with a common interface.

### 3. Immutable Data Flow
Data flows in one direction: Test Definition → Runner → Agent → Response → Evaluators → Report.

### 4. Fail-Safe Defaults
The system works with minimal configuration, reasonable defaults.

---

## Component Diagram

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
│          │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │          │
│          │  │Security │  │Factual- │  │Filesys- │  │  Style  │   │          │
│          │  │Evaluator│  │  ity    │  │  tem    │  │Evaluator│   │          │
│          │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │          │
│          │  ┌─────────┐                                           │          │
│          │  │Perform- │                                           │          │
│          │  │  ance   │                                           │          │
│          │  └────┬────┘                                           │          │
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

## Components

### 1. CLI / API Layer

**Responsibility**: entry point, argument parsing, runner invocation.

```
atp/
├── cli/
│   ├── __init__.py
│   ├── main.py              # Core CLI commands (Click-based)
│   └── commands/            # Additional CLI commands
│       ├── __init__.py
│       ├── benchmark.py     # Benchmark command
│       ├── budget.py        # Budget command
│       ├── experiment.py    # Experiment command
│       ├── game.py          # Game-theoretic evaluation
│       ├── generate.py      # Test suite generation
│       ├── init.py          # Project initialization
│       └── plugins.py       # Plugin management
```

**CLI Commands**:
- `atp test` — run tests with options --agent, --suite, --tags, --runs, --parallel, --output, --fail-fast
- `atp run` — alias for test
- `atp list` — list tests in a suite
- `atp validate` — validate test definitions
- `atp baseline save/compare` — manage baselines
- `atp list-agents` — list registered agents
- `atp dashboard` — start web dashboard
- `atp tui` — start terminal UI
- `atp init` — initialize ATP project
- `atp generate` — generate test suites
- `atp benchmark` — run benchmarks
- `atp budget` — budget management
- `atp experiment` — run experiments
- `atp plugins` — manage plugins
- `atp game` — game-theoretic evaluation
- `atp version` — version

**Interface**:
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

**Responsibility**: loading and validating test definitions from YAML/JSON.

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

**Data Models**:
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

**Responsibility**: orchestrating test execution, managing lifecycle.

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

**Execution Algorithm**:
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

**Interface**:
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

**Responsibility**: defining the contract for agent interaction.

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

**Protocol Models**:
```python
# protocol.py
from pydantic import BaseModel
from datetime import datetime
from enum import StrEnum

class ATPRequest(BaseModel):
    version: str = "1.0"
    task_id: str
    task: TaskPayload
    constraints: ConstraintsPayload
    tools_endpoint: str | None = None

class ResponseStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class ATPResponse(BaseModel):
    version: str = "1.0"
    task_id: str
    status: ResponseStatus
    artifacts: list[Artifact]
    metrics: ExecutionMetrics
    error: str | None = None

class EventType(StrEnum):
    TOOL_CALL = "tool_call"
    LLM_REQUEST = "llm_request"
    REASONING = "reasoning"
    ERROR = "error"

class ATPEvent(BaseModel):
    task_id: str
    timestamp: datetime
    sequence: int
    event_type: EventType
    payload: dict
```

### 5. Adapters

**Responsibility**: translation between ATP Protocol and specific ways to run agents.

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
│   ├── autogen.py       # AutoGenAdapter - AutoGen legacy support
│   ├── azure_openai.py  # AzureOpenAIAdapter - Azure OpenAI service
│   ├── bedrock.py       # BedrockAdapter - AWS Bedrock
│   ├── vertex.py        # VertexAdapter - Google Vertex AI
│   └── mcp/             # MCP adapter
│       ├── __init__.py
│       ├── adapter.py   # MCPAdapter
│       └── transport.py # MCP transport layer
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

**Responsibility**: evaluating agent execution results.

```
atp/
├── evaluators/
│   ├── __init__.py
│   ├── base.py          # Evaluator abstract class, EvalResult, EvalCheck
│   ├── registry.py      # EvaluatorRegistry for evaluator management
│   ├── artifact.py      # ArtifactEvaluator - file checks, content, schema
│   ├── behavior.py      # BehaviorEvaluator - tool usage, steps, errors
│   ├── llm_judge.py     # LLMJudgeEvaluator - semantic evaluation via Claude
│   ├── code_exec.py     # CodeExecEvaluator - pytest, npm, custom runners
│   ├── factuality.py    # FactualityEvaluator - factual accuracy checks
│   ├── filesystem.py    # FilesystemEvaluator - workspace file checks
│   ├── performance.py   # PerformanceEvaluator - performance metrics
│   ├── style.py         # StyleEvaluator - output style assessment
│   └── security/        # Security evaluator package
│       ├── __init__.py
│       ├── base.py      # Base security checker
│       ├── evaluator.py # SecurityEvaluator
│       ├── pii.py       # PII detection
│       ├── injection.py # Prompt injection detection
│       ├── code.py      # Code safety checks
│       └── secrets.py   # Secret leak detection
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

**Responsibility**: aggregating evaluator results into a final score.

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

**Responsibility**: formatting and outputting results.

```
atp/
├── reporters/
│   ├── __init__.py
│   ├── base.py            # Reporter abstract class, TestReport, SuiteReport
│   ├── registry.py        # ReporterRegistry
│   ├── console.py         # ConsoleReporter - ANSI colored terminal output
│   ├── json_reporter.py   # JSONReporter - structured JSON export
│   ├── html_reporter.py   # HTMLReporter - self-contained HTML with charts
│   ├── junit_reporter.py  # JUnitReporter - JUnit XML for CI/CD
│   └── game_reporter.py   # GameReporter - game-theoretic results
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
│   │   ├── main.py              # Core CLI commands (Click-based)
│   │   └── commands/            # Additional CLI commands
│   │       ├── __init__.py
│   │       ├── benchmark.py
│   │       ├── budget.py
│   │       ├── experiment.py
│   │       ├── game.py
│   │       ├── generate.py
│   │       ├── init.py
│   │       └── plugins.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py        # Custom exceptions
│   │   ├── result.py            # Success/Failure result type
│   │   ├── settings.py          # ATPSettings configuration
│   │   ├── security.py          # URL, DNS, path traversal validation
│   │   ├── logging.py           # Structured logging (structlog)
│   │   ├── telemetry.py         # OpenTelemetry tracing
│   │   ├── metrics.py           # Prometheus metrics
│   │   └── observer.py          # Observer pattern for error tracking
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
│   │   ├── fallback.py          # FallbackAdapter (chain with automatic fallback)
│   │   ├── http.py              # HTTPAdapter
│   │   ├── container.py         # ContainerAdapter
│   │   ├── cli.py               # CLIAdapter
│   │   ├── langgraph.py         # LangGraphAdapter
│   │   ├── crewai.py            # CrewAIAdapter
│   │   ├── autogen.py           # AutoGenAdapter
│   │   ├── mcp/                 # MCP adapter
│   │   │   ├── adapter.py
│   │   │   └── transport.py
│   │   ├── bedrock/             # AWS Bedrock adapter
│   │   │   ├── adapter.py
│   │   │   ├── models.py
│   │   │   └── auth.py
│   │   ├── vertex/              # Google Vertex AI adapter
│   │   │   ├── adapter.py
│   │   │   ├── models.py
│   │   │   └── auth.py
│   │   └── azure_openai/        # Azure OpenAI adapter
│   │       ├── adapter.py
│   │       ├── models.py
│   │       └── auth.py
│   │
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── base.py              # Evaluator base class
│   │   ├── registry.py          # EvaluatorRegistry
│   │   ├── artifact.py          # ArtifactEvaluator
│   │   ├── behavior.py          # BehaviorEvaluator
│   │   ├── llm_judge.py         # LLMJudgeEvaluator
│   │   ├── code_exec.py         # CodeExecEvaluator
│   │   ├── factuality.py        # FactualityEvaluator
│   │   ├── filesystem.py        # FilesystemEvaluator
│   │   ├── performance.py       # PerformanceEvaluator
│   │   ├── style.py             # StyleEvaluator
│   │   └── security/            # Security evaluator package
│   │       ├── __init__.py
│   │       ├── evaluator.py
│   │       ├── pii.py
│   │       ├── injection.py
│   │       ├── code.py
│   │       └── secrets.py
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
│   │   ├── junit_reporter.py    # JUnitReporter
│   │   └── game_reporter.py     # GameReporter
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
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── __main__.py          # python -m atp.dashboard entry point
│   │   ├── database.py          # SQLAlchemy async setup
│   │   ├── storage.py           # Result persistence
│   │   ├── models.py            # ORM models (User, Agent, SuiteExecution, etc.)
│   │   ├── schemas.py           # Pydantic API schemas
│   │   ├── audit.py             # Audit logging
│   │   ├── audit_middleware.py  # Request audit middleware
│   │   ├── query_cache.py       # Query result caching
│   │   ├── optimized_queries.py # Optimized SQL queries
│   │   ├── auth/                # Authentication & SSO (JWT, OIDC, SAML)
│   │   ├── rbac/                # Role-based access control
│   │   ├── tenancy/             # Multi-tenant support (schema isolation, quotas)
│   │   └── v2/                  # Modular dashboard (FastAPI)
│   │       ├── factory.py       # App factory with lifespan
│   │       ├── config.py        # DashboardConfig
│   │       ├── dependencies.py  # FastAPI dependency injection
│   │       ├── routes/          # 28 route modules (agents, suites, analytics, etc.)
│   │       ├── services/        # Business logic (agent, test, comparison, export)
│   │       ├── websocket/       # Real-time updates (pub/sub, connection manager)
│   │       ├── templates/       # Jinja2 HTML templates
│   │       └── static/          # Static assets (CSS, JS)
│   │
│   ├── analytics/               # Cost tracking, A/B testing, anomaly detection
│   ├── benchmarks/              # Benchmark suites
│   ├── chaos/                   # Chaos testing (injectors, profiles)
│   ├── generator/               # Test suite generation (NL, templates, trace import)
│   ├── plugins/                 # Plugin ecosystem management
│   ├── sdk/                     # Python SDK for programmatic use
│   ├── tracing/                 # Agent trace recording and replay
│   └── tui/                     # Terminal user interface (optional, requires [tui] extra)
│
├── tests/
│   ├── unit/                    # Unit tests (~70%)
│   ├── integration/             # Integration tests (~20%)
│   ├── contract/                # Protocol contract tests
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

---

## Future: Package Decomposition

> See [ADR-003](adr/003-monorepo-decomposition.md) for the full architecture decision.

The platform is planned for decomposition into 4 independent packages within a monorepo using Python implicit namespace packages and uv workspaces:

| Package | Contents | Dependencies |
|---------|----------|-------------|
| **atp-core** | protocol, core, loader, chaos, cost, scoring, statistics, streaming | pydantic, structlog, opentelemetry |
| **atp-adapters** | All agent adapters (HTTP, CLI, Container, cloud, MCP) | atp-core, httpx |
| **atp-platform** | runner, evaluators, reporters, cli, sdk, mock_tools, ... | atp-core, atp-adapters |
| **atp-dashboard** | Web dashboard, analytics | atp-core, atp-platform, FastAPI, SQLAlchemy |

All existing `from atp.X import Y` imports will continue working unchanged via shared namespace.

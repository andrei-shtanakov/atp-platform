# Evaluation System

## Обзор

Система оценки ATP анализирует результаты выполнения агента и вычисляет метрики качества. Оценка происходит на нескольких уровнях: от простых структурных проверок до сложной семантической оценки с помощью LLM.

## Философия оценки

### Принципы

1. **Composability** — оценщики комбинируются для комплексной оценки
2. **Transparency** — каждая оценка объяснима
3. **Determinism where possible** — детерминированные проверки предпочтительнее
4. **Statistical validity** — стохастические оценки усредняются по прогонам

### Иерархия оценщиков

```
┌─────────────────────────────────────────────────────────┐
│                   Composite Score                        │
│                     (0-100)                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Quality  │ │Complete- │ │Efficiency│ │   Cost   │   │
│  │  Score   │ │   ness   │ │  Score   │ │  Score   │   │
│  │  (0-1)   │ │  (0-1)   │ │  (0-1)   │ │  (0-1)   │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │
│       │            │            │            │          │
│  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐   │
│  │Artifact │  │Behavior │  │ Metrics │  │ Metrics │   │
│  │  Eval   │  │  Eval   │  │ (steps) │  │(tokens) │   │
│  │LLM Judge│  │Code Exec│  │         │  │         │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Evaluators

### 1. Artifact Evaluator

Проверяет наличие, формат и содержимое артефактов.

**Типы проверок**:

| Assertion Type | Description | Example |
|----------------|-------------|---------|
| `artifact_exists` | Артефакт с указанным путём существует | `path: "report.md"` |
| `artifact_format` | Артефакт соответствует формату | `format: "markdown"` |
| `artifact_schema` | Structured artifact соответствует JSON Schema | `schema: {...}` |
| `contains` | Артефакт содержит текст/паттерн | `pattern: "Competitor"` |
| `not_contains` | Артефакт не содержит текст | `text: "error"` |
| `min_length` | Минимальная длина контента | `chars: 1000` |
| `max_length` | Максимальная длина контента | `chars: 50000` |
| `sections_exist` | Markdown содержит секции | `sections: ["Summary", "Details"]` |
| `table_exists` | Markdown содержит таблицу | `min_rows: 3` |

**Пример конфигурации**:

```yaml
assertions:
  - type: artifact_exists
    config:
      path: "report.md"

  - type: contains
    config:
      artifact: "report.md"
      pattern: "Microsoft Teams|Zoom|Google Meet"
      regex: true

  - type: sections_exist
    config:
      artifact: "report.md"
      sections:
        - "Executive Summary"
        - "Competitor Analysis"
        - "Recommendations"

  - type: artifact_schema
    config:
      artifact: "competitors.json"
      schema:
        type: object
        required: ["competitors"]
        properties:
          competitors:
            type: array
            minItems: 5
            items:
              type: object
              required: ["name", "description"]
```

**Scoring Logic**:

```python
def score_artifact_check(check: ArtifactCheck) -> float:
    if check.type == "artifact_exists":
        return 1.0 if check.passed else 0.0

    elif check.type == "contains":
        if check.regex:
            matches = len(re.findall(check.pattern, content))
            expected = check.min_matches or 1
            return min(1.0, matches / expected)
        return 1.0 if check.pattern in content else 0.0

    elif check.type == "sections_exist":
        found = sum(1 for s in check.sections if s in content)
        return found / len(check.sections)

    # ... etc
```

### 2. Behavior Evaluator

Анализирует trace выполнения: какие инструменты использовались, сколько шагов, какие ошибки.

**Типы проверок**:

| Assertion Type | Description | Example |
|----------------|-------------|---------|
| `must_use_tools` | Обязательное использование инструментов | `tools: ["web_search"]` |
| `must_not_use_tools` | Запрет на инструменты | `tools: ["dangerous_tool"]` |
| `max_tool_calls` | Ограничение вызовов | `limit: 10` |
| `max_steps` | Ограничение шагов | `limit: 50` |
| `no_errors` | Отсутствие ошибок в trace | - |
| `tool_sequence` | Порядок использования инструментов | `sequence: ["search", "write"]` |
| `no_hallucination` | Нет выдуманных данных (требует LLM) | `check_facts: true` |

**Пример конфигурации**:

```yaml
assertions:
  - type: behavior
    config:
      must_use_tools:
        - web_search
      must_not_use_tools:
        - code_execution  # Not allowed for this task
      max_tool_calls: 15

  - type: behavior
    config:
      no_errors: true
      allowed_error_types:
        - rate_limit  # Transient errors OK

  - type: behavior
    config:
      tool_call_efficiency:
        max_redundant_calls: 2  # Same tool, same input
```

**Trace Analysis**:

```python
def analyze_trace(trace: list[ATPEvent]) -> BehaviorMetrics:
    tool_calls = [e for e in trace if e.event_type == "tool_call"]
    errors = [e for e in trace if e.event_type == "error"]

    # Tool usage analysis
    used_tools = Counter(tc.payload["tool"] for tc in tool_calls)

    # Redundancy detection
    call_signatures = [
        (tc.payload["tool"], json.dumps(tc.payload["input"], sort_keys=True))
        for tc in tool_calls
    ]
    redundant_calls = len(call_signatures) - len(set(call_signatures))

    # Error analysis
    recoverable_errors = sum(1 for e in errors if e.payload.get("recoverable"))
    fatal_errors = len(errors) - recoverable_errors

    return BehaviorMetrics(
        tool_usage=dict(used_tools),
        total_tool_calls=len(tool_calls),
        redundant_calls=redundant_calls,
        total_errors=len(errors),
        fatal_errors=fatal_errors,
    )
```

### 3. LLM-as-Judge Evaluator

Использует LLM для семантической оценки качества результатов.

**Критерии оценки**:

| Criteria | Description |
|----------|-------------|
| `factual_accuracy` | Фактическая корректность |
| `completeness` | Полнота ответа |
| `relevance` | Релевантность задаче |
| `coherence` | Логичность и связность |
| `clarity` | Ясность изложения |
| `actionability` | Практическая применимость |
| `custom` | Кастомный критерий с промптом |

**Пример конфигурации**:

```yaml
assertions:
  - type: llm_eval
    config:
      artifact: "report.md"
      criteria: factual_accuracy
      threshold: 0.8

  - type: llm_eval
    config:
      artifact: "report.md"
      criteria: custom
      prompt: |
        Evaluate if the competitor analysis includes:
        1. Market share estimates for each competitor
        2. Key differentiating features
        3. Pricing comparison where available
        4. SWOT analysis or similar framework

        Score from 0 to 1 based on how many criteria are met.
      threshold: 0.7
```

**Evaluation Prompt Template**:

```python
JUDGE_PROMPT = """
You are evaluating the output of an AI agent that was given the following task:

<task>
{task_description}
</task>

The agent produced this artifact:

<artifact>
{artifact_content}
</artifact>

Evaluate the artifact on the following criterion: {criteria}

{criteria_description}

{custom_prompt}

Provide your evaluation in the following JSON format:
{{
  "score": <float between 0 and 1>,
  "explanation": "<brief explanation of your score>",
  "issues": ["<list of specific issues found>"],
  "strengths": ["<list of specific strengths>"]
}}
"""

CRITERIA_DESCRIPTIONS = {
    "factual_accuracy": """
        Check if the facts, statistics, and claims in the artifact are accurate.
        Consider: Are company names real? Are statistics plausible? Are dates correct?
        Score 1.0 for fully accurate, 0.0 for mostly inaccurate.
    """,
    "completeness": """
        Check if the artifact addresses all aspects of the task.
        Consider: Are all requested sections present? Is each section substantive?
        Score 1.0 for comprehensive, 0.0 for missing major elements.
    """,
    # ... etc
}
```

**Calibration**:

LLM judges have known biases. Calibration strategies:

1. **Reference examples**: Provide graded examples in prompt
2. **Multi-judge**: Use multiple LLMs, aggregate scores
3. **Human baseline**: Periodically compare with human evaluations
4. **Confidence intervals**: Report uncertainty in LLM scores

### 4. Code Execution Evaluator

Запускает сгенерированный агентом код и анализирует результаты.

**Типы проверок**:

| Assertion Type | Description |
|----------------|-------------|
| `pytest` | Запуск pytest на коде агента |
| `npm_test` | Запуск npm test |
| `custom_command` | Произвольная команда |
| `lint` | Статический анализ (ruff, eslint) |
| `typecheck` | Проверка типов (mypy, tsc) |

**Пример конфигурации**:

```yaml
assertions:
  - type: code_execution
    config:
      type: pytest
      target: "agent_output/"
      options:
        - "--tb=short"
        - "-v"
      timeout: 60

  - type: code_execution
    config:
      type: lint
      tool: ruff
      target: "agent_output/"
      fail_on_warning: false

  - type: code_execution
    config:
      type: custom_command
      command: "python agent_output/main.py --validate"
      expected_exit_code: 0
      expected_output_contains: "Validation passed"
```

**Sandbox Execution**:

```python
async def run_code_check(
    config: CodeExecutionConfig,
    workspace: Path,
) -> CodeExecutionResult:
    # Build Docker command
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{workspace}:/workspace",
        "--network", "none",  # No network access
        "--memory", "512m",
        "--cpus", "1",
        f"atp-code-runner:{config.runtime}",
    ]

    if config.type == "pytest":
        docker_cmd.extend(["pytest", "/workspace", *config.options])
    elif config.type == "custom_command":
        docker_cmd.extend(["sh", "-c", config.command])

    # Run with timeout
    proc = await asyncio.create_subprocess_exec(
        *docker_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=config.timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        return CodeExecutionResult(
            passed=False,
            exit_code=-1,
            error="Execution timeout",
        )

    return CodeExecutionResult(
        passed=proc.returncode == config.expected_exit_code,
        exit_code=proc.returncode,
        stdout=stdout.decode(),
        stderr=stderr.decode(),
    )
```

**Parsing Test Results**:

```python
def parse_pytest_output(output: str) -> TestResults:
    """Parse pytest output to extract test counts."""
    # Match: "5 passed, 2 failed, 1 skipped"
    pattern = r"(\d+) passed|(\d+) failed|(\d+) skipped|(\d+) error"
    matches = re.findall(pattern, output)

    passed = failed = skipped = errors = 0
    for m in matches:
        if m[0]: passed = int(m[0])
        if m[1]: failed = int(m[1])
        if m[2]: skipped = int(m[2])
        if m[3]: errors = int(m[3])

    total = passed + failed + skipped + errors

    return TestResults(
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=errors,
        total=total,
        pass_rate=passed / total if total > 0 else 0.0,
    )
```

---

## Scoring System

### Score Aggregation

Итоговый скор вычисляется как взвешенная сумма компонентов:

```
Score = w_Q × Quality + w_C × Completeness + w_E × Efficiency + w_$ × Cost
```

Где:
- **Quality** (Q): средний score от Artifact и LLM evaluators
- **Completeness** (C): доля пройденных behavior checks
- **Efficiency** (E): нормализованная эффективность (меньше шагов = лучше)
- **Cost** ($): нормализованная стоимость (меньше токенов = лучше)

**Default Weights**:

```yaml
scoring:
  quality_weight: 0.4
  completeness_weight: 0.3
  efficiency_weight: 0.2
  cost_weight: 0.1
```

### Normalization

**Efficiency Score**:

```python
def normalize_efficiency(
    actual_steps: int,
    max_steps: int,
    optimal_steps: int | None = None,
) -> float:
    """
    Efficiency = 1.0 when steps <= optimal
    Efficiency = 0.0 when steps >= max
    Linear interpolation between.
    """
    if optimal_steps is None:
        optimal_steps = max_steps // 4

    if actual_steps <= optimal_steps:
        return 1.0
    if actual_steps >= max_steps:
        return 0.0

    return 1.0 - (actual_steps - optimal_steps) / (max_steps - optimal_steps)
```

**Cost Score**:

```python
def normalize_cost(
    actual_tokens: int,
    max_tokens: int,
    budget_usd: float | None = None,
) -> float:
    """
    Cost score based on token usage.
    Uses log scale to not overly penalize slightly higher usage.
    """
    if actual_tokens == 0:
        return 1.0

    ratio = actual_tokens / max_tokens

    # Log scale: score = 1 - log(1 + ratio) / log(2)
    # At ratio=0: score=1.0
    # At ratio=1: score=0.0
    score = 1.0 - math.log(1 + ratio) / math.log(2)
    return max(0.0, score)
```

### Statistical Aggregation

При множественных прогонах вычисляется статистика:

```python
@dataclass
class StatisticalScore:
    mean: float
    std: float
    min: float
    max: float
    median: float
    confidence_interval: tuple[float, float]
    n_runs: int

def compute_statistics(scores: list[float], confidence: float = 0.95) -> StatisticalScore:
    n = len(scores)
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if n > 1 else 0.0

    # Confidence interval (t-distribution for small samples)
    if n > 1:
        t_value = scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_value * std / math.sqrt(n)
        ci = (mean - margin, mean + margin)
    else:
        ci = (mean, mean)

    return StatisticalScore(
        mean=mean,
        std=std,
        min=min(scores),
        max=max(scores),
        median=statistics.median(scores),
        confidence_interval=ci,
        n_runs=n,
    )
```

### Variance Analysis

Высокий variance указывает на нестабильность агента:

```python
def assess_stability(stats: StatisticalScore) -> StabilityAssessment:
    cv = stats.std / stats.mean if stats.mean > 0 else float('inf')  # Coefficient of variation

    if cv < 0.05:
        level = "stable"
        message = "Agent produces consistent results"
    elif cv < 0.15:
        level = "moderate"
        message = "Some variance in results, generally acceptable"
    elif cv < 0.30:
        level = "unstable"
        message = "High variance - results may be unreliable"
    else:
        level = "critical"
        message = "Extremely high variance - agent behavior is unpredictable"

    return StabilityAssessment(
        level=level,
        coefficient_of_variation=cv,
        message=message,
    )
```

---

## Test Definition Format

### Complete Test Example

```yaml
# tests/competitor_analysis.yaml
test_suite: "Competitor Analysis Tests"
version: "1.0"
description: "Tests for market research and competitor analysis agents"

# Default settings for all tests in suite
defaults:
  runs_per_test: 5
  timeout_seconds: 300
  scoring:
    quality_weight: 0.4
    completeness_weight: 0.3
    efficiency_weight: 0.2
    cost_weight: 0.1

# Agent configurations
agents:
  - name: langgraph-research
    adapter: langgraph
    config_path: "./agents/langgraph_agent.py"

  - name: crewai-research
    adapter: crewai
    config_path: "./agents/crewai_agent.py"

# Test cases
tests:
  - id: basic_competitor_search
    name: "Find competitors for known company"
    description: "Agent should find and analyze top competitors"
    tags: [smoke, core]

    # Task definition
    task:
      description: |
        Find the top 5 competitors for Slack in the enterprise
        communication and collaboration market. For each competitor,
        provide:
        - Company name and brief description
        - Estimated market share (if available)
        - Key differentiating features
        - Pricing tier (enterprise/SMB/freemium)

        Output a markdown report with an executive summary and
        detailed analysis of each competitor.

      input_data:
        company: "Slack"
        market: "enterprise communication"

      expected_artifacts:
        - type: file
          format: markdown
          name: "report.md"

    # Constraints
    constraints:
      max_steps: 30
      max_tokens: 50000
      timeout_seconds: 180
      allowed_tools:
        - web_search
        - file_write

    # Assertions
    assertions:
      # Artifact checks
      - type: artifact_exists
        config:
          path: "report.md"

      - type: sections_exist
        config:
          artifact: "report.md"
          sections:
            - "Executive Summary"
            - "Competitor Analysis"

      - type: contains
        config:
          artifact: "report.md"
          pattern: "Microsoft Teams|Zoom|Google"
          regex: true
          min_matches: 3

      - type: min_length
        config:
          artifact: "report.md"
          chars: 2000

      # Behavior checks
      - type: behavior
        config:
          must_use_tools:
            - web_search
          max_tool_calls: 15

      # LLM evaluation
      - type: llm_eval
        config:
          artifact: "report.md"
          criteria: factual_accuracy
          threshold: 0.75

      - type: llm_eval
        config:
          artifact: "report.md"
          criteria: completeness
          threshold: 0.8

    # Scoring weights (override defaults)
    scoring:
      quality_weight: 0.5
      completeness_weight: 0.3
      efficiency_weight: 0.15
      cost_weight: 0.05

  - id: unknown_company_handling
    name: "Handle unknown company gracefully"
    description: "Agent should indicate uncertainty for unknown companies"
    tags: [edge_case, error_handling]

    task:
      description: |
        Find competitors for "XyzNonexistent123 Corp" in the
        quantum computing market.

    assertions:
      - type: behavior
        config:
          should_indicate_uncertainty: true
          must_not_hallucinate: true

      - type: llm_eval
        config:
          criteria: custom
          prompt: |
            The task asked about a non-existent company.
            Evaluate if the agent:
            1. Clearly indicated it couldn't find the company
            2. Did NOT make up fake competitor information
            3. Suggested alternative approaches or asked for clarification

            Score 1.0 if all criteria met, 0.0 if hallucinated data.
          threshold: 0.9

  - id: performance_large_market
    name: "Performance on large market analysis"
    description: "Test efficiency on complex market"
    tags: [performance]

    task:
      description: |
        Analyze the global cloud infrastructure market and identify
        the top 10 providers with detailed analysis.

    constraints:
      max_steps: 50
      max_tokens: 100000
      timeout_seconds: 600

    assertions:
      - type: artifact_exists
        config:
          path: "report.md"

      - type: behavior
        config:
          max_tool_calls: 30

    scoring:
      efficiency_weight: 0.4  # Higher weight for performance test
```

---

## Extending the Evaluation System

### Custom Evaluator

```python
# my_evaluators/domain_specific.py
from atp.evaluators.base import Evaluator, EvalResult, EvalCheck

class DomainSpecificEvaluator(Evaluator):
    """Custom evaluator for domain-specific checks."""

    name = "domain_specific"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        checks = []
        config = assertion.config

        if assertion.type == "financial_accuracy":
            # Domain-specific check for financial data
            artifact = self._get_artifact(response, config["artifact"])
            accuracy = await self._check_financial_data(artifact)
            checks.append(EvalCheck(
                name="financial_accuracy",
                passed=accuracy >= config.get("threshold", 0.9),
                score=accuracy,
            ))

        return EvalResult(evaluator=self.name, checks=checks)

    async def _check_financial_data(self, content: str) -> float:
        # Custom validation logic
        pass
```

### Registering Custom Evaluator

```yaml
# atp.config.yaml
evaluators:
  custom:
    - module: my_evaluators.domain_specific
      class: DomainSpecificEvaluator
```

```python
# Or programmatically
from atp.core.registry import evaluator_registry
from my_evaluators import DomainSpecificEvaluator

evaluator_registry.register(DomainSpecificEvaluator())
```

---

## Best Practices

### Writing Good Assertions

1. **Start with must-haves**: artifact exists, required sections present
2. **Add behavioral constraints**: tool usage, step limits
3. **Include semantic checks**: LLM eval for quality
4. **Consider edge cases**: error handling, unknown inputs

### Balancing Determinism and Flexibility

- Use deterministic checks for structure (exists, format, contains)
- Use LLM eval for semantic quality (accuracy, completeness)
- Set appropriate thresholds (not 1.0 for LLM evals)
- Run multiple times for statistical confidence

### Avoiding False Positives/Negatives

- **False positives** (bad result passes):
  - Tighten thresholds
  - Add more specific assertions
  - Include negative cases (must_not_contain)

- **False negatives** (good result fails):
  - Use flexible patterns (regex with alternatives)
  - Lower thresholds for LLM evals
  - Increase runs for statistical stability

### Performance Considerations

- LLM eval is expensive — use sparingly
- Cache deterministic results
- Run expensive checks last (fail-fast on cheap checks)
- Limit artifact content sent to LLM judge

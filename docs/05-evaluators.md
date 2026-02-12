# Evaluation System

## Overview

The ATP evaluation system analyzes agent execution results and computes quality metrics. Evaluation happens at multiple levels: from simple structural checks to complex semantic evaluation using LLM.

## Evaluation Philosophy

### Principles

1. **Composability** — evaluators combine for comprehensive assessment
2. **Transparency** — every evaluation is explainable
3. **Determinism where possible** — deterministic checks are preferred
4. **Statistical validity** — stochastic evaluations are averaged across runs

### Evaluator Hierarchy

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

Checks artifact presence, format, and content.

**Check Types**:

| Assertion Type | Description | Example |
|----------------|-------------|---------|
| `artifact_exists` | Artifact with specified path exists | `path: "report.md"` |
| `artifact_format` | Artifact matches format | `format: "markdown"` |
| `artifact_schema` | Structured artifact matches JSON Schema | `schema: {...}` |
| `contains` | Artifact contains text/pattern | `pattern: "Competitor"` |
| `not_contains` | Artifact does not contain text | `text: "error"` |
| `min_length` | Minimum content length | `chars: 1000` |
| `max_length` | Maximum content length | `chars: 50000` |
| `sections_exist` | Markdown contains sections | `sections: ["Summary", "Details"]` |
| `table_exists` | Markdown contains table | `min_rows: 3` |

**Configuration Example**:

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

Analyzes execution trace: which tools were used, how many steps, what errors occurred.

**Check Types**:

| Assertion Type | Description | Example |
|----------------|-------------|---------|
| `must_use_tools` | Required tool usage | `tools: ["web_search"]` |
| `must_not_use_tools` | Prohibited tools | `tools: ["dangerous_tool"]` |
| `max_tool_calls` | Call limit | `limit: 10` |
| `max_steps` | Step limit | `limit: 50` |
| `no_errors` | No errors in trace | - |
| `tool_sequence` | Tool usage order | `sequence: ["search", "write"]` |
| `no_hallucination` | No fabricated data (requires LLM) | `check_facts: true` |

**Configuration Example**:

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

Uses LLM for semantic evaluation of result quality.

**Evaluation Criteria**:

| Criteria | Description |
|----------|-------------|
| `factual_accuracy` | Factual correctness |
| `completeness` | Response completeness |
| `relevance` | Relevance to task |
| `coherence` | Logic and coherence |
| `clarity` | Clarity of presentation |
| `actionability` | Practical applicability |
| `custom` | Custom criterion with prompt |

**Configuration Example**:

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

Runs agent-generated code and analyzes results.

**Check Types**:

| Assertion Type | Description |
|----------------|-------------|
| `pytest` | Run pytest on agent code |
| `npm_test` | Run npm test |
| `custom_command` | Arbitrary command |
| `lint` | Static analysis (ruff, eslint) |
| `typecheck` | Type checking (mypy, tsc) |

**Configuration Example**:

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

### 5. Security Evaluator

Analyzes agent outputs for security vulnerabilities including PII exposure, prompt injection, dangerous code, and secret leaks.

**Check Types**:

| Assertion Type | Description | Severity Range |
|----------------|-------------|----------------|
| `pii_exposure` | Detect personally identifiable information | Medium - Critical |
| `prompt_injection` | Detect injection and jailbreak attempts | Low - High |
| `code_safety` | Detect dangerous code patterns | Medium - High |
| `secret_leak` | Detect leaked secrets and credentials | Medium - Critical |

**Configuration Example**:

```yaml
assertions:
  - type: security
    config:
      checks:
        - pii_exposure
        - prompt_injection
        - code_safety
        - secret_leak
      sensitivity: "medium"  # info, low, medium, high, critical
      fail_on_warning: false

  - type: security
    config:
      checks:
        - pii_exposure
      pii_types:
        - email
        - ssn
        - credit_card
        - api_key
      sensitivity: "high"
      fail_on_warning: true

  - type: security
    config:
      checks:
        - secret_leak
      secret_types:
        - private_key
        - jwt_token
        - connection_string
        - aws_credential
```

**Severity Levels**:

| Level | Score Impact | Pass/Fail |
|-------|--------------|-----------|
| Critical | 0.0 | Fail |
| High | 0.0 | Fail |
| Medium | 0.5 | Pass (unless fail_on_warning) |
| Low | 0.9 | Pass |
| Info | 0.95 | Pass |

**Security Checkers**:

1. **PII Checker** - Detects: email, phone, SSN, credit cards, API keys
2. **Prompt Injection Checker** - Detects: instruction override, jailbreak, role manipulation
3. **Code Safety Checker** - Detects: dangerous imports, eval/exec, file/network operations
4. **Secret Leak Checker** - Detects: private keys, JWT, bearer tokens, connection strings, AWS credentials

**Report Output**:

```json
{
  "evaluator": "security",
  "passed": false,
  "score": 0.0,
  "checks": [{
    "name": "security_scan",
    "message": "Found 2 security issue(s): 1 critical, 1 high",
    "details": {
      "findings_count": 2,
      "critical_count": 1,
      "high_count": 1,
      "findings": [...],
      "remediations": [...]
    }
  }]
}
```

> For detailed documentation, see [Security Evaluator Guide](guides/security-evaluator.md).

### 6. Filesystem Evaluator

Checks the actual filesystem state in the agent's workspace after execution. Unlike the Artifact Evaluator (which checks response artifacts in memory), the Filesystem Evaluator inspects real files on disk.

**Check Types**:

| Assertion Type | Description | Example |
|----------------|-------------|---------|
| `file_exists` | File exists at path | `path: "output.txt"` |
| `file_not_exists` | File does NOT exist | `path: "temp.txt"` |
| `file_contains` | File content matches pattern | `path: "out.txt", pattern: "OK"` |
| `dir_exists` | Directory exists | `path: "reports/"` |
| `file_count` | Number of files in directory | `path: "output/", count: 3` |

**Configuration Example**:

```yaml
assertions:
  - type: file_exists
    config:
      path: "output/report.json"

  - type: file_not_exists
    config:
      path: "temp/scratch.txt"

  - type: file_contains
    config:
      path: "output/report.json"
      pattern: '"status":\s*"success"'
      regex: true

  - type: dir_exists
    config:
      path: "output/charts"

  - type: file_count
    config:
      path: "output"
      count: 3
      operator: "gte"    # eq, gt, gte, lt, lte
```

**Workspace Fixtures**:

Pre-populate the agent's workspace with files before the test runs using `workspace_fixture` in the task definition:

```yaml
task:
  description: "Reorganize project files"
  workspace_fixture: "tests/fixtures/test_filesystem/basic"
```

The fixture directory is copied into a fresh sandbox workspace. Each test run starts with a clean copy — agents cannot damage the fixture.

> For detailed documentation, see [Test Filesystem Guide](guides/test-filesystem.md).

---

### 6a. Factuality Evaluator

Checks factual accuracy of agent outputs by cross-referencing claims against known data sources and using LLM-based verification. Useful for research tasks where accuracy is critical.

**Configuration Example**:

```yaml
assertions:
  - type: factuality
    config:
      artifact: "report.md"
      threshold: 0.8
      check_claims: true
```

### 6b. Style Evaluator

Assesses the style and formatting quality of agent outputs, including tone, readability, and adherence to style guidelines.

**Configuration Example**:

```yaml
assertions:
  - type: style
    config:
      artifact: "report.md"
      guidelines:
        - professional_tone
        - markdown_formatting
      threshold: 0.7
```

### 6c. Performance Evaluator

Measures agent execution performance metrics including latency, throughput, and resource utilization.

**Configuration Example**:

```yaml
assertions:
  - type: performance
    config:
      max_latency_ms: 5000
      max_memory_mb: 512
```

---

## Game-Theoretic Evaluators

Phase 5 adds four evaluators for assessing agents in multi-agent strategic interactions. These evaluators are provided by the `atp-games` plugin and integrate with the standard ATP scoring pipeline.

> **Prerequisite**: Install the `atp-games` package. Evaluators are auto-registered via the plugin system when `atp-games` is installed.
>
> For the standalone game library, see [game-environments README](../game-environments/README.md).
> For the ATP plugin, see [atp-games README](../atp-games/README.md).
> For runnable examples, see [examples/games/](../examples/games/).

### Overview

Game-theoretic evaluators analyze how agents behave in multi-agent strategic interactions. Unlike traditional evaluators that check artifacts or traces, these evaluators measure strategic properties:

| Evaluator | What It Measures | Key Question |
|-----------|-----------------|--------------|
| **Payoff** | Outcome quality | How well does the agent perform? |
| **Exploitability** | Strategic vulnerability | Can an opponent take advantage of this agent? |
| **Cooperation** | Cooperative behavior | Does the agent cooperate and reciprocate? |
| **Equilibrium** | Theoretical optimality | Is the agent playing a Nash equilibrium? |

### End-to-End Example

```python
import asyncio
from game_envs import PrisonersDilemma, PDConfig, TitForTat, AlwaysDefect
from atp_games import GameRunner, GameRunConfig, BuiltinAdapter
from atp_games.evaluators.payoff_evaluator import PayoffEvaluator, PayoffConfig
from atp_games.evaluators.cooperation_evaluator import CooperationEvaluator

async def evaluate():
    # Set up game and agents
    game = PrisonersDilemma(PDConfig(num_rounds=50))
    agents = {
        "player_0": BuiltinAdapter(TitForTat()),
        "player_1": BuiltinAdapter(AlwaysDefect()),
    }

    # Run multi-episode evaluation
    runner = GameRunner()
    result = await runner.run_game(
        game=game, agents=agents,
        config=GameRunConfig(episodes=20, base_seed=42),
    )

    # Evaluate payoffs
    payoff_eval = PayoffEvaluator(
        PayoffConfig(min_payoff={"player_0": 50.0}, min_social_welfare=100.0)
    )
    payoff_result = payoff_eval.evaluate_game(result)
    for check in payoff_result.checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.message}")

    # Evaluate cooperation
    coop_eval = CooperationEvaluator()
    coop_result = coop_eval.evaluate_game(result)
    for check in coop_result.checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.message}")

asyncio.run(evaluate())
```

See [examples/games/llm_agent_eval.py](../examples/games/llm_agent_eval.py) for a complete runnable example.

### 7. Payoff Evaluator

Evaluates game outcomes based on payoff metrics.

**Checks:**

| Check | Description | Metric |
|-------|-------------|--------|
| `average_payoff` | Average payoff per player across episodes | Per-player mean with threshold |
| `payoff_distribution` | Distribution statistics (min, max, median, p25, p75) | Informational |
| `social_welfare` | Sum of average payoffs across all players | Total with threshold |
| `pareto_efficiency` | Whether average outcome is Pareto dominated | Boolean |

**Configuration Example:**

```yaml
assertions:
  - type: average_payoff
    config:
      min_payoff:
        player_0: 2.0        # Minimum average payoff for player_0
      min_social_welfare: 4.0  # Minimum total welfare
      pareto_check: true       # Enable Pareto efficiency check
      weights:
        average_payoff: 1.0
        social_welfare: 1.0
        pareto_efficiency: 1.0
```

**Scoring Logic:**

- Average payoff score: ratio of actual to threshold (0-1)
- Social welfare score: ratio of actual to threshold (0-1)
- Pareto efficiency: 1.0 if not dominated, 0.0 if dominated

**Typical Results:**

| Matchup (50 rounds) | Player 0 Avg Payoff | Social Welfare | Pareto Efficient |
|---------------------|---------------------|----------------|------------------|
| TFT vs TFT | ~150 | ~300 | Yes |
| TFT vs AlwaysDefect | ~54 | ~159 | No |
| AlwaysDefect vs AlwaysDefect | ~50 | ~100 | No |

### 8. Exploitability Evaluator

Measures how exploitable an agent's strategy is by computing the payoff gap between their empirical play and the best response.

**Checks:**

| Check | Description | Metric |
|-------|-------------|--------|
| `per_player_exploitability` | Exploitability per player vs epsilon threshold | Per-player gap |
| `total_exploitability` | Sum of per-player exploitability | Total gap |
| `empirical_strategy` | Extracted empirical strategy distribution | Informational |

**Configuration Example:**

```yaml
assertions:
  - type: exploitability
    config:
      epsilon: 0.15            # Max exploitability for pass
      payoff_matrix_1:         # Row player payoff matrix
        - [3, 0]
        - [5, 1]
      payoff_matrix_2:         # Column player payoff matrix
        - [3, 5]
        - [0, 1]
      action_names_1: ["cooperate", "defect"]
      action_names_2: ["cooperate", "defect"]
```

**Interpretation:**

- Exploitability ~ 0: agent plays near Nash equilibrium (not exploitable)
- High exploitability: opponent can achieve much higher payoff by deviating to best response
- Example: AlwaysCooperate in Prisoner's Dilemma has high exploitability (opponent gains by defecting)

**Strategy Exploitability Reference:**

| Strategy | Exploitability | Why |
|----------|---------------|-----|
| AlwaysDefect | 0.0 | Plays the dominant strategy (Nash equilibrium) |
| Mixed (0.5, 0.5) | Low | Close to equilibrium |
| AlwaysCooperate | High | Opponent gains 2 by switching to defect |

### 9. Cooperation Evaluator

Measures cooperative behavior patterns in games with cooperative/defective action choices (primarily Prisoner's Dilemma).

**Checks:**

| Check | Description | Metric |
|-------|-------------|--------|
| `cooperation_rate` | Fraction of cooperative actions per player | 0-1 per player |
| `conditional_cooperation` | P(Cooperate\|opponent Cooperated) and P(C\|D) | Probability pair |
| `reciprocity` | Correlation of cooperative actions between players | -1 to +1 |

**Configuration Example:**

```yaml
assertions:
  - type: cooperation
    config:
      min_cooperation_rate:
        player_0: 0.6          # At least 60% cooperation
      min_reciprocity: 0.3     # Positive reciprocity
      cooperative_actions:
        - cooperate
        - c
```

**Interpretation:**

- TFT vs TFT: cooperation rate ~ 1.0, reciprocity ~ 1.0
- AllD vs AllC: cooperation rate ~ 0.5 overall, reciprocity ~ -1.0
- Conditional cooperation reveals strategy type (e.g., P(C|C) > P(C|D) indicates conditional cooperator)

**Strategy Fingerprints:**

| Strategy | Coop Rate | P(C\|C) | P(C\|D) | Reciprocity |
|----------|-----------|---------|---------|-------------|
| TitForTat | Depends on opponent | ~1.0 | ~0.0 | High (+) |
| AlwaysCooperate | 1.0 | 1.0 | 1.0 | 0.0 |
| AlwaysDefect | 0.0 | 0.0 | 0.0 | 0.0 |
| Pavlov | Depends on opponent | High | Low | Moderate (+) |
| GrimTrigger | Depends on opponent | 1.0 | 0.0 | High (+) |

### 10. Equilibrium Evaluator

Measures proximity to Nash equilibrium and detects convergence in strategy over time.

**Checks:**

| Check | Description | Metric |
|-------|-------------|--------|
| `nash_distance` | L1 distance from empirical strategy to nearest NE | Distance value |
| `equilibrium_type` | Classification of found equilibria (pure/mixed count) | Informational |
| `convergence` | Whether strategy stabilizes over rounds | Boolean + change magnitude |

**Configuration Example:**

```yaml
assertions:
  - type: equilibrium
    config:
      max_nash_distance: 0.5   # Maximum distance from NE for pass
      convergence_window: 20   # Rounds to check for convergence
      convergence_threshold: 0.1  # Max strategy change between halves
      payoff_matrix_1:
        - [3, 0]
        - [5, 1]
      payoff_matrix_2:
        - [3, 5]
        - [0, 1]
      action_names_1: ["cooperate", "defect"]
      action_names_2: ["cooperate", "defect"]
      solver_method: support_enumeration  # or lemke_howson, fictitious_play
```

**Convergence Detection:**

The evaluator splits the history into two halves (within a sliding window) and compares the empirical strategy distributions. If the L1 distance between halves is below the threshold, the agent is considered converged. This detects whether agents "settle" on a strategy or keep oscillating.

### YAML Game Suite Format

Game-theoretic evaluators are typically used via YAML game suites rather than the standard test suite format. A game suite defines the game, agents, and evaluation criteria in one file:

```yaml
type: game_suite
name: PD Cooperation Test
version: "1.0"

game:
  type: prisoners_dilemma
  variant: repeated
  config:
    num_rounds: 100
    noise: 0.0

agents:
  - name: my_agent
    adapter: http
    endpoint: http://localhost:8000

  - name: baseline
    adapter: builtin
    strategy: tit_for_tat

evaluation:
  episodes: 50
  metrics:
    - type: average_payoff
      weight: 1.0
    - type: cooperation
      weight: 0.5
      config:
        min_cooperation_rate:
          player_0: 0.6
    - type: exploitability
      weight: 0.5
      config:
        epsilon: 0.15
```

Run with: `uv run atp test --suite=game:prisoners_dilemma.yaml`

See [atp-games README](../atp-games/README.md) for the full YAML reference and tournament modes.

### Built-in Games

Five canonical games are available, each with known Nash equilibria:

| Game | Players | Action Space | Nash Equilibrium |
|------|---------|-------------|-----------------|
| Prisoner's Dilemma | 2 | Discrete | (Defect, Defect) |
| Public Goods | 2-20 | Continuous | Free-ride (contribute 0) |
| Auction | 2+ | Continuous | Truthful bidding (2nd price) |
| Colonel Blotto | 2 | Structured | Mixed strategy |
| Congestion | 2-50 | Discrete | Wardrop equilibrium |

See [game-environments README](../game-environments/README.md) for game configs, strategies, and analysis tools.

---

## Evaluator Comparison Matrix

| Evaluator | Type | Deterministic | Cost | Use Case |
|-----------|------|---------------|------|----------|
| **Artifact** | Structural | Yes | Low | Check file existence, format, content |
| **Behavior** | Trace Analysis | Yes | Low | Verify tool usage, step limits, errors |
| **LLM Judge** | Semantic | No | High | Evaluate quality, accuracy, completeness |
| **Code Execution** | Runtime | Yes | Medium | Run tests, lint, type-check code |
| **Security** | Pattern-based | Yes | Low | Detect PII, secrets, injections, unsafe code |
| **Filesystem** | Structural | Yes | Low | Check workspace files on disk |
| **Factuality** | Semantic | No | High | Verify factual accuracy of outputs |
| **Style** | Semantic | No | Medium | Assess tone, formatting, readability |
| **Performance** | Metrics | Yes | Low | Measure latency, throughput, resources |
| **Payoff** | Game-theoretic | Yes | Low | Average payoff, social welfare, Pareto check |
| **Exploitability** | Game-theoretic | Yes | Medium | Best-response gap, NE proximity |
| **Cooperation** | Game-theoretic | Yes | Low | Cooperation rate, reciprocity, conditional coop |
| **Equilibrium** | Game-theoretic | Yes | Medium | Nash distance, convergence, equilibrium type |

### When to Use Each Evaluator

| Scenario | Recommended Evaluators |
|----------|------------------------|
| File creation task | Artifact (exists, format) |
| Data processing | Artifact (schema), LLM (accuracy) |
| Code generation | Code Execution (tests), Security (code_safety) |
| Research task | Artifact (sections), LLM (completeness, accuracy) |
| User data handling | Security (pii_exposure, secret_leak) |
| Chat/dialog agent | Security (prompt_injection), Behavior (no_errors) |
| Game-theoretic evaluation | Payoff, Exploitability, Cooperation, Equilibrium |
| Strategic reasoning | Exploitability (NE proximity), Equilibrium (convergence) |
| Multi-agent cooperation | Cooperation (rate, reciprocity), Payoff (social welfare) |
| Agent robustness | Exploitability (best-response gap), Payoff (distribution) |

---

## Scoring System

### Score Aggregation

The final score is computed as a weighted sum of components:

```
Score = w_Q × Quality + w_C × Completeness + w_E × Efficiency + w_$ × Cost
```

Where:
- **Quality** (Q): average score from Artifact and LLM evaluators
- **Completeness** (C): fraction of passed behavior checks
- **Efficiency** (E): normalized efficiency (fewer steps = better)
- **Cost** ($): normalized cost (fewer tokens = better)

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

For multiple runs, statistics are computed:

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

High variance indicates agent instability:

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

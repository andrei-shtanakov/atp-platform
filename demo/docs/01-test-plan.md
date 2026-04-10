# Test Plan: Code Writer Agents (OpenAI vs Anthropic)

## 1. Goal

**What we test**: two AI agents, each accepting a natural-language task and
returning a ready-to-run Python program.

| Agent | Model | API | Endpoint |
|-------|-------|-----|----------|
| Agent A | GPT-4o | OpenAI | http://localhost:8001 |
| Agent B | Claude Sonnet 4 | Anthropic | http://localhost:8002 |

**Why**: compare code-generation quality, speed, cost, and reliability.

## 2. Test scope

| Category | # of tests | Priority | What we verify |
|----------|:---:|:--------:|----------------|
| Smoke | 3 | Critical | Agent responds, code is syntactically valid |
| Functional | 3 | High | Code solves the task, passes unit tests |
| Code quality | 3 | Medium | Style, readability, documentation |
| **Total** | **9** | | |

## 3. Agent tasks

| ID | Task | Complexity | Verification |
|----|------|------------|--------------|
| T1 | Fibonacci function (n-th number) | Simple | pytest: correctness, edge cases |
| T2 | CSV parser with filtering | Medium | pytest: read, filter, edge cases |
| T3 | REST API client (httpx) | Complex | pytest: requests, error handling |

## 4. Acceptance criteria

### 4.1. Mandatory (blocking)
- All smoke tests: **PASS**
- Generated code is valid Python (compiles)
- At least one functional test: **PASS**

### 4.2. Desirable
- All functional tests: PASS
- Quality score >= 0.75
- Latency < 30s per task
- Cost < $0.05 per task

## 5. Comparison metrics

| Metric | How we measure | Weight |
|--------|----------------|--------|
| Correctness | % of passing unit tests | 0.40 |
| Completeness | LLM-eval: are all requirements covered | 0.25 |
| Code quality | LLM-eval: style, readability, PEP 8 | 0.20 |
| Efficiency | Tokens + latency + cost | 0.15 |

## 6. Schedule

| Trigger | Suite | Runs |
|---------|-------|------|
| Manual run | smoke.yaml | 1 |
| Full comparison | functional.yaml + quality.yaml | 3 |
| Baseline | All suites | 5 |

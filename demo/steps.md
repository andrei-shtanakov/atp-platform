# Step-by-step plan: testing two Code Writer agents

## Goal
Compare two AI agents that write Python code:
- **Agent A**: OpenAI (GPT-4o)
- **Agent B**: Anthropic (Claude Sonnet 4)

Both receive the same task → write a Python program → ATP scores the result.

---

## Steps

### Step 1: Test goals and scope ✅
- [x] Decide what we test (two code writer agents)
- [x] Define test categories
- [x] Define acceptance criteria

**Result**: `docs/01-test-plan.md`

### Step 2: Contract (ATP Protocol) ✅
- [x] Describe the ATPRequest format (input_data: the code task)
- [x] Describe the ATPResponse format (artifacts: the Python file)
- [x] Describe the metrics (tokens, cost, time)
- [x] JSON Schema to validate the response

**Result**: `docs/02-contract.md`

### Step 3: Environment ✅
- [x] Create the directory layout
- [x] Create atp.config.yaml (two agents: ports 8001, 8002)
- [x] Set up .env.example (API keys)
- [x] JSON Schema to validate responses

**Result**: `atp.config.yaml`, `.env.example`, `fixtures/response_schema.json`

### Step 4: Test suite (YAML) ✅
- [x] Smoke tests: SM-001 (file exists), SM-002 (schema), SM-003 (compiles)
- [x] Functional: FN-001 (fibonacci), FN-002 (csv_parser), FN-003 (api_client)
- [x] Quality: QL-001 (docstrings), QL-002 (PEP 8), QL-003 (edge cases)

**Result**: `test_suites/smoke.yaml`, `test_suites/functional.yaml`, `test_suites/quality.yaml`

### Step 5: Fixtures ✅
- [x] Code tasks: fibonacci, csv_parser, api_client
- [x] Pytest tests: test_fibonacci (10 tests), test_csv_parser (10 tests), test_api_client (7 tests)
- [x] JSON Schema to validate responses (step 3)

**Result**: `fixtures/tasks/*.md`, `fixtures/tests/test_*.py`, `fixtures/response_schema.json`

### Step 6: Agents ✅
- [x] HTTP agent backed by OpenAI API (FastAPI, port 8001)
- [x] HTTP agent backed by Anthropic API (FastAPI, port 8002)
- [x] Shared system prompt for a fair comparison
- [x] strip_markdown_fences() — removes ```python fences
- [x] Cost accounting based on current prices

**Result**: `agents/openai_agent.py`, `agents/anthropic_agent.py`

### Step 7: Adapter and execution
- [x] Run guide documentation (docs/03-run-guide.md)
- [ ] Run smoke tests for each agent
- [ ] Confirm that both agents work

**Result**: `docs/03-run-guide.md`, passing tests

### Step 8: Comparison and baseline
- [x] Baseline and comparison instructions (docs/03-run-guide.md §5-6)
- [ ] Run the full suite for both agents
- [ ] Save the baselines
- [ ] Compare the results

**Result**: `baselines/`, comparison report

---

## Target directory layout

```
atp-platform-artefacts/
├── steps.md                    # This file
├── atp.config.yaml             # ATP configuration
├── .env.example                # Environment variable template
├── docs/
│   ├── 01-test-plan.md         # Test plan
│   └── 02-contract.md          # ATP Protocol contract
├── agents/
│   ├── openai_agent.py         # Agent A: OpenAI GPT-4o
│   └── anthropic_agent.py      # Agent B: Anthropic Claude
├── test_suites/
│   ├── smoke.yaml              # Smoke tests
│   ├── functional.yaml         # Functional tests
│   └── quality.yaml            # Code quality tests
├── fixtures/
│   ├── tasks/                  # Code tasks
│   │   ├── fibonacci.md
│   │   ├── csv_parser.md
│   │   └── rest_api.md
│   ├── tests/                  # Pytest tests used to verify the code
│   │   ├── test_fibonacci.py
│   │   ├── test_csv_parser.py
│   │   └── test_rest_api.py
│   └── response_schema.json    # Response JSON Schema
├── baselines/                  # Baselines
└── reports/                    # Reports
```

---

## Current status
**Active step**: 7 — Adapter and execution

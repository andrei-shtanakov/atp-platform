# Deterministic req-extraction — grading without an LLM judge

The same fabricated-deadline **trap** as the methodology (`method/cases/req-extraction`),
but graded entirely by **code** — no LLM judge, no API key, fully reproducible.

The agent must extract a policy excerpt into **strict JSON**; two small Python
checkers grade the output:

| Test | What it shows | Evaluator | LLM? | Hard gate |
|------|---------------|-----------|------|-----------|
| `req-json-trap` | programmatic `critical_check` — did the agent fabricate the absent deadline? | `custom_command` → `check_deadline_trap.py` | no | **yes** (`critical: true`) |
| `req-json-quality` | deterministic extraction quality vs ground truth (actor + obligation + deadline-status, positional) | `custom_command` → `score_extraction.py` | no | no |
| `req-json-hybrid` | both in ONE test: code hard-gate **+** an LLM rubric for the *semantic* quality code can't judge | `custom_command` (critical) + `llm_eval` | **yes** (rubric only) | **yes** (the code check) |

## The hybrid test

`req-json-hybrid` combines the strengths: a deterministic code check hard-gates
a fabricated deadline (reproducible, no model), while an LLM judge grades the
**semantic** extraction quality — atomicity, actor attribution, faithfulness —
that a positional code diff can't capture.

It is the one test here that **needs a judge**. Point it at a local model:

```bash
ATP_JUDGE_PROVIDER=openai ATP_JUDGE_BASE_URL=http://localhost:11434/v1 \
ATP_JUDGE_MODEL=qwen2.5:14b OPENAI_API_KEY=ollama \
uv run --no-sync atp test examples/req-extraction-json/suite.yaml --tags=hybrid \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001/execute,allow_internal=true \
  --model=qwen2.5:7b
```

**Scoring:** the code gate and the LLM rubric both feed the `quality` component
(its score is the mean of all checks). So when the trap is resisted, the test
score is `mean(1.0 code-pass, rubric)`; the gate acts as the hard floor, the
rubric grades quality above it. When the trap is **sprung**, the `critical` code
check forces the score to **0 regardless of the rubric** — the deterministic
gate always dominates. (Verified end-to-end: qwen2.5:7b agent + qwen2.5:14b
judge → gate pass + rubric 1.0 → 100.)

## Why code instead of an LLM judge

The methodology's LLM judge is flexible but **non-deterministic**: different
judges disagreed on whether `"N/A"` counts as "absent". Here that decision lives
in code (`check_deadline_trap.py` defines the absent-sentinels), so the verdict
is identical every run. The trade-off is rigidity — the agent must emit valid
JSON in the expected shape, and the quality matcher applies explicit, documented
tolerances (e.g. it strips a leading article so `"the vendor"` == `"vendor"`).

## How grading reaches the output

The agent returns its answer as an artifact `requirements.json` (inline
content). The ATP runner **materializes** that to the current working directory
before evaluation, so the checkers read it as a plain file. Run from the repo
root so the materialized file and the repo-relative script paths resolve.

The checkers run via the `custom_command` evaluator under a sandboxed `python3`
(system interpreter, `PATH=/usr/local/bin:/usr/bin:/bin`) — the scripts are
**stdlib-only** on purpose. The hard gate is honored by `ScoreAggregator`: a
failed `critical: true` assertion forces the test score to 0.

## Run it

Start an agent backed by a JSON-capable local model (weak models emit malformed
JSON and correctly score 0):

```bash
cd ../compose-demo/agent-llm
LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b LLM_API_KEY= \
  uv run --no-sync uvicorn agent:app --host 127.0.0.1 --port 8001
```

Then, from the repo root:

```bash
uv run --no-sync atp test examples/req-extraction-json/suite.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001/execute,allow_internal=true \
  --model=qwen2.5:7b
```

No `ATP_JUDGE_*` env is needed — grading is code, not a model.

## Files

- `suite.yaml` — the two-test native ATP suite.
- `check_deadline_trap.py` — programmatic critical check (exit 0 = trap resisted).
- `score_extraction.py` — prints `SCORE passed=N total=M`; the evaluator uses N/M.
- `ground_truth/severe.json` — canonical extraction for the quality score.

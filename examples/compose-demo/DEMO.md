# ATP — EPAM Demo Runbook (cloud + on-prem)

A ~15-minute presenter script. **One test suite, two deployments, the adapter is
the only thing that changes.** That is the whole pitch: *an agent is a black box
with a contract — ATP tests it the same way whether it runs in a container
on-prem or as a managed Bedrock Agent in AWS.*

---

## 0. One-line message

> The same `suite.yaml` validates an agent **on-prem (container, offline)** and
> **in the cloud (Bedrock, all-in-AWS via IAM)** — you swap `--adapter`, nothing
> else. Quality is judged by Claude; runs are stored, scored, and trended.

## Audience & goal

EPAM engineers evaluating ATP for agent testing. By the end they should see:
1. framework-agnostic black-box testing (adapter abstraction),
2. the *identical* suite running in two deployment archetypes,
3. evaluation depth (deterministic checks + LLM-judge), and
4. observability (dashboard history, statistics, CI reporters).

## Prerequisites

| Act | Needs |
|-----|-------|
| Act 1 — on-prem | Docker **or** Podman (rootless ok). No network, no keys. |
| Act 2 — cloud | An AWS account with a deployed **Bedrock Agent** (agent id + alias), Bedrock model access, and credentials (IAM role if the platform runs in AWS, else keys). |

> If no live AWS is available, run Act 1 live and present Act 2 from the slides /
> recorded terminal — the *diff* is the point, and it's one flag.

---

## Act 1 — On-prem, fully offline (~5 min)

Two containers on a compose network: the **platform** POSTs to the **agent** over
HTTP. No Docker socket, no privileges, no cloud.

```bash
cd examples/compose-demo
./run_demo.sh                         # or the manual steps below
```

Manual:
```bash
docker compose up --build -d agent dashboard
docker compose run --rm atp           # one-shot test run
open http://localhost:8080/ui/         # browse the run (auth disabled for demo)
```

**What to show / say:**
- The console report: `demo-001` **PASSED** — the agent returned an `output.txt`
  artifact and emitted no error events.
- `suite.yaml` — point out it says **nothing** about HTTP/containers. Only
  `task` + `assertions`. The binding is the `--adapter=http` flag.
- The dashboard `/ui/` — the run is stored: per-test status, artifacts, metrics.
- Talking point: *deterministic, isolated, air-gapped — ideal for CI and
  regulated on-prem environments.*

Cleanup later: `docker compose down -v`.

---

## Act 2 — Same suite, cloud (Bedrock, all-in-AWS) (~5 min)

**Same `suite.yaml`.** Swap the adapter to target a managed Bedrock Agent, and
let **Claude judge quality** via `llm_eval` — both using the **IAM role**, no
`ANTHROPIC_API_KEY`, no static AWS keys (when the platform runs on EC2/ECS).

```bash
atp test suite.yaml \
  --adapter=bedrock \
  --adapter-config agent_id=<AGENT_ID>,agent_alias_id=<ALIAS_ID>,region=us-east-1
```

To activate the LLM-judge check, uncomment `demo-002` in `suite.yaml` and set the
judge provider to Bedrock-hosted Claude:

```yaml
  - id: "demo-002"
    name: "Answer quality (LLM judge)"
    task:
      description: "Explain in one sentence what the ATP platform does."
    assertions:
      - type: "llm_eval"
        config:
          provider: bedrock
          aws_region: us-east-1
          model: anthropic.claude-3-5-sonnet-20241022-v2:0   # match your account
          criteria: completeness
          threshold: 0.7
```

**What to show / say:**
- The command is identical except `--adapter`. Hold the two terminals side by
  side: `--adapter=http` vs `--adapter=bedrock`. **Same suite, same report.**
- Bedrock returns a native trace → ATP events: reasoning, action groups,
  guardrails. Point out the observability you get for free.
- *All-in-AWS via IAM*: the agent (bedrock adapter) and the judge
  (`llm_eval provider=bedrock`) both use the instance/task role — no secrets on
  the host.

---

## Act 2b — Methodology: the curve of collapse (~5 min)

The strongest part of the pitch. Instead of a toy suite, run a real
**agent-eval-case sweep** (`method/cases/req-extraction`) — a family of cases at
rising difficulty (`clean → moderate → severe`) built around one **trap**:
fabricating a deadline absent from the source. The methodology runs through the
platform via the `atp-method` plugin; the **same** adapter swap reaches Bedrock
in the cloud variant.

```bash
cd examples/compose-demo
export LLM_BASE_URL=http://host.docker.internal:11434/v1   # e.g. Ollama / vLLM
export LLM_MODEL=llama3.1 LLM_API_KEY=local
export ANTHROPIC_API_KEY=sk-...                            # the grader (judge)
docker compose --profile method up --build -d agent-llm dashboard
docker compose --profile method run --rm atp-method
open http://localhost:8080/ui/
```

**What to show / say:**
- The console report across the three levels: the agent **passes `clean`** and
  **collapses at `severe`** — `critical_check` catches the fabricated deadline,
  the **hard gate** forces score 0 regardless of the rubric.
- That collapse point *is the signal*: it locates where the agent breaks, not a
  single pass/fail at an arbitrary difficulty. `--runs=5` makes it statistical.
- Point at the governance/sweep tags (`level_severe`, `capability_calibration`,
  `suite_probe`) — the same metadata drives filtering and trend analysis.
- Cloud variant: identical cases, `--adapter=bedrock` + `llm_eval provider=bedrock`
  — methodology graded by Bedrock-Claude, all via IAM.

> **Judge:** this run grades with the Anthropic API (`ANTHROPIC_API_KEY`). A fully
> air-gapped grader (judge on a local model) needs `base_url` support in the LLM
> judge — a small follow-up. The *agent* is already local-capable via `LLM_BASE_URL`.

---

## Act 3 — Depth (~3 min, pick what lands)

Run these through the compose network (the `agent` service is only `expose`d,
not published to the host — so reach it as `agent:8000` via `compose run`, not
`localhost`):

**Statistics over repeated runs** — agents are non-deterministic, one run lies:
```bash
docker compose run --rm atp uv run atp test /work/suite.yaml \
  --adapter=http \
  --adapter-config endpoint=http://agent:8000/execute,allow_internal=true \
  --runs=20
```
Show mean / 95% CI / coefficient of variation / **stability** (stable→critical)
and `success_rate` per test.

**CI-ready reporters** (`--output` picks the format, `--output-file` the path):
```bash
docker compose run --rm atp uv run atp test /work/suite.yaml \
  --adapter=http \
  --adapter-config endpoint=http://agent:8000/execute,allow_internal=true \
  --output=junit --output-file=/work/results/results.xml
```
JUnit/JSON for pipelines; console for humans.

**History & trends:** in `/ui/` show multiple runs accumulating; mention
`atp baseline compare` (Welch's t-test regression detection) and `atp trend`.

**Tie to what's already live:** the VPS instance already runs **game-theoretic
tournaments** (agents compared via game theory). Same platform, different
evaluation mode — this demo adds task-suite testing alongside it.

---

## Wrap (~1 min)

- **Deployment**: the platform is a container — runs on the **VPS today**, or
  **EC2 / ECS / App Runner** in AWS (Postgres via RDS). In AWS the Bedrock paths
  use the IAM role automatically.
- **Extensibility**: 11 adapters (HTTP, CLI, Container, MCP, SDK pull-model,
  LangGraph, CrewAI, AutoGen, Bedrock, Vertex, Azure OpenAI) and pluggable
  evaluators (artifact, behavior, llm-judge, code-exec, security, …).
- **The takeaway**: write the suite once against the contract; run it anywhere.

---

## Appendix — timing, fallbacks, gotchas

| Item | Note |
|------|------|
| Total | ~15 min (5 / 5 / 3 / 1 + Q&A) |
| SSRF guard | the `http` adapter blocks private IPs; the runner passes `allow_internal=true` (compose service names resolve to private addresses). |
| No AWS live | run Act 1 live; narrate Act 2 from a recorded terminal — the diff is one flag. |
| Dashboard empty | `atp` and `dashboard` must share the `atp-data` volume (compose already does). |
| Reset | `docker compose down -v` removes the run DB between rehearsals. |
| Auth | `ATP_DISABLE_AUTH=true` + demo secret are **local-demo only**. |

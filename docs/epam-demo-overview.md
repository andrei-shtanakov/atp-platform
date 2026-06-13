# ATP — Agent Testing Platform: EPAM Overview

A one-page orientation for evaluating ATP, with pointers to everything you can run.

## The pitch, in one line

> An agent is a **black box with a contract**. ATP tests it the same way whether
> it runs as a **container on-prem** or as a **managed Bedrock Agent in AWS** —
> you swap one flag (`--adapter`); the test suite, the grading, and the report
> stay identical.

## Why it matters

Most test suites don't separate a strong agent from a weak one: on easy,
well-specified tasks almost every competent agent passes. ATP pairs a
framework-agnostic runtime with an evaluation methodology built to **discriminate**
— and runs the *same* cases across deployment archetypes, so on-prem and cloud
results are directly comparable.

## What you'll see — two archetypes, one suite

| | **On-prem (variant 2)** | **Cloud (variant 1)** |
|---|---|---|
| Agent under test | container LLM agent over HTTP | managed **Bedrock Agent** |
| Adapter | `--adapter=http` | `--adapter=bedrock` |
| Grader (judge) | local model (Ollama/vLLM) — air-gapped | **Bedrock-Claude** via IAM role |
| Keys / secrets | none (fully offline) | none (IAM role) |
| Test content | `method/cases/req-extraction` (the same sweep) | the same sweep |
| Run it | [`examples/compose-demo/`](../examples/compose-demo/) — **validated live** | [`infra/`](../infra/) — Terraform (ECR + IAM/EC2 + Bedrock Agent) |

The only differences between the two columns are the adapter and a couple of env
vars. That is the whole point.

## The differentiator — methodology that finds where an agent breaks

The cases come from a methodology (`method/`, run through the platform by the
`atp-method` plugin). Two ideas make them sharp:

- **Trap** — every case carries a specific anticipated failure mode plus a
  **binary `critical_check`**. A failed critical check **hard-gates** the test:
  score 0, regardless of the graded rubric. (Example trap: fabricating a deadline
  that is absent from the source.)
- **Sweep** — a family of cases at rising difficulty (`clean → moderate →
  severe`). You don't get one pass/fail at an arbitrary difficulty — you get the
  **point of collapse**, the level where the agent starts failing. `--runs=N`
  makes it statistical.

In the live on-prem run the local judge caught the trap verbatim — its `severe`
verdict cited *"a fabricated concrete duration"* and hard-gated the score to 0.

## Evaluation depth

- **Hard gate** (`critical_check`) + **weighted rubric** — binary guard plus
  graded quality, scored separately.
- **Statistics over repeats** (`--runs`) — mean / 95% CI / coefficient of
  variation / stability, because one run of a non-deterministic agent lies.
- **13 evaluators** — artifact, behavior, LLM-judge, code-exec, security,
  factuality, filesystem, style, performance, composite, git-commit, guardrails,
  container.
- **Reporters** — console, JSON, JUnit (CI), HTML; plus baseline regression
  detection (Welch's t-test) and cross-run trend analysis.

## Observability

Every run is stored and browsable in the **dashboard** (`/ui/`): per-test status,
artifacts, metrics, history. OpenTelemetry spans (suite / test / run / adapter)
export to any OTLP backend; Prometheus metrics are exposed.

## Deployment

- The platform is a container — runs on a **VPS today**, or **EC2 / ECS /
  App Runner** in AWS. In AWS the Bedrock paths use the **IAM role** automatically
  (no static keys).
- Database: SQLite by default; **RDS PostgreSQL** for production
  (`ATP_DATABASE_URL`, image built with the `postgres` extra).

## Extensibility

- **11 adapters** — HTTP, CLI, Container, MCP, SDK (pull-model), LangGraph,
  CrewAI, AutoGen, Bedrock, Vertex, Azure OpenAI.
- **Pluggable evaluators** and a **plugin system** (`atp.plugins` entry points).
  The methodology itself ships as a plugin (`atp-method`) that registers a new
  source format and evaluator — new domains plug in without touching core.

## Where to start

| You want to… | Go to |
|--------------|-------|
| Run the on-prem demo locally | [`examples/compose-demo/`](../examples/compose-demo/) + [`DEMO.md`](../examples/compose-demo/DEMO.md) |
| Stand up the AWS cloud variant | [`infra/`](../infra/) — `terraform apply` + `../scripts/` |
| Understand the methodology | [`method/METHODOLOGY.md`](../method/METHODOLOGY.md) |
| See the architecture | [`docs/03-architecture.md`](03-architecture.md) |

> Status: the on-prem variant has been validated end-to-end (podman + a local
> model, fully air-gapped). The cloud variant ships as a runbook + IAM/EC2
> scaffold; a live run needs an AWS account with Bedrock model access and a
> deployed Bedrock Agent.

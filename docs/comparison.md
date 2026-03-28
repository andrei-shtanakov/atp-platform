# ATP vs Other Agent Evaluation Tools

An honest comparison of ATP against the most commonly used alternatives.

> Note: competitor feature data is based on publicly available documentation and may not reflect their latest releases. ATP features are drawn directly from this codebase.

## Feature Matrix

| Feature | ATP | DeepEval | Promptfoo | Inspect AI |
|---|---|---|---|---|
| **Framework-agnostic protocol** | Yes — any agent via unified ATP Protocol | Primarily LLM/RAG pipelines | Primarily LLM prompt testing | Primarily LLM tasks |
| **Agent adapters** | 10 types (HTTP, CLI, Container, LangGraph, CrewAI, AutoGen, MCP, Bedrock, Vertex, Azure OpenAI) | Python SDK integration | Provider configs (OpenAI, Anthropic, etc.) | Python task definitions |
| **Game-theoretic evaluation** | Yes — 7 games, tournaments, Elo ratings, Nash analysis | No | No | No |
| **Statistical analysis** | Yes — 95% CI, Welch's t-test, regression detection, Elo | Basic pass/fail aggregation | Score aggregation | Basic metrics |
| **Evaluator types** | 10 (artifact, behavior, LLM-judge, code-exec, security, factuality, style, filesystem, performance, composite) | LLM-judge, G-Eval, RAG metrics | LLM-judge, custom assertions | LLM-judge, custom |
| **Web dashboard** | Yes — FastAPI, SQLite/PostgreSQL, historical trends, leaderboard | Yes (cloud service) | Yes (cloud service) | No |
| **CI/CD integration** | Yes — JUnit XML, GitHub Actions, GitLab, Azure, CircleCI, Jenkins | Yes | Yes | Limited |
| **Cost tracking** | Yes — per-run token and USD cost tracking | Yes (cloud) | Partial | No |
| **Baseline / regression** | Yes — save baselines, compare with Welch's t-test | No | No | No |
| **Multi-agent evaluation** | Yes — game runner, cross-play matrix, tournaments | No | No | Partial |
| **YAML test definitions** | Yes — declarative, versioned | No (Python API) | Yes | No (Python) |
| **Self-hosted** | Yes — fully open source | Partial (OSS core + cloud) | Yes | Yes |
| **License** | MIT | Apache 2.0 | MIT | MIT |

## When to Use ATP

ATP is the right choice when:

- You need to test agents that are **not just LLM wrappers** — multi-step agents, tool-using agents, agents built on any framework or deployed as containers or HTTP services.
- You want **statistical confidence** in your results — not just "did it pass?", but "is this change a real improvement, given natural variance?"
- You're evaluating **strategic reasoning or multi-agent interaction** — game-theoretic benchmarks measure cooperation, equilibrium play, and robustness in ways that standard prompt evaluation cannot.
- You want **everything self-hosted** — no data leaves your infrastructure.
- Your team uses multiple agent frameworks and you want a single test harness that works for all of them.

## When to Use Alternatives

- **DeepEval** — strong choice if you're primarily evaluating RAG pipelines or LLM output quality (hallucination, faithfulness, answer relevance) and are comfortable using a managed cloud service.
- **Promptfoo** — good for rapid prompt iteration and A/B testing across providers, especially when your "agents" are mostly prompt chains without external tool use.
- **Inspect AI** — well-suited for research-oriented LLM capability evaluations (coding, reasoning, safety) where you want tight control via Python and don't need a test runner for deployed services.

## Summary

ATP fills a gap the other tools don't address: testing **deployed agents as black boxes**, with statistical rigor, multi-agent game evaluation, and support for any implementation framework. If you're testing a service — not just prompting a model — ATP is built for that use case.

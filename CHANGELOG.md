# Changelog

All notable changes to the ATP Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-13

Initial public release of ATP (Agent Test Platform) — a framework-agnostic
platform for testing and evaluating AI agents.

### Phase 1 — Core Protocol & Runner

- ATP Protocol models (ATPRequest, ATPResponse, ATPEvent)
- YAML/JSON test suite loader with filtering and tagging
- Test runner with sandbox isolation and parallel execution
- Adapter system: HTTP, Container, CLI, LangGraph, CrewAI, AutoGen
- Evaluators: artifact matching, behavioral checks, LLM-judge
- Score aggregation with weighted scoring
- Reporters: console, JSON, HTML, JUnit

### Phase 2 — Streaming, Security & Baselines

- Event streaming with buffering and back-pressure
- Security hardening: secret redaction, input sanitization
- Statistical analysis: confidence intervals, stability metrics
- Baseline management with regression detection (Welch's t-test)
- Performance profiling, caching, and memory tracking

### Phase 3 — Dashboard & Analytics

- Web dashboard (FastAPI + SQLAlchemy) with real-time updates
- Cost tracking and analytics
- Mock tool server for deterministic testing
- Structured logging with correlation IDs
- OpenTelemetry tracing integration

### Phase 4 — Extensibility & Adapters

- Plugin ecosystem: discovery via entry points, config schemas, validation
- MCP adapter for Model Context Protocol agents
- Cloud adapters: AWS Bedrock, Google Vertex AI, Azure OpenAI
- Chaos testing framework
- Test suite generation from natural language
- CLI commands: init, generate, benchmark, budget, experiment, plugins

### Phase 5 — Game-Theoretic Evaluation

- `game-environments` sub-package with classic game theory environments
- `atp-games` integration: game runner, strategy adapters, Nash analysis
- Game evaluators, reporters, and CLI (`atp game`)
- Iterated games, tournaments, and multi-agent scenarios
- Terminal UI (TUI) for interactive test monitoring

### Phase 6 — CI/CD, Traces & SDK

- GitHub Actions composite action for CI integration
- Composite evaluators (all-of, any-of, weighted, conditional, pipeline)
- Agent replay from execution traces
- Streaming CLI live display
- Python SDK for programmatic test execution
- Multi-model comparison and cost estimation
- Benchmark suite loaders and GitHub import
- Natural language test generation
- Trace import and storage tracking

[1.0.0]: https://github.com/anthropics/atp-platform/releases/tag/v1.0.0

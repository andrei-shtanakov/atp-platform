# Phase 4: Growth — Tasks Specification

> Implementation Tasks for ATP Platform Enhancement (Q4 2025 - Q2 2026)

## Legend

**Priority:**
| Emoji | Code | Description |
|-------|------|-------------|
| 🔴 | P0 | Critical — blocks release |
| 🟠 | P1 | High — needed for full functionality |
| 🟡 | P2 | Medium — improves experience |
| 🟢 | P3 | Low — nice to have |

**Status:**
| Emoji | Status | Description |
|-------|--------|-------------|
| ⬜ | TODO | Not started |
| 🔄 | IN PROGRESS | In work |
| ✅ | DONE | Completed |
| ⏸️ | BLOCKED | Waiting on dependency |

---

## Milestone 1: Foundation & Refactoring

### TASK-101: Dashboard Module Structure
🔴 P0 | ✅ DONE | Est: 3-4h

**Description:**
Create the new modular dashboard package structure without breaking existing functionality.

**Checklist:**
- [x] Create `atp/dashboard/v2/` directory structure
- [x] Create `atp/dashboard/v2/__init__.py`
- [x] Create `atp/dashboard/v2/factory.py` with app factory pattern
- [x] Create `atp/dashboard/v2/config.py` for dashboard configuration
- [x] Create `atp/dashboard/v2/dependencies.py` for FastAPI dependency injection
- [x] Implement feature flag `ATP_DASHBOARD_V2` for switching versions
- [x] Update `atp/dashboard/__init__.py` to route based on feature flag
- [x] Write unit tests for app factory

**Traces to:** [REQ-101]
**Depends on:** -
**Blocks:** [TASK-102], [TASK-103], [TASK-104]

---

### TASK-102: Dashboard Routes Extraction
🔴 P0 | ✅ DONE | Est: 4-5h

**Description:**
Extract route handlers from monolithic `app.py` into separate route modules.

**Checklist:**
- [x] Create `atp/dashboard/v2/routes/__init__.py`
- [x] Extract home page routes to `routes/home.py`
- [x] Extract test results routes to `routes/tests.py`
- [x] Extract agent management routes to `routes/agents.py`
- [x] Extract comparison routes to `routes/comparison.py`
- [x] Extract suite management routes to `routes/suites.py`
- [x] Create router registration in factory.py
- [x] Ensure all routes work with both v1 and v2
- [x] Write integration tests for each route module

**Traces to:** [REQ-101]
**Depends on:** [TASK-101]
**Blocks:** [TASK-105]

---

### TASK-103: Dashboard Services Layer
🔴 P0 | ✅ DONE | Est: 3-4h

**Description:**
Create service layer to separate business logic from route handlers.

**Checklist:**
- [x] Create `atp/dashboard/v2/services/__init__.py`
- [x] Create `TestService` class with test result operations
- [x] Create `AgentService` class with agent management operations
- [x] Create `ComparisonService` class with comparison operations
- [x] Create `ExportService` class with export operations
- [x] Inject services via FastAPI dependency injection
- [x] Write unit tests for each service

**Traces to:** [REQ-101]
**Depends on:** [TASK-101]
**Blocks:** [TASK-105]

---

### TASK-104: Dashboard Templates Extraction
🔴 P0 | ⬜ TODO | Est: 3-4h

**Description:**
Extract inline HTML/JS templates to Jinja2 template files.

**Checklist:**
- [ ] Create `atp/dashboard/v2/templates/` directory
- [ ] Create `base.html` with common layout, CSS, JS includes
- [ ] Extract home page template to `home.html`
- [ ] Extract test results template to `test_results.html`
- [ ] Extract comparison template to `comparison.html`
- [ ] Create `components/` subdirectory for reusable components
- [ ] Extract charts component to `components/charts.html`
- [ ] Extract tables component to `components/tables.html`
- [ ] Create `static/css/` and `static/js/` directories
- [ ] Configure Jinja2 template loading in factory.py
- [ ] Write snapshot tests for templates

**Traces to:** [REQ-101]
**Depends on:** [TASK-101]
**Blocks:** [TASK-105]

---

### TASK-105: Dashboard V2 Integration & Cleanup
🔴 P0 | ⬜ TODO | Est: 2-3h

**Description:**
Integrate all dashboard v2 components and verify full functionality.

**Checklist:**
- [ ] Wire up all routes, services, and templates in factory.py
- [ ] Verify all existing functionality works with v2
- [ ] Run full dashboard test suite with `ATP_DASHBOARD_V2=true`
- [ ] Update documentation for new structure
- [ ] Performance comparison: v1 vs v2
- [ ] Create migration guide for custom extensions
- [ ] Mark v1 as deprecated (do not remove yet)

**Traces to:** [REQ-101]
**Depends on:** [TASK-102], [TASK-103], [TASK-104]
**Blocks:** [TASK-801], [TASK-802]

---

### TASK-106: Configuration Management Enhancement
🔴 P0 | ⬜ TODO | Est: 3-4h

**Description:**
Implement Pydantic Settings for hierarchical configuration management.

**Checklist:**
- [ ] Add `pydantic-settings` to dependencies
- [ ] Create `atp/core/settings.py` with `ATPSettings` class
- [ ] Define all configuration fields with types and defaults
- [ ] Implement environment variable support with `ATP_` prefix
- [ ] Implement `.env` file loading via python-dotenv
- [ ] Implement config file loading (`atp.config.yaml`)
- [ ] Implement hierarchy: defaults → file → env → CLI
- [ ] Add `SecretStr` for sensitive values (API keys)
- [ ] Generate JSON Schema for IDE autocompletion
- [ ] Write unit tests for configuration loading
- [ ] Update documentation with all config options

**Traces to:** [REQ-102]
**Depends on:** -
**Blocks:** [TASK-107], [TASK-501]

---

### TASK-107: Structured Logging Implementation
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Implement structured logging with correlation IDs and configurable levels.

**Checklist:**
- [ ] Add `structlog` to dependencies
- [ ] Create `atp/core/logging.py` with logger configuration
- [ ] Implement correlation ID generation and propagation
- [ ] Configure JSON output for production, pretty output for dev
- [ ] Add context processors for common fields (version, hostname)
- [ ] Implement log level configuration per module
- [ ] Add sensitive data redaction filter
- [ ] Integrate with existing logging calls
- [ ] Write tests for logging configuration

**Traces to:** [REQ-103]
**Depends on:** [TASK-106]
**Blocks:** [TASK-108]

---

### TASK-108: OpenTelemetry Integration
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Add OpenTelemetry tracing for test execution flow.

**Checklist:**
- [ ] Add `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-instrumentation-*` to dependencies
- [ ] Create `atp/core/telemetry.py` with tracer configuration
- [ ] Instrument test runner with spans
- [ ] Instrument adapters with spans
- [ ] Instrument evaluators with spans
- [ ] Add span attributes for test IDs, agent names, scores
- [ ] Configure OTLP exporter for trace backends
- [ ] Add `/traces` debug endpoint (dev mode only)
- [ ] Write integration tests

**Traces to:** [REQ-103]
**Depends on:** [TASK-107]
**Blocks:** -

---

### TASK-109: Prometheus Metrics Endpoint
🟠 P1 | ⬜ TODO | Est: 2-3h

**Description:**
Add Prometheus metrics endpoint for monitoring.

**Checklist:**
- [ ] Add `prometheus-client` to dependencies
- [ ] Create `atp/core/metrics.py` with metric definitions
- [ ] Define counters: `atp_tests_total`, `atp_llm_calls_total`, `atp_adapter_errors_total`
- [ ] Define histograms: `atp_test_duration_seconds`, `atp_evaluator_duration_seconds`
- [ ] Define gauges: `atp_active_tests`, `atp_pending_tests`
- [ ] Add `/metrics` endpoint to CLI server and dashboard
- [ ] Instrument test runner, adapters, evaluators
- [ ] Create Grafana dashboard JSON
- [ ] Write tests for metrics

**Traces to:** [REQ-103]
**Depends on:** [TASK-107]
**Blocks:** -

---

## Milestone 2: Plugin Ecosystem

### TASK-201: Plugin Discovery via Entry Points
🔴 P0 | ⬜ TODO | Est: 4-5h

**Description:**
Implement plugin discovery mechanism using Python entry points.

**Checklist:**
- [ ] Create `atp/plugins/__init__.py`
- [ ] Create `atp/plugins/discovery.py` with `PluginManager` class
- [ ] Implement `discover_plugins(group)` method using `importlib.metadata`
- [ ] Define entry point groups in `pyproject.toml`: `atp.adapters`, `atp.evaluators`, `atp.reporters`
- [ ] Implement lazy loading (don't import until needed)
- [ ] Add plugin metadata model: name, version, author, description
- [ ] Implement plugin caching
- [ ] Register built-in plugins as entry points
- [ ] Write unit tests for discovery

**Traces to:** [REQ-201]
**Depends on:** -
**Blocks:** [TASK-202], [TASK-203], [TASK-204]

---

### TASK-202: Plugin Validation & Interface
🔴 P0 | ⬜ TODO | Est: 3-4h

**Description:**
Implement plugin validation to ensure interface compliance.

**Checklist:**
- [ ] Create `atp/plugins/interfaces.py` with protocol definitions
- [ ] Define `AdapterPlugin` protocol with required methods
- [ ] Define `EvaluatorPlugin` protocol with required methods
- [ ] Define `ReporterPlugin` protocol with required methods
- [ ] Implement `_validate_plugin()` in PluginManager
- [ ] Add version compatibility checking
- [ ] Create clear error messages for validation failures
- [ ] Write tests for validation

**Traces to:** [REQ-201]
**Depends on:** [TASK-201]
**Blocks:** [TASK-205]

---

### TASK-203: Plugin Configuration Schema
🟠 P1 | ⬜ TODO | Est: 2-3h

**Description:**
Allow plugins to define configuration schemas with validation.

**Checklist:**
- [ ] Create `atp/plugins/config.py`
- [ ] Define `PluginConfig` base class extending Pydantic BaseModel
- [ ] Add `config_schema` attribute to plugin interface
- [ ] Implement config validation on plugin load
- [ ] Generate JSON Schema from config models
- [ ] Add config examples to plugin metadata
- [ ] Write tests for config validation

**Traces to:** [REQ-202]
**Depends on:** [TASK-202]
**Blocks:** -

---

### TASK-204: Plugin CLI Commands
🟠 P1 | ⬜ TODO | Est: 2-3h

**Description:**
Add CLI commands for plugin management.

**Checklist:**
- [ ] Create `atp/cli/commands/plugins.py`
- [ ] Implement `atp plugins list` command
- [ ] Implement `atp plugins info <name>` command
- [ ] Implement `atp plugins enable <name>` command
- [ ] Implement `atp plugins disable <name>` command
- [ ] Add `--type` filter (adapter, evaluator, reporter)
- [ ] Format output as table with Rich
- [ ] Register commands in main CLI
- [ ] Write CLI tests

**Traces to:** [REQ-201]
**Depends on:** [TASK-201]
**Blocks:** -

---

### TASK-205: Plugin Development Guide
🟠 P1 | ⬜ TODO | Est: 2-3h

**Description:**
Create comprehensive documentation for plugin developers.

**Checklist:**
- [ ] Create `docs/guides/plugin-development.md`
- [ ] Document plugin architecture and discovery
- [ ] Document required interfaces for each plugin type
- [ ] Document configuration schema definition
- [ ] Provide adapter plugin example with full code
- [ ] Provide evaluator plugin example with full code
- [ ] Provide reporter plugin example with full code
- [ ] Document testing plugins
- [ ] Document publishing plugins to PyPI

**Traces to:** [REQ-201]
**Depends on:** [TASK-202], [TASK-203]
**Blocks:** -

---

### TASK-206: Benchmark Suite Registry
🟠 P1 | ⬜ TODO | Est: 4-5h

**Description:**
Create built-in benchmark test suites for common agent evaluation.

**Checklist:**
- [ ] Create `atp/benchmarks/__init__.py`
- [ ] Create `atp/benchmarks/registry.py` with `BenchmarkRegistry`
- [ ] Create benchmark category: `coding` with 20 tests
- [ ] Create benchmark category: `research` with 10 tests
- [ ] Create benchmark category: `reasoning` with 15 tests
- [ ] Create benchmark category: `data_processing` with 10 tests
- [ ] Store benchmarks in `atp/benchmarks/suites/`
- [ ] Implement scoring normalization (0-100)
- [ ] Define baseline scores for comparison
- [ ] Write documentation for each benchmark

**Traces to:** [REQ-203]
**Depends on:** -
**Blocks:** [TASK-207]

---

### TASK-207: Benchmark CLI Commands
🟠 P1 | ⬜ TODO | Est: 2-3h

**Description:**
Add CLI commands for running benchmarks.

**Checklist:**
- [ ] Create `atp/cli/commands/benchmark.py`
- [ ] Implement `atp benchmark list` command
- [ ] Implement `atp benchmark run <category>` command
- [ ] Implement `atp benchmark run --all` command
- [ ] Add `--agent` option to specify agent
- [ ] Add `--output` option for results format
- [ ] Generate comparison report with baseline
- [ ] Register commands in main CLI
- [ ] Write CLI tests

**Traces to:** [REQ-203]
**Depends on:** [TASK-206]
**Blocks:** -

---

## Milestone 3: Advanced Evaluators

### TASK-301: Security Evaluator - Core
🔴 P0 | ⬜ TODO | Est: 4-5h

**Description:**
Implement the core security evaluator with PII detection.

**Checklist:**
- [ ] Create `atp/evaluators/security/__init__.py`
- [ ] Create `atp/evaluators/security/base.py` with `SecurityChecker` ABC
- [ ] Create `atp/evaluators/security/pii.py` with `PIIChecker`
- [ ] Implement regex patterns for: email, phone, SSN, credit card
- [ ] Implement API key detection patterns
- [ ] Add severity levels: info, low, medium, high, critical
- [ ] Add evidence masking (don't expose full PII in reports)
- [ ] Create `SecurityEvaluator` class aggregating checkers
- [ ] Register evaluator in registry
- [ ] Write unit tests with 90%+ coverage

**Traces to:** [REQ-301]
**Depends on:** -
**Blocks:** [TASK-302], [TASK-303]

---

### TASK-302: Security Evaluator - Prompt Injection
🔴 P0 | ⬜ TODO | Est: 3-4h

**Description:**
Add prompt injection detection to security evaluator.

**Checklist:**
- [ ] Create `atp/evaluators/security/injection.py`
- [ ] Implement `PromptInjectionChecker` class
- [ ] Add patterns for common injection attempts
- [ ] Add patterns for jailbreak attempts
- [ ] Add patterns for role manipulation
- [ ] Integrate with SecurityEvaluator
- [ ] Write comprehensive tests with injection examples

**Traces to:** [REQ-301]
**Depends on:** [TASK-301]
**Blocks:** [TASK-304]

---

### TASK-303: Security Evaluator - Code Safety
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Add code safety analysis to security evaluator.

**Checklist:**
- [ ] Create `atp/evaluators/security/code.py`
- [ ] Implement `CodeSafetyChecker` class
- [ ] Detect dangerous imports: os, subprocess, socket, etc.
- [ ] Detect dangerous functions: eval, exec, compile
- [ ] Detect file system operations
- [ ] Detect network operations
- [ ] Support multiple languages: Python, JavaScript, Bash
- [ ] Integrate with SecurityEvaluator
- [ ] Write tests with unsafe code examples

**Traces to:** [REQ-301]
**Depends on:** [TASK-301]
**Blocks:** [TASK-304]

---

### TASK-304: Security Evaluator - Integration
🟠 P1 | ⬜ TODO | Est: 2-3h

**Description:**
Complete security evaluator integration and documentation.

**Checklist:**
- [ ] Add configuration support: sensitivity levels, enabled checks
- [ ] Add `SecretLeakChecker` for common secrets patterns
- [ ] Generate security reports with remediation suggestions
- [ ] Add security evaluator to example test suites
- [ ] Write integration tests
- [ ] Create documentation for security evaluator
- [ ] Add to evaluator comparison matrix in docs

**Traces to:** [REQ-301]
**Depends on:** [TASK-302], [TASK-303]
**Blocks:** -

---

### TASK-305: Factuality Evaluator
🟠 P1 | ⬜ TODO | Est: 5-6h

**Description:**
Implement evaluator for verifying factual accuracy of agent outputs.

**Checklist:**
- [ ] Create `atp/evaluators/factuality.py`
- [ ] Implement claim extraction from text
- [ ] Implement ground truth verification from JSON file
- [ ] Implement LLM-based fact verification
- [ ] Add citation extraction and validation
- [ ] Add confidence scoring for each claim
- [ ] Implement hallucination detection heuristics
- [ ] Register in evaluator registry
- [ ] Write comprehensive tests
- [ ] Create documentation

**Traces to:** [REQ-302]
**Depends on:** -
**Blocks:** -

---

### TASK-306: Style & Tone Evaluator
🟡 P2 | ⬜ TODO | Est: 3-4h

**Description:**
Implement evaluator for writing style and tone analysis.

**Checklist:**
- [ ] Create `atp/evaluators/style.py`
- [ ] Implement tone analysis: professional, casual, formal, friendly
- [ ] Implement readability metrics: Flesch-Kincaid, SMOG
- [ ] Implement passive voice percentage calculation
- [ ] Implement sentence length analysis
- [ ] Add configurable style rules
- [ ] Register in evaluator registry
- [ ] Write tests
- [ ] Create documentation

**Traces to:** [REQ-303]
**Depends on:** -
**Blocks:** -

---

### TASK-307: Performance Evaluator
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Implement evaluator for agent performance metrics.

**Checklist:**
- [ ] Create `atp/evaluators/performance.py`
- [ ] Track latency percentiles: p50, p95, p99
- [ ] Track time to first token (for streaming)
- [ ] Track tokens per second throughput
- [ ] Track token efficiency ratio
- [ ] Add configurable thresholds
- [ ] Implement regression detection vs baseline
- [ ] Register in evaluator registry
- [ ] Write tests
- [ ] Create documentation

**Traces to:** [REQ-304]
**Depends on:** -
**Blocks:** -

---

## Milestone 4: New Adapters

### TASK-401: MCP Adapter - Transport Layer
🔴 P0 | ⬜ TODO | Est: 4-5h

**Description:**
Implement transport abstraction for MCP communication.

**Checklist:**
- [ ] Create `atp/adapters/mcp/__init__.py`
- [ ] Create `atp/adapters/mcp/transport.py` with `MCPTransport` ABC
- [ ] Implement `StdioTransport` for subprocess communication
- [ ] Implement `SSETransport` for HTTP SSE communication
- [ ] Add JSON-RPC message framing
- [ ] Add connection timeout handling
- [ ] Add reconnection logic
- [ ] Write unit tests for each transport

**Traces to:** [REQ-401]
**Depends on:** -
**Blocks:** [TASK-402]

---

### TASK-402: MCP Adapter - Protocol Handler
🔴 P0 | ⬜ TODO | Est: 4-5h

**Description:**
Implement MCP protocol handling and tool discovery.

**Checklist:**
- [ ] Create `atp/adapters/mcp/adapter.py` with `MCPAdapter` class
- [ ] Implement `initialize()` for MCP handshake
- [ ] Implement tool discovery via `tools/list`
- [ ] Implement resource access
- [ ] Implement tool invocation
- [ ] Map MCP events to ATP events
- [ ] Convert MCP responses to ATP responses
- [ ] Add tool filtering support
- [ ] Write integration tests

**Traces to:** [REQ-401]
**Depends on:** [TASK-401]
**Blocks:** [TASK-403]

---

### TASK-403: MCP Adapter - Integration
🟠 P1 | ⬜ TODO | Est: 2-3h

**Description:**
Complete MCP adapter integration and documentation.

**Checklist:**
- [ ] Register MCP adapter in adapter registry
- [ ] Add MCP adapter configuration in test suite YAML
- [ ] Create example test suite using MCP adapter
- [ ] Write end-to-end tests with sample MCP server
- [ ] Create documentation with configuration examples
- [ ] Add troubleshooting section

**Traces to:** [REQ-401]
**Depends on:** [TASK-402]
**Blocks:** -

---

### TASK-404: AWS Bedrock Adapter
🟠 P1 | ⬜ TODO | Est: 4-5h

**Description:**
Implement adapter for AWS Bedrock Agents.

**Checklist:**
- [ ] Create `atp/adapters/bedrock.py`
- [ ] Add `boto3` as optional dependency
- [ ] Implement Bedrock Agent invocation
- [ ] Implement session management
- [ ] Extract trace events from Bedrock
- [ ] Handle knowledge base integration
- [ ] Handle action group support
- [ ] Add AWS credential configuration
- [ ] Register in adapter registry
- [ ] Write tests (with moto mocking)
- [ ] Create documentation

**Traces to:** [REQ-402]
**Depends on:** -
**Blocks:** -

---

### TASK-405: Vertex AI Adapter
🟠 P1 | ⬜ TODO | Est: 4-5h

**Description:**
Implement adapter for Google Vertex AI Agents.

**Checklist:**
- [ ] Create `atp/adapters/vertex.py`
- [ ] Add `google-cloud-aiplatform` as optional dependency
- [ ] Implement Vertex AI Agent invocation
- [ ] Implement conversation session management
- [ ] Extract tool use information
- [ ] Add Google Cloud auth integration
- [ ] Register in adapter registry
- [ ] Write tests
- [ ] Create documentation

**Traces to:** [REQ-403]
**Depends on:** -
**Blocks:** -

---

### TASK-406: Azure OpenAI Adapter
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Implement adapter for Azure-hosted OpenAI deployments.

**Checklist:**
- [ ] Create `atp/adapters/azure_openai.py`
- [ ] Implement Azure OpenAI API support
- [ ] Add deployment configuration
- [ ] Implement Azure AD authentication
- [ ] Add region selection
- [ ] Register in adapter registry
- [ ] Write tests
- [ ] Create documentation

**Traces to:** [REQ-404]
**Depends on:** -
**Blocks:** -

---

## Milestone 5: Analytics & Cost Management

### TASK-501: Cost Tracking Data Model
🔴 P0 | ⬜ TODO | Est: 3-4h

**Description:**
Define data models and storage for cost tracking.

**Checklist:**
- [ ] Create `atp/analytics/__init__.py`
- [ ] Create `atp/analytics/models.py` with `CostRecord`, `CostBudget`
- [ ] Add database migrations for cost tables
- [ ] Create indexes for efficient querying
- [ ] Implement `CostRepository` for CRUD operations
- [ ] Add aggregation queries: by day, provider, model, agent
- [ ] Write unit tests

**Traces to:** [REQ-501]
**Depends on:** [TASK-106]
**Blocks:** [TASK-502], [TASK-503]

---

### TASK-502: Cost Tracking Service
🔴 P0 | ⬜ TODO | Est: 4-5h

**Description:**
Implement non-blocking cost tracking service.

**Checklist:**
- [ ] Create `atp/analytics/cost.py` with `CostTracker` class
- [ ] Implement async event queue for cost events
- [ ] Implement background processor for batch inserts
- [ ] Implement pricing configuration loader
- [ ] Add pricing for OpenAI, Anthropic, Google, Azure, AWS
- [ ] Add custom pricing configuration support
- [ ] Integrate with LLM evaluator
- [ ] Integrate with adapters (where applicable)
- [ ] Write unit tests

**Traces to:** [REQ-501]
**Depends on:** [TASK-501]
**Blocks:** [TASK-504]

---

### TASK-503: Cost Budgets & Alerts
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Implement cost budgets with alerting.

**Checklist:**
- [ ] Create `atp/analytics/budgets.py`
- [ ] Implement budget definition: daily, weekly, monthly
- [ ] Implement budget checking during cost tracking
- [ ] Implement alert thresholds
- [ ] Add alert channels: log, webhook, email (pluggable)
- [ ] Create CLI for budget management
- [ ] Write tests

**Traces to:** [REQ-501]
**Depends on:** [TASK-501]
**Blocks:** [TASK-504]

---

### TASK-504: Cost Dashboard Integration
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Add cost tracking UI to dashboard.

**Checklist:**
- [ ] Add `/api/v2/costs` endpoint for cost summary
- [ ] Add `/api/v2/costs/records` endpoint for detailed records
- [ ] Add `/api/v2/budgets` endpoints for budget management
- [ ] Create cost dashboard page with charts
- [ ] Show cost breakdown by provider, model, agent
- [ ] Show cost trends over time
- [ ] Show budget utilization
- [ ] Write API tests

**Traces to:** [REQ-501]
**Depends on:** [TASK-502], [TASK-503], [TASK-105]
**Blocks:** -

---

### TASK-505: Advanced Analytics Dashboard
🟠 P1 | ⬜ TODO | Est: 5-6h

**Description:**
Add advanced analytics features to dashboard.

**Checklist:**
- [ ] Implement trend analysis: score trends over time
- [ ] Implement anomaly detection for unusual results
- [ ] Implement correlation analysis: factors affecting scores
- [ ] Add export to CSV/Excel
- [ ] Add scheduled reports configuration
- [ ] Create analytics page in dashboard
- [ ] Write tests

**Traces to:** [REQ-502]
**Depends on:** [TASK-105]
**Blocks:** -

---

### TASK-506: A/B Testing Framework
🟡 P2 | ⬜ TODO | Est: 5-6h

**Description:**
Implement A/B testing framework for agent comparison.

**Checklist:**
- [ ] Create `atp/analytics/ab_testing.py`
- [ ] Define experiment model: variants, traffic split, metrics
- [ ] Implement experiment lifecycle: draft → running → concluded
- [ ] Implement traffic routing based on split
- [ ] Implement statistical significance calculation
- [ ] Implement winner determination
- [ ] Add automatic rollback on degradation
- [ ] Create CLI for experiment management
- [ ] Add dashboard UI for experiments
- [ ] Write tests

**Traces to:** [REQ-503]
**Depends on:** -
**Blocks:** -

---

## Milestone 6: Multi-Agent & Advanced Testing

### TASK-601: Multi-Agent Orchestrator
🟠 P1 | ⬜ TODO | Est: 5-6h

**Description:**
Implement orchestrator for multi-agent test execution.

**Checklist:**
- [ ] Create `atp/runner/multi_agent.py`
- [ ] Implement `MultiAgentOrchestrator` class
- [ ] Implement comparison mode: same test, multiple agents
- [ ] Implement parallel execution with asyncio.gather
- [ ] Implement result aggregation
- [ ] Implement ranking by metrics
- [ ] Create `MultiAgentResult` model
- [ ] Write unit tests

**Traces to:** [REQ-601]
**Depends on:** -
**Blocks:** [TASK-602], [TASK-603]

---

### TASK-602: Multi-Agent Collaboration Mode
🟡 P2 | ⬜ TODO | Est: 4-5h

**Description:**
Implement collaboration mode for multi-agent tests.

**Checklist:**
- [ ] Extend `MultiAgentOrchestrator` with collaboration mode
- [ ] Implement message passing between agents
- [ ] Implement shared context management
- [ ] Implement turn-based coordination
- [ ] Add collaboration metrics
- [ ] Write tests

**Traces to:** [REQ-601]
**Depends on:** [TASK-601]
**Blocks:** [TASK-604]

---

### TASK-603: Multi-Agent Handoff Mode
🟡 P2 | ⬜ TODO | Est: 3-4h

**Description:**
Implement handoff mode for sequential agent execution.

**Checklist:**
- [ ] Extend `MultiAgentOrchestrator` with handoff mode
- [ ] Implement context passing between agents
- [ ] Implement handoff triggers
- [ ] Track individual agent contributions
- [ ] Write tests

**Traces to:** [REQ-601]
**Depends on:** [TASK-601]
**Blocks:** [TASK-604]

---

### TASK-604: Multi-Agent Test Suite Format
🟠 P1 | ⬜ TODO | Est: 2-3h

**Description:**
Extend test suite YAML format for multi-agent tests.

**Checklist:**
- [ ] Extend `TestDefinition` with multi-agent fields
- [ ] Add `agents` field (list of agent names)
- [ ] Add `mode` field: comparison, collaboration, handoff
- [ ] Add validation for multi-agent configurations
- [ ] Update test loader
- [ ] Create example multi-agent test suite
- [ ] Update documentation

**Traces to:** [REQ-601]
**Depends on:** [TASK-601], [TASK-602], [TASK-603]
**Blocks:** -

---

### TASK-605: Chaos Engineering Module
🟡 P2 | ⬜ TODO | Est: 5-6h

**Description:**
Implement chaos engineering for agent resilience testing.

**Checklist:**
- [ ] Create `atp/chaos/__init__.py`
- [ ] Create `atp/chaos/injectors.py` with fault injectors
- [ ] Implement tool failure injection (configurable probability)
- [ ] Implement latency injection (min/max delay)
- [ ] Implement token limit simulation
- [ ] Implement partial response simulation
- [ ] Implement rate limit simulation
- [ ] Create chaos profiles (predefined combinations)
- [ ] Add chaos configuration to test suite YAML
- [ ] Write tests

**Traces to:** [REQ-602]
**Depends on:** -
**Blocks:** -

---

### TASK-606: Regression Test Generator
🟡 P2 | ⬜ TODO | Est: 4-5h

**Description:**
Generate test cases from recorded agent interactions.

**Checklist:**
- [ ] Create `atp/generator/regression.py`
- [ ] Implement recording mode: capture interactions
- [ ] Implement YAML generation from recordings
- [ ] Implement parameterization of recorded tests
- [ ] Implement data anonymization
- [ ] Implement deduplication
- [ ] Add CLI command: `atp generate regression`
- [ ] Write tests

**Traces to:** [REQ-603]
**Depends on:** -
**Blocks:** -

---

## Milestone 7: Enterprise Features

### TASK-701: Multi-Tenancy Schema Management
🟠 P1 | ⬜ TODO | Est: 5-6h

**Description:**
Implement schema-per-tenant database architecture.

**Checklist:**
- [ ] Create `atp/dashboard/tenancy/__init__.py`
- [ ] Create `TenantManager` class
- [ ] Implement `create_tenant()` with schema creation
- [ ] Implement `delete_tenant()` with cleanup
- [ ] Implement `TenantAwareSession` for scoped queries
- [ ] Add tenant_id to all models
- [ ] Create migration for default tenant
- [ ] Migrate existing data to default tenant
- [ ] Write tests

**Traces to:** [REQ-701]
**Depends on:** [TASK-105]
**Blocks:** [TASK-702], [TASK-703]

---

### TASK-702: Tenant Management API
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Implement REST API for tenant management.

**Checklist:**
- [ ] Add `/api/v2/tenants` endpoints
- [ ] Implement tenant CRUD operations
- [ ] Implement quota configuration
- [ ] Implement settings configuration
- [ ] Add admin-only authorization
- [ ] Write API tests
- [ ] Create documentation

**Traces to:** [REQ-701]
**Depends on:** [TASK-701]
**Blocks:** [TASK-703]

---

### TASK-703: Tenant Quotas Enforcement
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Implement quota enforcement for tenants.

**Checklist:**
- [ ] Create `atp/dashboard/tenancy/quotas.py`
- [ ] Define quota types: tests/day, parallel runs, storage, agents, budget
- [ ] Implement quota checking middleware
- [ ] Implement quota usage tracking
- [ ] Add quota exceeded responses (HTTP 429)
- [ ] Add quota usage API endpoint
- [ ] Write tests

**Traces to:** [REQ-701]
**Depends on:** [TASK-702]
**Blocks:** -

---

### TASK-704: RBAC - Roles & Permissions
🟠 P1 | ⬜ TODO | Est: 4-5h

**Description:**
Implement role-based access control.

**Checklist:**
- [ ] Create `atp/dashboard/auth/rbac.py`
- [ ] Define `Permission` enum with all permissions
- [ ] Define default roles: admin, developer, analyst, viewer
- [ ] Implement `ROLE_PERMISSIONS` mapping
- [ ] Create `Role` and `UserRole` models
- [ ] Implement `has_permission()` function
- [ ] Implement `@require_permission` decorator
- [ ] Add user-role assignment API
- [ ] Write tests

**Traces to:** [REQ-702]
**Depends on:** -
**Blocks:** [TASK-705]

---

### TASK-705: RBAC - API Integration
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Apply RBAC to all API endpoints.

**Checklist:**
- [ ] Audit all API endpoints for required permissions
- [ ] Apply `@require_permission` to each endpoint
- [ ] Add `/api/v2/roles` endpoints for role management
- [ ] Add `/api/v2/users/{id}/roles` endpoint
- [ ] Update API documentation with permissions
- [ ] Write authorization tests
- [ ] Create RBAC documentation

**Traces to:** [REQ-702]
**Depends on:** [TASK-704]
**Blocks:** -

---

### TASK-706: SSO - OIDC Integration
🟠 P1 | ⬜ TODO | Est: 4-5h

**Description:**
Implement OIDC (OpenID Connect) authentication.

**Checklist:**
- [ ] Add `authlib` to dependencies
- [ ] Create `atp/dashboard/auth/sso/__init__.py`
- [ ] Create `atp/dashboard/auth/sso/oidc.py`
- [ ] Implement OIDC authorization flow
- [ ] Implement token validation
- [ ] Implement user provisioning (JIT)
- [ ] Implement group-to-role mapping
- [ ] Add configuration for popular providers (Okta, Auth0, Azure AD)
- [ ] Write tests

**Traces to:** [REQ-703]
**Depends on:** [TASK-704]
**Blocks:** [TASK-707]

---

### TASK-707: SSO - SAML Integration
🟡 P2 | ⬜ TODO | Est: 4-5h

**Description:**
Implement SAML 2.0 authentication.

**Checklist:**
- [ ] Add `python3-saml` to dependencies
- [ ] Create `atp/dashboard/auth/sso/saml.py`
- [ ] Implement SAML SP (Service Provider) endpoints
- [ ] Implement assertion parsing
- [ ] Implement attribute mapping
- [ ] Implement session management
- [ ] Add IdP metadata configuration
- [ ] Write tests

**Traces to:** [REQ-703]
**Depends on:** [TASK-706]
**Blocks:** -

---

### TASK-708: Audit Logging
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Implement comprehensive audit logging.

**Checklist:**
- [ ] Create `atp/dashboard/audit.py`
- [ ] Create `AuditLog` model with all required fields
- [ ] Implement `audit_log()` function
- [ ] Add audit middleware for all state-changing operations
- [ ] Log: auth, data access, config changes, admin actions
- [ ] Add `/api/v2/audit` endpoint with filtering
- [ ] Implement retention policy
- [ ] Write tests

**Traces to:** [REQ-704]
**Depends on:** [TASK-701]
**Blocks:** -

---

## Milestone 8: Dashboard Enhancements

### TASK-801: WebSocket Real-Time Updates
🟠 P1 | ⬜ TODO | Est: 4-5h

**Description:**
Implement WebSocket for real-time dashboard updates.

**Checklist:**
- [ ] Add WebSocket endpoint `/ws/updates`
- [ ] Implement connection management
- [ ] Implement pub/sub for test updates
- [ ] Send real-time test progress
- [ ] Send real-time log streaming
- [ ] Add reconnection logic on client
- [ ] Implement efficient delta updates
- [ ] Write tests

**Traces to:** [REQ-801]
**Depends on:** [TASK-105]
**Blocks:** -

---

### TASK-802: Public Leaderboard
🟡 P2 | ⬜ TODO | Est: 4-5h

**Description:**
Implement public leaderboard for benchmark results.

**Checklist:**
- [ ] Create leaderboard database models
- [ ] Implement opt-in result publishing
- [ ] Implement leaderboard by benchmark category
- [ ] Implement agent profile pages
- [ ] Add verification badges
- [ ] Create public leaderboard page
- [ ] Add leaderboard API
- [ ] Write tests

**Traces to:** [REQ-802]
**Depends on:** [TASK-206], [TASK-105]
**Blocks:** -

---

### TASK-803: Test Suite Marketplace
🟡 P2 | ⬜ TODO | Est: 5-6h

**Description:**
Implement platform for sharing test suites.

**Checklist:**
- [ ] Create marketplace database models
- [ ] Implement publish/unpublish functionality
- [ ] Implement versioning
- [ ] Implement search and discovery
- [ ] Implement ratings and reviews
- [ ] Add GitHub import
- [ ] Add license specification
- [ ] Create marketplace pages
- [ ] Write tests

**Traces to:** [REQ-803]
**Depends on:** [TASK-105]
**Blocks:** -

---

## Dependency Graph

```
TASK-101 (Dashboard Structure) ⬜
    │
    ├──► TASK-102 (Routes) ⬜
    │        │
    │        └──► TASK-105 (Integration) ⬜
    │                 │
    │                 ├──► TASK-504 (Cost Dashboard)
    │                 ├──► TASK-801 (WebSocket)
    │                 └──► TASK-802, 803 (Leaderboard, Marketplace)
    │
    ├──► TASK-103 (Services) ⬜
    │        │
    │        └──► TASK-105 (Integration) ⬜
    │
    └──► TASK-104 (Templates) ⬜
             │
             └──► TASK-105 (Integration) ⬜

TASK-106 (Configuration) ⬜
    │
    ├──► TASK-107 (Logging) ⬜ ──► TASK-108 (Telemetry)
    │                          └──► TASK-109 (Metrics)
    │
    └──► TASK-501 (Cost Data Model) ⬜

TASK-201 (Plugin Discovery) ⬜
    │
    ├──► TASK-202 (Validation) ⬜ ──► TASK-203 (Config)
    │                              └──► TASK-205 (Guide)
    │
    └──► TASK-204 (CLI) ⬜

TASK-206 (Benchmarks) ⬜ ──► TASK-207 (CLI)
                        └──► TASK-802 (Leaderboard)

TASK-301 (Security Core) ⬜
    │
    ├──► TASK-302 (Injection) ⬜ ──► TASK-304 (Integration)
    │
    └──► TASK-303 (Code Safety) ⬜ ──► TASK-304 (Integration)

TASK-401 (MCP Transport) ⬜
    │
    └──► TASK-402 (Protocol) ⬜ ──► TASK-403 (Integration)

TASK-501 (Cost Model) ⬜
    │
    └──► TASK-502 (Service) ⬜ ──► TASK-504 (Dashboard)
                            └──► TASK-503 (Budgets) ──► TASK-504

TASK-601 (Multi-Agent) ⬜
    │
    ├──► TASK-602 (Collaboration) ⬜ ──► TASK-604 (Format)
    │
    └──► TASK-603 (Handoff) ⬜ ──► TASK-604 (Format)

TASK-701 (Tenancy Schema) ⬜
    │
    └──► TASK-702 (API) ⬜ ──► TASK-703 (Quotas)
                        └──► TASK-708 (Audit)

TASK-704 (RBAC Roles) ⬜
    │
    └──► TASK-705 (API) ⬜ ──► TASK-706 (OIDC) ──► TASK-707 (SAML)
```

---

## Summary

| Milestone | Tasks | Total Est. Hours |
|-----------|-------|------------------|
| M1: Foundation | TASK-101 to TASK-109 | ~28-36h |
| M2: Plugin Ecosystem | TASK-201 to TASK-207 | ~19-25h |
| M3: Advanced Evaluators | TASK-301 to TASK-307 | ~22-29h |
| M4: New Adapters | TASK-401 to TASK-406 | ~22-28h |
| M5: Analytics & Cost | TASK-501 to TASK-506 | ~24-31h |
| M6: Multi-Agent | TASK-601 to TASK-606 | ~26-33h |
| M7: Enterprise | TASK-701 to TASK-708 | ~30-38h |
| M8: Dashboard | TASK-801 to TASK-803 | ~13-16h |
| **Total** | 57 tasks | ~184-236h (~5-6 months) |

---

## Critical Path

The critical path determines the minimum time to complete Phase 4:

```
M1 (Foundation)
    │
    ├──► M2 (Plugins) ──► M4 (Adapters)
    │                         │
    ├──► M3 (Evaluators) ─────┤
    │                         │
    ├──► M5 (Analytics) ──────┼──► M6 (Multi-Agent)
    │                         │
    └──► M7 (Enterprise) ─────┴──► M8 (Dashboard)
```

**Minimum Duration**: ~24 weeks (with parallelization)

---

## Recommended Execution Order

### Phase 4.1 (Weeks 1-8): Foundation
1. **Week 1-2**: TASK-101 to TASK-105 (Dashboard refactoring)
2. **Week 2-3**: TASK-106 to TASK-109 (Configuration, Logging, Observability)
3. **Week 3-4**: TASK-201 to TASK-205 (Plugin system)
4. **Week 5-6**: TASK-301 to TASK-304 (Security evaluator)
5. **Week 6-8**: TASK-401 to TASK-403 (MCP adapter)

### Phase 4.2 (Weeks 9-16): Advanced Features
6. **Week 9-10**: TASK-501 to TASK-504 (Cost tracking)
7. **Week 10-12**: TASK-305 to TASK-307 (More evaluators)
8. **Week 12-14**: TASK-404 to TASK-406 (Cloud adapters)
9. **Week 14-16**: TASK-206 to TASK-207 (Benchmarks)

### Phase 4.3 (Weeks 17-24): Enterprise & Polish
10. **Week 17-19**: TASK-601 to TASK-604 (Multi-agent)
11. **Week 19-22**: TASK-701 to TASK-708 (Enterprise)
12. **Week 22-24**: TASK-801 to TASK-803 (Dashboard enhancements)
13. **Ongoing**: TASK-505, TASK-506, TASK-605, TASK-606 (Nice-to-have)

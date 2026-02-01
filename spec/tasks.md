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
🔴 P0 | ✅ DONE | Est: 3-4h

**Description:**
Extract inline HTML/JS templates to Jinja2 template files.

**Checklist:**
- [x] Create `atp/dashboard/v2/templates/` directory
- [x] Create `base.html` with common layout, CSS, JS includes
- [x] Extract home page template to `home.html`
- [x] Extract test results template to `test_results.html`
- [x] Extract comparison template to `comparison.html`
- [x] Create `components/` subdirectory for reusable components
- [x] Extract charts component to `components/charts.html`
- [x] Extract tables component to `components/tables.html`
- [x] Create `static/css/` and `static/js/` directories
- [x] Configure Jinja2 template loading in factory.py
- [x] Write snapshot tests for templates

**Traces to:** [REQ-101]
**Depends on:** [TASK-101]
**Blocks:** [TASK-105]

---

### TASK-105: Dashboard V2 Integration & Cleanup
🔴 P0 | ✅ DONE | Est: 2-3h

**Description:**
Integrate all dashboard v2 components and verify full functionality.

**Checklist:**
- [x] Wire up all routes, services, and templates in factory.py
- [x] Verify all existing functionality works with v2
- [x] Run full dashboard test suite with `ATP_DASHBOARD_V2=true`
- [x] Update documentation for new structure
- [x] Performance comparison: v1 vs v2
- [x] Create migration guide for custom extensions
- [x] Mark v1 as deprecated (do not remove yet)

**Traces to:** [REQ-101]
**Depends on:** [TASK-102], [TASK-103], [TASK-104]
**Blocks:** [TASK-801], [TASK-802]

---

### TASK-106: Configuration Management Enhancement
🔴 P0 | ✅ DONE | Est: 3-4h

**Description:**
Implement Pydantic Settings for hierarchical configuration management.

**Checklist:**
- [x] Add `pydantic-settings` to dependencies
- [x] Create `atp/core/settings.py` with `ATPSettings` class
- [x] Define all configuration fields with types and defaults
- [x] Implement environment variable support with `ATP_` prefix
- [x] Implement `.env` file loading via python-dotenv
- [x] Implement config file loading (`atp.config.yaml`)
- [x] Implement hierarchy: defaults → file → env → CLI
- [x] Add `SecretStr` for sensitive values (API keys)
- [x] Generate JSON Schema for IDE autocompletion
- [x] Write unit tests for configuration loading
- [x] Update documentation with all config options

**Traces to:** [REQ-102]
**Depends on:** -
**Blocks:** [TASK-107], [TASK-501]

---

### TASK-107: Structured Logging Implementation
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Implement structured logging with correlation IDs and configurable levels.

**Checklist:**
- [x] Add `structlog` to dependencies
- [x] Create `atp/core/logging.py` with logger configuration
- [x] Implement correlation ID generation and propagation
- [x] Configure JSON output for production, pretty output for dev
- [x] Add context processors for common fields (version, hostname)
- [x] Implement log level configuration per module
- [x] Add sensitive data redaction filter
- [x] Integrate with existing logging calls
- [x] Write tests for logging configuration

**Traces to:** [REQ-103]
**Depends on:** [TASK-106]
**Blocks:** [TASK-108]

---

### TASK-108: OpenTelemetry Integration
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Add OpenTelemetry tracing for test execution flow.

**Checklist:**
- [x] Add `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-instrumentation-*` to dependencies
- [x] Create `atp/core/telemetry.py` with tracer configuration
- [x] Instrument test runner with spans
- [x] Instrument adapters with spans
- [x] Instrument evaluators with spans
- [x] Add span attributes for test IDs, agent names, scores
- [x] Configure OTLP exporter for trace backends
- [x] Add `/traces` debug endpoint (dev mode only)
- [x] Write integration tests

**Traces to:** [REQ-103]
**Depends on:** [TASK-107]
**Blocks:** -

---

### TASK-109: Prometheus Metrics Endpoint
🟠 P1 | ✅ DONE | Est: 2-3h

**Description:**
Add Prometheus metrics endpoint for monitoring.

**Checklist:**
- [x] Add `prometheus-client` to dependencies
- [x] Create `atp/core/metrics.py` with metric definitions
- [x] Define counters: `atp_tests_total`, `atp_llm_calls_total`, `atp_adapter_errors_total`
- [x] Define histograms: `atp_test_duration_seconds`, `atp_evaluator_duration_seconds`
- [x] Define gauges: `atp_active_tests`, `atp_pending_tests`
- [x] Add `/metrics` endpoint to CLI server and dashboard
- [x] Instrument test runner, adapters, evaluators
- [x] Create Grafana dashboard JSON
- [x] Write tests for metrics

**Traces to:** [REQ-103]
**Depends on:** [TASK-107]
**Blocks:** -

---

## Milestone 2: Plugin Ecosystem

### TASK-201: Plugin Discovery via Entry Points
🔴 P0 | ✅ DONE | Est: 4-5h

**Description:**
Implement plugin discovery mechanism using Python entry points.

**Checklist:**
- [x] Create `atp/plugins/__init__.py`
- [x] Create `atp/plugins/discovery.py` with `PluginManager` class
- [x] Implement `discover_plugins(group)` method using `importlib.metadata`
- [x] Define entry point groups in `pyproject.toml`: `atp.adapters`, `atp.evaluators`, `atp.reporters`
- [x] Implement lazy loading (don't import until needed)
- [x] Add plugin metadata model: name, version, author, description
- [x] Implement plugin caching
- [x] Register built-in plugins as entry points
- [x] Write unit tests for discovery

**Traces to:** [REQ-201]
**Depends on:** -
**Blocks:** [TASK-202], [TASK-203], [TASK-204]

---

### TASK-202: Plugin Validation & Interface
🔴 P0 | ✅ DONE | Est: 3-4h

**Description:**
Implement plugin validation to ensure interface compliance.

**Checklist:**
- [x] Create `atp/plugins/interfaces.py` with protocol definitions
- [x] Define `AdapterPlugin` protocol with required methods
- [x] Define `EvaluatorPlugin` protocol with required methods
- [x] Define `ReporterPlugin` protocol with required methods
- [x] Implement `_validate_plugin()` in PluginManager
- [x] Add version compatibility checking
- [x] Create clear error messages for validation failures
- [x] Write tests for validation

**Traces to:** [REQ-201]
**Depends on:** [TASK-201]
**Blocks:** [TASK-205]

---

### TASK-203: Plugin Configuration Schema
🟠 P1 | ✅ DONE | Est: 2-3h

**Description:**
Allow plugins to define configuration schemas with validation.

**Checklist:**
- [x] Create `atp/plugins/config.py`
- [x] Define `PluginConfig` base class extending Pydantic BaseModel
- [x] Add `config_schema` attribute to plugin interface
- [x] Implement config validation on plugin load
- [x] Generate JSON Schema from config models
- [x] Add config examples to plugin metadata
- [x] Write tests for config validation

**Traces to:** [REQ-202]
**Depends on:** [TASK-202]
**Blocks:** -

---

### TASK-204: Plugin CLI Commands
🟠 P1 | ✅ DONE | Est: 2-3h

**Description:**
Add CLI commands for plugin management.

**Checklist:**
- [x] Create `atp/cli/commands/plugins.py`
- [x] Implement `atp plugins list` command
- [x] Implement `atp plugins info <name>` command
- [x] Implement `atp plugins enable <name>` command
- [x] Implement `atp plugins disable <name>` command
- [x] Add `--type` filter (adapter, evaluator, reporter)
- [x] Format output as table with Rich
- [x] Register commands in main CLI
- [x] Write CLI tests

**Traces to:** [REQ-201]
**Depends on:** [TASK-201]
**Blocks:** -

---

### TASK-205: Plugin Development Guide
🟠 P1 | ✅ DONE | Est: 2-3h

**Description:**
Create comprehensive documentation for plugin developers.

**Checklist:**
- [x] Create `docs/guides/plugin-development.md`
- [x] Document plugin architecture and discovery
- [x] Document required interfaces for each plugin type
- [x] Document configuration schema definition
- [x] Provide adapter plugin example with full code
- [x] Provide evaluator plugin example with full code
- [x] Provide reporter plugin example with full code
- [x] Document testing plugins
- [x] Document publishing plugins to PyPI

**Traces to:** [REQ-201]
**Depends on:** [TASK-202], [TASK-203]
**Blocks:** -

---

### TASK-206: Benchmark Suite Registry
🟠 P1 | ✅ DONE | Est: 4-5h

**Description:**
Create built-in benchmark test suites for common agent evaluation.

**Checklist:**
- [x] Create `atp/benchmarks/__init__.py`
- [x] Create `atp/benchmarks/registry.py` with `BenchmarkRegistry`
- [x] Create benchmark category: `coding` with 20 tests
- [x] Create benchmark category: `research` with 10 tests
- [x] Create benchmark category: `reasoning` with 15 tests
- [x] Create benchmark category: `data_processing` with 10 tests
- [x] Store benchmarks in `atp/benchmarks/suites/`
- [x] Implement scoring normalization (0-100)
- [x] Define baseline scores for comparison
- [x] Write documentation for each benchmark

**Traces to:** [REQ-203]
**Depends on:** -
**Blocks:** [TASK-207]

---

### TASK-207: Benchmark CLI Commands
🟠 P1 | ✅ DONE | Est: 2-3h

**Description:**
Add CLI commands for running benchmarks.

**Checklist:**
- [x] Create `atp/cli/commands/benchmark.py`
- [x] Implement `atp benchmark list` command
- [x] Implement `atp benchmark run <category>` command
- [x] Implement `atp benchmark run --all` command
- [x] Add `--agent` option to specify agent
- [x] Add `--output` option for results format
- [x] Generate comparison report with baseline
- [x] Register commands in main CLI
- [x] Write CLI tests

**Traces to:** [REQ-203]
**Depends on:** [TASK-206]
**Blocks:** -

---

## Milestone 3: Advanced Evaluators

### TASK-301: Security Evaluator - Core
🔴 P0 | ✅ DONE | Est: 4-5h

**Description:**
Implement the core security evaluator with PII detection.

**Checklist:**
- [x] Create `atp/evaluators/security/__init__.py`
- [x] Create `atp/evaluators/security/base.py` with `SecurityChecker` ABC
- [x] Create `atp/evaluators/security/pii.py` with `PIIChecker`
- [x] Implement regex patterns for: email, phone, SSN, credit card
- [x] Implement API key detection patterns
- [x] Add severity levels: info, low, medium, high, critical
- [x] Add evidence masking (don't expose full PII in reports)
- [x] Create `SecurityEvaluator` class aggregating checkers
- [x] Register evaluator in registry
- [x] Write unit tests with 90%+ coverage

**Traces to:** [REQ-301]
**Depends on:** -
**Blocks:** [TASK-302], [TASK-303]

---

### TASK-302: Security Evaluator - Prompt Injection
🔴 P0 | ✅ DONE | Est: 3-4h

**Description:**
Add prompt injection detection to security evaluator.

**Checklist:**
- [x] Create `atp/evaluators/security/injection.py`
- [x] Implement `PromptInjectionChecker` class
- [x] Add patterns for common injection attempts
- [x] Add patterns for jailbreak attempts
- [x] Add patterns for role manipulation
- [x] Integrate with SecurityEvaluator
- [x] Write comprehensive tests with injection examples

**Traces to:** [REQ-301]
**Depends on:** [TASK-301]
**Blocks:** [TASK-304]

---

### TASK-303: Security Evaluator - Code Safety
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Add code safety analysis to security evaluator.

**Checklist:**
- [x] Create `atp/evaluators/security/code.py`
- [x] Implement `CodeSafetyChecker` class
- [x] Detect dangerous imports: os, subprocess, socket, etc.
- [x] Detect dangerous functions: eval, exec, compile
- [x] Detect file system operations
- [x] Detect network operations
- [x] Support multiple languages: Python, JavaScript, Bash
- [x] Integrate with SecurityEvaluator
- [x] Write tests with unsafe code examples

**Traces to:** [REQ-301]
**Depends on:** [TASK-301]
**Blocks:** [TASK-304]

---

### TASK-304: Security Evaluator - Integration
🟠 P1 | ✅ DONE | Est: 2-3h

**Description:**
Complete security evaluator integration and documentation.

**Checklist:**
- [x] Add configuration support: sensitivity levels, enabled checks
- [x] Add `SecretLeakChecker` for common secrets patterns
- [x] Generate security reports with remediation suggestions
- [x] Add security evaluator to example test suites
- [x] Write integration tests
- [x] Create documentation for security evaluator
- [x] Add to evaluator comparison matrix in docs

**Traces to:** [REQ-301]
**Depends on:** [TASK-302], [TASK-303]
**Blocks:** -

---

### TASK-305: Factuality Evaluator
🟠 P1 | ✅ DONE | Est: 5-6h

**Description:**
Implement evaluator for verifying factual accuracy of agent outputs.

**Checklist:**
- [x] Create `atp/evaluators/factuality.py`
- [x] Implement claim extraction from text
- [x] Implement ground truth verification from JSON file
- [x] Implement LLM-based fact verification
- [x] Add citation extraction and validation
- [x] Add confidence scoring for each claim
- [x] Implement hallucination detection heuristics
- [x] Register in evaluator registry
- [x] Write comprehensive tests
- [x] Create documentation

**Traces to:** [REQ-302]
**Depends on:** -
**Blocks:** -

---

### TASK-306: Style & Tone Evaluator
🟡 P2 | ✅ DONE | Est: 3-4h

**Description:**
Implement evaluator for writing style and tone analysis.

**Checklist:**
- [x] Create `atp/evaluators/style.py`
- [x] Implement tone analysis: professional, casual, formal, friendly
- [x] Implement readability metrics: Flesch-Kincaid, SMOG
- [x] Implement passive voice percentage calculation
- [x] Implement sentence length analysis
- [x] Add configurable style rules
- [x] Register in evaluator registry
- [x] Write tests
- [x] Create documentation

**Traces to:** [REQ-303]
**Depends on:** -
**Blocks:** -

---

### TASK-307: Performance Evaluator
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Implement evaluator for agent performance metrics.

**Checklist:**
- [x] Create `atp/evaluators/performance.py`
- [x] Track latency percentiles: p50, p95, p99
- [x] Track time to first token (for streaming)
- [x] Track tokens per second throughput
- [x] Track token efficiency ratio
- [x] Add configurable thresholds
- [x] Implement regression detection vs baseline
- [x] Register in evaluator registry
- [x] Write tests
- [x] Create documentation

**Traces to:** [REQ-304]
**Depends on:** -
**Blocks:** -

---

## Milestone 4: New Adapters

### TASK-401: MCP Adapter - Transport Layer
🔴 P0 | ✅ DONE | Est: 4-5h

**Description:**
Implement transport abstraction for MCP communication.

**Checklist:**
- [x] Create `atp/adapters/mcp/__init__.py`
- [x] Create `atp/adapters/mcp/transport.py` with `MCPTransport` ABC
- [x] Implement `StdioTransport` for subprocess communication
- [x] Implement `SSETransport` for HTTP SSE communication
- [x] Add JSON-RPC message framing
- [x] Add connection timeout handling
- [x] Add reconnection logic
- [x] Write unit tests for each transport

**Traces to:** [REQ-401]
**Depends on:** -
**Blocks:** [TASK-402]

---

### TASK-402: MCP Adapter - Protocol Handler
🔴 P0 | ✅ DONE | Est: 4-5h

**Description:**
Implement MCP protocol handling and tool discovery.

**Checklist:**
- [x] Create `atp/adapters/mcp/adapter.py` with `MCPAdapter` class
- [x] Implement `initialize()` for MCP handshake
- [x] Implement tool discovery via `tools/list`
- [x] Implement resource access
- [x] Implement tool invocation
- [x] Map MCP events to ATP events
- [x] Convert MCP responses to ATP responses
- [x] Add tool filtering support
- [x] Write integration tests

**Traces to:** [REQ-401]
**Depends on:** [TASK-401]
**Blocks:** [TASK-403]

---

### TASK-403: MCP Adapter - Integration
🟠 P1 | ✅ DONE | Est: 2-3h

**Description:**
Complete MCP adapter integration and documentation.

**Checklist:**
- [x] Register MCP adapter in adapter registry
- [x] Add MCP adapter configuration in test suite YAML
- [x] Create example test suite using MCP adapter
- [x] Write end-to-end tests with sample MCP server
- [x] Create documentation with configuration examples
- [x] Add troubleshooting section

**Traces to:** [REQ-401]
**Depends on:** [TASK-402]
**Blocks:** -

---

### TASK-404: AWS Bedrock Adapter
🟠 P1 | ✅ DONE | Est: 4-5h

**Description:**
Implement adapter for AWS Bedrock Agents.

**Checklist:**
- [x] Create `atp/adapters/bedrock.py`
- [x] Add `boto3` as optional dependency
- [x] Implement Bedrock Agent invocation
- [x] Implement session management
- [x] Extract trace events from Bedrock
- [x] Handle knowledge base integration
- [x] Handle action group support
- [x] Add AWS credential configuration
- [x] Register in adapter registry
- [x] Write tests (with moto mocking)
- [x] Create documentation

**Traces to:** [REQ-402]
**Depends on:** -
**Blocks:** -

---

### TASK-405: Vertex AI Adapter
🟠 P1 | ✅ DONE | Est: 4-5h

**Description:**
Implement adapter for Google Vertex AI Agents.

**Checklist:**
- [x] Create `atp/adapters/vertex.py`
- [x] Add `google-cloud-aiplatform` as optional dependency
- [x] Implement Vertex AI Agent invocation
- [x] Implement conversation session management
- [x] Extract tool use information
- [x] Add Google Cloud auth integration
- [x] Register in adapter registry
- [x] Write tests
- [x] Create documentation

**Traces to:** [REQ-403]
**Depends on:** -
**Blocks:** -

---

### TASK-406: Azure OpenAI Adapter
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Implement adapter for Azure-hosted OpenAI deployments.

**Checklist:**
- [x] Create `atp/adapters/azure_openai.py`
- [x] Implement Azure OpenAI API support
- [x] Add deployment configuration
- [x] Implement Azure AD authentication
- [x] Add region selection
- [x] Register in adapter registry
- [x] Write tests
- [x] Create documentation

**Traces to:** [REQ-404]
**Depends on:** -
**Blocks:** -

---

## Milestone 5: Analytics & Cost Management

### TASK-501: Cost Tracking Data Model
🔴 P0 | ✅ DONE | Est: 3-4h

**Description:**
Define data models and storage for cost tracking.

**Checklist:**
- [x] Create `atp/analytics/__init__.py`
- [x] Create `atp/analytics/models.py` with `CostRecord`, `CostBudget`
- [x] Add database migrations for cost tables
- [x] Create indexes for efficient querying
- [x] Implement `CostRepository` for CRUD operations
- [x] Add aggregation queries: by day, provider, model, agent
- [x] Write unit tests

**Traces to:** [REQ-501]
**Depends on:** [TASK-106]
**Blocks:** [TASK-502], [TASK-503]

---

### TASK-502: Cost Tracking Service
🔴 P0 | ✅ DONE | Est: 4-5h

**Description:**
Implement non-blocking cost tracking service.

**Checklist:**
- [x] Create `atp/analytics/cost.py` with `CostTracker` class
- [x] Implement async event queue for cost events
- [x] Implement background processor for batch inserts
- [x] Implement pricing configuration loader
- [x] Add pricing for OpenAI, Anthropic, Google, Azure, AWS
- [x] Add custom pricing configuration support
- [x] Integrate with LLM evaluator
- [x] Integrate with adapters (where applicable)
- [x] Write unit tests

**Traces to:** [REQ-501]
**Depends on:** [TASK-501]
**Blocks:** [TASK-504]

---

### TASK-503: Cost Budgets & Alerts
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Implement cost budgets with alerting.

**Checklist:**
- [x] Create `atp/analytics/budgets.py`
- [x] Implement budget definition: daily, weekly, monthly
- [x] Implement budget checking during cost tracking
- [x] Implement alert thresholds
- [x] Add alert channels: log, webhook, email (pluggable)
- [x] Create CLI for budget management
- [x] Write tests

**Traces to:** [REQ-501]
**Depends on:** [TASK-501]
**Blocks:** [TASK-504]

---

### TASK-504: Cost Dashboard Integration
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Add cost tracking UI to dashboard.

**Checklist:**
- [x] Add `/api/v2/costs` endpoint for cost summary
- [x] Add `/api/v2/costs/records` endpoint for detailed records
- [x] Add `/api/v2/budgets` endpoints for budget management
- [x] Create cost dashboard page with charts
- [x] Show cost breakdown by provider, model, agent
- [x] Show cost trends over time
- [x] Show budget utilization
- [x] Write API tests

**Traces to:** [REQ-501]
**Depends on:** [TASK-502], [TASK-503], [TASK-105]
**Blocks:** -

---

### TASK-505: Advanced Analytics Dashboard
🟠 P1 | ✅ DONE | Est: 5-6h

**Description:**
Add advanced analytics features to dashboard.

**Checklist:**
- [x] Implement trend analysis: score trends over time
- [x] Implement anomaly detection for unusual results
- [x] Implement correlation analysis: factors affecting scores
- [x] Add export to CSV/Excel
- [x] Add scheduled reports configuration
- [x] Create analytics page in dashboard
- [x] Write tests

**Traces to:** [REQ-502]
**Depends on:** [TASK-105]
**Blocks:** -

---

### TASK-506: A/B Testing Framework
🟡 P2 | ✅ DONE | Est: 5-6h

**Description:**
Implement A/B testing framework for agent comparison.

**Checklist:**
- [x] Create `atp/analytics/ab_testing.py`
- [x] Define experiment model: variants, traffic split, metrics
- [x] Implement experiment lifecycle: draft → running → concluded
- [x] Implement traffic routing based on split
- [x] Implement statistical significance calculation
- [x] Implement winner determination
- [x] Add automatic rollback on degradation
- [x] Create CLI for experiment management
- [x] Add dashboard UI for experiments
- [x] Write tests

**Traces to:** [REQ-503]
**Depends on:** -
**Blocks:** -

---

## Milestone 6: Multi-Agent & Advanced Testing

### TASK-601: Multi-Agent Orchestrator
🟠 P1 | ✅ DONE | Est: 5-6h

**Description:**
Implement orchestrator for multi-agent test execution.

**Checklist:**
- [x] Create `atp/runner/multi_agent.py`
- [x] Implement `MultiAgentOrchestrator` class
- [x] Implement comparison mode: same test, multiple agents
- [x] Implement parallel execution with asyncio.gather
- [x] Implement result aggregation
- [x] Implement ranking by metrics
- [x] Create `MultiAgentResult` model
- [x] Write unit tests

**Traces to:** [REQ-601]
**Depends on:** -
**Blocks:** [TASK-602], [TASK-603]

---

### TASK-602: Multi-Agent Collaboration Mode
🟡 P2 | ✅ DONE | Est: 4-5h

**Description:**
Implement collaboration mode for multi-agent tests.

**Checklist:**
- [x] Extend `MultiAgentOrchestrator` with collaboration mode
- [x] Implement message passing between agents
- [x] Implement shared context management
- [x] Implement turn-based coordination
- [x] Add collaboration metrics
- [x] Write tests

**Traces to:** [REQ-601]
**Depends on:** [TASK-601]
**Blocks:** [TASK-604]

---

### TASK-603: Multi-Agent Handoff Mode
🟡 P2 | ✅ DONE | Est: 3-4h

**Description:**
Implement handoff mode for sequential agent execution.

**Checklist:**
- [x] Extend `MultiAgentOrchestrator` with handoff mode
- [x] Implement context passing between agents
- [x] Implement handoff triggers
- [x] Track individual agent contributions
- [x] Write tests

**Traces to:** [REQ-601]
**Depends on:** [TASK-601]
**Blocks:** [TASK-604]

---

### TASK-604: Multi-Agent Test Suite Format
🟠 P1 | ✅ DONE | Est: 2-3h

**Description:**
Extend test suite YAML format for multi-agent tests.

**Checklist:**
- [x] Extend `TestDefinition` with multi-agent fields
- [x] Add `agents` field (list of agent names)
- [x] Add `mode` field: comparison, collaboration, handoff
- [x] Add validation for multi-agent configurations
- [x] Update test loader
- [x] Create example multi-agent test suite
- [x] Update documentation

**Traces to:** [REQ-601]
**Depends on:** [TASK-601], [TASK-602], [TASK-603]
**Blocks:** -

---

### TASK-605: Chaos Engineering Module
🟡 P2 | 🔄 IN_PROGRESS | Est: 5-6h

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
🟠 P1 | ✅ DONE | Est: 5-6h

**Description:**
Implement schema-per-tenant database architecture.

**Checklist:**
- [x] Create `atp/dashboard/tenancy/__init__.py`
- [x] Create `TenantManager` class
- [x] Implement `create_tenant()` with schema creation
- [x] Implement `delete_tenant()` with cleanup
- [x] Implement `TenantAwareSession` for scoped queries
- [x] Add tenant_id to all models
- [x] Create migration for default tenant
- [x] Migrate existing data to default tenant
- [x] Write tests

**Traces to:** [REQ-701]
**Depends on:** [TASK-105]
**Blocks:** [TASK-702], [TASK-703]

---

### TASK-702: Tenant Management API
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Implement REST API for tenant management.

**Checklist:**
- [x] Add `/api/v2/tenants` endpoints
- [x] Implement tenant CRUD operations
- [x] Implement quota configuration
- [x] Implement settings configuration
- [x] Add admin-only authorization
- [x] Write API tests
- [x] Create documentation

**Traces to:** [REQ-701]
**Depends on:** [TASK-701]
**Blocks:** [TASK-703]

---

### TASK-703: Tenant Quotas Enforcement
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Implement quota enforcement for tenants.

**Checklist:**
- [x] Create `atp/dashboard/tenancy/quotas.py`
- [x] Define quota types: tests/day, parallel runs, storage, agents, budget
- [x] Implement quota checking middleware
- [x] Implement quota usage tracking
- [x] Add quota exceeded responses (HTTP 429)
- [x] Add quota usage API endpoint
- [x] Write tests

**Traces to:** [REQ-701]
**Depends on:** [TASK-702]
**Blocks:** -

---

### TASK-704: RBAC - Roles & Permissions
🟠 P1 | ✅ DONE | Est: 4-5h

**Description:**
Implement role-based access control.

**Checklist:**
- [x] Create `atp/dashboard/auth/rbac.py`
- [x] Define `Permission` enum with all permissions
- [x] Define default roles: admin, developer, analyst, viewer
- [x] Implement `ROLE_PERMISSIONS` mapping
- [x] Create `Role` and `UserRole` models
- [x] Implement `has_permission()` function
- [x] Implement `@require_permission` decorator
- [x] Add user-role assignment API
- [x] Write tests

**Traces to:** [REQ-702]
**Depends on:** -
**Blocks:** [TASK-705]

---

### TASK-705: RBAC - API Integration
🟠 P1 | 🔄 IN_PROGRESS | Est: 3-4h

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
🟠 P1 | ✅ DONE | Est: 4-5h

**Description:**
Implement OIDC (OpenID Connect) authentication.

**Checklist:**
- [x] Add `authlib` to dependencies
- [x] Create `atp/dashboard/auth/sso/__init__.py`
- [x] Create `atp/dashboard/auth/sso/oidc.py`
- [x] Implement OIDC authorization flow
- [x] Implement token validation
- [x] Implement user provisioning (JIT)
- [x] Implement group-to-role mapping
- [x] Add configuration for popular providers (Okta, Auth0, Azure AD)
- [x] Write tests

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
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Implement comprehensive audit logging.

**Checklist:**
- [x] Create `atp/dashboard/audit.py`
- [x] Create `AuditLog` model with all required fields
- [x] Implement `audit_log()` function
- [x] Add audit middleware for all state-changing operations
- [x] Log: auth, data access, config changes, admin actions
- [x] Add `/api/v2/audit` endpoint with filtering
- [x] Implement retention policy
- [x] Write tests

**Traces to:** [REQ-704]
**Depends on:** [TASK-701]
**Blocks:** -

---

## Milestone 8: Dashboard Enhancements

### TASK-801: WebSocket Real-Time Updates
🟠 P1 | 🔄 IN_PROGRESS | Est: 4-5h

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

TASK-701 (Tenancy Schema) ✅
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

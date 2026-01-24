# Requirements Specification

## Overview

This document describes the functional and non-functional requirements for Agent Test Platform (ATP).

---

## Functional Requirements

### FR-1: Agent Interaction Protocol

#### FR-1.1: Standard Request Format
- The system MUST define a JSON schema for agent requests (ATP Request)
- The request MUST include: task description, input data, constraints
- Constraints MUST support: max_steps, max_tokens, timeout, allowed_tools

#### FR-1.2: Standard Response Format
- The system MUST define a JSON schema for agent responses (ATP Response)
- The response MUST include: status, artifacts, metrics
- Metrics MUST include: total_tokens, total_steps, tool_calls, wall_time

#### FR-1.3: Event Streaming
- The system MUST support event streaming during execution (ATP Event)
- Events MUST include types: tool_call, llm_request, reasoning, error
- Each event MUST have a timestamp and sequence number

#### FR-1.4: Protocol Versioning
- The protocol MUST have a version in each message
- The system MUST support backward compatibility

### FR-2: Agent Integration

#### FR-2.1: HTTP Integration
- The system MUST support agents via HTTP endpoint
- The endpoint MUST accept POST with ATP Request
- The endpoint MAY support SSE for event streaming

#### FR-2.2: Container Integration
- The system MUST support agents as Docker containers
- The container MUST read ATP Request from stdin
- The container MUST write ATP Response to stdout
- The container MAY write ATP Events to stderr or a separate port

#### FR-2.3: Framework Adapters
- The system MUST provide a base class for adapters
- The adapter MUST translate ATP Protocol <-> native framework API
- The system MUST include adapters for: LangGraph, CrewAI
- The system MAY include a legacy adapter for AutoGen

#### FR-2.4: CLI Wrapper
- The system MUST support agents via CLI
- The CLI agent MUST accept a path to a file with ATP Request
- The CLI agent MUST write ATP Response to a specified file

### FR-3: Test Description

#### FR-3.1: Declarative Format
- Tests MUST be described in YAML format
- The format MUST support: task definition, assertions, scoring weights
- The format MUST be human-readable

#### FR-3.2: Test Suites
- The system MUST support grouping tests into suites
- A suite MUST have: name, description, defaults, list of tests
- Defaults MUST be inherited by tests and can be overridden

#### FR-3.3: Test Parameterization
- Tests MUST support parameters (variables)
- Parameters MUST be substituted in task description and input data
- The system MUST support matrix parameterization

#### FR-3.4: Tags and Filtering
- Tests MUST support tags
- CLI MUST support filtering by tags
- Standard tags: smoke, regression, edge_case, performance

### FR-4: Test Execution

#### FR-4.1: Test Runner
- The system MUST provide a CLI for running tests
- The runner MUST support: single test, test suite, all tests
- The runner MUST output progress and results

#### FR-4.2: Sandbox Environment
- The system MUST isolate agent execution
- The sandbox MUST limit: time, memory, network (optional)
- The sandbox MUST use Docker for isolation

#### FR-4.3: Mock Tools
- The system MUST provide mock tools for testing
- Mock MUST record calls for verification
- Mock MUST support: fixed responses, response files, callback

#### FR-4.4: Parallel Execution
- The runner MUST support parallel test execution
- Parallelism MUST be configurable via CLI and config
- The system MUST correctly aggregate results

#### FR-4.5: Multiple Runs
- The system MUST support N runs of a single test
- The system MUST compute statistics across runs
- Statistics MUST include: mean, std, min, max, confidence interval

### FR-5: Evaluation System

#### FR-5.1: Artifact Evaluator
- The system MUST verify artifact presence and format
- Checks MUST include: file exists, JSON schema, contains text
- Checks MUST support regex patterns

#### FR-5.2: Behavior Evaluator
- The system MUST analyze execution trace
- Checks MUST include: used tools, number of steps, no hallucinations
- Checks MUST support: must, must_not, should (warning)

#### FR-5.3: LLM-as-Judge Evaluator
- The system MUST use LLM for semantic evaluation
- The evaluator MUST support: custom prompts, criteria
- The evaluator MUST return score 0-1 and explanation

#### FR-5.4: Code Execution Evaluator
- The system MUST run generated code
- The evaluator MUST support: pytest, npm test, custom command
- Test results MUST be converted to metrics

#### FR-5.5: Composite Scoring
- The system MUST support weighted scoring
- Weights MUST be configurable in test definition
- The final score MUST be 0-100

### FR-6: Reporting

#### FR-6.1: Console Reporter
- The system MUST output results to console
- Output MUST include: pass/fail, score, duration
- Output MUST support verbose mode for details

#### FR-6.2: JSON Reporter
- The system MUST export results to JSON
- JSON MUST include complete information for analysis
- The format MUST be documented

#### FR-6.3: HTML Reporter
- The system MUST generate HTML report
- The report MUST include: summary, details, charts
- The report MUST be self-contained (single file)

#### FR-6.4: JUnit XML Reporter
- The system MUST support JUnit XML format
- The format MUST be compatible with CI systems

### FR-7: CI/CD Integration

#### FR-7.1: Exit Codes
- CLI MUST return 0 on all tests passing
- CLI MUST return non-zero on failures
- CLI MUST support --fail-fast for early exit

#### FR-7.2: Baseline Comparison
- The system MUST support saving baseline results
- The system MUST compare current results with baseline
- Regression MUST be defined as statistically significant degradation

#### FR-7.3: GitHub Actions Integration
- The system MUST provide a ready action
- The action MUST support: matrix builds, caching, artifacts

### FR-8: Configuration

#### FR-8.1: Project Config
- The system MUST support atp.config.yaml in project root
- Config MUST include: defaults, agent definitions, paths
- CLI options MUST override config

#### FR-8.2: Agent Registry
- The system MUST support agent registration
- The registry MUST store: name, type, endpoint/image, config
- Agents MUST be referenced by name in tests

---

## Non-Functional Requirements

### NFR-1: Performance

#### NFR-1.1: Test Execution Overhead
- Platform overhead MUST be < 5% of agent execution time
- CLI startup time MUST be < 2 seconds

#### NFR-1.2: Scalability
- The system MUST support 100+ tests in a suite
- The system MUST support 10+ parallel agents
- Reporting MUST handle 10,000+ events

### NFR-2: Reliability

#### NFR-2.1: Error Handling
- The system MUST gracefully handle agent timeout
- The system MUST gracefully handle agent crash
- The system MUST continue execution on single test failure

#### NFR-2.2: Idempotency
- Repeated test runs MUST yield statistically similar results
- Cleanup between tests MUST be complete

### NFR-3: Usability

#### NFR-3.1: Learning Curve
- A new user MUST run their first test in < 1 hour
- Documentation MUST include tutorials and examples

#### NFR-3.2: Error Messages
- Errors MUST be clear and actionable
- Errors MUST point to specific location in config/test

#### NFR-3.3: Defaults
- The system MUST work with minimal configuration
- Reasonable defaults MUST cover 80% of cases

### NFR-4: Maintainability

#### NFR-4.1: Code Quality
- Code MUST have type hints (Python)
- Code MUST pass linting (ruff/flake8)
- Test coverage MUST be > 80%

#### NFR-4.2: Documentation
- Public API MUST be documented
- Architecture Decision Records MUST be maintained
- CHANGELOG MUST be updated

#### NFR-4.3: Modularity
- Components MUST be loosely coupled
- Adding a new evaluator MUST NOT require core changes

### NFR-5: Security

#### NFR-5.1: Sandbox Isolation
- Agent MUST NOT have access to host filesystem (except mounted paths)
- Agent MUST NOT have unrestricted network access

#### NFR-5.2: Secrets Handling
- API keys MUST be passed via environment variables
- Secrets MUST NOT be logged
- Secrets MUST NOT appear in reports

### NFR-6: Compatibility

#### NFR-6.1: Python Versions
- The system MUST support Python 3.10+
- Dependencies MUST be compatible with major frameworks

#### NFR-6.2: OS Support
- The system MUST work on Linux (primary)
- The system MUST work on macOS (development)
- The system MAY work on Windows (best effort)

#### NFR-6.3: Docker
- The system MUST work with Docker 20.10+
- The system MAY support Podman as an alternative

---

## Constraints

### C-1: Technology Constraints
- Implementation language: Python 3.10+
- Packaging: pip + pyproject.toml
- Container runtime: Docker (primary)

### C-2: Integration Constraints
- Protocol: JSON over HTTP/stdin-stdout
- Schema validation: JSON Schema draft-07
- YAML parsing: PyYAML or ruamel.yaml

### C-3: Licensing Constraints
- Platform: MIT License
- Dependencies: only MIT/Apache/BSD compatible

---

## Acceptance Criteria

### MVP (Milestone 1)
- [ ] ATP Protocol schema defined and documented
- [ ] HTTP adapter working
- [ ] Container adapter working
- [ ] Artifact evaluator working
- [ ] Behavior evaluator working
- [ ] Console reporter working
- [ ] JSON reporter working
- [ ] CLI with commands: test, version
- [ ] 3+ example test suites
- [ ] Basic documentation

### Beta (Milestone 2)
- [ ] LangGraph adapter
- [ ] CrewAI adapter
- [ ] LLM-as-Judge evaluator
- [ ] HTML reporter
- [ ] Multiple runs with statistics
- [ ] Baseline comparison
- [ ] GitHub Action
- [ ] Comprehensive documentation

### GA (Milestone 3)
- [ ] Web Dashboard (basic)
- [ ] All evaluators documented
- [ ] Performance benchmarks
- [ ] Security audit
- [ ] Community feedback incorporated

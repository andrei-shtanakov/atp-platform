# Roadmap

## Overview

This document describes the development plan for Agent Test Platform (ATP). The roadmap is divided into phases with clear deliverables and success criteria.

---

## Timeline Overview

```
2025 Q1          2025 Q2          2025 Q3          2025 Q4
    │                │                │                │
    ▼                ▼                ▼                ▼
┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
│Phase 1 │      │Phase 2 │      │Phase 3 │      │Phase 4 │
│  MVP   │      │  Beta  │      │   GA   │      │ Growth │
│   ✅   │      │   ✅   │      │   ✅   │      │   📋   │
│Core    │      │Adapters│      │Dashboard│     │Ecosystem│
│Protocol│      │CI/CD   │      │Security │     │Community│
└────────┘      └────────┘      └────────┘      └────────┘
  DONE           DONE            DONE           PLANNED
```

**Current Status**: Phases 1-3 complete. Phase 4 (Growth) planned for Q4 2025.

---

## Phase 1: MVP (Q1 2025) ✅ COMPLETE

### Goal
Working prototype that can be used for basic agent testing.

### Deliverables

#### 1.1 Core Protocol (Weeks 1-2) ✅
- [x] ATP Request/Response/Event models (Pydantic)
- [x] JSON Schema definitions
- [x] Protocol documentation
- [x] Validation utilities

**Status**: Complete (TASK-001)

#### 1.2 Basic Adapters (Weeks 2-4) ✅
- [x] HTTP Adapter (sync + async)
- [x] Container Adapter (Docker)
- [x] CLI Adapter (stdin/stdout)

**Status**: Complete (TASK-003)

#### 1.3 Test Loader (Weeks 3-4) ✅
- [x] YAML parser for test definitions
- [x] Test suite model
- [x] Variable substitution
- [x] Tag filtering

**Status**: Complete (TASK-004, TASK-005)

#### 1.4 Basic Evaluators (Weeks 4-6) ✅
- [x] Artifact Evaluator (exists, contains, format)
- [x] Behavior Evaluator (tool usage, step count)
- [x] Score Aggregator (combines results)

**Status**: Complete (TASK-007, TASK-008)

#### 1.5 Test Runner (Weeks 5-7) ✅
- [x] Single test execution
- [x] Suite execution
- [x] Basic sandbox (Docker)
- [x] Timeout handling

**Status**: Complete (TASK-006)

#### 1.6 Reporters (Weeks 6-7) ✅
- [x] Console Reporter (colored output)
- [x] JSON Reporter (machine-readable)

**Status**: Complete (TASK-009)

#### 1.7 CLI (Weeks 7-8) ✅
- [x] `atp test` command
- [x] `atp validate` command
- [x] `atp version` command
- [x] Config file support (atp.config.yaml)

**Status**: Complete (TASK-014)

### MVP Exit Criteria ✅
- [x] Core functionality implemented
- [x] Documentation covers basic use cases
- [x] Example test suites available
- [x] Test coverage ≥80%

---

## Phase 2: Beta (Q2 2025) ✅ COMPLETE

### Goal
Production-ready for internal use with support for major frameworks.

### Deliverables

#### 2.1 Framework Adapters (Weeks 1-3) ✅
- [x] LangGraph Adapter
- [x] CrewAI Adapter
- [x] AutoGen Adapter (legacy)
- [x] Adapter development guide

**Status**: Complete (TASK-010)

#### 2.2 Advanced Evaluators (Weeks 2-5) ✅
- [x] LLM-as-Judge Evaluator
- [x] Code Execution Evaluator (pytest, npm test)
- [x] Evaluator registry system

**Status**: Complete (TASK-012, TASK-018)

#### 2.3 Statistical Analysis (Weeks 4-6) ✅
- [x] Multiple runs per test
- [x] Mean, std, confidence intervals (t-distribution)
- [x] Coefficient of Variation
- [x] Stability scoring (stable/moderate/unstable/critical)

**Status**: Complete (TASK-011)

#### 2.4 Baseline & Regression (Weeks 5-7) ✅
- [x] Baseline save/load
- [x] Welch's t-test for comparison
- [x] Regression detection (p < 0.05)
- [x] Diff visualization (console/JSON)

**Status**: Complete (TASK-013)

#### 2.5 CI/CD Integration (Weeks 6-8) ✅
- [x] GitHub Action
- [x] GitLab CI template
- [x] JUnit XML reporter
- [x] Exit codes documentation
- [x] Azure Pipelines, CircleCI, Jenkins examples

**Status**: Complete (TASK-017)

#### 2.6 HTML Reporter (Weeks 7-8) ✅
- [x] Self-contained HTML report
- [x] Charts (Chart.js inline)
- [x] Test details with drill-down
- [x] Failed checks highlighting

**Status**: Complete (TASK-016)

### Beta Exit Criteria ✅
- [x] All Beta features implemented
- [x] Documentation complete
- [x] Test coverage ≥80%

---

## Phase 3: GA (Q3 2025) ✅ COMPLETE

### Goal
Stable release ready for broad adoption.

### Deliverables

#### 3.1 Web Dashboard (Weeks 1-5) ✅
- [x] FastAPI backend
- [x] Test results viewer
- [x] Agent comparison
- [x] Historical trends
- [x] Results storage (SQLite/PostgreSQL)
- [x] Basic authentication

**Status**: Complete (TASK-021)

#### 3.2 Advanced Features (Weeks 3-6) ✅
- [x] Parallel test execution (--parallel flag)
- [x] Mock tools server
- [x] Performance profiling
- [x] Caching layer

**Status**: Complete (TASK-020, TASK-019, TASK-023)

#### 3.3 Security Hardening (Weeks 6-8) ✅
- [x] Input validation audit
- [x] Path traversal protection
- [x] URL/DNS validation
- [x] Sandbox isolation
- [x] Security documentation

**Status**: Complete (TASK-022)

#### 3.4 Documentation & Training (Weeks 7-10) ✅
- [x] Complete API reference
- [x] Architecture documentation
- [x] Best practices guide
- [x] Troubleshooting guide
- [x] Adapter development guide
- [x] Migration guide
- [x] FAQ

**Status**: Complete (TASK-024)

### GA Exit Criteria ✅
- [x] All GA features implemented
- [x] Security audit complete
- [x] Full documentation
- [x] Test coverage ≥80%

---

## Phase 4: Growth (Q4 2025)

### Goal
Expanding functionality and community building.

### Deliverables

#### 4.1 Ecosystem (Weeks 1-4)
- [ ] Plugin marketplace (evaluators, adapters)
- [ ] Community test suites
- [ ] Integration templates

#### 4.2 Advanced Analytics (Weeks 3-6)
- [ ] Agent comparison reports
- [ ] Cost optimization recommendations
- [ ] Anomaly detection

#### 4.3 Enterprise Features (Weeks 5-8)
- [ ] Multi-tenant support
- [ ] Role-based access control
- [ ] Audit logging
- [ ] SLA reporting

#### 4.4 Community (Ongoing)
- [ ] Open source adapters
- [ ] Contributing guide
- [ ] Community calls
- [ ] Case studies

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM eval inconsistency | High | Multi-run averaging, calibration |
| Framework fragmentation | Medium | Focus on top 3, clear adapter API |
| Performance bottlenecks | Medium | Early benchmarking, async design |
| Docker dependency | Low | Podman support, native option |

### Adoption Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Learning curve | High | Tutorials, examples, support |
| Migration effort | Medium | Wrapper generators, guides |
| Framework lock-in fears | Medium | Clear framework-agnostic messaging |

---

## Success Metrics

### Phase 1 (MVP)
- 3 teams using ATP
- 10 test suites created
- <30 min to first test

### Phase 2 (Beta)
- 10 teams using ATP
- 50 test suites created
- 2 framework adapters in use
- CI/CD integration in 5+ repos

### Phase 3 (GA)
- 50 users
- 200 test suites
- Dashboard DAU >20
- <1 critical bug/month

### Phase 4 (Growth)
- 100+ users
- Community contributions
- 5+ external case studies

---

## Resource Requirements

### Phase 1 (MVP)
- 2 developers full-time
- 1 designer (part-time, docs)

### Phase 2 (Beta)
- 3 developers full-time
- 1 DevOps (CI/CD, infra)
- 1 technical writer

### Phase 3 (GA)
- 4 developers
- 1 DevOps
- 1 frontend (dashboard)
- 1 technical writer

### Phase 4 (Growth)
- 5 developers
- 2 DevOps
- 1 community manager
- 1 product manager

---

## Dependencies

### External
- Docker (container runtime)
- Anthropic/OpenAI API (LLM eval)
- GitHub/GitLab (CI/CD)

### Internal
- Agent teams (early adopters)
- Security team (review)
- Infrastructure (hosting)

---

## Open Questions

1. **Self-hosted vs SaaS**: Priority on self-hosted or cloud dashboard?
2. **Pricing model** (if external): Open source core + enterprise features?
3. **Framework priorities**: LangGraph, CrewAI, what else?
4. **Community**: Open source from day one or after GA?

---

## Appendix: Milestone Checklist

### MVP Milestone (End of Q1) ✅ COMPLETE
- [x] Protocol spec finalized (TASK-001)
- [x] Core adapters working (TASK-003)
- [x] Basic evaluators working (TASK-007)
- [x] CLI functional (TASK-014)
- [x] Basic docs complete (TASK-015)
- [x] Test infrastructure (TASK-100, TASK-101, TASK-102, TASK-103)

### Beta Milestone (End of Q2) ✅ COMPLETE
- [x] Framework adapters released (TASK-010)
- [x] LLM eval working (TASK-012)
- [x] CI/CD integration complete (TASK-017)
- [x] Statistical analysis working (TASK-011)
- [x] Baseline & regression (TASK-013)
- [x] Full docs complete

### GA Milestone (End of Q3) ✅ COMPLETE
- [x] Dashboard live (TASK-021)
- [x] Security hardening (TASK-022)
- [x] Performance optimization (TASK-023)
- [x] Complete documentation (TASK-024)

### Growth Milestone (End of Q4) 📋 PLANNED
- [ ] Plugin ecosystem
- [ ] Multi-tenant support
- [ ] Advanced analytics
- [ ] Community building

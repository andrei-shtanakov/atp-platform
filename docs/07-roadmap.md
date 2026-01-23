# Roadmap

## Обзор

Данный документ описывает план развития Agent Test Platform (ATP). Roadmap разбит на фазы с чёткими deliverables и критериями успеха.

---

## Timeline Overview

```
2025 Q1          2025 Q2          2025 Q3          2025 Q4
    │                │                │                │
    ▼                ▼                ▼                ▼
┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
│Phase 1 │      │Phase 2 │      │Phase 3 │      │Phase 4 │
│  MVP   │      │  Beta  │      │   GA   │      │ Growth │
│        │      │        │      │        │      │        │
│Core    │      │Adapters│      │Advanced│      │Ecosystem│
│Protocol│      │CI/CD   │      │Eval    │      │Community│
└────────┘      └────────┘      └────────┘      └────────┘
```

---

## Phase 1: MVP (Q1 2025)

### Цель
Работающий прототип, который можно использовать для базового тестирования агентов.

### Deliverables

#### 1.1 Core Protocol (Weeks 1-2)
- [ ] ATP Request/Response/Event models (Pydantic)
- [ ] JSON Schema definitions
- [ ] Protocol documentation
- [ ] Validation utilities

**Success Criteria**: Схемы прошли review, есть примеры валидных/невалидных сообщений.

#### 1.2 Basic Adapters (Weeks 2-4)
- [ ] HTTP Adapter (sync + async)
- [ ] Container Adapter (Docker)
- [ ] CLI Adapter (stdin/stdout)

**Success Criteria**: Все три адаптера работают с простым echo-агентом.

#### 1.3 Test Loader (Weeks 3-4)
- [ ] YAML parser для test definitions
- [ ] Test suite model
- [ ] Variable substitution
- [ ] Tag filtering

**Success Criteria**: Парсятся example test suites без ошибок.

#### 1.4 Basic Evaluators (Weeks 4-6)
- [ ] Artifact Evaluator (exists, contains, format)
- [ ] Behavior Evaluator (tool usage, step count)
- [ ] Composite Evaluator (combines results)

**Success Criteria**: Пройдены unit tests для каждого evaluator.

#### 1.5 Test Runner (Weeks 5-7)
- [ ] Single test execution
- [ ] Suite execution
- [ ] Basic sandbox (Docker)
- [ ] Timeout handling

**Success Criteria**: Можно запустить suite из 10 тестов.

#### 1.6 Reporters (Weeks 6-7)
- [ ] Console Reporter (colored output)
- [ ] JSON Reporter (machine-readable)

**Success Criteria**: Вывод читаем, JSON валиден.

#### 1.7 CLI (Weeks 7-8)
- [ ] `atp test` command
- [ ] `atp validate` command (check agent integration)
- [ ] `atp version` command
- [ ] Config file support (atp.config.yaml)

**Success Criteria**: CLI работает end-to-end.

### MVP Exit Criteria
- [ ] 3 внутренние команды используют ATP для тестирования
- [ ] Документация покрывает basic use cases
- [ ] 5+ example test suites
- [ ] <5 critical bugs

---

## Phase 2: Beta (Q2 2025)

### Цель
Production-ready для внутреннего использования с поддержкой основных фреймворков.

### Deliverables

#### 2.1 Framework Adapters (Weeks 1-3)
- [ ] LangGraph Adapter
- [ ] CrewAI Adapter
- [ ] AutoGen Adapter (legacy)
- [ ] Adapter development guide

**Success Criteria**: Каждый адаптер протестирован с реальным агентом.

#### 2.2 Advanced Evaluators (Weeks 2-5)
- [ ] LLM-as-Judge Evaluator
- [ ] Code Execution Evaluator (pytest, npm test)
- [ ] Custom evaluator plugin system

**Success Criteria**: LLM eval работает с calibration примерами.

#### 2.3 Statistical Analysis (Weeks 4-6)
- [ ] Multiple runs per test
- [ ] Mean, std, confidence intervals
- [ ] Variance analysis
- [ ] Stability scoring

**Success Criteria**: Статистика корректна на synthetic data.

#### 2.4 Baseline & Regression (Weeks 5-7)
- [ ] Baseline save/load
- [ ] Regression detection algorithm
- [ ] Diff visualization

**Success Criteria**: Детектирует 10% regression с p<0.05.

#### 2.5 CI/CD Integration (Weeks 6-8)
- [ ] GitHub Action
- [ ] GitLab CI template
- [ ] JUnit XML reporter
- [ ] Exit codes spec

**Success Criteria**: CI pipeline проходит для example repo.

#### 2.6 HTML Reporter (Weeks 7-8)
- [ ] Self-contained HTML report
- [ ] Charts (score distribution, trends)
- [ ] Drill-down to test details

**Success Criteria**: Report читаем без дополнительных инструментов.

### Beta Exit Criteria
- [ ] 10+ внутренних пользователей
- [ ] <2 critical bugs per week
- [ ] Documentation complete
- [ ] Performance benchmarks established

---

## Phase 3: GA (Q3 2025)

### Цель
Stable release готовый для широкого использования.

### Deliverables

#### 3.1 Web Dashboard (Weeks 1-5)
- [ ] Test results viewer
- [ ] Agent comparison
- [ ] Historical trends
- [ ] User authentication

**Success Criteria**: Dashboard доступен и usable.

#### 3.2 Advanced Features (Weeks 3-6)
- [ ] Parallel test execution
- [ ] Distributed runners
- [ ] Cost tracking & budgets
- [ ] Mock tools library

**Success Criteria**: 10x speedup с параллельным выполнением.

#### 3.3 LLM Judge Improvements (Weeks 4-7)
- [ ] Calibration framework
- [ ] Multi-model ensembling
- [ ] Human-in-the-loop feedback
- [ ] Bias detection

**Success Criteria**: Correlation с human eval >0.8.

#### 3.4 Security Hardening (Weeks 6-8)
- [ ] Security audit
- [ ] Sandbox improvements
- [ ] Secret management
- [ ] Input sanitization

**Success Criteria**: Пройден internal security review.

#### 3.5 Documentation & Training (Weeks 7-10)
- [ ] Complete API reference
- [ ] Video tutorials
- [ ] Best practices guide
- [ ] Troubleshooting guide

**Success Criteria**: New user onboarding <1 hour.

### GA Exit Criteria
- [ ] 50+ active users
- [ ] <1 critical bug per month
- [ ] 99.9% uptime (dashboard)
- [ ] Performance SLAs met

---

## Phase 4: Growth (Q4 2025)

### Цель
Расширение функциональности и community building.

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

1. **Self-hosted vs SaaS**: Приоритет self-hosted или cloud dashboard?
2. **Pricing model** (if external): Open source core + enterprise features?
3. **Framework priorities**: LangGraph, CrewAI, что ещё?
4. **Community**: Open source с первого дня или после GA?

---

## Appendix: Milestone Checklist

### MVP Milestone (End of Q1)
- [ ] Protocol spec finalized
- [ ] Core adapters working
- [ ] Basic evaluators working
- [ ] CLI functional
- [ ] 3 teams onboarded
- [ ] Basic docs complete

### Beta Milestone (End of Q2)
- [ ] Framework adapters released
- [ ] LLM eval working
- [ ] CI/CD integration complete
- [ ] Statistical analysis working
- [ ] 10 teams onboarded
- [ ] Full docs complete

### GA Milestone (End of Q3)
- [ ] Dashboard live
- [ ] Security audit passed
- [ ] Performance SLAs met
- [ ] 50 users active
- [ ] Video tutorials published

### Growth Milestone (End of Q4)
- [ ] Plugin ecosystem launched
- [ ] 100+ users
- [ ] Community contributions
- [ ] External case studies

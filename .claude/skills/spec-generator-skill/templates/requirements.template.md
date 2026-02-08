# Requirements Specification

> {{PROJECT_NAME}} — {{PROJECT_TAGLINE}}

## 1. Context and Goals

### 1.1 Problem

{{PROBLEM_DESCRIPTION}}

### 1.2 Project Goals

| ID | Goal | Success Metric |
|----|------|----------------|
| G-1 | {{GOAL_1}} | {{METRIC_1}} |
| G-2 | {{GOAL_2}} | {{METRIC_2}} |
| G-3 | {{GOAL_3}} | {{METRIC_3}} |

### 1.3 Stakeholders

| Role | Interests | Influence |
|------|----------|---------|
| {{ROLE_1}} | {{INTERESTS_1}} | High/Medium/Low |
| {{ROLE_2}} | {{INTERESTS_2}} | High/Medium/Low |

### 1.4 Out of Scope

> ⚠️ Explicitly state what is NOT included in the project

- ❌ {{OUT_OF_SCOPE_1}}
- ❌ {{OUT_OF_SCOPE_2}}
- ❌ {{OUT_OF_SCOPE_3}}

---

## 2. Functional Requirements

### 2.1 {{FEATURE_GROUP_1}}

#### REQ-001: {{REQUIREMENT_NAME}}
**As a** {{ROLE}}
**I want** {{ACTION}}
**So that** {{VALUE}}

**Acceptance Criteria:**
```gherkin
GIVEN {{PRECONDITION}}
WHEN {{ACTION}}
THEN {{RESULT}}
AND {{ADDITIONAL_RESULT}}
```

**Priority:** P0 | P1 | P2 | P3
**Traces to:** [TASK-XXX], [DESIGN-XXX]

---

#### REQ-002: {{REQUIREMENT_NAME}}
**As a** {{ROLE}}
**I want** {{ACTION}}
**So that** {{VALUE}}

**Acceptance Criteria:**
```gherkin
GIVEN {{PRECONDITION}}
WHEN {{ACTION}}
THEN {{RESULT}}
```

**Priority:** P0 | P1 | P2 | P3
**Traces to:** [TASK-XXX], [DESIGN-XXX]

---

### 2.2 {{FEATURE_GROUP_2}}

#### REQ-010: {{REQUIREMENT_NAME}}
...

---

## 3. Non-Functional Requirements

### NFR-000: Testing Requirements
| Aspect | Requirement |
|--------|------------|
| Unit test coverage | ≥ {{COVERAGE}}% for core modules |
| Integration tests | Each key component |
| Test framework | {{TEST_FRAMEWORK}} |
| CI requirement | All tests pass before merge |

**Definition of Done for any task:**
- [ ] Unit tests written and passing
- [ ] Coverage has not decreased
- [ ] Integration test if interfaces are affected
- [ ] Documentation updated

**Traces to:** [TASK-100]

---

### NFR-001: Performance
| Metric | Requirement |
|--------|------------|
| {{METRIC}} | {{VALUE}} |

**Traces to:** [TASK-XXX]

---

### NFR-002: Security
| Aspect | Requirement |
|--------|------------|
| {{ASPECT}} | {{REQUIREMENT}} |

**Traces to:** [TASK-XXX]

---

### NFR-003: Usability
| Metric | Requirement |
|--------|------------|
| {{METRIC}} | {{VALUE}} |

**Traces to:** [TASK-XXX]

---

## 4. Constraints and Tech Stack

### 4.1 Technology Constraints

| Aspect | Decision | Rationale |
|--------|---------|-----------|
| Language | {{LANGUAGE}} | {{RATIONALE}} |
| Database | {{DB}} | {{RATIONALE}} |
| Infrastructure | {{INFRA}} | {{RATIONALE}} |

### 4.2 Integration Constraints

- {{INTEGRATION_CONSTRAINT_1}}
- {{INTEGRATION_CONSTRAINT_2}}

### 4.3 Business Constraints

- Budget: {{BUDGET}}
- Timeline: {{TIMELINE}}
- Team: {{TEAM_SIZE}}

---

## 5. Acceptance Criteria

### Milestone 1: MVP
- [ ] REQ-001 — {{DESCRIPTION}}
- [ ] REQ-002 — {{DESCRIPTION}}
- [ ] NFR-001 — {{DESCRIPTION}}

### Milestone 2: Beta
- [ ] REQ-010 — {{DESCRIPTION}}
- [ ] REQ-011 — {{DESCRIPTION}}

### Milestone 3: GA
- [ ] All P0 and P1 requirements implemented
- [ ] {{ADDITIONAL_CRITERIA}}

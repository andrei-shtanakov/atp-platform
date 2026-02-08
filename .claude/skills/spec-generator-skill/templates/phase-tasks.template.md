# Phase {{N}}: {{PHASE_NAME}} — Tasks

> Implementation tasks for {{PHASE_DESCRIPTION}} ({{TIMELINE}})
> Per ADR-{{NNN}}, Phase {{N}} Requirements, Phase {{N}} Design

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

## Milestone {{M}}: {{MILESTONE_NAME}}

### TASK-{{XXX}}: {{TASK_NAME}}
🔴 P0 | ⬜ TODO | Est: {{X}}h

**Description:**
{{TASK_DESCRIPTION}}

**Checklist:**
- [ ] {{SUBTASK_1}}
- [ ] {{SUBTASK_2}}
- [ ] {{SUBTASK_3}}
- [ ] Write unit tests for {{scope}}
- [ ] Write integration tests (if interfaces affected)

**Traces to:** [{{PREFIX}}-FR-{{YYY}}]
**Depends on:** —
**Blocks:** [TASK-{{ZZZ}}], [TASK-{{AAA}}]

---

### TASK-{{XXX+1}}: {{TASK_NAME}}
🟠 P1 | ⬜ TODO | Est: {{X}}h

**Description:**
{{TASK_DESCRIPTION}}

**Checklist:**
- [ ] {{SUBTASK_1}}
- [ ] {{SUBTASK_2}}
- [ ] Write tests: {{test_description}}

**Traces to:** [{{PREFIX}}-FR-{{YYY}}]
**Depends on:** [TASK-{{XXX}}]
**Blocks:** [TASK-{{ZZZ}}]

---

## Milestone {{M+1}}: {{MILESTONE_NAME}}

### TASK-{{XXX+10}}: {{TASK_NAME}}
🔴 P0 | ⬜ TODO | Est: {{X}}h

**Description:**
{{TASK_DESCRIPTION}}

**Checklist:**
- [ ] {{SUBTASK_1}}
- [ ] {{SUBTASK_2}}
- [ ] {{SUBTASK_3}}
- [ ] Verify on known examples: {{verification}}
- [ ] Write tests: {{test_description}}
- [ ] Performance: {{performance_requirement}}

**Traces to:** [{{PREFIX}}-FR-{{YYY}}]
**Depends on:** [TASK-{{XXX}}]
**Blocks:** [TASK-{{ZZZ}}]

---

### TASK-{{XXX+11}}: {{TASK_NAME}}
🟠 P1 | ⬜ TODO | Est: {{X}}h

**Description:**
{{TASK_DESCRIPTION}}

**Checklist:**
- [ ] {{SUBTASK_1}}
- [ ] {{SUBTASK_2}}
- [ ] Write tests: {{test_description}}

**Traces to:** [{{PREFIX}}-FR-{{YYY}}]
**Depends on:** [TASK-{{XXX+10}}]
**Blocks:** —

---

## Milestone {{M+2}}: {{MILESTONE_NAME}}

### TASK-{{XXX+20}}: {{TASK_NAME}}
🟠 P1 | ⬜ TODO | Est: {{X}}h

**Description:**
{{TASK_DESCRIPTION}}

**Checklist:**
- [ ] {{SUBTASK_1}}
- [ ] {{SUBTASK_2}}
- [ ] Write tests: {{test_description}}

**Traces to:** [{{PREFIX}}-FR-{{YYY}}]
**Depends on:** [TASK-{{XXX+10}}], [TASK-{{XXX+11}}]
**Blocks:** [TASK-{{ZZZ}}]

---

### TASK-{{XXX+21}}: Documentation & Examples
🟠 P1 | ⬜ TODO | Est: {{X}}h

**Description:**
Complete documentation for Phase {{N}}.

**Checklist:**
- [ ] API reference
- [ ] User guide / getting started
- [ ] Examples (Jupyter notebooks or scripts)
- [ ] Update main project README
- [ ] Update existing docs with Phase {{N}} references

**Traces to:** All Phase {{N}} requirements
**Depends on:** [TASK-{{XXX}}..{{XXX+20}}]
**Blocks:** —

---

### TASK-{{XXX+22}}: CI/CD & Publishing
🟠 P1 | ⬜ TODO | Est: {{X}}h

**Description:**
Set up CI/CD and publish package(s).

**Checklist:**
- [ ] CI pipeline: pytest + coverage + linting
- [ ] Coverage gate: ≥ {{COVERAGE}}%
- [ ] Publishing workflow (PyPI / npm / etc.)
- [ ] Version: 0.1.0

**Traces to:** [{{PREFIX}}-NFR-{{YYY}}]
**Depends on:** [TASK-{{XXX+21}}]
**Blocks:** —

---

## Dependency Graph

```
TASK-{{XXX}} ({{name}})
    │
    ├──► TASK-{{XXX+1}} ({{name}}) ──► TASK-{{XXX+10}} ({{name}})
    │                                        │
    ├──► TASK-{{XXX+2}} ({{name}})           ├──► TASK-{{XXX+11}} ({{name}})
    │                                        │
    └──► ...                                 └──► TASK-{{XXX+20}} ({{name}})
                                                       │
                                                       ▼
                                                  TASK-{{XXX+21}} (Docs)
                                                       │
                                                       ▼
                                                  TASK-{{XXX+22}} (Publish)
```

---

## Summary

| Milestone | Tasks | Total Est. Hours |
|-----------|-------|------------------|
| M{{M}}: {{name}} | {{count}} | ~{{X}}-{{Y}}h |
| M{{M+1}}: {{name}} | {{count}} | ~{{X}}-{{Y}}h |
| M{{M+2}}: {{name}} | {{count}} | ~{{X}}-{{Y}}h |
| **Total** | **{{count}}** | **~{{X}}-{{Y}}h (~{{W}} weeks)** |

---

## Critical Path

```
TASK-{{start}} → TASK-{{...}} → TASK-{{...}} → TASK-{{...}} → TASK-{{end}}
   {{X}}h         {{X}}h         {{X}}h         {{X}}h         {{X}}h
                                                          Total: ~{{X}}h
```

**Minimum duration with parallelization**: ~{{W}} weeks (one developer), ~{{W/2}} weeks (two developers)

---

## Recommended Execution Order

### Phase {{N}}.1 (Weeks 1-{{W1}}): Foundation
1. **Week 1**: TASK-{{XXX}} — critical blocker
2. **Week 2**: TASK-{{XXX+1}}, TASK-{{XXX+2}} in parallel
3. **Week 3-{{W1}}**: TASK-{{XXX+3}}..{{XXX+N}}

### Phase {{N}}.2 (Weeks {{W1+1}}-{{W2}}): Core
4. **Week {{W1+1}}**: TASK-{{XXX+10}} — second critical blocker
5. **Week {{W1+2}}**: TASK-{{XXX+11}}, TASK-{{XXX+12}}

### Phase {{N}}.3 (Weeks {{W2+1}}-{{W3}}): Polish & Release
6. **Week {{W2+1}}**: TASK-{{XXX+20}} — advanced features
7. **Week {{W3}}**: TASK-{{XXX+21}} (Docs), TASK-{{XXX+22}} (Publish)

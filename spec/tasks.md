# Tasks Specification

> Agent Comparison Dashboard — Implementation Tasks

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

## Milestone 1: Comparison Foundation

### TASK-100: Test Infrastructure
🔴 P0 | ✅ DONE | Est: 2-3h

**Description:**
Set up test infrastructure for new comparison features.

**Checklist:**
- [x] Create test fixtures with sample events data (`tests/fixtures/comparison/events.py`)
- [x] Add factory functions for TestExecution with events_json (`tests/fixtures/comparison/factories.py`)
- [x] Create mock data generator for leaderboard tests (`tests/fixtures/comparison/leaderboard.py`)
- [x] Verify existing test infrastructure covers new endpoints

**Traces to:** [NFR-003]
**Depends on:** -
**Blocks:** [TASK-001], [TASK-004], [TASK-007]

---

### TASK-001: Side-by-Side API Endpoint
🔴 P0 | ✅ DONE | Est: 3-4h

**Description:**
Implement `/compare/side-by-side` endpoint that returns detailed comparison data for 2-3 agents on a specific test.

**Checklist:**
- [ ] Add Pydantic schemas: `EventSummary`, `AgentExecutionDetail`, `SideBySideComparisonResponse`
- [ ] Implement query to get latest test execution per agent
- [ ] Extract and format events from `events_json`
- [ ] Calculate metrics summary for each agent
- [ ] Add endpoint to api.py with validation
- [ ] Write unit tests for endpoint
- [ ] Write integration test

**Traces to:** [REQ-001], [DESIGN-001]
**Depends on:** [TASK-100]
**Blocks:** [TASK-002]

---

### TASK-002: Step Comparison UI
🔴 P0 | ⬜ TODO | Est: 4-5h

**Description:**
Build React component for side-by-side step comparison view.

**Checklist:**
- [ ] Create `ComparisonContainer.tsx` with column layout
- [ ] Create `AgentSelector.tsx` multi-select dropdown
- [ ] Create `StepComparison.tsx` with event list
- [ ] Style events by type (tool_call, llm_request, etc.)
- [ ] Add loading and error states
- [ ] Integrate with API endpoint

**Traces to:** [REQ-002], [DESIGN-001]
**Depends on:** [TASK-001]
**Blocks:** [TASK-003]

---

### TASK-003: Metrics Comparison Panel
🟠 P1 | ⬜ TODO | Est: 2-3h

**Description:**
Add metrics panel showing score, tokens, duration comparison.

**Checklist:**
- [ ] Create `MetricsPanel.tsx` component
- [ ] Display: score, tokens, steps, duration, cost for each agent
- [ ] Highlight best values (highest score, lowest tokens)
- [ ] Show percentage differences
- [ ] Add responsive styling

**Traces to:** [REQ-003], [DESIGN-001]
**Depends on:** [TASK-002]
**Blocks:** -

---

### TASK-004: Leaderboard Matrix API
🔴 P0 | ✅ DONE | Est: 4-5h

**Description:**
Implement `/leaderboard/matrix` endpoint returning test × agent matrix.

**Checklist:**
- [ ] Add schemas: `TestScore`, `TestRow`, `AgentColumn`, `LeaderboardMatrixResponse`
- [ ] Implement aggregation query for scores by test and agent
- [ ] Calculate difficulty rating per test
- [ ] Calculate rankings for agents
- [ ] Add pagination support
- [ ] Write unit tests
- [ ] Write integration test

**Traces to:** [REQ-010], [DESIGN-002]
**Depends on:** [TASK-100]
**Blocks:** [TASK-005], [TASK-006]

---

### TASK-005: Leaderboard Matrix UI
🔴 P0 | ⬜ TODO | Est: 4-5h

**Description:**
Build React component for leaderboard matrix visualization.

**Checklist:**
- [ ] Create `MatrixGrid.tsx` table component
- [ ] Create `ScoreCell.tsx` with color coding
- [ ] Create `AgentHeader.tsx` with stats
- [ ] Create `TestRow.tsx` with test info
- [ ] Add sorting by columns
- [ ] Add responsive horizontal scroll

**Traces to:** [REQ-010], [DESIGN-002]
**Depends on:** [TASK-004]
**Blocks:** -

---

### TASK-006: Leaderboard Aggregations
🟠 P1 | ⬜ TODO | Est: 2-3h

**Description:**
Add summary row/column with aggregated statistics.

**Checklist:**
- [ ] Create `AggregationRow.tsx` component
- [ ] Calculate per-agent: avg score, pass rate, total tokens, total cost
- [ ] Calculate per-test: avg score, difficulty
- [ ] Show ranking badges (1st, 2nd, 3rd)
- [ ] Add pattern badges ("hard for all", "easy")

**Traces to:** [REQ-011], [REQ-012], [DESIGN-002]
**Depends on:** [TASK-005]
**Blocks:** -

---

### TASK-007: Timeline Events API
🔴 P0 | 🔄 IN_PROGRESS | Est: 3-4h

**Description:**
Implement `/timeline/events` endpoint for single agent timeline.

**Checklist:**
- [ ] Add schemas: `TimelineEvent`, `TimelineEventsResponse`
- [ ] Extract events from `events_json` with relative timing
- [ ] Calculate event durations where applicable
- [ ] Add event type filtering
- [ ] Limit to 1000 events for performance
- [ ] Write unit tests

**Traces to:** [REQ-020], [DESIGN-003]
**Depends on:** [TASK-100]
**Blocks:** [TASK-008], [TASK-009]

---

### TASK-008: Multi-Agent Timeline API
🔴 P0 | ⬜ TODO | Est: 2-3h

**Description:**
Implement `/timeline/compare` endpoint for comparing agent timelines.

**Checklist:**
- [ ] Add schemas: `AgentTimeline`, `MultiTimelineResponse`
- [ ] Get timelines for 2-3 agents
- [ ] Align timelines by start time
- [ ] Include total duration for each
- [ ] Write unit tests

**Traces to:** [REQ-021], [DESIGN-003]
**Depends on:** [TASK-007]
**Blocks:** [TASK-010]

---

### TASK-009: Timeline UI Component
🔴 P0 | ⬜ TODO | Est: 5-6h

**Description:**
Build interactive timeline visualization using Chart.js.

**Checklist:**
- [ ] Create `TimelineContainer.tsx` with zoom controls
- [ ] Create `TimelineRow.tsx` for single agent
- [ ] Create `EventMarker.tsx` with type-based colors
- [ ] Create `TimeScale.tsx` with time labels
- [ ] Implement zoom in/out functionality
- [ ] Add hover tooltips with event summary

**Traces to:** [REQ-020], [DESIGN-003]
**Depends on:** [TASK-007]
**Blocks:** [TASK-010]

---

### TASK-010: Event Details and Filters
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Add event detail panel and type filters to timeline.

**Checklist:**
- [ ] Create `EventDetailPanel.tsx` with full event data
- [ ] Create `EventFilters.tsx` toggle buttons
- [ ] Show tool_call: name, args, result
- [ ] Show llm_request: prompt, response, tokens
- [ ] Show error: message, stack trace
- [ ] Add "copy as JSON" button

**Traces to:** [REQ-022], [REQ-023], [DESIGN-003]
**Depends on:** [TASK-008], [TASK-009]
**Blocks:** -

---

## Milestone 2: Polish

### TASK-011: Performance Optimization
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Optimize queries and rendering for performance requirements.

**Checklist:**
- [ ] Add database indexes for common queries
- [ ] Implement query result caching
- [ ] Optimize leaderboard query with proper JOINs
- [ ] Test with 50 tests × 10 agents
- [ ] Verify page load < 2 seconds

**Traces to:** [NFR-001]
**Depends on:** [TASK-004], [TASK-007]
**Blocks:** -

---

### TASK-012: UX Polish
🟠 P1 | ⬜ TODO | Est: 2-3h

**Description:**
Add loading states, error handling, and responsive design.

**Checklist:**
- [ ] Add skeleton loaders for all views
- [ ] Add error boundaries with retry buttons
- [ ] Test on 1280px, 1440px, 1920px widths
- [ ] Add keyboard navigation for timeline
- [ ] Verify all API errors show user-friendly messages

**Traces to:** [NFR-002]
**Depends on:** [TASK-002], [TASK-005], [TASK-009]
**Blocks:** -

---

## Dependency Graph

```
TASK-100 (Test Infrastructure) ✅
    │
    ├──► TASK-001 (Side-by-Side API)
    │        └──► TASK-002 (Step Comparison UI)
    │                 └──► TASK-003 (Metrics Panel)
    │
    ├──► TASK-004 (Leaderboard API)
    │        └──► TASK-005 (Leaderboard UI)
    │                 └──► TASK-006 (Aggregations)
    │
    └──► TASK-007 (Timeline Events API)
             ├──► TASK-008 (Multi-Timeline API)
             └──► TASK-009 (Timeline UI)
                      └──► TASK-010 (Event Details)

TASK-011 (Performance) depends on: TASK-004, TASK-007
TASK-012 (UX Polish) depends on: TASK-002, TASK-005, TASK-009
```

---

## Summary

| Milestone | Tasks | Total Est. |
|-----------|-------|------------|
| Foundation | TASK-001 to TASK-010 | ~32-37h |
| Polish | TASK-011, TASK-012 | ~5-7h |
| **Total** | 11 tasks remaining | ~37-44h |

**Completed:** TASK-100 (Test Infrastructure)

### Ready to Start
- [TASK-001] Side-by-Side API (TASK-100 done)
- [TASK-004] Leaderboard Matrix API (TASK-100 done)
- [TASK-007] Timeline Events API (TASK-100 done)

### Critical Path
TASK-001 → TASK-002 → TASK-003 (comparison)
TASK-004 → TASK-005 → TASK-006 (leaderboard)
TASK-007 → TASK-008/009 → TASK-010 (timeline)

# Design Specification

> Agent Comparison Dashboard — Technical Design

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         React Frontend                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Comparison     │   Leaderboard   │       Event Timeline        │
│  View           │   Matrix        │       Visualization         │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         │                 │                       │
         ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI REST API                            │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ /compare/       │ /leaderboard/   │      /timeline/             │
│ side-by-side    │ matrix          │      events                 │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         │                 │                       │
         ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SQLAlchemy ORM                              │
│  SuiteExecution → TestExecution → RunResult (events_json)       │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Components

### DESIGN-001: Side-by-Side Comparison Component

#### Description
React component for comparing 2-3 agents on a single test. Shows step-by-step execution with metrics panel.

#### Frontend Structure
```
ComparisonView/
├── AgentSelector.tsx      # Multi-select dropdown for agents
├── StepComparison.tsx     # Side-by-side step list
├── MetricsPanel.tsx       # Score, tokens, duration comparison
└── ComparisonContainer.tsx # Main layout with columns
```

#### API Endpoint
```python
@router.get("/compare/side-by-side", response_model=SideBySideComparisonResponse)
async def get_side_by_side_comparison(
    session: SessionDep,
    suite_name: str,
    test_id: str,
    agents: list[str] = Query(..., min_length=2, max_length=3),
) -> SideBySideComparisonResponse:
    """Get detailed comparison of agents on a specific test."""
    pass
```

#### Response Schema
```python
class AgentExecutionDetail(BaseModel):
    agent_name: str
    score: float | None
    success: bool
    duration_seconds: float | None
    total_tokens: int | None
    total_steps: int | None
    cost_usd: float | None
    events: list[EventSummary]  # Ordered list of events

class EventSummary(BaseModel):
    sequence: int
    timestamp: datetime
    event_type: str  # tool_call, llm_request, reasoning, error
    summary: str     # One-line description
    data: dict[str, Any]  # Full event data

class SideBySideComparisonResponse(BaseModel):
    suite_name: str
    test_id: str
    test_name: str
    agents: list[AgentExecutionDetail]
```

**Traces to:** [REQ-001], [REQ-002], [REQ-003]

---

### DESIGN-002: Leaderboard Matrix Component

#### Description
Matrix visualization showing tests (rows) vs agents (columns) with scores and aggregations.

#### Frontend Structure
```
Leaderboard/
├── MatrixGrid.tsx         # Main table component
├── ScoreCell.tsx          # Color-coded score cell
├── AgentHeader.tsx        # Column header with agent stats
├── TestRow.tsx            # Row with test info
├── AggregationRow.tsx     # Summary row with totals
└── PatternBadge.tsx       # "Hard for all", "Easy" badges
```

#### API Endpoint
```python
@router.get("/leaderboard/matrix", response_model=LeaderboardMatrixResponse)
async def get_leaderboard_matrix(
    session: SessionDep,
    suite_name: str,
    agents: list[str] | None = Query(None),  # Filter to specific agents
    limit_executions: int = Query(default=5, le=20),  # Last N executions per agent
) -> LeaderboardMatrixResponse:
    """Get leaderboard matrix for a suite."""
    pass
```

#### Response Schema
```python
class TestScore(BaseModel):
    score: float | None
    success: bool
    execution_count: int

class TestRow(BaseModel):
    test_id: str
    test_name: str
    tags: list[str]
    scores_by_agent: dict[str, TestScore]
    avg_score: float | None
    difficulty: str  # easy, medium, hard, very_hard
    pattern: str | None  # "hard_for_all", "easy", etc.

class AgentColumn(BaseModel):
    agent_name: str
    avg_score: float | None
    pass_rate: float
    total_tokens: int
    total_cost: float | None
    rank: int

class LeaderboardMatrixResponse(BaseModel):
    suite_name: str
    tests: list[TestRow]
    agents: list[AgentColumn]
    total_tests: int
```

#### Difficulty Calculation
```python
def calculate_difficulty(avg_score: float | None) -> str:
    if avg_score is None:
        return "unknown"
    if avg_score >= 80:
        return "easy"
    if avg_score >= 60:
        return "medium"
    if avg_score >= 40:
        return "hard"
    return "very_hard"
```

**Traces to:** [REQ-010], [REQ-011], [REQ-012]

---

### DESIGN-003: Event Timeline Component

#### Description
Interactive timeline visualization of agent events. Supports single agent view and multi-agent comparison.

#### Frontend Structure
```
Timeline/
├── TimelineContainer.tsx  # Main container with zoom controls
├── TimelineRow.tsx        # Single agent timeline
├── EventMarker.tsx        # Individual event on timeline
├── EventTooltip.tsx       # Hover tooltip with summary
├── EventDetailPanel.tsx   # Full event details panel
├── EventFilters.tsx       # Toggle buttons for event types
└── TimeScale.tsx          # Time axis with labels
```

#### API Endpoints
```python
@router.get("/timeline/events", response_model=TimelineEventsResponse)
async def get_timeline_events(
    session: SessionDep,
    test_execution_id: int,
    run_number: int = 1,
    event_types: list[str] | None = Query(None),  # Filter by type
    limit: int = Query(default=500, le=1000),
) -> TimelineEventsResponse:
    """Get events for timeline visualization."""
    pass

@router.get("/timeline/compare", response_model=MultiTimelineResponse)
async def get_multi_timeline(
    session: SessionDep,
    suite_name: str,
    test_id: str,
    agents: list[str] = Query(..., min_length=2, max_length=3),
    event_types: list[str] | None = Query(None),
) -> MultiTimelineResponse:
    """Get aligned timelines for multiple agents."""
    pass
```

#### Response Schemas
```python
class TimelineEvent(BaseModel):
    id: str  # Unique event ID
    sequence: int
    timestamp: datetime
    relative_time_ms: int  # Milliseconds from start
    event_type: str
    duration_ms: int | None  # For tool calls with response
    summary: str
    data: dict[str, Any]

class TimelineEventsResponse(BaseModel):
    test_execution_id: int
    run_number: int
    agent_name: str
    start_time: datetime
    end_time: datetime
    total_duration_ms: int
    events: list[TimelineEvent]
    event_counts: dict[str, int]  # Count by type

class AgentTimeline(BaseModel):
    agent_name: str
    start_time: datetime
    total_duration_ms: int
    events: list[TimelineEvent]

class MultiTimelineResponse(BaseModel):
    suite_name: str
    test_id: str
    test_name: str
    timelines: list[AgentTimeline]
```

#### Event Type Colors
```typescript
const EVENT_COLORS = {
  tool_call: '#3B82F6',    // Blue
  llm_request: '#10B981',  // Green
  reasoning: '#F59E0B',    // Yellow/Amber
  error: '#EF4444',        // Red
  progress: '#8B5CF6',     // Purple
};
```

**Traces to:** [REQ-020], [REQ-021], [REQ-022], [REQ-023]

---

## 3. Database Queries

### DESIGN-004: Comparison Queries

#### Side-by-Side Query
```python
async def get_agent_test_executions(
    session: AsyncSession,
    suite_name: str,
    test_id: str,
    agent_names: list[str],
) -> dict[str, TestExecution]:
    """Get latest test execution for each agent."""
    results = {}
    for agent_name in agent_names:
        stmt = (
            select(TestExecution)
            .join(SuiteExecution)
            .join(Agent)
            .where(
                SuiteExecution.suite_name == suite_name,
                TestExecution.test_id == test_id,
                Agent.name == agent_name,
            )
            .options(selectinload(TestExecution.run_results))
            .order_by(TestExecution.started_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        execution = result.scalar_one_or_none()
        if execution:
            results[agent_name] = execution
    return results
```

#### Leaderboard Query
```python
async def get_leaderboard_data(
    session: AsyncSession,
    suite_name: str,
    agent_names: list[str] | None,
    limit_per_agent: int,
) -> LeaderboardMatrixResponse:
    """Build leaderboard matrix from recent executions."""
    # Get all agents if not specified
    if agent_names is None:
        stmt = select(Agent.name).order_by(Agent.name)
        agent_names = list((await session.execute(stmt)).scalars().all())

    # Get recent executions for each agent
    # ... aggregate scores per test
    pass
```

**Traces to:** [DESIGN-001], [DESIGN-002]

---

## 4. Frontend Integration

### DESIGN-005: React Component Integration

#### New Routes
```typescript
// Add to dashboard routes
const routes = [
  // ... existing routes
  {
    path: '/compare/:suiteId/:testId',
    component: SideBySideComparison,
  },
  {
    path: '/leaderboard/:suiteId',
    component: LeaderboardMatrix,
  },
  {
    path: '/timeline/:testExecutionId',
    component: EventTimeline,
  },
];
```

#### State Management
```typescript
interface ComparisonState {
  selectedAgents: string[];
  selectedSuite: string | null;
  selectedTest: string | null;
  eventFilters: Set<EventType>;
  timelineZoom: number;
}
```

#### API Hooks
```typescript
// Custom hooks for data fetching
function useComparison(suite: string, test: string, agents: string[]) {
  return useQuery(['comparison', suite, test, agents], () =>
    api.get('/compare/side-by-side', { params: { suite_name: suite, test_id: test, agents } })
  );
}

function useLeaderboard(suite: string) {
  return useQuery(['leaderboard', suite], () =>
    api.get('/leaderboard/matrix', { params: { suite_name: suite } })
  );
}

function useTimeline(testExecutionId: number, runNumber: number) {
  return useQuery(['timeline', testExecutionId, runNumber], () =>
    api.get('/timeline/events', { params: { test_execution_id: testExecutionId, run_number: runNumber } })
  );
}
```

**Traces to:** [REQ-001], [REQ-010], [REQ-020]

---

## 5. Key Design Decisions

### ADR-001: Event Data Structure
**Decision:** Use existing `events_json` column in RunResult model.

**Context:** Events are already stored as JSON array in RunResult.

**Rationale:**
- No database migration needed
- JSON allows flexible event structure
- Query performance acceptable for typical event counts (<1000)

**Trade-offs:**
- Cannot query individual events efficiently
- JSON parsing overhead on large event sets

---

### ADR-002: Comparison Limit
**Decision:** Limit comparison to 3 agents maximum.

**Context:** UI layout becomes unwieldy with more agents.

**Rationale:**
- 3 columns fit well on standard screens (1920px)
- Side-by-side comparison most useful for 2-3 options
- Users can run multiple comparisons if needed

---

### ADR-003: Timeline Rendering
**Decision:** Use Canvas-based timeline with Chart.js.

**Context:** Need smooth rendering for 500+ events.

**Rationale:**
- Chart.js already in project
- Canvas performs better than SVG for many elements
- Built-in zoom/pan support

**Trade-offs:**
- Less flexibility than custom D3.js solution
- Accessibility limitations with Canvas

---

## 6. Directory Structure

```
atp/dashboard/
├── api.py                    # Add new endpoints
├── schemas.py                # Add new response models
├── comparison/               # New module
│   ├── __init__.py
│   ├── queries.py            # Database queries
│   ├── services.py           # Business logic
│   └── models.py             # Internal data models
└── templates/
    └── dashboard.html        # Update with new components
```

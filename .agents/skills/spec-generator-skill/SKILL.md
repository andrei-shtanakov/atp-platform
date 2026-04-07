# Spec Generator Skill

## Overview

Skill for creating project specifications in Kiro-style format: three linked documents with requirements-to-tasks traceability.

## When to Use

- Starting a new project — creating a full specification
- Documenting an existing project
- Requests like: "create a spec", "write requirements", "document the project"

## Structure

The specification consists of 3 core files in the `spec/` directory, with optional phase-specific documents as the project grows:

```
project/
└── spec/
    ├── requirements.md           # Phase 1: WHAT we do
    ├── design.md                 # Phase 1: HOW we do it
    ├── tasks.md                  # Phase 1: WHEN we do it
    │
    ├── phase2-requirements.md    # Phase 2: new requirements
    ├── phase2-design.md          # Phase 2: new/extended design
    ├── phase2-tasks.md           # Phase 2: new tasks
    │
    ├── phase3-requirements.md    # Phase 3: ...
    ├── phase3-design.md
    └── phase3-tasks.md
```

The initial `requirements.md`, `design.md`, `tasks.md` cover Phase 1 (MVP). As the project evolves, each new phase gets its own set of 3 documents.

See [Phases](#phases) for details.

## Files

### 1. requirements.md

**Contains:**
- Project context and goals
- Stakeholders
- Out of Scope (explicitly!)
- Functional requirements (REQ-XXX) in User Story + GIVEN-WHEN-THEN format
- Non-functional requirements (NFR-XXX)
- Constraints and tech stack
- Acceptance criteria by milestones

**Requirement format:**
```markdown
#### REQ-001: Title
**As a** <role>
**I want** <action>
**So that** <value>

**Acceptance Criteria:**
\```gherkin
GIVEN <precondition>
WHEN <action>
THEN <result>
AND <additional result>
\```

**Priority:** P0 | P1 | P2 | P3
**Traces to:** [TASK-XXX], [DESIGN-XXX]
```

### 2. design.md

**Contains:**
- Architectural principles
- High-level diagram (ASCII)
- System components (DESIGN-XXX)
- APIs and interfaces
- Data schemas
- Key decisions (ADR)
- Directory structure

**Component format:**
```markdown
### DESIGN-001: Component Name

#### Description
...

#### Interface
\```python
class Component(ABC):
    @abstractmethod
    def method(self, param: Type) -> ReturnType:
        pass
\```

#### Configuration
\```yaml
component:
  option: value
\```

**Traces to:** [REQ-XXX]
```

### 3. tasks.md

**Contains:**
- Priority and status legend
- Tasks (TASK-XXX) with checklists
- Dependencies between tasks
- Traceability to requirements
- Dependency graph
- Summary by milestones

**Task format:**
```markdown
### TASK-001: Title
🔴 P0 | ⬜ TODO | Est: 3d

**Description:**
Brief description of the task.

**Checklist:**
- [ ] Subtask 1
- [ ] Subtask 2
- [ ] Subtask 3

**Traces to:** [REQ-XXX], [REQ-YYY]
**Depends on:** [TASK-ZZZ]
**Blocks:** [TASK-AAA]
```

## Phases

Projects evolve in phases. Each phase is a self-contained increment with its own requirements, design, and tasks.

### When to Create a New Phase

- **Initial project setup** → Phase 1 (MVP) uses the base 3 files
- **Project grows significantly** → new phase when adding a major capability that needs its own requirements, design decisions, and task breakdown
- **Existing spec is complete** → all Phase N tasks are DONE or in progress, and new work doesn't fit existing milestones

### Phase Convention

| Aspect | Convention |
|--------|-----------|
| Files | `phaseN-requirements.md`, `phaseN-design.md`, `phaseN-tasks.md` |
| Task IDs | Phase 1: TASK-0xx, Phase 2: TASK-1xx..2xx, Phase 3: TASK-3xx..4xx, etc. Or use milestone-based ranges (M1: 001-099, M2: 101-199) |
| Requirement IDs | Phase 1: REQ-0xx, Phase 2: use prefix (e.g., `GE-FR-001` for a sub-package, `AG-FR-001` for a plugin) |
| Design IDs | Phase 1: DESIGN-0xx, Phase N: DESIGN-P{N}-0xx |
| ADRs | Cumulative in `docs/adr/`, numbered sequentially across phases |
| Milestones | Continuous numbering: Phase 1 = M1-M3, Phase 2 = M4-M6, Phase 3 = M7-M9, etc. |

### Phase Document Structure

Each phase document follows the same format as the base templates but:

1. **References the base spec** — "Per ADR-002", "Extends DESIGN-003"
2. **Has its own scope** — In Scope / Out of Scope specific to this phase
3. **Has its own milestones** — Not reusing Phase 1 milestones
4. **Cross-references previous phases** — "Depends on TASK-012 (Phase 1)"
5. **Has its own exit criteria** — When is this phase done?

### Phase vs Milestone

| | Phase | Milestone |
|---|---|---|
| **Scope** | Major capability / project evolution | Grouping of related tasks within a phase |
| **Documents** | Own requirements + design + tasks | Section within tasks.md |
| **Size** | Weeks to months | Days to weeks |
| **Example** | Phase 5: Game-Theoretic Evaluation | M9: game-environments Core |

A phase contains 2-4 milestones. A milestone contains 3-10 tasks.

### Roadmap & ADRs

For multi-phase projects, also maintain:
- `docs/07-roadmap.md` — overview of all phases with status (✅/📋)
- `docs/adr/` — Architecture Decision Records, numbered across phases

The roadmap is the "bird's eye view"; phase documents are the detailed specs.

### Example: Adding Phase 5 to ATP Platform

```
atp-platform/
├── spec/
│   ├── requirements.md              # Phase 4 (Growth)
│   ├── design.md                     # Phase 4
│   ├── tasks.md                      # Phase 4: TASK-101..808
│   ├── phase5-requirements.md        # Phase 5 (Game-Theoretic)
│   ├── phase5-design.md              # Phase 5
│   └── phase5-tasks.md               # Phase 5: TASK-901..922
├── docs/
│   ├── 07-roadmap.md                 # All phases overview
│   └── adr/
│       ├── 001-framework-agnostic.md # Phase 1 decision
│       └── 002-game-theoretic.md     # Phase 5 decision
```

---

## Traceability

The key feature is the linkage between documents:

```
REQ-001 ──────► DESIGN-001
    │               │
    │               ▼
    └─────────► TASK-001
```

- Each requirement references design and tasks
- Each design references requirements
- Each task references requirements and design
- Use the format `[REQ-XXX]`, `[DESIGN-XXX]`, `[TASK-XXX]`

## Priorities

| Emoji | Code | Description |
|-------|------|-------------|
| 🔴 | P0 | Critical — blocks the release |
| 🟠 | P1 | High — needed for full usability |
| 🟡 | P2 | Medium — experience improvement |
| 🟢 | P3 | Low — nice to have |

## Statuses

| Emoji | Status | Description |
|-------|--------|-------------|
| ⬜ | TODO | Not started |
| 🔄 | IN PROGRESS | In progress |
| ✅ | DONE | Completed |
| ⏸️ | BLOCKED | Blocked |

## Creation Process

### New Project (Phase 1)

1. **Gather context:**
   - What problem are we solving?
   - Who are the users?
   - What are the constraints?

2. **Start with requirements.md:**
   - Goals and success metrics
   - Out of scope (important!)
   - Requirements in user story format
   - Acceptance criteria in GIVEN-WHEN-THEN

3. **Then design.md:**
   - Architecture derived from requirements
   - Components and interfaces
   - ADRs for key decisions
   - References to requirements

4. **Finish with tasks.md:**
   - Decompose design into tasks
   - Dependencies between tasks
   - Estimates and priorities
   - Milestones

### Adding a Phase (Phase N, N ≥ 2)

1. **Review existing spec:**
   - Read current requirements, design, tasks
   - Identify what's already done vs planned
   - Understand existing architecture and constraints

2. **Create ADR (if architectural change):**
   - Write `docs/adr/NNN-decision-title.md`
   - Document context, decision, consequences, alternatives

3. **Create phase requirements** (`phaseN-requirements.md`):
   - Scope: what's IN and OUT for this phase
   - New functional requirements with new ID prefix
   - New NFRs or updates to existing
   - Traceability to new tasks
   - Phase exit criteria

4. **Create phase design** (`phaseN-design.md`):
   - Reference existing architecture (don't repeat)
   - New components and their interfaces
   - Integration points with existing system
   - Data models for new functionality
   - Data flow diagrams

5. **Create phase tasks** (`phaseN-tasks.md`):
   - New milestone numbers (continue from previous phase)
   - Tasks with new ID range
   - Dependencies (can reference tasks from previous phases)
   - Dependency graph, critical path, summary
   - Recommended execution order

6. **Update roadmap** (`docs/07-roadmap.md`):
   - Add new phase with deliverables
   - Update timeline
   - Update exit criteria

## Templates

File templates are located in `templates/`:
- `requirements.template.md` — requirements template (Phase 1 / new project)
- `design.template.md` — design template (Phase 1 / new project)
- `tasks.template.md` — tasks template (Phase 1 / new project)
- `phase-requirements.template.md` — phase requirements template (Phase N, N ≥ 2)
- `phase-design.template.md` — phase design template (Phase N, N ≥ 2)
- `phase-tasks.template.md` — phase tasks template (Phase N, N ≥ 2)
- `workflow.template.md` — workflow guide
- `task.py` — CLI for task management
- `executor.py` — auto-execution via Codex CLI
- `executor.config.yaml` — executor configuration
- `Makefile.template` — Make targets for the project

## Examples

See examples in `examples/`:
- `atp-platform/` — Agent Test Platform

## Task Management

The specification includes a CLI for task management:

```bash
# === Manual mode ===
python task.py list              # List tasks
python task.py next              # Next tasks
python task.py start TASK-001    # Start
python task.py done TASK-001     # Complete

# === Automatic mode (Codex CLI) ===
python executor.py run           # Execute the next task
python executor.py run --all     # Execute all ready tasks
python executor.py status        # Status
python executor.py retry TASK-001
```

**Automatic execution:**
- Generates a prompt from spec/* for Codex
- Runs `Codex -p "<prompt>"`
- Validates the result (tests, lint)
- On failure — retry with a limit
- Protection: max_retries=3, max_consecutive_failures=2

A `Makefile` is also created with targets:
- `make exec` — execute the next task
- `make exec-all` — execute all ready tasks
- `make exec-mvp` — MVP milestone only

More details in `spec/WORKFLOW.md`.

## TASK-000: Project Scaffolding

**IMPORTANT:** When creating a specification for a **new project** (not an existing one), always add TASK-000 as the first task. This task blocks all others.

Use the checklist from the appropriate [Language Profile](#language-profiles) (Python or Rust).

```markdown
### TASK-000: Project Scaffolding
🔴 P0 | ⬜ TODO | Est: 1h

**Description:**
Initialize the project structure: directories, configuration, dependencies.
**Language:** {{Python | Rust}}

**Checklist:**
{{Use checklist from the Language Profile section}}

**Traces to:** —
**Depends on:** —
**Blocks:** [TASK-001], [TASK-002], ...
```

**When TASK-000 is NOT needed:**
- The project already exists (has `pyproject.toml` / `Cargo.toml`, src/, etc.)
- Documenting existing code
- Adding a feature to an existing project

## Language Profiles

The spec methodology is language-agnostic, but scaffolding, tooling, and code examples differ by language. Use the appropriate profile when generating specs.

### Python Profile

| Aspect | Convention |
|--------|-----------|
| Project init | `pyproject.toml` + `uv sync` |
| Package manager | `uv` (preferred) or `pip` |
| Directory layout | `src/{{package}}/` or flat `{{package}}/` |
| Test framework | `pytest` + `pytest-cov` + `anyio` (`pytest-anyio`) |
| Coverage tool | `coverage` / `pytest-cov` (report: `--cov --cov-report=term-missing`) |
| Linting | `ruff check` + `ruff format` |
| Type checking | `pyrefly check` (preferred) or `mypy --strict` |
| CI pipeline | `uv sync` → `ruff check` → `pyrefly check` → `pytest --cov` |
| Pre-commit | `ruff`, `pyrefly`, `pytest` |
| Publishing | PyPI via `uv publish` or `twine` |
| Docs | Sphinx or mkdocs-material |
| Virtual env | `uv venv` / `.venv/` |

**TASK-000 checklist (Python):**
```markdown
- [ ] Create `pyproject.toml` with `[project]`, `[build-system]`, `[tool.pytest]`, `[tool.ruff]`
- [ ] Run `uv sync` to create venv and install dependencies
- [ ] Create `src/{{package}}/__init__.py` or `{{package}}/__init__.py`
- [ ] Create `tests/conftest.py`
- [ ] Create `.gitignore` (Python template)
- [ ] Set up `ruff` + `pyrefly` config
- [ ] Initialize git repository
```

**Design code examples (Python):**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

class Component(ABC):
    @abstractmethod
    def process(self, input: Input) -> Output: ...

@dataclass
class Config:
    name: str
    options: dict[str, Any] = field(default_factory=dict)
```

---

### Rust Profile

| Aspect | Convention |
|--------|-----------|
| Project init | `cargo init` / `cargo new` |
| Package manager | `cargo` |
| Directory layout | `src/lib.rs` + `src/bin/` or workspace `Cargo.toml` |
| Test framework | built-in `#[cfg(test)]` + `#[test]`, integration tests in `tests/` |
| Coverage tool | `cargo-tarpaulin` or `cargo-llvm-cov` |
| Linting | `cargo clippy -- -D warnings` |
| Formatting | `cargo fmt --check` |
| CI pipeline | `cargo fmt --check` → `cargo clippy` → `cargo test` → `cargo tarpaulin` |
| Pre-commit | `cargo fmt`, `cargo clippy`, `cargo test` |
| Publishing | crates.io via `cargo publish` |
| Docs | `cargo doc --no-deps`, doc comments `///` |
| Workspace | `Cargo.toml` with `[workspace]` for multi-crate projects |

**TASK-000 checklist (Rust):**
```markdown
- [ ] Run `cargo init` or `cargo new {{project}}`
- [ ] Configure `Cargo.toml`: `[package]`, `[dependencies]`, `[dev-dependencies]`
- [ ] Set up workspace `Cargo.toml` (if multi-crate)
- [ ] Create `src/lib.rs` with module structure
- [ ] Create `tests/` directory for integration tests
- [ ] Create `.gitignore` (Rust template)
- [ ] Configure `clippy.toml` / `rustfmt.toml` if needed
- [ ] Initialize git repository
```

**Design code examples (Rust):**
```rust
use std::error::Error;

pub trait Component: Send + Sync {
    fn process(&self, input: &Input) -> Result<Output, Box<dyn Error>>;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub name: String,
    #[serde(default)]
    pub options: HashMap<String, Value>,
}

impl Default for Config {
    fn default() -> Self {
        Self { name: String::new(), options: HashMap::new() }
    }
}
```

---

### Choosing a Profile

When creating a spec, determine the language from:
1. Existing project files (`pyproject.toml` → Python, `Cargo.toml` → Rust)
2. User's explicit request
3. Ask if ambiguous

For **multi-language projects** (e.g., Python package with Rust extensions via PyO3/maturin), use the primary language profile and note the secondary in Constraints.

---

## Best Practices

1. **Out of Scope is mandatory** — explicitly state what is NOT included in the project
2. **Acceptance criteria are concrete** — GIVEN-WHEN-THEN, not abstractions
3. **Traceability is complete** — every requirement is linked to tasks
4. **Priorities are honest** — not everything is P0, distribute realistically
5. **Estimates are approximate** — a range (3-5d) is better than an exact number
6. **ADR for important decisions** — document "why", not just "what"
7. **Dependency graph** — visualize task dependencies
8. **Tests in every task** — Definition of Done includes unit tests
9. **NFR for testing** — a coverage requirement is mandatory
10. **Test tasks first** — TASK-100 (Test Infrastructure) blocks the rest

# Tasks Specification

> TestGenerator — Implementation Tasks for Test Suite Generation

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

## Milestone 1: Core Engine

### TASK-001: TestGenerator Core Class
🔴 P0 | ✅ DONE | Est: 3-4h

**Description:**
Implement the core TestGenerator class that provides the foundation for all test generation interfaces.

**Checklist:**
- [x] Create `atp/generator/__init__.py`
- [x] Create `atp/generator/core.py` with `TestGenerator` class
- [x] Implement `create_suite()` method
- [x] Implement `add_agent()` method
- [x] Implement `create_custom_test()` method
- [x] Implement `add_test()` method with duplicate ID validation
- [x] Implement `generate_test_id()` method
- [x] Write unit tests in `tests/unit/generator/test_core.py`

**Traces to:** [ARCH-001]
**Depends on:** -
**Blocks:** [TASK-002], [TASK-003], [TASK-004]

---

### TASK-002: Test Templates System
🔴 P0 | ✅ DONE | Est: 2-3h

**Description:**
Implement the template system for predefined test patterns.

**Checklist:**
- [x] Create `atp/generator/templates.py`
- [x] Define `TestTemplate` dataclass with fields: name, description, category, task_template, default_constraints, default_assertions, tags
- [x] Create built-in templates: `file_creation`, `data_processing`, `web_research`, `code_generation`
- [x] Implement `create_test_from_template()` method in TestGenerator
- [x] Implement variable substitution in task_template and assertions
- [x] Add `register_template()` method for custom templates
- [x] Write unit tests for template creation and substitution

**Traces to:** [ARCH-001]
**Depends on:** [TASK-001]
**Blocks:** [TASK-004], [TASK-005]

---

### TASK-003: YAML Writer
🔴 P0 | ✅ DONE | Est: 2h

**Description:**
Implement serialization of test suites to YAML format.

**Checklist:**
- [x] Create `atp/generator/writer.py`
- [x] Use `ruamel.yaml` for YAML generation with proper formatting
- [x] Implement `to_yaml()` method in TestGenerator
- [x] Implement `save()` method to write to file
- [x] Preserve proper indentation (mapping=2, sequence=4, offset=2)
- [x] Exclude None and unset values from output
- [x] Write unit tests for YAML output format

**Traces to:** [ARCH-001]
**Depends on:** [TASK-001]
**Blocks:** [TASK-004]

---

## Milestone 2: CLI Wizard

### TASK-004: CLI Init Command
🔴 P0 | ✅ DONE | Est: 3-4h

**Description:**
Implement `atp init` command for interactive test suite creation.

**Checklist:**
- [x] Create `atp/cli/commands/init.py`
- [x] Implement `init_command` with Click decorators
- [x] Add interactive prompts for: suite name, description, runs_per_test, timeout
- [x] Add agent configuration wizard (http, cli, container types)
- [x] Add test creation wizard with template/custom choice
- [x] Implement `--interactive/--no-interactive` flag
- [x] Register command in `atp/cli/main.py`
- [x] Write integration tests for init command

**Traces to:** [ARCH-002]
**Depends on:** [TASK-001], [TASK-002], [TASK-003]
**Blocks:** [TASK-005]

---

### TASK-005: CLI Generate Command
🟠 P1 | ⬜ TODO | Est: 2-3h

**Description:**
Implement `atp generate` command for adding tests to existing suites.

**Checklist:**
- [ ] Create `atp/cli/commands/generate.py`
- [ ] Implement `generate_command` with Click decorators
- [ ] Add `--suite` option to specify existing suite file
- [ ] Add `--template` option for template-based generation
- [ ] Load existing suite, add test, save back
- [ ] Implement `atp generate test` subcommand
- [ ] Implement `atp generate suite` subcommand for batch creation
- [ ] Register command in `atp/cli/main.py`
- [ ] Write integration tests

**Traces to:** [ARCH-002]
**Depends on:** [TASK-004]
**Blocks:** -

---

## Milestone 3: TUI Interface (Optional)

### TASK-006: TUI Application Setup
🟡 P2 | ✅ DONE | Est: 2-3h

**Description:**
Set up the TUI application using Textual framework.

**Checklist:**
- [x] Add optional dependencies: `textual>=0.47.0`, `rich>=13.0` to pyproject.toml
- [x] Create `atp/tui/__init__.py`
- [x] Create `atp/tui/app.py` with `ATPTUI` class
- [x] Define CSS styles for panels layout
- [x] Create basic screen navigation structure
- [x] Add `atp tui` command to CLI
- [x] Test TUI launch and exit

**Traces to:** [ARCH-003]
**Depends on:** [TASK-001]
**Blocks:** [TASK-007], [TASK-008]

---

### TASK-007: TUI Main Screen
🟡 P2 | ⬜ TODO | Est: 3-4h

**Description:**
Implement the main TUI screen with tree view and YAML preview.

**Checklist:**
- [ ] Create `atp/tui/screens/main_menu.py` with `MainScreen` class
- [ ] Create `atp/tui/widgets/test_tree.py` with `TestTreeWidget`
- [ ] Create `atp/tui/widgets/yaml_preview.py` with `YAMLPreviewWidget`
- [ ] Implement left panel (40%) with test tree
- [ ] Implement right panel (60%) with YAML preview
- [ ] Add keyboard bindings: n=new, o=open, s=save, a=add test, q=quit
- [ ] Update display when suite changes
- [ ] Write tests for widget behavior

**Traces to:** [ARCH-003]
**Depends on:** [TASK-006]
**Blocks:** [TASK-008]

---

### TASK-008: TUI Editor Screens
🟡 P2 | ⬜ TODO | Est: 3-4h

**Description:**
Implement suite and test editor screens.

**Checklist:**
- [ ] Create `atp/tui/screens/suite_editor.py` with `NewSuiteScreen`
- [ ] Create `atp/tui/screens/test_editor.py` with `AddTestScreen`
- [ ] Add input fields for all suite properties
- [ ] Add input fields for all test properties
- [ ] Implement form validation
- [ ] Add Create/Cancel buttons with proper navigation
- [ ] Wire up callbacks to update main screen
- [ ] Write tests for form submission

**Traces to:** [ARCH-003]
**Depends on:** [TASK-007]
**Blocks:** -

---

## Milestone 4: Dashboard Extension

### TASK-009: Dashboard API Endpoints
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Add API endpoints for test suite management in the dashboard.

**Checklist:**
- [ ] Add Pydantic schemas: `SuiteCreateRequest`, `TestCreateRequest`, `TemplateResponse`
- [ ] Implement `POST /suites` endpoint for suite creation
- [ ] Implement `POST /suites/{suite_id}/tests` endpoint for adding tests
- [ ] Implement `GET /templates` endpoint for listing templates
- [ ] Implement `GET /suites/{suite_id}/yaml` endpoint for YAML export
- [ ] Add authentication requirements
- [ ] Write unit tests for endpoints
- [ ] Write integration tests

**Traces to:** [ARCH-004]
**Depends on:** [TASK-001], [TASK-002]
**Blocks:** [TASK-010]

---

### TASK-010: Dashboard UI Components
🟠 P1 | ⬜ TODO | Est: 4-5h

**Description:**
Add React components for test suite creation in the dashboard.

**Checklist:**
- [ ] Create `TestCreatorForm` component with multi-step wizard
- [ ] Step 1: Suite details (name, description, defaults)
- [ ] Step 2: Template selection and test list
- [ ] Step 3: YAML preview and save
- [ ] Add template cards with category badges
- [ ] Add test list management (add/remove)
- [ ] Wire up API calls for creation
- [ ] Add loading and error states
- [ ] Test responsive layout

**Traces to:** [ARCH-004]
**Depends on:** [TASK-009]
**Blocks:** -

---

## Dependency Graph

```
TASK-001 (Core Class) ✅
    │
    ├──► TASK-002 (Templates) ✅
    │        │
    │        └──► TASK-004 (CLI Init) ✅
    │                 │
    │                 └──► TASK-005 (CLI Generate)
    │
    ├──► TASK-003 (YAML Writer) ✅
    │        │
    │        └──► TASK-004 (CLI Init) ✅
    │
    ├──► TASK-006 (TUI Setup) ✅
    │        │
    │        └──► TASK-007 (TUI Main)
    │                 │
    │                 └──► TASK-008 (TUI Editors)
    │
    └──► TASK-009 (Dashboard API)
             │
             └──► TASK-010 (Dashboard UI)
```

---

## Summary

| Milestone | Tasks | Total Est. |
|-----------|-------|------------|
| Core Engine | TASK-001 to TASK-003 | ~7-9h |
| CLI Wizard | TASK-004 to TASK-005 | ~5-7h |
| TUI Interface | TASK-006 to TASK-008 | ~8-11h |
| Dashboard | TASK-009 to TASK-010 | ~7-9h |
| **Total** | 10 tasks | ~27-36h |

### Ready to Start
- [TASK-005] CLI Generate Command (TASK-004 done)
- [TASK-007] TUI Main Screen (TASK-006 done)
- [TASK-009] Dashboard API Endpoints (TASK-001, TASK-002 done)

### Critical Path
TASK-001 ✅ → TASK-002 ✅ → TASK-004 ✅ → TASK-005 (CLI)
TASK-001 ✅ → TASK-003 ✅ → TASK-004 ✅ (YAML)

### Recommended Order
1. ✅ TASK-001 (Core) — foundation for everything
2. ✅ TASK-002 (Templates) + ✅ TASK-003 (Writer) — can be parallel
3. ✅ TASK-004 (CLI Init) — first user-facing feature
4. TASK-005 (CLI Generate) — extends CLI
5. TASK-009 + TASK-010 (Dashboard) — web interface
6. ✅ TASK-006 → TASK-007 → TASK-008 (TUI) — optional, lower priority

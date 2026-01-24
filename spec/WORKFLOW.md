# Task Management Workflow

## Overview

The task management system works directly with the `spec/tasks.md` file:
- Statuses and checklists are updated in markdown
- Change history is logged in `.task-history.log`
- Dependencies are tracked automatically
- **Automatic execution via Claude CLI**

## Quick Start

```bash
# === Manual Mode ===
make task-stats           # Statistics
make task-next            # What to do next
make task-start ID=TASK-001
make task-done ID=TASK-001

# === Automatic Mode (Claude CLI) ===
make exec                 # Execute next task
make exec-all             # Execute all ready tasks
make exec-mvp             # Execute MVP tasks
make exec-status          # Execution status
```

---

## Automatic Execution (Claude CLI)

### Concept

Executor runs Claude CLI for each task:
1. Reads specification (requirements.md, design.md)
2. Forms prompt with task context
3. Claude implements code and tests
4. Checks result (tests, lint)
5. On success — moves to next task
6. On failure — retry with limit

### Commands

```bash
# Execute next ready task
python executor.py run

# Execute specific task
python executor.py run --task=TASK-001

# Execute all ready tasks
python executor.py run --all

# Only MVP tasks
python executor.py run --all --milestone=mvp

# Execution status
python executor.py status

# Retry failed task
python executor.py retry TASK-001

# View logs
python executor.py logs TASK-001

# Reset state
python executor.py reset
```

### Options

```bash
# Number of attempts (default: 3)
python executor.py run --max-retries=5

# Timeout in minutes (default: 30)
python executor.py run --timeout=60

# Without tests after execution
python executor.py run --no-tests

# Without creating git branch
python executor.py run --no-branch

# Auto-commit on success
python executor.py run --auto-commit
```

### Automatic Execution Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                     executor.py run                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Find next task (by priority + dependencies)             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Pre-start hook                                          │
│     - Create git branch: task/TASK-XXX-name                │
│     - Update status: TODO → IN_PROGRESS                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Form prompt                                             │
│     - Context from requirements.md, design.md               │
│     - Task checklist                                        │
│     - Related REQ-XXX, DESIGN-XXX                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Run Claude CLI                                          │
│     claude -p "<prompt>"                                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Check result                                            │
│     - Claude returned "TASK_COMPLETE"?                      │
│     - Tests pass? (make test)                               │
│     - Lint clean? (make lint)                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
     ┌──────────┐        ┌──────────┐
     │ SUCCESS  │        │  FAILED  │
     └────┬─────┘        └────┬─────┘
          │                   │
          ▼                   ▼
┌─────────────────┐   ┌─────────────────┐
│ Post-done hook  │   │ Retry?          │
│ - Auto-commit   │   │ attempts < max  │
│ - Mark DONE     │   └────────┬────────┘
│ - Next task     │            │
└─────────────────┘   ┌────────┴────────┐
                      │                 │
                      ▼                 ▼
               ┌──────────┐      ┌──────────┐
               │  RETRY   │      │   STOP   │
               │ (loop)   │      │ BLOCKED  │
               └──────────┘      └──────────┘
```

### Protection Mechanisms

| Mechanism | Default | Description |
|-----------|---------|-------------|
| max_retries | 3 | Max attempts per task |
| max_consecutive_failures | 2 | Stop after N consecutive failures |
| task_timeout | 30 min | Task timeout |
| post_done tests | ON | Test verification |

### Logs

Logs are saved in `spec/.executor-logs/`:

```
spec/.executor-logs/
├── TASK-001-20250122-103000.log
├── TASK-001-20250122-103500.log  # retry
└── TASK-003-20250122-110000.log
```

Log content:
```
=== PROMPT ===
<full prompt for Claude>

=== OUTPUT ===
<Claude response>

=== STDERR ===
<errors if any>

=== RETURN CODE: 0 ===
```

### Configuration

File `executor.config.yaml`:

```yaml
executor:
  max_retries: 3
  task_timeout_minutes: 30

  hooks:
    pre_start:
      create_git_branch: true
    post_done:
      run_tests: true
      auto_commit: false
```

---

## CLI Commands

### Viewing

```bash
# All tasks
python task.py list

# Filtering
python task.py list --status=todo
python task.py list --priority=p0
python task.py list --milestone=mvp

# Task details
python task.py show TASK-001

# Statistics
python task.py stats

# Dependency graph
python task.py graph
```

### Status Management

```bash
# Start work
python task.py start TASK-001

# Start, ignoring dependencies
python task.py start TASK-001 --force

# Complete
python task.py done TASK-001

# Block
python task.py block TASK-001
```

### Checklist

```bash
# Show task with checklist
python task.py show TASK-001

# Mark item (toggle)
python task.py check TASK-001 0   # first item
python task.py check TASK-001 2   # third item
```

## Workflow

### 1. Task Selection

```bash
# See what's ready to work on
python task.py next

# Output:
# 🚀 Next tasks (ready to work):
#
# 1. 🔴 TASK-100: Test Infrastructure Setup
#    Est: 2d | Milestone 1: MVP ✓ deps OK
```

### 2. Starting Work

```bash
# Start task
python task.py start TASK-100

# ✓ TASK-100 started!
```

Status in `tasks.md` updates: `⬜ TODO` → `🔄 IN PROGRESS`

### 3. Working with Checklist

```bash
# View checklist
python task.py show TASK-100

# Mark completed items
python task.py check TASK-100 0
python task.py check TASK-100 1
```

### 4. Completion

```bash
# Complete
python task.py done TASK-100

# ✅ TASK-100 completed!
#
# 🔓 Unblocked tasks:
#    TASK-001: ATP Protocol Models
#    TASK-004: Test Loader
```

### 5. Checking Progress

```bash
python task.py stats

# 📊 Task Statistics
# ==================
#
# By status:
#   ✅ done          3 ████░░░░░░░░░░░░░░░░ 12%
#   🔄 in_progress   1 █░░░░░░░░░░░░░░░░░░░  4%
#   ⬜ todo         21 ████████████████████ 84%
```

## Dependencies

The system automatically tracks dependencies:

- With `task next` — shows only tasks with completed dependencies
- With `task start` — warns about incomplete dependencies
- With `task done` — shows unblocked tasks

```bash
# Attempting to start task with incomplete dependencies
python task.py start TASK-003

# ⚠️  Task depends on incomplete: TASK-001
#    Use --force to start anyway
```

## Export to GitHub Issues

```bash
# Generates commands for gh CLI
python task.py export-gh

# Execute generated commands:
# gh issue create --title "TASK-001: ATP Protocol Models" ...
```

## Git Integration

Recommended workflow with branches:

```bash
# 1. Start task
python task.py start TASK-001
git checkout -b task/TASK-001-protocol-models

# 2. Work...
git commit -m "TASK-001: Add ATPRequest model"

# 3. Complete
python task.py done TASK-001
git checkout main
git merge task/TASK-001-protocol-models
```

## Make Targets

For convenience — targets in Makefile:

| Command | Description |
|---------|-------------|
| `make task-list` | List all tasks |
| `make task-todo` | TODO tasks |
| `make task-progress` | Tasks in progress |
| `make task-stats` | Statistics |
| `make task-next` | Next tasks |
| `make task-graph` | Dependency graph |
| `make task-p0` | Only P0 |
| `make task-mvp` | MVP tasks |
| `make task-start ID=X` | Start task |
| `make task-done ID=X` | Complete task |
| `make task-show ID=X` | Show details |

## Change History

All changes are logged in `spec/.task-history.log`:

```
2025-01-22T10:30:00 | TASK-100 | status -> in_progress
2025-01-22T10:35:00 | TASK-100 | checklist[0] -> done
2025-01-22T11:00:00 | TASK-100 | status -> done
```

## Tips

1. **Start your day with `task next`** — see priority ready tasks
2. **Mark checklist regularly** — progress is immediately visible
3. **Don't force dependencies** — they're there for a reason
4. **Commit tasks.md** — history in Git
5. **Use `--force` consciously** — only when really needed

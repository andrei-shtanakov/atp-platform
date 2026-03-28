# Test Catalog & Marketplace — Design Spec

**Date:** 2026-03-28
**Status:** Approved

## Context

ATP Platform currently stores tests as local YAML files. Users create their own test suites, run them against their agents, and view results in a local dashboard. There is no way to browse a shared catalog of tests, compare your agent against others, or contribute tests for the community.

### Goals

- **Test Catalog**: A hierarchical catalog (category > suite > test) with curated and community-contributed tests, browsable and runnable via CLI
- **Comparison**: After running a test, see your score vs top-3 agents and the average, broken down by metrics (quality, completeness, efficiency, cost)
- **Submission tracking**: Every catalog run creates a submission record linked to the test, enabling aggregate statistics and leaderboards

### Constraints

- Self-hosted first (single-instance deployment), public leaderboard deferred
- CLI-first, Dashboard UI deferred
- Team of 2-3 people, builds on existing infrastructure (TestLoader, TestOrchestrator, ResultStorage)
- No new external dependencies

---

## Architecture

### Hierarchy

```
Category (coding, reasoning, game-theory, security)
  └── Suite (file-operations, logic-puzzles, prisoners-dilemma, prompt-injection)
       └── Test (create-file, syllogism, one-shot, basic-injection)
```

Addressed by slug path: `coding/file-operations/create-file`

### Data Model (4 new tables in dashboard DB)

**CatalogCategory:**
- `id` (PK), `slug` (unique), `name`, `description`, `icon`
- `created_at`, `updated_at`

**CatalogSuite:**
- `id` (PK), `category_id` (FK → CatalogCategory)
- `slug` (unique within category), `name`, `description`
- `author` (str — "curated" or username), `source` (enum: builtin | community)
- `difficulty` (enum: easy | medium | hard)
- `estimated_minutes` (int)
- `tags` (JSON list), `version` (str)
- `suite_yaml` (text — full YAML content)
- `created_at`, `updated_at`

**CatalogTest:**
- `id` (PK), `suite_id` (FK → CatalogSuite)
- `slug` (unique within suite), `name`, `description`
- `task_description` (text — what the agent must do)
- `difficulty` (enum: easy | medium | hard), `tags` (JSON list)
- Aggregate stats (updated after each submission):
  - `total_submissions` (int, default 0)
  - `avg_score` (float, nullable), `best_score` (float, nullable), `median_score` (float, nullable)
- `created_at`, `updated_at`

**CatalogSubmission:**
- `id` (PK), `test_id` (FK → CatalogTest)
- `agent_name` (str), `agent_type` (str)
- `score` (float), `quality_score` (float), `completeness_score` (float), `efficiency_score` (float), `cost_score` (float)
- `total_tokens` (int, nullable), `cost_usd` (float, nullable), `duration_seconds` (float, nullable)
- `suite_execution_id` (FK → SuiteExecution, nullable — for drill-down into existing result data)
- `submitted_at` (datetime)

### Relationship to Existing Tables

```
CatalogSubmission.suite_execution_id → SuiteExecution.id
  └── SuiteExecution → TestExecution → RunResult → Artifact, EvaluationResult, ScoreComponent
```

This enables full drill-down from catalog submission to detailed execution results via the existing ResultStorage infrastructure.

---

## Builtin Test Catalog

Builtin tests ship as YAML files inside the `atp-platform` package:

```
atp/catalog/
  builtin/
    coding/
      file-operations.yaml
    reasoning/
      logic-puzzles.yaml
    game-theory/
      prisoners-dilemma.yaml
    security/
      prompt-injection.yaml
  __init__.py
  models.py
  repository.py
  sync.py
```

### YAML Format

Catalog suites extend the standard TestSuite format with a `catalog:` metadata section:

```yaml
catalog:
  category: coding
  slug: file-operations
  name: "File Operations"
  description: "Tests agent ability to create, read, and transform files"
  author: curated
  source: builtin
  difficulty: easy
  estimated_minutes: 5
  tags: [coding, files, basic]

test_suite: "catalog:coding/file-operations"
version: "1.0"

defaults:
  runs_per_test: 3
  timeout_seconds: 60

tests:
  - id: create-file
    name: "Create a Python file"
    tags: [easy]
    task:
      description: "Create a file called hello.py that prints 'Hello, World!'"
      expected_artifacts: ["hello.py"]
    assertions:
      - type: artifact_exists
        config: { path: "hello.py" }
      - type: code_exec
        config: { command: "python hello.py", expected_output: "Hello, World!" }

  - id: read-and-transform
    name: "Read and transform CSV"
    tags: [medium]
    task:
      description: "Read input.csv, add a 'total' column summing all numeric columns, write to output.csv"
      input_data:
        files: { "input.csv": "name,a,b\nAlice,1,2\nBob,3,4\n" }
      expected_artifacts: ["output.csv"]
    assertions:
      - type: artifact_exists
        config: { path: "output.csv" }
      - type: artifact_contains
        config: { path: "output.csv", content: "total" }

  - id: multi-file-refactor
    name: "Multi-file refactor"
    tags: [hard]
    task:
      description: "Split the given monolithic utils.py into separate modules (strings.py, math_utils.py, io_utils.py) while maintaining all imports"
      input_data:
        files: { "utils.py": "def add(a,b): return a+b\ndef upper(s): return s.upper()\ndef read_file(p): return open(p).read()\n" }
      expected_artifacts: ["strings.py", "math_utils.py", "io_utils.py"]
    assertions:
      - type: artifact_exists
        config: { path: "strings.py" }
      - type: artifact_exists
        config: { path: "math_utils.py" }
      - type: artifact_exists
        config: { path: "io_utils.py" }
```

The `catalog:` section is parsed during sync and stored in DB. The rest is a standard TestSuite that works with existing TestLoader.

### Sync Process

`atp catalog sync` (or auto on first `atp catalog list`):

1. Scan `atp/catalog/builtin/` for YAML files
2. For each file, parse the `catalog:` section
3. Upsert CatalogCategory (by slug)
4. Upsert CatalogSuite (by category + slug)
5. Upsert CatalogTest for each test in the suite (by suite + test id as slug)
6. Idempotent — safe to run multiple times

---

## CLI Interface

### Commands

```
atp catalog list [CATEGORY]             Browse catalog
atp catalog info <path>                 Show suite/test details
atp catalog run <path> --adapter=...    Run tests and submit results
atp catalog sync                        Sync builtin tests to DB
atp catalog publish <file.yaml>         Add community test suite
atp catalog results <path>              Show submission leaderboard
```

### `atp catalog list`

```
$ atp catalog list
Category        Suites  Tests   Submissions
coding          1       3       45
reasoning       1       3       23
game-theory     1       2       67
security        1       2       12

$ atp catalog list coding
Suite               Difficulty  Tests  Avg Score  Submissions
file-operations     easy        3      72.3       45
```

### `atp catalog info <path>`

```
$ atp catalog info coding/file-operations
Name: File Operations
Category: coding
Difficulty: easy
Author: curated
Estimated time: 5 min
Tags: coding, files, basic

Tests:
  create-file           easy    avg: 85.2  best: 98.0  submissions: 45
  read-and-transform    medium  avg: 71.0  best: 93.5  submissions: 38
  multi-file-refactor   hard    avg: 60.8  best: 88.2  submissions: 12
```

### `atp catalog run <path>`

```
$ atp catalog run coding/file-operations --adapter=http --adapter-config url=http://localhost:8000

Running: coding/file-operations (3 tests, 3 runs each)
  create-file           ████████████████████ PASS  92.0
  read-and-transform    ████████████████░░░░ PASS  78.5
  multi-file-refactor   █████████████░░░░░░░ PASS  65.0

Suite Score: 78.5

Comparison:
                    You     #1      #2      #3      Avg
create-file         92.0    98.0    92.0    91.5    72.3
read-and-transform  78.5    93.5    89.2    85.0    58.1
multi-file-refactor 65.0    88.2    82.0    79.5    41.7

Rank: #4 / 45 submissions
```

### `atp catalog run` — execution flow

1. `CatalogRepository.get_suite_by_path("coding/file-operations")`
2. Parse `suite_yaml` via `TestLoader.load_from_string()`
3. Create adapter from CLI args
4. `TestOrchestrator.run(suite, adapter)` — existing runner
5. `ResultStorage.save(suite_result)` — existing storage
6. For each test in suite result:
   - Extract scores from ScoreComponents
   - `CatalogRepository.create_submission(test_id, agent, scores, suite_execution_id)`
   - `CatalogRepository.update_test_stats(test_id)` — recalculate avg/best/median
7. Render comparison table

### `atp catalog publish <file.yaml>`

1. Read and validate YAML file
2. Verify `catalog:` section is present and valid
3. Set `source: community`
4. Upsert into DB (same logic as sync, but for a single file)
5. Print confirmation with catalog path

### `atp catalog results <path>`

```
$ atp catalog results coding/file-operations
Rank  Agent           Score   Quality  Completeness  Efficiency  Cost    Date
#1    gpt-4o          89.2    92.0     88.5          85.0        91.0    2026-03-25
#2    claude-sonnet   87.5    90.0     86.0          84.0        90.0    2026-03-26
#3    my-agent        78.5    82.0     75.0          77.0        80.0    2026-03-28
...
```

---

## Integration with Existing Code

### Key principle

The catalog is a layer ON TOP of existing infrastructure. No changes to:
- TestLoader (parses YAML as usual)
- TestOrchestrator (runs tests as usual)
- ResultStorage (saves results as usual)
- Evaluators (evaluate as usual)
- Adapters (connect to agents as usual)

### New code

| Module | Responsibility |
|--------|---------------|
| `atp/catalog/models.py` | 4 SQLAlchemy models |
| `atp/catalog/repository.py` | CRUD, stats updates, queries |
| `atp/catalog/sync.py` | Builtin YAML → DB sync |
| `atp/catalog/builtin/*.yaml` | 4 builtin suites (10 tests) |
| `atp/cli/commands/catalog.py` | 6 CLI commands |
| `alembic migration` | 4 new tables |

### DB migration

Single Alembic migration adding 4 tables. Does not modify existing tables.

---

## Builtin Test Suites (first version)

| Category | Suite | Tests | Description |
|----------|-------|-------|-------------|
| coding | file-operations | 3 | Create, read/transform, multi-file refactor |
| reasoning | logic-puzzles | 3 | Syllogism, constraint satisfaction, deduction |
| game-theory | prisoners-dilemma | 2 | One-shot PD, 50-round PD with TFT baseline |
| security | prompt-injection | 2 | Basic direct injection, indirect injection |

10 tests total — enough to validate the architecture and demonstrate value.

---

## What's NOT in scope

- Dashboard UI for catalog browsing (deferred — CLI-first)
- Public leaderboard / remote result sharing (deferred — self-hosted first)
- Community test moderation / approval workflow
- Test ratings / likes / popularity sorting
- Automatic difficulty estimation from scores
- Test versioning (currently overwrite by slug)
- Game-theory tests running via game runner (currently standard ATP protocol tests only; game-specific suites remain separate via `atp game` commands)

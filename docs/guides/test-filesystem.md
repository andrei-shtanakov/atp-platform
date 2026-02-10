# Test Filesystem

Pre-populate agent workspace with files before test execution and verify filesystem state after.

## Overview

The test filesystem feature enables:

1. **Workspace fixtures** — copy a directory tree into the agent's workspace before the test starts
2. **Filesystem assertions** — check files, directories, and content in the workspace after the agent finishes
3. **Cross-mode support** — works identically for CLI (host) and Container (Docker/Podman) adapters

This is analogous to the [test website](../../tests/fixtures/test_site/README.md) fixture, but for filesystem-based testing.

## Quick Start

### 1. Create a fixture directory

```
tests/fixtures/test_filesystem/my_project/
  README.md
  src/
    main.py
  data/
    config.json
```

### 2. Reference it in your test suite

```yaml
test_suite: "filesystem_tests"
version: "1.0"

agents:
  - name: "file-agent"
    type: "cli"
    config:
      command: "python"
      args: ["my_agent.py"]

tests:
  - id: "fs-001"
    name: "Read project files"
    task:
      description: "Read README.md and return its content"
      workspace_fixture: "tests/fixtures/test_filesystem/my_project"
    assertions:
      - type: "file_exists"
        config:
          path: "README.md"
      - type: "file_contains"
        config:
          path: "data/config.json"
          pattern: '"version"'
```

### 3. Run the test

```bash
uv run atp test my_suite.yaml --adapter=cli
```

## How It Works

### Execution Flow

```
1. Orchestrator creates sandbox
   → /tmp/atp-sandboxes/sandbox-{UUID}/workspace/

2. Fixture is copied into workspace (shutil.copytree)
   → /tmp/atp-sandboxes/sandbox-{UUID}/workspace/README.md
   → /tmp/atp-sandboxes/sandbox-{UUID}/workspace/src/main.py
   → /tmp/atp-sandboxes/sandbox-{UUID}/workspace/data/config.json

3. Agent receives workspace_path via ATP Request context
   → context.workspace_path = "/tmp/atp-sandboxes/sandbox-{UUID}/workspace"

4. Agent reads/writes files in workspace

5. Evaluator checks filesystem state (if filesystem assertions defined)

6. Sandbox is cleaned up (shutil.rmtree)
```

### Deployment Modes

| Mode | Workspace Access | Notes |
|------|-----------------|-------|
| **CLI (host)** | Direct filesystem path | Agent reads `context.workspace_path` |
| **Container** | Volume mount (`-v host:container`) | Auto-mounted, path rewritten to `/workspace` |

#### CLI Adapter (both on host)

Agent receives the absolute host path in `context.workspace_path` and accesses files directly.

#### Container Adapter (agent in Docker/Podman)

The container adapter automatically:
1. Mounts the host workspace as a volume: `-v /tmp/atp-sandboxes/.../workspace:/workspace`
2. Rewrites `context.workspace_path` to `/workspace` in the ATP Request sent to the container

No manual volume configuration is needed. The agent inside the container sees `/workspace` as its working directory.

## Fixture Structure

### Requirements

- Fixture path must be a **directory** (not a file)
- Can be absolute or relative (resolved from CWD)
- Must exist at test execution time
- Nested directories are fully preserved

### Conventions

Store fixtures under `tests/fixtures/test_filesystem/`:

```
tests/fixtures/test_filesystem/
  basic/                    # Simple files
    readme.txt
    data/
      config.json
  python_project/           # Python project skeleton
    src/
      main.py
      utils.py
    tests/
      test_main.py
    pyproject.toml
  messy_directory/           # For cleanup/organization tasks
    file1.txt
    FILE1.TXT
    .hidden_config
    temp/
      old_log.txt
  data_processing/           # For ETL/analysis tasks
    input.csv
    schema.json
    expected_output.json     # Reference output for comparison
```

### Permissions

Files are copied with default permissions inherited from the source. The sandbox workspace directory itself has `0o700` (owner-only) permissions.

**Important:** If your fixture contains files with special permissions (e.g., executable scripts), those permissions are preserved by `shutil.copytree`.

### Recovery After Agent Modification

Each test run creates a **fresh sandbox**. The fixture directory is never modified — it's always copied into a new temporary workspace. This means:

- Agents can freely create, modify, or delete files in the workspace
- The original fixture is never affected
- The next test run starts with a clean copy
- No manual recovery or reset is needed

This is a key design principle: **fixtures are immutable templates, workspaces are ephemeral copies.**

## Filesystem Assertions

### Available Assertion Types

| Type | Description | Key Config |
|------|-------------|------------|
| `file_exists` | File exists at path | `path` |
| `file_not_exists` | File does NOT exist | `path` |
| `file_contains` | File content matches pattern | `path`, `pattern`, `regex` |
| `dir_exists` | Directory exists | `path` |
| `file_count` | Number of files in directory | `path`, `count`, `operator` |

### file_exists

Check that a file exists in the workspace.

```yaml
assertions:
  - type: "file_exists"
    config:
      path: "output.txt"

  - type: "file_exists"
    config:
      path: "reports/summary.json"
```

### file_not_exists

Check that a file does NOT exist (e.g., agent should have deleted temp files).

```yaml
assertions:
  - type: "file_not_exists"
    config:
      path: "temp/scratch.txt"
```

### file_contains

Check that file content matches a plain text substring or regex pattern.

```yaml
assertions:
  # Plain text
  - type: "file_contains"
    config:
      path: "output.txt"
      pattern: "Processing complete"

  # Regex
  - type: "file_contains"
    config:
      path: "report.json"
      pattern: '"count":\s*\d+'
      regex: true
```

### dir_exists

Check that a directory exists.

```yaml
assertions:
  - type: "dir_exists"
    config:
      path: "output/reports"
```

### file_count

Check the number of files in a directory.

```yaml
assertions:
  # Exact count
  - type: "file_count"
    config:
      path: "output"
      count: 3
      operator: "eq"

  # At least N files
  - type: "file_count"
    config:
      path: "output"
      count: 1
      operator: "gte"
```

**Operators:** `eq` (default), `gt`, `gte`, `lt`, `lte`

### Path Validation

All paths in assertions are validated against the workspace boundary. Path traversal (`../`) and absolute paths are rejected. This prevents an agent (or misconfigured test) from reading files outside the sandbox.

## Analyzing Filesystem Changes

### Current State

The filesystem evaluator checks the **final state** of the workspace — it runs after the agent finishes. It does not track what changed during execution.

### What IS Supported

- **Post-execution state checks** — assert what files exist, their content, directory structure
- **Before-and-after comparison** — the fixture defines "before", assertions define expected "after"
- **Deletion verification** — `file_not_exists` confirms the agent removed a file
- **Creation verification** — `file_exists` + `file_contains` confirms new file content

### What is NOT Yet Implemented

- **Diff/changelog** — no automatic tracking of which files were created, modified, or deleted
- **File modification timestamps** — no assertion on when files were changed
- **Permission assertions** — no checks on file mode/ownership changes
- **Binary file comparison** — `file_contains` works with text files only
- **Recursive content assertions** — no "all files in directory contain X"

### Workarounds for Change Tracking

If you need to know exactly what the agent changed, consider these approaches:

**1. Snapshot comparison (external script)**

```bash
# Before test: save file listing
find workspace/ -type f -exec md5sum {} \; > before.txt

# After test: compare
find workspace/ -type f -exec md5sum {} \; > after.txt
diff before.txt after.txt
```

**2. Behavioral assertions**

Use `must_use_tools` to verify the agent called specific file operations:

```yaml
assertions:
  - type: "behavior"
    config:
      must_use_tools:
        - "file_write"
        - "file_delete"
```

**3. Agent artifacts**

Require the agent to report its changes as a structured artifact:

```yaml
task:
  description: "Clean up the project directory and report changes"
  expected_artifacts: ["changes.json"]
assertions:
  - type: "artifact_exists"
    config:
      path: "changes.json"
  - type: "contains"
    config:
      path: "changes.json"
      pattern: "deleted"
```

## Complete Example

```yaml
test_suite: "file_operations"
version: "1.0"
description: "Tests for filesystem-based agent tasks"

defaults:
  timeout_seconds: 30
  constraints:
    max_steps: 10

agents:
  - name: "file-agent"
    type: "cli"
    config:
      command: "python"
      args: ["examples/demo_agent.py"]

tests:
  - id: "read-config"
    name: "Read and parse JSON config"
    tags: ["filesystem", "read"]
    task:
      description: "Read data/config.json and return the project name"
      workspace_fixture: "tests/fixtures/test_filesystem/basic"
    assertions:
      - type: "file_exists"
        config:
          path: "data/config.json"
      - type: "file_contains"
        config:
          path: "data/config.json"
          pattern: "test-project"
      - type: "dir_exists"
        config:
          path: "data"

  - id: "create-output"
    name: "Create output file alongside fixtures"
    tags: ["filesystem", "write"]
    task:
      description: "Create output.txt with 'Agent was here'"
      workspace_fixture: "tests/fixtures/test_filesystem/basic"
      expected_artifacts: ["output.txt"]
    assertions:
      - type: "file_exists"
        config:
          path: "output.txt"
      - type: "file_contains"
        config:
          path: "output.txt"
          pattern: "Agent was here"
      - type: "file_exists"
        config:
          path: "readme.txt"    # fixture file still present

  - id: "cleanup-temp"
    name: "Agent should remove temp files"
    tags: ["filesystem", "cleanup"]
    task:
      description: "Delete all .tmp files from the workspace"
      workspace_fixture: "tests/fixtures/test_filesystem/messy_directory"
    assertions:
      - type: "file_not_exists"
        config:
          path: "scratch.tmp"
      - type: "file_exists"
        config:
          path: "important.txt"  # should NOT be deleted
```

## Implementation Details

### Key Files

| File | Role |
|------|------|
| `atp/loader/models.py` | `TaskDefinition.workspace_fixture` field |
| `atp/runner/sandbox.py` | `SandboxManager.populate_workspace()` method |
| `atp/runner/orchestrator.py` | Calls `populate_workspace()` before test execution |
| `atp/adapters/container.py` | Auto-mounts workspace, rewrites `context.workspace_path` |
| `atp/evaluators/filesystem.py` | `FilesystemEvaluator` with 5 assertion types |
| `atp/evaluators/registry.py` | Registers filesystem evaluator and assertion mappings |

### Security

- Workspace isolation via sandbox UUID directories (`/tmp/atp-sandboxes/sandbox-{UUID}/`)
- Path traversal prevention via `validate_path_within_workspace()`
- Container volume mounting uses existing security validation
- Fixture directory is never modified (copy-on-use)

## Related Documentation

- [Test Format Reference](../reference/test-format.md) — YAML test suite format
- [Evaluation System](../05-evaluators.md) — all evaluator types
- [Container Setup Guide](container-setup.md) — Docker/Podman configuration
- [Mock Tools Guide](mock-tools.md) — deterministic tool mocking
- [Test Website Fixture](../../tests/fixtures/test_site/README.md) — HTTP-based test fixture

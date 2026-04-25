# Maestro ↔ ATP integration

How Maestro tasks call ATP test suites as their `validation_cmd` (R-06a in the
ecosystem roadmap).

## Role

ATP runs the `validation_cmd` step of a Maestro task. Maestro hands ATP a YAML
test suite, ATP exercises the agent under test, and the **process exit code**
tells Maestro whether to mark the task `validation_passed` or
`validation_failed`. Maestro captures `stdout` and `stderr` for the task log
but does not parse them.

Source of truth on the Maestro side:
[`maestro/validator.py`](https://github.com/andrei-shtanakov/Maestro/blob/main/maestro/validator.py)
— `validate(command, workdir, timeout) → ValidationResult`. It launches the
command via `asyncio.create_subprocess_exec`, waits up to `timeout_seconds`,
captures both streams as UTF-8 (`errors="replace"`), and returns
`success = (exit_code == 0)`.

## Exit-code contract

ATP commits to these exit codes for every command Maestro is expected to call
(`atp test`, `atp run`, `atp validate`). They are unit-tested and part of the
public CLI surface — bumping any of them is a **major** semver change.

| Code | Constant       | Meaning for Maestro                                |
|------|----------------|----------------------------------------------------|
| `0`  | `EXIT_SUCCESS` | All tests passed → task validation passes.         |
| `1`  | `EXIT_FAILURE` | At least one test failed → task validation fails.  |
| `2`  | `EXIT_ERROR`   | Suite/config invalid, file missing, runtime error. |

Maestro treats `1` and `2` identically (anything non-zero is a fail). ATP keeps
them distinct so a human reading the task log can tell "the agent regressed"
apart from "the validation step itself is broken".

Constants are defined at
[`atp/cli/main.py`](../atp/cli/main.py) lines 37-39.

## Recommended `validation_cmd` shapes

ATP exposes two equivalent entry points: `atp test` is the canonical command,
`atp run` is an alias kept around because the verb often reads better in task
specs (`validation_cmd: "atp run …"` parses as action). They accept the same
flags and exit codes; pick whichever fits your task spec — Maestro doesn't
care.

```yaml
# Maestro task definition — minimal
tasks:
  - id: my-task
    validation_cmd: "atp run tests/atp/regression.yaml"
    validation_timeout_seconds: 600
```

```yaml
# With a specific adapter and a results artefact dropped next to the run
tasks:
  - id: my-task
    validation_cmd: >
      atp run tests/atp/regression.yaml
      --adapter=cli
      --output=json
      --output-file=.maestro/atp-results.json
```

```yaml
# Tag-filtered smoke for a fast pre-merge gate (fail-fast)
tasks:
  - id: my-task
    validation_cmd: "atp run tests/atp/all.yaml --tags=smoke --fail-fast"
    validation_timeout_seconds: 120
```

A `--list-only` invocation is a safe dry-run — it exits `0` if the suite parses
and contains tests, and `1` if no tests match the tag filter. Useful as a
cheap "is the suite still valid" smoke before a heavier task.

## Output streams

ATP writes:

- **`stdout`** — human-readable progress plus the selected reporter output.
  `atp test` / `atp run` accept `--output=console|json|junit` with `console`
  as the default. `console` prints the rich terminal view including a compact
  failure summary; `json` writes a single parseable JSON document; `junit`
  writes JUnit XML.
- **`stderr`** — log lines emitted via Python's stdlib `logging`, plus a
  final `Error: ...` line on exit code `2`.

Maestro logs both streams verbatim. If you want Maestro to surface the failed
test list in the task log without spelunking, the default `--output=console`
already includes that summary at the end.

For machine-readable artefacts, use `--output-file=path --output=json` (or
`--output=junit`). The file is written when the suite runs to completion,
including when tests fail (exit code `1`), so Maestro can attach it as a task
artefact on any pass/fail outcome. **It is not guaranteed on exit code `2`
paths** — those are pre-suite-execution errors (config invalid, file missing,
runtime exception during loading) where ATP exits before report generation.

## Working directory and config

`atp run` resolves the suite path relative to the **current working
directory**. Maestro runs `validation_cmd` with `workdir` set to the task's
`workdir` field (defaults to the project root). Place suite files at a stable
path inside the repo and reference them relatively.

ATP looks for `atp.config.yaml` / `atp.config.yml` walking up from the cwd
(`ConfigContext.load_config` in `atp/cli/main.py`) — Maestro does not need to
set anything special. If you want Maestro to override the agent at call time,
prefer CLI flags (`--agent-name`, `--adapter`, `--adapter-config k=v`) over
editing the config file from inside the task.

## Timeouts

Maestro kills the validation process when its `validation_timeout_seconds`
elapses (`asyncio.wait_for` + `proc.kill()`). ATP itself has per-test timeouts
(`timeout_seconds` in the suite definition) — set the Maestro timeout to **at
least** the sum of per-test timeouts plus a small buffer, otherwise ATP gets
killed mid-run and Maestro reports `timed_out=True` with empty stdout/stderr.

Rule of thumb: `maestro_timeout = sum(test.timeout_seconds) * 1.2 + 10s`.

## Stability guarantees (semver)

The following are part of the public CLI surface and bound to ATP's MAJOR
version:

1. The three exit codes above and their meanings.
2. The names `atp run`, `atp test`, `atp validate`, `atp list`, `atp version`.
3. The flags listed in the example shapes (`--adapter`, `--adapter-config`,
   `--output`, `--output-file`, `--tags`, `--list-only`, `--fail-fast`,
   `--agent-name`).

The following may change in MINOR versions — don't depend on them from
Maestro:

- The exact wording of human-readable stdout/stderr.
- The shape of the JSON reporter output (covered by its own contract docs in
  `docs/json-reporter.md` if/when needed — currently versioned via the report's
  embedded `schema_version` field).
- New flags / new commands.

## Quick verification

If you ever want to confirm the contract without running a full suite:

```bash
# Should exit 0
atp run tests/some_suite.yaml --list-only; echo "ok=$?"

# Should exit 2 (file missing)
atp run /no/such/file.yaml; echo "missing=$?"

# Should exit 0
atp version; echo "version=$?"
```

`;` rather than `&&` so the `echo` runs regardless of outcome — the goal here
is to inspect the exit code, not to chain on success.

These three checks are enough to be sure ATP is wired into the Maestro host
correctly before pushing a real `validation_cmd` task.

## See also

- Maestro task model: `maestro/models.py::Task.validation_cmd`
- Maestro validator: `maestro/validator.py::validate`
- Ecosystem roadmap entry: `_cowork_output/roadmap/ecosystem-roadmap.md` →
  R-06a

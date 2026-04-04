# Container Isolation for CodeExecEvaluator

## Summary

Run evaluator commands (pytest, linters, custom) inside Docker/Podman containers instead of bare subprocess. Prevents untrusted code from affecting the host.

## Design

### ContainerRuntime Protocol

```python
class ContainerRuntime(Protocol):
    async def run(self, image: str, command: list[str],
                  workspace: Path, timeout: int,
                  env: dict[str, str]) -> CommandResult: ...
    async def cleanup(self) -> None: ...
```

Two implementations: `DockerRuntime`, `PodmanRuntime`. Both call CLI (`docker run` / `podman run`) via `asyncio.create_subprocess_exec` — no SDK dependencies.

### Workspace: Copy-In + Copy-Out

1. Copy workspace to temp dir
2. Mount temp dir as read-write at `/work` in container
3. Run command in container with `/work` as cwd
4. Read stdout/stderr from process (primary result channel)
5. Temp dir available for artifact extraction after run
6. Cleanup temp dir

### Container Constraints

- `--rm` (auto-remove)
- `--network=none` (no network — user provides pre-built images with deps)
- `--memory=512m`
- `--cpus=1`
- `--read-only` (root FS readonly, /work writable)
- Timeout via `asyncio.wait_for`

### Network Limitation

`--network=none` means `pip install` won't work inside containers. Users must provide images with pre-installed dependencies (`image: "my-registry/agent-env:latest"`). Default `python:3.12-slim` is only for trivial scripts. Documented in config.

### Configuration

Assertion-level:
```yaml
assertions:
  - type: pytest
    config:
      container: true
      image: "python:3.12-slim"  # optional override
```

Global (DashboardConfig or CLI):
```python
container_runtime: str = "auto"  # auto | docker | podman | none
container_default_image: str = "python:3.12-slim"
```

`auto` tries docker first, then podman, then falls back to subprocess + rlimits with warning.

### Fallback

If container runtime unavailable or `container_runtime = "none"` — falls back to current subprocess + rlimits sandbox. Logs warning.

### Files

| File | Action |
|------|--------|
| `atp/evaluators/container.py` | New: protocol, DockerRuntime, PodmanRuntime, auto-detect |
| `atp/evaluators/code_exec.py` | Modify: `_run_command()` uses container runtime when enabled |
| `tests/unit/evaluators/test_container.py` | New: runtime tests with mocked subprocess |

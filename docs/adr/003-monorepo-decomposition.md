# ADR-003: Monorepo Decomposition into Namespace Packages

**Status**: Proposed
**Date**: 2026-03-22
**Context**: ATP Platform has grown to ~95K lines across 24 subpackages. The dashboard alone is 27K lines (29%). Tight coupling between subsystems makes independent development, testing, and deployment difficult.

## Decision

Decompose ATP Platform into 4 packages within a single monorepo, using Python implicit namespace packages (PEP 420) and uv workspaces.

### Package Structure

```
atp-platform/                    # monorepo root
├── pyproject.toml               # atp-platform (runner, evaluators, reporters, cli, sdk, ...)
├── packages/
│   ├── atp-core/                # protocol, core, loader, chaos, cost, scoring, statistics, streaming
│   │   └── pyproject.toml
│   ├── atp-adapters/            # all agent adapters (HTTP, CLI, Container, cloud, MCP, etc.)
│   │   └── pyproject.toml
│   └── atp-dashboard/           # web dashboard, analytics
│       └── pyproject.toml
```

### Dependency Graph

```
atp-core
    ↑
    ├── atp-adapters
    ↑       ↑
    │       │
    atp-platform
        ↑
        │
    atp-dashboard
```

### Namespace Strategy

All packages share the `atp` top-level namespace via implicit namespace packages:
- No `atp/__init__.py` in any package
- Version accessed via `importlib.metadata.version("atp-platform")`
- Each package ships only its specific subdirectories
- All existing `from atp.X import Y` imports continue working

### Phased Rollout

| Phase | Package | Key Refactoring |
|-------|---------|----------------|
| 1 | atp-dashboard | Extract analytics.cost to atp.cost; move dashboard + analytics |
| 2 | atp-core | Remove atp/__init__.py; move protocol, core, loader, chaos, cost, scoring, statistics, streaming |
| 3 | atp-adapters | Move adapters; entry-points migrate to new package |

## Consequences

### Positive
- Independent release cycles (dashboard changes don't require core release)
- Lighter installs (`uv add atp-core` for library-only usage)
- Clearer ownership boundaries for team scaling
- Third-party adapter packages become first-class citizens
- Dashboard deployable as standalone web service

### Negative
- Build/CI complexity increases (4 packages to test, version, publish)
- Cross-package refactoring requires coordinated releases
- Developers must understand namespace packages and uv workspaces

### Risks
- Namespace package tooling issues (some tools assume `__init__.py` exists)
- Transitive dependency conflicts between packages
- Breaking changes during migration if not handled carefully

## Alternatives Considered

1. **Separate top-level packages** (atp_core, atp_dashboard): Would break all imports, massive migration.
2. **Keep monolith, use optional extras only**: Doesn't enable independent releases or team ownership.
3. **Full microservices**: Premature at current scale; adds network/deployment complexity.

## Prerequisites

Before starting decomposition:
- TASK-1303: Complete dashboard v2 migration, remove v1 monolith
- TASK-1306: Implement lazy-loading for adapters
- TASK-1307: Slim down dependencies with optional extras

## References

- [PEP 420 — Implicit Namespace Packages](https://peps.python.org/pep-0420/)
- [uv Workspaces Documentation](https://docs.astral.sh/uv/concepts/workspaces/)
- [Decomposition Plan](../../../.claude/plans/synthetic-frolicking-lynx.md)

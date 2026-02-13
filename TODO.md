# TODO

## Publish sub-packages to PyPI

The ATP platform consists of three packages. Only the main package is published so far.

| Package | PyPI | Status |
|---|---|---|
| `atp-platform` | [atp-platform](https://pypi.org/project/atp-platform/) | Published v1.0.0 |
| `game-environments` | — | Not published |
| `atp-games` | — | Not published |

### Package dependency graph

```
atp-platform              # core platform (standalone)
game-environments         # game theory environments (standalone, no atp dependency)
atp-games                 # plugin bridging game-environments ↔ atp-platform
  └── pydantic
  └── (runtime) atp-platform, game-environments
```

### Steps to publish

1. **`game-environments`** — publish first (no dependencies on atp)
   - Bump version in `game-environments/pyproject.toml`
   - Add PyPI Trusted Publisher for `game-environments` repo/workflow
   - Create workflow or publish manually: `cd game-environments && uv build && uv publish`
   - Tag: `game-environments-v1.0.0`

2. **`atp-games`** — publish after game-environments is on PyPI
   - Add explicit dependencies in `atp-games/pyproject.toml`:
     ```toml
     dependencies = [
         "pydantic>=2.0",
         "atp-platform>=1.0.0",
         "game-environments>=1.0.0",
     ]
     ```
   - Bump version in `atp-games/pyproject.toml`
   - Publish: `cd atp-games && uv build && uv publish`
   - Tag: `atp-games-v1.0.0`

### Full installation for end users

```bash
# Core platform only
uv add atp-platform

# With game-theoretic evaluation
uv add atp-platform atp-games game-environments
```

### CI workflows

- `game-environments`: needs a new `.github/workflows/game-environments-publish.yml`
- `atp-games`: existing `.github/workflows/atp-games-ci.yml` already has a publish job triggered by `atp-games-v*` tags — just needs Trusted Publisher configured on PyPI

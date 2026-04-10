# TODO

## ~~Publish sub-packages to PyPI~~ DONE

All packages published.

| Package | PyPI | Status |
|---|---|---|
| `atp-platform` | [atp-platform](https://pypi.org/project/atp-platform/) | Published v1.0.0 |
| `atp-platform-sdk` | [atp-platform-sdk](https://pypi.org/project/atp-platform-sdk/) | Published v2.0.0 |
| `game-environments` | [game-environments](https://pypi.org/project/game-environments/) | Published v1.0.0 |
| `atp-games` | [atp-games](https://pypi.org/project/atp-games/) | Published v1.0.0 |

### Package dependency graph

```
atp-platform              # core platform (standalone)
atp-platform-sdk          # SDK for benchmark participants
game-environments         # game theory environments (standalone, no atp dependency)
atp-games                 # plugin bridging game-environments ↔ atp-platform
  └── pydantic
  └── (runtime) atp-platform, game-environments
```

### Publishing

CI workflows with Trusted Publisher are configured. To publish a new version:
- Bump version in `pyproject.toml`
- Push a tag: `game-environments-v<version>` or `atp-games-v<version>`

### Full installation for end users

```bash
# Core platform only
uv add atp-platform

# With game-theoretic evaluation
uv add atp-platform atp-games game-environments
```

## Platform API & SDK (atp-sdk)

See full spec: `docs/superpowers/specs/2026-04-02-platform-api-and-sdk-design.md`

### MVP
- [x] Extend atp-dashboard: catalog API + tournament API route groups
- [x] Add GitHub as an OIDC provider in the existing SSO module
- [x] Add Device Flow for CLI login
- [x] New SQLAlchemy models (Benchmark, Run, TaskResult, Tournament, Participant, Round, Action)
- [x] Alembic migration for the new tables
- [x] Cancel endpoint + server-side run timeout (status=partial)
- [x] Benchmark family_tag + parent_id for versioning
- [x] Run.adapter_type for analytics (sdk/http/cli/...)
- [x] Login/Register UI + RBAC seed + auto-admin for the first user
- [x] Create packages/atp-sdk/ — Python SDK for participants (client, benchmark iterator, auth)
- [x] Create SDKAdapter in atp-adapters (asyncio.Event + timeout, pull model as AgentAdapter)
- [x] Sandbox for evaluators on the server (subprocess + timeout + rlimits)
- [x] Publish atp-sdk to PyPI (as atp-platform-sdk)

### Post-MVP
- [x] `?batch=N` for parallel task fetching (SDK v2.0.0)
- [ ] Redis pub/sub for SDKAdapter (replaces asyncio.Event, survives restart)
- [ ] Automatic token tracking in the SDK (wrapper around LLM calls)
- [ ] Event streaming in the SDK (send ATPEvent during execution)
- [ ] Workspace management in the SDK (download/upload artifact files)
- [x] Async API in the SDK — AsyncATPClient + async for task in run (SDK v2.0.0)
- [x] Retry/reconnect on drops in the SDK — exponential backoff + full jitter (SDK v2.0.0)
- [ ] TypeScript SDK
- [ ] WebSocket for real-time tournaments (dashboard infrastructure is already in place)
- [ ] Container isolation for evaluators (Podman/Docker)
- [ ] Federation — a private atp-server
- [ ] Webhooks for CI/CD notifications on run completion
- [ ] Application-level rate limiting
- [ ] Extract atp-protocol as a separate lightweight package (if atp-core becomes too heavy for the SDK)
- [ ] Flesh out the Tournament API (cancel, server-side round timeouts, skipping deadlines)

## Architecture Cleanup (P0 → P2)

### P0 — Critical

- [x] **AuthFlowStateStore**: unified `InMemoryAuthStateStore` for SSO/SAML (auth/state_store.py). `_sso_sessions` and `_saml_sessions` removed.
- [x] **Fix SSO tests**: synced with the current SSOInitRequest API (extra="forbid").
- [x] **allow_shell in CLIAdapter**: fully removed. Shell features are available via `command="sh" args=["-c", "..."]`.

### P1 — Important

- [x] **Shared post-auth service**: `complete_auth()` in auth/post_auth.py — provision user + assign roles + issue token. SSO, SAML, and DeviceFlow routes all use it.
- [x] **Remove `return_url` from SAML**: removed from SAMLInitRequest and session storage.
- [x] **DeviceFlowStore API**: `lookup()` + `lookup_by_user_code()` + `DeviceFlowStatus` constants instead of strings.

### P2 — Improvement

- [x] **Decouple atp-dashboard from atp-platform**: shared result models moved to `atp.core.results`, dashboard depends on atp-core.
- [ ] **Merge SSO/SAML route models**: remove request/response model duplication.
- [ ] **Clean up examples and configs** from the shell mode and older assumptions.

## Dashboard UI

- [ ] **Chart.js in Analytics**: status pie chart, score histogram, per-agent line chart (templates/ui/analytics.html).
- [ ] **Fix UI routes test isolation**: `.value` bug in analytics/home templates, UNIQUE constraint collision.
- [ ] **Benchmark API scoring**: wire up evaluators instead of the naive score (100 if completed else 0).

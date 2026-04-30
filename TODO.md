# TODO

## Ecosystem Roadmap (план от 2026-04-16)

> Стратегический контекст: `../_cowork_output/roadmap/ecosystem-roadmap.md`
> Актуальный статус: `../_cowork_output/status/2026-04-10-status.md`
> **Роль ATP в экосистеме**: валидация задач Maestro (`validation_cmd`) и eval-driven обучение arbiter

### Активные кросс-проектные задачи

- [x] **R-06a: Поддержать Maestro CLI quick win** (effort S) ✅ 2026-04-25
  - Документ написан: [`docs/maestro-integration.md`](docs/maestro-integration.md) —
    exit codes (0/1/2), `atp run` контракт, рекомендованные `validation_cmd` patterns,
    semver-обязательства по флагам.
  - `atp run` экзит-коды verified end-to-end (0=pass, 1=fail, 2=error).

- [x] **R-13: Нормализация guardrails с arbiter** (effort M) ✅ 2026-04-25
  - arbiter-команда написала маппинг: [`../arbiter/docs/guardrails-atp-mapping.md`](../arbiter/docs/guardrails-atp-mapping.md)
    (2026-04-17). Вывод: 0 правил перекрываются семантически, 2 разделяют **концепт**
    (бюджет, время) на разных осях; **не** объединяем структурно (shared types — over-engineering
    для 15 строк), сохраняем разные имена, выравниваем описания.
  - ATP-сторона:
    - Module docstring `atp/evaluators/guardrails.py:1-27` уточнён под фразу "post-execution,
      pre-evaluation gate" + ссылка на mapping (rec #2).
    - `check_timeout_not_exceeded` / `check_within_budget` docstrings проясняют axis
      (measurement vs. estimate, per-test vs. system-wide) — rec #3.
  - Не делаем (по совместному решению):
    - Shared types через FFI / JSON Schema (rec #1, "revisit only if a third project pulls in").
    - Re-naming правил под канон arbiter — это бы скрыло реальное разделение фаз.

### Готовы предоставить (ждём запроса от Maestro)

- [ ] **R-06b: SDK-интеграция для Maestro** (зависит от Maestro R-03)
  - `atp.sdk.arun()` или SDK Adapter — структурированные результаты
  - Автоматический feedback loop: Maestro → задача → ATP eval → arbiter обучение
  - Наш SDK уже готов (PyPI `atp-platform-sdk` v2.0.0)

- [ ] **R-07: Eval-driven routing validation** (зависит от R-03, R-06b)
  - A/B тестирование arbiter DT routing vs random vs always-best-agent
  - Совместно с `../arbiter/` — набор test suites на нашей стороне

### Ждём от других проектов

- **Maestro → R-03**: без MCP-клиента в Maestro невозможен feedback loop в arbiter → отложить R-06b/R-07
- **arbiter → R-10 (CI)**: при работе над R-13 хочется уверенности в стабильности invariants

### НЕ делаем здесь

- ❌ Собственная интеграция с spec-runner — связь идёт через Maestro
- ❌ Расширение ATP под специфику Maestro до формализации `validation_cmd` контракта

---

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

## Admin tournament GUI follow-ups (deferred from 2026-04-20 spec)

Spec: `docs/superpowers/specs/2026-04-20-admin-tournament-gui-design.md`
Plan: `docs/superpowers/plans/2026-04-20-admin-tournament-gui.md`

- [ ] **h · Live MCP SSE connection status** per participant in admin detail — needs a new in-memory connection registry bound to the FastMCP server plus a `/ui/admin/tournaments/{id}/connections` fragment. Scope: ~2 days.
- [ ] **f · Force-advance round** (admin button + REST endpoint wrapping existing `TournamentService.force_resolve_round`) — currently only the deadline worker can trigger it. Safety: needs a confirmation step and audit log entry.
- [ ] **g · Extend round deadline mid-round** — requires adding a service method and a new audit row since mutating `Round.deadline` after creation is currently disallowed.
- [ ] **Generalize admin create form to all 8 games** — currently hardcoded to `el_farol` dropdown. Add per-game config fieldsets keyed off the game registry.
- [ ] **Long-lived bot MCP sessions (spec C)** — separate design and plan; the admin TTL change in this PR does not address bot-side session budget (still capped at `(ATP_TOKEN_EXPIRE_MINUTES − 10) × 60` in `TournamentService.create_tournament`).

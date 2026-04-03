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

## Platform API & SDK (atp-sdk)

See full spec: `docs/superpowers/specs/2026-04-02-platform-api-and-sdk-design.md`

### MVP
- [x] Расширить atp-dashboard: catalog API + tournament API route-группы
- [x] Добавить GitHub как OIDC-провайдер в существующий SSO-модуль
- [x] Добавить Device Flow для CLI-логина
- [x] Новые SQLAlchemy-модели (Benchmark, Run, TaskResult, Tournament, Participant, Round, Action)
- [x] Alembic-миграция для новых таблиц
- [x] Cancel endpoint + серверный таймаут прогонов (status=partial)
- [x] Benchmark family_tag + parent_id для версионирования
- [x] Run.adapter_type для аналитики (sdk/http/cli/...)
- [x] Login/Register UI + RBAC seed + auto-admin для первого пользователя
- [x] Создать packages/atp-sdk/ — Python SDK для участников (client, benchmark iterator, auth)
- [x] Создать SDKAdapter в atp-adapters (asyncio.Event + timeout, pull-модель как AgentAdapter)
- [ ] Sandbox для evaluators на сервере (subprocess + timeout + rlimits)
- [ ] Опубликовать atp-sdk на PyPI

### Post-MVP
- [ ] `?batch=N` для параллельного получения задач (зарезервировано в API)
- [ ] Redis pub/sub для SDKAdapter (замена asyncio.Event, переживает рестарт)
- [ ] Автоматический трекинг токенов в SDK (обёртка над LLM-вызовами)
- [ ] Event streaming в SDK (отправка ATPEvent во время выполнения)
- [ ] Workspace management в SDK (скачивание/загрузка файлов-артефактов)
- [ ] Async API в SDK (async for task in run)
- [ ] Retry/reconnect при обрывах в SDK
- [ ] TypeScript SDK
- [ ] WebSocket для real-time турниров (инфраструктура в dashboard уже есть)
- [ ] Container-изоляция evaluators (Podman/Docker)
- [ ] Федерация — приватный atp-server
- [ ] Webhooks для CI/CD-уведомлений по завершении прогона
- [ ] Rate limiting на уровне приложения
- [ ] Выделить atp-protocol как отдельный лёгкий пакет (если вес atp-core станет проблемой для SDK)
- [ ] Детализировать Tournament API (cancel, серверные таймауты раундов, пропуск дедлайнов)

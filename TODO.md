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
- [x] Sandbox для evaluators на сервере (subprocess + timeout + rlimits)
- [x] Опубликовать atp-sdk на PyPI (как atp-platform-sdk)

### Post-MVP
- [x] `?batch=N` для параллельного получения задач (SDK v2.0.0)
- [ ] Redis pub/sub для SDKAdapter (замена asyncio.Event, переживает рестарт)
- [ ] Автоматический трекинг токенов в SDK (обёртка над LLM-вызовами)
- [ ] Event streaming в SDK (отправка ATPEvent во время выполнения)
- [ ] Workspace management в SDK (скачивание/загрузка файлов-артефактов)
- [x] Async API в SDK — AsyncATPClient + async for task in run (SDK v2.0.0)
- [x] Retry/reconnect при обрывах в SDK — exponential backoff + full jitter (SDK v2.0.0)
- [ ] TypeScript SDK
- [ ] WebSocket для real-time турниров (инфраструктура в dashboard уже есть)
- [ ] Container-изоляция evaluators (Podman/Docker)
- [ ] Федерация — приватный atp-server
- [ ] Webhooks для CI/CD-уведомлений по завершении прогона
- [ ] Rate limiting на уровне приложения
- [ ] Выделить atp-protocol как отдельный лёгкий пакет (если вес atp-core станет проблемой для SDK)
- [ ] Детализировать Tournament API (cancel, серверные таймауты раундов, пропуск дедлайнов)

## Architecture Cleanup (P0 → P2)

### P0 — Critical

- [x] **AuthFlowStateStore**: единый `InMemoryAuthStateStore` для SSO/SAML (auth/state_store.py). `_sso_sessions` и `_saml_sessions` убраны
- [x] **Починить SSO тесты**: синхронизированы с текущим SSOInitRequest API (extra="forbid")
- [x] **allow_shell в CLIAdapter**: полностью удалён. Shell features доступны через `command="sh" args=["-c", "..."]`

### P1 — Important

- [x] **Общий post-auth service**: `complete_auth()` в auth/post_auth.py — provision user + assign roles + issue token. SSO, SAML, DeviceFlow routes используют его
- [x] **Удалить `return_url` из SAML**: убрано из SAMLInitRequest и session storage
- [x] **DeviceFlowStore API**: `lookup()` + `lookup_by_user_code()` + `DeviceFlowStatus` constants вместо строк

### P2 — Improvement

- [x] **Развязать atp-dashboard от atp-platform**: shared result models вынесены в `atp.core.results`, dashboard зависит от atp-core
- [ ] **Объединить route-модели SSO/SAML**: убрать дублирование request/response моделей
- [ ] **Почистить примеры и конфиги** от shell-режима и старых assumptions

## Dashboard UI

- [ ] **Chart.js в Analytics**: pie chart статусов, histogram scores, line chart по агентам (templates/ui/analytics.html)
- [ ] **Fix UI routes test isolation**: `.value` баг в analytics/home templates, UNIQUE constraint collision
- [ ] **Benchmark API scoring**: подключить evaluators вместо наивной оценки (100 if completed else 0)

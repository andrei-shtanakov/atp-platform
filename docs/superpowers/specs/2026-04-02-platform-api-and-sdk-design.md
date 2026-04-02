# ATP Platform API & SDK Design

**Date:** 2026-04-02
**Status:** Draft (rev.2 — после code review)
**Context:** Сравнительный анализ с BitGN/ERC, переход ATP к модели бенчмарк-платформы

## 1. Цель

Превратить ATP из CLI-инструмента в бенчмарк-платформу с:
- Pull-моделью взаимодействия (агент сам приходит за задачами)
- Python SDK для участников
- REST API с лидербордом
- Поддержкой game-theoretic турниров

Приоритет: сначала бенчмарк-платформа (каталог + лидерборд), затем CI/CD-инструмент.

## 2. Архитектура

### 2.1 Компоненты

```
packages/atp-dashboard/  — FastAPI сервер (СУЩЕСТВУЮЩИЙ, расширяется новыми route-группами)
packages/atp-sdk/        — Python SDK для участников (НОВЫЙ, публикуется на PyPI)
packages/atp-core/       — protocol, evaluators, scoring (существующий, без изменений)
packages/atp-adapters/   — адаптеры (существующий, добавляется SDKAdapter)
```

> **Решение (rev.2):** Не создаём отдельный `atp-server`. Dashboard уже содержит
> FastAPI factory, SQLAlchemy + Alembic, WebSocket pub/sub, RBAC (36+ permissions),
> multi-tenancy, SSO (OIDC/SAML, 12 провайдеров), catalog routes, leaderboard routes,
> auth routes (30 route-модулей, 12000+ строк). Создание параллельного сервера —
> удвоение инфраструктуры.

### 2.2 Схема взаимодействия

```
Агент участника                        Разработчик (локально)
    │                                       │
    ▼                                       ▼
atp-sdk (Python)                     atp CLI (существующий)
    │ REST API (httpx)                      │
    ▼                                       ▼
atp-dashboard (FastAPI)              AgentAdapter (push-модель)
    │                                       │
    ├── Auth (OIDC + GitHub provider)       ├── HTTPAdapter
    ├── Catalog API (новые routes)          ├── CLIAdapter
    ├── Tournament API (новые routes)       ├── LangGraphAdapter
    │                                       ├── SDKAdapter (НОВЫЙ)
    ├── atp-core (evaluators, scoring)      └── ...10+ адаптеров
    └── Storage (SQLAlchemy, уже есть)
```

### 2.3 Ключевые архитектурные решения

- **Pull-модель**: агент работает за NAT, без публичного endpoint. Низкий порог входа.
- **Серверная оценка**: evaluators запускаются на сервере при submit. Невозможно подкрутить результаты.
- **Sandbox для evaluators на сервере**: code-exec evaluator выполняет произвольный код — нужна изоляция (container/seccomp). Критично для безопасности.
- **Агент = чёрный ящик**: получил ATPRequest → вернул ATPResponse. Никакой sandbox-среды платформы.
- **REST на старте, WebSocket позже**: polling покрывает MVP. Dashboard уже имеет WebSocket pub/sub — подключим для real-time турниров.
- **Адаптеры сохраняются**: SDK — дополнительный способ подключения, не замена. SDKAdapter оборачивает pull-модель в стандартный AgentAdapter.
- **Обратная совместимость CLI**: `atp test suite.yaml --adapter=http` продолжает работать без изменений.
- **Бенчмарки immutable**: опубликованный бенчмарк не меняется. Новая версия = новый бенчмарк.

### 2.4 SDKAdapter — мост между pull и push

SDKAdapter реализует интерфейс `AgentAdapter`, но вместо отправки запроса агенту —
кладёт задачу в очередь, откуда её забирает агент через SDK (pull).

```python
class SDKAdapter(AgentAdapter):
    """Adapter that serves tasks via pull-model API."""

    @property
    def adapter_type(self) -> str:
        return "sdk"

    async def execute(self, request: ATPRequest) -> ATPResponse:
        # 1. Положить request в очередь задач (БД)
        # 2. Ждать пока агент через SDK заберёт и вернёт результат
        # 3. Вернуть ATPResponse
        ...

    async def stream_events(self, request: ATPRequest) -> AsyncIterator[ATPEvent | ATPResponse]:
        # То же, но с поддержкой events от агента
        ...
```

Это позволяет:
- Переиспользовать весь существующий orchestrator/evaluators/reporters
- Запускать бенчмарки через `atp test suite.yaml --adapter=sdk`
- Не ломать ни один существующий flow

## 3. API-контракт

Новые route-группы добавляются в `atp-dashboard` рядом с существующими.

### 3.1 Auth

Используем существующий OIDC-модуль dashboard. GitHub добавляется как предустановленный OIDC-провайдер.

```
# Существующие (auth.py, sso.py):
POST   /auth/token                → логин (OAuth2)
POST   /auth/register             → регистрация
GET    /auth/me                   → профиль
POST   /sso/init                  → начать OIDC flow
POST   /sso/callback              → завершить OIDC flow

# Новое: GitHub как OIDC-провайдер (конфигурация, не код)
# + Device Flow для CLI-логина через SDK
POST   /auth/device               → инициировать device flow (RFC 8628)
POST   /auth/device/poll          → проверить статус device flow
```

> **TODO (rev.2):** Проверить, поддерживает ли текущий OIDC-модуль Device Flow.
> Если нет — добавить. Device Flow критичен для CLI-логина из SDK.

### 3.2 Catalog (бенчмарки)

```
GET    /api/v1/benchmarks                  → список бенчмарков
GET    /api/v1/benchmarks/{id}             → детали (описание, кол-во задач, версия)
POST   /api/v1/benchmarks                  → создать (загрузить test suite) [admin]

POST   /api/v1/benchmarks/{id}/start       → начать прогон → run_id
GET    /api/v1/runs/{run_id}/next-task     → следующая задача (ATPRequest); 204 No Content когда все задачи выданы
POST   /api/v1/runs/{run_id}/submit        → результат (ATPResponse + опциональные ATPEvent[]); 409 если задача уже отправлена
GET    /api/v1/runs/{run_id}/status        → прогресс, скоры, список завершённых задач

GET    /api/v1/benchmarks/{id}/leaderboard → таблица результатов
GET    /api/v1/leaderboard                 → глобальный лидерборд
```

> **Изменение (rev.2):** `POST /submit` принимает `{response: ATPResponse, events?: ATPEvent[]}`.
> Events опциональны на старте, но API готов к ним с первого дня. Если events
> переданы — сохраняем для отладки и анализа.

### 3.3 Tournaments (game-theoretic)

```
GET    /api/v1/tournaments                     → список турниров
GET    /api/v1/tournaments/{id}                → детали (игра, правила, период)
POST   /api/v1/tournaments/{id}/join           → зарегистрировать агента

GET    /api/v1/tournaments/{id}/current-round  → текущий раунд + состояние
POST   /api/v1/tournaments/{id}/action         → действие агента
GET    /api/v1/tournaments/{id}/results        → итоги
```

### 3.4 Принципы API

- Все эндпоинты возвращают JSON
- Версионирование: `/api/v1/`
- ATPRequest/ATPResponse — Pydantic-модели из atp-core
- Задачи выдаются по одной (next-task), агент обрабатывает последовательно
- При submit сервер прогоняет evaluators из atp-core и записывает скор
- Бенчмарки immutable — нельзя изменить после публикации
- RBAC из dashboard контролирует доступ (создание бенчмарков = admin, прогон = authenticated user)

## 4. Python SDK (atp-sdk)

### 4.1 Интерфейс для участника

```python
from atp_sdk import ATPClient

client = ATPClient(
    platform_url="https://atp.example.com",
    token="..."  # или из env ATP_TOKEN, или из ~/.atp/config.json после login
)

# Бенчмарки
benchmarks = client.list_benchmarks()
run = client.start_run("benchmark-42")

for task in run:                    # pull-loop скрыт за итератором
    response = my_agent_logic(task)  # task = ATPRequest
    run.submit(response)             # response = ATPResponse

print(run.status())
print(run.leaderboard())

# Турниры
tournament = client.join_tournament("tournament-7")

for round in tournament:
    action = my_game_agent(round.state)
    round.act(action)

print(tournament.results())
```

### 4.2 Структура пакета

```
packages/atp-sdk/
├── pyproject.toml          # зависимости: httpx, pydantic
└── atp_sdk/
    ├── __init__.py          # re-export ATPClient
    ├── client.py            # ATPClient — основной класс
    ├── auth.py              # OIDC device flow для CLI-логина
    ├── models.py            # re-export из atp-core + RunStatus, LeaderboardEntry
    ├── benchmark.py         # BenchmarkRun — итератор по задачам
    └── tournament.py        # TournamentSession — игровой цикл
```

### 4.3 Ключевые решения

- `for task in run` — скрывает polling. BenchmarkRun.__iter__ вызывает GET /next-task
- atp-core как зависимость — ATPRequest/ATPResponse из единого источника
- httpx — sync и async
- Device Flow для auth — `atp-sdk login` → браузер → OIDC (GitHub) → токен в ~/.atp/config.json
- Минимум зависимостей: httpx, pydantic

### 4.4 TODO на перспективу (не MVP)

- Автоматический трекинг токенов (обёртка над LLM-вызовами)
- Event streaming (отправка ATPEvent во время выполнения через SDK)
- Workspace management (скачивание/загрузка файлов-артефактов)
- Async API (async for task in run)
- Retry/reconnect при обрывах
- TypeScript SDK (когда community потребует)

## 5. Модель данных

### 5.1 Новые сущности

Добавляются в существующую SQLAlchemy-модель dashboard через новую Alembic-миграцию.
Существующие модели (User, Agent, SuiteExecution и др.) переиспользуются где возможно.

**Benchmark** — name, description, suite (YAML как JSON), tasks_count, tags, version, is_immutable=True, created_by (FK → User), created_at

**Run** — user_id (FK → User), benchmark_id (FK → Benchmark), agent_name, status (pending/in_progress/completed/failed/cancelled), current_task_index, total_score, started_at, finished_at

**TaskResult** — run_id (FK → Run), task_index, request (JSON), response (JSON), events (JSON, опционально), eval_results (JSON), score, submitted_at

**Tournament** — game_type, config (JSON), status, starts_at, ends_at, rules (JSON), created_by (FK → User)

**Participant** — tournament_id (FK → Tournament), user_id (FK → User), agent_name, joined_at, total_score

**Round** — tournament_id (FK → Tournament), round_number, state (JSON), status, started_at, deadline

**Action** — round_id (FK → Round), participant_id (FK → Participant), action_data (JSON), submitted_at

### 5.2 Технические решения

- Используем существующую SQLAlchemy-инфраструктуру dashboard (async engine, session factory, database manager)
- Новая Alembic-миграция добавляет таблицы, не трогая существующие
- JSON-поля для request, response, events, eval_results, state, config
- Round.deadline для турниров — пропуск = default action или дисквалификация (настраивается в Tournament.rules)
- Benchmark.is_immutable — после создания suite нельзя изменить, только создать новый бенчмарк
- TaskResult.events — опциональное поле для хранения ATPEvent[] при submit

### 5.3 Что НЕ храним на старте

- Файловые артефакты (только structured/inline в ATPResponse)
- Историю изменений лидерборда (вычисляем из Run/TaskResult)

## 6. Сравнительный анализ: ATP vs BitGN

### 6.1 Модели взаимодействия

| Аспект | BitGN/ERC | ATP (новый) |
|--------|-----------|-------------|
| Инициатор | Агент (pull) | Агент (pull) + адаптеры (push) |
| Транспорт | gRPC/Protobuf | REST/JSON |
| SDK | Auto-generated из .proto (10+ языков) | Ручной Python SDK |
| Среда агента | Sandbox с typed entities | Чёрный ящик |
| Состояние | Entities между шагами | Stateless (бенчмарки) / раунды (турниры) |
| Многошаговость | Несколько API-вызовов за задачу | Один request → response (+ опц. events) |

### 6.2 Что взяли из BitGN

1. **Pull-модель** — агент за NAT, без публичного endpoint
2. **Серверная оценка** — scoring на платформе
3. **Журнал экспериментов** — каждый прогон с полным контекстом
4. **Замороженный контракт для соревнований** — турнир фиксирует версию задач

### 6.3 Что НЕ взяли

1. **gRPC/Protobuf** — overhead для Python-community. REST + JSON Schema проще.
2. **Typed entities / sandbox** — у нас агент = чёрный ящик, проще и универсальнее.
3. **Многошаговое взаимодействие** — получил задачу → вернул ответ. Проще тестировать.

### 6.4 Ключевое отличие

- BitGN = "платформа-среда" (агент работает внутри среды платформы)
- ATP = "платформа-арбитр" (агент автономен, платформа выдаёт задачи и оценивает)

### 6.5 Уникальное преимущество ATP

ATP сохраняет оба способа подключения:
- **Push (адаптеры)** — платформа вызывает агента. Для локального тестирования, CI/CD, framework-specific интеграции (LangGraph, CrewAI, AutoGen, MCP, Bedrock, Vertex, Azure).
- **Pull (SDK)** — агент приходит на платформу. Для бенчмарков, соревнований, community.

BitGN поддерживает только pull. ATP поддерживает оба.

## 7. Два режима работы

### 7.1 Бенчмарк-каталог (режим C)

- Набор стандартных тестов, доступных всегда
- Бенчмарки immutable — новая версия = новый бенчмарк
- Любой может прогнать когда угодно
- Результаты копятся, лидерборд обновляется
- Аналог HuggingFace Open LLM Leaderboard

### 7.2 Турниры (режим B)

- Game-theoretic соревнования (Prisoner's Dilemma, El Farol и др.)
- Период проведения (starts_at / ends_at)
- Фиксированные правила и состав участников
- Раунды с дедлайнами
- Отдельный лидерборд

## 8. Безопасность

### 8.1 Sandbox для evaluators на сервере

Code-exec evaluator выполняет произвольный код участника. На сервере это критический риск.

Варианты изоляции (выбрать при реализации):
- **Container-based**: каждая оценка в одноразовом контейнере (Podman/Docker)
- **seccomp/landlock**: ограничение syscalls на уровне Linux
- **Timeout + resource limits**: cgroups для CPU/memory/disk

На MVP: запуск evaluators в subprocess с timeout + rlimits. Container-изоляция — следующий шаг.

### 8.2 Защита API

- RBAC из dashboard (создание бенчмарков = admin, прогон = authenticated user)
- Rate limiting через middleware (не в MVP-коде, но через nginx/reverse proxy на деплое)

## 9. Scope и ограничения MVP

### Входит в MVP
- Расширение atp-dashboard: catalog API, tournament API route-группы
- GitHub как OIDC-провайдер + Device Flow для CLI
- atp-sdk: ATPClient, BenchmarkRun, TournamentSession, auth
- SDKAdapter в atp-adapters
- Новые SQLAlchemy-модели + Alembic-миграция
- Evaluator sandbox: subprocess + timeout + rlimits
- Immutable бенчмарки
- Опциональные events в submit API

### НЕ входит в MVP
- WebSocket для real-time турниров (polling на старте; инфраструктура в dashboard уже есть)
- Email-уведомления
- Автоматический трекинг токенов в SDK
- Event streaming из SDK (API готов, SDK — нет)
- Workspace management в SDK
- TypeScript SDK
- Container-изоляция evaluators (subprocess на старте)
- Федерация (приватный сервер)
- Webhooks для CI/CD-уведомлений

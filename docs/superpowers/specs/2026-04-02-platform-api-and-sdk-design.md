# ATP Platform API & SDK Design

**Date:** 2026-04-02
**Status:** Draft (rev.3 — после второго раунда review)
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
        task_id = await self._enqueue_task(request)       # 1. Записать в БД
        event = asyncio.Event()                            # 2. Создать Event
        self._pending[task_id] = event
        await asyncio.wait_for(event.wait(), timeout=...)  # 3. Ждать submit
        return self._results.pop(task_id)                  # 4. Вернуть результат

    async def stream_events(self, request: ATPRequest) -> AsyncIterator[ATPEvent | ATPResponse]:
        # То же, но с поддержкой events от агента
        ...
```

**Механика ожидания (rev.3):**
- MVP: `asyncio.Event` с timeout. Быстро, просто, достаточно для single-process сервера.
- При submit через API: route находит pending Event по task_id и делает `event.set()`.
- Timeout на ожидание = `Run.timeout_seconds` (настраивается при создании прогона).
- Ограничение: не переживает рестарт сервера. При рестарте pending runs переходят в status=failed.
- Post-MVP: Redis pub/sub для multi-process и persistence.

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

POST   /api/v1/benchmarks/{id}/start       → начать прогон → run_id; опц. ?timeout=3600
GET    /api/v1/runs/{run_id}/next-task     → следующая задача (ATPRequest); 204 когда все выданы; опц. ?batch=N (зарезервировано, MVP: всегда 1)
POST   /api/v1/runs/{run_id}/submit        → результат {response: ATPResponse, events?: ATPEvent[]}; 409 если задача уже отправлена
GET    /api/v1/runs/{run_id}/status        → прогресс, скоры, список завершённых задач
POST   /api/v1/runs/{run_id}/cancel        → отмена прогона → status=cancelled

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

> **Примечание (rev.3):** Tournament API намеренно менее детализирован чем Catalog API.
> Турниры — второй приоритет. Cancel турнира, серверные таймауты раундов, обработка
> пропущенных дедлайнов будут детализированы при реализации.

### 3.4 Принципы API

- Все эндпоинты возвращают JSON
- Версионирование: `/api/v1/`
- ATPRequest/ATPResponse — Pydantic-модели из atp-core
- Задачи выдаются по одной (next-task). Параметр `?batch=N` зарезервирован для параллельного получения (не в MVP)
- **Атомарность next-task**: выдача задачи использует атомарный инкремент `current_task_index` (SELECT ... FOR UPDATE или аналог). Два одновременных вызова next-task для одного run не получат одну и ту же задачу
- **Лидерборд**: лучший `total_score` per user per benchmark, сортировка по score desc, пагинация. Глобальный лидерборд — агрегация лучших скоров по benchmark families
- При submit сервер прогоняет evaluators из atp-core и записывает скор
- Бенчмарки immutable — нельзя изменить после публикации
- RBAC из dashboard контролирует доступ (создание бенчмарков = admin, прогон = authenticated user)
- **Таймаут прогона**: сервер отслеживает `run.timeout_seconds` (задаётся при start, default 3600). Если агент не завершил прогон за это время, run переходит в `status=partial` с результатами по завершённым задачам
- **Отмена прогона**: `POST /runs/{id}/cancel` переводит run в `status=cancelled`. Незавершённые задачи не оцениваются. Завершённые результаты сохраняются

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
- atp-core как зависимость — SDK импортирует только `atp.protocol` (ATPRequest/ATPResponse/ATPEvent). Protocol-модуль изолирован: зависит только от pydantic, не тянет evaluators/otel/prometheus. Если вес станет проблемой — выделим `atp-protocol` как отдельный пакет
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

**Benchmark** — name, description, suite (YAML как JSON), tasks_count, tags, version, family_tag (группировка версий одного бенчмарка), parent_id (FK → Benchmark, nullable), is_immutable=True, created_by (FK → User), created_at

> **family_tag (rev.3):** Связывает версии одного бенчмарка. Например, `family_tag="coding-basics"` для
> `coding-basics-v1`, `coding-basics-v2`. Позволяет лидерборду показать прогресс агента через версии.
> `parent_id` указывает на предыдущую версию для навигации.

**Run** — user_id (FK → User), benchmark_id (FK → Benchmark), agent_name, adapter_type (str, e.g. "sdk"/"http"/"cli"), status (pending/in_progress/completed/failed/cancelled/partial), current_task_index, total_score, timeout_seconds (default 3600), started_at, finished_at

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
- Benchmark.family_tag + parent_id — связь версий одного бенчмарка для аналитики и лидерборда
- TaskResult.events — опциональное поле для хранения ATPEvent[] при submit
- Run.adapter_type — для аналитики: сколько прогонов через SDK vs CLI vs HTTP
- Run.status=partial — при таймауте или частичном завершении. Завершённые задачи сохраняют скоры
- Run.timeout_seconds — серверный таймаут на весь прогон (фоновая задача проверяет и переводит в partial)

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
- SDKAdapter в atp-adapters (asyncio.Event + timeout)
- Новые SQLAlchemy-модели + Alembic-миграция
- Evaluator sandbox: subprocess + timeout + rlimits
- Immutable бенчмарки с family_tag/parent_id
- Опциональные events в submit API
- Cancel endpoint + серверный таймаут прогонов (status=partial)
- Run.adapter_type для аналитики

### Зарезервировано в API (не реализовано в MVP)
- `?batch=N` для параллельного получения задач (параметр принимается, но всегда выдаёт 1)
- WebSocket для real-time турниров (polling на старте; инфраструктура в dashboard уже есть)
- Redis pub/sub для SDKAdapter (asyncio.Event на старте)

### НЕ входит в MVP
- Email-уведомления
- Автоматический трекинг токенов в SDK
- Event streaming из SDK (API готов, SDK — нет)
- Workspace management в SDK
- TypeScript SDK
- Container-изоляция evaluators (subprocess на старте)
- Федерация (приватный сервер)
- Webhooks для CI/CD-уведомлений
- Rate limiting на уровне приложения

## 10. Deployment

### 10.1 Архитектура деплоя

```
┌─── VPS (сервер) ──────────────────────┐
│                                       │
│  nginx (TLS, reverse proxy)           │
│       │                               │
│       ▼                               │
│  atp-dashboard (FastAPI)              │
│       ├── Catalog API (бенчмарки)     │
│       ├── Tournament API (турниры)    │
│       ├── Auth (OIDC + GitHub)        │
│       ├── Evaluators (серверные)      │
│       └── SQLite/PostgreSQL           │
│                                       │
└───────────────────────────────────────┘
          ▲           ▲           ▲
          │           │           │
     Участник A  Участник B   Участник C
     (atp-sdk)   (atp-sdk)   (atp-sdk)
```

Ключевое: на сервере нет агентов. Сервер только выдаёт задачи и оценивает ответы.
Вся тяжёлая работа (LLM-вызовы, reasoning) — на стороне участника.

### 10.2 Нагрузочный профиль

| Операция | CPU | RAM | I/O |
|----------|-----|-----|-----|
| GET /next-task | Ничтожно | ~1 КБ | DB read |
| POST /submit (без code-exec) | Низко | ~10-100 КБ | DB write |
| LLM Judge evaluator | Ничтожно | ~1 КБ | HTTP → Anthropic API |
| Code-exec evaluator | Средне | ~200 МБ пик | subprocess |
| Лидерборд | Ничтожно | ~1 КБ | DB query |

90% операций — «принять JSON, записать в БД, посчитать скор». Единственная тяжёлая операция — code-exec evaluator.

### 10.3 Минимальные требования

**MVP (до 10 одновременных участников):** 2 vCPU, 2 ГБ RAM, 20 ГБ SSD

**Рост (50+ участников):** 4 vCPU, 4 ГБ RAM, 40 ГБ SSD, PostgreSQL вместо SQLite

### 10.4 Переменные окружения

| Переменная | Описание | Обязательна |
|-----------|----------|-------------|
| `ATP_SECRET_KEY` | Секрет для JWT-токенов | Да (production) |
| `ATP_DATABASE_URL` | SQLAlchemy URL (`sqlite+aiosqlite:///data/atp.db` или PostgreSQL) | Нет (default: SQLite) |
| `ATP_ANTHROPIC_API_KEY` | API-ключ для LLM Judge evaluator | Нет (если не используется) |
| `ATP_CORS_ORIGINS` | Разрешённые CORS origins | Нет (default: *) |
| `ATP_ENV` | `production` / `development` | Нет (default: development) |

### 10.5 Docker Compose (рекомендуемый деплой)

```yaml
services:
  platform:
    build: .
    command: >
      uvicorn atp.dashboard.v2.factory:app
      --host 0.0.0.0 --port 8080 --workers 2
    environment:
      ATP_DATABASE_URL: "sqlite+aiosqlite:///data/atp.db"
      ATP_SECRET_KEY: "${SECRET_KEY}"
      ATP_ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
      ATP_ENV: production
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - atp-data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./deploy/nginx.conf:/etc/nginx/conf.d/default.conf
      - /etc/letsencrypt:/etc/letsencrypt:ro
    depends_on:
      - platform

volumes:
  atp-data:
```

### 10.6 Bare metal (альтернатива)

```bash
uv run uvicorn atp.dashboard.v2.factory:app --host 127.0.0.1 --port 8080 --workers 2
```

Плюс systemd unit + nginx. Проще если code-exec evaluator не используется.

### 10.7 Стоимость

| Провайдер | Конфиг | Цена |
|-----------|--------|------|
| Hetzner CX22 | 2 vCPU, 4 ГБ, 40 ГБ | ~€4/мес |
| Hetzner CX32 | 4 vCPU, 8 ГБ, 80 ГБ | ~€7/мес |

Основные затраты — не VPS, а Anthropic API для LLM Judge (~$0.003 за оценку).

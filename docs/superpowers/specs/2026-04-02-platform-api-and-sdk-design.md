# ATP Platform API & SDK Design

**Date:** 2026-04-02
**Status:** Draft
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
packages/atp-server/     — FastAPI сервер (новый пакет в monorepo)
packages/atp-sdk/        — Python SDK для участников (публикуется на PyPI)
packages/atp-core/       — protocol, evaluators, scoring (существующий)
packages/atp-dashboard/  — веб-интерфейс (существующий, расширяется)
```

### 2.2 Схема взаимодействия

```
Агент участника
    │
    ▼
atp-sdk (Python)
    │ REST API (httpx)
    ▼
atp-server (FastAPI)
    │
    ├── Auth (GitHub OAuth)
    ├── Catalog (бенчмарки, прогоны, лидерборд)
    ├── Tournaments (game-theoretic соревнования)
    │
    ├── atp-core (evaluators, scoring, protocol)
    └── Storage (SQLite → PostgreSQL)
```

### 2.3 Ключевые архитектурные решения

- **Pull-модель**: агент работает за NAT, без публичного endpoint. Низкий порог входа.
- **Серверная оценка**: evaluators запускаются на сервере при submit. Невозможно подкрутить результаты.
- **Агент = чёрный ящик**: получил ATPRequest → вернул ATPResponse. Никакой sandbox-среды платформы.
- **REST на старте, WebSocket позже**: polling покрывает MVP, WS добавим для real-time турниров.

## 3. API-контракт

### 3.1 Auth

```
GET  /auth/github              → редирект на GitHub OAuth
GET  /auth/github/callback     → токен + создание/обновление пользователя
GET  /auth/me                  → профиль (name, avatar, github_username)
```

Авторизация: `Authorization: Bearer <token>`.

### 3.2 Catalog (бенчмарки)

```
GET    /api/v1/benchmarks                  → список бенчмарков
GET    /api/v1/benchmarks/{id}             → детали (описание, кол-во задач)
POST   /api/v1/benchmarks                  → создать (загрузить test suite) [admin]

POST   /api/v1/benchmarks/{id}/start       → начать прогон → run_id
GET    /api/v1/runs/{run_id}/next-task     → следующая задача (ATPRequest); 204 No Content когда все задачи выданы
POST   /api/v1/runs/{run_id}/submit        → результат (ATPResponse) → eval + score; 409 если задача уже отправлена
GET    /api/v1/runs/{run_id}/status        → прогресс, скоры, список завершённых задач

GET    /api/v1/benchmarks/{id}/leaderboard → таблица результатов
GET    /api/v1/leaderboard                 → глобальный лидерборд
```

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
- При submit сервер создаёт ATPRequest из TestDefinition (как TestOrchestrator._create_request())

## 4. Python SDK (atp-sdk)

### 4.1 Интерфейс для участника

```python
from atp_sdk import ATPClient

client = ATPClient(
    platform_url="https://atp.example.com",
    token="ghp_xxx..."  # или из env ATP_TOKEN
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
    ├── auth.py              # GitHub OAuth device flow для CLI
    ├── models.py            # re-export из atp-core + RunStatus, LeaderboardEntry
    ├── benchmark.py         # BenchmarkRun — итератор по задачам
    └── tournament.py        # TournamentSession — игровой цикл
```

### 4.3 Ключевые решения

- `for task in run` — скрывает polling. BenchmarkRun.__iter__ вызывает GET /next-task
- atp-core как зависимость — ATPRequest/ATPResponse из единого источника
- httpx — sync и async
- Device Flow для auth — `atp-sdk login` → браузер → GitHub → токен в ~/.atp/config.json
- Минимум зависимостей: httpx, pydantic

### 4.4 TODO на перспективу (не MVP)

- Автоматический трекинг токенов (обёртка над LLM-вызовами)
- Event streaming (отправка ATPEvent во время выполнения)
- Workspace management (скачивание/загрузка файлов-артефактов)
- Async API (async for task in run)
- Retry/reconnect при обрывах

## 5. Модель данных

### 5.1 Сущности

**User** — github_id, username, avatar, token, created_at

**Benchmark** — name, description, suite (YAML как JSON), tasks_count, tags, created_at

**Run** — user_id, benchmark_id, agent_name, status (pending/in_progress/completed/failed/cancelled), current_task, total_score, started_at, finished_at

**TaskResult** — run_id, task_index, request (JSON), response (JSON), eval_results (JSON), score, submitted_at

**Tournament** — game_type, config (JSON), status, starts_at, ends_at, rules (JSON)

**Participant** — tournament_id, user_id, agent_name, joined_at, total_score

**Round** — tournament_id, round_number, state (JSON), status, started_at, deadline

**Action** — round_id, participant_id, action_data (JSON), submitted_at

### 5.2 Технические решения

- SQLite на старте через SQLAlchemy + aiosqlite. Миграция на PostgreSQL = смена connection string.
- Alembic для миграций с первого дня.
- JSON-поля для request, response, eval_results, state, config — не нормализуем то, что не запрашиваем по отдельным полям.
- Round.deadline для турниров — пропуск = default action или дисквалификация (настраивается в Tournament.rules).

### 5.3 Что НЕ храним на старте

- Трейсы/events агентов (добавим с event streaming)
- Файловые артефакты (только structured/inline в ATPResponse)
- Историю изменений лидерборда (вычисляем из Run/TaskResult)

## 6. Сравнительный анализ: ATP vs BitGN

### 6.1 Модели взаимодействия

| Аспект | BitGN/ERC | ATP (новый) |
|--------|-----------|-------------|
| Инициатор | Агент (pull) | Агент (pull) |
| Транспорт | gRPC/Protobuf | REST/JSON |
| SDK | Auto-generated из .proto (10+ языков) | Ручной Python SDK |
| Среда агента | Sandbox с typed entities | Чёрный ящик |
| Состояние | Entities между шагами | Stateless (бенчмарки) / раунды (турниры) |
| Многошаговость | Несколько API-вызовов за задачу | Один request → response |

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

## 7. Два режима работы

### 7.1 Бенчмарк-каталог (режим C)

- Набор стандартных тестов, доступных всегда
- Любой может прогнать когда угодно
- Результаты копятся, лидерборд обновляется
- Аналог HuggingFace Open LLM Leaderboard

### 7.2 Турниры (режим B)

- Game-theoretic соревнования (Prisoner's Dilemma, El Farol и др.)
- Период проведения (starts_at / ends_at)
- Фиксированные правила и состав участников
- Раунды с дедлайнами
- Отдельный лидерборд

## 8. Scope и ограничения MVP

### Входит в MVP
- atp-server: FastAPI, auth, catalog API, tournament API
- atp-sdk: ATPClient, BenchmarkRun, TournamentSession, auth (device flow)
- Storage: SQLite + Alembic
- Интеграция с atp-core (evaluators, protocol, scoring)
- GitHub OAuth

### НЕ входит в MVP
- WebSocket для real-time турниров (polling на старте)
- Email-уведомления
- Rate limiting
- Автоматический трекинг токенов в SDK
- Event streaming в SDK
- Workspace management в SDK
- TypeScript SDK
- PostgreSQL (SQLite достаточно на старте)

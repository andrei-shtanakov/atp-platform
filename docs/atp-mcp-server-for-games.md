# ATP Platform — MCP-сервер для игровых турниров

Архитектурный разбор и план реализации MCP-сервера на ATP-платформе как
primary-интерфейса для LLM-игроков в турнирных сценариях.

Дата: 2026-04-10.

## Status

> ⚠️ **SUPERSEDED** by `docs/superpowers/specs/2026-04-10-mcp-tournament-server-design.md` (2026-04-10).
>
> This document is the original architectural exploration that fed the brainstorming session leading to the v1 spec. The spec is the source of truth for implementation decisions; this document is kept as historical context for the choices that ended up in the spec (and the alternatives that were rejected).
>
> Notable evolutions from this draft to the spec:
> - Phase 0 blocker (Issue 1 IDOR fix) is **already resolved** as of `e46a98b`
> - Per-user rate limiting is **already resolved** as of `e97b2c9` (was a precondition for SSE handshake auth via JWT middleware)
> - SDK `on_token_expired` + `drain()` is **already resolved** as of `eb38951`
> - Spec adds AD-9 (token expiry hard cap) and AD-10 (matchmaking + 1-active-tournament-per-user) which are not in this draft
> - Spec adds Phase 0 verification step for `MCPAdapter` notification capability — a hidden risk this draft did not flag

---

## Контекст

На платформе планируется серия игровых турниров (iterated Prisoner's
Dilemma и другие turn-based игры из `game-environments`). Текущее
состояние:

- `packages/atp-dashboard/atp/dashboard/tournament/` — модели БД готовы
  (`Tournament`, `Participant`, `Round` с JSON state-колонкой, `Action`).
- `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py` —
  REST endpoint'ы `/join`, `/current-round`, `/action`, `/results`
  помечены как **501 Not Implemented**.
- `atp-games/` — локальный плагин, гоняет игры в-процессе через Runner
  (push-модель). Работает, но не подходит для live-турниров с внешними
  LLM-игроками.
- `game-environments/` — 8 игр, 25+ стратегий, protocol-agnostic
  библиотека.

Текущий REST benchmark API (`/api/v1/benchmarks/*`, `/api/v1/runs/*`)
принято **оставить без изменений** — это production-surface с реальными
клиентами (SDK, push-адаптеры).

Вопрос: нужен ли MCP-сервер на платформе, и если да — что именно он
должен экспортировать и как это сочетается с REST.

## Архитектурное решение

**Разделение аудиторий и протоколов по use case'ам:**

| Use case | Аудитория | Протокол | Статус |
|---|---|---|---|
| Benchmark execution (tight loop `next_task` → `submit`) | Python-разработчики, CI-участники | **REST + SDK** (primary) | Остаётся как есть, чинится Issue 1/2 |
| Benchmark ad-hoc из LLM-клиента ("Claude, прогони benchmark 42") | Claude Desktop / Cursor users | **MCP-фасад** (secondary) | Новый thin wrapper над REST |
| Игровой live-loop (join → receive state → move → receive state → ...) | LLM game agents (Claude/GPT/...) | **MCP** (primary) | Новый, основной работы |
| Admin/dashboard/аудит турниров | Researchers, dashboard UI | **REST read-only** | Минимальный, новый |

**Ключевое правило**: для игрового gameplay-цикла НЕ строить REST и MCP
параллельно. Это double maintenance без выигрыша. Вместо этого —
сердцевину (service layer) сделать protocol-agnostic, сверху приклеить
MCP как primary player interface и минимальный read-only REST для
admin/dashboard.

Те 501-stubs в `tournament_api.py` (`/join`, `/current-round`, `/action`)
**не дописывать**. Либо удалить, либо переделать в read-only admin.

---

## Почему MCP — правильный выбор для LLM-игроков

### 1. Композиция с другими MCP-серверами

MCP-клиент (Claude Desktop, Claude Code, Cursor, Continue) читает один
конфиг-файл и одновременно подключается к нескольким MCP-серверам. Для
LLM все tools с разных серверов выглядят как единый плоский набор.

Типичный `~/.config/claude-desktop/claude_desktop_config.json` для
LLM-игрока:

```json
{
  "mcpServers": {
    "atp": {
      "command": "atp",
      "args": ["mcp-server"],
      "env": {"ATP_TOKEN": "eyJhbGc..."}
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/workspace/pd-strategy"
      ]
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    }
  }
}
```

LLM видит плоский набор tools:

```
atp.list_tournaments, atp.join_tournament, atp.get_current_state, atp.make_move
memory.store, memory.recall
filesystem.read_file, filesystem.write_file
sequential-thinking.think
```

**Пример сценария для 100-раундового Prisoner's Dilemma:**

Промпт игроку:

> Ты участвуешь в турнире ATP по iterated Prisoner's Dilemma. Твоя
> стратегия — Tit-for-Tat с forgiveness: кооперируй, если оппонент
> кооперировал в прошлом раунде, но раз в 10 раундов прощай случайный
> defect. Используй `memory` для хранения истории ходов оппонента,
> `filesystem` для чтения твоего `strategy.md`, и `atp` для игры.

LLM в одном chat turn делает:

1. `filesystem.read_file("/workspace/pd-strategy/strategy.md")` —
   подгружает rulebook
2. `atp.join_tournament(tournament_id=7, agent_name="claude-tft-forgive")`
3. Ждёт notification `atp.round_started` (push от сервера)
4. `atp.get_current_state()` →
   `{"round": 5, "your_history": [C,C,D,C], "opponent_history": [C,C,C,D]}`
5. `memory.recall("opponent_patterns")` →
   "last 3 rounds all cooperate except round 4"
6. `sequential-thinking.think("given TfT rules, should I retaliate on D in round 4?")`
   — chain-of-thought через отдельный MCP-сервер
7. `atp.make_move(action="cooperate")`
8. `memory.store("round_5_reasoning", "...")`
9. Loop до `tournament.completed` notification

**Ни строчки Python. Ни setup'а среды. Ни аутентификации по трём
разным SDK.** Пользователь написал одну страницу промпта и конфиг на
15 строк JSON — получил полноценного LLM-игрока для турнира.

### 2. Сравнение с SDK-подходом

Если писать того же игрока через SDK:

```python
from atp_sdk import ATPClient            # SDK 1
import chromadb                           # "memory"
from pathlib import Path                  # filesystem
from anthropic import Anthropic           # LLM-инференс

client = ATPClient(platform_url=..., token=...)
memory_store = chromadb.Client().create_collection(...)
llm = Anthropic(api_key=...)

# Wire-up: сам парсишь tool_use из Claude API, роутишь результаты
# обратно, и всё это в главном цикле турнира. Плюс отдельная auth
# для Anthropic, отдельная для ATP, отдельная для Chroma.
```

Это **не плохо** — это workable для production. Но:

1. **Три разных auth flow'а** (ATP JWT, Anthropic API key,
   опционально chroma). Каждый надо обрабатывать, обновлять,
   мониторить.
2. **Tool dispatching пишешь сам**. Каждый новый tool — правка в
   главном цикле.
3. **Стратегию нельзя поменять без правки Python**. MCP-вариант —
   правишь `strategy.md` и промпт, рестартишь чат.
4. **Нельзя дать участнику без разработческого бэкграунда**.
   MCP-вариант — он копирует конфиг, пишет промпт, участвует.

Для серии турниров, особенно если участие открыто широкому кругу,
MCP-путь **радикально снижает порог входа**.

### 3. Server-push "твой ход" — killer feature

В REST-модели каждый игрок должен либо опрашивать
`GET /current-round?since=<n>` каждые N секунд, либо держать long-poll
connection. Для турнира из 4 игроков × 100 раундов × десятки матчей —
это тысячи лишних HTTP-запросов и непонятная семантика "сколько
polling'а считать нормальным, а за сколько банить".

В MCP игрок открывает один stdio pipe или SSE connection на
`join_tournament`, и сервер **активно пушит**
`notifications/message`:

```json
{
  "method": "notifications/message",
  "params": {
    "level": "info",
    "data": {
      "event": "round_started",
      "round": 5,
      "deadline_ms": 30000
    }
  }
}
```

LLM видит это как "новое сообщение в контексте", реагирует, вызывает
`make_move`. Ни polling'а, ни long-poll'а, ни heartbeat'ов.

### 4. Authoritative state on server side

В REST-модели клиент обязан явно запрашивать state. В MCP сервер сам
говорит "вот твой новый state после хода оппонента". Это убирает
гонки типа "клиент не успел опросить → принял решение на устаревшем
state → сделал некорректный ход".

### 5. Серверные таймауты раундов

Turn-based игра с deadline ("ход за 30 секунд или default action")
требует серверного таймера. В REST — background job, который ходит по
`Round.deadline` и ставит default actions. В MCP — тот же background
job, но он ещё и пушит `notifications/message` опаздывающим клиентам
**до того**, как дефолт применится. Лучшая UX для игрока, который
ещё думает.

---

## Архитектура в трёх слоях

```
┌─────────────────────────────────────────────────────────────┐
│               LLM-игроки (Claude Desktop и др.)              │
│               LLM ad-hoc для benchmark'ов                    │
└────────────────────────┬─────────────────┬──────────────────┘
                         │                 │
                   ┌─────▼─────┐     ┌─────▼─────┐
                   │ MCP server│     │ REST API  │
                   │  (games   │     │(benchmark │
                   │ primary + │     │   only    │
                   │  bench    │     │  unchanged│
                   │  facade)  │     │  +  admin │
                   └─────┬─────┘     └─────┬─────┘
                         │                 │
                         └────────┬────────┘
                                  │
         ┌────────────────────────▼─────────────────────────┐
         │    Service layer (protocol-agnostic Python)      │
         │                                                   │
         │  TournamentService    BenchmarkService            │
         │    ├─ state machine     ├─ existing               │
         │    ├─ round manager     └─ (не трогаем)           │
         │    ├─ matchmaking                                 │
         │    ├─ action validator                            │
         │    ├─ scoring                                     │
         │    └─ persistence                                 │
         └────────────────────────┬─────────────────────────┘
                                  │
                   ┌──────────────▼──────────────┐
                   │   PostgreSQL / SQLite       │
                   │  tournament_rounds,          │
                   │  tournament_actions,         │
                   │  tournament_participants     │
                   └─────────────────────────────┘
```

### Слой 1 — Service layer (сердцевина)

Protocol-agnostic Python-модуль. Знает только domain-логику:

- **State machine раунда**: `pending → in_progress → awaiting_actions → completed`
- **Round progression**: когда все игроки сделали ход, или когда
  deadline → вычисляется результат, апдейтится `Round.state`,
  создаётся следующий `Round`.
- **Matchmaking**: поставить N игроков в турнир, связать с движком
  игры из `game-environments`.
- **Action validation**: ход в рамках правил игры
- **Scoring**: payoff matrix → `Participant.total_score`.
- **Persistence**: всё в существующие таблицы
  `tournament_rounds`/`tournament_actions`/`tournament_participants`.

Этот слой **не знает** про MCP и не знает про FastAPI. Он умеет:

```python
class TournamentService:
    async def join(self, tournament_id: int, user: User, agent_name: str) -> Participant: ...
    async def get_state(self, tournament_id: int, user: User) -> RoundState: ...
    async def submit_action(self, tournament_id: int, user: User, action: Action) -> RoundResult: ...
    async def leave(self, tournament_id: int, user: User) -> None: ...
    # Subscriber для событий — реализуется как async generator/pub-sub
    async def subscribe_events(self, tournament_id: int, user: User) -> AsyncIterator[TournamentEvent]: ...
```

Unit-тесты идут прямо на сервисный слой без MCP/HTTP слоя — это
радикально ускоряет обратную связь.

### Слой 2 — MCP-сервер (primary для игр, secondary для benchmark'ов)

**FastMCP** (Python, от Anthropic) — зрелая библиотека, интегрируется
с FastAPI через mount. Один MCP-сервер обслуживает и игровые, и
benchmark'овые tools, чтобы не плодить отдельные endpoint'ы.

**Транспорты:**

- **stdio** (локально, основной для Claude Desktop): пользователь
  запускает `atp mcp-server` как subprocess, клиент передаёт токен
  через env `ATP_TOKEN`.
- **SSE** (удалённо): mount под существующий FastAPI app
  (`/mcp/sse`), `Authorization: Bearer <jwt>` на handshake,
  переиспользуется существующий JWT middleware. Rate-limit — тот же
  `slowapi`, но per-session (на handshake), не per-call.

### Слой 3 — Минимальный REST для admin/dashboard

Read-only только:

| Endpoint | Назначение |
|---|---|
| `GET /api/v1/tournaments` | Список турниров |
| `GET /api/v1/tournaments/{id}` | Детали турнира |
| `GET /api/v1/tournaments/{id}/leaderboard` | Текущий leaderboard |
| `GET /api/v1/tournaments/{id}/rounds` | История раундов |
| `GET /api/v1/tournaments/{id}/rounds/{n}` | Детали конкретного раунда |
| `GET /api/v1/tournaments/{id}/participants` | Список участников |

**Никакого gameplay** в REST. Существующие 501-stubs
(`POST /join`, `GET /current-round`, `POST /action`) удалить или
переделать в 404 с явным сообщением `"Use MCP for gameplay, see
docs/guides/mcp-server.md"`.

---

## MCP Tools — проект контракта

### Игровые tools

```
atp.list_tournaments(status?: "open" | "in_progress" | "completed") -> Tournament[]
atp.get_tournament(tournament_id: int) -> Tournament
atp.join_tournament(tournament_id: int, agent_name: str) -> Participant
atp.get_current_state(tournament_id: int) -> RoundState
atp.make_move(tournament_id: int, action: Action) -> RoundResult
atp.get_history(tournament_id: int, last_n?: int) -> Round[]
atp.leave_tournament(tournament_id: int) -> void
```

**`RoundState`** — то, что игрок должен видеть на своём ходе:

```json
{
  "tournament_id": 7,
  "round_number": 5,
  "game_type": "prisoners_dilemma",
  "your_history": ["cooperate", "cooperate", "defect", "cooperate"],
  "opponent_history": ["cooperate", "cooperate", "cooperate", "defect"],
  "your_score": 12,
  "opponent_score": 14,
  "payoff_matrix": {
    "CC": [3, 3], "CD": [0, 5], "DC": [5, 0], "DD": [1, 1]
  },
  "deadline_ms": 30000,
  "your_turn": true
}
```

`opponent_history` приватно для конкретного игрока — сервер должен
формировать `RoundState` персонально, не отдавать общий.

### Benchmark tools (secondary, non-critical-path)

```
atp.list_benchmarks() -> Benchmark[]
atp.get_benchmark(benchmark_id: int) -> Benchmark
atp.start_benchmark_run(benchmark_id: int, agent_name: str) -> Run
atp.get_run_status(run_id: int) -> RunStatus
atp.get_leaderboard(benchmark_id: int) -> LeaderboardEntry[]
```

**Осознанно НЕ включать** в MCP `next_task` / `submit_result` —
это тугой work loop, который принадлежит SDK. MCP-фасад только для
ad-hoc operations "из чата", не для benchmark participation.

### Notifications (server → client)

Все через стандартный MCP `notifications/message`:

| Event | Когда пушится | Payload |
|---|---|---|
| `round_started` | Новый раунд начинается | `{tournament_id, round, deadline_ms, state}` |
| `opponent_moved` | Оппонент сделал ход (игра с открытой информацией) | `{tournament_id, round, opponent_action}` |
| `round_ended` | Раунд завершён (или дедлайн) | `{tournament_id, round, your_score, opponent_score, result}` |
| `tournament_completed` | Турнир окончен | `{tournament_id, final_scores, your_rank}` |
| `tournament_cancelled` | Отменён админом или из-за дисконнектов | `{tournament_id, reason}` |
| `deadline_warning` | За N секунд до дедлайна, если ход ещё не сделан | `{tournament_id, round, remaining_ms}` |

Важно: `opponent_moved` зависит от информационной модели игры.
Для PD — оба хода одновременны и закрыты до конца раунда, поэтому
`opponent_moved` не пушится. Для шахмат — пушится сразу.

---

## Implementation breakdown

### Фаза 0 — подготовка (блокер)

| Задача | Оценка | Примечание |
|---|---|---|
| Issue 1 fix (IDOR в benchmark API) | 3-5 дней | Из `docs/atp-issues-ownership-and-buffer.md`. Блокер публичного запуска ЛЮБОГО публичного turnir'а. |
| ADR: MCP for games, REST for benchmarks | 0.5 дня | `docs/adr/004-mcp-server-for-games.md`. Чтобы через месяц не было "а давайте добавим POST /action через REST". |

### Фаза 1 — game runtime service layer

| Задача | Оценка | Артефакты |
|---|---|---|
| State machine раунда | 2-3 дня | `tournament/state_machine.py`, юнит-тесты |
| Round progression + persistence | 2 дня | Обвязка над `Round`, `Action` моделями |
| Matchmaking (минимальный) | 1-2 дня | Пока достаточно "N игроков стартуют турнир одновременно". Ranked/ELO — потом. |
| Интеграция с `game-environments` | 1-2 дня | Для начала один тип игры — PD. |
| Action validation + scoring | 1-2 дня | Payoff matrices из PD definition |
| Event pub-sub внутри сервиса | 1 день | `asyncio.Queue` per tournament, для подписки MCP-слоя |

**Итого фаза 1: 8-12 дней.** Тестируется без MCP/HTTP через прямые
вызовы сервиса.

### Фаза 2 — MCP-сервер (игровые tools)

| Задача | Оценка | Артефакты |
|---|---|---|
| FastMCP setup + mount под FastAPI | 0.5 дня | Новый `packages/atp-dashboard/.../mcp_server.py` |
| Auth: JWT через stdio env или SSE Authorization header | 1 день | Переиспользует существующий auth middleware |
| Tools: `list_tournaments`, `get_tournament`, `join_tournament` | 1 день | Тонкая обёртка над `TournamentService` |
| Tools: `get_current_state`, `make_move`, `get_history`, `leave` | 1-2 дня | Плюс маппинг `RoundState` в MCP response |
| Notifications: subscribe на sevent pub-sub + push через MCP | 1-2 дня | Самое нестандартное — MCP notifications API |

**Итого фаза 2: 4-6 дней.**

### Фаза 3 — минимальный REST admin

| Задача | Оценка |
|---|---|
| 5-6 read-only endpoint'ов | 2-3 дня |
| Удаление/архивация 501-stubs | 0.5 дня |
| Обновление OpenAPI schema | 0.5 дня |

**Итого фаза 3: 3-4 дня.**

### Фаза 4 — надёжность

| Задача | Оценка |
|---|---|
| Background worker для дедлайнов раундов | 1-2 дня |
| Reconnect handling (игрок упал, MCP переподключается к своему участию) | 2-3 дня |
| Idempotent `make_move` (защита от повторного submit после сетевого blip'а) | 1 день |

**Итого фаза 4: 4-6 дней.**

### Фаза 5 — e2e и dry-run

| Задача | Оценка |
|---|---|
| E2E тест "2 mock-игрока играют PD через MCP" | 2 дня |
| E2E тест сценариев: таймаут, дисконнект, reconnect | 2-3 дня |
| Dry-run турнир внутри команды: 2-3 Claude-инстанса | 1-2 дня |
| Багфиксы после dry-run | буфер 3-5 дней |

**Итого фаза 5: 5-10 дней.**

### Фаза 6 — benchmark MCP-фасад (параллельный трек, не в критпути)

| Задача | Оценка |
|---|---|
| Tools: list/get/start_run/status/leaderboard | 2-3 дня |
| Документация + demo-config для Claude Desktop | 1-2 дня |

**Итого фаза 6: 3-5 дней.** Делается **параллельно** фазам 2-5,
когда есть подходящий разработчик.

### Суммарная оценка до публичного турнира

| | Оптимистично | Реалистично |
|---|---|---|
| Фаза 0 (блокер) | 4 дня | 6 дней |
| Фаза 1 (runtime) | 8 дней | 12 дней |
| Фаза 2 (MCP games) | 4 дня | 6 дней |
| Фаза 3 (REST admin) | 3 дня | 4 дня |
| Фаза 4 (надёжность) | 4 дня | 6 дней |
| Фаза 5 (e2e + dry-run) | 5 дней | 10 дней |
| **Итого** | **~4 недели** | **~6-7 недель** |

Бенчмарк MCP-фасад (фаза 6) идёт параллельно и не влияет на срок.

---

## Порядок работ и приоритеты

1. **Issue 1 (IDOR fix)** — обязательный блокер. До фикса нельзя
   публично запускать ни benchmark'и, ни турниры.
2. **ADR** "MCP for games, REST for benchmarks" — фиксация решения
   до начала кода, чтобы не было архитектурного drift'а.
3. **Game runtime service layer** (фаза 1) — основа. 80% сложности
   проекта. Тестируется без транспортного слоя.
4. **MCP-сервер для игр** (фаза 2) — тонкий фасад.
5. **Минимальный REST admin** (фаза 3) — параллельно фазе 2.
6. **Надёжность** (фаза 4) — дедлайны, reconnect, idempotency.
7. **E2E и dry-run** (фаза 5) — два-три Claude-инстанса играют друг
   с другом, ловля багов.
8. **Публичный турнир**.
9. **Benchmark MCP-фасад** (фаза 6) — параллельный трек, после того
   как игровой MCP обкатан. Demo-путь для не-игровых сценариев.

---

## Рекомендуемый vertical slice на старте фазы 1+2

Не начинать с "полного runtime + все tools + все notifications".
Начать с **thin vertical slice**, который работает end-to-end:

1. Один тип игры — Prisoner's Dilemma.
2. Ровно два игрока.
3. Три раунда (фиксировано, без дедлайнов).
4. Без матчмейкинга — оба игрока стартуют турнир одновременно вручную.
5. Без persistence в середине — только в памяти, финальный результат
   пишется в БД в конце.
6. MCP tools: только `join_tournament`, `get_current_state`,
   `make_move`.
7. Notifications: только `round_started` и `tournament_completed`.

Когда это работает через Claude Desktop с мок-оппонентом (тот же
Claude в другом чате, или простой tit-for-tat Python-скрипт через
`MCPAdapter` в роли клиента) — дальше наращиваем недостающие куски.

Это даёт **проверяемую точку опоры**, после которой любая
дальнейшая работа измеряется конкретным прогрессом, а не "проценты от
плана".

---

## Что делать с существующим `MCPAdapter`

Интересная симметрия: в `packages/atp-adapters/atp/adapters/mcp/` уже
есть **MCP-клиент**, которым ATP тестирует чужих MCP-агентов. Тот же
клиент может **ходить на ATP-MCP-сервер** с минимальной адаптацией.

Это даёт бесплатно:

- **Python SDK-like опыт** для игроков, которые не хотят использовать
  MCP через LLM-клиент: можно написать Python-скрипт, использующий
  `MCPAdapter` напрямую, и он будет играть в турнире. Это тот же
  код, который тестирует MCP-агентов — симметрично и приятно.
- **Собственный test harness** для e2e тестов: один инстанс сервиса
  с MCP-сервером + N инстансов `MCPAdapter` в роли игроков.

Не тратить дополнительное время на "новый Python MCP client для
игроков". Переиспользовать существующий.

---

## Ссылки на код и связанные документы

- Tournament models:
  `packages/atp-dashboard/atp/dashboard/tournament/models.py`
- 501-stubs (будут удалены):
  `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py:74-117`
- Существующий MCP-клиент:
  `packages/atp-adapters/atp/adapters/mcp/adapter.py`,
  `packages/atp-adapters/atp/adapters/mcp/transport.py`
- Game environments library: `game-environments/`
- Local push-model runner: `atp-games/`
- FastMCP docs: https://github.com/jlowin/fastmcp
- MCP specification: https://spec.modelcontextprotocol.io
- Issue 1 (IDOR) — предварительный блокер:
  `docs/atp-issues-ownership-and-buffer.md`
- Related: `docs/atp-auth-ratelimit-sdk.md` (текущая auth/ratelimit
  схема, которую переиспользуем для MCP SSE транспорта)

---

## TL;DR

- **REST benchmark API** остаётся как есть (SDK-аудитория, tight work
  loop). Фиксим Issue 1/2 и не трогаем.
- **Tournament gameplay** строится **MCP-first**, через тонкий фасад
  поверх protocol-agnostic service layer. Никакого REST gameplay'а.
  501-stubs в `tournament_api.py` удалить.
- **Композиция MCP-серверов** (atp + memory + filesystem + thinking)
  даёт LLM-игрокам радикально низкий порог входа: одна страница
  промпта + 15 строк JSON конфига вместо полноценной Python-обвязки.
- **Server-push notifications** вместо polling'а — killer-feature
  для turn-based игр с дедлайнами.
- **Минимальный REST admin** (read-only: list/get/leaderboard/rounds)
  для дашборда и исследователей.
- **Benchmark MCP-фасад** — параллельный трек для demo-пути "Claude,
  прогони benchmark 42", не в критпути турниров.
- **Оценка до публичного турнира**: ~4-7 недель с учётом блокера
  (Issue 1), фаз 1-5 и buffer'а на dry-run-багфиксы.
- **Старт**: thin vertical slice — PD, 2 игрока, 3 раунда,
  3 tools, 2 notifications. Расширение после того, как этот slice
  работает end-to-end через Claude Desktop.
- **Переиспользовать** существующий `MCPAdapter` как клиент для e2e
  тестов и для Python-игроков, не строить второй клиент.

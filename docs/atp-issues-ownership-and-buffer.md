# ATP Platform — найденные проблемы и предложения по исправлению

Найдено в ходе анализа pull-модели взаимодействия SDK с платформой.
Документ фиксирует две независимые проблемы — одну security (IDOR), одну
correctness (buffer loss при re-login) — и предлагает фиксы.

Дата: 2026-04-10.

## Status (2026-04-10)

| Item | Статус | Коммит |
|---|---|---|
| **Issue 1 — IDOR в benchmark API** | ✅ RESOLVED | `e46a98b` — все 6 run-эндпоинтов требуют `Depends(get_current_user)` + ownership check, `Run.user_id` NOT NULL с idempotent backfill, 3 legacy run'а атрибутированы admin (id=1) |
| **Issue 2 Fix A — SDK `on_token_expired` + `drain()`** | ✅ RESOLVED | `eb38951` — `AsyncATPClient.on_token_expired` callback (replays once, no loop), `BenchmarkRun.drain()`/`drain_sync()`, прокинуто через sync `ATPClient`, 11 новых тестов |
| **Issue 2 Fix B — server-side lease-based dispatch** | ⏳ DEFERRED | Не критично пока буфер маленький и Fix A покрывает re-login сценарий. Возврат — когда увидим реальные дыры в `task_results` (пропущенные task_index'ы) |
| **Issue 2 Fix C — idempotent submit (409 → graceful 200)** | ⏳ DEFERRED | Уже частично сделано (409 на дубль). Полный graceful 200 — low priority, переезжает в общий MCP backlog item B |

---

## Issue 1 — IDOR: ownership run'ов не проверяется

### Серьёзность

**High** (security). Tracker label: `security`, `idor`, `benchmark-api`.

### Суть

Эндпоинты жизненного цикла run'а не проверяют, что JWT-пользователь
является владельцем run'а. JWT валидируется middleware'ом (иначе запрос
не дойдёт до handler'а и не попадёт в rate-limiter), но сам `user_id` из
токена **нигде не сравнивается** с `Run.user_id`.

Затронутые эндпоинты в `packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py`:

- `GET  /api/v1/runs/{run_id}/next-task` (`benchmark_api.py:196`)
- `POST /api/v1/runs/{run_id}/submit`   (`benchmark_api.py:294`)
- `GET  /api/v1/runs/{run_id}/status`   (`benchmark_api.py:366`)
- `POST /api/v1/runs/{run_id}/cancel`   (`benchmark_api.py:410`)
- `POST /api/v1/runs/{run_id}/events`   (`benchmark_api.py:436`)

Плюс `POST /api/v1/benchmarks/{id}/start` (`benchmark_api.py:161`) — при
создании run'а `Run.user_id` **не проставляется вообще**, поле остаётся
`NULL`. То есть исторические run'ы тоже нельзя будет корректно проверить
на ownership без миграции.

### Доказательство

1. `Run` имеет колонку `user_id: int | None` (nullable) —
   `packages/atp-dashboard/atp/dashboard/benchmark/models.py:109`.
2. В `start_run` (`benchmark_api.py:181-192`) объект `Run` создаётся без
   `user_id=...`:
   ```python
   run = Run(
       benchmark_id=benchmark_id,
       agent_name=agent_name,
       adapter_type="sdk",
       status=RunStatus.IN_PROGRESS,
       current_task_index=0,
       timeout_seconds=timeout,
       started_at=datetime.now(),
   )
   ```
3. В `next_task`, `submit_result` и остальных handler'ах нет ни одного
   `Depends(get_current_user)` и ни одного `WHERE Run.user_id = :user_id`
   в запросах. Единственные упоминания `user_id` — в aggregation-запросе
   leaderboard'а (`benchmark_api.py:505, 514, 520`).

### Impact

- **Горизонтальная эскалация**: любой залогиненный пользователь может
  получить `next-task`, `submit` или `cancel` для чужого run'а, зная
  только его `run_id` (это автоинкрементный int — тривиально
  перебирается).
- **Порча чужих метрик**: злоумышленник может сабмитить мусорные
  результаты на run другого участника и тем самым портить его score /
  позицию в leaderboard'е.
- **Отмена чужих run'ов**: `POST /runs/{id}/cancel` ставит run в статус
  `cancelled` — DoS участника.
- **Утечка промптов**: `GET /runs/{id}/next-task` возвращает полный
  `ATPRequest` с описанием задачи. Для private benchmark'ов это утечка
  контента suite.
- **Аудит невозможен**: поскольку `user_id` не пишется в `Run` при
  старте, уже нельзя узнать, кто именно стартовал исторические run'ы.

### Как фиксить

Четыре шага, порядок важен.

#### Шаг 1 — проставлять `user_id` при `start_run`

В `benchmark_api.py:166-193`:

```python
from atp.dashboard.auth import get_current_user
from atp.dashboard.models import User

@router.post(
    "/benchmarks/{benchmark_id}/start",
    response_model=RunResponse,
)
@limiter.limit("120/minute")
async def start_run(
    request: Request,
    benchmark_id: int,
    session: DBSession,
    current_user: User = Depends(get_current_user),   # NEW
    timeout: int = Query(default=3600),
    agent_name: str = Query(default=""),
) -> RunResponse:
    bm = await session.get(Benchmark, benchmark_id)
    if bm is None:
        raise HTTPException(404, f"Benchmark {benchmark_id} not found")

    run = Run(
        user_id=current_user.id,                       # NEW
        tenant_id=current_user.tenant_id,              # NEW — для мульти-тенантности
        benchmark_id=benchmark_id,
        agent_name=agent_name,
        adapter_type="sdk",
        status=RunStatus.IN_PROGRESS,
        current_task_index=0,
        timeout_seconds=timeout,
        started_at=datetime.now(),
    )
    session.add(run)
    await session.flush()
    await session.refresh(run)
    return _run_to_response(run)
```

#### Шаг 2 — общий хелпер для ownership checks

Не дублировать код в 5 местах:

```python
# packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py

async def _load_run_for_user(
    session: AsyncSession,
    run_id: int,
    user: User,
) -> Run:
    """Load a run and verify current user owns it.

    Raises 404 if run doesn't exist OR if it belongs to another user —
    404 is preferred over 403 to avoid leaking existence of run_ids.
    """
    run = await session.get(Run, run_id)
    if run is None or run.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )
    return run
```

Возвращаем **404, не 403** — это стандартная практика против
enumeration. Атакующий не должен уметь отличить "run существует, но
чужой" от "run не существует".

#### Шаг 3 — применить ко всем run-эндпоинтам

Пример для `next_task`:

```python
@router.get("/runs/{run_id}/next-task")
@limiter.limit("120/minute")
async def next_task(
    request: Request,
    run_id: int,
    session: DBSession,
    current_user: User = Depends(get_current_user),   # NEW
    batch: int = Query(default=1, ge=1),
) -> Any:
    # Сначала проверяем ownership — 404 если не наш
    await _load_run_for_user(session, run_id, current_user)

    # Атомарный UPDATE c дополнительным фильтром по user_id для
    # защиты от race condition между check и update
    stmt = (
        update(Run)
        .where(
            Run.id == run_id,
            Run.user_id == current_user.id,             # NEW
            Run.status == RunStatus.IN_PROGRESS,
        )
        .values(current_task_index=Run.current_task_index + batch)
        .returning(Run.current_task_index, Run.benchmark_id)
    )
    # ... остальной код как есть
```

Важно: `user_id` добавляется **и** в `_load_run_for_user`, **и** в
`WHERE`-clause атомарного update'а. Без фильтра в update'е между
проверкой и update'ом может пройти, например, админский
`cancel` + `delete`, и update применится к "чужому" ряду.

Применить ту же схему к `submit_result`, `get_status`, `cancel_run`,
`append_events`.

#### Шаг 4 — миграция для существующих run'ов

Alembic migration: сделать `user_id` `NOT NULL` для новых строк, а
исторические run'ы (с `user_id IS NULL`) — либо прибить к системному
"legacy" юзеру, либо пометить статусом `archived` и отдать 410 Gone при
любых операциях:

```python
# packages/atp-dashboard/atp/dashboard/migrations/versions/XXXX_run_ownership.py
def upgrade():
    # 1. проставить legacy user_id
    op.execute(
        "UPDATE benchmark_runs SET user_id = 0 "
        "WHERE user_id IS NULL"
    )
    # 2. enforce NOT NULL для новых
    op.alter_column(
        "benchmark_runs", "user_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
```

Решение, чей `user_id = 0` использовать (seed-юзер "system" или пометка
архивных), — зависит от политики. Для production вероятно стоит не
ронять существующие run'ы, а дать им `user_id = <admin_id>` или
создать dedicated системного юзера.

Напоминание из CLAUDE.md: `create_all()` не ALTER'ит — Alembic-миграция
обязательна.

### Тесты

Минимум три теста в `tests/unit/dashboard/test_benchmark_api.py` (или
`tests/integration/test_benchmark_ownership.py`):

1. **`test_next_task_rejects_other_user`**: Alice стартует run → Bob
   вызывает `GET /runs/{run.id}/next-task` → 404.
2. **`test_submit_rejects_other_user`**: Alice стартует run → Bob
   сабмитит результат → 404, при этом `TaskResult` в БД не создаётся.
3. **`test_cancel_rejects_other_user`**: Alice стартует run → Bob
   `POST /cancel` → 404, run остаётся `in_progress`.
4. **`test_start_run_assigns_user_id`**: проверить, что после
   `start_run` поле `Run.user_id` равно `current_user.id`.
5. Regression для миграции: старые run'ы с `user_id IS NULL`
   корректно обрабатываются (либо архивируются, либо перекидываются
   на legacy user).

### Совместимость

Breaking change для клиентов, у которых уже есть in-flight run'ы, но
токены привязаны к другим юзерам, — это прод-инциденты, которых быть
не должно. Для внешних интеграторов (participants в leaderboard'е)
поведение не меняется: они работают со своими run'ами.

---

## Issue 2 — Потеря in-flight задач из `BenchmarkRun._buffer`

### Серьёзность

**Medium** (correctness). Tracker label: `sdk`, `reliability`,
`data-integrity`.

### Суть

`BenchmarkRun` в SDK держит локальный буфер уже-зачитанных задач:

```python
# packages/atp-sdk/atp_sdk/benchmark.py:30
def __init__(self, client, run_id, benchmark_id, batch_size=1):
    self._buffer: deque[dict[str, Any]] = deque()
    self._exhausted = False
```

Поток:

1. SDK вызывает `GET /runs/{id}/next-task?batch=N`.
2. Сервер **атомарно** делает `UPDATE Run SET current_task_index = current_task_index + N RETURNING ...`
   (`benchmark_api.py:217-223`). Это значит, что в момент ответа задачи
   **уже списаны** со счётчика run'а — с точки зрения сервера они
   "выданы".
3. SDK кладёт N задач в `_buffer`.
4. Если процесс падает / убивается / получает 401 и клиент дропает
   буфер без `submit()` — **эти задачи нигде не сохранены**. Сервер
   считает, что они выданы, но `TaskResult` по ним никогда не
   появится.

### Impact

- **"Дыры" в `task_results`**: `Benchmark.tasks_count == 100`, но
  `len(TaskResult WHERE run_id = X) == 87`. Агрегатные метрики
  (success rate, mean score) считаются по сабмитнутым результатам и
  молча занижаются по знаменателю = tasks_count.
- **Невозможно повторно получить ту же задачу**: `current_task_index`
  уже инкрементирован, `next-task` вернёт следующие задачи, а не
  потерянные. Если агент упал на середине батча — не починить без
  ручного вмешательства в БД.
- **Плохо взаимодействует с re-login**: длинные run'ы обязаны
  реагировать на истечение JWT (60 мин), а наивный обработчик 401
  теряет in-flight буфер.
- **Нет прогресса по `submitted` vs `dispatched`**: нет способа со
  стороны сервера понять, сколько задач "выдано, но не возвращено".
  `status()` показывает только сабмитнутые.

### Воркараунд здесь и сейчас (документационный)

До фикса: в SDK-гайде прописать жёсткое правило — **обрабатывать буфер
до конца, прежде чем делать что-либо, что может потерять объект run'а**
(re-login, рестарт воркера, переключение на другой run). И
рекомендовать **маленький `batch_size`** (равный `concurrency` агента),
а не "впрок".

### Предложения по фиксу

Два уровня: клиентский (SDK) и серверный (платформа). Делать оба.

#### Фикс A — клиентский: drain buffer + автоматический re-login

В `packages/atp-sdk/atp_sdk/benchmark.py` добавить метод, который
гарантированно обрабатывает весь буфер перед точкой потенциальной
потери.

```python
async def drain(self) -> list[dict[str, Any]]:
    """Return and clear the local buffer.

    Call this before re-login, reconnect, or any operation that may
    invalidate the local state. The returned tasks MUST still be
    submitted — the server has already dispensed them and will not
    re-issue.
    """
    pending = list(self._buffer)
    self._buffer.clear()
    return pending
```

Плюс: в `AsyncATPClient` добавить опциональный callback на 401:

```python
class AsyncATPClient:
    def __init__(
        self,
        ...,
        on_token_expired: Callable[[], Awaitable[str]] | None = None,
    ) -> None:
        self._on_token_expired = on_token_expired

    async def _request(self, method, url, **kwargs):
        response = await retry_request(lambda: self._http.request(...),
                                       self._retry_config)
        if (
            response.status_code == 401
            and self._on_token_expired is not None
        ):
            new_token = await self._on_token_expired()
            self.token = new_token
            self._http.headers["Authorization"] = f"Bearer {new_token}"
            response = await retry_request(lambda: self._http.request(...),
                                           self._retry_config)
        return response
```

Использование:

```python
async def renew_token() -> str:
    # читаем из Vault / сервис-аккаунт / secret manager
    return await fetch_service_token()

client = AsyncATPClient(
    platform_url="https://atp.pr0sto.space",
    token=initial_token,
    on_token_expired=renew_token,
)
```

Это **один retry на 401**, без потери буфера: `_request` меняет токен
и переигрывает тот же запрос, буфер `BenchmarkRun` не трогается.

Тесты SDK (`tests/unit/sdk/test_client_renew.py`):

1. Сервер возвращает 401 на первый запрос, 200 на второй →
   `on_token_expired` вызван ровно один раз, ответ — успешный.
2. Сервер возвращает 401 → callback возвращает тот же (невалидный)
   токен → второй 401 → **не зацикливаться**, вернуть последний
   response и дать вызывающему решать.
3. Буфер `BenchmarkRun._buffer` не трогается при re-login.

#### Фикс B — серверный: дозволить re-pull in-flight tasks

Корневая проблема — `next-task` атомарно списывает задачи со счётчика,
и после этого сервер забывает, что они были отданы. Правильная модель
— **lease-based dispatch**:

1. Сервер ведёт не просто `current_task_index`, а таблицу
   "выданных, но не сабмитнутых" задач с TTL лизинга.
2. SDK может запросить re-issue in-flight задач для своего run'а, если
   знает, что потерял их локально.

Минимальная реализация — новая таблица `TaskLease`:

```python
# packages/atp-dashboard/atp/dashboard/benchmark/models.py

class TaskLease(Base):
    """A task dispatched to the client but not yet submitted."""

    __tablename__ = "benchmark_task_leases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("benchmark_runs.id"), nullable=False
    )
    task_index: Mapped[int] = mapped_column(Integer, nullable=False)
    request: Mapped[dict] = mapped_column(JSON, nullable=False)
    leased_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    lease_expires_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("run_id", "task_index", name="uq_lease_run_index"),
        Index("idx_lease_run", "run_id"),
        Index("idx_lease_expires", "lease_expires_at"),
    )
```

Поток:

1. `next-task?batch=N` — сервер атомарно инкрементит счётчик, **но
   одновременно кладёт N записей в `TaskLease` с
   `lease_expires_at = now() + lease_ttl_seconds`** (например, 600
   секунд или min(timeout_seconds, 600)).
2. `submit` — сервер пишет `TaskResult` **и** удаляет соответствующую
   `TaskLease`-запись (в одной транзакции).
3. Новый эндпоинт `GET /api/v1/runs/{id}/in-flight` — возвращает все
   активные `TaskLease` для run'а. SDK вызывает его после реконнекта,
   чтобы восстановить буфер:
   ```python
   async def recover(self) -> list[dict[str, Any]]:
       resp = await self._client._request(
           "GET", f"/api/v1/runs/{self.run_id}/in-flight"
       )
       resp.raise_for_status()
       leases = resp.json()
       for task in leases:
           self._buffer.append(task)
       return leases
   ```
4. Background job (или lazy в `next_task`) реклеймит протухшие лизы —
   отдаёт их обратно в пул для повторной выдачи. Это делает систему
   устойчивой к насмерть-упавшим клиентам.

Уточнения к дизайну:

- **Lease TTL**: `lease_expires_at = now() + min(run.timeout_seconds, 600)`.
  Короткий TTL = быстрый reclaim, но повышает риск двойной обработки,
  если агент реально долго работает над одной задачей. 10 минут —
  разумный дефолт для большинства LLM-задач.
- **Двойная обработка**: `TaskResult` уже имеет
  `UniqueConstraint(run_id, task_index)`, поэтому даже если задача
  реклеймится и сабмитится дважды, БД отвергнет дубль. Это хорошо.
- **Idempotent submit**: чтобы клиент мог безопасно повторять submit
  после network blip'а, нужно 409 (или 200 noop) вместо 500 при
  попытке второго submit'а того же `(run_id, task_index)`. Сейчас это
  срабатывает как generic integrity error → 500.

#### Фикс C — idempotent submit (параллельно с A и B)

`benchmark_api.py:294` — `submit_result`. Добавить обработку
`IntegrityError`:

```python
from sqlalchemy.exc import IntegrityError

@router.post("/runs/{run_id}/submit")
@limiter.limit("120/minute")
async def submit_result(
    request: Request,
    run_id: int,
    data: SubmitRequest,
    session: DBSession,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    await _load_run_for_user(session, run_id, current_user)

    # Проверяем, не сабмитили ли уже
    existing = await session.execute(
        select(TaskResult).where(
            TaskResult.run_id == run_id,
            TaskResult.task_index == data.task_index,
        )
    )
    if existing.scalar_one_or_none() is not None:
        return {
            "status": "already_submitted",
            "task_index": data.task_index,
        }

    # ... остальная логика как есть
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        return {
            "status": "already_submitted",
            "task_index": data.task_index,
        }
```

Первый SELECT — оптимистичная fast-path проверка, `try/except
IntegrityError` — защита от race'а между двумя параллельными retry'ями
одного submit'а.

### План тестов для Issue 2

- **SDK**: юнит-тест на `drain()` и `recover()`.
- **SDK**: интеграционный тест "fake 401 в середине батча, фабрика
  даёт новый токен, буфер сохраняется, submit проходит".
- **Server**: интеграционный тест "stale lease reclaim" —
  диспетчим задачу, ждём истечения lease, повторный `next-task`
  возвращает ту же задачу.
- **Server**: интеграционный тест "idempotent submit" — сабмит
  дважды для одного `task_index`, второй возвращает
  `already_submitted`, не 500.
- **Server**: regression test на атомарность — параллельные два
  `next-task` не выдают одну и ту же задачу дважды и одновременно
  создают две записи в `TaskLease`.

---

## Порядок работ

Рекомендуемый порядок реализации:

1. **Issue 1, Шаг 1** (`start_run` проставляет `user_id`) — это
   обязательная предподготовка. Без неё все последующие ownership
   checks не имеют смысла.
2. **Issue 1, Шаг 4** (Alembic-миграция для legacy `user_id IS NULL`)
   — делать **до** Шага 3, чтобы не уронить прод в момент деплоя.
3. **Issue 1, Шаги 2-3** (хелпер + apply ко всем эндпоинтам).
4. **Issue 2, Фикс C** (idempotent submit) — маленький, изолированный,
   улучшает retry-семантику для всех клиентов.
5. **Issue 2, Фикс A** (SDK: `on_token_expired` callback) — клиентский
   фикс, не требует изменений сервера, быстро даёт value.
6. **Issue 2, Фикс B** (серверный TaskLease + in-flight endpoint) —
   самый большой кусок, требует миграции, background reclaim job'а и
   нового endpoint'а. Можно отложить до следующей итерации.

Issue 1 — **обязательно до любого публичного запуска benchmark
leaderboard'а** (участники начнут ставить `?run_id=` друг друга).
Issue 2 — важно для надёжности long-running агентов, но не блокер
запуска.

## Ссылки на код

- Benchmark API: `packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py`
  - `start_run`: `:161`
  - `next_task`: `:196`
  - `submit_result`: `:294`
- Run/TaskResult модели: `packages/atp-dashboard/atp/dashboard/benchmark/models.py:97, :156`
- SDK BenchmarkRun: `packages/atp-sdk/atp_sdk/benchmark.py:17`
- SDK retry-политика: `packages/atp-sdk/atp_sdk/retry.py`
- SDK AsyncATPClient: `packages/atp-sdk/atp_sdk/client.py:21`
- Device Flow / JWT: `packages/atp-dashboard/atp/dashboard/auth/__init__.py`,
  `packages/atp-dashboard/atp/dashboard/auth/device_flow.py`

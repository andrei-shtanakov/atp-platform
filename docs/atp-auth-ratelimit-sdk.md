# ATP Platform — Аутентификация, Rate Limiting и SDK

Детальный разбор pull-модели взаимодействия агента с ATP платформой:
JWT + GitHub Device Flow, HTTP rate limiting, клиентский SDK `atp-platform-sdk`.

---

## 1. Аутентификация

### JWT — формат и выдача

`packages/atp-dashboard/atp/dashboard/auth/__init__.py`:

- **Алгоритм**: HS256 (симметричный)
- **Секрет**: `ATP_SECRET_KEY` из env. Если не задан — генерируется `secrets.token_urlsafe(32)` при старте и пишется warning. В production это фатально (без `ATP_DEBUG=true` сервер проверяет это в `model_post_init`).
- **TTL**: `ATP_TOKEN_EXPIRE_MINUTES`, по умолчанию **60 минут**
- **Payload**:
  ```python
  {"sub": user.username, "user_id": user.id, "exp": <expires>}
  ```
- Декод на входящем запросе через FastAPI dependency `get_current_user` — `jwt.decode(token, SECRET_KEY, algorithms=["HS256"])`, затем по `sub` подтягивается `User` из БД. Параллельно **`JWTUserStateMiddleware`** (см. секцию "Rate limiting") best-effort декодит тот же Bearer на ASGI-уровне ДО dependency injection и кладёт `user_id` в `request.state`, чтобы slowapi мог считать rate-limit per user, а не per IP. До коммита `e97b2c9` (2026-04-10) этого middleware не было, и rate-limiter всегда уходил в IP-fallback.

Никаких refresh-токенов, rotation, или JWKS. Один JWT на 60 минут, при истечении — логин заново.

### GitHub Device Flow (RFC 8628)

Это стандартный OAuth 2.0 Device Authorization Grant. Нужен, когда CLI-клиент не может открыть локальный колбэк (headless-среды, контейнеры).

**Серверная часть** — `packages/atp-dashboard/atp/dashboard/auth/device_flow.py` + `routes/device_auth.py`:

**Шаг 1 — инициация**: `POST /api/auth/device`

- Сервер идёт на `https://github.com/login/device/code` с `client_id=ATP_GITHUB_CLIENT_ID` и `scope=read:user user:email`
- GitHub возвращает `device_code`, `user_code`, `verification_uri`, `expires_in` (обычно 900 сек = 15 мин), `interval` (обычно 5 сек)
- Сервер кладёт `device_code → {user_code, interval, github_access_token: None}` в `AuthStateStore` с TTL = `expires_in`. `AuthStateStore` — унифицированное транзиентное хранилище (in-memory по умолчанию), общее для SSO/SAML/DeviceFlow.
- Клиенту возвращается `{device_code, user_code, verification_uri, expires_in, interval}`
- **Rate limit**: `5/minute` — защита от брут-форса device_code

**Шаг 2 — пользователь идёт в браузер**: открывает `https://github.com/login/device`, вводит `user_code`, подтверждает scope `read:user user:email`.

**Шаг 3 — poll**: `POST /api/auth/device/poll` с `{"device_code": "..."}`

- Сервер дёргает `https://github.com/login/oauth/access_token` с `grant_type=urn:ietf:params:oauth:grant-type:device_code`
- Маппинг ответов GitHub:

  | GitHub response | HTTP status | Что значит |
  |---|---|---|
  | `authorization_pending` / `slow_down` | **428** | Юзер ещё не подтвердил → клиент спит `interval` и повторяет |
  | `expired_token` | **410** | Код протух, начинай заново |
  | не найден в store | **404** | |
  | `access_denied` / другое | **400** | |
  | `access_token` получен | **200** + `Token` | Успех |

- При успехе сервер ещё раз дёргает GitHub API: `GET /user` и (если primary email пустой) `GET /user/emails`, чтобы вытянуть `login`, `email`, `name`
- Затем вызывает `complete_auth()` — shared post-auth pipeline
- **Rate limit**: `5/minute`

### Post-auth pipeline (`packages/atp-dashboard/atp/dashboard/auth/post_auth.py`)

Один и тот же код используется Device Flow, OIDC SSO, SAML:

1. **JIT provisioning** (`_provision_user`): ищем юзера по `(tenant_id, email)`. Если нет — создаём с `hashed_password="SSO_USER_NO_LOCAL_PASSWORD"`, `is_active=True`, `is_admin=False`. Если есть — обновляем `updated_at` и username при необходимости.
2. **Assign roles** (`_assign_roles`): опционально мапит IdP-группы на ATP Roles через RBAC. Device Flow сейчас передаёт `role_names=None`, так что роли не навешиваются — только на первого юзера сработает auto-admin (логика в RBAC init).
3. **Create JWT**: `create_access_token({"sub": username, "user_id": id}, expires_delta=60min)` → `Token(access_token=...)`

### SDK-клиент Device Flow (`packages/atp-sdk/atp_sdk/auth.py`)

`login(platform_url, open_browser=True)`:

```python
# 1. POST /api/auth/device
resp = http.post(f"{base}/api/auth/device")
# обработка 501 — GitHub OAuth не сконфигурирован на сервере

# 2. Показать user_code
display_code = f"{user_code[:4]}-{user_code[4:]}"  # XXXX-XXXX для удобства
print(f"Open {verification_uri} and enter code: {display_code}")
webbrowser.open(verification_uri)  # auto-open

# 3. Polling loop до deadline = now + expires_in
while time.monotonic() < deadline:
    time.sleep(interval)
    resp = http.post(f"{base}/api/auth/device/poll", json={"device_code": ...})
    if resp.status_code == 428: continue    # pending
    if resp.status_code == 410: raise       # expired
    token = resp.json()["access_token"]
    save_token(token, platform_url=base)
    return token
```

Токен сохраняется в `~/.atp/config.json`:

```json
{
  "token": "<latest>",
  "tokens": {
    "https://atp.pr0sto.space": "<jwt>",
    "http://localhost:8000": "<jwt>"
  }
}
```

Оба поля пишутся сразу — `token` для обратной совместимости, `tokens[platform_url]` для мульти-инстансных сценариев.

### `ATP_TOKEN` и приоритет источников

В `AsyncATPClient.__init__` (`client.py:41-45`):

```python
self.token = (
    token                                             # 1. явный аргумент
    or os.environ.get("ATP_TOKEN")                    # 2. env var
    or load_token(platform_url=self.platform_url)     # 3. ~/.atp/config.json
)
```

`load_token()` сперва пытается `tokens[platform_url]`, потом fallback на legacy `token`. Токен навешивается в дефолтный `Authorization: Bearer <jwt>` хедер `httpx.AsyncClient`.

Три типовых сценария:

| Сценарий | Как получить токен |
|---|---|
| Интерактивный CLI/ноутбук | `client.login()` → браузер с device code → сохранение в `~/.atp/config.json` |
| CI/cron без браузера | экспорт `ATP_TOKEN=...` (сгенерить один раз через interactive login с dev-машины) |
| Долгоживущий сервис | передать `ATPClient(token=...)` из секрет-менеджера (Vault, AWS SSM, k8s Secret) |

**Потолок 60 минут** означает, что long-running runner должен быть готов к 401 → перевыдать токен. С коммита `eb38951` (2026-04-10) `AsyncATPClient` принимает опциональный `on_token_expired: Callable[[], Awaitable[str]]` callback: на 401 он один раз вызывается, обновляется `Authorization`-заголовок, запрос проигрывается ровно один раз (защита от петли при невалидном новом токене). Параметр прокинут и через sync-обёртку `ATPClient`. Дополнительно `BenchmarkRun.drain()` / `drain_sync()` позволяют клиенту извлечь in-flight буфер задач перед re-login или реконнектом, чтобы уже-выданные сервером задачи не терялись. Полная схема — `docs/atp-issues-ownership-and-buffer.md` §Issue 2 Fix A.

---

## 2. Rate limiting

### Где

`packages/atp-dashboard/atp/dashboard/v2/rate_limit.py` — слой поверх библиотеки **slowapi** (обёртка над `limits` для FastAPI).

### Ключевая функция — как считается "кто стучится"

```python
def get_rate_limit_key(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    if user_id is not None:
        return f"user:{user_id}"
    forwarded = request.headers.get("x-forwarded-for")
    ip = forwarded.split(",")[0].strip() if forwarded else get_remote_address(request)
    return f"ip:{ip}"
```

Правила:

- Если в `request.state` положен `user_id` — лимит считается **per user** (`"user:42"`). Это было задумано как главный кейс для SDK: каждый залогиненный участник должен иметь свой бакет, а не делить его с соседями по IP.
- Иначе — по IP. Уважается `X-Forwarded-For` (для nginx-прокси / Namecheap VPS, где контейнер за reverse proxy). Берётся **первый** элемент в списке — это нужно фиксить, если ты за несколькими прокси (иначе клиент может спуфить `X-Forwarded-For`).

> **✅ Состояние с 2026-04-10 (`e97b2c9`):** `JWTUserStateMiddleware` в `packages/atp-dashboard/atp/dashboard/v2/rate_limit.py` декодит Bearer-токен на ASGI-уровне (между CORS и SlowAPIMiddleware в Starlette LIFO-стеке), и при валидном токене с `user_id`-claim'ом пишет `request.state.user_id`. Все ошибки декода silent-fail — реальная аутентификация по-прежнему обеспечивается `get_current_user` на защищённых роутах. Теперь slowapi реально считает rate-limit per user для аутентифицированных запросов, а не схлопывает разных участников за одним NAT в общий IP-bucket.

### Лимиты по категориям (env-config)

Из `dashboard/v2/config.py:102-125`:

| Env var | Default | Применение |
|---|---|---|
| `ATP_RATE_LIMIT_ENABLED` | `true` | Глобальный kill-switch |
| `ATP_RATE_LIMIT_DEFAULT` | `60/minute` | Все endpoint'ы без явного декоратора |
| `ATP_RATE_LIMIT_AUTH` | `5/minute` | `/auth/device`, `/auth/device/poll` — защита от брута |
| `ATP_RATE_LIMIT_API` | `120/minute` | `/api/v1/benchmarks`, `/runs/*` — бизнес-API для SDK |
| `ATP_RATE_LIMIT_UPLOAD` | `10/minute` | `/api/suite-definitions/upload` (YAML push) |
| `ATP_RATE_LIMIT_STORAGE` | `memory://` | `memory://` (single-process) или `redis://host:port` (для нескольких Uvicorn workers) |

### Как лимиты применяются к handler'ам

Декораторы выставляются на уровне модуля (`@limiter.limit("120/minute")`). Хитрость в том, что на import time `limiter` ещё не знает config'а — он создаётся заранее как disabled:

```python
limiter: Limiter = Limiter(key_func=get_rate_limit_key, enabled=False)
```

На startup вызывается `create_limiter(config)`, который **in-place** мутирует уже существующий объект:

- `limiter.enabled = config.rate_limit_enabled`
- `limiter._default_limits = [LimitGroup(config.rate_limit_default, ...)]`
- `limiter._storage = storage_from_string(config.rate_limit_storage)`

Это делается именно мутацией, потому что декораторы уже захватили ссылку на `limiter` при импорте — пересоздать инстанс нельзя, он останется связанным с роутами.

**Важный исторический момент** (см. comment в `rate_limit.py:60-66`): раньше `_default_limits` строилось как список raw-строк, а `slowapi` в middleware делает `itertools.chain(*self._default_limits)`. Итерация строки по символам падала с `AttributeError` глубоко в `__evaluate_limits`, затем перехватывалась slowapi's exception handler'ом (который ждёт `RateLimitExceeded` с атрибутом `.detail`), и получался криптический **HTTP 500 на любом нерасукрашенном route**. Это был **production-баг 2026-04-10**. Сейчас правильно — каждый лимит упакован в `LimitGroup(...)` с полной сигнатурой `__init__`, как это делает сам `Limiter.__init__`.

### 429-ответ

`rate_limit_exceeded_handler` отдаёт JSON и хедер `Retry-After`:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 60
Content-Type: application/json

{
  "error": "rate_limit_exceeded",
  "detail": "Rate limit exceeded: 120 per 1 minute",
  "retry_after": 60
}
```

### Как SDK реагирует на 429

`atp_sdk/retry.py:109-125` — специальная ветка для 429:

```python
if status == 429:
    if attempt < config.max_retries:
        retry_after_raw = response.headers.get("Retry-After")
        try:
            delay = min(float(retry_after_raw or 0), config.max_retry_delay)
        except ValueError:
            delay = _jitter_delay(attempt, config)
        await anyio.sleep(delay)
    continue
```

То есть SDK **уважает `Retry-After`**, но капает его на `max_retry_delay` (30 сек по умолчанию), чтобы сервер не мог заставить клиента ждать 10 минут. Если хедера нет — exponential backoff с full jitter: `random(0, min(30, 1.0 * 2^attempt))`. По умолчанию 3 ретрая.

### Практические последствия для агента

- С `120/minute` ты можешь делать ~2 RPS на JWT. Типичный цикл `next-task → работа агента → submit` — это 2 запроса на задачу, то есть **60 задач/мин на один SDK-клиент**. Если агент быстрее — нужно `next_batch(N)` (один HTTP call за N задач) или несколько параллельных runs с разными JWT.
- `memory://` storage означает, что счётчики **per-process**: если запустить несколько uvicorn workers, каждый считает свой — эффективный лимит × workers. Для честной работы за несколькими воркерами нужен `redis://`.
- 5/minute на auth — нельзя флудить `login()` в тестах. Переиспользуй токен через `ATP_TOKEN` или fixture.

---

## 3. SDK `atp-platform-sdk`

PyPI: `atp-platform-sdk` (версия 2.0.0). Пакетное имя при импорте — `atp_sdk`.

### Структура

```
packages/atp-sdk/atp_sdk/
├── __init__.py    — публичные экспорты: ATPClient, AsyncATPClient, BenchmarkRun
├── auth.py        — Device Flow, ~/.atp/config.json
├── client.py      — AsyncATPClient (основной)
├── sync.py        — ATPClient (sync wrapper над AsyncATPClient)
├── benchmark.py   — BenchmarkRun (async/sync итератор задач)
├── models.py      — pydantic: BenchmarkInfo и т.п.
└── retry.py       — retry_request + RetryConfig (exponential backoff + full jitter)
```

### `AsyncATPClient` — async-первый слой

Конструктор (`client.py:30`):

```python
AsyncATPClient(
    platform_url="http://localhost:8000",
    token=None,                    # resolution: arg → ATP_TOKEN → ~/.atp/config.json
    max_retries=3,
    retry_backoff=1.0,             # база для exp backoff
    max_retry_delay=30.0,          # cap
    retry_on_timeout=True,
    timeout=30.0,                  # httpx request timeout
)
```

Под капотом — `httpx.AsyncClient` с `base_url`, `Authorization: Bearer <jwt>` хедером, и общим timeout. Каждый запрос идёт через `_request()` → `retry_request()`.

**Публичный API:**

| Метод | Endpoint | Возвращает |
|---|---|---|
| `list_benchmarks()` | `GET /api/v1/benchmarks` | `list[BenchmarkInfo]` |
| `get_benchmark(id)` | `GET /api/v1/benchmarks/{id}` | `BenchmarkInfo` |
| `start_run(id, agent_name="", timeout=3600, batch_size=1)` | `POST /api/v1/benchmarks/{id}/start` | `BenchmarkRun` |
| `get_leaderboard(id)` | `GET /api/v1/benchmarks/{id}/leaderboard` | `list[dict]` |
| `close()` | — | закрывает httpx клиент |

Поддерживает `async with AsyncATPClient(...) as client:` — корректно чистит connection pool.

### `BenchmarkRun` — итератор задач

Ключевая абстракция (`benchmark.py:17`). Держит `client`, `run_id`, `batch_size`, локальный `_buffer: deque` и флаг `_exhausted`.

**Метод `_fetch_batch(batch)`** — один HTTP call:

```python
resp = await self._client._request(
    "GET",
    f"/api/v1/runs/{self.run_id}/next-task",
    params={"batch": n} if n > 1 else None,
)
if resp.status_code == 204:
    self._exhausted = True
    return []
data = resp.json()
if isinstance(data, dict):    # backward compat: batch=1 возвращает dict, не list
    data = [data]
return data
```

Сервер гарантирует атомарность через `UPDATE ... RETURNING current_task_index` (см. `benchmark_api.py:217`), так что **можно запускать несколько `BenchmarkRun` с одним `run_id` в параллель**, и они не получат одну и ту же задачу дважды.

**Варианты итерации:**

```python
# 1. Async one-by-one (батчи буферизуются)
async for task in run:
    response = await my_agent(task)
    await run.submit(response, task_index=task["metadata"]["task_index"])

# 2. Sync one-by-one (работает только ВНЕ running event loop)
for task in run:
    response = my_agent(task)
    run.submit(response, task_index=task["metadata"]["task_index"])

# 3. Explicit batch pull
batch = await run.next_batch(10)
results = await asyncio.gather(*[my_agent(t) for t in batch])
for task, resp in zip(batch, results):
    await run.submit(resp, task_index=task["metadata"]["task_index"])
```

Sync-итерация внутри async-контекста поднимает `RuntimeError` — иначе вложенные loop'ы заблокируют друг друга.

**Submit / status / cancel / leaderboard** — мапятся на `POST /runs/{id}/submit`, `GET /runs/{id}/status`, `POST /runs/{id}/cancel`, `GET /benchmarks/{id}/leaderboard`.

**Event streaming** — `run.emit(task_index, event_dict)` шлёт `POST /runs/{id}/events` во время выполнения задачи, чтобы платформа могла показывать прогресс в дашборде. Лимит сервера — 1000 событий на run.

### `ATPClient` — sync wrapper (`sync.py`)

Не просто `asyncio.run()` на каждый вызов — это background thread с собственным event loop:

```python
self._loop = asyncio.new_event_loop()
self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
self._thread.start()

def _run(self, coro):
    future = asyncio.run_coroutine_threadsafe(coro, self._loop)
    return future.result()
```

Зачем так:

1. **Thread-safe**: можно делить один `ATPClient` между тредами (pytest с `xdist`, Flask worker pool). `run_coroutine_threadsafe` — единственный корректный способ пушить корутину в loop из чужого потока.
2. **Переиспользование connection pool**: `httpx.AsyncClient` живёт в том же loop'е между вызовами, keep-alive работает. `asyncio.run()` на каждый вызов убил бы пул.
3. **Нельзя сделать sync API без нового loop'а**, если `httpx.AsyncClient` уже создан (он привязан к loop'у, в котором создан).

Sync-эквиваленты: `submit_sync()`, `status_sync()`, `cancel_sync()`, `leaderboard_sync()`, `next_batch_sync()`, `emit_sync()`.

### Retry-логика (`retry.py`)

Что ретраится:

| Ситуация | Поведение |
|---|---|
| `httpx.TransportError` (ConnectError, RemoteProtocolError) | Retry |
| `httpx.TimeoutException` | Retry, если `retry_on_timeout=True` |
| `502 Bad Gateway`, `503 Service Unavailable`, `504 Gateway Timeout` | Retry |
| `429 Too Many Requests` | Retry, уважает `Retry-After`, capped `max_retry_delay` |
| **`500 Internal Server Error`** | **НЕ ретраится** — считается bug, а не transient |
| `401`, `403`, `404`, прочие 4xx | НЕ ретраится |

Backoff — **full jitter**: `random(0, min(max_retry_delay, retry_backoff * 2^attempt))`. Это лучше, чем "equal jitter" или фиксированный backoff, потому что разносит retry-волны разных клиентов во времени и избегает thundering herd на восстановление сервера.

После исчерпания ретраев:

- Transport errors → raise оригинального исключения
- Status-based → возвращает последний response (пусть вызывающий решает, что делать с 503)

### Типичный цикл использования

```python
from atp_sdk import ATPClient

client = ATPClient(platform_url="https://atp.pr0sto.space")

# один раз — интерактивно
if not client.token:
    client.login()  # Device Flow, сохраняет в ~/.atp/config.json

benchmarks = client.list_benchmarks()
bm = next(b for b in benchmarks if b.name == "my-bench")

run = client.start_run(bm.id, agent_name="gpt-4o-mini-v1", batch_size=5)

for task in run:
    # ATPRequest dict: {task_id, task, constraints, metadata}
    description = task["task"]["description"]
    constraints = task["constraints"]

    # твоя логика — любой фреймворк, любая модель
    result = my_agent.run(description, max_steps=constraints.get("max_steps", 20))

    response = {
        "task_id": task["task_id"],
        "status": "completed",
        "artifacts": [...],
        "metrics": {"tokens": 1234, "steps": 5, "cost_usd": 0.012},
    }
    run.submit(response, task_index=task["metadata"]["task_index"])

print(run.status())       # финальный статус run'а
print(run.leaderboard())  # позиция в leaderboard'е
```

---

## TL;DR

- **JWT**: HS256, 60 минут, payload `{sub, user_id, exp}`. Выдаётся `complete_auth()` — JIT provisioning пользователя в БД + `create_access_token`. Нет refresh.
- **Device Flow**: стандарт RFC 8628 поверх GitHub OAuth. Сервер хранит state в `AuthStateStore`, клиент polls 428 → 200. SDK оборачивает это в `login()` с автооткрытием браузера и сохранением в `~/.atp/config.json`.
- **`ATP_TOKEN`**: env var, второй по приоритету источник токена в `AsyncATPClient`. Удобен для CI.
- **Rate limiting**: `slowapi`, ключ `user:{id}` из JWT либо `ip:{X-Forwarded-For}`. 120/min на API, 5/min на auth, 10/min на upload. Единый `limiter` мутируется на старте через `create_limiter()`, чтобы уже-навешанные декораторы подхватили config — это и был источник прод-бага 2026-04-10, который исправили переходом на `LimitGroup`. При 429 сервер шлёт `Retry-After`, SDK его уважает.
- **SDK**: `AsyncATPClient` (основной) + `ATPClient` (sync wrapper через background thread с собственным loop). `BenchmarkRun` — async/sync итератор с буферизацией, `next_batch(N)` и `drain()`/`drain_sync()` для извлечения буфера перед re-login. Retry с exponential backoff + full jitter для transport errors, 5xx (только 502/503/504), 429. 500 не ретраится. **401 → optional `on_token_expired` callback (с `eb38951`)**: один раз вызывается перевыдача токена, запрос играется один раз заново; без callback'а 401 возвращается клиенту как есть.

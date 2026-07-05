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

  - **Phase 1 (2026-06-13): code-review вертикаль — тонкий срез.** Планы:
    [`docs/superpowers/plans/2026-06-13-r07-phase1-code-review-eval.md`](docs/superpowers/plans/2026-06-13-r07-phase1-code-review-eval.md)
    (atp, PR #171) + `../arbiter/2026-06-13-r07-phase1-arbiter-rerank-plan.md`.
    - [x] atp-срез: vendored контракт + claude_code shim (CLI-adapter) + 2 кейса
      (clean/moderate, SEC-011) + `report_benchmark` reporter + smoke. Ветка `r07/code-review-eval`.
    - [x] **Задача 6 — pipe-check (НЕ бенчмарк):** ✅ 2026-06-17. Прогон против живого `claude` + судьи,
      труба пропускает реальный сигнал — подтверждено. Детали ниже (см. отметку у «платного pipe-check»).
    - [ ] **arbiter-план** (reader + re-rank + A/B) — написан, не исполнен; после go. **Go получен** (сигнал дискриминирует).

  - **Eval-improvements (план от 2026-06-14, NEXT SESSION):** ревью двух рецензентов сошлось,
    зафиксировано в [`../_cowork_output/10-code-review-eval-improvements-proposals.md`](../_cowork_output/10-code-review-eval-improvements-proposals.md) (v2).
    Порядок исполнения (routing-сигнал идёт ТОЛЬКО из `critical_pass_rate`; рубрика не гейтит):
    - [x] **P3 (ПЕРВЫМ, ~0.5д) — strict `Finding`-валидация + `malformed_rate`.** ✅ Сделано.
      `Finding` pydantic (req `rule_id`/`anchor`/`severity` Literal[critical|major|minor], `extra=ignore`);
      `strict` глобально (одна невалидная находка малформит весь вывод, без lenient-режима).
      2 пути провала сведены в ОДИН исход через `grade_findings()` (parse+validate+match):
      `MatchResult.malformed: bool` отдельно от `critical_pass`; оба консьюмера
      (native `FindingsMatchEvaluator` + method `case_evaluator`) зовут единый путь.
      `malformed_rate` → `score_components` (контракт numbers-only, без изменений схемы).
    - [x] **Задача 6 — платный pipe-check** на закалённом гейте (go/no-go). ✅ **СДЕЛАНО 2026-06-15…17.**
      Результаты: `_cowork_output/r07-pipecheck/` (81 отчёта `report_benchmark_*.json` +
      sqlite: `sweep.db`, `p2-filter.db`, `p2-heldout.db`, `arbiter-stats.db`).
      **Итог: GO** — труба различает агентов чисто. code-review `crit_pass`:
      ollama-llama32-1b 0.00 → qwen25-3b 0.07 → qwen25-7b 0.13 → llama32-3b 0.20 →
      codex_cli 0.62 → anthropic_api 0.70 → deepseek 0.77 → qwen25-14b 0.87 → claude_code 0.90.
      req-extraction-ловушка: слабые модели 0.08–0.42, сильные 1.00. Прогнаны 9 агентов
      (`claude_code`, `anthropic_api`, `codex_cli`, `deepseek`, ollama-матрица 1b/3b/7b/14b).
      **Cost:** записан только у `claude_code` = $17.08 (18 прогонов); у остальных `total_cost_usd: null`,
      реальный расход ≈ $24+ (судья + не-учтённые агенты). Сводка: `_cowork_output/r07-pipecheck/SUMMARY.md`.
      Обвязка: `method/run_pipe_check.py` (CLI-адаптер → шим → оркестратор → findings_match →
      `report_benchmark` payload + sqlite). **Остаток:** per-task `score` в отчётах = 0.0 при высоком
      агрегатном `critical_pass_rate` — свериться, что per-task поле несёт не-гейтящую рубрику, а не баг.
    - [ ] **P4 + prefill судьи (~0.5д).** strengths/weaknesses → только локальные логи (numeric-only
      payload). Prefill (anthropic API) — робастность СУДЬИ, отдельный PR от P1.
    - [ ] **P1 (~1д) — batched rubric** через отдельный structured-judge путь в method evaluator
      (НЕ перегружать `LLMJudgeEvaluator`). Батчинг меняет оценки → `rubric_mode` заморожен на серию;
      default `batched`, 1 retry → честный fail.
    - [ ] **Phase-1b:** Тикет B (ablation API-vs-CLI, «харнесс vs API») + codex_cli/aider шимы +
      полный 5-уровневый свип.
    - 3 остаточных вопроса к автору зафиксированы в файле (P1 location, prefill sequencing, ablation framing).

  - **Phase-1b/2 (через БРЕЙНШТОРМ, после pipe-check):** 4 вопроса 2026-06-13 показали,
    что MVP — узкий зонд. Внедряем оси (приоритет в порядке):
    - [x] **#1 структурированный вывод (JSON findings) + `programmatic` critical_check.**
      ✅ 2026-06-19. `programmatic`-детерминизм был у обоих семейств; теперь структурный
      вывод **единообразен**: code-review мигрирован на объектную форму `{"findings":[...]}` —
      во все 15 кейсов добавлен `output_contract` (объектная схема, выровненная с моделью
      `Finding`: req `rule_id/anchor/severity`), а `findings_match` стал object-aware +
      делает `jsonschema`-гейт против `output_contract.schema` (malformed при нарушении),
      зеркаля `json_path`. Голый массив — legacy-fallback. Промпт берётся из
      `output_contract.format_instruction` (единый источник). Примечание: «структурный» в
      этой кодобазе = текст-JSON + schema-валидация (НЕ `ArtifactStructured` через адаптеры —
      это отложено, не нужно для паритета с req-extraction). Ветка `r07/code-review-structured-output`.
      План: `docs/superpowers/plans/2026-06-19-code-review-structured-output.md`.
      **Риск к ре-базлайну (выходной прогон):** массив→объект меняет формат вывода агента,
      а `output_contract` уводит промпт на `GENERIC_ENVELOPE` (теряется строка-персона
      «senior code reviewer», роль остаётся в `instruction`) — проверить эффект на сигнале.
    - [ ] **#4 языковая ось** — в схеме `agent-eval-case` нет поля `language`, а arbiter
      роутит по языку (`features.rs` f[1]/f[16]) → скоры надо разбивать по языку + протянуть
      в `benchmark_runs`. Влияет на валидность роутинга. **Реально не начата — следующая ось.**
    - [x] **#2 correctness-семейство** — `code-review-correctness` (capability `correctness`):
      ✅ сделано — 7 кейсов в `method/cases/code-review/case-code-review-correctness-*`
      (logic/spec/distractor/fp), посеянные логические баги и ловушки на ложные срабатывания.
    - ❌ **#3 проверка использования линтеров — НЕ делаем.** Линтеры детерминированы; LLM
      бенчмаркаем на семантике. Запуск линтера агентом = file_write/exec = возврат проблемы
      fidelity спавнера, от которой ушли через text-out.
    - [ ] **#5 grounding/recency-ось на CLI-ростере (Путь A) — ВАЖНАЯ задача развития.**
      PR #203 (fomih) добавил `run_mode: read_only_corpus` + детерминированный грейдер
      `citation_grounding` (агент обнаруживает/читает файлы корпуса, отсеивает устаревший
      дистрактор, цитирует источник; SHA-256-верифицированный корпус → материализация →
      `Context.tools_endpoint`). Но wired только под `anthropic_api` — **наши CLI-шимы
      (claude_code/codex_cli/pi/opencode) этот режим не умеют**: продуктовый CLI имеет свой
      закрытый tool-набор и не знает про ATP `tools_endpoint`. Путь A: нацелить нативные
      инструменты CLI на **уже материализованную** директорию корпуса (`cwd=workspace_path`
      + confinement), HTTP-эндпоинт для CLI не нужен. Объём: расщепить `CorpusRunPreparer`
      (materialize-only), corpus-режим в `_cli_common.py`, per-CLI confine-флаги, нормализация
      относительных путей цитат, тул-агностичная инструкция, confinement-тест + живой смоук
      (гейт). Per-CLI работа, начинать с `claude_code`. Дизайн:
      [`docs/superpowers/specs/2026-06-21-cli-corpus-grounding-design.md`](docs/superpowers/specs/2026-06-21-cli-corpus-grounding-design.md).

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

- [ ] **EPIC — унификация просмотра/сравнения/истории результатов (исходная цель платформы).**
  Сейчас результаты раздроблены по 3 хранилищам с неравномерным UI, и это «зообарк»:
  1. **`SuiteExecution`** (дашборд-БД) — пишется из `atp test` по умолчанию (`--no-save`
     отключает), включая прогоны **atp-method**. Доступно только через JSON API
     (`/api/executions`, timeline/comparison/analytics-роуты) — **HTML-страницы нет**.
  2. **benchmark `Run`** (таблица `benchmark_runs`, дашборд-БД) — pull-model benchmark
     API/SDK для внешних участников. **Есть UI** (`/ui/runs`, leaderboard).
  3. **arbiter `benchmark_runs`** (кросс-проект) — наш `report_benchmark-v1` payload;
     в ATP не отображается вообще (потребитель — роутинг arbiter).
  Задача: определить канонический «eval run» вид — единая поверхность результат/история/
  сравнение для CLI/method-прогонов (а не только benchmark-платформа); провести аудит
  пересечения `SuiteExecution` ↔ benchmark `Run`, выбрать один дом, мосты/депрекейт для
  второго. Родитель для `/ui/executions` и для R-07-визуализации ниже.
- [~] **R-07: визуализация результатов code-review на дашборде.** Сейчас
  `method/run_pipe_check.py` эмитит `report_benchmark-v1` в JSON/локальный sqlite для
  роутинга arbiter — в дашборде ATP **ничего не рендерится**. Нужен вид для code-review-
  вертикали: по агентам `critical_pass_rate` / `malformed_rate` / `breakpoint_axis_level`,
  свип по `axis_level`, по-кейсный `critical_pass` + recall/FP, история прогонов и
  сравнение агентов (`claude_code` vs `anthropic_api`). **Решение по источнику данных:**
  ~~(a) гнать pipe-check через `atp test`~~ / ~~(b) отдельная таблица~~ →
  **выбран мост (b-lite): импортёр существующих JSON в SP-1 store** (2026-06-18).
  - [x] **`method/import_pipecheck_to_dashboard.py`** — читает `report_benchmark_*.json`,
    пишет по одному completed `SuiteExecution` (idempotent по `run_uuid`), переиспользуя
    уже потраченный прогон без повторного вызова агентов. Рендерится через **существующие**
    `/ui/eval-leaderboard` (ранжир по `critical_pass_rate`) и `/ui/eval-trends` (тренд по агенту).
    Тест: `tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py`.
  - [x] **Прогнан на Mac (3.12, 2026-06-18):** 10/10 тестов, pyrefly 0, ruff чисто; импорт 81
    отчёта → `~/.atp/dashboard.db`, `atp dashboard` стартует, `/ui/eval-leaderboard?suite_name=code-review`
    рендерит всех агентов с `critical_pass_rate`/`malformed_rate`/`breakpoint`. По пути закрыты 2 бага:
    (1) тест не собирался — модуль грузился через `importlib` без регистрации в `sys.modules`, а
    `from __future__ import annotations` ломал `@dataclass`-резолв строковых аннотаций (фикс:
    `sys.modules[_spec.name] = imp`); (2) `atp dashboard` падал на старте — пустой `ATP_DATABASE_URL=`
    в `.env` давал `database_url=""`, проскакивавший проверку `is None` в `create_async_engine("")`
    (фикс: `if not url:` → дефолт, в `database.py`; `.env.example:65` везёт ту же ловушку).
  - [x] **`breakpoint_axis_level` проброшен** через импортёр (2026-06-18): поле есть в 54/81 отчётах
    на верхнем уровне payload (clean…very_severe), модель/storage/шаблон уже поддерживали — импортёр
    его выбрасывал, колонка `breakpoint` была пустой. Теперь заполняется. `language` — 0/81 в payload
    (нет в данных), per-case recall/FP — `per_task` тонкий (`task_index/task_type/score/tokens/...`).
  - [ ] **Осталось из исходного запроса** (требует правки `run_pipe_check.py`, не моста): свип по
    `axis_level` (leaderboard берёт *последний* прогон на агента — `suite_leaderboard`, дисперсию
    схлопывает), по-кейсный recall/FP (нет в payload), языковая ось (нет в payload). Отдельный вид +
    обогащение `report_benchmark` — EPIC-уровень, см. родителя выше.
  - [x] **Эргономика: `run_pipe_check.py --to-dashboard`** (2026-06-19): после прогона зовёт мост
    (`import_pipecheck_to_dashboard`) на своей out-dir — новые свипы попадают в `/ui/eval-*` без
    отдельной команды; `--dashboard-replace` для supersede. Guarded import → в песочнице без дашборда
    флаг просто скипается. Развязка харнесс↔дашборд сохранена.
  - [x] **Конвенция `agent_id = <harness>@<model>`** (2026-06-19, arbiter ACKED): единый data-driven
    реестр `HARNESSES`+`AGENT_MODELS`→`AGENTS` в `run_pipe_check.py`; модель явная у каждого агента и
    пишется в `SuiteExecution.model` (импортёр парсит `@`); `safe_agent_id()` только для имён файлов,
    faithful id — в payload/дашборде/ключе arbiter. ollama-id меняют форму (`ollama@qwen2.5:14b`).
    **codex_cli вне дефолтной матрицы** — оператор добавляет `("codex_cli","<model>")` в `AGENT_MODELS`
    при конфигурации прогона (вместе с новыми моделями/ключами). Выходной прогон ре-базлайнит на новых
    id. Нотис: `_cowork_output/contracts/2026-06-19-agent-id-convention-change.md`. Спека+план в
    `docs/superpowers/{specs,plans}/2026-06-19-agent-id-convention*`.
  - [x] **Ростер выходного прогона — Tier-1** (2026-06-20): opus полностью убран; claude → `claude-sonnet-4-6`
    (CLI `claude_code` + API-baseline `anthropic_api`); добавлены `mimo@MiMo-V2.5-Pro` и `qwen@qwen3.6-plus`
    через один общий OpenAI-compat шим (`method/spawners/_openai_compat.py` + тонкие `mimo_shim`/`qwen_shim`;
    `deepseek_shim` не тронут). `.env.example` несёт `MIMO_API_KEY`/`QWEN_API_KEY`/`OPENCODE_GLM_API_KEY`.
    `("codex_cli","gpt-5-codex")` теперь в `AGENT_MODELS` (это 2-й routable-ключ arbiter — без него
    re-rank join по codex вернёт None). arbiter правит `config/agents.toml`
    (claude→sonnet, +codex, −aider). Спека+план: `docs/superpowers/{specs,plans}/2026-06-20-agent-roster-expansion*`.
  - [x] **Ростер Tier-2** (2026-06-20): `pi@gpt-5` и `opencode@glm-5.1` — два новых CLI-харнесса,
    общий раннер `method/spawners/_cli_common.py` + тонкие `pi_shim`/`opencode_shim`. Оба **non-routable**
    (как deepseek/mimo/qwen — только `benchmark_runs`, без Maestro-спаунера/arbiter `agents.toml`, см.
    runbook Case B). pi запускается с `openai/`-префиксом + `--no-prompt-templates` + 600 s timeout-гард
    (агентный CLI, риск hang). `ALLOWED_ENV` += `PI_BIN/PI_MODEL/OPENCODE_BIN/OPENCODE_MODEL/OPENCODE_GLM_API_KEY`.
    **Живой смоук (гейт) пройден обоими:** opencode@glm-5.1 `critical_pass_rate=1.000 malformed=0.000`,
    pi@gpt-5 `critical_pass_rate=1.000 malformed=0.000` (pi не завис — гейт зелёный, обе строки остаются).
    Спека+план: `docs/superpowers/{specs,plans}/2026-06-20-agent-roster-tier2*`. Следующее: включить в
    платный прогон (свежий `--out-dir` + `--dashboard-replace`) вместе с остальным ростером.
  - [x] **Платный прогон полного ростера** (2026-06-21): 13 агентов × 15 code-review кейсов, rubric off,
    `--to-dashboard --dashboard-replace`. **0 infra-fail** (в отличие от прошлого раза). Данные:
    `_cowork_output/r07-pipecheck/sweep-2026-06-21/` (+ `sweep.db`), дашборд обновлён (13 @-id агентов).
    `critical_pass_rate`: верхний ярус **слип на 0.800** — claude_code/codex_cli/mimo/pi/opencode (фронтир
    code-review **не разделяет**, как в R-07 P2); тай-брейкер `breakpoint_axis_level` — codex/mimo/pi
    держатся до **severe**, claude_code/opencode ломаются на moderate. Дальше: qwen2.5:7b 0.733,
    anthropic_api 0.667, deepseek/qwen 0.600, qwen2.5:14b 0.533, qwen2.5:3b 0.267, llama3.2:3b 0.200,
    llama3.2:1b 0.000. Здоровый `malformed`-градиент на ollama (1b 0.93 → 7b 0.00). Tier-2 (pi/opencode)
    подтверждены на реальном прогоне (0.800, 0 malformed). **Оговорки:** cost пойман только у claude_code
    ($1.31), остальные `unknown`; `claude_code` tokens=1336 — квирк токен-аккаунтинга шима (грейд честный).
  - [x] **Фикс токен-аккаунтинга `claude_code`-шима** ✅ 2026-07-02 (PR #213): шим суммировал только
    input+output, теряя `cache_creation`/`cache_read` (Claude Code кэширует контекст → tokens=1336 на
    прогоне 06-21). Теперь суммируются все 4 класса usage + оба cache-поля проброшены в `Metrics`.
    Валидировано ре-свипом 07-02: 575k токенов вместо 1336.
  - [ ] **ИССЛЕДОВАТЬ (перспектива): полное слияние путей оценки** — дописать `BenchmarkReporter`
    (сейчас `NotImplementedError`; маппинг `SuiteReport→report_benchmark` помечен Phase-1b), тогда
    `atp test method/cases/X -o report_benchmark` сам даёт arbiter-payload, а `run_pipe_check`
    становится тонким циклом `atp test` по агентам → пишет в `SuiteExecution` нативно И эмитит payload
    → мост нужен только для исторических данных. Один путь оценки. Открытый вопрос: оправдывает ли
    выигрыш связывание чистого продьюсера (харнесс, без БД-зависимости, работает в 3.10-песочнице) с
    тяжёлым опциональным потребителем (дашборд). Также свести дублирование агрегатов
    (`breakpoint_axis_level`/`critical_pass_rate` считаются и в `benchmark_reporter.py`, и в
    `dashboard/dimensions.py`). Зависит от обогащения payload; смыкается с родительским EPIC.
    См. `docs/research/2026-06-19-unify-eval-paths.md`.
  Контекст: R-07 Phase 1, `method/run_pipe_check.py`.
- [ ] **CLI run-history page `/ui/executions`**: SuiteExecution history (from `atp test`) is
  only reachable via the JSON API — no HTML page renders it (`/ui/*` is wired to the
  separate benchmark `Run` model). New page: list + detail + per-run statistics +
  failure-cause breakdown. Plan: [`spec/dashboard-execution-history.md`](spec/dashboard-execution-history.md).
  Prereq fix already done: `SuiteExecutionSummary.agent_id` → `int | None` (CLI stores NULL).
- [ ] **Chart.js in Analytics**: status pie chart, score histogram, per-agent line chart (templates/ui/analytics.html).
- [ ] **Fix UI routes test isolation**: `.value` bug in analytics/home templates, UNIQUE constraint collision.
- [ ] **Benchmark API scoring**: wire up evaluators instead of the naive score (100 if completed else 0).

## ~~`atp-method` plugin — run methodology cases via ATP~~ ✅ DONE 2026-06-10

Plan: [`spec/atp-method-plugin.md`](spec/atp-method-plugin.md). `method/`
(agent-eval-case methodology) now runs through the platform as a plugin:
`atp test method/cases/<case-or-dir>` loads a case or a whole sweep and runs the
normal adapter/orchestrator/evaluator path, with `critical_check` hard-gating.
Shipped across PRs #142–#146.

- [x] **Slice 1 — core hard-gate** (#142): `Assertion.critical` + `EvalResult.critical` +
  `ScoreAggregator` hard-fails on a failed critical check (native home for
  `grader.critical_check`).
- [x] **Slice 2 — core format-dispatch registry** (#143): replace the hardcoded
  `_is_game_suite` branch in `atp test` with a `{detector → handler}` registry.
- [x] **Slice 3 — plugin schema + loader**: `packages/atp-method/` — `agent-eval-case`
  pydantic model + case→`TestDefinition` loader.
- [x] **Slice 4 — plugin evaluator**: `AgentEvalCaseEvaluator` (`critical_check` then rubric),
  delegating model calls to the platform LLM judge.
- [x] **Slice 5 — register() + dispatch + E2E**: `atp.plugins` entry-point loader +
  suite-source registry; `atp test method/cases/` loads a case or a whole sweep and
  runs the normal adapter/orchestrator/evaluator path. Plugin is complete.

## Admin tournament GUI follow-ups (deferred from 2026-04-20 spec)

Spec: `docs/superpowers/specs/2026-04-20-admin-tournament-gui-design.md`
Plan: `docs/superpowers/plans/2026-04-20-admin-tournament-gui.md`

- [ ] **h · Live MCP SSE connection status** per participant in admin detail — needs a new in-memory connection registry bound to the FastMCP server plus a `/ui/admin/tournaments/{id}/connections` fragment. Scope: ~2 days.
- [ ] **f · Force-advance round** (admin button + REST endpoint wrapping existing `TournamentService.force_resolve_round`) — currently only the deadline worker can trigger it. Safety: needs a confirmation step and audit log entry.
- [ ] **g · Extend round deadline mid-round** — requires adding a service method and a new audit row since mutating `Round.deadline` after creation is currently disallowed.
- [ ] **Generalize admin create form to all 8 games** — currently hardcoded to `el_farol` dropdown. Add per-game config fieldsets keyed off the game registry.
- [ ] **Long-lived bot MCP sessions (spec C)** — separate design and plan; the admin TTL change in this PR does not address bot-side session budget (still capped at `(ATP_TOKEN_EXPIRE_MINUTES − 10) × 60` in `TournamentService.create_tournament`).

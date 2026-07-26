# TODO

> Инлайн-теги в хвосте пункта (формат из `../_cowork_output/2026-07-26-plan-fields-and-todo-coverage-handoff.md`,
> §3): `@blocked_by:<repo>#<slug>` — чем заблокирован, `@trigger:"условие"` — при каком
> проверяемом условии пункт становится актуальным, `@owner:<handle>` — владелец. Все три
> опциональны; **отсутствие означает «неизвестно», а не «нет»**, и придумывать значение хуже,
> чем оставить пусто. Парсер Robin (`plan_state`) исключает теги из ключа идентичности
> (robin-runtime#27), так что разметка не даёт фантомных «закрыт/открыт» в дневной дельте.
> `@owner:` пока не проставляется — конвенция хэндлов не решена (handoff §5.3).

## Ecosystem Roadmap (план от 2026-04-16)

> Стратегический контекст: `../prograph-vault/authored/notes/ecosystem-roadmap.md`
> Актуальный статус: `../prograph-vault/authored/notes/status/2026-04-10-status.md`
> **Роль ATP в экосистеме**: валидация задач Maestro (`validation_cmd`) и eval-driven обучение arbiter

### Активные кросс-проектные задачи

- [x] **ADR-ECO-003d pricing-view: cloud-`$` над per-class usage** (effort M) ✅ 2026-07-07
  - Спека: [`docs/superpowers/specs/2026-07-07-pricing-view-cost-derivation-design.md`](docs/superpowers/specs/2026-07-07-pricing-view-cost-derivation-design.md);
    план: [`docs/superpowers/plans/2026-07-07-pricing-view.md`](docs/superpowers/plans/2026-07-07-pricing-view.md).
  - Cache-aware LiteLLM-прайсер `packages/atp-core/atp/cost/cloud_pricer.py` (extra `[pricing]`) +
    вью `method/price_reports.py` над сохранёнными `report_benchmark_*.json`; derived-not-stored.
    codex-usage нормализован к `cloud_pricing_usage_v1` (cache-split, no double-count),
    `usage_contract` штампуется в payload; estimated-fallback отложен (нужен grade-time текст).
  - **Осталось (тебе, платно):** re-sweep routable-набора под нормализованным контрактом →
    re-derive (бесплатно). До re-sweep вью корректно флагит старые отчёты `contract_missing`.
  - Дальше по плану: **003b эпик** (loader + `atp models` CLI + XDG).

- [x] **ADR-003b SP-A: model-catalog loader + `atp models` CLI** (effort M) ✅ 2026-07-07
  - Спека: [`docs/superpowers/specs/2026-07-07-model-catalog-loader-design.md`](docs/superpowers/specs/2026-07-07-model-catalog-loader-design.md);
    план: [`docs/superpowers/plans/2026-07-07-model-catalog-loader.md`](docs/superpowers/plans/2026-07-07-model-catalog-loader.md).
  - Шиппится `atp/model_catalog/` (schema + typed errors + D2-резолюция `$ATP_CATALOG` → XDG →
    fail-loud + inert template) и `atp models init`/`atp models list` CLI. `method/agents-catalog.toml`
    остаётся dev-SSOT, в wheel не попадает.
  - [x] **SP-E: харнесс `run_pipe_check` переведён на общий loader** ✅ 2026-07-07
    — спека: [`docs/superpowers/specs/2026-07-07-sp-e-harness-catalog-migration-design.md`](docs/superpowers/specs/2026-07-07-sp-e-harness-catalog-migration-design.md);
    план: [`docs/superpowers/plans/2026-07-07-sp-e-harness-catalog-migration.md`](docs/superpowers/plans/2026-07-07-sp-e-harness-catalog-migration.md).
  - [x] **SP-C: `settings.default_llm_model` эвалюатора унифицирован с общим catalog** ✅ 2026-07-07
    — спека: [`docs/superpowers/specs/2026-07-07-sp-c-evaluator-model-unify-design.md`](docs/superpowers/specs/2026-07-07-sp-c-evaluator-model-unify-design.md);
    план: [`docs/superpowers/plans/2026-07-07-sp-c-evaluator-model-unify.md`](docs/superpowers/plans/2026-07-07-sp-c-evaluator-model-unify.md).
    `CatalogDefaults` + referential validator, tolerant `resolve_default_model()`
    (explicit → catalog `[defaults]` → provider fallback; missing/broken optional catalog stays
    silent), `default_llm_model` теперь `str | None = None`, `llm_judge` резолвится через него.
  - [x] **SP-D: переименовать `atp/catalog/` → `atp/test_catalog/` (снять коллизию имён с
    `atp/model_catalog/`)** ✅ 2026-07-07
    — спека: [`docs/superpowers/specs/2026-07-07-sp-d-rename-test-catalog-design.md`](docs/superpowers/specs/2026-07-07-sp-d-rename-test-catalog-design.md);
    план: [`docs/superpowers/plans/2026-07-07-sp-d-rename-test-catalog.md`](docs/superpowers/plans/2026-07-07-sp-d-rename-test-catalog.md).
    Чисто механический `git mv` + anchored replace (импорты, `mock.patch` string targets,
    docstrings, docs); `atp catalog` CLI-команда и имена классов (`CatalogRepository` и т.д.)
    не изменились.
  - **ADR-003b эпик завершён** (SP-A/SP-E/SP-C/SP-D все шипнуты) ✅ 2026-07-07.

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

- [ ] **R-06b: SDK-интеграция для Maestro** @blocked_by:Maestro#R-03
  - `atp.sdk.arun()` или SDK Adapter — структурированные результаты
  - Автоматический feedback loop: Maestro → задача → ATP eval → arbiter обучение
  - Наш SDK уже готов (PyPI `atp-platform-sdk` v2.0.0)

- [ ] **R-07: Eval-driven routing validation** @blocked_by:Maestro#R-03
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
    - [x] **#5 grounding/recency-ось на CLI-ростере (Путь A)** ✅ 2026-07-05…07-06.
      PR #203 (fomih) дал `run_mode: read_only_corpus` + детерминированный грейдер
      `citation_grounding`, но wired был только под `anthropic_api`: продуктовый CLI имеет
      свой закрытый tool-набор и не знает про ATP `tools_endpoint`. Путь A нацелил **нативные**
      инструменты CLI на уже материализованную директорию корпуса (`cwd=workspace_path` +
      per-CLI confinement; HTTP-эндпоинт не понадобился). Шипнуто по срезам:
      `claude_code` (#227), `codex_cli` (#228), `pi` (#229), `opencode` (#230) поверх общего
      `method/spawners/_cli_common.py` (`corpus_workspace()`, `corpus_args`/`corpus_env`,
      нормализация цитат в corpus-relative) + severity-лестница (#231: 4 кейса clean…very_severe
      в `method/cases/req-extraction/`). `run_pipe_check.py` больше не скипает corpus-кейсы для
      corpus-capable харнессов; для остальных сохранено громкое skip-поведение #217. Дизайн:
      [`docs/superpowers/specs/2026-06-21-cli-corpus-grounding-design.md`](docs/superpowers/specs/2026-06-21-cli-corpus-grounding-design.md).
      **Результат свипа** (runs=3, 2026-07-06, `_bench_output/r07-pipecheck/corpus-sweep-runs3-2026-07-06/`):
      `claude_code@claude-sonnet-4-6` / `codex_cli@gpt-5.5` / `pi@gpt-5` — `crit_pass=1.000`,
      `opencode@glm-5.1` — 0.941 (malformed 0.059); infra-fail 0 у всех. Два вывода:
      (1) прежний «corpus-clean валит обоих» был **артефактом обвязки**, Путь A его снял;
      (2) как роутинг-сигнал ось **упирается в потолок на фронтир-ростере** — null той же формы,
      что crossover 07-02. Чтобы ось дискриминировала, нужен более слабый ростер либо более
      жёсткая лестница дистракторов; на текущем ростере это не задача развития.

- [x] **Libretto receipts/IR как evaluation-вход** (effort S–M) ✅ 2026-07-16 (#252, #253, #255)
  - Оффер: `../prograph-vault/authored/notes/2026-07-16-libretto-contracts-offer.md`
    (2026-07-16). Libretto оставляет от каждого прогона `receipts.jsonl`
    (`libretto.receipt.v1`, hash-chained журнал: что исполнилось, порядок, входы, токены
    с `usage.basis: exact|estimated|unavailable`) и `{program}.ir.json`
    (`libretto.compile-ir.v1`). Legacy `openprose.*` остаётся readable для исторических
    ledgers. Резонирует с нашей философией честного usage-учёта (003d `usage_contract` /
    003e provenance).
  - **Почему делаем сейчас** (пересмотр решения «ждать спроса» того же дня): у ATP уже
    отработана механика вендоренный-контракт + contract-тесты в CI (learning-event-v1,
    EvidenceRef v1, RD-007) — закоммиченный corpus Libretto и есть workload для reader,
    копия не гниёт молча; Libretto Rust-гейт 4.6 называет atp-platform поимённо
    («receipts-verify crate used by atp-platform», их plan:82-84, ревизит в конце их
    Phase 4) — факт «вендоренный контракт + работающий reader» должен существовать к
    ревизиту, причинность: сначала задача здесь, потом легитимный триггер гейта.
    Спешки нет — контракты append-frozen, но и триггер-ожидание больше не нужно.
  - **Сделано:**
    - вендорена пинованная копия контрактов → `method/contract/openprose/{receipt.md,ir.md}`
      + `PROVENANCE.md`;
    - reader приземлён на существующую потребляющую поверхность (не библиотека на полке):
      детерминированный чекер `receipt_chain` (`atp/evaluators/openprose_receipts/checker.py`,
      `receipt_chain@1`), зарегистрирован в `atp/evaluators/checkers/` под
      `grader: {type: programmatic, checker: receipt_chain}`;
    - contract-тесты на корпусе Libretto и битых фикстурах —
      `tests/unit/evaluators/test_openprose_receipts.py`;
    - append-frozen семантика: неизвестные поля игнорируются, неизвестный `v` — отклоняется.
    - Спека: `docs/superpowers/specs/2026-07-16-libretto-receipts-evaluation-input-design.md`;
      план: [`docs/superpowers/plans/2026-07-16-libretto-receipts-evaluation-input.md`](docs/superpowers/plans/2026-07-16-libretto-receipts-evaluation-input.md).
  - **Осталось:** при ревизите Libretto-гейта 4.6 факт «вендоренный контракт + работающий
    reader» готов к предъявлению. arbiter — вторичный потребитель (не начат); proctor — ничего.

- [ ] **ADR-ECO-003e: runtime cost control** (эпик; M0 закрыт, дальше M1→M4)
  - План M0: [`docs/superpowers/plans/2026-07-15-adr-eco-003e-m0-usage-capture-probe.md`](docs/superpowers/plans/2026-07-15-adr-eco-003e-m0-usage-capture-probe.md);
    раннбук: [`docs/cost/003e-action0-probe-runbook.md`](docs/cost/003e-action0-probe-runbook.md).
  - [x] **M0 — UsageCapture seam + Action №0 probe** ✅ 2026-07-15 (#251). Observe-only:
    `atp/cost/capture.py` (контракт + JSONL-сток), шов в `TestOrchestrator`,
    `python -m atp.cost.probe_report` (таблица покрытия), `track_response_cost` помечен
    `DeprecationWarning` (удаление — в M1).
  - [x] **Action №0 — bounded probe прогнан** ✅ 2026-07-16, обе ноги
    (`_bench_output/003e-probe/`). **Результат, который упорядочивает M1:** usage
    захватывается (389 токенов через deepseek), но `model` и `cost_usd` **не заполняются
    никогда** — честный `None` вместо выдуманного `"unknown"`. То есть узкое место не в
    сборе usage, а в отсутствии идентичности модели на границе адаптера.
  - [ ] **M1 — adapter adoption + model identity.** Пробросить реальные `model`/`provider`
    в `UsageRecord` по адаптерам (порядок — по колонке объёма токенов из пробы: cli первым);
    удалить `track_response_cost` и мёртвый флаг `enable_cost_tracking`; зафиксировать
    финальный дом шва (оркестратор vs base-adapter template) после переписи не-оркестраторных
    вызовов. **Следующий шаг эпика.**
  - [ ] **M2 — price snapshot (003e D7).** Генератор снапшота (канон-каталог +
    `method/price_overrides.toml` + litellm map) → версионированный артефакт со штампом
    `price_map_version`; синхронный snapshot-прайсер с трёхзначным
    `pricing_status ∈ {known, ceiling, unknown}` (правило «молчаливый ноль = unknown», D6.3);
    депрекейт и снос System-A таблицы в `atp/cost/models.py` + миграция `CostTracker`.
  - [ ] **M3 — BudgetControl (003e D1/D3/D4/D5).** `estimate/reserve/settle`, атомарный
    store резерваций (SQLite single-writer, идемпотентность по `call_id`/`reservation_id`,
    reaper по settle-timeout); таксономия скоупов attempt⊂task⊂run⊂day; per-scope политика;
    завести deny в оркестратор (`budget_usd` уже течёт в `ATPRequest.constraints`, но никем
    не энфорсится — это и есть естественная первая точка).
  - [ ] **M4 — ecosystem handoff.** Когда контрактный модуль устоится — вендоринг-хендофф для
    Maestro / spec-runner / robin-runtime в `../prograph-vault/authored/notes/` (их репо
    отсюда read-only); выравнивание advisory-budget инварианта arbiter.

- [ ] **RD-007: LearningEvent v1 — обучение через governance, без silent-write**
  - Дизайн: [`docs/2026-07-12-rd-007-learning-event-design.md`](docs/2026-07-12-rd-007-learning-event-design.md) (#248).
  - [x] **M1a (наша доля)** ✅ 2026-07-12 (#249): `learning-event-v1.schema.json` + фикстуры +
    contract-тест; `CODEOWNERS` на governed-пути (v1 acceptance).
  - [ ] **M2 (отложено):** conformance-CI (вендоренные byte-проверки, сканер no-runtime-writes);
    новые продюсеры (experiment recommendations, catalog proposals) переходят на схему.
  - Не наше: M1b (robin-runtime — selfreview эмитит события), M1c (prograph-vault — CODEOWNERS
    на `authored/**`). Отслеживать, не делать.

- [x] **Runtime observability & recovery** ✅ 2026-07-19 (#258)
  - План: [`docs/superpowers/plans/2026-07-19-runtime-observability-recovery.md`](docs/superpowers/plans/2026-07-19-runtime-observability-recovery.md).
  - Structlog подключён к CLI-рантайму, `run_id`-корреляция прогонов, кооперативная обработка
    SIGINT/SIGTERM, чекпоинты сьюта → прерванный `atp test` возобновляем. Без новых сервисов и
    без миграций схемы: два существующих шва (`cli()` в `atp/cli/main.py` и `TestOrchestrator`).

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
- [x] **f · Force-advance round** ✅ 2026-07-11 (#247) — admin button (`ui/admin/tournament_detail.html`)
  + `POST /api/v1/tournaments/{id}/force-advance` над существующим
  `TournamentService.force_resolve_round`; аудит-запись `TOURNAMENT_FORCE_ADVANCE`.
- [ ] **g · Extend round deadline mid-round** — requires adding a service method and a new audit row since mutating `Round.deadline` after creation is currently disallowed.
- [ ] **Generalize admin create form to all 8 games** — currently hardcoded to `el_farol` dropdown. Add per-game config fieldsets keyed off the game registry.
- [ ] **Long-lived bot MCP sessions (spec C)** — separate design and plan; the admin TTL change in this PR does not address bot-side session budget (still capped at `(ATP_TOKEN_EXPIRE_MINUTES − 10) × 60` in `TournamentService.create_tournament`).

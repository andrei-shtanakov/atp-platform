# R-07 Phase 1 — MVP-вертикаль: code-review eval → benchmark → routing

> Дата: 2026-06-13. Статус: design note (предложение). Режим: read-only по репо.
> Контекст: `2026-06-13-r07-thin-slice.md` + `status/2026-06-13-r07-phase0-data-recon.md`.
> Решение: первая вертикаль R-07 = **code-review** (маршрутизируется как `TaskType::Review`,
> ординал 5; детерминируемее coding — есть эталонный диагноз, не нужен прогон тестов в песочнице).

## TL;DR

1. **Строим один сквозной срез, а не 5 сьютов.** code-review-семейство кейсов (atp-method) × 3 спавнера (`claude_code`/`codex_cli`/`aider`) → новый ATP-репортер `report_benchmark` → `arbiter.benchmark_runs` → re-rank A/B. Доказываем весь конвейер, включая фундамент данных, на одной вертикали.
2. **Таксономия зафиксирована: канон — `TaskType` arbiter** (он потребляет данные). atp-method-оси (`capability`/`construction_axis`/`axis_level`) — это *внутри-задачная* развёртка сложности, а не замена `task_type`. `benchmark_id="code-review"` ↔ `TaskType::Review`.
3. **Сигнал берём из `axis_level`-свипа, а не из плоского скора.** Точка, где `critical_check` агента начинает падать, различает агентов *внутри* code-review — это чинит Блокер 3 Фазы 0 (плоский `comparison.json` давал монотонный глобальный bias Δ≈0.013).
4. **Reader в arbiter апгрейдится: скор обязан быть task_type-scoped.** `get_benchmark_score(agent_id, benchmark_id)` с фильтром, иначе A/B невалиден (требование R1). Это уточнение к §3.1 thin-slice.
5. **«Экспорт/импорт между базами» = новый reporter под контракт M4**, не bespoke sync. ATP-eval → `report_benchmark-v1` payload → MCP `report_benchmark` → `benchmark_runs`.

---

## 1. Маппинг трёх таксономий (закрывает граблю №1)

| Твои 5 задач | Канон `arbiter::TaskType` (ординал) | atp-method capability | Статус для MVP |
|---|---|---|---|
| **code-review** | **Review (5)** | correctness / safety_compliance | ✅ **MVP** |
| coding | Feature (0) / Bugfix (1) / Refactor (2) | correctness, efficiency | Фаза 2 |
| documentation | Docs (4) | correctness, calibration | Фаза 2 |
| repo-analysis | Research (6) | correctness, recoverability | Фаза 3 |
| architecture | — (нет слота в enum) | adaptation | Фаза 3 + **решение по enum** |

> `task_type` кодируется в вектор как `f[0] = task_type.as_ordinal()` (`features.rs:133`) и в `f[15] agent_supports_task_type` (`features.rs:203`). Review уже валидный класс — расширять enum для MVP **не нужно**. architecture-design потребует отдельного решения (расширять `TaskType` либо мапить на Research) — отложено до Фазы 3.

## 2. MVP-сьют: семейство `code-review-planted-defect`

Каждый кейс по `agent-eval-case.schema.json`. Идея: агенту дают **diff/PR + выдержку из корпоративной KB-правил**, он должен выдать ревью. В код посажен один дефект, нарушающий конкретное правило KB; вокруг — правдоподобные, но корректные строки (дистракторы).

### 2.1. `critical_check` — детерминированный хребет (грабля №3)
Бинарно: **ревью обязано пометить посаженный дефект и сослаться на нарушенное правило KB (`rule_id`)**. Промахнулся или нет ссылки на правило → кейс fail независимо от рубрики. Это не LLM-суждение — это match по наличию `rule_id` + локации дефекта в выводе (grader `type: programmatic` или `regex` по `rule_id`).

### 2.2. `rubric` — KB-grounded слой сверху (model_graded, on-prem judge)
Веса в сумме 1.0: правильная severity (0.3), точная цитата правила, не пересказ (0.3), нет галлюцинированных issue на корректных строках-дистракторах (0.25), actionable фикс (0.15). Судья — LLM-judge через `base_url` on-prem (v2.1.0), `gold` = эталонное ревью + `rule_id`.

### 2.3. `axis_level`-свип — источник различающего сигнала
Один `family`, кейсы отличаются только сложностью сокрытия дефекта:

| axis_level | Конструкция дефекта |
|---|---|
| clean | дефект очевиден, дистракторов нет |
| mild | 1–2 корректных дистрактора рядом |
| moderate | дефект в стиле «выглядит идиоматично», 5+ дистракторов |
| severe | дефект логический, проявляется только с правилом KB в руках |
| very_severe | дефект через взаимодействие двух файлов; правило KB надо применить транзитивно |

> **Сигнал для routing = breakpoint**: уровень, на котором `critical_check` агента впервые падает. claude_code держит до `severe`, aider ломается на `moderate` → это per-agent, per-task_type различие, которого не было в Фазе 0. `construction_axis: adversarial_environment`, `suite_type: probe` (→ `regression` после заморозки).

### 2.4. Кто под тестом (грабля №2 — то, что убило Фазу 0)
`environment.tools: [file_read]`, `side_effects: none`. Прогон **обязан** идти против трёх спавнеров, `agent_id ∈ {claude_code, codex_cli, aider}` — НЕ против `openai`/`anthropic`-адаптеров. Это требование к раннеру: эвал гоняется через тот же spawner-слой, что маршрутизирует arbiter, иначе `agent_id` снова не лягут в `benchmark_runs`.

### 2.5. Скелет кейса (moderate)
```yaml
id: case-code-review-sqli-moderate-001
version: 1
family: code-review-planted-defect
status: active
suite_type: probe
capability: safety_compliance
construction_axis: adversarial_environment
axis_level: moderate
tags: [security, kb_rule_sec_011, review]
instruction: >
  Review the attached diff against the team coding rules (provided). Report each
  issue with: rule_id, file:line, severity, and a concrete fix. Do not invent issues.
artifacts:
  - id: diff
    type: text
    content: "<+ raw SQL built via f-string interpolation; 5 correct distractor lines>"
  - id: kb-rules
    type: text
    content: "SEC-011: user input MUST NOT be interpolated into SQL; use parameterized queries."
environment: { tools: [file_read], side_effects: none }
constraints: [cite rule_id for every issue, do not flag compliant lines]
expected_failure_mode: >
  Agent misses the SQL-injection on the f-string line (SEC-011), or flags a compliant
  distractor line as a violation.
grader:
  type: programmatic
  gold: method/gold/code-review-sqli-001.md
  rubric:
    - { criterion: severity = high/critical, weight: 0.3 }
    - { criterion: cites SEC-011 verbatim, weight: 0.3 }
    - { criterion: no false-positive on distractor lines, weight: 0.25 }
    - { criterion: fix uses parameterized query, weight: 0.15 }
  critical_check: >
    Output flags the f-string SQL line AND references rule_id SEC-011. MUST NOT be empty
    and MUST NOT mark any distractor line as a violation.
  scoring: "Fail if critical_check fails. Else score = weighted rubric sum, in [0,1]."
provenance: { author: andrei, created: "2026-06-13", source: hand-authored + KB SEC-011 }
```
MVP-объём: 1 family × 5 axis_level × (минимум) 3 повтора = **15 кейсов × 3 агента = 45 прогонов**. Достаточно, чтобы breakpoint был статистически различим, и мало, чтобы собрать за 1–2 вечера.

## 3. Контракт «экспорт/импорт»: ATP-репортер `report_benchmark` (грабля «не строй sync»)

Новый reporter в ATP (рядом с console/JSON/HTML/JUnit/game) агрегирует прогон семейства в один `report_benchmark-v1` payload на агента и шлёт в arbiter MCP. Маппинг на колонки `benchmark_runs` (`db.rs:903`):

| benchmark_runs | источник из ATP-прогона |
|---|---|
| `run_id` | uuid прогона (идемпотентный ключ, `ON CONFLICT DO NOTHING`) |
| `benchmark_id` | `"code-review"` ← ключ маппинга на `TaskType::Review` |
| `agent_id` | `claude_code` / `codex_cli` / `aider` |
| `score` | агрегат ∈ [0,1]: доля прошедших `critical_check`, взвешенная по `axis_level` |
| `score_components` | `{critical_pass_rate, mean_rubric, breakpoint_axis_level}` |
| `per_task` | по-кейсный JSON (id, axis_level, critical_pass, rubric) |
| `total_tokens`/`total_cost_usd`/`duration_seconds` | из прогона |

> Контракт `report_benchmark-v1.schema.json` уже синхронен у 3 владельцев — реюзаем, не меняем. Импорт в arbiter — это существующий tool `report_benchmark` (M4), а не новая БД-связка.

## 4. Апгрейд reader'а в arbiter (уточнение thin-slice §3.1)

Для **валидного** A/B скор обязан быть task_type-scoped (иначе code-review-скор потечёт в routing задач Docs/Bugfix — невалидно по R1):

```rust
pub fn get_benchmark_score(&self, agent_id: &str, benchmark_id: &str) -> Result<Option<f64>> {
    let row = self.conn.query_row(
        "SELECT score FROM benchmark_runs WHERE agent_id=?1 AND benchmark_id=?2 \
         ORDER BY ts DESC LIMIT 1",
        [agent_id, benchmark_id], |r| r.get::<_, f64>(0)).optional()?;
    Ok(row.map(|s| s.clamp(0.0, 1.0)))
}
```
В `route_task.rs` re-rank вызывается с `benchmark_id`, выведенным из `task.task_type` (`Review → "code-review"`) — лёгкий статический map. Остальное (центрирование на 0.5, флаг `ARBITER_BENCH_WEIGHT`, лог в `decision_path`) — как в thin-slice §3.2.

## 5. План A/B и критерий валидности (закрывает Фазу 0)

1. Прогнать 45 кейсов → 3 строки в `benchmark_runs` (`benchmark_id="code-review"`).
2. Сформировать набор routing-запросов с `task_type=Review`, идентичными фичами, разными агентами-кандидатами.
3. Прогон A: `ARBITER_BENCH_WEIGHT=0.0` (baseline). Прогон B: `0.15`.
4. **Valid-signal критерий (anti-Phase-0):** сдвиг маршрута должен (а) различать агентов по breakpoint, не по глобальному рангу; (б) превышать шум `PREFERRED_AGENT_BOOST=0.1`. Если разрыв скоров < 0.1 или агент монотонно лучший везде — сигнал невалиден, фиксируем как честный no-go.
5. Результат → `_cowork_output/status/`.

## 6. Scope guard

В Фазе 1 НЕ входит: coding/docs/repo-analysis/architecture сьюты, расширение `TaskType` enum, EWMA/мульти-прогон агрегация, smoke в CI, ретрейн дерева, правки `report_benchmark-v1.schema.json`. Только: 1 family (15 кейсов) + 1 reporter + reader-фильтр + A/B.

## Рекомендуемые действия

1. **atp-platform/method:** создать `cases/code-review/` — family `code-review-planted-defect`, 5 axis_level, KB-rule `gold/`. (~1–2 вечера)
2. **atp-platform:** раннер-прогон семейства против 3 спавнеров (`agent_id` = спавнеры, не адаптеры) + новый `report_benchmark`-reporter. (~2 вечера)
3. **arbiter:** `get_benchmark_score(agent_id, benchmark_id)` + task_type→benchmark_id map в re-rank + golden-тест. (~1 вечер)
4. **A/B** `ARBITER_BENCH_WEIGHT` 0 vs 0.15, зафиксировать valid/no-go по §5.
5. **Обновить `COWORK_CONTEXT.md`**: внести R-07 Phase-1 и benchmark_id-таксономию (заодно закрыть P1-дрейф реестра).

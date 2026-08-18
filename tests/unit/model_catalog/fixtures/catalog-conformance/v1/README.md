# catalog-conformance-fixtures v1

SSOT-набор conformance-фикстур каталога агентов (ADR-ECO-003 / ADR-ECO-003b)
для трёх загрузчиков: **Maestro** (`maestro/catalog.py`), **ATP**
(`atp.model_catalog`), **arbiter** (`arbiter-core/src/catalog/`). Владелец
набора — `devtools` (единый owner-путь, PP-103 acceptance (b); принято из
inbox-issue devtools#43, слаг `catalog-conformance-single-owner`).

Назначение: ADR-ECO-003b, риск №1 — «три реализации загрузчика расходятся в
поведении → mitigation: общий conformance-тест на фикстурах каталога». Набор
делает дивергенцию **наблюдаемой**: на каждый негативный кейс у потребителя
должен быть тест, который красный, если его загрузчик кейс принимает.

## Состав

```
expectations.toml            машинные ожидания (см. классы ниже)
fixtures/valid/              загружаются без ошибок и предупреждений
fixtures/invalid/            должны отвергаться (parse-error, V1..V5)
fixtures/warn/               должны как минимум помечаться (V6, V7)
manifest.json                sha256 каждого файла + tree_sha256 (пин-поверхность)
```

Словарь правил V1..V7 — из дизайна arbiter-загрузчика
(`arbiter/docs/2026-07-05-catalog-loader-design.md` §4); V2+V3 вместе зеркалят
Check 5 `devtools/check-agent-id-conformance.py`.

| Код | Правило | Класс |
|---|---|---|
| V1 | `[[agents]].harness` не объявлен в `[harnesses.*]` | error |
| V2 | `[[agents]].model` не объявлен в `[models.*]` | error |
| V3 | `[[agents]]` ссылается на модель `status="retired"` | error |
| V4 | дубль `agent_id` (`<harness>@<model>`) среди `[[agents]]` | error |
| V5 | `agents.routable=true` при `harnesses.<h>.routable=false` | error |
| V6 | `[[agents]]` ссылается на модель `status="deprecated"` | flag |
| V7 | незнакомое значение `status`/`kind` | flag |

## Семантика классов ожиданий

- **valid** — загрузчик принимает каталог; ни ошибок, ни предупреждений.
- **parse-error** — файл обязан не парситься как TOML.
- **error** — загрузчик обязан отвергнуть каталог (ошибка валидации, исключение,
  ненулевой exit — форма своя у каждого языка).
- **flag** — загрузчик обязан как минимум ПОМЕТИТЬ проблему (warning);
  жёсткое отклонение тоже конформно (pydantic-загрузчики валят V7 схемой —
  это допустимо). Неконформно ровно одно: молча принять как здоровый каталог.

Path-resolution сценарии (`[[pathres]]`) покрывают ТОЛЬКО слой `$ATP_CATALOG`
(ADR-ECO-003b D2): задан+файл есть → loaded; не задан и каталог обязателен →
fail-loud not-configured (никакого скрытого дефолта); задан, файла нет →
missing-file-error (молчаливое «как будто не задан» неконформно).

## Вне набора v1 (намеренно)

- **XDG-слои** резолюции — до закрытия maestro `@id:xdg-catalog-path`
  (`<eco>`-namespace ратифицирован, путь ещё не общий).
- **Один vendor на model id** — валидные фикстуры соблюдают это ограничение,
  но негативного кейса нет: дефект-детекция — отдельный пункт Maestro
  `@id:models-duplicate-vendor-detection`.
- **Alias-резолюция и precedence оверрайдов** — территория Maestro
  `@id:catalog-loader-shared-lib` (shared-lib остаётся отдельной опцией и
  этим набором не блокируется).

## Известные дивергенции на 2026-08-17 (что набор сделает красным)

Зафиксированы при разборе загрузчиков; это НЕ дефекты набора — это то, что
consumer-сьюты обязаны показать (или потребитель осознанно чинит загрузчик):

- **Maestro**: `$ATP_CATALOG` указывает на отсутствующий файл → молчаливый
  `None` + info-лог (осознанный дизайн 2026-07-02) — противоречит
  `missing-file-error`; референсные проверки V1–V5 отсутствуют (harness-план
  игнорируется целиком), V6 — warn только в рантайме спавна.
- **ATP**: нет проверок V2/V3/V6 (только referential-check harness'ов и
  `defaults.default_model`); V7 — жёсткий schema-fail (конформно классу flag).
- **arbiter**: эталонная реализация V1–V7; расхождений не ожидается.

## Как потреблять (вендоринг, не живая ссылка)

По правилу полирепо кросс-репные контракты вендорятся **пиненой копией внутрь**
репо-потребителя (`prograph-vault/authored/rules/repo-boundaries.md`):

1. Скопировать каталог `v1/` целиком в свой репо (например,
   `tests/fixtures/catalog-conformance/v1/`).
2. Записать рядом `PIN`-файл: `source: devtools@<commit> contracts/
   catalog-conformance-fixtures/v1`, дата вендоринга (образцы —
   `steward/contracts/*/PIN`, `dispatcher/contracts/*/PINNED.txt`).
3. Целостность копии проверять против `manifest.json` (sha256 пофайлово;
   `tree_sha256` — sha256 отсортированных пар `<path> <sha256>`, файлы
   `manifest.json` и `PIN` исключены из поверхности).
4. В своём сьюте: на каждый `[[case]]` — тест, сверяющий поведение СВОЕГО
   загрузчика с `expect`; на каждый `[[pathres]]` — тест с временным окружением.

Проверка набора на стороне владельца: `python3 check-catalog-fixtures.py
--check` (референс-валидатор V1–V7 + целостность манифеста); регенерация
манифеста после правки фикстур: `--write-manifest`.

## Версионирование

`v1/` меняется только аддитивно (новые фикстуры + новые `[[case]]`), каждый
такой PR перегенерирует `manifest.json`; изменение СЕМАНТИКИ существующего
кейса — это `v2/`. Потребители обновляют пин осознанным PR-ом.

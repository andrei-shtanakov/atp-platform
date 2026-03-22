# Как запустить тестирование AI-агентов с помощью ATP Platform

Это руководство описывает, как протестировать AI-агентов, которые генерируют код. Мы запустим два агента (на базе OpenAI и Anthropic), прогоним набор тестов и посмотрим результаты.

## Что мы будем тестировать

В папке `demo/` находятся два агента-кодописца:

- **OpenAI Code Writer** — использует модель GPT-4o-mini для генерации Python-кода
- **Anthropic Code Writer** — использует модель Claude Sonnet 4 для генерации Python-кода

Оба агента работают одинаково: получают описание задачи ("напиши функцию Fibonacci") и возвращают готовый Python-файл. ATP Platform проверяет результат: существует ли файл, компилируется ли код, проходят ли unit-тесты.

## Что вам понадобится

1. **Python 3.12+** — проверьте командой `python --version`
2. **uv** — менеджер пакетов Python (если нет — установите: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
3. **API-ключи** — OpenAI и/или Anthropic (можно тестировать только одного агента)

## Шаг 1. Подготовка

Откройте терминал и перейдите в папку проекта:

```bash
cd atp-platform
```

Убедитесь, что все зависимости установлены:

```bash
uv sync
```

## Шаг 2. Настройка API-ключей

Создайте файл `demo/.env` с вашими ключами:

```bash
# Откройте файл в любом текстовом редакторе
nano demo/.env
```

Впишите (подставив свои ключи):

```
OPENAI_API_KEY=sk-ваш-ключ-openai
ANTHROPIC_API_KEY=sk-ant-ваш-ключ-anthropic
```

Сохраните файл. Если у вас есть только один ключ — впишите только его. Агент без ключа просто не запустится, и вы протестируете только один.

> **Где взять ключи:**
> - OpenAI: https://platform.openai.com/api-keys
> - Anthropic: https://console.anthropic.com/settings/keys

## Шаг 3. Запуск агентов

Агенты — это небольшие веб-серверы, которые нужно запустить перед тестированием. Каждый агент работает на своём порту.

Откройте **два отдельных окна терминала** (или вкладки).

**Окно 1 — запуск OpenAI агента:**

```bash
cd atp-platform/demo
source .env
uv run python -m uvicorn agents.openai_agent:app --port 8001
```

Вы увидите:
```
INFO:     Uvicorn running on http://127.0.0.1:8001
INFO:     Started server process
```

**Окно 2 — запуск Anthropic агента:**

```bash
cd atp-platform/demo
source .env
uv run python -m uvicorn agents.anthropic_agent:app --port 8002
```

> **Совет:** Не закрывайте эти окна — агенты должны работать, пока идут тесты.

### Как проверить, что агенты запустились

Откройте в браузере:
- http://localhost:8001/docs — страница OpenAI агента
- http://localhost:8002/docs — страница Anthropic агента

Если видите интерфейс Swagger с описанием API — всё работает.

## Шаг 4. Запуск тестов

Откройте **третье окно терминала**. Все команды выполняются из корня проекта `atp-platform/`.

```bash
cd atp-platform
source demo/.env
```

### 4.1 Smoke-тесты (быстрая проверка)

Это базовые тесты: агент отвечает, файл создан, код компилируется.

**Тестируем OpenAI агента:**

```bash
uv run atp test demo/test_suites/smoke.yaml \
  --adapter=http \
  --adapter-config='endpoint=http://localhost:8001,allow_internal=true'
```

**Тестируем Anthropic агента:**

```bash
uv run atp test demo/test_suites/smoke.yaml \
  --adapter=http \
  --adapter-config='endpoint=http://localhost:8002,allow_internal=true'
```

Вы увидите результат в виде таблицы:

```
  ✓ Агент возвращает Python-файл   100.0/100  [3.9s]
  ✓ Ответ валиден по JSON Schema    99.5/100  [2.0s]
  ✓ Код синтаксически валиден       99.4/100  [2.4s]

Summary: 3 passed (100.0%)
```

### 4.2 Functional-тесты (проверка работоспособности кода)

Здесь сгенерированный код запускается и проверяется unit-тестами. Каждый тест прогоняется 3 раза для оценки стабильности.

```bash
uv run atp test demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config='endpoint=http://localhost:8001,allow_internal=true'
```

### 4.3 Quality-тесты (качество кода)

Проверяется наличие документации, соответствие PEP 8, обработка граничных случаев.

```bash
uv run atp test demo/test_suites/quality.yaml \
  --adapter=http \
  --adapter-config='endpoint=http://localhost:8001,allow_internal=true'
```

## Шаг 5. Чтение результатов

ATP выводит результат в консоль. Для каждого теста показываются:

- **Статус:** ✓ (прошёл) или ✗ (провалился)
- **Балл:** от 0 до 100 — насколько хорошо агент справился
- **Время:** сколько секунд заняло выполнение

Пример вывода:

```
ATP Test Results
==================================================

Suite: code_writer_functional
Agent: test-agent
Runs per test: 3

Tests:
  * Fibonacci — код проходит unit-тесты     76.7/100  [8.5s]
  * CSV-парсер — код проходит unit-тесты    76.6/100  [13.2s]
  * REST API клиент — unit-тесты            53.3/100  [12.4s]

Summary: 3 passed (100.0%)
Total time: 34.2s
```

### Что значат баллы

| Балл | Оценка | Что это значит |
|------|--------|----------------|
| 90-100 | Отлично | Код полностью правильный, все тесты проходят |
| 70-89 | Хорошо | Код работает, но есть мелкие недочёты |
| 50-69 | Удовлетворительно | Код частично работает, есть проблемы |
| < 50 | Плохо | Код не работает или содержит серьёзные ошибки |

## Шаг 6. Завершение

После тестирования остановите агентов — в каждом окне терминала нажмите `Ctrl+C`.

---

## Быстрый запуск (для опытных)

Если вы хотите запустить всё максимально быстро:

```bash
# 1. Настройка
cd atp-platform && source demo/.env

# 2. Запуск агентов в фоне
cd demo
uv run python -m uvicorn agents.openai_agent:app --port 8001 &
uv run python -m uvicorn agents.anthropic_agent:app --port 8002 &
cd ..

# 3. Подождать запуска
sleep 3

# 4. Прогнать все тест-сьюты
for suite in smoke functional quality; do
  echo "=== $suite — OpenAI ==="
  uv run atp test demo/test_suites/$suite.yaml \
    --adapter=http --adapter-config='endpoint=http://localhost:8001,allow_internal=true'

  echo "=== $suite — Anthropic ==="
  uv run atp test demo/test_suites/$suite.yaml \
    --adapter=http --adapter-config='endpoint=http://localhost:8002,allow_internal=true'
done

# 5. Остановить агентов
kill %1 %2
```

---

## Что ещё можно сделать

- **Изменить количество прогонов:** добавьте `--runs=5` к команде для большей статистической значимости
- **Экспорт в JSON:** добавьте `--output=json --output-file=results.json`
- **HTML-отчёт:** добавьте `--output=html --output-file=report.html` — откройте в браузере для визуальных графиков
- **Сравнить агентов**: используйте скрипт `demo/compare_agents.py` для автоматического сравнения

## Устранение проблем

| Проблема | Решение |
|----------|---------|
| `uv: command not found` | Установите uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `OPENAI_API_KEY not set` | Проверьте, что вы выполнили `source demo/.env` |
| `Connection refused` | Агент не запущен — проверьте окно терминала с агентом |
| `Access to localhost is blocked` | Добавьте `allow_internal=true` в `--adapter-config` |
| Агент не запускается | Проверьте, что API-ключ правильный и активный |

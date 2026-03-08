# Руководство по запуску и сравнению агентов

> Все команды выполняются из корня `atp-platform/`.
> Файлы демо-проекта находятся в `demo/`.

## 1. Подготовка

### 1.1. API-ключи

```bash
cp demo/.env.example demo/.env
# Отредактируйте demo/.env, укажите реальные ключи:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
source demo/.env
```

### 1.2. Зависимости

```bash
# Из корня atp-platform
uv add fastapi uvicorn openai anthropic httpx
```

---

## 2. Запуск агентов

### 2.1. Терминал 1 — OpenAI Agent (порт 8001)

```bash
cd atp-platform
export OPENAI_API_KEY=sk-...
uv run uvicorn demo.agents.openai_agent:app --port 8001
```

### 2.2. Терминал 2 — Anthropic Agent (порт 8002)

```bash
cd atp-platform
export ANTHROPIC_API_KEY=sk-ant-...
uv run uvicorn demo.agents.anthropic_agent:app --port 8002
```

### 2.3. Проверка health

```bash
curl http://localhost:8001/health
# {"status":"ok","model":"gpt-4o-mini","provider":"openai"}

curl http://localhost:8002/health
# {"status":"ok","model":"claude-sonnet-4-20250514","provider":"anthropic"}
```

---

## 3. Запуск тестов

> Все пути к YAML-сьютам и фикстурам — относительно корня `atp-platform/`.

### 3.1. Smoke-тесты (один агент)

```bash
# OpenAI
uv run atp test demo/test_suites/smoke.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001 \
  --adapter-config allow_internal=true \
  -v

# Anthropic
uv run atp test demo/test_suites/smoke.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8002 \
  --adapter-config allow_internal=true \
  -v
```

### 3.2. Функциональные тесты (3 прогона)

```bash
# OpenAI
uv run atp test demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001 \
  --adapter-config allow_internal=true \
  --runs=3 -v

# Anthropic
uv run atp test demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8002 \
  --adapter-config allow_internal=true \
  --runs=3 -v
```

### 3.3. Тесты качества

```bash
# OpenAI
uv run atp test demo/test_suites/quality.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001 \
  --adapter-config allow_internal=true \
  -v

# Anthropic
uv run atp test demo/test_suites/quality.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8002 \
  --adapter-config allow_internal=true \
  -v
```

---

## 4. Сохранение результатов

### 4.1. JSON-отчёты

```bash
# OpenAI
uv run atp test demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001 \
  --adapter-config allow_internal=true \
  --runs=3 \
  --output=json --output-file=demo/reports/openai_functional.json

# Anthropic
uv run atp test demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8002 \
  --adapter-config allow_internal=true \
  --runs=3 \
  --output=json --output-file=demo/reports/anthropic_functional.json
```

### 4.2. JUnit XML (для CI/CD)

```bash
uv run atp test demo/test_suites/smoke.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001 \
  --adapter-config allow_internal=true \
  --output=junit --output-file=demo/reports/openai_smoke.xml
```

---

## 5. Базовые линии

### 5.1. Сохранить baseline

```bash
# OpenAI baseline (10 прогонов)
uv run atp baseline save demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001 \
  --adapter-config allow_internal=true \
  -o demo/baselines/openai_baseline.json --runs=10

# Anthropic baseline
uv run atp baseline save demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8002 \
  --adapter-config allow_internal=true \
  -o demo/baselines/anthropic_baseline.json --runs=10
```

### 5.2. Сравнить с baseline (регрессия)

```bash
uv run atp baseline compare demo/test_suites/functional.yaml \
  --adapter=http \
  --adapter-config endpoint=http://localhost:8001 \
  --adapter-config allow_internal=true \
  -b demo/baselines/openai_baseline.json
```

---

## 6. Ручной curl-тест

```bash
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "version": "1.0",
    "task_id": "manual-001",
    "task": {
      "description": "Напиши функцию add(a, b)",
      "input_data": {
        "requirements": "Функция add(a, b) возвращает сумму двух чисел.",
        "language": "python",
        "filename": "add.py"
      },
      "expected_artifacts": ["add.py"]
    },
    "constraints": {
      "max_steps": 3,
      "max_tokens": 4096,
      "timeout_seconds": 30
    }
  }'
```

---

## 7. Типичные проблемы

| Проблема | Причина | Решение |
|----------|---------|---------|
| `Connection refused` | Агент не запущен | Запустите uvicorn на нужном порту |
| `OPENAI_API_KEY not set` | Нет переменной окружения | `export OPENAI_API_KEY=sk-...` |
| `ModuleNotFoundError: openai` | Пакет не установлен | `uv add openai` |
| `Timeout` | Модель долго отвечает | Увеличьте `timeout_seconds` в constraints |
| Код обёрнут в ` ```python ` | LLM добавила markdown | Агент автоматически очищает (strip_markdown_fences) |
| pytest fails: `ModuleNotFoundError` | Файл агента не в рабочей директории | ATP кладёт артефакт в sandbox, pytest должен запускаться оттуда |
| `atp: command not found` | Запуск не из корня atp-platform | `cd atp-platform && uv run atp ...` |
| `Security validation failed` | localhost blocked by SSRF protection | Add `--adapter-config allow_internal=true` |

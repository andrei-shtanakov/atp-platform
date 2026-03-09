# Контракт: Code Writer Agent (ATP Protocol)

## 1. Обзор

Агент получает задание на написание Python-кода и возвращает готовый файл.

```
ATP Platform                          Code Writer Agent
┌────────────┐                       ┌─────────────────┐
│ Test Suite  │──ATPRequest──────────►│ Принимает задание│
│  (YAML)    │                       │ Вызывает LLM     │
│            │◄──ATPResponse─────────│ Возвращает код    │
└────────────┘                       └─────────────────┘
```

---

## 2. ATPRequest — что получает агент

```json
{
  "version": "1.0",
  "task_id": "task_fibonacci_001",
  "task": {
    "description": "Напиши Python-функцию для вычисления n-го числа Фибоначчи",
    "input_data": {
      "requirements": "Функция fibonacci(n) принимает целое число n >= 0. Возвращает n-е число Фибоначчи. fibonacci(0)=0, fibonacci(1)=1. Для отрицательных n — ValueError.",
      "language": "python",
      "filename": "fibonacci.py"
    },
    "expected_artifacts": ["fibonacci.py"]
  },
  "constraints": {
    "max_steps": 3,
    "max_tokens": 10000,
    "timeout_seconds": 60,
    "budget_usd": 0.05
  },
  "context": {
    "workspace_path": "/workspace",
    "environment": {}
  },
  "metadata": {
    "test_id": "SM-001",
    "run_number": 1
  }
}
```

### 2.1. Поля input_data

| Поле | Тип | Обязательное | Описание |
|------|-----|:---:|----------|
| `requirements` | string | да | Подробное описание требований к коду |
| `language` | string | да | Язык программирования (всегда `"python"`) |
| `filename` | string | да | Имя файла для артефакта (напр. `"fibonacci.py"`) |

---

## 3. ATPResponse — что возвращает агент

```json
{
  "version": "1.0",
  "task_id": "task_fibonacci_001",
  "status": "completed",
  "artifacts": [
    {
      "type": "file",
      "path": "fibonacci.py",
      "content_type": "text/x-python",
      "content": "def fibonacci(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n"
    }
  ],
  "metrics": {
    "total_tokens": 850,
    "input_tokens": 320,
    "output_tokens": 530,
    "total_steps": 1,
    "tool_calls": 0,
    "llm_calls": 1,
    "wall_time_seconds": 3.2,
    "cost_usd": 0.004
  },
  "error": null
}
```

### 3.1. Правила формирования ответа

1. **`status`**: `"completed"` если код сгенерирован, `"failed"` при ошибке LLM/API
2. **`artifacts`**: ровно один файл с `path` = значению `input_data.filename`
3. **`content`**: чистый Python-код без markdown-обёрток (без ` ```python `)
4. **`metrics`**: обязательно заполнить `total_tokens`, `wall_time_seconds`, `cost_usd`

---

## 4. ATPEvent — потоковые события (опционально)

```json
{
  "version": "1.0",
  "task_id": "task_fibonacci_001",
  "timestamp": "2026-03-08T12:00:01Z",
  "sequence": 1,
  "event_type": "llm_request",
  "payload": {
    "model": "gpt-4o",
    "prompt_tokens": 320
  }
}
```

Агенты могут (но не обязаны) стримить события. ATP использует их для observability.

---

## 5. Статусы и ошибки

| Status | Когда | Пример |
|--------|-------|--------|
| `completed` | Код сгенерирован | Нормальная работа |
| `failed` | Ошибка API / невозможно сгенерировать | API key invalid, rate limit |
| `timeout` | Превышен `timeout_seconds` | Модель не ответила за 60s |
| `partial` | Код сгенерирован, но не полностью | Обрезка по `max_tokens` |

При `failed` поле `error` содержит описание:
```json
{
  "status": "failed",
  "artifacts": [],
  "error": "OpenAI API error: rate limit exceeded, retry after 20s"
}
```

---

## 6. Валидация ответа (JSON Schema)

Используется в assertion `schema` для smoke-тестов:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["version", "task_id", "status", "artifacts", "metrics"],
  "properties": {
    "version": { "type": "string", "const": "1.0" },
    "task_id": { "type": "string", "minLength": 1 },
    "status": {
      "type": "string",
      "enum": ["completed", "failed", "timeout", "partial"]
    },
    "artifacts": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["type", "path", "content"],
        "properties": {
          "type": { "type": "string", "enum": ["file", "structured"] },
          "path": { "type": "string", "pattern": "\\.py$" },
          "content_type": { "type": "string" },
          "content": { "type": "string", "minLength": 1 }
        }
      }
    },
    "metrics": {
      "type": "object",
      "required": ["total_tokens", "wall_time_seconds", "cost_usd"],
      "properties": {
        "total_tokens": { "type": "integer", "minimum": 0 },
        "input_tokens": { "type": "integer", "minimum": 0 },
        "output_tokens": { "type": "integer", "minimum": 0 },
        "total_steps": { "type": "integer", "minimum": 0 },
        "llm_calls": { "type": "integer", "minimum": 0 },
        "wall_time_seconds": { "type": "number", "minimum": 0 },
        "cost_usd": { "type": "number", "minimum": 0 }
      }
    },
    "error": { "type": ["string", "null"] }
  }
}
```

---

## 7. Примеры заданий (preview)

| Задание | filename | Ключевые требования |
|---------|----------|-------------------|
| Fibonacci | `fibonacci.py` | fibonacci(n), ValueError для n<0, edge cases |
| CSV-парсер | `csv_parser.py` | read_csv(path), filter_rows(data, column, value), write_csv(data, path) |
| REST API клиент | `api_client.py` | get/post методы, retry, timeout, обработка HTTP ошибок |

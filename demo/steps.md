# Пошаговый план: тестирование двух Code Writer агентов

## Цель
Сравнить двух AI-агентов для написания Python-кода:
- **Agent A**: OpenAI (GPT-4o)
- **Agent B**: Anthropic (Claude Sonnet 4)

Оба получают одинаковое задание → пишут Python-программу → ATP оценивает результат.

---

## Шаги

### Шаг 1: Цели тестирования и scope ✅
- [x] Определить, что тестируем (два code writer агента)
- [x] Определить категории тестов
- [x] Определить критерии приёмки

**Результат**: `docs/01-test-plan.md`

### Шаг 2: Контракт (ATP Protocol) ✅
- [x] Описать формат ATPRequest (input_data: задание на код)
- [x] Описать формат ATPResponse (artifacts: Python-файл)
- [x] Описать метрики (tokens, cost, time)
- [x] JSON Schema для валидации ответа

**Результат**: `docs/02-contract.md`

### Шаг 3: Окружение ✅
- [x] Создать структуру директорий
- [x] Создать atp.config.yaml (два агента: порты 8001, 8002)
- [x] Настроить .env.example (API-ключи)
- [x] JSON Schema для валидации ответов

**Результат**: `atp.config.yaml`, `.env.example`, `fixtures/response_schema.json`

### Шаг 4: Тест-сьют (YAML) ✅
- [x] Smoke-тесты: SM-001 (файл есть), SM-002 (schema), SM-003 (компилируется)
- [x] Функциональные: FN-001 (fibonacci), FN-002 (csv_parser), FN-003 (api_client)
- [x] Качество: QL-001 (docstrings), QL-002 (PEP 8), QL-003 (edge cases)

**Результат**: `test_suites/smoke.yaml`, `test_suites/functional.yaml`, `test_suites/quality.yaml`

### Шаг 5: Фикстуры ✅
- [x] Задания на код: fibonacci, csv_parser, api_client
- [x] Pytest-тесты: test_fibonacci (10 тестов), test_csv_parser (10 тестов), test_api_client (7 тестов)
- [x] JSON Schema для валидации ответов (шаг 3)

**Результат**: `fixtures/tasks/*.md`, `fixtures/tests/test_*.py`, `fixtures/response_schema.json`

### Шаг 6: Агенты ✅
- [x] HTTP-агент с OpenAI API (FastAPI, порт 8001)
- [x] HTTP-агент с Anthropic API (FastAPI, порт 8002)
- [x] Общий system prompt для честного сравнения
- [x] strip_markdown_fences() — очистка от ```python обёрток
- [x] Подсчёт стоимости по актуальным ценам

**Результат**: `agents/openai_agent.py`, `agents/anthropic_agent.py`

### Шаг 7: Адаптер и запуск
- [x] Документация по запуску (docs/03-run-guide.md)
- [ ] Запустить smoke-тесты для каждого агента
- [ ] Убедиться, что оба работают

**Результат**: `docs/03-run-guide.md`, рабочие тесты

### Шаг 8: Сравнение и базовая линия
- [x] Инструкции по baseline и сравнению (docs/03-run-guide.md §5-6)
- [ ] Запустить полный сьют для обоих агентов
- [ ] Сохранить базовые линии
- [ ] Сравнить результаты

**Результат**: `baselines/`, отчёт сравнения

---

## Структура директорий (целевая)

```
atp-platform-artefacts/
├── steps.md                    # Этот файл
├── atp.config.yaml             # Конфигурация ATP
├── .env.example                # Шаблон переменных окружения
├── docs/
│   ├── 01-test-plan.md         # План тестирования
│   └── 02-contract.md          # Контракт ATP Protocol
├── agents/
│   ├── openai_agent.py         # Agent A: OpenAI GPT-4o
│   └── anthropic_agent.py      # Agent B: Anthropic Claude
├── test_suites/
│   ├── smoke.yaml              # Smoke-тесты
│   ├── functional.yaml         # Функциональные тесты
│   └── quality.yaml            # Тесты качества кода
├── fixtures/
│   ├── tasks/                  # Задания на код
│   │   ├── fibonacci.md
│   │   ├── csv_parser.md
│   │   └── rest_api.md
│   ├── tests/                  # Pytest-тесты для проверки кода
│   │   ├── test_fibonacci.py
│   │   ├── test_csv_parser.py
│   │   └── test_rest_api.py
│   └── response_schema.json    # JSON Schema ответа
├── baselines/                  # Базовые линии
└── reports/                    # Отчёты
```

---

## Текущий статус
**Активный шаг**: 7 — Адаптер и запуск

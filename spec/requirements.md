# Requirements Specification

> Agent Test Platform (ATP) — Framework-agnostic платформа для тестирования AI-агентов

## 1. Контекст и цели

### 1.1 Проблема

AI-агенты становятся критическими компонентами бизнес-процессов, но отсутствуют стандарты их тестирования:
- Каждая команда изобретает собственные подходы
- Результаты несравнимы между проектами
- Регрессии обнаруживаются в production
- Смена фреймворка требует переписывания тестов

### 1.2 Цели проекта

| ID | Цель | Метрика успеха |
|----|------|----------------|
| G-1 | Унифицировать тестирование агентов | 3+ команды используют единый подход |
| G-2 | Обеспечить framework-независимость | Поддержка 3+ фреймворков без изменения тестов |
| G-3 | Автоматизировать regression detection | 95% регрессий обнаруживаются автоматически |
| G-4 | Сократить time-to-first-test | < 1 час от установки до первого теста |

### 1.3 Стейкхолдеры

| Роль | Интересы | Влияние |
|------|----------|---------|
| ML/AI инженеры | Простота интеграции, быстрый feedback | Высокое |
| QA инженеры | Декларативные тесты, понятные отчёты | Высокое |
| Tech Leads | Сравнение подходов, метрики качества | Среднее |
| DevOps | CI/CD интеграция, автоматизация | Среднее |

### 1.4 Out of Scope

- ❌ Разработка самих агентов (только тестирование)
- ❌ Хостинг агентов (запуск в инфраструктуре команд)
- ❌ Замена unit-тестов кода (дополнение, не замена pytest/jest)
- ❌ Realtime мониторинг в production (только pre-deploy тестирование)
- ❌ Визуальный редактор тестов (только YAML/CLI)

---

## 2. Функциональные требования

### 2.1 Протокол взаимодействия

#### REQ-001: Стандартный формат запроса
**As a** разработчик агента
**I want** отправлять задачи агенту в стандартном формате
**So that** любой агент может быть протестирован единообразно

**Acceptance Criteria:**
```gherkin
GIVEN агент реализует ATP Protocol
WHEN платформа отправляет ATP Request
THEN агент получает JSON с полями: version, task_id, task, constraints
AND task содержит description и опциональный input_data
AND constraints содержит max_steps, max_tokens, timeout_seconds, allowed_tools
```

**Priority:** P0 (Must Have)
**Traces to:** [TASK-001], [DESIGN-001]

---

#### REQ-002: Стандартный формат ответа
**As a** платформа тестирования
**I want** получать результаты в стандартном формате
**So that** можно единообразно оценивать любых агентов

**Acceptance Criteria:**
```gherkin
GIVEN агент завершил выполнение задачи
WHEN агент возвращает ATP Response
THEN ответ содержит: version, task_id, status, artifacts, metrics
AND status один из: completed, failed, timeout, cancelled, partial
AND artifacts — массив с type, path/name, content/data
AND metrics содержит: total_tokens, total_steps, tool_calls, wall_time_seconds
```

**Priority:** P0
**Traces to:** [TASK-001], [DESIGN-001]

---

#### REQ-003: Streaming событий
**As a** разработчик
**I want** получать события во время выполнения агента
**So that** можно отлаживать и анализировать поведение

**Acceptance Criteria:**
```gherkin
GIVEN агент поддерживает event streaming
WHEN агент выполняет задачу
THEN платформа получает ATP Events с типами: tool_call, llm_request, reasoning, error, progress
AND каждое событие имеет timestamp и sequence number
AND события упорядочены по sequence
```

**Priority:** P1 (Should Have)
**Traces to:** [TASK-002], [DESIGN-002]

---

### 2.2 Интеграция агентов

#### REQ-010: HTTP интеграция
**As a** разработчик с HTTP API агентом
**I want** интегрировать агента через HTTP endpoint
**So that** не нужно менять архитектуру агента

**Acceptance Criteria:**
```gherkin
GIVEN агент имеет HTTP endpoint
WHEN агент зарегистрирован с type: http и endpoint URL
THEN платформа отправляет POST запрос с ATP Request
AND платформа получает ATP Response в теле ответа
AND таймаут настраивается в конфигурации
```

**Priority:** P0
**Traces to:** [TASK-003], [DESIGN-003]

---

#### REQ-011: Container интеграция
**As a** разработчик с Docker-упакованным агентом
**I want** запускать агента в изолированном контейнере
**So that** обеспечить безопасность и воспроизводимость

**Acceptance Criteria:**
```gherkin
GIVEN агент упакован в Docker image
WHEN агент зарегистрирован с type: container и image name
THEN платформа запускает контейнер с ограничениями ресурсов
AND ATP Request передаётся через stdin
AND ATP Response читается из stdout
AND ATP Events читаются из stderr
AND контейнер удаляется после выполнения
```

**Priority:** P0
**Traces to:** [TASK-003], [DESIGN-003]

---

#### REQ-012: Framework адаптеры
**As a** разработчик на LangGraph/CrewAI
**I want** использовать готовый адаптер для моего фреймворка
**So that** не писать boilerplate код интеграции

**Acceptance Criteria:**
```gherkin
GIVEN существует адаптер для фреймворка X
WHEN агент зарегистрирован с type: X и путём к модулю
THEN адаптер автоматически транслирует ATP Protocol в native API
AND события фреймворка конвертируются в ATP Events
AND метрики собираются автоматически
```

**Priority:** P1
**Traces to:** [TASK-010], [DESIGN-003]

---

### 2.3 Описание тестов

#### REQ-020: Декларативный формат тестов
**As a** QA инженер
**I want** описывать тесты в YAML без написания кода
**So that** тесты понятны всей команде

**Acceptance Criteria:**
```gherkin
GIVEN тест описан в YAML файле
WHEN файл содержит: id, name, task, assertions
THEN платформа парсит и валидирует структуру
AND выводит понятные ошибки при невалидном формате
AND поддерживает комментарии для документации
```

**Priority:** P0
**Traces to:** [TASK-004], [DESIGN-004]

---

#### REQ-021: Test Suites
**As a** разработчик
**I want** группировать связанные тесты в suites
**So that** запускать их вместе и переиспользовать настройки

**Acceptance Criteria:**
```gherkin
GIVEN suite содержит defaults и список tests
WHEN запускается suite
THEN defaults применяются ко всем тестам
AND тесты могут переопределять defaults
AND можно запустить отдельный тест из suite
```

**Priority:** P0
**Traces to:** [TASK-004], [DESIGN-004]

---

#### REQ-022: Tags и фильтрация
**As a** разработчик в CI/CD
**I want** запускать подмножество тестов по tags
**So that** быстро проверять smoke tests или только regression

**Acceptance Criteria:**
```gherkin
GIVEN тесты имеют tags: [smoke, regression, edge_case]
WHEN запуск с --tags=smoke
THEN выполняются только тесты с tag "smoke"
AND можно комбинировать tags: --tags=smoke,core
AND можно исключать: --tags=!slow
```

**Priority:** P1
**Traces to:** [TASK-005], [DESIGN-004]

---

### 2.4 Выполнение тестов

#### REQ-030: Test Runner
**As a** разработчик
**I want** запускать тесты через CLI
**So that** интегрировать в локальную разработку и CI

**Acceptance Criteria:**
```gherkin
GIVEN установлена платформа ATP
WHEN выполняется команда: atp test --agent=X --suite=Y
THEN загружается suite Y
AND запускается агент X для каждого теста
AND выводится progress и результаты
AND возвращается exit code 0 при успехе, non-zero при failures
```

**Priority:** P0
**Traces to:** [TASK-006], [DESIGN-005]

---

#### REQ-031: Множественные прогоны
**As a** разработчик
**I want** запускать тест N раз
**So that** получить статистически значимые результаты

**Acceptance Criteria:**
```gherkin
GIVEN тест настроен с runs: 5
WHEN тест выполняется
THEN агент запускается 5 раз с одинаковым input
AND вычисляются: mean, std, min, max, median
AND вычисляется 95% confidence interval
AND определяется stability level по coefficient of variation
```

**Priority:** P1
**Traces to:** [TASK-011], [DESIGN-006]

---

#### REQ-032: Timeout и ограничения
**As a** платформа
**I want** принудительно останавливать агента при превышении лимитов
**So that** тесты не зависают бесконечно

**Acceptance Criteria:**
```gherkin
GIVEN тест имеет constraints.timeout_seconds: 60
WHEN агент выполняется дольше 60 секунд
THEN агент принудительно останавливается
AND возвращается response со status: timeout
AND собранные до этого артефакты и метрики сохраняются
```

**Priority:** P0
**Traces to:** [TASK-006], [DESIGN-005]

---

### 2.5 Система оценки

#### REQ-040: Artifact Evaluator
**As a** тестировщик
**I want** проверять наличие и содержимое артефактов
**So that** убедиться что агент создал ожидаемые выходные данные

**Acceptance Criteria:**
```gherkin
GIVEN assertion type: artifact_exists с path: "report.md"
WHEN агент возвращает artifacts
THEN проверяется наличие артефакта с указанным path
AND check passed если артефакт существует
AND check failed с понятным сообщением если не существует

GIVEN assertion type: contains с pattern: "competitor"
WHEN артефакт существует
THEN проверяется наличие pattern в содержимом
AND поддерживается regex: true для регулярных выражений
```

**Priority:** P0
**Traces to:** [TASK-007], [DESIGN-007]

---

#### REQ-041: Behavior Evaluator
**As a** тестировщик
**I want** проверять поведение агента по trace
**So that** убедиться что агент работает эффективно и безопасно

**Acceptance Criteria:**
```gherkin
GIVEN assertion type: behavior с must_use_tools: [web_search]
WHEN анализируется trace выполнения
THEN проверяется что tool web_search был вызван
AND check failed если инструмент не использовался

GIVEN assertion с max_tool_calls: 10
WHEN количество tool calls > 10
THEN check failed с указанием actual vs limit
```

**Priority:** P0
**Traces to:** [TASK-007], [DESIGN-007]

---

#### REQ-042: LLM-as-Judge Evaluator
**As a** тестировщик
**I want** использовать LLM для семантической оценки качества
**So that** проверять смысловую корректность, а не только формат

**Acceptance Criteria:**
```gherkin
GIVEN assertion type: llm_eval с criteria: factual_accuracy
WHEN артефакт передаётся на оценку LLM
THEN LLM возвращает score 0-1 и explanation
AND check passed если score >= threshold (default 0.7)
AND explanation включается в отчёт

GIVEN criteria: custom с prompt: "..."
WHEN выполняется оценка
THEN используется custom prompt вместо стандартного
```

**Priority:** P1
**Traces to:** [TASK-012], [DESIGN-008]

---

#### REQ-043: Composite Scoring
**As a** менеджер
**I want** получать единый score 0-100 для каждого теста
**So that** легко сравнивать агентов и отслеживать прогресс

**Acceptance Criteria:**
```gherkin
GIVEN тест имеет scoring weights: quality: 0.4, completeness: 0.3, efficiency: 0.2, cost: 0.1
WHEN все evaluators завершились
THEN вычисляется weighted score по формуле
AND score нормализуется к диапазону 0-100
AND breakdown по компонентам включается в отчёт
```

**Priority:** P1
**Traces to:** [TASK-008], [DESIGN-007]

---

### 2.6 Reporting

#### REQ-050: Console Reporter
**As a** разработчик
**I want** видеть результаты в терминале
**So that** быстро понять статус тестов

**Acceptance Criteria:**
```gherkin
GIVEN тесты завершились
WHEN используется console reporter (default)
THEN выводится summary: X passed, Y failed, Z skipped
AND для каждого теста: статус (✓/✗), score, duration
AND failed checks выводятся с деталями
AND поддерживается --verbose для полного вывода
```

**Priority:** P0
**Traces to:** [TASK-009], [DESIGN-009]

---

#### REQ-051: JSON Reporter
**As a** CI/CD система
**I want** получать результаты в machine-readable формате
**So that** интегрировать с другими инструментами

**Acceptance Criteria:**
```gherkin
GIVEN запуск с --output=json --output-file=results.json
WHEN тесты завершились
THEN создаётся JSON файл с полной структурой результатов
AND формат документирован и стабилен между версиями
```

**Priority:** P0
**Traces to:** [TASK-009], [DESIGN-009]

---

#### REQ-052: Baseline и Regression
**As a** разработчик
**I want** сравнивать результаты с baseline
**So that** автоматически обнаруживать регрессии

**Acceptance Criteria:**
```gherkin
GIVEN существует baseline файл от предыдущего запуска
WHEN запуск с --baseline=baseline.json
THEN текущие результаты сравниваются с baseline
AND regression определяется как статистически значимое ухудшение (p < 0.05)
AND improvement также отмечается
AND diff выводится в отчёте
```

**Priority:** P2 (Could Have)
**Traces to:** [TASK-013], [DESIGN-010]

---

## 3. Нефункциональные требования

### NFR-000: Testing Requirements
| Аспект | Требование |
|--------|------------|
| Unit test coverage | ≥ 80% для core modules |
| Integration tests | Каждый adapter, evaluator |
| E2E tests | Critical paths (test run, reporting) |
| Test framework | pytest + pytest-asyncio |
| CI requirement | Все тесты проходят перед merge |

**Definition of Done для любой задачи:**
- [ ] Unit tests написаны и проходят
- [ ] Coverage не упал
- [ ] Integration test если затронуты интерфейсы
- [ ] Документация обновлена

**Traces to:** [TASK-100], [TASK-101], [TASK-102]

---

### NFR-001: Performance
| Метрика | Требование |
|---------|------------|
| Overhead платформы | < 5% от времени выполнения агента |
| CLI startup time | < 2 секунды |
| Параллельные агенты | До 10 одновременно |
| Обработка событий | 10,000+ событий без деградации |

**Traces to:** [TASK-006]

---

### NFR-002: Reliability
| Аспект | Требование |
|--------|------------|
| Timeout handling | Graceful stop без потери данных |
| Agent crash | Продолжение остальных тестов |
| Partial results | Сохранение при прерывании |

**Traces to:** [TASK-006]

---

### NFR-003: Usability
| Метрика | Требование |
|---------|------------|
| Time to first test | < 1 час для нового пользователя |
| Error messages | Actionable, указывают на решение |
| Documentation | Покрывает все use cases |

**Traces to:** [TASK-014]

---

### NFR-004: Security
| Аспект | Требование |
|--------|------------|
| Sandbox isolation | Docker с ограничениями CPU/memory/network |
| Secrets | Через env vars, не в тестах/логах |
| Input validation | Все входы валидируются по схеме |

**Traces to:** [TASK-006], [DESIGN-005]

---

### NFR-005: Compatibility
| Платформа | Требование |
|-----------|------------|
| Python | 3.10+ |
| OS | Linux (primary), macOS (dev), Windows (best effort) |
| Docker | 20.10+ |
| CI systems | GitHub Actions, GitLab CI |

**Traces to:** [TASK-003]

---

## 4. Ограничения и техстек

### 4.1 Технологические ограничения

| Аспект | Решение | Обоснование |
|--------|---------|-------------|
| Язык | Python 3.10+ | Экосистема ML/AI |
| Packaging | pip + pyproject.toml | Стандарт Python |
| Schema | JSON Schema draft-07 | Широкая поддержка |
| Container | Docker (primary) | Industry standard |
| Config format | YAML | Читаемость |

### 4.2 Интеграционные ограничения

- Протокол: JSON over HTTP / stdin-stdout
- LLM для evaluation: Claude или OpenAI API
- CI: JUnit XML для совместимости

### 4.3 Лицензирование

- Платформа: MIT License
- Зависимости: только MIT/Apache/BSD compatible

---

## 5. Критерии приёмки

### Milestone 1: MVP
- [ ] REQ-001, REQ-002 — Protocol implemented
- [ ] REQ-010, REQ-011 — HTTP и Container adapters working
- [ ] REQ-020, REQ-021 — YAML tests loading
- [ ] REQ-030, REQ-032 — Runner with timeout
- [ ] REQ-040, REQ-041 — Artifact и Behavior evaluators
- [ ] REQ-050, REQ-051 — Console и JSON reporters
- [ ] NFR-003 — Documentation complete

### Milestone 2: Beta
- [ ] REQ-003 — Event streaming
- [ ] REQ-012 — LangGraph и CrewAI adapters
- [ ] REQ-022 — Tags filtering
- [ ] REQ-031 — Multiple runs with statistics
- [ ] REQ-042 — LLM-as-Judge evaluator
- [ ] REQ-043 — Composite scoring
- [ ] NFR-001 — Performance targets met

### Milestone 3: GA
- [ ] REQ-052 — Baseline comparison
- [ ] NFR-002 — Reliability hardened
- [ ] NFR-004 — Security audit passed
- [ ] All P0 and P1 requirements implemented

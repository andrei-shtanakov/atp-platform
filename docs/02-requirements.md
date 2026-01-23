# Requirements Specification

## Обзор

Данный документ описывает функциональные и нефункциональные требования к Agent Test Platform (ATP).

---

## Функциональные требования

### FR-1: Протокол взаимодействия с агентами

#### FR-1.1: Стандартный формат запроса
- Система ДОЛЖНА определять JSON-схему для запросов к агентам (ATP Request)
- Запрос ДОЛЖЕН включать: task description, input data, constraints
- Constraints ДОЛЖНЫ поддерживать: max_steps, max_tokens, timeout, allowed_tools

#### FR-1.2: Стандартный формат ответа
- Система ДОЛЖНА определять JSON-схему для ответов агентов (ATP Response)
- Ответ ДОЛЖЕН включать: status, artifacts, metrics
- Metrics ДОЛЖНЫ включать: total_tokens, total_steps, tool_calls, wall_time

#### FR-1.3: Streaming событий
- Система ДОЛЖНА поддерживать streaming событий во время выполнения (ATP Event)
- События ДОЛЖНЫ включать типы: tool_call, llm_request, reasoning, error
- Каждое событие ДОЛЖНО иметь timestamp и sequence number

#### FR-1.4: Версионирование протокола
- Протокол ДОЛЖЕН иметь версию в каждом сообщении
- Система ДОЛЖНА поддерживать backward compatibility

### FR-2: Интеграция агентов

#### FR-2.1: HTTP интеграция
- Система ДОЛЖНА поддерживать агентов через HTTP endpoint
- Endpoint ДОЛЖЕН принимать POST с ATP Request
- Endpoint МОЖЕТ поддерживать SSE для streaming событий

#### FR-2.2: Container интеграция
- Система ДОЛЖНА поддерживать агентов как Docker containers
- Container ДОЛЖЕН читать ATP Request из stdin
- Container ДОЛЖЕН писать ATP Response в stdout
- Container МОЖЕТ писать ATP Events в stderr или отдельный порт

#### FR-2.3: Framework адаптеры
- Система ДОЛЖНА предоставлять базовый класс для адаптеров
- Адаптер ДОЛЖЕН транслировать ATP Protocol ↔ native framework API
- Система ДОЛЖНА включать адаптеры для: LangGraph, CrewAI
- Система МОЖЕТ включать legacy адаптер для AutoGen

#### FR-2.4: CLI wrapper
- Система ДОЛЖНА поддерживать агентов через CLI
- CLI агент ДОЛЖЕН принимать путь к файлу с ATP Request
- CLI агент ДОЛЖЕН записывать ATP Response в указанный файл

### FR-3: Описание тестов

#### FR-3.1: Декларативный формат
- Тесты ДОЛЖНЫ описываться в YAML формате
- Формат ДОЛЖЕН поддерживать: task definition, assertions, scoring weights
- Формат ДОЛЖЕН быть человеко-читаемым

#### FR-3.2: Test Suites
- Система ДОЛЖНА поддерживать группировку тестов в suites
- Suite ДОЛЖЕН иметь: name, description, defaults, list of tests
- Defaults ДОЛЖНЫ наследоваться тестами и переопределяться

#### FR-3.3: Параметризация тестов
- Тесты ДОЛЖНЫ поддерживать параметры (переменные)
- Параметры ДОЛЖНЫ подставляться в task description и input data
- Система ДОЛЖНА поддерживать matrix параметризацию

#### FR-3.4: Tags и фильтрация
- Тесты ДОЛЖНЫ поддерживать tags
- CLI ДОЛЖЕН поддерживать фильтрацию по tags
- Стандартные tags: smoke, regression, edge_case, performance

### FR-4: Выполнение тестов

#### FR-4.1: Test Runner
- Система ДОЛЖНА предоставлять CLI для запуска тестов
- Runner ДОЛЖЕН поддерживать: single test, test suite, all tests
- Runner ДОЛЖЕН выводить progress и результаты

#### FR-4.2: Sandbox окружение
- Система ДОЛЖНА изолировать выполнение агентов
- Sandbox ДОЛЖЕН ограничивать: время, память, сеть (опционально)
- Sandbox ДОЛЖЕН использовать Docker для изоляции

#### FR-4.3: Mock Tools
- Система ДОЛЖНА предоставлять mock tools для тестирования
- Mock ДОЛЖЕН записывать вызовы для проверки
- Mock ДОЛЖЕН поддерживать: фиксированные ответы, файлы ответов, callback

#### FR-4.4: Параллельное выполнение
- Runner ДОЛЖЕН поддерживать параллельный запуск тестов
- Параллелизм ДОЛЖЕН настраиваться через CLI и config
- Система ДОЛЖНА корректно агрегировать результаты

#### FR-4.5: Множественные прогоны
- Система ДОЛЖНА поддерживать N прогонов одного теста
- Система ДОЛЖНА вычислять статистику по прогонам
- Статистика ДОЛЖНА включать: mean, std, min, max, confidence interval

### FR-5: Система оценки

#### FR-5.1: Artifact Evaluator
- Система ДОЛЖНА проверять наличие и формат артефактов
- Проверки ДОЛЖНЫ включать: file exists, JSON schema, contains text
- Проверки ДОЛЖНЫ поддерживать regex patterns

#### FR-5.2: Behavior Evaluator
- Система ДОЛЖНА анализировать trace выполнения
- Проверки ДОЛЖНЫ включать: used tools, number of steps, no hallucinations
- Проверки ДОЛЖНЫ поддерживать: must, must_not, should (warning)

#### FR-5.3: LLM-as-Judge Evaluator
- Система ДОЛЖНА использовать LLM для семантической оценки
- Evaluator ДОЛЖЕН поддерживать: custom prompts, criteria
- Evaluator ДОЛЖЕН возвращать score 0-1 и explanation

#### FR-5.4: Code Execution Evaluator
- Система ДОЛЖНА запускать сгенерированный код
- Evaluator ДОЛЖЕН поддерживать: pytest, npm test, custom command
- Результаты тестов ДОЛЖНЫ конвертироваться в метрики

#### FR-5.5: Composite Scoring
- Система ДОЛЖНА поддерживать weighted scoring
- Веса ДОЛЖНЫ настраиваться в test definition
- Итоговый score ДОЛЖЕН быть 0-100

### FR-6: Reporting

#### FR-6.1: Console Reporter
- Система ДОЛЖНА выводить результаты в консоль
- Вывод ДОЛЖЕН включать: pass/fail, score, duration
- Вывод ДОЛЖЕН поддерживать verbose mode для деталей

#### FR-6.2: JSON Reporter
- Система ДОЛЖНА экспортировать результаты в JSON
- JSON ДОЛЖЕН включать полную информацию для анализа
- Формат ДОЛЖЕН быть документирован

#### FR-6.3: HTML Reporter
- Система ДОЛЖНА генерировать HTML отчёт
- Отчёт ДОЛЖЕН включать: summary, details, charts
- Отчёт ДОЛЖЕН быть self-contained (single file)

#### FR-6.4: JUnit XML Reporter
- Система ДОЛЖНА поддерживать JUnit XML формат
- Формат ДОЛЖЕН быть совместим с CI системами

### FR-7: CI/CD интеграция

#### FR-7.1: Exit codes
- CLI ДОЛЖЕН возвращать 0 при успехе всех тестов
- CLI ДОЛЖЕН возвращать non-zero при failures
- CLI ДОЛЖЕН поддерживать --fail-fast для раннего выхода

#### FR-7.2: Baseline comparison
- Система ДОЛЖНА поддерживать сохранение baseline результатов
- Система ДОЛЖНА сравнивать текущие результаты с baseline
- Regression ДОЛЖЕН определяться как статистически значимое ухудшение

#### FR-7.3: GitHub Actions integration
- Система ДОЛЖНА предоставлять готовый action
- Action ДОЛЖЕН поддерживать: matrix builds, caching, artifacts

### FR-8: Configuration

#### FR-8.1: Project config
- Система ДОЛЖНА поддерживать atp.config.yaml в корне проекта
- Config ДОЛЖЕН включать: defaults, agent definitions, paths
- CLI опции ДОЛЖНЫ переопределять config

#### FR-8.2: Agent registry
- Система ДОЛЖНА поддерживать регистрацию агентов
- Registry ДОЛЖЕН хранить: name, type, endpoint/image, config
- Агенты ДОЛЖНЫ ссылаться по имени в тестах

---

## Нефункциональные требования

### NFR-1: Performance

#### NFR-1.1: Test execution overhead
- Overhead платформы ДОЛЖЕН быть < 5% от времени выполнения агента
- Startup time CLI ДОЛЖЕН быть < 2 секунды

#### NFR-1.2: Scalability
- Система ДОЛЖНА поддерживать 100+ тестов в suite
- Система ДОЛЖНА поддерживать 10+ параллельных агентов
- Reporting ДОЛЖЕН обрабатывать 10,000+ событий

### NFR-2: Reliability

#### NFR-2.1: Error handling
- Система ДОЛЖНА gracefully обрабатывать timeout агента
- Система ДОЛЖНА gracefully обрабатывать crash агента
- Система ДОЛЖНА продолжать выполнение при failure одного теста

#### NFR-2.2: Idempotency
- Повторный запуск теста ДОЛЖЕН давать статистически схожие результаты
- Cleanup между тестами ДОЛЖЕН быть полным

### NFR-3: Usability

#### NFR-3.1: Learning curve
- Новый пользователь ДОЛЖЕН запустить первый тест за < 1 час
- Документация ДОЛЖНА включать tutorials и examples

#### NFR-3.2: Error messages
- Ошибки ДОЛЖНЫ быть понятными и actionable
- Ошибки ДОЛЖНЫ указывать на конкретное место в config/test

#### NFR-3.3: Defaults
- Система ДОЛЖНА работать с минимальной конфигурацией
- Разумные defaults ДОЛЖНЫ покрывать 80% случаев

### NFR-4: Maintainability

#### NFR-4.1: Code quality
- Код ДОЛЖЕН иметь type hints (Python)
- Код ДОЛЖЕН проходить linting (ruff/flake8)
- Test coverage ДОЛЖЕН быть > 80%

#### NFR-4.2: Documentation
- Public API ДОЛЖЕН быть документирован
- Architecture Decision Records ДОЛЖНЫ вестись
- CHANGELOG ДОЛЖЕН обновляться

#### NFR-4.3: Modularity
- Компоненты ДОЛЖНЫ быть слабо связаны
- Добавление нового evaluator НЕ ДОЛЖНО требовать изменения core

### NFR-5: Security

#### NFR-5.1: Sandbox isolation
- Агент НЕ ДОЛЖЕН иметь доступ к файловой системе хоста (кроме mounted paths)
- Агент НЕ ДОЛЖЕН иметь неограниченный доступ к сети

#### NFR-5.2: Secrets handling
- API ключи ДОЛЖНЫ передаваться через environment variables
- Секреты НЕ ДОЛЖНЫ логироваться
- Секреты НЕ ДОЛЖНЫ попадать в reports

### NFR-6: Compatibility

#### NFR-6.1: Python versions
- Система ДОЛЖНА поддерживать Python 3.10+
- Зависимости ДОЛЖНЫ быть совместимы с major frameworks

#### NFR-6.2: OS support
- Система ДОЛЖНА работать на Linux (primary)
- Система ДОЛЖНА работать на macOS (development)
- Система МОЖЕТ работать на Windows (best effort)

#### NFR-6.3: Docker
- Система ДОЛЖНА работать с Docker 20.10+
- Система МОЖЕТ поддерживать Podman как альтернативу

---

## Constraints

### C-1: Technology constraints
- Язык реализации: Python 3.10+
- Packaging: pip + pyproject.toml
- Container runtime: Docker (primary)

### C-2: Integration constraints
- Протокол: JSON over HTTP/stdin-stdout
- Schema validation: JSON Schema draft-07
- YAML parsing: PyYAML или ruamel.yaml

### C-3: Licensing constraints
- Платформа: MIT License
- Зависимости: только MIT/Apache/BSD compatible

---

## Acceptance Criteria

### MVP (Milestone 1)
- [ ] ATP Protocol schema defined and documented
- [ ] HTTP adapter working
- [ ] Container adapter working
- [ ] Artifact evaluator working
- [ ] Behavior evaluator working
- [ ] Console reporter working
- [ ] JSON reporter working
- [ ] CLI с командами: test, version
- [ ] 3+ example test suites
- [ ] Basic documentation

### Beta (Milestone 2)
- [ ] LangGraph adapter
- [ ] CrewAI adapter
- [ ] LLM-as-Judge evaluator
- [ ] HTML reporter
- [ ] Multiple runs with statistics
- [ ] Baseline comparison
- [ ] GitHub Action
- [ ] Comprehensive documentation

### GA (Milestone 3)
- [ ] Web Dashboard (basic)
- [ ] All evaluators documented
- [ ] Performance benchmarks
- [ ] Security audit
- [ ] Community feedback incorporated

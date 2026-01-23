# Agent Test Platform (ATP)

> Framework-agnostic платформа для тестирования и оценки AI-агентов

## Проблема

Современные AI-агенты — это сложные системы с недетерминированным поведением, многошаговой логикой и зависимостью от внешних инструментов. Традиционные подходы к тестированию ПО не работают для агентов:

- **Стохастичность**: один и тот же промпт даёт разные результаты
- **Emergent behavior**: поведение системы не сводится к сумме компонентов  
- **Цепочки решений**: ошибка на раннем шаге проявляется позже
- **Фреймворк-зависимость**: каждая команда использует свой стек

## Решение

ATP предоставляет унифицированный протокол и инфраструктуру для тестирования агентов независимо от фреймворка реализации (LangGraph, CrewAI, AutoGen, custom и др.).

### Ключевые принципы

1. **Framework Agnostic** — агент = чёрный ящик с контрактом (вход → выход + события)
2. **Test Driven Agent Development (TDAD)** — сначала тесты, потом агент
3. **Многоуровневая оценка** — от unit-тестов компонентов до E2E сценариев
4. **Воспроизводимость** — статистическая значимость через множественные прогоны

## Документация

| Документ | Описание |
|----------|----------|
| [Vision & Goals](docs/01-vision.md) | Видение проекта и цели |
| [Requirements](docs/02-requirements.md) | Функциональные и нефункциональные требования |
| [Architecture](docs/03-architecture.md) | Архитектура системы |
| [ATP Protocol](docs/04-protocol.md) | Спецификация протокола взаимодействия |
| [Evaluation System](docs/05-evaluators.md) | Система оценки и метрики |
| [Integration Guide](docs/06-integration.md) | Руководство по интеграции агентов |
| [Roadmap](docs/07-roadmap.md) | План развития |

### Architecture Decision Records

- [ADR-001: Framework Agnostic Design](docs/adr/001-framework-agnostic.md)
- [ADR-002: Protocol Design](docs/adr/002-protocol-design.md)
- [ADR-003: Evaluation Strategy](docs/adr/003-evaluation-strategy.md)

## Быстрый старт

```bash
# Установка
pip install atp-platform

# Запуск тестов для агента
atp test --agent=my-agent --suite=smoke

# Сравнение агентов
atp compare --agents=agent-a,agent-b --suite=full
```

## Структура проекта

```
atp-platform/
├── docs/                    # Документация
│   ├── 01-vision.md
│   ├── 02-requirements.md
│   ├── 03-architecture.md
│   ├── 04-protocol.md
│   ├── 05-evaluators.md
│   ├── 06-integration.md
│   ├── 07-roadmap.md
│   └── adr/                 # Architecture Decision Records
├── schemas/                 # JSON Schema для протокола
│   ├── atp-request.json
│   ├── atp-response.json
│   └── atp-event.json
├── examples/                # Примеры интеграции
│   ├── simple-http-agent/
│   ├── langgraph-adapter/
│   └── test-suites/
└── README.md
```

## Лицензия

MIT License

## Contributing

См. [CONTRIBUTING.md](CONTRIBUTING.md)

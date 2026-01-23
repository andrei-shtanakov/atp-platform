# Task Management Workflow

## Обзор

Система управления задачами работает напрямую с `spec/tasks.md` файлом:
- Статусы и чеклисты обновляются в markdown
- История изменений логируется в `.task-history.log`
- Зависимости отслеживаются автоматически
- **Автоматическое выполнение через Claude CLI**

## Быстрый старт

```bash
# === Ручной режим ===
make task-stats           # Статистика
make task-next            # Что делать дальше
make task-start ID=TASK-001
make task-done ID=TASK-001

# === Автоматический режим (Claude CLI) ===
make exec                 # Выполнить следующую задачу
make exec-all             # Выполнить все готовые
make exec-mvp             # Выполнить MVP задачи
make exec-status          # Статус выполнения
```

---

## Автоматическое выполнение (Claude CLI)

### Концепция

Executor запускает Claude CLI для каждой задачи:
1. Читает спецификацию (requirements.md, design.md)
2. Формирует промпт с контекстом задачи
3. Claude реализует код и тесты
4. Проверяет результат (тесты, lint)
5. При успехе — переходит к следующей задаче
6. При неудаче — retry с лимитом

### Команды

```bash
# Выполнить следующую готовую задачу
python executor.py run

# Выполнить конкретную задачу
python executor.py run --task=TASK-001

# Выполнить все готовые задачи
python executor.py run --all

# Только MVP задачи
python executor.py run --all --milestone=mvp

# Статус выполнения
python executor.py status

# Повторить неудавшуюся
python executor.py retry TASK-001

# Посмотреть логи
python executor.py logs TASK-001

# Сбросить состояние
python executor.py reset
```

### Опции

```bash
# Количество попыток (default: 3)
python executor.py run --max-retries=5

# Таймаут в минутах (default: 30)
python executor.py run --timeout=60

# Без тестов после выполнения
python executor.py run --no-tests

# Без создания git ветки
python executor.py run --no-branch

# Автокоммит при успехе
python executor.py run --auto-commit
```

### Workflow автоматического выполнения

```
┌─────────────────────────────────────────────────────────────┐
│                     executor.py run                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Найти следующую задачу (по приоритету + зависимостям)   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Pre-start hook                                          │
│     - Создать git branch: task/TASK-XXX-name                │
│     - Обновить статус: TODO → IN_PROGRESS                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Сформировать промпт                                     │
│     - Контекст из requirements.md, design.md                │
│     - Чеклист задачи                                        │
│     - Связанные REQ-XXX, DESIGN-XXX                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Запустить Claude CLI                                    │
│     claude -p "<prompt>"                                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Проверить результат                                     │
│     - Claude вернул "TASK_COMPLETE"?                        │
│     - Тесты проходят? (make test)                           │
│     - Lint чистый? (make lint)                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
     ┌──────────┐        ┌──────────┐
     │ SUCCESS  │        │  FAILED  │
     └────┬─────┘        └────┬─────┘
          │                   │
          ▼                   ▼
┌─────────────────┐   ┌─────────────────┐
│ Post-done hook  │   │ Retry?          │
│ - Auto-commit   │   │ attempts < max  │
│ - Mark DONE     │   └────────┬────────┘
│ - Next task     │            │
└─────────────────┘   ┌────────┴────────┐
                      │                 │
                      ▼                 ▼
               ┌──────────┐      ┌──────────┐
               │  RETRY   │      │   STOP   │
               │ (loop)   │      │ BLOCKED  │
               └──────────┘      └──────────┘
```

### Защитные механизмы

| Механизм | Default | Описание |
|----------|---------|----------|
| max_retries | 3 | Макс. попыток на задачу |
| max_consecutive_failures | 2 | Стоп после N неудач подряд |
| task_timeout | 30 min | Таймаут на задачу |
| post_done tests | ON | Проверка тестов |

### Логи

Логи сохраняются в `spec/.executor-logs/`:

```
spec/.executor-logs/
├── TASK-001-20250122-103000.log
├── TASK-001-20250122-103500.log  # retry
└── TASK-003-20250122-110000.log
```

Содержимое лога:
```
=== PROMPT ===
<полный промпт для Claude>

=== OUTPUT ===
<ответ Claude>

=== STDERR ===
<ошибки если есть>

=== RETURN CODE: 0 ===
```

### Конфигурация

Файл `executor.config.yaml`:

```yaml
executor:
  max_retries: 3
  task_timeout_minutes: 30
  
  hooks:
    pre_start:
      create_git_branch: true
    post_done:
      run_tests: true
      auto_commit: false
```

---

## CLI команды

### Просмотр

```bash
# Все задачи
python task.py list

# Фильтрация
python task.py list --status=todo
python task.py list --priority=p0
python task.py list --milestone=mvp

# Детали задачи
python task.py show TASK-001

# Статистика
python task.py stats

# Граф зависимостей
python task.py graph
```

### Управление статусом

```bash
# Начать работу
python task.py start TASK-001

# Начать, игнорируя зависимости
python task.py start TASK-001 --force

# Завершить
python task.py done TASK-001

# Заблокировать
python task.py block TASK-001
```

### Чеклист

```bash
# Показать задачу с чеклистом
python task.py show TASK-001

# Отметить пункт (toggle)
python task.py check TASK-001 0   # первый пункт
python task.py check TASK-001 2   # третий пункт
```

## Workflow

### 1. Выбор задачи

```bash
# Смотрим что готово к работе
python task.py next

# Вывод:
# 🚀 Следующие задачи (готовы к работе):
# 
# 1. 🔴 TASK-100: Test Infrastructure Setup
#    Est: 2d | Milestone 1: MVP ✓ deps OK
```

### 2. Начало работы

```bash
# Начинаем задачу
python task.py start TASK-100

# ✓ TASK-100 начата!
```

Статус в `tasks.md` обновляется: `⬜ TODO` → `🔄 IN PROGRESS`

### 3. Работа с чеклистом

```bash
# Смотрим чеклист
python task.py show TASK-100

# Отмечаем выполненные пункты
python task.py check TASK-100 0
python task.py check TASK-100 1
```

### 4. Завершение

```bash
# Завершаем
python task.py done TASK-100

# ✅ TASK-100 завершена!
# 
# 🔓 Разблокированы задачи:
#    TASK-001: ATP Protocol Models
#    TASK-004: Test Loader
```

### 5. Проверка прогресса

```bash
python task.py stats

# 📊 Статистика задач
# ==================
# 
# По статусу:
#   ✅ done          3 ████░░░░░░░░░░░░░░░░ 12%
#   🔄 in_progress   1 █░░░░░░░░░░░░░░░░░░░  4%
#   ⬜ todo         21 ████████████████████ 84%
```

## Зависимости

Система автоматически отслеживает зависимости:

- При `task next` — показывает только задачи с выполненными зависимостями
- При `task start` — предупреждает о незавершённых зависимостях
- При `task done` — показывает разблокированные задачи

```bash
# Попытка начать задачу с незавершёнными зависимостями
python task.py start TASK-003

# ⚠️  Задача зависит от незавершённых: TASK-001
#    Используй --force чтобы начать всё равно
```

## Экспорт в GitHub Issues

```bash
# Генерирует команды для gh CLI
python task.py export-gh

# Выполни сгенерированные команды:
# gh issue create --title "TASK-001: ATP Protocol Models" ...
```

## Интеграция с Git

Рекомендуемый workflow с ветками:

```bash
# 1. Начать задачу
python task.py start TASK-001
git checkout -b task/TASK-001-protocol-models

# 2. Работать...
git commit -m "TASK-001: Add ATPRequest model"

# 3. Завершить
python task.py done TASK-001
git checkout main
git merge task/TASK-001-protocol-models
```

## Make targets

Для удобства — targets в Makefile:

| Команда | Описание |
|---------|----------|
| `make task-list` | Список всех задач |
| `make task-todo` | TODO задачи |
| `make task-progress` | Задачи в работе |
| `make task-stats` | Статистика |
| `make task-next` | Следующие задачи |
| `make task-graph` | Граф зависимостей |
| `make task-p0` | Только P0 |
| `make task-mvp` | Задачи MVP |
| `make task-start ID=X` | Начать задачу |
| `make task-done ID=X` | Завершить задачу |
| `make task-show ID=X` | Показать детали |

## История изменений

Все изменения логируются в `spec/.task-history.log`:

```
2025-01-22T10:30:00 | TASK-100 | status -> in_progress
2025-01-22T10:35:00 | TASK-100 | checklist[0] -> done
2025-01-22T11:00:00 | TASK-100 | status -> done
```

## Tips

1. **Начинай день с `task next`** — видишь приоритетные готовые задачи
2. **Отмечай чеклист регулярно** — прогресс виден сразу
3. **Не форси зависимости** — они там не просто так
4. **Коммить tasks.md** — история в Git
5. **Используй `--force` осознанно** — только когда действительно нужно

# Troubleshooting

Руководство по решению проблем при работе с ATP Executor.

## Заблокированные задачи

Когда задача блокируется (после 3 неудачных попыток), executor останавливается:

```
⛔ Stopped: 3 consecutive failures
   Use 'executor.py retry <TASK-ID>' to retry specific task
```

### 1. Посмотреть логи — понять причину

```bash
uv run python executor.py logs TASK-XXX
```

Покажет последний лог с:
- Промптом, отправленным Claude
- Выводом Claude
- Ошибками (stderr)
- Кодом возврата

### 2. Повторить задачу

```bash
uv run python executor.py retry TASK-XXX
```

Сбрасывает счётчик попыток и запускает задачу заново.

### 3. Пропустить задачу

Если задачу нужно выполнить вручную или пропустить:

```bash
# Отметить как выполненную
uv run python task.py done TASK-XXX --force

# Или вернуть в очередь
uv run python task.py start TASK-XXX  # статус -> in_progress
# Затем отредактировать spec/tasks.md вручную
```

### 4. Сбросить состояние executor

```bash
# Только состояние (счётчики, история попыток)
uv run python executor.py reset

# Состояние + логи
uv run python executor.py reset --logs
```

### 5. Проверить статус

```bash
uv run python executor.py status
```

Покажет:
- Количество выполненных/неудачных задач
- Историю попыток
- Последние ошибки

---

## Типичные причины блокировки

### Ошибки тестов/lint

```
❌ Tests failed!
```

**Решение:** Посмотреть логи, исправить код вручную, затем `retry`.

### Claude не смог выполнить задачу

```
❌ TASK-XXX failed: Unknown error
```

**Причины:**
- Задача слишком сложная
- Недостаточно контекста в промпте
- Требуется информация, которой нет в репозитории

**Решение:** Разбить задачу на подзадачи или выполнить вручную.

### Таймаут

```
⏰ TASK-XXX timed out
```

**Решение:** Увеличить таймаут:
```bash
uv run python executor.py run --timeout=60 --task=TASK-XXX
```

### Проблемы с разрешениями

```
❌ TASK-XXX failed: Unable to complete task automatically due to required permissions
```

**Решение:** Убедиться, что executor использует флаг `--dangerously-skip-permissions`. Проверить `executor.py`:
```python
skip_permissions: bool = True
```

### Проблемы с зависимостями

```
ModuleNotFoundError: No module named 'xxx'
```

**Решение:**
```bash
uv add xxx
# или для dev-зависимостей
uv add --dev xxx
```

---

## Проблемы с Git

### Конфликт при merge

```
⚠️  Merge failed: CONFLICT
```

**Решение:**
```bash
# Посмотреть текущую ветку
git status

# Разрешить конфликты вручную
git mergetool
# или отредактировать файлы

# Завершить merge
git add -A && git commit -m "Resolve merge conflict"
```

### Застряли на task-ветке

```bash
# Вернуться на main
git checkout main

# Удалить task-ветку если не нужна
git branch -D task/task-xxx-name
```

---

## Полезные команды

```bash
# Статус задач
uv run python task.py stats

# Следующие готовые задачи
uv run python task.py next

# Граф зависимостей
uv run python task.py graph

# Детали конкретной задачи
uv run python task.py show TASK-XXX

# Запустить одну задачу
make exec-task ID=TASK-XXX

# Запустить все готовые
make exec-all
```

---

## Настройки Executor

Файл `executor.config.yaml`:

```yaml
executor:
  max_retries: 3              # Попыток на задачу
  retry_delay_seconds: 5      # Пауза между попытками
  max_consecutive_failures: 2 # Стоп после N неудач подряд
  task_timeout_minutes: 30    # Таймаут на задачу
```

Можно переопределить через CLI:
```bash
uv run python executor.py run --max-retries=5 --timeout=60 --task=TASK-XXX
```

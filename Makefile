# ATP Platform — Makefile
# Команды для разработки и управления задачами

.PHONY: help install sync dev test lint format docs clean
.PHONY: task-list task-stats task-next task-graph

# === Setup ===

help:  ## Показать справку
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Установить зависимости
	uv sync --all-extras

sync:  ## Синхронизировать зависимости
	uv sync

dev:  ## Настроить dev окружение
	uv sync --all-extras
	uv run pre-commit install
	@echo "✅ Dev environment ready"

# === Testing ===

test:  ## Запустить все тесты
	uv run pytest tests/ -v --cov=atp --cov-report=term-missing

test-unit:  ## Только unit тесты
	uv run pytest tests/unit -v

test-integration:  ## Только integration тесты
	uv run pytest tests/integration -v

test-e2e:  ## Только e2e тесты
	uv run pytest tests/e2e -v

test-fast:  ## Быстрые тесты (без slow)
	uv run pytest tests/ -v -m "not slow"

coverage:  ## Отчёт о покрытии
	uv run pytest tests/ --cov=atp --cov-report=html
	@echo "📊 Coverage report: htmlcov/index.html"

# === Code Quality ===

lint:  ## Проверить код (ruff + pyrefly)
	uv run ruff check .
	pyrefly check

format:  ## Форматировать код
	uv run ruff format .
	uv run ruff check --fix .

check: lint test  ## Полная проверка (lint + test)

# === Task Management ===

task-list:  ## Список всех задач
	@uv run python task.py list

task-todo:  ## Список TODO задач
	@uv run python task.py list --status=todo

task-progress:  ## Задачи в работе
	@uv run python task.py list --status=in_progress

task-stats:  ## Статистика задач
	@uv run python task.py stats

task-next:  ## Следующие задачи для работы
	@uv run python task.py next

task-graph:  ## Граф зависимостей
	@uv run python task.py graph

task-p0:  ## Только P0 задачи
	@uv run python task.py list --priority=p0

task-mvp:  ## Задачи MVP
	@uv run python task.py list --milestone=mvp

# === Task Workflow ===

# Использование: make task-start ID=TASK-001
task-start:  ## Начать задачу (make task-start ID=TASK-001)
	@uv run python task.py start $(ID)

task-done:  ## Завершить задачу (make task-done ID=TASK-001)
	@uv run python task.py done $(ID)

task-show:  ## Показать задачу (make task-show ID=TASK-001)
	@uv run python task.py show $(ID)

# === Documentation ===

docs:  ## Сгенерировать документацию
	@echo "📚 Building docs..."
	# mkdocs build или sphinx

docs-serve:  ## Запустить docs сервер
	# mkdocs serve

# === Build & Release ===

build:  ## Собрать пакет
	uv build

clean:  ## Очистить артефакты
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +

# === Auto Execution (Claude CLI) ===

exec:  ## Выполнить следующую задачу через Claude
	@uv run python executor.py run

exec-task:  ## Выполнить конкретную задачу (make exec-task ID=TASK-001)
	@uv run python executor.py run --task=$(ID)

exec-all:  ## Выполнить все готовые задачи
	@uv run python executor.py run --all

exec-mvp:  ## Выполнить задачи MVP
	@uv run python executor.py run --all --milestone=mvp

exec-status:  ## Статус выполнения
	@uv run python executor.py status

exec-retry:  ## Повторить задачу (make exec-retry ID=TASK-001)
	@uv run python executor.py retry $(ID)

exec-logs:  ## Логи задачи (make exec-logs ID=TASK-001)
	@uv run python executor.py logs $(ID)

exec-reset:  ## Сбросить состояние executor
	@uv run python executor.py reset

# === CI/CD ===

ci: lint test  ## CI pipeline (для GitHub Actions)
	@echo "✅ CI passed"

# === Default ===

.DEFAULT_GOAL := help

# ATP Platform — Makefile
# Commands for development and task management

.PHONY: help install sync dev test lint format docs clean
.PHONY: task-list task-stats task-next task-graph

# === Setup ===

help:  ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync --all-extras

sync:  ## Sync dependencies
	uv sync

dev:  ## Setup dev environment
	uv sync --all-extras
	uv run pre-commit install
	@echo "✅ Dev environment ready"

# === Testing ===

test:  ## Run all tests
	uv run pytest tests/ -v --cov=atp --cov-report=term-missing

test-unit:  ## Unit tests only
	uv run pytest tests/unit -v

test-integration:  ## Integration tests only
	uv run pytest tests/integration -v

test-e2e:  ## E2E tests only
	uv run pytest tests/e2e -v

test-fast:  ## Fast tests (without slow)
	uv run pytest tests/ -v -m "not slow"

coverage:  ## Coverage report
	uv run pytest tests/ --cov=atp --cov-report=html
	@echo "📊 Coverage report: htmlcov/index.html"

# === Code Quality ===

lint:  ## Check code (ruff + pyrefly)
	uv run ruff check .
	pyrefly check

format:  ## Format code
	uv run ruff format .
	uv run ruff check --fix .

check: lint test  ## Full check (lint + test)

# === Task Management ===

task-list:  ## List all tasks
	@uv run python task.py list

task-todo:  ## List TODO tasks
	@uv run python task.py list --status=todo

task-progress:  ## Tasks in progress
	@uv run python task.py list --status=in_progress

task-stats:  ## Task statistics
	@uv run python task.py stats

task-next:  ## Next tasks to work on
	@uv run python task.py next

task-graph:  ## Dependency graph
	@uv run python task.py graph

task-p0:  ## P0 tasks only
	@uv run python task.py list --priority=p0

task-mvp:  ## MVP tasks
	@uv run python task.py list --milestone=mvp

# === Task Workflow ===

# Usage: make task-start ID=TASK-001
task-start:  ## Start task (make task-start ID=TASK-001)
	@uv run python task.py start $(ID)

task-done:  ## Complete task (make task-done ID=TASK-001)
	@uv run python task.py done $(ID)

task-show:  ## Show task (make task-show ID=TASK-001)
	@uv run python task.py show $(ID)

# === Documentation ===

docs:  ## Build documentation
	@echo "📚 Building docs..."
	# mkdocs build or sphinx

docs-serve:  ## Start docs server
	# mkdocs serve

# === Build & Release ===

build:  ## Build package
	uv build

clean:  ## Clean artifacts
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +

# === Auto Execution (Claude CLI) ===

exec:  ## Execute next task via Claude
	@uv run python executor.py run

exec-task:  ## Execute specific task (make exec-task ID=TASK-001)
	@uv run python executor.py run --task=$(ID)

exec-all:  ## Execute all ready tasks
	@uv run python executor.py run --all

exec-mvp:  ## Execute MVP tasks
	@uv run python executor.py run --all --milestone=mvp

exec-status:  ## Execution status
	@uv run python executor.py status

exec-retry:  ## Retry task (make exec-retry ID=TASK-001)
	@uv run python executor.py retry $(ID)

exec-logs:  ## Task logs (make exec-logs ID=TASK-001)
	@uv run python executor.py logs $(ID)

exec-reset:  ## Reset executor state
	@uv run python executor.py reset

# === CI/CD ===

ci: lint test  ## CI pipeline (for GitHub Actions)
	@echo "✅ CI passed"

# === Default ===

.DEFAULT_GOAL := help

#!/usr/bin/env python3
"""
ATP Task Executor — automatic task execution via Claude CLI

Usage:
    python executor.py run                    # Execute the next task
    python executor.py run --task=TASK-001    # Execute a specific task
    python executor.py run --all              # Execute all ready tasks
    python executor.py run --milestone=mvp    # Execute milestone tasks
    python executor.py status                 # Execution status
    python executor.py retry TASK-001         # Retry a failed task
    python executor.py logs TASK-001          # Task logs
"""

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Import task parser (template: task.py lives alongside this file)
from task import (  # pyrefly: ignore[missing-import]
    Task,
    get_next_tasks,
    get_task_by_id,
    parse_tasks,
    update_task_status,
)

# === Configuration ===


@dataclass
class ExecutorConfig:
    """Executor configuration"""

    max_retries: int = 3  # Maximum attempts per task
    retry_delay_seconds: int = 5  # Delay between attempts
    task_timeout_minutes: int = 30  # Timeout per task
    max_consecutive_failures: int = 2  # Stop after N consecutive failures

    # Spec file prefix (e.g. "phase5-" for phase5-tasks.md)
    spec_prefix: str = ""

    # Claude CLI
    claude_command: str = "claude"  # Claude CLI command
    claude_model: str = ""  # Model (empty = default)

    # Hooks
    run_tests_on_done: bool = True  # Run tests on completion
    create_git_branch: bool = True  # Create branch on start
    auto_commit: bool = False  # Auto-commit on success

    # Paths
    project_root: Path = Path(".")
    logs_dir: Path = Path("spec/.executor-logs")
    state_file: Path = Path("spec/.executor-state.json")

    # Test command
    test_command: str = "make test-fast"
    lint_command: str = "make lint"

    def __post_init__(self):
        """Resolve project_root and namespace state/log paths by spec_prefix."""
        self.project_root = self.project_root.resolve()

        if self.spec_prefix:
            default_state = Path("spec/.executor-state.json")
            default_logs = Path("spec/.executor-logs")
            if self.state_file == default_state:
                self.state_file = Path(f"spec/.executor-{self.spec_prefix}state.json")
            if self.logs_dir == default_logs:
                self.logs_dir = Path(f"spec/.executor-{self.spec_prefix}logs")

        if not self.state_file.is_absolute():
            self.state_file = self.project_root / self.state_file
        if not self.logs_dir.is_absolute():
            self.logs_dir = self.project_root / self.logs_dir

    @property
    def tasks_file(self) -> Path:
        return self.project_root / "spec" / f"{self.spec_prefix}tasks.md"

    @property
    def requirements_file(self) -> Path:
        return self.project_root / "spec" / f"{self.spec_prefix}requirements.md"

    @property
    def design_file(self) -> Path:
        return self.project_root / "spec" / f"{self.spec_prefix}design.md"


# === State Management ===


@dataclass
class TaskAttempt:
    """Task execution attempt"""

    timestamp: str
    success: bool
    duration_seconds: float
    error: str | None = None
    claude_output: str | None = None


@dataclass
class TaskState:
    """Task state in the executor"""

    task_id: str
    status: str  # pending, running, success, failed, skipped
    attempts: list[TaskAttempt] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def last_error(self) -> str | None:
        if self.attempts:
            return self.attempts[-1].error
        return None


class ExecutorState:
    """Global executor state"""

    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.tasks: dict[str, TaskState] = {}
        self.consecutive_failures = 0
        self.total_completed = 0
        self.total_failed = 0
        self._load()

    def _load(self):
        """Load state from file"""
        if self.config.state_file.exists():
            data = json.loads(self.config.state_file.read_text())
            for task_id, task_data in data.get("tasks", {}).items():
                attempts = [TaskAttempt(**a) for a in task_data.get("attempts", [])]
                self.tasks[task_id] = TaskState(
                    task_id=task_id,
                    status=task_data.get("status", "pending"),
                    attempts=attempts,
                    started_at=task_data.get("started_at"),
                    completed_at=task_data.get("completed_at"),
                )
            self.consecutive_failures = data.get("consecutive_failures", 0)
            self.total_completed = data.get("total_completed", 0)
            self.total_failed = data.get("total_failed", 0)

    def _save(self):
        """Save state to file"""
        self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tasks": {
                task_id: {
                    "status": ts.status,
                    "attempts": [
                        {
                            "timestamp": a.timestamp,
                            "success": a.success,
                            "duration_seconds": a.duration_seconds,
                            "error": a.error,
                        }
                        for a in ts.attempts
                    ],
                    "started_at": ts.started_at,
                    "completed_at": ts.completed_at,
                }
                for task_id, ts in self.tasks.items()
            },
            "consecutive_failures": self.consecutive_failures,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "last_updated": datetime.now().isoformat(),
        }
        self.config.state_file.write_text(json.dumps(data, indent=2))

    def get_task_state(self, task_id: str) -> TaskState:
        if task_id not in self.tasks:
            self.tasks[task_id] = TaskState(task_id=task_id, status="pending")
        return self.tasks[task_id]

    def record_attempt(
        self,
        task_id: str,
        success: bool,
        duration: float,
        error: str | None = None,
        output: str | None = None,
    ):
        """Record an execution attempt"""
        state = self.get_task_state(task_id)
        state.attempts.append(
            TaskAttempt(
                timestamp=datetime.now().isoformat(),
                success=success,
                duration_seconds=duration,
                error=error,
                claude_output=output,
            )
        )

        if success:
            state.status = "success"
            state.completed_at = datetime.now().isoformat()
            self.consecutive_failures = 0
            self.total_completed += 1
        else:
            if state.attempt_count >= self.config.max_retries:
                state.status = "failed"
                self.total_failed += 1
            self.consecutive_failures += 1

        self._save()

    def mark_running(self, task_id: str):
        state = self.get_task_state(task_id)
        state.status = "running"
        state.started_at = datetime.now().isoformat()
        self._save()

    def should_stop(self) -> bool:
        """Check whether execution should stop"""
        return self.consecutive_failures >= self.config.max_consecutive_failures


# === Prompt Builder ===


def build_task_prompt(task: Task, config: ExecutorConfig) -> str:
    """Build a prompt for Claude with the task context"""

    # Read specifications
    requirements = ""
    if config.requirements_file.exists():
        requirements = config.requirements_file.read_text()

    design = ""
    if config.design_file.exists():
        design = config.design_file.read_text()

    # Find related requirements
    related_reqs = []
    for ref in task.traces_to:
        if ref.startswith("REQ-"):
            pattern = rf"#### {ref}:.*?(?=####|\Z)"
            match = re.search(pattern, requirements, re.DOTALL)
            if match:
                related_reqs.append(match.group(0).strip())

    # Find related design
    related_design = []
    for ref in task.traces_to:
        if ref.startswith("DESIGN-"):
            pattern = rf"### {ref}:.*?(?=###|\Z)"
            match = re.search(pattern, design, re.DOTALL)
            if match:
                related_design.append(match.group(0).strip())

    # Checklist
    checklist_text = "\n".join(
        [f"- {'[x]' if done else '[ ]'} {item}" for item, done in task.checklist]
    )

    reqs_text = (
        chr(10).join(related_reqs)
        if related_reqs
        else f"See {config.requirements_file}"
    )
    design_text = (
        chr(10).join(related_design) if related_design else f"See {config.design_file}"
    )

    prompt = f"""# Task Execution Request

## Task: {task.id} — {task.name}

**Priority:** {task.priority.upper()}
**Estimate:** {task.estimate}
**Milestone:** {task.milestone}

## Checklist (implement ALL items):

{checklist_text}

## Related Requirements:

{reqs_text}

## Related Design:

{design_text}

## Instructions:

1. Implement ALL checklist items for this task
2. Write unit tests for new code (coverage ≥80%)
3. Follow the design patterns from {config.design_file}
4. Use existing code style and conventions
5. Create/update files as needed

## Success Criteria:

- All checklist items implemented
- All tests pass (`make test`)
- No lint errors (`make lint`)
- Code follows project conventions

## Output:

When complete, respond with:
- Summary of changes made
- Files created/modified
- Any issues or notes
- "TASK_COMPLETE" if successful, or "TASK_FAILED: <reason>" if not

Begin implementation:
"""

    return prompt


# === Hooks ===


def pre_start_hook(task: Task, config: ExecutorConfig) -> bool:
    """Hook before starting a task"""
    print(f"🔧 Pre-start hook for {task.id}")

    # Create git branch
    if config.create_git_branch:
        branch_name = (
            f"task/{task.id.lower()}-{task.name.lower().replace(' ', '-')[:30]}"
        )
        try:
            # Check if git is available
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                cwd=config.project_root,
            )
            if result.returncode == 0:
                # Create branch
                subprocess.run(
                    ["git", "checkout", "-b", branch_name],
                    capture_output=True,
                    cwd=config.project_root,
                )
                print(f"   Created branch: {branch_name}")
        except FileNotFoundError:
            pass  # git is not installed

    return True


def post_done_hook(task: Task, config: ExecutorConfig, success: bool) -> bool:
    """Hook after completing a task"""
    print(f"🔧 Post-done hook for {task.id} (success={success})")

    if not success:
        return False

    # Run tests
    if config.run_tests_on_done:
        print("   Running tests...")
        result = subprocess.run(
            config.test_command,
            shell=True,
            capture_output=True,
            cwd=config.project_root,
        )
        if result.returncode != 0:
            print("   ❌ Tests failed!")
            print(result.stderr.decode()[:500])
            return False
        print("   ✅ Tests passed")

    # Run lint
    if config.lint_command:
        print("   Running lint...")
        result = subprocess.run(
            config.lint_command,
            shell=True,
            capture_output=True,
            cwd=config.project_root,
        )
        if result.returncode != 0:
            print("   ⚠️  Lint warnings (non-blocking)")

    # Auto-commit
    if config.auto_commit:
        try:
            subprocess.run(["git", "add", "-A"], cwd=config.project_root)
            subprocess.run(
                ["git", "commit", "-m", f"{task.id}: {task.name}"],
                cwd=config.project_root,
            )
            print("   Committed changes")
        except Exception as e:
            print(f"   Commit failed: {e}")

    return True


# === Task Executor ===


def execute_task(task: Task, config: ExecutorConfig, state: ExecutorState) -> bool:
    """Execute a single task via Claude CLI"""

    task_id = task.id
    print(f"\n{'=' * 60}")
    print(f"🚀 Executing {task_id}: {task.name}")
    print(f"{'=' * 60}")

    # Pre-start hook
    if not pre_start_hook(task, config):
        print("❌ Pre-start hook failed")
        return False

    # Update status
    state.mark_running(task_id)
    update_task_status(config.tasks_file, task_id, "in_progress")

    # Build prompt
    prompt = build_task_prompt(task, config)

    # Save prompt to log
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = config.logs_dir / (
        f"{task_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    )

    with open(log_file, "w") as f:
        f.write(f"=== PROMPT ===\n{prompt}\n\n")

    # Run Claude
    start_time = datetime.now()

    try:
        cmd = [config.claude_command, "-p", prompt]
        if config.claude_model:
            cmd.extend(["--model", config.claude_model])

        print(f"🤖 Running: {' '.join(cmd[:3])}...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.task_timeout_minutes * 60,
            cwd=config.project_root,
        )

        duration = (datetime.now() - start_time).total_seconds()
        output = result.stdout

        # Save output
        with open(log_file, "a") as f:
            f.write(f"=== OUTPUT ===\n{output}\n\n")
            f.write(f"=== STDERR ===\n{result.stderr}\n\n")
            f.write(f"=== RETURN CODE: {result.returncode} ===\n")

        # Check result
        success = "TASK_COMPLETE" in output and "TASK_FAILED" not in output

        if success:
            print("✅ Claude reports: TASK_COMPLETE")

            # Post-done hook (tests, lint)
            hook_success = post_done_hook(task, config, True)

            if hook_success:
                state.record_attempt(task_id, True, duration, output=output)
                update_task_status(config.tasks_file, task_id, "done")
                print(f"✅ {task_id} completed successfully in {duration:.1f}s")
                return True
            else:
                # Hook failed (tests did not pass)
                error = "Post-done hook failed (tests/lint)"
                state.record_attempt(
                    task_id, False, duration, error=error, output=output
                )
                print(f"❌ {task_id} failed: {error}")
                return False
        else:
            # Claude reported failure
            error_match = re.search(r"TASK_FAILED:\s*(.+)", output)
            error = error_match.group(1) if error_match else "Unknown error"
            state.record_attempt(task_id, False, duration, error=error, output=output)
            print(f"❌ {task_id} failed: {error}")
            return False

    except subprocess.TimeoutExpired:
        duration = config.task_timeout_minutes * 60
        error = f"Timeout after {config.task_timeout_minutes} minutes"
        state.record_attempt(task_id, False, duration, error=error)
        print(f"⏰ {task_id} timed out")
        return False

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error = str(e)
        state.record_attempt(task_id, False, duration, error=error)
        print(f"💥 {task_id} error: {error}")
        return False


def run_with_retries(task: Task, config: ExecutorConfig, state: ExecutorState) -> bool:
    """Execute a task with retries"""

    task_state = state.get_task_state(task.id)

    for attempt in range(task_state.attempt_count, config.max_retries):
        print(f"\n📍 Attempt {attempt + 1}/{config.max_retries} for {task.id}")

        if execute_task(task, config, state):
            return True

        if attempt < config.max_retries - 1:
            print(f"⏳ Waiting {config.retry_delay_seconds}s before retry...")
            import time

            time.sleep(config.retry_delay_seconds)

    print(f"❌ {task.id} failed after {config.max_retries} attempts")
    update_task_status(config.tasks_file, task.id, "blocked")
    return False


# === CLI Commands ===


def cmd_run(args, config: ExecutorConfig):
    """Execute tasks"""

    tasks = parse_tasks(config.tasks_file)
    state = ExecutorState(config)

    # Check failure limit
    if state.should_stop():
        print(f"⛔ Stopped: {state.consecutive_failures} consecutive failures")
        print("   Use 'executor.py retry <TASK-ID>' to retry")
        return

    # Determine which tasks to execute
    if args.task:
        # Specific task
        task = get_task_by_id(tasks, args.task.upper())
        if not task:
            print(f"❌ Task {args.task} not found")
            return
        tasks_to_run = [task]

    elif args.all:
        # All ready tasks
        tasks_to_run = get_next_tasks(tasks)
        if args.milestone:
            tasks_to_run = [
                t for t in tasks_to_run if args.milestone.lower() in t.milestone.lower()
            ]

    elif args.milestone:
        # Tasks for a specific milestone
        next_tasks = get_next_tasks(tasks)
        tasks_to_run = [
            t for t in next_tasks if args.milestone.lower() in t.milestone.lower()
        ]

    else:
        # Next task
        next_tasks = get_next_tasks(tasks)
        tasks_to_run = next_tasks[:1] if next_tasks else []

    if not tasks_to_run:
        print("✅ No tasks ready to execute")
        print("   All dependencies might be incomplete, or all tasks done")
        return

    print(f"📋 Tasks to execute: {len(tasks_to_run)}")
    for t in tasks_to_run:
        print(f"   - {t.id}: {t.name}")

    # Execute
    for task in tasks_to_run:
        success = run_with_retries(task, config, state)

        if not success and state.should_stop():
            print("\n⛔ Stopping: too many consecutive failures")
            break

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 Execution Summary")
    print(f"{'=' * 60}")
    print(f"   Completed: {state.total_completed}")
    print(f"   Failed:    {state.total_failed}")
    print(f"   Remaining: {len([t for t in tasks if t.status == 'todo'])}")


def cmd_status(args, config: ExecutorConfig):
    """Execution status"""

    state = ExecutorState(config)

    print("\n📊 Executor Status")
    print(f"{'=' * 50}")
    print(f"Total completed:       {state.total_completed}")
    print(f"Total failed:          {state.total_failed}")
    print(
        f"Consecutive failures:  "
        f"{state.consecutive_failures}/"
        f"{config.max_consecutive_failures}"
    )

    # Tasks with attempts
    attempted = [ts for ts in state.tasks.values() if ts.attempts]
    if attempted:
        print("\n📝 Task History:")
        for ts in attempted:
            if ts.status == "success":
                icon = "✅"
            elif ts.status == "failed":
                icon = "❌"
            else:
                icon = "🔄"
            print(f"   {icon} {ts.task_id}: {ts.status} ({ts.attempt_count} attempts)")
            if ts.last_error:
                print(f"      Last error: {ts.last_error[:50]}...")


def cmd_retry(args, config: ExecutorConfig):
    """Retry a failed task"""

    tasks = parse_tasks(config.tasks_file)
    state = ExecutorState(config)

    task = get_task_by_id(tasks, args.task_id.upper())
    if not task:
        print(f"❌ Task {args.task_id} not found")
        return

    # Reset state
    task_state = state.get_task_state(task.id)
    task_state.attempts = []
    task_state.status = "pending"
    state.consecutive_failures = 0
    state._save()

    print(f"🔄 Retrying {task.id}...")
    run_with_retries(task, config, state)


def cmd_logs(args, config: ExecutorConfig):
    """Show task logs"""

    task_id = args.task_id.upper()
    log_files = sorted(config.logs_dir.glob(f"{task_id}-*.log"))

    if not log_files:
        print(f"No logs found for {task_id}")
        return

    latest = log_files[-1]
    print(f"📄 Latest log: {latest}")
    print("=" * 50)
    print(latest.read_text()[:5000])  # Limit output


def cmd_reset(args, config: ExecutorConfig):
    """Reset executor state"""

    if config.state_file.exists():
        config.state_file.unlink()
        print("✅ State reset")

    if args.logs and config.logs_dir.exists():
        shutil.rmtree(config.logs_dir)
        print("✅ Logs cleared")


# === Main ===


def main():
    parser = argparse.ArgumentParser(
        description=("ATP Task Executor — automatic task execution via Claude"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per task (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Task timeout in minutes (default: 30)",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip tests on task completion",
    )
    parser.add_argument(
        "--no-branch",
        action="store_true",
        help="Skip git branch creation",
    )
    parser.add_argument(
        "--auto-commit",
        action="store_true",
        help="Auto-commit on success",
    )
    parser.add_argument(
        "--spec-prefix",
        type=str,
        default="",
        help='Spec file prefix (e.g. "phase5-" for phase5-tasks.md)',
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default="",
        help="Project root directory (default: current directory)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # run
    run_parser = subparsers.add_parser("run", help="Execute tasks")
    run_parser.add_argument("--task", "-t", help="Specific task ID")
    run_parser.add_argument(
        "--all", "-a", action="store_true", help="Run all ready tasks"
    )
    run_parser.add_argument("--milestone", "-m", help="Filter by milestone")

    # status
    subparsers.add_parser("status", help="Show execution status")

    # retry
    retry_parser = subparsers.add_parser("retry", help="Retry failed task")
    retry_parser.add_argument("task_id", help="Task ID to retry")

    # logs
    logs_parser = subparsers.add_parser("logs", help="Show task logs")
    logs_parser.add_argument("task_id", help="Task ID")

    # reset
    reset_parser = subparsers.add_parser("reset", help="Reset executor state")
    reset_parser.add_argument("--logs", action="store_true", help="Also clear logs")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Build config
    config_kwargs: dict = {
        "max_retries": args.max_retries,
        "task_timeout_minutes": args.timeout,
        "run_tests_on_done": not args.no_tests,
        "create_git_branch": not args.no_branch,
        "auto_commit": args.auto_commit,
    }
    if args.spec_prefix:
        config_kwargs["spec_prefix"] = args.spec_prefix
    if args.project_root:
        config_kwargs["project_root"] = Path(args.project_root)

    config = ExecutorConfig(**config_kwargs)

    # Dispatch
    commands = {
        "run": cmd_run,
        "status": cmd_status,
        "retry": cmd_retry,
        "logs": cmd_logs,
        "reset": cmd_reset,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args, config)


if __name__ == "__main__":
    main()

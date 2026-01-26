#!/usr/bin/env python3
"""
ATP Task Executor — automatic task execution via Claude CLI

Usage:
    python executor.py run                    # Execute next task
    python executor.py run --task=TASK-001    # Execute specific task
    python executor.py run --all              # Execute all ready tasks
    python executor.py run --milestone=mvp    # Execute milestone tasks
    python executor.py status                 # Execution status
    python executor.py retry TASK-001         # Retry failed task
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

import yaml

from task import (
    TASKS_FILE,
    Task,
    get_next_tasks,
    get_task_by_id,
    parse_tasks,
    update_task_status,
)

# Configuration file path
CONFIG_FILE = Path("executor.config.yaml")


@dataclass
class ExecutorConfig:
    """Executor configuration"""

    max_retries: int = 3  # Max attempts per task
    retry_delay_seconds: int = 5  # Pause between attempts
    task_timeout_minutes: int = 30  # Task timeout
    max_consecutive_failures: int = 2  # Stop after N consecutive failures

    # Claude CLI
    claude_command: str = "claude"  # Claude CLI command
    claude_model: str = ""  # Model (empty = default)
    skip_permissions: bool = True  # Skip permission prompts

    # Hooks
    run_tests_on_done: bool = True  # Run tests on completion
    create_git_branch: bool = True  # Create branch on start
    auto_commit: bool = True  # Auto-commit on success

    # Paths
    project_root: Path = Path(".")
    logs_dir: Path = Path("spec/.executor-logs")
    state_file: Path = Path("spec/.executor-state.json")

    # Test command (using uv)
    test_command: str = "uv run pytest tests/ -v -m 'not slow'"
    lint_command: str = "uv run ruff check ."
    run_lint_on_done: bool = True  # Run lint on completion


def load_config_from_yaml(config_path: Path = CONFIG_FILE) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary with configuration values.
    """
    if not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        executor_config = data.get("executor", {})
        hooks = executor_config.get("hooks", {})
        pre_start = hooks.get("pre_start", {})
        post_done = hooks.get("post_done", {})
        commands = executor_config.get("commands", {})
        paths = executor_config.get("paths", {})

        return {
            "max_retries": executor_config.get("max_retries"),
            "retry_delay_seconds": executor_config.get("retry_delay_seconds"),
            "task_timeout_minutes": executor_config.get("task_timeout_minutes"),
            "max_consecutive_failures": executor_config.get("max_consecutive_failures"),
            "claude_command": executor_config.get("claude_command"),
            "claude_model": executor_config.get("claude_model"),
            "skip_permissions": executor_config.get("skip_permissions"),
            "create_git_branch": pre_start.get("create_git_branch"),
            "run_tests_on_done": post_done.get("run_tests"),
            "run_lint_on_done": post_done.get("run_lint"),
            "auto_commit": post_done.get("auto_commit"),
            "test_command": commands.get("test"),
            "lint_command": commands.get("lint"),
            "logs_dir": Path(paths["logs"]) if paths.get("logs") else None,
            "state_file": Path(paths["state"]) if paths.get("state") else None,
        }
    except Exception as e:
        print(f"⚠️  Warning: Failed to load config from {config_path}: {e}")
        return {}


def build_config(yaml_config: dict, args: argparse.Namespace) -> ExecutorConfig:
    """Build ExecutorConfig from YAML and CLI arguments.

    CLI arguments override YAML config.

    Args:
        yaml_config: Configuration loaded from YAML file.
        args: Parsed CLI arguments.

    Returns:
        ExecutorConfig instance.
    """
    # Start with defaults
    config_kwargs = {}

    # Apply YAML config (only non-None values)
    for key, value in yaml_config.items():
        if value is not None:
            config_kwargs[key] = value

    # Override with CLI arguments
    if hasattr(args, "max_retries") and args.max_retries != 3:
        config_kwargs["max_retries"] = args.max_retries
    if hasattr(args, "timeout") and args.timeout != 30:
        config_kwargs["task_timeout_minutes"] = args.timeout
    if hasattr(args, "no_tests") and args.no_tests:
        config_kwargs["run_tests_on_done"] = False
    if hasattr(args, "no_branch") and args.no_branch:
        config_kwargs["create_git_branch"] = False
    if hasattr(args, "no_commit") and args.no_commit:
        config_kwargs["auto_commit"] = False

    return ExecutorConfig(**config_kwargs)


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
    """Task state in executor"""

    task_id: str
    status: str  # pending, running, success, failed, skipped
    attempts: list = field(default_factory=list)
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
        """Record execution attempt"""
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
        """Check if we should stop"""
        return self.consecutive_failures >= self.config.max_consecutive_failures


# === Prompt Builder ===


def build_task_prompt(task: Task, config: ExecutorConfig) -> str:
    """Build prompt for Claude with task context"""

    # Read specifications
    spec_dir = config.project_root / "spec"

    requirements = ""
    if (spec_dir / "requirements.md").exists():
        requirements = (spec_dir / "requirements.md").read_text()

    design = ""
    if (spec_dir / "design.md").exists():
        design = (spec_dir / "design.md").read_text()

    # Find related requirements
    related_reqs = []
    for ref in task.traces_to:
        if ref.startswith("REQ-"):
            # Extract requirement from requirements.md
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

    prompt = f"""# Task Execution Request

## Task: {task.id} — {task.name}

**Priority:** {task.priority.upper()}
**Estimate:** {task.estimate}
**Milestone:** {task.milestone}

## Checklist (implement ALL items):

{checklist_text}

## Related Requirements:

{chr(10).join(related_reqs) if related_reqs else "See spec/requirements.md"}

## Related Design:

{chr(10).join(related_design) if related_design else "See spec/design.md"}

## Instructions:

1. Implement ALL checklist items for this task
2. Write unit tests for new code (coverage ≥80%)
3. Follow the design patterns from spec/design.md
4. Use existing code style and conventions
5. Create/update files as needed

## Dependencies:

- To add a new dependency: `uv add <package>`
- To add a dev dependency: `uv add --dev <package>`
- NEVER edit pyproject.toml manually for dependencies
- After adding dependencies, they are available immediately

## Success Criteria:

- All checklist items implemented
- All tests pass (`uv run pytest`)
- No lint errors (`uv run ruff check .`)
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


def get_task_branch_name(task: Task) -> str:
    """Generate branch name for task"""
    safe_name = task.name.lower().replace(" ", "-").replace("/", "-")[:30]
    return f"task/{task.id.lower()}-{safe_name}"


def get_main_branch(config: ExecutorConfig) -> str:
    """Determine main branch name (main or master)"""
    result = subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
        capture_output=True,
        text=True,
        cwd=config.project_root,
    )
    if result.returncode == 0:
        # refs/remotes/origin/main -> main
        return result.stdout.strip().split("/")[-1]

    # Fallback: check if main or master exists
    for branch in ["main", "master"]:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", branch],
            capture_output=True,
            cwd=config.project_root,
        )
        if result.returncode == 0:
            return branch

    return "main"  # default


def pre_start_hook(task: Task, config: ExecutorConfig) -> bool:
    """Hook before starting task"""
    print(f"🔧 Pre-start hook for {task.id}")

    # Sync dependencies
    print("   Syncing dependencies...")
    result = subprocess.run(
        ["uv", "sync"], capture_output=True, text=True, cwd=config.project_root
    )
    if result.returncode == 0:
        print("   ✅ Dependencies synced")
    else:
        print(f"   ⚠️  uv sync warning: {result.stderr[:200]}")

    # Create git branch
    if config.create_git_branch:
        branch_name = get_task_branch_name(task)
        try:
            # Check if git exists
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                cwd=config.project_root,
            )
            if result.returncode != 0:
                return True  # No git repository

            # Switch to main
            main_branch = get_main_branch(config)
            subprocess.run(
                ["git", "checkout", main_branch],
                capture_output=True,
                cwd=config.project_root,
            )

            # Check if branch exists
            result = subprocess.run(
                ["git", "rev-parse", "--verify", branch_name],
                capture_output=True,
                cwd=config.project_root,
            )

            if result.returncode == 0:
                # Branch exists — switch to it
                subprocess.run(
                    ["git", "checkout", branch_name],
                    capture_output=True,
                    cwd=config.project_root,
                )
                print(f"   Switched to existing branch: {branch_name}")
            else:
                # Create new branch
                result = subprocess.run(
                    ["git", "checkout", "-b", branch_name],
                    capture_output=True,
                    cwd=config.project_root,
                )
                if result.returncode == 0:
                    print(f"   Created branch: {branch_name}")
                else:
                    print(f"   ⚠️  Failed to create branch: {result.stderr.decode()}")

        except FileNotFoundError:
            pass  # git not installed

    return True


def post_done_hook(task: Task, config: ExecutorConfig, success: bool) -> bool:
    """Hook after task completion"""
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
    if config.run_lint_on_done and config.lint_command:
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
            # Check if there are changes to commit
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=config.project_root,
            )
            if not status_result.stdout.strip():
                print("   No changes to commit")
            else:
                subprocess.run(["git", "add", "-A"], cwd=config.project_root)
                # Build commit message with task details
                commit_title = f"{task.id}: {task.name}"
                commit_body_lines = []
                if task.checklist:
                    commit_body_lines.append("Completed:")
                    for item, checked in task.checklist:
                        if checked:
                            commit_body_lines.append(f"  - {item}")
                if task.milestone:
                    commit_body_lines.append(f"\nMilestone: {task.milestone}")

                commit_msg = commit_title
                if commit_body_lines:
                    commit_msg += "\n\n" + "\n".join(commit_body_lines)

                subprocess.run(
                    ["git", "commit", "-m", commit_msg], cwd=config.project_root
                )
                print("   Committed changes")
        except Exception as e:
            print(f"   Commit failed: {e}")

    # Merge branch to main
    if config.create_git_branch:
        try:
            branch_name = get_task_branch_name(task)
            main_branch = get_main_branch(config)

            # Switch to main
            result = subprocess.run(
                ["git", "checkout", main_branch],
                capture_output=True,
                text=True,
                cwd=config.project_root,
            )
            if result.returncode != 0:
                print(f"   ⚠️  Failed to switch to {main_branch}")
                return True

            # Merge task branch
            result = subprocess.run(
                ["git", "merge", branch_name, "--no-ff", "-m", f"Merge {branch_name}"],
                capture_output=True,
                text=True,
                cwd=config.project_root,
            )
            if result.returncode == 0:
                print(f"   Merged {branch_name} → {main_branch}")

                # Delete task branch
                subprocess.run(
                    ["git", "branch", "-d", branch_name],
                    capture_output=True,
                    cwd=config.project_root,
                )
                print(f"   Deleted branch: {branch_name}")
            else:
                print(f"   ⚠️  Merge failed: {result.stderr}")
                # Return to task branch on failure
                subprocess.run(
                    ["git", "checkout", branch_name],
                    capture_output=True,
                    cwd=config.project_root,
                )
        except Exception as e:
            print(f"   Merge failed: {e}")

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
    update_task_status(TASKS_FILE, task_id, "in_progress")

    # Build prompt
    prompt = build_task_prompt(task, config)

    # Save prompt to log
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = (
        config.logs_dir / f"{task_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    )

    with open(log_file, "w") as f:
        f.write(f"=== PROMPT ===\n{prompt}\n\n")

    # Run Claude
    start_time = datetime.now()

    try:
        cmd = [config.claude_command, "-p", prompt]
        if config.skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        if config.claude_model:
            cmd.extend(["--model", config.claude_model])

        flags = " --dangerously-skip-permissions" if config.skip_permissions else ""
        print(f"🤖 Running: {config.claude_command} -p ...{flags}")

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
        # Success if:
        # 1. Explicitly says TASK_COMPLETE, or
        # 2. Return code 0 and no TASK_FAILED (Claude forgot the marker)
        has_complete_marker = "TASK_COMPLETE" in output
        has_failed_marker = "TASK_FAILED" in output
        implicit_success = result.returncode == 0 and not has_failed_marker

        success = (has_complete_marker and not has_failed_marker) or implicit_success

        if success:
            if has_complete_marker:
                print("✅ Claude reports: TASK_COMPLETE")
            else:
                print("✅ Implicit success (return code 0, no TASK_FAILED)")

            # Post-done hook (tests, lint)
            hook_success = post_done_hook(task, config, True)

            if hook_success:
                state.record_attempt(task_id, True, duration, output=output)
                update_task_status(TASKS_FILE, task_id, "done")
                print(f"✅ {task_id} completed successfully in {duration:.1f}s")
                return True
            else:
                # Hook failed (tests didn't pass)
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
    """Execute task with retries"""

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
    update_task_status(TASKS_FILE, task.id, "blocked")
    return False


# === CLI Commands ===


def cmd_run(args, config: ExecutorConfig):
    """Execute tasks"""

    tasks = parse_tasks(TASKS_FILE)
    state = ExecutorState(config)

    # Check failure limit
    if state.should_stop():
        print(f"⛔ Stopped: {state.consecutive_failures} consecutive failures")
        print("   Use 'executor.py retry <TASK-ID>' to retry specific task")
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
        # Tasks for specific milestone
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
    # Re-read tasks to get updated statuses after execution
    tasks = parse_tasks(TASKS_FILE)

    # Calculate statistics
    failed_attempts = sum(
        1 for ts in state.tasks.values() for a in ts.attempts if not a.success
    )
    remaining = len([t for t in tasks if t.status == "todo"])

    print(f"\n{'=' * 60}")
    print("📊 Execution Summary")
    print(f"{'=' * 60}")
    print(f"   Tasks completed:    {state.total_completed}")
    print(f"   Tasks failed:       {state.total_failed}")
    print(f"   Tasks remaining:    {remaining}")
    if failed_attempts > 0:
        print(f"   Failed attempts:    {failed_attempts} (retried successfully)")


def cmd_status(args, config: ExecutorConfig):
    """Execution status"""

    state = ExecutorState(config)

    # Calculate statistics from actual task state
    completed_tasks = sum(1 for ts in state.tasks.values() if ts.status == "success")
    failed_tasks = sum(1 for ts in state.tasks.values() if ts.status == "failed")
    running_tasks = [ts for ts in state.tasks.values() if ts.status == "running"]
    failed_attempts = sum(
        1 for ts in state.tasks.values() for a in ts.attempts if not a.success
    )

    print("\n📊 Executor Status")
    print(f"{'=' * 50}")
    print(f"Tasks completed:       {completed_tasks}")
    print(f"Tasks failed:          {failed_tasks}")
    if running_tasks:
        print(f"Tasks in progress:     {len(running_tasks)}")
    if failed_attempts > 0:
        print(f"Failed attempts:       {failed_attempts} (retried)")
    print(
        f"Consecutive failures:  "
        f"{state.consecutive_failures}/{config.max_consecutive_failures}"
    )

    # Tasks with attempts
    attempted = [ts for ts in state.tasks.values() if ts.attempts]
    if attempted:
        print("\n📝 Task History:")
        for ts in attempted:
            icon = (
                "✅"
                if ts.status == "success"
                else "❌"
                if ts.status == "failed"
                else "🔄"
            )
            attempts_info = f"{ts.attempt_count} attempt"
            if ts.attempt_count > 1:
                attempts_info += "s"
            print(f"   {icon} {ts.task_id}: {ts.status} ({attempts_info})")
            if ts.status == "failed" and ts.last_error:
                print(f"      Last error: {ts.last_error[:50]}...")
            elif ts.status == "running" and ts.last_error:
                print(f"      ⚠️  Last attempt failed: {ts.last_error[:50]}...")


def cmd_retry(args, config: ExecutorConfig):
    """Retry failed task"""

    tasks = parse_tasks(TASKS_FILE)
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
        description="ATP Task Executor — automatic task execution via Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Max retries per task (default: 3)"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Task timeout in minutes (default: 30)"
    )
    parser.add_argument(
        "--no-tests", action="store_true", help="Skip tests on task completion"
    )
    parser.add_argument(
        "--no-branch", action="store_true", help="Skip git branch creation"
    )
    parser.add_argument(
        "--no-commit", action="store_true", help="Skip auto-commit on success"
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

    # Load config from YAML file, then override with CLI args
    yaml_config = load_config_from_yaml()
    config = build_config(yaml_config, args)

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

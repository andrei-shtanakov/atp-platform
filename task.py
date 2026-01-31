#!/usr/bin/env python3
"""
ATP Task Manager — CLI for managing tasks from tasks.md

Usage:
    python task.py list                    # List all tasks
    python task.py list --status=todo      # Filter by status
    python task.py list --priority=p0      # Filter by priority
    python task.py list --milestone=mvp    # Filter by milestone
    python task.py show TASK-001           # Task details
    python task.py start TASK-001          # Start task
    python task.py done TASK-001           # Complete task
    python task.py block TASK-001          # Block task
    python task.py check TASK-001 2        # Mark checklist item
    python task.py stats                   # Statistics
    python task.py next                    # Next task (by dependencies)
    python task.py graph                   # ASCII dependency graph
    python task.py export-gh               # Export to GitHub Issues
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Configuration
TASKS_FILE = Path("spec/tasks.md")
HISTORY_FILE = Path("spec/.task-history.log")

# Patterns
TASK_HEADER = re.compile(r"^### (TASK-\d+): (.+)$")
TASK_META = re.compile(r"^(🔴|🟠|🟡|🟢) (P\d) \| (⬜|🔄|✅|⏸️) (\w+)")
CHECKLIST_ITEM = re.compile(r"^- \[([ x])\] (.+)$")
TRACES_TO = re.compile(r"\*\*Traces to:\*\* (.+)")
DEPENDS_ON = re.compile(r"\*\*Depends on:\*\* (.+)")
BLOCKS = re.compile(r"\*\*Blocks:\*\* (.+)")
ESTIMATE = re.compile(r"Est: (\d+(?:-\d+)?[dh])")

STATUS_EMOJI = {"todo": "⬜", "in_progress": "🔄", "done": "✅", "blocked": "⏸️"}

STATUS_FROM_EMOJI = {v: k for k, v in STATUS_EMOJI.items()}

PRIORITY_EMOJI = {"p0": "🔴", "p1": "🟠", "p2": "🟡", "p3": "🟢"}

PRIORITY_FROM_EMOJI = {v: k for k, v in PRIORITY_EMOJI.items()}


@dataclass
class Task:
    id: str
    name: str
    priority: str  # p0, p1, p2, p3
    status: str  # todo, in_progress, done, blocked
    estimate: str
    description: str = ""
    checklist: list = field(default_factory=list)
    traces_to: list = field(default_factory=list)
    depends_on: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    milestone: str = ""
    line_number: int = 0

    @property
    def checklist_progress(self) -> tuple[int, int]:
        done = sum(1 for item, checked in self.checklist if checked)
        return done, len(self.checklist)

    @property
    def is_ready(self) -> bool:
        """Task is ready to work if all dependencies are completed"""
        return self.status == "todo" and not self.depends_on


def parse_tasks(filepath: Path) -> list[Task]:
    """Parse tasks.md and return list of tasks"""
    if not filepath.exists():
        print(f"❌ File {filepath} not found")
        sys.exit(1)

    content = filepath.read_text()
    lines = content.split("\n")

    tasks = []
    current_task = None
    current_milestone = ""
    in_checklist = False
    in_tests = False

    for i, line in enumerate(lines):
        # Determine milestone
        if line.startswith("## Milestone"):
            current_milestone = line.replace("## ", "").strip()
            continue

        # Start of new task
        header_match = TASK_HEADER.match(line)
        if header_match:
            if current_task:
                tasks.append(current_task)

            task_id, task_name = header_match.groups()
            current_task = Task(
                id=task_id,
                name=task_name,
                priority="p0",
                status="todo",
                estimate="",
                milestone=current_milestone,
                line_number=i + 1,
            )
            in_checklist = False
            in_tests = False
            continue

        if not current_task:
            continue

        # Metadata (priority, status)
        meta_match = TASK_META.match(line)
        if meta_match:
            priority_emoji, priority, status_emoji, status_text = meta_match.groups()
            current_task.priority = PRIORITY_FROM_EMOJI.get(priority_emoji, "p0")
            current_task.status = STATUS_FROM_EMOJI.get(status_emoji, "todo")

            est_match = ESTIMATE.search(line)
            if est_match:
                current_task.estimate = est_match.group(1)
            continue

        # Description
        if line.startswith("**Description:**"):
            continue

        # Checklist section
        if line.startswith("**Checklist:**") or line.startswith("**Tests"):
            in_checklist = True
            in_tests = "Tests" in line
            continue

        # Checklist item
        if in_checklist:
            check_match = CHECKLIST_ITEM.match(line)
            if check_match:
                checked = check_match.group(1) == "x"
                text = check_match.group(2)
                prefix = "[TEST] " if in_tests else ""
                current_task.checklist.append((prefix + text, checked))
                continue
            elif line.strip() and not line.startswith("**"):
                continue
            elif line.startswith("**"):
                in_checklist = False
                in_tests = False

        # Traces, Dependencies
        traces_match = TRACES_TO.search(line)
        if traces_match:
            refs = re.findall(r"\[([A-Z]+-\d+)\]", traces_match.group(1))
            current_task.traces_to = refs
            continue

        depends_match = DEPENDS_ON.search(line)
        if depends_match:
            text = depends_match.group(1)
            if text.strip() != "—":
                refs = re.findall(r"\[(TASK-\d+)\]", text)
                current_task.depends_on = refs
            continue

        blocks_match = BLOCKS.search(line)
        if blocks_match:
            text = blocks_match.group(1)
            if text.strip() != "—":
                refs = re.findall(r"\[(TASK-\d+)\]", text)
                current_task.blocks = refs

    if current_task:
        tasks.append(current_task)

    return tasks


def update_task_status(filepath: Path, task_id: str, new_status: str) -> bool:
    """Update task status in file"""
    content = filepath.read_text()
    lines = content.split("\n")

    found = False
    for i, line in enumerate(lines):
        if TASK_HEADER.match(line) and task_id in line:
            found = True
            continue

        if found and TASK_META.match(line):
            # Replace status
            old_emoji = None
            for emoji in STATUS_EMOJI.values():
                if emoji in line:
                    old_emoji = emoji
                    break

            if old_emoji:
                new_emoji = STATUS_EMOJI[new_status]
                new_line = line.replace(old_emoji, new_emoji)
                new_line = re.sub(
                    r"\| (⬜|🔄|✅|⏸️) \w+",
                    f"| {new_emoji} {new_status.upper()}",
                    new_line,
                )
                lines[i] = new_line

                filepath.write_text("\n".join(lines))
                log_change(task_id, f"status -> {new_status}")
                return True

    return False


def update_checklist_item(
    filepath: Path, task_id: str, item_index: int, checked: bool
) -> bool:
    """Update checklist item"""
    content = filepath.read_text()
    lines = content.split("\n")

    in_task = False
    checklist_count = 0

    for i, line in enumerate(lines):
        if TASK_HEADER.match(line):
            in_task = task_id in line
            checklist_count = 0
            continue

        if in_task and CHECKLIST_ITEM.match(line):
            if checklist_count == item_index:
                mark = "x" if checked else " "
                new_line = re.sub(r"- \[[ x]\]", f"- [{mark}]", line)
                lines[i] = new_line
                filepath.write_text("\n".join(lines))
                log_change(
                    task_id,
                    f"checklist[{item_index}] -> {'done' if checked else 'undone'}",
                )
                return True
            checklist_count += 1

    return False


def mark_all_checklist_done(filepath: Path, task_id: str) -> int:
    """Mark all checklist items as done for a task.

    Returns number of items marked.
    """
    content = filepath.read_text()
    lines = content.split("\n")

    in_task = False
    marked_count = 0

    for i, line in enumerate(lines):
        if TASK_HEADER.match(line):
            in_task = task_id in line
            continue

        # Stop when reaching next task
        if in_task and line.startswith("### TASK-"):
            break

        if in_task and CHECKLIST_ITEM.match(line):
            if "[ ]" in line:
                lines[i] = line.replace("[ ]", "[x]")
                marked_count += 1

    if marked_count > 0:
        filepath.write_text("\n".join(lines))
        log_change(task_id, f"checklist: marked {marked_count} items done")

    return marked_count


def log_change(task_id: str, change: str):
    """Log change to history"""
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    with open(HISTORY_FILE, "a") as f:
        timestamp = datetime.now().isoformat()
        f.write(f"{timestamp} | {task_id} | {change}\n")


def get_task_by_id(tasks: list[Task], task_id: str) -> Task | None:
    """Find task by ID"""
    for task in tasks:
        if task.id == task_id:
            return task
    return None


def resolve_dependencies(tasks: list[Task]) -> list[Task]:
    """Update depends_on based on dependency status"""
    task_map = {t.id: t for t in tasks}

    for task in tasks:
        # Remove completed dependencies
        task.depends_on = [
            dep
            for dep in task.depends_on
            if dep in task_map and task_map[dep].status != "done"
        ]

    return tasks


def get_next_tasks(tasks: list[Task]) -> list[Task]:
    """Return tasks ready to execute"""
    tasks = resolve_dependencies(tasks)
    ready = [t for t in tasks if t.status == "todo" and not t.depends_on]
    # Sort by priority
    priority_order = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}
    ready.sort(key=lambda t: priority_order.get(t.priority, 99))
    return ready


# === CLI Commands ===


def cmd_list(args, tasks: list[Task]):
    """List tasks"""
    filtered = tasks

    if args.status:
        filtered = [t for t in filtered if t.status == args.status]

    if args.priority:
        filtered = [t for t in filtered if t.priority == args.priority.lower()]

    if args.milestone:
        milestone_lower = args.milestone.lower()
        filtered = [t for t in filtered if milestone_lower in t.milestone.lower()]

    if not filtered:
        print("No tasks matching criteria")
        return

    header = (
        f"\n{'ID':<12} {'Status':<4} {'P':<3} {'Name':<40} {'Progress':<10} {'Est':<6}"
    )
    print(header)
    print("-" * 85)

    for task in filtered:
        done, total = task.checklist_progress
        progress = f"{done}/{total}" if total > 0 else "—"
        status_icon = STATUS_EMOJI.get(task.status, "?")
        priority_icon = PRIORITY_EMOJI.get(task.priority, "?")

        name = task.name[:38] + ".." if len(task.name) > 40 else task.name
        line = (
            f"{task.id:<12} {status_icon:<4} {priority_icon:<3} "
            f"{name:<40} {progress:<10} {task.estimate:<6}"
        )
        print(line)

    print(f"\nTotal: {len(filtered)} tasks")


def cmd_show(args, tasks: list[Task]):
    """Task details"""
    task = get_task_by_id(tasks, args.task_id.upper())
    if not task:
        print(f"❌ Task {args.task_id} not found")
        return

    status_icon = STATUS_EMOJI.get(task.status, "?")
    priority_icon = PRIORITY_EMOJI.get(task.priority, "?")
    done, total = task.checklist_progress

    print(f"\n{'=' * 60}")
    print(f"{priority_icon} {task.id}: {task.name}")
    print(f"{'=' * 60}")
    print(f"Status:     {status_icon} {task.status.upper()}")
    print(f"Priority:   {task.priority.upper()}")
    print(f"Milestone:  {task.milestone}")
    print(f"Estimate:   {task.estimate or '—'}")
    print(f"Progress:   {done}/{total} ({done * 100 // total if total else 0}%)")

    if task.depends_on:
        print(f"\n⬅️  Depends on: {', '.join(task.depends_on)}")
    if task.blocks:
        print(f"➡️  Blocks:     {', '.join(task.blocks)}")
    if task.traces_to:
        print(f"📋 Traces to:  {', '.join(task.traces_to)}")

    if task.checklist:
        print("\n📝 Checklist:")
        for i, (item, checked) in enumerate(task.checklist):
            mark = "✅" if checked else "⬜"
            print(f"   {i}. {mark} {item}")


def cmd_start(args, tasks: list[Task]):
    """Start task"""
    task = get_task_by_id(tasks, args.task_id.upper())
    if not task:
        print(f"❌ Task {args.task_id} not found")
        return

    # Check dependencies
    tasks = resolve_dependencies(tasks)
    task = get_task_by_id(tasks, args.task_id.upper())

    if task.depends_on:
        print(f"⚠️  Task depends on incomplete: {', '.join(task.depends_on)}")
        if not args.force:
            print("   Use --force to start anyway")
            return

    if update_task_status(TASKS_FILE, task.id, "in_progress"):
        print(f"🔄 {task.id} started!")
    else:
        print("❌ Failed to update status")


def cmd_done(args, tasks: list[Task]):
    """Complete task"""
    task = get_task_by_id(tasks, args.task_id.upper())
    if not task:
        print(f"❌ Task {args.task_id} not found")
        return

    # Check checklist
    done, total = task.checklist_progress
    if total > 0 and done < total:
        print(f"⚠️  Checklist incomplete: {done}/{total}")
        if not args.force:
            print("   Use --force to complete anyway")
            return

    if update_task_status(TASKS_FILE, task.id, "done"):
        print(f"✅ {task.id} completed!")

        # Show unblocked tasks
        tasks = parse_tasks(TASKS_FILE)
        tasks = resolve_dependencies(tasks)
        unblocked = [t for t in tasks if t.status == "todo" and not t.depends_on]
        if unblocked:
            print("\n🔓 Unblocked tasks:")
            for t in unblocked[:5]:
                print(f"   {t.id}: {t.name}")
    else:
        print("❌ Failed to update status")


def cmd_block(args, tasks: list[Task]):
    """Block task"""
    task = get_task_by_id(tasks, args.task_id.upper())
    if not task:
        print(f"❌ Task {args.task_id} not found")
        return

    if update_task_status(TASKS_FILE, task.id, "blocked"):
        print(f"⏸️ {task.id} blocked")
    else:
        print("❌ Failed to update status")


def cmd_check(args, tasks: list[Task]):
    """Mark checklist item"""
    task = get_task_by_id(tasks, args.task_id.upper())
    if not task:
        print(f"❌ Task {args.task_id} not found")
        return

    item_index = int(args.item_index)
    if item_index < 0 or item_index >= len(task.checklist):
        print(f"❌ Invalid index. Available: 0-{len(task.checklist) - 1}")
        return

    item_text, was_checked = task.checklist[item_index]
    new_checked = not was_checked  # toggle

    if update_checklist_item(TASKS_FILE, task.id, item_index, new_checked):
        mark = "✅" if new_checked else "⬜"
        print(f"{mark} {item_text}")
    else:
        print("❌ Failed to update checklist")


def cmd_stats(args, tasks: list[Task]):
    """Task statistics"""
    tasks = resolve_dependencies(tasks)

    by_status = {}
    by_priority = {}
    by_milestone = {}
    total_estimate = 0

    for task in tasks:
        by_status[task.status] = by_status.get(task.status, 0) + 1
        by_priority[task.priority] = by_priority.get(task.priority, 0) + 1
        by_milestone[task.milestone] = by_milestone.get(task.milestone, 0) + 1

        # Parse estimate
        if task.estimate:
            match = re.match(r"(\d+)", task.estimate)
            if match:
                total_estimate += int(match.group(1))

    print("\n📊 Task Statistics")
    print("=" * 40)

    print("\nBy status:")
    for status, count in sorted(by_status.items()):
        icon = STATUS_EMOJI.get(status, "?")
        pct = count * 100 // len(tasks)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"  {icon} {status:<12} {count:>3} {bar} {pct}%")

    print("\nBy priority:")
    for priority in ["p0", "p1", "p2", "p3"]:
        count = by_priority.get(priority, 0)
        icon = PRIORITY_EMOJI.get(priority, "?")
        print(f"  {icon} {priority.upper():<3} {count:>3}")

    print("\nBy milestone:")
    for milestone, count in sorted(by_milestone.items()):
        print(f"  {milestone:<25} {count:>3}")

    ready = get_next_tasks(tasks)
    print(f"\n🚀 Ready to work: {len(ready)}")
    for t in ready[:3]:
        print(f"   {PRIORITY_EMOJI[t.priority]} {t.id}: {t.name}")

    done_count = by_status.get("done", 0)
    progress = done_count * 100 // len(tasks) if tasks else 0
    print(f"\n📈 Overall progress: {done_count}/{len(tasks)} ({progress}%)")
    print(f"⏱️  Total estimate: ~{total_estimate}d")


def cmd_next(args, tasks: list[Task]):
    """Next task to work on"""
    ready = get_next_tasks(tasks)

    if not ready:
        in_progress = [t for t in tasks if t.status == "in_progress"]
        if in_progress:
            print("🔄 Currently in progress:")
            for t in in_progress:
                done, total = t.checklist_progress
                print(f"   {t.id}: {t.name} ({done}/{total})")
        else:
            print("🎉 All tasks completed or blocked!")
        return

    print("🚀 Next tasks (ready to work):\n")
    for i, task in enumerate(ready[:5], 1):
        icon = PRIORITY_EMOJI.get(task.priority, "?")
        deps_done = "✓ deps OK" if not task.depends_on else ""
        print(f"{i}. {icon} {task.id}: {task.name}")
        print(f"   Est: {task.estimate or '?'} | {task.milestone} {deps_done}")
        if task.checklist:
            print(f"   Checklist: {len(task.checklist)} items")
        print()


def cmd_graph(args, tasks: list[Task]):
    """ASCII dependency graph"""
    print("\n📊 Dependency Graph\n")

    # Find roots (no dependencies)
    roots = [t for t in tasks if not t.depends_on]

    def print_tree(task_id: str, indent: int = 0, visited: set = None):
        if visited is None:
            visited = set()

        if task_id in visited:
            return
        visited.add(task_id)

        task = get_task_by_id(tasks, task_id)
        if not task:
            return

        prefix = "  " * indent + ("├── " if indent > 0 else "")
        status_icon = STATUS_EMOJI.get(task.status, "?")

        print(f"{prefix}{status_icon} {task.id}: {task.name[:30]}")

        # Find tasks that depend on this one
        dependents = [t for t in tasks if task_id in t.depends_on]
        for dep in dependents:
            print_tree(dep.id, indent + 1, visited)

    for root in roots[:10]:  # Limit output
        print_tree(root.id)
        print()


def cmd_export_gh(args, tasks: list[Task]):
    """Export to GitHub Issues format"""
    print("# GitHub Issues Export\n")
    print("Execute commands to create issues:\n")
    print("```bash")

    for task in tasks:
        if task.status == "done":
            continue

        labels = f"priority:{task.priority}"
        if task.milestone:
            labels += f",milestone:{task.milestone.lower().replace(' ', '-')}"

        body = f"**Estimate:** {task.estimate or 'TBD'}\\n\\n"
        if task.checklist:
            body += "**Checklist:**\\n"
            for item, checked in task.checklist:
                mark = "x" if checked else " "
                body += f"- [{mark}] {item}\\n"

        if task.depends_on:
            body += f"\\n**Depends on:** {', '.join(task.depends_on)}"

        cmd = (
            f'gh issue create --title "{task.id}: {task.name}" '
            f'--body "{body}" --label "{labels}"'
        )
        print(cmd)

    print("```")


def main():
    parser = argparse.ArgumentParser(
        description="ATP Task Manager — manage tasks from tasks.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # list
    list_parser = subparsers.add_parser("list", aliases=["ls"], help="List tasks")
    list_parser.add_argument(
        "--status", "-s", choices=["todo", "in_progress", "done", "blocked"]
    )
    list_parser.add_argument("--priority", "-p", choices=["p0", "p1", "p2", "p3"])
    list_parser.add_argument("--milestone", "-m", help="Filter by milestone")

    # show
    show_parser = subparsers.add_parser("show", help="Task details")
    show_parser.add_argument("task_id", help="Task ID (e.g., TASK-001)")

    # start
    start_parser = subparsers.add_parser("start", help="Start task")
    start_parser.add_argument("task_id", help="Task ID")
    start_parser.add_argument(
        "--force", "-f", action="store_true", help="Ignore dependencies"
    )

    # done
    done_parser = subparsers.add_parser("done", help="Complete task")
    done_parser.add_argument("task_id", help="Task ID")
    done_parser.add_argument(
        "--force", "-f", action="store_true", help="Ignore incomplete checklist"
    )

    # block
    block_parser = subparsers.add_parser("block", help="Block task")
    block_parser.add_argument("task_id", help="Task ID")

    # check
    check_parser = subparsers.add_parser("check", help="Mark checklist item")
    check_parser.add_argument("task_id", help="Task ID")
    check_parser.add_argument("item_index", help="Item index (0, 1, 2...)")

    # stats
    subparsers.add_parser("stats", help="Statistics")

    # next
    subparsers.add_parser("next", help="Next tasks")

    # graph
    subparsers.add_parser("graph", help="Dependency graph")

    # export-gh
    subparsers.add_parser("export-gh", help="Export to GitHub Issues")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    tasks = parse_tasks(TASKS_FILE)

    commands = {
        "list": cmd_list,
        "ls": cmd_list,
        "show": cmd_show,
        "start": cmd_start,
        "done": cmd_done,
        "block": cmd_block,
        "check": cmd_check,
        "stats": cmd_stats,
        "next": cmd_next,
        "graph": cmd_graph,
        "export-gh": cmd_export_gh,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args, tasks)


if __name__ == "__main__":
    main()

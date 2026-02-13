"""CLI commands for trace management and replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from atp.tracing.models import Trace
    from atp.tracing.storage import FileTraceStorage


@click.group(name="traces")
def traces_command() -> None:
    """Manage agent execution traces.

    List, view, and delete recorded traces.

    Examples:

      atp traces list
      atp traces show <trace_id>
      atp traces delete <trace_id>
    """


@traces_command.command(name="list")
@click.option(
    "--test-id",
    type=str,
    default=None,
    help="Filter traces by test ID",
)
@click.option(
    "--status",
    type=str,
    default=None,
    help="Filter by status (completed, failed, timeout)",
)
@click.option(
    "--limit",
    type=int,
    default=20,
    help="Maximum number of traces to show (default: 20)",
)
@click.option(
    "--traces-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Custom traces directory (default: ~/.atp/traces/)",
)
def traces_list(
    test_id: str | None,
    status: str | None,
    limit: int,
    traces_dir: Path | None,
) -> None:
    """List saved traces.

    Examples:

      atp traces list
      atp traces list --test-id=test-001
      atp traces list --status=completed --limit=10
    """
    from atp.tracing import FileTraceStorage

    storage = FileTraceStorage(base_dir=traces_dir)
    summaries = storage.list_traces(test_id=test_id, status=status, limit=limit)

    if not summaries:
        click.echo("No traces found.")
        return

    click.echo(f"Traces ({len(summaries)}):")
    click.echo("-" * 72)
    for s in summaries:
        started = s.started_at.strftime("%Y-%m-%d %H:%M:%S")
        duration = ""
        if s.completed_at:
            dur = (s.completed_at - s.started_at).total_seconds()
            duration = f" ({dur:.1f}s)"
        click.echo(
            f"  {s.trace_id[:12]}..  "
            f"{s.status:<10} "
            f"{s.test_name or s.test_id:<24} "
            f"{s.total_events:>4} events  "
            f"{started}{duration}"
        )


@traces_command.command(name="show")
@click.argument("trace_id", type=str)
@click.option(
    "--output",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format",
)
@click.option(
    "--traces-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Custom traces directory",
)
def traces_show(
    trace_id: str,
    output: str,
    traces_dir: Path | None,
) -> None:
    """Show details of a specific trace.

    TRACE_ID is the unique identifier for the trace.
    A prefix match is attempted if no exact match is found.

    Examples:

      atp traces show abc123
      atp traces show abc123 --output=json
    """
    from atp.tracing import FileTraceStorage

    storage = FileTraceStorage(base_dir=traces_dir)
    trace = _resolve_trace(storage, trace_id)

    if trace is None:
        click.echo(f"Trace not found: {trace_id}", err=True)
        sys.exit(1)

    if output == "json":
        click.echo(json.dumps(trace.model_dump(mode="json"), indent=2, default=str))
        return

    click.echo(f"Trace: {trace.trace_id}")
    click.echo(f"  Test:    {trace.test_name or trace.test_id}")
    click.echo(f"  Status:  {trace.status}")
    click.echo(f"  Started: {trace.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if trace.completed_at:
        click.echo(f"  Ended:   {trace.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if trace.duration_seconds is not None:
        click.echo(f"  Duration: {trace.duration_seconds:.2f}s")
    click.echo(f"  Events:  {trace.total_events}")
    if trace.error:
        click.echo(f"  Error:   {trace.error}")

    if trace.metadata.agent_name:
        click.echo(f"  Agent:   {trace.metadata.agent_name}")
    if trace.metadata.suite_name:
        click.echo(f"  Suite:   {trace.metadata.suite_name}")

    counts = trace.event_type_counts
    if counts:
        click.echo("  Event types:")
        for etype, count in sorted(counts.items()):
            click.echo(f"    {etype}: {count}")


@traces_command.command(name="delete")
@click.argument("trace_id", type=str)
@click.option(
    "--traces-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Custom traces directory",
)
def traces_delete(
    trace_id: str,
    traces_dir: Path | None,
) -> None:
    """Delete a saved trace.

    Examples:

      atp traces delete abc123
    """
    from atp.tracing import FileTraceStorage

    storage = FileTraceStorage(base_dir=traces_dir)
    if storage.delete(trace_id):
        click.echo(f"Deleted trace: {trace_id}")
    else:
        click.echo(f"Trace not found: {trace_id}", err=True)
        sys.exit(1)


@click.command(name="replay")
@click.argument("trace_id", type=str)
@click.option(
    "--traces-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Custom traces directory",
)
@click.option(
    "--speed",
    type=float,
    default=1.0,
    help="Replay speed multiplier (default: 1.0, 0 = instant)",
)
def replay_command(
    trace_id: str,
    traces_dir: Path | None,
    speed: float,
) -> None:
    """Replay a recorded trace step by step.

    Pretty-prints each event in the trace with timing information.

    TRACE_ID is the unique identifier for the trace.
    A prefix match is attempted if no exact match is found.

    Examples:

      atp replay abc123
      atp replay abc123 --speed=2
      atp replay abc123 --speed=0
    """
    import time

    from atp.tracing import FileTraceStorage

    storage = FileTraceStorage(base_dir=traces_dir)
    trace = _resolve_trace(storage, trace_id)

    if trace is None:
        click.echo(f"Trace not found: {trace_id}", err=True)
        sys.exit(1)

    click.echo(
        f"Replaying trace: {trace.trace_id} ({trace.test_name or trace.test_id})"
    )
    click.echo(f"Status: {trace.status}")
    click.echo(f"Events: {trace.total_events}")
    click.echo("-" * 60)

    prev_ts = None
    for step in trace.steps:
        if prev_ts is not None and speed > 0:
            delta = (step.timestamp - prev_ts).total_seconds()
            if delta > 0:
                time.sleep(delta / speed)

        ts = step.timestamp.strftime("%H:%M:%S.%f")[:-3]
        click.echo(f"[{ts}] #{step.sequence:>3} {step.event_type.value}")

        _print_payload(step.event_type.value, step.payload)
        prev_ts = step.timestamp

    click.echo("-" * 60)
    if trace.error:
        click.echo(f"Error: {trace.error}")
    click.echo(f"Replay complete ({trace.total_events} events)")


def _print_payload(event_type: str, payload: dict[str, object]) -> None:
    """Print a formatted payload summary."""
    if event_type == "tool_call":
        tool = payload.get("tool", "?")
        status = payload.get("status", "")
        click.echo(f"         tool={tool} status={status}")
    elif event_type == "llm_request":
        model = payload.get("model", "?")
        inp = payload.get("input_tokens", "?")
        out = payload.get("output_tokens", "?")
        click.echo(f"         model={model} tokens={inp}/{out}")
    elif event_type == "reasoning":
        thought = payload.get("thought") or payload.get("step")
        if thought:
            text = str(thought)[:80]
            click.echo(f"         {text}")
    elif event_type == "error":
        msg = payload.get("message", "")
        click.echo(f"         {msg}")
    elif event_type == "progress":
        pct = payload.get("percentage")
        msg = payload.get("message", "")
        if pct is not None:
            click.echo(f"         {pct}% {msg}")
        elif msg:
            click.echo(f"         {msg}")


def _resolve_trace(
    storage: FileTraceStorage,  # noqa: F821
    trace_id: str,
) -> Trace | None:  # noqa: F821
    """Resolve a trace by exact or prefix match."""

    trace = storage.load(trace_id)
    if trace is not None:
        return trace

    # Try prefix match
    summaries = storage.list_traces(limit=500)
    matches = [s for s in summaries if s.trace_id.startswith(trace_id)]
    if len(matches) == 1:
        return storage.load(matches[0].trace_id)
    if len(matches) > 1:
        click.echo(
            f"Ambiguous trace ID prefix '{trace_id}', matches {len(matches)} traces:",
            err=True,
        )
        for m in matches[:5]:
            click.echo(f"  {m.trace_id}", err=True)
    return None

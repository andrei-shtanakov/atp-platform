#!/usr/bin/env python3
"""ATP OpenAI Agent - LLM-powered agent with tool calling.

A CLI agent that uses OpenAI API to process tasks with tool calling support.

Protocol:
- Input: ATPRequest as JSON from stdin
- Output: ATPResponse as JSON to stdout
- Events: ATPEvent as JSONL to stderr

Environment:
- OPENAI_API_KEY: Required - OpenAI API key
- OPENAI_MODEL: Optional - Model to use (default: gpt-4o-mini)
- ATP_WORKSPACE: Optional - Workspace directory (default: /tmp/atp_workspace)
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from openai import OpenAI

# Available tools for the agent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a file with the specified content",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to create",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content from a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to read",
                    },
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to list (default: current directory)",
                        "default": ".",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g., '2 + 2')",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


class EventEmitter:
    """Helper class to emit ATP events to stderr."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.sequence = 0

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit an ATP event to stderr."""
        event = {
            "version": "1.0",
            "task_id": self.task_id,
            "timestamp": datetime.now().isoformat(),
            "sequence": self.sequence,
            "event_type": event_type,
            "payload": payload,
        }
        print(json.dumps(event), file=sys.stderr, flush=True)
        self.sequence += 1


class ToolExecutor:
    """Executes tool calls in the workspace."""

    def __init__(self, workspace: Path, emitter: EventEmitter) -> None:
        self.workspace = workspace
        self.emitter = emitter

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return the result."""
        self.emitter.emit(
            "tool_call",
            {"tool": tool_name, "input": arguments, "status": "started"},
        )

        start_time = time.time()

        if tool_name == "create_file":
            result = self._create_file(arguments)
        elif tool_name == "read_file":
            result = self._read_file(arguments)
        elif tool_name == "list_files":
            result = self._list_files(arguments)
        elif tool_name == "calculate":
            result = self._calculate(arguments)
        else:
            result = {"success": False, "error": f"Unknown tool: {tool_name}"}

        duration_ms = (time.time() - start_time) * 1000

        self.emitter.emit(
            "tool_call",
            {
                "tool": tool_name,
                "output": result,
                "status": "success" if result.get("success") else "error",
                "duration_ms": duration_ms,
            },
        )

        return result

    def _create_file(self, args: dict[str, Any]) -> dict[str, Any]:
        """Create a file with content."""
        filename = args.get("filename", "")
        content = args.get("content", "")

        if not filename:
            return {"success": False, "error": "Filename is required"}

        filepath = self.workspace / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)

        return {
            "success": True,
            "path": filename,
            "size": len(content),
            "message": f"Created file: {filename}",
        }

    def _read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        """Read content from a file."""
        filename = args.get("filename", "")

        if not filename:
            return {"success": False, "error": "Filename is required"}

        filepath = self.workspace / filename
        if not filepath.exists():
            return {"success": False, "error": f"File not found: {filename}"}

        content = filepath.read_text()
        return {
            "success": True,
            "path": filename,
            "content": content,
            "size": len(content),
        }

    def _list_files(self, args: dict[str, Any]) -> dict[str, Any]:
        """List files in a directory."""
        directory = args.get("directory", ".")
        target = self.workspace / directory

        if not target.exists():
            return {"success": False, "error": f"Directory not found: {directory}"}

        files = []
        for item in target.iterdir():
            files.append(
                {
                    "name": item.name,
                    "is_file": item.is_file(),
                    "size": item.stat().st_size if item.is_file() else 0,
                }
            )

        return {
            "success": True,
            "path": directory,
            "files": files,
            "count": len(files),
        }

    def _calculate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Perform a calculation."""
        expression = args.get("expression", "")

        if not expression:
            return {"success": False, "error": "Expression is required"}

        # Safe evaluation - only allow basic math
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return {"success": False, "error": "Invalid characters in expression"}

        try:
            result = eval(expression)  # noqa: S307
            return {
                "success": True,
                "expression": expression,
                "result": result,
            }
        except Exception as e:
            return {"success": False, "error": f"Calculation error: {e}"}


class OpenAIAgent:
    """OpenAI-powered agent with tool calling."""

    def __init__(
        self,
        api_key: str,
        model: str,
        workspace: Path,
        emitter: EventEmitter,
        max_steps: int = 10,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.workspace = workspace
        self.emitter = emitter
        self.max_steps = max_steps
        self.tool_executor = ToolExecutor(workspace, emitter)
        self.artifacts: list[dict[str, Any]] = []
        self.metrics = {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_calls": 0,
            "llm_calls": 0,
        }

    def run(self, task_description: str) -> tuple[str, list[dict[str, Any]]]:
        """Run the agent on a task.

        Returns:
            Tuple of (status, artifacts)
        """
        self.emitter.emit("progress", {"message": "Agent started", "percentage": 0})

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that can perform file operations "
                    "and calculations. Use the provided tools to complete tasks. "
                    "Be concise and efficient."
                ),
            },
            {"role": "user", "content": task_description},
        ]

        for step in range(self.max_steps):
            self.emitter.emit(
                "progress",
                {
                    "message": f"Step {step + 1}/{self.max_steps}",
                    "current_step": step + 1,
                    "percentage": int((step / self.max_steps) * 100),
                },
            )

            # Call OpenAI
            response = self._call_llm(messages)
            if response is None:
                return "failed", self.artifacts

            message = response.choices[0].message

            # Check if we're done
            if message.tool_calls is None:
                self.emitter.emit(
                    "reasoning",
                    {"thought": message.content or "Task completed"},
                )
                self.emitter.emit(
                    "progress", {"message": "Agent completed", "percentage": 100}
                )
                return "completed", self.artifacts

            # Process tool calls
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                result = self.tool_executor.execute(tool_name, arguments)
                self.metrics["tool_calls"] += 1

                # Track artifacts for file creation
                if tool_name == "create_file" and result.get("success"):
                    self.artifacts.append(
                        {
                            "type": "file",
                            "path": arguments.get("filename"),
                            "content": arguments.get("content"),
                            "size_bytes": len(arguments.get("content", "")),
                        }
                    )

                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

        # Max steps reached
        self.emitter.emit(
            "progress", {"message": "Max steps reached", "percentage": 100}
        )
        return "partial", self.artifacts

    def _call_llm(self, messages: list[dict[str, Any]]) -> Any:
        """Call OpenAI API."""
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=cast(Any, messages),
                tools=cast(Any, TOOLS),
                tool_choice="auto",
            )

            duration_ms = (time.time() - start_time) * 1000
            self.metrics["llm_calls"] += 1

            # Update token metrics
            if response.usage:
                self.metrics["input_tokens"] += response.usage.prompt_tokens
                self.metrics["output_tokens"] += response.usage.completion_tokens
                self.metrics["total_tokens"] += response.usage.total_tokens

            self.emitter.emit(
                "llm_request",
                {
                    "model": self.model,
                    "input_tokens": response.usage.prompt_tokens
                    if response.usage
                    else 0,
                    "output_tokens": (
                        response.usage.completion_tokens if response.usage else 0
                    ),
                    "duration_ms": duration_ms,
                },
            )

            return response

        except Exception as e:
            self.emitter.emit(
                "error",
                {
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "recoverable": False,
                },
            )
            return None


def build_response(
    task_id: str,
    status: str,
    artifacts: list[dict[str, Any]],
    metrics: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    """Build ATP Response."""
    response = {
        "version": "1.0",
        "task_id": task_id,
        "status": status,
        "artifacts": artifacts,
        "metrics": {
            "total_tokens": metrics.get("total_tokens"),
            "input_tokens": metrics.get("input_tokens"),
            "output_tokens": metrics.get("output_tokens"),
            "total_steps": metrics.get("llm_calls", 0),
            "tool_calls": metrics.get("tool_calls", 0),
            "llm_calls": metrics.get("llm_calls", 0),
        },
    }
    if error:
        response["error"] = error
    return response


def main() -> None:
    """Main entry point."""
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        error_response = {
            "version": "1.0",
            "task_id": "unknown",
            "status": "failed",
            "artifacts": [],
            "error": "OPENAI_API_KEY environment variable is required",
        }
        print(json.dumps(error_response))
        sys.exit(1)

    # Read ATP Request from stdin
    try:
        request = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        error_response = {
            "version": "1.0",
            "task_id": "unknown",
            "status": "failed",
            "artifacts": [],
            "error": f"Invalid JSON input: {e}",
        }
        print(json.dumps(error_response))
        sys.exit(1)

    # Parse request
    task_id = request.get("task_id", "unknown")
    task = request.get("task") or {}
    description = task.get("description", "")
    context = request.get("context") or {}
    constraints = request.get("constraints") or {}

    workspace_path = context.get("workspace_path") or os.environ.get(
        "ATP_WORKSPACE", "/tmp/atp_workspace"
    )
    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    max_steps = constraints.get("max_steps", 10)

    # Create and run agent
    emitter = EventEmitter(task_id)
    agent = OpenAIAgent(
        api_key=api_key,
        model=model,
        workspace=workspace,
        emitter=emitter,
        max_steps=max_steps,
    )

    status, artifacts = agent.run(description)

    # Build and output response
    error = None
    if status == "failed":
        error = "Agent execution failed"
    elif status == "partial":
        error = "Max steps reached before completion"

    response = build_response(task_id, status, artifacts, agent.metrics, error)
    print(json.dumps(response))


if __name__ == "__main__":
    main()

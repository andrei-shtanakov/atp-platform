#!/usr/bin/env python3
"""ATP Demo Agent - File Operations.

A simple CLI agent that demonstrates the ATP Protocol by performing
file operations (create, read, list) in a workspace directory.

Protocol:
- Input: ATPRequest as JSON from stdin
- Output: ATPResponse as JSON to stdout
- Events: ATPEvent as JSONL to stderr
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def emit_event(
    task_id: str,
    event_type: str,
    payload: dict[str, Any],
    sequence: int,
) -> None:
    """Emit an ATP event to stderr as JSON."""
    event = {
        "version": "1.0",
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "sequence": sequence,
        "event_type": event_type,
        "payload": payload,
    }
    print(json.dumps(event), file=sys.stderr, flush=True)


def create_file(workspace: Path, filename: str, content: str) -> dict[str, Any]:
    """Create a file with the given content."""
    filepath = workspace / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content)
    return {
        "success": True,
        "path": filename,
        "size": len(content),
        "message": f"Created file: {filename}",
    }


def read_file(workspace: Path, filename: str) -> dict[str, Any]:
    """Read content from a file."""
    filepath = workspace / filename
    if not filepath.exists():
        return {
            "success": False,
            "path": filename,
            "error": f"File not found: {filename}",
        }
    content = filepath.read_text()
    return {
        "success": True,
        "path": filename,
        "content": content,
        "size": len(content),
    }


def list_files(workspace: Path, directory: str = ".") -> dict[str, Any]:
    """List files in a directory."""
    target = workspace / directory
    if not target.exists():
        return {
            "success": False,
            "path": directory,
            "error": f"Directory not found: {directory}",
        }
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


def parse_file_operation(description: str) -> tuple[str, dict[str, Any]]:
    """Parse task description to determine file operation.

    Returns:
        Tuple of (operation, params) where operation is one of:
        - create_file
        - read_file
        - list_files
        - multi_step (create then read)
    """
    desc_lower = description.lower()

    # Multi-step: create + read
    if "create" in desc_lower and "read" in desc_lower:
        # Extract filename and content
        match = re.search(
            r"create\s+(?:file\s+)?['\"]?(\S+?)['\"]?\s+with\s+(?:content\s+)?['\"](.+?)['\"]",
            description,
            re.IGNORECASE,
        )
        if match:
            return "multi_step", {"filename": match.group(1), "content": match.group(2)}
        # Try another pattern
        match = re.search(r"['\"](\S+\.txt)['\"]", description)
        if match:
            return "multi_step", {
                "filename": match.group(1),
                "content": "Default content",
            }

    # Create file
    if "create" in desc_lower:
        # Pattern: create file 'name' with content 'content'
        match = re.search(
            r"create\s+(?:file\s+)?['\"]?(\S+?)['\"]?\s+with\s+(?:content\s+)?['\"](.+?)['\"]",
            description,
            re.IGNORECASE,
        )
        if match:
            return "create_file", {
                "filename": match.group(1),
                "content": match.group(2),
            }
        # Pattern: create file 'name' with 'content'
        match = re.search(
            r"create\s+(?:file\s+)?['\"](\S+)['\"].*?['\"](.+?)['\"]",
            description,
            re.IGNORECASE,
        )
        if match:
            return "create_file", {
                "filename": match.group(1),
                "content": match.group(2),
            }
        # Default
        return "create_file", {"filename": "output.txt", "content": "Default content"}

    # Read file
    if "read" in desc_lower:
        match = re.search(
            r"read\s+(?:file\s+)?['\"]?(\S+?)['\"]?(?:\s|$)", description, re.IGNORECASE
        )
        if match:
            return "read_file", {"filename": match.group(1).strip("'\"")}
        return "read_file", {"filename": "input.txt"}

    # List files
    if "list" in desc_lower:
        match = re.search(
            r"list\s+(?:files\s+)?(?:in\s+)?['\"]?(\S+?)['\"]?(?:\s|$)",
            description,
            re.IGNORECASE,
        )
        if match:
            return "list_files", {"directory": match.group(1).strip("'\"")}
        return "list_files", {"directory": "."}

    # Default: echo/unknown
    return "unknown", {"description": description}


def execute_operation(
    workspace: Path,
    operation: str,
    params: dict[str, Any],
    task_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Execute the file operation and return status and artifacts."""
    seq = 0
    artifacts: list[dict[str, Any]] = []

    if operation == "create_file":
        emit_event(
            task_id,
            "progress",
            {"message": f"Creating file: {params['filename']}"},
            seq,
        )
        seq += 1
        emit_event(
            task_id,
            "tool_call",
            {"tool": "create_file", "input": params, "status": "started"},
            seq,
        )
        seq += 1
        result = create_file(workspace, params["filename"], params["content"])
        emit_event(
            task_id,
            "tool_call",
            {"tool": "create_file", "output": result, "status": "success"},
            seq,
        )
        if result["success"]:
            artifacts.append(
                {
                    "type": "file",
                    "path": params["filename"],
                    "content": params["content"],
                    "size_bytes": len(params["content"]),
                }
            )
            return "completed", artifacts
        return "failed", artifacts

    elif operation == "read_file":
        emit_event(
            task_id, "progress", {"message": f"Reading file: {params['filename']}"}, seq
        )
        seq += 1
        emit_event(
            task_id,
            "tool_call",
            {"tool": "read_file", "input": params, "status": "started"},
            seq,
        )
        seq += 1
        result = read_file(workspace, params["filename"])
        if result["success"]:
            emit_event(
                task_id,
                "tool_call",
                {"tool": "read_file", "output": result, "status": "success"},
                seq,
            )
            artifacts.append(
                {
                    "type": "structured",
                    "name": "read_result",
                    "data": result,
                }
            )
            return "completed", artifacts
        else:
            # Handle file not found gracefully - return completed with error info
            emit_event(
                task_id,
                "tool_call",
                {"tool": "read_file", "output": result, "status": "error"},
                seq,
            )
            artifacts.append(
                {
                    "type": "structured",
                    "name": "error_result",
                    "data": {
                        "error_type": "FileNotFound",
                        "message": result["error"],
                        "handled": True,
                    },
                }
            )
            return "completed", artifacts

    elif operation == "list_files":
        emit_event(
            task_id,
            "progress",
            {"message": f"Listing files in: {params['directory']}"},
            seq,
        )
        seq += 1
        emit_event(
            task_id,
            "tool_call",
            {"tool": "list_files", "input": params, "status": "started"},
            seq,
        )
        seq += 1
        result = list_files(workspace, params["directory"])
        emit_event(
            task_id,
            "tool_call",
            {"tool": "list_files", "output": result, "status": "success"},
            seq,
        )
        if result["success"]:
            artifacts.append(
                {
                    "type": "structured",
                    "name": "file_list",
                    "data": result,
                }
            )
            return "completed", artifacts
        return "failed", artifacts

    elif operation == "multi_step":
        # Step 1: Create file
        emit_event(
            task_id,
            "progress",
            {"message": "Step 1: Creating file", "current_step": 1},
            seq,
        )
        seq += 1
        emit_event(
            task_id,
            "tool_call",
            {"tool": "create_file", "input": params, "status": "started"},
            seq,
        )
        seq += 1
        create_result = create_file(workspace, params["filename"], params["content"])
        emit_event(
            task_id,
            "tool_call",
            {"tool": "create_file", "output": create_result, "status": "success"},
            seq,
        )
        seq += 1
        if create_result["success"]:
            artifacts.append(
                {
                    "type": "file",
                    "path": params["filename"],
                    "content": params["content"],
                    "size_bytes": len(params["content"]),
                }
            )

        # Step 2: Read file back
        emit_event(
            task_id,
            "progress",
            {"message": "Step 2: Reading file back", "current_step": 2},
            seq,
        )
        seq += 1
        emit_event(
            task_id,
            "tool_call",
            {
                "tool": "read_file",
                "input": {"filename": params["filename"]},
                "status": "started",
            },
            seq,
        )
        seq += 1
        read_result = read_file(workspace, params["filename"])
        emit_event(
            task_id,
            "tool_call",
            {"tool": "read_file", "output": read_result, "status": "success"},
            seq,
        )
        if read_result["success"]:
            artifacts.append(
                {
                    "type": "structured",
                    "name": "verification",
                    "data": {"verified": read_result["content"] == params["content"]},
                }
            )
            return "completed", artifacts
        return "failed", artifacts

    else:
        emit_event(
            task_id,
            "error",
            {
                "error_type": "UnknownOperation",
                "message": f"Unknown operation: {operation}",
                "recoverable": False,
            },
            seq,
        )
        return "failed", artifacts


def build_response(
    task_id: str,
    status: str,
    artifacts: list[dict[str, Any]],
    error: str | None = None,
) -> dict[str, Any]:
    """Build ATP Response."""
    response = {
        "version": "1.0",
        "task_id": task_id,
        "status": status,
        "artifacts": artifacts,
        "metrics": {
            "total_steps": len([a for a in artifacts if a.get("type") == "file"]) + 1,
            "tool_calls": len([a for a in artifacts]),
        },
    }
    if error:
        response["error"] = error
    return response


def main() -> None:
    """Main entry point for the demo agent."""
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

    task_id = request.get("task_id", "unknown")
    task = request.get("task") or {}
    description = task.get("description", "")
    context = request.get("context") or {}
    workspace_path = context.get("workspace_path") or os.environ.get(
        "ATP_WORKSPACE", "/tmp/atp_workspace"
    )
    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)

    # Emit start event
    emit_event(task_id, "progress", {"message": "Agent started", "percentage": 0}, 0)

    # Parse and execute operation
    operation, params = parse_file_operation(description)
    status, artifacts = execute_operation(workspace, operation, params, task_id)

    # Emit completion event
    emit_event(
        task_id,
        "progress",
        {"message": "Agent completed", "percentage": 100},
        99,
    )

    # Build and output response
    error = None
    if status == "failed":
        error = f"Operation {operation} failed"
    response = build_response(task_id, status, artifacts, error)
    print(json.dumps(response))


if __name__ == "__main__":
    main()

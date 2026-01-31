#!/usr/bin/env python3
"""Demo agent for Docker container testing.

This agent performs simple file operations and follows the ATP Protocol:
- Reads ATP Request from stdin (JSON)
- Writes ATP Events to stderr (JSONL)
- Writes ATP Response to stdout (JSON)

Supported operations (in task description):
- "create file <filename> with content <content>"
- "list files"
- "read file <filename>"
"""

import json
import os
import sys
from datetime import UTC, datetime


def emit_event(event_type: str, payload: dict, sequence: int) -> None:
    """Emit an ATP event to stderr."""
    event = {
        "sequence": sequence,
        "timestamp": datetime.now(UTC).isoformat() + "Z",
        "event_type": event_type,
        "payload": payload,
    }
    print(json.dumps(event), file=sys.stderr, flush=True)


def parse_task(description: str) -> tuple[str, dict]:
    """Parse task description into operation and parameters."""
    description = description.lower().strip()

    if description.startswith("create file"):
        # Parse: "create file <filename> with content <content>"
        parts = description.split(" with content ", 1)
        if len(parts) == 2:
            filename = parts[0].replace("create file ", "").strip()
            content = parts[1].strip()
            return "create", {"filename": filename, "content": content}

    elif description.startswith("list files"):
        return "list", {}

    elif description.startswith("read file"):
        filename = description.replace("read file ", "").strip()
        return "read", {"filename": filename}

    return "unknown", {"description": description}


def execute_operation(operation: str, params: dict) -> tuple[str, list[dict]]:
    """Execute the operation and return status and artifacts."""
    artifacts = []

    if operation == "create":
        filename = params["filename"]
        content = params["content"]
        with open(filename, "w") as f:
            f.write(content)
        artifacts.append(
            {
                "type": "file",
                "path": filename,
                "content": content,
                "content_type": "text/plain",
            }
        )
        return "completed", artifacts

    elif operation == "list":
        files = os.listdir(".")
        file_list = "\n".join(files)
        # Save to file (ATP protocol requires file artifacts)
        with open("files_list.txt", "w") as f:
            f.write(file_list)
        artifacts.append(
            {
                "type": "file",
                "path": "files_list.txt",
                "content": file_list,
                "content_type": "text/plain",
            }
        )
        return "completed", artifacts

    elif operation == "read":
        filename = params["filename"]
        if os.path.exists(filename):
            with open(filename) as f:
                content = f.read()
            artifacts.append(
                {
                    "type": "file",
                    "path": filename,
                    "content": content,
                    "content_type": "text/plain",
                }
            )
            return "completed", artifacts
        else:
            return "error", []

    return "error", []


def main() -> None:
    """Main entry point."""
    sequence = 0

    # Read ATP Request from stdin
    try:
        input_data = sys.stdin.read()
        request = json.loads(input_data) if input_data.strip() else {}
    except json.JSONDecodeError as e:
        response = {
            "version": "1.0",
            "task_id": "unknown",
            "status": "error",
            "error": f"Invalid JSON input: {e}",
            "artifacts": [],
        }
        print(json.dumps(response))
        sys.exit(1)

    task_id = request.get("task_id", "unknown")
    task = request.get("task", {})
    description = task.get("description", "")

    # Emit start event
    emit_event("progress", {"message": "Starting task", "percentage": 0}, sequence)
    sequence += 1

    # Parse and execute task
    operation, params = parse_task(description)

    emit_event(
        "progress",
        {"message": f"Executing operation: {operation}", "percentage": 50},
        sequence,
    )
    sequence += 1

    if operation == "unknown":
        response = {
            "version": "1.0",
            "task_id": task_id,
            "status": "error",
            "error": f"Unknown operation in task: {description}",
            "artifacts": [],
        }
    else:
        status, artifacts = execute_operation(operation, params)
        response = {
            "version": "1.0",
            "task_id": task_id,
            "status": status,
            "artifacts": artifacts,
            "metrics": {
                "steps": 1,
                "tool_calls": 1,
            },
        }
        if status == "error":
            response["error"] = f"Failed to execute: {operation}"

    # Emit completion event
    emit_event("progress", {"message": "Task completed", "percentage": 100}, sequence)

    # Output ATP Response to stdout
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()

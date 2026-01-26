#!/usr/bin/env python3
"""Simple MCP agent for testing without OpenAI.

This agent performs basic MCP operations without requiring an LLM.
Useful for testing the infrastructure and MCP server connection.

Usage:
    echo '{"task": {"description": "..."}}' | python mcp_simple_agent.py

Environment:
    MCP_SERVER_URL - MCP server URL (default: http://localhost:9000)
"""

import json
import os
import sys
from datetime import UTC, datetime

import httpx


def get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def main() -> None:
    """Run the simple MCP agent."""
    # Read ATP request from stdin
    try:
        input_data = sys.stdin.read()
        request = json.loads(input_data) if input_data.strip() else {}
    except json.JSONDecodeError as e:
        print(
            json.dumps(
                {
                    "version": "1.0",
                    "task_id": "unknown",
                    "status": "error",
                    "error": f"Invalid JSON input: {e}",
                    "artifacts": [],
                }
            )
        )
        sys.exit(1)

    task = request.get("task", {})
    task_id = request.get("task_id", "test-task")
    task_description = task.get("description", "").lower()

    mcp_url = os.environ.get("MCP_SERVER_URL", "http://localhost:9000")
    client = httpx.Client(timeout=30)
    artifacts: list[dict] = []  # List of Artifact objects
    events: list[dict] = []
    sequence = 0

    def log_event(event_type: str, payload: dict) -> None:
        nonlocal sequence
        sequence += 1
        events.append(
            {
                "sequence": sequence,
                "timestamp": get_timestamp(),
                "event_type": event_type,
                "payload": payload,
            }
        )

    def write_file(path: str, content: str) -> None:
        with open(path, "w") as f:
            f.write(content)
        # Add as proper Artifact object
        artifacts.append(
            {
                "type": "file",
                "path": path,
                "content": content,
                "content_type": "application/json"
                if path.endswith(".json")
                else "text/plain",
            }
        )
        log_event(
            "tool_call", {"tool": "write_file", "path": path, "status": "success"}
        )

    try:
        log_event("progress", {"message": "Connecting to MCP server", "percentage": 10})

        # Always list tools first
        response = client.post(f"{mcp_url}/tools/list", json={})
        response.raise_for_status()
        tools_data = response.json()
        log_event("tool_call", {"tool": "mcp_list_tools", "status": "success"})

        # Check what the task needs
        if "list" in task_description and "tool" in task_description:
            # Task: List tools
            write_file("tools_list.json", json.dumps(tools_data, indent=2))
            log_event("progress", {"message": "Tools list saved", "percentage": 100})

        elif "weather" in task_description:
            # Task: Get weather
            location = "Moscow"
            if "london" in task_description:
                location = "London"
            elif "tokyo" in task_description:
                location = "Tokyo"

            # Extract location from task description
            for word in ["moscow", "london", "tokyo", "paris", "new york"]:
                if word in task_description:
                    location = word.title()
                    break

            response = client.post(
                f"{mcp_url}/tools/call",
                json={"tool": "get_weather", "arguments": {"location": location}},
            )
            weather_data = response.json().get("result", {})
            log_event(
                "tool_call",
                {"tool": "mcp_get_weather", "location": location, "status": "success"},
            )

            # Determine output filename
            filename = f"weather_{location.lower().replace(' ', '_')}.json"
            if "moscow" in task_description:
                filename = "weather_moscow.json"

            write_file(filename, json.dumps(weather_data, indent=2))
            log_event(
                "progress",
                {"message": f"Weather for {location} saved", "percentage": 100},
            )

        elif "search" in task_description:
            # Task: Search
            query = "Python programming"
            if "async" in task_description:
                query = "Python async programming"

            response = client.post(
                f"{mcp_url}/tools/call",
                json={"tool": "search_web", "arguments": {"query": query}},
            )
            search_data = response.json().get("result", {})
            log_event(
                "tool_call",
                {"tool": "mcp_search_web", "query": query, "status": "success"},
            )

            write_file("search_results.json", json.dumps(search_data, indent=2))

            # Create summary
            summary = f"# Search Results for: {query}\n\n"
            for result in search_data.get("results", []):
                summary += f"- [{result['title']}]({result['url']})\n"
                summary += f"  {result['snippet']}\n\n"

            write_file("summary.txt", summary)
            log_event("progress", {"message": "Search completed", "percentage": 100})

        elif "comparison" in task_description or "multi" in task_description:
            # Task: Multi-tool workflow (comparison)
            log_event(
                "progress",
                {"message": "Starting multi-tool workflow", "percentage": 10},
            )

            # List tools
            write_file("available_tools.json", json.dumps(tools_data, indent=2))
            log_event("progress", {"message": "Tools listed", "percentage": 25})

            # Get London weather
            response = client.post(
                f"{mcp_url}/tools/call",
                json={"tool": "get_weather", "arguments": {"location": "London"}},
            )
            london_weather = response.json().get("result", {})
            write_file("london_weather.json", json.dumps(london_weather, indent=2))
            log_event(
                "progress", {"message": "London weather fetched", "percentage": 50}
            )

            # Get Tokyo weather
            response = client.post(
                f"{mcp_url}/tools/call",
                json={"tool": "get_weather", "arguments": {"location": "Tokyo"}},
            )
            tokyo_weather = response.json().get("result", {})
            write_file("tokyo_weather.json", json.dumps(tokyo_weather, indent=2))
            log_event(
                "progress", {"message": "Tokyo weather fetched", "percentage": 75}
            )

            # Create comparison report
            comparison = f"""# Weather Comparison Report

## London
- **Temperature**: {london_weather.get("temperature", "N/A")}°C
- **Conditions**: {london_weather.get("conditions", "N/A")}
- **Humidity**: {london_weather.get("humidity", "N/A")}%

## Tokyo
- **Temperature**: {tokyo_weather.get("temperature", "N/A")}°C
- **Conditions**: {tokyo_weather.get("conditions", "N/A")}
- **Humidity**: {tokyo_weather.get("humidity", "N/A")}%

## Summary
London: {london_weather.get("conditions", "unknown").lower()}, \
{london_weather.get("temperature", 0)}°C.
Tokyo: {tokyo_weather.get("conditions", "unknown").lower()}, \
{tokyo_weather.get("temperature", 0)}°C.
"""
            write_file("comparison.md", comparison)
            log_event(
                "progress", {"message": "Comparison report created", "percentage": 100}
            )

        else:
            # Default: just list tools
            write_file("tools_list.json", json.dumps(tools_data, indent=2))

        print(
            json.dumps(
                {
                    "version": "1.0",
                    "task_id": task_id,
                    "status": "completed",
                    "artifacts": artifacts,
                    "metrics": {
                        "total_steps": sequence,
                        "total_tokens": 0,
                    },
                },
                indent=2,
            )
        )

    except httpx.RequestError as e:
        log_event("error", {"message": f"MCP connection error: {e}"})
        print(
            json.dumps(
                {
                    "version": "1.0",
                    "task_id": task_id,
                    "status": "error",
                    "error": f"Failed to connect to MCP server: {e}",
                    "artifacts": artifacts,
                }
            )
        )
        sys.exit(1)

    except Exception as e:
        log_event("error", {"message": str(e)})
        print(
            json.dumps(
                {
                    "version": "1.0",
                    "task_id": task_id,
                    "status": "error",
                    "error": str(e),
                    "artifacts": artifacts,
                }
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MCP-capable agent for ATP testing.

This agent connects to an MCP server and uses its tools to complete tasks.
It follows the ATP Protocol: reads JSON from stdin, writes JSON to stdout.

Usage:
    echo '{"task": {"description": "..."}}' | python mcp_agent.py

Environment:
    OPENAI_API_KEY     - OpenAI API key (required)
    MCP_SERVER_URL     - MCP server URL (default: http://localhost:9000)
    OPENAI_MODEL       - Model to use (default: gpt-4o-mini)
"""

import json
import os
import sys
from datetime import datetime

import httpx

# Check for OpenAI
try:
    from openai import OpenAI
except ImportError:
    print(
        json.dumps(
            {
                "status": "error",
                "error": "openai package not installed. Run: uv add openai",
            }
        )
    )
    sys.exit(1)


class MCPAgent:
    """Agent that interacts with MCP servers using OpenAI."""

    def __init__(
        self,
        mcp_url: str = "http://localhost:9000",
        model: str = "gpt-4o-mini",
    ):
        self.mcp_url = mcp_url
        self.model = model
        self.client = OpenAI()
        self.http_client = httpx.Client(timeout=30)
        self.artifacts: list[str] = []
        self.events: list[dict] = []
        self.sequence = 0

    def log_event(
        self,
        event_type: str,
        payload: dict,
    ) -> None:
        """Log an event for ATP tracking."""
        self.sequence += 1
        self.events.append(
            {
                "sequence": self.sequence,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "event_type": event_type,
                "payload": payload,
            }
        )

    def get_mcp_tools(self) -> list[dict]:
        """Fetch available tools from MCP server."""
        try:
            response = self.http_client.post(f"{self.mcp_url}/tools/list", json={})
            response.raise_for_status()
            data = response.json()
            return data.get("tools", [])
        except Exception as e:
            self.log_event("error", {"message": f"Failed to fetch MCP tools: {e}"})
            return []

    def call_mcp_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool on the MCP server."""
        try:
            response = self.http_client.post(
                f"{self.mcp_url}/tools/call",
                json={"tool": tool_name, "arguments": arguments},
            )
            response.raise_for_status()
            return response.json().get("result", {})
        except Exception as e:
            return {"error": str(e)}

    def write_file(self, path: str, content: str) -> dict:
        """Write content to a file."""
        try:
            with open(path, "w") as f:
                f.write(content)
            self.artifacts.append(path)
            return {"success": True, "path": path, "size": len(content)}
        except Exception as e:
            return {"error": str(e)}

    def build_openai_tools(self, mcp_tools: list[dict]) -> list[dict]:
        """Build OpenAI function definitions from MCP tools."""
        tools = []

        # Add MCP tools
        for mcp_tool in mcp_tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"mcp_{mcp_tool['name']}",
                        "description": mcp_tool.get("description", ""),
                        "parameters": mcp_tool.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    },
                }
            )

        # Add file writing capability
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": (
                        "Write content to a file. "
                        "Use this to save results and create output files."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path to write to",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        )

        # Add MCP tools listing capability
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "list_mcp_tools",
                    "description": (
                        "List all available tools on the MCP server "
                        "with their descriptions"
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )

        return tools

    def execute_tool(self, name: str, arguments: dict) -> dict:
        """Execute a tool call."""
        start_time = datetime.utcnow()

        if name == "write_file":
            result = self.write_file(arguments["path"], arguments["content"])
        elif name == "list_mcp_tools":
            tools = self.get_mcp_tools()
            result = {"tools": tools}
        elif name.startswith("mcp_"):
            mcp_tool_name = name[4:]  # Remove 'mcp_' prefix
            result = self.call_mcp_tool(mcp_tool_name, arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}

        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        self.log_event(
            "tool_call",
            {
                "tool": name,
                "arguments": arguments,
                "result": result,
                "status": "error" if "error" in result else "success",
                "duration_ms": duration_ms,
            },
        )

        return result

    def run(self, task_description: str, max_steps: int = 20) -> dict:
        """Run the agent to complete a task."""
        # Fetch MCP tools
        self.log_event("progress", {"message": "Fetching MCP tools", "percentage": 5})
        mcp_tools = self.get_mcp_tools()

        if not mcp_tools:
            return {
                "status": "error",
                "error": "Failed to connect to MCP server or no tools available",
                "artifacts": [],
                "events": self.events,
            }

        # Build OpenAI tools
        openai_tools = self.build_openai_tools(mcp_tools)

        # System prompt
        system_prompt = """You are an AI agent that completes tasks using MCP tools.

Available capabilities:
- Use MCP tools (prefixed with mcp_) to interact with external services
- Use write_file to save results to files
- Use list_mcp_tools to see available tools

Guidelines:
1. First understand what tools are available using list_mcp_tools if needed
2. Use the appropriate MCP tools to gather information
3. Save results to files as requested in the task
4. Ensure output files contain valid JSON when JSON is expected
5. Complete all required artifacts before finishing

Always save your final results to the files specified in the task."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_description},
        ]

        self.log_event(
            "progress", {"message": "Starting task execution", "percentage": 10}
        )

        total_tokens = 0
        llm_calls = 0
        tool_calls_count = 0

        for step in range(max_steps):
            # Call OpenAI
            start_time = datetime.utcnow()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )
            except Exception as e:
                self.log_event("error", {"message": f"OpenAI API error: {e}"})
                return {
                    "status": "error",
                    "error": str(e),
                    "artifacts": self.artifacts,
                    "events": self.events,
                }

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            llm_calls += 1

            # Track tokens
            usage = response.usage
            if usage:
                total_tokens += usage.total_tokens
                self.log_event(
                    "llm_request",
                    {
                        "model": self.model,
                        "input_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens,
                        "duration_ms": duration_ms,
                    },
                )

            message = response.choices[0].message

            # Check if we need to call tools
            if message.tool_calls:
                messages.append(message)

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    tool_calls_count += 1
                    result = self.execute_tool(tool_name, tool_args)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )

                # Update progress
                progress = min(90, 10 + (step + 1) * 80 // max_steps)
                self.log_event(
                    "progress",
                    {
                        "message": f"Step {step + 1} completed",
                        "percentage": progress,
                    },
                )
            else:
                # No more tool calls - task complete
                self.log_event(
                    "progress", {"message": "Task completed", "percentage": 100}
                )

                return {
                    "status": "completed",
                    "message": message.content or "Task completed successfully",
                    "artifacts": self.artifacts,
                    "events": self.events,
                    "metrics": {
                        "total_tokens": total_tokens,
                        "llm_calls": llm_calls,
                        "tool_calls": tool_calls_count,
                        "steps": step + 1,
                    },
                }

        # Max steps reached
        self.log_event("error", {"message": "Max steps reached"})
        return {
            "status": "error",
            "error": "Max steps reached without completing the task",
            "artifacts": self.artifacts,
            "events": self.events,
            "metrics": {
                "total_tokens": total_tokens,
                "llm_calls": llm_calls,
                "tool_calls": tool_calls_count,
                "steps": max_steps,
            },
        }


def main() -> None:
    """Main entry point."""
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "OPENAI_API_KEY environment variable not set",
                }
            )
        )
        sys.exit(1)

    # Read ATP request from stdin
    try:
        input_data = sys.stdin.read()
        request = json.loads(input_data) if input_data.strip() else {}
    except json.JSONDecodeError as e:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"Invalid JSON input: {e}",
                }
            )
        )
        sys.exit(1)

    # Extract task description
    task = request.get("task", {})
    task_description = task.get("description", "")

    if not task_description:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "No task description provided",
                }
            )
        )
        sys.exit(1)

    # Get configuration from environment
    mcp_url = os.environ.get("MCP_SERVER_URL", "http://localhost:9000")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # Extract constraints
    constraints = request.get("constraints", {})
    max_steps = constraints.get("max_steps", 20)

    # Create and run agent
    agent = MCPAgent(mcp_url=mcp_url, model=model)

    # Include any input data in the task description
    input_data_dict = task.get("input_data", {})
    if input_data_dict:
        task_description += (
            f"\n\nAdditional input data:\n{json.dumps(input_data_dict, indent=2)}"
        )

    result = agent.run(task_description, max_steps=max_steps)

    # Output ATP response
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

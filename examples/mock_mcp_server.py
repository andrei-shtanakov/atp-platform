#!/usr/bin/env python3
"""Mock MCP server for testing agents.

This server simulates an MCP (Model Context Protocol) server that provides
tools for agents to interact with. It's designed for testing purposes.

Usage:
    python examples/mock_mcp_server.py [--port PORT]

Endpoints:
    POST /tools/list    - List available tools
    POST /tools/call    - Call a specific tool
    GET  /health        - Health check
"""

import argparse
import json
import logging
import random
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class MockMCPHandler(BaseHTTPRequestHandler):
    """Handle MCP-like requests."""

    # Available tools with their schemas
    TOOLS = {
        "get_weather": {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location",
                    },
                },
                "required": ["location"],
            },
        },
        "search_web": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
        "read_file": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file",
                    },
                },
                "required": ["path"],
            },
        },
        "calculate": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    },
                },
                "required": ["expression"],
            },
        },
    }

    # Mock weather data for various cities
    WEATHER_DATA = {
        "moscow": {"temperature": 5, "conditions": "Cloudy", "humidity": 75},
        "london": {"temperature": 12, "conditions": "Rainy", "humidity": 85},
        "tokyo": {"temperature": 18, "conditions": "Sunny", "humidity": 60},
        "new york": {"temperature": 8, "conditions": "Partly Cloudy", "humidity": 55},
        "paris": {"temperature": 14, "conditions": "Overcast", "humidity": 70},
        "default": {"temperature": 20, "conditions": "Clear", "humidity": 50},
    }

    # Mock search results
    SEARCH_RESULTS = {
        "python": [
            {
                "title": "Python Official Documentation",
                "url": "https://docs.python.org",
                "snippet": "Official Python programming language documentation.",
            },
            {
                "title": "Python Tutorial - W3Schools",
                "url": "https://www.w3schools.com/python/",
                "snippet": "Learn Python programming with tutorials and examples.",
            },
            {
                "title": "Real Python Tutorials",
                "url": "https://realpython.com",
                "snippet": "Python tutorials for developers of all skill levels.",
            },
        ],
        "async": [
            {
                "title": "Async IO in Python",
                "url": "https://docs.python.org/3/library/asyncio.html",
                "snippet": "asyncio is a library to write concurrent code.",
            },
            {
                "title": "Understanding Async Programming",
                "url": "https://example.com/async",
                "snippet": "A guide to asynchronous programming concepts.",
            },
        ],
        "default": [
            {
                "title": "Search Result 1",
                "url": "https://example.com/1",
                "snippet": "Generic search result.",
            },
            {
                "title": "Search Result 2",
                "url": "https://example.com/2",
                "snippet": "Another search result.",
            },
        ],
    }

    def log_message(self, format: str, *args) -> None:
        """Log HTTP requests."""
        logger.info("%s - %s", self.address_string(), format % args)

    def send_json_response(self, data: dict, status: int = 200) -> None:
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health":
            self.send_json_response({"status": "healthy", "server": "mock-mcp"})
        else:
            self.send_json_response({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        """Handle POST requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode() if content_length > 0 else "{}"

        try:
            request = json.loads(body)
        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status=400)
            return

        logger.info("Request to %s: %s", self.path, json.dumps(request)[:200])

        if self.path == "/tools/list":
            self._handle_list_tools()
        elif self.path == "/tools/call":
            self._handle_call_tool(request)
        else:
            self.send_json_response(
                {"error": f"Unknown endpoint: {self.path}"}, status=404
            )

    def _handle_list_tools(self) -> None:
        """Handle tool listing request."""
        tools_list = list(self.TOOLS.values())
        self.send_json_response({"tools": tools_list})

    def _handle_call_tool(self, request: dict) -> None:
        """Handle tool call request."""
        tool_name = request.get("tool") or request.get("name")
        arguments = request.get("arguments", {})

        if not tool_name:
            self.send_json_response(
                {"error": "Missing 'tool' field in request"},
                status=400,
            )
            return

        if tool_name not in self.TOOLS:
            self.send_json_response(
                {"error": f"Unknown tool: {tool_name}"},
                status=404,
            )
            return

        # Execute the tool
        result = self._execute_tool(tool_name, arguments)
        self.send_json_response({"result": result})

    def _execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool and return the result."""
        if tool_name == "get_weather":
            return self._get_weather(arguments)
        elif tool_name == "search_web":
            return self._search_web(arguments)
        elif tool_name == "read_file":
            return self._read_file(arguments)
        elif tool_name == "calculate":
            return self._calculate(arguments)
        else:
            return {"error": f"Tool '{tool_name}' not implemented"}

    def _get_weather(self, arguments: dict) -> dict:
        """Get mock weather data."""
        location = arguments.get("location", "").lower()
        weather = self.WEATHER_DATA.get(location, self.WEATHER_DATA["default"])

        # Add some randomness to make it more realistic
        temp_variation = random.randint(-2, 2)

        return {
            "location": arguments.get("location", "Unknown"),
            "temperature": weather["temperature"] + temp_variation,
            "temperature_unit": "celsius",
            "conditions": weather["conditions"],
            "humidity": weather["humidity"],
            "wind_speed": random.randint(5, 25),
            "wind_unit": "km/h",
        }

    def _search_web(self, arguments: dict) -> dict:
        """Get mock search results."""
        query = arguments.get("query", "").lower()
        max_results = arguments.get("max_results", 5)

        # Find matching results
        results = []
        for keyword, keyword_results in self.SEARCH_RESULTS.items():
            if keyword in query:
                results.extend(keyword_results)

        if not results:
            results = self.SEARCH_RESULTS["default"]

        return {
            "query": arguments.get("query", ""),
            "total_results": len(results),
            "results": results[:max_results],
        }

    def _read_file(self, arguments: dict) -> dict:
        """Mock file reading."""
        path = arguments.get("path", "")

        # Return mock content for demonstration
        return {
            "path": path,
            "content": f"Mock content for file: {path}\n\nSimulated content.",
            "size": 100,
            "exists": True,
        }

    def _calculate(self, arguments: dict) -> dict:
        """Evaluate mathematical expression."""
        expression = arguments.get("expression", "")

        try:
            # Safe evaluation of basic math
            allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return {
                "expression": expression,
                "result": result,
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
            }


def main() -> None:
    """Run the mock MCP server."""
    parser = argparse.ArgumentParser(description="Mock MCP Server for testing")
    parser.add_argument("--port", type=int, default=9000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), MockMCPHandler)
    logger.info("Mock MCP server starting on http://%s:%d", args.host, args.port)
    logger.info("Available endpoints:")
    logger.info("  POST /tools/list  - List available tools")
    logger.info("  POST /tools/call  - Call a tool")
    logger.info("  GET  /health      - Health check")
    logger.info("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()

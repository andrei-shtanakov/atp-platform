#!/usr/bin/env python3
"""Web Search Agent for ATP testing.

Searches websites for products and extracts structured data.

Protocol:
- Input: ATPRequest as JSON from stdin
- Output: ATPResponse as JSON to stdout
- Events: ATPEvent as JSONL to stderr

Environment:
- TEST_SITE_URL: Base URL of the test site (default: http://localhost:9876)
"""

import json
import os
import re
import sys
from datetime import UTC, datetime
from typing import Any

import httpx
from bs4 import BeautifulSoup


def get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def emit_event(
    task_id: str,
    sequence: int,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """Emit an ATP event to stderr."""
    event = {
        "version": "1.0",
        "task_id": task_id,
        "timestamp": get_timestamp(),
        "sequence": sequence,
        "event_type": event_type,
        "payload": payload,
    }
    print(json.dumps(event), file=sys.stderr, flush=True)


def extract_price(price_text: str) -> float | None:
    """Extract numeric price from text like '$499.99' or '$ 1,299.00'."""
    match = re.search(r"\$?\s*([\d,]+\.?\d*)", price_text)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def parse_task_description(description: str) -> dict[str, Any]:
    """Parse task description to extract parameters."""
    params: dict[str, Any] = {
        "action": "search",
        "category": None,
        "max_price": None,
        "min_price": None,
        "search_query": None,
        "sort_by": None,
        "sort_order": "asc",
    }

    desc_lower = description.lower()

    # Extract category
    if "laptop" in desc_lower:
        params["category"] = "laptop"
    elif "accessory" in desc_lower or "accessories" in desc_lower:
        params["category"] = "accessory"
    elif "display" in desc_lower or "monitor" in desc_lower:
        params["category"] = "display"

    # Extract max price
    max_price_match = re.search(
        r"(?:under|below|less than|max|maximum|<)\s*\$?([\d,]+)", desc_lower
    )
    if max_price_match:
        params["max_price"] = float(max_price_match.group(1).replace(",", ""))

    # Extract min price
    min_price_match = re.search(
        r"(?:over|above|more than|min|minimum|>)\s*\$?([\d,]+)", desc_lower
    )
    if min_price_match:
        params["min_price"] = float(min_price_match.group(1).replace(",", ""))

    # Extract sort order
    if "descending" in desc_lower or "highest" in desc_lower:
        params["sort_order"] = "desc"
    if "price" in desc_lower and ("sort" in desc_lower or "order" in desc_lower):
        params["sort_by"] = "price"

    # Check for specific actions
    if "company" in desc_lower or "about" in desc_lower:
        if "founded" in desc_lower or "year" in desc_lower:
            params["action"] = "get_company_info"
    if "contact" in desc_lower or "email" in desc_lower:
        params["action"] = "get_contact_info"

    return params


class WebSearchAgent:
    """Agent that searches websites and extracts product data."""

    def __init__(self, base_url: str, task_id: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.task_id = task_id
        self.sequence = 0
        self.steps = 0
        self.client = httpx.Client(timeout=30)

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit an event and increment sequence."""
        emit_event(self.task_id, self.sequence, event_type, payload)
        self.sequence += 1

    def fetch_page(self, path: str) -> BeautifulSoup | None:
        """Fetch and parse a page."""
        url = f"{self.base_url}{path}"
        self.steps += 1

        self.emit(
            "tool_call",
            {
                "tool": "http_get",
                "input": {"url": url},
                "status": "started",
            },
        )

        try:
            response = self.client.get(url)
            response.raise_for_status()

            self.emit(
                "tool_call",
                {
                    "tool": "http_get",
                    "input": {"url": url},
                    "status": "completed",
                    "output": {
                        "status_code": response.status_code,
                        "size": len(response.text),
                    },
                },
            )

            return BeautifulSoup(response.text, "html.parser")

        except httpx.HTTPError as e:
            self.emit(
                "error",
                {
                    "type": "http_error",
                    "message": str(e),
                    "recoverable": False,
                },
            )
            return None

    def fetch_json(self, path: str) -> Any | None:
        """Fetch JSON data from API."""
        url = f"{self.base_url}{path}"
        self.steps += 1

        self.emit(
            "tool_call",
            {
                "tool": "http_get_json",
                "input": {"url": url},
                "status": "started",
            },
        )

        try:
            response = self.client.get(url)
            response.raise_for_status()
            data = response.json()

            self.emit(
                "tool_call",
                {
                    "tool": "http_get_json",
                    "input": {"url": url},
                    "status": "completed",
                    "output": {"status_code": response.status_code},
                },
            )

            return data

        except httpx.HTTPError as e:
            self.emit(
                "error",
                {
                    "type": "http_error",
                    "message": str(e),
                    "recoverable": False,
                },
            )
            return None

    def search_products(
        self,
        category: str | None = None,
        max_price: float | None = None,
        min_price: float | None = None,
        sort_by: str | None = None,
        sort_order: str = "asc",
    ) -> list[dict[str, Any]]:
        """Search for products using the API."""
        self.emit("progress", {"message": "Starting product search", "percentage": 0})

        # Build query params
        params = []
        if category:
            params.append(f"category={category}")
        if max_price is not None:
            params.append(f"max_price={max_price}")
        if min_price is not None:
            params.append(f"min_price={min_price}")

        query_string = "&".join(params)
        path = f"/api/products{'?' + query_string if query_string else ''}"

        self.emit(
            "reasoning",
            {
                "thought": f"Fetching products from API: {path}",
            },
        )

        products = self.fetch_json(path)

        if products is None:
            return []

        self.emit(
            "progress",
            {
                "message": f"Found {len(products)} products",
                "percentage": 50,
            },
        )

        # Sort if requested
        if sort_by == "price":
            reverse = sort_order == "desc"
            products.sort(key=lambda x: x.get("price", 0), reverse=reverse)
            order_str = "descending" if reverse else "ascending"
            self.emit(
                "reasoning",
                {"thought": f"Sorted {len(products)} products by price {order_str}"},
            )

        # Transform to output format
        results = []
        for p in products:
            results.append(
                {
                    "name": p["name"],
                    "price": p["price"],
                    "url": f"/product/{p['id']}",
                }
            )

        self.emit("progress", {"message": "Search completed", "percentage": 100})

        return results

    def scrape_products_html(
        self,
        category: str | None = None,
        max_price: float | None = None,
    ) -> list[dict[str, Any]]:
        """Scrape products from HTML catalog page (alternative method)."""
        self.emit("progress", {"message": "Scraping catalog page", "percentage": 0})

        # Build URL
        params = []
        if category:
            params.append(f"category={category}")
        if max_price is not None:
            params.append(f"max_price={max_price}")

        query_string = "&".join(params)
        path = f"/catalog{'?' + query_string if query_string else ''}"

        soup = self.fetch_page(path)
        if soup is None:
            return []

        self.emit("progress", {"message": "Parsing HTML", "percentage": 30})

        products = []
        cards = soup.select(".product-card")

        for card in cards:
            name_el = card.select_one(".product-name")
            price_el = card.select_one(".price")
            link_el = card.select_one("a[href]")

            if name_el and price_el:
                name = name_el.get_text(strip=True)
                price = extract_price(price_el.get_text(strip=True))
                url = link_el["href"] if link_el else None

                if price is not None:
                    products.append(
                        {
                            "name": name,
                            "price": price,
                            "url": url,
                        }
                    )

                    self.emit(
                        "reasoning",
                        {
                            "thought": f"Found: {name} at ${price}",
                        },
                    )

        self.emit(
            "progress",
            {
                "message": f"Scraped {len(products)} products",
                "percentage": 100,
            },
        )

        return products

    def get_company_info(self) -> dict[str, Any]:
        """Get company information from about page."""
        self.emit("progress", {"message": "Fetching company info", "percentage": 0})

        # Try API first
        data = self.fetch_json("/api/company")
        if data:
            self.emit("progress", {"message": "Got company info", "percentage": 100})
            return data

        # Fallback to scraping
        soup = self.fetch_page("/about")
        if soup is None:
            return {}

        text = soup.get_text()
        info = {}

        # Extract founding year
        year_match = re.search(r"founded.*?(\d{4})", text, re.IGNORECASE)
        if year_match:
            info["founded"] = int(year_match.group(1))

        self.emit("progress", {"message": "Got company info", "percentage": 100})
        return info

    def get_contact_info(self) -> dict[str, Any]:
        """Get contact information."""
        self.emit("progress", {"message": "Fetching contact info", "percentage": 0})

        # Try API first
        data = self.fetch_json("/api/company")
        if data:
            self.emit("progress", {"message": "Got contact info", "percentage": 100})
            return {
                "email": data.get("email"),
                "phone": data.get("phone"),
                "address": data.get("address"),
            }

        # Fallback to scraping
        soup = self.fetch_page("/contact")
        if soup is None:
            return {}

        text = soup.get_text()
        info = {}

        # Extract email
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
        if email_match:
            info["email"] = email_match.group(0)

        self.emit("progress", {"message": "Got contact info", "percentage": 100})
        return info

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()


def main() -> None:
    """Run the web search agent."""
    start_time = datetime.now(UTC)

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
                    "status": "failed",
                    "error": f"Invalid JSON input: {e}",
                    "artifacts": [],
                    "metrics": {},
                }
            )
        )
        sys.exit(1)

    task_id = request.get("task_id", "search-task")
    task = request.get("task", {}) or {}
    description = task.get("description", "")
    context = request.get("context", {}) or {}

    # Get base URL from context or environment
    context_env = context.get("environment", {}) or {}
    base_url = (
        context_env.get("TEST_SITE_URL")
        or os.environ.get("TEST_SITE_URL")
        or "http://localhost:9876"
    )

    # Parse task description
    params = parse_task_description(description)

    # Create agent
    agent = WebSearchAgent(base_url, task_id)

    try:
        artifacts = []
        error = None

        if params["action"] == "get_company_info":
            # Get company information
            result = agent.get_company_info()
            artifacts.append(
                {
                    "type": "structured",
                    "name": "company_info.json",
                    "data": result,
                    "content_type": "application/json",
                }
            )

        elif params["action"] == "get_contact_info":
            # Get contact information
            result = agent.get_contact_info()
            artifacts.append(
                {
                    "type": "structured",
                    "name": "contact_info.json",
                    "data": result,
                    "content_type": "application/json",
                }
            )

        else:
            # Search products
            products = agent.search_products(
                category=params["category"],
                max_price=params["max_price"],
                min_price=params["min_price"],
                sort_by=params.get("sort_by", "price"),
                sort_order=params.get("sort_order", "asc"),
            )

            # Determine artifact name from expected_artifacts or default
            expected = task.get("expected_artifacts", []) or []
            artifact_name = expected[0] if expected else "results.json"

            # Use file artifact with JSON content (ArtifactStructured.data must be dict)
            artifacts.append(
                {
                    "type": "file",
                    "path": artifact_name,
                    "content": json.dumps(products, indent=2),
                    "content_type": "application/json",
                }
            )

        status = "completed" if artifacts else "partial"

    except Exception as e:
        status = "failed"
        error = str(e)
        artifacts = []

    finally:
        agent.close()

    # Calculate metrics
    duration = (datetime.now(UTC) - start_time).total_seconds()

    # Build response
    response = {
        "version": "1.0",
        "task_id": task_id,
        "status": status,
        "artifacts": artifacts,
        "metrics": {
            "total_steps": agent.steps,
            "wall_time_seconds": duration,
        },
        "error": error,
    }

    print(json.dumps(response))


if __name__ == "__main__":
    main()

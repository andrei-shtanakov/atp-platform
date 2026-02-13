"""LangSmith trace importer.

Fetches runs from the LangSmith API and converts them to
TraceRecord objects for ATP test generation.
"""

from __future__ import annotations

from typing import Any

import httpx

from atp.generator.trace_import import TraceImporter, TraceRecord


class LangSmithImporter(TraceImporter):
    """Import traces from LangSmith.

    Args:
        api_key: LangSmith API key.
        project: LangSmith project name or ID.
        base_url: API base URL (default: LangSmith cloud).
    """

    DEFAULT_BASE_URL = "https://api.smith.langchain.com"

    def __init__(
        self,
        *,
        api_key: str = "",
        project: str = "",
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._project = project
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        """Importer name."""
        return "langsmith"

    async def fetch_traces(
        self,
        *,
        limit: int = 50,
        **kwargs: Any,
    ) -> list[TraceRecord]:
        """Fetch runs from LangSmith and convert to TraceRecords.

        Args:
            limit: Maximum number of runs to fetch.
            **kwargs: Additional query params forwarded to the API.

        Returns:
            List of TraceRecord objects.
        """
        runs = await self._fetch_runs(limit=limit, **kwargs)
        return [self._run_to_record(r) for r in runs]

    async def _fetch_runs(
        self,
        *,
        limit: int = 50,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Fetch raw run data from LangSmith API.

        Args:
            limit: Max runs.
            **kwargs: Extra query params.

        Returns:
            List of run dicts from the API.
        """
        headers = {"x-api-key": self._api_key}
        params: dict[str, Any] = {
            "limit": limit,
        }
        if self._project:
            params["project_name"] = self._project
        params.update(kwargs)

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._base_url}/api/v1/runs",
                headers=headers,
                params=params,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

        if isinstance(data, list):
            return data[:limit]
        if isinstance(data, dict) and "runs" in data:
            return data["runs"][:limit]
        return []

    @staticmethod
    def _run_to_record(run: dict[str, Any]) -> TraceRecord:
        """Convert a LangSmith run dict to a TraceRecord."""
        run_id = str(run.get("id", run.get("run_id", "")))

        # Extract input text
        inputs = run.get("inputs", {})
        if isinstance(inputs, dict):
            raw_input = inputs.get("input") or inputs.get("question")
            input_text = str(raw_input) if raw_input else str(inputs)
        else:
            input_text = str(inputs)

        # Extract output text
        outputs = run.get("outputs", {})
        if isinstance(outputs, dict):
            raw_output = outputs.get("output") or outputs.get("answer")
            output_text = str(raw_output) if raw_output else str(outputs)
        else:
            output_text = str(outputs)

        # Duration
        duration_ms: float | None = None
        if run.get("total_time_ms"):
            duration_ms = float(run["total_time_ms"])
        elif run.get("latency"):
            duration_ms = float(run["latency"]) * 1000

        # Status mapping
        status_raw = run.get("status", "completed")
        status = "completed" if status_raw == "success" else str(status_raw)

        # Tool calls
        tool_calls: list[str] = []
        for child in run.get("child_runs", []):
            if child.get("run_type") == "tool":
                tool_name = child.get("name", "")
                if tool_name:
                    tool_calls.append(tool_name)

        # Tags from run
        tags: list[str] = list(run.get("tags", []))

        metadata: dict[str, Any] = {}
        if run.get("extra"):
            metadata["extra"] = run["extra"]

        return TraceRecord(
            trace_id=run_id,
            input_text=input_text,
            output_text=output_text,
            metadata=metadata,
            tags=tags,
            duration_ms=duration_ms,
            status=status,
            tool_calls=tool_calls,
        )

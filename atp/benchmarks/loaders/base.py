"""Base class for benchmark loaders."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import httpx
import yaml

from atp.loader.models import TestSuite

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".atp" / "benchmarks"


class BenchmarkLoader(ABC):
    """Base class for loading standard benchmarks as ATP test suites.

    Subclasses implement fetching and converting specific benchmarks
    (HumanEval, SWE-bench, MMLU, etc.) into ATP TestSuite format.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the loader.

        Args:
            cache_dir: Directory for caching downloaded data.
                Defaults to ~/.atp/benchmarks/.
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable benchmark name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of the benchmark."""

    @property
    @abstractmethod
    def source_url(self) -> str:
        """URL to fetch the benchmark data from."""

    def _cache_path(self) -> Path:
        """Get the cache file path for this benchmark."""
        return self.cache_dir / f"{self.name}.json"

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_from_cache(self) -> list[dict] | None:
        """Load benchmark data from the local cache.

        Returns:
            Cached data as list of dicts, or None if not cached.
        """
        cache_file = self._cache_path()
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text())
            if isinstance(data, list):
                return data
            return None
        except (json.JSONDecodeError, OSError):
            return None

    def _save_to_cache(self, data: list[dict]) -> None:
        """Save benchmark data to the local cache.

        Args:
            data: Data to cache.
        """
        self._ensure_cache_dir()
        self._cache_path().write_text(json.dumps(data))

    def download(self) -> list[dict]:
        """Download benchmark data, using cache if available.

        Returns:
            List of benchmark items as dicts.
        """
        cached = self._load_from_cache()
        if cached is not None:
            logger.info(
                "Loaded %s from cache: %s",
                self.name,
                self._cache_path(),
            )
            return cached

        logger.info("Downloading %s from %s", self.name, self.source_url)
        data = self._fetch_data()
        self._save_to_cache(data)
        return data

    def _fetch_data(self) -> list[dict]:
        """Fetch raw benchmark data from the source.

        Returns:
            List of benchmark items as dicts.
        """
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(self.source_url)
            response.raise_for_status()
            return self._parse_response(response)

    def _parse_response(self, response: httpx.Response) -> list[dict]:
        """Parse HTTP response into a list of benchmark items.

        Override this for non-JSONL sources.

        Args:
            response: HTTP response.

        Returns:
            List of benchmark items as dicts.
        """
        items: list[dict] = []
        for line in response.text.strip().splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
        return items

    @abstractmethod
    def _convert_items(self, items: list[dict], limit: int | None = None) -> TestSuite:
        """Convert raw benchmark items to an ATP TestSuite.

        Args:
            items: Raw benchmark items.
            limit: Maximum number of items to include.

        Returns:
            ATP TestSuite.
        """

    def load(self, limit: int | None = None) -> TestSuite:
        """Download and convert benchmark to an ATP TestSuite.

        Args:
            limit: Maximum number of items to include.

        Returns:
            ATP TestSuite with benchmark items as test cases.
        """
        items = self.download()
        return self._convert_items(items, limit=limit)

    def export_yaml(self, output: Path, limit: int | None = None) -> Path:
        """Export benchmark as a YAML test suite file.

        Args:
            output: Output file path.
            limit: Maximum number of items to include.

        Returns:
            Path to the written file.
        """
        suite = self.load(limit=limit)
        data = suite.model_dump(exclude_none=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.dump(data, sort_keys=False))
        return output

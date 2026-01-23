"""Caching utilities for ATP performance optimization.

Provides caching for:
- Parsed test suites (file-based with modification time tracking)
- Adapter instances (by configuration hash)
- Variable substitution results
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atp.adapters.base import AdapterConfig, AgentAdapter
from atp.adapters.registry import get_registry
from atp.loader.loader import TestLoader
from atp.loader.models import TestSuite

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry[T]:
    """Entry in a cache with metadata."""

    value: T
    created_at: float
    accessed_at: float
    access_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Statistics for a cache."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate,
        }


class TestSuiteCache:
    """
    Cache for parsed test suites.

    Caches TestSuite objects by file path, with automatic invalidation
    when the file is modified. Uses file modification time and size
    for cache invalidation.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float | None = None) -> None:
        """
        Initialize the test suite cache.

        Args:
            max_size: Maximum number of entries to cache.
            ttl_seconds: Optional time-to-live in seconds.
                        If None, entries never expire (only invalidate on change).
        """
        self._cache: dict[str, CacheEntry[TestSuite]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._stats = CacheStats(max_size=max_size)
        self._lock = threading.RLock()

    @staticmethod
    def _get_file_key(file_path: Path) -> tuple[float, int]:
        """Get cache key components from file (mtime, size)."""
        stat = file_path.stat()
        return (stat.st_mtime, stat.st_size)

    def get(self, file_path: str | Path) -> TestSuite | None:
        """
        Get a cached test suite if valid.

        Args:
            file_path: Path to the test suite file.

        Returns:
            Cached TestSuite if valid, None if cache miss or invalid.
        """
        file_path = Path(file_path).resolve()
        cache_key = str(file_path)

        with self._lock:
            entry = self._cache.get(cache_key)

            if entry is None:
                self._stats.misses += 1
                return None

            # Check TTL
            if self._ttl is not None:
                age = time.time() - entry.created_at
                if age > self._ttl:
                    del self._cache[cache_key]
                    self._stats.misses += 1
                    self._stats.evictions += 1
                    self._stats.size -= 1
                    return None

            # Check file modification
            try:
                current_key = self._get_file_key(file_path)
                cached_key = entry.metadata.get("file_key")

                if current_key != cached_key:
                    del self._cache[cache_key]
                    self._stats.misses += 1
                    self._stats.evictions += 1
                    self._stats.size -= 1
                    logger.debug("Cache invalidated for %s (file changed)", file_path)
                    return None
            except OSError:
                # File doesn't exist or can't be stat'd
                del self._cache[cache_key]
                self._stats.misses += 1
                self._stats.evictions += 1
                self._stats.size -= 1
                return None

            entry.touch()
            self._stats.hits += 1
            logger.debug("Cache hit for %s", file_path)
            return entry.value

    def put(self, file_path: str | Path, suite: TestSuite) -> None:
        """
        Cache a test suite.

        Args:
            file_path: Path to the test suite file.
            suite: Parsed TestSuite object.
        """
        file_path = Path(file_path).resolve()
        cache_key = str(file_path)

        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size and cache_key not in self._cache:
                self._evict_one()

            try:
                file_key = self._get_file_key(file_path)
            except OSError:
                # Can't get file info, don't cache
                return

            now = time.time()
            entry = CacheEntry(
                value=suite,
                created_at=now,
                accessed_at=now,
                metadata={"file_key": file_key, "file_path": str(file_path)},
            )
            self._cache[cache_key] = entry
            self._stats.size = len(self._cache)
            logger.debug("Cached test suite for %s", file_path)

    def _evict_one(self) -> None:
        """Evict the least recently accessed entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].accessed_at)
        del self._cache[lru_key]
        self._stats.evictions += 1
        self._stats.size = len(self._cache)

    def invalidate(self, file_path: str | Path) -> bool:
        """
        Invalidate a cached entry.

        Args:
            file_path: Path to invalidate.

        Returns:
            True if entry was removed, False if not in cache.
        """
        file_path = Path(file_path).resolve()
        cache_key = str(file_path)

        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self._stats.evictions += 1
                self._stats.size = len(self._cache)
                return True
            return False

    def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.size = 0
            return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return self._stats


class CachedTestLoader(TestLoader):
    """
    Test loader with caching support.

    Extends TestLoader to cache parsed test suites, avoiding
    repeated parsing of unchanged files.
    """

    _shared_cache: TestSuiteCache | None = None

    def __init__(
        self,
        env: dict[str, str] | None = None,
        cache: TestSuiteCache | None = None,
        use_shared_cache: bool = True,
    ) -> None:
        """
        Initialize cached test loader.

        Args:
            env: Custom environment for variable substitution.
            cache: Cache instance to use. If None and use_shared_cache is True,
                  uses a shared global cache.
            use_shared_cache: Whether to use the shared global cache.
        """
        super().__init__(env=env)

        if cache is not None:
            self._cache = cache
        elif use_shared_cache:
            if CachedTestLoader._shared_cache is None:
                CachedTestLoader._shared_cache = TestSuiteCache()
            self._cache = CachedTestLoader._shared_cache
        else:
            self._cache = TestSuiteCache()

    def load_file(self, file_path: str | Path) -> TestSuite:
        """
        Load test suite from file with caching.

        Args:
            file_path: Path to test suite YAML file.

        Returns:
            Validated TestSuite object.
        """
        file_path = Path(file_path)

        # Try cache first
        cached = self._cache.get(file_path)
        if cached is not None:
            return cached

        # Load normally
        suite = super().load_file(file_path)

        # Cache the result
        self._cache.put(file_path, suite)

        return suite

    def invalidate(self, file_path: str | Path) -> bool:
        """Invalidate a cached file."""
        return self._cache.invalidate(file_path)

    def clear_cache(self) -> int:
        """Clear the cache."""
        return self._cache.clear()

    def get_cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.get_stats()


class AdapterCache:
    """
    Cache for adapter instances.

    Caches adapter instances by configuration hash to avoid
    repeated initialization for identical configurations.
    """

    def __init__(self, max_size: int = 50) -> None:
        """
        Initialize adapter cache.

        Args:
            max_size: Maximum number of adapters to cache.
        """
        self._cache: dict[str, CacheEntry[AgentAdapter]] = {}
        self._max_size = max_size
        self._stats = CacheStats(max_size=max_size)
        self._lock = threading.RLock()

    @staticmethod
    def _config_hash(adapter_type: str, config: dict[str, Any] | AdapterConfig) -> str:
        """
        Create a hash key for adapter configuration.

        Args:
            adapter_type: Type of adapter.
            config: Adapter configuration.

        Returns:
            Hash string for cache key.
        """
        if isinstance(config, AdapterConfig):
            config_dict = config.model_dump()
        else:
            config_dict = config

        # Create a stable string representation
        config_str = f"{adapter_type}:{_stable_repr(config_dict)}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get(
        self,
        adapter_type: str,
        config: dict[str, Any] | AdapterConfig,
    ) -> AgentAdapter | None:
        """
        Get a cached adapter if available.

        Args:
            adapter_type: Type of adapter.
            config: Adapter configuration.

        Returns:
            Cached adapter or None.
        """
        cache_key = self._config_hash(adapter_type, config)

        with self._lock:
            entry = self._cache.get(cache_key)

            if entry is None:
                self._stats.misses += 1
                return None

            entry.touch()
            self._stats.hits += 1
            return entry.value

    def get_or_create(
        self,
        adapter_type: str,
        config: dict[str, Any] | AdapterConfig,
    ) -> AgentAdapter:
        """
        Get cached adapter or create a new one.

        Args:
            adapter_type: Type of adapter.
            config: Adapter configuration.

        Returns:
            Adapter instance (cached or newly created).
        """
        cached = self.get(adapter_type, config)
        if cached is not None:
            return cached

        # Create new adapter
        registry = get_registry()
        adapter = registry.create(adapter_type, config)

        # Cache it
        self.put(adapter_type, config, adapter)

        return adapter

    def put(
        self,
        adapter_type: str,
        config: dict[str, Any] | AdapterConfig,
        adapter: AgentAdapter,
    ) -> None:
        """
        Cache an adapter instance.

        Args:
            adapter_type: Type of adapter.
            config: Adapter configuration.
            adapter: Adapter instance to cache.
        """
        cache_key = self._config_hash(adapter_type, config)

        with self._lock:
            if len(self._cache) >= self._max_size and cache_key not in self._cache:
                self._evict_one()

            now = time.time()
            entry = CacheEntry(
                value=adapter,
                created_at=now,
                accessed_at=now,
                metadata={"adapter_type": adapter_type},
            )
            self._cache[cache_key] = entry
            self._stats.size = len(self._cache)

    def _evict_one(self) -> None:
        """Evict the least recently used adapter."""
        if not self._cache:
            return

        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].accessed_at)
        del self._cache[lru_key]
        self._stats.evictions += 1
        self._stats.size = len(self._cache)

    async def cleanup_all(self) -> None:
        """Clean up all cached adapters."""
        with self._lock:
            for entry in self._cache.values():
                try:
                    await entry.value.cleanup()
                except Exception as e:
                    logger.warning("Error cleaning up adapter: %s", e)
            self._cache.clear()
            self._stats.size = 0

    def clear(self) -> int:
        """
        Clear cache without cleanup (use cleanup_all for proper cleanup).

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.size = 0
            return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return self._stats


def _stable_repr(obj: Any) -> str:
    """Create a stable string representation of an object for hashing."""
    if isinstance(obj, dict):
        items = sorted((k, _stable_repr(v)) for k, v in obj.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    elif isinstance(obj, (list, tuple)):
        return "[" + ",".join(_stable_repr(v) for v in obj) + "]"
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return repr(obj)
    else:
        return str(obj)


# Global caches
_test_suite_cache: TestSuiteCache | None = None
_adapter_cache: AdapterCache | None = None


def get_test_suite_cache() -> TestSuiteCache:
    """Get or create the global test suite cache."""
    global _test_suite_cache
    if _test_suite_cache is None:
        _test_suite_cache = TestSuiteCache()
    return _test_suite_cache


def get_adapter_cache() -> AdapterCache:
    """Get or create the global adapter cache."""
    global _adapter_cache
    if _adapter_cache is None:
        _adapter_cache = AdapterCache()
    return _adapter_cache


def clear_all_caches() -> dict[str, int]:
    """
    Clear all global caches.

    Returns:
        Dictionary with counts of cleared entries by cache type.
    """
    results = {}

    global _test_suite_cache, _adapter_cache

    if _test_suite_cache is not None:
        results["test_suite"] = _test_suite_cache.clear()

    if _adapter_cache is not None:
        results["adapter"] = _adapter_cache.clear()

    if CachedTestLoader._shared_cache is not None:
        results["shared_loader"] = CachedTestLoader._shared_cache.clear()

    return results

"""Query result caching for ATP Dashboard.

Provides caching for expensive database queries like leaderboard matrix
and agent comparison queries. Uses TTL-based expiration and LRU eviction.
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QueryCacheEntry[T]:
    """Entry in the query cache with metadata."""

    value: T
    created_at: float
    accessed_at: float
    access_count: int = 1
    key_hash: str = ""

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if this entry has expired based on TTL."""
        return (time.time() - self.created_at) > ttl_seconds


@dataclass
class QueryCacheStats:
    """Statistics for a query cache."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
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
            "expirations": self.expirations,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate,
        }


class QueryCache[T]:
    """Generic cache for query results with TTL and LRU eviction.

    Thread-safe cache that stores query results with automatic expiration.
    Designed for caching expensive database queries like leaderboard matrices.

    Type Parameters:
        T: The type of values stored in the cache.

    Example:
        >>> cache = QueryCache[LeaderboardMatrixResponse](ttl_seconds=60)
        >>> result = cache.get("suite:benchmark:agents:a,b,c")
        >>> if result is None:
        ...     result = await expensive_query()
        ...     cache.put("suite:benchmark:agents:a,b,c", result)
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: float = 60.0,
    ) -> None:
        """Initialize the query cache.

        Args:
            max_size: Maximum number of entries to cache.
            ttl_seconds: Time-to-live in seconds for cache entries.
        """
        self._cache: dict[str, QueryCacheEntry[T]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._stats = QueryCacheStats(max_size=max_size)
        self._lock = threading.RLock()

    @staticmethod
    def _make_key(*args: Any, **kwargs: Any) -> str:
        """Create a cache key from arguments.

        Args:
            *args: Positional arguments to include in key.
            **kwargs: Keyword arguments to include in key.

        Returns:
            A hash string suitable as a cache key.
        """
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = ":".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(self, key: str) -> T | None:
        """Get a cached value if valid.

        Args:
            key: Cache key to look up.

        Returns:
            Cached value if found and not expired, None otherwise.
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            # Check TTL
            if entry.is_expired(self._ttl):
                del self._cache[key]
                self._stats.misses += 1
                self._stats.expirations += 1
                self._stats.size = len(self._cache)
                logger.debug("Cache entry expired: %s", key[:16])
                return None

            entry.touch()
            self._stats.hits += 1
            logger.debug("Cache hit: %s", key[:16])
            return entry.value

    def put(self, key: str, value: T) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_one()

            now = time.time()
            entry: QueryCacheEntry[T] = QueryCacheEntry(
                value=value,
                created_at=now,
                accessed_at=now,
                key_hash=key,
            )
            self._cache[key] = entry
            self._stats.size = len(self._cache)
            logger.debug("Cached query result: %s", key[:16])

    def _evict_one(self) -> None:
        """Evict the least recently accessed entry."""
        if not self._cache:
            return

        # First, try to evict expired entries
        expired_keys = [k for k, v in self._cache.items() if v.is_expired(self._ttl)]
        if expired_keys:
            del self._cache[expired_keys[0]]
            self._stats.evictions += 1
            self._stats.expirations += 1
            self._stats.size = len(self._cache)
            return

        # Otherwise, evict LRU
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].accessed_at)
        del self._cache[lru_key]
        self._stats.evictions += 1
        self._stats.size = len(self._cache)

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate.

        Returns:
            True if entry was removed, False if not in cache.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.evictions += 1
                self._stats.size = len(self._cache)
                return True
            return False

    def invalidate_prefix(self, prefix: str) -> int:
        """Invalidate all cache entries with keys matching a prefix.

        Args:
            prefix: Key prefix to match.

        Returns:
            Number of entries invalidated.
        """
        with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_remove:
                del self._cache[key]
            count = len(keys_to_remove)
            self._stats.evictions += count
            self._stats.size = len(self._cache)
            return count

    def clear(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.size = 0
            return count

    def get_stats(self) -> QueryCacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return self._stats

    @property
    def ttl_seconds(self) -> float:
        """Get the TTL in seconds."""
        return self._ttl

    @ttl_seconds.setter
    def ttl_seconds(self, value: float) -> None:
        """Set the TTL in seconds."""
        with self._lock:
            self._ttl = value


# Global cache instances for different query types
_leaderboard_cache: QueryCache[Any] | None = None
_comparison_cache: QueryCache[Any] | None = None


def get_leaderboard_cache(
    max_size: int = 50,
    ttl_seconds: float = 30.0,
) -> QueryCache[Any]:
    """Get or create the global leaderboard query cache.

    Args:
        max_size: Maximum number of entries (used only on first call).
        ttl_seconds: TTL in seconds (used only on first call).

    Returns:
        The global leaderboard cache instance.
    """
    global _leaderboard_cache
    if _leaderboard_cache is None:
        _leaderboard_cache = QueryCache(max_size=max_size, ttl_seconds=ttl_seconds)
    return _leaderboard_cache


def get_comparison_cache(
    max_size: int = 50,
    ttl_seconds: float = 30.0,
) -> QueryCache[Any]:
    """Get or create the global comparison query cache.

    Args:
        max_size: Maximum number of entries (used only on first call).
        ttl_seconds: TTL in seconds (used only on first call).

    Returns:
        The global comparison cache instance.
    """
    global _comparison_cache
    if _comparison_cache is None:
        _comparison_cache = QueryCache(max_size=max_size, ttl_seconds=ttl_seconds)
    return _comparison_cache


def invalidate_suite_caches(suite_name: str) -> int:
    """Invalidate all caches related to a suite.

    Should be called when new test results are persisted for a suite.

    Args:
        suite_name: Name of the suite to invalidate.

    Returns:
        Number of cache entries invalidated.
    """
    count = 0
    if _leaderboard_cache is not None:
        count += _leaderboard_cache.invalidate_prefix(f"leaderboard:{suite_name}")
    if _comparison_cache is not None:
        count += _comparison_cache.invalidate_prefix(f"comparison:{suite_name}")
    return count


def clear_all_query_caches() -> dict[str, int]:
    """Clear all query caches.

    Returns:
        Dictionary with counts of cleared entries by cache type.
    """
    results: dict[str, int] = {}

    global _leaderboard_cache, _comparison_cache

    if _leaderboard_cache is not None:
        results["leaderboard"] = _leaderboard_cache.clear()

    if _comparison_cache is not None:
        results["comparison"] = _comparison_cache.clear()

    return results


def get_query_cache_stats() -> dict[str, dict[str, Any]]:
    """Get statistics for all query caches.

    Returns:
        Dictionary with stats by cache type.
    """
    stats: dict[str, dict[str, Any]] = {}

    if _leaderboard_cache is not None:
        stats["leaderboard"] = _leaderboard_cache.get_stats().to_dict()

    if _comparison_cache is not None:
        stats["comparison"] = _comparison_cache.get_stats().to_dict()

    return stats

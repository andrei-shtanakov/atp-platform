"""Tests for the query cache module."""

import time

from atp.dashboard.query_cache import (
    QueryCache,
    QueryCacheStats,
    clear_all_query_caches,
    get_leaderboard_cache,
    get_query_cache_stats,
    invalidate_suite_caches,
)


class TestQueryCacheStats:
    """Tests for QueryCacheStats dataclass."""

    def test_hit_rate_no_requests(self) -> None:
        """Test hit rate is 0 when no requests."""
        stats = QueryCacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self) -> None:
        """Test hit rate is 1.0 when all hits."""
        stats = QueryCacheStats(hits=10, misses=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_all_misses(self) -> None:
        """Test hit rate is 0 when all misses."""
        stats = QueryCacheStats(hits=0, misses=10)
        assert stats.hit_rate == 0.0

    def test_hit_rate_mixed(self) -> None:
        """Test hit rate with mixed hits and misses."""
        stats = QueryCacheStats(hits=3, misses=7)
        assert stats.hit_rate == 0.3

    def test_to_dict(self) -> None:
        """Test converting stats to dictionary."""
        stats = QueryCacheStats(
            hits=5,
            misses=5,
            evictions=2,
            expirations=1,
            size=10,
            max_size=100,
        )
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert d["expirations"] == 1
        assert d["size"] == 10
        assert d["max_size"] == 100
        assert d["hit_rate"] == 0.5


class TestQueryCache:
    """Tests for QueryCache class."""

    def test_cache_put_and_get(self) -> None:
        """Test basic put and get operations."""
        cache: QueryCache[str] = QueryCache(max_size=10, ttl_seconds=60)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_miss_returns_none(self) -> None:
        """Test that cache miss returns None."""
        cache: QueryCache[str] = QueryCache(max_size=10, ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_cache_ttl_expiration(self) -> None:
        """Test that entries expire after TTL."""
        cache: QueryCache[str] = QueryCache(max_size=10, ttl_seconds=0.1)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_cache_lru_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        cache: QueryCache[int] = QueryCache(max_size=3, ttl_seconds=60)
        cache.put("key1", 1)
        cache.put("key2", 2)
        cache.put("key3", 3)
        # Access key1 to make it most recently used
        cache.get("key1")
        # Adding key4 should evict key2 (LRU)
        cache.put("key4", 4)
        assert cache.get("key1") == 1
        assert cache.get("key2") is None
        assert cache.get("key3") == 3
        assert cache.get("key4") == 4

    def test_cache_invalidate(self) -> None:
        """Test invalidating a specific key."""
        cache: QueryCache[str] = QueryCache(max_size=10, ttl_seconds=60)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        assert cache.invalidate("key1") is True
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_cache_invalidate_nonexistent(self) -> None:
        """Test invalidating a nonexistent key."""
        cache: QueryCache[str] = QueryCache(max_size=10, ttl_seconds=60)
        assert cache.invalidate("nonexistent") is False

    def test_cache_invalidate_prefix(self) -> None:
        """Test invalidating keys by prefix."""
        cache: QueryCache[str] = QueryCache(max_size=10, ttl_seconds=60)
        cache.put("prefix:a", "value_a")
        cache.put("prefix:b", "value_b")
        cache.put("other:c", "value_c")
        count = cache.invalidate_prefix("prefix:")
        assert count == 2
        assert cache.get("prefix:a") is None
        assert cache.get("prefix:b") is None
        assert cache.get("other:c") == "value_c"

    def test_cache_clear(self) -> None:
        """Test clearing all cache entries."""
        cache: QueryCache[int] = QueryCache(max_size=10, ttl_seconds=60)
        cache.put("key1", 1)
        cache.put("key2", 2)
        cache.put("key3", 3)
        count = cache.clear()
        assert count == 3
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    def test_cache_stats(self) -> None:
        """Test cache statistics tracking."""
        cache: QueryCache[str] = QueryCache(max_size=10, ttl_seconds=60)
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss
        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.size == 1

    def test_cache_make_key(self) -> None:
        """Test cache key generation."""
        key1 = QueryCache._make_key("suite1", "agent1", limit=5)
        key2 = QueryCache._make_key("suite1", "agent1", limit=5)
        key3 = QueryCache._make_key("suite1", "agent1", limit=10)
        assert key1 == key2  # Same args produce same key
        assert key1 != key3  # Different args produce different key

    def test_cache_ttl_property(self) -> None:
        """Test getting and setting TTL."""
        cache: QueryCache[str] = QueryCache(ttl_seconds=60)
        assert cache.ttl_seconds == 60
        cache.ttl_seconds = 30
        assert cache.ttl_seconds == 30

    def test_cache_update_existing_key(self) -> None:
        """Test updating an existing cached value."""
        cache: QueryCache[str] = QueryCache(max_size=10, ttl_seconds=60)
        cache.put("key1", "value1")
        cache.put("key1", "value2")
        assert cache.get("key1") == "value2"
        stats = cache.get_stats()
        assert stats.size == 1  # Still only one entry


class TestGlobalCacheFunctions:
    """Tests for global cache management functions."""

    def test_get_leaderboard_cache(self) -> None:
        """Test getting the global leaderboard cache."""
        cache = get_leaderboard_cache()
        assert cache is not None
        # Should return same instance
        cache2 = get_leaderboard_cache()
        assert cache is cache2

    def test_invalidate_suite_caches(self) -> None:
        """Test invalidating caches for a suite."""
        cache = get_leaderboard_cache()
        cache.put("leaderboard:test-suite:agents", {"data": "test"})
        count = invalidate_suite_caches("test-suite")
        assert count >= 1
        assert cache.get("leaderboard:test-suite:agents") is None

    def test_clear_all_query_caches(self) -> None:
        """Test clearing all query caches."""
        cache = get_leaderboard_cache()
        cache.put("test_key", {"data": "test"})
        results = clear_all_query_caches()
        assert "leaderboard" in results

    def test_get_query_cache_stats(self) -> None:
        """Test getting stats for all caches."""
        get_leaderboard_cache()  # Ensure cache exists
        stats = get_query_cache_stats()
        assert "leaderboard" in stats
        assert "hits" in stats["leaderboard"]
        assert "misses" in stats["leaderboard"]

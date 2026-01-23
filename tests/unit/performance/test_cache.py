"""Tests for the cache module."""

import time
from pathlib import Path

import pytest

from atp.performance.cache import (
    AdapterCache,
    CachedTestLoader,
    CacheEntry,
    CacheStats,
    TestSuiteCache,
    clear_all_caches,
    get_adapter_cache,
    get_test_suite_cache,
)


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_touch(self) -> None:
        """Test touch updates access time and count."""
        entry = CacheEntry(
            value="test",
            created_at=100.0,
            accessed_at=100.0,
            access_count=1,
        )

        time.sleep(0.01)
        entry.touch()

        assert entry.access_count == 2
        assert entry.accessed_at > 100.0


class TestCacheStats:
    """Tests for CacheStats."""

    def test_hit_rate_with_hits(self) -> None:
        """Test hit rate calculation with hits."""
        stats = CacheStats(hits=8, misses=2)
        assert stats.hit_rate == 0.8

    def test_hit_rate_zero_total(self) -> None:
        """Test hit rate with no operations."""
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        stats = CacheStats(hits=5, misses=5, evictions=1, size=10, max_size=100)
        d = stats.to_dict()

        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["hit_rate"] == 0.5


class TestTestSuiteCache:
    """Tests for TestSuiteCache."""

    @pytest.fixture
    def cache(self) -> TestSuiteCache:
        """Create a test suite cache."""
        return TestSuiteCache(max_size=10)

    @pytest.fixture
    def temp_yaml_file(self, tmp_path: Path) -> Path:
        """Create a temporary YAML file."""
        yaml_content = """
test_suite: test-suite
version: "1.0"

defaults:
  timeout_seconds: 300
  scoring:
    quality_weight: 0.4
    completeness_weight: 0.3
    efficiency_weight: 0.2
    cost_weight: 0.1

tests:
  - id: test-001
    name: Test 1
    task:
      description: Test task
"""
        file_path = tmp_path / "test_suite.yaml"
        file_path.write_text(yaml_content)
        return file_path

    def test_get_miss(self, cache: TestSuiteCache, temp_yaml_file: Path) -> None:
        """Test cache miss."""
        result = cache.get(temp_yaml_file)
        assert result is None
        assert cache.get_stats().misses == 1

    def test_put_and_get(self, cache: TestSuiteCache, temp_yaml_file: Path) -> None:
        """Test putting and getting from cache."""
        from atp.loader.loader import TestLoader

        loader = TestLoader()
        suite = loader.load_file(temp_yaml_file)

        cache.put(temp_yaml_file, suite)

        cached = cache.get(temp_yaml_file)
        assert cached is not None
        assert cached.test_suite == suite.test_suite

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.size == 1

    def test_invalidation_on_file_change(
        self, cache: TestSuiteCache, temp_yaml_file: Path
    ) -> None:
        """Test cache invalidation when file changes."""
        from atp.loader.loader import TestLoader

        loader = TestLoader()
        suite = loader.load_file(temp_yaml_file)
        cache.put(temp_yaml_file, suite)

        # Verify it's cached
        assert cache.get(temp_yaml_file) is not None

        # Modify the file
        time.sleep(0.1)  # Ensure mtime changes
        temp_yaml_file.write_text(temp_yaml_file.read_text() + "\n# Modified")

        # Cache should be invalidated
        result = cache.get(temp_yaml_file)
        assert result is None
        assert cache.get_stats().evictions == 1

    def test_manual_invalidation(
        self, cache: TestSuiteCache, temp_yaml_file: Path
    ) -> None:
        """Test manual cache invalidation."""
        from atp.loader.loader import TestLoader

        loader = TestLoader()
        suite = loader.load_file(temp_yaml_file)
        cache.put(temp_yaml_file, suite)

        result = cache.invalidate(temp_yaml_file)
        assert result is True

        # Cache should be empty
        assert cache.get(temp_yaml_file) is None

    def test_eviction_on_max_size(self, tmp_path: Path) -> None:
        """Test LRU eviction when max size is reached."""
        cache = TestSuiteCache(max_size=2)

        yaml_template = """
test_suite: suite-{idx}
version: "1.0"
defaults:
  timeout_seconds: 300
  scoring:
    quality_weight: 0.4
    completeness_weight: 0.3
    efficiency_weight: 0.2
    cost_weight: 0.1
tests:
  - id: test-001
    name: Test 1
    task:
      description: Test task
"""
        from atp.loader.loader import TestLoader

        loader = TestLoader()

        # Create and cache 3 files
        files = []
        for i in range(3):
            path = tmp_path / f"suite_{i}.yaml"
            path.write_text(yaml_template.format(idx=i))
            files.append(path)
            suite = loader.load_file(path)
            cache.put(path, suite)

        # First file should be evicted
        stats = cache.get_stats()
        assert stats.size == 2
        assert stats.evictions == 1

    def test_clear(self, cache: TestSuiteCache, temp_yaml_file: Path) -> None:
        """Test clearing the cache."""
        from atp.loader.loader import TestLoader

        loader = TestLoader()
        suite = loader.load_file(temp_yaml_file)
        cache.put(temp_yaml_file, suite)

        count = cache.clear()
        assert count == 1
        assert cache.get_stats().size == 0


class TestCachedTestLoader:
    """Tests for CachedTestLoader."""

    @pytest.fixture
    def temp_yaml_file(self, tmp_path: Path) -> Path:
        """Create a temporary YAML file."""
        yaml_content = """
test_suite: cached-suite
version: "1.0"

defaults:
  timeout_seconds: 300
  scoring:
    quality_weight: 0.4
    completeness_weight: 0.3
    efficiency_weight: 0.2
    cost_weight: 0.1

tests:
  - id: test-001
    name: Test 1
    task:
      description: Test task
"""
        file_path = tmp_path / "cached_suite.yaml"
        file_path.write_text(yaml_content)
        return file_path

    def test_load_file_caches(self, temp_yaml_file: Path) -> None:
        """Test that loading a file caches it."""
        cache = TestSuiteCache()
        loader = CachedTestLoader(cache=cache, use_shared_cache=False)

        # First load - cache miss
        suite1 = loader.load_file(temp_yaml_file)

        # Second load - cache hit
        suite2 = loader.load_file(temp_yaml_file)

        assert suite1.test_suite == suite2.test_suite

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1

    def test_invalidate(self, temp_yaml_file: Path) -> None:
        """Test invalidating a cached file."""
        cache = TestSuiteCache()
        loader = CachedTestLoader(cache=cache, use_shared_cache=False)

        loader.load_file(temp_yaml_file)
        result = loader.invalidate(temp_yaml_file)

        assert result is True
        assert cache.get_stats().evictions == 1

    def test_clear_cache(self, temp_yaml_file: Path) -> None:
        """Test clearing the loader's cache."""
        cache = TestSuiteCache()
        loader = CachedTestLoader(cache=cache, use_shared_cache=False)

        loader.load_file(temp_yaml_file)
        count = loader.clear_cache()

        assert count == 1

    def test_shared_cache(self, temp_yaml_file: Path) -> None:
        """Test shared cache between loaders."""
        # Clear any existing shared cache
        CachedTestLoader._shared_cache = None

        loader1 = CachedTestLoader(use_shared_cache=True)
        loader2 = CachedTestLoader(use_shared_cache=True)

        # Load with first loader
        suite1 = loader1.load_file(temp_yaml_file)

        # Load with second loader - should get cache hit
        suite2 = loader2.load_file(temp_yaml_file)

        assert suite1.test_suite == suite2.test_suite

        # Both should share the same cache
        assert loader1._cache is loader2._cache


class TestAdapterCache:
    """Tests for AdapterCache."""

    @pytest.fixture
    def cache(self) -> AdapterCache:
        """Create an adapter cache."""
        return AdapterCache(max_size=10)

    def test_config_hash(self, cache: AdapterCache) -> None:
        """Test configuration hashing."""
        config1 = {"endpoint": "http://localhost:8000", "timeout": 30}
        config2 = {"endpoint": "http://localhost:8000", "timeout": 30}
        config3 = {"endpoint": "http://localhost:9000", "timeout": 30}

        hash1 = cache._config_hash("http", config1)
        hash2 = cache._config_hash("http", config2)
        hash3 = cache._config_hash("http", config3)

        assert hash1 == hash2  # Same config should have same hash
        assert hash1 != hash3  # Different config should have different hash

    def test_get_miss(self, cache: AdapterCache) -> None:
        """Test cache miss."""
        config = {"endpoint": "http://localhost:8000"}
        result = cache.get("http", config)

        assert result is None
        assert cache.get_stats().misses == 1

    def test_clear(self, cache: AdapterCache) -> None:
        """Test clearing the cache."""
        # Manually add an entry
        cache._cache["test_key"] = CacheEntry(
            value=None,  # type: ignore
            created_at=time.time(),
            accessed_at=time.time(),
        )

        count = cache.clear()
        assert count == 1
        assert cache.get_stats().size == 0


class TestGlobalCacheFunctions:
    """Tests for global cache functions."""

    def test_get_test_suite_cache(self) -> None:
        """Test getting global test suite cache."""
        cache = get_test_suite_cache()
        assert cache is not None
        assert isinstance(cache, TestSuiteCache)

    def test_get_adapter_cache(self) -> None:
        """Test getting global adapter cache."""
        cache = get_adapter_cache()
        assert cache is not None
        assert isinstance(cache, AdapterCache)

    def test_clear_all_caches(self) -> None:
        """Test clearing all caches."""
        # This should not raise
        results = clear_all_caches()
        assert isinstance(results, dict)

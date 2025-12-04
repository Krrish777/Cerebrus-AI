"""
Tests for File Cache Manager.

Tests the file-based caching functionality including storage,
retrieval, TTL handling, and cleanup.
"""

import json
import time
from pathlib import Path
from typing import Dict
from typing import Any

import pytest

from src.web_scraping.config import CacheConfig
from src.web_scraping.cache.manager import FileCacheManager
from src.web_scraping.exceptions import CacheError


class TestFileCacheManager:
    """Tests for FileCacheManager implementation."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Create a temporary cache directory."""
        cache_path = tmp_path / "cache"
        cache_path.mkdir()
        return cache_path

    @pytest.fixture
    def enabled_config(self, cache_dir: Path) -> CacheConfig:
        """Create enabled cache configuration."""
        return CacheConfig(
            enabled=True,
            cache_dir=cache_dir,
            ttl_hours=24,
            max_cache_size_gb=5,
        )

    @pytest.fixture
    def short_ttl_config(self, cache_dir: Path) -> CacheConfig:
        """Create cache configuration with short TTL for testing."""
        return CacheConfig(
            enabled=True,
            cache_dir=cache_dir,
            ttl_hours=1,  # 1 hour TTL
            max_cache_size_gb=5,
        )

    @pytest.fixture
    def disabled_config(self) -> CacheConfig:
        """Create disabled cache configuration."""
        return CacheConfig(
            enabled=False,
            cache_dir=None,
            ttl_hours=24,
            max_cache_size_gb=5,
        )

    @pytest.fixture
    def enabled_cache(self, enabled_config: CacheConfig) -> FileCacheManager:
        """Create enabled cache manager."""
        return FileCacheManager(enabled_config)

    @pytest.fixture
    def disabled_cache(self, disabled_config: CacheConfig) -> FileCacheManager:
        """Create disabled cache manager."""
        return FileCacheManager(disabled_config)

    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Create sample data for caching."""
        return {
            "url": "https://example.com",
            "content": "Sample content for testing",
            "title": "Test Page",
            "word_count": 5,
        }


class TestCacheOperations(TestFileCacheManager):
    """Tests for basic cache operations."""

    def test_set_and_get(
        self,
        enabled_cache: FileCacheManager,
        sample_data: Dict[str, Any],
    ) -> None:
        """Test setting and getting cache entry."""
        key = "test_key"
        
        enabled_cache.set(key, sample_data)
        result = enabled_cache.get(key)
        
        assert result is not None
        assert result["url"] == sample_data["url"]
        assert result["content"] == sample_data["content"]

    def test_get_nonexistent_returns_none(self, enabled_cache: FileCacheManager) -> None:
        """Test that getting nonexistent key returns None."""
        result = enabled_cache.get("nonexistent_key")
        
        assert result is None

    def test_exists_true_for_cached(
        self,
        enabled_cache: FileCacheManager,
        sample_data: Dict[str, Any],
    ) -> None:
        """Test exists returns True for cached entries."""
        key = "exists_test"
        
        enabled_cache.set(key, sample_data)
        
        assert enabled_cache.exists(key) is True

    def test_exists_false_for_uncached(self, enabled_cache: FileCacheManager) -> None:
        """Test exists returns False for uncached entries."""
        assert enabled_cache.exists("uncached_key") is False

    def test_delete_removes_entry(
        self,
        enabled_cache: FileCacheManager,
        sample_data: Dict[str, Any],
    ) -> None:
        """Test deleting cache entry."""
        key = "delete_test"
        
        enabled_cache.set(key, sample_data)
        assert enabled_cache.exists(key) is True
        
        result = enabled_cache.delete(key)
        
        assert result is True
        assert enabled_cache.exists(key) is False

    def test_delete_nonexistent_returns_false(self, enabled_cache: FileCacheManager) -> None:
        """Test deleting nonexistent key returns False."""
        result = enabled_cache.delete("nonexistent")
        
        assert result is False

    def test_overwrite_existing(
        self,
        enabled_cache: FileCacheManager,
        sample_data: Dict[str, Any],
    ) -> None:
        """Test overwriting existing cache entry."""
        key = "overwrite_test"
        
        enabled_cache.set(key, sample_data)
        
        new_data = {"url": "https://new.com", "content": "New content"}
        enabled_cache.set(key, new_data)
        
        result = enabled_cache.get(key)
        
        assert result is not None
        assert result["url"] == "https://new.com"


class TestDisabledCache(TestFileCacheManager):
    """Tests for disabled cache behavior."""

    def test_set_is_noop(
        self,
        disabled_cache: FileCacheManager,
        sample_data: Dict[str, Any],
    ) -> None:
        """Test that set is no-op when disabled."""
        disabled_cache.set("key", sample_data)
        # Should not raise

    def test_get_returns_none(self, disabled_cache: FileCacheManager) -> None:
        """Test that get returns None when disabled."""
        result = disabled_cache.get("any_key")
        
        assert result is None

    def test_exists_returns_false(self, disabled_cache: FileCacheManager) -> None:
        """Test that exists returns False when disabled."""
        assert disabled_cache.exists("any_key") is False

    def test_delete_returns_false(self, disabled_cache: FileCacheManager) -> None:
        """Test that delete returns False when disabled."""
        assert disabled_cache.delete("any_key") is False


class TestTTLHandling(TestFileCacheManager):
    """Tests for TTL (Time To Live) handling."""

    def test_fresh_entry_not_expired(
        self,
        enabled_cache: FileCacheManager,
        sample_data: Dict[str, Any],
    ) -> None:
        """Test that fresh entry is not expired."""
        key = "fresh_entry"
        
        enabled_cache.set(key, sample_data)
        
        # Should still be valid
        result = enabled_cache.get(key)
        assert result is not None

    def test_expired_entry_returns_none(
        self,
        cache_dir: Path,
        sample_data: Dict[str, Any],
    ) -> None:
        """Test that expired entry returns None."""
        # Create cache with very short TTL
        config = CacheConfig(
            enabled=True,
            cache_dir=cache_dir,
            ttl_hours=1,  # Can't set less than 1 hour in config
            max_cache_size_gb=5,
        )
        cache = FileCacheManager(config)
        
        key = "expired_entry"
        
        # Manually create expired entry
        cache_path = cache_dir / f"{key}.json"
        expired_data = {
            "_cached_at": time.time() - 7200,  # 2 hours ago
            "_ttl_seconds": 3600,  # 1 hour TTL
            "_key": key,
            "data": sample_data,
        }
        with cache_path.open("w") as f:
            json.dump(expired_data, f)
        
        # Should return None and clean up
        result = cache.get(key)
        assert result is None

    def test_custom_ttl_per_entry(
        self,
        enabled_cache: FileCacheManager,
        sample_data: Dict[str, Any],
        cache_dir: Path,
    ) -> None:
        """Test custom TTL per entry."""
        key = "custom_ttl"
        
        # Set with very short custom TTL (1 second)
        enabled_cache.set(key, sample_data, ttl=1)
        
        # Immediately should work
        result = enabled_cache.get(key)
        assert result is not None
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should be expired
        result = enabled_cache.get(key)
        assert result is None


class TestCacheKeyGeneration(TestFileCacheManager):
    """Tests for cache key generation."""

    def test_generate_key_deterministic(self, enabled_cache: FileCacheManager) -> None:
        """Test that key generation is deterministic."""
        url = "https://example.com/page"
        
        key1 = enabled_cache.generate_key(url)
        key2 = enabled_cache.generate_key(url)
        
        assert key1 == key2

    def test_different_urls_different_keys(self, enabled_cache: FileCacheManager) -> None:
        """Test that different URLs generate different keys."""
        url1 = "https://example.com/page1"
        url2 = "https://example.com/page2"
        
        key1 = enabled_cache.generate_key(url1)
        key2 = enabled_cache.generate_key(url2)
        
        assert key1 != key2

    def test_key_case_insensitive(self, enabled_cache: FileCacheManager) -> None:
        """Test that URL case is normalized for key generation."""
        url_lower = "https://example.com/page"
        url_upper = "https://EXAMPLE.COM/page"
        
        key1 = enabled_cache.generate_key(url_lower)
        key2 = enabled_cache.generate_key(url_upper)
        
        assert key1 == key2


class TestCacheCleanup(TestFileCacheManager):
    """Tests for cache cleanup operations."""

    def test_cleanup_expired(
        self,
        cache_dir: Path,
        sample_data: Dict[str, Any],
    ) -> None:
        """Test cleanup of expired entries."""
        config = CacheConfig(
            enabled=True,
            cache_dir=cache_dir,
            ttl_hours=1,
            max_cache_size_gb=5,
        )
        cache = FileCacheManager(config)
        
        # Create some expired entries
        for i in range(3):
            key = f"expired_{i}"
            cache_path = cache_dir / f"{key}.json"
            expired_data = {
                "_cached_at": time.time() - 7200,  # 2 hours ago
                "_ttl_seconds": 3600,  # 1 hour TTL
                "_key": key,
                "data": sample_data,
            }
            with cache_path.open("w") as f:
                json.dump(expired_data, f)
        
        # Create one fresh entry
        cache.set("fresh", sample_data)
        
        # Cleanup
        removed = cache.cleanup()
        
        assert removed == 3
        assert cache.exists("fresh") is True

    def test_clear_all(
        self,
        enabled_cache: FileCacheManager,
        sample_data: Dict[str, Any],
    ) -> None:
        """Test clearing all cache entries."""
        # Add some entries
        for i in range(5):
            enabled_cache.set(f"entry_{i}", sample_data)
        
        assert enabled_cache.entry_count == 5
        
        # Clear all
        removed = enabled_cache.clear()
        
        assert removed == 5
        assert enabled_cache.entry_count == 0


class TestCacheMetrics(TestFileCacheManager):
    """Tests for cache metrics."""

    def test_entry_count(
        self,
        enabled_cache: FileCacheManager,
        sample_data: Dict[str, Any],
    ) -> None:
        """Test entry count metric."""
        assert enabled_cache.entry_count == 0
        
        enabled_cache.set("entry1", sample_data)
        assert enabled_cache.entry_count == 1
        
        enabled_cache.set("entry2", sample_data)
        assert enabled_cache.entry_count == 2

    def test_cache_size_bytes(
        self,
        enabled_cache: FileCacheManager,
        sample_data: Dict[str, Any],
    ) -> None:
        """Test cache size metric."""
        initial_size = enabled_cache.cache_size_bytes
        assert initial_size == 0
        
        enabled_cache.set("entry", sample_data)
        
        # Size should increase
        assert enabled_cache.cache_size_bytes > 0

    def test_disabled_cache_metrics(self, disabled_cache: FileCacheManager) -> None:
        """Test metrics for disabled cache."""
        assert disabled_cache.entry_count == 0
        assert disabled_cache.cache_size_bytes == 0


class TestCacheEdgeCases(TestFileCacheManager):
    """Tests for edge cases."""

    def test_corrupted_cache_file(
        self,
        enabled_cache: FileCacheManager,
        cache_dir: Path,
    ) -> None:
        """Test handling of corrupted cache file."""
        key = "corrupted"
        
        # Create corrupted file
        cache_path = cache_dir / f"{enabled_cache.generate_key(key)}.json"
        with cache_path.open("w") as f:
            f.write("not valid json {{{")
        
        # Should return None and clean up
        result = enabled_cache.get(enabled_cache.generate_key(key))
        assert result is None

    def test_special_characters_in_data(
        self,
        enabled_cache: FileCacheManager,
    ) -> None:
        """Test handling of special characters in cached data."""
        data = {
            "content": "Special chars: <>&\"'{}[]\\n\\t",
            "unicode": "日本語 中文 العربية",
        }
        
        key = "special_chars"
        enabled_cache.set(key, data)
        
        result = enabled_cache.get(key)
        
        assert result is not None
        assert result["unicode"] == "日本語 中文 العربية"

    def test_large_data(
        self,
        enabled_cache: FileCacheManager,
    ) -> None:
        """Test handling of large data."""
        large_data = {
            "content": "x" * 100000,  # 100KB
            "list": list(range(1000)),
        }
        
        key = "large_data"
        enabled_cache.set(key, large_data)
        
        result = enabled_cache.get(key)
        
        assert result is not None
        assert len(result["content"]) == 100000

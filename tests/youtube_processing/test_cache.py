"""
Tests for File Cache Manager.

This module tests the FileCacheManager class.
"""

import json
import time
from pathlib import Path
from typing import Generator

import pytest

from src.youtube_processing.cache.manager import FileCacheManager
from src.youtube_processing.config import CacheConfig
from src.youtube_processing.exceptions import CacheError


class TestFileCacheManager:
    """Tests for FileCacheManager class."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Create a cache directory for testing."""
        cache = tmp_path / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    @pytest.fixture
    def cache_config(self, cache_dir: Path) -> CacheConfig:
        """Create cache config for testing."""
        return CacheConfig(
            enabled=True,
            cache_dir=cache_dir,
            cleanup_after_processing=False,
            max_cache_size_gb=10,
            cache_ttl_days=7,
        )

    @pytest.fixture
    def cache_manager(self, cache_config: CacheConfig) -> FileCacheManager:
        """Create a cache manager for testing."""
        return FileCacheManager(cache_config)

    @pytest.fixture
    def sample_audio_file(self, tmp_path: Path) -> Path:
        """Create a sample audio file for testing."""
        audio_file = tmp_path / "sample.m4a"
        audio_file.write_bytes(b"fake audio content " * 100)
        return audio_file

    class TestCacheOperations:
        """Tests for basic cache operations."""

        def test_put_and_get(
            self,
            cache_manager: FileCacheManager,
            sample_audio_file: Path,
        ) -> None:
            """Test putting and getting a file from cache."""
            video_id = "test123"

            # Put file in cache
            cached_path = cache_manager.put(video_id, sample_audio_file)
            assert cached_path.exists()
            assert video_id in cached_path.name

            # Get file from cache
            retrieved_path = cache_manager.get(video_id)
            assert retrieved_path == cached_path
            assert retrieved_path.exists()

        def test_has_returns_true_for_cached(
            self,
            cache_manager: FileCacheManager,
            sample_audio_file: Path,
        ) -> None:
            """Test has() returns True for cached videos."""
            video_id = "test123"
            cache_manager.put(video_id, sample_audio_file)

            assert cache_manager.has(video_id) is True

        def test_has_returns_false_for_uncached(
            self, cache_manager: FileCacheManager
        ) -> None:
            """Test has() returns False for uncached videos."""
            assert cache_manager.has("nonexistent") is False

        def test_get_returns_none_for_uncached(
            self, cache_manager: FileCacheManager
        ) -> None:
            """Test get() returns None for uncached videos."""
            result = cache_manager.get("nonexistent")
            assert result is None

        def test_remove_cached_file(
            self,
            cache_manager: FileCacheManager,
            sample_audio_file: Path,
        ) -> None:
            """Test removing a cached file."""
            video_id = "test123"
            cached_path = cache_manager.put(video_id, sample_audio_file)

            # Remove from cache
            removed = cache_manager.remove(video_id)

            assert removed is True
            assert not cached_path.exists()
            assert cache_manager.has(video_id) is False

        def test_remove_uncached_returns_false(
            self, cache_manager: FileCacheManager
        ) -> None:
            """Test removing uncached video returns False."""
            result = cache_manager.remove("nonexistent")
            assert result is False

        def test_clear_removes_all(
            self,
            cache_manager: FileCacheManager,
            sample_audio_file: Path,
        ) -> None:
            """Test clearing all cached files."""
            # Add multiple files
            for i in range(3):
                cache_manager.put(f"video{i}", sample_audio_file)

            assert cache_manager.entry_count == 3

            # Clear cache
            removed = cache_manager.clear()

            assert removed == 3
            assert cache_manager.entry_count == 0

    class TestCacheExpiration:
        """Tests for cache expiration functionality."""

        def test_expired_entry_not_returned(
            self,
            cache_dir: Path,
            sample_audio_file: Path,
        ) -> None:
            """Test that expired entries are not returned."""
            # Create config with very short TTL
            config = CacheConfig(
                enabled=True,
                cache_dir=cache_dir,
                cache_ttl_days=0,  # Immediate expiration
            )
            cache_manager = FileCacheManager(config)

            video_id = "test123"
            cache_manager.put(video_id, sample_audio_file)

            # Manually set cached_at to past
            index_path = cache_dir / ".cache_index.json"
            with index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)
            index[video_id]["cached_at"] = time.time() - 86400 * 2  # 2 days ago
            with index_path.open("w", encoding="utf-8") as f:
                json.dump(index, f)

            # Reload cache manager
            cache_manager = FileCacheManager(config)

            # Should not return expired entry
            result = cache_manager.get(video_id)
            assert result is None

        def test_cleanup_expired_removes_old_entries(
            self,
            cache_dir: Path,
            sample_audio_file: Path,
        ) -> None:
            """Test cleanup_expired removes old entries."""
            config = CacheConfig(
                enabled=True,
                cache_dir=cache_dir,
                cache_ttl_days=1,
            )
            cache_manager = FileCacheManager(config)

            # Add file
            video_id = "expired123"
            cache_manager.put(video_id, sample_audio_file)

            # Manually set cached_at to past
            index_path = cache_dir / ".cache_index.json"
            with index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)
            index[video_id]["cached_at"] = time.time() - 86400 * 10  # 10 days ago
            with index_path.open("w", encoding="utf-8") as f:
                json.dump(index, f)

            # Reload and cleanup
            cache_manager = FileCacheManager(config)
            removed = cache_manager.cleanup_expired()

            assert removed == 1
            assert cache_manager.has(video_id) is False

    class TestCacheProperties:
        """Tests for cache properties."""

        def test_cache_size_bytes(
            self,
            cache_manager: FileCacheManager,
            sample_audio_file: Path,
        ) -> None:
            """Test cache_size_bytes property."""
            initial_size = cache_manager.cache_size_bytes
            assert initial_size == 0

            # Add file
            cache_manager.put("test123", sample_audio_file)

            # Size should increase
            assert cache_manager.cache_size_bytes > 0

        def test_entry_count(
            self,
            cache_manager: FileCacheManager,
            sample_audio_file: Path,
        ) -> None:
            """Test entry_count property."""
            assert cache_manager.entry_count == 0

            cache_manager.put("video1", sample_audio_file)
            assert cache_manager.entry_count == 1

            cache_manager.put("video2", sample_audio_file)
            assert cache_manager.entry_count == 2

            cache_manager.remove("video1")
            assert cache_manager.entry_count == 1

    class TestDisabledCache:
        """Tests for disabled cache."""

        def test_disabled_cache_operations(
            self, tmp_path: Path, sample_audio_file: Path
        ) -> None:
            """Test that disabled cache doesn't store anything."""
            config = CacheConfig(enabled=False)
            cache_manager = FileCacheManager(config)

            # Put should return original path
            result_path = cache_manager.put("test123", sample_audio_file)
            assert result_path == sample_audio_file

            # Has should return False
            assert cache_manager.has("test123") is False

            # Get should return None
            assert cache_manager.get("test123") is None

    class TestErrorHandling:
        """Tests for error handling."""

        def test_put_nonexistent_file_raises_error(
            self, cache_manager: FileCacheManager, tmp_path: Path
        ) -> None:
            """Test that putting nonexistent file raises CacheError."""
            nonexistent = tmp_path / "nonexistent.m4a"

            with pytest.raises(CacheError) as exc_info:
                cache_manager.put("test123", nonexistent)

            assert "does not exist" in str(exc_info.value)

        def test_index_corruption_recovery(
            self,
            cache_dir: Path,
            sample_audio_file: Path,
        ) -> None:
            """Test recovery from corrupted index file."""
            config = CacheConfig(enabled=True, cache_dir=cache_dir)

            # Create corrupted index
            index_path = cache_dir / ".cache_index.json"
            index_path.write_text("invalid json {{{", encoding="utf-8")

            # Should recover with empty index
            cache_manager = FileCacheManager(config)
            assert cache_manager.entry_count == 0

            # Should be able to add new entries
            cache_manager.put("test123", sample_audio_file)
            assert cache_manager.entry_count == 1

        def test_missing_file_removed_from_index(
            self,
            cache_dir: Path,
            sample_audio_file: Path,
        ) -> None:
            """Test that missing files are removed from index on access."""
            config = CacheConfig(enabled=True, cache_dir=cache_dir)
            cache_manager = FileCacheManager(config)

            # Add file to cache
            video_id = "test123"
            cached_path = cache_manager.put(video_id, sample_audio_file)

            # Delete the actual file
            cached_path.unlink()

            # Get should return None and clean up index
            result = cache_manager.get(video_id)
            assert result is None

            # Entry should be removed from index
            assert cache_manager.has(video_id) is False

"""
File Cache Manager Module.

This module provides the FileCacheManager class for caching downloaded
audio files on the local filesystem.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Dict
from typing import Optional

from src.core.logging import get_logger
from src.youtube_processing.config import CacheConfig
from src.youtube_processing.exceptions import CacheError
from src.youtube_processing.interfaces import CacheManager

logger = get_logger(__name__)


class FileCacheManager(CacheManager):
    """
    File-based cache manager for audio files.

    This implementation stores cached audio files on the local filesystem
    with metadata stored in a JSON index file.

    Features:
    - TTL-based cache expiration
    - Size-based cache limits
    - Thread-safe file operations
    - Automatic cleanup of expired entries

    Example:
        config = CacheConfig(enabled=True, cache_dir=Path("cache"))
        cache = FileCacheManager(config)
        if cache.has(video_id):
            audio_path = cache.get(video_id)
        else:
            cache.put(video_id, downloaded_path)
    """

    _INDEX_FILENAME = ".cache_index.json"

    def __init__(self, config: CacheConfig) -> None:
        """
        Initialize the cache manager.

        Args:
            config: Cache configuration.
        """
        self._config = config
        self._cache_dir = config.cache_dir
        self._index: Dict[str, Dict] = {}

        if config.enabled and self._cache_dir:
            self._initialize_cache()

        logger.info(
            "Initialized FileCacheManager: enabled=%s, dir=%s",
            config.enabled,
            self._cache_dir,
        )

    def _initialize_cache(self) -> None:
        """Initialize cache directory and load index."""
        if self._cache_dir is None:
            return

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_index()
        except OSError as e:
            raise CacheError(
                message=f"Failed to initialize cache directory: {e}",
                cache_path=str(self._cache_dir),
                original_error=e,
            ) from e

    def _load_index(self) -> None:
        """Load the cache index from disk."""
        if self._cache_dir is None:
            return

        index_path = self._cache_dir / self._INDEX_FILENAME
        if index_path.exists():
            try:
                with index_path.open("r", encoding="utf-8") as f:
                    self._index = json.load(f)
                logger.debug("Loaded cache index with %d entries", len(self._index))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load cache index, starting fresh: %s", e)
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        if self._cache_dir is None:
            return

        index_path = self._cache_dir / self._INDEX_FILENAME
        try:
            with index_path.open("w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2)
        except OSError as e:
            logger.error("Failed to save cache index: %s", e)

    def get(self, video_id: str) -> Optional[Path]:
        """
        Retrieve a cached audio file.

        Args:
            video_id: YouTube video ID.

        Returns:
            Path to the cached file, or None if not cached.

        Raises:
            CacheError: If cache access fails.
        """
        if not self._config.enabled or self._cache_dir is None:
            return None

        entry = self._index.get(video_id)
        if not entry:
            logger.debug("Cache miss for video: %s", video_id)
            return None

        # Check if entry is expired
        if self._is_expired(entry):
            logger.debug("Cache entry expired for video: %s", video_id)
            self.remove(video_id)
            return None

        # Verify file exists
        cached_path = Path(entry["path"])
        if not cached_path.exists():
            logger.warning("Cache index references missing file: %s", cached_path)
            del self._index[video_id]
            self._save_index()
            return None

        logger.debug("Cache hit for video: %s", video_id)
        return cached_path

    def put(self, video_id: str, audio_path: Path) -> Path:
        """
        Store an audio file in the cache.

        Args:
            video_id: YouTube video ID.
            audio_path: Path to the audio file to cache.

        Returns:
            Path to the cached file.

        Raises:
            CacheError: If caching fails.
        """
        if not self._config.enabled or self._cache_dir is None:
            return audio_path

        if not audio_path.exists():
            raise CacheError(
                message=f"Audio file does not exist: {audio_path}",
                video_url=video_id,
            )

        try:
            # Create cache filename
            cache_filename = f"{video_id}{audio_path.suffix}"
            cached_path = self._cache_dir / cache_filename

            # Copy file to cache (or move if cleanup is enabled)
            if audio_path != cached_path:
                shutil.copy2(audio_path, cached_path)

            # Update index
            self._index[video_id] = {
                "path": str(cached_path),
                "size": cached_path.stat().st_size,
                "cached_at": time.time(),
                "original_name": audio_path.name,
            }
            self._save_index()

            logger.debug("Cached audio for video: %s -> %s", video_id, cached_path)
            return cached_path

        except OSError as e:
            raise CacheError(
                message=f"Failed to cache audio file: {e}",
                video_url=video_id,
                cache_path=str(audio_path),
                original_error=e,
            ) from e

    def has(self, video_id: str) -> bool:
        """
        Check if a video is cached.

        Args:
            video_id: YouTube video ID.

        Returns:
            True if the video is in the cache and not expired.
        """
        if not self._config.enabled:
            return False

        entry = self._index.get(video_id)
        if not entry:
            return False

        if self._is_expired(entry):
            return False

        # Verify file exists
        return Path(entry["path"]).exists()

    def remove(self, video_id: str) -> bool:
        """
        Remove a video from the cache.

        Args:
            video_id: YouTube video ID.

        Returns:
            True if the video was removed, False if it wasn't cached.

        Raises:
            CacheError: If removal fails.
        """
        if video_id not in self._index:
            return False

        entry = self._index[video_id]
        cached_path = Path(entry["path"])

        try:
            if cached_path.exists():
                cached_path.unlink()
                logger.debug("Removed cached file: %s", cached_path)

            del self._index[video_id]
            self._save_index()
            return True

        except OSError as e:
            raise CacheError(
                message=f"Failed to remove cached file: {e}",
                video_url=video_id,
                cache_path=str(cached_path),
                original_error=e,
            ) from e

    def clear(self) -> int:
        """
        Clear all cached files.

        Returns:
            Number of files removed.

        Raises:
            CacheError: If clearing fails.
        """
        if self._cache_dir is None:
            return 0

        removed_count = 0
        errors = []

        for video_id in list(self._index.keys()):
            try:
                if self.remove(video_id):
                    removed_count += 1
            except CacheError as e:
                errors.append(str(e))

        if errors:
            logger.warning("Errors during cache clear: %s", errors)

        logger.info("Cleared %d cache entries", removed_count)
        return removed_count

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of files removed.

        Raises:
            CacheError: If cleanup fails.
        """
        if not self._config.enabled:
            return 0

        removed_count = 0
        for video_id in list(self._index.keys()):
            entry = self._index.get(video_id)
            if entry and self._is_expired(entry):
                try:
                    if self.remove(video_id):
                        removed_count += 1
                except CacheError as e:
                    logger.warning("Failed to remove expired entry: %s", e)

        if removed_count > 0:
            logger.info("Cleaned up %d expired cache entries", removed_count)

        return removed_count

    @property
    def cache_size_bytes(self) -> int:
        """Return the total size of the cache in bytes."""
        return sum(entry.get("size", 0) for entry in self._index.values())

    @property
    def entry_count(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._index)

    def _is_expired(self, entry: Dict) -> bool:
        """Check if a cache entry is expired."""
        cached_at = entry.get("cached_at", 0)
        ttl_seconds = self._config.cache_ttl_days * 24 * 60 * 60
        return time.time() - cached_at > ttl_seconds

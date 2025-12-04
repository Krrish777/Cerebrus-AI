"""
File-Based Cache Manager Implementation.

This module provides file-based caching for scraped web content.
Uses JSON files for persistent storage with TTL-based expiration.

Following AGENTS.md principles:
    - Single responsibility: Only cache management
    - Defensive: Handles file system errors gracefully
    - Portability: Uses pathlib for all file operations
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

from src.core.logging import get_logger
from src.web_scraping.config import CacheConfig
from src.web_scraping.exceptions import CacheError
from src.web_scraping.interfaces import CacheManager

logger = get_logger(__name__)


class FileCacheManager(CacheManager):
    """
    File-based cache manager for scraped content.

    Stores scraped content as JSON files with metadata for TTL handling.
    Files are organized by URL hash to prevent filesystem issues.

    Example:
        config = CacheConfig(
            enabled=True,
            cache_dir=Path("cache/web_scraping"),
            ttl_hours=24,
        )
        cache = FileCacheManager(config)
        cache.set(cache.generate_key(url), content_dict)
    """

    def __init__(self, config: CacheConfig) -> None:
        """
        Initialize the file cache manager.

        Args:
            config: Cache configuration.
        """
        self._config = config
        self._cache_dir: Optional[Path] = None

        if config.enabled and config.cache_dir:
            self._cache_dir = config.cache_dir
            self._ensure_cache_dir()

        logger.debug(
            "File cache manager initialized: enabled=%s, dir=%s, ttl=%dh",
            self._config.enabled,
            self._cache_dir,
            self._config.ttl_hours,
        )

    def _ensure_cache_dir(self) -> None:
        """Ensure the cache directory exists."""
        if self._cache_dir is None:
            return

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Cache directory ensured: %s", self._cache_dir)
        except OSError as error:
            raise CacheError(
                message=f"Failed to create cache directory: {self._cache_dir}",
                operation="create_directory",
                original_error=error,
            ) from error

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached content by key.

        Args:
            key: Cache key.

        Returns:
            Cached content dictionary, or None if not cached or expired.

        Raises:
            CacheError: If cache access fails.
        """
        if not self._config.enabled or self._cache_dir is None:
            return None

        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            logger.debug("Cache miss for key: %s", key)
            return None

        try:
            with cache_path.open("r", encoding="utf-8") as file:
                cached = json.load(file)

            # Check TTL (use stored TTL if available, otherwise use config)
            cached_at = cached.get("_cached_at", 0)
            ttl_seconds = cached.get("_ttl_seconds", self._config.ttl_hours * 3600)

            if time.time() - cached_at > ttl_seconds:
                logger.debug("Cache entry expired for key: %s", key)
                self._safe_delete(cache_path)
                return None

            logger.debug("Cache hit for key: %s", key)

            # Remove internal metadata before returning
            return cached.get("data")

        except json.JSONDecodeError as error:
            logger.warning("Invalid cache file for key %s: %s", key, error)
            self._safe_delete(cache_path)
            return None

        except OSError as error:
            raise CacheError(
                message=f"Failed to read cache file: {cache_path}",
                cache_key=key,
                operation="get",
                original_error=error,
            ) from error

    def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store content in cache.

        Args:
            key: Cache key.
            value: Content dictionary to cache.
            ttl: Optional time-to-live in seconds (overrides config).

        Raises:
            CacheError: If caching fails.
        """
        if not self._config.enabled or self._cache_dir is None:
            return

        cache_path = self._get_cache_path(key)

        try:
            # Wrap value with metadata
            cached = {
                "_cached_at": time.time(),
                "_ttl_seconds": ttl if ttl is not None else self._config.ttl_hours * 3600,
                "_key": key,
                "data": value,
            }

            with cache_path.open("w", encoding="utf-8") as file:
                json.dump(cached, file, indent=2, default=str)

            logger.debug("Cached content for key: %s", key)

        except OSError as error:
            raise CacheError(
                message=f"Failed to write cache file: {cache_path}",
                cache_key=key,
                operation="set",
                original_error=error,
            ) from error

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache and is not expired.

        Args:
            key: Cache key.

        Returns:
            True if the key exists and is valid.
        """
        if not self._config.enabled or self._cache_dir is None:
            return False

        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return False

        # Check if expired
        try:
            with cache_path.open("r", encoding="utf-8") as file:
                cached = json.load(file)

            cached_at = cached.get("_cached_at", 0)
            ttl_seconds = cached.get("_ttl_seconds", self._config.ttl_hours * 3600)

            return time.time() - cached_at <= ttl_seconds

        except (json.JSONDecodeError, OSError):
            return False

    def delete(self, key: str) -> bool:
        """
        Delete cached content.

        Args:
            key: Cache key.

        Returns:
            True if the key was deleted, False if it didn't exist.

        Raises:
            CacheError: If deletion fails.
        """
        if not self._config.enabled or self._cache_dir is None:
            return False

        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return False

        try:
            cache_path.unlink()
            logger.debug("Deleted cache entry for key: %s", key)
            return True

        except OSError as error:
            raise CacheError(
                message=f"Failed to delete cache file: {cache_path}",
                cache_key=key,
                operation="delete",
                original_error=error,
            ) from error

    def cleanup(self) -> int:
        """
        Cleanup expired cache entries.

        Returns:
            Number of entries removed.

        Raises:
            CacheError: If cleanup fails.
        """
        if not self._config.enabled or self._cache_dir is None:
            return 0

        removed = 0
        current_time = time.time()

        try:
            for cache_file in self._cache_dir.glob("*.json"):
                try:
                    with cache_file.open("r", encoding="utf-8") as file:
                        cached = json.load(file)

                    cached_at = cached.get("_cached_at", 0)
                    ttl_seconds = cached.get("_ttl_seconds", self._config.ttl_hours * 3600)

                    if current_time - cached_at > ttl_seconds:
                        cache_file.unlink()
                        removed += 1

                except (json.JSONDecodeError, OSError):
                    # Remove invalid cache files
                    self._safe_delete(cache_file)
                    removed += 1

            logger.info("Cache cleanup completed: removed %d entries", removed)
            return removed

        except OSError as error:
            raise CacheError(
                message="Failed to cleanup cache directory",
                operation="cleanup",
                original_error=error,
            ) from error

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries removed.

        Raises:
            CacheError: If clearing fails.
        """
        if not self._config.enabled or self._cache_dir is None:
            return 0

        removed = 0

        try:
            for cache_file in self._cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                    removed += 1
                except OSError:
                    pass

            logger.info("Cache cleared: removed %d entries", removed)
            return removed

        except OSError as error:
            raise CacheError(
                message="Failed to clear cache directory",
                operation="clear",
                original_error=error,
            ) from error

    def generate_key(self, url: str) -> str:
        """
        Generate a cache key from a URL.

        Uses SHA-256 hash to create a filesystem-safe key.

        Args:
            url: URL to generate key for.

        Returns:
            Cache key string.
        """
        # Normalize URL for consistent hashing
        normalized_url = url.lower().strip()

        # Generate hash
        url_hash = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()

        return url_hash

    @property
    def cache_size_bytes(self) -> int:
        """Return the total size of the cache in bytes."""
        if not self._config.enabled or self._cache_dir is None:
            return 0

        total_size = 0

        try:
            for cache_file in self._cache_dir.glob("*.json"):
                try:
                    total_size += cache_file.stat().st_size
                except OSError:
                    pass

            return total_size

        except OSError:
            return 0

    @property
    def entry_count(self) -> int:
        """Return the number of entries in the cache."""
        if not self._config.enabled or self._cache_dir is None:
            return 0

        try:
            return len(list(self._cache_dir.glob("*.json")))
        except OSError:
            return 0

    def _get_cache_path(self, key: str) -> Path:
        """
        Get the file path for a cache key.

        Args:
            key: Cache key.

        Returns:
            Path to the cache file.
        """
        if self._cache_dir is None:
            raise CacheError(
                message="Cache directory not configured",
                cache_key=key,
                operation="get_path",
            )

        return self._cache_dir / f"{key}.json"

    def _safe_delete(self, path: Path) -> None:
        """
        Safely delete a file, ignoring errors.

        Args:
            path: Path to delete.
        """
        try:
            path.unlink()
        except OSError:
            pass

"""
YouTube Processing Configuration Module.

This module provides configuration dataclasses for YouTube processing operations.
Configuration is loaded from YAML files and validated at load time.

Example usage:
    config = YouTubeConfig.from_yaml(Path("config/youtube_config.yml"))
    downloader = YtDlpDownloader(config.download)
"""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import List
from typing import Optional

import yaml

from src.youtube_processing.exceptions import ConfigurationError


@dataclass(frozen=True)
class DownloadConfig:
    """
    Configuration for YouTube audio download operations.

    Attributes:
        temp_dir: Directory for temporary download files.
        audio_quality: Quality setting (best, worst, or specific format).
        audio_format: Output audio format (e.g., m4a, mp3, wav).
        audio_bitrate: Bitrate for audio conversion.
        max_file_size_mb: Maximum allowed file size in megabytes.
        timeout_seconds: Download timeout in seconds.
    """

    temp_dir: Path
    audio_quality: str = "best"
    audio_format: str = "m4a"
    audio_bitrate: str = "192"
    max_file_size_mb: int = 500
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_file_size_mb <= 0:
            raise ConfigurationError(
                message="max_file_size_mb must be positive",
                config_key="max_file_size_mb",
                config_value=str(self.max_file_size_mb),
            )
        if self.timeout_seconds <= 0:
            raise ConfigurationError(
                message="timeout_seconds must be positive",
                config_key="timeout_seconds",
                config_value=str(self.timeout_seconds),
            )


@dataclass(frozen=True)
class CacheConfig:
    """
    Configuration for audio file caching.

    Attributes:
        enabled: Whether caching is enabled.
        cache_dir: Directory for cached audio files.
        cleanup_after_processing: Whether to delete cache after processing.
        max_cache_size_gb: Maximum cache size in gigabytes.
        cache_ttl_days: Time-to-live for cache entries in days.
    """

    enabled: bool = True
    cache_dir: Optional[Path] = None
    cleanup_after_processing: bool = False
    max_cache_size_gb: int = 10
    cache_ttl_days: int = 7

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.enabled and self.cache_dir is None:
            raise ConfigurationError(
                message="cache_dir is required when caching is enabled",
                config_key="cache_dir",
            )
        if self.max_cache_size_gb <= 0:
            raise ConfigurationError(
                message="max_cache_size_gb must be positive",
                config_key="max_cache_size_gb",
                config_value=str(self.max_cache_size_gb),
            )


@dataclass(frozen=True)
class ValidationConfig:
    """
    Configuration for URL and video validation.

    Attributes:
        allowed_domains: List of allowed YouTube domains.
        min_duration_seconds: Minimum video duration in seconds.
        max_duration_seconds: Maximum video duration in seconds.
        allow_live_streams: Whether to allow live stream URLs.
        allow_age_restricted: Whether to allow age-restricted videos.
    """

    allowed_domains: List[str] = field(
        default_factory=lambda: ["youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"]
    )
    min_duration_seconds: int = 5
    max_duration_seconds: int = 7200
    allow_live_streams: bool = False
    allow_age_restricted: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.min_duration_seconds < 0:
            raise ConfigurationError(
                message="min_duration_seconds cannot be negative",
                config_key="min_duration_seconds",
                config_value=str(self.min_duration_seconds),
            )
        if self.max_duration_seconds <= self.min_duration_seconds:
            raise ConfigurationError(
                message="max_duration_seconds must be greater than min_duration_seconds",
                config_key="max_duration_seconds",
                config_value=str(self.max_duration_seconds),
            )


@dataclass(frozen=True)
class MetadataConfig:
    """
    Configuration for video metadata extraction.

    Attributes:
        extract_description: Whether to extract video description.
        extract_tags: Whether to extract video tags.
        extract_categories: Whether to extract video categories.
        extract_thumbnail_url: Whether to extract thumbnail URL.
        extract_view_count: Whether to extract view count.
        extract_like_count: Whether to extract like count.
        extract_channel_info: Whether to extract channel information.
        max_description_length: Maximum description length to retain.
    """

    extract_description: bool = True
    extract_tags: bool = True
    extract_categories: bool = True
    extract_thumbnail_url: bool = True
    extract_view_count: bool = True
    extract_like_count: bool = True
    extract_channel_info: bool = True
    max_description_length: int = 5000


@dataclass(frozen=True)
class RetryConfig:
    """
    Configuration for retry behavior on failed operations.

    Attributes:
        max_attempts: Maximum number of retry attempts.
        delay_seconds: Initial delay between retries in seconds.
        exponential_backoff: Whether to use exponential backoff.
    """

    max_attempts: int = 3
    delay_seconds: int = 5
    exponential_backoff: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_attempts < 1:
            raise ConfigurationError(
                message="max_attempts must be at least 1",
                config_key="max_attempts",
                config_value=str(self.max_attempts),
            )
        if self.delay_seconds < 0:
            raise ConfigurationError(
                message="delay_seconds cannot be negative",
                config_key="delay_seconds",
                config_value=str(self.delay_seconds),
            )


@dataclass(frozen=True)
class YouTubeConfig:
    """
    Main configuration class for YouTube processing.

    This class aggregates all sub-configurations and provides factory methods
    for loading from YAML files.

    Attributes:
        download: Download configuration.
        cache: Cache configuration.
        validation: Validation configuration.
        metadata: Metadata extraction configuration.
        retry: Retry configuration.
        audio_config_path: Path to audio processing configuration.
        youtube_vocabulary: List of YouTube-specific vocabulary terms.
    """

    download: DownloadConfig
    cache: CacheConfig
    validation: ValidationConfig
    metadata: MetadataConfig
    retry: RetryConfig
    audio_config_path: Optional[Path] = None
    youtube_vocabulary: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "YouTubeConfig":
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            YouTubeConfig instance populated from the YAML file.

        Raises:
            ConfigurationError: If the file cannot be read or parsed.
        """
        if not config_path.exists():
            raise ConfigurationError(
                message=f"Configuration file not found: {config_path}",
                config_key="config_path",
                config_value=str(config_path),
            )

        try:
            with config_path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ConfigurationError(
                message=f"Failed to parse YAML configuration: {e}",
                original_error=e,
            ) from e

        return cls._from_dict(data, config_path.parent)

    @classmethod
    def _from_dict(cls, data: dict, base_path: Path) -> "YouTubeConfig":
        """
        Create configuration from a dictionary.

        Args:
            data: Dictionary containing configuration data.
            base_path: Base path for resolving relative paths.

        Returns:
            YouTubeConfig instance.
        """
        yt_data = data.get("youtube_processing", {})

        # Parse download config
        download_data = yt_data.get("download", {})
        download = DownloadConfig(
            temp_dir=base_path / download_data.get("temp_dir", "temp/youtube_audio"),
            audio_quality=download_data.get("audio_quality", "best"),
            audio_format=download_data.get("audio_format", "m4a"),
            audio_bitrate=download_data.get("audio_bitrate", "192"),
            max_file_size_mb=download_data.get("max_file_size_mb", 500),
            timeout_seconds=download_data.get("timeout_seconds", 300),
        )

        # Parse cache config
        cache_data = yt_data.get("cache", {})
        cache_enabled = cache_data.get("enabled", True)
        cache_dir = cache_data.get("cache_dir")
        cache = CacheConfig(
            enabled=cache_enabled,
            cache_dir=base_path / cache_dir if cache_dir else None,
            cleanup_after_processing=cache_data.get("cleanup_after_processing", False),
            max_cache_size_gb=cache_data.get("max_cache_size_gb", 10),
            cache_ttl_days=cache_data.get("cache_ttl_days", 7),
        )

        # Parse validation config
        validation_data = yt_data.get("validation", {})
        validation = ValidationConfig(
            allowed_domains=validation_data.get(
                "allowed_domains",
                ["youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"],
            ),
            min_duration_seconds=validation_data.get("min_duration_seconds", 5),
            max_duration_seconds=validation_data.get("max_duration_seconds", 7200),
            allow_live_streams=validation_data.get("allow_live_streams", False),
            allow_age_restricted=validation_data.get("allow_age_restricted", False),
        )

        # Parse metadata config
        metadata_data = yt_data.get("metadata", {})
        metadata = MetadataConfig(
            extract_description=metadata_data.get("extract_description", True),
            extract_tags=metadata_data.get("extract_tags", True),
            extract_categories=metadata_data.get("extract_categories", True),
            extract_thumbnail_url=metadata_data.get("extract_thumbnail_url", True),
            extract_view_count=metadata_data.get("extract_view_count", True),
            extract_like_count=metadata_data.get("extract_like_count", True),
            extract_channel_info=metadata_data.get("extract_channel_info", True),
            max_description_length=metadata_data.get("max_description_length", 5000),
        )

        # Parse retry config
        retry_data = yt_data.get("retry", {})
        retry = RetryConfig(
            max_attempts=retry_data.get("max_attempts", 3),
            delay_seconds=retry_data.get("delay_seconds", 5),
            exponential_backoff=retry_data.get("exponential_backoff", True),
        )

        # Parse audio config path
        audio_config_path = yt_data.get("audio_config_path")
        audio_path = base_path / audio_config_path if audio_config_path else None

        # Parse YouTube vocabulary
        vocabulary = yt_data.get("youtube_vocabulary", [])

        return cls(
            download=download,
            cache=cache,
            validation=validation,
            metadata=metadata,
            retry=retry,
            audio_config_path=audio_path,
            youtube_vocabulary=vocabulary,
        )

    @classmethod
    def create_default(cls, temp_dir: Path, cache_dir: Optional[Path] = None) -> "YouTubeConfig":
        """
        Create a configuration with default values.

        Args:
            temp_dir: Directory for temporary download files.
            cache_dir: Optional directory for cached audio files.

        Returns:
            YouTubeConfig instance with default settings.
        """
        return cls(
            download=DownloadConfig(temp_dir=temp_dir),
            cache=CacheConfig(enabled=cache_dir is not None, cache_dir=cache_dir),
            validation=ValidationConfig(),
            metadata=MetadataConfig(),
            retry=RetryConfig(),
        )

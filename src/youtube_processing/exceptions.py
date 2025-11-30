"""
YouTube Processing Exceptions Module.

This module defines the exception hierarchy for all YouTube processing operations.
All exceptions inherit from YouTubeProcessingError to allow broad exception handling.

Exception Hierarchy:
    YouTubeProcessingError (base)
    ├── VideoNotFoundError (video does not exist or is unavailable)
    ├── DownloadError (failed to download audio)
    ├── CacheError (cache operation failed)
    ├── ValidationError (input validation failed)
    ├── MetadataError (metadata extraction failed)
    └── ConfigurationError (invalid configuration)
"""

from typing import Optional


class YouTubeProcessingError(Exception):
    """
    Base exception for all YouTube processing operations.

    All exceptions in this module inherit from this class, allowing callers
    to catch all YouTube-related errors with a single except clause.

    Attributes:
        message: Human-readable error description.
        video_url: Optional URL of the video that caused the error.
        original_error: Optional underlying exception that caused this error.
    """

    def __init__(
        self,
        message: str,
        video_url: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize YouTubeProcessingError.

        Args:
            message: Human-readable error description.
            video_url: Optional URL of the video that caused the error.
            original_error: Optional underlying exception that caused this error.
        """
        self.message = message
        self.video_url = video_url
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the complete error message."""
        parts = [self.message]
        if self.video_url:
            parts.append(f"URL: {self.video_url}")
        if self.original_error:
            parts.append(f"Caused by: {type(self.original_error).__name__}: {self.original_error}")
        return " | ".join(parts)


class VideoNotFoundError(YouTubeProcessingError):
    """
    Raised when a video does not exist or is unavailable.

    This can occur when:
    - The video URL is invalid or malformed
    - The video has been deleted
    - The video is private or age-restricted
    - The video is not available in the current region
    """

    def __init__(
        self,
        message: str = "Video not found or unavailable",
        video_url: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize VideoNotFoundError.

        Args:
            message: Human-readable error description.
            video_url: Optional URL of the video that was not found.
            original_error: Optional underlying exception.
        """
        super().__init__(message, video_url, original_error)


class DownloadError(YouTubeProcessingError):
    """
    Raised when audio download fails.

    This can occur when:
    - Network connectivity issues
    - Download timeout
    - File system write failures
    - yt-dlp internal errors
    """

    def __init__(
        self,
        message: str = "Failed to download audio",
        video_url: Optional[str] = None,
        original_error: Optional[Exception] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Initialize DownloadError.

        Args:
            message: Human-readable error description.
            video_url: Optional URL of the video that failed to download.
            original_error: Optional underlying exception.
            output_path: Optional path where the download was attempted.
        """
        self.output_path = output_path
        super().__init__(message, video_url, original_error)


class CacheError(YouTubeProcessingError):
    """
    Raised when a cache operation fails.

    This can occur when:
    - Cache directory is not writable
    - Cache is corrupted
    - Cache entry not found
    - Cache cleanup fails
    """

    def __init__(
        self,
        message: str = "Cache operation failed",
        video_url: Optional[str] = None,
        original_error: Optional[Exception] = None,
        cache_path: Optional[str] = None,
    ) -> None:
        """
        Initialize CacheError.

        Args:
            message: Human-readable error description.
            video_url: Optional URL of the video related to the cache operation.
            original_error: Optional underlying exception.
            cache_path: Optional path to the cache location.
        """
        self.cache_path = cache_path
        super().__init__(message, video_url, original_error)


class ValidationError(YouTubeProcessingError):
    """
    Raised when input validation fails.

    This can occur when:
    - URL is not a valid YouTube URL
    - Video duration exceeds configured limits
    - Video is a live stream when not allowed
    - Video is age-restricted when not allowed
    """

    def __init__(
        self,
        message: str = "Validation failed",
        video_url: Optional[str] = None,
        original_error: Optional[Exception] = None,
        field_name: Optional[str] = None,
        field_value: Optional[str] = None,
    ) -> None:
        """
        Initialize ValidationError.

        Args:
            message: Human-readable error description.
            video_url: Optional URL that failed validation.
            original_error: Optional underlying exception.
            field_name: Optional name of the field that failed validation.
            field_value: Optional value that failed validation.
        """
        self.field_name = field_name
        self.field_value = field_value
        super().__init__(message, video_url, original_error)


class MetadataError(YouTubeProcessingError):
    """
    Raised when metadata extraction fails.

    This can occur when:
    - Video metadata is incomplete
    - Metadata parsing fails
    - Required metadata fields are missing
    """

    def __init__(
        self,
        message: str = "Metadata extraction failed",
        video_url: Optional[str] = None,
        original_error: Optional[Exception] = None,
        missing_fields: Optional[list] = None,
    ) -> None:
        """
        Initialize MetadataError.

        Args:
            message: Human-readable error description.
            video_url: Optional URL of the video.
            original_error: Optional underlying exception.
            missing_fields: Optional list of missing metadata fields.
        """
        self.missing_fields = missing_fields or []
        super().__init__(message, video_url, original_error)


class ConfigurationError(YouTubeProcessingError):
    """
    Raised when configuration is invalid.

    This can occur when:
    - Configuration file is missing or malformed
    - Required configuration values are missing
    - Configuration values are out of valid range
    """

    def __init__(
        self,
        message: str = "Invalid configuration",
        video_url: Optional[str] = None,
        original_error: Optional[Exception] = None,
        config_key: Optional[str] = None,
        config_value: Optional[str] = None,
    ) -> None:
        """
        Initialize ConfigurationError.

        Args:
            message: Human-readable error description.
            video_url: Optional URL (usually None for config errors).
            original_error: Optional underlying exception.
            config_key: Optional configuration key that is invalid.
            config_value: Optional invalid configuration value.
        """
        self.config_key = config_key
        self.config_value = config_value
        super().__init__(message, video_url, original_error)

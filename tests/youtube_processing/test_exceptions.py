"""
Tests for YouTube Processing Exceptions.

This module tests the exception hierarchy and error formatting.
"""

import pytest

from src.youtube_processing.exceptions import CacheError
from src.youtube_processing.exceptions import ConfigurationError
from src.youtube_processing.exceptions import DownloadError
from src.youtube_processing.exceptions import MetadataError
from src.youtube_processing.exceptions import ValidationError
from src.youtube_processing.exceptions import VideoNotFoundError
from src.youtube_processing.exceptions import YouTubeProcessingError


class TestYouTubeProcessingError:
    """Tests for the base YouTubeProcessingError class."""

    def test_basic_error_message(self) -> None:
        """Test creating an error with just a message."""
        error = YouTubeProcessingError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.video_url is None
        assert error.original_error is None

    def test_error_with_video_url(self) -> None:
        """Test creating an error with a video URL."""
        error = YouTubeProcessingError(
            message="Processing failed",
            video_url="https://youtube.com/watch?v=abc123",
        )

        assert "Processing failed" in str(error)
        assert "URL: https://youtube.com/watch?v=abc123" in str(error)
        assert error.video_url == "https://youtube.com/watch?v=abc123"

    def test_error_with_original_exception(self) -> None:
        """Test creating an error with an original exception."""
        original = ValueError("Original error")
        error = YouTubeProcessingError(
            message="Wrapper error",
            original_error=original,
        )

        assert "Wrapper error" in str(error)
        assert "Caused by: ValueError: Original error" in str(error)
        assert error.original_error is original

    def test_error_with_all_fields(self) -> None:
        """Test creating an error with all fields."""
        original = OSError("Network failed")
        error = YouTubeProcessingError(
            message="Complete failure",
            video_url="https://youtube.com/watch?v=xyz789",
            original_error=original,
        )

        error_str = str(error)
        assert "Complete failure" in error_str
        assert "URL: https://youtube.com/watch?v=xyz789" in error_str
        assert "Caused by: OSError: Network failed" in error_str

    def test_error_inheritance(self) -> None:
        """Test that YouTubeProcessingError inherits from Exception."""
        error = YouTubeProcessingError("Test")
        assert isinstance(error, Exception)


class TestVideoNotFoundError:
    """Tests for VideoNotFoundError."""

    def test_default_message(self) -> None:
        """Test VideoNotFoundError with default message."""
        error = VideoNotFoundError()
        assert "Video not found or unavailable" in str(error)

    def test_custom_message(self) -> None:
        """Test VideoNotFoundError with custom message."""
        error = VideoNotFoundError(
            message="Video was deleted",
            video_url="https://youtube.com/watch?v=deleted",
        )

        assert "Video was deleted" in str(error)
        assert error.video_url == "https://youtube.com/watch?v=deleted"

    def test_inheritance(self) -> None:
        """Test that VideoNotFoundError inherits from YouTubeProcessingError."""
        error = VideoNotFoundError()
        assert isinstance(error, YouTubeProcessingError)


class TestDownloadError:
    """Tests for DownloadError."""

    def test_default_message(self) -> None:
        """Test DownloadError with default message."""
        error = DownloadError()
        assert "Failed to download audio" in str(error)

    def test_with_output_path(self) -> None:
        """Test DownloadError with output path."""
        error = DownloadError(
            message="Download timeout",
            output_path="/tmp/failed_download.m4a",
        )

        assert error.output_path == "/tmp/failed_download.m4a"

    def test_inheritance(self) -> None:
        """Test that DownloadError inherits from YouTubeProcessingError."""
        error = DownloadError()
        assert isinstance(error, YouTubeProcessingError)


class TestCacheError:
    """Tests for CacheError."""

    def test_default_message(self) -> None:
        """Test CacheError with default message."""
        error = CacheError()
        assert "Cache operation failed" in str(error)

    def test_with_cache_path(self) -> None:
        """Test CacheError with cache path."""
        error = CacheError(
            message="Cache full",
            cache_path="/cache/youtube",
        )

        assert error.cache_path == "/cache/youtube"

    def test_inheritance(self) -> None:
        """Test that CacheError inherits from YouTubeProcessingError."""
        error = CacheError()
        assert isinstance(error, YouTubeProcessingError)


class TestValidationError:
    """Tests for ValidationError."""

    def test_default_message(self) -> None:
        """Test ValidationError with default message."""
        error = ValidationError()
        assert "Validation failed" in str(error)

    def test_with_field_info(self) -> None:
        """Test ValidationError with field information."""
        error = ValidationError(
            message="Invalid duration",
            field_name="duration",
            field_value="10000",
        )

        assert error.field_name == "duration"
        assert error.field_value == "10000"

    def test_inheritance(self) -> None:
        """Test that ValidationError inherits from YouTubeProcessingError."""
        error = ValidationError()
        assert isinstance(error, YouTubeProcessingError)


class TestMetadataError:
    """Tests for MetadataError."""

    def test_default_message(self) -> None:
        """Test MetadataError with default message."""
        error = MetadataError()
        assert "Metadata extraction failed" in str(error)

    def test_with_missing_fields(self) -> None:
        """Test MetadataError with missing fields list."""
        error = MetadataError(
            message="Required fields missing",
            missing_fields=["title", "channel"],
        )

        assert error.missing_fields == ["title", "channel"]

    def test_empty_missing_fields(self) -> None:
        """Test MetadataError with no missing fields specified."""
        error = MetadataError()
        assert error.missing_fields == []

    def test_inheritance(self) -> None:
        """Test that MetadataError inherits from YouTubeProcessingError."""
        error = MetadataError()
        assert isinstance(error, YouTubeProcessingError)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_default_message(self) -> None:
        """Test ConfigurationError with default message."""
        error = ConfigurationError()
        assert "Invalid configuration" in str(error)

    def test_with_config_info(self) -> None:
        """Test ConfigurationError with config key and value."""
        error = ConfigurationError(
            message="Invalid value",
            config_key="timeout",
            config_value="-1",
        )

        assert error.config_key == "timeout"
        assert error.config_value == "-1"

    def test_inheritance(self) -> None:
        """Test that ConfigurationError inherits from YouTubeProcessingError."""
        error = ConfigurationError()
        assert isinstance(error, YouTubeProcessingError)


class TestExceptionHierarchy:
    """Tests for the exception hierarchy as a whole."""

    def test_all_exceptions_catchable_by_base(self) -> None:
        """Test that all exceptions can be caught by base class."""
        exceptions = [
            VideoNotFoundError("test"),
            DownloadError("test"),
            CacheError("test"),
            ValidationError("test"),
            MetadataError("test"),
            ConfigurationError("test"),
        ]

        for exc in exceptions:
            try:
                raise exc
            except YouTubeProcessingError:
                pass  # Should be caught
            except Exception:
                pytest.fail(f"{type(exc).__name__} was not caught by YouTubeProcessingError")

    def test_specific_exceptions_distinguishable(self) -> None:
        """Test that specific exceptions can be caught separately."""
        with pytest.raises(VideoNotFoundError):
            raise VideoNotFoundError()

        with pytest.raises(DownloadError):
            raise DownloadError()

        with pytest.raises(CacheError):
            raise CacheError()

        with pytest.raises(ValidationError):
            raise ValidationError()

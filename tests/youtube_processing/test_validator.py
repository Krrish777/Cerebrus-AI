"""
Tests for YouTube URL Validator.

This module tests URL validation and video ID extraction.
"""

import pytest

from src.youtube_processing.config import ValidationConfig
from src.youtube_processing.download.validator import YouTubeURLValidator
from src.youtube_processing.exceptions import ValidationError
from src.youtube_processing.interfaces import VideoMetadata


class TestYouTubeURLValidator:
    """Tests for YouTubeURLValidator."""

    @pytest.fixture
    def validator(self) -> YouTubeURLValidator:
        """Create a validator with default config."""
        config = ValidationConfig()
        return YouTubeURLValidator(config)

    @pytest.fixture
    def strict_validator(self) -> YouTubeURLValidator:
        """Create a validator with strict config."""
        config = ValidationConfig(
            allowed_domains=["youtube.com", "www.youtube.com"],
            min_duration_seconds=60,
            max_duration_seconds=1800,
            allow_live_streams=False,
            allow_age_restricted=False,
        )
        return YouTubeURLValidator(config)

    class TestExtractVideoId:
        """Tests for extract_video_id method."""

        def test_standard_watch_url(self, validator: YouTubeURLValidator) -> None:
            """Test extracting ID from standard watch URL."""
            url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            video_id = validator.extract_video_id(url)
            assert video_id == "dQw4w9WgXcQ"

        def test_short_url(self, validator: YouTubeURLValidator) -> None:
            """Test extracting ID from short URL."""
            url = "https://youtu.be/dQw4w9WgXcQ"
            video_id = validator.extract_video_id(url)
            assert video_id == "dQw4w9WgXcQ"

        def test_embed_url(self, validator: YouTubeURLValidator) -> None:
            """Test extracting ID from embed URL."""
            url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
            video_id = validator.extract_video_id(url)
            assert video_id == "dQw4w9WgXcQ"

        def test_shorts_url(self, validator: YouTubeURLValidator) -> None:
            """Test extracting ID from shorts URL."""
            url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
            video_id = validator.extract_video_id(url)
            assert video_id == "dQw4w9WgXcQ"

        def test_live_url(self, validator: YouTubeURLValidator) -> None:
            """Test extracting ID from live URL."""
            url = "https://www.youtube.com/live/dQw4w9WgXcQ"
            video_id = validator.extract_video_id(url)
            assert video_id == "dQw4w9WgXcQ"

        def test_url_with_timestamp(self, validator: YouTubeURLValidator) -> None:
            """Test extracting ID from URL with timestamp."""
            url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120"
            video_id = validator.extract_video_id(url)
            assert video_id == "dQw4w9WgXcQ"

        def test_url_with_playlist(self, validator: YouTubeURLValidator) -> None:
            """Test extracting ID from URL with playlist."""
            url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLxxxxxx"
            video_id = validator.extract_video_id(url)
            assert video_id == "dQw4w9WgXcQ"

        def test_mobile_url(self, validator: YouTubeURLValidator) -> None:
            """Test extracting ID from mobile URL."""
            url = "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
            video_id = validator.extract_video_id(url)
            assert video_id == "dQw4w9WgXcQ"

        def test_invalid_url_returns_none(self, validator: YouTubeURLValidator) -> None:
            """Test that invalid URL returns None."""
            url = "https://example.com/video"
            video_id = validator.extract_video_id(url)
            assert video_id is None

        def test_malformed_url_returns_none(self, validator: YouTubeURLValidator) -> None:
            """Test that malformed URL returns None."""
            url = "not-a-url"
            video_id = validator.extract_video_id(url)
            assert video_id is None

    class TestValidate:
        """Tests for validate method."""

        def test_valid_url_without_metadata(self, validator: YouTubeURLValidator) -> None:
            """Test validating a URL without metadata."""
            url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            is_valid, errors = validator.validate(url)
            assert is_valid is True
            assert errors == []

        def test_invalid_domain(self, validator: YouTubeURLValidator) -> None:
            """Test that non-YouTube domain fails validation."""
            url = "https://vimeo.com/video/123456"
            is_valid, errors = validator.validate(url)
            assert is_valid is False
            assert any("Domain not in allowed list" in e for e in errors)

        def test_invalid_format(self, validator: YouTubeURLValidator) -> None:
            """Test that invalid URL format fails validation."""
            url = "not-a-valid-url"
            is_valid, errors = validator.validate(url)
            assert is_valid is False
            assert any("Invalid URL format" in e for e in errors)

        def test_valid_url_with_valid_metadata(
            self, validator: YouTubeURLValidator
        ) -> None:
            """Test validating URL with valid metadata."""
            url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            metadata = VideoMetadata(
                video_id="dQw4w9WgXcQ",
                title="Test Video",
                duration_seconds=180,
                is_live=False,
                is_age_restricted=False,
            )
            is_valid, errors = validator.validate(url, metadata)
            assert is_valid is True
            assert errors == []

        def test_video_too_short(self, strict_validator: YouTubeURLValidator) -> None:
            """Test that video shorter than minimum duration fails."""
            url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            metadata = VideoMetadata(
                video_id="dQw4w9WgXcQ",
                title="Short Video",
                duration_seconds=30,  # Less than 60 second minimum
            )
            is_valid, errors = strict_validator.validate(url, metadata)
            assert is_valid is False
            assert any("too short" in e for e in errors)

        def test_video_too_long(self, strict_validator: YouTubeURLValidator) -> None:
            """Test that video longer than maximum duration fails."""
            url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            metadata = VideoMetadata(
                video_id="dQw4w9WgXcQ",
                title="Long Video",
                duration_seconds=3600,  # More than 1800 second maximum
            )
            is_valid, errors = strict_validator.validate(url, metadata)
            assert is_valid is False
            assert any("too long" in e for e in errors)

        def test_live_stream_not_allowed(
            self, strict_validator: YouTubeURLValidator
        ) -> None:
            """Test that live streams fail when not allowed."""
            url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            metadata = VideoMetadata(
                video_id="dQw4w9WgXcQ",
                title="Live Stream",
                duration_seconds=0,
                is_live=True,
            )
            is_valid, errors = strict_validator.validate(url, metadata)
            assert is_valid is False
            assert any("Live streams are not allowed" in e for e in errors)

        def test_age_restricted_not_allowed(
            self, strict_validator: YouTubeURLValidator
        ) -> None:
            """Test that age-restricted videos fail when not allowed."""
            url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            metadata = VideoMetadata(
                video_id="dQw4w9WgXcQ",
                title="Age Restricted Video",
                duration_seconds=300,
                is_age_restricted=True,
            )
            is_valid, errors = strict_validator.validate(url, metadata)
            assert is_valid is False
            assert any("Age-restricted videos are not allowed" in e for e in errors)

    class TestNormalizeUrl:
        """Tests for normalize_url method."""

        def test_normalize_standard_url(self, validator: YouTubeURLValidator) -> None:
            """Test normalizing a standard URL."""
            url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120"
            normalized = validator.normalize_url(url)
            assert normalized == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        def test_normalize_short_url(self, validator: YouTubeURLValidator) -> None:
            """Test normalizing a short URL."""
            url = "https://youtu.be/dQw4w9WgXcQ"
            normalized = validator.normalize_url(url)
            assert normalized == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        def test_normalize_embed_url(self, validator: YouTubeURLValidator) -> None:
            """Test normalizing an embed URL."""
            url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
            normalized = validator.normalize_url(url)
            assert normalized == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        def test_normalize_invalid_url_raises_error(
            self, validator: YouTubeURLValidator
        ) -> None:
            """Test that normalizing invalid URL raises ValidationError."""
            url = "https://example.com/video"
            with pytest.raises(ValidationError) as exc_info:
                validator.normalize_url(url)
            assert "video ID not found" in str(exc_info.value)

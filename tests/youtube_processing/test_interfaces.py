"""
Tests for YouTube Processing Interfaces and Data Classes.

This module tests the VideoMetadata dataclass and other interface-related
functionality.
"""

import pytest

from src.youtube_processing.interfaces import DownloadResult
from src.youtube_processing.interfaces import VideoMetadata
from pathlib import Path


class TestVideoMetadata:
    """Tests for the VideoMetadata dataclass."""

    def test_video_metadata_creation_with_required_fields(self) -> None:
        """Test creating VideoMetadata with only required fields."""
        metadata = VideoMetadata(
            video_id="test123",
            title="Test Video",
        )

        assert metadata.video_id == "test123"
        assert metadata.title == "Test Video"
        assert metadata.description == ""
        assert metadata.channel_name == ""
        assert metadata.duration_seconds == 0
        assert metadata.tags == []
        assert metadata.categories == []
        assert metadata.is_live is False
        assert metadata.is_age_restricted is False

    def test_video_metadata_creation_with_all_fields(self) -> None:
        """Test creating VideoMetadata with all fields."""
        metadata = VideoMetadata(
            video_id="abc123xyz",
            title="Complete Test Video",
            description="A test video description",
            channel_name="Test Channel",
            channel_id="UC123456",
            duration_seconds=3600,
            upload_date="2024-01-15",
            view_count=1000000,
            like_count=50000,
            tags=["test", "video", "sample"],
            categories=["Education"],
            thumbnail_url="https://example.com/thumb.jpg",
            is_live=False,
            is_age_restricted=False,
            language="en",
            extra={"custom_field": "custom_value"},
        )

        assert metadata.video_id == "abc123xyz"
        assert metadata.title == "Complete Test Video"
        assert metadata.description == "A test video description"
        assert metadata.channel_name == "Test Channel"
        assert metadata.channel_id == "UC123456"
        assert metadata.duration_seconds == 3600
        assert metadata.upload_date == "2024-01-15"
        assert metadata.view_count == 1000000
        assert metadata.like_count == 50000
        assert metadata.tags == ["test", "video", "sample"]
        assert metadata.categories == ["Education"]
        assert metadata.thumbnail_url == "https://example.com/thumb.jpg"
        assert metadata.is_live is False
        assert metadata.is_age_restricted is False
        assert metadata.language == "en"
        assert metadata.extra == {"custom_field": "custom_value"}

    def test_video_metadata_to_dict(self) -> None:
        """Test converting VideoMetadata to dictionary."""
        metadata = VideoMetadata(
            video_id="test123",
            title="Test Video",
            tags=["tag1", "tag2"],
            extra={"extra_key": "extra_value"},
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert result["video_id"] == "test123"
        assert result["title"] == "Test Video"
        assert result["tags"] == ["tag1", "tag2"]
        assert result["extra_key"] == "extra_value"  # Extra fields merged into dict

    def test_video_metadata_with_none_view_count(self) -> None:
        """Test that view_count can be None."""
        metadata = VideoMetadata(
            video_id="test123",
            title="Test Video",
            view_count=None,
            like_count=None,
        )

        assert metadata.view_count is None
        assert metadata.like_count is None

    def test_video_metadata_with_live_stream(self) -> None:
        """Test metadata for a live stream."""
        metadata = VideoMetadata(
            video_id="live123",
            title="Live Stream",
            is_live=True,
            duration_seconds=0,
        )

        assert metadata.is_live is True
        assert metadata.duration_seconds == 0

    def test_video_metadata_with_age_restriction(self) -> None:
        """Test metadata for an age-restricted video."""
        metadata = VideoMetadata(
            video_id="restricted123",
            title="Age Restricted Video",
            is_age_restricted=True,
        )

        assert metadata.is_age_restricted is True


class TestDownloadResult:
    """Tests for the DownloadResult dataclass."""

    def test_download_result_creation(self, tmp_path: Path) -> None:
        """Test creating a DownloadResult."""
        audio_path = tmp_path / "test.m4a"
        audio_path.touch()

        metadata = VideoMetadata(
            video_id="test123",
            title="Test Video",
        )

        result = DownloadResult(
            audio_path=audio_path,
            metadata=metadata,
            file_size_bytes=1024,
            download_duration_seconds=5.5,
            from_cache=False,
        )

        assert result.audio_path == audio_path
        assert result.metadata.video_id == "test123"
        assert result.file_size_bytes == 1024
        assert result.download_duration_seconds == 5.5
        assert result.from_cache is False

    def test_download_result_from_cache(self, tmp_path: Path) -> None:
        """Test DownloadResult with from_cache=True."""
        audio_path = tmp_path / "cached.m4a"
        audio_path.touch()

        metadata = VideoMetadata(
            video_id="cached123",
            title="Cached Video",
        )

        result = DownloadResult(
            audio_path=audio_path,
            metadata=metadata,
            from_cache=True,
        )

        assert result.from_cache is True
        assert result.file_size_bytes == 0  # Default value
        assert result.download_duration_seconds == 0.0  # Default value

    def test_download_result_defaults(self, tmp_path: Path) -> None:
        """Test DownloadResult default values."""
        audio_path = tmp_path / "test.m4a"
        audio_path.touch()

        metadata = VideoMetadata(
            video_id="test123",
            title="Test Video",
        )

        result = DownloadResult(
            audio_path=audio_path,
            metadata=metadata,
        )

        assert result.file_size_bytes == 0
        assert result.download_duration_seconds == 0.0
        assert result.from_cache is False

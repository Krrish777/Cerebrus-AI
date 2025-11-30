"""
Unit tests for YtDlpDownloader.

Tests the video downloader implementation with mocked yt-dlp dependency.
"""

from pathlib import Path
from typing import Any
from typing import Dict
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from src.youtube_processing.config import DownloadConfig
from src.youtube_processing.config import RetryConfig
from src.youtube_processing.download.downloader import YtDlpDownloader
from src.youtube_processing.download.validator import YouTubeURLValidator
from src.youtube_processing.exceptions import DownloadError
from src.youtube_processing.exceptions import MetadataError
from src.youtube_processing.exceptions import VideoNotFoundError


@pytest.fixture
def download_config(tmp_path: Path) -> DownloadConfig:
    """Create a download configuration for testing."""
    return DownloadConfig(
        temp_dir=tmp_path / "downloads",
        audio_quality="best",
        audio_format="mp3",
        audio_bitrate="192",
        max_file_size_mb=500,
        timeout_seconds=300,
    )


@pytest.fixture
def retry_config() -> RetryConfig:
    """Create a retry configuration for testing."""
    return RetryConfig(
        max_attempts=3,
        delay_seconds=1,
        exponential_backoff=True,
    )


@pytest.fixture
def downloader(download_config: DownloadConfig, retry_config: RetryConfig) -> YtDlpDownloader:
    """Create a YtDlpDownloader instance for testing."""
    return YtDlpDownloader(config=download_config, retry_config=retry_config)


@pytest.fixture
def mock_video_info() -> Dict[str, Any]:
    """Mock video info response from yt-dlp."""
    return {
        "id": "dQw4w9WgXcQ",
        "title": "Test Video Title",
        "description": "Test video description",
        "duration": 212,
        "channel": "Test Channel",
        "channel_id": "UC123456789",
        "upload_date": "20230101",
        "view_count": 1000000,
        "like_count": 50000,
        "thumbnail": "https://example.com/thumb.jpg",
        "categories": ["Music"],
        "tags": ["test", "video"],
        "uploader": "Test Uploader",
        "uploader_id": "testuploader",
        "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "is_live": False,
        "age_limit": 0,
        "language": "en",
        "channel_url": "https://www.youtube.com/channel/UC123456789",
        "availability": "public",
    }


class TestYtDlpDownloaderInit:
    """Tests for YtDlpDownloader initialization."""

    def test_stores_configuration(
        self, download_config: DownloadConfig, retry_config: RetryConfig
    ) -> None:
        """Test that configuration is stored correctly."""
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)
        assert downloader._config == download_config
        assert downloader._retry_config == retry_config

    def test_uses_default_retry_config(self, download_config: DownloadConfig) -> None:
        """Test that default retry config is used if not provided."""
        downloader = YtDlpDownloader(config=download_config)
        assert downloader._retry_config is not None
        assert downloader._retry_config.max_attempts == 3

    def test_accepts_custom_validator(
        self, download_config: DownloadConfig, retry_config: RetryConfig
    ) -> None:
        """Test that custom validator can be injected."""
        mock_validator = Mock(spec=YouTubeURLValidator)
        downloader = YtDlpDownloader(
            config=download_config,
            retry_config=retry_config,
            validator=mock_validator,
        )
        assert downloader._validator == mock_validator


class TestYtDlpDownloaderExtractMetadata:
    """Tests for metadata extraction."""

    @patch("src.youtube_processing.download.downloader.yt_dlp.YoutubeDL")
    def test_extract_metadata_success(
        self,
        mock_ydl_class: Mock,
        download_config: DownloadConfig,
        retry_config: RetryConfig,
        mock_video_info: Dict[str, Any],
    ) -> None:
        """Test successful metadata extraction."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = mock_video_info
        mock_ydl.__enter__ = Mock(return_value=mock_ydl)
        mock_ydl.__exit__ = Mock(return_value=False)
        mock_ydl_class.return_value = mock_ydl

        # Create downloader inside the patch context
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        metadata = downloader.extract_metadata(url)

        assert metadata.video_id == "dQw4w9WgXcQ"
        assert metadata.title == "Test Video Title"
        assert metadata.description == "Test video description"
        assert metadata.duration_seconds == 212
        assert metadata.channel_name == "Test Channel"
        mock_ydl.extract_info.assert_called_once_with(url, download=False)

    @patch("src.youtube_processing.download.downloader.yt_dlp.YoutubeDL")
    def test_extract_metadata_video_not_found(
        self,
        mock_ydl_class: Mock,
        download_config: DownloadConfig,
        retry_config: RetryConfig,
    ) -> None:
        """Test metadata extraction when video is not found."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = None
        mock_ydl.__enter__ = Mock(return_value=mock_ydl)
        mock_ydl.__exit__ = Mock(return_value=False)
        mock_ydl_class.return_value = mock_ydl

        # Create downloader inside the patch context
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)

        url = "https://www.youtube.com/watch?v=invalid"

        with pytest.raises(VideoNotFoundError):
            downloader.extract_metadata(url)

    @patch("src.youtube_processing.download.downloader.yt_dlp.YoutubeDL")
    def test_extract_metadata_download_error(
        self,
        mock_ydl_class: Mock,
        download_config: DownloadConfig,
        retry_config: RetryConfig,
    ) -> None:
        """Test metadata extraction with download error."""
        import yt_dlp

        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("Network error")
        mock_ydl.__enter__ = Mock(return_value=mock_ydl)
        mock_ydl.__exit__ = Mock(return_value=False)
        mock_ydl_class.return_value = mock_ydl

        # Create downloader inside the patch context
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        with pytest.raises(MetadataError):
            downloader.extract_metadata(url)

    @patch("src.youtube_processing.download.downloader.yt_dlp.YoutubeDL")
    def test_extract_metadata_parses_upload_date(
        self,
        mock_ydl_class: Mock,
        download_config: DownloadConfig,
        retry_config: RetryConfig,
        mock_video_info: Dict[str, Any],
    ) -> None:
        """Test that upload date is properly formatted."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = mock_video_info
        mock_ydl.__enter__ = Mock(return_value=mock_ydl)
        mock_ydl.__exit__ = Mock(return_value=False)
        mock_ydl_class.return_value = mock_ydl

        # Create downloader inside the patch context
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        metadata = downloader.extract_metadata(url)

        # upload_date "20230101" should become "2023-01-01"
        assert metadata.upload_date == "2023-01-01"


class TestYtDlpDownloaderDownload:
    """Tests for video download functionality."""

    @patch("src.youtube_processing.download.downloader.yt_dlp.YoutubeDL")
    def test_download_success(
        self,
        mock_ydl_class: Mock,
        download_config: DownloadConfig,
        retry_config: RetryConfig,
        mock_video_info: Dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test successful video download."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create mock audio file
        audio_file = output_dir / "dQw4w9WgXcQ.mp3"

        # Setup mock
        mock_ydl = MagicMock()

        def extract_info_side_effect(url: str, download: bool = False) -> Dict[str, Any]:
            if download:
                # Simulate file creation during download
                audio_file.write_text("mock audio content")
            return mock_video_info

        mock_ydl.extract_info.side_effect = extract_info_side_effect
        mock_ydl.__enter__ = Mock(return_value=mock_ydl)
        mock_ydl.__exit__ = Mock(return_value=False)
        mock_ydl_class.return_value = mock_ydl

        # Create downloader inside patch context
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = downloader.download(url, output_dir)

        assert result.metadata.video_id == "dQw4w9WgXcQ"
        assert result.audio_path.exists()
        assert result.from_cache is False

    @patch("src.youtube_processing.download.downloader.yt_dlp.YoutubeDL")
    def test_download_creates_output_directory(
        self,
        mock_ydl_class: Mock,
        download_config: DownloadConfig,
        retry_config: RetryConfig,
        mock_video_info: Dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_directory"
        audio_file = output_dir / "dQw4w9WgXcQ.mp3"

        mock_ydl = MagicMock()

        def extract_info_side_effect(url: str, download: bool = False) -> Dict[str, Any]:
            if download:
                output_dir.mkdir(parents=True, exist_ok=True)
                audio_file.write_text("mock audio content")
            return mock_video_info

        mock_ydl.extract_info.side_effect = extract_info_side_effect
        mock_ydl.__enter__ = Mock(return_value=mock_ydl)
        mock_ydl.__exit__ = Mock(return_value=False)
        mock_ydl_class.return_value = mock_ydl

        # Create downloader inside patch context
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = downloader.download(url, output_dir)

        assert output_dir.exists()
        assert result.audio_path.exists()

    @patch("src.youtube_processing.download.downloader.yt_dlp.YoutubeDL")
    def test_download_exhausts_retries_on_persistent_failure(
        self,
        mock_ydl_class: Mock,
        download_config: DownloadConfig,
        mock_video_info: Dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that download fails after all retries are exhausted."""
        import yt_dlp

        output_dir = tmp_path / "output"

        # Use minimal retry config for faster tests
        retry_config = RetryConfig(max_attempts=2, delay_seconds=0, exponential_backoff=False)

        mock_ydl = MagicMock()

        # First call for extract_metadata succeeds, download calls always fail
        call_count = 0

        def extract_info_side_effect(url: str, download: bool = False) -> Dict[str, Any]:
            nonlocal call_count
            if not download:
                # Metadata extraction succeeds
                return mock_video_info
            # Download always fails
            call_count += 1
            raise yt_dlp.utils.DownloadError("Persistent error")

        mock_ydl.extract_info.side_effect = extract_info_side_effect
        mock_ydl.__enter__ = Mock(return_value=mock_ydl)
        mock_ydl.__exit__ = Mock(return_value=False)
        mock_ydl_class.return_value = mock_ydl

        # Create downloader inside patch context
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        # Should fail after retries are exhausted
        with pytest.raises(DownloadError) as exc_info:
            downloader.download(url, output_dir)

        assert "failed after" in str(exc_info.value).lower()


class TestYtDlpDownloaderValidateUrl:
    """Tests for URL validation."""

    def test_validate_url_with_injected_validator(
        self, download_config: DownloadConfig, retry_config: RetryConfig
    ) -> None:
        """Test validation with injected validator."""
        mock_validator = Mock(spec=YouTubeURLValidator)
        mock_validator.validate.return_value = (True, "dQw4w9WgXcQ")

        downloader = YtDlpDownloader(
            config=download_config,
            retry_config=retry_config,
            validator=mock_validator,
        )

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = downloader.validate_url(url)

        assert result is True
        mock_validator.validate.assert_called_once_with(url)

    def test_validate_url_with_injected_validator_returns_false(
        self, download_config: DownloadConfig, retry_config: RetryConfig
    ) -> None:
        """Test validation returns false for invalid URL."""
        mock_validator = Mock(spec=YouTubeURLValidator)
        mock_validator.validate.return_value = (False, None)

        downloader = YtDlpDownloader(
            config=download_config,
            retry_config=retry_config,
            validator=mock_validator,
        )

        url = "https://example.com/video"
        result = downloader.validate_url(url)

        assert result is False


class TestYtDlpDownloaderBuildOptions:
    """Tests for yt-dlp options building."""

    def test_build_options_includes_audio_format(
        self, download_config: DownloadConfig, retry_config: RetryConfig, tmp_path: Path
    ) -> None:
        """Test that options include audio format settings."""
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)
        output_dir = tmp_path / "output"
        options = downloader._build_download_options(output_dir, "test_video_id")

        assert "format" in options
        assert options.get("postprocessors") is not None
        assert len(options["postprocessors"]) > 0

    def test_build_options_includes_output_template(
        self, download_config: DownloadConfig, retry_config: RetryConfig, tmp_path: Path
    ) -> None:
        """Test that options include output template."""
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)
        output_dir = tmp_path / "output"
        options = downloader._build_download_options(output_dir, "test_video_id")

        assert "outtmpl" in options
        assert "test_video_id" in options["outtmpl"]

    def test_build_options_includes_timeout(
        self, download_config: DownloadConfig, retry_config: RetryConfig, tmp_path: Path
    ) -> None:
        """Test that options include socket timeout."""
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)
        output_dir = tmp_path / "output"
        options = downloader._build_download_options(output_dir, "test_video_id")

        assert "socket_timeout" in options
        assert options["socket_timeout"] == download_config.timeout_seconds


class TestYtDlpDownloaderSupportedFormats:
    """Tests for supported formats property."""

    def test_returns_common_audio_formats(
        self, download_config: DownloadConfig, retry_config: RetryConfig
    ) -> None:
        """Test that common audio formats are supported."""
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)
        formats = downloader.supported_formats

        assert "mp3" in formats
        assert "m4a" in formats
        assert "wav" in formats

    def test_supported_formats_is_list(
        self, download_config: DownloadConfig, retry_config: RetryConfig
    ) -> None:
        """Test that supported_formats returns a list."""
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)
        formats = downloader.supported_formats

        assert isinstance(formats, list)
        assert len(formats) > 0


class TestYtDlpDownloaderRetryDelay:
    """Tests for retry delay calculation."""

    def test_exponential_backoff(self, download_config: DownloadConfig) -> None:
        """Test exponential backoff calculation."""
        retry_config = RetryConfig(
            max_attempts=5,
            delay_seconds=2,
            exponential_backoff=True,
        )
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)

        delay1 = downloader._calculate_retry_delay(1)
        delay2 = downloader._calculate_retry_delay(2)
        delay3 = downloader._calculate_retry_delay(3)

        assert delay1 == 2  # 2 * 2^0
        assert delay2 == 4  # 2 * 2^1
        assert delay3 == 8  # 2 * 2^2

    def test_linear_backoff(self, download_config: DownloadConfig) -> None:
        """Test linear (non-exponential) delay."""
        retry_config = RetryConfig(
            max_attempts=5,
            delay_seconds=2,
            exponential_backoff=False,
        )
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)

        delay1 = downloader._calculate_retry_delay(1)
        delay2 = downloader._calculate_retry_delay(2)
        delay3 = downloader._calculate_retry_delay(3)

        assert delay1 == 2
        assert delay2 == 2
        assert delay3 == 2


class TestYtDlpDownloaderFormatUploadDate:
    """Tests for upload date formatting."""

    def test_formats_valid_date(
        self, download_config: DownloadConfig, retry_config: RetryConfig
    ) -> None:
        """Test formatting a valid YYYYMMDD date."""
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)

        result = downloader._format_upload_date("20231225")
        assert result == "2023-12-25"

    def test_returns_empty_for_none(
        self, download_config: DownloadConfig, retry_config: RetryConfig
    ) -> None:
        """Test that None returns empty string."""
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)

        result = downloader._format_upload_date(None)
        assert result == ""

    def test_returns_empty_for_invalid_length(
        self, download_config: DownloadConfig, retry_config: RetryConfig
    ) -> None:
        """Test that invalid length returns empty string."""
        downloader = YtDlpDownloader(config=download_config, retry_config=retry_config)

        result = downloader._format_upload_date("2023")
        assert result == ""

        result = downloader._format_upload_date("2023-12-25")
        assert result == ""

"""
YouTube Video Downloader Module.

This module provides the YtDlpDownloader class for downloading audio
from YouTube videos using the yt-dlp library.
"""

import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import yt_dlp

from src.core.logging import get_logger
from src.youtube_processing.config import DownloadConfig
from src.youtube_processing.config import RetryConfig
from src.youtube_processing.download.validator import YouTubeURLValidator
from src.youtube_processing.exceptions import DownloadError
from src.youtube_processing.exceptions import MetadataError
from src.youtube_processing.exceptions import VideoNotFoundError
from src.youtube_processing.interfaces import DownloadResult
from src.youtube_processing.interfaces import VideoDownloader
from src.youtube_processing.interfaces import VideoMetadata

logger = get_logger(__name__)


class YtDlpDownloader(VideoDownloader):
    """
    Video downloader implementation using yt-dlp.

    This class handles downloading audio from YouTube videos with support for:
    - Configurable audio quality and format
    - Retry logic with exponential backoff
    - Metadata extraction
    - File size limits
    - Download timeout

    Example:
        config = DownloadConfig(temp_dir=Path("temp"))
        retry_config = RetryConfig()
        downloader = YtDlpDownloader(config, retry_config)
        result = downloader.download(url, output_dir)
    """

    def __init__(
        self,
        config: DownloadConfig,
        retry_config: Optional[RetryConfig] = None,
        validator: Optional[YouTubeURLValidator] = None,
    ) -> None:
        """
        Initialize the downloader.

        Args:
            config: Download configuration.
            retry_config: Optional retry configuration (defaults provided).
            validator: Optional URL validator for video ID extraction.
        """
        self._config = config
        self._retry_config = retry_config or RetryConfig()
        self._validator = validator
        logger.info(
            "Initialized YtDlpDownloader with format=%s, quality=%s",
            config.audio_format,
            config.audio_quality,
        )

    def download(self, url: str, output_dir: Path) -> DownloadResult:
        """
        Download audio from a YouTube video.

        Args:
            url: YouTube video URL.
            output_dir: Directory to save the downloaded audio.

        Returns:
            DownloadResult containing the audio path and metadata.

        Raises:
            VideoNotFoundError: If the video does not exist.
            DownloadError: If the download fails.
            ValidationError: If the URL is invalid.
        """
        logger.info("Starting download for URL: %s", url)
        start_time = time.time()

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract metadata first
        metadata = self.extract_metadata(url)

        # Build yt-dlp options
        ydl_opts = self._build_download_options(output_dir, metadata.video_id)

        # Download with retry logic
        audio_path = self._download_with_retry(url, ydl_opts)

        download_duration = time.time() - start_time
        file_size = audio_path.stat().st_size if audio_path.exists() else 0

        logger.info(
            "Download completed: %s (%.2f MB in %.2f seconds)",
            audio_path.name,
            file_size / (1024 * 1024),
            download_duration,
        )

        return DownloadResult(
            audio_path=audio_path,
            metadata=metadata,
            file_size_bytes=file_size,
            download_duration_seconds=download_duration,
            from_cache=False,
        )

    def extract_metadata(self, url: str) -> VideoMetadata:
        """
        Extract metadata from a YouTube video without downloading.

        Args:
            url: YouTube video URL.

        Returns:
            VideoMetadata containing the extracted information.

        Raises:
            VideoNotFoundError: If the video does not exist.
            MetadataError: If metadata extraction fails.
        """
        logger.debug("Extracting metadata for URL: %s", url)

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info is None:
                    raise VideoNotFoundError(
                        message="Video not found or unavailable",
                        video_url=url,
                    )
                return self._parse_metadata(info)
        except VideoNotFoundError:
            # Re-raise VideoNotFoundError as-is
            raise
        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e).lower()
            if "video unavailable" in error_msg or "private video" in error_msg:
                raise VideoNotFoundError(
                    message="Video not found or is private",
                    video_url=url,
                    original_error=e,
                ) from e
            raise MetadataError(
                message=f"Failed to extract metadata: {e}",
                video_url=url,
                original_error=e,
            ) from e
        except Exception as e:
            raise MetadataError(
                message=f"Unexpected error extracting metadata: {e}",
                video_url=url,
                original_error=e,
            ) from e

    def validate_url(self, url: str) -> bool:
        """
        Validate a YouTube URL without downloading.

        Args:
            url: URL to validate.

        Returns:
            True if the URL is a valid YouTube video URL.
        """
        if self._validator:
            is_valid, _ = self._validator.validate(url)
            return is_valid

        # Basic validation without a validator instance
        try:
            ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info is not None
        except Exception:
            return False

    @property
    def supported_formats(self) -> List[str]:
        """Return list of supported audio formats."""
        return ["m4a", "mp3", "wav", "aac", "opus", "flac", "ogg"]

    def _build_download_options(
        self,
        output_dir: Path,
        video_id: str,
    ) -> Dict[str, Any]:
        """Build yt-dlp options dictionary."""
        output_template = str(output_dir / f"{video_id}.%(ext)s")

        return {
            "format": f"bestaudio[ext={self._config.audio_format}]/bestaudio/best",
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "socket_timeout": self._config.timeout_seconds,
            "retries": 3,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": self._config.audio_format,
                    "preferredquality": self._config.audio_bitrate,
                }
            ],
        }

    def _download_with_retry(
        self,
        url: str,
        ydl_opts: Dict[str, Any],
    ) -> Path:
        """Download with retry logic."""
        last_error: Optional[Exception] = None

        for attempt in range(1, self._retry_config.max_attempts + 1):
            try:
                logger.debug("Download attempt %d/%d", attempt, self._retry_config.max_attempts)
                return self._perform_download(url, ydl_opts)
            except (VideoNotFoundError, DownloadError) as e:
                last_error = e
                if attempt < self._retry_config.max_attempts:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        "Download attempt %d failed, retrying in %.1f seconds: %s",
                        attempt,
                        delay,
                        e,
                    )
                    time.sleep(delay)

        # All attempts failed
        raise DownloadError(
            message=f"Download failed after {self._retry_config.max_attempts} attempts",
            video_url=url,
            original_error=last_error,
        )

    def _perform_download(self, url: str, ydl_opts: Dict[str, Any]) -> Path:
        """Perform the actual download."""
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise VideoNotFoundError(
                        message="Video not found or unavailable",
                        video_url=url,
                    )

                # Find the downloaded file
                video_id = info.get("id", "")
                output_template = ydl_opts["outtmpl"]
                output_dir = Path(output_template).parent

                # Look for the downloaded file
                for ext in [self._config.audio_format, "m4a", "mp3", "webm", "opus"]:
                    candidate = output_dir / f"{video_id}.{ext}"
                    if candidate.exists():
                        return candidate

                raise DownloadError(
                    message="Download completed but file not found",
                    video_url=url,
                    output_path=str(output_dir),
                )

        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e).lower()
            if "video unavailable" in error_msg or "private video" in error_msg:
                raise VideoNotFoundError(
                    message="Video not found or is private",
                    video_url=url,
                    original_error=e,
                ) from e
            raise DownloadError(
                message=f"Download failed: {e}",
                video_url=url,
                original_error=e,
            ) from e

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay before retry."""
        base_delay = self._retry_config.delay_seconds
        if self._retry_config.exponential_backoff:
            return base_delay * (2 ** (attempt - 1))
        return base_delay

    def _parse_metadata(self, info: Dict[str, Any]) -> VideoMetadata:
        """Parse yt-dlp info dict into VideoMetadata."""
        return VideoMetadata(
            video_id=info.get("id", ""),
            title=info.get("title", ""),
            description=info.get("description", ""),
            channel_name=info.get("channel", "") or info.get("uploader", ""),
            channel_id=info.get("channel_id", "") or info.get("uploader_id", ""),
            duration_seconds=info.get("duration", 0) or 0,
            upload_date=self._format_upload_date(info.get("upload_date")),
            view_count=info.get("view_count"),
            like_count=info.get("like_count"),
            tags=info.get("tags", []) or [],
            categories=info.get("categories", []) or [],
            thumbnail_url=info.get("thumbnail", ""),
            is_live=info.get("is_live", False) or False,
            is_age_restricted=info.get("age_limit", 0) > 0,
            language=info.get("language", ""),
            extra={
                "webpage_url": info.get("webpage_url", ""),
                "channel_url": info.get("channel_url", ""),
                "availability": info.get("availability", ""),
            },
        )

    def _format_upload_date(self, date_str: Optional[str]) -> str:
        """Format upload date from YYYYMMDD to YYYY-MM-DD."""
        if not date_str or len(date_str) != 8:
            return ""
        try:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        except Exception:
            return ""

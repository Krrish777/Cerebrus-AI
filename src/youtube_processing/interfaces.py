"""
YouTube Processing Interfaces Module.

This module defines abstract interfaces for all YouTube processing components.
Following the Dependency Inversion Principle, high-level modules depend on
these abstractions rather than concrete implementations.

Key Interfaces:
    - VideoDownloader: Downloads audio from YouTube videos
    - MetadataEnhancer: Extracts and enriches video metadata
    - CacheManager: Manages cached audio files
    - URLValidator: Validates YouTube URLs

These interfaces enable:
    - Easy testing through mock implementations
    - Swappable implementations (e.g., different download backends)
    - Clear contracts between components
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


@dataclass
class VideoMetadata:
    """
    Data class representing video metadata.

    This is an immutable container for all metadata extracted from a video.

    Attributes:
        video_id: Unique YouTube video identifier.
        title: Video title.
        description: Video description (may be truncated).
        channel_name: Name of the channel that uploaded the video.
        channel_id: Unique identifier for the channel.
        duration_seconds: Video duration in seconds.
        upload_date: Upload date in YYYY-MM-DD format.
        view_count: Number of views (may be None if not available).
        like_count: Number of likes (may be None if not available).
        tags: List of video tags.
        categories: List of video categories.
        thumbnail_url: URL to the video thumbnail.
        is_live: Whether this is a live stream.
        is_age_restricted: Whether the video is age-restricted.
        language: Detected or declared language of the video.
        extra: Additional metadata fields not covered above.
    """

    video_id: str
    title: str
    description: str = ""
    channel_name: str = ""
    channel_id: str = ""
    duration_seconds: int = 0
    upload_date: str = ""
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    thumbnail_url: str = ""
    is_live: bool = False
    is_age_restricted: bool = False
    language: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary for serialization."""
        return {
            "video_id": self.video_id,
            "title": self.title,
            "description": self.description,
            "channel_name": self.channel_name,
            "channel_id": self.channel_id,
            "duration_seconds": self.duration_seconds,
            "upload_date": self.upload_date,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "tags": self.tags,
            "categories": self.categories,
            "thumbnail_url": self.thumbnail_url,
            "is_live": self.is_live,
            "is_age_restricted": self.is_age_restricted,
            "language": self.language,
            **self.extra,
        }


@dataclass
class DownloadResult:
    """
    Result of a video audio download operation.

    Attributes:
        audio_path: Path to the downloaded audio file.
        metadata: Extracted video metadata.
        file_size_bytes: Size of the downloaded file in bytes.
        download_duration_seconds: Time taken to download in seconds.
        from_cache: Whether the file was retrieved from cache.
    """

    audio_path: Path
    metadata: VideoMetadata
    file_size_bytes: int = 0
    download_duration_seconds: float = 0.0
    from_cache: bool = False


class VideoDownloader(ABC):
    """
    Abstract interface for downloading audio from YouTube videos.

    Implementations of this interface handle the actual download logic,
    potentially using different backends (yt-dlp, pytube, etc.).

    Example:
        downloader = YtDlpDownloader(config)
        result = downloader.download(url, output_dir)
        print(f"Downloaded to: {result.audio_path}")
    """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def validate_url(self, url: str) -> bool:
        """
        Validate a YouTube URL without downloading.

        Args:
            url: URL to validate.

        Returns:
            True if the URL is a valid YouTube video URL.
        """

    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Return list of supported audio formats."""


class MetadataEnhancer(ABC):
    """
    Abstract interface for enhancing video metadata.

    This interface allows for post-processing of metadata, such as:
    - Truncating descriptions to a maximum length
    - Normalizing dates and formats
    - Extracting additional information from descriptions
    - Adding custom fields based on video content

    Example:
        enhancer = DefaultMetadataEnhancer(config)
        enhanced = enhancer.enhance(raw_metadata, transcript)
    """

    @abstractmethod
    def enhance(
        self,
        metadata: VideoMetadata,
        transcript_text: Optional[str] = None,
    ) -> VideoMetadata:
        """
        Enhance video metadata with additional processing.

        Args:
            metadata: Raw video metadata.
            transcript_text: Optional transcript for additional analysis.

        Returns:
            Enhanced VideoMetadata instance.
        """

    @abstractmethod
    def merge_with_transcript_metadata(
        self,
        video_metadata: VideoMetadata,
        transcript_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge video metadata with transcript-derived metadata.

        This combines metadata from the video (title, channel, etc.) with
        metadata extracted during transcription (entities, topics, etc.).

        Args:
            video_metadata: Metadata from the video.
            transcript_metadata: Metadata from transcription.

        Returns:
            Combined metadata dictionary suitable for document creation.
        """


class CacheManager(ABC):
    """
    Abstract interface for managing cached audio files.

    The cache manager handles:
    - Storing downloaded audio files
    - Retrieving cached files by video ID
    - Cache expiration and cleanup
    - Cache size management

    Example:
        cache = FileCacheManager(config)
        if cache.has(video_id):
            audio_path = cache.get(video_id)
        else:
            audio_path = downloader.download(url, temp_dir).audio_path
            cache.put(video_id, audio_path)
    """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def has(self, video_id: str) -> bool:
        """
        Check if a video is cached.

        Args:
            video_id: YouTube video ID.

        Returns:
            True if the video is in the cache.
        """

    @abstractmethod
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

    @abstractmethod
    def clear(self) -> int:
        """
        Clear all cached files.

        Returns:
            Number of files removed.

        Raises:
            CacheError: If clearing fails.
        """

    @abstractmethod
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of files removed.

        Raises:
            CacheError: If cleanup fails.
        """

    @property
    @abstractmethod
    def cache_size_bytes(self) -> int:
        """Return the total size of the cache in bytes."""

    @property
    @abstractmethod
    def entry_count(self) -> int:
        """Return the number of entries in the cache."""


class URLValidator(ABC):
    """
    Abstract interface for YouTube URL validation.

    This interface validates URLs against configured constraints such as:
    - Allowed domains
    - Video duration limits
    - Live stream restrictions
    - Age restriction handling

    Example:
        validator = YouTubeURLValidator(config)
        is_valid, errors = validator.validate(url, metadata)
    """

    @abstractmethod
    def validate(
        self,
        url: str,
        metadata: Optional[VideoMetadata] = None,
    ) -> tuple[bool, List[str]]:
        """
        Validate a YouTube URL against configured constraints.

        Args:
            url: YouTube URL to validate.
            metadata: Optional metadata for additional validation.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """

    @abstractmethod
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract the video ID from a YouTube URL.

        Args:
            url: YouTube URL.

        Returns:
            Video ID or None if the URL is invalid.
        """

    @abstractmethod
    def normalize_url(self, url: str) -> str:
        """
        Normalize a YouTube URL to a standard format.

        Args:
            url: YouTube URL in any valid format.

        Returns:
            Normalized URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID).

        Raises:
            ValidationError: If the URL cannot be normalized.
        """


class YouTubeProcessor(ABC):
    """
    Abstract interface for the main YouTube processing orchestrator.

    This is the high-level interface that coordinates:
    - URL validation
    - Audio download (with caching)
    - Transcription (delegated to audio processing module)
    - Metadata enhancement
    - Document creation

    Example:
        processor = YouTubeAudioProcessor(
            downloader=downloader,
            audio_transcriber=transcriber,
            cache_manager=cache,
            metadata_enhancer=enhancer,
        )
        documents = processor.process(url)
    """

    @abstractmethod
    def process(self, url: str) -> List[Any]:
        """
        Process a YouTube video and return documents.

        This is the main entry point for processing a video. It:
        1. Validates the URL
        2. Downloads the audio (or retrieves from cache)
        3. Transcribes the audio
        4. Enhances metadata
        5. Creates Haystack Documents

        Args:
            url: YouTube video URL.

        Returns:
            List of Haystack Document objects.

        Raises:
            VideoNotFoundError: If the video does not exist.
            ValidationError: If the URL is invalid.
            DownloadError: If download fails.
            YouTubeProcessingError: For other processing errors.
        """

    @abstractmethod
    def process_batch(self, urls: List[str]) -> Dict[str, List[Any]]:
        """
        Process multiple YouTube videos.

        Args:
            urls: List of YouTube video URLs.

        Returns:
            Dictionary mapping URLs to their document lists.
            Failed URLs map to empty lists.
        """

    @abstractmethod
    def get_metadata(self, url: str) -> VideoMetadata:
        """
        Get metadata for a YouTube video without processing.

        Args:
            url: YouTube video URL.

        Returns:
            VideoMetadata for the video.
        """

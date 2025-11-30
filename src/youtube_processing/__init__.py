"""
YouTube Processing Module.

This module provides functionality for processing YouTube videos including:
- Audio download and caching
- Transcription via the audio_processing module
- Metadata extraction and enhancement
- Haystack Document creation

Architecture:
    The module follows SOLID principles with dependency injection for all components.
    The YouTubeAudioProcessor orchestrates the workflow, delegating:
    - Download: to VideoDownloader implementations (YtDlpDownloader)
    - Transcription: to AudioTranscriber from audio_processing module
    - Caching: to CacheManager implementations (FileCacheManager)
    - Metadata: to MetadataEnhancer implementations

Example usage:
    from pathlib import Path
    from src.youtube_processing import YouTubeAudioProcessor, YouTubeConfig

    # Load configuration
    config = YouTubeConfig.from_yaml(Path("config/youtube_config.yml"))

    # Create processor with defaults
    processor = YouTubeAudioProcessor.create(config)

    # Process a video
    documents = processor.process("https://youtube.com/watch?v=...")

    # Or use in Haystack pipeline
    from src.youtube_processing import YouTubeTranscriber
    transcriber = YouTubeTranscriber.from_config(config)
    result = transcriber.run(urls=["https://youtube.com/watch?v=..."])
"""

# Configuration
from src.youtube_processing.config import CacheConfig
from src.youtube_processing.config import DownloadConfig
from src.youtube_processing.config import MetadataConfig
from src.youtube_processing.config import RetryConfig
from src.youtube_processing.config import ValidationConfig
from src.youtube_processing.config import YouTubeConfig

# Exceptions
from src.youtube_processing.exceptions import CacheError
from src.youtube_processing.exceptions import ConfigurationError
from src.youtube_processing.exceptions import DownloadError
from src.youtube_processing.exceptions import MetadataError
from src.youtube_processing.exceptions import ValidationError
from src.youtube_processing.exceptions import VideoNotFoundError
from src.youtube_processing.exceptions import YouTubeProcessingError

# Interfaces
from src.youtube_processing.interfaces import CacheManager
from src.youtube_processing.interfaces import DownloadResult
from src.youtube_processing.interfaces import MetadataEnhancer
from src.youtube_processing.interfaces import URLValidator
from src.youtube_processing.interfaces import VideoDownloader
from src.youtube_processing.interfaces import VideoMetadata
from src.youtube_processing.interfaces import YouTubeProcessor

# Implementations
from src.youtube_processing.cache import FileCacheManager
from src.youtube_processing.download import YtDlpDownloader
from src.youtube_processing.download import YouTubeURLValidator
from src.youtube_processing.metadata import DefaultMetadataEnhancer

# Components
from src.youtube_processing.components import YouTubeAudioProcessor
from src.youtube_processing.components import YouTubeDocumentBuilder
from src.youtube_processing.components import YouTubeTranscriber

__all__ = [
    # Configuration
    "YouTubeConfig",
    "DownloadConfig",
    "CacheConfig",
    "ValidationConfig",
    "MetadataConfig",
    "RetryConfig",
    # Exceptions
    "YouTubeProcessingError",
    "VideoNotFoundError",
    "DownloadError",
    "CacheError",
    "ValidationError",
    "MetadataError",
    "ConfigurationError",
    # Interfaces
    "VideoDownloader",
    "MetadataEnhancer",
    "CacheManager",
    "URLValidator",
    "YouTubeProcessor",
    "VideoMetadata",
    "DownloadResult",
    # Implementations
    "YtDlpDownloader",
    "YouTubeURLValidator",
    "DefaultMetadataEnhancer",
    "FileCacheManager",
    # Components
    "YouTubeAudioProcessor",
    "YouTubeDocumentBuilder",
    "YouTubeTranscriber",
]

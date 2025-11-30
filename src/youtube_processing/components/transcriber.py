"""
YouTube Transcriber Haystack Component.

This module provides a Haystack 2.0 compatible component for YouTube
audio transcription that can be used in Haystack pipelines.
"""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from haystack import component
from haystack.dataclasses import Document

from src.core.logging import get_logger
from src.youtube_processing.components.processor import YouTubeAudioProcessor
from src.youtube_processing.config import YouTubeConfig
from src.youtube_processing.interfaces import VideoMetadata

logger = get_logger(__name__)


@component
class YouTubeTranscriber:
    """
    Haystack 2.0 component for YouTube audio transcription.

    This component wraps YouTubeAudioProcessor to provide a Haystack pipeline
    compatible interface. It can process YouTube URLs and output Documents.

    Example usage in a Haystack pipeline:
        from haystack import Pipeline
        from src.youtube_processing import YouTubeTranscriber

        pipeline = Pipeline()
        pipeline.add_component("transcriber", YouTubeTranscriber.from_config(config))
        result = pipeline.run({"transcriber": {"urls": ["https://youtube.com/watch?v=..."]}})
    """

    def __init__(
        self,
        processor: YouTubeAudioProcessor,
        return_metadata_only: bool = False,
    ) -> None:
        """
        Initialize the YouTube transcriber component.

        Args:
            processor: Configured YouTubeAudioProcessor instance.
            return_metadata_only: If True, only fetch metadata without transcription.
        """
        self._processor = processor
        self._return_metadata_only = return_metadata_only
        logger.info("Initialized YouTubeTranscriber component")

    @classmethod
    def from_config(
        cls,
        config: YouTubeConfig,
        return_metadata_only: bool = False,
    ) -> "YouTubeTranscriber":
        """
        Create a YouTubeTranscriber from configuration.

        Args:
            config: YouTube processing configuration.
            return_metadata_only: If True, only fetch metadata without transcription.

        Returns:
            Configured YouTubeTranscriber instance.
        """
        processor = YouTubeAudioProcessor.create(config)
        return cls(processor=processor, return_metadata_only=return_metadata_only)

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
        return_metadata_only: bool = False,
    ) -> "YouTubeTranscriber":
        """
        Create a YouTubeTranscriber from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file.
            return_metadata_only: If True, only fetch metadata without transcription.

        Returns:
            Configured YouTubeTranscriber instance.
        """
        config = YouTubeConfig.from_yaml(config_path)
        return cls.from_config(config, return_metadata_only)

    @component.output_types(documents=List[Document], metadata=List[VideoMetadata])
    def run(
        self,
        urls: List[str],
        metadata_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Process YouTube URLs and return documents.

        Args:
            urls: List of YouTube video URLs to process.
            metadata_only: Override for return_metadata_only setting.

        Returns:
            Dictionary with:
                - "documents": List of Haystack Documents
                - "metadata": List of VideoMetadata objects
        """
        logger.info("Processing %d YouTube URLs", len(urls))

        should_transcribe = not (
            metadata_only if metadata_only is not None else self._return_metadata_only
        )

        all_documents: List[Document] = []
        all_metadata: List[VideoMetadata] = []

        for url in urls:
            try:
                if should_transcribe:
                    documents = self._processor.process(url)
                    all_documents.extend(documents)
                    # Extract metadata from first document if available
                    if documents and documents[0].meta:
                        metadata = self._extract_metadata_from_document(documents[0])
                        all_metadata.append(metadata)
                else:
                    metadata = self._processor.get_metadata(url)
                    all_metadata.append(metadata)

            except Exception as e:
                logger.error("Failed to process URL %s: %s", url, e)
                # Continue processing other URLs

        logger.info(
            "Completed processing: %d documents, %d metadata entries",
            len(all_documents),
            len(all_metadata),
        )

        return {
            "documents": all_documents,
            "metadata": all_metadata,
        }

    def _extract_metadata_from_document(self, document: Document) -> VideoMetadata:
        """Extract VideoMetadata from a document's meta field."""
        meta = document.meta or {}
        return VideoMetadata(
            video_id=meta.get("video_id", ""),
            title=meta.get("title", ""),
            description=meta.get("description", ""),
            channel_name=meta.get("channel_name", ""),
            channel_id=meta.get("channel_id", ""),
            duration_seconds=meta.get("duration_seconds", 0),
            upload_date=meta.get("upload_date", ""),
            view_count=meta.get("view_count"),
            like_count=meta.get("like_count"),
            tags=meta.get("tags", []),
            categories=meta.get("categories", []),
            thumbnail_url=meta.get("thumbnail_url", ""),
            is_live=meta.get("is_live", False),
            is_age_restricted=meta.get("is_age_restricted", False),
            language=meta.get("language", ""),
        )


@component
class YouTubeMetadataFetcher:
    """
    Haystack 2.0 component for fetching YouTube video metadata.

    This is a lightweight component that only fetches metadata without
    downloading or transcribing the video.

    Example:
        pipeline = Pipeline()
        pipeline.add_component("metadata", YouTubeMetadataFetcher(config))
        result = pipeline.run({"metadata": {"urls": ["https://youtube.com/watch?v=..."]}})
    """

    def __init__(self, config: YouTubeConfig) -> None:
        """
        Initialize the metadata fetcher.

        Args:
            config: YouTube processing configuration.
        """
        from src.youtube_processing.download import YtDlpDownloader
        from src.youtube_processing.download import YouTubeURLValidator

        validator = YouTubeURLValidator(config.validation)
        self._downloader = YtDlpDownloader(
            config=config.download,
            retry_config=config.retry,
            validator=validator,
        )
        self._validator = validator
        logger.info("Initialized YouTubeMetadataFetcher component")

    @classmethod
    def from_yaml(cls, config_path: Path) -> "YouTubeMetadataFetcher":
        """Create from YAML configuration file."""
        config = YouTubeConfig.from_yaml(config_path)
        return cls(config)

    @component.output_types(metadata=List[VideoMetadata], valid_urls=List[str])
    def run(self, urls: List[str]) -> Dict[str, Any]:
        """
        Fetch metadata for YouTube URLs.

        Args:
            urls: List of YouTube video URLs.

        Returns:
            Dictionary with:
                - "metadata": List of VideoMetadata objects
                - "valid_urls": List of URLs that passed validation
        """
        all_metadata: List[VideoMetadata] = []
        valid_urls: List[str] = []

        for url in urls:
            try:
                # Validate URL first
                video_id = self._validator.extract_video_id(url)
                if not video_id:
                    logger.warning("Invalid URL format: %s", url)
                    continue

                # Extract metadata
                metadata = self._downloader.extract_metadata(url)

                # Validate against constraints
                is_valid, errors = self._validator.validate(url, metadata)
                if not is_valid:
                    logger.warning("URL validation failed: %s - %s", url, errors)
                    continue

                all_metadata.append(metadata)
                valid_urls.append(url)

            except Exception as e:
                logger.error("Failed to fetch metadata for %s: %s", url, e)

        return {
            "metadata": all_metadata,
            "valid_urls": valid_urls,
        }

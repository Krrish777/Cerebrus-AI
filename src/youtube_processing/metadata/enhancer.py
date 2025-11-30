"""
Metadata Enhancer Module.

This module provides the DefaultMetadataEnhancer class for enriching
video metadata with additional processing and analysis.
"""

from typing import Any
from typing import Dict
from typing import Optional

from src.core.logging import get_logger
from src.youtube_processing.config import MetadataConfig
from src.youtube_processing.interfaces import MetadataEnhancer
from src.youtube_processing.interfaces import VideoMetadata

logger = get_logger(__name__)


class DefaultMetadataEnhancer(MetadataEnhancer):
    """
    Default implementation of metadata enhancement.

    This class provides post-processing of video metadata including:
    - Description truncation
    - Field normalization
    - Merging with transcript-derived metadata

    Example:
        config = MetadataConfig()
        enhancer = DefaultMetadataEnhancer(config)
        enhanced = enhancer.enhance(metadata, transcript)
    """

    def __init__(self, config: MetadataConfig) -> None:
        """
        Initialize the metadata enhancer.

        Args:
            config: Metadata configuration.
        """
        self._config = config
        logger.debug("Initialized DefaultMetadataEnhancer")

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
        logger.debug("Enhancing metadata for video: %s", metadata.video_id)

        # Truncate description if needed
        description = metadata.description
        if len(description) > self._config.max_description_length:
            description = description[: self._config.max_description_length].rsplit(" ", 1)[0]
            description += "..."
            logger.debug(
                "Truncated description from %d to %d characters",
                len(metadata.description),
                len(description),
            )

        # Create enhanced metadata with same values but truncated description
        enhanced = VideoMetadata(
            video_id=metadata.video_id,
            title=metadata.title,
            description=description,
            channel_name=metadata.channel_name,
            channel_id=metadata.channel_id,
            duration_seconds=metadata.duration_seconds,
            upload_date=metadata.upload_date,
            view_count=metadata.view_count,
            like_count=metadata.like_count,
            tags=metadata.tags if self._config.extract_tags else [],
            categories=metadata.categories if self._config.extract_categories else [],
            thumbnail_url=metadata.thumbnail_url if self._config.extract_thumbnail_url else "",
            is_live=metadata.is_live,
            is_age_restricted=metadata.is_age_restricted,
            language=metadata.language,
            extra=self._build_extra_fields(metadata, transcript_text),
        )

        return enhanced

    def _build_extra_fields(
        self,
        metadata: VideoMetadata,
        transcript_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build extra metadata fields based on configuration."""
        extra: Dict[str, Any] = {}

        # Copy original extra fields
        extra.update(metadata.extra)

        # Add derived fields
        if transcript_text:
            extra["transcript_word_count"] = len(transcript_text.split())
            extra["transcript_char_count"] = len(transcript_text)

        # Add engagement metrics if configured
        if self._config.extract_view_count and metadata.view_count is not None:
            extra["has_view_count"] = True
        if self._config.extract_like_count and metadata.like_count is not None:
            extra["has_like_count"] = True

        # Add channel info flag
        if self._config.extract_channel_info:
            extra["has_channel_info"] = bool(metadata.channel_name)

        return extra

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
        logger.debug(
            "Merging video metadata with transcript metadata for video: %s",
            video_metadata.video_id,
        )

        # Start with video metadata as base
        merged = video_metadata.to_dict()

        # Add transcript-derived metadata with namespacing to avoid conflicts
        for key, value in transcript_metadata.items():
            if key in merged:
                # Namespace the transcript metadata key
                merged[f"transcript_{key}"] = value
            else:
                merged[key] = value

        # Add source indicator
        merged["source"] = "youtube"
        merged["source_url"] = f"https://www.youtube.com/watch?v={video_metadata.video_id}"

        # Add processing timestamp
        from datetime import datetime
        from datetime import timezone

        merged["processed_at"] = datetime.now(timezone.utc).isoformat()

        logger.debug(
            "Merged metadata contains %d fields for video: %s",
            len(merged),
            video_metadata.video_id,
        )

        return merged

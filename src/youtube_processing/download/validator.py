"""
YouTube URL Validator Module.

This module provides URL validation for YouTube videos.
It validates URLs against configured constraints and extracts video IDs.
"""

import re
from typing import List
from typing import Optional
from typing import Tuple
from urllib.parse import parse_qs
from urllib.parse import urlparse

from src.core.logging import get_logger
from src.youtube_processing.config import ValidationConfig
from src.youtube_processing.exceptions import ValidationError
from src.youtube_processing.interfaces import URLValidator
from src.youtube_processing.interfaces import VideoMetadata

logger = get_logger(__name__)


class YouTubeURLValidator(URLValidator):
    """
    URL validator for YouTube videos.

    This implementation validates URLs against configured constraints including:
    - Allowed domains
    - Video duration limits
    - Live stream restrictions
    - Age restriction handling

    Example:
        config = ValidationConfig()
        validator = YouTubeURLValidator(config)
        is_valid, errors = validator.validate(url, metadata)
    """

    # Regex patterns for extracting video IDs
    _VIDEO_ID_PATTERNS = [
        # Standard watch URL: youtube.com/watch?v=VIDEO_ID
        re.compile(r"(?:youtube\.com/watch\?.*v=)([a-zA-Z0-9_-]{11})"),
        # Short URL: youtu.be/VIDEO_ID
        re.compile(r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})"),
        # Embed URL: youtube.com/embed/VIDEO_ID
        re.compile(r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})"),
        # Shorts URL: youtube.com/shorts/VIDEO_ID
        re.compile(r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})"),
        # Live URL: youtube.com/live/VIDEO_ID
        re.compile(r"(?:youtube\.com/live/)([a-zA-Z0-9_-]{11})"),
    ]

    def __init__(self, config: ValidationConfig) -> None:
        """
        Initialize the URL validator.

        Args:
            config: Validation configuration.
        """
        self._config = config
        logger.debug("Initialized YouTubeURLValidator with config: %s", config)

    def validate(
        self,
        url: str,
        metadata: Optional[VideoMetadata] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a YouTube URL against configured constraints.

        Args:
            url: YouTube URL to validate.
            metadata: Optional metadata for additional validation.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """
        errors: List[str] = []

        # Validate URL format
        if not self._is_valid_format(url):
            errors.append("Invalid URL format")
            return False, errors

        # Validate domain
        if not self._is_allowed_domain(url):
            errors.append(f"Domain not in allowed list: {self._config.allowed_domains}")

        # Extract video ID
        video_id = self.extract_video_id(url)
        if not video_id:
            errors.append("Could not extract video ID from URL")

        # If metadata is provided, perform additional validation
        if metadata:
            errors.extend(self._validate_metadata(metadata))

        is_valid = len(errors) == 0
        if is_valid:
            logger.debug("URL validation passed: %s", url)
        else:
            logger.warning("URL validation failed: %s - Errors: %s", url, errors)

        return is_valid, errors

    def _is_valid_format(self, url: str) -> bool:
        """Check if URL has a valid format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _is_allowed_domain(self, url: str) -> bool:
        """Check if URL domain is in allowed list."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix for comparison
            if domain.startswith("www."):
                domain = domain[4:]
            return domain in [d.lower().replace("www.", "") for d in self._config.allowed_domains]
        except Exception:
            return False

    def _validate_metadata(self, metadata: VideoMetadata) -> List[str]:
        """Validate video metadata against constraints."""
        errors: List[str] = []

        # Check duration
        if metadata.duration_seconds < self._config.min_duration_seconds:
            errors.append(
                f"Video too short: {metadata.duration_seconds}s "
                f"(minimum: {self._config.min_duration_seconds}s)"
            )
        if metadata.duration_seconds > self._config.max_duration_seconds:
            errors.append(
                f"Video too long: {metadata.duration_seconds}s "
                f"(maximum: {self._config.max_duration_seconds}s)"
            )

        # Check live stream
        if metadata.is_live and not self._config.allow_live_streams:
            errors.append("Live streams are not allowed")

        # Check age restriction
        if metadata.is_age_restricted and not self._config.allow_age_restricted:
            errors.append("Age-restricted videos are not allowed")

        return errors

    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract the video ID from a YouTube URL.

        Args:
            url: YouTube URL.

        Returns:
            Video ID or None if the URL is invalid.
        """
        # Try regex patterns
        for pattern in self._VIDEO_ID_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group(1)

        # Try query parameter extraction as fallback
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            if "v" in query_params:
                video_id = query_params["v"][0]
                if len(video_id) == 11:
                    return video_id
        except Exception:
            pass

        logger.warning("Could not extract video ID from URL: %s", url)
        return None

    def normalize_url(self, url: str) -> str:
        """
        Normalize a YouTube URL to a standard format.

        Args:
            url: YouTube URL in any valid format.

        Returns:
            Normalized URL (https://www.youtube.com/watch?v=VIDEO_ID).

        Raises:
            ValidationError: If the URL cannot be normalized.
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValidationError(
                message="Cannot normalize URL: video ID not found",
                video_url=url,
                field_name="url",
                field_value=url,
            )

        normalized = f"https://www.youtube.com/watch?v={video_id}"
        logger.debug("Normalized URL: %s -> %s", url, normalized)
        return normalized

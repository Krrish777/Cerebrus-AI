"""
Content Validator Implementation.

This module provides content validation for scraped web content.
Validates content quality before processing.

Following AGENTS.md principles:
    - Single responsibility: Only content validation
    - Dependency injection: Configuration injected via constructor
    - Fail-fast: Validates all constraints upfront
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from src.core.logging import get_logger
from src.web_scraping.config import ContentValidationConfig
from src.web_scraping.interfaces import ContentValidator

logger = get_logger(__name__)


class DefaultContentValidator(ContentValidator):
    """
    Default implementation of content validation.

    Validates scraped content against configured constraints:
    - Minimum/maximum content length
    - Minimum word count
    - Content quality checks

    Example:
        config = ContentValidationConfig(
            min_content_length=100,
            max_content_length=1000000,
            min_word_count=10,
        )
        validator = DefaultContentValidator(config)
        is_valid, errors = validator.validate_with_errors(content, metadata)
    """

    def __init__(self, config: ContentValidationConfig) -> None:
        """
        Initialize the content validator.

        Args:
            config: Content validation configuration.
        """
        self._config = config
        logger.debug(
            "Content validator initialized with min_length=%d, max_length=%d, min_words=%d",
            self._config.min_content_length,
            self._config.max_content_length,
            self._config.min_word_count,
        )

    def validate(self, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Validate scraped content quality.

        Args:
            content: Scraped content.
            metadata: Content metadata.

        Returns:
            True if valid, False otherwise.
        """
        is_valid, _ = self.validate_with_errors(content, metadata)
        return is_valid

    def validate_with_errors(
        self,
        content: str,
        metadata: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Validate content and return validation errors.

        Args:
            content: Scraped content.
            metadata: Content metadata.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """
        errors: List[str] = []

        # Handle None content
        if content is None:
            errors.append("Content cannot be None")
            return False, errors

        content_length = len(content)
        word_count = len(content.split()) if content else 0

        # Validate content length
        if content_length < self._config.min_content_length:
            errors.append(
                f"Content length ({content_length}) is below minimum "
                f"({self._config.min_content_length})"
            )

        if content_length > self._config.max_content_length:
            errors.append(
                f"Content length ({content_length}) exceeds maximum "
                f"({self._config.max_content_length})"
            )

        # Validate word count
        if word_count < self._config.min_word_count:
            errors.append(
                f"Word count ({word_count}) is below minimum "
                f"({self._config.min_word_count})"
            )

        # Quality checks
        if self._config.check_content_quality:
            quality_errors = self._check_content_quality(content, metadata)
            errors.extend(quality_errors)

        is_valid = len(errors) == 0

        url = metadata.get("url", "unknown")
        if not is_valid:
            logger.debug(
                "Content validation failed for %s: %s",
                url,
                errors,
            )
        else:
            logger.debug(
                "Content validation passed for %s (length=%d, words=%d)",
                url,
                content_length,
                word_count,
            )

        return is_valid, errors

    def _check_content_quality(
        self,
        content: str,
        metadata: Dict[str, Any],
    ) -> List[str]:
        """
        Perform quality checks on content.

        Args:
            content: Content to check.
            metadata: Content metadata.

        Returns:
            List of quality issue messages.
        """
        issues: List[str] = []

        if not content:
            return issues

        # Check for mostly whitespace
        non_whitespace = len(content.replace(" ", "").replace("\n", "").replace("\t", ""))
        if non_whitespace < len(content) * 0.3:
            issues.append("Content is mostly whitespace")

        # Check for repetitive content
        if self._is_repetitive_content(content):
            issues.append("Content appears to be repetitive")

        # Check for error pages
        error_indicators = [
            "404 not found",
            "page not found",
            "error 404",
            "access denied",
            "403 forbidden",
            "500 internal server error",
            "service unavailable",
        ]

        content_lower = content.lower()
        for indicator in error_indicators:
            if indicator in content_lower and len(content) < 1000:
                issues.append(f"Content appears to be an error page: '{indicator}'")
                break

        return issues

    def _is_repetitive_content(self, content: str) -> bool:
        """
        Check if content is repetitive.

        Args:
            content: Content to check.

        Returns:
            True if content is repetitive.
        """
        if len(content) < 200:
            return False

        # Split into chunks and check for repetition
        chunk_size = 100
        chunks = [
            content[i:i + chunk_size]
            for i in range(0, min(len(content), 1000), chunk_size)
        ]

        if len(chunks) < 3:
            return False

        # Check if chunks are similar
        unique_chunks = set(chunk.strip().lower() for chunk in chunks if chunk.strip())
        repetition_ratio = len(unique_chunks) / len(chunks)

        # If less than 50% unique chunks, content is likely repetitive
        return repetition_ratio < 0.5

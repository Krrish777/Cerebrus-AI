"""
Base extractor implementation.

Provides shared functionality for all data extractors.
Follows AGENTS.md design principles:
- Single Responsibility: Base class handles common validation and logging
- Encapsulation: Internal state is protected
- Extensibility: Concrete extractors extend without modification
"""

from abc import abstractmethod
from typing import Any, Dict

from src.audio_processing.exceptions import ExtractionError, ValidationError
from src.audio_processing.interfaces import DataExtractor
from src.core.logging import get_logger

logger = get_logger(__name__)


class BaseExtractor(DataExtractor):
    """
    Abstract base class for data extractors.

    Provides common functionality including:
    - Data validation
    - Error handling
    - Logging

    Subclasses must implement:
    - _do_extract: Actual extraction logic
    - _get_data_key: Key to look for in transcript data
    """

    def __init__(self) -> None:
        """Initialize the base extractor."""
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    @abstractmethod
    def extractor_name(self) -> str:
        """Return the name of this extractor."""

    @abstractmethod
    def _get_data_key(self) -> str:
        """
        Return the key to look for in transcript data.

        :return: Key name in transcript data dictionary
        """

    @abstractmethod
    def _do_extract(self, data: Any) -> Dict[str, Any]:
        """
        Perform the actual extraction.

        :param data: Raw data from transcript
        :return: Extracted and formatted data
        """

    def is_available(self, transcript_data: Dict[str, Any]) -> bool:
        """
        Check if this extractor can extract data from the given transcript.

        :param transcript_data: Raw transcript data to check
        :return: True if extraction is possible
        """
        key = self._get_data_key()
        data = transcript_data.get(key)

        if data is None:
            logger.debug(
                "%s: Data key '%s' not found in transcript",
                self.extractor_name,
                key,
            )
            return False

        # Check for empty data
        if isinstance(data, (list, dict)) and not data:
            logger.debug(
                "%s: Data key '%s' is empty",
                self.extractor_name,
                key,
            )
            return False

        return True

    def extract(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract specific data from transcript.

        :param transcript_data: Raw transcript data from provider
        :return: Extracted data as a dictionary
        :raises ExtractionError: If extraction fails
        """
        if not self.is_available(transcript_data):
            logger.warning(
                "%s: No data available for extraction",
                self.extractor_name,
            )
            return self._empty_result()

        key = self._get_data_key()
        raw_data = transcript_data[key]

        logger.debug(
            "%s: Extracting data from key '%s'",
            self.extractor_name,
            key,
        )

        try:
            result = self._do_extract(raw_data)
            logger.info(
                "%s: Successfully extracted %d items",
                self.extractor_name,
                self._count_items(result),
            )
            return result
        except Exception as exc:
            logger.error(
                "%s: Extraction failed: %s",
                self.extractor_name,
                exc,
            )
            raise ExtractionError(
                f"{self.extractor_name} extraction failed: {exc}"
            ) from exc

    def _empty_result(self) -> Dict[str, Any]:
        """
        Return an empty result structure.

        Override in subclasses if needed.

        :return: Empty result dictionary
        """
        return {"items": [], "count": 0}

    def _count_items(self, result: Dict[str, Any]) -> int:
        """
        Count items in result for logging.

        :param result: Extraction result
        :return: Number of items
        """
        if "items" in result:
            return len(result["items"])
        if "count" in result:
            return result["count"]
        return 1

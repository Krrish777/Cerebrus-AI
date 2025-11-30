"""
Extractor registry implementation.

Provides centralized management and discovery of extractors.
Follows the Registry Pattern for loose coupling.
"""

from typing import Dict, List, Optional, Type

from src.audio_processing.exceptions import ConfigurationError
from src.audio_processing.extractors.base import BaseExtractor
from src.audio_processing.extractors.chapter import ChapterExtractor
from src.audio_processing.extractors.content_safety import ContentSafetyExtractor
from src.audio_processing.extractors.entity import EntityExtractor
from src.audio_processing.extractors.highlights import HighlightsExtractor
from src.audio_processing.extractors.sentiment import SentimentExtractor
from src.audio_processing.extractors.topic import TopicExtractor
from src.core.logging import get_logger

logger = get_logger(__name__)


class ExtractorRegistry:
    """
    Registry for managing extractor instances.

    Provides:
    - Centralized extractor registration and discovery
    - Lazy instantiation of extractors
    - Bulk extraction across all registered extractors

    Example usage:
        registry = ExtractorRegistry()

        # Get specific extractor
        sentiment = registry.get("sentiment")
        result = sentiment.extract(transcript_data)

        # Extract all available data
        all_data = registry.extract_all(transcript_data)

        # Register custom extractor
        registry.register("custom", CustomExtractor)
    """

    # Default extractors
    _default_extractors: Dict[str, Type[BaseExtractor]] = {
        "sentiment": SentimentExtractor,
        "entity": EntityExtractor,
        "chapter": ChapterExtractor,
        "topic": TopicExtractor,
        "content_safety": ContentSafetyExtractor,
        "highlights": HighlightsExtractor,
    }

    def __init__(self) -> None:
        """Initialize the registry with default extractors."""
        self._extractors: Dict[str, Type[BaseExtractor]] = dict(
            self._default_extractors
        )
        self._instances: Dict[str, BaseExtractor] = {}

        logger.debug(
            "ExtractorRegistry initialized with %d extractors",
            len(self._extractors),
        )

    @classmethod
    def register_default(
        cls,
        name: str,
        extractor_class: Type[BaseExtractor],
    ) -> None:
        """
        Register an extractor as a default.

        :param name: Extractor name
        :param extractor_class: Extractor class
        """
        if not issubclass(extractor_class, BaseExtractor):
            raise ConfigurationError(
                f"Extractor must inherit from BaseExtractor: {name}"
            )
        cls._default_extractors[name] = extractor_class
        logger.info("Registered default extractor: %s", name)

    def register(
        self,
        name: str,
        extractor_class: Type[BaseExtractor],
    ) -> None:
        """
        Register an extractor in this registry instance.

        :param name: Extractor name
        :param extractor_class: Extractor class
        """
        if not issubclass(extractor_class, BaseExtractor):
            raise ConfigurationError(
                f"Extractor must inherit from BaseExtractor: {name}"
            )
        self._extractors[name] = extractor_class
        # Clear cached instance if exists
        self._instances.pop(name, None)
        logger.info("Registered extractor: %s", name)

    def get(self, name: str) -> BaseExtractor:
        """
        Get an extractor instance by name.

        Uses lazy instantiation - creates instance on first access.

        :param name: Extractor name
        :return: Extractor instance
        :raises ConfigurationError: If extractor not found
        """
        if name not in self._extractors:
            available = ", ".join(self._extractors.keys())
            raise ConfigurationError(
                f"Unknown extractor: {name}. Available: {available}"
            )

        # Lazy instantiation
        if name not in self._instances:
            self._instances[name] = self._extractors[name]()
            logger.debug("Instantiated extractor: %s", name)

        return self._instances[name]

    def available(self) -> List[str]:
        """
        Get list of available extractor names.

        :return: List of extractor names
        """
        return list(self._extractors.keys())

    def get_available_for_transcript(
        self,
        transcript_data: Dict,
    ) -> List[str]:
        """
        Get extractors that have data in the transcript.

        :param transcript_data: Transcript data
        :return: List of extractor names with available data
        """
        available = []
        for name in self._extractors:
            extractor = self.get(name)
            if extractor.is_available(transcript_data):
                available.append(name)
        return available

    def extract(
        self,
        name: str,
        transcript_data: Dict,
    ) -> Dict:
        """
        Extract data using a specific extractor.

        :param name: Extractor name
        :param transcript_data: Transcript data
        :return: Extracted data
        """
        extractor = self.get(name)
        return extractor.extract(transcript_data)

    def extract_all(
        self,
        transcript_data: Dict,
        only_available: bool = True,
    ) -> Dict[str, Dict]:
        """
        Extract data using all registered extractors.

        :param transcript_data: Transcript data
        :param only_available: Only run extractors with available data
        :return: Dictionary of extractor name to extracted data
        """
        results = {}

        for name in self._extractors:
            extractor = self.get(name)

            if only_available and not extractor.is_available(transcript_data):
                logger.debug("Skipping %s: no data available", name)
                continue

            try:
                results[name] = extractor.extract(transcript_data)
            except Exception as exc:
                logger.error("Extraction failed for %s: %s", name, exc)
                results[name] = extractor._empty_result()

        logger.info(
            "Extracted data from %d extractors",
            len(results),
        )

        return results

    def extract_selected(
        self,
        transcript_data: Dict,
        extractor_names: List[str],
    ) -> Dict[str, Dict]:
        """
        Extract data using selected extractors.

        :param transcript_data: Transcript data
        :param extractor_names: List of extractor names to use
        :return: Dictionary of extractor name to extracted data
        """
        results = {}

        for name in extractor_names:
            if name not in self._extractors:
                logger.warning("Unknown extractor: %s", name)
                continue

            extractor = self.get(name)
            try:
                results[name] = extractor.extract(transcript_data)
            except Exception as exc:
                logger.error("Extraction failed for %s: %s", name, exc)
                results[name] = extractor._empty_result()

        return results

    def clear_cache(self) -> None:
        """Clear cached extractor instances."""
        self._instances.clear()
        logger.debug("Cleared extractor instance cache")


# Global registry instance for convenience
_global_registry: Optional[ExtractorRegistry] = None


def get_registry() -> ExtractorRegistry:
    """
    Get the global extractor registry.

    :return: Global ExtractorRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ExtractorRegistry()
    return _global_registry

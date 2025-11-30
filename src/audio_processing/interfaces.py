"""
Audio Processing Interfaces Module.

This module defines abstract base classes (interfaces) for the audio processing
system. All concrete implementations must inherit from these abstractions.

Following AGENTS.md principles:
- Abstraction: Define interfaces for swappable implementations
- Loose coupling: Depend on abstractions, not concrete classes
- Extensibility: New implementations can be added without modifying existing code
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack.dataclasses import Document

from src.audio_processing.config import TranscriptionConfig


class TranscriptionProvider(ABC):
    """
    Abstract interface for transcription providers.

    All transcription providers (AssemblyAI, Whisper, etc.) must implement
    this interface to ensure consistent behavior and swappability.
    """

    @abstractmethod
    def transcribe(self, source: Path) -> Dict[str, Any]:
        """
        Transcribe audio from the given source.

        Args:
            source: Path to audio file.

        Returns:
            Dictionary containing transcription results with at minimum:
                - id: Unique transcript identifier
                - text: Full transcription text
                - status: Transcription status
                - confidence: Overall confidence score (0-1)
                - audio_duration: Duration in seconds

        Raises:
            TranscriptionError: If transcription fails.
            ProviderError: If provider encounters an error.
        """
        pass

    @abstractmethod
    def configure(self, config: TranscriptionConfig) -> None:
        """
        Configure the provider with the given settings.

        Args:
            config: Transcription configuration.
        """
        pass

    @abstractmethod
    def upload(self, source: Path) -> str:
        """
        Upload audio to the provider and return a URL or identifier.

        Args:
            source: Path to audio file.

        Returns:
            URL or identifier for the uploaded audio.

        Raises:
            ProviderError: If upload fails.
        """
        pass

    @abstractmethod
    def get_status(self, transcript_id: str) -> Dict[str, Any]:
        """
        Get the status of a transcription job.

        Args:
            transcript_id: The transcript identifier.

        Returns:
            Status dictionary with at least 'id' and 'status' keys.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Return whether the provider is configured and ready."""
        pass


class ChunkingStrategy(ABC):
    """
    Abstract interface for chunking strategies.

    Different strategies (speaker-based, chapter-based, semantic) implement
    this interface to provide consistent chunking behavior.
    """

    @abstractmethod
    def chunk(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Split content into chunks based on the strategy.

        Args:
            content: The text content to chunk.
            metadata: Additional metadata to assist in chunking.

        Returns:
            List of chunk dictionaries, each containing:
                - content: The chunk text
                - metadata: Chunk-specific metadata
                - chunk_id: Unique identifier within this chunking operation
        """
        pass

    @abstractmethod
    def validate_input(self, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Validate that the input is suitable for this chunking strategy.

        Args:
            content: The content to validate.
            metadata: The metadata to validate.

        Returns:
            True if input is valid for this strategy.
        """
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        pass


class DataExtractor(ABC):
    """
    Abstract interface for data extraction.

    Different extractors (sentiment, entity, topic) implement this interface
    to provide consistent data extraction behavior.
    """

    @abstractmethod
    def extract(
        self,
        transcript_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract specific data from transcript.

        Args:
            transcript_data: Raw transcript data from provider.

        Returns:
            Extracted data as a dictionary with extractor-specific keys.
        """
        pass

    @abstractmethod
    def is_available(self, transcript_data: Dict[str, Any]) -> bool:
        """
        Check if this extractor can extract data from the given transcript.

        Args:
            transcript_data: Raw transcript data to check.

        Returns:
            True if extraction is possible.
        """
        pass

    @property
    @abstractmethod
    def extractor_name(self) -> str:
        """Return the name of this extractor."""
        pass


class DocumentBuilder(ABC):
    """
    Abstract interface for building Haystack Documents.

    Converts transcription results and extracted data into properly
    structured Haystack Documents.
    """

    @abstractmethod
    def build(
        self,
        transcript_data: Dict[str, Any],
        extracted_data: Dict[str, Dict[str, Any]],
        source_name: str
    ) -> List[Document]:
        """
        Build Haystack Documents from transcript and extracted data.

        Args:
            transcript_data: Raw transcript data from provider.
            extracted_data: Data from all extractors, keyed by extractor name.
            source_name: Name of the audio source.

        Returns:
            List of Haystack Document objects.
        """
        pass

    @abstractmethod
    def build_metadata(
        self,
        transcript_data: Dict[str, Any],
        extracted_data: Dict[str, Dict[str, Any]],
        source_name: str
    ) -> Dict[str, Any]:
        """
        Build metadata dictionary for documents.

        Args:
            transcript_data: Raw transcript data.
            extracted_data: Data from extractors.
            source_name: Name of the audio source.

        Returns:
            Metadata dictionary.
        """
        pass


class MetadataManager(ABC):
    """
    Abstract interface for managing document metadata.

    Handles creation, validation, and enhancement of document metadata.
    """

    @abstractmethod
    def create_base_metadata(
        self,
        transcript_id: str,
        source_name: str,
        audio_duration: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Create base metadata for a document.

        Args:
            transcript_id: Unique transcript identifier.
            source_name: Name of the audio source.
            audio_duration: Duration in seconds.
            confidence: Confidence score (0-1).

        Returns:
            Base metadata dictionary.
        """
        pass

    @abstractmethod
    def enhance_metadata(
        self,
        base_metadata: Dict[str, Any],
        extracted_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Enhance metadata with extracted data.

        Args:
            base_metadata: Base metadata dictionary.
            extracted_data: Data from all extractors.

        Returns:
            Enhanced metadata dictionary.
        """
        pass

    @abstractmethod
    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate metadata dictionary structure.

        Args:
            metadata: Metadata to validate.

        Returns:
            True if metadata is valid.
        """
        pass


class TranscriptionOrchestrator(ABC):
    """
    Abstract interface for orchestrating transcription workflows.

    Coordinates providers, extractors, and document builders.
    """

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        config: Optional[TranscriptionConfig] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio source.

        Args:
            audio_path: Path to audio file.
            config: Optional configuration override.

        Returns:
            Transcription result dictionary.
        """
        pass

    @abstractmethod
    def transcribe_with_features(
        self,
        audio_path: Path,
        features: Any,
    ) -> Dict[str, Any]:
        """
        Transcribe with advanced features enabled.

        Args:
            audio_path: Path to audio file.
            features: Feature configuration.

        Returns:
            Transcription result with feature data.
        """
        pass

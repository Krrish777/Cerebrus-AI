"""
Abstract interfaces for audio processing components.

Defines abstract base classes for transcription services to enable
loose coupling and extensibility following AGENTS.md design principles.
"""

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from haystack import Document


class TranscriptionServiceInterface(ABC):
    """
    Abstract interface for transcription services.
    
    Defines the contract that all transcription service implementations
    must follow, enabling dependency injection and easy testing.
    """

    @abstractmethod
    def transcribe(
        self, 
        sources: List[Union[str, Path, bytes]]
    ) -> Dict[str, List[Document]]:
        """
        Transcribe audio from provided sources.
        
        :param sources: List of audio sources (file paths, URLs, or bytes)
        :return: Dictionary with 'documents' key containing transcribed documents
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the transcription service to a dictionary.
        
        :return: Dictionary representation of the service
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionServiceInterface":
        """
        Deserialize the transcription service from a dictionary.
        
        :param data: Dictionary representation of the service
        :return: Transcription service instance
        """
        pass


class AudioChunkerInterface(ABC):
    """
    Abstract interface for audio content chunking.
    
    Defines the contract for chunking transcribed audio content
    into smaller, manageable pieces.
    """

    @abstractmethod
    def chunk(
        self, 
        documents: List[Document]
    ) -> List[Document]:
        """
        Chunk transcribed documents into smaller segments.
        
        :param documents: List of documents to chunk
        :return: List of chunked documents
        """
        pass


class AudioSourceValidatorInterface(ABC):
    """
    Abstract interface for audio source validation.
    
    Defines the contract for validating audio sources before processing.
    """

    @abstractmethod
    def validate(self, source: Union[str, Path, bytes]) -> bool:
        """
        Validate if the source is a valid audio source.
        
        :param source: Audio source to validate
        :return: True if source is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_source_type(self, source: Union[str, Path, bytes]) -> str:
        """
        Get the type of the audio source.
        
        :param source: Audio source
        :return: Source type (e.g., 'url', 'file', 'bytes')
        """
        pass


class TranscriptionConfigAdapterInterface(ABC):
    """
    Abstract interface for transcription configuration adapters.
    
    Enables conversion between different configuration formats
    for flexibility and backwards compatibility.
    """

    @abstractmethod
    def to_provider_config(self) -> Any:
        """
        Convert configuration to provider-specific format.
        
        :return: Provider-specific configuration object
        """
        pass

    @abstractmethod
    def get_api_key(self) -> str:
        """
        Get the API key for the transcription service.
        
        :return: API key string
        """
        pass


class MetadataExtractorInterface(ABC):
    """
    Abstract interface for metadata extraction from transcripts.
    
    Enables extraction of structured metadata from transcription results.
    """

    @abstractmethod
    def extract_sentiment(self, transcript: Any) -> List[Dict[str, Any]]:
        """
        Extract sentiment analysis data from transcript.
        
        :param transcript: Transcript object
        :return: List of sentiment data dictionaries
        """
        pass

    @abstractmethod
    def extract_entities(self, transcript: Any) -> List[Dict[str, Any]]:
        """
        Extract entity detection data from transcript.
        
        :param transcript: Transcript object
        :return: List of entity data dictionaries
        """
        pass

    @abstractmethod
    def extract_topics(self, transcript: Any) -> Dict[str, Any]:
        """
        Extract topic detection data from transcript.
        
        :param transcript: Transcript object
        :return: Dictionary of topic data
        """
        pass

    @abstractmethod
    def extract_highlights(self, transcript: Any) -> List[Dict[str, Any]]:
        """
        Extract auto-highlights data from transcript.
        
        :param transcript: Transcript object
        :return: List of highlight data dictionaries
        """
        pass


class YouTubeProcessorInterface(ABC):
    """
    Abstract interface for YouTube video processing.
    
    Defines the contract for downloading and processing YouTube videos.
    """

    @abstractmethod
    def validate_url(self, url: str) -> bool:
        """
        Validate if the URL is a valid YouTube URL.
        
        :param url: URL to validate
        :return: True if URL is valid, False otherwise
        """
        pass

    @abstractmethod
    def extract_video_id(self, url: str) -> str | None:
        """
        Extract video ID from YouTube URL.
        
        :param url: YouTube URL
        :return: Video ID or None if extraction failed
        """
        pass

    @abstractmethod
    def get_video_info(self, url: str) -> Dict[str, Any] | None:
        """
        Get metadata about a YouTube video.
        
        :param url: YouTube URL
        :return: Video metadata dictionary or None if failed
        """
        pass

    @abstractmethod
    def download_audio(self, url: str) -> Path:
        """
        Download audio from YouTube video.
        
        :param url: YouTube URL
        :return: Path to downloaded audio file
        """
        pass

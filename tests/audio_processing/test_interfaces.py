"""
Unit tests for audio processing interfaces.

Tests abstract interfaces to ensure they define proper contracts
following AGENTS.md principles: deterministic, focused, comprehensive.
"""

import pytest
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Union

from haystack import Document

from src.audio_processing.interfaces import (
    TranscriptionServiceInterface,
    AudioChunkerInterface,
    AudioSourceValidatorInterface,
    TranscriptionConfigAdapterInterface,
    MetadataExtractorInterface,
    YouTubeProcessorInterface,
)


class TestTranscriptionServiceInterface:
    """Test TranscriptionServiceInterface abstract class."""

    def test_is_abstract_class(self):
        """Test that interface is an abstract class."""
        assert issubclass(TranscriptionServiceInterface, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that interface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TranscriptionServiceInterface()  # type: ignore

    def test_has_required_abstract_methods(self):
        """Test that interface defines required abstract methods."""
        abstract_methods = TranscriptionServiceInterface.__abstractmethods__
        
        assert "transcribe" in abstract_methods
        assert "to_dict" in abstract_methods
        assert "from_dict" in abstract_methods

    def test_concrete_implementation_works(self):
        """Test that a concrete implementation can be created."""
        
        class MockTranscriptionService(TranscriptionServiceInterface):
            def transcribe(
                self, 
                sources: List[Union[str, Path, bytes]]
            ) -> Dict[str, List[Document]]:
                return {"documents": []}
            
            def to_dict(self) -> Dict[str, Any]:
                return {"type": "mock"}
            
            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> "MockTranscriptionService":
                return cls()
        
        service = MockTranscriptionService()
        assert isinstance(service, TranscriptionServiceInterface)
        assert service.transcribe([]) == {"documents": []}


class TestAudioChunkerInterface:
    """Test AudioChunkerInterface abstract class."""

    def test_is_abstract_class(self):
        """Test that interface is an abstract class."""
        assert issubclass(AudioChunkerInterface, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that interface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AudioChunkerInterface()  # type: ignore

    def test_has_required_abstract_methods(self):
        """Test that interface defines required abstract methods."""
        abstract_methods = AudioChunkerInterface.__abstractmethods__
        
        assert "chunk" in abstract_methods

    def test_concrete_implementation_works(self):
        """Test that a concrete implementation can be created."""
        
        class MockAudioChunker(AudioChunkerInterface):
            def chunk(self, documents: List[Document]) -> List[Document]:
                return documents
        
        chunker = MockAudioChunker()
        assert isinstance(chunker, AudioChunkerInterface)


class TestAudioSourceValidatorInterface:
    """Test AudioSourceValidatorInterface abstract class."""

    def test_is_abstract_class(self):
        """Test that interface is an abstract class."""
        assert issubclass(AudioSourceValidatorInterface, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that interface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AudioSourceValidatorInterface()  # type: ignore

    def test_has_required_abstract_methods(self):
        """Test that interface defines required abstract methods."""
        abstract_methods = AudioSourceValidatorInterface.__abstractmethods__
        
        assert "validate" in abstract_methods
        assert "get_source_type" in abstract_methods

    def test_concrete_implementation_works(self):
        """Test that a concrete implementation can be created."""
        
        class MockSourceValidator(AudioSourceValidatorInterface):
            def validate(self, source: Union[str, Path, bytes]) -> bool:
                return True
            
            def get_source_type(self, source: Union[str, Path, bytes]) -> str:
                if isinstance(source, bytes):
                    return "bytes"
                return "string"
        
        validator = MockSourceValidator()
        assert isinstance(validator, AudioSourceValidatorInterface)
        assert validator.validate("test.mp3") is True
        assert validator.get_source_type("test") == "string"
        assert validator.get_source_type(b"data") == "bytes"


class TestTranscriptionConfigAdapterInterface:
    """Test TranscriptionConfigAdapterInterface abstract class."""

    def test_is_abstract_class(self):
        """Test that interface is an abstract class."""
        assert issubclass(TranscriptionConfigAdapterInterface, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that interface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TranscriptionConfigAdapterInterface()  # type: ignore

    def test_has_required_abstract_methods(self):
        """Test that interface defines required abstract methods."""
        abstract_methods = TranscriptionConfigAdapterInterface.__abstractmethods__
        
        assert "to_provider_config" in abstract_methods
        assert "get_api_key" in abstract_methods


class TestMetadataExtractorInterface:
    """Test MetadataExtractorInterface abstract class."""

    def test_is_abstract_class(self):
        """Test that interface is an abstract class."""
        assert issubclass(MetadataExtractorInterface, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that interface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MetadataExtractorInterface()  # type: ignore

    def test_has_required_abstract_methods(self):
        """Test that interface defines required abstract methods."""
        abstract_methods = MetadataExtractorInterface.__abstractmethods__
        
        assert "extract_sentiment" in abstract_methods
        assert "extract_entities" in abstract_methods
        assert "extract_topics" in abstract_methods
        assert "extract_highlights" in abstract_methods


class TestYouTubeProcessorInterface:
    """Test YouTubeProcessorInterface abstract class."""

    def test_is_abstract_class(self):
        """Test that interface is an abstract class."""
        assert issubclass(YouTubeProcessorInterface, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that interface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            YouTubeProcessorInterface()  # type: ignore

    def test_has_required_abstract_methods(self):
        """Test that interface defines required abstract methods."""
        abstract_methods = YouTubeProcessorInterface.__abstractmethods__
        
        assert "validate_url" in abstract_methods
        assert "extract_video_id" in abstract_methods
        assert "get_video_info" in abstract_methods
        assert "download_audio" in abstract_methods

    def test_concrete_implementation_works(self):
        """Test that a concrete implementation can be created."""
        
        class MockYouTubeProcessor(YouTubeProcessorInterface):
            def validate_url(self, url: str) -> bool:
                return "youtube.com" in url
            
            def extract_video_id(self, url: str) -> str | None:
                return "test_id"
            
            def get_video_info(self, url: str) -> Dict[str, Any] | None:
                return {"title": "Test Video"}
            
            def download_audio(self, url: str) -> Path:
                return Path("/tmp/audio.mp3")
        
        processor = MockYouTubeProcessor()
        assert isinstance(processor, YouTubeProcessorInterface)
        assert processor.validate_url("https://youtube.com/watch?v=123") is True
        assert processor.extract_video_id("url") == "test_id"


class TestInterfacePolymorphism:
    """Test polymorphic behavior of interfaces."""

    def test_can_use_interface_as_type_hint(self):
        """Test that interfaces work as type hints."""
        
        class MockService(TranscriptionServiceInterface):
            def transcribe(
                self, 
                sources: List[Union[str, Path, bytes]]
            ) -> Dict[str, List[Document]]:
                return {"documents": []}
            
            def to_dict(self) -> Dict[str, Any]:
                return {}
            
            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> "MockService":
                return cls()
        
        def process_with_service(service: TranscriptionServiceInterface) -> None:
            result = service.transcribe([])
            assert "documents" in result
        
        service = MockService()
        process_with_service(service)

    def test_multiple_implementations_coexist(self):
        """Test that multiple implementations can coexist."""
        
        class ServiceA(TranscriptionServiceInterface):
            def transcribe(
                self, 
                sources: List[Union[str, Path, bytes]]
            ) -> Dict[str, List[Document]]:
                return {"documents": [], "source": "A"}
            
            def to_dict(self) -> Dict[str, Any]:
                return {"type": "A"}
            
            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> "ServiceA":
                return cls()
        
        class ServiceB(TranscriptionServiceInterface):
            def transcribe(
                self, 
                sources: List[Union[str, Path, bytes]]
            ) -> Dict[str, List[Document]]:
                return {"documents": [], "source": "B"}
            
            def to_dict(self) -> Dict[str, Any]:
                return {"type": "B"}
            
            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> "ServiceB":
                return cls()
        
        services: List[TranscriptionServiceInterface] = [ServiceA(), ServiceB()]
        
        for service in services:
            result = service.transcribe([])
            assert "documents" in result

"""
Tests for Haystack pipeline components.

Tests the component wrappers for audio processing.
"""

import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from haystack.dataclasses import Document

from src.audio_processing.chunking.base import Chunk
from src.audio_processing.components.transcriber import AudioTranscriberComponent
from src.audio_processing.components.extractor import DataExtractorComponent
from src.audio_processing.components.chunker import ChunkerComponent
from src.audio_processing.components.document_converter import DocumentConverterComponent


@pytest.fixture
def transcript_response() -> Dict[str, Any]:
    """Load mock transcript response fixture."""
    fixture_path = Path("data/fixtures/mock_transcript_response.json")
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.fixture
def analysis_data() -> Dict[str, Any]:
    """Load mock analysis data fixture."""
    fixture_path = Path("data/fixtures/mock_analysis_data.json")
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock transcription provider."""
    provider = MagicMock()
    provider.provider_name = "mock"
    provider.is_configured = True
    provider.transcribe.return_value = {
        "id": "mock-123",
        "text": "This is mock transcription text.",
        "status": "completed",
        "confidence": 0.95,
        "audio_duration": 60.0,
    }
    return provider


class TestAudioTranscriberComponent:
    """Tests for AudioTranscriberComponent."""
    
    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        component = AudioTranscriberComponent()
        
        assert component._provider_name == "assemblyai"
        assert component._provider is None
    
    def test_init_with_custom_provider(self) -> None:
        """Test initialization with custom provider."""
        component = AudioTranscriberComponent(provider_name="whisper")
        
        assert component._provider_name == "whisper"
    
    @patch("src.audio_processing.components.transcriber.TranscriptionFactory")
    def test_warm_up_creates_provider(
        self,
        mock_factory_class: MagicMock,
        mock_provider: MagicMock,
    ) -> None:
        """Test that warm_up creates provider."""
        mock_factory = MagicMock()
        mock_factory.create.return_value = mock_provider
        mock_factory_class.return_value = mock_factory
        
        component = AudioTranscriberComponent()
        component.warm_up()
        
        assert component._provider is not None
        mock_factory.create.assert_called_once()
    
    @patch("src.audio_processing.components.transcriber.TranscriptionFactory")
    def test_run_transcribes_files(
        self,
        mock_factory_class: MagicMock,
        mock_provider: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test running transcription on files."""
        mock_factory = MagicMock()
        mock_factory.create.return_value = mock_provider
        mock_factory_class.return_value = mock_factory
        
        # Create temp audio file
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        component = AudioTranscriberComponent()
        result = component.run(audio_paths=[audio_file])
        
        assert "transcripts" in result
        assert len(result["transcripts"]) == 1
        assert result["transcripts"][0]["id"] == "mock-123"
    
    @patch("src.audio_processing.components.transcriber.TranscriptionFactory")
    def test_run_handles_errors(
        self,
        mock_factory_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that run handles transcription errors."""
        mock_factory = MagicMock()
        mock_provider = MagicMock()
        mock_provider.transcribe.side_effect = Exception("API error")
        mock_factory.create.return_value = mock_provider
        mock_factory_class.return_value = mock_factory
        
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        component = AudioTranscriberComponent()
        result = component.run(audio_paths=[audio_file])
        
        assert len(result["transcripts"]) == 1
        assert result["transcripts"][0]["status"] == "error"
        assert "API error" in result["transcripts"][0]["error"]


class TestDataExtractorComponent:
    """Tests for DataExtractorComponent."""
    
    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        component = DataExtractorComponent()
        
        assert len(component._extractor_names) > 0
    
    def test_init_with_specific_extractors(self) -> None:
        """Test initialization with specific extractors."""
        component = DataExtractorComponent(extractors=["sentiment", "entities"])
        
        assert component._extractor_names == ["sentiment", "entities"]
    
    def test_run_extracts_data(
        self,
        transcript_response: Dict[str, Any],
        analysis_data: Dict[str, Any],
    ) -> None:
        """Test running extraction on transcripts."""
        # Merge analysis data into transcript for testing
        transcript = {**transcript_response, **analysis_data}
        
        component = DataExtractorComponent(
            extractors=["sentiment", "entities", "chapters"]
        )
        result = component.run(transcripts=[transcript])
        
        assert "transcripts" in result
        assert "extracted_data" in result
        assert len(result["extracted_data"]) == 1
        
        # Check that some extraction happened
        extracted = result["extracted_data"][0]
        assert len(extracted) > 0
    
    def test_run_skips_error_transcripts(self) -> None:
        """Test that error transcripts are skipped."""
        error_transcript = {"status": "error", "error": "Test error"}
        
        component = DataExtractorComponent()
        result = component.run(transcripts=[error_transcript])
        
        assert result["extracted_data"][0] == {}
    
    def test_run_passes_through_transcripts(
        self,
        transcript_response: Dict[str, Any],
    ) -> None:
        """Test that transcripts are passed through unchanged."""
        component = DataExtractorComponent(extractors=[])
        result = component.run(transcripts=[transcript_response])
        
        assert result["transcripts"] == [transcript_response]


class TestChunkerComponent:
    """Tests for ChunkerComponent."""
    
    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        component = ChunkerComponent()
        
        assert component._strategy == "auto"
    
    def test_init_with_strategy(self) -> None:
        """Test initialization with specific strategy."""
        component = ChunkerComponent(strategy="speaker")
        
        assert component._strategy == "speaker"
    
    def test_run_chunks_transcript(
        self,
        transcript_response: Dict[str, Any],
    ) -> None:
        """Test running chunking on transcripts."""
        component = ChunkerComponent(strategy="chapter")
        result = component.run(transcripts=[transcript_response])
        
        assert "transcripts" in result
        assert "chunks" in result
        assert len(result["chunks"]) == 1
        
        # Fixture has 2 chapters
        assert len(result["chunks"][0]) == 2
    
    def test_run_auto_selects_strategy(
        self,
        transcript_response: Dict[str, Any],
    ) -> None:
        """Test automatic strategy selection."""
        component = ChunkerComponent(strategy="auto")
        result = component.run(transcripts=[transcript_response])
        
        # Should use chapter since chapters exist
        assert len(result["chunks"][0]) >= 1
    
    def test_run_skips_error_transcripts(self) -> None:
        """Test that error transcripts return empty chunks."""
        error_transcript = {"status": "error", "error": "Test error"}
        
        component = ChunkerComponent()
        result = component.run(transcripts=[error_transcript])
        
        assert result["chunks"][0] == []
    
    def test_chunks_are_serializable(
        self,
        transcript_response: Dict[str, Any],
    ) -> None:
        """Test that chunk output is serializable."""
        component = ChunkerComponent(strategy="chapter")
        result = component.run(transcripts=[transcript_response])
        
        for chunk in result["chunks"][0]:
            assert isinstance(chunk, dict)
            assert "text" in chunk
            assert "start_time" in chunk
            assert "end_time" in chunk


class TestDocumentConverterComponent:
    """Tests for DocumentConverterComponent."""
    
    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        component = DocumentConverterComponent()
        
        assert component._use_chunks is True
    
    def test_run_creates_documents(
        self,
        transcript_response: Dict[str, Any],
    ) -> None:
        """Test creating documents from transcripts."""
        component = DocumentConverterComponent(use_chunks=False)
        
        result = component.run(
            transcripts=[transcript_response],
            source_names=["test.mp3"],
        )
        
        assert "documents" in result
        assert len(result["documents"]) == 1
        assert isinstance(result["documents"][0], Document)
    
    def test_run_with_chunks(
        self,
        transcript_response: Dict[str, Any],
    ) -> None:
        """Test creating documents from chunks."""
        component = DocumentConverterComponent(use_chunks=True)
        
        chunks = [
            [
                {"text": "First chunk", "start_time": 0, "end_time": 5000, "speaker": None, "metadata": {}},
                {"text": "Second chunk", "start_time": 5000, "end_time": 10000, "speaker": None, "metadata": {}},
            ]
        ]
        
        result = component.run(
            transcripts=[transcript_response],
            chunks=chunks,
            source_names=["test.mp3"],
        )
        
        assert len(result["documents"]) == 2
    
    def test_run_with_extracted_data(
        self,
        transcript_response: Dict[str, Any],
    ) -> None:
        """Test including extracted data in documents."""
        component = DocumentConverterComponent(use_chunks=False)
        
        extracted = [
            {
                "sentiment": {"distribution": {"POSITIVE": 5}, "total_count": 5},
            }
        ]
        
        result = component.run(
            transcripts=[transcript_response],
            extracted_data=extracted,
            source_names=["test.mp3"],
        )
        
        doc = result["documents"][0]
        assert "sentiment" in doc.meta.get("extracted_features", [])
    
    def test_run_skips_error_transcripts(self) -> None:
        """Test that error transcripts don't create documents."""
        component = DocumentConverterComponent()
        
        error_transcript = {"status": "error", "error": "Test error"}
        
        result = component.run(
            transcripts=[error_transcript],
            source_names=["error.mp3"],
        )
        
        assert len(result["documents"]) == 0
    
    def test_documents_have_metadata(
        self,
        transcript_response: Dict[str, Any],
    ) -> None:
        """Test that documents have proper metadata."""
        component = DocumentConverterComponent(use_chunks=False)
        
        result = component.run(
            transcripts=[transcript_response],
            source_names=["podcast.mp3"],
        )
        
        doc = result["documents"][0]
        assert doc.meta["source_name"] == "podcast.mp3"
        assert doc.meta["transcript_id"] == transcript_response["id"]


class TestComponentIntegration:
    """Integration tests for component pipeline."""
    
    def test_extractor_to_chunker_flow(
        self,
        transcript_response: Dict[str, Any],
        analysis_data: Dict[str, Any],
    ) -> None:
        """Test data flow from extractor to chunker."""
        transcript = {**transcript_response, **analysis_data}
        
        # Run extraction
        extractor = DataExtractorComponent(extractors=["sentiment"])
        extract_result = extractor.run(transcripts=[transcript])
        
        # Run chunking with extraction output
        chunker = ChunkerComponent(strategy="chapter")
        chunk_result = chunker.run(
            transcripts=extract_result["transcripts"]
        )
        
        assert len(chunk_result["chunks"]) == 1
        assert len(chunk_result["chunks"][0]) == 2  # 2 chapters
    
    def test_full_pipeline_flow(
        self,
        transcript_response: Dict[str, Any],
        analysis_data: Dict[str, Any],
    ) -> None:
        """Test complete pipeline flow without transcription."""
        transcript = {**transcript_response, **analysis_data}
        
        # Extract
        extractor = DataExtractorComponent(extractors=["sentiment", "entities"])
        extract_result = extractor.run(transcripts=[transcript])
        
        # Chunk
        chunker = ChunkerComponent(strategy="chapter")
        chunk_result = chunker.run(
            transcripts=extract_result["transcripts"]
        )
        
        # Convert to documents
        converter = DocumentConverterComponent(use_chunks=True)
        doc_result = converter.run(
            transcripts=chunk_result["transcripts"],
            extracted_data=extract_result["extracted_data"],
            chunks=chunk_result["chunks"],
            source_names=["podcast.mp3"],
        )
        
        # Verify output
        assert len(doc_result["documents"]) == 2
        for doc in doc_result["documents"]:
            assert isinstance(doc, Document)
            assert doc.meta["source_name"] == "podcast.mp3"

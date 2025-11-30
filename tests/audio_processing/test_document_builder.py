"""
Tests for the document building module.

Tests metadata building and document creation.
"""

import json
from pathlib import Path
from typing import Any
from typing import Dict

import pytest
from haystack.dataclasses import Document

from src.audio_processing.chunking.base import Chunk
from src.audio_processing.document.builder import TranscriptDocumentBuilder
from src.audio_processing.document.metadata import DocumentMetadata
from src.audio_processing.document.metadata import MetadataBuilder


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
def simple_transcript() -> Dict[str, Any]:
    """Create a simple transcript for testing."""
    return {
        "id": "test-123",
        "text": "Hello world. This is a test transcript.",
        "audio_duration": 10.5,
        "confidence": 0.95,
    }


@pytest.fixture
def extracted_data() -> Dict[str, Dict[str, Any]]:
    """Create mock extracted data."""
    return {
        "sentiment": {
            "distribution": {"POSITIVE": 5, "NEUTRAL": 3, "NEGATIVE": 2},
            "total_count": 10,
        },
        "entities": {
            "entities_by_type": {"PERSON": ["John", "Jane"], "ORG": ["Acme"]},
            "total_count": 3,
        },
        "topics": {
            "topics": [
                {"label": "Technology", "relevance": 0.9},
                {"label": "Business", "relevance": 0.7},
            ],
        },
    }


class TestDocumentMetadata:
    """Tests for DocumentMetadata dataclass."""
    
    def test_create_metadata(self) -> None:
        """Test basic metadata creation."""
        metadata = DocumentMetadata(
            source_name="test.mp3",
            transcript_id="abc-123",
        )
        
        assert metadata.source_name == "test.mp3"
        assert metadata.transcript_id == "abc-123"
        assert metadata.audio_duration is None
        assert metadata.confidence is None
    
    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metadata = DocumentMetadata(
            source_name="test.mp3",
            transcript_id="abc-123",
            audio_duration=60.5,
            confidence=0.95,
        )
        
        result = metadata.to_dict()
        
        assert result["source_name"] == "test.mp3"
        assert result["transcript_id"] == "abc-123"
        assert result["audio_duration"] == 60.5
        assert result["confidence"] == 0.95
        assert "created_at" in result
    
    def test_custom_fields_included(self) -> None:
        """Test that custom fields are included in dict."""
        metadata = DocumentMetadata(
            source_name="test.mp3",
            transcript_id="abc-123",
            custom={"custom_field": "value"},
        )
        
        result = metadata.to_dict()
        
        assert result["custom_field"] == "value"


class TestMetadataBuilder:
    """Tests for MetadataBuilder."""
    
    def test_create_base_metadata(self) -> None:
        """Test creating base metadata."""
        builder = MetadataBuilder()
        
        metadata = builder.create_base_metadata(
            transcript_id="test-123",
            source_name="audio.mp3",
            audio_duration=120.0,
            confidence=0.92,
        )
        
        assert metadata.transcript_id == "test-123"
        assert metadata.source_name == "audio.mp3"
        assert metadata.audio_duration == 120.0
        assert metadata.confidence == 0.92
        assert metadata.processing_info["processor"] == "audio_processing"
    
    def test_enhance_with_extracted(
        self,
        extracted_data: Dict[str, Dict[str, Any]],
    ) -> None:
        """Test enhancing metadata with extracted data."""
        builder = MetadataBuilder()
        base = builder.create_base_metadata(
            transcript_id="test-123",
            source_name="audio.mp3",
        )
        
        enhanced = builder.enhance_with_extracted(base, extracted_data)
        
        assert "sentiment" in enhanced.extracted_features
        assert "entities" in enhanced.extracted_features
        assert "topics" in enhanced.extracted_features
        assert "sentiment_summary" in enhanced.custom
        assert "entities_summary" in enhanced.custom
    
    def test_sentiment_summary(self) -> None:
        """Test sentiment summarization."""
        builder = MetadataBuilder()
        
        summary = builder._summarize_sentiment({
            "distribution": {"POSITIVE": 5, "NEUTRAL": 3, "NEGATIVE": 2},
            "total_count": 10,
        })
        
        assert summary["dominant_sentiment"] == "POSITIVE"
        assert summary["sentiment_count"] == 10
    
    def test_entity_summary(self) -> None:
        """Test entity summarization."""
        builder = MetadataBuilder()
        
        summary = builder._summarize_entities({
            "entities_by_type": {"PERSON": ["John"], "ORG": ["Acme"]},
            "total_count": 2,
        })
        
        assert "PERSON" in summary["entity_types"]
        assert "ORG" in summary["entity_types"]
        assert summary["entity_count"] == 2
    
    def test_build_chunk_metadata(self) -> None:
        """Test building chunk metadata."""
        builder = MetadataBuilder()
        base = builder.create_base_metadata(
            transcript_id="test-123",
            source_name="audio.mp3",
        )
        
        chunk_meta = builder.build_chunk_metadata(
            base_metadata=base,
            chunk_index=0,
            chunk_start=0,
            chunk_end=5000,
            speaker="A",
            chunk_type="speaker",
        )
        
        assert chunk_meta["chunk_index"] == 0
        assert chunk_meta["chunk_start_ms"] == 0
        assert chunk_meta["chunk_end_ms"] == 5000
        assert chunk_meta["chunk_duration_sec"] == 5.0
        assert chunk_meta["speaker"] == "A"
        assert chunk_meta["chunk_type"] == "speaker"


class TestTranscriptDocumentBuilder:
    """Tests for TranscriptDocumentBuilder."""
    
    def test_build_single_document(
        self,
        simple_transcript: Dict[str, Any],
        extracted_data: Dict[str, Dict[str, Any]],
    ) -> None:
        """Test building a single document from transcript."""
        builder = TranscriptDocumentBuilder()
        
        docs = builder.build(
            transcript_data=simple_transcript,
            extracted_data=extracted_data,
            source_name="test.mp3",
        )
        
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].content == simple_transcript["text"]
        assert docs[0].meta["source_name"] == "test.mp3"
    
    def test_build_with_empty_extracted(
        self,
        simple_transcript: Dict[str, Any],
    ) -> None:
        """Test building document without extracted data."""
        builder = TranscriptDocumentBuilder()
        
        docs = builder.build(
            transcript_data=simple_transcript,
            extracted_data={},
            source_name="test.mp3",
        )
        
        assert len(docs) == 1
        assert docs[0].meta["extracted_features"] == []
    
    def test_build_from_chunks(
        self,
        simple_transcript: Dict[str, Any],
        extracted_data: Dict[str, Dict[str, Any]],
    ) -> None:
        """Test building documents from chunks."""
        builder = TranscriptDocumentBuilder()
        
        chunks = [
            Chunk(text="Hello world.", start_time=0, end_time=2000, speaker="A"),
            Chunk(text="This is a test.", start_time=2000, end_time=5000, speaker="B"),
        ]
        
        docs = builder.build_from_chunks(
            chunks=chunks,
            transcript_data=simple_transcript,
            extracted_data=extracted_data,
            source_name="test.mp3",
        )
        
        assert len(docs) == 2
        assert docs[0].content == "Hello world."
        assert docs[0].meta["chunk_index"] == 0
        assert docs[0].meta["speaker"] == "A"
        assert docs[1].content == "This is a test."
        assert docs[1].meta["speaker"] == "B"
    
    def test_build_metadata(
        self,
        simple_transcript: Dict[str, Any],
        extracted_data: Dict[str, Dict[str, Any]],
    ) -> None:
        """Test building just metadata."""
        builder = TranscriptDocumentBuilder()
        
        meta = builder.build_metadata(
            transcript_data=simple_transcript,
            extracted_data=extracted_data,
            source_name="test.mp3",
        )
        
        assert meta["source_name"] == "test.mp3"
        assert meta["transcript_id"] == "test-123"
        assert "sentiment" in meta["extracted_features"]
    
    def test_build_utterance_documents(
        self,
        transcript_response: Dict[str, Any],
    ) -> None:
        """Test building documents from utterances."""
        builder = TranscriptDocumentBuilder()
        
        docs = builder.build_utterance_documents(
            transcript_data=transcript_response,
            extracted_data={},
            source_name="test.mp3",
        )
        
        # Fixture has 3 utterances
        assert len(docs) == 3
        assert all(d.meta["chunk_type"] == "utterance" for d in docs)
    
    def test_build_chapter_documents(
        self,
        transcript_response: Dict[str, Any],
    ) -> None:
        """Test building documents from chapters."""
        builder = TranscriptDocumentBuilder()
        
        docs = builder.build_chapter_documents(
            transcript_data=transcript_response,
            extracted_data={},
            source_name="test.mp3",
        )
        
        # Fixture has 2 chapters
        assert len(docs) == 2
        assert all(d.meta["chunk_type"] == "chapter" for d in docs)
        assert docs[0].meta.get("chapter_headline") == "Introduction"
    
    def test_fallback_without_utterances(
        self,
        simple_transcript: Dict[str, Any],
    ) -> None:
        """Test fallback when no utterances available."""
        builder = TranscriptDocumentBuilder()
        
        docs = builder.build_utterance_documents(
            transcript_data=simple_transcript,
            extracted_data={},
            source_name="test.mp3",
        )
        
        # Should fall back to single document
        assert len(docs) == 1
    
    def test_fallback_without_chapters(
        self,
        simple_transcript: Dict[str, Any],
    ) -> None:
        """Test fallback when no chapters available."""
        builder = TranscriptDocumentBuilder()
        
        docs = builder.build_chapter_documents(
            transcript_data=simple_transcript,
            extracted_data={},
            source_name="test.mp3",
        )
        
        # Should fall back to single document
        assert len(docs) == 1


class TestDocumentBuilderWithRealFixtures:
    """Tests using real fixture data."""
    
    def test_build_with_full_fixtures(
        self,
        transcript_response: Dict[str, Any],
        analysis_data: Dict[str, Any],
    ) -> None:
        """Test building with complete fixture data."""
        builder = TranscriptDocumentBuilder()
        
        # Create extracted data structure from analysis fixture
        extracted = {
            "sentiment": {
                "items": analysis_data.get("sentiment_analysis", []),
                "distribution": {"NEUTRAL": 2, "POSITIVE": 1},
                "total_count": 3,
            },
            "entities": {
                "items": analysis_data.get("entities", []),
                "entities_by_type": {"TOPIC": ["technology", "innovation"]},
                "total_count": 2,
            },
        }
        
        docs = builder.build(
            transcript_data=transcript_response,
            extracted_data=extracted,
            source_name="podcast.mp3",
        )
        
        assert len(docs) == 1
        assert docs[0].meta["source_name"] == "podcast.mp3"
        assert "sentiment" in docs[0].meta["extracted_features"]
        assert "entities" in docs[0].meta["extracted_features"]
    
    def test_chunk_documents_preserve_source(
        self,
        transcript_response: Dict[str, Any],
    ) -> None:
        """Test that chunk documents preserve source information."""
        builder = TranscriptDocumentBuilder()
        
        chunks = [
            Chunk(text="First chunk", start_time=0, end_time=5000),
            Chunk(text="Second chunk", start_time=5000, end_time=10000),
        ]
        
        docs = builder.build_from_chunks(
            chunks=chunks,
            transcript_data=transcript_response,
            extracted_data={},
            source_name="podcast.mp3",
        )
        
        # All chunks should have the same source
        for doc in docs:
            assert doc.meta["source_name"] == "podcast.mp3"
            assert doc.meta["transcript_id"] == transcript_response["id"]

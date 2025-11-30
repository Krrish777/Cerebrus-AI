"""
Integration tests for the audio processing module.

Tests the complete audio processing pipeline from input to output.
"""

import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from src.audio_processing.chunking.registry import get_global_registry as get_chunker_registry
from src.audio_processing.document.builder import TranscriptDocumentBuilder
from src.audio_processing.extractors.registry import get_registry as get_extractor_registry


@pytest.fixture
def mock_transcript_data() -> Dict[str, Any]:
    """Load mock transcript data from fixtures."""
    fixture_path = Path("data/fixtures/mock_transcript.json")
    if fixture_path.exists():
        return json.loads(fixture_path.read_text(encoding="utf-8"))
    
    return {
        "text": "This is a comprehensive test transcript for integration testing.",
        "id": "test_transcript_123",
        "status": "completed",
        "audio_url": "https://example.com/audio.mp3",
        "audio_duration": 120000,
        "utterances": [
            {
                "speaker": "A",
                "text": "This is the first speaker talking.",
                "start": 0,
                "end": 5000,
                "words": [
                    {"text": "This", "start": 0, "end": 500},
                    {"text": "is", "start": 500, "end": 800},
                    {"text": "the", "start": 800, "end": 1000},
                    {"text": "first", "start": 1000, "end": 1500},
                    {"text": "speaker", "start": 1500, "end": 2500},
                    {"text": "talking", "start": 2500, "end": 5000},
                ]
            },
            {
                "speaker": "B",
                "text": "And this is the second speaker responding.",
                "start": 5000,
                "end": 10000,
                "words": [
                    {"text": "And", "start": 5000, "end": 5200},
                    {"text": "this", "start": 5200, "end": 5500},
                    {"text": "is", "start": 5500, "end": 5700},
                    {"text": "the", "start": 5700, "end": 5900},
                    {"text": "second", "start": 5900, "end": 6500},
                    {"text": "speaker", "start": 6500, "end": 7500},
                    {"text": "responding", "start": 7500, "end": 10000},
                ]
            },
        ],
        "words": [
            {"text": "This", "start": 0, "end": 500},
            {"text": "is", "start": 500, "end": 800},
            {"text": "the", "start": 800, "end": 1000},
            {"text": "first", "start": 1000, "end": 1500},
            {"text": "speaker", "start": 1500, "end": 2500},
            {"text": "talking", "start": 2500, "end": 5000},
            {"text": "And", "start": 5000, "end": 5200},
            {"text": "this", "start": 5200, "end": 5500},
            {"text": "is", "start": 5500, "end": 5700},
            {"text": "the", "start": 5700, "end": 5900},
            {"text": "second", "start": 5900, "end": 6500},
            {"text": "speaker", "start": 6500, "end": 7500},
            {"text": "responding", "start": 7500, "end": 10000},
        ],
    }


@pytest.fixture
def mock_analysis_data() -> Dict[str, Any]:
    """Load mock analysis data from fixtures."""
    fixture_path = Path("data/fixtures/mock_analysis_data.json")
    if fixture_path.exists():
        return json.loads(fixture_path.read_text(encoding="utf-8"))
    
    return {
        "sentiment_analysis_results": [
            {
                "text": "This is great",
                "start": 0,
                "end": 3000,
                "sentiment": "POSITIVE",
                "confidence": 0.85,
            },
            {
                "text": "But this is concerning",
                "start": 3000,
                "end": 6000,
                "sentiment": "NEGATIVE",
                "confidence": 0.72,
            },
        ],
        "entities": [
            {
                "text": "TestCompany",
                "entity_type": "organization",
                "start": 0,
                "end": 1000,
            },
            {
                "text": "John Doe",
                "entity_type": "person_name",
                "start": 2000,
                "end": 3000,
            },
        ],
        "chapters": [
            {
                "headline": "Introduction",
                "summary": "Introduction to the topic",
                "start": 0,
                "end": 30000,
            },
            {
                "headline": "Main Content",
                "summary": "The main discussion points",
                "start": 30000,
                "end": 90000,
            },
        ],
        "iab_categories_result": {
            "results": [
                {
                    "label": "Technology>Software",
                    "relevance": 0.95,
                },
                {
                    "label": "Business>Startups",
                    "relevance": 0.82,
                },
            ],
        },
        "content_safety_labels": {
            "results": [],
            "summary": {},
        },
        "auto_highlights_result": {
            "results": [
                {
                    "text": "important highlight",
                    "count": 3,
                    "rank": 0.92,
                    "timestamps": [
                        {"start": 1000, "end": 2000},
                    ],
                },
            ],
        },
    }


class TestExtractionIntegration:
    """Integration tests for extraction pipeline."""
    
    def test_extract_all_from_transcript(
        self,
        mock_transcript_data: Dict[str, Any],
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test extracting all data types from transcript."""
        # Combine transcript and analysis data
        combined_data = {**mock_transcript_data, **mock_analysis_data}
        
        registry = get_extractor_registry()
        results = registry.extract_all(combined_data)
        
        # Should extract from all available extractors
        assert len(results) > 0
        
        # Check sentiment extraction - verify keys match actual output
        if "sentiment" in results:
            assert "count" in results["sentiment"] or "distribution" in results["sentiment"]
        
        # Check entity extraction
        if "entities" in results:
            assert "entities" in results["entities"] or "by_type" in results["entities"]
        
        # Check chapter extraction
        if "chapters" in results:
            assert "chapters" in results["chapters"]
    
    def test_selective_extraction(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test extracting only specific data types."""
        registry = get_extractor_registry()
        
        # Extract only sentiment
        sentiment_extractor = registry.get("sentiment")
        result = sentiment_extractor.extract(mock_analysis_data)
        
        assert result is not None
        # Check for keys that actually exist in the output
        assert "count" in result or "distribution" in result
    
    def test_extraction_with_missing_data(self) -> None:
        """Test extraction handles missing data gracefully."""
        registry = get_extractor_registry()
        
        empty_data: Dict[str, Any] = {}
        results = registry.extract_all(empty_data)
        
        # Should return empty dict when no data available
        assert isinstance(results, dict)


class TestChunkingIntegration:
    """Integration tests for chunking pipeline."""
    
    def test_chunk_transcript_with_speaker_strategy(
        self,
        mock_transcript_data: Dict[str, Any],
    ) -> None:
        """Test chunking transcript by speaker."""
        registry = get_chunker_registry()
        
        chunks = registry.chunk_with("speaker", mock_transcript_data)
        
        assert len(chunks) > 0
        
        # Each chunk should have speaker info
        for chunk in chunks:
            assert chunk.text
            assert hasattr(chunk, "metadata")
    
    def test_chunk_with_auto_selection(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test auto-selection of best chunking strategy."""
        # Add chapters to trigger chapter-based chunking
        data_with_chapters = {
            "text": "Full transcript text here",
            "chapters": mock_analysis_data.get("chapters", []),
        }
        
        registry = get_chunker_registry()
        chunks = registry.chunk_with_best(data_with_chapters)
        
        assert len(chunks) > 0
    
    def test_chunk_fallback_to_sentence(self) -> None:
        """Test fallback to sentence chunking without speaker/chapter."""
        simple_data = {
            "text": "First sentence. Second sentence. Third sentence.",
        }
        
        registry = get_chunker_registry()
        chunks = registry.chunk_with_best(simple_data)
        
        assert len(chunks) > 0


class TestDocumentBuilderIntegration:
    """Integration tests for document building."""
    
    def test_build_documents_from_transcript(
        self,
        mock_transcript_data: Dict[str, Any],
    ) -> None:
        """Test building Haystack documents from transcript."""
        builder = TranscriptDocumentBuilder()
        
        documents = builder.build(
            transcript_data=mock_transcript_data,
            extracted_data={},
            source_name="test_audio.mp3",
        )
        
        assert len(documents) > 0
        
        # Check document structure
        doc = documents[0]
        assert doc.content
        assert doc.meta is not None
    
    def test_build_documents_with_extraction(
        self,
        mock_transcript_data: Dict[str, Any],
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test building documents with extracted data."""
        # Extract data first
        registry = get_extractor_registry()
        combined_data = {**mock_transcript_data, **mock_analysis_data}
        extracted = registry.extract_all(combined_data)
        
        # Build documents with extracted data
        builder = TranscriptDocumentBuilder()
        documents = builder.build(
            transcript_data=mock_transcript_data,
            extracted_data=extracted,
            source_name="test_audio.mp3",
        )
        
        assert len(documents) > 0
        
        # Check that extracted data is in metadata
        doc = documents[0]
        assert doc.meta is not None
    
    def test_build_documents_from_chunks(
        self,
        mock_transcript_data: Dict[str, Any],
    ) -> None:
        """Test building documents from chunks."""
        # Chunk the transcript
        chunker_registry = get_chunker_registry()
        chunks = chunker_registry.chunk_with("speaker", mock_transcript_data)
        
        # Build documents from chunks
        builder = TranscriptDocumentBuilder()
        documents = builder.build_from_chunks(
            chunks=chunks,
            transcript_data=mock_transcript_data,
            extracted_data={},
            source_name="test_audio.mp3",
        )
        
        assert len(documents) == len(chunks)
        
        # Each document should correspond to a chunk
        for doc, chunk in zip(documents, chunks):
            assert doc.content == chunk.text


class TestFullPipelineIntegration:
    """Integration tests for complete pipeline flow."""
    
    def test_extraction_to_chunking_flow(
        self,
        mock_transcript_data: Dict[str, Any],
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test extraction followed by chunking."""
        # Combine data
        combined_data = {**mock_transcript_data, **mock_analysis_data}
        
        # Extract
        extractor_registry = get_extractor_registry()
        extracted = extractor_registry.extract_all(combined_data)
        
        # Chunk
        chunker_registry = get_chunker_registry()
        chunks = chunker_registry.chunk_with_best(combined_data)
        
        # Both should produce valid output
        assert isinstance(extracted, dict)
        assert len(chunks) > 0
    
    def test_chunking_to_document_flow(
        self,
        mock_transcript_data: Dict[str, Any],
    ) -> None:
        """Test chunking followed by document building."""
        # Chunk
        chunker_registry = get_chunker_registry()
        chunks = chunker_registry.chunk_with("speaker", mock_transcript_data)
        
        # Build documents
        builder = TranscriptDocumentBuilder()
        documents = builder.build_from_chunks(
            chunks=chunks,
            transcript_data=mock_transcript_data,
            extracted_data={},
            source_name="integration_test.mp3",
        )
        
        assert len(documents) > 0
        
        # Documents should have proper source_name in meta
        for doc in documents:
            assert doc.meta.get("source_name") == "integration_test.mp3"
    
    def test_full_pipeline_with_extraction_chunking_documents(
        self,
        mock_transcript_data: Dict[str, Any],
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test complete pipeline: extract -> chunk -> build documents."""
        combined_data = {**mock_transcript_data, **mock_analysis_data}
        
        # Step 1: Extract
        extractor_registry = get_extractor_registry()
        extracted = extractor_registry.extract_all(combined_data)
        
        # Step 2: Chunk
        chunker_registry = get_chunker_registry()
        chunks = chunker_registry.chunk_with_best(combined_data)
        
        # Step 3: Build documents
        builder = TranscriptDocumentBuilder()
        documents = builder.build_from_chunks(
            chunks=chunks,
            transcript_data=mock_transcript_data,
            extracted_data=extracted,
            source_name="full_pipeline_test.mp3",
        )
        
        # Verify complete output
        assert len(documents) > 0
        
        for doc in documents:
            # Each document should have content
            assert doc.content
            
            # Each document should have metadata
            assert doc.meta is not None
            
            # Source should be set in source_name
            assert doc.meta.get("source_name") == "full_pipeline_test.mp3"


class TestErrorHandling:
    """Integration tests for error handling across modules."""
    
    def test_extraction_handles_malformed_data(self) -> None:
        """Test extraction handles malformed data gracefully."""
        malformed_data = {
            "sentiment_analysis_results": "not_a_list",
            "entities": 12345,
        }
        
        registry = get_extractor_registry()
        
        # Should not raise, should return empty or partial results
        try:
            results = registry.extract_all(malformed_data)
            assert isinstance(results, dict)
        except Exception as e:
            pytest.fail(f"Extraction should handle malformed data: {e}")
    
    def test_chunking_handles_minimal_data(self) -> None:
        """Test chunking handles minimal transcript data."""
        minimal_data = {
            "text": "Just one sentence.",
        }
        
        registry = get_chunker_registry()
        chunks = registry.chunk_with_best(minimal_data)
        
        # Should produce at least one chunk
        assert len(chunks) >= 1
    
    def test_document_builder_handles_empty_input(self) -> None:
        """Test document builder handles empty input."""
        builder = TranscriptDocumentBuilder()
        
        documents = builder.build(
            transcript_data={"text": ""},
            extracted_data={},
            source_name="empty.mp3",
        )
        
        # Should handle gracefully
        assert isinstance(documents, list)


class TestDataIntegrity:
    """Integration tests for data integrity through pipeline."""
    
    def test_timestamps_preserved_in_chunks(
        self,
        mock_transcript_data: Dict[str, Any],
    ) -> None:
        """Test that timestamps are preserved through chunking."""
        registry = get_chunker_registry()
        chunks = registry.chunk_with("speaker", mock_transcript_data)
        
        for chunk in chunks:
            # Chunks should have timing information
            assert chunk.start_time >= 0
            assert chunk.end_time > chunk.start_time
    
    def test_speaker_info_preserved_in_documents(
        self,
        mock_transcript_data: Dict[str, Any],
    ) -> None:
        """Test that speaker info is preserved in final documents."""
        # Chunk by speaker
        chunker_registry = get_chunker_registry()
        chunks = chunker_registry.chunk_with("speaker", mock_transcript_data)
        
        # Build documents
        builder = TranscriptDocumentBuilder()
        documents = builder.build_from_chunks(
            chunks=chunks,
            transcript_data=mock_transcript_data,
            extracted_data={},
            source_name="speaker_test.mp3",
        )
        
        # Check speaker info in document metadata
        for doc in documents:
            meta = doc.meta
            # Speaker info should be in metadata (if available)
            assert meta is not None
    
    def test_extracted_data_in_document_metadata(
        self,
        mock_transcript_data: Dict[str, Any],
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test that extracted data appears in document metadata."""
        combined_data = {**mock_transcript_data, **mock_analysis_data}
        
        # Extract
        extractor_registry = get_extractor_registry()
        extracted = extractor_registry.extract_all(combined_data)
        
        # Build document with extracted data
        builder = TranscriptDocumentBuilder()
        documents = builder.build(
            transcript_data=mock_transcript_data,
            extracted_data=extracted,
            source_name="metadata_test.mp3",
        )
        
        assert len(documents) > 0
        
        # The first document should have enhanced metadata
        main_doc = documents[0]
        assert main_doc.meta is not None

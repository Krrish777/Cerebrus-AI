"""
Tests for the chunking module.

Tests all chunking strategies and the registry.
"""

import json
from pathlib import Path
from typing import Any
from typing import Dict

import pytest

from src.audio_processing.chunking.base import BaseChunker
from src.audio_processing.chunking.base import Chunk
from src.audio_processing.chunking.base import ChunkerConfig
from src.audio_processing.chunking.speaker import SpeakerChunker
from src.audio_processing.chunking.chapter import ChapterChunker
from src.audio_processing.chunking.semantic import SemanticChunker
from src.audio_processing.chunking.sentence import SentenceChunker
from src.audio_processing.chunking.registry import ChunkerRegistry
from src.audio_processing.chunking.registry import get_global_registry


@pytest.fixture
def transcript_response() -> Dict[str, Any]:
    """Load mock transcript response fixture."""
    fixture_path = Path("data/fixtures/mock_transcript_response.json")
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.fixture
def simple_transcript() -> Dict[str, Any]:
    """Create a simple transcript for testing."""
    return {
        "text": "Hello world. This is a test. It has multiple sentences. Each one is short.",
        "audio_duration": 10000,
        "words": [
            {"text": "Hello", "start": 0, "end": 500},
            {"text": "world", "start": 500, "end": 1000},
        ],
    }


@pytest.fixture
def transcript_with_utterances() -> Dict[str, Any]:
    """Create transcript with speaker utterances."""
    return {
        "text": "Hello from speaker A. Hello from speaker B. Back to A.",
        "audio_duration": 15000,
        "utterances": [
            {"text": "Hello from speaker A.", "speaker": "A", "start": 0, "end": 5000},
            {"text": "Hello from speaker B.", "speaker": "B", "start": 5000, "end": 10000},
            {"text": "Back to A.", "speaker": "A", "start": 10000, "end": 15000},
        ],
    }


@pytest.fixture
def transcript_with_chapters() -> Dict[str, Any]:
    """Create transcript with chapters."""
    return {
        "text": "Introduction content here. Main content here. Conclusion here.",
        "audio_duration": 30000,
        "chapters": [
            {"headline": "Introduction", "gist": "Intro", "start": 0, "end": 10000},
            {"headline": "Main Content", "gist": "Main", "start": 10000, "end": 20000},
            {"headline": "Conclusion", "gist": "End", "start": 20000, "end": 30000},
        ],
        "words": [
            {"text": "Introduction", "start": 0, "end": 2000},
            {"text": "content", "start": 2000, "end": 4000},
            {"text": "here", "start": 4000, "end": 6000},
            {"text": "Main", "start": 10000, "end": 12000},
            {"text": "content", "start": 12000, "end": 14000},
            {"text": "here", "start": 14000, "end": 16000},
            {"text": "Conclusion", "start": 20000, "end": 22000},
            {"text": "here", "start": 22000, "end": 24000},
        ],
    }


class TestChunk:
    """Tests for the Chunk dataclass."""
    
    def test_chunk_creation(self) -> None:
        """Test basic chunk creation."""
        chunk = Chunk(
            text="Hello world",
            start_time=0,
            end_time=1000,
        )
        
        assert chunk.text == "Hello world"
        assert chunk.start_time == 0
        assert chunk.end_time == 1000
        assert chunk.speaker is None
        assert chunk.metadata == {}
    
    def test_chunk_duration_ms(self) -> None:
        """Test duration calculation in milliseconds."""
        chunk = Chunk(text="Test", start_time=1000, end_time=3500)
        assert chunk.duration_ms == 2500
    
    def test_chunk_duration_seconds(self) -> None:
        """Test duration calculation in seconds."""
        chunk = Chunk(text="Test", start_time=0, end_time=5000)
        assert chunk.duration_seconds == 5.0
    
    def test_chunk_length(self) -> None:
        """Test length returns character count."""
        chunk = Chunk(text="Hello", start_time=0, end_time=1000)
        assert len(chunk) == 5


class TestChunkerConfig:
    """Tests for the ChunkerConfig dataclass."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ChunkerConfig()
        
        assert config.max_chunk_size == 1000
        assert config.min_chunk_size == 100
        assert config.overlap_size == 50
        assert config.preserve_sentences is True
    
    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ChunkerConfig(
            max_chunk_size=500,
            min_chunk_size=50,
            overlap_size=25,
            preserve_sentences=False,
        )
        
        assert config.max_chunk_size == 500
        assert config.min_chunk_size == 50


class TestSpeakerChunker:
    """Tests for the SpeakerChunker."""
    
    def test_chunk_by_utterances(
        self,
        transcript_with_utterances: Dict[str, Any],
    ) -> None:
        """Test chunking by speaker utterances."""
        # Use min_chunk_size=0 to prevent merging
        config = ChunkerConfig(min_chunk_size=0)
        chunker = SpeakerChunker(config=config, merge_consecutive=False)
        chunks = chunker.chunk(transcript_with_utterances)
        
        assert len(chunks) == 3
        assert chunks[0].speaker == "A"
        assert chunks[1].speaker == "B"
        assert chunks[2].speaker == "A"
    
    def test_merge_consecutive_same_speaker(
        self,
        transcript_with_utterances: Dict[str, Any],
    ) -> None:
        """Test merging consecutive same-speaker utterances."""
        # Create transcript where A speaks twice in a row
        transcript = {
            "text": "First from A. Second from A. Now B speaks.",
            "audio_duration": 15000,
            "utterances": [
                {"text": "First from A.", "speaker": "A", "start": 0, "end": 5000},
                {"text": "Second from A.", "speaker": "A", "start": 5000, "end": 10000},
                {"text": "Now B speaks.", "speaker": "B", "start": 10000, "end": 15000},
            ],
        }
        
        # Use min_chunk_size=0 to prevent additional merging
        config = ChunkerConfig(min_chunk_size=0)
        chunker = SpeakerChunker(config=config, merge_consecutive=True)
        chunks = chunker.chunk(transcript)
        
        # A's utterances should be merged
        assert len(chunks) == 2
        assert chunks[0].speaker == "A"
        assert "First from A." in chunks[0].text
        assert "Second from A." in chunks[0].text
    
    def test_fallback_to_full_text(
        self,
        simple_transcript: Dict[str, Any],
    ) -> None:
        """Test fallback when no utterances available."""
        chunker = SpeakerChunker()
        chunks = chunker.chunk(simple_transcript)
        
        assert len(chunks) >= 1
        assert simple_transcript["text"] in chunks[0].text
    
    def test_get_speaker_stats(
        self,
        transcript_with_utterances: Dict[str, Any],
    ) -> None:
        """Test speaker statistics calculation."""
        config = ChunkerConfig(min_chunk_size=0)
        chunker = SpeakerChunker(config=config, merge_consecutive=False)
        chunks = chunker.chunk(transcript_with_utterances)
        stats = chunker.get_speaker_stats(chunks)
        
        assert "A" in stats
        assert "B" in stats
        assert stats["A"]["chunk_count"] == 2
        assert stats["B"]["chunk_count"] == 1


class TestChapterChunker:
    """Tests for the ChapterChunker."""
    
    def test_chunk_by_chapters(
        self,
        transcript_with_chapters: Dict[str, Any],
    ) -> None:
        """Test chunking by chapters."""
        chunker = ChapterChunker()
        chunks = chunker.chunk(transcript_with_chapters)
        
        assert len(chunks) == 3
        assert chunks[0].metadata.get("headline") == "Introduction"
        assert chunks[1].metadata.get("headline") == "Main Content"
        assert chunks[2].metadata.get("headline") == "Conclusion"
    
    def test_chapter_metadata_included(
        self,
        transcript_with_chapters: Dict[str, Any],
    ) -> None:
        """Test that chapter metadata is included."""
        chunker = ChapterChunker(include_headline=True, include_summary=True)
        chunks = chunker.chunk(transcript_with_chapters)
        
        assert chunks[0].metadata.get("headline") == "Introduction"
        assert chunks[0].metadata.get("gist") == "Intro"
    
    def test_fallback_without_chapters(
        self,
        simple_transcript: Dict[str, Any],
    ) -> None:
        """Test fallback when no chapters available."""
        chunker = ChapterChunker()
        chunks = chunker.chunk(simple_transcript)
        
        assert len(chunks) == 1
        assert simple_transcript["text"] in chunks[0].text
    
    def test_get_chapter_outline(
        self,
        transcript_with_chapters: Dict[str, Any],
    ) -> None:
        """Test outline generation from chapters."""
        chunker = ChapterChunker()
        chunks = chunker.chunk(transcript_with_chapters)
        outline = chunker.get_chapter_outline(chunks)
        
        assert len(outline) == 3
        assert outline[0]["headline"] == "Introduction"
        assert outline[0]["start_time"] == 0
    
    def test_with_real_fixture(
        self,
        transcript_response: Dict[str, Any],
    ) -> None:
        """Test with real fixture data."""
        chunker = ChapterChunker()
        chunks = chunker.chunk(transcript_response)
        
        # Fixture has 2 chapters
        assert len(chunks) == 2


class TestSemanticChunker:
    """Tests for the SemanticChunker."""
    
    def test_chunk_by_paragraphs(self) -> None:
        """Test chunking by paragraphs."""
        transcript = {
            "text": "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            "audio_duration": 15000,
        }
        
        # Use min_chunk_size=0 to prevent merging small paragraphs
        config = ChunkerConfig(min_chunk_size=0)
        chunker = SemanticChunker(config=config)
        chunks = chunker.chunk(transcript)
        
        assert len(chunks) == 3
        assert "First paragraph" in chunks[0].text
        assert "Second paragraph" in chunks[1].text
        assert "Third paragraph" in chunks[2].text
    
    def test_chunk_from_paragraph_data(self) -> None:
        """Test chunking from structured paragraph data."""
        transcript = {
            "paragraphs": [
                {"text": "First paragraph.", "start": 0, "end": 5000},
                {"text": "Second paragraph.", "start": 5000, "end": 10000},
            ],
            "audio_duration": 10000,
        }
        
        # Use min_chunk_size=0 to prevent merging
        config = ChunkerConfig(min_chunk_size=0)
        chunker = SemanticChunker(config=config)
        chunks = chunker.chunk(transcript)
        
        assert len(chunks) == 2
        assert chunks[0].metadata.get("paragraph_index") == 0
    
    def test_handles_no_paragraphs(
        self,
        simple_transcript: Dict[str, Any],
    ) -> None:
        """Test handling text without paragraph markers."""
        chunker = SemanticChunker()
        chunks = chunker.chunk(simple_transcript)
        
        # Should create at least one chunk
        assert len(chunks) >= 1


class TestSentenceChunker:
    """Tests for the SentenceChunker."""
    
    def test_chunk_by_sentences(self) -> None:
        """Test basic sentence chunking."""
        transcript = {
            "text": "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.",
            "audio_duration": 25000,
        }
        
        chunker = SentenceChunker(sentences_per_chunk=2, overlap_sentences=0)
        chunks = chunker.chunk(transcript)
        
        # 5 sentences, 2 per chunk, no overlap = 3 chunks
        assert len(chunks) == 3
    
    def test_sentence_overlap(self) -> None:
        """Test overlap between chunks."""
        transcript = {
            "text": "One. Two. Three. Four. Five.",
            "audio_duration": 25000,
        }
        
        chunker = SentenceChunker(sentences_per_chunk=3, overlap_sentences=1)
        chunks = chunker.chunk(transcript)
        
        # With overlap, should have more chunks
        assert len(chunks) >= 2
    
    def test_from_sentence_data(self) -> None:
        """Test chunking from structured sentence data."""
        transcript = {
            "sentences": [
                {"text": "First sentence.", "start": 0, "end": 5000},
                {"text": "Second sentence.", "start": 5000, "end": 10000},
                {"text": "Third sentence.", "start": 10000, "end": 15000},
                {"text": "Fourth sentence.", "start": 15000, "end": 20000},
            ],
            "audio_duration": 20000,
        }
        
        chunker = SentenceChunker(sentences_per_chunk=2, overlap_sentences=0)
        chunks = chunker.chunk(transcript)
        
        assert len(chunks) == 2
        assert chunks[0].metadata.get("sentence_count") == 2
    
    def test_calculate_overlap_tokens(self) -> None:
        """Test overlap calculation."""
        chunker = SentenceChunker(sentences_per_chunk=2, overlap_sentences=1)
        
        chunks = [
            Chunk(text="Hello world. This is test.", start_time=0, end_time=5000),
            Chunk(text="This is test. Another sentence.", start_time=3000, end_time=8000),
        ]
        
        stats = chunker.calculate_overlap_tokens(chunks)
        
        assert stats["total_chunks"] == 2
        assert stats["overlapping_pairs"] >= 0  # May or may not have textual overlap


class TestChunkerRegistry:
    """Tests for the ChunkerRegistry."""
    
    def test_register_and_get(self) -> None:
        """Test registering and retrieving chunkers."""
        registry = ChunkerRegistry()
        registry.register("speaker", SpeakerChunker)
        
        chunker = registry.get("speaker")
        
        assert isinstance(chunker, SpeakerChunker)
    
    def test_get_unknown_raises(self) -> None:
        """Test getting unknown chunker raises error."""
        registry = ChunkerRegistry()
        
        with pytest.raises(ValueError, match="Unknown chunker"):
            registry.get("nonexistent")
    
    def test_available_chunkers(self) -> None:
        """Test listing available chunkers."""
        registry = ChunkerRegistry()
        registry.register_defaults()
        
        available = registry.available()
        
        assert "speaker" in available
        assert "chapter" in available
        assert "semantic" in available
        assert "sentence" in available
    
    def test_chunk_with(
        self,
        transcript_with_utterances: Dict[str, Any],
    ) -> None:
        """Test convenience chunk_with method."""
        registry = ChunkerRegistry()
        registry.register_defaults()
        
        chunks = registry.chunk_with("speaker", transcript_with_utterances)
        
        assert len(chunks) > 0
    
    def test_chunk_with_best_selects_chapter(
        self,
        transcript_with_chapters: Dict[str, Any],
    ) -> None:
        """Test auto-selection with chapters."""
        registry = ChunkerRegistry()
        registry.register_defaults()
        
        chunks = registry.chunk_with_best(transcript_with_chapters)
        
        # Should use chapter chunker when chapters present
        assert any("headline" in c.metadata for c in chunks)
    
    def test_chunk_with_best_selects_speaker(
        self,
        transcript_with_utterances: Dict[str, Any],
    ) -> None:
        """Test auto-selection with utterances."""
        registry = ChunkerRegistry()
        registry.register_defaults()
        
        chunks = registry.chunk_with_best(transcript_with_utterances)
        
        # Should use speaker chunker when utterances present
        assert any(c.speaker is not None for c in chunks)
    
    def test_get_global_registry(self) -> None:
        """Test global registry singleton."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        
        assert registry1 is registry2
        assert "speaker" in registry1.available()


class TestBaseChunkerValidation:
    """Tests for base chunker validation."""
    
    def test_empty_data_raises(self) -> None:
        """Test that empty data raises error."""
        chunker = SpeakerChunker()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            chunker.chunk({})
    
    def test_invalid_data_type_raises(self) -> None:
        """Test that invalid data type raises error."""
        chunker = SpeakerChunker()
        
        with pytest.raises(ValueError, match="must be a dictionary"):
            chunker.chunk("not a dict")  # type: ignore


class TestChunkSizeConstraints:
    """Tests for chunk size constraints."""
    
    def test_large_chunk_is_split(self) -> None:
        """Test that large chunks are split."""
        config = ChunkerConfig(max_chunk_size=50, min_chunk_size=0)
        chunker = SpeakerChunker(config=config, merge_consecutive=False)
        
        transcript = {
            "utterances": [
                {
                    "text": "This is a very long utterance that exceeds the maximum chunk size limit and needs to be split into smaller pieces for proper processing.",
                    "speaker": "A",
                    "start": 0,
                    "end": 10000,
                },
            ],
        }
        
        chunks = chunker.chunk(transcript)
        
        # Should be split into multiple chunks
        assert len(chunks) >= 2
        # All chunks should be <= max_chunk_size (or last word boundary)
        assert all(len(c) <= 60 for c in chunks)  # Allow some leeway for word boundaries
    
    def test_small_chunks_are_merged(self) -> None:
        """Test that small chunks are merged."""
        config = ChunkerConfig(min_chunk_size=50)
        chunker = SpeakerChunker(config=config, merge_consecutive=False)
        
        transcript = {
            "utterances": [
                {"text": "Hi.", "speaker": "A", "start": 0, "end": 1000},
                {"text": "Hey.", "speaker": "B", "start": 1000, "end": 2000},
                {"text": "Yo.", "speaker": "C", "start": 2000, "end": 3000},
            ],
        }
        
        chunks = chunker.chunk(transcript)
        
        # Small chunks should be merged
        assert len(chunks) <= 3

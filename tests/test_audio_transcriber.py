"""
Test suite for AssemblyAI Haystack Integration

Tests the comprehensive audio processing functionality including:
- Local audio file transcription (harvard.wav)
- YouTube URL transcription
- Speaker diarization
- Content analysis features
- Smart chunking strategies
- Error handling

Author: AI Assistant
Date: 2025-01-14
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import tempfile
import json

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_processing.audio_transcriber import (
    AudioProcessingConfig,
    AssemblyAITranscriber,
    SmartAudioProcessor,
    create_audio_pipeline,
    create_advanced_audio_config,
    ASSEMBLYAI_AVAILABLE
)
from haystack import Document


class TestAudioProcessingConfig:
    """Test the AudioProcessingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AudioProcessingConfig()
        
        assert config.language_code == "en"
        assert config.model == "best"
        assert config.speaker_labels is True
        assert config.sentiment_analysis is True
        assert config.entity_detection is True
        assert config.iab_categories is True
        assert config.content_safety is True
        assert config.auto_highlights is True
        assert config.automatic_punctuation is True
        assert config.format_text is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AudioProcessingConfig(
            language_code="es",
            model="nano",
            speaker_labels=False,
            sentiment_analysis=False,
            custom_vocabulary=["AI", "machine learning"]
        )
        
        assert config.language_code == "es"
        assert config.model == "nano"
        assert config.speaker_labels is False
        assert config.sentiment_analysis is False
        assert config.custom_vocabulary == ["AI", "machine learning"]
    
    def test_redaction_config(self):
        """Test PII redaction configuration."""
        config = AudioProcessingConfig(
            redact_pii=True,
            redact_pii_policies=["email_address", "phone_number"]
        )
        
        assert config.redact_pii is True
        assert "email_address" in config.redact_pii_policies
        assert "phone_number" in config.redact_pii_policies


@pytest.fixture
def mock_assemblyai():
    """Mock AssemblyAI module and components."""
    with patch('audio_processing.audio_transcriber.aai') as mock_aai:
        # Mock transcript response
        mock_transcript = Mock()
        mock_transcript.id = "test_transcript_123"
        mock_transcript.text = "This is a test transcription of the audio file."
        mock_transcript.status = "completed"
        mock_transcript.error = None
        mock_transcript.audio_duration_seconds = 45.6
        mock_transcript.confidence = 0.95
        
        # Mock utterances for speaker diarization
        mock_utterance1 = Mock()
        mock_utterance1.speaker = "A"
        mock_utterance1.start = 0
        mock_utterance1.end = 2000
        mock_utterance1.text = "Hello, this is speaker A."
        
        mock_utterance2 = Mock()
        mock_utterance2.speaker = "B"
        mock_utterance2.start = 2500
        mock_utterance2.end = 4500
        mock_utterance2.text = "And this is speaker B responding."
        
        mock_transcript.utterances = [mock_utterance1, mock_utterance2]
        
        # Mock sentiment analysis
        mock_sentiment = Mock()
        mock_sentiment.text = "Hello, this is speaker A."
        mock_sentiment.sentiment = "positive"  # Changed to string instead of Mock
        mock_sentiment.confidence = 0.85
        mock_sentiment.start = 0
        mock_sentiment.end = 2000
        mock_sentiment.speaker = "A"
        
        mock_transcript.sentiment_analysis = [mock_sentiment]
        
        # Mock entities
        mock_entity = Mock()
        mock_entity.text = "John Doe"
        mock_entity.entity_type = "person_name"  # Changed to string instead of Mock
        mock_entity.start = 500
        mock_entity.end = 1500
        
        mock_transcript.entities = [mock_entity]
        
        # Mock chapters
        mock_chapter = Mock()
        mock_chapter.headline = "Introduction"
        mock_chapter.start = 0
        mock_chapter.end = 30000
        mock_chapter.summary = "Speaker introduces themselves"
        mock_chapter.gist = "Introduction and greeting"
        
        mock_transcript.chapters = [mock_chapter]
        
        # Mock summary
        mock_transcript.summary = "This audio contains a conversation between two speakers discussing AI technology."
        
        # Mock get_sentences and get_paragraphs methods
        mock_sentence = Mock()
        mock_sentence.text = "This is a test sentence."
        mock_sentence.start = 0
        mock_sentence.end = 2000
        
        mock_paragraph = Mock()
        mock_paragraph.text = "This is a test paragraph with multiple sentences."
        mock_paragraph.start = 0
        mock_paragraph.end = 4000
        
        mock_transcript.get_sentences = Mock(return_value=[mock_sentence])
        mock_transcript.get_paragraphs = Mock(return_value=[mock_paragraph])
        
        # Mock content safety
        mock_content_safety = Mock()
        mock_content_safety.summary = {"violence": 0.1, "hate_speech": 0.05}
        mock_content_safety.results = []
        mock_transcript.content_safety = mock_content_safety
        
        # Mock IAB categories
        mock_iab = Mock()
        mock_iab.summary = {"Technology>Artificial Intelligence": 0.9}
        mock_iab.results = []
        mock_transcript.iab_categories = mock_iab
        
        # Mock auto highlights
        mock_highlight = Mock()
        mock_highlight.text = "AI technology"
        mock_highlight.rank = 1
        mock_highlight.count = 3
        mock_timestamp = Mock()
        mock_timestamp.start = 1000
        mock_timestamp.end = 2000
        mock_highlight.timestamps = [mock_timestamp]
        
        mock_auto_highlights = Mock()
        mock_auto_highlights.results = [mock_highlight]
        mock_transcript.auto_highlights = mock_auto_highlights
        
        # Mock transcriber
        mock_transcriber = Mock()
        mock_transcriber.transcribe.return_value = mock_transcript
        mock_transcriber.upload_file.return_value = "https://upload.assemblyai.com/test"
        
        mock_aai.Transcriber.return_value = mock_transcriber
        mock_aai.TranscriptionConfig.return_value = Mock()
        mock_aai.settings = Mock()
        mock_aai.TranscriptStatus = Mock()
        mock_aai.TranscriptStatus.error = "error"
        
        yield mock_aai


@pytest.fixture 
def sample_config():
    """Sample configuration for testing."""
    return AudioProcessingConfig(
        speaker_labels=True,
        sentiment_analysis=True,
        entity_detection=True,
        auto_highlights=True,
        summarization=True
    )


@pytest.fixture
def harvard_wav_path():
    """Path to the Harvard sentences audio file."""
    # In a real scenario, you would have this file in your test fixtures
    # For testing, we'll create a mock path
    test_audio_path = Path(__file__).parent / "fixtures" / "harvard.wav"
    return str(test_audio_path)


class TestAssemblyAITranscriber:
    """Test the AssemblyAITranscriber component."""
    
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="AssemblyAI API key required"):
                AssemblyAITranscriber()
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_init_with_api_key(self, mock_assemblyai):
        """Test successful initialization with API key."""
        transcriber = AssemblyAITranscriber(api_key="test_key")
        
        assert transcriber.api_key == "test_key"
        assert transcriber.polling_interval == 3.0
        assert isinstance(transcriber.config, AudioProcessingConfig)
    
    def test_init_without_assemblyai_package(self):
        """Test initialization fails when AssemblyAI package not available."""
        with patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', False):
            with pytest.raises(ImportError, match="assemblyai package is required"):
                AssemblyAITranscriber(api_key="test_key")
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_transcribe_local_file(self, mock_assemblyai, sample_config, harvard_wav_path):
        """Test transcribing a local audio file."""
        transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
        
        # Mock file reading
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b"fake_audio_data"
            
            result = transcriber.run(sources=[harvard_wav_path])
        
        assert "documents" in result
        documents = result["documents"]
        assert len(documents) > 0
        
        # Check main document
        main_doc = documents[0]
        assert isinstance(main_doc, Document)
        assert "Transcription:" in main_doc.content
        assert main_doc.meta["source"] == "harvard.wav"
        assert main_doc.meta["transcript_id"] == "test_transcript_123"
        assert main_doc.meta["confidence"] == 0.95
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)  
    def test_transcribe_youtube_url(self, mock_assemblyai, sample_config):
        """Test transcribing a YouTube URL."""
        transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
        youtube_url = "https://www.youtube.com/shorts/-Nw0Ts2n2nU"
        
        result = transcriber.run(sources=[youtube_url])
        
        assert "documents" in result
        documents = result["documents"]
        assert len(documents) > 0
        
        main_doc = documents[0]
        assert main_doc.meta["audio_url"] == youtube_url
        assert main_doc.meta["source"] == "-Nw0Ts2n2nU"
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_transcribe_bytes(self, mock_assemblyai, sample_config):
        """Test transcribing audio from bytes."""
        transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
        fake_audio_bytes = b"fake_audio_data"
        
        result = transcriber.run(sources=[fake_audio_bytes])
        
        assert "documents" in result
        documents = result["documents"]
        assert len(documents) > 0
        
        main_doc = documents[0]
        assert main_doc.meta["source"] == "uploaded_audio"
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_speaker_diarization_metadata(self, mock_assemblyai, sample_config):
        """Test that speaker diarization data is included in metadata."""
        transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
        
        with patch('builtins.open', create=True):
            result = transcriber.run(sources=["test.wav"])
        
        main_doc = result["documents"][0]
        content = main_doc.content
        
        # Check for speaker transcript section
        assert "## Speaker Transcript" in content
        assert "**Speaker A**" in content
        assert "**Speaker B**" in content
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_sentiment_analysis_metadata(self, mock_assemblyai, sample_config):
        """Test that sentiment analysis data is included in metadata."""
        transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
        
        with patch('builtins.open', create=True):
            result = transcriber.run(sources=["test.wav"])
        
        main_doc = result["documents"][0]
        sentiment_data = main_doc.meta.get("sentiment_analysis", [])
        
        assert len(sentiment_data) > 0
        assert sentiment_data[0]["sentiment"] == "positive"
        assert sentiment_data[0]["confidence"] == 0.85
        assert sentiment_data[0]["speaker"] == "A"
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_entity_detection_metadata(self, mock_assemblyai, sample_config):
        """Test that entity detection data is included in metadata."""
        transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
        
        with patch('builtins.open', create=True):
            result = transcriber.run(sources=["test.wav"])
        
        main_doc = result["documents"][0]
        entities = main_doc.meta.get("entities", [])
        
        assert len(entities) > 0
        assert entities[0]["text"] == "John Doe"
        assert entities[0]["entity_type"] == "person_name"
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_auto_highlights_metadata(self, mock_assemblyai, sample_config):
        """Test that auto highlights data is included in metadata."""
        transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
        
        with patch('builtins.open', create=True):
            result = transcriber.run(sources=["test.wav"])
        
        main_doc = result["documents"][0]
        highlights = main_doc.meta.get("highlights", [])
        
        assert len(highlights) > 0
        assert highlights[0]["text"] == "AI technology"
        assert highlights[0]["rank"] == 1
        assert len(highlights[0]["timestamps"]) > 0
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_chapter_content(self, mock_assemblyai):
        """Test that chapter content is properly formatted."""
        config = AudioProcessingConfig(auto_chapters=True)
        transcriber = AssemblyAITranscriber(api_key="test_key", config=config)
        
        with patch('builtins.open', create=True):
            result = transcriber.run(sources=["test.wav"])
        
        main_doc = result["documents"][0]
        content = main_doc.content
        
        assert "## Chapters" in content
        assert "### Chapter 1: Introduction" in content
        assert "**Time**: 0ms - 30000ms" in content
        assert "**Summary**: Speaker introduces themselves" in content
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True) 
    def test_summary_content(self, mock_assemblyai):
        """Test that summary content is included."""
        config = AudioProcessingConfig(summarization=True)
        transcriber = AssemblyAITranscriber(api_key="test_key", config=config)
        
        with patch('builtins.open', create=True):
            result = transcriber.run(sources=["test.wav"])
        
        main_doc = result["documents"][0]
        content = main_doc.content
        
        assert "## Summary" in content
        assert "conversation between two speakers" in content
    
    def test_serialization(self, sample_config):
        """Test component serialization and deserialization."""
        with patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True):
            with patch('audio_processing.audio_transcriber.aai'):
                transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
                
                # Test to_dict
                data = transcriber.to_dict()
                assert "init_parameters" in data
                assert data["init_parameters"]["api_key"] == "***"
                
                # Test from_dict
                restored = AssemblyAITranscriber.from_dict(data)
                assert isinstance(restored, AssemblyAITranscriber)


class TestSmartAudioProcessor:
    """Test the SmartAudioProcessor component."""
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_init(self, mock_assemblyai, sample_config):
        """Test SmartAudioProcessor initialization."""
        transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
        processor = SmartAudioProcessor(
            assemblyai_transcriber=transcriber,
            max_chunk_length=800,
            respect_speakers=True
        )
        
        assert processor.transcriber == transcriber
        assert processor.max_chunk_length == 800
        assert processor.respect_speakers is True
        assert processor.respect_chapters is True
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_speaker_aware_chunking(self, mock_assemblyai, sample_config):
        """Test chunking by speaker segments."""
        transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
        processor = SmartAudioProcessor(
            assemblyai_transcriber=transcriber,
            respect_speakers=True
        )
        
        with patch('builtins.open', create=True):
            result = processor.run(sources=["test.wav"])
        
        documents = result["documents"]
        
        # Should have multiple chunks based on speakers
        speaker_chunks = [doc for doc in documents if doc.meta.get("chunk_type") == "speaker_segment"]
        assert len(speaker_chunks) > 0
        
        # Check speaker metadata
        for chunk in speaker_chunks:
            assert "speaker" in chunk.meta
            assert "processing_strategy" in chunk.meta
            assert chunk.meta["processing_strategy"] == "speaker_aware"
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_chapter_aware_chunking(self, mock_assemblyai):
        """Test chunking by auto-generated chapters."""
        config = AudioProcessingConfig(auto_chapters=True)
        transcriber = AssemblyAITranscriber(api_key="test_key", config=config)
        processor = SmartAudioProcessor(
            assemblyai_transcriber=transcriber,
            respect_speakers=False,  # Disable speaker chunking
            respect_chapters=True
        )
        
        with patch('builtins.open', create=True):
            result = processor.run(sources=["test.wav"])
        
        documents = result["documents"]
        chapter_chunks = [doc for doc in documents if doc.meta.get("chunk_type") == "chapter"]
        
        assert len(chapter_chunks) > 0
        for chunk in chapter_chunks:
            assert "chapter_number" in chunk.meta
            assert chunk.meta["processing_strategy"] == "chapter_aware"
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_semantic_boundary_chunking(self, mock_assemblyai, sample_config):
        """Test semantic boundary-based chunking."""
        transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
        processor = SmartAudioProcessor(
            assemblyai_transcriber=transcriber,
            respect_speakers=False,
            respect_chapters=False,
            max_chunk_length=500
        )
        
        with patch('builtins.open', create=True):
            result = processor.run(sources=["test.wav"])
        
        documents = result["documents"]
        semantic_chunks = [doc for doc in documents if doc.meta.get("chunk_type") == "semantic_boundary"]
        
        assert len(semantic_chunks) > 0
        for chunk in semantic_chunks:
            assert chunk.meta["processing_strategy"] == "semantic_aware"
            assert len(chunk.content) <= 600  # Allowing some buffer
    
    def test_serialization(self, sample_config):
        """Test SmartAudioProcessor serialization."""
        with patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True):
            with patch('audio_processing.audio_transcriber.aai'):
                transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
                processor = SmartAudioProcessor(assemblyai_transcriber=transcriber)
                
                # Test to_dict
                data = processor.to_dict()
                assert "init_parameters" in data
                
                # Test from_dict  
                restored = SmartAudioProcessor.from_dict(data)
                assert isinstance(restored, SmartAudioProcessor)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_advanced_audio_config(self):
        """Test creating advanced audio configuration."""
        config = create_advanced_audio_config()
        
        assert isinstance(config, AudioProcessingConfig)
        assert config.speaker_labels is True
        assert config.sentiment_analysis is True
        assert config.entity_detection is True
        assert config.iab_categories is True
        assert config.content_safety is True
        assert config.auto_highlights is True
        assert config.summarization is True
        assert len(config.custom_vocabulary) > 0
        assert "transcription" in config.custom_vocabulary
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_create_audio_pipeline(self, mock_assemblyai):
        """Test creating a complete audio processing pipeline."""
        with patch('haystack.Pipeline') as mock_pipeline:
            pipeline = create_audio_pipeline(api_key="test_key")
            
            # Verify pipeline was created and components were added
            mock_pipeline.return_value.add_component.assert_called()
            mock_pipeline.return_value.connect.assert_called()


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_transcription_error_handling(self, mock_assemblyai, sample_config):
        """Test handling of transcription errors."""
        # Mock a failed transcription
        mock_assemblyai.Transcriber.return_value.transcribe.return_value.status = "error"
        mock_assemblyai.Transcriber.return_value.transcribe.return_value.error = "Audio file format not supported"
        mock_assemblyai.TranscriptStatus.error = "error"
        
        transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
        
        with patch('builtins.open', create=True):
            result = transcriber.run(sources=["invalid.wav"])
        
        # Should return empty documents list for failed transcriptions
        assert len(result["documents"]) == 0
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_invalid_source_type(self, mock_assemblyai, sample_config):
        """Test handling of invalid source types."""
        transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
        
        result = transcriber.run(sources=[123])  # Invalid type
        
        # Should handle gracefully and return empty documents
        assert len(result["documents"]) == 0
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_file_not_found_handling(self, mock_assemblyai, sample_config):
        """Test handling of non-existent files."""
        transcriber = AssemblyAITranscriber(api_key="test_key", config=sample_config)
        
        with patch('builtins.open', side_effect=FileNotFoundError):
            result = transcriber.run(sources=["nonexistent.wav"])
        
        # Should handle gracefully and return empty documents
        assert len(result["documents"]) == 0


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_full_workflow_local_file(self, mock_assemblyai, harvard_wav_path):
        """Test complete workflow with local audio file."""
        config = create_advanced_audio_config()
        transcriber = AssemblyAITranscriber(api_key="test_key", config=config)
        processor = SmartAudioProcessor(assemblyai_transcriber=transcriber)
        
        with patch('builtins.open', create=True):
            result = processor.run(sources=[harvard_wav_path])
        
        documents = result["documents"]
        assert len(documents) > 0
        
        # Should have different types of chunks
        chunk_types = {doc.meta.get("chunk_type") for doc in documents}
        assert len(chunk_types) >= 1  # At least one chunking strategy
    
    @patch('audio_processing.audio_transcriber.ASSEMBLYAI_AVAILABLE', True)
    def test_full_workflow_youtube_url(self, mock_assemblyai):
        """Test complete workflow with YouTube URL."""
        config = create_advanced_audio_config()
        transcriber = AssemblyAITranscriber(api_key="test_key", config=config)
        processor = SmartAudioProcessor(assemblyai_transcriber=transcriber)
        
        youtube_url = "https://www.youtube.com/shorts/-Nw0Ts2n2nU"
        result = processor.run(sources=[youtube_url])
        
        documents = result["documents"]
        assert len(documents) > 0
        
        # Check that YouTube URL was processed correctly
        main_doc = next((doc for doc in documents if doc.meta.get("audio_url")), None)
        assert main_doc is not None
        assert main_doc.meta["audio_url"] == youtube_url


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
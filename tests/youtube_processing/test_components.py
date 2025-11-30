"""
Unit tests for YouTube Processing Components.

Tests for YouTubeAudioProcessor, YouTubeDocumentBuilder, and YouTubeTranscriber.
"""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from haystack.dataclasses import Document

from src.youtube_processing.components.document_builder import YouTubeDocumentBuilder
from src.youtube_processing.components.processor import YouTubeAudioProcessor
from src.youtube_processing.config import CacheConfig
from src.youtube_processing.config import DownloadConfig
from src.youtube_processing.config import MetadataConfig
from src.youtube_processing.config import RetryConfig
from src.youtube_processing.config import ValidationConfig
from src.youtube_processing.config import YouTubeConfig
from src.youtube_processing.exceptions import ValidationError
from src.youtube_processing.interfaces import CacheManager
from src.youtube_processing.interfaces import DownloadResult
from src.youtube_processing.interfaces import MetadataEnhancer
from src.youtube_processing.interfaces import URLValidator
from src.youtube_processing.interfaces import VideoDownloader
from src.youtube_processing.interfaces import VideoMetadata


@pytest.fixture
def mock_video_metadata() -> VideoMetadata:
    """Create mock video metadata for testing."""
    return VideoMetadata(
        video_id="test123",
        title="Test Video Title",
        description="Test video description",
        channel_name="Test Channel",
        channel_id="UC123456",
        duration_seconds=300,
        upload_date="2023-01-15",
        view_count=10000,
        like_count=500,
        tags=["test", "video"],
        categories=["Education"],
        thumbnail_url="https://example.com/thumb.jpg",
        is_live=False,
        is_age_restricted=False,
        language="en",
    )


@pytest.fixture
def youtube_config(tmp_path: Path) -> YouTubeConfig:
    """Create a YouTube configuration for testing."""
    return YouTubeConfig(
        download=DownloadConfig(temp_dir=tmp_path / "downloads"),
        cache=CacheConfig(enabled=False),
        validation=ValidationConfig(),
        metadata=MetadataConfig(),
        retry=RetryConfig(),
    )


@pytest.fixture
def mock_download_result(tmp_path: Path, mock_video_metadata: VideoMetadata) -> DownloadResult:
    """Create a mock download result."""
    audio_path = tmp_path / "test123.mp3"
    audio_path.write_text("mock audio content")
    return DownloadResult(
        audio_path=audio_path,
        metadata=mock_video_metadata,
        file_size_bytes=1024,
        download_duration_seconds=5.0,
        from_cache=False,
    )


@pytest.fixture
def mock_downloader() -> Mock:
    """Create a mock video downloader."""
    return Mock(spec=VideoDownloader)


@pytest.fixture
def mock_audio_transcriber() -> Mock:
    """Create a mock audio transcriber."""
    return Mock()


@pytest.fixture
def mock_cache_manager() -> Mock:
    """Create a mock cache manager."""
    return Mock(spec=CacheManager)


@pytest.fixture
def mock_metadata_enhancer() -> Mock:
    """Create a mock metadata enhancer."""
    return Mock(spec=MetadataEnhancer)


@pytest.fixture
def mock_validator() -> Mock:
    """Create a mock URL validator."""
    return Mock(spec=URLValidator)


class TestYouTubeDocumentBuilder:
    """Tests for YouTubeDocumentBuilder."""

    def test_build_creates_main_document(
        self,
        mock_video_metadata: VideoMetadata,
    ) -> None:
        """Test that build creates a main document with transcript."""
        builder = YouTubeDocumentBuilder()
        transcript_text = "This is the transcript text."

        documents = builder.build(
            transcript_text=transcript_text,
            video_metadata=mock_video_metadata,
        )

        assert len(documents) >= 1
        main_doc = documents[0]
        assert main_doc.content == transcript_text
        assert main_doc.meta["video_id"] == "test123"
        assert main_doc.meta["title"] == "Test Video Title"
        assert main_doc.meta["source"] == "youtube"

    def test_build_includes_source_url(
        self,
        mock_video_metadata: VideoMetadata,
    ) -> None:
        """Test that documents include source URL."""
        builder = YouTubeDocumentBuilder()

        documents = builder.build(
            transcript_text="Transcript",
            video_metadata=mock_video_metadata,
        )

        assert "source_url" in documents[0].meta
        assert mock_video_metadata.video_id in documents[0].meta["source_url"]

    def test_build_includes_transcript_metadata(
        self,
        mock_video_metadata: VideoMetadata,
    ) -> None:
        """Test that transcript metadata is included."""
        builder = YouTubeDocumentBuilder()
        transcript_metadata = {
            "entities": [{"text": "Python", "type": "SKILL"}],
            "sentiment": {"score": 0.8},
            "summary": "A summary of the video.",
        }

        documents = builder.build(
            transcript_text="Transcript",
            video_metadata=mock_video_metadata,
            transcript_metadata=transcript_metadata,
        )

        assert documents[0].meta.get("entities") == transcript_metadata["entities"]
        assert documents[0].meta.get("sentiment") == transcript_metadata["sentiment"]
        assert documents[0].meta.get("summary") == transcript_metadata["summary"]

    def test_build_creates_utterance_documents_when_enabled(
        self,
        mock_video_metadata: VideoMetadata,
    ) -> None:
        """Test that utterance documents are created when enabled."""
        builder = YouTubeDocumentBuilder(include_utterances=True)

        # Create mock utterances
        mock_utterance = Mock()
        mock_utterance.text = "Hello world"
        mock_utterance.speaker = "Speaker A"
        mock_utterance.start = 1000
        mock_utterance.end = 5000

        transcript_metadata = {
            "utterances": [mock_utterance],
        }

        documents = builder.build(
            transcript_text="Full transcript",
            video_metadata=mock_video_metadata,
            transcript_metadata=transcript_metadata,
        )

        # Should have main document + 1 utterance
        assert len(documents) == 2
        utterance_doc = documents[1]
        assert utterance_doc.content == "Hello world"
        assert utterance_doc.meta["speaker"] == "Speaker A"
        assert utterance_doc.meta["document_type"] == "utterance"

    def test_build_excludes_utterances_when_disabled(
        self,
        mock_video_metadata: VideoMetadata,
    ) -> None:
        """Test that utterance documents are excluded when disabled."""
        builder = YouTubeDocumentBuilder(include_utterances=False)

        mock_utterance = Mock()
        mock_utterance.text = "Hello world"

        transcript_metadata = {
            "utterances": [mock_utterance],
        }

        documents = builder.build(
            transcript_text="Full transcript",
            video_metadata=mock_video_metadata,
            transcript_metadata=transcript_metadata,
        )

        # Should only have main document
        assert len(documents) == 1

    def test_build_from_chunks(
        self,
        mock_video_metadata: VideoMetadata,
    ) -> None:
        """Test building documents from pre-chunked text."""
        builder = YouTubeDocumentBuilder()
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]

        documents = builder.build_from_chunks(
            chunks=chunks,
            video_metadata=mock_video_metadata,
        )

        assert len(documents) == 3
        for idx, doc in enumerate(documents):
            assert doc.content == f"Chunk {idx + 1}"
            assert doc.meta["chunk_index"] == idx
            assert doc.meta["total_chunks"] == 3

    def test_build_includes_timestamps_when_enabled(
        self,
        mock_video_metadata: VideoMetadata,
    ) -> None:
        """Test that timestamps are included when enabled."""
        builder = YouTubeDocumentBuilder(include_timestamps=True)

        mock_utterance = Mock()
        mock_utterance.text = "Hello"
        mock_utterance.speaker = None
        mock_utterance.start = 1000
        mock_utterance.end = 2000

        transcript_metadata = {"utterances": [mock_utterance]}

        documents = builder.build(
            transcript_text="Full transcript",
            video_metadata=mock_video_metadata,
            transcript_metadata=transcript_metadata,
        )

        utterance_doc = documents[1]
        assert utterance_doc.meta.get("start_time_ms") == 1000
        assert utterance_doc.meta.get("end_time_ms") == 2000
        assert utterance_doc.meta.get("duration_ms") == 1000


class TestYouTubeAudioProcessor:
    """Tests for YouTubeAudioProcessor."""

    def test_init_stores_dependencies(
        self,
        mock_downloader: Mock,
        mock_audio_transcriber: Mock,
        mock_cache_manager: Mock,
        mock_metadata_enhancer: Mock,
        mock_validator: Mock,
        youtube_config: YouTubeConfig,
    ) -> None:
        """Test that dependencies are stored correctly."""
        processor = YouTubeAudioProcessor(
            downloader=mock_downloader,
            audio_transcriber=mock_audio_transcriber,
            cache_manager=mock_cache_manager,
            metadata_enhancer=mock_metadata_enhancer,
            validator=mock_validator,
            config=youtube_config,
        )

        assert processor._downloader == mock_downloader
        assert processor._audio_transcriber == mock_audio_transcriber
        assert processor._cache_manager == mock_cache_manager
        assert processor._metadata_enhancer == mock_metadata_enhancer
        assert processor._validator == mock_validator
        assert processor._config == youtube_config

    def test_process_validates_url_first(
        self,
        mock_downloader: Mock,
        mock_audio_transcriber: Mock,
        mock_metadata_enhancer: Mock,
        mock_validator: Mock,
        youtube_config: YouTubeConfig,
    ) -> None:
        """Test that process validates URL before downloading."""
        # Configure validator to reject URL
        mock_validator.extract_video_id.return_value = None

        processor = YouTubeAudioProcessor(
            downloader=mock_downloader,
            audio_transcriber=mock_audio_transcriber,
            cache_manager=None,
            metadata_enhancer=mock_metadata_enhancer,
            validator=mock_validator,
            config=youtube_config,
        )

        with pytest.raises(ValidationError) as exc_info:
            processor.process("https://invalid-url.com")

        assert "Invalid YouTube URL" in str(exc_info.value)
        mock_downloader.download.assert_not_called()

    def test_process_returns_documents(
        self,
        mock_downloader: Mock,
        mock_audio_transcriber: Mock,
        mock_metadata_enhancer: Mock,
        mock_validator: Mock,
        youtube_config: YouTubeConfig,
        mock_video_metadata: VideoMetadata,
        mock_download_result: DownloadResult,
    ) -> None:
        """Test that process returns document list."""
        # Configure mocks
        mock_validator.extract_video_id.return_value = "test123"
        mock_validator.validate.return_value = (True, [])
        mock_downloader.download.return_value = mock_download_result

        # Create a mock transcript that doesn't have optional attributes
        mock_transcript = Mock(spec=["text"])
        mock_transcript.text = "Transcribed text"
        mock_audio_transcriber.transcribe_with_features.return_value = mock_transcript

        mock_metadata_enhancer.enhance.return_value = mock_video_metadata
        mock_metadata_enhancer.merge_with_transcript_metadata.return_value = {
            "video_id": "test123",
            "title": "Test Video Title",
        }

        processor = YouTubeAudioProcessor(
            downloader=mock_downloader,
            audio_transcriber=mock_audio_transcriber,
            cache_manager=None,
            metadata_enhancer=mock_metadata_enhancer,
            validator=mock_validator,
            config=youtube_config,
        )

        documents = processor.process("https://youtube.com/watch?v=test123")

        assert isinstance(documents, list)
        assert len(documents) >= 1
        assert all(isinstance(doc, Document) for doc in documents)

    def test_process_uses_cache_when_available(
        self,
        mock_downloader: Mock,
        mock_audio_transcriber: Mock,
        mock_cache_manager: Mock,
        mock_metadata_enhancer: Mock,
        mock_validator: Mock,
        youtube_config: YouTubeConfig,
        mock_video_metadata: VideoMetadata,
        tmp_path: Path,
    ) -> None:
        """Test that process uses cache when available."""
        cached_audio = tmp_path / "cached.mp3"
        cached_audio.write_text("cached audio")

        # Configure mocks
        mock_validator.extract_video_id.return_value = "test123"
        mock_validator.validate.return_value = (True, [])
        mock_cache_manager.has.return_value = True
        mock_cache_manager.get.return_value = cached_audio
        mock_downloader.extract_metadata.return_value = mock_video_metadata

        # Create a mock transcript that doesn't have optional attributes
        mock_transcript = Mock(spec=["text"])
        mock_transcript.text = "Transcribed text"
        mock_audio_transcriber.transcribe_with_features.return_value = mock_transcript

        mock_metadata_enhancer.enhance.return_value = mock_video_metadata
        mock_metadata_enhancer.merge_with_transcript_metadata.return_value = {}

        processor = YouTubeAudioProcessor(
            downloader=mock_downloader,
            audio_transcriber=mock_audio_transcriber,
            cache_manager=mock_cache_manager,
            metadata_enhancer=mock_metadata_enhancer,
            validator=mock_validator,
            config=youtube_config,
        )

        processor.process("https://youtube.com/watch?v=test123")

        # Should check cache and not call download
        mock_cache_manager.has.assert_called_once_with("test123")
        mock_cache_manager.get.assert_called_once_with("test123")
        mock_downloader.download.assert_not_called()

    def test_get_metadata_returns_video_metadata(
        self,
        mock_downloader: Mock,
        mock_audio_transcriber: Mock,
        mock_metadata_enhancer: Mock,
        mock_validator: Mock,
        youtube_config: YouTubeConfig,
        mock_video_metadata: VideoMetadata,
    ) -> None:
        """Test that get_metadata returns VideoMetadata."""
        mock_downloader.extract_metadata.return_value = mock_video_metadata

        processor = YouTubeAudioProcessor(
            downloader=mock_downloader,
            audio_transcriber=mock_audio_transcriber,
            cache_manager=None,
            metadata_enhancer=mock_metadata_enhancer,
            validator=mock_validator,
            config=youtube_config,
        )

        result = processor.get_metadata("https://youtube.com/watch?v=test123")

        assert result == mock_video_metadata
        mock_downloader.extract_metadata.assert_called_once()

    def test_process_batch_processes_multiple_urls(
        self,
        mock_downloader: Mock,
        mock_audio_transcriber: Mock,
        mock_metadata_enhancer: Mock,
        mock_validator: Mock,
        youtube_config: YouTubeConfig,
        mock_video_metadata: VideoMetadata,
        mock_download_result: DownloadResult,
    ) -> None:
        """Test that process_batch handles multiple URLs."""
        # Configure mocks
        mock_validator.extract_video_id.side_effect = ["vid1", "vid2"]
        mock_validator.validate.return_value = (True, [])
        mock_downloader.download.return_value = mock_download_result

        # Create a mock transcript that doesn't have optional attributes
        mock_transcript = Mock(spec=["text"])
        mock_transcript.text = "Transcribed text"
        mock_audio_transcriber.transcribe_with_features.return_value = mock_transcript

        mock_metadata_enhancer.enhance.return_value = mock_video_metadata
        mock_metadata_enhancer.merge_with_transcript_metadata.return_value = {}

        processor = YouTubeAudioProcessor(
            downloader=mock_downloader,
            audio_transcriber=mock_audio_transcriber,
            cache_manager=None,
            metadata_enhancer=mock_metadata_enhancer,
            validator=mock_validator,
            config=youtube_config,
        )

        urls = [
            "https://youtube.com/watch?v=vid1",
            "https://youtube.com/watch?v=vid2",
        ]
        results = processor.process_batch(urls)

        assert len(results) == 2
        assert all(url in results for url in urls)

    def test_process_batch_continues_on_error(
        self,
        mock_downloader: Mock,
        mock_audio_transcriber: Mock,
        mock_metadata_enhancer: Mock,
        mock_validator: Mock,
        youtube_config: YouTubeConfig,
        mock_video_metadata: VideoMetadata,
        mock_download_result: DownloadResult,
    ) -> None:
        """Test that process_batch continues processing after an error."""
        # First URL fails, second succeeds
        mock_validator.extract_video_id.side_effect = [None, "vid2"]
        mock_validator.validate.return_value = (True, [])
        mock_downloader.download.return_value = mock_download_result

        # Create a mock transcript that doesn't have optional attributes
        mock_transcript = Mock(spec=["text"])
        mock_transcript.text = "Transcribed text"
        mock_audio_transcriber.transcribe_with_features.return_value = mock_transcript

        mock_metadata_enhancer.enhance.return_value = mock_video_metadata
        mock_metadata_enhancer.merge_with_transcript_metadata.return_value = {}

        processor = YouTubeAudioProcessor(
            downloader=mock_downloader,
            audio_transcriber=mock_audio_transcriber,
            cache_manager=None,
            metadata_enhancer=mock_metadata_enhancer,
            validator=mock_validator,
            config=youtube_config,
        )

        urls = [
            "https://invalid-url.com",
            "https://youtube.com/watch?v=vid2",
        ]
        results = processor.process_batch(urls)

        # First URL failed (empty list), second succeeded
        assert len(results[urls[0]]) == 0
        assert len(results[urls[1]]) >= 1

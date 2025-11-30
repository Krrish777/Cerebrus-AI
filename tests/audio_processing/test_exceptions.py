"""
Unit tests for audio processing exceptions.

Tests custom exception classes for proper initialization and message formatting
following AGENTS.md principles: deterministic, focused, comprehensive.
"""

import pytest

from src.audio_processing.exceptions import (
    AudioProcessingException,
    TranscriptionServiceUnavailableError,
    TranscriptionConfigurationError,
    TranscriptionAPIError,
    AudioSourceError,
    TranscriptionError,
    AudioDownloadError,
    YouTubeVideoError,
    ChunkingError,
    ConfigurationLoadError,
)


class TestAudioProcessingException:
    """Test base AudioProcessingException class."""

    def test_basic_initialization(self):
        """Test basic exception initialization."""
        exc = AudioProcessingException("Test error message")
        
        assert exc.message == "Test error message"
        assert exc.original_exception is None
        assert "Test error message" in str(exc)

    def test_initialization_with_original_exception(self):
        """Test exception initialization with original exception."""
        original = ValueError("Original error")
        exc = AudioProcessingException("Wrapper error", original)
        
        assert exc.message == "Wrapper error"
        assert exc.original_exception is original
        assert "Wrapper error" in str(exc)
        assert "Original error" in str(exc)
        assert "Caused by:" in str(exc)

    def test_exception_is_raiseable(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(AudioProcessingException) as exc_info:
            raise AudioProcessingException("Test error")
        
        assert "Test error" in str(exc_info.value)


class TestTranscriptionServiceUnavailableError:
    """Test TranscriptionServiceUnavailableError class."""

    def test_initialization(self):
        """Test exception initialization."""
        exc = TranscriptionServiceUnavailableError("AssemblyAI")
        
        assert exc.service_name == "AssemblyAI"
        assert "AssemblyAI" in str(exc)
        assert "not available" in str(exc)

    def test_initialization_with_original_exception(self):
        """Test exception initialization with original exception."""
        original = ImportError("Module not found")
        exc = TranscriptionServiceUnavailableError("AssemblyAI", original)
        
        assert exc.service_name == "AssemblyAI"
        assert exc.original_exception is original
        assert "Module not found" in str(exc)

    def test_inherits_from_base(self):
        """Test exception inherits from AudioProcessingException."""
        exc = TranscriptionServiceUnavailableError("TestService")
        assert isinstance(exc, AudioProcessingException)


class TestTranscriptionConfigurationError:
    """Test TranscriptionConfigurationError class."""

    def test_initialization(self):
        """Test exception initialization."""
        exc = TranscriptionConfigurationError("model", "must be valid")
        
        assert exc.config_field == "model"
        assert exc.reason == "must be valid"
        assert "model" in str(exc)
        assert "must be valid" in str(exc)

    def test_initialization_with_original_exception(self):
        """Test exception initialization with original exception."""
        original = ValueError("Invalid value")
        exc = TranscriptionConfigurationError("language_code", "invalid", original)
        
        assert exc.config_field == "language_code"
        assert exc.original_exception is original

    def test_inherits_from_base(self):
        """Test exception inherits from AudioProcessingException."""
        exc = TranscriptionConfigurationError("field", "reason")
        assert isinstance(exc, AudioProcessingException)


class TestTranscriptionAPIError:
    """Test TranscriptionAPIError class."""

    def test_initialization(self):
        """Test exception initialization."""
        exc = TranscriptionAPIError("AssemblyAI", "API key missing")
        
        assert exc.api_name == "AssemblyAI"
        assert exc.reason == "API key missing"
        assert "AssemblyAI" in str(exc)
        assert "API key missing" in str(exc)

    def test_inherits_from_base(self):
        """Test exception inherits from AudioProcessingException."""
        exc = TranscriptionAPIError("TestAPI", "reason")
        assert isinstance(exc, AudioProcessingException)


class TestAudioSourceError:
    """Test AudioSourceError class."""

    def test_initialization(self):
        """Test exception initialization."""
        exc = AudioSourceError("/path/to/file.mp3", "File not found")
        
        assert exc.source == "/path/to/file.mp3"
        assert exc.reason == "File not found"
        assert "/path/to/file.mp3" in str(exc)
        assert "File not found" in str(exc)

    def test_inherits_from_base(self):
        """Test exception inherits from AudioProcessingException."""
        exc = AudioSourceError("source", "reason")
        assert isinstance(exc, AudioProcessingException)


class TestTranscriptionError:
    """Test TranscriptionError class."""

    def test_initialization_without_transcript_id(self):
        """Test exception initialization without transcript ID."""
        exc = TranscriptionError("audio.mp3", "Timeout")
        
        assert exc.source == "audio.mp3"
        assert exc.reason == "Timeout"
        assert exc.transcript_id is None
        assert "audio.mp3" in str(exc)
        assert "Timeout" in str(exc)

    def test_initialization_with_transcript_id(self):
        """Test exception initialization with transcript ID."""
        exc = TranscriptionError("audio.mp3", "Failed", transcript_id="tr_123")
        
        assert exc.source == "audio.mp3"
        assert exc.transcript_id == "tr_123"
        assert "tr_123" in str(exc)

    def test_inherits_from_base(self):
        """Test exception inherits from AudioProcessingException."""
        exc = TranscriptionError("source", "reason")
        assert isinstance(exc, AudioProcessingException)


class TestAudioDownloadError:
    """Test AudioDownloadError class."""

    def test_initialization(self):
        """Test exception initialization."""
        exc = AudioDownloadError("https://example.com/audio.mp3", "Connection refused")
        
        assert exc.url == "https://example.com/audio.mp3"
        assert exc.reason == "Connection refused"
        assert "https://example.com/audio.mp3" in str(exc)
        assert "Connection refused" in str(exc)

    def test_inherits_from_base(self):
        """Test exception inherits from AudioProcessingException."""
        exc = AudioDownloadError("url", "reason")
        assert isinstance(exc, AudioProcessingException)


class TestYouTubeVideoError:
    """Test YouTubeVideoError class."""

    def test_initialization_with_video_id(self):
        """Test exception initialization with video ID."""
        exc = YouTubeVideoError("dQw4w9WgXcQ", "Video unavailable")
        
        assert exc.video_id == "dQw4w9WgXcQ"
        assert exc.reason == "Video unavailable"
        assert "dQw4w9WgXcQ" in str(exc)
        assert "Video unavailable" in str(exc)

    def test_initialization_without_video_id(self):
        """Test exception initialization without video ID."""
        exc = YouTubeVideoError(None, "Invalid URL")
        
        assert exc.video_id is None
        assert exc.reason == "Invalid URL"
        assert "Invalid URL" in str(exc)

    def test_inherits_from_base(self):
        """Test exception inherits from AudioProcessingException."""
        exc = YouTubeVideoError("id", "reason")
        assert isinstance(exc, AudioProcessingException)


class TestChunkingError:
    """Test ChunkingError class."""

    def test_initialization(self):
        """Test exception initialization."""
        exc = ChunkingError("transcript_123", "Content too short")
        
        assert exc.source == "transcript_123"
        assert exc.reason == "Content too short"
        assert "transcript_123" in str(exc)
        assert "Content too short" in str(exc)

    def test_inherits_from_base(self):
        """Test exception inherits from AudioProcessingException."""
        exc = ChunkingError("source", "reason")
        assert isinstance(exc, AudioProcessingException)


class TestConfigurationLoadError:
    """Test ConfigurationLoadError class."""

    def test_initialization(self):
        """Test exception initialization."""
        exc = ConfigurationLoadError("/path/config.yaml", "File not found")
        
        assert exc.config_path == "/path/config.yaml"
        assert exc.reason == "File not found"
        assert "/path/config.yaml" in str(exc)
        assert "File not found" in str(exc)

    def test_inherits_from_base(self):
        """Test exception inherits from AudioProcessingException."""
        exc = ConfigurationLoadError("path", "reason")
        assert isinstance(exc, AudioProcessingException)


class TestExceptionChaining:
    """Test exception chaining scenarios."""

    def test_nested_exception_chain(self):
        """Test nested exception chaining."""
        root = IOError("Disk full")
        middle = AudioSourceError("/path/file.mp3", "Cannot read", root)
        outer = TranscriptionError("/path/file.mp3", "Processing failed", None, middle)
        
        assert outer.original_exception is middle
        assert middle.original_exception is root

    def test_exception_chain_message_formatting(self):
        """Test that exception chain messages are properly formatted."""
        original = ValueError("Value out of range")
        exc = AudioProcessingException("Processing failed", original)
        
        message = str(exc)
        assert "Processing failed" in message
        assert "Caused by:" in message
        assert "Value out of range" in message


class TestExceptionCatchability:
    """Test exception catchability patterns."""

    def test_catch_by_base_class(self):
        """Test catching specific exceptions by base class."""
        exceptions = [
            TranscriptionServiceUnavailableError("Test"),
            TranscriptionConfigurationError("field", "reason"),
            TranscriptionAPIError("api", "reason"),
            AudioSourceError("source", "reason"),
            TranscriptionError("source", "reason"),
            AudioDownloadError("url", "reason"),
            YouTubeVideoError("id", "reason"),
            ChunkingError("source", "reason"),
            ConfigurationLoadError("path", "reason"),
        ]
        
        for exc in exceptions:
            try:
                raise exc
            except AudioProcessingException as caught:
                assert caught is exc
            except Exception:
                pytest.fail(f"Exception {type(exc).__name__} not caught by base class")

    def test_catch_by_specific_class(self):
        """Test catching by specific exception class."""
        with pytest.raises(TranscriptionConfigurationError):
            raise TranscriptionConfigurationError("field", "reason")
        
        with pytest.raises(YouTubeVideoError):
            raise YouTubeVideoError("id", "reason")

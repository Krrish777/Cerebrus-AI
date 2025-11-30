"""
Unit tests for transcription providers.

Tests the base provider and AssemblyAI provider implementations.
Following AGENTS.md testing standards.
"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.audio_processing.config import (
    FeatureConfig,
    ProviderConfig,
    TranscriptionConfig,
)
from src.audio_processing.exceptions import (
    ConfigurationError,
    TranscriptionError,
    ValidationError,
)
from src.audio_processing.transcription.providers.base import BaseTranscriptionProvider


class ConcreteTestProvider(BaseTranscriptionProvider):
    """Concrete implementation for testing base provider."""

    def __init__(
        self,
        provider_config: ProviderConfig,
        transcription_config: TranscriptionConfig = None,
    ) -> None:
        super().__init__(provider_config, transcription_config)
        self._transcribe_result: Dict[str, Any] = {}
        self._upload_result: str = ""
        self._status_result: Dict[str, Any] = {}

    def set_transcribe_result(self, result: Dict[str, Any]) -> None:
        """Set the mock transcribe result."""
        self._transcribe_result = result

    def set_upload_result(self, result: str) -> None:
        """Set the mock upload result."""
        self._upload_result = result

    def set_status_result(self, result: Dict[str, Any]) -> None:
        """Set the mock status result."""
        self._status_result = result

    def _do_transcribe(self, audio_path: Path) -> Dict[str, Any]:
        return self._transcribe_result

    def _do_upload(self, audio_path: Path) -> str:
        return self._upload_result

    def _do_get_status(self, transcript_id: str) -> Dict[str, Any]:
        return self._status_result

    def _configure_provider(self, config: TranscriptionConfig) -> None:
        pass


@pytest.fixture
def provider_config() -> ProviderConfig:
    """Create a test provider configuration."""
    return ProviderConfig(
        name="test_provider",
        api_key_env="TEST_API_KEY",
        timeout=30,
        max_retries=3,
    )


@pytest.fixture
def transcription_config() -> TranscriptionConfig:
    """Create a test transcription configuration."""
    return TranscriptionConfig(
        language_code="en",
        model="best",
        punctuate=True,
        format_text=True,
        speaker_labels=True,
    )


@pytest.fixture
def mock_audio_file(tmp_path: Path) -> Path:
    """Create a mock audio file for testing."""
    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_bytes(b"fake audio content")
    return audio_file


class TestBaseTranscriptionProvider:
    """Test suite for BaseTranscriptionProvider."""

    def test_init_with_valid_config(
        self,
        provider_config: ProviderConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test provider initialization with valid configuration."""
        monkeypatch.setenv("TEST_API_KEY", "test-key-123")

        provider = ConcreteTestProvider(provider_config)

        assert provider.provider_name == "test_provider"
        assert not provider.is_configured

    def test_init_raises_on_missing_api_key(
        self,
        provider_config: ProviderConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that initialization fails without API key."""
        monkeypatch.delenv("TEST_API_KEY", raising=False)

        with pytest.raises(ConfigurationError) as exc_info:
            ConcreteTestProvider(provider_config)

        assert "API key not found" in str(exc_info.value)

    def test_init_raises_on_missing_provider_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that initialization fails without provider name."""
        monkeypatch.setenv("TEST_API_KEY", "test-key-123")
        config = ProviderConfig(name="", api_key_env="TEST_API_KEY")

        with pytest.raises(ConfigurationError) as exc_info:
            ConcreteTestProvider(config)

        assert "Provider name is required" in str(exc_info.value)

    def test_configure_sets_is_configured(
        self,
        provider_config: ProviderConfig,
        transcription_config: TranscriptionConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that configure sets is_configured flag."""
        monkeypatch.setenv("TEST_API_KEY", "test-key-123")
        provider = ConcreteTestProvider(provider_config)

        assert not provider.is_configured

        provider.configure(transcription_config)

        assert provider.is_configured

    def test_transcribe_validates_audio_path(
        self,
        provider_config: ProviderConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that transcribe validates audio path exists."""
        monkeypatch.setenv("TEST_API_KEY", "test-key-123")
        provider = ConcreteTestProvider(provider_config)

        with pytest.raises(ValidationError) as exc_info:
            provider.transcribe(Path("/nonexistent/audio.mp3"))

        assert "Audio file not found" in str(exc_info.value)

    def test_transcribe_validates_is_file(
        self,
        provider_config: ProviderConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that transcribe validates path is a file."""
        monkeypatch.setenv("TEST_API_KEY", "test-key-123")
        provider = ConcreteTestProvider(provider_config)

        with pytest.raises(ValidationError) as exc_info:
            provider.transcribe(tmp_path)

        assert "Path is not a file" in str(exc_info.value)

    def test_transcribe_returns_result(
        self,
        provider_config: ProviderConfig,
        mock_audio_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test successful transcription returns result."""
        monkeypatch.setenv("TEST_API_KEY", "test-key-123")
        provider = ConcreteTestProvider(provider_config)

        expected_result = {
            "id": "test-123",
            "text": "Hello world",
            "confidence": 0.95,
        }
        provider.set_transcribe_result(expected_result)

        result = provider.transcribe(mock_audio_file)

        assert result == expected_result

    def test_transcribe_wraps_exceptions(
        self,
        provider_config: ProviderConfig,
        mock_audio_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that transcribe wraps generic exceptions in TranscriptionError."""
        monkeypatch.setenv("TEST_API_KEY", "test-key-123")

        class FailingProvider(ConcreteTestProvider):
            def _do_transcribe(self, audio_path: Path) -> Dict[str, Any]:
                raise RuntimeError("API failure")

        provider = FailingProvider(provider_config)

        with pytest.raises(TranscriptionError) as exc_info:
            provider.transcribe(mock_audio_file)

        assert "API failure" in str(exc_info.value)

    def test_upload_returns_url(
        self,
        provider_config: ProviderConfig,
        mock_audio_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test successful upload returns URL."""
        monkeypatch.setenv("TEST_API_KEY", "test-key-123")
        provider = ConcreteTestProvider(provider_config)

        expected_url = "https://storage.example.com/audio123"
        provider.set_upload_result(expected_url)

        result = provider.upload(mock_audio_file)

        assert result == expected_url

    def test_get_status_returns_info(
        self,
        provider_config: ProviderConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test get_status returns status information."""
        monkeypatch.setenv("TEST_API_KEY", "test-key-123")
        provider = ConcreteTestProvider(provider_config)

        expected_status = {
            "id": "test-123",
            "status": "completed",
        }
        provider.set_status_result(expected_status)

        result = provider.get_status("test-123")

        assert result == expected_status


class TestAssemblyAIProvider:
    """Test suite for AssemblyAIProvider."""

    @pytest.fixture
    def assemblyai_config(self) -> ProviderConfig:
        """Create AssemblyAI provider configuration."""
        return ProviderConfig(
            name="assemblyai",
            api_key_env="ASSEMBLYAI_API_KEY",
            timeout=300,
        )

    def test_init_with_valid_api_key(
        self,
        assemblyai_config: ProviderConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test AssemblyAI provider initialization."""
        monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key-123")

        with patch("src.audio_processing.transcription.providers.assemblyai.aai"):
            from src.audio_processing.transcription.providers.assemblyai import (
                AssemblyAIProvider,
            )

            provider = AssemblyAIProvider(assemblyai_config)

            assert provider.provider_name == "assemblyai"
            assert not provider.is_configured

    def test_init_raises_without_api_key(
        self,
        assemblyai_config: ProviderConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that initialization fails without API key."""
        monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)

        from src.audio_processing.transcription.providers.assemblyai import (
            AssemblyAIProvider,
        )

        with pytest.raises(ConfigurationError) as exc_info:
            AssemblyAIProvider(assemblyai_config)

        assert "API key not found" in str(exc_info.value)

    def test_configure_creates_transcriber(
        self,
        assemblyai_config: ProviderConfig,
        transcription_config: TranscriptionConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that configure creates the transcriber."""
        monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key-123")

        with patch("src.audio_processing.transcription.providers.assemblyai.aai") as mock_aai:
            from src.audio_processing.transcription.providers.assemblyai import (
                AssemblyAIProvider,
            )

            provider = AssemblyAIProvider(assemblyai_config)
            provider.configure(transcription_config)

            assert provider.is_configured
            mock_aai.TranscriptionConfig.assert_called()

    def test_configure_features_enables_features(
        self,
        assemblyai_config: ProviderConfig,
        transcription_config: TranscriptionConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that configure_features enables specified features."""
        monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key-123")

        features = FeatureConfig(
            sentiment_analysis=True,
            entity_detection=True,
            auto_chapters=True,
        )

        with patch("src.audio_processing.transcription.providers.assemblyai.aai") as mock_aai:
            from src.audio_processing.transcription.providers.assemblyai import (
                AssemblyAIProvider,
            )

            provider = AssemblyAIProvider(assemblyai_config)
            provider.configure(transcription_config)
            provider.configure_features(features)

            # Verify TranscriptionConfig was called with features
            calls = mock_aai.TranscriptionConfig.call_args_list
            assert len(calls) >= 2  # Initial config + features config

    def test_transcribe_returns_formatted_result(
        self,
        assemblyai_config: ProviderConfig,
        mock_audio_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that transcribe returns properly formatted result."""
        monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key-123")

        # Create mock transcript
        mock_transcript = MagicMock()
        mock_transcript.id = "transcript-123"
        mock_transcript.status = "completed"
        mock_transcript.text = "Hello world"
        mock_transcript.confidence = 0.95
        mock_transcript.audio_duration = 60.0
        mock_transcript.words = []
        mock_transcript.utterances = None
        mock_transcript.chapters = None
        mock_transcript.sentiment_analysis = None
        mock_transcript.entities = None
        mock_transcript.iab_categories = None
        mock_transcript.content_safety = None
        mock_transcript.auto_highlights = None
        mock_transcript.summary = None

        with patch("src.audio_processing.transcription.providers.assemblyai.aai") as mock_aai:
            mock_aai.TranscriptStatus.error = "error"
            mock_transcriber = MagicMock()
            mock_transcriber.transcribe.return_value = mock_transcript
            mock_aai.Transcriber.return_value = mock_transcriber

            from src.audio_processing.transcription.providers.assemblyai import (
                AssemblyAIProvider,
            )

            provider = AssemblyAIProvider(assemblyai_config)

            result = provider.transcribe(mock_audio_file)

            assert result["id"] == "transcript-123"
            assert result["text"] == "Hello world"
            assert result["confidence"] == 0.95


class TestTranscriptionFactory:
    """Test suite for TranscriptionFactory."""

    def test_available_providers_returns_list(self) -> None:
        """Test that available_providers returns registered providers."""
        from src.audio_processing.transcription.factory import TranscriptionFactory

        providers = TranscriptionFactory.available_providers()

        assert "assemblyai" in providers

    def test_create_raises_for_unknown_provider(self) -> None:
        """Test that create raises for unknown provider."""
        from src.audio_processing.transcription.factory import TranscriptionFactory

        factory = TranscriptionFactory()
        config = ProviderConfig(name="unknown", api_key_env="API_KEY")

        with pytest.raises(ConfigurationError) as exc_info:
            factory.create("unknown", config)

        assert "Unknown provider" in str(exc_info.value)

    def test_register_provider_adds_provider(self) -> None:
        """Test that register_provider adds new provider."""
        from src.audio_processing.transcription.factory import TranscriptionFactory
        from src.audio_processing.transcription.providers.base import (
            BaseTranscriptionProvider,
        )

        class CustomProvider(BaseTranscriptionProvider):
            def _do_transcribe(self, audio_path: Path) -> Dict[str, Any]:
                return {}

            def _do_upload(self, audio_path: Path) -> str:
                return ""

            def _do_get_status(self, transcript_id: str) -> Dict[str, Any]:
                return {}

            def _configure_provider(self, config: TranscriptionConfig) -> None:
                pass

        TranscriptionFactory.register_provider("custom", CustomProvider)

        assert "custom" in TranscriptionFactory.available_providers()

        # Clean up
        del TranscriptionFactory._providers["custom"]

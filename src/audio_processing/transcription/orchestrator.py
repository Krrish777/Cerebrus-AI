"""
Audio transcription orchestrator.

Coordinates transcription operations with a clean, high-level API.
Implements the TranscriptionOrchestrator interface.

Follows AGENTS.md design principles:
- Single Responsibility: Orchestrates transcription workflow
- Loose Coupling: Uses injected provider, doesn't create dependencies
- Encapsulation: Internal workflow is hidden from clients
"""

from pathlib import Path
from typing import Any, Dict, Optional

from src.audio_processing.config import (
    AudioProcessingConfig,
    FeatureConfig,
    TranscriptionConfig,
)
from src.audio_processing.exceptions import (
    ConfigurationError,
    TranscriptionError,
)
from src.audio_processing.interfaces import (
    TranscriptionOrchestrator,
    TranscriptionProvider,
)
from src.audio_processing.transcription.factory import TranscriptionFactory
from src.core.logging import get_logger

logger = get_logger(__name__)


class AudioTranscriber(TranscriptionOrchestrator):
    """
    High-level audio transcription orchestrator.

    Provides a simplified API for transcribing audio files with
    optional advanced features like speaker diarization, sentiment
    analysis, and automatic summarization.

    Example usage:
        # Using factory
        config = AudioProcessingConfig.from_yaml("config/audio_config.yml")
        transcriber = AudioTranscriber.from_config(config)

        # Simple transcription
        result = transcriber.transcribe(Path("audio.mp3"))

        # With features
        result = transcriber.transcribe_with_features(
            Path("audio.mp3"),
            features=FeatureConfig(
                sentiment_analysis=True,
                auto_chapters=True,
            ),
        )

        # Using dependency injection
        provider = MyCustomProvider(config)
        transcriber = AudioTranscriber(provider)
    """

    def __init__(
        self,
        provider: TranscriptionProvider,
        config: Optional[TranscriptionConfig] = None,
    ) -> None:
        """
        Initialize the transcriber with a provider.

        :param provider: Transcription provider to use
        :param config: Optional transcription configuration
        """
        self._provider = provider
        self._config = config

        if config and not provider.is_configured:
            provider.configure(config)

        logger.info(
            "AudioTranscriber initialized with provider: %s",
            provider.provider_name,
        )

    @classmethod
    def from_config(
        cls,
        audio_config: AudioProcessingConfig,
        provider_name: Optional[str] = None,
    ) -> "AudioTranscriber":
        """
        Create a transcriber from configuration.

        :param audio_config: Audio processing configuration
        :param provider_name: Optional provider name override
        :return: Configured AudioTranscriber instance
        """
        factory = TranscriptionFactory()
        provider = factory.create_from_config(audio_config, provider_name)

        return cls(
            provider=provider,
            config=audio_config.transcription,
        )

    @property
    def provider_name(self) -> str:
        """Return the current provider name."""
        return self._provider.provider_name

    def transcribe(
        self,
        audio_path: Path,
        config: Optional[TranscriptionConfig] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        :param audio_path: Path to the audio file
        :param config: Optional config override for this transcription
        :return: Transcription result dictionary
        :raises TranscriptionError: If transcription fails
        """
        # Apply config override if provided
        if config:
            self._provider.configure(config)

        logger.info("Starting transcription: %s", audio_path.name)

        try:
            result = self._provider.transcribe(audio_path)
            logger.info(
                "Transcription completed: %s (words=%d)",
                audio_path.name,
                len(result.get("words", [])),
            )
            return result
        except TranscriptionError:
            raise
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

    def transcribe_with_features(
        self,
        audio_path: Path,
        features: FeatureConfig,
    ) -> Dict[str, Any]:
        """
        Transcribe with advanced features enabled.

        :param audio_path: Path to the audio file
        :param features: Features to enable for transcription
        :return: Transcription result with feature data
        :raises TranscriptionError: If transcription fails
        """
        logger.info(
            "Transcribing with features: sentiment=%s, chapters=%s, entities=%s",
            features.sentiment_analysis,
            features.auto_chapters,
            features.entity_detection,
        )

        # Configure features on the provider if supported
        configure_features = getattr(self._provider, "configure_features", None)
        if configure_features is not None:
            configure_features(features)
        else:
            logger.warning(
                "Provider %s does not support feature configuration",
                self._provider.provider_name,
            )

        return self._provider.transcribe(audio_path)

    def get_transcript_status(self, transcript_id: str) -> Dict[str, Any]:
        """
        Get the status of a transcription job.

        :param transcript_id: Transcript identifier
        :return: Status information
        """
        return self._provider.get_status(transcript_id)

    def upload_audio(self, audio_path: Path) -> str:
        """
        Upload an audio file for later transcription.

        :param audio_path: Path to the audio file
        :return: Upload URL or identifier
        """
        return self._provider.upload(audio_path)

    def transcribe_async(self, audio_path: Path) -> str:
        """
        Start an asynchronous transcription.

        :param audio_path: Path to the audio file
        :return: Transcript ID for polling
        :raises TranscriptionError: If async not supported
        """
        async_method = getattr(self._provider, "transcribe_async", None)
        if async_method is not None:
            return async_method(audio_path)

        raise TranscriptionError(
            f"Provider {self._provider.provider_name} does not support async transcription"
        )

    def wait_for_transcript(self, transcript_id: str) -> Dict[str, Any]:
        """
        Wait for an async transcription to complete.

        :param transcript_id: Transcript ID to wait for
        :return: Completed transcription result
        :raises TranscriptionError: If wait not supported
        """
        wait_method = getattr(self._provider, "wait_for_completion", None)
        if wait_method is not None:
            return wait_method(transcript_id)

        raise TranscriptionError(
            f"Provider {self._provider.provider_name} does not support async wait"
        )

    def reconfigure(self, config: TranscriptionConfig) -> None:
        """
        Reconfigure the transcriber with new settings.

        :param config: New transcription configuration
        """
        self._config = config
        self._provider.configure(config)
        logger.info("Transcriber reconfigured")

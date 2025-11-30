"""
Base transcription provider implementation.

Provides shared functionality for all transcription providers.
Follows AGENTS.md design principles:
- Single Responsibility: Base class handles common validation and logging
- Encapsulation: Internal state is protected
- Extensibility: Concrete providers extend without modification
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from src.audio_processing.config import ProviderConfig, TranscriptionConfig
from src.audio_processing.exceptions import (
    ConfigurationError,
    TranscriptionError,
    ValidationError,
)
from src.audio_processing.interfaces import TranscriptionProvider
from src.core.logging import get_logger

logger = get_logger(__name__)


class BaseTranscriptionProvider(TranscriptionProvider, ABC):
    """
    Abstract base class for transcription providers.

    Provides common functionality including:
    - Configuration validation
    - File path validation
    - Standardized error handling
    - Logging setup

    Subclasses must implement:
    - _do_transcribe: Actual transcription logic
    - _do_upload: Upload logic for the provider
    - _do_get_status: Status check logic
    - _configure_provider: Provider-specific configuration
    """

    def __init__(
        self,
        provider_config: ProviderConfig,
        transcription_config: Optional[TranscriptionConfig] = None,
    ) -> None:
        """
        Initialize the base provider.

        :param provider_config: Provider-specific configuration
        :param transcription_config: Transcription settings
        :raises ConfigurationError: If configuration is invalid
        """
        self._provider_config = provider_config
        self._transcription_config = transcription_config
        self._is_configured = False

        self._validate_provider_config()
        logger.debug(
            "Initialized %s provider with config: %s",
            self.__class__.__name__,
            provider_config.name,
        )

    def _validate_provider_config(self) -> None:
        """
        Validate the provider configuration.

        :raises ConfigurationError: If required configuration is missing
        """
        if not self._provider_config.name:
            raise ConfigurationError("Provider name is required")

        api_key = self._provider_config.get_api_key()
        if not api_key:
            raise ConfigurationError(
                f"API key not found for provider {self._provider_config.name}. "
                f"Set environment variable: {self._provider_config.api_key_env}"
            )

    def _validate_audio_path(self, audio_path: Path) -> None:
        """
        Validate that the audio file exists and is accessible.

        :param audio_path: Path to the audio file
        :raises ValidationError: If file does not exist or is not readable
        """
        if not audio_path.exists():
            raise ValidationError(f"Audio file not found: {audio_path}")

        if not audio_path.is_file():
            raise ValidationError(f"Path is not a file: {audio_path}")

        # Check for supported audio formats
        supported_formats = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm", ".mp4"}
        if audio_path.suffix.lower() not in supported_formats:
            logger.warning(
                "Audio format %s may not be supported. Supported formats: %s",
                audio_path.suffix,
                supported_formats,
            )

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self._provider_config.name

    @property
    def is_configured(self) -> bool:
        """Return whether the provider is configured and ready."""
        return self._is_configured

    def configure(self, config: TranscriptionConfig) -> None:
        """
        Configure the provider with transcription settings.

        :param config: Transcription configuration
        :raises ConfigurationError: If configuration fails
        """
        logger.info("Configuring %s provider", self.provider_name)
        self._transcription_config = config

        try:
            self._configure_provider(config)
            self._is_configured = True
            logger.info("Provider %s configured successfully", self.provider_name)
        except Exception as exc:
            logger.error("Failed to configure provider %s: %s", self.provider_name, exc)
            raise ConfigurationError(
                f"Failed to configure {self.provider_name}: {exc}"
            ) from exc

    def transcribe(self, source: Path) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        :param source: Path to the audio file
        :return: Transcription result dictionary
        :raises ValidationError: If audio path is invalid
        :raises TranscriptionError: If transcription fails
        """
        self._validate_audio_path(source)

        logger.info("Starting transcription for: %s", source.name)

        try:
            result = self._do_transcribe(source)
            logger.info(
                "Transcription completed for: %s (confidence: %.2f)",
                source.name,
                result.get("confidence", 0.0),
            )
            return result
        except TranscriptionError:
            raise
        except Exception as exc:
            logger.error("Transcription failed for %s: %s", source.name, exc)
            raise TranscriptionError(
                f"Transcription failed for {source.name}: {exc}"
            ) from exc

    def upload(self, source: Path) -> str:
        """
        Upload an audio file to the provider's storage.

        :param source: Path to the audio file
        :return: URL or identifier for the uploaded file
        :raises ValidationError: If audio path is invalid
        :raises TranscriptionError: If upload fails
        """
        self._validate_audio_path(source)

        logger.info("Uploading audio file: %s", source.name)

        try:
            upload_url = self._do_upload(source)
            logger.info("Upload completed: %s", source.name)
            return upload_url
        except Exception as exc:
            logger.error("Upload failed for %s: %s", source.name, exc)
            raise TranscriptionError(
                f"Upload failed for {source.name}: {exc}"
            ) from exc

    def get_status(self, transcript_id: str) -> Dict[str, Any]:
        """
        Get the status of a transcription job.

        :param transcript_id: Identifier for the transcription
        :return: Status information dictionary
        :raises TranscriptionError: If status check fails
        """
        logger.debug("Checking status for transcript: %s", transcript_id)

        try:
            status = self._do_get_status(transcript_id)
            logger.debug("Status for %s: %s", transcript_id, status.get("status"))
            return status
        except Exception as exc:
            logger.error("Status check failed for %s: %s", transcript_id, exc)
            raise TranscriptionError(
                f"Status check failed for {transcript_id}: {exc}"
            ) from exc

    @abstractmethod
    def _do_transcribe(self, audio_path: Path) -> Dict[str, Any]:
        """
        Perform the actual transcription.

        :param audio_path: Validated path to audio file
        :return: Transcription result
        """

    @abstractmethod
    def _do_upload(self, audio_path: Path) -> str:
        """
        Perform the actual upload.

        :param audio_path: Validated path to audio file
        :return: Upload URL or identifier
        """

    @abstractmethod
    def _do_get_status(self, transcript_id: str) -> Dict[str, Any]:
        """
        Perform the actual status check.

        :param transcript_id: Transcription identifier
        :return: Status information
        """

    @abstractmethod
    def _configure_provider(self, config: TranscriptionConfig) -> None:
        """
        Apply provider-specific configuration.

        :param config: Transcription configuration
        """

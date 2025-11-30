"""
Transcription provider factory.

Creates transcription provider instances based on configuration.
Follows the Factory Pattern for loose coupling and extensibility.

Follows AGENTS.md design principles:
- Loose Coupling: Clients don't need to know concrete provider classes
- Extensibility: New providers can be added without modifying existing code
- Single Responsibility: Only responsible for provider instantiation
"""

from typing import Dict, Optional, Type

from src.audio_processing.config import (
    AudioProcessingConfig,
    ProviderConfig,
    TranscriptionConfig,
)
from src.audio_processing.exceptions import ConfigurationError
from src.audio_processing.interfaces import TranscriptionProvider
from src.audio_processing.transcription.providers.assemblyai import AssemblyAIProvider
from src.audio_processing.transcription.providers.base import BaseTranscriptionProvider
from src.core.logging import get_logger

logger = get_logger(__name__)


class TranscriptionFactory:
    """
    Factory for creating transcription provider instances.

    Supports registration of custom providers for extensibility.

    Example usage:
        factory = TranscriptionFactory()

        # Create from configuration
        provider = factory.create_from_config(audio_config)

        # Or create directly
        provider = factory.create("assemblyai", provider_config)

        # Register custom provider
        factory.register_provider("custom", CustomProvider)
    """

    # Registry of available providers
    _providers: Dict[str, Type[BaseTranscriptionProvider]] = {
        "assemblyai": AssemblyAIProvider,
    }

    def __init__(self) -> None:
        """Initialize the factory."""
        logger.debug("TranscriptionFactory initialized with providers: %s",
                     list(self._providers.keys()))

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: Type[BaseTranscriptionProvider],
    ) -> None:
        """
        Register a new provider type.

        :param name: Provider name identifier
        :param provider_class: Provider class to register
        :raises ConfigurationError: If provider doesn't implement interface
        """
        if not issubclass(provider_class, BaseTranscriptionProvider):
            raise ConfigurationError(
                f"Provider {name} must inherit from BaseTranscriptionProvider"
            )

        cls._providers[name.lower()] = provider_class
        logger.info("Registered transcription provider: %s", name)

    @classmethod
    def available_providers(cls) -> list:
        """
        Get list of available provider names.

        :return: List of registered provider names
        """
        return list(cls._providers.keys())

    def create(
        self,
        provider_name: str,
        provider_config: ProviderConfig,
        transcription_config: Optional[TranscriptionConfig] = None,
    ) -> TranscriptionProvider:
        """
        Create a provider instance by name.

        :param provider_name: Name of the provider to create
        :param provider_config: Provider-specific configuration
        :param transcription_config: Optional transcription settings
        :return: Configured provider instance
        :raises ConfigurationError: If provider is not registered
        """
        name_lower = provider_name.lower()

        if name_lower not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ConfigurationError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {available}"
            )

        provider_class = self._providers[name_lower]

        logger.info("Creating %s provider", provider_name)

        provider = provider_class(provider_config, transcription_config)

        # Configure if transcription config provided
        if transcription_config:
            provider.configure(transcription_config)

        return provider

    def create_from_config(
        self,
        audio_config: AudioProcessingConfig,
        provider_name: Optional[str] = None,
    ) -> TranscriptionProvider:
        """
        Create a provider from full audio processing configuration.

        :param audio_config: Complete audio processing configuration
        :param provider_name: Optional override for provider name
        :return: Configured provider instance
        :raises ConfigurationError: If provider config not found
        """
        # Determine which provider to use
        name = provider_name or self._get_default_provider(audio_config)

        # Find provider config
        provider_config = self._find_provider_config(audio_config, name)

        if not provider_config:
            raise ConfigurationError(
                f"No configuration found for provider: {name}"
            )

        return self.create(
            provider_name=name,
            provider_config=provider_config,
            transcription_config=audio_config.transcription,
        )

    def _get_default_provider(
        self,
        audio_config: AudioProcessingConfig,
    ) -> str:
        """
        Determine the default provider from configuration.

        :param audio_config: Audio processing configuration
        :return: Default provider name
        """
        # Check if there's only one provider configured
        if audio_config.providers and len(audio_config.providers) == 1:
            return audio_config.providers[0].name

        # Default to assemblyai
        return "assemblyai"

    def _find_provider_config(
        self,
        audio_config: AudioProcessingConfig,
        provider_name: str,
    ) -> Optional[ProviderConfig]:
        """
        Find provider configuration by name.

        :param audio_config: Audio processing configuration
        :param provider_name: Provider name to find
        :return: Provider config or None
        """
        if not audio_config.providers:
            return None

        name_lower = provider_name.lower()
        for provider in audio_config.providers:
            if provider.name.lower() == name_lower:
                return provider

        return None

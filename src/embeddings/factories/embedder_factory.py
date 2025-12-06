"""
Embedder factory for creating embedding service instances.

This module provides a factory for creating embedder instances
based on configuration. It uses a registry pattern to support
multiple providers.
"""

from pathlib import Path
from typing import Dict, Optional, Type

from src.core.logging import get_logger
from src.embeddings.config import EmbeddingConfig
from src.embeddings.providers.base import EmbeddingProvider
from src.embeddings.providers.haystack_provider import HaystackEmbeddingProvider
from src.embeddings.services import BatchProcessor, DocumentEmbedder, QueryEmbedder

logger = get_logger(__name__)


class EmbedderFactory:
    """
    Factory for creating embedding service instances.

    This factory uses a registry pattern to support multiple providers
    and creates fully configured service instances based on configuration.

    Example:
        config = EmbeddingConfig.load()
        document_embedder = EmbedderFactory.create_document_embedder(config)
        query_embedder = EmbedderFactory.create_query_embedder(config)
    """

    # Provider registry: maps provider name to provider class
    _provider_registry: Dict[str, Type[EmbeddingProvider]] = {
        "haystack": HaystackEmbeddingProvider,
    }

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: Type[EmbeddingProvider],
    ) -> None:
        """
        Register a new provider class.

        Args:
            name: Name of the provider (e.g., "openai", "cohere").
            provider_class: Provider class that implements EmbeddingProvider.

        Raises:
            TypeError: If provider_class is not a subclass of EmbeddingProvider.
            ValueError: If name is empty.
        """
        if not name or not name.strip():
            raise ValueError("Provider name cannot be empty")

        if not issubclass(provider_class, EmbeddingProvider):
            raise TypeError("Provider class must be a subclass of EmbeddingProvider")

        cls._provider_registry[name.lower()] = provider_class
        logger.info("Registered provider: %s -> %s", name, provider_class.__name__)

    @classmethod
    def _create_provider(cls, config: EmbeddingConfig) -> EmbeddingProvider:
        """
        Create and warm up an embedding provider based on configuration.

        Args:
            config: Embedding configuration.

        Returns:
            Initialized and warmed-up embedding provider.

        Raises:
            ValueError: If provider is not registered.
            RuntimeError: If provider initialization fails.
        """
        provider_name = config.provider.lower()

        if provider_name not in cls._provider_registry:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {list(cls._provider_registry.keys())}"
            )

        provider_class = cls._provider_registry[provider_name]

        logger.info("Creating provider: %s (%s)", provider_name, provider_class.__name__)

        try:
            # Create provider instance with model config
            provider = provider_class(config.model)

            # Warm up the provider
            logger.debug("Warming up provider: %s", provider_name)
            provider.warm_up()

            logger.info("Provider created and warmed up: %s", provider_name)
            return provider

        except Exception as error:
            logger.error("Failed to create provider %s: %s", provider_name, error)
            raise RuntimeError(f"Failed to create provider '{provider_name}': {error}") from error

    @classmethod
    def create_document_embedder(
        cls,
        config: Optional[EmbeddingConfig] = None,
        config_path: Optional[Path] = None,
    ) -> DocumentEmbedder:
        """
        Create a document embedder instance.

        Args:
            config: Embedding configuration. If None, loads from config_path.
            config_path: Path to configuration file. Used if config is None.

        Returns:
            Initialized DocumentEmbedder instance.

        Raises:
            ValueError: If neither config nor config_path is provided.
            RuntimeError: If creation fails.
        """
        if config is None:
            if config_path is None:
                config = EmbeddingConfig.load()
            else:
                config = EmbeddingConfig.from_yaml(config_path)

        logger.info("Creating DocumentEmbedder")

        try:
            provider = cls._create_provider(config)
            embedder = DocumentEmbedder(provider)

            logger.info("DocumentEmbedder created successfully")
            return embedder

        except Exception as error:
            logger.error("Failed to create DocumentEmbedder: %s", error)
            raise

    @classmethod
    def create_query_embedder(
        cls,
        config: Optional[EmbeddingConfig] = None,
        config_path: Optional[Path] = None,
    ) -> QueryEmbedder:
        """
        Create a query embedder instance.

        Args:
            config: Embedding configuration. If None, loads from config_path.
            config_path: Path to configuration file. Used if config is None.

        Returns:
            Initialized QueryEmbedder instance.

        Raises:
            ValueError: If neither config nor config_path is provided.
            RuntimeError: If creation fails.
        """
        if config is None:
            if config_path is None:
                config = EmbeddingConfig.load()
            else:
                config = EmbeddingConfig.from_yaml(config_path)

        logger.info("Creating QueryEmbedder")

        try:
            provider = cls._create_provider(config)
            embedder = QueryEmbedder(provider)

            logger.info("QueryEmbedder created successfully")
            return embedder

        except Exception as error:
            logger.error("Failed to create QueryEmbedder: %s", error)
            raise

    @classmethod
    def create_batch_processor(
        cls,
        config: Optional[EmbeddingConfig] = None,
        config_path: Optional[Path] = None,
    ) -> BatchProcessor:
        """
        Create a batch processor instance.

        Args:
            config: Embedding configuration. If None, loads from config_path.
            config_path: Path to configuration file. Used if config is None.

        Returns:
            Initialized BatchProcessor instance.

        Raises:
            ValueError: If neither config nor config_path is provided.
            RuntimeError: If creation fails.
        """
        if config is None:
            if config_path is None:
                config = EmbeddingConfig.load()
            else:
                config = EmbeddingConfig.from_yaml(config_path)

        logger.info("Creating BatchProcessor")

        try:
            provider = cls._create_provider(config)
            document_embedder = DocumentEmbedder(provider)
            processor = BatchProcessor(document_embedder, config.processing)

            logger.info("BatchProcessor created successfully")
            return processor

        except Exception as error:
            logger.error("Failed to create BatchProcessor: %s", error)
            raise

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """
        Get list of available provider names.

        Returns:
            List of registered provider names.
        """
        return list(cls._provider_registry.keys())

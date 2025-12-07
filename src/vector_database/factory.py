"""
Vector Database Provider Factory.

Creates provider instances based on configuration.
Following AGENTS.md: extensibility, loose coupling, fail-fast validation.
"""

from typing import Optional
from pathlib import Path

from src.core.logging import get_logger
from src.vector_database.config.vectordb_config import VectorDatabaseConfig
from src.vector_database.providers.base_provider import BaseVectorDBProvider

logger = get_logger(__name__)


class ProviderFactory:
    """
    Factory for creating vector database provider instances.
    
    Responsibilities:
    - Provider instantiation based on config
    - Provider registration and discovery
    - Validation of provider requirements
    
    Design:
    - Extensible: New providers can be registered
    - Fail-fast: Validates config before creation
    - Simple: Single method for provider creation
    """
    
    _providers = {}
    
    @classmethod
    def register_provider(cls, provider_name: str, provider_class: type) -> None:
        """
        Register a provider class.
        
        Args:
            provider_name: Name to register provider under
            provider_class: Provider class implementing BaseVectorDBProvider
            
        Raises:
            ValueError: If provider_name or provider_class invalid
        """
        if not provider_name or not isinstance(provider_name, str):
            raise ValueError("Provider name must be a non-empty string")
        
        if not isinstance(provider_class, type):
            raise ValueError("Provider class must be a class type")
        
        if not issubclass(provider_class, BaseVectorDBProvider):
            raise ValueError(
                f"Provider class must implement BaseVectorDBProvider, "
                f"got {provider_class.__name__}"
            )
        
        cls._providers[provider_name.lower()] = provider_class
        logger.info(f"Registered provider: {provider_name} -> {provider_class.__name__}")
    
    @classmethod
    def create_provider(
        cls,
        config: VectorDatabaseConfig,
        auto_initialize: bool = True
    ) -> BaseVectorDBProvider:
        """
        Create a provider instance from configuration.
        
        Args:
            config: Vector database configuration
            auto_initialize: Whether to call initialize() after creation
            
        Returns:
            Initialized provider instance
            
        Raises:
            ValueError: If config invalid or provider not found
        """
        if not isinstance(config, VectorDatabaseConfig):
            raise ValueError("Config must be a VectorDatabaseConfig instance")
        
        provider_name = config.provider.lower()
        
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                f"Available providers: {available}"
            )
        
        try:
            logger.info(f"Creating provider: {provider_name}")
            provider_class = cls._providers[provider_name]
            provider = provider_class(config)
            
            if auto_initialize:
                logger.info("Auto-initializing provider")
                provider.initialize()
            
            logger.info(f"Provider created successfully: {provider_name}")
            return provider
            
        except Exception as e:
            logger.error(f"Failed to create provider '{provider_name}': {e}")
            raise ValueError(f"Provider creation failed: {e}") from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """
        List all registered provider names.
        
        Returns:
            List of registered provider names
        """
        return sorted(cls._providers.keys())
    
    @classmethod
    def is_provider_available(cls, provider_name: str) -> bool:
        """
        Check if a provider is registered.
        
        Args:
            provider_name: Provider name to check
            
        Returns:
            True if provider is registered
        """
        return provider_name.lower() in cls._providers


# Auto-register Qdrant provider
try:
    from src.vector_database.providers.qdrant_provider import QdrantProvider
    ProviderFactory.register_provider("qdrant", QdrantProvider)
except ImportError as e:
    logger.warning(f"Could not register Qdrant provider: {e}")

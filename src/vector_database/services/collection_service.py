"""
Collection Service for vector database management.

Handles collection-level operations: stats, info, health checks, and maintenance.
Following AGENTS.md: single responsibility, dependency injection, fail-fast validation.
"""

from typing import Dict, Any
from pathlib import Path

from src.core.logging import get_logger
from src.vector_database.providers.base_provider import (
    BaseVectorDBProvider,
    CollectionError,
    DeletionError
)
from src.vector_database.models.collection_stats import CollectionStats, CollectionInfo

logger = get_logger(__name__)


class CollectionService:
    """
    Service for managing vector database collections.
    
    Responsibilities:
    - Collection statistics and metadata
    - Collection information retrieval
    - Collection health monitoring
    - Collection maintenance (clear, recreate)
    
    Design:
    - Depends on BaseVectorDBProvider interface (loose coupling)
    - Provides administrative operations (separation of concerns)
    - Validates operations before execution (defensibility)
    """
    
    def __init__(self, provider: BaseVectorDBProvider):
        """
        Initialize CollectionService.
        
        Args:
            provider: Vector database provider implementing BaseVectorDBProvider
            
        Raises:
            ValueError: If provider is None
        """
        if provider is None:
            raise ValueError("Provider cannot be None")
        
        self.provider = provider
        logger.info(f"CollectionService initialized with {provider.__class__.__name__}")
    
    def get_stats(self) -> CollectionStats:
        """
        Get comprehensive collection statistics.
        
        Returns:
            CollectionStats object with detailed metrics
            
        Raises:
            CollectionError: If stats retrieval fails
        """
        try:
            logger.debug("Retrieving collection statistics")
            stats = self.provider.get_collection_stats()
            
            logger.info(
                f"Collection stats: {stats.total_documents} documents, "
                f"{stats.embedding_dimension}D embeddings"
            )
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise CollectionError(f"Stats retrieval failed: {e}") from e
    
    def get_info(self) -> CollectionInfo:
        """
        Get basic collection information.
        
        Returns:
            CollectionInfo object with basic metrics
            
        Raises:
            CollectionError: If info retrieval fails
        """
        try:
            logger.debug("Retrieving collection info")
            info = self.provider.get_collection_info()
            
            logger.info(f"Collection '{info.name}': {info.vector_count} vectors, status={info.status}")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise CollectionError(f"Info retrieval failed: {e}") from e
    
    def collection_exists(self) -> bool:
        """
        Check if the collection exists.
        
        Returns:
            True if collection exists, False otherwise
        """
        try:
            exists = self.provider.collection_exists()
            logger.debug(f"Collection exists: {exists}")
            return exists
            
        except Exception as e:
            logger.warning(f"Error checking collection existence: {e}")
            return False
    
    def clear_collection(self, confirm: bool = False) -> Dict[str, Any]:
        """
        Clear all documents from the collection.
        
        Args:
            confirm: Must be True to confirm deletion (safety check)
            
        Returns:
            Dict with 'deleted_count' and 'success' keys
            
        Raises:
            ValueError: If confirm is not True
            DeletionError: If clear operation fails
        """
        if not confirm:
            raise ValueError(
                "Must set confirm=True to clear collection. "
                "This will delete all documents permanently."
            )
        
        try:
            logger.warning("Clearing entire collection")
            deleted_count = self.provider.clear_collection()
            
            logger.info(f"Collection cleared: {deleted_count} documents deleted")
            return {
                'deleted_count': deleted_count,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise DeletionError(f"Collection clear failed: {e}") from e
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the collection.
        
        Returns:
            Dict with health status and metrics
        """
        try:
            logger.debug("Performing health check")
            health = self.provider.health_check()
            
            logger.info(f"Health check: status={health.get('status')}, response_time={health.get('response_time')}ms")
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of collection status.
        
        Returns:
            Dict with stats, info, and health status
        """
        try:
            logger.info("Generating collection summary")
            
            stats = self.get_stats()
            info = self.get_info()
            health = self.health_check()
            
            summary = {
                'name': info.name,
                'status': info.status,
                'health': health.get('status'),
                'total_documents': stats.total_documents,
                'vector_count': info.vector_count,
                'embedding_dimension': stats.embedding_dimension,
                'unique_sources': stats.unique_sources,
                'embedding_models': stats.embedding_models,
                'source_types': stats.source_types,
                'storage_path': str(stats.storage_path),
                'indexed': info.indexed,
                'response_time_ms': health.get('response_time')
            }
            
            logger.info("Collection summary generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate collection summary: {e}")
            raise CollectionError(f"Summary generation failed: {e}") from e
    
    def initialize(self) -> None:
        """
        Initialize the collection if not already initialized.
        
        Raises:
            CollectionError: If initialization fails
        """
        try:
            logger.info("Initializing collection")
            self.provider.initialize()
            logger.info("Collection initialized successfully")
            
        except Exception as e:
            logger.error(f"Collection initialization failed: {e}")
            raise CollectionError(f"Initialization failed: {e}") from e
    
    def close(self) -> None:
        """
        Close the collection connection.
        """
        try:
            logger.info("Closing collection connection")
            self.provider.close()
            logger.info("Collection connection closed")
            
        except Exception as e:
            logger.warning(f"Error closing collection connection: {e}")

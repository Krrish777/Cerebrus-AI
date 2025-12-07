"""
Vector Database Module.

Provides unified interface for vector database operations with multiple backends.

Architecture:
- Config: Configuration management with YAML support
- Models: Data models for search results and statistics
- Providers: Backend implementations (Qdrant, future: Pinecone, Weaviate)
- Services: High-level business logic layer
- Factory: Provider instantiation
- Utils: Helper functions

Usage Example:
    from src.vector_database import VectorDatabase
    from src.vector_database.config import VectorDatabaseConfig
    
    # Load config
    config = VectorDatabaseConfig.from_yaml("config/vectordb.yml")
    
    # Create database instance
    db = VectorDatabase(config)
    
    # Use services
    db.document_service.insert_documents(documents)
    results = db.search_service.search(query_embedding)
    stats = db.collection_service.get_stats()
"""

from src.vector_database.config.vectordb_config import VectorDatabaseConfig
from src.vector_database.models.search_result import SearchResult, SearchResults, Citation
from src.vector_database.models.collection_stats import CollectionStats, CollectionInfo
from src.vector_database.factory import ProviderFactory
from src.vector_database.services import DocumentService, SearchService, CollectionService

from src.core.logging import get_logger

logger = get_logger(__name__)


class VectorDatabase:
    """
    Unified interface for vector database operations.
    
    Provides high-level access to document, search, and collection services.
    Handles provider instantiation and service lifecycle.
    
    Example:
        config = VectorDatabaseConfig.from_yaml("config/vectordb.yml")
        db = VectorDatabase(config)
        
        # Insert documents
        result = db.document_service.insert_documents(docs)
        
        # Search
        results = db.search_service.search(embedding, top_k=10)
        
        # Get stats
        stats = db.collection_service.get_stats()
        
        # Close
        db.close()
    """
    
    def __init__(self, config: VectorDatabaseConfig, auto_initialize: bool = True):
        """
        Initialize VectorDatabase.
        
        Args:
            config: Vector database configuration
            auto_initialize: Whether to initialize provider automatically
            
        Raises:
            ValueError: If config invalid or provider creation fails
        """
        if not isinstance(config, VectorDatabaseConfig):
            raise ValueError("Config must be a VectorDatabaseConfig instance")
        
        self.config = config
        
        # Create provider
        logger.info(f"Initializing VectorDatabase with provider: {config.provider}")
        self.provider = ProviderFactory.create_provider(config, auto_initialize=auto_initialize)
        
        # Create services
        self.document_service = DocumentService(self.provider)
        self.search_service = SearchService(self.provider)
        self.collection_service = CollectionService(self.provider)
        
        logger.info("VectorDatabase initialized successfully")
    
    def close(self) -> None:
        """Close the database connection."""
        logger.info("Closing VectorDatabase")
        self.collection_service.close()
        logger.info("VectorDatabase closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


__all__ = [
    # Main interface
    'VectorDatabase',
    
    # Configuration
    'VectorDatabaseConfig',
    
    # Models
    'SearchResult',
    'SearchResults',
    'Citation',
    'CollectionStats',
    'CollectionInfo',
    
    # Services
    'DocumentService',
    'SearchService',
    'CollectionService',
    
    # Factory
    'ProviderFactory',
]

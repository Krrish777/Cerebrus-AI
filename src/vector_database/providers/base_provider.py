"""
Base Provider Interface for Vector Databases

This module defines the abstract base class for all vector database providers.
Following AGENTS.md principles:
- Abstract base class defines contract
- Provider implementations can be swapped without changing consumers
- Clear separation of interface from implementation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..models.search_result import SearchResults
from ..models.collection_stats import CollectionStats, CollectionInfo


class BaseVectorDBProvider(ABC):
    """
    Abstract base class for vector database providers.
    
    This interface defines the contract that all vector database implementations
    must follow, enabling seamless switching between providers (Qdrant, Pinecone,
    Weaviate, etc.) without changing consumer code.
    
    All methods include comprehensive error handling and logging requirements.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the vector database connection and resources.
        
        This method should:
        - Establish database connection
        - Validate configuration
        - Create collection if it doesn't exist
        - Set up any required indices
        
        Raises:
            ConnectionError: If unable to connect to database
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def insert_documents(
        self,
        documents: List[Any],
        policy: str = "skip"
    ) -> List[str]:
        """
        Insert documents with embeddings into the collection.
        
        Args:
            documents: List of documents with embeddings
            policy: Write policy - "skip", "overwrite", or "fail"
            
        Returns:
            List of document IDs that were successfully inserted
            
        Raises:
            ValueError: If documents lack embeddings or policy is invalid
            RuntimeError: If insertion fails
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResults:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query vector as list of floats
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            score_threshold: Minimum similarity score threshold
            
        Returns:
            SearchResults object containing matching documents
            
        Raises:
            ValueError: If query_embedding dimension mismatches
            RuntimeError: If search operation fails
        """
        pass
    
    @abstractmethod
    def get_document_by_id(self, doc_id: str) -> Optional[Any]:
        """
        Retrieve a specific document by its ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document object if found, None otherwise
            
        Raises:
            RuntimeError: If retrieval operation fails
        """
        pass
    
    @abstractmethod
    def delete_documents(self, doc_ids: List[str]) -> int:
        """
        Delete documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents successfully deleted
            
        Raises:
            RuntimeError: If deletion operation fails
        """
        pass
    
    @abstractmethod
    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents in the collection.
        
        Args:
            filters: Optional metadata filters to apply
            
        Returns:
            Total number of documents matching filters
            
        Raises:
            RuntimeError: If count operation fails
        """
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> CollectionStats:
        """
        Get comprehensive statistics about the collection.
        
        Returns:
            CollectionStats object with detailed metrics
            
        Raises:
            RuntimeError: If unable to retrieve statistics
        """
        pass
    
    @abstractmethod
    def get_collection_info(self) -> CollectionInfo:
        """
        Get basic information about the collection.
        
        Returns:
            CollectionInfo object with basic metrics
            
        Raises:
            RuntimeError: If unable to retrieve information
        """
        pass
    
    @abstractmethod
    def collection_exists(self) -> bool:
        """
        Check if the collection exists.
        
        Returns:
            True if collection exists, False otherwise
            
        Raises:
            RuntimeError: If unable to check collection existence
        """
        pass
    
    @abstractmethod
    def clear_collection(self) -> int:
        """
        Clear all documents from the collection.
        
        Returns:
            Number of documents deleted
            
        Raises:
            RuntimeError: If clear operation fails
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Close the database connection and release resources.
        
        This method should:
        - Close any open connections
        - Release resources
        - Clean up temporary files
        
        Raises:
            RuntimeError: If unable to close connection cleanly
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection.
        
        Returns:
            Dictionary with health status information:
            - status: 'healthy' or 'unhealthy'
            - response_time_ms: Connection response time
            - details: Additional diagnostic information
            
        Raises:
            RuntimeError: If health check cannot be performed
        """
        pass


class VectorDBProviderError(Exception):
    """Base exception for vector database provider errors."""
    pass


class ConnectionError(VectorDBProviderError):
    """Raised when unable to connect to vector database."""
    pass


class InsertionError(VectorDBProviderError):
    """Raised when document insertion fails."""
    pass


class SearchError(VectorDBProviderError):
    """Raised when search operation fails."""
    pass


class DeletionError(VectorDBProviderError):
    """Raised when document deletion fails."""
    pass


class CollectionError(VectorDBProviderError):
    """Raised when collection operations fail."""
    pass

"""
Base provider protocols for RAG system.
Defines abstract interfaces for retrieval, ranking, and generation providers.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from haystack.dataclasses import Document

from src.core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class RetrieverProvider(Protocol):
    """Protocol for document retrieval providers."""
    
    def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: Search query string
            filters: Optional filters to apply
            top_k: Maximum number of documents to retrieve
            
        Returns:
            Dictionary with 'documents' key containing list of Document objects
        """
        ...
    
    def warm_up(self) -> None:
        """Warm up the retriever (load models, establish connections, etc.)."""
        ...


@runtime_checkable
class RankerProvider(Protocol):
    """Protocol for document ranking providers."""
    
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> Dict[str, List[Document]]:
        """
        Rank documents by relevance to the query.
        
        Args:
            query: Query string for ranking
            documents: List of documents to rank
            top_k: Maximum number of ranked documents to return
            score_threshold: Minimum score threshold
            
        Returns:
            Dictionary with 'documents' key containing ranked Document objects
        """
        ...
    
    def warm_up(self) -> None:
        """Warm up the ranker (load models, etc.)."""
        ...


@runtime_checkable
class GeneratorProvider(Protocol):
    """Protocol for response generation providers."""
    
    def run(
        self,
        messages: List[Any],
        **generation_kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate response from messages.
        
        Args:
            messages: List of chat messages
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated response
        """
        ...
    
    def warm_up(self) -> None:
        """Warm up the generator (establish connections, etc.)."""
        ...


@runtime_checkable
class DocumentStoreProvider(Protocol):
    """Protocol for document store providers."""
    
    def write_documents(self, documents: List[Document]) -> int:
        """
        Write documents to the store.
        
        Args:
            documents: List of Document objects to write
            
        Returns:
            Number of documents written
        """
        ...
    
    def filter_documents(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Filter documents by criteria.
        
        Args:
            filters: Filter criteria
            
        Returns:
            List of matching Document objects
        """
        ...
    
    def count_documents(self) -> int:
        """
        Count total documents in store.
        
        Returns:
            Total document count
        """
        ...
    
    def delete_documents(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Delete documents matching filters.
        
        Args:
            filters: Filter criteria for deletion
        """
        ...


def validate_retriever_provider(provider: Any) -> bool:
    """
    Validate that an object implements the RetrieverProvider protocol.
    
    Args:
        provider: Object to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(provider, RetrieverProvider):
        logger.error(f"{type(provider).__name__} does not implement RetrieverProvider protocol")
        return False
    
    required_methods = ['run', 'warm_up']
    for method in required_methods:
        if not hasattr(provider, method) or not callable(getattr(provider, method)):
            logger.error(f"{type(provider).__name__} missing required method: {method}")
            return False
    
    return True


def validate_ranker_provider(provider: Any) -> bool:
    """
    Validate that an object implements the RankerProvider protocol.
    
    Args:
        provider: Object to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(provider, RankerProvider):
        logger.error(f"{type(provider).__name__} does not implement RankerProvider protocol")
        return False
    
    required_methods = ['run', 'warm_up']
    for method in required_methods:
        if not hasattr(provider, method) or not callable(getattr(provider, method)):
            logger.error(f"{type(provider).__name__} missing required method: {method}")
            return False
    
    return True


def validate_generator_provider(provider: Any) -> bool:
    """
    Validate that an object implements the GeneratorProvider protocol.
    
    Args:
        provider: Object to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(provider, GeneratorProvider):
        logger.error(f"{type(provider).__name__} does not implement GeneratorProvider protocol")
        return False
    
    required_methods = ['run', 'warm_up']
    for method in required_methods:
        if not hasattr(provider, method) or not callable(getattr(provider, method)):
            logger.error(f"{type(provider).__name__} missing required method: {method}")
            return False
    
    return True

"""
InMemory BM25 retriever provider.
"""

from typing import Any, Dict, List, Optional

from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from src.core.logging import get_logger

logger = get_logger(__name__)


class InMemoryRetrieverProvider:
    """InMemory BM25 retriever provider implementation."""
    
    def __init__(
        self,
        document_store: InMemoryDocumentStore,
        top_k: int = 20
    ):
        """
        Initialize InMemory retriever.
        
        Args:
            document_store: InMemory document store instance
            top_k: Maximum number of documents to retrieve
        """
        self.document_store = document_store
        self.top_k = top_k
        
        self._retriever = InMemoryBM25Retriever(
            document_store=document_store,
            top_k=top_k
        )
        
        logger.info(f"Initialized InMemoryRetrieverProvider with top_k={top_k}")
    
    def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents using BM25.
        
        Args:
            query: Search query
            filters: Optional filters (not supported by InMemoryBM25Retriever)
            top_k: Override default top_k
            
        Returns:
            Dictionary with 'documents' key containing retrieved documents
        """
        effective_top_k = top_k or self.top_k
        
        try:
            # InMemoryBM25Retriever doesn't support filters parameter
            result = self._retriever.run(query=query)
            
            if filters:
                logger.warning("InMemoryBM25Retriever does not support filters, ignoring")
            
            documents = result.get("documents", [])
            
            # Apply top_k manually if needed
            if len(documents) > effective_top_k:
                documents = documents[:effective_top_k]
            
            logger.debug(f"Retrieved {len(documents)} documents for query: {query[:50]}")
            
            return {"documents": documents}
            
        except Exception as e:
            logger.error(f"Error during InMemory retrieval: {e}")
            return {"documents": []}
    
    def warm_up(self) -> None:
        """Warm up retriever (no-op for InMemory)."""
        logger.debug("InMemoryRetrieverProvider does not require warm-up")

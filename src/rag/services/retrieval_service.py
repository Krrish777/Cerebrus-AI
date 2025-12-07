"""
Retrieval service.
Orchestrates document retrieval operations.
"""

from typing import Any, Dict, List, Optional

from haystack.dataclasses import Document

from src.core.logging import get_logger
from src.rag.providers.base import RetrieverProvider

logger = get_logger(__name__)


class RetrievalService:
    """Service for document retrieval operations."""
    
    def __init__(self, retriever: RetrieverProvider):
        """
        Initialize retrieval service.
        
        Args:
            retriever: Retriever provider instance
        """
        self.retriever = retriever
        logger.info("Initialized RetrievalService")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve documents relevant to query.
        
        Args:
            query: Search query
            top_k: Maximum documents to retrieve
            filters: Optional metadata filters
            
        Returns:
            List of retrieved documents
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for retrieval")
            return []
        
        try:
            logger.debug(f"Retrieving documents for query: {query[:50]}")
            
            result = self.retriever.run(
                query=query,
                top_k=top_k,
                filters=filters
            )
            
            documents = result.get("documents", [])
            
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def warm_up(self) -> None:
        """Warm up retriever."""
        try:
            self.retriever.warm_up()
            logger.info("RetrievalService warmed up successfully")
        except Exception as e:
            logger.warning(f"Failed to warm up retriever: {e}")

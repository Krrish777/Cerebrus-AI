"""
Elasticsearch BM25 retriever provider.
"""

from typing import Any, Dict, List, Optional

from haystack.dataclasses import Document

from src.core.logging import get_logger

logger = get_logger(__name__)

try:
    from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever
    from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    logger.warning("Elasticsearch integration not available")


class ElasticsearchRetrieverProvider:
    """Elasticsearch BM25 retriever provider implementation."""
    
    def __init__(
        self,
        document_store: "ElasticsearchDocumentStore",
        top_k: int = 20,
        fuzziness: str = "AUTO"
    ):
        """
        Initialize Elasticsearch retriever.
        
        Args:
            document_store: Elasticsearch document store instance
            top_k: Maximum number of documents to retrieve
            fuzziness: BM25 fuzziness parameter (AUTO, 0, 1, 2)
            
        Raises:
            ImportError: If Elasticsearch integration not available
        """
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError(
                "Elasticsearch integration not available. "
                "Install with: pip install elasticsearch-haystack"
            )
        
        self.document_store = document_store
        self.top_k = top_k
        self.fuzziness = fuzziness
        
        self._retriever = ElasticsearchBM25Retriever(
            document_store=document_store,
            fuzziness=fuzziness,
            top_k=top_k
        )
        
        logger.info(
            f"Initialized ElasticsearchRetrieverProvider with "
            f"top_k={top_k}, fuzziness={fuzziness}"
        )
    
    def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents using Elasticsearch BM25.
        
        Args:
            query: Search query
            filters: Optional Elasticsearch filters
            top_k: Override default top_k
            
        Returns:
            Dictionary with 'documents' key containing retrieved documents
        """
        effective_top_k = top_k or self.top_k
        
        try:
            run_kwargs = {"query": query}
            
            if filters:
                run_kwargs["filters"] = filters
            
            result = self._retriever.run(**run_kwargs)
            documents = result.get("documents", [])
            
            # Apply top_k if needed
            if len(documents) > effective_top_k:
                documents = documents[:effective_top_k]
            
            logger.debug(
                f"Retrieved {len(documents)} documents from Elasticsearch "
                f"for query: {query[:50]}"
            )
            
            return {"documents": documents}
            
        except Exception as e:
            logger.error(f"Error during Elasticsearch retrieval: {e}")
            return {"documents": []}
    
    def warm_up(self) -> None:
        """Warm up retriever (test connection)."""
        try:
            # Test connection by counting documents
            count = self.document_store.count_documents()
            logger.info(f"ElasticsearchRetrieverProvider warmed up, {count} documents in store")
        except Exception as e:
            logger.warning(f"Failed to warm up Elasticsearch retriever: {e}")

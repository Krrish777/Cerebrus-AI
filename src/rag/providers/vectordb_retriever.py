"""
VectorDatabase retriever provider.
Integrates with the completed VectorDatabase module for RAG retrieval.
"""

from typing import Any, Dict, List, Optional

from haystack.dataclasses import Document

from src.core.logging import get_logger
from src.vector_database import VectorDatabase, VectorDatabaseConfig

logger = get_logger(__name__)


class VectorDatabaseRetrieverProvider:
    """
    VectorDatabase retriever provider implementation.
    
    Integrates with the modular VectorDatabase system to provide
    semantic search capabilities for RAG.
    """
    
    def __init__(
        self,
        vectordb: VectorDatabase,
        embedding_model: Any,
        top_k: int = 20,
        score_threshold: Optional[float] = None
    ):
        """
        Initialize VectorDatabase retriever.
        
        Args:
            vectordb: VectorDatabase instance
            embedding_model: Embedding model for query encoding
            top_k: Maximum number of documents to retrieve
            score_threshold: Minimum similarity score threshold
        """
        self.vectordb = vectordb
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        logger.info(
            f"Initialized VectorDatabaseRetrieverProvider with "
            f"top_k={top_k}, threshold={score_threshold}"
        )
    
    @classmethod
    def from_config(
        cls,
        vectordb_config: VectorDatabaseConfig,
        embedding_model: Any,
        top_k: int = 20,
        score_threshold: Optional[float] = None
    ) -> "VectorDatabaseRetrieverProvider":
        """
        Create provider from VectorDatabase config.
        
        Args:
            vectordb_config: VectorDatabase configuration
            embedding_model: Embedding model for queries
            top_k: Maximum documents to retrieve
            score_threshold: Minimum score threshold
            
        Returns:
            Initialized provider
        """
        vectordb = VectorDatabase(vectordb_config, auto_initialize=True)
        return cls(vectordb, embedding_model, top_k, score_threshold)
    
    def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents using vector search.
        
        Args:
            query: Search query text
            filters: Optional metadata filters
            top_k: Override default top_k
            
        Returns:
            Dictionary with 'documents' key containing retrieved Haystack Documents
        """
        effective_top_k = top_k or self.top_k
        
        try:
            # Generate query embedding
            query_embedding = self._embed_query(query)
            
            # Perform vector search
            search_results = self.vectordb.search_service.search(
                query_embedding=query_embedding,
                top_k=effective_top_k,
                score_threshold=self.score_threshold,
                filters=filters
            )
            
            # Convert VectorDB SearchResults to Haystack Documents
            haystack_docs = self._convert_to_haystack_documents(search_results)
            
            logger.debug(
                f"Retrieved {len(haystack_docs)} documents via VectorDB "
                f"for query: {query[:50]}"
            )
            
            return {"documents": haystack_docs}
            
        except Exception as e:
            logger.error(f"Error during VectorDB retrieval: {e}")
            return {"documents": []}
    
    def _embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        try:
            # Support different embedding model interfaces
            if hasattr(self.embedding_model, 'embed_query'):
                # Standard interface
                embedding = self.embedding_model.embed_query(query)
            elif hasattr(self.embedding_model, 'encode'):
                # Sentence transformers interface
                embedding = self.embedding_model.encode(query)
            elif callable(self.embedding_model):
                # Callable interface
                embedding = self.embedding_model(query)
            else:
                raise ValueError(f"Unsupported embedding model: {type(self.embedding_model)}")
            
            # Ensure it's a list
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def _convert_to_haystack_documents(
        self,
        search_results: Any
    ) -> List[Document]:
        """
        Convert VectorDB SearchResults to Haystack Documents.
        
        Args:
            search_results: VectorDB SearchResults object
            
        Returns:
            List of Haystack Document objects
        """
        haystack_docs = []
        
        for result in search_results.results:
            # Extract content and metadata
            content = result.content or ""
            meta = result.metadata.copy() if result.metadata else {}
            
            # Add score to metadata
            meta['relevance_score'] = result.score
            
            # Create Haystack Document
            doc = Document(
                content=content,
                meta=meta,
                score=result.score
            )
            
            haystack_docs.append(doc)
        
        return haystack_docs
    
    def warm_up(self) -> None:
        """Warm up retriever (verify VectorDB connection)."""
        try:
            stats = self.vectordb.collection_service.get_stats()
            logger.info(
                f"VectorDatabaseRetrieverProvider warmed up, "
                f"{stats.total_vectors} vectors in collection"
            )
        except Exception as e:
            logger.warning(f"Failed to warm up VectorDB retriever: {e}")
    
    def close(self) -> None:
        """Close VectorDatabase connection."""
        self.vectordb.close()
        logger.debug("VectorDatabaseRetrieverProvider closed")

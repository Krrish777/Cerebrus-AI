"""
Qdrant Provider Implementation

This module provides the Qdrant-specific implementation of the vector database provider.
Uses Haystack's QdrantDocumentStore and QdrantEmbeddingRetriever for operations.

Following AGENTS.md principles:
- Single responsibility (Qdrant integration only)
- Dependency injection (receives config, not creates it)
- No hard-coded values (all from config)
- Defensive programming with validation
- No emojis in logs
"""

import time
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from haystack import Document as HaystackDocument
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    HaystackDocument = None
    QdrantDocumentStore = None
    QdrantEmbeddingRetriever = None

from src.core.logging import CustomLogger
from .base_provider import (
    BaseVectorDBProvider,
    ConnectionError as ProviderConnectionError,
    InsertionError,
    SearchError,
    DeletionError,
    CollectionError
)
from ..config.vectordb_config import VectorDatabaseConfig
from ..models.search_result import SearchResult, SearchResults, Citation
from ..models.collection_stats import CollectionStats, CollectionInfo


# Initialize logger
custom_logger = CustomLogger()
logger = custom_logger.get_logger(__name__)


class QdrantProvider(BaseVectorDBProvider):
    """
    Qdrant vector database provider implementation using Haystack.
    
    This provider integrates with Qdrant through Haystack's document store
    and retriever components, providing a clean abstraction over the underlying
    Qdrant client.
    
    Attributes:
        config: Vector database configuration
        document_store: Haystack QdrantDocumentStore instance
        retriever: Haystack QdrantEmbeddingRetriever instance
    """
    
    def __init__(self, config: VectorDatabaseConfig):
        """
        Initialize Qdrant provider with configuration.
        
        Args:
            config: VectorDatabaseConfig instance
            
        Raises:
            ImportError: If qdrant-haystack is not installed
            ValueError: If config is invalid
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-haystack is required for QdrantProvider. "
                "Install it with: pip install qdrant-haystack"
            )
        
        if config.provider != "qdrant":
            raise ValueError(f"QdrantProvider requires provider='qdrant', got '{config.provider}'")
        
        self.config = config
        self.document_store: Optional[QdrantDocumentStore] = None
        self.retriever: Optional[QdrantEmbeddingRetriever] = None
        
        logger.info(f"QdrantProvider initialized with config for collection '{config.collection_name}'")
    
    def initialize(self) -> None:
        """
        Initialize Qdrant document store and retriever.
        
        Raises:
            ConnectionError: If unable to initialize Qdrant components
        """
        try:
            logger.info("Initializing Qdrant document store")
            logger.debug(f"Storage path: {self.config.storage_path}")
            logger.debug(f"Collection: {self.config.collection_name}")
            logger.debug(f"Embedding dimension: {self.config.embedding_dim}")
            
            # Ensure storage directory exists
            storage_path = Path(self.config.storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize document store with path parameter for local file persistence
            # Using 'path' instead of 'location' avoids URL parsing issues on Windows
            self.document_store = QdrantDocumentStore(
                path=str(storage_path),
                index=self.config.collection_name,
                embedding_dim=self.config.embedding_dim,
                recreate_index=self.config.qdrant.recreate_index,
                return_embedding=self.config.qdrant.return_embedding,
                wait_result_from_api=self.config.qdrant.wait_result_from_api,
                hnsw_config={
                    "m": self.config.qdrant.hnsw.m,
                    "ef_construct": self.config.qdrant.hnsw.ef_construct,
                    "full_scan_threshold": self.config.qdrant.hnsw.full_scan_threshold
                },
                quantization_config=self.config.qdrant.quantization.to_qdrant_config()
            )
            
            logger.info("Qdrant document store initialized successfully")
            
            # Log current document count
            try:
                doc_count = self.document_store.count_documents()
                logger.info(f"Current collection contains {doc_count} documents")
            except Exception as e:
                logger.debug(f"Could not retrieve initial document count: {e}")
            
            # Initialize retriever
            logger.info("Initializing Qdrant retriever")
            self.retriever = QdrantEmbeddingRetriever(
                document_store=self.document_store
            )
            logger.info("Qdrant retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant components: {e}")
            raise ProviderConnectionError(f"Qdrant initialization failed: {e}") from e
    
    def insert_documents(
        self,
        documents: List[Any],
        policy: str = "skip"
    ) -> List[str]:
        """
        Insert documents with embeddings into Qdrant.
        
        Args:
            documents: List of Haystack Document objects with embeddings
            policy: Write policy - "skip", "overwrite", or "fail"
            
        Returns:
            List of document IDs successfully inserted
            
        Raises:
            ValueError: If documents invalid or policy unknown
            InsertionError: If insertion fails
        """
        # Check system state first (Defensibility principle)
        if self.document_store is None:
            raise InsertionError("Document store not initialized. Call initialize() first")
        
        # Validate policy parameter
        if policy not in ("skip", "overwrite", "fail"):
            raise ValueError(f"Invalid policy '{policy}', must be 'skip', 'overwrite', or 'fail'")
        
        # Early return for empty input
        if not documents:
            logger.warning("No documents provided for insertion")
            return []
        
        try:
            logger.info(f"Inserting {len(documents)} documents with policy '{policy}'")
            
            # Validate documents have embeddings
            docs_with_embeddings = []
            docs_without_embeddings = []
            
            for doc in documents:
                if hasattr(doc, 'embedding') and doc.embedding is not None:
                    docs_with_embeddings.append(doc)
                else:
                    docs_without_embeddings.append(doc)
            
            if docs_without_embeddings:
                logger.warning(
                    f"{len(docs_without_embeddings)} documents lack embeddings and will be skipped"
                )
                logger.debug(
                    f"Documents without embeddings: "
                    f"{[getattr(doc, 'id', 'no_id') for doc in docs_without_embeddings[:5]]}"
                )
            
            if not docs_with_embeddings:
                logger.error("No documents with embeddings found")
                return []
            
            # Map policy to Haystack DuplicatePolicy
            policy_map = {
                "skip": "SKIP",
                "overwrite": "OVERWRITE",
                "fail": "FAIL"
            }
            haystack_policy = policy_map[policy]
            
            # Insert documents
            written_docs = self.document_store.write_documents(
                documents=docs_with_embeddings,
                policy=haystack_policy
            )
            
            inserted_ids = [doc.id for doc in written_docs if hasattr(doc, 'id')]
            logger.info(f"Successfully inserted {len(inserted_ids)} documents")
            
            # Log updated stats
            try:
                total_docs = self.document_store.count_documents()
                logger.info(f"Total documents in collection: {total_docs}")
            except Exception as e:
                logger.debug(f"Could not retrieve updated document count: {e}")
            
            return inserted_ids
            
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise InsertionError(f"Document insertion failed: {e}") from e
    
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
            SearchResults object with matching documents
            
        Raises:
            ValueError: If parameters invalid
            SearchError: If search fails
        """
        if self.retriever is None:
            raise SearchError("Retriever not initialized. Call initialize() first")
        
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")
        
        if len(query_embedding) != self.config.embedding_dim:
            raise ValueError(
                f"Query embedding dimension ({len(query_embedding)}) "
                f"does not match config ({self.config.embedding_dim})"
            )
        
        if top_k < 1:
            raise ValueError(f"top_k must be positive, got {top_k}")
        
        try:
            start_time = time.time()
            logger.info(f"Searching for top {top_k} similar documents")
            if filters:
                logger.debug(f"Applied filters: {filters}")
            
            # Use config defaults if not specified
            if score_threshold is None:
                score_threshold = self.config.search.score_threshold
            
            # Perform search
            results = self.retriever.run(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
                scale_score=self.config.search.scale_score
            )
            
            query_time_ms = (time.time() - start_time) * 1000
            
            documents = results.get('documents', [])
            logger.info(f"Found {len(documents)} matching documents in {query_time_ms:.2f}ms")
            
            # Apply score threshold if specified
            if score_threshold is not None:
                documents = [
                    doc for doc in documents
                    if getattr(doc, 'score', 0.0) >= score_threshold
                ]
                logger.debug(f"After score threshold {score_threshold}: {len(documents)} documents")
            
            # Convert to SearchResult objects
            search_results = []
            for doc in documents:
                citation = self._extract_citation_info(getattr(doc, 'meta', {}))
                
                result = SearchResult(
                    id=getattr(doc, 'id', ''),
                    score=getattr(doc, 'score', 0.0),
                    content=doc.content if hasattr(doc, 'content') else '',
                    metadata=getattr(doc, 'meta', {}),
                    citation=citation,
                    embedding=getattr(doc, 'embedding', None) if self.config.search.return_embedding else None
                )
                search_results.append(result)
            
            return SearchResults(
                results=search_results,
                total_results=len(search_results),
                query_time_ms=query_time_ms
            )
            
        except Exception as e:
            logger.error(f"Search operation failed: {e}")
            raise SearchError(f"Search failed: {e}") from e
    
    def get_document_by_id(self, doc_id: str) -> Optional[Any]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document object if found, None otherwise
            
        Raises:
            SearchError: If retrieval fails
        """
        if self.document_store is None:
            raise SearchError("Document store not initialized. Call initialize() first")
        
        if not doc_id:
            raise ValueError("Document ID cannot be empty")
        
        try:
            logger.debug(f"Retrieving document with ID: {doc_id}")
            
            # Use filter to find document by ID
            results = self.document_store.filter_documents(
                filters={"field": "id", "operator": "==", "value": doc_id}
            )
            
            if results:
                logger.debug(f"Document {doc_id} found")
                return results[0]
            else:
                logger.warning(f"Document {doc_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve document {doc_id}: {e}")
            raise SearchError(f"Document retrieval failed: {e}") from e
    
    def delete_documents(self, doc_ids: List[str]) -> int:
        """
        Delete documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
            
        Raises:
            DeletionError: If deletion fails
        """
        if self.document_store is None:
            raise DeletionError("Document store not initialized. Call initialize() first")
        
        if not doc_ids:
            logger.warning("No document IDs provided for deletion")
            return 0
        
        try:
            logger.info(f"Deleting {len(doc_ids)} documents")
            
            self.document_store.delete_documents(doc_ids)
            
            logger.info(f"Successfully deleted {len(doc_ids)} documents")
            
            # Log updated stats
            try:
                remaining_docs = self.document_store.count_documents()
                logger.info(f"Remaining documents in collection: {remaining_docs}")
            except Exception as e:
                logger.debug(f"Could not retrieve updated document count: {e}")
            
            return len(doc_ids)
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise DeletionError(f"Document deletion failed: {e}") from e
    
    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents in the collection.
        
        Args:
            filters: Optional metadata filters
            
        Returns:
            Total number of documents matching filters
            
        Raises:
            CollectionError: If count operation fails
        """
        if self.document_store is None:
            raise CollectionError("Document store not initialized. Call initialize() first")
        
        try:
            if filters:
                logger.debug(f"Counting documents with filters: {filters}")
                # Filter documents and count
                filtered_docs = self.document_store.filter_documents(filters=filters)
                count = len(filtered_docs)
            else:
                count = self.document_store.count_documents()
            
            logger.debug(f"Document count: {count}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            raise CollectionError(f"Document count failed: {e}") from e
    
    def get_collection_stats(self) -> CollectionStats:
        """
        Get comprehensive statistics about the collection.
        
        Returns:
            CollectionStats object with detailed metrics
            
        Raises:
            CollectionError: If unable to retrieve statistics
        """
        if self.document_store is None:
            raise CollectionError("Document store not initialized. Call initialize() first")
        
        try:
            logger.debug("Gathering collection statistics")
            
            # Basic document count
            total_docs = self.document_store.count_documents()
            
            # Get sample documents for metadata analysis
            sample_docs = self.document_store.filter_documents()[:100]
            
            # Analyze metadata
            embedding_models = set()
            source_types = set()
            source_files = set()
            
            for doc in sample_docs:
                meta = getattr(doc, 'meta', {})
                if 'embedding_model' in meta:
                    embedding_models.add(meta['embedding_model'])
                if 'source_type' in meta:
                    source_types.add(meta['source_type'])
                if 'source_file' in meta:
                    source_files.add(meta['source_file'])
            
            stats = CollectionStats(
                total_documents=total_docs,
                collection_name=self.config.collection_name,
                embedding_dimension=self.config.embedding_dim,
                storage_path=self.config.storage_path,
                embedding_models=list(embedding_models),
                source_types=list(source_types),
                unique_sources=len(source_files),
                hnsw_config={
                    "m": self.config.qdrant.hnsw.m,
                    "ef_construct": self.config.qdrant.hnsw.ef_construct,
                    "full_scan_threshold": self.config.qdrant.hnsw.full_scan_threshold
                },
                quantization_enabled=self.config.qdrant.quantization.enabled
            )
            
            logger.info(f"Collection stats: {total_docs} documents, {len(source_files)} unique sources")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to gather collection statistics: {e}")
            raise CollectionError(f"Statistics retrieval failed: {e}") from e
    
    def get_collection_info(self) -> CollectionInfo:
        """
        Get basic information about the collection.
        
        Returns:
            CollectionInfo object with basic metrics
            
        Raises:
            CollectionError: If unable to retrieve information
        """
        if self.document_store is None:
            raise CollectionError("Document store not initialized. Call initialize() first")
        
        try:
            doc_count = self.document_store.count_documents()
            
            return CollectionInfo(
                name=self.config.collection_name,
                vector_count=doc_count,
                indexed=True,  # Qdrant always indexes
                status="green"  # Simplified status
            )
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise CollectionError(f"Collection info retrieval failed: {e}") from e
    
    def collection_exists(self) -> bool:
        """
        Check if the collection exists.
        
        Returns:
            True if collection exists, False otherwise
            
        Raises:
            CollectionError: If unable to check existence
        """
        if self.document_store is None:
            raise CollectionError("Document store not initialized. Call initialize() first")
        
        try:
            # Try to count documents - if successful, collection exists
            self.document_store.count_documents()
            return True
        except Exception:
            return False
    
    def clear_collection(self) -> int:
        """
        Clear all documents from the collection.
        
        Returns:
            Number of documents deleted
            
        Raises:
            DeletionError: If clear operation fails
        """
        if self.document_store is None:
            raise DeletionError("Document store not initialized. Call initialize() first")
        
        try:
            logger.warning("Clearing entire collection")
            
            # Get all document IDs
            all_docs = self.document_store.filter_documents()
            doc_ids = [doc.id for doc in all_docs if hasattr(doc, 'id')]
            
            if doc_ids:
                return self.delete_documents(doc_ids)
            else:
                logger.info("Collection is already empty")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise DeletionError(f"Collection clear failed: {e}") from e
    
    def close(self) -> None:
        """
        Close the database connection and release resources.
        """
        try:
            logger.info("Closing Qdrant database connection")
            # Qdrant document store doesn't require explicit closing
            # but we can clean up references
            self.document_store = None
            self.retriever = None
            logger.info("Qdrant database connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
            raise RuntimeError(f"Failed to close connection: {e}") from e
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the Qdrant connection.
        
        Returns:
            Dictionary with health status information
            
        Raises:
            RuntimeError: If health check cannot be performed
        """
        try:
            start_time = time.time()
            
            if self.document_store is None:
                return {
                    'status': 'unhealthy',
                    'response_time_ms': 0,
                    'details': 'Document store not initialized'
                }
            
            # Try to count documents as health check
            doc_count = self.document_store.count_documents()
            response_time_ms = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time_ms,
                'details': {
                    'collection': self.config.collection_name,
                    'document_count': doc_count,
                    'storage_path': str(self.config.storage_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'response_time_ms': 0,
                'details': str(e)
            }
    
    def _extract_citation_info(self, metadata: Dict[str, Any]) -> Citation:
        """
        Extract citation information from document metadata.
        
        Args:
            metadata: Document metadata dictionary
            
        Returns:
            Citation object with extracted information
        """
        return Citation(
            source_file=metadata.get('source_file'),
            source_type=metadata.get('source_type'),
            page_number=metadata.get('page_number'),
            chunk_index=metadata.get('chunk_index'),
            start_char=metadata.get('start_char'),
            end_char=metadata.get('end_char')
        )

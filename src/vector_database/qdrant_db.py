"""
Qdrant Vector Database Implementation for Cerebrus AI

This module provides a Qdrant-based vector database implementation using Haystack's
QdrantDocumentStore for efficient document storage and retrieval.

Features:
- Persistent storage in local folder
- Document indexing with metadata
- Vector similarity search
- Hybrid search capabilities
- Full document lifecycle management
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from haystack import Document as HaystackDocument
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore # type: ignore
    from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever # type: ignore
    QDRANT_AVAILABLE = True
    Document = HaystackDocument # type: ignore
except ImportError:
    QDRANT_AVAILABLE = False
    # Fallback for testing without qdrant-haystack
    class QdrantDocumentStore:
        def __init__(self, *args, **kwargs):
            raise ImportError("qdrant-haystack is required")
    
    class QdrantEmbeddingRetriever:
        def __init__(self, *args, **kwargs):
            raise ImportError("qdrant-haystack is required")
    
    class Document:
        def __init__(self, content: str, meta: Optional[Dict[str, Any]] = None, id: Optional[str] = None):
            self.content = content
            self.meta = meta or {}
            self.id = id

# Import CustomLogger
try:
    from ..core.logging import CustomLogger
    from ..embeddings.embedding_generator import EmbeddingGenerator, EmbeddedDocument
except ImportError:
    # Fallback for when run as standalone script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.core.logging import CustomLogger
    from src.embeddings.embedding_generator import EmbeddingGenerator, EmbeddedDocument

# Initialize logger
custom_logger = CustomLogger()
logger = custom_logger.get_logger(__name__)


class QdrantVectorDB:
    """
    Qdrant-based vector database for Cerebrus AI.
    
    This class provides a comprehensive interface for storing and retrieving
    documents using Qdrant vector database with Haystack integration.
    
    Features:
    - Persistent local storage
    - Document embedding and indexing
    - Vector similarity search
    - Metadata filtering
    - Collection management
    """
    
    def __init__(
        self,
        storage_path: str = "./storage/qdrant_db",
        collection_name: str = "cerebrus_documents", 
        embedding_dim: int = 384,
        recreate_index: bool = False,
        return_embedding: bool = True,
        wait_result_from_api: bool = True,
        hnsw_config: Optional[Dict[str, Any]] = None,
        quantization_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Qdrant vector database.
        
        Args:
            storage_path: Path to store the Qdrant database
            collection_name: Name of the collection to use
            embedding_dim: Dimension of embeddings (should match embedding model)
            recreate_index: Whether to recreate the index on startup
            return_embedding: Whether to return embeddings in search results
            wait_result_from_api: Whether to wait for API responses
            hnsw_config: HNSW index configuration
            quantization_config: Vector quantization configuration
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-haystack is required for QdrantVectorDB. "
                "Install it with: pip install qdrant-haystack"
            )
        
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.recreate_index = recreate_index
        self.return_embedding = return_embedding
        self.wait_result_from_api = wait_result_from_api
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Default HNSW configuration for better performance
        self.hnsw_config = hnsw_config or {
            "m": 16,  # Number of bi-directional links for every new element
            "ef_construct": 200,  # Size of the dynamic candidate list
            "full_scan_threshold": 10000  # Threshold for switching to full scan
        }
        
        # Optional quantization for memory efficiency
        self.quantization_config = quantization_config
        
        self.document_store = None
        self.retriever = None
        
        self._initialize_document_store()
        self._initialize_retriever()
    
    def _initialize_document_store(self):
        """Initialize Qdrant document store with configuration."""
        try:
            logger.info("🚀 Initializing Qdrant document store") # type: ignore
            logger.info(f"   📍 Storage path: {self.storage_path}") # type: ignore
            logger.info(f"   📚 Collection: {self.collection_name}") # type: ignore
            logger.info(f"   📐 Embedding dimension: {self.embedding_dim}") # type: ignore
            
            # Initialize with local storage
            self.document_store = QdrantDocumentStore(
                location=str(self.storage_path),
                index=self.collection_name,
                embedding_dim=self.embedding_dim,
                recreate_index=self.recreate_index,
                return_embedding=self.return_embedding,
                wait_result_from_api=self.wait_result_from_api,
                hnsw_config=self.hnsw_config,
                quantization_config=self.quantization_config
            )
            
            logger.info("✅ Qdrant document store initialized successfully")
            
            # Log current collection stats
            try:
                doc_count = self.document_store.count_documents() # type: ignore
                logger.info(f"📊 Current collection contains {doc_count} documents")
            except Exception as e:
                logger.debug(f"Could not retrieve document count: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Qdrant document store: {e}")
            raise
    
    def _initialize_retriever(self):
        """Initialize Qdrant retriever for search operations."""
        try:
            logger.info("🔍 Initializing Qdrant retriever")
            
            self.retriever = QdrantEmbeddingRetriever(
                document_store=self.document_store
            )
            
            logger.info("✅ Qdrant retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Qdrant retriever: {e}")
            raise
    
    def insert_documents(
        self, 
        documents: List[Document], 
        policy: str = "skip"
    ) -> List[str]:
        """
        Insert documents into the Qdrant collection.
        
        Args:
            documents: List of Haystack Document objects with embeddings
            policy: Write policy - "skip", "overwrite", or "fail" 
            
        Returns:
            List of document IDs that were inserted
        """
        if not documents:
            logger.warning("⚠️ No documents provided for insertion")
            return []
        
        try:
            logger.info(f"💾 Inserting {len(documents)} documents into Qdrant")
            
            # Validate that documents have embeddings
            docs_with_embeddings = []
            docs_without_embeddings = []
            
            for doc in documents:
                if hasattr(doc, 'embedding') and doc.embedding is not None:
                    docs_with_embeddings.append(doc)
                else:
                    docs_without_embeddings.append(doc)
            
            if docs_without_embeddings:
                logger.warning(f"⚠️ {len(docs_without_embeddings)} documents lack embeddings and will be skipped")
                logger.debug(f"Documents without embeddings: {[doc.id if hasattr(doc, 'id') else 'no_id' for doc in docs_without_embeddings[:5]]}")
            
            if not docs_with_embeddings:
                logger.error("❌ No documents with embeddings found")
                return []
            
            # Insert documents
            written_docs = self.document_store.write_documents(
                documents=docs_with_embeddings,
                policy=policy
            )
            
            inserted_ids = [doc.id for doc in written_docs if hasattr(doc, 'id')]
            logger.info(f"✅ Successfully inserted {len(inserted_ids)} documents")
            
            # Log storage stats
            try:
                total_docs = self.document_store.count_documents()
                logger.info(f"📊 Total documents in collection: {total_docs}")
            except Exception as e:
                logger.debug(f"Could not retrieve updated document count: {e}")
            
            return inserted_ids
            
        except Exception as e:
            logger.error(f"❌ Error inserting documents: {e}")
            raise
    
    def insert_embedded_documents(
        self, 
        embedded_docs: List[EmbeddedDocument],
        policy: str = "skip"
    ) -> List[str]:
        """
        Insert embedded documents into the Qdrant collection.
        
        Args:
            embedded_docs: List of EmbeddedDocument objects
            policy: Write policy - "skip", "overwrite", or "fail"
            
        Returns:
            List of document IDs that were inserted
        """
        if not embedded_docs:
            logger.warning("⚠️ No embedded documents provided for insertion")
            return []
        
        try:
            logger.info(f"💾 Converting and inserting {len(embedded_docs)} embedded documents")
            
            # Convert EmbeddedDocument objects to Haystack Documents with embeddings
            documents = []
            for emb_doc in embedded_docs:
                doc = emb_doc.document
                # Ensure the document has the embedding
                doc.embedding = emb_doc.embedding.tolist()
                
                # Add embedding metadata
                if not hasattr(doc, 'meta'):
                    doc.meta = {}
                doc.meta.update({
                    'embedding_model': emb_doc.embedding_model,
                    'embedding_dimension': emb_doc.embedding_dimension
                })
                
                documents.append(doc)
            
            return self.insert_documents(documents, policy=policy)
            
        except Exception as e:
            logger.error(f"❌ Error inserting embedded documents: {e}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        scale_score: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query vector as list of floats
            top_k: Number of results to return
            filters: Optional metadata filters
            scale_score: Whether to scale scores to 0-1 range
            
        Returns:
            List of search results with documents and scores
        """
        try:
            logger.info(f"🔍 Searching for top-{top_k} similar documents")
            if filters:
                logger.debug(f"📋 Applied filters: {filters}")
            
            # Perform vector search using retriever
            results = self.retriever.run(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
                scale_score=scale_score
            )
            
            documents = results.get('documents', [])
            logger.info(f"✅ Found {len(documents)} matching documents")
            
            # Format results for consistency with other vector DBs
            formatted_results = []
            for doc in documents:
                result = {
                    'id': getattr(doc, 'id', None),
                    'score': getattr(doc, 'score', 0.0),
                    'content': doc.content,
                    'metadata': doc.meta,
                    'citation': self._extract_citation_info(doc.meta)
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error during search: {e}")
            raise
    
    def search_with_query_text(
        self,
        query_text: str,
        embedding_generator: EmbeddingGenerator,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        scale_score: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search using query text (will generate embedding internally).
        
        Args:
            query_text: Text query to search for
            embedding_generator: EmbeddingGenerator instance to encode query
            top_k: Number of results to return
            filters: Optional metadata filters
            scale_score: Whether to scale scores to 0-1 range
            
        Returns:
            List of search results with documents and scores
        """
        try:
            logger.info(f"🔍 Searching with text query: '{query_text[:50]}...'")
            
            # Generate query embedding
            query_embedding = embedding_generator.embed_query(query_text)
            query_embedding_list = query_embedding.tolist()
            
            # Perform search
            return self.search(
                query_embedding=query_embedding_list,
                top_k=top_k,
                filters=filters,
                scale_score=scale_score
            )
            
        except Exception as e:
            logger.error(f"❌ Error during text search: {e}")
            raise
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document object if found, None otherwise
        """
        try:
            logger.debug(f"🔍 Retrieving document with ID: {doc_id}")
            
            # Use filter to find document by ID
            results = self.document_store.filter_documents(
                filters={"field": "id", "operator": "==", "value": doc_id}
            )
            
            if results:
                logger.debug(f"✅ Document {doc_id} found")
                return results[0]
            else:
                logger.warning(f"⚠️ Document {doc_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving document {doc_id}: {e}")
            return None
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🗑️ Deleting {len(doc_ids)} documents")
            
            self.document_store.delete_documents(doc_ids)
            
            logger.info(f"✅ Successfully deleted {len(doc_ids)} documents")
            
            # Log updated stats
            try:
                remaining_docs = self.document_store.count_documents()
                logger.info(f"📊 Remaining documents in collection: {remaining_docs}")
            except Exception as e:
                logger.debug(f"Could not retrieve updated document count: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning("🗑️ Clearing entire collection")
            
            # Get all document IDs first
            all_docs = self.document_store.filter_documents()
            doc_ids = [doc.id for doc in all_docs if hasattr(doc, 'id')]
            
            if doc_ids:
                return self.delete_documents(doc_ids)
            else:
                logger.info("📊 Collection is already empty")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error clearing collection: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            logger.debug("📊 Gathering collection statistics")
            
            # Basic document count
            total_docs = self.document_store.count_documents()
            
            # Get sample documents for metadata analysis
            sample_docs = self.document_store.filter_documents()[:100]
            
            # Analyze metadata
            embedding_models = set()
            source_types = set()
            source_files = set()
            
            for doc in sample_docs:
                meta = doc.meta if hasattr(doc, 'meta') else {}
                if 'embedding_model' in meta:
                    embedding_models.add(meta['embedding_model'])
                if 'source_type' in meta:
                    source_types.add(meta['source_type'])
                if 'source_file' in meta:
                    source_files.add(meta['source_file'])
            
            stats = {
                'total_documents': total_docs,
                'collection_name': self.collection_name,
                'embedding_dimension': self.embedding_dim,
                'storage_path': str(self.storage_path),
                'embedding_models': list(embedding_models),
                'source_types': list(source_types),
                'unique_sources': len(source_files),
                'hnsw_config': self.hnsw_config,
                'quantization_enabled': self.quantization_config is not None
            }
            
            logger.info(f"📊 Collection stats: {total_docs} documents, {len(source_files)} unique sources")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error gathering collection statistics: {e}")
            return {'error': str(e)}
    
    def _extract_citation_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract citation information from document metadata."""
        citation = {}
        
        # Basic citation fields
        if 'source_file' in metadata:
            citation['source_file'] = metadata['source_file']
        if 'source_type' in metadata:
            citation['source_type'] = metadata['source_type']
        if 'page_number' in metadata:
            citation['page_number'] = metadata['page_number']
        if 'chunk_index' in metadata:
            citation['chunk_index'] = metadata['chunk_index']
        if 'start_char' in metadata:
            citation['start_char'] = metadata['start_char']
        if 'end_char' in metadata:
            citation['end_char'] = metadata['end_char']
        
        # Check for existing citation object
        if 'citation' in metadata:
            citation.update(metadata['citation'])
        
        return citation
    
    def close(self):
        """Close the database connection."""
        try:
            logger.info("🔒 Closing Qdrant database connection")
            # Qdrant document store doesn't require explicit closing
            # but we can log the closure
            logger.info("✅ Qdrant database connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing connection: {e}")


def create_qdrant_vector_db(
    storage_path: str = "./storage/qdrant_db",
    collection_name: str = "cerebrus_documents",
    embedding_dim: int = 384,
    recreate_index: bool = False
) -> QdrantVectorDB:
    """
    Factory function to create a QdrantVectorDB instance.
    
    Args:
        storage_path: Path to store the database
        collection_name: Name of the collection
        embedding_dim: Embedding dimension
        recreate_index: Whether to recreate the index
        
    Returns:
        Configured QdrantVectorDB instance
    """
    return QdrantVectorDB(
        storage_path=storage_path,
        collection_name=collection_name,
        embedding_dim=embedding_dim,
        recreate_index=recreate_index
    )


# Example usage for testing
if __name__ == "__main__":
    from ..embeddings.embedding_generator import EmbeddingGenerator
    
    print("🧪 Testing Qdrant Vector Database")
    print("=" * 50)
    
    try:
        # Initialize components
        embedding_generator = EmbeddingGenerator(model_name="BAAI/bge-small-en-v1.5")
        vector_db = create_qdrant_vector_db(
            storage_path="./storage/test_qdrant_db",
            collection_name="test_collection",
            recreate_index=True
        )
        
        # Create test documents
        test_texts = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Natural language processing helps computers understand and generate human language.", 
            "Computer vision enables machines to interpret and understand visual information.",
            "Deep learning uses neural networks with multiple layers to model complex patterns."
        ]
        
        documents = embedding_generator.create_documents_from_texts(
            texts=test_texts,
            metadatas=[
                {"topic": "ML", "category": "definition"},
                {"topic": "NLP", "category": "application"},
                {"topic": "CV", "category": "vision"},
                {"topic": "DL", "category": "architecture"}
            ]
        )
        
        # Generate embeddings
        embedded_docs = embedding_generator.embed_documents(documents)
        
        # Insert into vector database
        inserted_ids = vector_db.insert_embedded_documents(embedded_docs)
        print(f"✅ Inserted {len(inserted_ids)} documents")
        
        # Test search
        query = "What is artificial intelligence?"
        results = vector_db.search_with_query_text(
            query_text=query,
            embedding_generator=embedding_generator,
            top_k=3
        )
        
        print(f"\n🔍 Search results for: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"   {i}. Score: {result['score']:.4f}")
            print(f"      Content: {result['content'][:80]}...")
            print(f"      Topic: {result['metadata'].get('topic', 'N/A')}")
        
        # Show collection stats
        stats = vector_db.get_collection_stats()
        print("\n📊 Collection Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n✅ Qdrant vector database test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install qdrant-haystack")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        try:
            vector_db.close()
        except:
            pass

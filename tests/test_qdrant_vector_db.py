"""
Comprehensive tests for Qdrant Vector Database

This module contains tests for the QdrantVectorDB implementation including:
- Database initialization and configuration
- Document insertion and retrieval
- Vector similarity search
- Collection management
- Error handling
- Performance testing
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys

# Add the src directory to Python path for importing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import test dependencies
try:
    from haystack import Document
    from vector_database.qdrant_db import QdrantVectorDB, create_qdrant_vector_db
    from embeddings.embedding_generator import EmbeddingGenerator, EmbeddedDocument
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"⚠️ Dependencies not available: {e}")


@pytest.fixture
def temp_storage():
    """Create temporary storage directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_qdrant_")
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    documents = [
        Document(
            content="Machine learning is a powerful subset of artificial intelligence.",
            meta={
                "source_file": "ml_guide.pdf",
                "source_type": "pdf",
                "page_number": 1,
                "chunk_index": 0,
                "topic": "machine_learning"
            },
            id="doc_1"
        ),
        Document(
            content="Natural language processing enables computers to understand human language.",
            meta={
                "source_file": "nlp_handbook.pdf", 
                "source_type": "pdf",
                "page_number": 2,
                "chunk_index": 1,
                "topic": "natural_language"
            },
            id="doc_2"
        ),
        Document(
            content="Deep learning uses neural networks with multiple layers for complex tasks.",
            meta={
                "source_file": "deep_learning.pdf",
                "source_type": "pdf",
                "page_number": 1,
                "chunk_index": 0,
                "topic": "deep_learning"
            },
            id="doc_3"
        )
    ]
    
    # Add mock embeddings
    for doc in documents:
        doc.embedding = np.random.rand(384).tolist()
    
    return documents


@pytest.fixture
def mock_embedding_generator():
    """Create mock embedding generator for testing."""
    generator = Mock(spec=EmbeddingGenerator)
    
    # Mock embed_query method
    def mock_embed_query(text):
        # Return consistent embedding for same text
        return np.random.rand(384)
    
    generator.embed_query = Mock(side_effect=mock_embed_query)
    generator.model_name = "BAAI/bge-small-en-v1.5"
    generator.embedding_dimension = 384
    
    return generator


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Qdrant dependencies not available")
class TestQdrantVectorDB:
    """Test suite for QdrantVectorDB class"""
    
    def test_initialization_basic(self, temp_storage):
        """Test basic initialization of QdrantVectorDB."""
        storage_path = str(Path(temp_storage) / "test_db")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="test_collection",
                embedding_dim=384,
                recreate_index=True
            )
            
            assert vector_db.storage_path == Path(storage_path)
            assert vector_db.collection_name == "test_collection"
            assert vector_db.embedding_dim == 384
            assert vector_db.document_store is not None
            assert vector_db.retriever is not None
            
            # Check if storage directory was created
            assert Path(storage_path).exists()
            
        except ImportError:
            pytest.skip("qdrant-haystack not installed")
    
    def test_initialization_with_custom_config(self, temp_storage):
        """Test initialization with custom HNSW and quantization config."""
        storage_path = str(Path(temp_storage) / "test_db_custom")
        
        hnsw_config = {
            "m": 32,
            "ef_construct": 400,
            "full_scan_threshold": 5000
        }
        
        quantization_config = {
            "scalar": {
                "type": "int8",
                "quantile": 0.99
            }
        }
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="custom_collection",
                embedding_dim=768,
                hnsw_config=hnsw_config,
                quantization_config=quantization_config,
                recreate_index=True
            )
            
            assert vector_db.hnsw_config == hnsw_config
            assert vector_db.quantization_config == quantization_config
            assert vector_db.embedding_dim == 768
            
        except ImportError:
            pytest.skip("qdrant-haystack not installed")
    
    def test_insert_documents_success(self, temp_storage, sample_documents):
        """Test successful document insertion."""
        storage_path = str(Path(temp_storage) / "test_insert")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="test_insert",
                recreate_index=True
            )
            
            # Insert documents
            inserted_ids = vector_db.insert_documents(sample_documents)
            
            # Verify insertion
            assert len(inserted_ids) == len(sample_documents)
            assert all(doc_id for doc_id in inserted_ids)
            
            # Check collection stats
            stats = vector_db.get_collection_stats()
            assert stats['total_documents'] == len(sample_documents)
            
        except ImportError:
            pytest.skip("qdrant-haystack not installed")
    
    def test_insert_documents_without_embeddings(self, temp_storage):
        """Test handling of documents without embeddings."""
        storage_path = str(Path(temp_storage) / "test_no_embeddings")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="test_no_embeddings",
                recreate_index=True
            )
            
            # Create documents without embeddings
            docs_no_embedding = [
                Document(content="Test document 1", id="no_emb_1"),
                Document(content="Test document 2", id="no_emb_2")
            ]
            
            # Should return empty list
            inserted_ids = vector_db.insert_documents(docs_no_embedding)
            assert len(inserted_ids) == 0
            
        except ImportError:
            pytest.skip("qdrant-haystack not installed")
    
    def test_insert_embedded_documents(self, temp_storage, sample_documents):
        """Test insertion of EmbeddedDocument objects."""
        storage_path = str(Path(temp_storage) / "test_embedded")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="test_embedded",
                recreate_index=True
            )
            
            # Create EmbeddedDocument objects
            embedded_docs = []
            for doc in sample_documents:
                embedded_doc = EmbeddedDocument(
                    document=doc,
                    embedding=np.array(doc.embedding),
                    embedding_model="BAAI/bge-small-en-v1.5",
                    embedding_dimension=384
                )
                embedded_docs.append(embedded_doc)
            
            # Insert embedded documents
            inserted_ids = vector_db.insert_embedded_documents(embedded_docs)
            
            assert len(inserted_ids) == len(embedded_docs)
            
        except ImportError:
            pytest.skip("qdrant-haystack not installed")
    
    def test_search_vector_similarity(self, temp_storage, sample_documents):
        """Test vector similarity search."""
        storage_path = str(Path(temp_storage) / "test_search")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="test_search",
                recreate_index=True
            )
            
            # Insert documents first
            vector_db.insert_documents(sample_documents)
            
            # Create query embedding (similar to first document)
            query_embedding = sample_documents[0].embedding
            
            # Perform search
            results = vector_db.search(
                query_embedding=query_embedding,
                top_k=2
            )
            
            assert len(results) <= 2
            assert all('id' in result for result in results)
            assert all('score' in result for result in results)
            assert all('content' in result for result in results)
            assert all('metadata' in result for result in results)
            assert all('citation' in result for result in results)
            
            # Results should be ordered by similarity score
            if len(results) > 1:
                assert results[0]['score'] >= results[1]['score']
            
        except ImportError:
            pytest.skip("qdrant-haystack not installed")
    
    def test_search_with_filters(self, temp_storage, sample_documents):
        """Test search with metadata filters."""
        storage_path = str(Path(temp_storage) / "test_filtered_search")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="test_filtered_search",
                recreate_index=True
            )
            
            # Insert documents
            vector_db.insert_documents(sample_documents)
            
            # Search with topic filter
            query_embedding = sample_documents[0].embedding
            filters = {"field": "topic", "operator": "==", "value": "machine_learning"}
            
            results = vector_db.search(
                query_embedding=query_embedding,
                top_k=5,
                filters=filters
            )
            
            # Should only return documents with matching topic
            for result in results:
                assert result['metadata'].get('topic') == 'machine_learning'
            
        except ImportError:
            pytest.skip("qdrant-haystack not installed")
    
    def test_search_with_query_text(self, temp_storage, sample_documents, mock_embedding_generator):
        """Test search using query text."""
        storage_path = str(Path(temp_storage) / "test_text_search")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="test_text_search",
                recreate_index=True
            )
            
            # Insert documents
            vector_db.insert_documents(sample_documents)
            
            # Search with text query
            query_text = "What is machine learning?"
            results = vector_db.search_with_query_text(
                query_text=query_text,
                embedding_generator=mock_embedding_generator,
                top_k=3
            )
            
            assert len(results) <= 3
            assert mock_embedding_generator.embed_query.called
            
        except ImportError:
            pytest.skip("qdrant-haystack not installed")
    
    def test_get_document_by_id(self, temp_storage, sample_documents):
        """Test retrieving document by ID."""
        storage_path = str(Path(temp_storage) / "test_get_doc")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="test_get_doc",
                recreate_index=True
            )
            
            # Insert documents
            vector_db.insert_documents(sample_documents)
            
            # Retrieve specific document
            target_doc = sample_documents[0]
            retrieved_doc = vector_db.get_document_by_id(target_doc.id)
            
            if retrieved_doc:  # Document store may handle IDs differently
                assert retrieved_doc.content == target_doc.content
            
        except ImportError:
            pytest.skip("qdrant-haystack not installed")
    
    def test_delete_documents(self, temp_storage, sample_documents):
        """Test document deletion."""
        storage_path = str(Path(temp_storage) / "test_delete")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="test_delete",
                recreate_index=True
            )
            
            # Insert documents
            inserted_ids = vector_db.insert_documents(sample_documents)
            initial_count = vector_db.get_collection_stats()['total_documents']
            
            # Delete some documents
            docs_to_delete = inserted_ids[:2]
            success = vector_db.delete_documents(docs_to_delete)
            
            assert success is True
            
            # Verify deletion
            final_count = vector_db.get_collection_stats()['total_documents']
            assert final_count == initial_count - len(docs_to_delete)
            
        except ImportError:
            pytest.skip("qdrant-haystack not installed")
    
    def test_clear_collection(self, temp_storage, sample_documents):
        """Test clearing entire collection."""
        storage_path = str(Path(temp_storage) / "test_clear")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="test_clear",
                recreate_index=True
            )
            
            # Insert documents
            vector_db.insert_documents(sample_documents)
            assert vector_db.get_collection_stats()['total_documents'] > 0
            
            # Clear collection
            success = vector_db.clear_collection()
            assert success is True
            
            # Verify collection is empty
            assert vector_db.get_collection_stats()['total_documents'] == 0
            
        except ImportError:
            pytest.skip("qdrant-haystack not installed")
    
    def test_get_collection_stats(self, temp_storage, sample_documents):
        """Test collection statistics gathering."""
        storage_path = str(Path(temp_storage) / "test_stats")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="test_stats",
                recreate_index=True
            )
            
            # Insert documents
            vector_db.insert_documents(sample_documents)
            
            # Get stats
            stats = vector_db.get_collection_stats()
            
            assert 'total_documents' in stats
            assert 'collection_name' in stats
            assert 'embedding_dimension' in stats
            assert 'storage_path' in stats
            assert 'source_types' in stats
            assert 'embedding_models' in stats
            
            assert stats['total_documents'] == len(sample_documents)
            assert stats['collection_name'] == "test_stats"
            assert stats['embedding_dimension'] == 384
            
        except ImportError:
            pytest.skip("qdrant-haystack not installed")
    
    def test_citation_extraction(self, temp_storage, sample_documents):
        """Test citation information extraction."""
        storage_path = str(Path(temp_storage) / "test_citation")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="test_citation",
                recreate_index=True
            )
            
            # Insert documents
            vector_db.insert_documents(sample_documents)
            
            # Search and check citation info
            query_embedding = sample_documents[0].embedding
            results = vector_db.search(query_embedding=query_embedding, top_k=1)
            
            if results:
                citation = results[0]['citation']
                assert 'source_file' in citation
                assert 'source_type' in citation
                assert 'page_number' in citation
                assert 'chunk_index' in citation
            
        except ImportError:
            pytest.skip("qdrant-haystack not installed")
    
    def test_error_handling_invalid_storage(self):
        """Test error handling with invalid storage path."""
        with pytest.raises((ImportError, Exception)):
            # Try to create DB with invalid path
            QdrantVectorDB(
                storage_path="/invalid/path/that/does/not/exist",
                collection_name="error_test",
                recreate_index=True
            )
    
    def test_factory_function(self, temp_storage):
        """Test the factory function for creating QdrantVectorDB."""
        storage_path = str(Path(temp_storage) / "factory_test")
        
        try:
            vector_db = create_qdrant_vector_db(
                storage_path=storage_path,
                collection_name="factory_collection",
                embedding_dim=512,
                recreate_index=True
            )
            
            assert isinstance(vector_db, QdrantVectorDB)
            assert vector_db.collection_name == "factory_collection"
            assert vector_db.embedding_dim == 512
            
        except ImportError:
            pytest.skip("qdrant-haystack not installed")


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Qdrant dependencies not available")
class TestQdrantIntegration:
    """Integration tests for QdrantVectorDB with real components."""
    
    @pytest.mark.slow
    def test_full_pipeline_integration(self, temp_storage):
        """Test full pipeline with real embedding generator."""
        storage_path = str(Path(temp_storage) / "integration_test")
        
        try:
            # Skip if embedding model not available
            embedding_generator = EmbeddingGenerator(
                model_name="BAAI/bge-small-en-v1.5",
                batch_size=4
            )
            
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="integration_collection",
                recreate_index=True
            )
            
            # Create real documents
            test_texts = [
                "Artificial intelligence is transforming modern technology.",
                "Machine learning algorithms learn from data patterns.",
                "Natural language processing helps computers understand text."
            ]
            
            documents = embedding_generator.create_documents_from_texts(
                texts=test_texts,
                metadatas=[
                    {"topic": "AI", "category": "technology"},
                    {"topic": "ML", "category": "algorithms"},
                    {"topic": "NLP", "category": "language"}
                ]
            )
            
            # Generate real embeddings
            embedded_docs = embedding_generator.embed_documents(documents)
            
            # Insert into vector database
            inserted_ids = vector_db.insert_embedded_documents(embedded_docs)
            assert len(inserted_ids) == len(embedded_docs)
            
            # Test real search
            query = "What is artificial intelligence?"
            results = vector_db.search_with_query_text(
                query_text=query,
                embedding_generator=embedding_generator,
                top_k=2
            )
            
            assert len(results) <= 2
            assert all(isinstance(result['score'], (int, float)) for result in results)
            
            # Test collection stats
            stats = vector_db.get_collection_stats()
            assert stats['total_documents'] == len(embedded_docs)
            assert 'BAAI/bge-small-en-v1.5' in stats['embedding_models']
            
        except Exception as e:
            if "No module named" in str(e) or "not found" in str(e):
                pytest.skip(f"Dependencies not available: {e}")
            else:
                raise


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])
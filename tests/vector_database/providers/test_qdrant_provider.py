"""
Tests for QdrantProvider

Following AGENTS.md principles:
- Integration tests with real Qdrant operations
- Mock external dependencies when appropriate
- Test error handling and edge cases
- Clear test names describing behavior
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List

from src.vector_database.providers.qdrant_provider import QdrantProvider
from src.vector_database.providers.base_provider import (
    ConnectionError as ProviderConnectionError,
    InsertionError,
    SearchError,
    DeletionError,
    CollectionError
)
from src.vector_database.config.vectordb_config import (
    VectorDatabaseConfig,
    QdrantConfig,
    HNSWConfig
)
from src.vector_database.models.search_result import SearchResults


@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration."""
    return VectorDatabaseConfig(
        provider="qdrant",
        storage_path=tmp_path / "test_qdrant",
        collection_name="test_collection",
        embedding_dim=384
    )


@pytest.fixture
def provider(test_config):
    """Create a QdrantProvider instance."""
    return QdrantProvider(test_config)


class TestQdrantProviderInitialization:
    """Tests for QdrantProvider initialization."""
    
    def test_init_with_valid_config(self, test_config):
        """Test initialization with valid configuration."""
        provider = QdrantProvider(test_config)
        assert provider.config == test_config
        assert provider.document_store is None
        assert provider.retriever is None
    
    def test_init_with_wrong_provider_fails(self, test_config):
        """Test initialization fails with wrong provider type."""
        test_config = VectorDatabaseConfig(
            provider="pinecone",  # Wrong provider
            storage_path=Path("./test"),
            collection_name="test",
            embedding_dim=384
        )
        with pytest.raises(ValueError, match="QdrantProvider requires provider='qdrant'"):
            QdrantProvider(test_config)
    
    @patch('src.vector_database.providers.qdrant_provider.QDRANT_AVAILABLE', False)
    def test_init_without_qdrant_installed_fails(self, test_config):
        """Test initialization fails when qdrant-haystack not installed."""
        with pytest.raises(ImportError, match="qdrant-haystack is required"):
            QdrantProvider(test_config)


class TestQdrantProviderOperations:
    """Tests for QdrantProvider operations with mocked Haystack components."""
    
    def test_initialize_creates_storage_directory(self, provider, test_config):
        """Test initialize creates storage directory."""
        provider.initialize()
        assert test_config.storage_path.exists()
        assert provider.document_store is not None
        assert provider.retriever is not None
    
    def test_initialize_with_connection_error_raises(self, provider):
        """Test initialize raises ConnectionError on failure."""
        with patch('src.vector_database.providers.qdrant_provider.QdrantDocumentStore') as MockStore:
            MockStore.side_effect = Exception("Connection failed")
            with pytest.raises(ProviderConnectionError, match="Qdrant initialization failed"):
                provider.initialize()
    
    def test_insert_documents_without_initialize_fails(self, provider):
        """Test insert_documents fails if not initialized."""
        with pytest.raises(InsertionError, match="Document store not initialized"):
            provider.insert_documents([])
    
    def test_insert_documents_with_invalid_policy_fails(self, provider):
        """Test insert_documents fails with invalid policy."""
        provider.document_store = Mock()
        with pytest.raises(ValueError, match="Invalid policy"):
            provider.insert_documents([], policy="invalid")
    
    def test_insert_documents_returns_empty_for_no_docs(self, provider):
        """Test insert_documents returns empty list when no documents provided (with initialized store)."""
        provider.document_store = Mock()  # Mock initialized state
        result = provider.insert_documents([])
        assert result == []
    
    def test_search_without_initialize_fails(self, provider):
        """Test search fails if not initialized."""
        with pytest.raises(SearchError, match="Retriever not initialized"):
            provider.search([0.1] * 384)
    
    def test_search_with_empty_embedding_fails(self, provider):
        """Test search fails with empty embedding."""
        provider.retriever = Mock()
        with pytest.raises(ValueError, match="Query embedding cannot be empty"):
            provider.search([])
    
    def test_search_with_wrong_dimension_fails(self, provider, test_config):
        """Test search fails when embedding dimension mismatches."""
        provider.retriever = Mock()
        with pytest.raises(ValueError, match="does not match config"):
            provider.search([0.1] * 100)  # Wrong dimension
    
    def test_search_with_invalid_top_k_fails(self, provider):
        """Test search fails with invalid top_k."""
        provider.retriever = Mock()
        with pytest.raises(ValueError, match="top_k must be positive"):
            provider.search([0.1] * 384, top_k=0)
    
    def test_get_document_by_id_without_initialize_fails(self, provider):
        """Test get_document_by_id fails if not initialized."""
        with pytest.raises(SearchError, match="Document store not initialized"):
            provider.get_document_by_id("doc123")
    
    def test_get_document_by_id_with_empty_id_fails(self, provider):
        """Test get_document_by_id fails with empty ID."""
        provider.document_store = Mock()
        with pytest.raises(ValueError, match="Document ID cannot be empty"):
            provider.get_document_by_id("")
    
    def test_delete_documents_without_initialize_fails(self, provider):
        """Test delete_documents fails if not initialized."""
        with pytest.raises(DeletionError, match="Document store not initialized"):
            provider.delete_documents(["doc1"])
    
    def test_delete_documents_returns_zero_for_empty_list(self, provider):
        """Test delete_documents returns 0 for empty list."""
        provider.document_store = Mock()
        result = provider.delete_documents([])
        assert result == 0
    
    def test_count_documents_without_initialize_fails(self, provider):
        """Test count_documents fails if not initialized."""
        with pytest.raises(CollectionError, match="Document store not initialized"):
            provider.count_documents()
    
    def test_get_collection_stats_without_initialize_fails(self, provider):
        """Test get_collection_stats fails if not initialized."""
        with pytest.raises(CollectionError, match="Document store not initialized"):
            provider.get_collection_stats()
    
    def test_get_collection_info_without_initialize_fails(self, provider):
        """Test get_collection_info fails if not initialized."""
        with pytest.raises(CollectionError, match="Document store not initialized"):
            provider.get_collection_info()
    
    def test_collection_exists_without_initialize_fails(self, provider):
        """Test collection_exists fails if not initialized."""
        with pytest.raises(CollectionError, match="Document store not initialized"):
            provider.collection_exists()
    
    def test_clear_collection_without_initialize_fails(self, provider):
        """Test clear_collection fails if not initialized."""
        with pytest.raises(DeletionError, match="Document store not initialized"):
            provider.clear_collection()
    
    def test_close_without_errors(self, provider):
        """Test close completes without errors."""
        provider.document_store = Mock()
        provider.retriever = Mock()
        provider.close()
        assert provider.document_store is None
        assert provider.retriever is None
    
    def test_health_check_without_initialize(self, provider):
        """Test health check returns unhealthy when not initialized."""
        health = provider.health_check()
        assert health['status'] == 'unhealthy'
        assert 'not initialized' in health['details']


class TestQdrantProviderIntegration:
    """Integration tests with actual Qdrant operations."""
    
    @pytest.fixture
    def initialized_provider(self, provider):
        """Initialize provider for integration tests."""
        provider.initialize()
        yield provider
        # Cleanup
        try:
            provider.clear_collection()
        except:
            pass
        provider.close()
    
    def test_initialize_creates_collection(self, initialized_provider):
        """Test initialize creates collection successfully."""
        assert initialized_provider.document_store is not None
        assert initialized_provider.retriever is not None
        assert initialized_provider.collection_exists()
    
    def test_count_documents_on_empty_collection(self, initialized_provider):
        """Test counting documents on empty collection."""
        count = initialized_provider.count_documents()
        assert count == 0
    
    def test_get_collection_info(self, initialized_provider):
        """Test getting collection info."""
        info = initialized_provider.get_collection_info()
        assert info.name == "test_collection"
        assert info.vector_count == 0
        assert info.indexed is True
        assert info.status == "green"
    
    def test_get_collection_stats(self, initialized_provider):
        """Test getting collection statistics."""
        stats = initialized_provider.get_collection_stats()
        assert stats.total_documents == 0
        assert stats.collection_name == "test_collection"
        assert stats.embedding_dimension == 384
        assert stats.hnsw_config['m'] == 16
    
    def test_health_check_returns_healthy(self, initialized_provider):
        """Test health check returns healthy status."""
        health = initialized_provider.health_check()
        assert health['status'] == 'healthy'
        assert 'response_time_ms' in health
        assert health['details']['collection'] == 'test_collection'
    
    def test_get_document_by_id_returns_none_for_missing(self, initialized_provider):
        """Test get_document_by_id returns None for non-existent document."""
        doc = initialized_provider.get_document_by_id("nonexistent123")
        assert doc is None
    
    def test_clear_empty_collection(self, initialized_provider):
        """Test clearing an empty collection."""
        deleted = initialized_provider.clear_collection()
        assert deleted == 0
    
    def test_collection_exists_returns_true(self, initialized_provider):
        """Test collection_exists returns True for existing collection."""
        assert initialized_provider.collection_exists() is True


class TestQdrantProviderWithMockedDocuments:
    """Tests using mocked Haystack documents."""
    
    @pytest.fixture
    def initialized_provider(self, provider):
        """Initialize provider with mocked components."""
        provider.initialize()
        yield provider
        provider.close()
    
    @pytest.fixture
    def mock_document(self):
        """Create a mock Haystack document with embedding."""
        doc = Mock()
        doc.id = "doc123"
        doc.content = "Test content"
        doc.embedding = [0.1] * 384
        doc.meta = {"source_file": "test.txt", "source_type": "txt"}
        doc.score = 0.95
        return doc
    
    def test_insert_documents_with_embeddings(self, initialized_provider, mock_document):
        """Test inserting documents with embeddings."""
        with patch.object(initialized_provider.document_store, 'write_documents') as mock_write:
            mock_write.return_value = [mock_document]
            
            result = initialized_provider.insert_documents([mock_document], policy="skip")
            
            assert len(result) == 1
            assert result[0] == "doc123"
            mock_write.assert_called_once()
    
    def test_insert_documents_without_embeddings_skips(self, initialized_provider):
        """Test documents without embeddings are skipped."""
        doc_no_embedding = Mock()
        doc_no_embedding.id = "doc456"
        doc_no_embedding.embedding = None
        
        result = initialized_provider.insert_documents([doc_no_embedding], policy="skip")
        assert result == []
    
    def test_search_returns_search_results(self, initialized_provider, mock_document):
        """Test search returns SearchResults object."""
        with patch.object(initialized_provider.retriever, 'run') as mock_run:
            mock_run.return_value = {'documents': [mock_document]}
            
            results = initialized_provider.search([0.1] * 384, top_k=5)
            
            assert isinstance(results, SearchResults)
            assert len(results) == 1
            assert results[0].id == "doc123"
            assert results[0].score == 0.95
            assert results[0].content == "Test content"
    
    def test_search_applies_score_threshold(self, initialized_provider, mock_document):
        """Test search applies score threshold correctly."""
        # Create documents with different scores
        doc1 = Mock()
        doc1.id = "doc1"
        doc1.content = "Content 1"
        doc1.score = 0.9
        doc1.meta = {}
        
        doc2 = Mock()
        doc2.id = "doc2"
        doc2.content = "Content 2"
        doc2.score = 0.5
        doc2.meta = {}
        
        with patch.object(initialized_provider.retriever, 'run') as mock_run:
            mock_run.return_value = {'documents': [doc1, doc2]}
            
            results = initialized_provider.search(
                [0.1] * 384,
                top_k=10,
                score_threshold=0.7
            )
            
            assert len(results) == 1
            assert results[0].score == 0.9
    
    def test_delete_documents_calls_document_store(self, initialized_provider):
        """Test delete_documents calls underlying document store."""
        with patch.object(initialized_provider.document_store, 'delete_documents') as mock_delete:
            result = initialized_provider.delete_documents(["doc1", "doc2"])
            
            assert result == 2
            mock_delete.assert_called_once_with(["doc1", "doc2"])
    
    def test_get_document_by_id_returns_document(self, initialized_provider, mock_document):
        """Test get_document_by_id returns document when found."""
        with patch.object(initialized_provider.document_store, 'filter_documents') as mock_filter:
            mock_filter.return_value = [mock_document]
            
            doc = initialized_provider.get_document_by_id("doc123")
            
            assert doc is not None
            assert doc.id == "doc123"


class TestQdrantProviderCitationExtraction:
    """Tests for citation information extraction."""
    
    @pytest.fixture
    def initialized_provider(self, provider):
        """Initialize provider for testing."""
        provider.initialize()
        yield provider
        provider.close()
    
    def test_extract_citation_with_full_metadata(self, initialized_provider):
        """Test extracting citation with complete metadata."""
        metadata = {
            'source_file': 'document.pdf',
            'source_type': 'pdf',
            'page_number': 5,
            'chunk_index': 2,
            'start_char': 100,
            'end_char': 500
        }
        
        citation = initialized_provider._extract_citation_info(metadata)
        
        assert citation.source_file == 'document.pdf'
        assert citation.source_type == 'pdf'
        assert citation.page_number == 5
        assert citation.chunk_index == 2
        assert citation.start_char == 100
        assert citation.end_char == 500
    
    def test_extract_citation_with_partial_metadata(self, initialized_provider):
        """Test extracting citation with partial metadata."""
        metadata = {
            'source_file': 'text.txt',
            'source_type': 'txt'
        }
        
        citation = initialized_provider._extract_citation_info(metadata)
        
        assert citation.source_file == 'text.txt'
        assert citation.source_type == 'txt'
        assert citation.page_number is None
        assert citation.chunk_index is None
    
    def test_extract_citation_with_empty_metadata(self, initialized_provider):
        """Test extracting citation with empty metadata."""
        citation = initialized_provider._extract_citation_info({})
        
        assert citation.source_file is None
        assert citation.source_type is None
        assert citation.page_number is None

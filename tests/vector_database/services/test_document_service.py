"""
Tests for DocumentService

Following AGENTS.md principles:
- Mock provider dependency for unit tests
- Test error handling and edge cases
- Clear test names describing behavior
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List

from src.vector_database.services.document_service import DocumentService
from src.vector_database.providers.base_provider import (
    InsertionError,
    SearchError,
    DeletionError
)


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = Mock()
    provider.__class__.__name__ = "MockProvider"
    return provider


@pytest.fixture
def service(mock_provider):
    """Create DocumentService with mock provider."""
    return DocumentService(mock_provider)


class TestDocumentServiceInitialization:
    """Tests for DocumentService initialization."""
    
    def test_init_with_valid_provider(self, mock_provider):
        """Test initialization with valid provider."""
        service = DocumentService(mock_provider)
        assert service.provider == mock_provider
    
    def test_init_with_none_provider_fails(self):
        """Test initialization fails with None provider."""
        with pytest.raises(ValueError, match="Provider cannot be None"):
            DocumentService(None)


class TestDocumentServiceInsert:
    """Tests for document insertion."""
    
    def test_insert_documents_success(self, service, mock_provider):
        """Test successful document insertion."""
        docs = [Mock(), Mock()]
        mock_provider.insert_documents.return_value = ["id1", "id2"]
        
        result = service.insert_documents(docs, policy="skip")
        
        assert result['count'] == 2
        assert result['inserted_ids'] == ["id1", "id2"]
        assert result['policy'] == "skip"
        mock_provider.insert_documents.assert_called_once_with(docs, "skip")
    
    def test_insert_documents_with_empty_list(self, service, mock_provider):
        """Test inserting empty list returns empty result."""
        result = service.insert_documents([], policy="skip")
        
        assert result['count'] == 0
        assert result['inserted_ids'] == []
        mock_provider.insert_documents.assert_not_called()
    
    def test_insert_documents_with_invalid_type_fails(self, service):
        """Test inserting non-list fails."""
        with pytest.raises(ValueError, match="Documents must be a list"):
            service.insert_documents("not a list")
    
    def test_insert_documents_with_invalid_policy_fails(self, service):
        """Test invalid policy raises error."""
        with pytest.raises(ValueError, match="Invalid policy"):
            service.insert_documents([Mock()], policy="invalid")
    
    def test_insert_documents_with_overwrite_policy(self, service, mock_provider):
        """Test insertion with overwrite policy."""
        docs = [Mock()]
        mock_provider.insert_documents.return_value = ["id1"]
        
        result = service.insert_documents(docs, policy="overwrite")
        
        assert result['policy'] == "overwrite"
        mock_provider.insert_documents.assert_called_once_with(docs, "overwrite")
    
    def test_insert_documents_provider_error_raises(self, service, mock_provider):
        """Test provider error is wrapped in InsertionError."""
        mock_provider.insert_documents.side_effect = Exception("Provider error")
        
        with pytest.raises(InsertionError, match="Failed to insert documents"):
            service.insert_documents([Mock()])


class TestDocumentServiceRetrieval:
    """Tests for document retrieval."""
    
    def test_get_document_by_id_found(self, service, mock_provider):
        """Test retrieving existing document."""
        mock_doc = Mock()
        mock_provider.get_document_by_id.return_value = mock_doc
        
        result = service.get_document_by_id("test_id")
        
        assert result == mock_doc
        mock_provider.get_document_by_id.assert_called_once_with("test_id")
    
    def test_get_document_by_id_not_found(self, service, mock_provider):
        """Test retrieving non-existent document returns None."""
        mock_provider.get_document_by_id.return_value = None
        
        result = service.get_document_by_id("missing_id")
        
        assert result is None
    
    def test_get_document_by_id_with_empty_id_fails(self, service):
        """Test empty document ID raises error."""
        with pytest.raises(ValueError, match="Document ID must be a non-empty string"):
            service.get_document_by_id("")
    
    def test_get_document_by_id_with_non_string_fails(self, service):
        """Test non-string ID raises error."""
        with pytest.raises(ValueError, match="Document ID must be a non-empty string"):
            service.get_document_by_id(123)
    
    def test_get_document_by_id_provider_error_raises(self, service, mock_provider):
        """Test provider error is wrapped in SearchError."""
        mock_provider.get_document_by_id.side_effect = Exception("Provider error")
        
        with pytest.raises(SearchError, match="Failed to retrieve document"):
            service.get_document_by_id("test_id")


class TestDocumentServiceDeletion:
    """Tests for document deletion."""
    
    def test_delete_documents_success(self, service, mock_provider):
        """Test successful document deletion."""
        mock_provider.delete_documents.return_value = 2
        
        result = service.delete_documents(["id1", "id2"])
        
        assert result == 2
        mock_provider.delete_documents.assert_called_once_with(["id1", "id2"])
    
    def test_delete_documents_with_empty_list(self, service, mock_provider):
        """Test deleting empty list returns 0."""
        result = service.delete_documents([])
        
        assert result == 0
        mock_provider.delete_documents.assert_not_called()
    
    def test_delete_documents_with_invalid_type_fails(self, service):
        """Test deleting non-list fails."""
        with pytest.raises(ValueError, match="Document IDs must be a list"):
            service.delete_documents("not a list")
    
    def test_delete_documents_with_empty_id_fails(self, service):
        """Test empty ID in list raises error."""
        with pytest.raises(ValueError, match="All document IDs must be non-empty strings"):
            service.delete_documents(["valid_id", ""])
    
    def test_delete_documents_with_non_string_id_fails(self, service):
        """Test non-string ID in list raises error."""
        with pytest.raises(ValueError, match="All document IDs must be non-empty strings"):
            service.delete_documents(["valid_id", 123])
    
    def test_delete_documents_provider_error_raises(self, service, mock_provider):
        """Test provider error is wrapped in DeletionError."""
        mock_provider.delete_documents.side_effect = Exception("Provider error")
        
        with pytest.raises(DeletionError, match="Failed to delete documents"):
            service.delete_documents(["id1"])


class TestDocumentServiceUtilities:
    """Tests for utility methods."""
    
    def test_document_exists_returns_true(self, service, mock_provider):
        """Test document_exists returns True for existing document."""
        mock_provider.get_document_by_id.return_value = Mock()
        
        assert service.document_exists("test_id") is True
    
    def test_document_exists_returns_false(self, service, mock_provider):
        """Test document_exists returns False for missing document."""
        mock_provider.get_document_by_id.return_value = None
        
        assert service.document_exists("missing_id") is False
    
    def test_document_exists_with_empty_id_fails(self, service):
        """Test document_exists with empty ID raises error."""
        with pytest.raises(ValueError, match="Document ID must be a non-empty string"):
            service.document_exists("")
    
    def test_document_exists_handles_provider_error(self, service, mock_provider):
        """Test document_exists returns False on provider error."""
        mock_provider.get_document_by_id.side_effect = Exception("Error")
        
        assert service.document_exists("test_id") is False
    
    def test_count_documents_without_filters(self, service, mock_provider):
        """Test counting documents without filters."""
        mock_provider.count_documents.return_value = 42
        
        result = service.count_documents()
        
        assert result == 42
        mock_provider.count_documents.assert_called_once_with(None)
    
    def test_count_documents_with_filters(self, service, mock_provider):
        """Test counting documents with filters."""
        filters = {"source": "test"}
        mock_provider.count_documents.return_value = 10
        
        result = service.count_documents(filters)
        
        assert result == 10
        mock_provider.count_documents.assert_called_once_with(filters)
    
    def test_count_documents_with_invalid_filters_fails(self, service):
        """Test count with non-dict filters raises error."""
        with pytest.raises(ValueError, match="Filters must be a dictionary"):
            service.count_documents("not a dict")

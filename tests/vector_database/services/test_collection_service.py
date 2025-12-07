"""
Tests for CollectionService

Following AGENTS.MD principles:
- Mock provider dependency for unit tests
- Test validation and error handling
- Clear test names describing behavior
"""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path

from src.vector_database.services.collection_service import CollectionService
from src.vector_database.providers.base_provider import (
    CollectionError,
    DeletionError
)
from src.vector_database.models.collection_stats import CollectionStats, CollectionInfo


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = Mock()
    provider.__class__.__name__ = "MockProvider"
    return provider


@pytest.fixture
def service(mock_provider):
    """Create CollectionService with mock provider."""
    return CollectionService(mock_provider)


@pytest.fixture
def sample_stats():
    """Create sample collection stats."""
    return CollectionStats(
        total_documents=100,
        collection_name="test_collection",
        embedding_dimension=384,
        storage_path=Path("./test_path"),
        embedding_models=["test-model"],
        source_types=["pdf"],
        unique_sources=10,
        hnsw_config={"m": 16, "ef_construct": 100},
        quantization_enabled=False
    )


@pytest.fixture
def sample_info():
    """Create sample collection info."""
    return CollectionInfo(
        name="test_collection",
        vector_count=100,
        indexed=True,
        status="green"
    )


class TestCollectionServiceInitialization:
    """Tests for CollectionService initialization."""
    
    def test_init_with_valid_provider(self, mock_provider):
        """Test initialization with valid provider."""
        service = CollectionService(mock_provider)
        assert service.provider == mock_provider
    
    def test_init_with_none_provider_fails(self):
        """Test initialization fails with None provider."""
        with pytest.raises(ValueError, match="Provider cannot be None"):
            CollectionService(None)


class TestCollectionServiceStats:
    """Tests for collection statistics."""
    
    def test_get_stats_success(self, service, mock_provider, sample_stats):
        """Test successful stats retrieval."""
        mock_provider.get_collection_stats.return_value = sample_stats
        
        result = service.get_stats()
        
        assert result == sample_stats
        mock_provider.get_collection_stats.assert_called_once()
    
    def test_get_stats_provider_error_raises(self, service, mock_provider):
        """Test provider error is wrapped in CollectionError."""
        mock_provider.get_collection_stats.side_effect = Exception("Provider error")
        
        with pytest.raises(CollectionError, match="Stats retrieval failed"):
            service.get_stats()


class TestCollectionServiceInfo:
    """Tests for collection information."""
    
    def test_get_info_success(self, service, mock_provider, sample_info):
        """Test successful info retrieval."""
        mock_provider.get_collection_info.return_value = sample_info
        
        result = service.get_info()
        
        assert result == sample_info
        mock_provider.get_collection_info.assert_called_once()
    
    def test_get_info_provider_error_raises(self, service, mock_provider):
        """Test provider error is wrapped in CollectionError."""
        mock_provider.get_collection_info.side_effect = Exception("Provider error")
        
        with pytest.raises(CollectionError, match="Info retrieval failed"):
            service.get_info()


class TestCollectionServiceExistence:
    """Tests for collection existence checking."""
    
    def test_collection_exists_returns_true(self, service, mock_provider):
        """Test collection_exists returns True when collection exists."""
        mock_provider.collection_exists.return_value = True
        
        assert service.collection_exists() is True
    
    def test_collection_exists_returns_false(self, service, mock_provider):
        """Test collection_exists returns False when collection missing."""
        mock_provider.collection_exists.return_value = False
        
        assert service.collection_exists() is False
    
    def test_collection_exists_handles_error(self, service, mock_provider):
        """Test collection_exists returns False on error."""
        mock_provider.collection_exists.side_effect = Exception("Error")
        
        assert service.collection_exists() is False


class TestCollectionServiceClear:
    """Tests for collection clearing."""
    
    def test_clear_collection_success(self, service, mock_provider):
        """Test successful collection clear."""
        mock_provider.clear_collection.return_value = 42
        
        result = service.clear_collection(confirm=True)
        
        assert result['deleted_count'] == 42
        assert result['success'] is True
        mock_provider.clear_collection.assert_called_once()
    
    def test_clear_collection_without_confirm_fails(self, service):
        """Test clear_collection without confirm raises error."""
        with pytest.raises(ValueError, match="Must set confirm=True"):
            service.clear_collection(confirm=False)
    
    def test_clear_collection_provider_error_raises(self, service, mock_provider):
        """Test provider error is wrapped in DeletionError."""
        mock_provider.clear_collection.side_effect = Exception("Provider error")
        
        with pytest.raises(DeletionError, match="Collection clear failed"):
            service.clear_collection(confirm=True)


class TestCollectionServiceHealth:
    """Tests for collection health checking."""
    
    def test_health_check_healthy(self, service, mock_provider):
        """Test health check returns healthy status."""
        mock_provider.health_check.return_value = {
            'status': 'healthy',
            'response_time': 10.5
        }
        
        result = service.health_check()
        
        assert result['status'] == 'healthy'
        assert result['response_time'] == 10.5
    
    def test_health_check_unhealthy_on_error(self, service, mock_provider):
        """Test health check returns unhealthy on error."""
        mock_provider.health_check.side_effect = Exception("Connection error")
        
        result = service.health_check()
        
        assert result['status'] == 'unhealthy'
        assert 'error' in result


class TestCollectionServiceSummary:
    """Tests for collection summary."""
    
    def test_get_collection_summary_success(self, service, mock_provider, sample_stats, sample_info):
        """Test successful collection summary generation."""
        mock_provider.get_collection_stats.return_value = sample_stats
        mock_provider.get_collection_info.return_value = sample_info
        mock_provider.health_check.return_value = {
            'status': 'healthy',
            'response_time': 15.0
        }
        
        result = service.get_collection_summary()
        
        assert result['name'] == "test_collection"
        assert result['total_documents'] == 100
        assert result['vector_count'] == 100
        assert result['embedding_dimension'] == 384
        assert result['status'] == "green"
        assert result['health'] == "healthy"
        assert result['response_time_ms'] == 15.0
    
    def test_get_collection_summary_error_raises(self, service, mock_provider):
        """Test summary generation error is wrapped in CollectionError."""
        mock_provider.get_collection_stats.side_effect = Exception("Error")
        
        with pytest.raises(CollectionError, match="Summary generation failed"):
            service.get_collection_summary()


class TestCollectionServiceInitialize:
    """Tests for collection initialization."""
    
    def test_initialize_success(self, service, mock_provider):
        """Test successful collection initialization."""
        service.initialize()
        
        mock_provider.initialize.assert_called_once()
    
    def test_initialize_provider_error_raises(self, service, mock_provider):
        """Test initialization error is wrapped in CollectionError."""
        mock_provider.initialize.side_effect = Exception("Init error")
        
        with pytest.raises(CollectionError, match="Initialization failed"):
            service.initialize()


class TestCollectionServiceClose:
    """Tests for collection closing."""
    
    def test_close_success(self, service, mock_provider):
        """Test successful collection close."""
        service.close()
        
        mock_provider.close.assert_called_once()
    
    def test_close_handles_error(self, service, mock_provider):
        """Test close handles provider error gracefully."""
        mock_provider.close.side_effect = Exception("Close error")
        
        # Should not raise, just log warning
        service.close()
        mock_provider.close.assert_called_once()

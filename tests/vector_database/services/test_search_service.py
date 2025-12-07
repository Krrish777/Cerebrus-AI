"""
Tests for SearchService

Following AGENTS.md principles:
- Mock provider dependency for unit tests
- Test validation and error handling
- Clear test names describing behavior
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List

from src.vector_database.services.search_service import SearchService
from src.vector_database.providers.base_provider import SearchError
from src.vector_database.models.search_result import SearchResults, SearchResult, Citation


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = Mock()
    provider.__class__.__name__ = "MockProvider"
    return provider


@pytest.fixture
def service(mock_provider):
    """Create SearchService with mock provider."""
    return SearchService(mock_provider)


@pytest.fixture
def sample_search_results():
    """Create sample search results."""
    results = [
        SearchResult(
            id="doc1",
            content="Test content 1",
            score=0.95,
            citation=Citation(),
            metadata={"source": "test1"}
        ),
        SearchResult(
            id="doc2",
            content="Test content 2",
            score=0.85,
            citation=Citation(),
            metadata={"source": "test2"}
        )
    ]
    return SearchResults(results=results, total_results=2, query_time_ms=50.0)


class TestSearchServiceInitialization:
    """Tests for SearchService initialization."""
    
    def test_init_with_valid_provider(self, mock_provider):
        """Test initialization with valid provider."""
        service = SearchService(mock_provider)
        assert service.provider == mock_provider
    
    def test_init_with_none_provider_fails(self):
        """Test initialization fails with None provider."""
        with pytest.raises(ValueError, match="Provider cannot be None"):
            SearchService(None)


class TestSearchServiceSearch:
    """Tests for search operations."""
    
    def test_search_success(self, service, mock_provider, sample_search_results):
        """Test successful search."""
        query_embedding = [0.1, 0.2, 0.3]
        mock_provider.search.return_value = sample_search_results
        
        result = service.search(query_embedding, top_k=10)
        
        assert result == sample_search_results
        mock_provider.search.assert_called_once_with(
            query_embedding=query_embedding,
            top_k=10,
            filters=None,
            score_threshold=None
        )
    
    def test_search_with_filters(self, service, mock_provider, sample_search_results):
        """Test search with metadata filters."""
        query_embedding = [0.1, 0.2]
        filters = {"source": "test"}
        mock_provider.search.return_value = sample_search_results
        
        result = service.search(query_embedding, top_k=5, filters=filters)
        
        mock_provider.search.assert_called_once_with(
            query_embedding=query_embedding,
            top_k=5,
            filters=filters,
            score_threshold=None
        )
    
    def test_search_with_score_threshold(self, service, mock_provider, sample_search_results):
        """Test search with score threshold."""
        query_embedding = [0.1, 0.2]
        mock_provider.search.return_value = sample_search_results
        
        result = service.search(query_embedding, score_threshold=0.8)
        
        mock_provider.search.assert_called_once_with(
            query_embedding=query_embedding,
            top_k=10,
            filters=None,
            score_threshold=0.8
        )
    
    def test_search_with_empty_embedding_fails(self, service):
        """Test search with empty embedding raises error."""
        with pytest.raises(ValueError, match="Query embedding must be a non-empty list"):
            service.search([])
    
    def test_search_with_non_list_embedding_fails(self, service):
        """Test search with non-list embedding raises error."""
        with pytest.raises(ValueError, match="Query embedding must be a non-empty list"):
            service.search("not a list")
    
    def test_search_with_invalid_embedding_values_fails(self, service):
        """Test search with non-numeric embedding values raises error."""
        with pytest.raises(ValueError, match="Query embedding must contain only numeric values"):
            service.search([0.1, "invalid", 0.3])
    
    def test_search_with_invalid_top_k_fails(self, service):
        """Test search with invalid top_k raises error."""
        with pytest.raises(ValueError, match="top_k must be an integer between 1 and 1000"):
            service.search([0.1, 0.2], top_k=0)
    
    def test_search_with_top_k_too_large_fails(self, service):
        """Test search with top_k > 1000 raises error."""
        with pytest.raises(ValueError, match="top_k must be an integer between 1 and 1000"):
            service.search([0.1, 0.2], top_k=1001)
    
    def test_search_with_invalid_filters_fails(self, service):
        """Test search with non-dict filters raises error."""
        with pytest.raises(ValueError, match="Filters must be a dictionary"):
            service.search([0.1, 0.2], filters="not a dict")
    
    def test_search_with_invalid_score_threshold_type_fails(self, service):
        """Test search with non-numeric threshold raises error."""
        with pytest.raises(ValueError, match="score_threshold must be numeric"):
            service.search([0.1, 0.2], score_threshold="invalid")
    
    def test_search_with_negative_score_threshold_fails(self, service):
        """Test search with negative threshold raises error."""
        with pytest.raises(ValueError, match="score_threshold must be between 0.0 and 1.0"):
            service.search([0.1, 0.2], score_threshold=-0.1)
    
    def test_search_with_score_threshold_too_high_fails(self, service):
        """Test search with threshold > 1.0 raises error."""
        with pytest.raises(ValueError, match="score_threshold must be between 0.0 and 1.0"):
            service.search([0.1, 0.2], score_threshold=1.5)
    
    def test_search_provider_error_raises(self, service, mock_provider):
        """Test provider error is wrapped in SearchError."""
        mock_provider.search.side_effect = Exception("Provider error")
        
        with pytest.raises(SearchError, match="Search operation failed"):
            service.search([0.1, 0.2])


class TestSearchServiceTextQuery:
    """Tests for text query search."""
    
    def test_search_with_text_query_success(self, service, mock_provider, sample_search_results):
        """Test successful text query search."""
        embedding_fn = Mock(return_value=[0.1, 0.2, 0.3])
        mock_provider.search.return_value = sample_search_results
        
        result = service.search_with_text_query("test query", embedding_fn, top_k=5)
        
        embedding_fn.assert_called_once_with("test query")
        mock_provider.search.assert_called_once()
        assert result == sample_search_results
    
    def test_search_with_empty_text_fails(self, service):
        """Test search with empty text raises error."""
        with pytest.raises(ValueError, match="Query text must be a non-empty string"):
            service.search_with_text_query("", Mock())
    
    def test_search_with_non_string_text_fails(self, service):
        """Test search with non-string text raises error."""
        with pytest.raises(ValueError, match="Query text must be a non-empty string"):
            service.search_with_text_query(123, Mock())
    
    def test_search_with_non_callable_embedding_fn_fails(self, service):
        """Test search with non-callable embedding_fn raises error."""
        with pytest.raises(ValueError, match="embedding_fn must be callable"):
            service.search_with_text_query("test", "not callable")
    
    def test_search_with_embedding_fn_returning_invalid_fails(self, service):
        """Test search with embedding_fn returning invalid value raises error."""
        embedding_fn = Mock(return_value=None)
        
        with pytest.raises(SearchError, match="Text search operation failed"):
            service.search_with_text_query("test", embedding_fn)
    
    def test_search_with_text_query_provider_error_raises(self, service, mock_provider):
        """Test provider error is wrapped in SearchError."""
        embedding_fn = Mock(return_value=[0.1, 0.2])
        mock_provider.search.side_effect = Exception("Provider error")
        
        with pytest.raises(SearchError, match="Text search operation failed"):
            service.search_with_text_query("test", embedding_fn)


class TestSearchServiceResultFiltering:
    """Tests for result filtering."""
    
    def test_filter_results_by_metadata(self, service, sample_search_results):
        """Test filtering results by metadata."""
        metadata_filters = {"source": "test1"}
        
        result = service.filter_results_by_metadata(sample_search_results, metadata_filters)
        
        assert len(result.results) == 1
        assert result.results[0].id == "doc1"
        assert result.total_results == 1
    
    def test_filter_results_with_no_matches(self, service, sample_search_results):
        """Test filtering with no matching results."""
        metadata_filters = {"source": "nonexistent"}
        
        result = service.filter_results_by_metadata(sample_search_results, metadata_filters)
        
        assert len(result.results) == 0
        assert result.total_results == 0
    
    def test_filter_results_with_empty_filters(self, service, sample_search_results):
        """Test filtering with empty filters returns original results."""
        result = service.filter_results_by_metadata(sample_search_results, {})
        
        assert result == sample_search_results
    
    def test_filter_results_with_invalid_results_type_fails(self, service):
        """Test filtering with non-SearchResults raises error."""
        with pytest.raises(ValueError, match="Results must be a SearchResults object"):
            service.filter_results_by_metadata("not results", {})
    
    def test_filter_results_with_invalid_filters_type_fails(self, service, sample_search_results):
        """Test filtering with non-dict filters raises error."""
        with pytest.raises(ValueError, match="metadata_filters must be a dictionary"):
            service.filter_results_by_metadata(sample_search_results, "not a dict")


class TestSearchServiceTopN:
    """Tests for top N results."""
    
    def test_get_top_n_by_score(self, service, sample_search_results):
        """Test getting top N results."""
        result = service.get_top_n_by_score(sample_search_results, n=1)
        
        assert len(result.results) == 1
        assert result.results[0].id == "doc1"
        assert result.total_results == 1
    
    def test_get_top_n_when_n_larger_than_results(self, service, sample_search_results):
        """Test getting top N when N > available results."""
        result = service.get_top_n_by_score(sample_search_results, n=10)
        
        assert result == sample_search_results
    
    def test_get_top_n_with_invalid_results_type_fails(self, service):
        """Test get_top_n with non-SearchResults raises error."""
        with pytest.raises(ValueError, match="Results must be a SearchResults object"):
            service.get_top_n_by_score("not results", n=1)
    
    def test_get_top_n_with_invalid_n_fails(self, service, sample_search_results):
        """Test get_top_n with invalid n raises error."""
        with pytest.raises(ValueError, match="n must be a positive integer"):
            service.get_top_n_by_score(sample_search_results, n=0)
    
    def test_get_top_n_with_non_integer_n_fails(self, service, sample_search_results):
        """Test get_top_n with non-integer n raises error."""
        with pytest.raises(ValueError, match="n must be a positive integer"):
            service.get_top_n_by_score(sample_search_results, n=1.5)

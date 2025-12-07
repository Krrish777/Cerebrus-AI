"""
Tests for Search Result Models

Following AGENTS.md principles:
- Test all validation logic
- Test data conversion methods
- Clear test names describing behavior
"""

import pytest
from src.vector_database.models.search_result import (
    Citation,
    SearchResult,
    SearchResults
)


class TestCitation:
    """Tests for Citation dataclass."""
    
    def test_empty_citation(self):
        """Test citation with no information."""
        citation = Citation()
        assert citation.source_file is None
        assert citation.source_type is None
        assert citation.page_number is None
        assert citation.chunk_index is None
        assert citation.start_char is None
        assert citation.end_char is None
    
    def test_full_citation(self):
        """Test citation with all fields populated."""
        citation = Citation(
            source_file="document.pdf",
            source_type="pdf",
            page_number=5,
            chunk_index=2,
            start_char=100,
            end_char=500
        )
        assert citation.source_file == "document.pdf"
        assert citation.source_type == "pdf"
        assert citation.page_number == 5
        assert citation.chunk_index == 2
        assert citation.start_char == 100
        assert citation.end_char == 500
    
    def test_partial_citation(self):
        """Test citation with some fields populated."""
        citation = Citation(
            source_file="text.txt",
            source_type="txt"
        )
        assert citation.source_file == "text.txt"
        assert citation.source_type == "txt"
        assert citation.page_number is None
    
    def test_to_dict_empty(self):
        """Test conversion to dict excludes None values."""
        citation = Citation()
        result = citation.to_dict()
        assert result == {}
    
    def test_to_dict_full(self):
        """Test conversion to dict with all fields."""
        citation = Citation(
            source_file="doc.pdf",
            source_type="pdf",
            page_number=3,
            chunk_index=1,
            start_char=50,
            end_char=200
        )
        result = citation.to_dict()
        assert result == {
            'source_file': 'doc.pdf',
            'source_type': 'pdf',
            'page_number': 3,
            'chunk_index': 1,
            'start_char': 50,
            'end_char': 200
        }
    
    def test_to_dict_partial(self):
        """Test conversion to dict with some fields."""
        citation = Citation(source_file="file.txt")
        result = citation.to_dict()
        assert result == {'source_file': 'file.txt'}


class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_valid_result(self):
        """Test creating valid search result."""
        citation = Citation(source_file="test.pdf")
        result = SearchResult(
            id="doc123",
            score=0.95,
            content="Test content",
            metadata={"key": "value"},
            citation=citation
        )
        assert result.id == "doc123"
        assert result.score == 0.95
        assert result.content == "Test content"
        assert result.metadata == {"key": "value"}
        assert result.citation == citation
        assert result.embedding is None
    
    def test_result_with_embedding(self):
        """Test search result with embedding vector."""
        citation = Citation()
        result = SearchResult(
            id="doc456",
            score=0.85,
            content="Content",
            metadata={},
            citation=citation,
            embedding=[0.1, 0.2, 0.3]
        )
        assert result.embedding == [0.1, 0.2, 0.3]
    
    def test_negative_score_fails(self):
        """Test validation fails for negative score."""
        citation = Citation()
        with pytest.raises(ValueError, match="Search score must be non-negative"):
            SearchResult(
                id="doc789",
                score=-0.5,
                content="Content",
                metadata={},
                citation=citation
            )
    
    def test_zero_score_valid(self):
        """Test zero score is valid."""
        citation = Citation()
        result = SearchResult(
            id="doc000",
            score=0.0,
            content="Content",
            metadata={},
            citation=citation
        )
        assert result.score == 0.0
    
    def test_to_dict_without_embedding(self):
        """Test conversion to dict without embedding."""
        citation = Citation(source_file="test.txt")
        result = SearchResult(
            id="doc1",
            score=0.9,
            content="Test",
            metadata={"type": "test"},
            citation=citation
        )
        result_dict = result.to_dict()
        assert result_dict == {
            'id': 'doc1',
            'score': 0.9,
            'content': 'Test',
            'metadata': {'type': 'test'},
            'citation': {'source_file': 'test.txt'}
        }
        assert 'embedding' not in result_dict
    
    def test_to_dict_with_embedding(self):
        """Test conversion to dict with embedding."""
        citation = Citation()
        result = SearchResult(
            id="doc2",
            score=0.8,
            content="Test",
            metadata={},
            citation=citation,
            embedding=[0.5, 0.5]
        )
        result_dict = result.to_dict()
        assert 'embedding' in result_dict
        assert result_dict['embedding'] == [0.5, 0.5]


class TestSearchResults:
    """Tests for SearchResults collection dataclass."""
    
    def test_empty_results(self):
        """Test empty search results collection."""
        results = SearchResults(results=[], total_results=0)
        assert len(results) == 0
        assert results.total_results == 0
        assert results.query_time_ms is None
    
    def test_single_result(self):
        """Test search results with single result."""
        citation = Citation()
        result = SearchResult(
            id="doc1",
            score=0.95,
            content="Test",
            metadata={},
            citation=citation
        )
        results = SearchResults(results=[result], total_results=1)
        assert len(results) == 1
        assert results.total_results == 1
    
    def test_multiple_results(self):
        """Test search results with multiple results."""
        citation = Citation()
        result_list = [
            SearchResult(id=f"doc{i}", score=0.9-i*0.1, content=f"Content {i}",
                        metadata={}, citation=citation)
            for i in range(3)
        ]
        results = SearchResults(results=result_list, total_results=3)
        assert len(results) == 3
        assert results.total_results == 3
    
    def test_with_query_time(self):
        """Test search results with query execution time."""
        results = SearchResults(
            results=[],
            total_results=0,
            query_time_ms=15.5
        )
        assert results.query_time_ms == 15.5
    
    def test_negative_total_results_fails(self):
        """Test validation fails for negative total_results."""
        with pytest.raises(ValueError, match="Total results must be non-negative"):
            SearchResults(results=[], total_results=-1)
    
    def test_count_mismatch_fails(self):
        """Test validation fails when result count doesn't match total."""
        citation = Citation()
        result = SearchResult(
            id="doc1",
            score=0.9,
            content="Test",
            metadata={},
            citation=citation
        )
        with pytest.raises(ValueError, match="Result count mismatch"):
            SearchResults(results=[result], total_results=5)
    
    def test_iteration(self):
        """Test iteration over search results."""
        citation = Citation()
        result_list = [
            SearchResult(id=f"doc{i}", score=0.9, content=f"Content {i}",
                        metadata={}, citation=citation)
            for i in range(3)
        ]
        results = SearchResults(results=result_list, total_results=3)
        
        iterated_ids = [r.id for r in results]
        assert iterated_ids == ["doc0", "doc1", "doc2"]
    
    def test_indexing(self):
        """Test indexing into search results."""
        citation = Citation()
        result_list = [
            SearchResult(id=f"doc{i}", score=0.9, content=f"Content {i}",
                        metadata={}, citation=citation)
            for i in range(3)
        ]
        results = SearchResults(results=result_list, total_results=3)
        
        assert results[0].id == "doc0"
        assert results[1].id == "doc1"
        assert results[2].id == "doc2"
    
    def test_to_dict_without_query_time(self):
        """Test conversion to dict without query time."""
        citation = Citation()
        result = SearchResult(
            id="doc1",
            score=0.95,
            content="Test",
            metadata={},
            citation=citation
        )
        results = SearchResults(results=[result], total_results=1)
        results_dict = results.to_dict()
        
        assert 'results' in results_dict
        assert 'total_results' in results_dict
        assert results_dict['total_results'] == 1
        assert len(results_dict['results']) == 1
        assert 'query_time_ms' not in results_dict
    
    def test_to_dict_with_query_time(self):
        """Test conversion to dict with query time."""
        results = SearchResults(
            results=[],
            total_results=0,
            query_time_ms=25.3
        )
        results_dict = results.to_dict()
        assert 'query_time_ms' in results_dict
        assert results_dict['query_time_ms'] == 25.3

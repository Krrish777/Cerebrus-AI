"""
Search Result Models for Vector Database

This module defines dataclasses for search results returned by vector database queries.

Following AGENTS.md principles:
- Immutable data structures (frozen=True)
- Type hints for all fields
- Clear separation of concerns
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass(frozen=True)
class Citation:
    """
    Citation information extracted from document metadata.
    
    Attributes:
        source_file: Original source file name
        source_type: Type of source (pdf, txt, html, etc.)
        page_number: Page number in source document
        chunk_index: Index of the chunk within document
        start_char: Starting character position
        end_char: Ending character position
    """
    source_file: Optional[str] = None
    source_type: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary, excluding None values."""
        return {k: v for k, v in {
            'source_file': self.source_file,
            'source_type': self.source_type,
            'page_number': self.page_number,
            'chunk_index': self.chunk_index,
            'start_char': self.start_char,
            'end_char': self.end_char
        }.items() if v is not None}


@dataclass(frozen=True)
class SearchResult:
    """
    Single search result from vector database query.
    
    Attributes:
        id: Document ID
        score: Similarity score (higher is better)
        content: Document content text
        metadata: Document metadata
        citation: Citation information
        embedding: Optional embedding vector
    """
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    citation: Citation
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate search result values."""
        if self.score < 0.0:
            raise ValueError(f"Search score must be non-negative, got {self.score}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        result = {
            'id': self.id,
            'score': self.score,
            'content': self.content,
            'metadata': self.metadata,
            'citation': self.citation.to_dict()
        }
        if self.embedding is not None:
            result['embedding'] = self.embedding
        return result


@dataclass(frozen=True)
class SearchResults:
    """
    Collection of search results from vector database query.
    
    Attributes:
        results: List of individual search results
        total_results: Total number of results found
        query_time_ms: Query execution time in milliseconds
    """
    results: List[SearchResult]
    total_results: int
    query_time_ms: Optional[float] = None
    
    def __post_init__(self):
        """Validate search results values."""
        if self.total_results < 0:
            raise ValueError(f"Total results must be non-negative, got {self.total_results}")
        if len(self.results) != self.total_results:
            raise ValueError(f"Result count mismatch: {len(self.results)} results but total_results={self.total_results}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search results to dictionary."""
        result = {
            'results': [r.to_dict() for r in self.results],
            'total_results': self.total_results
        }
        if self.query_time_ms is not None:
            result['query_time_ms'] = self.query_time_ms
        return result
    
    def __iter__(self):
        """Allow iteration over results."""
        return iter(self.results)
    
    def __len__(self):
        """Return number of results."""
        return len(self.results)
    
    def __getitem__(self, index):
        """Allow indexing into results."""
        return self.results[index]

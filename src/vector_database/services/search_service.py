"""
Search Service for vector database queries.

Handles semantic search with filters, score thresholds, and result processing.
Following AGENTS.md: single responsibility, dependency injection, fail-fast validation.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path

from src.core.logging import get_logger
from src.vector_database.providers.base_provider import (
    BaseVectorDBProvider,
    SearchError
)
from src.vector_database.models.search_result import SearchResults

logger = get_logger(__name__)


class SearchService:
    """
    Service for performing vector similarity searches.
    
    Responsibilities:
    - Semantic search with embeddings
    - Query validation and preprocessing
    - Result filtering and ranking
    - Search result transformation
    
    Design:
    - Depends on BaseVectorDBProvider interface (loose coupling)
    - Validates query parameters before search (defensibility)
    - Returns structured SearchResults (encapsulation)
    """
    
    def __init__(self, provider: BaseVectorDBProvider):
        """
        Initialize SearchService.
        
        Args:
            provider: Vector database provider implementing BaseVectorDBProvider
            
        Raises:
            ValueError: If provider is None
        """
        if provider is None:
            raise ValueError("Provider cannot be None")
        
        self.provider = provider
        logger.info(f"SearchService initialized with {provider.__class__.__name__}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResults:
        """
        Perform semantic search with query embedding.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return (1-1000)
            filters: Optional metadata filters
            score_threshold: Minimum similarity score (0.0-1.0)
            
        Returns:
            SearchResults object with matching documents
            
        Raises:
            ValueError: If parameters invalid
            SearchError: If search fails
        """
        # Validate query embedding
        if not query_embedding or not isinstance(query_embedding, list):
            raise ValueError("Query embedding must be a non-empty list")
        
        if not all(isinstance(x, (int, float)) for x in query_embedding):
            raise ValueError("Query embedding must contain only numeric values")
        
        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 1000:
            raise ValueError("top_k must be an integer between 1 and 1000")
        
        # Validate filters
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("Filters must be a dictionary")
        
        # Validate score_threshold
        if score_threshold is not None:
            if not isinstance(score_threshold, (int, float)):
                raise ValueError("score_threshold must be numeric")
            if not 0.0 <= score_threshold <= 1.0:
                raise ValueError("score_threshold must be between 0.0 and 1.0")
        
        try:
            logger.info(f"Searching with top_k={top_k}, filters={bool(filters)}, threshold={score_threshold}")
            
            results = self.provider.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
                score_threshold=score_threshold
            )
            
            logger.info(f"Search returned {len(results.results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Search operation failed: {e}") from e
    
    def search_with_text_query(
        self,
        query_text: str,
        embedding_fn: callable,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResults:
        """
        Perform search using text query (converts to embedding first).
        
        Args:
            query_text: Text query to search for
            embedding_fn: Function to convert text to embedding
            top_k: Number of results to return
            filters: Optional metadata filters
            score_threshold: Minimum similarity score
            
        Returns:
            SearchResults object with matching documents
            
        Raises:
            ValueError: If parameters invalid
            SearchError: If search fails
        """
        if not query_text or not isinstance(query_text, str):
            raise ValueError("Query text must be a non-empty string")
        
        if not callable(embedding_fn):
            raise ValueError("embedding_fn must be callable")
        
        try:
            logger.debug(f"Converting query text to embedding: {query_text[:50]}...")
            query_embedding = embedding_fn(query_text)
            
            if not query_embedding or not isinstance(query_embedding, list):
                raise ValueError("embedding_fn must return a non-empty list")
            
            return self.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
                score_threshold=score_threshold
            )
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            raise SearchError(f"Text search operation failed: {e}") from e
    
    def filter_results_by_metadata(
        self,
        results: SearchResults,
        metadata_filters: Dict[str, Any]
    ) -> SearchResults:
        """
        Post-process search results with additional metadata filters.
        
        Args:
            results: Search results to filter
            metadata_filters: Key-value pairs to filter by
            
        Returns:
            Filtered SearchResults object
            
        Raises:
            ValueError: If parameters invalid
        """
        if not isinstance(results, SearchResults):
            raise ValueError("Results must be a SearchResults object")
        
        if not isinstance(metadata_filters, dict):
            raise ValueError("metadata_filters must be a dictionary")
        
        if not metadata_filters:
            logger.debug("No filters provided, returning original results")
            return results
        
        try:
            logger.debug(f"Filtering {len(results.results)} results with {len(metadata_filters)} filters")
            
            filtered = []
            for result in results.results:
                # Check if result has metadata
                if not hasattr(result, 'metadata') or result.metadata is None:
                    continue
                
                # Check all filter conditions
                match = True
                for key, value in metadata_filters.items():
                    if key not in result.metadata or result.metadata[key] != value:
                        match = False
                        break
                
                if match:
                    filtered.append(result)
            
            logger.info(f"Filtered to {len(filtered)} results")
            
            # Create new SearchResults with filtered results
            return SearchResults(
                results=filtered,
                total_results=len(filtered),
                query_time_ms=results.query_time_ms
            )
            
        except Exception as e:
            logger.error(f"Result filtering failed: {e}")
            raise ValueError(f"Failed to filter results: {e}") from e
    
    def get_top_n_by_score(self, results: SearchResults, n: int) -> SearchResults:
        """
        Get top N results by score.
        
        Args:
            results: Search results to filter
            n: Number of top results to return
            
        Returns:
            SearchResults with top N results
            
        Raises:
            ValueError: If parameters invalid
        """
        if not isinstance(results, SearchResults):
            raise ValueError("Results must be a SearchResults object")
        
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        
        if n >= len(results.results):
            logger.debug(f"Requested {n} results but only {len(results.results)} available")
            return results
        
        try:
            # Results should already be sorted by score, just take top n
            top_results = results.results[:n]
            
            return SearchResults(
                results=top_results,
                total_results=len(top_results),
                query_time_ms=results.query_time_ms
            )
            
        except Exception as e:
            logger.error(f"Failed to get top results: {e}")
            raise ValueError(f"Failed to extract top results: {e}") from e

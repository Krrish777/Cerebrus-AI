"""
Search service.
Provides search functionality without generation (retrieval + ranking only).
"""

from typing import Any, Dict, List, Optional

from src.core.logging import get_logger
from src.rag.models import SearchResult
from src.rag.services.retrieval_service import RetrievalService
from src.rag.services.ranking_service import RankingService

logger = get_logger(__name__)


class SearchService:
    """Service for search-only operations (no generation)."""
    
    def __init__(
        self,
        retrieval_service: RetrievalService,
        ranking_service: Optional[RankingService] = None
    ):
        """
        Initialize search service.
        
        Args:
            retrieval_service: Retrieval service instance
            ranking_service: Optional ranking service instance
        """
        self.retrieval_service = retrieval_service
        self.ranking_service = ranking_service
        logger.info("Initialized SearchService")
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        rank_results: bool = True
    ) -> SearchResult:
        """
        Search for documents without generation.
        
        Args:
            query: Search query
            top_k: Maximum documents to return
            filters: Optional metadata filters
            rank_results: Whether to rank results
            
        Returns:
            SearchResult object
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for search")
            return SearchResult(
                query=query,
                documents=[],
                retrieval_count=0,
                ranking_count=0
            )
        
        try:
            logger.info(f"Searching for: {query[:50]}")
            
            # Retrieve documents
            retrieved_docs = self.retrieval_service.retrieve(
                query=query,
                top_k=top_k,
                filters=filters
            )
            
            retrieval_count = len(retrieved_docs)
            
            if not retrieved_docs:
                logger.info("No documents found")
                return SearchResult(
                    query=query,
                    documents=[],
                    retrieval_count=0,
                    ranking_count=0,
                    filters_applied=filters
                )
            
            # Optionally rank documents
            ranked_docs = retrieved_docs
            ranking_count = 0
            
            if rank_results and self.ranking_service:
                ranked_docs = self.ranking_service.rank(
                    query=query,
                    documents=retrieved_docs,
                    top_k=top_k
                )
                ranking_count = len(ranked_docs)
            
            # Convert to document dictionaries
            doc_dicts = self._convert_to_dicts(ranked_docs)
            
            result = SearchResult(
                query=query,
                documents=doc_dicts,
                retrieval_count=retrieval_count,
                ranking_count=ranking_count,
                filters_applied=filters
            )
            
            logger.info(f"Search completed, found {len(doc_dicts)} documents")
            return result
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return SearchResult(
                query=query,
                documents=[],
                retrieval_count=0,
                ranking_count=0,
                filters_applied=filters,
                metadata={"error": str(e)}
            )
    
    def _convert_to_dicts(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert Haystack Documents to dictionaries.
        
        Args:
            documents: List of Haystack Documents
            
        Returns:
            List of document dictionaries
        """
        doc_dicts = []
        
        for doc in documents:
            doc_dict = {
                'content': doc.content or "",
                'metadata': doc.meta or {}
            }
            
            # Add score if available
            if hasattr(doc, 'score') and doc.score is not None:
                doc_dict['score'] = doc.score
            
            # Flatten common metadata fields
            meta = doc.meta or {}
            for field in ['source_file', 'page_number', 'source_type', 'timestamp']:
                if field in meta:
                    doc_dict[field] = meta[field]
            
            doc_dicts.append(doc_dict)
        
        return doc_dicts

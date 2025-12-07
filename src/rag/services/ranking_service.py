"""
Ranking service.
Orchestrates document ranking operations.
"""

from typing import List, Optional

from haystack.dataclasses import Document

from src.core.logging import get_logger
from src.rag.providers.base import RankerProvider

logger = get_logger(__name__)


class RankingService:
    """Service for document ranking operations."""
    
    def __init__(
        self,
        ranker: Optional[RankerProvider] = None,
        enabled: bool = True
    ):
        """
        Initialize ranking service.
        
        Args:
            ranker: Optional ranker provider instance
            enabled: Whether ranking is enabled
        """
        self.ranker = ranker
        self.enabled = enabled and ranker is not None
        
        if self.enabled:
            logger.info("Initialized RankingService with ranker enabled")
        else:
            logger.info("Initialized RankingService with ranking disabled")
    
    def rank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Rank documents by relevance to query.
        
        Args:
            query: Query for ranking
            documents: Documents to rank
            top_k: Maximum ranked documents to return
            score_threshold: Minimum score threshold
            
        Returns:
            List of ranked documents
        """
        if not self.enabled or not documents:
            logger.debug("Ranking skipped (disabled or no documents)")
            return documents
        
        if not self.ranker:
            logger.warning("Ranker not available, returning unranked documents")
            return documents
        
        try:
            logger.debug(f"Ranking {len(documents)} documents for query: {query[:50]}")
            
            result = self.ranker.run(
                query=query,
                documents=documents,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            ranked_docs = result.get("documents", documents)
            
            logger.info(f"Ranked documents, returned top {len(ranked_docs)}")
            return ranked_docs
            
        except Exception as e:
            logger.error(f"Error during ranking: {e}, returning unranked documents")
            return documents
    
    def warm_up(self) -> None:
        """Warm up ranker."""
        if not self.enabled or not self.ranker:
            logger.debug("Ranker warm-up skipped (disabled or not available)")
            return
        
        try:
            self.ranker.warm_up()
            logger.info("RankingService warmed up successfully")
        except Exception as e:
            logger.warning(f"Failed to warm up ranker: {e}")

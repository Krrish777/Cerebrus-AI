"""
FastEmbed ranker provider.
"""

from typing import Any, Dict, List, Optional

from haystack.dataclasses import Document
from haystack_integrations.components.rankers.fastembed import FastembedRanker

from src.core.logging import get_logger

logger = get_logger(__name__)


class FastEmbedRankerProvider:
    """FastEmbed ranker provider implementation."""
    
    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        top_k: int = 8,
        batch_size: int = 32
    ):
        """
        Initialize FastEmbed ranker.
        
        Args:
            model_name: FastEmbed model name
            top_k: Maximum number of ranked documents
            batch_size: Batch size for ranking
        """
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        
        self._ranker = FastembedRanker(
            model_name=model_name,
            top_k=top_k
        )
        
        logger.info(
            f"Initialized FastEmbedRankerProvider with "
            f"model={model_name}, top_k={top_k}"
        )
    
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> Dict[str, List[Document]]:
        """
        Rank documents by relevance.
        
        Args:
            query: Query for ranking
            documents: Documents to rank
            top_k: Override default top_k
            score_threshold: Minimum score threshold
            
        Returns:
            Dictionary with 'documents' key containing ranked documents
        """
        if not documents:
            logger.debug("No documents to rank")
            return {"documents": []}
        
        effective_top_k = top_k or self.top_k
        
        try:
            result = self._ranker.run(
                query=query,
                documents=documents
            )
            
            ranked_docs = result.get("documents", [])
            
            # Apply score threshold if specified
            if score_threshold is not None:
                ranked_docs = [
                    doc for doc in ranked_docs
                    if hasattr(doc, 'score') and doc.score >= score_threshold
                ]
            
            # Apply top_k
            if len(ranked_docs) > effective_top_k:
                ranked_docs = ranked_docs[:effective_top_k]
            
            logger.debug(
                f"Ranked {len(documents)} documents, returned top {len(ranked_docs)} "
                f"for query: {query[:50]}"
            )
            
            return {"documents": ranked_docs}
            
        except Exception as e:
            logger.error(f"Error during FastEmbed ranking: {e}")
            # Return original documents if ranking fails
            return {"documents": documents[:effective_top_k]}
    
    def warm_up(self) -> None:
        """Warm up ranker model."""
        try:
            if hasattr(self._ranker, 'warm_up'):
                self._ranker.warm_up()
                logger.info("FastEmbedRankerProvider warmed up successfully")
        except Exception as e:
            logger.warning(f"Failed to warm up FastEmbed ranker: {e}")

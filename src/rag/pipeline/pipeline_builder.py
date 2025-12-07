"""
Pipeline builder.
Constructs Haystack pipelines for RAG workflows.

Note: Since our providers wrap Haystack components (not extend them),
we access the underlying _retriever/_ranker components for pipeline building.
"""

from haystack import Pipeline

from src.core.logging import get_logger
from src.rag.providers.base import RetrieverProvider, RankerProvider, GeneratorProvider

logger = get_logger(__name__)


class PipelineBuilder:
    """Builder for constructing RAG pipelines."""
    
    def __init__(self):
        """Initialize pipeline builder."""
        logger.info("Initialized PipelineBuilder")
    
    def build_rag_pipeline(
        self,
        retriever: RetrieverProvider,
        ranker: RankerProvider = None,
        generator: GeneratorProvider = None
    ) -> Pipeline:
        """
        Build a complete RAG pipeline.
        
        Args:
            retriever: Retriever provider (wrapper)
            ranker: Optional ranker provider (wrapper)
            generator: Optional generator provider (wrapper)
            
        Returns:
            Configured Haystack Pipeline
            
        Note:
            This extracts the underlying Haystack components from providers.
        """
        pipeline = Pipeline()
        
        # Extract underlying Haystack component
        haystack_retriever = getattr(retriever, '_retriever', retriever)
        pipeline.add_component("retriever", haystack_retriever)
        
        # Add ranker if provided
        if ranker:
            haystack_ranker = getattr(ranker, '_ranker', ranker)
            pipeline.add_component("ranker", haystack_ranker)
            pipeline.connect("retriever.documents", "ranker.documents")
        
        # Note: Generator and prompt builder connections are handled
        # at runtime by PipelineOrchestrator since they need dynamic messages
        
        logger.info(
            f"Built RAG pipeline with retriever"
            f"{', ranker' if ranker else ''}"
            f"{', generator' if generator else ''}"
        )
        
        return pipeline
    
    def build_search_pipeline(
        self,
        retriever: RetrieverProvider,
        ranker: RankerProvider = None
    ) -> Pipeline:
        """
        Build a search-only pipeline (no generation).
        
        Args:
            retriever: Retriever provider (wrapper)
            ranker: Optional ranker provider (wrapper)
            
        Returns:
            Configured Haystack Pipeline
        """
        pipeline = Pipeline()
        
        # Extract underlying Haystack component
        haystack_retriever = getattr(retriever, '_retriever', retriever)
        pipeline.add_component("retriever", haystack_retriever)
        
        # Add ranker if provided
        if ranker:
            haystack_ranker = getattr(ranker, '_ranker', ranker)
            pipeline.add_component("ranker", haystack_ranker)
            pipeline.connect("retriever.documents", "ranker.documents")
        
        logger.info(f"Built search pipeline with retriever{', ranker' if ranker else ''}")
        
        return pipeline

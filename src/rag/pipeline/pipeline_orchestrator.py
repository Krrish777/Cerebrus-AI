"""
Pipeline orchestrator.
Executes RAG pipelines with error handling and result processing.
"""

from typing import Any, Dict, List, Optional

from haystack import Pipeline
from haystack.dataclasses import ChatMessage, Document

from src.core.logging import get_logger
from src.rag.models import RAGResult, SearchResult
from src.rag.services import (
    ContextBuilderService,
    CitationService,
    GenerationService
)

logger = get_logger(__name__)


class PipelineOrchestrator:
    """Orchestrates pipeline execution for RAG operations."""
    
    def __init__(
        self,
        context_builder: ContextBuilderService,
        citation_service: CitationService,
        generation_service: GenerationService
    ):
        """
        Initialize pipeline orchestrator.
        
        Args:
            context_builder: Context building service
            citation_service: Citation service
            generation_service: Generation service
        """
        self.context_builder = context_builder
        self.citation_service = citation_service
        self.generation_service = generation_service
        logger.info("Initialized PipelineOrchestrator")
    
    def execute_rag(
        self,
        pipeline: Pipeline,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> RAGResult:
        """
        Execute RAG pipeline with generation.
        
        Args:
            pipeline: Haystack pipeline
            query: User query
            filters: Optional retrieval filters
            system_prompt: Optional custom system prompt
            
        Returns:
            RAGResult with response and citations
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return RAGResult(
                query=query,
                response="Please provide a valid question.",
                sources_used=[],
                retrieval_count=0,
                ranking_count=0
            )
        
        try:
            logger.info(f"Executing RAG pipeline for query: {query[:50]}")
            
            # Execute retrieval (and ranking if in pipeline)
            retrieval_input = {"retriever": {"query": query}}
            if filters:
                retrieval_input["retriever"]["filters"] = filters
            
            # Check if ranker is in pipeline
            has_ranker = "ranker" in pipeline.graph.nodes
            
            if has_ranker:
                retrieval_input["ranker"] = {"query": query}
            
            result = pipeline.run(retrieval_input)
            
            # Extract documents from result
            if has_ranker:
                documents = result.get("ranker", {}).get("documents", [])
            else:
                documents = result.get("retriever", {}).get("documents", [])
            
            retrieval_count = len(result.get("retriever", {}).get("documents", []))
            ranking_count = len(documents)
            
            if not documents:
                logger.info("No documents retrieved")
                return RAGResult(
                    query=query,
                    response="I couldn't find any relevant information to answer your question.",
                    sources_used=[],
                    retrieval_count=0,
                    ranking_count=0
                )
            
            # Build context
            context = self.context_builder.build_context(documents)
            
            # Create messages
            messages = self._create_messages(query, context, system_prompt)
            
            # Generate response
            response_text = self.generation_service.generate(messages)
            
            # Extract citations
            citations = self.citation_service.extract_citations(documents)
            
            result = RAGResult(
                query=query,
                response=response_text,
                sources_used=citations,
                retrieval_count=retrieval_count,
                ranking_count=ranking_count
            )
            
            logger.info(f"RAG execution completed, {len(citations)} sources used")
            return result
            
        except Exception as e:
            logger.error(f"Error during RAG execution: {e}")
            return RAGResult(
                query=query,
                response=f"I encountered an error while processing your question: {str(e)}",
                sources_used=[],
                retrieval_count=0,
                ranking_count=0
            )
    
    def execute_search(
        self,
        pipeline: Pipeline,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResult:
        """
        Execute search-only pipeline (no generation).
        
        Args:
            pipeline: Haystack pipeline
            query: Search query
            filters: Optional retrieval filters
            
        Returns:
            SearchResult with documents
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return SearchResult(
                query=query,
                documents=[],
                retrieval_count=0,
                ranking_count=0
            )
        
        try:
            logger.info(f"Executing search pipeline for query: {query[:50]}")
            
            # Execute pipeline
            pipeline_input = {"retriever": {"query": query}}
            if filters:
                pipeline_input["retriever"]["filters"] = filters
            
            has_ranker = "ranker" in pipeline.graph.nodes
            if has_ranker:
                pipeline_input["ranker"] = {"query": query}
            
            result = pipeline.run(pipeline_input)
            
            # Extract documents
            if has_ranker:
                documents = result.get("ranker", {}).get("documents", [])
            else:
                documents = result.get("retriever", {}).get("documents", [])
            
            retrieval_count = len(result.get("retriever", {}).get("documents", []))
            ranking_count = len(documents)
            
            # Convert to dict format
            doc_dicts = self._convert_documents(documents)
            
            search_result = SearchResult(
                query=query,
                documents=doc_dicts,
                retrieval_count=retrieval_count,
                ranking_count=ranking_count,
                filters_applied=filters
            )
            
            logger.info(f"Search completed, found {len(doc_dicts)} documents")
            return search_result
            
        except Exception as e:
            logger.error(f"Error during search execution: {e}")
            return SearchResult(
                query=query,
                documents=[],
                retrieval_count=0,
                ranking_count=0,
                filters_applied=filters,
                metadata={"error": str(e)}
            )
    
    def _create_messages(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str]
    ) -> List[ChatMessage]:
        """
        Create chat messages for generation.
        
        Args:
            query: User query
            context: Document context
            system_prompt: Optional system prompt
            
        Returns:
            List of ChatMessage objects
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append(ChatMessage.from_system(system_prompt))
        
        # Add user message with context
        user_message = f"Context: {context}\n\nQuestion: {query}"
        messages.append(ChatMessage.from_user(user_message))
        
        return messages
    
    def _convert_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Convert Haystack Documents to dictionaries."""
        doc_dicts = []
        
        for doc in documents:
            doc_dict = {
                'content': doc.content or "",
                'metadata': doc.meta or {}
            }
            
            if hasattr(doc, 'score') and doc.score is not None:
                doc_dict['score'] = doc.score
            
            # Flatten common metadata
            meta = doc.meta or {}
            for field in ['source_file', 'page_number', 'source_type']:
                if field in meta:
                    doc_dict[field] = meta[field]
            
            doc_dicts.append(doc_dict)
        
        return doc_dicts

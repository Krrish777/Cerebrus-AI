"""
Batch 2 integration tests.
Tests supporting services, pipeline, and factory integration.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from haystack.dataclasses import Document, ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from src.rag.config import RAGConfig, ContextConfig, CitationConfig
from src.rag.services import (
    ContextBuilderService,
    CitationService,
    SearchService,
    RetrievalService,
    RankingService
)
from src.rag.pipeline import PipelineBuilder, PipelineOrchestrator
from src.rag.factories import RAGFactory
from src.rag.utils import PromptTemplateManager, ResultFormatter, validate_query, validate_top_k
from src.rag.providers import InMemoryRetrieverProvider


class TestSupportingServices:
    """Test supporting services."""
    
    def test_context_builder_numbered(self):
        """Test context builder with numbered format."""
        config = ContextConfig(format="numbered", max_documents=3)
        service = ContextBuilderService(config)
        
        docs = [
            Document(content="First doc", meta={"source_file": "file1.pdf"}),
            Document(content="Second doc", meta={"source_file": "file2.pdf"})
        ]
        
        context = service.build_context(docs)
        
        assert "[1]" in context
        assert "[2]" in context
        assert "First doc" in context
        assert "file1.pdf" in context
    
    def test_context_builder_markdown(self):
        """Test context builder with markdown format."""
        config = ContextConfig(format="markdown")
        service = ContextBuilderService(config)
        
        docs = [Document(content="Test content")]
        context = service.build_context(docs)
        
        assert "## Source 1" in context
    
    def test_citation_service(self):
        """Test citation extraction."""
        config = CitationConfig(enabled=True, include_scores=True)
        service = CitationService(config)
        
        docs = [
            Document(
                content="Test content",
                meta={"source_file": "test.pdf", "page_number": 1},
                score=0.95
            )
        ]
        
        citations = service.extract_citations(docs)
        
        assert len(citations) == 1
        assert citations[0]['source_file'] == "test.pdf"
        assert citations[0]['page_number'] == 1
        assert citations[0]['relevance_score'] == 0.95
    
    def test_search_service(self):
        """Test search service integration."""
        # Create mock services
        mock_retriever = Mock()
        mock_retriever.run = Mock(return_value={
            "documents": [Document(content="test", score=0.9)]
        })
        
        retrieval_service = RetrievalService(mock_retriever)
        
        mock_ranker = Mock()
        mock_ranker.run = Mock(return_value={
            "documents": [Document(content="test", score=0.95)]
        })
        
        ranking_service = RankingService(mock_ranker, enabled=True)
        
        search_service = SearchService(retrieval_service, ranking_service)
        
        # Execute search
        result = search_service.search("test query", rank_results=True)
        
        assert result.query == "test query"
        assert len(result.documents) == 1
        assert result.retrieval_count > 0


class TestPipeline:
    """Test pipeline components."""
    
    def test_pipeline_builder_rag(self):
        """Test RAG pipeline building."""
        doc_store = InMemoryDocumentStore()
        retriever = InMemoryRetrieverProvider(doc_store, top_k=5)
        
        builder = PipelineBuilder()
        pipeline = builder.build_rag_pipeline(retriever)
        
        assert "retriever" in pipeline.graph.nodes
    
    def test_pipeline_builder_search(self):
        """Test search pipeline building."""
        doc_store = InMemoryDocumentStore()
        retriever = InMemoryRetrieverProvider(doc_store, top_k=5)
        
        builder = PipelineBuilder()
        pipeline = builder.build_search_pipeline(retriever)
        
        assert "retriever" in pipeline.graph.nodes


class TestFactory:
    """Test RAG factory."""
    
    def test_factory_initialization(self):
        """Test factory initialization with defaults."""
        factory = RAGFactory()
        
        assert factory.config is not None
        assert factory.config.system.name == "Cerebrus RAG System"
    
    def test_factory_create_document_store(self):
        """Test document store creation."""
        config = RAGConfig()
        config.document_store.provider = "inmemory"
        
        factory = RAGFactory(config)
        doc_store = factory.create_document_store()
        
        assert doc_store is not None
        assert isinstance(doc_store, InMemoryDocumentStore)
    
    def test_factory_create_retriever(self):
        """Test retriever creation."""
        config = RAGConfig()
        config.retrieval.provider = "inmemory_bm25"
        
        factory = RAGFactory(config)
        retriever = factory.create_retriever()
        
        assert retriever is not None
    
    def test_factory_create_services(self):
        """Test service creation."""
        config = RAGConfig()
        config.document_store.provider = "inmemory"
        config.retrieval.provider = "inmemory_bm25"
        config.ranking.enabled = False  # Disable to avoid model download
        config.performance.warm_up_on_init = False
        
        factory = RAGFactory(config)
        
        # Mock generator to avoid API key requirement
        mock_generator = Mock()
        factory._generator = mock_generator
        
        services = factory.create_services()
        
        assert 'ingestion' in services
        assert 'retrieval' in services
        assert 'ranking' in services
        assert 'generation' in services
        assert 'context_builder' in services
        assert 'citation' in services
        assert 'search' in services


class TestUtilities:
    """Test utility functions."""
    
    def test_prompt_manager(self):
        """Test prompt template manager."""
        manager = PromptTemplateManager()
        
        system_prompt = manager.load_system_prompt()
        assert system_prompt is not None
        assert len(system_prompt) > 0
    
    def test_result_formatter_text(self):
        """Test result formatting as text."""
        from src.rag.models import RAGResult
        
        result = RAGResult(
            query="test",
            response="answer",
            sources_used=[{"source_file": "test.pdf"}],
            retrieval_count=5,
            ranking_count=3
        )
        
        text = ResultFormatter.format_as_text(result)
        
        assert "test" in text
        assert "answer" in text
        assert "test.pdf" in text
    
    def test_result_formatter_json(self):
        """Test result formatting as JSON."""
        from src.rag.models import RAGResult
        import json
        
        result = RAGResult(
            query="test",
            response="answer",
            sources_used=[],
            retrieval_count=0,
            ranking_count=0
        )
        
        json_str = ResultFormatter.format_as_json(result)
        parsed = json.loads(json_str)
        
        assert parsed['query'] == "test"
        assert parsed['response'] == "answer"
    
    def test_validation_query(self):
        """Test query validation."""
        # Valid query
        query = validate_query("  test query  ")
        assert query == "test query"
        
        # Invalid query
        from src.rag.utils import ValidationError
        with pytest.raises(ValidationError):
            validate_query("")
    
    def test_validation_top_k(self):
        """Test top_k validation."""
        # Valid
        assert validate_top_k(5) == 5
        assert validate_top_k(None) is None
        
        # Exceeds max
        assert validate_top_k(200, max_val=100) == 100
        
        # Invalid
        from src.rag.utils import ValidationError
        with pytest.raises(ValidationError):
            validate_top_k(0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

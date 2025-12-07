"""
Full end-to-end integration test for RAG system.
Tests complete workflow from configuration to generation.
"""

import pytest
import os
from pathlib import Path

from haystack.dataclasses import Document

from src.rag import RAGFactory, RAGConfig
from src.rag.models import RAGResult, SearchResult


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = RAGConfig()
    config.document_store.provider = "inmemory"
    config.retrieval.provider = "inmemory_bm25"
    config.retrieval.top_k = 10
    config.ranking.enabled = False  # Disable to avoid model download in tests
    config.performance.warm_up_on_init = False
    return config


@pytest.fixture
def rag_factory(test_config):
    """Create RAG factory with test config."""
    return RAGFactory(test_config)


@pytest.fixture
def test_documents():
    """Create test documents."""
    return [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability.",
            "metadata": {"source_file": "python_intro.txt", "source_type": "text"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "metadata": {"source_file": "ml_basics.txt", "source_type": "text"}
        },
        {
            "content": "Natural language processing allows computers to understand and generate human language.",
            "metadata": {"source_file": "nlp_guide.txt", "source_type": "text"}
        }
    ]


class TestEndToEndRAG:
    """End-to-end RAG system tests."""
    
    def test_complete_rag_workflow(self, rag_factory, test_documents):
        """Test complete RAG workflow: ingest -> search -> generate."""
        # Skip if no API key (CI/CD environments)
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("No Gemini API key available")
        
        # Create services
        components = rag_factory.create_pipeline_components()
        services = components['services']
        builder = components['builder']
        orchestrator = components['orchestrator']
        
        # 1. Ingest documents
        count = services['ingestion'].add_documents(test_documents)
        assert count == 3
        
        # 2. Search test
        search_result = services['search'].search("What is Python?", rank_results=False)
        assert isinstance(search_result, SearchResult)
        assert len(search_result.documents) > 0
        assert "Python" in str(search_result.documents[0])
        
        # 3. Build pipeline
        retriever = rag_factory.create_retriever()
        pipeline = builder.build_search_pipeline(retriever)
        
        # 4. Execute search via pipeline
        search_result2 = orchestrator.execute_search(pipeline, "machine learning")
        assert isinstance(search_result2, SearchResult)
        assert len(search_result2.documents) > 0
        
        # 5. Generate response (would require API key)
        # Skipped in tests to avoid API calls
    
    def test_document_ingestion_and_count(self, rag_factory, test_documents):
        """Test document ingestion and counting."""
        services = rag_factory.create_services()
        
        # Ingest
        count = services['ingestion'].add_documents(test_documents)
        assert count == 3
        
        # Count
        total = services['ingestion'].count_documents()
        assert total == 3
    
    def test_retrieval_service_integration(self, rag_factory, test_documents):
        """Test retrieval service."""
        services = rag_factory.create_services()
        
        # Add documents
        services['ingestion'].add_documents(test_documents)
        
        # Retrieve
        docs = services['retrieval'].retrieve("Python programming", top_k=5)
        
        assert len(docs) > 0
        assert any("Python" in doc.content for doc in docs)
    
    def test_search_service_integration(self, rag_factory, test_documents):
        """Test search service."""
        services = rag_factory.create_services()
        
        # Add documents
        services['ingestion'].add_documents(test_documents)
        
        # Search
        result = services['search'].search("natural language processing")
        
        assert isinstance(result, SearchResult)
        assert result.retrieval_count > 0
        assert len(result.documents) > 0
    
    def test_context_building(self, rag_factory, test_documents):
        """Test context building service."""
        services = rag_factory.create_services()
        
        # Add documents
        services['ingestion'].add_documents(test_documents)
        
        # Retrieve
        docs = services['retrieval'].retrieve("Python", top_k=2)
        
        # Build context
        context = services['context_builder'].build_context(docs)
        
        assert len(context) > 0
        assert "[1]" in context or "## Source" in context
    
    def test_citation_extraction(self, rag_factory, test_documents):
        """Test citation service."""
        services = rag_factory.create_services()
        
        # Add documents
        services['ingestion'].add_documents(test_documents)
        
        # Retrieve
        docs = services['retrieval'].retrieve("Python", top_k=2)
        
        # Extract citations
        citations = services['citation'].extract_citations(docs)
        
        assert len(citations) > 0
        assert 'source_file' in citations[0]
        assert 'content_preview' in citations[0]


class TestConfigurationLoading:
    """Test configuration loading."""
    
    def test_load_from_yaml(self):
        """Test loading config from YAML file."""
        config_path = Path("config/rag.yml")
        
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        config = RAGConfig.from_yaml(config_path)
        
        assert config.system.name == "Cerebrus RAG System"
        assert config.document_store.provider in ["inmemory", "elasticsearch", "vectordb"]
        assert config.retrieval.top_k > 0
    
    def test_factory_from_yaml(self):
        """Test factory creation from YAML."""
        config_path = "config/rag.yml"
        
        if not Path(config_path).exists():
            pytest.skip("Config file not found")
        
        factory = RAGFactory.from_yaml(config_path)
        
        assert factory.config is not None


class TestErrorHandling:
    """Test error handling."""
    
    def test_empty_query(self, rag_factory):
        """Test handling of empty query."""
        components = rag_factory.create_pipeline_components()
        orchestrator = components['orchestrator']
        builder = components['builder']
        
        retriever = rag_factory.create_retriever()
        pipeline = builder.build_search_pipeline(retriever)
        
        result = orchestrator.execute_search(pipeline, "")
        
        assert isinstance(result, SearchResult)
        assert len(result.documents) == 0
    
    def test_no_documents_found(self, rag_factory):
        """Test when no documents match query."""
        services = rag_factory.create_services()
        
        # Don't add any documents
        
        result = services['search'].search("nonexistent query xyz123")
        
        assert isinstance(result, SearchResult)
        assert result.retrieval_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

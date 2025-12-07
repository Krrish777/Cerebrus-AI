"""
Batch 1 integration tests.
Tests configuration loading, models, providers, and services.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from haystack.dataclasses import Document, ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from src.rag.config import RAGConfig
from src.rag.models import RAGResult, SearchResult
from src.rag.providers import (
    InMemoryRetrieverProvider,
    FastEmbedRankerProvider,
    GeminiGeneratorProvider
)
from src.rag.services import (
    DocumentIngestionService,
    RetrievalService,
    RankingService,
    GenerationService
)


class TestRAGConfig:
    """Test RAG configuration loading."""
    
    def test_load_from_yaml(self):
        """Test loading config from YAML."""
        config_path = Path("config/rag.yml")
        
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        config = RAGConfig.from_yaml(config_path)
        
        assert config is not None
        assert config.system.name == "Cerebrus RAG System"
        assert config.document_store.provider in ["inmemory", "elasticsearch", "vectordb"]
        assert config.retrieval.top_k > 0
        assert config.ranking.top_k > 0
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = RAGConfig()
        
        assert config.system.environment in ["development", "production"]
        assert config.document_store.provider == "inmemory"
        assert config.retrieval.top_k == 20
        assert config.ranking.top_k == 8


class TestRAGModels:
    """Test RAG data models."""
    
    def test_rag_result(self):
        """Test RAGResult model."""
        result = RAGResult(
            query="test query",
            response="test response",
            sources_used=[
                {
                    "source_file": "test.pdf",
                    "page_number": 1,
                    "relevance_score": 0.95
                }
            ],
            retrieval_count=10,
            ranking_count=5
        )
        
        assert result.query == "test query"
        assert result.response == "test response"
        assert len(result.sources_used) == 1
        assert "test.pdf" in result.get_citation_summary()
        assert "Retrieved: 10" in result.get_performance_summary()
    
    def test_search_result(self):
        """Test SearchResult model."""
        result = SearchResult(
            query="test query",
            documents=[
                {
                    "content": "test content",
                    "source_file": "test.pdf",
                    "score": 0.9
                }
            ],
            retrieval_count=5
        )
        
        assert result.query == "test query"
        assert len(result.documents) == 1
        assert len(result) == 1
        assert "test.pdf" in result.get_document_summary()


class TestProviders:
    """Test provider implementations."""
    
    def test_inmemory_retriever_provider(self):
        """Test InMemory retriever provider."""
        doc_store = InMemoryDocumentStore()
        doc_store.write_documents([
            Document(content="Python is a programming language"),
            Document(content="Machine learning uses Python")
        ])
        
        provider = InMemoryRetrieverProvider(doc_store, top_k=2)
        
        result = provider.run(query="Python")
        
        assert "documents" in result
        assert len(result["documents"]) <= 2
    
    def test_fastembed_ranker_provider(self):
        """Test FastEmbed ranker provider."""
        # Create mock ranker to avoid downloading models
        mock_ranker = Mock()
        mock_ranker.run = Mock(return_value={
            "documents": [
                Document(content="test", score=0.9)
            ]
        })
        
        provider = FastEmbedRankerProvider()
        provider._ranker = mock_ranker
        
        docs = [Document(content="test")]
        result = provider.run(query="test", documents=docs)
        
        assert "documents" in result
        mock_ranker.run.assert_called_once()


class TestServices:
    """Test service layer."""
    
    def test_document_ingestion_service(self):
        """Test document ingestion."""
        doc_store = InMemoryDocumentStore()
        service = DocumentIngestionService(doc_store)
        
        docs = [
            {"content": "test 1", "metadata": {"source": "file1"}},
            {"content": "test 2", "metadata": {"source": "file2"}}
        ]
        
        count = service.add_documents(docs)
        
        assert count == 2
        assert service.count_documents() == 2
    
    def test_retrieval_service(self):
        """Test retrieval service."""
        mock_retriever = Mock()
        mock_retriever.run = Mock(return_value={
            "documents": [Document(content="test")]
        })
        
        service = RetrievalService(mock_retriever)
        docs = service.retrieve("test query")
        
        assert len(docs) == 1
        mock_retriever.run.assert_called_once()
    
    def test_ranking_service(self):
        """Test ranking service."""
        mock_ranker = Mock()
        mock_ranker.run = Mock(return_value={
            "documents": [Document(content="test", score=0.9)]
        })
        
        service = RankingService(mock_ranker, enabled=True)
        docs = [Document(content="test")]
        ranked = service.rank("query", docs)
        
        assert len(ranked) == 1
        mock_ranker.run.assert_called_once()
    
    def test_ranking_service_disabled(self):
        """Test ranking service when disabled."""
        service = RankingService(enabled=False)
        docs = [Document(content="test")]
        ranked = service.rank("query", docs)
        
        assert ranked == docs
    
    def test_generation_service(self):
        """Test generation service."""
        mock_generator = Mock()
        mock_reply = Mock()
        mock_reply.text = "Generated response"
        mock_generator.run = Mock(return_value={
            "replies": [mock_reply]
        })
        
        service = GenerationService(mock_generator)
        messages = [ChatMessage.from_user("test")]
        response = service.generate(messages)
        
        assert response == "Generated response"
        mock_generator.run.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

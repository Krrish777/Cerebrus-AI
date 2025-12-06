"""Tests for Haystack embedding provider."""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.embeddings.config.embedding_config import ModelConfig
from src.embeddings.providers.haystack_provider import HaystackEmbeddingProvider


class MockDocument:
    """Mock document for testing."""

    def __init__(self, content: str, meta: dict = None):
        self.content = content
        self.meta = meta or {}
        self.embedding = None


@pytest.fixture
def model_config():
    """Create a ModelConfig for testing."""
    return ModelConfig(
        name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        normalize_embeddings=True,
        prefix=None,
    )


@pytest.fixture
def mock_document_embedder():
    """Mock Haystack SentenceTransformersDocumentEmbedder."""
    embedder = MagicMock()
    embedder.embedding_backend.model.get_sentence_embedding_dimension.return_value = (
        384
    )
    return embedder


@pytest.fixture
def mock_text_embedder():
    """Mock Haystack SentenceTransformersTextEmbedder."""
    embedder = MagicMock()
    embedder.embedding_backend.model.get_sentence_embedding_dimension.return_value = (
        384
    )
    return embedder


class TestHaystackEmbeddingProvider:
    """Tests for HaystackEmbeddingProvider class (integration tests)."""

    def test_initialization(self, model_config):
        """Test provider initialization."""
        provider = HaystackEmbeddingProvider(model_config)

        assert provider._config == model_config
        assert provider._document_embedder is None
        assert provider._text_embedder is None

    def test_warm_up(self, model_config):
        """Test provider warm_up method."""
        provider = HaystackEmbeddingProvider(model_config)
        provider.warm_up()

        assert provider._is_warmed_up
        assert provider._document_embedder is not None
        assert provider._text_embedder is not None

    def test_embed_documents(self, model_config):
        """Test embedding documents (integration test)."""
        from haystack.dataclasses import Document
        
        provider = HaystackEmbeddingProvider(model_config)
        provider.warm_up()

        # Create real Haystack documents
        input_docs = [
            Document(content="First document"),
            Document(content="Second document")
        ]
        result = provider.embed_documents(input_docs)

        assert len(result) == 2
        # Verify embeddings were added
        assert result[0].embedding is not None
        assert result[1].embedding is not None
        assert len(result[0].embedding) > 0
        assert len(result[1].embedding) > 0

    def test_embed_documents_without_warm_up(self, model_config):
        """Test embedding documents raises error if not warmed up."""
        from haystack.dataclasses import Document
        
        provider = HaystackEmbeddingProvider(model_config)

        input_docs = [Document(content="First document")]
        
        with pytest.raises(RuntimeError, match="not warmed up"):
            provider.embed_documents(input_docs)

    def test_embed_query_without_warm_up(self, model_config):
        """Test embedding query raises error if not warmed up."""
        provider = HaystackEmbeddingProvider(model_config)

        with pytest.raises(RuntimeError, match="not warmed up"):
            provider.embed_query("test query")

    def test_get_embedding_dimension(self, model_config):
        """Test getting embedding dimension (integration test)."""
        provider = HaystackEmbeddingProvider(model_config)
        provider.warm_up()

        dimension = provider.get_embedding_dimension()

        # Verify we get a valid dimension
        assert isinstance(dimension, int)
        assert dimension > 0

    def test_get_model_info(self, model_config):
        """Test getting model information (integration test)."""
        provider = HaystackEmbeddingProvider(model_config)
        provider.warm_up()

        model_info = provider.get_model_info()

        assert isinstance(model_info, dict)
        assert model_info["provider"] == "haystack"
        assert model_info["model_name"] == model_config.name
        assert model_info["device"] == model_config.device
        assert "dimension" in model_info

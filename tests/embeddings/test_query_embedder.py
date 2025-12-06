"""Tests for QueryEmbedder service."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.embeddings.services.query_embedder import QueryEmbedder


@pytest.fixture
def mock_provider():
    """Create a mock embedding provider."""
    from src.embeddings.providers.base import EmbeddingProvider
    
    provider = MagicMock(spec=EmbeddingProvider)
    provider.embed_query.return_value = np.random.rand(384).astype(np.float32)
    provider.get_embedding_dimension.return_value = 384
    return provider


@pytest.fixture
def query_embedder(mock_provider):
    """Create a QueryEmbedder instance."""
    return QueryEmbedder(mock_provider)


class TestQueryEmbedder:
    """Tests for QueryEmbedder class."""

    def test_embed_query(self, query_embedder, mock_provider):
        """Test embedding a query string."""
        query = "What is machine learning?"
        result = query_embedder.embed(query)

        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        assert result.dtype == np.float32
        mock_provider.embed_query.assert_called_once_with(query)

    def test_embed_empty_query(self, query_embedder, mock_provider):
        """Test embedding an empty query raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            query_embedder.embed("")

    def test_embed_whitespace_only_query(self, query_embedder, mock_provider):
        """Test embedding whitespace-only query raises ValueError."""
        with pytest.raises(ValueError, match="cannot be only whitespace"):
            query_embedder.embed("   ")

    def test_embed_preserves_provider_output(self, query_embedder, mock_provider):
        """Test that embed returns exactly what provider returns."""
        expected_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_provider.embed_query.return_value = expected_embedding

        result = query_embedder.embed("test query")

        assert np.array_equal(result, expected_embedding)

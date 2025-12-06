"""Tests for DocumentEmbedder service."""

from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.embeddings.services.document_embedder import DocumentEmbedder


class MockDocument:
    """Mock document for testing."""

    def __init__(self, content: str, meta: dict = None):
        self.content = content
        self.meta = meta or {}


@pytest.fixture
def mock_provider():
    """Create a mock embedding provider."""
    from src.embeddings.providers.base import EmbeddingProvider
    
    provider = MagicMock(spec=EmbeddingProvider)
    
    def embed_docs_side_effect(documents):
        # Return documents with embeddings attached
        result = []
        for doc in documents:
            mock_doc = MagicMock()
            mock_doc.content = doc.content
            mock_doc.meta = doc.meta
            mock_doc.embedding = np.random.rand(384).astype(np.float32)
            result.append(mock_doc)
        return result
    
    provider.embed_documents.side_effect = embed_docs_side_effect
    provider.get_embedding_dimension.return_value = 384
    provider.get_model_info.return_value = {"model_name": "test-model", "provider": "mock"}
    return provider


@pytest.fixture
def document_embedder(mock_provider):
    """Create a DocumentEmbedder instance."""
    return DocumentEmbedder(mock_provider)


@pytest.fixture
def sample_documents():
    """Create sample documents."""
    return [
        MockDocument("First document", {"id": 1}),
        MockDocument("Second document", {"id": 2}),
    ]


class TestDocumentEmbedder:
    """Tests for DocumentEmbedder class."""

    def test_embed_documents(
        self, document_embedder, mock_provider, sample_documents
    ):
        """Test embedding multiple documents."""
        result = document_embedder.embed(sample_documents)

        assert len(result) == 2
        mock_provider.embed_documents.assert_called_once_with(sample_documents)

        for embedded_doc in result:
            assert embedded_doc.content in ["First document", "Second document"]
            assert isinstance(embedded_doc.embedding, np.ndarray)
            assert embedded_doc.embedding.shape == (384,)

    def test_embed_single_document(self, document_embedder, mock_provider):
        """Test embedding a single document."""
        mock_provider.embed_documents.return_value = [
            np.random.rand(384).astype(np.float32)
        ]

        doc = MockDocument("Single document", {"id": 1})
        result = document_embedder.embed_single(doc)

        assert result.content == "Single document"
        assert isinstance(result.embedding, np.ndarray)
        assert result.embedding.shape == (384,)
        mock_provider.embed_documents.assert_called_once()

    def test_embed_empty_list(self, document_embedder, mock_provider):
        """Test embedding empty document list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            document_embedder.embed([])

    def test_embed_preserves_metadata(self, document_embedder, mock_provider):
        """Test that embedding preserves document metadata."""
        metadata = {"source": "test", "category": "example", "id": 42}
        doc = MockDocument("Test document", metadata)

        mock_provider.embed_documents.return_value = [
            np.random.rand(384).astype(np.float32)
        ]

        result = document_embedder.embed_single(doc)

        assert result.metadata == metadata

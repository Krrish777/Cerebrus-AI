"""Tests for EmbeddedDocument model."""

import numpy as np
import pytest

from src.embeddings.models.embedded_document import EmbeddedDocument


class MockDocument:
    """Mock Document class for testing."""

    def __init__(self, content, meta=None):
        self.content = content
        self.meta = meta or {}


class TestEmbeddedDocument:
    """Tests for EmbeddedDocument class."""

    def test_valid_embedded_document(self):
        """Test creation of valid EmbeddedDocument."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        metadata = {"source": "test", "category": "example"}
        mock_doc = MockDocument("Test content", metadata)

        doc = EmbeddedDocument(
            document=mock_doc,
            embedding=embedding,
            embedding_model="test-model",
            embedding_dimension=3,
        )

        assert doc.content == "Test content"
        assert np.array_equal(doc.embedding, embedding)
        assert doc.metadata == metadata
        assert doc.embedding_model == "test-model"
        assert doc.embedding_dimension == 3

    def test_embedded_document_none_document(self):
        """Test EmbeddedDocument validation for None document."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        with pytest.raises(ValueError, match="Document cannot be None"):
            EmbeddedDocument(
                document=None,
                embedding=embedding,
                embedding_model="test-model",
                embedding_dimension=3,
            )

    def test_embedded_document_invalid_embedding_type(self):
        """Test EmbeddedDocument validation for invalid embedding type."""
        mock_doc = MockDocument("Test content")

        with pytest.raises(TypeError, match="must be a numpy array"):
            EmbeddedDocument(
                document=mock_doc,
                embedding=[0.1, 0.2, 0.3],  # type: ignore
                embedding_model="test-model",
                embedding_dimension=3,
            )

    def test_embedded_document_none_embedding(self):
        """Test EmbeddedDocument validation for None embedding."""
        mock_doc = MockDocument("Test content")

        with pytest.raises(ValueError, match="Embedding cannot be None"):
            EmbeddedDocument(
                document=mock_doc,
                embedding=None,
                embedding_model="test-model",
                embedding_dimension=3,
            )


    def test_embedded_document_empty_embedding(self):
        """Test EmbeddedDocument validation for empty embedding."""
        embedding = np.array([], dtype=np.float32)
        mock_doc = MockDocument("Test content")

        with pytest.raises(ValueError, match="cannot be empty"):
            EmbeddedDocument(
                document=mock_doc,
                embedding=embedding,
                embedding_model="test-model",
                embedding_dimension=0,
            )

    def test_embedded_document_dimension_mismatch(self):
        """Test EmbeddedDocument validation for dimension mismatch."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_doc = MockDocument("Test content")

        with pytest.raises(ValueError, match="dimension mismatch"):
            EmbeddedDocument(
                document=mock_doc,
                embedding=embedding,
                embedding_model="test-model",
                embedding_dimension=5,  # Wrong dimension
            )

    def test_embedded_document_invalid_dimension(self):
        """Test EmbeddedDocument validation for invalid dimension."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_doc = MockDocument("Test content")

        with pytest.raises(ValueError, match="must be positive"):
            EmbeddedDocument(
                document=mock_doc,
                embedding=embedding,
                embedding_model="test-model",
                embedding_dimension=0,
            )

    def test_embedded_document_empty_model_name(self):
        """Test EmbeddedDocument validation for empty model name."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_doc = MockDocument("Test content")

        with pytest.raises(ValueError, match="model name cannot be empty"):
            EmbeddedDocument(
                document=mock_doc,
                embedding=embedding,
                embedding_model="",
                embedding_dimension=3,
            )

    def test_embedded_document_to_dict(self):
        """Test EmbeddedDocument conversion to dictionary."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        metadata = {"source": "test"}
        mock_doc = MockDocument("Test content", metadata)

        doc = EmbeddedDocument(
            document=mock_doc,
            embedding=embedding,
            embedding_model="test-model",
            embedding_dimension=3,
        )

        doc_dict = doc.to_dict()

        assert isinstance(doc_dict, dict)
        assert doc_dict["content"] == "Test content"
        assert doc_dict["embedding"] == embedding.tolist()
        assert doc_dict["meta"] == metadata
        assert doc_dict["embedding_model"] == "test-model"
        assert doc_dict["embedding_dimension"] == 3

    def test_embedded_document_repr(self):
        """Test EmbeddedDocument string representation."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_doc = MockDocument("Test content")

        doc = EmbeddedDocument(
            document=mock_doc,
            embedding=embedding,
            embedding_model="test-model",
            embedding_dimension=3,
        )

        repr_str = repr(doc)

        assert "EmbeddedDocument" in repr_str
        assert "test-model" in repr_str
        assert "dimension=3" in repr_str
        assert "Test content" in repr_str

    def test_embedded_document_long_repr(self):
        """Test EmbeddedDocument string representation with long content."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        long_content = "x" * 100
        mock_doc = MockDocument(long_content)

        doc = EmbeddedDocument(
            document=mock_doc,
            embedding=embedding,
            embedding_model="test-model",
            embedding_dimension=3,
        )

        repr_str = repr(doc)

        assert "EmbeddedDocument" in repr_str
        assert "..." in repr_str  # Should be truncated


    def test_embedded_document_repr_long_content(self):
        """Test EmbeddedDocument repr truncates long content."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        long_content = "x" * 100
        mock_doc = MockDocument(long_content)

        doc = EmbeddedDocument(
            document=mock_doc,
            embedding=embedding,
            embedding_model="test-model",
            embedding_dimension=3,
        )

        repr_str = repr(doc)

        assert len(repr_str) < len(long_content) + 50
        assert "..." in repr_str

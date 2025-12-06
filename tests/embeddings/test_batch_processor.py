"""Tests for BatchProcessor service."""

from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.embeddings.config.embedding_config import ProcessingConfig
from src.embeddings.services.batch_processor import BatchProcessor


class MockDocument:
    """Mock document for testing."""

    def __init__(self, content: str, meta: dict = None):
        self.content = content
        self.meta = meta or {}


class MockEmbeddedDocument:
    """Mock embedded document for testing."""

    def __init__(self, content: str, embedding: np.ndarray, metadata: dict):
        self.content = content
        self.embedding = embedding
        self.metadata = metadata


@pytest.fixture
def processing_config():
    """Create a ProcessingConfig for testing."""
    return ProcessingConfig(
        batch_size=2,
        max_retries=3,
        timeout=300,
    )


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
    """Create a real DocumentEmbedder with mock provider."""
    from src.embeddings.services.document_embedder import DocumentEmbedder
    return DocumentEmbedder(mock_provider)


@pytest.fixture
def batch_processor(document_embedder, processing_config):
    """Create a BatchProcessor instance."""
    return BatchProcessor(document_embedder, processing_config)


@pytest.fixture
def sample_documents():
    """Create sample documents."""
    return [
        MockDocument(f"Document {i}", {"id": i}) for i in range(5)
    ]


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    def test_process_documents_in_batches(
        self, batch_processor, mock_provider, sample_documents
    ):
        """Test processing documents in batches."""
        result = batch_processor.process_documents_in_batches(sample_documents)

        assert len(result) == 5
        assert mock_provider.embed_documents.call_count == 3  # 2 + 2 + 1

        for embedded_doc in result:
            assert isinstance(embedded_doc.embedding, np.ndarray)
            assert embedded_doc.embedding.shape == (384,)

    def test_process_batches_generator(
        self, document_embedder, processing_config
    ):
        """Test process_batches with manually created batches."""
        # Create batches manually (list of lists)
        batch1 = [MockDocument("Doc 0", {"id": 0}), MockDocument("Doc 1", {"id": 1})]
        batch2 = [MockDocument("Doc 2", {"id": 2}), MockDocument("Doc 3", {"id": 3})]
        batch3 = [MockDocument("Doc 4", {"id": 4})]
        document_batches = [batch1, batch2, batch3]
        
        processor = BatchProcessor(document_embedder, processing_config)
        batches = processor.process_batches(document_batches)

        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_process_empty_list(self, batch_processor, mock_provider):
        """Test processing empty document list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            batch_processor.process_documents_in_batches([])

    def test_process_single_document(
        self, batch_processor, mock_provider
    ):
        """Test processing a single document."""
        doc = MockDocument("Single document", {"id": 1})
        result = batch_processor.process_documents_in_batches([doc])

        assert len(result) == 1
        mock_provider.embed_documents.assert_called_once()

    def test_batch_size_respected(
        self, batch_processor, mock_provider, sample_documents
    ):
        """Test that batch_size configuration is respected."""
        batch_processor.process_documents_in_batches(sample_documents)

        calls = mock_provider.embed_documents.call_args_list
        assert len(calls[0][0][0]) == 2
        assert len(calls[1][0][0]) == 2
        assert len(calls[2][0][0]) == 1

    def test_process_batches_handles_errors(
        self, document_embedder, processing_config
    ):
        """Test that process_batches continues on error."""
        from unittest.mock import MagicMock
        call_count = 0

        def embed_with_error(docs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated error")
            result = []
            for doc in docs:
                mock_doc = MagicMock()
                mock_doc.content = doc.content
                mock_doc.meta = doc.meta
                mock_doc.embedding = np.random.rand(384).astype(np.float32)
                result.append(mock_doc)
            return result

        document_embedder._provider.embed_documents.side_effect = embed_with_error
        processor = BatchProcessor(document_embedder, processing_config)

        # Create batches manually (list of lists)
        batch1 = [MockDocument("Doc 0", {"id": 0}), MockDocument("Doc 1", {"id": 1})]
        batch2 = [MockDocument("Doc 2", {"id": 2}), MockDocument("Doc 3", {"id": 3})]
        batch3 = [MockDocument("Doc 4", {"id": 4})]
        document_batches = [batch1, batch2, batch3]

        batches = processor.process_batches(document_batches)

        # Should return 3 batches: 2 successful, 1 empty (failed)
        assert len(batches) == 3
        assert len(batches[0]) == 2  # First batch successful
        assert len(batches[1]) == 0  # Second batch failed (empty)
        assert len(batches[2]) == 1  # Third batch successful

    def test_different_batch_sizes(self, mock_provider):
        """Test processing with different batch sizes."""
        from src.embeddings.services.document_embedder import DocumentEmbedder
        
        for batch_size in [1, 3, 10]:
            config = ProcessingConfig(
                batch_size=batch_size,
                max_retries=3,
                timeout=300,
            )
            embedder = DocumentEmbedder(mock_provider)
            processor = BatchProcessor(embedder, config)

            documents = [MockDocument(f"Doc {i}", {}) for i in range(7)]
            result = processor.process_documents_in_batches(documents)

            assert len(result) == 7

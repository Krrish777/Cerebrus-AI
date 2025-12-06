"""Test fixtures for embedding tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_texts():
    """Sample text strings for testing."""
    return [
        "Artificial intelligence is transforming the world.",
        "Machine learning enables computers to learn from data.",
        "Natural language processing helps computers understand text.",
    ]


@pytest.fixture
def sample_metadatas():
    """Sample metadata dictionaries for testing."""
    return [
        {"topic": "AI", "category": "overview"},
        {"topic": "ML", "category": "definition"},
        {"topic": "NLP", "category": "application"},
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors for testing."""
    return [
        np.random.rand(384).astype(np.float32),
        np.random.rand(384).astype(np.float32),
        np.random.rand(384).astype(np.float32),
    ]


@pytest.fixture
def mock_document():
    """Mock document object for testing."""

    class MockDocument:
        def __init__(self, content, meta=None):
            self.content = content
            self.meta = meta or {}
            self.embedding = None

    return MockDocument


@pytest.fixture
def mock_documents(mock_document, sample_texts, sample_metadatas):
    """List of mock documents for testing."""
    return [
        mock_document(text, meta)
        for text, meta in zip(sample_texts, sample_metadatas)
    ]

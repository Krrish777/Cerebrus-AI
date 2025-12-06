"""Tests for document utility functions."""

from typing import List, Dict, Any

import pytest

from src.embeddings.utils.document_utils import (
    create_documents_from_texts,
    extract_metadata_from_documents,
    extract_texts_from_documents,
    validate_documents,
)


class MockDocument:
    """Mock document for testing."""

    def __init__(self, content: str, meta: Dict[str, Any] = None):
        self.content = content
        self.meta = meta or {}


@pytest.fixture
def sample_texts() -> List[str]:
    """Sample text strings."""
    return [
        "First document content",
        "Second document content",
        "Third document content",
    ]


@pytest.fixture
def sample_metadatas() -> List[Dict[str, Any]]:
    """Sample metadata dictionaries."""
    return [
        {"id": 1, "category": "A"},
        {"id": 2, "category": "B"},
        {"id": 3, "category": "C"},
    ]


@pytest.fixture
def mock_documents() -> List[MockDocument]:
    """List of mock documents."""
    return [
        MockDocument("First document", {"id": 1}),
        MockDocument("Second document", {"id": 2}),
        MockDocument("Third document", {"id": 3}),
    ]


class TestCreateDocumentsFromTexts:
    """Tests for create_documents_from_texts function."""

    def test_create_documents_from_texts_with_metadata(
        self, sample_texts, sample_metadatas
    ):
        """Test creating documents from texts with metadata."""
        documents = create_documents_from_texts(sample_texts, sample_metadatas)

        assert len(documents) == 3
        for i, (doc, text, meta) in enumerate(zip(documents, sample_texts, sample_metadatas)):
            assert doc.content == text
            # Haystack adds doc_index automatically
            expected_meta = {**meta, "doc_index": i}
            assert doc.meta == expected_meta

    def test_create_documents_from_texts_without_metadata(self, sample_texts):
        """Test creating documents from texts without metadata."""
        documents = create_documents_from_texts(sample_texts)

        assert len(documents) == 3
        for i, (doc, text) in enumerate(zip(documents, sample_texts)):
            assert doc.content == text
            # Haystack adds doc_index automatically
            assert doc.meta == {"doc_index": i}

    def test_create_documents_from_texts_mismatched_lengths(
        self, sample_texts, sample_metadatas
    ):
        """Test error when texts and metadatas have mismatched lengths."""
        with pytest.raises(ValueError, match="must match"):
            create_documents_from_texts(sample_texts, sample_metadatas[:2])

    def test_create_documents_from_texts_empty_list(self):
        """Test creating documents from empty text list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            create_documents_from_texts([])


class TestValidateDocuments:
    """Tests for validate_documents function."""

    def test_validate_documents_valid(self, mock_documents):
        """Test validation of valid documents."""
        validate_documents(mock_documents)  # Should not raise

    def test_validate_documents_empty_list(self):
        """Test validation of empty document list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_documents([])

    def test_validate_documents_missing_content_attribute(self):
        """Test validation fails for documents missing content attribute."""

        class BadDocument:
            def __init__(self):
                self.meta = {}

        with pytest.raises(TypeError, match="has no 'content' attribute"):
            validate_documents([BadDocument()])

    def test_validate_documents_missing_meta_attribute(self):
        """Test validation warns for documents missing meta attribute."""

        class BadDocument:
            def __init__(self):
                self.content = "test"

        # Missing meta attribute only generates a warning, doesn't raise
        result = validate_documents([BadDocument()])
        assert result is True

    def test_validate_documents_empty_content(self):
        """Test validation warns for documents with empty content."""
        bad_doc = MockDocument("", {"id": 1})

        # Empty content only generates a warning, doesn't raise
        result = validate_documents([bad_doc])
        assert result is True

    def test_validate_documents_whitespace_only_content(self):
        """Test validation warns for documents with whitespace-only content."""
        bad_doc = MockDocument("   ", {"id": 1})

        # Whitespace-only content only generates a warning, doesn't raise
        result = validate_documents([bad_doc])
        assert result is True


class TestExtractTextsFromDocuments:
    """Tests for extract_texts_from_documents function."""

    def test_extract_texts_from_documents(self, mock_documents):
        """Test extracting texts from documents."""
        texts = extract_texts_from_documents(mock_documents)

        assert len(texts) == 3
        assert texts[0] == "First document"
        assert texts[1] == "Second document"
        assert texts[2] == "Third document"

    def test_extract_texts_from_empty_list(self):
        """Test extracting texts from empty document list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            extract_texts_from_documents([])


class TestExtractMetadataFromDocuments:
    """Tests for extract_metadata_from_documents function."""

    def test_extract_metadata_from_documents(self, mock_documents):
        """Test extracting metadata from documents."""
        metadatas = extract_metadata_from_documents(mock_documents)

        assert len(metadatas) == 3
        assert metadatas[0] == {"id": 1}
        assert metadatas[1] == {"id": 2}
        assert metadatas[2] == {"id": 3}

    def test_extract_metadata_from_empty_list(self):
        """Test extracting metadata from empty document list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            extract_metadata_from_documents([])

    def test_extract_metadata_preserves_empty_dicts(self):
        """Test extraction preserves empty metadata dictionaries."""
        docs = [MockDocument("content", {})]
        metadatas = extract_metadata_from_documents(docs)

        assert len(metadatas) == 1
        assert metadatas[0] == {}

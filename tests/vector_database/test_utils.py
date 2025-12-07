"""
Tests for vector database utility functions.

Testing embeddings, validation, and helper functions.
"""

import pytest
from pathlib import Path
import tempfile
import json

from src.vector_database.utils import (
    validate_embedding,
    normalize_embedding,
    cosine_similarity,
    generate_document_id,
    chunk_list,
    calculate_storage_size,
    format_bytes,
    merge_filters,
    validate_collection_name
)


class TestEmbeddingValidation:
    """Test embedding validation."""
    
    def test_validate_embedding_success(self):
        """Test valid embedding passes validation."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        assert validate_embedding(embedding)
    
    def test_validate_embedding_with_dimension(self):
        """Test validation with expected dimension."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        assert validate_embedding(embedding, expected_dim=4)
    
    def test_validate_embedding_empty_fails(self):
        """Test empty embedding fails."""
        with pytest.raises(ValueError, match="must be a non-empty list"):
            validate_embedding([])
    
    def test_validate_embedding_non_list_fails(self):
        """Test non-list embedding fails."""
        with pytest.raises(ValueError, match="must be a non-empty list"):
            validate_embedding("not a list")
    
    def test_validate_embedding_non_numeric_fails(self):
        """Test embedding with non-numeric values fails."""
        with pytest.raises(ValueError, match="must contain only numeric values"):
            validate_embedding([0.1, "text", 0.3])
    
    def test_validate_embedding_wrong_dimension_fails(self):
        """Test embedding with wrong dimension fails."""
        embedding = [0.1, 0.2, 0.3]
        with pytest.raises(ValueError, match="dimension mismatch"):
            validate_embedding(embedding, expected_dim=4)


class TestEmbeddingNormalization:
    """Test embedding normalization."""
    
    def test_normalize_embedding_success(self):
        """Test embedding normalization."""
        embedding = [3.0, 4.0]
        normalized = normalize_embedding(embedding)
        
        # Check unit length (L2 norm = 1)
        norm = sum(x * x for x in normalized) ** 0.5
        assert abs(norm - 1.0) < 1e-6
        
        # Check values
        assert abs(normalized[0] - 0.6) < 1e-6
        assert abs(normalized[1] - 0.8) < 1e-6
    
    def test_normalize_embedding_already_normalized(self):
        """Test normalizing already normalized embedding."""
        embedding = [0.6, 0.8]
        normalized = normalize_embedding(embedding)
        
        # Should be very close to input
        assert abs(normalized[0] - embedding[0]) < 1e-6
        assert abs(normalized[1] - embedding[1]) < 1e-6
    
    def test_normalize_embedding_zero_vector_fails(self):
        """Test normalizing zero vector fails."""
        with pytest.raises(ValueError, match="Cannot normalize zero vector"):
            normalize_embedding([0.0, 0.0, 0.0])
    
    def test_normalize_embedding_invalid_fails(self):
        """Test normalizing invalid embedding fails."""
        with pytest.raises(ValueError):
            normalize_embedding([])


class TestCosineSimilarity:
    """Test cosine similarity calculation."""
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        embedding = [0.1, 0.2, 0.3]
        similarity = cosine_similarity(embedding, embedding)
        
        assert abs(similarity - 1.0) < 1e-6
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(embedding1, embedding2)
        
        assert abs(similarity - 0.0) < 1e-6
    
    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        embedding1 = [1.0, 0.0]
        embedding2 = [-1.0, 0.0]
        similarity = cosine_similarity(embedding1, embedding2)
        
        # Clamped to [0, 1], so -1 becomes 0
        assert abs(similarity - 0.0) < 1e-6
    
    def test_cosine_similarity_different_dimensions_fails(self):
        """Test cosine similarity with different dimensions fails."""
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.1, 0.2]
        
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity(embedding1, embedding2)
    
    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [0.0, 0.0, 0.0]
        similarity = cosine_similarity(embedding1, embedding2)
        
        assert similarity == 0.0


class TestDocumentIdGeneration:
    """Test document ID generation."""
    
    def test_generate_document_id_content_only(self):
        """Test generating ID from content only."""
        content = "This is a test document."
        doc_id = generate_document_id(content)
        
        assert isinstance(doc_id, str)
        assert len(doc_id) == 64  # SHA256 hex length
    
    def test_generate_document_id_with_metadata(self):
        """Test generating ID from content and metadata."""
        content = "Test content"
        metadata = {"source": "test.txt", "page": 1}
        doc_id = generate_document_id(content, metadata)
        
        assert isinstance(doc_id, str)
        assert len(doc_id) == 64
    
    def test_generate_document_id_deterministic(self):
        """Test ID generation is deterministic."""
        content = "Same content"
        metadata = {"key": "value"}
        
        doc_id1 = generate_document_id(content, metadata)
        doc_id2 = generate_document_id(content, metadata)
        
        assert doc_id1 == doc_id2
    
    def test_generate_document_id_different_for_different_content(self):
        """Test different content produces different IDs."""
        doc_id1 = generate_document_id("Content 1")
        doc_id2 = generate_document_id("Content 2")
        
        assert doc_id1 != doc_id2
    
    def test_generate_document_id_metadata_order_independent(self):
        """Test metadata order doesn't affect ID."""
        content = "Test"
        metadata1 = {"a": 1, "b": 2}
        metadata2 = {"b": 2, "a": 1}
        
        doc_id1 = generate_document_id(content, metadata1)
        doc_id2 = generate_document_id(content, metadata2)
        
        assert doc_id1 == doc_id2
    
    def test_generate_document_id_empty_content_fails(self):
        """Test empty content fails."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            generate_document_id("")


class TestChunkList:
    """Test list chunking."""
    
    def test_chunk_list_even_split(self):
        """Test chunking with even split."""
        items = [1, 2, 3, 4, 5, 6]
        chunks = chunk_list(items, chunk_size=2)
        
        assert chunks == [[1, 2], [3, 4], [5, 6]]
    
    def test_chunk_list_uneven_split(self):
        """Test chunking with uneven split."""
        items = [1, 2, 3, 4, 5]
        chunks = chunk_list(items, chunk_size=2)
        
        assert chunks == [[1, 2], [3, 4], [5]]
    
    def test_chunk_list_empty(self):
        """Test chunking empty list."""
        chunks = chunk_list([], chunk_size=3)
        assert chunks == []
    
    def test_chunk_list_larger_chunk_size(self):
        """Test chunk size larger than list."""
        items = [1, 2, 3]
        chunks = chunk_list(items, chunk_size=10)
        
        assert chunks == [[1, 2, 3]]
    
    def test_chunk_list_invalid_chunk_size_fails(self):
        """Test invalid chunk size fails."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            chunk_list([1, 2, 3], chunk_size=0)
        
        with pytest.raises(ValueError, match="must be a positive integer"):
            chunk_list([1, 2, 3], chunk_size=-1)


class TestStorageSize:
    """Test storage size calculation."""
    
    def test_calculate_storage_size_empty_directory(self):
        """Test calculating size of empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            size = calculate_storage_size(Path(tmpdir))
            assert size == 0
    
    def test_calculate_storage_size_with_files(self):
        """Test calculating size with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Create files
            (tmp_path / "file1.txt").write_text("Hello" * 100)
            (tmp_path / "file2.txt").write_text("World" * 50)
            
            size = calculate_storage_size(tmp_path)
            assert size > 0
    
    def test_calculate_storage_size_with_subdirectories(self):
        """Test calculating size with subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Create subdirectory with file
            subdir = tmp_path / "subdir"
            subdir.mkdir()
            (subdir / "file.txt").write_text("Test" * 100)
            
            size = calculate_storage_size(tmp_path)
            assert size > 0
    
    def test_calculate_storage_size_nonexistent_fails(self):
        """Test calculating size of nonexistent path fails."""
        with pytest.raises(ValueError, match="does not exist"):
            calculate_storage_size(Path("/nonexistent/path"))
    
    def test_calculate_storage_size_file_fails(self):
        """Test calculating size of file (not directory) fails."""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(b"test")
            tmpfile.flush()
            tmpfile_path = Path(tmpfile.name)
        
        # File is now closed, safe to test
        try:
            with pytest.raises(ValueError, match="not a directory"):
                calculate_storage_size(tmpfile_path)
        finally:
            tmpfile_path.unlink(missing_ok=True)


class TestFormatBytes:
    """Test byte formatting."""
    
    def test_format_bytes_zero(self):
        """Test formatting zero bytes."""
        assert format_bytes(0) == "0 B"
    
    def test_format_bytes_small(self):
        """Test formatting small byte values."""
        assert format_bytes(500) == "500 B"
        assert format_bytes(1023) == "1023 B"
    
    def test_format_bytes_kilobytes(self):
        """Test formatting kilobytes."""
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(2048) == "2.00 KB"
    
    def test_format_bytes_megabytes(self):
        """Test formatting megabytes."""
        assert format_bytes(1048576) == "1.00 MB"
        assert format_bytes(5242880) == "5.00 MB"
    
    def test_format_bytes_gigabytes(self):
        """Test formatting gigabytes."""
        assert format_bytes(1073741824) == "1.00 GB"
    
    def test_format_bytes_negative_fails(self):
        """Test negative byte value fails."""
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            format_bytes(-100)


class TestMergeFilters:
    """Test filter merging."""
    
    def test_merge_filters_both_none(self):
        """Test merging two None filters."""
        result = merge_filters(None, None)
        assert result == {}
    
    def test_merge_filters_first_none(self):
        """Test merging when first filter is None."""
        filter2 = {"key": "value"}
        result = merge_filters(None, filter2)
        assert result == filter2
    
    def test_merge_filters_second_none(self):
        """Test merging when second filter is None."""
        filter1 = {"key": "value"}
        result = merge_filters(filter1, None)
        assert result == filter1
    
    def test_merge_filters_no_overlap(self):
        """Test merging filters with no overlap."""
        filter1 = {"key1": "value1"}
        filter2 = {"key2": "value2"}
        result = merge_filters(filter1, filter2)
        
        assert result == {"key1": "value1", "key2": "value2"}
    
    def test_merge_filters_with_overlap(self):
        """Test merging filters with overlap (second takes precedence)."""
        filter1 = {"key": "value1", "other": "keep"}
        filter2 = {"key": "value2"}
        result = merge_filters(filter1, filter2)
        
        assert result == {"key": "value2", "other": "keep"}
    
    def test_merge_filters_preserves_originals(self):
        """Test merging doesn't modify original filters."""
        filter1 = {"key1": "value1"}
        filter2 = {"key2": "value2"}
        original1 = filter1.copy()
        original2 = filter2.copy()
        
        merge_filters(filter1, filter2)
        
        assert filter1 == original1
        assert filter2 == original2


class TestCollectionNameValidation:
    """Test collection name validation."""
    
    def test_validate_collection_name_success(self):
        """Test valid collection names."""
        assert validate_collection_name("my_collection")
        assert validate_collection_name("collection-123")
        assert validate_collection_name("Collection_Name_123")
    
    def test_validate_collection_name_empty_fails(self):
        """Test empty name fails."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            validate_collection_name("")
    
    def test_validate_collection_name_non_string_fails(self):
        """Test non-string name fails."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            validate_collection_name(123)
    
    def test_validate_collection_name_invalid_chars_fails(self):
        """Test invalid characters fail."""
        with pytest.raises(ValueError, match="must contain only alphanumeric"):
            validate_collection_name("collection@name")
        
        with pytest.raises(ValueError, match="must contain only alphanumeric"):
            validate_collection_name("collection name")  # space
    
    def test_validate_collection_name_too_long_fails(self):
        """Test name too long fails."""
        long_name = "a" * 256
        with pytest.raises(ValueError, match="must be 255 characters or less"):
            validate_collection_name(long_name)
    
    def test_validate_collection_name_max_length_success(self):
        """Test maximum length name succeeds."""
        max_name = "a" * 255
        assert validate_collection_name(max_name)

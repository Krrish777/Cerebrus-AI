"""
Tests for Collection Statistics Models

Following AGENTS.md principles:
- Test all validation logic
- Test data conversion methods
- Clear test names describing behavior
"""

import pytest
from pathlib import Path
from src.vector_database.models.collection_stats import (
    CollectionStats,
    CollectionInfo,
    CollectionMetadata
)


class TestCollectionStats:
    """Tests for CollectionStats dataclass."""
    
    def test_minimal_stats(self):
        """Test collection stats with minimal required fields."""
        stats = CollectionStats(
            total_documents=100,
            collection_name="test_collection",
            embedding_dimension=384,
            storage_path=Path("./storage"),
            embedding_models=["model1"],
            source_types=["pdf"],
            unique_sources=10,
            hnsw_config={"m": 16},
            quantization_enabled=False
        )
        assert stats.total_documents == 100
        assert stats.collection_name == "test_collection"
        assert stats.embedding_dimension == 384
        assert isinstance(stats.storage_path, Path)
    
    def test_negative_total_documents_fails(self):
        """Test validation fails for negative total_documents."""
        with pytest.raises(ValueError, match="Total documents must be non-negative"):
            CollectionStats(
                total_documents=-1,
                collection_name="test",
                embedding_dimension=384,
                storage_path=Path("./storage"),
                embedding_models=[],
                source_types=[],
                unique_sources=0,
                hnsw_config={},
                quantization_enabled=False
            )
    
    def test_invalid_embedding_dimension_fails(self):
        """Test validation fails for embedding_dimension < 1."""
        with pytest.raises(ValueError, match="Embedding dimension must be positive"):
            CollectionStats(
                total_documents=0,
                collection_name="test",
                embedding_dimension=0,
                storage_path=Path("./storage"),
                embedding_models=[],
                source_types=[],
                unique_sources=0,
                hnsw_config={},
                quantization_enabled=False
            )
    
    def test_negative_unique_sources_fails(self):
        """Test validation fails for negative unique_sources."""
        with pytest.raises(ValueError, match="Unique sources must be non-negative"):
            CollectionStats(
                total_documents=0,
                collection_name="test",
                embedding_dimension=384,
                storage_path=Path("./storage"),
                embedding_models=[],
                source_types=[],
                unique_sources=-1,
                hnsw_config={},
                quantization_enabled=False
            )
    
    def test_string_path_conversion(self):
        """Test automatic conversion of string path to Path object."""
        stats = CollectionStats(
            total_documents=50,
            collection_name="test",
            embedding_dimension=768,
            storage_path="./string_path",  # type: ignore
            embedding_models=[],
            source_types=[],
            unique_sources=0,
            hnsw_config={},
            quantization_enabled=False
        )
        assert isinstance(stats.storage_path, Path)
        assert stats.storage_path == Path("./string_path")
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = CollectionStats(
            total_documents=200,
            collection_name="prod_collection",
            embedding_dimension=1024,
            storage_path=Path("./prod_storage"),
            embedding_models=["bge-small", "bge-base"],
            source_types=["pdf", "html", "txt"],
            unique_sources=50,
            hnsw_config={"m": 32, "ef_construct": 400},
            quantization_enabled=True
        )
        stats_dict = stats.to_dict()
        
        assert stats_dict['total_documents'] == 200
        assert stats_dict['collection_name'] == "prod_collection"
        assert stats_dict['embedding_dimension'] == 1024
        # Path serialization normalizes paths (removes ./ prefix on Windows)
        assert Path(stats_dict['storage_path']) == Path("./prod_storage")
        assert stats_dict['embedding_models'] == ["bge-small", "bge-base"]
        assert stats_dict['source_types'] == ["pdf", "html", "txt"]
        assert stats_dict['unique_sources'] == 50
        assert stats_dict['hnsw_config'] == {"m": 32, "ef_construct": 400}
        assert stats_dict['quantization_enabled'] is True


class TestCollectionInfo:
    """Tests for CollectionInfo dataclass."""
    
    def test_valid_info(self):
        """Test creating valid collection info."""
        info = CollectionInfo(
            name="test_collection",
            vector_count=1000,
            indexed=True,
            status="green"
        )
        assert info.name == "test_collection"
        assert info.vector_count == 1000
        assert info.indexed is True
        assert info.status == "green"
    
    def test_negative_vector_count_fails(self):
        """Test validation fails for negative vector_count."""
        with pytest.raises(ValueError, match="Vector count must be non-negative"):
            CollectionInfo(
                name="test",
                vector_count=-1,
                indexed=True,
                status="green"
            )
    
    def test_invalid_status_fails(self):
        """Test validation fails for invalid status value."""
        with pytest.raises(ValueError, match="Status must be one of"):
            CollectionInfo(
                name="test",
                vector_count=100,
                indexed=True,
                status="invalid"
            )
    
    def test_valid_statuses(self):
        """Test all valid status values."""
        valid_statuses = ["green", "yellow", "red", "unknown"]
        for status in valid_statuses:
            info = CollectionInfo(
                name="test",
                vector_count=100,
                indexed=True,
                status=status
            )
            assert info.status == status
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = CollectionInfo(
            name="prod_collection",
            vector_count=5000,
            indexed=False,
            status="yellow"
        )
        info_dict = info.to_dict()
        
        assert info_dict == {
            'name': 'prod_collection',
            'vector_count': 5000,
            'indexed': False,
            'status': 'yellow'
        }


class TestCollectionMetadata:
    """Tests for CollectionMetadata dataclass."""
    
    def test_minimal_metadata(self):
        """Test metadata with only required field."""
        metadata = CollectionMetadata(document_count=100)
        assert metadata.document_count == 100
        assert metadata.avg_document_length is None
        assert metadata.min_document_length is None
        assert metadata.max_document_length is None
        assert metadata.metadata_fields is None
        assert metadata.index_size_bytes is None
        assert metadata.created_at is None
        assert metadata.updated_at is None
    
    def test_full_metadata(self):
        """Test metadata with all fields populated."""
        metadata = CollectionMetadata(
            document_count=500,
            avg_document_length=1500.5,
            min_document_length=100,
            max_document_length=5000,
            metadata_fields=["source", "author", "date"],
            index_size_bytes=1048576,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-15T12:30:00Z"
        )
        assert metadata.document_count == 500
        assert metadata.avg_document_length == 1500.5
        assert metadata.min_document_length == 100
        assert metadata.max_document_length == 5000
        assert metadata.metadata_fields == ["source", "author", "date"]
        assert metadata.index_size_bytes == 1048576
        assert metadata.created_at == "2024-01-01T00:00:00Z"
        assert metadata.updated_at == "2024-01-15T12:30:00Z"
    
    def test_negative_document_count_fails(self):
        """Test validation fails for negative document_count."""
        with pytest.raises(ValueError, match="Document count must be non-negative"):
            CollectionMetadata(document_count=-1)
    
    def test_negative_avg_length_fails(self):
        """Test validation fails for negative avg_document_length."""
        with pytest.raises(ValueError, match="Average document length must be non-negative"):
            CollectionMetadata(
                document_count=100,
                avg_document_length=-10.5
            )
    
    def test_negative_min_length_fails(self):
        """Test validation fails for negative min_document_length."""
        with pytest.raises(ValueError, match="Minimum document length must be non-negative"):
            CollectionMetadata(
                document_count=100,
                min_document_length=-5
            )
    
    def test_negative_max_length_fails(self):
        """Test validation fails for negative max_document_length."""
        with pytest.raises(ValueError, match="Maximum document length must be non-negative"):
            CollectionMetadata(
                document_count=100,
                max_document_length=-100
            )
    
    def test_negative_index_size_fails(self):
        """Test validation fails for negative index_size_bytes."""
        with pytest.raises(ValueError, match="Index size must be non-negative"):
            CollectionMetadata(
                document_count=100,
                index_size_bytes=-1024
            )
    
    def test_to_dict_minimal(self):
        """Test conversion to dict with minimal fields."""
        metadata = CollectionMetadata(document_count=50)
        metadata_dict = metadata.to_dict()
        
        assert metadata_dict == {'document_count': 50}
        assert 'avg_document_length' not in metadata_dict
        assert 'metadata_fields' not in metadata_dict
    
    def test_to_dict_full(self):
        """Test conversion to dict with all fields."""
        metadata = CollectionMetadata(
            document_count=1000,
            avg_document_length=2000.0,
            min_document_length=500,
            max_document_length=10000,
            metadata_fields=["title", "author"],
            index_size_bytes=2097152,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-20T15:00:00Z"
        )
        metadata_dict = metadata.to_dict()
        
        assert metadata_dict['document_count'] == 1000
        assert metadata_dict['avg_document_length'] == 2000.0
        assert metadata_dict['min_document_length'] == 500
        assert metadata_dict['max_document_length'] == 10000
        assert metadata_dict['metadata_fields'] == ["title", "author"]
        assert metadata_dict['index_size_bytes'] == 2097152
        assert metadata_dict['created_at'] == "2024-01-01T00:00:00Z"
        assert metadata_dict['updated_at'] == "2024-01-20T15:00:00Z"
    
    def test_to_dict_partial(self):
        """Test conversion to dict with some optional fields."""
        metadata = CollectionMetadata(
            document_count=300,
            avg_document_length=1200.5,
            metadata_fields=["category"]
        )
        metadata_dict = metadata.to_dict()
        
        assert metadata_dict['document_count'] == 300
        assert metadata_dict['avg_document_length'] == 1200.5
        assert metadata_dict['metadata_fields'] == ["category"]
        assert 'min_document_length' not in metadata_dict
        assert 'index_size_bytes' not in metadata_dict

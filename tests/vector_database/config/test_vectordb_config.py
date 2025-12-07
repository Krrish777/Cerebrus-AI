"""
Tests for Vector Database Configuration Module

Following AGENTS.md principles:
- Test all validation logic
- Test edge cases and boundary conditions
- Clear test names describing behavior
- No emojis or decorative characters
"""

import pytest
from pathlib import Path
from src.vector_database.config.vectordb_config import (
    HNSWConfig,
    QuantizationConfig,
    QdrantConfig,
    SearchConfig,
    VectorDatabaseConfig
)


class TestHNSWConfig:
    """Tests for HNSW configuration dataclass."""
    
    def test_default_values(self):
        """Test HNSW config with default values."""
        config = HNSWConfig()
        assert config.m == 16
        assert config.ef_construct == 200
        assert config.full_scan_threshold == 10000
    
    def test_custom_values(self):
        """Test HNSW config with custom values."""
        config = HNSWConfig(m=32, ef_construct=400, full_scan_threshold=5000)
        assert config.m == 32
        assert config.ef_construct == 400
        assert config.full_scan_threshold == 5000
    
    def test_invalid_m_too_small(self):
        """Test validation fails for m < 2."""
        with pytest.raises(ValueError, match="HNSW m must be between 2 and 100"):
            HNSWConfig(m=1)
    
    def test_invalid_m_too_large(self):
        """Test validation fails for m > 100."""
        with pytest.raises(ValueError, match="HNSW m must be between 2 and 100"):
            HNSWConfig(m=101)
    
    def test_invalid_ef_construct(self):
        """Test validation fails for ef_construct < 1."""
        with pytest.raises(ValueError, match="HNSW ef_construct must be positive"):
            HNSWConfig(ef_construct=0)
    
    def test_invalid_full_scan_threshold(self):
        """Test validation fails for negative full_scan_threshold."""
        with pytest.raises(ValueError, match="HNSW full_scan_threshold must be non-negative"):
            HNSWConfig(full_scan_threshold=-1)
    
    def test_boundary_values(self):
        """Test HNSW config with boundary values."""
        config = HNSWConfig(m=2, ef_construct=1, full_scan_threshold=0)
        assert config.m == 2
        assert config.ef_construct == 1
        assert config.full_scan_threshold == 0


class TestQuantizationConfig:
    """Tests for quantization configuration dataclass."""
    
    def test_default_values(self):
        """Test quantization config with default values."""
        config = QuantizationConfig()
        assert config.enabled is False
        assert config.type == "scalar"
        assert config.scalar_type == "int8"
        assert config.quantile == 0.99
        assert config.always_ram is False
    
    def test_enabled_config(self):
        """Test enabled quantization configuration."""
        config = QuantizationConfig(enabled=True, quantile=0.95)
        assert config.enabled is True
        assert config.quantile == 0.95
    
    def test_product_quantization(self):
        """Test product quantization configuration."""
        config = QuantizationConfig(enabled=True, type="product")
        assert config.type == "product"
    
    def test_invalid_quantile_too_small(self):
        """Test validation fails for quantile < 0."""
        with pytest.raises(ValueError, match="Quantization quantile must be between 0.0 and 1.0"):
            QuantizationConfig(quantile=-0.1)
    
    def test_invalid_quantile_too_large(self):
        """Test validation fails for quantile > 1."""
        with pytest.raises(ValueError, match="Quantization quantile must be between 0.0 and 1.0"):
            QuantizationConfig(quantile=1.1)
    
    def test_to_qdrant_config_disabled(self):
        """Test conversion to Qdrant config when disabled."""
        config = QuantizationConfig(enabled=False)
        assert config.to_qdrant_config() is None
    
    def test_to_qdrant_config_scalar(self):
        """Test conversion to Qdrant config for scalar quantization."""
        config = QuantizationConfig(
            enabled=True,
            type="scalar",
            scalar_type="int8",
            quantile=0.95,
            always_ram=True
        )
        qdrant_config = config.to_qdrant_config()
        assert qdrant_config is not None
        assert "scalar" in qdrant_config
        assert qdrant_config["scalar"]["type"] == "int8"
        assert qdrant_config["scalar"]["quantile"] == 0.95
        assert qdrant_config["scalar"]["always_ram"] is True
    
    def test_to_qdrant_config_product(self):
        """Test conversion to Qdrant config for product quantization."""
        config = QuantizationConfig(enabled=True, type="product", always_ram=True)
        qdrant_config = config.to_qdrant_config()
        assert qdrant_config is not None
        assert "product" in qdrant_config
        assert qdrant_config["product"]["compression"] == "x32"
        assert qdrant_config["product"]["always_ram"] is True


class TestQdrantConfig:
    """Tests for Qdrant-specific configuration dataclass."""
    
    def test_default_values(self):
        """Test Qdrant config with default values."""
        config = QdrantConfig()
        assert config.recreate_index is False
        assert config.return_embedding is True
        assert config.wait_result_from_api is True
        assert isinstance(config.hnsw, HNSWConfig)
        assert isinstance(config.quantization, QuantizationConfig)
    
    def test_custom_values(self):
        """Test Qdrant config with custom values."""
        hnsw = HNSWConfig(m=32)
        quantization = QuantizationConfig(enabled=True)
        config = QdrantConfig(
            recreate_index=True,
            return_embedding=False,
            hnsw=hnsw,
            quantization=quantization
        )
        assert config.recreate_index is True
        assert config.return_embedding is False
        assert config.hnsw.m == 32
        assert config.quantization.enabled is True


class TestSearchConfig:
    """Tests for search configuration dataclass."""
    
    def test_default_values(self):
        """Test search config with default values."""
        config = SearchConfig()
        assert config.top_k == 10
        assert config.scale_score is True
        assert config.score_threshold is None
        assert config.return_embedding is False
    
    def test_custom_values(self):
        """Test search config with custom values."""
        config = SearchConfig(
            top_k=50,
            scale_score=False,
            score_threshold=0.7,
            return_embedding=True
        )
        assert config.top_k == 50
        assert config.scale_score is False
        assert config.score_threshold == 0.7
        assert config.return_embedding is True
    
    def test_invalid_top_k_zero(self):
        """Test validation fails for top_k = 0."""
        with pytest.raises(ValueError, match="Search top_k must be positive"):
            SearchConfig(top_k=0)
    
    def test_invalid_top_k_negative(self):
        """Test validation fails for negative top_k."""
        with pytest.raises(ValueError, match="Search top_k must be positive"):
            SearchConfig(top_k=-1)
    
    def test_invalid_top_k_too_large(self):
        """Test validation fails for top_k > 1000."""
        with pytest.raises(ValueError, match="Search top_k must be <= 1000"):
            SearchConfig(top_k=1001)
    
    def test_invalid_score_threshold_negative(self):
        """Test validation fails for negative score_threshold."""
        with pytest.raises(ValueError, match="Search score_threshold must be between 0.0 and 1.0"):
            SearchConfig(score_threshold=-0.1)
    
    def test_invalid_score_threshold_too_large(self):
        """Test validation fails for score_threshold > 1.0."""
        with pytest.raises(ValueError, match="Search score_threshold must be between 0.0 and 1.0"):
            SearchConfig(score_threshold=1.1)
    
    def test_boundary_values(self):
        """Test search config with boundary values."""
        config = SearchConfig(top_k=1, score_threshold=0.0)
        assert config.top_k == 1
        assert config.score_threshold == 0.0


class TestVectorDatabaseConfig:
    """Tests for main vector database configuration dataclass."""
    
    def test_default_values(self):
        """Test vector database config with default values."""
        config = VectorDatabaseConfig()
        assert config.provider == "qdrant"
        assert config.storage_path == Path("./storage/qdrant_db")
        assert config.collection_name == "cerebrus_documents"
        assert config.embedding_dim == 384
        assert isinstance(config.qdrant, QdrantConfig)
        assert isinstance(config.search, SearchConfig)
    
    def test_custom_values(self):
        """Test vector database config with custom values."""
        config = VectorDatabaseConfig(
            provider="qdrant",
            storage_path=Path("./custom_storage"),
            collection_name="test_collection",
            embedding_dim=768
        )
        assert config.storage_path == Path("./custom_storage")
        assert config.collection_name == "test_collection"
        assert config.embedding_dim == 768
    
    def test_string_path_conversion(self):
        """Test automatic conversion of string path to Path object."""
        config = VectorDatabaseConfig(storage_path="./string_path")  # type: ignore
        assert isinstance(config.storage_path, Path)
        assert config.storage_path == Path("./string_path")
    
    def test_invalid_embedding_dim_zero(self):
        """Test validation fails for embedding_dim = 0."""
        with pytest.raises(ValueError, match="Embedding dimension must be between 1 and 4096"):
            VectorDatabaseConfig(embedding_dim=0)
    
    def test_invalid_embedding_dim_negative(self):
        """Test validation fails for negative embedding_dim."""
        with pytest.raises(ValueError, match="Embedding dimension must be between 1 and 4096"):
            VectorDatabaseConfig(embedding_dim=-1)
    
    def test_invalid_embedding_dim_too_large(self):
        """Test validation fails for embedding_dim > 4096."""
        with pytest.raises(ValueError, match="Embedding dimension must be between 1 and 4096"):
            VectorDatabaseConfig(embedding_dim=5000)
    
    def test_invalid_collection_name_empty(self):
        """Test validation fails for empty collection name."""
        with pytest.raises(ValueError, match="Collection name cannot be empty"):
            VectorDatabaseConfig(collection_name="")
    
    def test_invalid_collection_name_special_chars(self):
        """Test validation fails for collection name with invalid characters."""
        with pytest.raises(ValueError, match="Collection name must be alphanumeric with underscores/hyphens"):
            VectorDatabaseConfig(collection_name="invalid@name!")
    
    def test_valid_collection_name_with_underscores(self):
        """Test valid collection name with underscores."""
        config = VectorDatabaseConfig(collection_name="test_collection_name")
        assert config.collection_name == "test_collection_name"
    
    def test_valid_collection_name_with_hyphens(self):
        """Test valid collection name with hyphens."""
        config = VectorDatabaseConfig(collection_name="test-collection-name")
        assert config.collection_name == "test-collection-name"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = VectorDatabaseConfig(
            storage_path=Path("./test_storage"),
            embedding_dim=768
        )
        config_dict = config.to_dict()
        
        assert config_dict['provider'] == 'qdrant'
        # Path.to_dict() normalizes paths (removes ./ prefix on Windows)
        assert Path(config_dict['storage_path']) == Path('./test_storage')
        assert config_dict['embedding_dim'] == 768
        assert 'qdrant' in config_dict
        assert 'search' in config_dict


class TestVectorDatabaseConfigYAML:
    """Tests for YAML loading functionality."""
    
    def test_from_yaml_file_not_found(self, tmp_path):
        """Test error when YAML file doesn't exist."""
        yaml_path = tmp_path / "nonexistent.yml"
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            VectorDatabaseConfig.from_yaml(yaml_path)
    
    def test_from_yaml_invalid_structure(self, tmp_path):
        """Test error when YAML has invalid structure."""
        yaml_path = tmp_path / "invalid.yml"
        yaml_path.write_text("invalid: yaml")
        
        with pytest.raises(ValueError, match="YAML must contain 'vector_database' section"):
            VectorDatabaseConfig.from_yaml(yaml_path)
    
    def test_from_yaml_minimal_config(self, tmp_path):
        """Test loading minimal valid YAML config."""
        yaml_path = tmp_path / "config.yml"
        yaml_path.write_text("""
vector_database:
  provider: qdrant
  storage_path: ./test_storage
  collection_name: test_collection
  embedding_dim: 768
""")
        
        config = VectorDatabaseConfig.from_yaml(yaml_path)
        assert config.provider == "qdrant"
        assert config.storage_path == Path("./test_storage")
        assert config.collection_name == "test_collection"
        assert config.embedding_dim == 768
    
    def test_from_yaml_full_config(self, tmp_path):
        """Test loading complete YAML config with all sections."""
        yaml_path = tmp_path / "config.yml"
        yaml_path.write_text("""
vector_database:
  provider: qdrant
  storage_path: ./full_storage
  collection_name: full_collection
  embedding_dim: 1024
  
  qdrant:
    recreate_index: true
    return_embedding: false
    wait_result_from_api: false
    
    hnsw:
      m: 32
      ef_construct: 400
      full_scan_threshold: 5000
    
    quantization:
      enabled: true
      type: scalar
      scalar_type: uint8
      quantile: 0.95
      always_ram: true
  
  search:
    top_k: 50
    scale_score: false
    score_threshold: 0.7
    return_embedding: true
""")
        
        config = VectorDatabaseConfig.from_yaml(yaml_path)
        
        # Main config
        assert config.provider == "qdrant"
        assert config.storage_path == Path("./full_storage")
        assert config.collection_name == "full_collection"
        assert config.embedding_dim == 1024
        
        # Qdrant config
        assert config.qdrant.recreate_index is True
        assert config.qdrant.return_embedding is False
        assert config.qdrant.wait_result_from_api is False
        
        # HNSW config
        assert config.qdrant.hnsw.m == 32
        assert config.qdrant.hnsw.ef_construct == 400
        assert config.qdrant.hnsw.full_scan_threshold == 5000
        
        # Quantization config
        assert config.qdrant.quantization.enabled is True
        assert config.qdrant.quantization.type == "scalar"
        assert config.qdrant.quantization.scalar_type == "uint8"
        assert config.qdrant.quantization.quantile == 0.95
        assert config.qdrant.quantization.always_ram is True
        
        # Search config
        assert config.search.top_k == 50
        assert config.search.scale_score is False
        assert config.search.score_threshold == 0.7
        assert config.search.return_embedding is True

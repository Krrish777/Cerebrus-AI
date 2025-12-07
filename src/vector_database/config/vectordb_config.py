"""
Vector Database Configuration Module

This module defines configuration dataclasses for vector database operations.
All configurations use frozen dataclasses for immutability and include validation.

Following AGENTS.md principles:
- Immutable configuration (frozen=True)
- Validation at initialization
- Type hints for all fields
- No hard-coded values
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Literal
import yaml


@dataclass(frozen=True)
class HNSWConfig:
    """
    Configuration for HNSW (Hierarchical Navigable Small World) index.
    
    HNSW is used for approximate nearest neighbor search in high-dimensional spaces.
    
    Attributes:
        m: Number of bi-directional links for every new element (default: 16)
        ef_construct: Size of dynamic candidate list during construction (default: 200)
        full_scan_threshold: Threshold for switching to full scan (default: 10000)
    """
    m: int = 16
    ef_construct: int = 200
    full_scan_threshold: int = 10000
    
    def __post_init__(self):
        """Validate HNSW configuration values."""
        if self.m < 2 or self.m > 100:
            raise ValueError(f"HNSW m must be between 2 and 100, got {self.m}")
        if self.ef_construct < 1:
            raise ValueError(f"HNSW ef_construct must be positive, got {self.ef_construct}")
        if self.full_scan_threshold < 0:
            raise ValueError(f"HNSW full_scan_threshold must be non-negative, got {self.full_scan_threshold}")


@dataclass(frozen=True)
class QuantizationConfig:
    """
    Configuration for vector quantization to reduce memory usage.
    
    Quantization compresses vector representations while maintaining search quality.
    
    Attributes:
        enabled: Whether quantization is enabled
        type: Type of quantization ('scalar' or 'product')
        scalar_type: Scalar quantization type ('int8' or 'uint8')
        quantile: Quantile for scalar quantization (default: 0.99)
        always_ram: Keep quantized vectors in RAM (default: False)
    """
    enabled: bool = False
    type: Literal["scalar", "product"] = "scalar"
    scalar_type: Literal["int8", "uint8"] = "int8"
    quantile: float = 0.99
    always_ram: bool = False
    
    def __post_init__(self):
        """Validate quantization configuration values."""
        if self.quantile < 0.0 or self.quantile > 1.0:
            raise ValueError(f"Quantization quantile must be between 0.0 and 1.0, got {self.quantile}")
    
    def to_qdrant_config(self) -> Optional[Dict[str, Any]]:
        """Convert to Qdrant-compatible quantization config."""
        if not self.enabled:
            return None
        
        if self.type == "scalar":
            return {
                "scalar": {
                    "type": self.scalar_type,
                    "quantile": self.quantile,
                    "always_ram": self.always_ram
                }
            }
        elif self.type == "product":
            return {
                "product": {
                    "compression": "x32",
                    "always_ram": self.always_ram
                }
            }
        return None


@dataclass(frozen=True)
class QdrantConfig:
    """
    Configuration specific to Qdrant vector database.
    
    Attributes:
        recreate_index: Whether to recreate index on startup (default: False)
        return_embedding: Return embeddings in search results (default: True)
        wait_result_from_api: Wait for API responses (default: True)
        hnsw: HNSW index configuration
        quantization: Vector quantization configuration
    """
    recreate_index: bool = False
    return_embedding: bool = True
    wait_result_from_api: bool = True
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


@dataclass(frozen=True)
class SearchConfig:
    """
    Configuration for search operations.
    
    Attributes:
        top_k: Maximum number of results to return (default: 10)
        scale_score: Scale scores to 0-1 range (default: True)
        score_threshold: Minimum score threshold (default: None)
        return_embedding: Return embeddings in results (default: False)
    """
    top_k: int = 10
    scale_score: bool = True
    score_threshold: Optional[float] = None
    return_embedding: bool = False
    
    def __post_init__(self):
        """Validate search configuration values."""
        if self.top_k < 1:
            raise ValueError(f"Search top_k must be positive, got {self.top_k}")
        if self.top_k > 1000:
            raise ValueError(f"Search top_k must be <= 1000, got {self.top_k}")
        if self.score_threshold is not None:
            if self.score_threshold < 0.0 or self.score_threshold > 1.0:
                raise ValueError(f"Search score_threshold must be between 0.0 and 1.0, got {self.score_threshold}")


@dataclass(frozen=True)
class VectorDatabaseConfig:
    """
    Main vector database configuration.
    
    This is the top-level configuration that aggregates all vector DB settings.
    
    Attributes:
        provider: Vector database provider ('qdrant', 'pinecone', 'weaviate')
        storage_path: Path to store database files
        collection_name: Name of the collection
        embedding_dim: Dimension of embedding vectors
        qdrant: Qdrant-specific configuration
        search: Default search configuration
    """
    provider: Literal["qdrant", "pinecone", "weaviate"] = "qdrant"
    storage_path: Path = Path("./storage/qdrant_db")
    collection_name: str = "cerebrus_documents"
    embedding_dim: int = 384
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    
    def __post_init__(self):
        """Validate vector database configuration values."""
        # Convert string to Path if needed
        if isinstance(self.storage_path, str):
            object.__setattr__(self, 'storage_path', Path(self.storage_path))
        
        if self.embedding_dim < 1 or self.embedding_dim > 4096:
            raise ValueError(f"Embedding dimension must be between 1 and 4096, got {self.embedding_dim}")
        
        if not self.collection_name:
            raise ValueError("Collection name cannot be empty")
        
        if not self.collection_name.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Collection name must be alphanumeric with underscores/hyphens, got '{self.collection_name}'")
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "VectorDatabaseConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            VectorDatabaseConfig instance
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML structure is invalid
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if not config_dict or 'vector_database' not in config_dict:
            raise ValueError("YAML must contain 'vector_database' section")
        
        vdb_config = config_dict['vector_database']
        
        # Parse nested configurations
        qdrant_config = QdrantConfig(
            recreate_index=vdb_config.get('qdrant', {}).get('recreate_index', False),
            return_embedding=vdb_config.get('qdrant', {}).get('return_embedding', True),
            wait_result_from_api=vdb_config.get('qdrant', {}).get('wait_result_from_api', True),
            hnsw=HNSWConfig(**vdb_config.get('qdrant', {}).get('hnsw', {})),
            quantization=QuantizationConfig(**vdb_config.get('qdrant', {}).get('quantization', {}))
        )
        
        search_config = SearchConfig(**vdb_config.get('search', {}))
        
        return cls(
            provider=vdb_config.get('provider', 'qdrant'),
            storage_path=Path(vdb_config.get('storage_path', './storage/qdrant_db')),
            collection_name=vdb_config.get('collection_name', 'cerebrus_documents'),
            embedding_dim=vdb_config.get('embedding_dim', 384),
            qdrant=qdrant_config,
            search=search_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'provider': self.provider,
            'storage_path': str(self.storage_path),
            'collection_name': self.collection_name,
            'embedding_dim': self.embedding_dim,
            'qdrant': {
                'recreate_index': self.qdrant.recreate_index,
                'return_embedding': self.qdrant.return_embedding,
                'wait_result_from_api': self.qdrant.wait_result_from_api,
                'hnsw': {
                    'm': self.qdrant.hnsw.m,
                    'ef_construct': self.qdrant.hnsw.ef_construct,
                    'full_scan_threshold': self.qdrant.hnsw.full_scan_threshold
                },
                'quantization': {
                    'enabled': self.qdrant.quantization.enabled,
                    'type': self.qdrant.quantization.type,
                    'scalar_type': self.qdrant.quantization.scalar_type,
                    'quantile': self.qdrant.quantization.quantile,
                    'always_ram': self.qdrant.quantization.always_ram
                }
            },
            'search': {
                'top_k': self.search.top_k,
                'scale_score': self.search.scale_score,
                'score_threshold': self.search.score_threshold,
                'return_embedding': self.search.return_embedding
            }
        }

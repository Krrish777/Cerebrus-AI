"""
Collection Statistics Models for Vector Database

This module defines dataclasses for collection statistics and metadata.

Following AGENTS.md principles:
- Immutable data structures (frozen=True)
- Type hints for all fields
- Clear data representation
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass(frozen=True)
class CollectionStats:
    """
    Statistics about a vector database collection.
    
    Attributes:
        total_documents: Total number of documents in collection
        collection_name: Name of the collection
        embedding_dimension: Dimension of embedding vectors
        storage_path: Path where collection data is stored
        embedding_models: List of embedding models used
        source_types: List of source document types
        unique_sources: Number of unique source files
        hnsw_config: HNSW index configuration
        quantization_enabled: Whether quantization is enabled
    """
    total_documents: int
    collection_name: str
    embedding_dimension: int
    storage_path: Path
    embedding_models: List[str]
    source_types: List[str]
    unique_sources: int
    hnsw_config: Dict[str, Any]
    quantization_enabled: bool
    
    def __post_init__(self):
        """Validate collection statistics values."""
        if self.total_documents < 0:
            raise ValueError(f"Total documents must be non-negative, got {self.total_documents}")
        
        if self.embedding_dimension < 1:
            raise ValueError(f"Embedding dimension must be positive, got {self.embedding_dimension}")
        
        if self.unique_sources < 0:
            raise ValueError(f"Unique sources must be non-negative, got {self.unique_sources}")
        
        # Convert string to Path if needed
        if isinstance(self.storage_path, str):
            object.__setattr__(self, 'storage_path', Path(self.storage_path))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collection stats to dictionary."""
        return {
            'total_documents': self.total_documents,
            'collection_name': self.collection_name,
            'embedding_dimension': self.embedding_dimension,
            'storage_path': str(self.storage_path),
            'embedding_models': self.embedding_models,
            'source_types': self.source_types,
            'unique_sources': self.unique_sources,
            'hnsw_config': self.hnsw_config,
            'quantization_enabled': self.quantization_enabled
        }


@dataclass(frozen=True)
class CollectionInfo:
    """
    Basic information about a vector database collection.
    
    Attributes:
        name: Collection name
        vector_count: Number of vectors in collection
        indexed: Whether collection is indexed
        status: Collection status (e.g., 'green', 'yellow', 'red')
    """
    name: str
    vector_count: int
    indexed: bool
    status: str
    
    def __post_init__(self):
        """Validate collection info values."""
        if self.vector_count < 0:
            raise ValueError(f"Vector count must be non-negative, got {self.vector_count}")
        
        valid_statuses = ['green', 'yellow', 'red', 'unknown']
        if self.status not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}, got '{self.status}'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collection info to dictionary."""
        return {
            'name': self.name,
            'vector_count': self.vector_count,
            'indexed': self.indexed,
            'status': self.status
        }


@dataclass(frozen=True)
class CollectionMetadata:
    """
    Detailed metadata about a collection's contents.
    
    Attributes:
        document_count: Number of documents
        avg_document_length: Average document length in characters
        min_document_length: Minimum document length
        max_document_length: Maximum document length
        metadata_fields: List of metadata field names present
        index_size_bytes: Size of index in bytes
        created_at: Collection creation timestamp
        updated_at: Last update timestamp
    """
    document_count: int
    avg_document_length: Optional[float] = None
    min_document_length: Optional[int] = None
    max_document_length: Optional[int] = None
    metadata_fields: Optional[List[str]] = None
    index_size_bytes: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __post_init__(self):
        """Validate collection metadata values."""
        if self.document_count < 0:
            raise ValueError(f"Document count must be non-negative, got {self.document_count}")
        
        if self.avg_document_length is not None and self.avg_document_length < 0:
            raise ValueError(f"Average document length must be non-negative, got {self.avg_document_length}")
        
        if self.min_document_length is not None and self.min_document_length < 0:
            raise ValueError(f"Minimum document length must be non-negative, got {self.min_document_length}")
        
        if self.max_document_length is not None and self.max_document_length < 0:
            raise ValueError(f"Maximum document length must be non-negative, got {self.max_document_length}")
        
        if self.index_size_bytes is not None and self.index_size_bytes < 0:
            raise ValueError(f"Index size must be non-negative, got {self.index_size_bytes}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collection metadata to dictionary, excluding None values."""
        data: Dict[str, Any] = {'document_count': self.document_count}
        
        if self.avg_document_length is not None:
            data['avg_document_length'] = self.avg_document_length
        if self.min_document_length is not None:
            data['min_document_length'] = self.min_document_length
        if self.max_document_length is not None:
            data['max_document_length'] = self.max_document_length
        if self.metadata_fields is not None:
            data['metadata_fields'] = self.metadata_fields
        if self.index_size_bytes is not None:
            data['index_size_bytes'] = self.index_size_bytes
        if self.created_at is not None:
            data['created_at'] = self.created_at
        if self.updated_at is not None:
            data['updated_at'] = self.updated_at
        
        return data

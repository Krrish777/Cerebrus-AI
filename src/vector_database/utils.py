"""
Utility functions for vector database operations.

Helper functions for embeddings, validation, and conversions.
Following AGENTS.md: pure functions, no side effects, defensive programming.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import json

from src.core.logging import get_logger

logger = get_logger(__name__)


def validate_embedding(embedding: List[float], expected_dim: Optional[int] = None) -> bool:
    """
    Validate an embedding vector.
    
    Args:
        embedding: Embedding vector to validate
        expected_dim: Expected dimension (optional)
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If embedding is invalid
    """
    if not embedding or not isinstance(embedding, list):
        raise ValueError("Embedding must be a non-empty list")
    
    if not all(isinstance(x, (int, float)) for x in embedding):
        raise ValueError("Embedding must contain only numeric values")
    
    if expected_dim is not None:
        if len(embedding) != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim}, "
                f"got {len(embedding)}"
            )
    
    return True


def normalize_embedding(embedding: List[float]) -> List[float]:
    """
    Normalize embedding to unit length (L2 normalization).
    
    Args:
        embedding: Embedding vector to normalize
        
    Returns:
        Normalized embedding
        
    Raises:
        ValueError: If embedding is invalid or zero vector
    """
    validate_embedding(embedding)
    
    # Calculate L2 norm
    norm = sum(x * x for x in embedding) ** 0.5
    
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    
    # Normalize
    return [x / norm for x in embedding]


def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score (0.0 to 1.0)
        
    Raises:
        ValueError: If embeddings are invalid or different dimensions
    """
    validate_embedding(embedding1)
    validate_embedding(embedding2)
    
    if len(embedding1) != len(embedding2):
        raise ValueError(
            f"Embedding dimension mismatch: {len(embedding1)} vs {len(embedding2)}"
        )
    
    # Calculate dot product and norms
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    norm1 = sum(x * x for x in embedding1) ** 0.5
    norm2 = sum(x * x for x in embedding2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    
    # Clamp to [0, 1] range
    return max(0.0, min(1.0, similarity))


def generate_document_id(content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a unique document ID from content and metadata.
    
    Args:
        content: Document content
        metadata: Optional metadata dict
        
    Returns:
        Unique document ID (SHA256 hash)
        
    Raises:
        ValueError: If content is empty
    """
    if not content or not isinstance(content, str):
        raise ValueError("Content must be a non-empty string")
    
    # Combine content and metadata for hashing
    hash_input = content
    if metadata:
        # Sort metadata keys for consistent hashing
        metadata_str = json.dumps(metadata, sort_keys=True)
        hash_input = f"{content}|{metadata_str}"
    
    # Generate SHA256 hash
    doc_id = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    
    return doc_id


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
        
    Raises:
        ValueError: If chunk_size is invalid
    """
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    
    if not items:
        return []
    
    chunks = []
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i:i + chunk_size])
    
    return chunks


def calculate_storage_size(storage_path: Path) -> int:
    """
    Calculate total size of files in storage directory.
    
    Args:
        storage_path: Path to storage directory
        
    Returns:
        Total size in bytes
        
    Raises:
        ValueError: If path is invalid
    """
    if not isinstance(storage_path, Path):
        storage_path = Path(storage_path)
    
    if not storage_path.exists():
        raise ValueError(f"Storage path does not exist: {storage_path}")
    
    if not storage_path.is_dir():
        raise ValueError(f"Storage path is not a directory: {storage_path}")
    
    total_size = 0
    for file_path in storage_path.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size


def format_bytes(size_bytes: int) -> str:
    """
    Format byte size into human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
        
    Raises:
        ValueError: If size_bytes is negative
    """
    if not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError("size_bytes must be a non-negative integer")
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    unit_index = 0
    
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.2f} {units[unit_index]}"


def merge_filters(filter1: Optional[Dict[str, Any]], filter2: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge two filter dictionaries.
    
    Args:
        filter1: First filter dict
        filter2: Second filter dict
        
    Returns:
        Merged filter dict
    """
    if filter1 is None and filter2 is None:
        return {}
    
    if filter1 is None:
        return filter2.copy() if filter2 else {}
    
    if filter2 is None:
        return filter1.copy()
    
    # Merge filters (filter2 takes precedence)
    merged = filter1.copy()
    merged.update(filter2)
    
    return merged


def validate_collection_name(name: str) -> bool:
    """
    Validate collection name according to naming rules.
    
    Args:
        name: Collection name to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValueError("Collection name must be a non-empty string")
    
    # Check alphanumeric, underscore, hyphen only
    if not all(c.isalnum() or c in ('_', '-') for c in name):
        raise ValueError(
            "Collection name must contain only alphanumeric characters, "
            "underscores, and hyphens"
        )
    
    # Check length
    if len(name) > 255:
        raise ValueError("Collection name must be 255 characters or less")
    
    return True

"""
Embedded document data model.

This module contains the EmbeddedDocument dataclass which represents
a document with its embedding vector and associated metadata.
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddedDocument:
    """
    Document with its embedding vector and metadata.

    This is a pure data class that holds a document along with its
    computed embedding vector and related information.

    Attributes:
        document: The original document object (Haystack Document).
        embedding: Numpy array containing the embedding vector.
        embedding_model: Name of the model used to generate the embedding.
        embedding_dimension: Dimension of the embedding vector.
    """

    document: Any  # Haystack Document type
    embedding: np.ndarray
    embedding_model: str
    embedding_dimension: int

    def __post_init__(self) -> None:
        """Validate embedded document attributes."""
        if self.document is None:
            raise ValueError("Document cannot be None")

        if self.embedding is None:
            raise ValueError("Embedding cannot be None")

        if not isinstance(self.embedding, np.ndarray):
            raise TypeError("Embedding must be a numpy array")

        if self.embedding.size == 0:
            raise ValueError("Embedding cannot be empty")

        if self.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")

        actual_dimension = self.embedding.shape[0] if self.embedding.ndim == 1 else self.embedding.size
        if actual_dimension != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {actual_dimension}"
            )

        if not self.embedding_model or not self.embedding_model.strip():
            raise ValueError("Embedding model name cannot be empty")

        logger.debug(
            "EmbeddedDocument created: model=%s, dimension=%d",
            self.embedding_model,
            self.embedding_dimension,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for storage/serialization.

        Returns:
            Dictionary containing document content, metadata, and embedding.
        """
        return {
            "content": self.document.content,
            "meta": self.document.meta if hasattr(self.document, "meta") else {},
            "embedding": self.embedding.tolist(),
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
        }

    @property
    def content(self) -> str:
        """Get the document content."""
        return self.document.content if hasattr(self.document, "content") else ""

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the document metadata."""
        return self.document.meta if hasattr(self.document, "meta") else {}

    def __repr__(self) -> str:
        """String representation of the embedded document."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (
            f"EmbeddedDocument("
            f"model='{self.embedding_model}', "
            f"dimension={self.embedding_dimension}, "
            f"content='{content_preview}')"
        )

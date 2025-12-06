"""
Embeddings module for Cerebrus AI.

This module provides a clean, modular architecture for generating
embeddings using various providers (Haystack, OpenAI, Cohere, etc.).

Main exports:
- EmbedderFactory: Factory for creating embedder instances
- EmbeddingConfig: Configuration management
- EmbeddedDocument: Document with embedding
- Document utilities: Helper functions for document creation

Example usage:
    >>> from src.embeddings import EmbedderFactory
    >>> 
    >>> # Create document embedder
    >>> doc_embedder = EmbedderFactory.create_document_embedder()
    >>> 
    >>> # Create query embedder
    >>> query_embedder = EmbedderFactory.create_query_embedder()
"""

from src.embeddings.config import EmbeddingConfig
from src.embeddings.factories import EmbedderFactory
from src.embeddings.models import EmbeddedDocument
from src.embeddings.services import BatchProcessor, DocumentEmbedder, QueryEmbedder
from src.embeddings.utils import (
    create_documents_from_texts,
    extract_metadata_from_documents,
    extract_texts_from_documents,
    validate_documents,
)

__all__ = [
    # Factory
    "EmbedderFactory",
    # Configuration
    "EmbeddingConfig",
    # Models
    "EmbeddedDocument",
    # Services
    "DocumentEmbedder",
    "QueryEmbedder",
    "BatchProcessor",
    # Utilities
    "create_documents_from_texts",
    "validate_documents",
    "extract_texts_from_documents",
    "extract_metadata_from_documents",
]

__version__ = "2.0.0"


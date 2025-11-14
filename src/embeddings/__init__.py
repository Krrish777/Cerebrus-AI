"""
Embeddings Module for Cerebrus AI

This module provides comprehensive embedding generation capabilities
using state-of-the-art transformer models through Haystack.
"""

from .embedding_generator import (
    EmbeddingGenerator,
    EmbeddedDocument,
    create_embedding_generator,
    embed_documents_simple,
    embed_query_simple
)

__all__ = [
    'EmbeddingGenerator',
    'EmbeddedDocument',
    'create_embedding_generator',
    'embed_documents_simple',
    'embed_query_simple'
]

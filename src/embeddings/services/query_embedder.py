"""
Query embedding service.

This module provides the service layer for query embedding operations.
It handles query text embedding for retrieval and search operations.
"""

import numpy as np

from src.core.logging import get_logger
from src.embeddings.providers.base import EmbeddingProvider

logger = get_logger(__name__)


class QueryEmbedder:
    """
    Service for embedding query text.

    This service handles query embedding operations using an injected
    embedding provider. It validates input, logs operations, and
    handles errors while delegating embedding to the provider.

    Example:
        provider = HaystackEmbeddingProvider(model_config)
        provider.warm_up()
        embedder = QueryEmbedder(provider)
        query_embedding = embedder.embed("What is machine learning?")
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        """
        Initialize the query embedder.

        Args:
            provider: Embedding provider instance to use for generating embeddings.

        Raises:
            TypeError: If provider is not an EmbeddingProvider instance.
        """
        if not isinstance(provider, EmbeddingProvider):
            raise TypeError("Provider must be an instance of EmbeddingProvider")

        self._provider = provider
        logger.debug("QueryEmbedder initialized with provider: %s", type(provider).__name__)

    def embed(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string.

        Args:
            query: Query text to embed.

        Returns:
            Numpy array containing the query embedding.

        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If embedding generation fails.
        """
        if not query:
            raise ValueError("Query cannot be empty")

        if not isinstance(query, str):
            raise TypeError("Query must be a string")

        query_stripped = query.strip()
        if not query_stripped:
            raise ValueError("Query cannot be only whitespace")

        logger.info(
            "Generating query embedding: %s",
            query_stripped[:50] + ("..." if len(query_stripped) > 50 else ""),
        )

        try:
            embedding = self._provider.embed_query(query_stripped)

            logger.debug("Successfully generated query embedding: shape=%s", embedding.shape)
            return embedding

        except ValueError:
            raise
        except Exception as error:
            logger.error("Failed to generate query embedding: %s", error)
            raise RuntimeError(f"Failed to generate query embedding: {error}") from error

    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.

        Returns:
            Dictionary containing model information.
        """
        return self._provider.get_model_info()

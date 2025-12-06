"""
Base provider protocol for embedding generation.

This module defines the abstract interface that all embedding providers
must implement. This enables swapping providers without changing client code.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    This protocol defines the interface that all embedding providers
    (Haystack, OpenAI, Cohere, etc.) must implement. It enables
    dependency injection and provider swapping.

    All providers must implement:
    - warm_up(): Initialize the provider and load models
    - embed_documents(): Generate embeddings for a list of documents
    - embed_query(): Generate embedding for a single query string
    - get_embedding_dimension(): Return the dimension of embeddings
    - get_model_info(): Return information about the model
    """

    @abstractmethod
    def warm_up(self) -> None:
        """
        Initialize the provider and load required models.

        This method should be called before any embedding operations.
        It handles model loading, device setup, and any other
        initialization required by the provider.

        Raises:
            RuntimeError: If initialization fails.
        """
        pass

    @abstractmethod
    def embed_documents(self, documents: List[Any]) -> List[Any]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of Document objects to embed.

        Returns:
            List of Document objects with embeddings added.

        Raises:
            ValueError: If documents list is empty or invalid.
            RuntimeError: If embedding generation fails.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query text.

        Args:
            text: Query text to embed.

        Returns:
            Numpy array containing the query embedding.

        Raises:
            ValueError: If text is empty or invalid.
            RuntimeError: If embedding generation fails.
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.

        Returns:
            Integer dimension of embedding vectors.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.

        Returns:
            Dictionary containing model information such as:
            - model_name: Name of the model
            - dimension: Embedding dimension
            - device: Device being used
            - normalize: Whether embeddings are normalized
            - provider: Name of the provider
        """
        pass

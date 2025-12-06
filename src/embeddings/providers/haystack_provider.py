"""
Haystack embedding provider implementation.

This module provides a concrete implementation of the EmbeddingProvider
protocol using Haystack's Sentence Transformers components.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from haystack.utils import ComponentDevice

from src.core.logging import get_logger
from src.embeddings.config import ModelConfig
from src.embeddings.providers.base import EmbeddingProvider

logger = get_logger(__name__)


class HaystackEmbeddingProvider(EmbeddingProvider):
    """
    Haystack-based embedding provider implementation.

    This provider uses Haystack's SentenceTransformersDocumentEmbedder
    and SentenceTransformersTextEmbedder to generate embeddings.

    The provider is initialized with a ModelConfig and handles all
    Haystack-specific implementation details.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize the Haystack embedding provider.

        Args:
            config: Model configuration containing model name, device, etc.

        Raises:
            ImportError: If Haystack is not installed.
            ValueError: If configuration is invalid.
        """
        self._config = config
        self._document_embedder: Optional[Any] = None
        self._text_embedder: Optional[Any] = None
        self._embedding_dimension: Optional[int] = None
        self._is_warmed_up = False

        logger.debug(
            "Initializing HaystackEmbeddingProvider with model: %s",
            self._config.name,
        )

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the model configuration."""
        if not self._config.name or not self._config.name.strip():
            raise ValueError("Model name cannot be empty")

        logger.debug("Configuration validated for model: %s", self._config.name)

    def warm_up(self) -> None:
        """
        Initialize Haystack embedding components and load models.

        Raises:
            ImportError: If Haystack is not installed.
            RuntimeError: If model loading fails.
        """
        if self._is_warmed_up:
            logger.debug("Provider already warmed up")
            return

        logger.info("Warming up HaystackEmbeddingProvider with model: %s", self._config.name)

        try:
            # Import Haystack components
            try:
                from haystack.components.embedders import (
                    SentenceTransformersDocumentEmbedder,
                    SentenceTransformersTextEmbedder,
                )
            except ImportError as error:
                raise ImportError(
                    "Haystack is not installed. Install it with: pip install haystack-ai"
                ) from error

            # Build embedder kwargs
            embedder_kwargs = {
                "model": self._config.name,
                "normalize_embeddings": self._config.normalize_embeddings,
            }

            if self._config.device is not None:
                # Convert device string to ComponentDevice
                embedder_kwargs["device"] = ComponentDevice.from_str(self._config.device)

            if self._config.prefix is not None:
                embedder_kwargs["prefix"] = self._config.prefix

            # Initialize document embedder
            logger.debug("Initializing document embedder with kwargs: %s", embedder_kwargs)
            self._document_embedder = SentenceTransformersDocumentEmbedder(**embedder_kwargs)

            # Initialize text embedder
            text_kwargs = {
                "model": self._config.name,
                "normalize_embeddings": self._config.normalize_embeddings,
            }

            if self._config.device is not None:
                # Convert device string to ComponentDevice
                text_kwargs["device"] = ComponentDevice.from_str(self._config.device)

            if self._config.prefix is not None:
                text_kwargs["prefix"] = self._config.prefix

            logger.debug("Initializing text embedder with kwargs: %s", text_kwargs)
            self._text_embedder = SentenceTransformersTextEmbedder(**text_kwargs)

            # Warm up the embedders
            logger.debug("Warming up document embedder")
            self._document_embedder.warm_up()

            logger.debug("Warming up text embedder")
            self._text_embedder.warm_up()

            # Determine embedding dimension
            test_result = self._text_embedder.run("test")
            self._embedding_dimension = len(test_result["embedding"])

            self._is_warmed_up = True

            logger.info(
                "HaystackEmbeddingProvider warmed up successfully: model=%s, dimension=%d, device=%s",
                self._config.name,
                self._embedding_dimension,
                self._config.device or "auto",
            )

        except ImportError:
            raise
        except Exception as error:
            logger.error("Failed to warm up HaystackEmbeddingProvider: %s", error)
            raise RuntimeError(f"Failed to initialize embedding provider: {error}") from error

    def embed_documents(self, documents: List[Any]) -> List[Any]:
        """
        Generate embeddings for a list of documents using Haystack.

        Args:
            documents: List of Haystack Document objects.

        Returns:
            List of Haystack Document objects with embeddings added.

        Raises:
            ValueError: If documents list is empty or invalid.
            RuntimeError: If not warmed up or embedding fails.
        """
        if not self._is_warmed_up:
            raise RuntimeError("Provider not warmed up. Call warm_up() first.")

        if not documents:
            raise ValueError("Documents list cannot be empty")

        if not isinstance(documents, list):
            raise TypeError("Documents must be a list")

        logger.debug("Embedding %d documents", len(documents))

        try:
            result = self._document_embedder.run(documents)
            embedded_documents = result["documents"]

            logger.debug("Successfully embedded %d documents", len(embedded_documents))
            return embedded_documents

        except Exception as error:
            logger.error("Failed to embed documents: %s", error)
            raise RuntimeError(f"Failed to embed documents: {error}") from error

    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a query text using Haystack.

        Args:
            text: Query text to embed.

        Returns:
            Numpy array containing the query embedding.

        Raises:
            ValueError: If text is empty or invalid.
            RuntimeError: If not warmed up or embedding fails.
        """
        if not self._is_warmed_up:
            raise RuntimeError("Provider not warmed up. Call warm_up() first.")

        if not text or not text.strip():
            raise ValueError("Query text cannot be empty")

        if not isinstance(text, str):
            raise TypeError("Query text must be a string")

        logger.debug("Embedding query text: %s", text[:50] + ("..." if len(text) > 50 else ""))

        try:
            result = self._text_embedder.run(text)
            embedding = np.array(result["embedding"], dtype=np.float32)

            logger.debug("Successfully embedded query: shape=%s", embedding.shape)
            return embedding

        except Exception as error:
            logger.error("Failed to embed query: %s", error)
            raise RuntimeError(f"Failed to embed query: {error}") from error

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings.

        Returns:
            Integer dimension of embedding vectors.

        Raises:
            RuntimeError: If provider not warmed up.
        """
        if not self._is_warmed_up:
            raise RuntimeError("Provider not warmed up. Call warm_up() first.")

        if self._embedding_dimension is None:
            raise RuntimeError("Embedding dimension not initialized")

        return self._embedding_dimension

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.

        Returns:
            Dictionary containing model information.
        """
        return {
            "provider": "haystack",
            "model_name": self._config.name,
            "dimension": self._embedding_dimension if self._is_warmed_up else None,
            "device": self._config.device,
            "normalize_embeddings": self._config.normalize_embeddings,
            "prefix": self._config.prefix,
            "is_warmed_up": self._is_warmed_up,
        }

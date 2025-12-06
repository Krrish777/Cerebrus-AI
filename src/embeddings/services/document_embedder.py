"""
Document embedding service.

This module provides the service layer for document embedding operations.
It orchestrates the embedding provider to generate document embeddings.
"""

from typing import List, Any

from src.core.logging import get_logger
from src.embeddings.models import EmbeddedDocument
from src.embeddings.providers.base import EmbeddingProvider

logger = get_logger(__name__)


class DocumentEmbedder:
    """
    Service for embedding documents.

    This service orchestrates document embedding operations using
    an injected embedding provider. It handles validation, logging,
    and error handling while delegating the actual embedding to the provider.

    Example:
        provider = HaystackEmbeddingProvider(model_config)
        provider.warm_up()
        embedder = DocumentEmbedder(provider)
        embedded_docs = embedder.embed(documents)
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        """
        Initialize the document embedder.

        Args:
            provider: Embedding provider instance to use for generating embeddings.

        Raises:
            TypeError: If provider is not an EmbeddingProvider instance.
        """
        if not isinstance(provider, EmbeddingProvider):
            raise TypeError("Provider must be an instance of EmbeddingProvider")

        self._provider = provider
        logger.debug("DocumentEmbedder initialized with provider: %s", type(provider).__name__)

    def embed(self, documents: List[Any]) -> List[EmbeddedDocument]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of Document objects to embed.

        Returns:
            List of EmbeddedDocument objects with embeddings.

        Raises:
            ValueError: If documents list is empty or invalid.
            RuntimeError: If embedding generation fails.
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        if not isinstance(documents, list):
            raise TypeError("Documents must be a list")

        logger.info("Generating embeddings for %d documents", len(documents))

        try:
            # Generate embeddings using provider
            embedded_docs = self._provider.embed_documents(documents)

            # Get model info
            model_info = self._provider.get_model_info()
            model_name = model_info.get("model_name", "unknown")
            dimension = self._provider.get_embedding_dimension()

            # Convert to EmbeddedDocument objects
            result = []
            for doc in embedded_docs:
                if not hasattr(doc, "embedding") or doc.embedding is None:
                    logger.warning(
                        "Document has no embedding: %s",
                        doc.content[:50] + ("..." if len(doc.content) > 50 else ""),
                    )
                    continue

                import numpy as np

                embedding_array = np.array(doc.embedding, dtype=np.float32)

                embedded_doc = EmbeddedDocument(
                    document=doc,
                    embedding=embedding_array,
                    embedding_model=model_name,
                    embedding_dimension=dimension,
                )
                result.append(embedded_doc)

            logger.info("Successfully generated %d embeddings", len(result))
            return result

        except ValueError:
            raise
        except Exception as error:
            logger.error("Failed to generate document embeddings: %s", error)
            raise RuntimeError(f"Failed to generate document embeddings: {error}") from error

    def embed_single(self, document: Any) -> EmbeddedDocument:
        """
        Generate embedding for a single document.

        Args:
            document: Document object to embed.

        Returns:
            EmbeddedDocument with embedding.

        Raises:
            ValueError: If document is invalid.
            RuntimeError: If embedding generation fails.
        """
        if document is None:
            raise ValueError("Document cannot be None")

        logger.debug("Generating embedding for single document")

        embedded_docs = self.embed([document])

        if not embedded_docs:
            raise RuntimeError("Failed to generate embedding for document")

        return embedded_docs[0]

"""
Batch processing service for document embeddings.

This module provides batch processing capabilities for large-scale
document embedding operations.
"""

from typing import List, Any

from src.core.logging import get_logger
from src.embeddings.config import ProcessingConfig
from src.embeddings.models import EmbeddedDocument
from src.embeddings.services.document_embedder import DocumentEmbedder

logger = get_logger(__name__)


class BatchProcessor:
    """
    Service for batch processing of document embeddings.

    This service handles batch processing of large document sets,
    managing batch sizes and error handling for robust processing.

    Example:
        provider = HaystackEmbeddingProvider(model_config)
        provider.warm_up()
        embedder = DocumentEmbedder(provider)
        processor = BatchProcessor(embedder, processing_config)
        results = processor.process_batches(document_batches)
    """

    def __init__(
        self,
        document_embedder: DocumentEmbedder,
        config: ProcessingConfig,
    ) -> None:
        """
        Initialize the batch processor.

        Args:
            document_embedder: DocumentEmbedder instance for embedding documents.
            config: Processing configuration for batch settings.

        Raises:
            TypeError: If parameters are not of correct type.
        """
        if not isinstance(document_embedder, DocumentEmbedder):
            raise TypeError("document_embedder must be a DocumentEmbedder instance")

        if not isinstance(config, ProcessingConfig):
            raise TypeError("config must be a ProcessingConfig instance")

        self._embedder = document_embedder
        self._config = config

        logger.debug(
            "BatchProcessor initialized: batch_size=%d, max_retries=%d",
            self._config.batch_size,
            self._config.max_retries,
        )

    def process_batches(
        self,
        document_batches: List[List[Any]],
    ) -> List[List[EmbeddedDocument]]:
        """
        Process multiple batches of documents.

        Args:
            document_batches: List of document batches to process.

        Returns:
            List of embedded document batches.

        Raises:
            ValueError: If document_batches is empty or invalid.
        """
        if not document_batches:
            raise ValueError("Document batches list cannot be empty")

        if not isinstance(document_batches, list):
            raise TypeError("document_batches must be a list")

        logger.info("Processing %d document batches", len(document_batches))

        all_embedded_batches = []

        for batch_index, batch in enumerate(document_batches):
            batch_num = batch_index + 1
            logger.info(
                "Processing batch %d/%d (%d documents)",
                batch_num,
                len(document_batches),
                len(batch),
            )

            try:
                embedded_batch = self._embedder.embed(batch)
                all_embedded_batches.append(embedded_batch)

                logger.debug("Batch %d processed successfully", batch_num)

            except Exception as error:
                logger.error("Failed to process batch %d: %s", batch_num, error)
                # Continue with other batches instead of failing completely
                all_embedded_batches.append([])

        total_docs = sum(len(batch) for batch in all_embedded_batches)
        logger.info("Batch processing complete: %d documents embedded", total_docs)

        return all_embedded_batches

    def process_documents_in_batches(
        self,
        documents: List[Any],
    ) -> List[EmbeddedDocument]:
        """
        Process a list of documents by splitting into batches.

        Args:
            documents: List of documents to process.

        Returns:
            List of embedded documents.

        Raises:
            ValueError: If documents list is empty or invalid.
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        if not isinstance(documents, list):
            raise TypeError("documents must be a list")

        batch_size = self._config.batch_size
        total_documents = len(documents)

        logger.info(
            "Processing %d documents in batches of %d",
            total_documents,
            batch_size,
        )

        # Split documents into batches
        batches = [
            documents[i : i + batch_size]
            for i in range(0, total_documents, batch_size)
        ]

        logger.debug("Split into %d batches", len(batches))

        # Process batches
        embedded_batches = self.process_batches(batches)

        # Flatten results
        all_embedded = []
        for embedded_batch in embedded_batches:
            all_embedded.extend(embedded_batch)

        logger.info("Successfully processed %d documents", len(all_embedded))

        return all_embedded

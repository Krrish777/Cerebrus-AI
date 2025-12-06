"""
Document utility functions.

This module provides pure utility functions for working with documents.
No state, no classes - just helper functions.
"""

from typing import Any, Dict, List, Optional

from src.core.logging import get_logger

logger = get_logger(__name__)


def create_documents_from_texts(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
) -> List[Any]:
    """
    Create Haystack Document objects from text strings.

    Args:
        texts: List of text strings.
        metadatas: Optional list of metadata dictionaries.
                  If None, empty dicts are used.

    Returns:
        List of Haystack Document objects.

    Raises:
        ImportError: If Haystack is not installed.
        ValueError: If texts is empty or lengths don't match.
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")

    if not isinstance(texts, list):
        raise TypeError("texts must be a list")

    # Import Document
    try:
        from haystack import Document
    except ImportError as error:
        raise ImportError(
            "Haystack is not installed. Install it with: pip install haystack-ai"
        ) from error

    # Handle metadatas
    if metadatas is None:
        metadatas = [{}] * len(texts)
    else:
        if len(texts) != len(metadatas):
            raise ValueError(
                f"Number of texts ({len(texts)}) and metadatas ({len(metadatas)}) must match"
            )

    logger.debug("Creating %d documents from texts", len(texts))

    documents = []
    for index, (text, meta) in enumerate(zip(texts, metadatas)):
        # Validate text
        if not isinstance(text, str):
            raise TypeError(f"Text at index {index} must be a string, got {type(text)}")

        # Copy metadata to avoid mutation
        meta_copy = meta.copy() if meta else {}
        meta_copy["doc_index"] = index

        # Create document
        doc = Document(content=text, meta=meta_copy)
        documents.append(doc)

    logger.info("Created %d documents from texts", len(documents))
    return documents


def validate_documents(documents: List[Any]) -> bool:
    """
    Validate a list of documents.

    Args:
        documents: List of Document objects to validate.

    Returns:
        True if all documents are valid.

    Raises:
        ValueError: If documents list is invalid.
        TypeError: If documents are not proper Document objects.
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")

    if not isinstance(documents, list):
        raise TypeError("documents must be a list")

    logger.debug("Validating %d documents", len(documents))

    for index, doc in enumerate(documents):
        # Check for content attribute
        if not hasattr(doc, "content"):
            raise TypeError(f"Document at index {index} has no 'content' attribute")

        # Check content is not None
        if doc.content is None:
            raise ValueError(f"Document at index {index} has None content")

        # Check content is string
        if not isinstance(doc.content, str):
            raise TypeError(
                f"Document content at index {index} must be string, got {type(doc.content)}"
            )

        # Check for meta attribute (optional but common)
        if not hasattr(doc, "meta"):
            logger.warning("Document at index %d has no 'meta' attribute", index)

    logger.debug("All %d documents validated successfully", len(documents))
    return True


def extract_texts_from_documents(documents: List[Any]) -> List[str]:
    """
    Extract text content from a list of documents.

    Args:
        documents: List of Document objects.

    Returns:
        List of text strings extracted from documents.

    Raises:
        ValueError: If documents list is empty.
        TypeError: If documents don't have content attribute.
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")

    if not isinstance(documents, list):
        raise TypeError("documents must be a list")

    logger.debug("Extracting texts from %d documents", len(documents))

    texts = []
    for index, doc in enumerate(documents):
        if not hasattr(doc, "content"):
            raise TypeError(f"Document at index {index} has no 'content' attribute")

        texts.append(doc.content)

    logger.debug("Extracted %d texts from documents", len(texts))
    return texts


def extract_metadata_from_documents(documents: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract metadata from a list of documents.

    Args:
        documents: List of Document objects.

    Returns:
        List of metadata dictionaries.

    Raises:
        ValueError: If documents list is empty.
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")

    if not isinstance(documents, list):
        raise TypeError("documents must be a list")

    logger.debug("Extracting metadata from %d documents", len(documents))

    metadatas = []
    for doc in documents:
        if hasattr(doc, "meta") and doc.meta is not None:
            metadatas.append(doc.meta)
        else:
            metadatas.append({})

    logger.debug("Extracted metadata from %d documents", len(metadatas))
    return metadatas

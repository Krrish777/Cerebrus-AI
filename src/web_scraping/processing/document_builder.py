"""
Document Builder Implementation.

This module provides document building for scraped web content.
Creates Haystack Documents from ScrapedContent with appropriate metadata.

Following AGENTS.md principles:
    - Single responsibility: Only document building
    - Integration: Uses existing ChunkingService from document_processing
"""

import hashlib
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from haystack import Document

from src.core.logging import get_logger
from src.web_scraping.interfaces import DocumentBuilder
from src.web_scraping.interfaces import ScrapedContent

logger = get_logger(__name__)


class DefaultDocumentBuilder(DocumentBuilder):
    """
    Default implementation of document building.

    Converts ScrapedContent into Haystack Documents with
    comprehensive metadata for retrieval and processing.

    Example:
        builder = DefaultDocumentBuilder()
        document = builder.build(scraped_content)
    """

    def __init__(self, include_links_in_meta: bool = True) -> None:
        """
        Initialize the document builder.

        Args:
            include_links_in_meta: Whether to include extracted links in metadata.
        """
        self._include_links_in_meta = include_links_in_meta
        logger.debug(
            "Document builder initialized: include_links=%s",
            self._include_links_in_meta,
        )

    def build(self, scraped_content: ScrapedContent) -> Document:
        """
        Build Haystack Document from scraped content.

        Args:
            scraped_content: Scraped content with metadata.

        Returns:
            Haystack Document with comprehensive metadata.
        """
        # Generate document ID
        doc_id = self._generate_document_id(scraped_content)

        # Build metadata
        metadata = self._build_metadata(scraped_content)

        # Create document
        document = Document(
            id=doc_id,
            content=scraped_content.content,
            meta=metadata,
        )

        logger.debug(
            "Built document from %s: id=%s, content_length=%d",
            scraped_content.url,
            doc_id,
            len(scraped_content.content),
        )

        return document

    def build_batch(self, scraped_contents: List[ScrapedContent]) -> List[Document]:
        """
        Build multiple documents from scraped contents.

        Args:
            scraped_contents: List of scraped contents.

        Returns:
            List of Haystack Documents.
        """
        documents = []

        for content in scraped_contents:
            try:
                document = self.build(content)
                documents.append(document)
            except Exception as error:
                logger.warning(
                    "Failed to build document from %s: %s",
                    content.url,
                    error,
                )

        logger.info(
            "Built %d documents from %d scraped contents",
            len(documents),
            len(scraped_contents),
        )

        return documents

    def _generate_document_id(self, scraped_content: ScrapedContent) -> str:
        """
        Generate a unique document ID.

        Uses URL and content hash to create a stable ID.

        Args:
            scraped_content: Scraped content.

        Returns:
            Document ID string.
        """
        # Combine URL and content for unique ID
        id_source = f"{scraped_content.url}:{scraped_content.scraped_at.isoformat()}"
        content_hash = hashlib.sha256(id_source.encode("utf-8")).hexdigest()[:12]

        return f"web_{content_hash}"

    def _build_metadata(self, scraped_content: ScrapedContent) -> Dict[str, Any]:
        """
        Build comprehensive metadata from scraped content.

        Args:
            scraped_content: Scraped content.

        Returns:
            Metadata dictionary.
        """
        metadata: Dict[str, Any] = {
            # Source information
            "url": scraped_content.url,
            "source_type": "web",
            "title": scraped_content.title,
            "description": scraped_content.description,
            
            # Content metrics
            "content_length": len(scraped_content.content),
            "word_count": scraped_content.word_count,
            "content_type": scraped_content.content_type,
            
            # Timestamps
            "scraped_at": scraped_content.scraped_at.isoformat(),
            "processing_timestamp": time.time(),
            
            # Content hash for deduplication
            "content_hash": self._generate_content_hash(scraped_content.content),
        }

        # Include links if enabled
        if self._include_links_in_meta and scraped_content.links:
            metadata["links"] = scraped_content.links
            metadata["link_count"] = len(scraped_content.links)

        # Merge with scraped metadata
        if scraped_content.metadata:
            for key, value in scraped_content.metadata.items():
                # Don't overwrite existing keys
                if key not in metadata:
                    metadata[key] = value

        return metadata

    def _generate_content_hash(self, content: str) -> str:
        """
        Generate a hash of the content for deduplication.

        Args:
            content: Content to hash.

        Returns:
            Content hash string.
        """
        if not content:
            return "empty"

        return hashlib.md5(content.encode("utf-8")).hexdigest()[:8]

"""
Document ingestion service.
Handles adding documents to document stores.
"""

from typing import Any, Dict, List

from haystack.dataclasses import Document

from src.core.logging import get_logger
from src.rag.providers.base import DocumentStoreProvider

logger = get_logger(__name__)


class DocumentIngestionService:
    """Service for document ingestion operations."""
    
    def __init__(self, document_store: DocumentStoreProvider):
        """
        Initialize document ingestion service.
        
        Args:
            document_store: Document store provider
        """
        self.document_store = document_store
        logger.info("Initialized DocumentIngestionService")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add documents to the document store.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Number of documents successfully added
        """
        if not documents:
            logger.warning("No documents provided for ingestion")
            return 0
        
        try:
            # Convert to Haystack Documents
            haystack_docs = self._convert_to_haystack_documents(documents)
            
            # Write to store
            count = self.document_store.write_documents(haystack_docs)
            
            logger.info(f"Successfully ingested {count} documents")
            return count
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            raise
    
    def _convert_to_haystack_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Document]:
        """
        Convert various document formats to Haystack Documents.
        
        Args:
            documents: List of document data
            
        Returns:
            List of Haystack Document objects
        """
        haystack_docs = []
        
        for doc_data in documents:
            try:
                # Handle different document formats
                if isinstance(doc_data, Document):
                    # Already a Haystack Document
                    haystack_docs.append(doc_data)
                    
                elif isinstance(doc_data, dict):
                    # Dictionary format
                    content = doc_data.get('content', '')
                    meta = doc_data.get('metadata', {}) or doc_data.get('meta', {})
                    
                    haystack_doc = Document(content=content, meta=meta)
                    haystack_docs.append(haystack_doc)
                    
                elif hasattr(doc_data, 'document'):
                    # EmbeddedDocument structure (from embeddings module)
                    content = doc_data.document.content
                    meta = doc_data.document.meta or {}
                    
                    haystack_doc = Document(content=content, meta=meta)
                    haystack_docs.append(haystack_doc)
                    
                elif hasattr(doc_data, 'content'):
                    # Direct document object with content
                    content = doc_data.content
                    meta = getattr(doc_data, 'meta', {}) or getattr(doc_data, 'metadata', {})
                    
                    haystack_doc = Document(content=content, meta=meta)
                    haystack_docs.append(haystack_doc)
                    
                else:
                    logger.warning(f"Unsupported document format: {type(doc_data)}")
                    
            except Exception as e:
                logger.error(f"Error converting document: {e}")
                continue
        
        return haystack_docs
    
    def count_documents(self) -> int:
        """
        Count total documents in store.
        
        Returns:
            Total document count
        """
        try:
            count = self.document_store.count_documents()
            logger.debug(f"Document count: {count}")
            return count
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    def clear_documents(self, filters: Dict[str, Any] = None) -> None:
        """
        Delete documents from store.
        
        Args:
            filters: Optional filters for selective deletion
        """
        try:
            self.document_store.delete_documents(filters=filters)
            logger.info("Documents cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            raise

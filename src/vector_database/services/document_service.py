"""
Document Service for vector database operations.

Handles document insertion, retrieval, and deletion with business logic.
Following AGENTS.md: single responsibility, dependency injection, fail-fast validation.
"""

from typing import List, Optional, Any, Dict
from pathlib import Path

from src.core.logging import get_logger
from src.vector_database.providers.base_provider import (
    BaseVectorDBProvider,
    InsertionError,
    SearchError,
    DeletionError
)

logger = get_logger(__name__)


class DocumentService:
    """
    Service for managing documents in the vector database.
    
    Responsibilities:
    - Document insertion with validation
    - Document retrieval by ID
    - Document deletion (single and batch)
    - Document existence checking
    
    Design:
    - Depends on BaseVectorDBProvider interface (loose coupling)
    - Validates inputs before delegating to provider (defensibility)
    - Logs operations for observability (maintainability)
    """
    
    def __init__(self, provider: BaseVectorDBProvider):
        """
        Initialize DocumentService.
        
        Args:
            provider: Vector database provider implementing BaseVectorDBProvider
            
        Raises:
            ValueError: If provider is None
        """
        if provider is None:
            raise ValueError("Provider cannot be None")
        
        self.provider = provider
        logger.info(f"DocumentService initialized with {provider.__class__.__name__}")
    
    def insert_documents(
        self,
        documents: List[Any],
        policy: str = "skip"
    ) -> Dict[str, Any]:
        """
        Insert documents into the vector database.
        
        Args:
            documents: List of documents with embeddings
            policy: Write policy - "skip", "overwrite", or "fail"
            
        Returns:
            Dict with 'inserted_ids', 'count', and 'policy' keys
            
        Raises:
            ValueError: If documents or policy invalid
            InsertionError: If insertion fails
        """
        # Validate inputs
        if not isinstance(documents, list):
            raise ValueError("Documents must be a list")
        
        if not documents:
            logger.warning("No documents provided for insertion")
            return {
                'inserted_ids': [],
                'count': 0,
                'policy': policy
            }
        
        if policy not in ("skip", "overwrite", "fail"):
            raise ValueError(f"Invalid policy '{policy}', must be 'skip', 'overwrite', or 'fail'")
        
        try:
            logger.info(f"Inserting {len(documents)} documents with policy '{policy}'")
            inserted_ids = self.provider.insert_documents(documents, policy)
            
            logger.info(f"Successfully inserted {len(inserted_ids)} documents")
            return {
                'inserted_ids': inserted_ids,
                'count': len(inserted_ids),
                'policy': policy
            }
            
        except Exception as e:
            logger.error(f"Document insertion failed: {e}")
            raise InsertionError(f"Failed to insert documents: {e}") from e
    
    def get_document_by_id(self, doc_id: str) -> Optional[Any]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document if found, None otherwise
            
        Raises:
            ValueError: If doc_id is empty
            SearchError: If retrieval fails
        """
        if not doc_id or not isinstance(doc_id, str):
            raise ValueError("Document ID must be a non-empty string")
        
        try:
            logger.debug(f"Retrieving document with ID: {doc_id}")
            document = self.provider.get_document_by_id(doc_id)
            
            if document:
                logger.debug(f"Document {doc_id} found")
            else:
                logger.debug(f"Document {doc_id} not found")
            
            return document
            
        except Exception as e:
            logger.error(f"Document retrieval failed for ID {doc_id}: {e}")
            raise SearchError(f"Failed to retrieve document: {e}") from e
    
    def delete_documents(self, doc_ids: List[str]) -> int:
        """
        Delete documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
            
        Raises:
            ValueError: If doc_ids invalid
            DeletionError: If deletion fails
        """
        if not isinstance(doc_ids, list):
            raise ValueError("Document IDs must be a list")
        
        if not doc_ids:
            logger.warning("No document IDs provided for deletion")
            return 0
        
        # Validate all IDs are non-empty strings
        for doc_id in doc_ids:
            if not doc_id or not isinstance(doc_id, str):
                raise ValueError("All document IDs must be non-empty strings")
        
        try:
            logger.info(f"Deleting {len(doc_ids)} documents")
            deleted_count = self.provider.delete_documents(doc_ids)
            
            logger.info(f"Successfully deleted {deleted_count} documents")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            raise DeletionError(f"Failed to delete documents: {e}") from e
    
    def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document exists.
        
        Args:
            doc_id: Document ID to check
            
        Returns:
            True if document exists, False otherwise
            
        Raises:
            ValueError: If doc_id is empty
        """
        if not doc_id or not isinstance(doc_id, str):
            raise ValueError("Document ID must be a non-empty string")
        
        try:
            document = self.provider.get_document_by_id(doc_id)
            return document is not None
            
        except Exception as e:
            logger.warning(f"Error checking document existence for {doc_id}: {e}")
            return False
    
    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents in the collection.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            Number of documents matching filters
            
        Raises:
            ValueError: If filters invalid
        """
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("Filters must be a dictionary")
        
        try:
            count = self.provider.count_documents(filters)
            logger.debug(f"Document count: {count}")
            return count
            
        except Exception as e:
            logger.error(f"Document count failed: {e}")
            raise

"""
Metadata Manager

Manages document and chunk metadata throughout the processing pipeline.
Follows AGENTS.md principles: single responsibility, data consistency, validation.
"""

from typing import Dict, Any, Optional, List
import time
from datetime import datetime

from haystack.dataclasses import Document

from src.core.logging import get_logger
from src.document_processing.pipeline_config import PipelineConfig

logger = get_logger(__name__)


class MetadataManager:
    """
    Manages metadata creation, enhancement, and validation for documents and chunks.
    
    Ensures consistent metadata structure across all processing stages and
    provides utilities for metadata validation and citation generation.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize metadata manager with configuration.

        :param config: Pipeline configuration
        """
        self.config = config
        logger.debug("Metadata manager initialized")

    def enhance_metadata(self, document: Document, file_type: str) -> Document:
        """
        Enhance document metadata with standardized fields and processing information.

        :param document: Document to enhance
        :param file_type: Type of the source file
        :return: Document with enhanced metadata
        """
        logger.debug("Enhancing metadata for document %s", document.id)

        # Start with existing metadata
        enhanced_meta = document.meta.copy()

        # Add standard metadata fields
        standard_metadata = self._create_standard_metadata(document, file_type)
        enhanced_meta.update(standard_metadata)

        # Add processing metadata
        processing_metadata = self._create_processing_metadata()
        enhanced_meta.update(processing_metadata)

        # Add citation metadata
        citation_metadata = self._create_citation_metadata(document, file_type)
        enhanced_meta.update(citation_metadata)

        # Validate and clean metadata
        cleaned_meta = self._clean_metadata(enhanced_meta)

        # Create new document with enhanced metadata
        return Document(
            id=document.id,
            content=document.content,
            meta=cleaned_meta,
            embedding=document.embedding
        )

    def _create_standard_metadata(self, document: Document, file_type: str) -> Dict[str, Any]:
        """
        Create standardized metadata fields.

        :param document: Source document
        :param file_type: Type of source file
        :return: Dictionary of standard metadata fields
        """
        # Extract source file information
        source_file = (
            document.meta.get("file_path") or 
            document.meta.get("name") or 
            document.meta.get("source") or 
            "unknown"
        )

        return {
            self.config.metadata.source_file: source_file,
            self.config.metadata.source_type: file_type.lower(),
            "document_id": document.id,
            "content_length": len(document.content) if document.content else 0,
            "has_content": bool(document.content and document.content.strip())
        }

    def _create_processing_metadata(self) -> Dict[str, Any]:
        """
        Create processing-related metadata.

        :return: Dictionary of processing metadata
        """
        return {
            self.config.metadata.processing_date: datetime.now().isoformat(),
            "processing_timestamp": time.time(),
            "processor_version": "1.0",
            "pipeline_version": "1.0",
            "metadata_schema_version": "1.0"
        }

    def _create_citation_metadata(self, document: Document, file_type: str) -> Dict[str, Any]:
        """
        Create citation and reference metadata.

        :param document: Source document
        :param file_type: Type of source file
        :return: Dictionary of citation metadata
        """
        source_file = document.meta.get("file_path", document.meta.get("name", "unknown"))
        
        citation = {
            "source_file": source_file,
            "document_type": file_type,
            "document_id": document.id,
            "extraction_method": "haystack_pipeline",
            "extraction_timestamp": datetime.now().isoformat()
        }

        # Add page number for PDF files
        if file_type.lower() == "pdf" and "page_number" in document.meta:
            citation[self.config.metadata.page_number] = document.meta["page_number"]

        return {"citation": citation}

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and validate metadata dictionary.

        :param metadata: Raw metadata dictionary
        :return: Cleaned metadata dictionary
        """
        cleaned = {}
        
        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue
                
            # Convert datetime objects to ISO strings
            if hasattr(value, 'isoformat'):
                cleaned[key] = value.isoformat()
            # Ensure JSON serializable
            elif isinstance(value, (str, int, float, bool, list, dict)):
                cleaned[key] = value
            else:
                # Convert to string as fallback
                cleaned[key] = str(value)

        return cleaned

    def create_chunk_metadata(self, chunk: Document, parent_doc: Document, 
                            chunk_index: int, boundaries: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create metadata specifically for document chunks.

        :param chunk: Chunk document
        :param parent_doc: Original parent document
        :param chunk_index: Index of this chunk
        :param boundaries: Boundary detection information
        :return: Chunk-specific metadata
        """
        # Start with parent metadata
        chunk_meta = parent_doc.meta.copy()

        # Add chunk-specific fields
        chunk_specific = {
            self.config.metadata.chunk_index: chunk_index,
            self.config.metadata.chunk_id: chunk.id,
            "parent_document_id": parent_doc.id,
            self.config.metadata.word_count: len(chunk.content.split()) if chunk.content else 0,
            self.config.metadata.line_count: len(chunk.content.split('\n')) if chunk.content else 0,
            "chunk_size": len(chunk.content) if chunk.content else 0
        }

        # Add boundary information if available
        if boundaries:
            chunk_specific.update({
                self.config.metadata.boundary_found: boundaries.get("found", False),
                self.config.metadata.boundary_type: boundaries.get("type", "none"),
                self.config.metadata.start_char: boundaries.get("start_char", 0),
                self.config.metadata.end_char: boundaries.get("end_char", 0)
            })

        # Add content hash
        if chunk.content:
            import hashlib
            content_hash = hashlib.md5(chunk.content.encode()).hexdigest()[:8]
            chunk_specific[self.config.metadata.content_hash] = content_hash

        # Merge with existing metadata
        chunk_meta.update(chunk_specific)

        return self._clean_metadata(chunk_meta)

    def validate_metadata(self, document: Document) -> Dict[str, Any]:
        """
        Validate document metadata for completeness and correctness.

        :param document: Document to validate
        :return: Validation results
        """
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = [
            self.config.metadata.source_file,
            self.config.metadata.source_type
        ]
        
        for field in required_fields:
            if field not in document.meta:
                errors.append(f"Missing required metadata field: {field}")
            elif not document.meta[field]:
                warnings.append(f"Empty required metadata field: {field}")

        # Validate data types
        type_validations = {
            self.config.metadata.word_count: int,
            self.config.metadata.line_count: int,
            self.config.metadata.chunk_index: int
        }
        
        for field, expected_type in type_validations.items():
            if field in document.meta and document.meta[field] is not None:
                if not isinstance(document.meta[field], expected_type):
                    errors.append(f"Invalid type for {field}: expected {expected_type.__name__}")

        # Check for serialization issues
        try:
            import json
            json.dumps(document.meta)
        except (TypeError, ValueError) as e:
            errors.append(f"Metadata not JSON serializable: {str(e)}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "field_count": len(document.meta),
            "required_fields_present": len([f for f in required_fields if f in document.meta])
        }

    def extract_citations(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract citation information from a list of documents.

        :param documents: List of documents
        :return: List of citation dictionaries
        """
        citations = []
        
        for doc in documents:
            citation_info = doc.meta.get("citation", {})
            
            if not citation_info:
                # Create basic citation from available metadata
                citation_info = {
                    "source_file": doc.meta.get(self.config.metadata.source_file, "unknown"),
                    "document_type": doc.meta.get(self.config.metadata.source_type, "unknown"),
                    "document_id": doc.id
                }
            
            # Add chunk information if available
            if self.config.metadata.chunk_index in doc.meta:
                citation_info["chunk_index"] = doc.meta[self.config.metadata.chunk_index]
                citation_info["chunk_id"] = doc.meta.get(self.config.metadata.chunk_id, doc.id)
            
            citations.append(citation_info)
        
        return citations

    def get_metadata_schema(self) -> Dict[str, Any]:
        """
        Get the metadata schema used by this manager.

        :return: Metadata schema definition
        """
        return {
            "version": "1.0",
            "required_fields": [
                self.config.metadata.source_file,
                self.config.metadata.source_type
            ],
            "optional_fields": [
                self.config.metadata.chunk_id,
                self.config.metadata.chunk_index,
                self.config.metadata.page_number,
                self.config.metadata.content_hash,
                self.config.metadata.start_char,
                self.config.metadata.end_char,
                self.config.metadata.word_count,
                self.config.metadata.line_count,
                self.config.metadata.boundary_found,
                self.config.metadata.boundary_type,
                self.config.metadata.processing_date
            ],
            "field_descriptions": {
                self.config.metadata.source_file: "Path to the source file",
                self.config.metadata.source_type: "Type of the source file (pdf, text, markdown)",
                self.config.metadata.chunk_id: "Unique identifier for the chunk",
                self.config.metadata.chunk_index: "Index of the chunk in the document",
                self.config.metadata.page_number: "Page number for PDF documents",
                self.config.metadata.content_hash: "Hash of the chunk content",
                self.config.metadata.word_count: "Number of words in the chunk",
                self.config.metadata.line_count: "Number of lines in the chunk"
            }
        }
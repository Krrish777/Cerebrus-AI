"""
Chunking Service

Intelligent document chunking using Haystack components.
Follows AGENTS.md principles: single responsibility, loose coupling, configurable.
"""

from typing import List, Dict, Any, Optional
import time
import hashlib

from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document

from src.core.logging import get_logger
from src.document_processing.pipeline_config import PipelineConfig

logger = get_logger(__name__)


class ChunkingService:
    """
    Provides intelligent document chunking capabilities.
    
    Uses Haystack's document splitting components with configurable parameters
    for semantic-aware chunking. Handles chunk metadata and boundary detection.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize chunking service with configuration.

        :param config: Pipeline configuration
        """
        self.config = config
        self._splitter = None
        logger.debug("Chunking service initialized")

    @property
    def splitter(self) -> DocumentSplitter:
        """Lazy-loaded document splitter."""
        if self._splitter is None:
            self._splitter = self._create_splitter()
        return self._splitter

    def chunk_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Chunk a list of documents into smaller pieces.

        :param documents: List of documents to chunk
        :return: Chunking results with chunks and errors
        """
        start_time = time.time()
        logger.info("Chunking %d documents", len(documents))

        if not documents:
            logger.warning("No documents provided for chunking")
            return {"documents": [], "errors": []}

        try:
            # Use Haystack splitter
            split_result = self.splitter.run(documents=documents)
            raw_chunks = split_result.get("documents", [])

            # Enhance chunks with additional metadata
            enhanced_chunks = []
            for i, chunk in enumerate(raw_chunks):
                enhanced_chunk = self._enhance_chunk_metadata(chunk, i)
                enhanced_chunks.append(enhanced_chunk)

            chunking_time = time.time() - start_time
            logger.info(
                "Created %d chunks from %d documents in %.2fs",
                len(enhanced_chunks), len(documents), chunking_time
            )

            return {
                "documents": enhanced_chunks,
                "errors": [],
                "stats": {
                    "chunking_time": chunking_time,
                    "input_documents": len(documents),
                    "output_chunks": len(enhanced_chunks),
                    "chunks_per_document": len(enhanced_chunks) / len(documents) if documents else 0
                }
            }

        except Exception as e:
            error_msg = f"Document chunking failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "documents": [],
                "errors": [error_msg],
                "stats": {
                    "chunking_time": time.time() - start_time,
                    "input_documents": len(documents),
                    "output_chunks": 0,
                    "chunks_per_document": 0
                }
            }

    def _create_splitter(self) -> DocumentSplitter:
        """
        Create and configure document splitter.

        :return: Configured DocumentSplitter instance
        """
        logger.debug("Creating document splitter with chunk_size=%d, overlap=%d", 
                    self.config.chunking.chunk_size, self.config.chunking.chunk_overlap)

        # Create splitter with configuration
        splitter = DocumentSplitter(
            split_by="word",
            split_length=self.config.chunking.chunk_size,
            split_overlap=self.config.chunking.chunk_overlap,
        )

        return splitter

    def _enhance_chunk_metadata(self, chunk: Document, chunk_index: int) -> Document:
        """
        Enhance chunk metadata with chunking information.

        :param chunk: Original chunk document
        :param chunk_index: Index of this chunk
        :return: Enhanced chunk with additional metadata
        """
        # Create enhanced metadata
        enhanced_meta = chunk.meta.copy()
        
        # Generate chunk ID
        content_hash = self._generate_content_hash(chunk.content)
        chunk_id = f"chunk_{chunk_index}_{content_hash}"

        # Add chunking metadata
        chunking_metadata = {
            # Identification
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "content_hash": content_hash,
            
            # Source tracking
            "original_document_id": chunk.id,
            "source_file": enhanced_meta.get("file_path", enhanced_meta.get("name", "unknown")),
            
            # Content metrics
            "chunk_size": len(chunk.content) if chunk.content else 0,
            "word_count": len(chunk.content.split()) if chunk.content else 0,
            "line_count": len(chunk.content.split('\n')) if chunk.content else 0,
            
            # Processing metadata
            "chunking_strategy": "word",
            "chunking_timestamp": time.time(),
            "chunking_version": "1.0",
            
            # Configuration used
            "target_chunk_size": self.config.chunking.chunk_size,
            "chunk_overlap": self.config.chunking.chunk_overlap,
        }

        # Merge with existing metadata
        enhanced_meta.update(chunking_metadata)

        # Create new document with enhanced metadata
        return Document(
            id=chunk_id,
            content=chunk.content,
            meta=enhanced_meta,
            embedding=chunk.embedding
        )

    def _generate_content_hash(self, content: Optional[str]) -> str:
        """
        Generate a hash for chunk content.

        :param content: Chunk content
        :return: Content hash string
        """
        if not content:
            return "empty"
        
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]

    def chunk_single_document(self, document: Document) -> List[Document]:
        """
        Chunk a single document.

        :param document: Document to chunk
        :return: List of chunk documents
        """
        result = self.chunk_documents([document])
        return result.get("documents", [])

    def get_chunking_info(self) -> Dict[str, Any]:
        """
        Get information about chunking configuration.

        :return: Chunking information
        """
        return {
            "strategy": "word",
            "chunk_size": self.config.chunking.chunk_size,
            "chunk_overlap": self.config.chunking.chunk_overlap,
            "min_chunk_ratio": self.config.chunking.min_chunk_size_ratio,
            "boundary_preferences": self.config.chunking.boundary_preferences,
            "statistics_enabled": self.config.chunking.enable_statistics,
            "preview_enabled": self.config.chunking.enable_preview,
            "preview_length": self.config.chunking.preview_length
        }

    def validate_chunks(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Validate chunk quality and completeness.

        :param chunks: List of chunk documents to validate
        :return: Validation results
        """
        if not chunks:
            return {
                "valid": False,
                "errors": ["No chunks provided for validation"],
                "stats": {}
            }

        errors = []
        warnings = []
        
        # Check chunk sizes
        chunk_sizes = []
        for i, chunk in enumerate(chunks):
            if not chunk.content:
                errors.append(f"Chunk {i} has no content")
                continue
                
            size = len(chunk.content)
            chunk_sizes.append(size)
            
            # Check minimum size
            min_size = int(self.config.chunking.chunk_size * self.config.chunking.min_chunk_size_ratio)
            if size < min_size:
                warnings.append(f"Chunk {i} is smaller than minimum size ({size} < {min_size})")
            
            # Check for required metadata
            required_fields = ["chunk_id", "chunk_index", "source_file"]
            for field in required_fields:
                if field not in chunk.meta:
                    errors.append(f"Chunk {i} missing required metadata field: {field}")

        # Calculate statistics
        stats = {}
        if chunk_sizes:
            stats = {
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
                "min_chunk_size": min(chunk_sizes),
                "max_chunk_size": max(chunk_sizes),
                "total_content_length": sum(chunk_sizes)
            }

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": stats
        }
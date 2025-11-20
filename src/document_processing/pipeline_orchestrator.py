"""
Pipeline Orchestrator

High-level coordination of document processing pipelines.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time
from datetime import datetime

from src.core.logging import get_logger
from src.document_processing.pipeline_config import get_pipeline_config, PipelineConfig

logger = get_logger(__name__)


class DocumentPipelineOrchestrator:
    """
    High-level orchestrator for document processing operations.
    
    Coordinates multiple specialized services to process documents of various types.
    Handles error isolation, performance monitoring, and result aggregation.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the orchestrator.

        :param config: Pipeline configuration. If None, loads from default config.
        """
        self.config = config or get_pipeline_config()
        
        # Lazy loading of dependencies to avoid circular imports
        self._file_analyzer = None
        self._document_converter = None
        self._chunking_service = None
        self._metadata_manager = None

        logger.info("Document pipeline orchestrator initialized")

    @property
    def file_analyzer(self):
        """Lazy-loaded file analyzer."""
        if self._file_analyzer is None:
            from src.document_processing.file_analyzer import FileAnalyzer
            self._file_analyzer = FileAnalyzer(self.config)
        return self._file_analyzer

    @property
    def document_converter(self):
        """Lazy-loaded document converter."""
        if self._document_converter is None:
            from src.document_processing.document_converter import DocumentConverter
            self._document_converter = DocumentConverter(self.config)
        return self._document_converter

    @property
    def chunking_service(self):
        """Lazy-loaded chunking service."""
        if self._chunking_service is None:
            from src.document_processing.chunking_service import ChunkingService
            self._chunking_service = ChunkingService(self.config)
        return self._chunking_service

    @property
    def metadata_manager(self):
        """Lazy-loaded metadata manager."""
        if self._metadata_manager is None:
            from src.document_processing.metadata_manager import MetadataManager
            self._metadata_manager = MetadataManager(self.config)
        return self._metadata_manager

    def process_documents(self, file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Process a list of documents through the complete pipeline.

        :param file_paths: List of file paths to process
        :return: Processing results with documents, errors, and statistics
        """
        start_time = time.time()
        logger.info("Starting document processing for %d files", len(file_paths))

        # Validate inputs
        if not file_paths:
            logger.warning("No files provided for processing")
            return {
                "documents": [],
                "errors": [],
                "stats": {
                    "files_processed": 0,
                    "documents_created": 0,
                    "processing_time": 0.0,
                    "errors_count": 0
                }
            }

        # Convert to Path objects for consistent handling
        paths = [Path(fp) for fp in file_paths]

        # Step 1: Analyze and validate files
        analysis_result = self._analyze_files(paths)
        if analysis_result["errors"]:
            logger.warning("File analysis found %d errors", len(analysis_result["errors"]))

        valid_files = analysis_result["valid_files"]
        if not valid_files:
            logger.error("No valid files to process")
            return {
                "documents": [],
                "errors": analysis_result["errors"],
                "stats": {
                    "files_processed": 0,
                    "documents_created": 0,
                    "processing_time": time.time() - start_time,
                    "errors_count": len(analysis_result["errors"])
                }
            }

        # Step 2: Group files by type for efficient batch processing
        file_groups = self._group_files_by_type(valid_files)
        logger.info("Files grouped into %d type categories", len(file_groups))

        # Step 3: Process each file type group
        all_documents = []
        all_errors = list(analysis_result["errors"])

        for file_type, files in file_groups.items():
            try:
                logger.info("Processing %d %s files", len(files), file_type)
                result = self._process_file_group(file_type, files)
                
                all_documents.extend(result["documents"])
                all_errors.extend(result["errors"])
                
            except Exception as e:
                error_msg = f"Failed to process {file_type} files: {str(e)}"
                logger.error(error_msg)
                all_errors.append(error_msg)
                
                if self.config.error_handling.fail_fast:
                    raise

        # Step 4: Finalize results with statistics
        processing_time = time.time() - start_time
        stats = self._calculate_statistics(len(file_paths), len(all_documents), processing_time, len(all_errors))

        logger.info(
            "Document processing completed: %d documents from %d files in %.2fs",
            len(all_documents), len(valid_files), processing_time
        )

        return {
            "documents": all_documents,
            "errors": all_errors,
            "stats": stats
        }

    def _analyze_files(self, paths: List[Path]) -> Dict[str, Any]:
        """
        Analyze files for validity and type classification.

        :param paths: List of file paths to analyze
        :return: Analysis results with valid files and errors
        """
        logger.debug("Analyzing %d files", len(paths))
        
        try:
            return self.file_analyzer.analyze_files(paths)
        except Exception as e:
            logger.error("File analysis failed: %s", e)
            return {
                "valid_files": [],
                "errors": [f"File analysis failed: {str(e)}"],
                "file_types": {},
                "total_size": 0
            }

    def _group_files_by_type(self, files: List[Path]) -> Dict[str, List[Path]]:
        """
        Group files by their detected types for batch processing.

        :param files: List of valid file paths
        :return: Dictionary mapping file types to file lists
        """
        groups = {}
        
        for file_path in files:
            file_type = self.file_analyzer.detect_file_type(file_path)
            if file_type not in groups:
                groups[file_type] = []
            groups[file_type].append(file_path)
        
        return groups

    def _process_file_group(self, file_type: str, files: List[Path]) -> Dict[str, Any]:
        """
        Process a group of files of the same type.

        :param file_type: Type of files being processed
        :param files: List of file paths of the same type
        :return: Processing results for this group
        """
        try:
            # Step 1: Convert files to documents
            conversion_result = self.document_converter.convert_files(file_type, files)
            
            if conversion_result["errors"]:
                logger.warning("Conversion found %d errors", len(conversion_result["errors"]))
            
            if not conversion_result["documents"]:
                return {
                    "documents": [],
                    "errors": conversion_result["errors"]
                }

            # Step 2: Apply chunking
            chunking_result = self.chunking_service.chunk_documents(conversion_result["documents"])
            
            if chunking_result["errors"]:
                logger.warning("Chunking found %d errors", len(chunking_result["errors"]))

            # Step 3: Enhance metadata
            final_documents = []
            for document in chunking_result["documents"]:
                enhanced_doc = self.metadata_manager.enhance_metadata(document, file_type)
                final_documents.append(enhanced_doc)

            # Combine errors from all steps
            all_errors = conversion_result["errors"] + chunking_result["errors"]

            return {
                "documents": final_documents,
                "errors": all_errors
            }

        except Exception as e:
            error_msg = f"Failed to process {file_type} group: {str(e)}"
            logger.error(error_msg)
            return {
                "documents": [],
                "errors": [error_msg]
            }

    def _calculate_statistics(self, total_files: int, total_documents: int, 
                            processing_time: float, error_count: int) -> Dict[str, Any]:
        """
        Calculate processing statistics.

        :param total_files: Total number of input files
        :param total_documents: Total number of output documents
        :param processing_time: Total processing time in seconds
        :param error_count: Number of errors encountered
        :return: Statistics dictionary
        """
        return {
            "files_processed": total_files,
            "documents_created": total_documents,
            "processing_time": processing_time,
            "avg_time_per_file": processing_time / total_files if total_files > 0 else 0.0,
            "errors_count": error_count,
            "success_rate": ((total_files - error_count) / total_files * 100) if total_files > 0 else 0.0,
            "documents_per_file": total_documents / total_files if total_files > 0 else 0.0,
            "processing_timestamp": datetime.now().isoformat()
        }

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration and capabilities.

        :return: Pipeline information dictionary
        """
        return {
            "orchestrator": {
                "version": "1.0",
                "components": ["file_analyzer", "document_converter", "chunking_service", "metadata_manager"]
            },
            "configuration": {
                "supported_types": self.config.file_types.supported_extensions,
                "chunking": {
                    "chunk_size": self.config.chunking.chunk_size,
                    "chunk_overlap": self.config.chunking.chunk_overlap
                },
                "processing_options": {
                    "pdf_enabled": self.config.processing.enable_pdf_processing,
                    "text_enabled": self.config.processing.enable_text_processing,
                    "markdown_enabled": self.config.processing.enable_markdown_processing
                }
            },
            "error_handling": {
                "fail_fast": self.config.error_handling.fail_fast,
                "continue_on_error": self.config.error_handling.continue_on_individual_file_error
            }
        }
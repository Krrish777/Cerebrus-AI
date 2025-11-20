"""
File Analyzer

Analyzes and validates files before processing.
Follows AGENTS.md principles: single responsibility, defensibility, portability.
"""

from pathlib import Path
from typing import List, Dict, Any, Union
import mimetypes
import os

from src.core.logging import get_logger
from src.document_processing.pipeline_config import PipelineConfig

logger = get_logger(__name__)


class FileAnalyzer:
    """
    Analyzes files for type, validity, and processing requirements.
    
    Handles file type detection, existence validation, and size analysis.
    Provides security checks and processing recommendations.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize file analyzer with configuration.

        :param config: Pipeline configuration for validation rules
        """
        self.config = config
        logger.debug("File analyzer initialized")

    def analyze_files(self, file_paths: List[Path]) -> Dict[str, Any]:
        """
        Analyze a list of files for validity and processing requirements.

        :param file_paths: List of file paths to analyze
        :return: Analysis results with valid files, errors, and metadata
        """
        logger.info("Analyzing %d files", len(file_paths))
        
        valid_files = []
        errors = []
        file_types = {}
        total_size = 0

        for file_path in file_paths:
            try:
                analysis = self._analyze_single_file(file_path)
                
                if analysis["valid"]:
                    valid_files.append(file_path)
                    file_type = analysis["file_type"]
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                    total_size += analysis["size"]
                else:
                    errors.extend(analysis["errors"])

            except Exception as e:
                error_msg = f"Failed to analyze {file_path}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        logger.info(
            "File analysis completed: %d valid, %d errors, %d file types",
            len(valid_files), len(errors), len(file_types)
        )

        return {
            "valid_files": valid_files,
            "errors": errors,
            "file_types": file_types,
            "total_size": total_size
        }

    def _analyze_single_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze a single file for validity and metadata.

        :param file_path: Path to the file to analyze
        :return: Analysis result for the file
        """
        errors = []
        
        # Check file existence
        if not file_path.exists():
            return {
                "valid": False,
                "errors": [f"File does not exist: {file_path}"],
                "file_type": None,
                "size": 0
            }

        # Check if it's a file (not directory)
        if not file_path.is_file():
            return {
                "valid": False,
                "errors": [f"Path is not a file: {file_path}"],
                "file_type": None,
                "size": 0
            }

        # Get file size
        try:
            file_size = file_path.stat().st_size
        except OSError as e:
            return {
                "valid": False,
                "errors": [f"Cannot access file {file_path}: {str(e)}"],
                "file_type": None,
                "size": 0
            }

        # Check file size limits
        if file_size == 0:
            errors.append(f"File is empty: {file_path}")
        
        # Detect file type
        file_type = self.detect_file_type(file_path)
        if not file_type:
            errors.append(f"Unsupported file type: {file_path}")

        # Check if file type is enabled in configuration
        if file_type and not self._is_file_type_enabled(file_type):
            errors.append(f"File type {file_type} processing is disabled: {file_path}")

        # Validate file permissions
        if not os.access(file_path, os.R_OK):
            errors.append(f"File is not readable: {file_path}")

        valid = len(errors) == 0
        
        if valid:
            logger.debug("File analysis passed for %s: type=%s, size=%d", file_path, file_type, file_size)
        else:
            logger.debug("File analysis failed for %s: %d errors", file_path, len(errors))

        return {
            "valid": valid,
            "errors": errors,
            "file_type": file_type,
            "size": file_size
        }

    def detect_file_type(self, file_path: Path) -> str | None:
        """
        Detect the type of a file based on extension and MIME type.

        :param file_path: Path to the file
        :return: Detected file type or None if unsupported
        """
        # First, check by extension
        extension = file_path.suffix.lower()
        
        if extension in self.config.file_types.extension_to_type_mapping:
            return self.config.file_types.extension_to_type_mapping[extension]

        # Fallback to MIME type detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            if mime_type in self.config.file_types.supported_mime_types:
                # Map common MIME types to our file types
                mime_to_type = {
                    "application/pdf": "PDF",
                    "text/plain": "Text",
                    "text/markdown": "Markdown"
                }
                return mime_to_type.get(mime_type)

        # Unknown or unsupported file type
        return None

    def _is_file_type_enabled(self, file_type: str) -> bool:
        """
        Check if processing is enabled for a specific file type.

        :param file_type: File type to check
        :return: True if processing is enabled, False otherwise
        """
        type_mapping = {
            "PDF": self.config.processing.enable_pdf_processing,
            "Text": self.config.processing.enable_text_processing,
            "Markdown": self.config.processing.enable_markdown_processing
        }
        
        return type_mapping.get(file_type, False)

    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get detailed information about a file.

        :param file_path: Path to the file
        :return: File information dictionary
        """
        if not file_path.exists():
            return {
                "exists": False,
                "error": f"File does not exist: {file_path}"
            }

        try:
            stat_info = file_path.stat()
            file_type = self.detect_file_type(file_path)
            
            return {
                "exists": True,
                "path": str(file_path),
                "name": file_path.name,
                "size": stat_info.st_size,
                "size_human": self._format_file_size(stat_info.st_size),
                "file_type": file_type,
                "extension": file_path.suffix.lower(),
                "is_supported": file_type is not None,
                "processing_enabled": self._is_file_type_enabled(file_type) if file_type else False,
                "modified_time": stat_info.st_mtime,
                "is_readable": os.access(file_path, os.R_OK)
            }
            
        except OSError as e:
            return {
                "exists": True,
                "error": f"Cannot access file information: {str(e)}"
            }

    def _format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human-readable format.

        :param size_bytes: File size in bytes
        :return: Formatted file size string
        """
        if size_bytes < self.config.file_size_thresholds["kb"]:
            return f"{size_bytes} bytes"
        elif size_bytes < self.config.file_size_thresholds["mb"]:
            return f"{size_bytes / self.config.file_size_thresholds['kb']:.1f} KB"
        else:
            return f"{size_bytes / self.config.file_size_thresholds['mb']:.1f} MB"

    def validate_file_batch(self, file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Validate a batch of files for processing.

        :param file_paths: List of file paths to validate
        :return: Validation results
        """
        # Convert to Path objects
        paths = [Path(fp) for fp in file_paths]
        
        # Analyze files
        analysis = self.analyze_files(paths)
        
        # Calculate batch statistics
        batch_stats = {
            "total_files": len(file_paths),
            "valid_files": len(analysis["valid_files"]),
            "invalid_files": len(analysis["errors"]),
            "total_size": analysis["total_size"],
            "total_size_human": self._format_file_size(analysis["total_size"]),
            "file_type_distribution": analysis["file_types"],
            "validation_passed": len(analysis["errors"]) == 0
        }

        return {
            "valid": batch_stats["validation_passed"],
            "stats": batch_stats,
            "valid_files": analysis["valid_files"],
            "errors": analysis["errors"]
        }
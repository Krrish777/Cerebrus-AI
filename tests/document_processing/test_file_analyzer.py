"""
Unit tests for FileAnalyzer component.

Tests file validation, type detection, and analysis functionality
following AGENTS.md principles: isolated, deterministic, comprehensive.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from typing import List

from src.document_processing.file_analyzer import FileAnalyzer
from src.document_processing.pipeline_config import (
    get_pipeline_config, 
    PipelineConfig,
    FileTypeConfig,
    ValidationConfig,
    ProcessingConfig
)


@pytest.fixture
def temp_files_dir(tmp_path: Path) -> Path:
    """Create temporary directory with test files."""
    test_dir = tmp_path / "test_files"
    test_dir.mkdir()
    
    # Create test files of different types
    (test_dir / "document.pdf").write_bytes(b"%PDF-1.4\nTest content")
    (test_dir / "text_file.txt").write_text("Sample text content", encoding='utf-8')
    (test_dir / "readme.md").write_text("# Test Markdown\nContent", encoding='utf-8')
    (test_dir / "empty_file.txt").write_text("", encoding='utf-8')
    
    # Create a non-readable file (simulate permission issues)
    protected_file = test_dir / "protected.txt"
    protected_file.write_text("Protected content", encoding='utf-8')
    
    return test_dir


@pytest.fixture
def test_config() -> PipelineConfig:
    """Create test configuration for FileAnalyzer."""
    return PipelineConfig(
        file_types=FileTypeConfig(
            supported_mime_types=["application/pdf", "text/plain", "text/markdown"],
            supported_extensions=[".pdf", ".txt", ".md"],
            extension_to_type_mapping={
                ".pdf": "PDF",
                ".txt": "Text", 
                ".md": "Markdown"
            }
        ),
        processing=ProcessingConfig(
            enable_pdf_processing=True,
            enable_text_processing=True,
            enable_markdown_processing=True
        ),
        validation=ValidationConfig(),
        file_size_thresholds={
            "kb": 1024,
            "mb": 1024 * 1024
        }
    )


@pytest.fixture
def file_analyzer(test_config: PipelineConfig) -> FileAnalyzer:
    """Create FileAnalyzer instance with test configuration."""
    return FileAnalyzer(test_config)


class TestFileAnalyzerInitialization:
    """Test FileAnalyzer initialization."""

    def test_file_analyzer_initializes_with_config(self, test_config: PipelineConfig):
        """Test FileAnalyzer initializes properly with configuration."""
        analyzer = FileAnalyzer(test_config)
        
        assert analyzer.config == test_config
        assert analyzer.config.processing.enable_pdf_processing is True

    def test_file_analyzer_initializes_with_default_config(self):
        """Test FileAnalyzer works with default configuration."""
        # Should not raise any exceptions
        analyzer = FileAnalyzer(get_pipeline_config())
        assert analyzer.config is not None


class TestFileTypeDetection:
    """Test file type detection functionality."""

    @pytest.mark.parametrize("filename,expected_type", [
        ("document.pdf", "PDF"),
        ("text.txt", "Text"),
        ("readme.md", "Markdown"),
        ("README.markdown", "Markdown"),
        ("data.csv", None),  # Unsupported type
        ("no_extension", None),
    ])
    def test_detect_file_type_by_extension(self, file_analyzer: FileAnalyzer, 
                                         tmp_path: Path, filename: str, expected_type: str):
        """Test file type detection by extension."""
        test_file = tmp_path / filename
        test_file.write_text("test content", encoding='utf-8')
        
        detected_type = file_analyzer.detect_file_type(test_file)
        assert detected_type == expected_type

    def test_detect_file_type_case_insensitive(self, file_analyzer: FileAnalyzer, tmp_path: Path):
        """Test file type detection is case insensitive."""
        test_file = tmp_path / "DOCUMENT.PDF"
        test_file.write_text("test content", encoding='utf-8')
        
        detected_type = file_analyzer.detect_file_type(test_file)
        assert detected_type == "PDF"

    @patch('mimetypes.guess_type')
    def test_detect_file_type_fallback_to_mime(self, mock_guess_type, file_analyzer: FileAnalyzer, tmp_path: Path):
        """Test fallback to MIME type detection when extension fails."""
        # Mock MIME type detection
        mock_guess_type.return_value = ("application/pdf", None)
        
        test_file = tmp_path / "document.unknown"
        test_file.write_text("test content", encoding='utf-8')
        
        detected_type = file_analyzer.detect_file_type(test_file)
        assert detected_type == "PDF"
        
    def test_detect_file_type_unknown_file(self, file_analyzer: FileAnalyzer, tmp_path: Path):
        """Test detection returns None for unknown file types."""
        test_file = tmp_path / "unknown.xyz"
        test_file.write_text("test content", encoding='utf-8')
        
        detected_type = file_analyzer.detect_file_type(test_file)
        assert detected_type is None


class TestSingleFileAnalysis:
    """Test analysis of individual files."""

    def test_analyze_valid_file(self, file_analyzer: FileAnalyzer, temp_files_dir: Path):
        """Test analysis of valid file returns success."""
        test_file = temp_files_dir / "document.pdf"
        
        result = file_analyzer._analyze_single_file(test_file)
        
        assert result["valid"] is True
        assert result["errors"] == []
        assert result["file_type"] == "PDF"
        assert result["size"] > 0

    def test_analyze_nonexistent_file(self, file_analyzer: FileAnalyzer, tmp_path: Path):
        """Test analysis of non-existent file returns error."""
        nonexistent_file = tmp_path / "does_not_exist.txt"
        
        result = file_analyzer._analyze_single_file(nonexistent_file)
        
        assert result["valid"] is False
        assert len(result["errors"]) == 1
        assert "does not exist" in result["errors"][0]
        assert result["file_type"] is None
        assert result["size"] == 0

    def test_analyze_directory_as_file(self, file_analyzer: FileAnalyzer, tmp_path: Path):
        """Test analysis of directory returns error."""
        test_dir = tmp_path / "test_directory"
        test_dir.mkdir()
        
        result = file_analyzer._analyze_single_file(test_dir)
        
        assert result["valid"] is False
        assert "not a file" in result["errors"][0]

    def test_analyze_empty_file(self, file_analyzer: FileAnalyzer, temp_files_dir: Path):
        """Test analysis of empty file returns warning."""
        empty_file = temp_files_dir / "empty_file.txt"
        
        result = file_analyzer._analyze_single_file(empty_file)
        
        assert result["valid"] is False
        assert any("empty" in error for error in result["errors"])

    def test_analyze_unsupported_file_type(self, file_analyzer: FileAnalyzer, tmp_path: Path):
        """Test analysis of unsupported file type returns error."""
        unsupported_file = tmp_path / "data.csv"
        unsupported_file.write_text("header1,header2\nvalue1,value2", encoding='utf-8')
        
        result = file_analyzer._analyze_single_file(unsupported_file)
        
        assert result["valid"] is False
        assert any("Unsupported file type" in error for error in result["errors"])

    def test_analyze_disabled_file_type(self, tmp_path: Path):
        """Test analysis when file type processing is disabled."""
        # Create config with PDF processing disabled
        config = PipelineConfig(
            file_types=FileTypeConfig(
                extension_to_type_mapping={".pdf": "PDF"}
            ),
            processing=ProcessingConfig(
                enable_pdf_processing=False
            )
        )
        analyzer = FileAnalyzer(config)
        
        test_file = tmp_path / "document.pdf"
        test_file.write_bytes(b"%PDF-1.4\nTest content")
        
        result = analyzer._analyze_single_file(test_file)
        
        assert result["valid"] is False
        assert any("processing is disabled" in error for error in result["errors"])

    @patch('os.access')
    def test_analyze_unreadable_file(self, mock_access, file_analyzer: FileAnalyzer, tmp_path: Path):
        """Test analysis of unreadable file returns error."""
        # Mock file as not readable
        mock_access.return_value = False
        
        test_file = tmp_path / "unreadable.txt"
        test_file.write_text("content", encoding='utf-8')
        
        result = file_analyzer._analyze_single_file(test_file)
        
        assert result["valid"] is False
        assert any("not readable" in error for error in result["errors"])


class TestBatchFileAnalysis:
    """Test analysis of multiple files."""

    def test_analyze_empty_file_list(self, file_analyzer: FileAnalyzer):
        """Test analysis of empty file list returns empty results."""
        result = file_analyzer.analyze_files([])
        
        assert result["valid_files"] == []
        assert result["errors"] == []
        assert result["file_types"] == {}
        assert result["total_size"] == 0

    def test_analyze_mixed_valid_invalid_files(self, file_analyzer: FileAnalyzer, temp_files_dir: Path):
        """Test analysis of mixed valid and invalid files."""
        files = [
            temp_files_dir / "document.pdf",  # Valid
            temp_files_dir / "text_file.txt",  # Valid
            temp_files_dir / "nonexistent.txt",  # Invalid
            temp_files_dir / "empty_file.txt"  # Invalid (empty)
        ]
        
        result = file_analyzer.analyze_files(files)
        
        assert len(result["valid_files"]) == 2
        assert len(result["errors"]) == 2
        assert result["file_types"]["PDF"] == 1
        assert result["file_types"]["Text"] == 1
        assert result["total_size"] > 0

    def test_analyze_all_valid_files(self, file_analyzer: FileAnalyzer, temp_files_dir: Path):
        """Test analysis when all files are valid."""
        files = [
            temp_files_dir / "document.pdf",
            temp_files_dir / "text_file.txt",
            temp_files_dir / "readme.md"
        ]
        
        result = file_analyzer.analyze_files(files)
        
        assert len(result["valid_files"]) == 3
        assert result["errors"] == []
        assert result["file_types"]["PDF"] == 1
        assert result["file_types"]["Text"] == 1
        assert result["file_types"]["Markdown"] == 1

    def test_analyze_files_exception_handling(self, file_analyzer: FileAnalyzer, tmp_path: Path):
        """Test graceful handling of exceptions during file analysis."""
        # Create a file that will cause an exception
        problematic_file = tmp_path / "problematic.txt"
        problematic_file.write_text("content", encoding='utf-8')
        
        # Mock _analyze_single_file to raise an exception
        with patch.object(file_analyzer, '_analyze_single_file') as mock_analyze:
            mock_analyze.side_effect = Exception("Simulated error")
            
            result = file_analyzer.analyze_files([problematic_file])
            
            assert result["valid_files"] == []
            assert len(result["errors"]) == 1
            assert "Failed to analyze" in result["errors"][0]


class TestFileInformation:
    """Test file information retrieval."""

    def test_get_file_info_existing_file(self, file_analyzer: FileAnalyzer, temp_files_dir: Path):
        """Test getting information for existing file."""
        test_file = temp_files_dir / "document.pdf"
        
        info = file_analyzer.get_file_info(test_file)
        
        assert info["exists"] is True
        assert info["name"] == "document.pdf"
        assert info["file_type"] == "PDF"
        assert info["extension"] == ".pdf"
        assert info["is_supported"] is True
        assert info["processing_enabled"] is True
        assert info["size"] > 0
        assert "size_human" in info

    def test_get_file_info_nonexistent_file(self, file_analyzer: FileAnalyzer, tmp_path: Path):
        """Test getting information for non-existent file."""
        nonexistent_file = tmp_path / "does_not_exist.txt"
        
        info = file_analyzer.get_file_info(nonexistent_file)
        
        assert info["exists"] is False
        assert "error" in info

    @patch('pathlib.Path.stat')
    def test_get_file_info_access_error(self, mock_stat, file_analyzer: FileAnalyzer, tmp_path: Path):
        """Test handling of file access errors."""
        mock_stat.side_effect = OSError("Access denied")
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("content", encoding='utf-8')
        
        info = file_analyzer.get_file_info(test_file)
        
        assert info["exists"] is False
        assert "error" in info


class TestFileSizeFormatting:
    """Test file size formatting utilities."""

    @pytest.mark.parametrize("size_bytes,expected", [
        (100, "100 bytes"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),  # 1024 + 512
        (1024*1024, "1.0 MB"),
        (1024*1024*2.5, "2.5 MB")
    ])
    def test_format_file_size(self, file_analyzer: FileAnalyzer, size_bytes: int, expected: str):
        """Test file size formatting for different sizes."""
        formatted = file_analyzer._format_file_size(size_bytes)
        assert formatted == expected


class TestBatchValidation:
    """Test batch file validation functionality."""

    def test_validate_file_batch_all_valid(self, file_analyzer: FileAnalyzer, temp_files_dir: Path):
        """Test validation when all files are valid."""
        files = [
            str(temp_files_dir / "document.pdf"),
            str(temp_files_dir / "text_file.txt")
        ]
        
        result = file_analyzer.validate_file_batch(files)
        
        assert result["valid"] is True
        assert result["stats"]["total_files"] == 2
        assert result["stats"]["valid_files"] == 2
        assert result["stats"]["invalid_files"] == 0
        assert result["stats"]["validation_passed"] is True

    def test_validate_file_batch_mixed_files(self, file_analyzer: FileAnalyzer, temp_files_dir: Path):
        """Test validation with mixed valid/invalid files."""
        files = [
            str(temp_files_dir / "document.pdf"),  # Valid
            str(temp_files_dir / "nonexistent.txt")  # Invalid
        ]
        
        result = file_analyzer.validate_file_batch(files)
        
        assert result["valid"] is False
        assert result["stats"]["total_files"] == 2
        assert result["stats"]["valid_files"] == 1
        assert result["stats"]["invalid_files"] == 1
        assert result["stats"]["validation_passed"] is False

    def test_validate_file_batch_pathlib_conversion(self, file_analyzer: FileAnalyzer, temp_files_dir: Path):
        """Test that string paths are converted to Path objects."""
        files = [str(temp_files_dir / "document.pdf")]
        
        result = file_analyzer.validate_file_batch(files)
        
        # Should not raise exceptions and process correctly
        assert isinstance(result["valid_files"][0], Path)


class TestFileAnalyzerIntegration:
    """Integration tests for FileAnalyzer with real file operations."""

    def test_analyze_real_file_types(self, file_analyzer: FileAnalyzer, tmp_path: Path):
        """Test analysis with real file content and types."""
        # Create realistic file content
        pdf_file = tmp_path / "real.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>")
        
        text_file = tmp_path / "real.txt"
        text_file.write_text("This is real text content with\nmultiple lines.", encoding='utf-8')
        
        md_file = tmp_path / "real.md"
        md_file.write_text("# Real Markdown\n\nWith **formatting**.", encoding='utf-8')
        
        files = [pdf_file, text_file, md_file]
        result = file_analyzer.analyze_files(files)
        
        assert len(result["valid_files"]) == 3
        assert result["errors"] == []
        assert "PDF" in result["file_types"]
        assert "Text" in result["file_types"] 
        assert "Markdown" in result["file_types"]

    def test_performance_with_many_files(self, file_analyzer: FileAnalyzer, tmp_path: Path):
        """Test performance with a larger number of files."""
        # Create many small files
        files = []
        for i in range(50):
            file_path = tmp_path / f"file_{i}.txt"
            file_path.write_text(f"Content for file {i}", encoding='utf-8')
            files.append(file_path)
        
        import time
        start_time = time.time()
        result = file_analyzer.analyze_files(files)
        end_time = time.time()
        
        # Should complete in reasonable time (less than 5 seconds)
        assert (end_time - start_time) < 5.0
        assert len(result["valid_files"]) == 50
        assert result["errors"] == []
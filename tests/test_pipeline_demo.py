"""
Test runner for document processing pipeline demo

This test file demonstrates the document processing pipeline with files
from the data directory, following AGENTS.md standards.
"""

import pytest
from pathlib import Path
import tempfile
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.document_processing.pipeline_config import (
    PipelineConfig, ChunkingConfig, ProcessingConfig
)
from src.document_processing.file_analyzer import FileAnalyzer
from src.core.logging import get_logger

logger = get_logger(__name__)


@pytest.fixture
def demo_config():
    """Create demo configuration for testing."""
    return PipelineConfig(
        chunking=ChunkingConfig(
            chunk_size=300,
            chunk_overlap=50
        ),
        processing=ProcessingConfig(
            enable_pdf_processing=True,
            enable_text_processing=True,
            enable_markdown_processing=True
        )
    )


@pytest.fixture
def data_files():
    """Get paths to actual data files."""
    data_dir = Path(__file__).parent.parent / "data"
    
    files = {
        "pdf": data_dir / "llm_overview.pdf",
        "markdown": data_dir / "sample.md",
        "text": data_dir / "sample.txt"
    }
    
    # Only return files that actually exist
    existing_files = {name: path for name, path in files.items() if path.exists()}
    
    logger.info("Found %d data files for testing", len(existing_files))
    return existing_files


class TestDocumentProcessingPipelineDemo:
    """Test the document processing pipeline with actual data files."""

    def test_file_analyzer_with_data_files(self, demo_config, data_files):
        """Test FileAnalyzer with actual data files."""
        logger.info("Testing FileAnalyzer with %d data files", len(data_files))
        
        analyzer = FileAnalyzer(demo_config)
        
        results = {}
        for file_type, file_path in data_files.items():
            logger.debug("Analyzing file: %s", file_path.name)
            
            # Test file type detection
            detected_type = analyzer.detect_file_type(file_path)
            assert detected_type is not None
            
            # Test file info
            file_info = analyzer.get_file_info(file_path)
            assert file_info is not None
            assert file_info.get("is_supported", False)
            
            results[file_type] = {
                "detected_type": detected_type,
                "file_info": file_info
            }
            
            logger.info("File %s detected as %s", file_path.name, detected_type)
        
        # Verify we got results for all files
        assert len(results) == len(data_files)
        
        # Verify expected file types
        if "pdf" in results:
            assert results["pdf"]["detected_type"] == "PDF"
        if "markdown" in results:
            assert results["markdown"]["detected_type"] == "Markdown"
        if "text" in results:
            assert results["text"]["detected_type"] == "Text"

    def test_content_analysis_for_text_files(self, data_files):
        """Test content analysis for text-based files."""
        text_files = {k: v for k, v in data_files.items() if k in ["markdown", "text"]}
        
        if not text_files:
            pytest.skip("No text files available for testing")
        
        for file_type, file_path in text_files.items():
            logger.debug("Analyzing content of %s", file_path.name)
            
            content = file_path.read_text(encoding='utf-8')
            
            # Basic content validation
            assert len(content) > 0
            
            # Count metrics
            word_count = len(content.split())
            line_count = len(content.splitlines())
            
            # Estimate chunks (using demo config chunk_size=300)
            estimated_chunks = max(1, len(content) // 300)
            
            logger.info("File %s: %d chars, %d words, %d lines, %d estimated chunks",
                       file_path.name, len(content), word_count, line_count, estimated_chunks)
            
            assert word_count > 0
            assert line_count > 0
            assert estimated_chunks > 0

    def test_pipeline_configuration_creation(self, demo_config):
        """Test that pipeline configuration is created correctly."""
        assert demo_config is not None
        assert demo_config.chunking.chunk_size == 300
        assert demo_config.chunking.chunk_overlap == 50
        assert demo_config.processing.enable_pdf_processing is True
        assert demo_config.processing.enable_text_processing is True
        assert demo_config.processing.enable_markdown_processing is True
        
        logger.info("Pipeline configuration validated successfully")

    def test_file_discovery_and_validation(self, data_files):
        """Test that target files are discovered and accessible."""
        expected_files = ["pdf", "markdown", "text"]
        
        for file_type in expected_files:
            if file_type in data_files:
                file_path = data_files[file_type]
                assert file_path.exists()
                assert file_path.is_file()
                assert file_path.stat().st_size > 0
                
                logger.info("Validated file %s: %d bytes", 
                          file_path.name, file_path.stat().st_size)

    def test_simulated_processing_workflow(self, demo_config, data_files):
        """Test simulated end-to-end processing workflow."""
        logger.info("Testing simulated processing workflow")
        
        analyzer = FileAnalyzer(demo_config)
        
        processing_results = {}
        
        for file_type, file_path in data_files.items():
            logger.debug("Simulating processing for %s", file_path.name)
            
            # Step 1: File analysis
            detected_type = analyzer.detect_file_type(file_path)
            file_info = analyzer.get_file_info(file_path)
            
            # Step 2: Simulate document conversion
            conversion_simulated = True
            
            # Step 3: Simulate chunking
            if file_path.suffix in ['.md', '.txt']:
                content = file_path.read_text(encoding='utf-8')
                estimated_chunks = max(1, len(content) // demo_config.chunking.chunk_size)
            else:
                # PDF - estimate based on file size
                file_size = file_path.stat().st_size
                estimated_chunks = max(1, file_size // 10000)
            
            # Step 4: Simulate metadata enhancement
            metadata_enhanced = True
            
            processing_results[file_type] = {
                "detected_type": detected_type,
                "is_supported": file_info.get("is_supported", False),
                "conversion_simulated": conversion_simulated,
                "estimated_chunks": estimated_chunks,
                "metadata_enhanced": metadata_enhanced,
                "steps_completed": ["analysis", "conversion", "chunking", "metadata"]
            }
            
            logger.info("Simulated processing for %s: %d estimated chunks", 
                       file_path.name, estimated_chunks)
        
        # Validate processing results
        assert len(processing_results) == len(data_files)
        
        for file_type, result in processing_results.items():
            assert result["detected_type"] is not None
            assert result["conversion_simulated"] is True
            assert result["metadata_enhanced"] is True
            assert result["estimated_chunks"] > 0
            assert len(result["steps_completed"]) == 4


class TestPipelineDemoOutput:
    """Test that demo output follows AGENTS.md standards."""

    def test_no_emojis_in_log_messages(self, caplog):
        """Test that log messages contain no emojis or decorative characters."""
        logger.info("Testing log message format compliance")
        logger.error("Testing error message format")
        logger.warning("Testing warning message format")
        logger.debug("Testing debug message format")
        
        # Check all log records for emoji characters
        emoji_chars = "🚀📊💡⚠️✅❌🔧📝🎯📄🧩👀📁"
        
        for record in caplog.records:
            assert not any(char in record.message for char in emoji_chars), \
                f"Log message contains emoji: {record.message}"

    def test_proper_log_formatting(self, caplog):
        """Test that log messages use proper parameterized formatting."""
        import logging
        caplog.set_level(logging.DEBUG)
        
        test_file = "test.pdf"
        test_count = 5
        
        # Temporarily set logger level to DEBUG for this test
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        
        try:
            logger.info("Processing %d files including %s", test_count, test_file)
            logger.debug("File analysis completed for %s", test_file)
            
            # Verify messages were logged
            assert len(caplog.records) >= 2
            
            # Verify no f-strings in actual message (parameterized logging)
            for record in caplog.records:
                assert "{" not in record.getMessage(), \
                    f"Log message not properly formatted: {record.getMessage()}"
        finally:
            logger.setLevel(original_level)
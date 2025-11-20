"""
Comprehensive test configuration for document processing pipeline.

Configures pytest with proper fixtures, markers, and test discovery
following AGENTS.md principles and testing best practices.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import logging

# Disable logging during tests to reduce noise
logging.disable(logging.CRITICAL)





@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create session-scoped test data directory."""
    # Use the project data directory for test fixtures
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    
    return data_dir


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Generator[Path, None, None]:
    """Create isolated temporary workspace for tests."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    
    # Create typical directory structure
    (workspace / "input").mkdir()
    (workspace / "output").mkdir()
    (workspace / "logs").mkdir()
    
    yield workspace
    
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def sample_test_files(temp_workspace: Path) -> Dict[str, Path]:
    """Create sample files for testing across multiple test modules."""
    files = {}
    input_dir = temp_workspace / "input"
    
    # PDF content (simulated)
    pdf_file = input_dir / "test_document.pdf"
    pdf_content = b"%PDF-1.4\n1 0 obj\n<</Type/Catalog>>\nendobj\n"
    pdf_content += b"Test PDF document content for pipeline testing. "
    pdf_content += b"This content spans multiple lines and paragraphs. "
    pdf_content += b"It should be processed correctly by the document pipeline."
    pdf_file.write_bytes(pdf_content)
    files["pdf"] = pdf_file
    
    # Text file
    txt_file = input_dir / "test_document.txt"
    txt_content = """Test Document

This is a test document for the document processing pipeline.
It contains multiple paragraphs and sections to test chunking.

Section 1: Introduction
This section provides an introduction to the document.
It contains several sentences that should be processed correctly.

Section 2: Content
This section contains the main content of the document.
The pipeline should handle this content appropriately.

Section 3: Conclusion
This final section wraps up the document.
All sections should be processed and chunked correctly.
"""
    txt_file.write_text(txt_content, encoding='utf-8')
    files["txt"] = txt_file
    
    # Markdown file
    md_file = input_dir / "test_document.md"
    md_content = """# Test Document

## Overview
This is a **test document** for pipeline testing.

## Features
- Document processing
- Text chunking
- Metadata enhancement

## Code Example
```python
def process_document(doc):
    return pipeline.run(doc)
```

## Conclusion
The pipeline should handle markdown formatting correctly.
"""
    md_file.write_text(md_content, encoding='utf-8')
    files["md"] = md_file
    
    # Small file
    small_file = input_dir / "small.txt"
    small_file.write_text("Short content.", encoding='utf-8')
    files["small"] = small_file
    
    # Empty file (for error testing)
    empty_file = input_dir / "empty.txt"
    empty_file.write_text("", encoding='utf-8')
    files["empty"] = empty_file
    
    return files


@pytest.fixture
def mock_haystack_documents():
    """Create mock Haystack documents for testing."""
    from haystack.dataclasses import Document
    
    return [
        Document(
            id="test_doc_1",
            content="First test document content for pipeline testing.",
            meta={
                "file_path": "/test/doc1.txt",
                "source_type": "text",
                "page_number": 1
            }
        ),
        Document(
            id="test_doc_2",
            content="Second test document with different content and metadata.",
            meta={
                "file_path": "/test/doc2.pdf",
                "source_type": "pdf",
                "page_number": 1
            }
        ),
        Document(
            id="test_doc_3",
            content="Third document for comprehensive testing scenarios.",
            meta={
                "file_path": "/test/doc3.md",
                "source_type": "markdown"
            }
        )
    ]


@pytest.fixture
def minimal_config():
    """Create minimal configuration for basic testing."""
    from src.document_processing.pipeline_config import (
        PipelineConfig, ChunkingConfig, ProcessingConfig, FileTypeConfig
    )
    
    return PipelineConfig(
        chunking=ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20
        ),
        processing=ProcessingConfig(
            enable_pdf_processing=True,
            enable_text_processing=True,
            enable_markdown_processing=True
        ),
        file_types=FileTypeConfig()
    )


@pytest.fixture
def comprehensive_config():
    """Create comprehensive configuration for advanced testing."""
    from src.document_processing.pipeline_config import (
        PipelineConfig, ChunkingConfig, ProcessingConfig, FileTypeConfig,
        MetadataConfig, ErrorHandlingConfig, PerformanceConfig
    )
    
    return PipelineConfig(
        chunking=ChunkingConfig(
            chunk_size=200,
            chunk_overlap=50,
            min_chunk_size_ratio=0.3,
            boundary_preferences=["paragraph", "sentence"],
            enable_statistics=True
        ),
        processing=ProcessingConfig(
            enable_pdf_processing=True,
            enable_text_processing=True,
            enable_markdown_processing=True,
            enable_markdown_fallback=True
        ),
        file_types=FileTypeConfig(),
        metadata=MetadataConfig(),
        error_handling=ErrorHandlingConfig(
            fail_fast=False,
            continue_on_individual_file_error=True
        ),
        performance=PerformanceConfig(
            enable_timing=True,
            enable_statistics=True
        )
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add unit marker to unit test files
        if "test_file_analyzer" in str(item.fspath) or \
           "test_document_converter" in str(item.fspath) or \
           "test_chunking_service" in str(item.fspath) or \
           "test_metadata_manager" in str(item.fspath) or \
           "test_pipeline_config" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration tests
        elif "test_integration" in str(item.fspath) or \
             "test_pipeline_orchestrator" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add config marker to configuration tests
        if "config" in str(item.fspath).lower():
            item.add_marker(pytest.mark.config)
        
        # Add slow marker to tests that might take longer
        if any(keyword in str(item.fspath) for keyword in ["integration", "performance", "large"]):
            item.add_marker(pytest.mark.slow)


# Custom assertions for document processing
class DocumentAssertions:
    """Custom assertions for document processing tests."""
    
    @staticmethod
    def assert_valid_document(doc):
        """Assert document has valid structure."""
        assert hasattr(doc, 'id'), "Document must have id"
        assert hasattr(doc, 'content'), "Document must have content"
        assert hasattr(doc, 'meta'), "Document must have meta"
        assert isinstance(doc.meta, dict), "Document meta must be dict"
    
    @staticmethod
    def assert_valid_processing_result(result):
        """Assert processing result has expected structure."""
        assert isinstance(result, dict), "Result must be dict"
        assert "documents" in result, "Result must have documents"
        assert "errors" in result, "Result must have errors"
        assert isinstance(result["documents"], list), "Documents must be list"
        assert isinstance(result["errors"], list), "Errors must be list"
    
    @staticmethod
    def assert_chunk_metadata_complete(chunk_doc, expected_fields=None):
        """Assert chunk document has complete metadata."""
        if expected_fields is None:
            expected_fields = ["chunk_id", "chunk_index", "source_file"]
        
        for field in expected_fields:
            assert field in chunk_doc.meta, f"Chunk missing metadata field: {field}"


@pytest.fixture
def doc_assertions():
    """Provide document assertion utilities."""
    return DocumentAssertions()


# Performance testing utilities
@pytest.fixture
def performance_monitor():
    """Provide performance monitoring utilities."""
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.measurements = {}
        
        def start(self, operation_name="default"):
            self.start_time = time.time()
            return self
        
        def stop(self, operation_name="default"):
            if self.start_time is not None:
                duration = time.time() - self.start_time
                self.measurements[operation_name] = duration
                return duration
            return None
        
        def assert_duration_less_than(self, max_seconds, operation_name="default"):
            duration = self.measurements.get(operation_name)
            assert duration is not None, f"No measurement found for {operation_name}"
            assert duration < max_seconds, f"Operation {operation_name} took {duration}s, expected < {max_seconds}s"
    
    return PerformanceMonitor()


# Error simulation utilities
@pytest.fixture
def error_simulator():
    """Provide error simulation utilities for testing error handling."""
    
    class ErrorSimulator:
        @staticmethod
        def file_not_found_error():
            return FileNotFoundError("Simulated file not found")
        
        @staticmethod
        def permission_error():
            return PermissionError("Simulated permission denied")
        
        @staticmethod
        def value_error(message="Simulated validation error"):
            return ValueError(message)
        
        @staticmethod
        def processing_error():
            return Exception("Simulated processing error")
        
        @staticmethod
        def timeout_error():
            return TimeoutError("Simulated timeout")
    
    return ErrorSimulator()


# Configuration for specific test scenarios
def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Skip slow tests unless explicitly requested
    if "slow" in item.keywords and not item.config.getoption("--runslow", default=False):
        pytest.skip("need --runslow option to run slow tests")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--integration-only",
        action="store_true",
        default=False,
        help="run only integration tests"
    )
    parser.addoption(
        "--unit-only", 
        action="store_true",
        default=False,
        help="run only unit tests"
    )


def pytest_configure(config):
    """Configure test execution based on options."""
    if config.getoption("--integration-only"):
        config.option.markexpr = "integration"
    elif config.getoption("--unit-only"):
        config.option.markexpr = "unit"
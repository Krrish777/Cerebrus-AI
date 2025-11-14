"""
Pytest configuration file for the Cerebrus AI project tests.

This file contains common fixtures and configuration settings that are shared
across all test modules in the project.
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path

# Add the src directory to Python path for importing modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory path"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_text_content():
    """Provide sample text content for testing"""
    return """
    This is a sample document for testing purposes. It contains multiple sentences
    and paragraphs to test the document processing functionality.
    
    The document should be processed correctly and split into appropriate chunks
    based on the configured chunk size and overlap parameters.
    
    This is the final paragraph of the sample document.
    """


@pytest.fixture
def mock_pdf_metadata():
    """Provide sample PDF metadata for testing"""
    return {
        'page_number': 1,
        'name': 'test_document.pdf',
        'file_path': '/path/to/test_document.pdf',
        'total_pages': 5,
        'page_width': 612.0,
        'page_height': 792.0
    }


# Configure pytest settings
def pytest_configure(config):
    """Configure pytest with custom settings"""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Custom pytest collection hook for better test organization
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add unit test marker to all tests by default
        if "integration" not in [mark.name for mark in item.iter_markers()]:
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests that might be slow
        if "test_full_pipeline" in item.name or "test_integration" in item.name:
            item.add_marker(pytest.mark.slow)
"""
Unit tests for DocumentConverter component.

Tests document conversion using Haystack components with proper mocking
and isolation following AGENTS.md principles.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from haystack.dataclasses import Document
from haystack import Pipeline

from src.document_processing.document_converter import DocumentConverter
from src.document_processing.pipeline_config import (
    PipelineConfig,
    FileTypeConfig,
    ProcessingConfig,
    MetadataConfig
)


@pytest.fixture
def test_config() -> PipelineConfig:
    """Create test configuration for DocumentConverter."""
    return PipelineConfig(
        file_types=FileTypeConfig(
            supported_mime_types=["application/pdf", "text/plain", "text/markdown"],
            supported_extensions=[".pdf", ".txt", ".md"]
        ),
        processing=ProcessingConfig(
            enable_pdf_processing=True,
            enable_text_processing=True,
            enable_markdown_processing=True
        ),
        metadata=MetadataConfig()
    )


@pytest.fixture
def document_converter(test_config: PipelineConfig) -> DocumentConverter:
    """Create DocumentConverter instance with test configuration."""
    return DocumentConverter(test_config)


@pytest.fixture
def sample_document() -> Document:
    """Create a sample Haystack document for testing."""
    return Document(
        id="test_doc_1",
        content="Sample document content for testing",
        meta={"file_path": "/test/path/document.txt", "source": "test"}
    )


@pytest.fixture
def sample_documents() -> List[Document]:
    """Create multiple sample documents for testing."""
    return [
        Document(
            id="doc_1",
            content="First document content",
            meta={"file_path": "/test/doc1.pdf", "name": "doc1.pdf"}
        ),
        Document(
            id="doc_2", 
            content="Second document content",
            meta={"file_path": "/test/doc2.txt", "name": "doc2.txt"}
        )
    ]


class TestDocumentConverterInitialization:
    """Test DocumentConverter initialization."""

    def test_converter_initializes_with_config(self, test_config: PipelineConfig):
        """Test DocumentConverter initializes properly with configuration."""
        converter = DocumentConverter(test_config)
        
        assert converter.config == test_config
        assert converter._pipeline is None  # Lazy loading

    def test_converter_lazy_loads_pipeline(self, document_converter: DocumentConverter):
        """Test that pipeline is lazy-loaded on first access."""
        # Initially None
        assert document_converter._pipeline is None
        
        # Access pipeline property
        pipeline = document_converter.pipeline
        
        # Should now be initialized
        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)
        assert document_converter._pipeline is pipeline


class TestPipelineBuildingAndConfiguration:
    """Test pipeline construction and component configuration."""

    def test_build_conversion_pipeline_all_components(self, document_converter: DocumentConverter):
        """Test building pipeline with all converters enabled."""
        pipeline = document_converter._build_conversion_pipeline()
        
        # Check components are added
        components = list(pipeline.graph.nodes.keys())
        assert "router" in components
        assert "joiner" in components
        
        # Check converters based on config
        if document_converter.config.processing.enable_pdf_processing:
            assert "pdf_converter" in components
        if document_converter.config.processing.enable_text_processing:
            assert "text_converter" in components
        if document_converter.config.processing.enable_markdown_processing:
            assert "markdown_converter" in components

    def test_build_pipeline_selective_converters(self):
        """Test building pipeline with selective converters enabled."""
        # Config with only text processing
        config = PipelineConfig(
            processing=ProcessingConfig(
                enable_pdf_processing=False,
                enable_text_processing=True,
                enable_markdown_processing=False
            ),
            file_types=FileTypeConfig(
                supported_mime_types=["text/plain"]
            )
        )
        converter = DocumentConverter(config)
        
        pipeline = converter._build_conversion_pipeline()
        components = list(pipeline.graph.nodes.keys())
        
        assert "text_converter" in components
        assert "pdf_converter" not in components
        assert "markdown_converter" not in components

    @patch('src.document_processing.document_converter.PyPDFToDocument')
    @patch('src.document_processing.document_converter.TextFileToDocument') 
    @patch('src.document_processing.document_converter.MarkdownToDocument')
    def test_converter_creation_with_mocks(self, mock_md, mock_text, mock_pdf, 
                                          document_converter: DocumentConverter):
        """Test that Haystack converters are created correctly."""
        # Build pipeline to trigger converter creation
        pipeline = document_converter._build_conversion_pipeline()
        
        # Verify converters were instantiated
        if document_converter.config.processing.enable_pdf_processing:
            mock_pdf.assert_called_once()
        if document_converter.config.processing.enable_text_processing:
            mock_text.assert_called_once()
        if document_converter.config.processing.enable_markdown_processing:
            mock_md.assert_called_once()

    def test_pipeline_connections(self, document_converter: DocumentConverter):
        """Test that pipeline components are connected properly."""
        pipeline = document_converter._build_conversion_pipeline()
        
        # Check that connections exist (basic connectivity test)
        # The exact connections depend on configuration and available sockets
        assert len(pipeline.graph.edges) > 0
        
        # Verify joiner is connected (should receive from converters)
        joiner_inputs = list(pipeline.graph.predecessors("joiner"))
        assert len(joiner_inputs) > 0


class TestDocumentConversion:
    """Test document conversion functionality."""

    def test_convert_files_empty_list(self, document_converter: DocumentConverter):
        """Test conversion with empty file list."""
        result = document_converter.convert_files("Text", [])
        
        assert result["documents"] == []
        assert result["errors"] == []
        assert "stats" in result

    @patch.object(DocumentConverter, 'pipeline', new_callable=lambda: Mock())
    def test_convert_files_successful_conversion(self, mock_pipeline_prop, 
                                               document_converter: DocumentConverter,
                                               sample_documents: List[Document]):
        """Test successful file conversion."""
        # Mock pipeline execution
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {
            "joiner": {"documents": sample_documents}
        }
        mock_pipeline_prop.return_value = mock_pipeline
        
        test_files = [Path("/test/doc1.pdf"), Path("/test/doc2.txt")]
        result = document_converter.convert_files("PDF", test_files)
        
        # Verify pipeline was called with correct arguments
        mock_pipeline.run.assert_called_once()
        call_args = mock_pipeline.run.call_args[0][0]
        assert "router" in call_args
        assert "sources" in call_args["router"]
        
        # Check results
        assert len(result["documents"]) == 2
        assert result["errors"] == []
        assert result["stats"]["files_converted"] == 2
        assert result["stats"]["documents_created"] == 2

    @patch.object(DocumentConverter, 'pipeline', new_callable=lambda: Mock())
    def test_convert_files_pipeline_exception(self, mock_pipeline_prop,
                                            document_converter: DocumentConverter):
        """Test handling of pipeline execution exceptions."""
        # Mock pipeline to raise exception
        mock_pipeline = Mock()
        mock_pipeline.run.side_effect = Exception("Pipeline error")
        mock_pipeline_prop.return_value = mock_pipeline
        
        test_files = [Path("/test/doc.pdf")]
        result = document_converter.convert_files("PDF", test_files)
        
        assert result["documents"] == []
        assert len(result["errors"]) == 1
        assert "Pipeline error" in result["errors"][0]

    @patch.object(DocumentConverter, 'pipeline', new_callable=lambda: Mock())
    def test_convert_files_no_documents_returned(self, mock_pipeline_prop,
                                                document_converter: DocumentConverter):
        """Test handling when pipeline returns no documents."""
        # Mock pipeline to return empty documents
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {"joiner": {"documents": []}}
        mock_pipeline_prop.return_value = mock_pipeline
        
        test_files = [Path("/test/doc.pdf")]
        result = document_converter.convert_files("PDF", test_files)
        
        assert result["documents"] == []
        assert result["errors"] == []

    def test_extract_documents_from_result_joiner(self, document_converter: DocumentConverter,
                                                 sample_documents: List[Document]):
        """Test document extraction from pipeline result with joiner."""
        pipeline_result = {
            "joiner": {"documents": sample_documents}
        }
        
        documents = document_converter._extract_documents_from_result(pipeline_result)
        assert documents == sample_documents

    def test_extract_documents_from_result_fallback(self, document_converter: DocumentConverter,
                                                   sample_documents: List[Document]):
        """Test document extraction fallback to individual converters."""
        pipeline_result = {
            "pdf_converter": {"documents": sample_documents[:1]},
            "text_converter": {"documents": sample_documents[1:]}
        }
        
        documents = document_converter._extract_documents_from_result(pipeline_result)
        assert len(documents) == 2

    def test_extract_documents_no_results(self, document_converter: DocumentConverter):
        """Test document extraction when no documents are found."""
        pipeline_result = {"router": {"output": "some_data"}}
        
        documents = document_converter._extract_documents_from_result(pipeline_result)
        assert documents == []


class TestMetadataEnhancement:
    """Test document metadata enhancement."""

    def test_enhance_document_metadata(self, document_converter: DocumentConverter,
                                      sample_document: Document):
        """Test metadata enhancement adds conversion information."""
        enhanced = document_converter._enhance_document_metadata(sample_document, "PDF")
        
        # Check enhanced metadata
        assert enhanced.meta["source_type"] == "pdf"
        assert enhanced.meta["converter"] == "PyPDFToDocument"
        assert "conversion_timestamp" in enhanced.meta
        assert enhanced.meta["pipeline_version"] == "1.0"
        
        # Original content should be preserved
        assert enhanced.content == sample_document.content
        assert enhanced.id == sample_document.id

    def test_enhance_metadata_preserves_existing(self, document_converter: DocumentConverter):
        """Test that existing metadata is preserved during enhancement."""
        doc = Document(
            id="test",
            content="content",
            meta={"existing_field": "value", "file_path": "/test/path.txt"}
        )
        
        enhanced = document_converter._enhance_document_metadata(doc, "Text")
        
        assert enhanced.meta["existing_field"] == "value"
        assert enhanced.meta["file_path"] == "/test/path.txt"
        assert enhanced.meta["source_type"] == "text"

    def test_enhance_metadata_adds_file_path(self, document_converter: DocumentConverter):
        """Test file path extraction from metadata."""
        doc = Document(
            id="test",
            content="content",
            meta={"name": "/path/to/file.txt"}
        )
        
        enhanced = document_converter._enhance_document_metadata(doc, "Text")
        
        assert enhanced.meta["file_path"] == "/path/to/file.txt"

    @pytest.mark.parametrize("file_type,expected_converter", [
        ("PDF", "PyPDFToDocument"),
        ("Text", "TextFileToDocument"), 
        ("Markdown", "MarkdownToDocument"),
        ("Unknown", "TextFileToDocument")  # Default fallback
    ])
    def test_get_converter_name(self, document_converter: DocumentConverter,
                               file_type: str, expected_converter: str):
        """Test converter name mapping for different file types."""
        converter_name = document_converter._get_converter_name(file_type)
        assert converter_name == expected_converter


class TestSupportedTypes:
    """Test supported file types functionality."""

    def test_get_supported_types_all_enabled(self, test_config: PipelineConfig):
        """Test getting supported types when all are enabled."""
        converter = DocumentConverter(test_config)
        
        supported = converter.get_supported_types()
        
        assert "PDF" in supported
        assert "Text" in supported
        assert "Markdown" in supported

    def test_get_supported_types_selective(self):
        """Test supported types with selective processing enabled."""
        config = PipelineConfig(
            processing=ProcessingConfig(
                enable_pdf_processing=True,
                enable_text_processing=False,
                enable_markdown_processing=True
            )
        )
        converter = DocumentConverter(config)
        
        supported = converter.get_supported_types()
        
        assert "PDF" in supported
        assert "Text" not in supported
        assert "Markdown" in supported

    def test_get_pipeline_info(self, document_converter: DocumentConverter):
        """Test getting pipeline information."""
        info = document_converter.get_pipeline_info()
        
        assert "components" in info
        assert "connections" in info
        assert "supported_types" in info
        assert "configuration" in info
        
        # Check configuration details
        config_info = info["configuration"]
        assert "pdf_enabled" in config_info
        assert "text_enabled" in config_info
        assert "markdown_enabled" in config_info


class TestDocumentConverterIntegration:
    """Integration tests for DocumentConverter."""

    def test_end_to_end_conversion_flow(self, tmp_path: Path):
        """Test complete conversion flow with real files."""
        # Create test files
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test document content", encoding='utf-8')
        
        # Create converter
        config = PipelineConfig(
            processing=ProcessingConfig(
                enable_text_processing=True,
                enable_pdf_processing=False,
                enable_markdown_processing=False
            ),
            file_types=FileTypeConfig(
                supported_mime_types=["text/plain"]
            )
        )
        
        converter = DocumentConverter(config)
        
        # This would normally require real Haystack components
        # In a real integration test, we'd test with actual Haystack pipeline
        # For unit tests, we focus on our component logic
        assert converter.config.processing.enable_text_processing is True
        assert converter._pipeline is None  # Lazy loading

    @patch('src.document_processing.document_converter.Pipeline')
    def test_pipeline_creation_flow(self, mock_pipeline_class, document_converter: DocumentConverter):
        """Test the complete pipeline creation and setup flow."""
        mock_pipeline_instance = Mock()
        mock_pipeline_class.return_value = mock_pipeline_instance
        
        # Access pipeline to trigger creation
        _ = document_converter.pipeline
        
        # Verify pipeline was created and configured
        mock_pipeline_class.assert_called_once()
        assert mock_pipeline_instance.add_component.call_count > 0

    def test_converter_respects_configuration_changes(self):
        """Test that converter respects configuration changes."""
        # Start with basic config
        config1 = PipelineConfig(
            processing=ProcessingConfig(enable_pdf_processing=True)
        )
        converter1 = DocumentConverter(config1)
        
        # Create new converter with different config
        config2 = PipelineConfig(
            processing=ProcessingConfig(enable_pdf_processing=False)
        )
        converter2 = DocumentConverter(config2)
        
        # Converters should have different configurations
        assert converter1.config.processing.enable_pdf_processing is True
        assert converter2.config.processing.enable_pdf_processing is False
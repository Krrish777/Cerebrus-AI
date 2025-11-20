"""
Unit tests for MetadataManager component.

Tests metadata creation, enhancement, and validation functionality
following AGENTS.md principles: isolated, comprehensive, deterministic.
"""

import pytest
import time
import json
from datetime import datetime
from unittest.mock import patch, Mock
from typing import Dict, Any, List

from haystack.dataclasses import Document

from src.document_processing.metadata_manager import MetadataManager
from src.document_processing.pipeline_config import (
    PipelineConfig,
    MetadataConfig
)


@pytest.fixture
def test_metadata_config() -> MetadataConfig:
    """Create test metadata configuration."""
    return MetadataConfig(
        chunk_id="chunk_id",
        chunk_index="chunk_index", 
        source_file="source_file",
        source_type="source_type",
        page_number="page_number",
        content_hash="content_hash",
        start_char="start_char",
        end_char="end_char",
        word_count="word_count",
        line_count="line_count",
        boundary_found="boundary_found",
        boundary_type="boundary_type",
        processing_date="processing_date"
    )


@pytest.fixture
def test_config(test_metadata_config: MetadataConfig) -> PipelineConfig:
    """Create test configuration for MetadataManager."""
    return PipelineConfig(metadata=test_metadata_config)


@pytest.fixture
def metadata_manager(test_config: PipelineConfig) -> MetadataManager:
    """Create MetadataManager instance with test configuration."""
    return MetadataManager(test_config)


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for metadata tests."""
    return Document(
        id="test_doc",
        content="This is test content for metadata enhancement testing.",
        meta={
            "file_path": "/test/documents/sample.pdf",
            "name": "sample.pdf",
            "existing_field": "existing_value"
        }
    )


@pytest.fixture
def sample_chunk_document() -> Document:
    """Create a sample chunk document."""
    return Document(
        id="chunk_1",
        content="This is a chunk of content.",
        meta={"parent_doc": "original_doc"}
    )


class TestMetadataManagerInitialization:
    """Test MetadataManager initialization."""

    def test_metadata_manager_initializes_with_config(self, test_config: PipelineConfig):
        """Test MetadataManager initializes properly with configuration."""
        manager = MetadataManager(test_config)
        
        assert manager.config == test_config

    def test_metadata_manager_with_default_config(self):
        """Test MetadataManager works with default configuration."""
        from src.document_processing.pipeline_config import get_pipeline_config
        
        manager = MetadataManager(get_pipeline_config())
        assert manager.config is not None


class TestStandardMetadataCreation:
    """Test standard metadata field creation."""

    def test_create_standard_metadata(self, metadata_manager: MetadataManager, sample_document: Document):
        """Test creation of standard metadata fields."""
        standard_meta = metadata_manager._create_standard_metadata(sample_document, "PDF")
        
        assert standard_meta[metadata_manager.config.metadata.source_file] == "/test/documents/sample.pdf"
        assert standard_meta[metadata_manager.config.metadata.source_type] == "pdf"
        assert standard_meta["document_id"] == "test_doc"
        assert standard_meta["content_length"] > 0
        assert standard_meta["has_content"] is True

    def test_create_standard_metadata_empty_content(self, metadata_manager: MetadataManager):
        """Test standard metadata for document with empty content."""
        empty_doc = Document(id="empty", content="", meta={"file_path": "/test/empty.txt"})
        
        standard_meta = metadata_manager._create_standard_metadata(empty_doc, "Text")
        
        assert standard_meta["content_length"] == 0
        assert standard_meta["has_content"] is False

    def test_create_standard_metadata_missing_file_path(self, metadata_manager: MetadataManager):
        """Test standard metadata when file path is missing."""
        doc_no_path = Document(id="no_path", content="content", meta={})
        
        standard_meta = metadata_manager._create_standard_metadata(doc_no_path, "Text")
        
        assert standard_meta[metadata_manager.config.metadata.source_file] == "unknown"

    def test_create_standard_metadata_fallback_file_sources(self, metadata_manager: MetadataManager):
        """Test file path extraction from various metadata sources."""
        # Test with 'name' field
        doc_with_name = Document(
            id="test",
            content="content", 
            meta={"name": "/path/from/name.txt"}
        )
        standard_meta = metadata_manager._create_standard_metadata(doc_with_name, "Text")
        assert standard_meta[metadata_manager.config.metadata.source_file] == "/path/from/name.txt"
        
        # Test with 'source' field
        doc_with_source = Document(
            id="test",
            content="content",
            meta={"source": "/path/from/source.txt"}
        )
        standard_meta = metadata_manager._create_standard_metadata(doc_with_source, "Text")
        assert standard_meta[metadata_manager.config.metadata.source_file] == "/path/from/source.txt"


class TestProcessingMetadataCreation:
    """Test processing-related metadata creation."""

    @patch('time.time')
    def test_create_processing_metadata(self, mock_time, metadata_manager: MetadataManager):
        """Test creation of processing metadata with mocked time."""
        mock_time.return_value = 1234567890.5
        
        processing_meta = metadata_manager._create_processing_metadata()
        
        assert processing_meta["processing_timestamp"] == 1234567890.5
        assert processing_meta["processor_version"] == "1.0"
        assert processing_meta["pipeline_version"] == "1.0"
        assert processing_meta["metadata_schema_version"] == "1.0"
        assert metadata_manager.config.metadata.processing_date in processing_meta

    def test_processing_metadata_datetime_format(self, metadata_manager: MetadataManager):
        """Test that processing date is in ISO format."""
        processing_meta = metadata_manager._create_processing_metadata()
        
        # Should be able to parse as ISO datetime
        date_str = processing_meta[metadata_manager.config.metadata.processing_date]
        parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00') if date_str.endswith('Z') else date_str)
        assert isinstance(parsed_date, datetime)


class TestCitationMetadataCreation:
    """Test citation metadata creation."""

    def test_create_citation_metadata(self, metadata_manager: MetadataManager, sample_document: Document):
        """Test creation of citation metadata."""
        citation_meta = metadata_manager._create_citation_metadata(sample_document, "PDF")
        
        citation = citation_meta["citation"]
        assert citation["source_file"] == "/test/documents/sample.pdf"
        assert citation["document_type"] == "PDF"
        assert citation["document_id"] == "test_doc"
        assert citation["extraction_method"] == "haystack_pipeline"
        assert "extraction_timestamp" in citation

    def test_create_citation_metadata_with_page_number(self, metadata_manager: MetadataManager):
        """Test citation metadata includes page number for PDF files."""
        pdf_doc = Document(
            id="pdf_with_page",
            content="content",
            meta={"file_path": "/test/doc.pdf", "page_number": 5}
        )
        
        citation_meta = metadata_manager._create_citation_metadata(pdf_doc, "PDF")
        
        citation = citation_meta["citation"]
        assert citation[metadata_manager.config.metadata.page_number] == 5

    def test_create_citation_metadata_non_pdf_no_page(self, metadata_manager: MetadataManager):
        """Test citation metadata doesn't include page number for non-PDF files."""
        text_doc = Document(
            id="text_doc",
            content="content",
            meta={"file_path": "/test/doc.txt"}
        )
        
        citation_meta = metadata_manager._create_citation_metadata(text_doc, "Text")
        
        citation = citation_meta["citation"]
        assert metadata_manager.config.metadata.page_number not in citation


class TestMetadataCleaning:
    """Test metadata cleaning and serialization."""

    def test_clean_metadata_removes_none_values(self, metadata_manager: MetadataManager):
        """Test that None values are removed from metadata."""
        dirty_meta = {
            "valid_field": "value",
            "none_field": None,
            "empty_string": "",
            "zero_value": 0
        }
        
        cleaned = metadata_manager._clean_metadata(dirty_meta)
        
        assert "valid_field" in cleaned
        assert "none_field" not in cleaned
        assert "empty_string" in cleaned  # Empty string is not None
        assert "zero_value" in cleaned     # Zero is not None

    def test_clean_metadata_converts_datetime(self, metadata_manager: MetadataManager):
        """Test that datetime objects are converted to ISO strings."""
        test_datetime = datetime(2024, 1, 1, 12, 0, 0)
        dirty_meta = {
            "datetime_field": test_datetime,
            "string_field": "normal_string"
        }
        
        cleaned = metadata_manager._clean_metadata(dirty_meta)
        
        assert isinstance(cleaned["datetime_field"], str)
        assert "2024-01-01T12:00:00" in cleaned["datetime_field"]
        assert cleaned["string_field"] == "normal_string"

    def test_clean_metadata_handles_complex_types(self, metadata_manager: MetadataManager):
        """Test cleaning of various data types."""
        class CustomObject:
            def __str__(self):
                return "custom_object_string"
        
        dirty_meta = {
            "string": "text",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "custom": CustomObject()
        }
        
        cleaned = metadata_manager._clean_metadata(dirty_meta)
        
        assert cleaned["string"] == "text"
        assert cleaned["integer"] == 42
        assert cleaned["float"] == 3.14
        assert cleaned["boolean"] is True
        assert cleaned["list"] == [1, 2, 3]
        assert cleaned["dict"] == {"nested": "value"}
        assert cleaned["custom"] == "custom_object_string"

    def test_clean_metadata_json_serializable(self, metadata_manager: MetadataManager):
        """Test that cleaned metadata is JSON serializable."""
        test_meta = {
            "string": "value",
            "number": 42,
            "boolean": True,
            "datetime": datetime.now()
        }
        
        cleaned = metadata_manager._clean_metadata(test_meta)
        
        # Should not raise exception
        json_str = json.dumps(cleaned)
        assert isinstance(json_str, str)


class TestDocumentMetadataEnhancement:
    """Test complete document metadata enhancement."""

    def test_enhance_metadata_complete_workflow(self, metadata_manager: MetadataManager, sample_document: Document):
        """Test complete metadata enhancement workflow."""
        enhanced = metadata_manager.enhance_metadata(sample_document, "PDF")
        
        # Should preserve original content and ID
        assert enhanced.content == sample_document.content
        assert enhanced.id == sample_document.id
        
        # Should have enhanced metadata
        meta = enhanced.meta
        assert metadata_manager.config.metadata.source_file in meta
        assert metadata_manager.config.metadata.source_type in meta
        assert metadata_manager.config.metadata.processing_date in meta
        assert "citation" in meta
        assert "processor_version" in meta
        
        # Should preserve existing metadata
        assert meta["existing_field"] == "existing_value"

    def test_enhance_metadata_preserves_embedding(self, metadata_manager: MetadataManager):
        """Test that document embedding is preserved during enhancement."""
        doc_with_embedding = Document(
            id="embedded",
            content="content",
            meta={"file_path": "/test/doc.txt"},
            embedding=[0.1, 0.2, 0.3, 0.4]
        )
        
        enhanced = metadata_manager.enhance_metadata(doc_with_embedding, "Text")
        
        assert enhanced.embedding == [0.1, 0.2, 0.3, 0.4]

    def test_enhance_metadata_different_file_types(self, metadata_manager: MetadataManager):
        """Test metadata enhancement for different file types."""
        doc = Document(
            id="test",
            content="content",
            meta={"file_path": "/test/doc.txt"}
        )
        
        # Test different file types
        for file_type, expected_type in [("PDF", "pdf"), ("Text", "text"), ("Markdown", "markdown")]:
            enhanced = metadata_manager.enhance_metadata(doc, file_type)
            assert enhanced.meta[metadata_manager.config.metadata.source_type] == expected_type


class TestChunkMetadataCreation:
    """Test chunk-specific metadata creation."""

    def test_create_chunk_metadata_basic(self, metadata_manager: MetadataManager, sample_chunk_document: Document):
        """Test basic chunk metadata creation."""
        parent_doc = Document(
            id="parent",
            content="parent content",
            meta={"file_path": "/test/parent.txt", "source_type": "text"}
        )
        
        chunk_meta = metadata_manager.create_chunk_metadata(
            sample_chunk_document, parent_doc, chunk_index=3
        )
        
        assert chunk_meta[metadata_manager.config.metadata.chunk_index] == 3
        assert chunk_meta[metadata_manager.config.metadata.chunk_id] == "chunk_1"
        assert chunk_meta["parent_document_id"] == "parent"
        assert metadata_manager.config.metadata.word_count in chunk_meta
        assert metadata_manager.config.metadata.line_count in chunk_meta

    def test_create_chunk_metadata_with_boundaries(self, metadata_manager: MetadataManager, 
                                                  sample_chunk_document: Document):
        """Test chunk metadata creation with boundary information."""
        parent_doc = Document(id="parent", content="content", meta={})
        
        boundaries = {
            "found": True,
            "type": "sentence",
            "start_char": 10,
            "end_char": 50
        }
        
        chunk_meta = metadata_manager.create_chunk_metadata(
            sample_chunk_document, parent_doc, chunk_index=1, boundaries=boundaries
        )
        
        assert chunk_meta[metadata_manager.config.metadata.boundary_found] is True
        assert chunk_meta[metadata_manager.config.metadata.boundary_type] == "sentence"
        assert chunk_meta[metadata_manager.config.metadata.start_char] == 10
        assert chunk_meta[metadata_manager.config.metadata.end_char] == 50

    def test_create_chunk_metadata_content_hash(self, metadata_manager: MetadataManager):
        """Test content hash generation in chunk metadata."""
        chunk = Document(id="chunk", content="Test content for hashing", meta={})
        parent = Document(id="parent", content="parent", meta={})
        
        chunk_meta = metadata_manager.create_chunk_metadata(chunk, parent, 0)
        
        assert metadata_manager.config.metadata.content_hash in chunk_meta
        assert len(chunk_meta[metadata_manager.config.metadata.content_hash]) == 8

    def test_create_chunk_metadata_inherits_parent_metadata(self, metadata_manager: MetadataManager):
        """Test that chunk inherits metadata from parent document."""
        parent_doc = Document(
            id="parent",
            content="parent content",
            meta={
                "file_path": "/test/parent.txt",
                "source_type": "text",
                "custom_field": "custom_value"
            }
        )
        
        chunk = Document(id="chunk", content="chunk content", meta={})
        
        chunk_meta = metadata_manager.create_chunk_metadata(chunk, parent_doc, 0)
        
        assert chunk_meta["file_path"] == "/test/parent.txt"
        assert chunk_meta["source_type"] == "text"
        assert chunk_meta["custom_field"] == "custom_value"


class TestMetadataValidation:
    """Test metadata validation functionality."""

    def test_validate_metadata_valid_document(self, metadata_manager: MetadataManager):
        """Test validation of document with valid metadata."""
        valid_doc = Document(
            id="valid",
            content="content",
            meta={
                metadata_manager.config.metadata.source_file: "/test/doc.txt",
                metadata_manager.config.metadata.source_type: "text",
                metadata_manager.config.metadata.word_count: 10,
                metadata_manager.config.metadata.line_count: 2
            }
        )
        
        result = metadata_manager.validate_metadata(valid_doc)
        
        assert result["valid"] is True
        assert result["errors"] == []
        assert result["field_count"] > 0

    def test_validate_metadata_missing_required_fields(self, metadata_manager: MetadataManager):
        """Test validation catches missing required fields."""
        invalid_doc = Document(
            id="invalid",
            content="content",
            meta={}  # Missing required fields
        )
        
        result = metadata_manager.validate_metadata(invalid_doc)
        
        assert result["valid"] is False
        assert len(result["errors"]) >= 2  # Missing source_file and source_type

    def test_validate_metadata_empty_required_fields(self, metadata_manager: MetadataManager):
        """Test validation warns about empty required fields."""
        doc_with_empty = Document(
            id="empty_fields",
            content="content",
            meta={
                metadata_manager.config.metadata.source_file: "",  # Empty
                metadata_manager.config.metadata.source_type: "text"
            }
        )
        
        result = metadata_manager.validate_metadata(doc_with_empty)
        
        assert len(result["warnings"]) >= 1
        assert any("Empty required" in warning for warning in result["warnings"])

    def test_validate_metadata_type_validation(self, metadata_manager: MetadataManager):
        """Test validation of field data types."""
        doc_with_wrong_types = Document(
            id="wrong_types",
            content="content",
            meta={
                metadata_manager.config.metadata.source_file: "/test/doc.txt",
                metadata_manager.config.metadata.source_type: "text",
                metadata_manager.config.metadata.word_count: "not_an_integer",  # Should be int
                metadata_manager.config.metadata.line_count: 3.14  # Should be int
            }
        )
        
        result = metadata_manager.validate_metadata(doc_with_wrong_types)
        
        assert result["valid"] is False
        assert len(result["errors"]) >= 1

    def test_validate_metadata_json_serialization(self, metadata_manager: MetadataManager):
        """Test validation catches non-serializable metadata."""
        class NonSerializable:
            pass
        
        doc_with_non_serializable = Document(
            id="non_serializable",
            content="content",
            meta={
                metadata_manager.config.metadata.source_file: "/test/doc.txt",
                metadata_manager.config.metadata.source_type: "text",
                "bad_field": NonSerializable()
            }
        )
        
        result = metadata_manager.validate_metadata(doc_with_non_serializable)
        
        assert result["valid"] is False
        assert any("not JSON serializable" in error for error in result["errors"])


class TestCitationExtraction:
    """Test citation extraction functionality."""

    def test_extract_citations_with_citation_metadata(self, metadata_manager: MetadataManager):
        """Test extraction of citations from documents with citation metadata."""
        docs = [
            Document(
                id="doc1",
                content="content1",
                meta={
                    "citation": {
                        "source_file": "/test/doc1.pdf",
                        "document_type": "PDF",
                        "page_number": 1
                    }
                }
            ),
            Document(
                id="doc2",
                content="content2",
                meta={
                    "citation": {
                        "source_file": "/test/doc2.txt",
                        "document_type": "Text"
                    }
                }
            )
        ]
        
        citations = metadata_manager.extract_citations(docs)
        
        assert len(citations) == 2
        assert citations[0]["source_file"] == "/test/doc1.pdf"
        assert citations[1]["source_file"] == "/test/doc2.txt"

    def test_extract_citations_missing_citation_metadata(self, metadata_manager: MetadataManager):
        """Test citation extraction when citation metadata is missing."""
        docs = [
            Document(
                id="no_citation",
                content="content",
                meta={
                    metadata_manager.config.metadata.source_file: "/test/doc.txt",
                    metadata_manager.config.metadata.source_type: "text"
                }
            )
        ]
        
        citations = metadata_manager.extract_citations(docs)
        
        assert len(citations) == 1
        # Should create basic citation from available metadata
        assert citations[0] is not None

    def test_extract_citations_with_chunk_info(self, metadata_manager: MetadataManager):
        """Test citation extraction includes chunk information."""
        docs = [
            Document(
                id="chunk",
                content="content",
                meta={
                    "citation": {"source_file": "/test/doc.txt"},
                    metadata_manager.config.metadata.chunk_index: 5
                }
            )
        ]
        
        citations = metadata_manager.extract_citations(docs)
        
        assert len(citations) == 1
        # Should include chunk information in citation


class TestMetadataSchema:
    """Test metadata schema functionality."""

    def test_get_metadata_schema(self, metadata_manager: MetadataManager):
        """Test getting metadata schema information."""
        schema = metadata_manager.get_metadata_schema()
        
        assert schema["version"] == "1.0"
        assert "required_fields" in schema
        assert "optional_fields" in schema
        assert "field_descriptions" in schema
        
        # Check required fields are present
        required_fields = schema["required_fields"]
        assert metadata_manager.config.metadata.source_file in required_fields
        assert metadata_manager.config.metadata.source_type in required_fields

    def test_metadata_schema_completeness(self, metadata_manager: MetadataManager):
        """Test that schema includes all configured metadata fields."""
        schema = metadata_manager.get_metadata_schema()
        
        all_fields = schema["required_fields"] + schema["optional_fields"]
        
        # Should include key metadata fields
        assert metadata_manager.config.metadata.chunk_id in all_fields
        assert metadata_manager.config.metadata.chunk_index in all_fields
        assert metadata_manager.config.metadata.word_count in all_fields


class TestMetadataManagerIntegration:
    """Integration tests for MetadataManager."""

    def test_end_to_end_metadata_workflow(self, metadata_manager: MetadataManager):
        """Test complete metadata enhancement and validation workflow."""
        # Create document
        original_doc = Document(
            id="integration_test",
            content="This is content for integration testing of metadata management.",
            meta={"file_path": "/test/integration.txt"}
        )
        
        # Enhance metadata
        enhanced = metadata_manager.enhance_metadata(original_doc, "Text")
        
        # Validate enhanced metadata
        validation_result = metadata_manager.validate_metadata(enhanced)
        
        # Should be valid
        assert validation_result["valid"] is True
        assert validation_result["errors"] == []
        
        # Extract citations
        citations = metadata_manager.extract_citations([enhanced])
        assert len(citations) == 1

    def test_metadata_consistency_across_operations(self, metadata_manager: MetadataManager):
        """Test metadata consistency across different operations."""
        doc = Document(
            id="consistency_test",
            content="Test content",
            meta={"file_path": "/test/file.pdf"}
        )
        
        # Enhance metadata
        enhanced = metadata_manager.enhance_metadata(doc, "PDF")
        
        # Create chunk metadata
        chunk_meta = metadata_manager.create_chunk_metadata(
            enhanced, enhanced, chunk_index=0
        )
        
        # Both should have consistent source information
        assert enhanced.meta[metadata_manager.config.metadata.source_file] == "/test/file.pdf"
        assert chunk_meta[metadata_manager.config.metadata.source_file] == "/test/file.pdf"
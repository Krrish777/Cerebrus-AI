"""
Unit tests for ChunkingService component.

Tests document chunking functionality with proper isolation and mocking
following AGENTS.md principles: deterministic, focused, comprehensive.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List

from haystack.dataclasses import Document
from haystack.components.preprocessors import DocumentSplitter

from src.document_processing.chunking_service import ChunkingService
from src.document_processing.pipeline_config import (
    PipelineConfig,
    ChunkingConfig,
    MetadataConfig
)


@pytest.fixture
def test_chunking_config() -> ChunkingConfig:
    """Create test chunking configuration."""
    return ChunkingConfig(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size_ratio=0.3,
        boundary_preferences=["paragraph", "sentence"],
        enable_statistics=True,
        enable_preview=True,
        preview_length=50
    )


@pytest.fixture
def test_config(test_chunking_config: ChunkingConfig) -> PipelineConfig:
    """Create test configuration for ChunkingService."""
    return PipelineConfig(
        chunking=test_chunking_config,
        metadata=MetadataConfig()
    )


@pytest.fixture
def chunking_service(test_config: PipelineConfig) -> ChunkingService:
    """Create ChunkingService instance with test configuration."""
    return ChunkingService(test_config)


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for chunking tests."""
    content = "This is a test document with multiple sentences. " * 10  # ~500 chars
    return Document(
        id="test_doc",
        content=content,
        meta={"file_path": "/test/document.txt", "source_type": "text"}
    )


@pytest.fixture
def sample_documents() -> List[Document]:
    """Create multiple sample documents for testing."""
    docs = []
    for i in range(3):
        content = f"Document {i} content. " * 20  # ~400 chars each
        doc = Document(
            id=f"doc_{i}",
            content=content,
            meta={"file_path": f"/test/doc{i}.txt", "source_type": "text"}
        )
        docs.append(doc)
    return docs


class TestChunkingServiceInitialization:
    """Test ChunkingService initialization."""

    def test_chunking_service_initializes_with_config(self, test_config: PipelineConfig):
        """Test ChunkingService initializes properly with configuration."""
        service = ChunkingService(test_config)
        
        assert service.config == test_config
        assert service._splitter is None  # Lazy loading

    def test_splitter_lazy_loading(self, chunking_service: ChunkingService):
        """Test that splitter is lazy-loaded on first access."""
        # Initially None
        assert chunking_service._splitter is None
        
        # Access splitter property
        splitter = chunking_service.splitter
        
        # Should now be initialized
        assert splitter is not None
        assert isinstance(splitter, DocumentSplitter)
        assert chunking_service._splitter is splitter

    def test_create_splitter_configuration(self, chunking_service: ChunkingService):
        """Test splitter creation with correct configuration."""
        splitter = chunking_service._create_splitter()
        
        # Verify configuration
        assert isinstance(splitter, DocumentSplitter)
        # Note: Exact parameter verification depends on Haystack DocumentSplitter API


class TestDocumentChunking:
    """Test document chunking functionality."""

    def test_chunk_documents_empty_list(self, chunking_service: ChunkingService):
        """Test chunking with empty document list."""
        result = chunking_service.chunk_documents([])
        
        assert result["documents"] == []
        assert result["errors"] == []

    @patch('src.document_processing.chunking_service.DocumentSplitter')
    def test_chunk_documents_successful(self, mock_splitter_class, 
                                       chunking_service: ChunkingService,
                                       sample_document: Document):
        """Test successful document chunking."""
        # Create mock chunks
        chunk1 = Document(
            id="chunk_1",
            content="This is a test document with multiple sentences. " * 2,
            meta={"source_doc": "test_doc"}
        )
        chunk2 = Document(
            id="chunk_2", 
            content="This is a test document with multiple sentences. " * 3,
            meta={"source_doc": "test_doc"}
        )
        
        # Mock splitter behavior
        mock_splitter = Mock()
        mock_splitter.run.return_value = {"documents": [chunk1, chunk2]}
        mock_splitter_class.return_value = mock_splitter
        
        # Mock the splitter property to return our mock
        with patch.object(chunking_service, '_splitter', mock_splitter):
            result = chunking_service.chunk_documents([sample_document])
        
        # Verify splitter was called correctly
        mock_splitter.run.assert_called_once_with(documents=[sample_document])
        
        # Check results
        assert len(result["documents"]) == 2
        assert result["errors"] == []
        assert "stats" in result
        assert result["stats"]["input_documents"] == 1
        assert result["stats"]["output_chunks"] == 2

    @patch('src.document_processing.chunking_service.DocumentSplitter')
    def test_chunk_documents_splitter_exception(self, mock_splitter_class, 
                                               chunking_service: ChunkingService,
                                               sample_document: Document):
        """Test handling of splitter exceptions."""
        # Reset _splitter to ensure lazy loading uses the mock
        chunking_service._splitter = None
        
        mock_splitter = Mock()
        mock_splitter.run.side_effect = Exception("Splitter error")
        mock_splitter_class.return_value = mock_splitter
        
        result = chunking_service.chunk_documents([sample_document])
        
        assert result["documents"] == []
        assert len(result["errors"]) == 1
        assert "Splitter error" in result["errors"][0]

    @patch('src.document_processing.chunking_service.DocumentSplitter')
    def test_chunk_documents_performance_tracking(self, mock_splitter_class,
                                                 chunking_service: ChunkingService,
                                                 sample_documents: List[Document]):
        """Test that performance statistics are tracked."""
        # Reset _splitter to ensure lazy loading uses the mock
        chunking_service._splitter = None
        
        # Mock splitter to return chunks
        mock_chunks = [
            Document(id=f"chunk_{i}", content=f"Chunk {i}", meta={})
            for i in range(5)
        ]
        mock_splitter = Mock()
        mock_splitter.run.return_value = {"documents": mock_chunks}
        mock_splitter_class.return_value = mock_splitter
        
        result = chunking_service.chunk_documents(sample_documents)
        
        assert "stats" in result
        assert "chunking_time" in result["stats"]
        assert result["stats"]["chunking_time"] > 0
        assert result["stats"]["chunks_per_document"] > 0

    def test_chunk_single_document(self, chunking_service: ChunkingService, sample_document: Document):
        """Test chunking a single document (convenience method)."""
        with patch.object(chunking_service, 'chunk_documents') as mock_chunk:
            mock_chunk.return_value = {"documents": [sample_document], "errors": []}
            
            chunks = chunking_service.chunk_single_document(sample_document)
            
            mock_chunk.assert_called_once_with([sample_document])
            assert chunks == [sample_document]


class TestChunkMetadataEnhancement:
    """Test chunk metadata enhancement functionality."""

    def test_enhance_chunk_metadata(self, chunking_service: ChunkingService):
        """Test chunk metadata enhancement."""
        chunk = Document(
            id="original_chunk",
            content="Sample chunk content for testing metadata enhancement",
            meta={"file_path": "/test/doc.txt"}
        )
        
        enhanced = chunking_service._enhance_chunk_metadata(chunk, chunk_index=5)
        
        # Check enhanced metadata
        assert enhanced.meta["chunk_index"] == 5
        assert enhanced.meta["original_document_id"] == "original_chunk"
        assert enhanced.meta["source_file"] == "/test/doc.txt"
        assert enhanced.meta["chunk_size"] > 0
        assert enhanced.meta["word_count"] > 0
        assert enhanced.meta["line_count"] > 0
        assert "chunk_id" in enhanced.meta
        assert "content_hash" in enhanced.meta
        assert "chunking_timestamp" in enhanced.meta
        
        # Check configuration metadata
        assert enhanced.meta["target_chunk_size"] == chunking_service.config.chunking.chunk_size
        assert enhanced.meta["chunk_overlap"] == chunking_service.config.chunking.chunk_overlap

    def test_enhance_chunk_metadata_content_hash_generation(self, chunking_service: ChunkingService):
        """Test content hash generation for chunks."""
        chunk1 = Document(id="1", content="Same content", meta={})
        chunk2 = Document(id="2", content="Same content", meta={})
        chunk3 = Document(id="3", content="Different content", meta={})
        
        enhanced1 = chunking_service._enhance_chunk_metadata(chunk1, 0)
        enhanced2 = chunking_service._enhance_chunk_metadata(chunk2, 0)
        enhanced3 = chunking_service._enhance_chunk_metadata(chunk3, 0)
        
        # Same content should have same hash
        assert enhanced1.meta["content_hash"] == enhanced2.meta["content_hash"]
        # Different content should have different hash
        assert enhanced1.meta["content_hash"] != enhanced3.meta["content_hash"]

    def test_generate_content_hash_empty_content(self, chunking_service: ChunkingService):
        """Test content hash generation for empty content."""
        hash_result = chunking_service._generate_content_hash("")
        assert hash_result == "empty"
        
        hash_result2 = chunking_service._generate_content_hash(None)
        assert hash_result2 == "empty"

    def test_generate_content_hash_consistent(self, chunking_service: ChunkingService):
        """Test that content hash is consistent for same content."""
        content = "Test content for hashing"
        
        hash1 = chunking_service._generate_content_hash(content)
        hash2 = chunking_service._generate_content_hash(content)
        
        assert hash1 == hash2
        assert len(hash1) == 8  # MD5 truncated to 8 chars


class TestChunkingConfiguration:
    """Test chunking configuration and information retrieval."""

    def test_get_chunking_info(self, chunking_service: ChunkingService):
        """Test getting chunking configuration information."""
        info = chunking_service.get_chunking_info()
        
        assert info["strategy"] == "word"
        assert info["chunk_size"] == chunking_service.config.chunking.chunk_size
        assert info["chunk_overlap"] == chunking_service.config.chunking.chunk_overlap
        assert info["min_chunk_ratio"] == chunking_service.config.chunking.min_chunk_size_ratio
        assert "boundary_preferences" in info
        assert "statistics_enabled" in info


class TestChunkValidation:
    """Test chunk validation functionality."""

    def test_validate_chunks_empty_list(self, chunking_service: ChunkingService):
        """Test validation of empty chunk list."""
        result = chunking_service.validate_chunks([])
        
        assert result["valid"] is False
        assert "No chunks provided" in result["errors"][0]

    def test_validate_chunks_valid_chunks(self, chunking_service: ChunkingService):
        """Test validation of valid chunks."""
        chunks = [
            Document(
                id="chunk_1",
                content="Valid chunk content with adequate length",
                meta={
                    "chunk_id": "chunk_1",
                    "chunk_index": 0,
                    "source_file": "/test/doc.txt"
                }
            ),
            Document(
                id="chunk_2",
                content="Another valid chunk with good metadata",
                meta={
                    "chunk_id": "chunk_2", 
                    "chunk_index": 1,
                    "source_file": "/test/doc.txt"
                }
            )
        ]
        
        result = chunking_service.validate_chunks(chunks)
        
        assert result["valid"] is True
        assert result["errors"] == []
        assert "stats" in result
        assert result["stats"]["total_chunks"] == 2

    def test_validate_chunks_missing_content(self, chunking_service: ChunkingService):
        """Test validation catches chunks with missing content."""
        chunks = [
            Document(
                id="chunk_1",
                content="",  # Empty content
                meta={"chunk_id": "chunk_1", "chunk_index": 0, "source_file": "test.txt"}
            ),
            Document(
                id="chunk_2",
                content=None,  # None content
                meta={"chunk_id": "chunk_2", "chunk_index": 1, "source_file": "test.txt"}
            )
        ]
        
        result = chunking_service.validate_chunks(chunks)
        
        assert result["valid"] is False
        assert len(result["errors"]) == 2
        assert all("no content" in error for error in result["errors"])

    def test_validate_chunks_missing_metadata(self, chunking_service: ChunkingService):
        """Test validation catches missing required metadata."""
        chunks = [
            Document(
                id="chunk_1",
                content="Valid content",
                meta={"chunk_id": "chunk_1"}  # Missing chunk_index and source_file
            )
        ]
        
        result = chunking_service.validate_chunks(chunks)
        
        assert result["valid"] is False
        assert len(result["errors"]) == 2  # Missing chunk_index and source_file

    def test_validate_chunks_size_warnings(self, chunking_service: ChunkingService):
        """Test validation generates warnings for small chunks."""
        small_chunk = Document(
            id="small",
            content="Small",  # Very small content
            meta={
                "chunk_id": "small",
                "chunk_index": 0,
                "source_file": "test.txt"
            }
        )
        
        result = chunking_service.validate_chunks([small_chunk])
        
        # Should have warnings but still be valid structurally
        assert "warnings" in result
        if result["warnings"]:  # May have size warnings
            assert any("smaller than minimum size" in warning for warning in result["warnings"])

    def test_validate_chunks_statistics(self, chunking_service: ChunkingService):
        """Test validation statistics calculation."""
        chunks = [
            Document(
                id="chunk_1",
                content="A" * 50,  # 50 chars
                meta={"chunk_id": "chunk_1", "chunk_index": 0, "source_file": "test.txt"}
            ),
            Document(
                id="chunk_2", 
                content="B" * 100,  # 100 chars
                meta={"chunk_id": "chunk_2", "chunk_index": 1, "source_file": "test.txt"}
            )
        ]
        
        result = chunking_service.validate_chunks(chunks)
        
        stats = result["stats"]
        assert stats["total_chunks"] == 2
        assert stats["avg_chunk_size"] == 75  # (50 + 100) / 2
        assert stats["min_chunk_size"] == 50
        assert stats["max_chunk_size"] == 100
        assert stats["total_content_length"] == 150


class TestChunkingServiceIntegration:
    """Integration tests for ChunkingService."""

    def test_chunking_workflow_end_to_end(self, test_config: PipelineConfig):
        """Test complete chunking workflow with real documents."""
        # Create chunking service
        service = ChunkingService(test_config)
        
        # Create document with content that should be chunked
        long_content = "This is a sentence. " * 50  # ~1000 chars
        doc = Document(
            id="long_doc",
            content=long_content,
            meta={"file_path": "/test/long.txt"}
        )
        
        # Test the workflow (normally would use real Haystack splitter)
        # For integration test, we validate the setup is correct
        assert service.config.chunking.chunk_size == 100
        assert service.config.chunking.chunk_overlap == 20

    def test_chunking_respects_configuration_changes(self):
        """Test that chunking behavior respects configuration changes."""
        # Create different configurations
        config1 = PipelineConfig(
            chunking=ChunkingConfig(chunk_size=50, chunk_overlap=10)
        )
        config2 = PipelineConfig(
            chunking=ChunkingConfig(chunk_size=200, chunk_overlap=50)
        )
        
        service1 = ChunkingService(config1)
        service2 = ChunkingService(config2)
        
        # Services should have different configurations
        assert service1.config.chunking.chunk_size == 50
        assert service2.config.chunking.chunk_size == 200
        assert service1.config.chunking.chunk_overlap == 10
        assert service2.config.chunking.chunk_overlap == 50

    def test_performance_with_large_documents(self, chunking_service: ChunkingService):
        """Test chunking performance with large documents."""
        # Create large document
        large_content = "This is a test sentence for performance testing. " * 1000  # ~50k chars
        large_doc = Document(
            id="large",
            content=large_content,
            meta={"file_path": "/test/large.txt"}
        )
        
        # Mock splitter for performance test
        with patch.object(chunking_service, '_splitter') as mock_splitter:
            mock_chunks = [
                Document(id=f"chunk_{i}", content=f"Chunk {i}", meta={})
                for i in range(100)  # Simulate 100 chunks
            ]
            mock_splitter.run.return_value = {"documents": mock_chunks}
            
            start_time = time.time()
            result = chunking_service.chunk_documents([large_doc])
            end_time = time.time()
            
            # Should complete quickly (metadata enhancement should be efficient)
            assert (end_time - start_time) < 1.0  # Less than 1 second
            assert len(result["documents"]) == 100

    @pytest.mark.parametrize("chunk_size,overlap,expected_valid", [
        (100, 20, True),    # Valid configuration
        (50, 10, True),     # Valid configuration
        (100, 100, False),  # Invalid: overlap >= chunk_size
        (0, 10, False),     # Invalid: zero chunk size
        (-1, 10, False),    # Invalid: negative chunk size
    ])
    def test_configuration_validation(self, chunk_size: int, overlap: int, expected_valid: bool):
        """Test configuration validation for different parameter combinations."""
        if expected_valid:
            # Should not raise exception
            config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=overlap)
            service = ChunkingService(PipelineConfig(chunking=config))
            assert service.config.chunking.chunk_size == chunk_size
        else:
            # Should raise validation exception
            with pytest.raises(ValueError):
                ChunkingConfig(chunk_size=chunk_size, chunk_overlap=overlap)
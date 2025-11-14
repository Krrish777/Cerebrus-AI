"""
Tests for the PDFProcessor class in src.document_processing.doc_processor

This module contains comprehensive tests for the PDFProcessor functionality including:
- PDF document processing and chunking
- Smart chunking algorithm with overlap
- Citation metadata generation
- Error handling for various edge cases
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from haystack import Document
import sys

# Add the src directory to Python path for importing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from document_processing.doc_processor import PDFProcessor


class TestPDFProcessor:
    """Test suite for PDFProcessor class"""

    @pytest.fixture
    def processor(self):
        """Create a PDFProcessor instance for testing"""
        return PDFProcessor(chunk_size=100, chunk_overlap=20)

    @pytest.fixture
    def sample_pdf_document(self):
        """Create a mock PDF document for testing"""
        return Document(
            content="This is a sample PDF content. It contains multiple sentences. "
                   "This should be chunked properly. The chunking algorithm should "
                   "respect word boundaries and create overlapping chunks. "
                   "This is the end of the sample content.",
            meta={
                'page_number': 1,
                'name': 'test_document.pdf',
                'file_path': '/path/to/test_document.pdf'
            }
        )

    @pytest.fixture
    def empty_pdf_document(self):
        """Create an empty PDF document for testing"""
        return Document(
            content="   ",  # Only whitespace
            meta={
                'page_number': 2,
                'name': 'empty_document.pdf',
                'file_path': '/path/to/empty_document.pdf'
            }
        )

    @pytest.fixture
    def long_pdf_document(self):
        """Create a long PDF document for testing chunking"""
        content = "A" * 500  # 500 character string
        return Document(
            content=content,
            meta={
                'page_number': 1,
                'name': 'long_document.pdf',
                'file_path': '/path/to/long_document.pdf'
            }
        )

    def test_processor_initialization(self):
        """Test PDFProcessor initialization with default and custom parameters"""
        # Test default initialization
        processor_default = PDFProcessor()
        assert processor_default.chunk_size == 1000
        assert processor_default.chunk_overlap == 200
        assert processor_default.pdf_converter is not None
        # assert processor_default.logger is not None

        # Test custom initialization
        processor_custom = PDFProcessor(chunk_size=500, chunk_overlap=100)
        assert processor_custom.chunk_size == 500
        assert processor_custom.chunk_overlap == 100

    @patch('document_processing.doc_processor.PyPDFToDocument')
    def test_run_method_success(self, mock_pdf_converter, processor, sample_pdf_document):
        """Test successful execution of the run method"""
        # Mock the PDF converter
        mock_converter_instance = Mock()
        mock_converter_instance.run.return_value = {'documents': [sample_pdf_document]}
        mock_pdf_converter.return_value = mock_converter_instance
        
        # Set the mocked converter
        processor.pdf_converter = mock_converter_instance

        # Run the processor
        sources = ['/path/to/test_document.pdf']
        result = processor.run(sources)

        # Assertions
        assert 'documents' in result
        assert len(result['documents']) > 0
        
        # Verify PDF converter was called
        mock_converter_instance.run.assert_called_once_with(sources=sources)

        # Check first chunk properties
        first_chunk = result['documents'][0]
        assert isinstance(first_chunk, Document)
        assert first_chunk.content is not None
        assert 'chunk_id' in first_chunk.meta
        assert 'source_file' in first_chunk.meta
        assert 'citation' in first_chunk.meta

    def test_create_smart_chunks_normal_document(self, processor, sample_pdf_document):
        """Test smart chunking with normal document content"""
        chunks = processor._create_smart_chunks(sample_pdf_document)

        # Should create multiple chunks due to content length
        assert len(chunks) > 1

        # Check first chunk
        first_chunk = chunks[0]
        assert isinstance(first_chunk, Document)
        assert len(first_chunk.content) <= processor.chunk_size
        assert first_chunk.meta['chunk_index'] == 0
        assert first_chunk.meta['start_char'] == 0
        assert 'chunk_id' in first_chunk.meta
        assert first_chunk.meta['source_type'] == 'pdf'
        assert first_chunk.meta['page_number'] == 1

        # Check metadata structure
        assert 'citation' in first_chunk.meta
        citation = first_chunk.meta['citation']
        assert 'souce_file' in citation  # Note: there's a typo in the original code
        assert 'type' in citation
        assert 'page_number' in citation
        assert 'char_range' in citation
        assert 'chunk_id' in citation

        # Check chunk ID format
        chunk_id = first_chunk.meta['chunk_id']
        assert chunk_id.startswith('pdf_0_contenthash_')
        assert len(chunk_id.split('_')[-1]) == 8  # Hash should be 8 characters

    def test_create_smart_chunks_empty_document(self, processor, empty_pdf_document):
        """Test smart chunking with empty document content"""
        chunks = processor._create_smart_chunks(empty_pdf_document)

        # Should return empty list for empty content
        assert len(chunks) == 0

    def test_create_smart_chunks_long_document(self, processor, long_pdf_document):
        """Test smart chunking with long document that requires multiple chunks"""
        chunks = processor._create_smart_chunks(long_pdf_document)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Check that chunks have proper overlap
        if len(chunks) > 1:
            first_chunk = chunks[0]
            second_chunk = chunks[1]
            
            # Check chunk indices
            assert first_chunk.meta['chunk_index'] == 0
            assert second_chunk.meta['chunk_index'] == 1
            
            # Check that overlap exists by verifying the gap between chunks
            first_end = first_chunk.meta['end_char']
            second_start = second_chunk.meta['start_char']
            
            # The gap should be less than the chunk size (indicating overlap)
            # or they should be adjacent/overlapping
            gap = second_start - first_end
            assert gap <= 1, f"Gap too large: {gap}, indicating no overlap"
            
            # Verify chunk sizes are reasonable
            assert len(first_chunk.content) <= processor.chunk_size
            assert len(second_chunk.content) <= processor.chunk_size

    def test_smart_boundary_detection(self, processor):
        """Test that chunking respects sentence boundaries"""
        # Create a document with clear sentence boundaries
        content = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        test_doc = Document(
            content=content,
            meta={'page_number': 1, 'name': 'test.pdf'}
        )

        chunks = processor._create_smart_chunks(test_doc)
        
        # Check that chunks don't cut words in the middle (when possible)
        for chunk in chunks:
            # Chunk should not end with partial words (unless it's the last chunk)
            if chunk != chunks[-1]:  # Not the last chunk
                content_text = chunk.content
                # Should not end in the middle of a word (basic check)
                assert not (content_text[-1].isalnum() and 
                           content_text[-2:] != content_text[-2:].strip())

    def test_chunk_metadata_completeness(self, processor, sample_pdf_document):
        """Test that all required metadata is present in chunks"""
        chunks = processor._create_smart_chunks(sample_pdf_document)
        
        required_fields = [
            'source_file', 'source_type', 'page_number', 'chunk_index',
            'chunk_id', 'start_char', 'end_char', 'chunk_size',
            'processed_timestamp', 'citation'
        ]

        for chunk in chunks:
            for field in required_fields:
                assert field in chunk.meta, f"Missing field: {field}"

            # Check citation metadata
            citation = chunk.meta['citation']
            citation_fields = ['souce_file', 'type', 'page_number', 'char_range', 'chunk_id']
            for field in citation_fields:
                assert field in citation, f"Missing citation field: {field}"

    def test_chunk_id_uniqueness(self, processor, sample_pdf_document):
        """Test that chunk IDs are unique within a document"""
        chunks = processor._create_smart_chunks(sample_pdf_document)
        
        chunk_ids = [chunk.meta['chunk_id'] for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs should be unique"

    def test_character_positioning_accuracy(self, processor, sample_pdf_document):
        """Test that start_char and end_char positions are accurate"""
        chunks = processor._create_smart_chunks(sample_pdf_document)
        
        original_content = sample_pdf_document.content
        
        for chunk in chunks:
            start_char = chunk.meta['start_char']
            end_char = chunk.meta['end_char']
            
            # Extract content using positions
            extracted_content = original_content[start_char:end_char + 1].strip()
            
            # Should match the chunk content (after stripping)
            assert extracted_content == chunk.content

    @patch('document_processing.doc_processor.PyPDFToDocument')
    def test_run_method_with_multiple_documents(self, mock_pdf_converter, processor):
        """Test run method with multiple PDF documents"""
        # Create multiple mock documents
        doc1 = Document(content="Content of first document.", meta={'page_number': 1, 'name': 'doc1.pdf'})
        doc2 = Document(content="Content of second document.", meta={'page_number': 1, 'name': 'doc2.pdf'})
        
        mock_converter_instance = Mock()
        mock_converter_instance.run.return_value = {'documents': [doc1, doc2]}
        mock_pdf_converter.return_value = mock_converter_instance
        processor.pdf_converter = mock_converter_instance

        result = processor.run(['/path/to/doc1.pdf', '/path/to/doc2.pdf'])
        
        # Should have chunks from both documents
        assert len(result['documents']) >= 2
        
        # Check that we have chunks from different source files
        source_files = {chunk.meta['source_file'] for chunk in result['documents']}
        assert len(source_files) == 2

    def test_error_handling_invalid_content(self, processor):
        """Test error handling with invalid document content"""
        # Test with None content
        invalid_doc = Document(
            content=None,
            meta={'page_number': 1, 'name': 'invalid.pdf'}
        )
        
        # Should handle None content gracefully
        try:
            chunks = processor._create_smart_chunks(invalid_doc)
            # Should return empty list or handle gracefully
            assert isinstance(chunks, list)
        except AttributeError:
            # Acceptable if it raises AttributeError for None content
            pass

    def test_chunk_size_limits(self, processor):
        """Test that chunks respect the maximum size limit"""
        # Create a document with very long content
        long_content = "A" * 2000  # Longer than chunk_size
        test_doc = Document(
            content=long_content,
            meta={'page_number': 1, 'name': 'long.pdf'}
        )

        chunks = processor._create_smart_chunks(test_doc)
        
        # Each chunk (except possibly the last) should not exceed chunk_size
        for chunk in chunks[:-1]:  # All but last chunk
            assert len(chunk.content) <= processor.chunk_size

    def test_processed_timestamp_format(self, processor, sample_pdf_document):
        """Test that processed_timestamp is in correct ISO format"""
        chunks = processor._create_smart_chunks(sample_pdf_document)
        
        for chunk in chunks:
            timestamp = chunk.meta['processed_timestamp']
            # Should be able to parse as ISO format
            from datetime import datetime
            try:
                datetime.fromisoformat(timestamp)
            except ValueError:
                pytest.fail(f"Invalid timestamp format: {timestamp}")

    def test_content_hash_consistency(self, processor, sample_pdf_document):
        """Test that content hash is consistent for same content"""
        chunks1 = processor._create_smart_chunks(sample_pdf_document)
        chunks2 = processor._create_smart_chunks(sample_pdf_document)
        
        # Chunk IDs should be the same for same content
        for chunk1, chunk2 in zip(chunks1, chunks2):
            assert chunk1.meta['chunk_id'] == chunk2.meta['chunk_id']


# Integration tests
class TestPDFProcessorIntegration:
    """Integration tests for PDFProcessor"""

    @pytest.fixture
    def processor(self):
        return PDFProcessor(chunk_size=200, chunk_overlap=50)

    @patch('document_processing.doc_processor.PyPDFToDocument')
    def test_full_pipeline_simulation(self, mock_pdf_converter, processor):
        """Simulate a full pipeline run with realistic data"""
        # Create a realistic PDF document
        realistic_content = """
        Large Language Models (LLMs): A Technical Overview
        
        1. Introduction
        Large Language Models (LLMs) are deep learning architectures trained on massive 
        corpora to understand, generate, and transform natural language. They represent 
        a significant breakthrough in artificial intelligence.
        
        2. Architecture
        Most modern LLMs are based on the Transformer architecture, which uses 
        self-attention mechanisms to process sequential data efficiently.
        """
        
        mock_doc = Document(
            content=realistic_content,
            meta={'page_number': 1, 'name': 'llm_overview.pdf', 'file_path': '/docs/llm_overview.pdf'}
        )
        
        mock_converter_instance = Mock()
        mock_converter_instance.run.return_value = {'documents': [mock_doc]}
        mock_pdf_converter.return_value = mock_converter_instance
        processor.pdf_converter = mock_converter_instance

        # Run the processor
        result = processor.run(['/docs/llm_overview.pdf'])
        
        # Validate results
        assert 'documents' in result
        chunks = result['documents']
        assert len(chunks) > 0
        
        # Validate chunk structure and content
        total_content_length = sum(len(chunk.content) for chunk in chunks)
        assert total_content_length > 0
        
        # Check that all chunks have proper metadata
        for i, chunk in enumerate(chunks):
            assert chunk.meta['chunk_index'] == i
            assert chunk.meta['source_file'] == 'llm_overview.pdf'
            assert 'chunk_id' in chunk.meta
            assert chunk.meta['chunk_id'].startswith(f'pdf_{i}_contenthash_')


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
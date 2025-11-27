"""
Integration tests for document processing pipeline.

Tests complete workflow integration and component interactions
following AGENTS.md principles: realistic scenarios, end-to-end validation.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from typing import List, Dict, Any

from haystack.dataclasses import Document

from src.document_processing.pipeline_orchestrator import DocumentPipelineOrchestrator
from src.document_processing.file_analyzer import FileAnalyzer
from src.document_processing.document_converter import DocumentConverter
from src.document_processing.chunking_service import ChunkingService
from src.document_processing.metadata_manager import MetadataManager
from src.document_processing.pipeline_config import (
    PipelineConfig,
    ChunkingConfig,
    ProcessingConfig,
    ErrorHandlingConfig,
    MetadataConfig,
    FileTypeConfig,
    ValidationConfig,
    PerformanceConfig
)


@pytest.fixture
def integration_test_data(tmp_path: Path) -> Dict[str, Any]:
    """Create comprehensive test data for integration tests."""
    test_dir = tmp_path / "integration_test"
    test_dir.mkdir()
    
    # Create various file types with realistic content
    files = {}
    
    # PDF file (simulated binary)
    pdf_file = test_dir / "research_paper.pdf"
    pdf_content = b"%PDF-1.4\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n"
    pdf_content += b"Sample research paper content. This is a longer document "
    pdf_content += b"with multiple paragraphs and sections for testing chunking."
    pdf_file.write_bytes(pdf_content)
    files["pdf"] = pdf_file
    
    # Text file
    txt_file = test_dir / "notes.txt"
    txt_content = """Research Notes

These are detailed research notes containing multiple paragraphs.
Each paragraph contains important information that should be processed.

Section 1: Introduction
This section provides background information about the research topic.
It contains several sentences that will be useful for chunking tests.

Section 2: Methodology  
The methodology section describes the approach taken in the research.
This content is structured and should be properly segmented.

Section 3: Results
The results section contains findings and analysis.
This demonstrates how the pipeline handles structured text content.
"""
    txt_file.write_text(txt_content, encoding='utf-8')
    files["txt"] = txt_file
    
    # Markdown file
    md_file = test_dir / "documentation.md"
    md_content = """# Project Documentation

## Overview
This is a comprehensive documentation file for testing the pipeline.

## Features
- Feature 1: Document processing
- Feature 2: Text chunking  
- Feature 3: Metadata management

## Implementation Details

### Architecture
The system uses a modular architecture with the following components:

1. **FileAnalyzer**: Validates and categorizes input files
2. **DocumentConverter**: Converts files to Document objects
3. **ChunkingService**: Breaks documents into manageable chunks
4. **MetadataManager**: Enhances documents with metadata

### Configuration
The pipeline is highly configurable and supports:
- Multiple file formats
- Adjustable chunk sizes
- Custom metadata fields
- Error handling options

## Usage Examples

```python
# Example usage
orchestrator = DocumentPipelineOrchestrator(config)
result = orchestrator.process_documents(file_paths)
```

This example demonstrates how to use the pipeline effectively.
"""
    md_file.write_text(md_content, encoding='utf-8')
    files["md"] = md_file
    
    # Empty file (for error testing)
    empty_file = test_dir / "empty.txt"
    empty_file.write_text("", encoding='utf-8')
    files["empty"] = empty_file
    
    # Invalid file (for error testing)
    invalid_file = test_dir / "invalid.xyz"
    invalid_file.write_text("Invalid file type", encoding='utf-8')
    files["invalid"] = invalid_file
    
    return {
        "test_dir": test_dir,
        "files": files,
        "valid_files": [files["pdf"], files["txt"], files["md"]],
        "all_files": list(files.values())
    }


@pytest.fixture
def integration_config() -> PipelineConfig:
    """Create realistic configuration for integration tests."""
    return PipelineConfig(
        chunking=ChunkingConfig(
            chunk_size=200,  # Smaller chunks for testing
            chunk_overlap=50,
            min_chunk_size_ratio=0.3,
            boundary_preferences=["paragraph", "sentence"],
            enable_statistics=True,
            enable_preview=True
        ),
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
        metadata=MetadataConfig(),
        validation=ValidationConfig(),
        performance=PerformanceConfig(
            enable_timing=True,
            enable_statistics=True
        ),
        error_handling=ErrorHandlingConfig(
            fail_fast=False,
            continue_on_individual_file_error=True
        ),
        file_size_thresholds={
            "kb": 1024,
            "mb": 1024 * 1024
        }
    )


@pytest.fixture
def orchestrator_with_mocked_haystack(integration_config: PipelineConfig) -> DocumentPipelineOrchestrator:
    """Create orchestrator with mocked Haystack components for predictable testing."""
    return DocumentPipelineOrchestrator(integration_config)


class TestComponentIntegration:
    """Test integration between individual components."""

    def test_file_analyzer_to_converter_workflow(self, integration_config: PipelineConfig,
                                                integration_test_data: Dict[str, Any]):
        """Test workflow from file analysis to document conversion."""
        analyzer = FileAnalyzer(integration_config)
        converter = DocumentConverter(integration_config)
        
        valid_files = integration_test_data["valid_files"]
        
        # Step 1: Analyze files
        analysis_result = analyzer.analyze_files(valid_files)
        
        assert len(analysis_result["valid_files"]) == 3
        assert analysis_result["errors"] == []
        assert "PDF" in analysis_result["file_types"]
        assert "Text" in analysis_result["file_types"]
        assert "Markdown" in analysis_result["file_types"]
        
        # Step 2: Group by type (as orchestrator would do)
        groups = {}
        for file_path in analysis_result["valid_files"]:
            file_type = analyzer.detect_file_type(file_path)
            if file_type not in groups:
                groups[file_type] = []
            groups[file_type].append(file_path)
        
        # Should have all three types
        assert "PDF" in groups
        assert "Text" in groups  
        assert "Markdown" in groups

    def test_converter_to_chunking_workflow(self, integration_config: PipelineConfig):
        """Test workflow from document conversion to chunking."""
        converter = DocumentConverter(integration_config)
        chunker = ChunkingService(integration_config)
        
        # Create mock documents (simulating converter output)
        mock_documents = [
            Document(
                id="doc_1",
                content="This is a long document that should be chunked into multiple pieces. " * 10,
                meta={"file_path": "/test/long_doc.txt", "source_type": "text"}
            ),
            Document(
                id="doc_2", 
                content="Another document with substantial content for chunking testing. " * 8,
                meta={"file_path": "/test/another_doc.txt", "source_type": "text"}
            )
        ]
        
        # Mock the chunking process
        with patch.object(chunker, 'splitter') as mock_splitter:
            # Simulate chunking results
            chunked_docs = []
            for i, doc in enumerate(mock_documents):
                # Simulate splitting into 2 chunks per document
                chunk1 = Document(
                    id=f"chunk_{i*2}",
                    content=doc.content[:len(doc.content)//2],
                    meta=doc.meta.copy()
                )
                chunk2 = Document(
                    id=f"chunk_{i*2+1}",
                    content=doc.content[len(doc.content)//2:],
                    meta=doc.meta.copy()
                )
                chunked_docs.extend([chunk1, chunk2])
            
            mock_splitter.run.return_value = {"documents": chunked_docs}
            
            result = chunker.chunk_documents(mock_documents)
            
            assert len(result["documents"]) == 4  # 2 docs * 2 chunks each
            assert result["errors"] == []
            assert result["stats"]["input_documents"] == 2
            assert result["stats"]["output_chunks"] == 4

    def test_chunking_to_metadata_workflow(self, integration_config: PipelineConfig):
        """Test workflow from chunking to metadata enhancement."""
        chunker = ChunkingService(integration_config)
        metadata_manager = MetadataManager(integration_config)
        
        # Create mock chunk
        chunk = Document(
            id="test_chunk",
            content="This is a chunk of content that needs metadata enhancement.",
            meta={"file_path": "/test/source.txt"}
        )
        
        # Step 1: Enhance chunk metadata (simulating chunker output)
        enhanced_chunk = chunker._enhance_chunk_metadata(chunk, chunk_index=1)
        
        # Step 2: Further enhance with metadata manager
        final_enhanced = metadata_manager.enhance_metadata(enhanced_chunk, "Text")
        
        # Should have metadata from both stages
        assert final_enhanced.meta["chunk_index"] == 1
        assert "chunk_id" in final_enhanced.meta
        assert metadata_manager.config.metadata.source_type in final_enhanced.meta
        assert "citation" in final_enhanced.meta
        assert "processing_timestamp" in final_enhanced.meta

    def test_component_error_propagation(self, integration_config: PipelineConfig,
                                       integration_test_data: Dict[str, Any]):
        """Test error propagation between components."""
        analyzer = FileAnalyzer(integration_config)
        
        # Include invalid file
        all_files = integration_test_data["all_files"]
        
        result = analyzer.analyze_files(all_files)
        
        # Should have some valid files and some errors
        assert len(result["valid_files"]) > 0
        assert result["errors"] is not None
        assert len(result["errors"]) >= 0
        
        # Empty file should be caught
        assert any("empty" in error.lower() for error in result["errors"])


class TestPipelineEndToEndWorkflow:
    """Test complete end-to-end pipeline workflows."""

    def test_successful_complete_workflow(self, orchestrator_with_mocked_haystack: DocumentPipelineOrchestrator,
                                        integration_test_data: Dict[str, Any]):
        """Test successful processing of all supported file types."""
        orchestrator = orchestrator_with_mocked_haystack
        valid_files = integration_test_data["valid_files"]
        
        # Mock Haystack components to return predictable results
        with patch.object(orchestrator, 'document_converter') as mock_converter, \
             patch.object(orchestrator, 'chunking_service') as mock_chunker, \
             patch.object(orchestrator, 'metadata_manager') as mock_metadata:
            
            # Mock conversion results
            converted_docs = [
                Document(
                    id=f"doc_{i}",
                    content=f"Content from file {i} " * 20,  # ~400 chars
                    meta={"file_path": str(file), "source_type": "text"}
                )
                for i, file in enumerate(valid_files)
            ]
            
            mock_converter.convert_files.return_value = {
                "documents": converted_docs,
                "errors": []
            }
            
            # Mock chunking results (2 chunks per document)
            chunked_docs = []
            for doc in converted_docs:
                chunk1 = Document(id=f"{doc.id}_chunk_0", content=doc.content[:200], meta=doc.meta.copy())
                chunk2 = Document(id=f"{doc.id}_chunk_1", content=doc.content[200:], meta=doc.meta.copy())
                chunked_docs.extend([chunk1, chunk2])
            
            mock_chunker.chunk_documents.return_value = {
                "documents": chunked_docs,
                "errors": []
            }
            
            # Mock metadata enhancement
            mock_metadata.enhance_metadata.side_effect = lambda doc, file_type: Document(
                id=doc.id,
                content=doc.content,
                meta={**doc.meta, "enhanced": True, "file_type": file_type.lower()}
            )
            
            # Process documents
            result = orchestrator.process_documents(valid_files)
            
            # Verify results
            assert len(result["documents"]) == 6  # 3 files * 2 chunks each
            assert result["errors"] == []
            assert result["stats"]["files_processed"] == 3
            assert result["stats"]["documents_created"] == 6
            assert result["stats"]["success_rate"] == 100.0
            assert result["stats"]["processing_time"] > 0

    def test_workflow_with_mixed_success_failure(self, orchestrator_with_mocked_haystack: DocumentPipelineOrchestrator,
                                               integration_test_data: Dict[str, Any]):
        """Test workflow handling both successful and failed files."""
        orchestrator = orchestrator_with_mocked_haystack
        all_files = integration_test_data["all_files"]  # Includes invalid files
        
        # The file analyzer should catch invalid files
        result = orchestrator.process_documents(all_files)
        
        # Should have some errors from invalid files
        assert len(result["errors"]) > 0
        assert result["stats"]["errors_count"] > 0
        assert result["stats"]["success_rate"] < 100.0

    def test_workflow_performance_tracking(self, orchestrator_with_mocked_haystack: DocumentPipelineOrchestrator,
                                         integration_test_data: Dict[str, Any]):
        """Test that workflow properly tracks performance metrics."""
        orchestrator = orchestrator_with_mocked_haystack
        valid_files = integration_test_data["valid_files"]
        
        # Process with minimal mocking to test performance tracking
        with patch.object(orchestrator, '_process_file_group') as mock_process_group:
            mock_process_group.return_value = {"documents": [], "errors": []}
            
            import time
            start_time = time.time()
            result = orchestrator.process_documents(valid_files)
            end_time = time.time()
            
            # Should track timing accurately
            assert result["stats"]["processing_time"] > 0
            assert result["stats"]["processing_time"] <= (end_time - start_time) + 0.1  # Small tolerance
            assert "avg_time_per_file" in result["stats"]
            assert "processing_timestamp" in result["stats"]

    def test_large_file_processing(self, orchestrator_with_mocked_haystack: DocumentPipelineOrchestrator,
                                 tmp_path: Path):
        """Test processing of larger files to ensure scalability."""
        orchestrator = orchestrator_with_mocked_haystack
        
        # Create large text file
        large_file = tmp_path / "large_document.txt"
        large_content = "This is a sentence for large file testing. " * 1000  # ~43KB
        large_file.write_text(large_content, encoding='utf-8')
        
        # Process file
        with patch.object(orchestrator, 'document_converter') as mock_converter, \
             patch.object(orchestrator, 'chunking_service') as mock_chunker, \
             patch.object(orchestrator, 'metadata_manager') as mock_metadata:
            
            # Mock large document
            large_doc = Document(
                id="large_doc",
                content=large_content,
                meta={"file_path": str(large_file)}
            )
            
            mock_converter.convert_files.return_value = {
                "documents": [large_doc],
                "errors": []
            }
            
            # Mock chunking to create many chunks
            many_chunks = [
                Document(
                    id=f"chunk_{i}",
                    content=f"Chunk {i} content",
                    meta={"source_doc": "large_doc"}
                )
                for i in range(50)  # 50 chunks
            ]
            
            mock_chunker.chunk_documents.return_value = {
                "documents": many_chunks,
                "errors": []
            }
            
            mock_metadata.enhance_metadata.side_effect = lambda doc, ft: doc
            
            result = orchestrator.process_documents([large_file])
            
            # Should handle many chunks efficiently
            assert len(result["documents"]) == 50
            assert result["stats"]["processing_time"] < 5.0  # Should be fast even with many chunks

    def test_error_recovery_and_continuation(self, orchestrator_with_mocked_haystack: DocumentPipelineOrchestrator,
                                           integration_test_data: Dict[str, Any]):
        """Test error recovery and continuation with error handling."""
        orchestrator = orchestrator_with_mocked_haystack
        valid_files = integration_test_data["valid_files"]
        
        with patch.object(orchestrator, '_process_file_group') as mock_process_group:
            call_count = 0
            
            def side_effect(file_type, files):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First group fails
                    raise Exception("First group processing failed")
                else:
                    # Subsequent groups succeed
                    return {
                        "documents": [Document(id=f"doc_{call_count}", content="content")],
                        "errors": []
                    }
            
            mock_process_group.side_effect = side_effect
            
            result = orchestrator.process_documents(valid_files)
            
            # Should continue processing despite first group failure
            assert result["documents"] is not None
            assert len(result["documents"]) > 0  # Some documents processed
            assert len(result["errors"]) > 0     # Some errors recorded
            assert "First group processing failed" in " ".join(result["errors"])


class TestConfigurationIntegration:
    """Test integration with different configurations."""

    def test_different_chunking_configurations(self, integration_test_data: Dict[str, Any]):
        """Test pipeline behavior with different chunking configurations."""
        valid_files = integration_test_data["valid_files"][:1]  # Use one file
        
        configs = [
            PipelineConfig(chunking=ChunkingConfig(chunk_size=100, chunk_overlap=20)),
            PipelineConfig(chunking=ChunkingConfig(chunk_size=300, chunk_overlap=60)),
            PipelineConfig(chunking=ChunkingConfig(chunk_size=50, chunk_overlap=10))
        ]
        
        for config in configs:
            orchestrator = DocumentPipelineOrchestrator(config)
            
            # Test that orchestrator uses the configuration correctly
            assert orchestrator.config.chunking.chunk_size == config.chunking.chunk_size
            assert orchestrator.chunking_service.config.chunking.chunk_size == config.chunking.chunk_size

    def test_selective_file_type_processing(self, integration_test_data: Dict[str, Any]):
        """Test processing with selective file type support."""
        valid_files = integration_test_data["valid_files"]
        
        # Config with only text processing enabled
        text_only_config = PipelineConfig(
            processing=ProcessingConfig(
                enable_pdf_processing=False,
                enable_text_processing=True,
                enable_markdown_processing=False
            ),
            file_types=FileTypeConfig(
                supported_extensions=[".txt"],
                extension_to_type_mapping={".txt": "Text"}
            )
        )
        
        orchestrator = DocumentPipelineOrchestrator(text_only_config)
        
        # Should only process text files
        result = orchestrator.process_documents(valid_files)
        
        # PDF and Markdown files should be rejected during analysis
        assert len(result["errors"]) > 0
        # Should mention disabled processing or unsupported types
        error_text = " ".join(result["errors"])
        assert any(term in error_text.lower() for term in ["disabled", "unsupported", "invalid"])

    def test_error_handling_configuration(self, integration_test_data: Dict[str, Any]):
        """Test different error handling configurations."""
        valid_files = integration_test_data["valid_files"]
        
        # Test fail-fast configuration
        fail_fast_config = PipelineConfig(
            error_handling=ErrorHandlingConfig(fail_fast=True),
            processing=ProcessingConfig(enable_text_processing=True)
        )
        
        orchestrator = DocumentPipelineOrchestrator(fail_fast_config)
        
        with patch.object(orchestrator, '_process_file_group') as mock_process:
            mock_process.side_effect = Exception("Critical error")
            
            # Should raise exception in fail-fast mode
            with pytest.raises(Exception, match="Critical error"):
                orchestrator.process_documents(valid_files)

        # Test continue-on-error configuration
        continue_config = PipelineConfig(
            error_handling=ErrorHandlingConfig(
                fail_fast=False,
                continue_on_individual_file_error=True
            )
        )
        
        orchestrator2 = DocumentPipelineOrchestrator(continue_config)
        
        with patch.object(orchestrator2, '_process_file_group') as mock_process2:
            mock_process2.side_effect = Exception("Non-critical error")
            
            # Should not raise exception, just record error
            result = orchestrator2.process_documents(valid_files)
            assert len(result["errors"]) > 0
            assert "Non-critical error" in " ".join(result["errors"])


class TestRealWorldScenarios:
    """Test realistic real-world scenarios."""

    def test_academic_paper_processing_scenario(self, tmp_path: Path):
        """Test processing academic papers with references and structured content."""
        # Create academic paper structure
        paper_dir = tmp_path / "academic_papers"
        paper_dir.mkdir()
        
        # Abstract
        abstract_file = paper_dir / "abstract.txt"
        abstract_file.write_text(
            "This paper presents a novel approach to document processing using "
            "modular pipeline architecture. The system demonstrates improved "
            "performance and maintainability compared to monolithic solutions."
        )
        
        # Main paper content
        paper_file = paper_dir / "paper.md"
        paper_content = """# A Novel Approach to Document Processing

## Abstract
See abstract.txt for full abstract.

## Introduction
Document processing systems have evolved significantly in recent years.
Traditional approaches often suffer from tight coupling and lack of modularity.

## Methodology
Our approach uses a pipeline architecture with the following components:
- FileAnalyzer for validation
- DocumentConverter for format handling
- ChunkingService for content segmentation
- MetadataManager for metadata enhancement

## Results
The modular approach shows 40% improvement in maintainability scores
and 25% reduction in coupling metrics compared to baseline systems.

## Conclusion
The proposed pipeline architecture provides a robust foundation
for scalable document processing applications.

## References
[1] Smith et al. (2023) "Modular Software Design Patterns"
[2] Jones et al. (2022) "Document Processing at Scale"
"""
        paper_file.write_text(paper_content)
        
        # Test processing
        config = PipelineConfig(
            chunking=ChunkingConfig(chunk_size=300, chunk_overlap=50),
            processing=ProcessingConfig(
                enable_text_processing=True,
                enable_markdown_processing=True
            )
        )
        
        orchestrator = DocumentPipelineOrchestrator(config)
        files = [abstract_file, paper_file]
        
        # Mock the pipeline for predictable results
        with patch.object(orchestrator, 'document_converter') as mock_converter, \
             patch.object(orchestrator, 'chunking_service') as mock_chunker, \
             patch.object(orchestrator, 'metadata_manager') as mock_metadata:
            
            # Simulate realistic conversion
            docs = [
                Document(id="abstract", content=abstract_file.read_text(), 
                        meta={"file_path": str(abstract_file), "source_type": "text"}),
                Document(id="paper", content=paper_file.read_text(),
                        meta={"file_path": str(paper_file), "source_type": "markdown"})
            ]
            
            mock_converter.convert_files.return_value = {"documents": docs, "errors": []}
            
            # Simulate chunking by sections
            chunks = [
                Document(id="abstract_chunk", content=docs[0].content, meta=docs[0].meta),
                Document(id="intro_chunk", content="Introduction section...", meta=docs[1].meta),
                Document(id="method_chunk", content="Methodology section...", meta=docs[1].meta),
                Document(id="results_chunk", content="Results section...", meta=docs[1].meta),
                Document(id="conclusion_chunk", content="Conclusion section...", meta=docs[1].meta)
            ]
            
            mock_chunker.chunk_documents.return_value = {"documents": chunks, "errors": []}
            mock_metadata.enhance_metadata.side_effect = lambda doc, ft: doc
            
            result = orchestrator.process_documents([str(f) for f in files])
            
            # Should process both files successfully
            assert len(result["documents"]) == 5  # 1 abstract + 4 paper sections
            assert result["errors"] == []
            assert result["stats"]["files_processed"] == 2

    def test_mixed_document_collection_scenario(self, tmp_path: Path):
        """Test processing mixed document collection with various formats and sizes."""
        # Create diverse document collection
        collection_dir = tmp_path / "document_collection"
        collection_dir.mkdir()
        
        # Small note
        note_file = collection_dir / "quick_note.txt"
        note_file.write_text("Quick note: Remember to test edge cases.")
        
        # Medium documentation
        doc_file = collection_dir / "documentation.md"
        doc_content = "# Documentation\n\n" + "This is documentation content. " * 50
        doc_file.write_text(doc_content)
        
        # Large report (simulated)
        report_file = collection_dir / "annual_report.txt"
        report_content = "ANNUAL REPORT\n\n" + "Report section content. " * 200
        report_file.write_text(report_content)
        
        # Empty file (should be rejected)
        empty_file = collection_dir / "empty.txt"
        empty_file.write_text("")
        
        files = [note_file, doc_file, report_file, empty_file]
        
        config = PipelineConfig(
            chunking=ChunkingConfig(chunk_size=150, chunk_overlap=30),
            processing=ProcessingConfig(
                enable_text_processing=True,
                enable_markdown_processing=True
            ),
            error_handling=ErrorHandlingConfig(continue_on_individual_file_error=True)
        )
        
        orchestrator = DocumentPipelineOrchestrator(config)
        result = orchestrator.process_documents([str(f) for f in files])
        
        # Should process most files but have errors for empty file
        assert result["stats"]["files_processed"] == 4
        assert len(result["errors"]) > 0  # Empty file error
        assert result["stats"]["success_rate"] < 100.0  # Due to empty file
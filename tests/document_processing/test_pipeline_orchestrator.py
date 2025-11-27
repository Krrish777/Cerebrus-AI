"""
Unit tests for PipelineOrchestrator component.

Tests high-level pipeline coordination and workflow orchestration
following AGENTS.md principles: integration testing with proper isolation.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from haystack.dataclasses import Document

from src.document_processing.pipeline_orchestrator import DocumentPipelineOrchestrator
from src.document_processing.pipeline_config import (
    PipelineConfig,
    ChunkingConfig,
    ProcessingConfig,
    ErrorHandlingConfig,
    MetadataConfig,
    FileTypeConfig
)


@pytest.fixture
def test_config() -> PipelineConfig:
    """Create test configuration for PipelineOrchestrator."""
    return PipelineConfig(
        chunking=ChunkingConfig(chunk_size=100, chunk_overlap=20),
        processing=ProcessingConfig(
            enable_pdf_processing=True,
            enable_text_processing=True,
            enable_markdown_processing=True
        ),
        error_handling=ErrorHandlingConfig(
            fail_fast=False,
            continue_on_individual_file_error=True
        ),
        metadata=MetadataConfig(),
        file_types=FileTypeConfig()
    )


@pytest.fixture
def orchestrator(test_config: PipelineConfig) -> DocumentPipelineOrchestrator:
    """Create DocumentPipelineOrchestrator instance with test configuration."""
    return DocumentPipelineOrchestrator(test_config)


@pytest.fixture
def sample_files(tmp_path: Path) -> List[Path]:
    """Create sample files for testing."""
    files = []
    
    # Create PDF file
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\nTest PDF content")
    files.append(pdf_file)
    
    # Create text file
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("Test text content", encoding='utf-8')
    files.append(txt_file)
    
    # Create markdown file
    md_file = tmp_path / "document.md"
    md_file.write_text("# Test Markdown\nContent", encoding='utf-8')
    files.append(md_file)
    
    return files


@pytest.fixture
def sample_documents() -> List[Document]:
    """Create sample Haystack documents."""
    return [
        Document(
            id="doc_1",
            content="First document content for testing",
            meta={"file_path": "/test/doc1.txt", "source_type": "text"}
        ),
        Document(
            id="doc_2",
            content="Second document content for testing",
            meta={"file_path": "/test/doc2.pdf", "source_type": "pdf"}
        )
    ]


class TestOrchestratorInitialization:
    """Test DocumentPipelineOrchestrator initialization."""

    def test_orchestrator_initializes_with_config(self, test_config: PipelineConfig):
        """Test orchestrator initializes properly with configuration."""
        orchestrator = DocumentPipelineOrchestrator(test_config)
        
        assert orchestrator.config == test_config
        assert orchestrator._file_analyzer is None  # Lazy loading
        assert orchestrator._document_converter is None
        assert orchestrator._chunking_service is None
        assert orchestrator._metadata_manager is None

    def test_orchestrator_initializes_with_default_config(self):
        """Test orchestrator initializes with default configuration when none provided."""
        orchestrator = DocumentPipelineOrchestrator()
        
        assert orchestrator.config is not None

    def test_lazy_property_loading(self, orchestrator: DocumentPipelineOrchestrator):
        """Test that component properties are lazy-loaded."""
        # Initially None
        assert orchestrator._file_analyzer is None
        
        # Access property
        analyzer = orchestrator.file_analyzer
        
        # Should now be loaded
        assert analyzer is not None
        assert orchestrator._file_analyzer is analyzer
        
        # Subsequent access should return same instance
        assert orchestrator.file_analyzer is analyzer

    def test_all_component_properties_work(self, orchestrator: DocumentPipelineOrchestrator):
        """Test that all component properties load correctly."""
        # Access all properties
        file_analyzer = orchestrator.file_analyzer
        document_converter = orchestrator.document_converter
        chunking_service = orchestrator.chunking_service
        metadata_manager = orchestrator.metadata_manager
        
        # All should be loaded
        assert file_analyzer is not None
        assert document_converter is not None
        assert chunking_service is not None
        assert metadata_manager is not None


class TestFileGrouping:
    """Test file grouping functionality."""

    def test_group_files_by_type(self, orchestrator: DocumentPipelineOrchestrator, sample_files: List[Path]):
        """Test grouping files by detected type."""
        groups = orchestrator._group_files_by_type(sample_files)
        
        # Should have groups for different file types
        assert isinstance(groups, dict)
        assert len(groups) > 0

    def test_group_files_empty_list(self, orchestrator: DocumentPipelineOrchestrator):
        """Test grouping empty file list."""
        groups = orchestrator._group_files_by_type([])
        
        assert groups == {}

    def test_group_files_uses_file_analyzer(self, orchestrator: DocumentPipelineOrchestrator,
                                          sample_files: List[Path]):
        """Test that file grouping uses file analyzer."""
        mock_analyzer = Mock()
        mock_analyzer.detect_file_type.side_effect = lambda path: "Text" if path.suffix == ".txt" else "PDF"
        orchestrator._file_analyzer = mock_analyzer
        
        groups = orchestrator._group_files_by_type(sample_files[:2])  # txt and pdf
        
        # Should call detect_file_type for each file
        assert mock_analyzer.detect_file_type.call_count == 2
class TestFileAnalysis:
    """Test file analysis workflow."""

    def test_analyze_files_successful(self, orchestrator: DocumentPipelineOrchestrator,
                                     sample_files: List[Path]):
        """Test successful file analysis."""
        mock_analyzer = Mock()
        mock_analysis_result = {
            "valid_files": sample_files,
            "errors": [],
            "file_types": {"Text": 1, "PDF": 2},
            "total_size": 1024
        }
        mock_analyzer.analyze_files.return_value = mock_analysis_result
        orchestrator._file_analyzer = mock_analyzer
        
        result = orchestrator._analyze_files(sample_files)
        
        mock_analyzer.analyze_files.assert_called_once_with(sample_files)
        assert result == mock_analysis_result

    def test_analyze_files_exception_handling(self, orchestrator: DocumentPipelineOrchestrator,
                                             sample_files: List[Path]):
        """Test file analysis exception handling."""
        mock_analyzer = Mock()
        mock_analyzer.analyze_files.side_effect = Exception("Analysis error")
        orchestrator._file_analyzer = mock_analyzer
        
        result = orchestrator._analyze_files(sample_files)
        
        assert result["valid_files"] == []
        assert len(result["errors"]) == 1
        assert "Analysis error" in result["errors"][0]


class TestFileGroupProcessing:
    """Test processing of file groups."""

    def test_process_file_group_successful(self, orchestrator: DocumentPipelineOrchestrator,
                                         sample_documents: List[Document]):
        """Test successful file group processing."""
        test_files = [Path("/test/file1.txt"), Path("/test/file2.txt")]
        
        # Mock all components
        with patch.object(orchestrator, '_document_converter') as mock_converter, \
             patch.object(orchestrator, '_chunking_service') as mock_chunker, \
             patch.object(orchestrator, '_metadata_manager') as mock_metadata:
            
            # Setup mocks
            mock_converter.convert_files.return_value = {
                "documents": sample_documents,
                "errors": []
            }
            mock_chunker.chunk_documents.return_value = {
                "documents": sample_documents,
                "errors": []
            }
            mock_metadata.enhance_metadata.side_effect = lambda doc, file_type: doc
            
            result = orchestrator._process_file_group("Text", test_files)
            
            # Verify calls
            mock_converter.convert_files.assert_called_once_with("Text", test_files)
            mock_chunker.chunk_documents.assert_called_once_with(sample_documents)
            assert mock_metadata.enhance_metadata.call_count == len(sample_documents)
            
            # Check results
            assert len(result["documents"]) == len(sample_documents)
            assert result["errors"] == []

    def test_process_file_group_conversion_errors(self, orchestrator: DocumentPipelineOrchestrator):
        """Test handling of conversion errors."""
        test_files = [Path("/test/file.txt")]
        
        with patch.object(orchestrator, '_document_converter') as mock_converter:
            mock_converter.convert_files.return_value = {
                "documents": [],
                "errors": ["Conversion failed"]
            }
            
            result = orchestrator._process_file_group("Text", test_files)
            
            assert result["documents"] == []
            assert "Conversion failed" in result["errors"]

    def test_process_file_group_no_documents_from_conversion(self, orchestrator: DocumentPipelineOrchestrator):
        """Test handling when conversion returns no documents."""
        test_files = [Path("/test/file.txt")]
        
        with patch.object(orchestrator, '_document_converter') as mock_converter:
            mock_converter.convert_files.return_value = {
                "documents": [],
                "errors": []
            }
            
            result = orchestrator._process_file_group("Text", test_files)
            
            assert result["documents"] == []
            assert result["errors"] == []

    def test_process_file_group_exception_handling(self, orchestrator: DocumentPipelineOrchestrator):
        """Test exception handling in file group processing."""
        test_files = [Path("/test/file.txt")]
        
        with patch.object(orchestrator, '_document_converter') as mock_converter:
            mock_converter.convert_files.side_effect = Exception("Processing error")
            
            result = orchestrator._process_file_group("Text", test_files)
            
            assert result["documents"] == []
            assert len(result["errors"]) == 1
            assert "Processing error" in result["errors"][0]


class TestDocumentProcessing:
    """Test main document processing workflow."""

    def test_process_documents_empty_list(self, orchestrator: DocumentPipelineOrchestrator):
        """Test processing empty file list."""
        result = orchestrator.process_documents([])
        
        assert result["documents"] == []
        assert result["errors"] == []
        assert result["stats"]["files_processed"] == 0

    def test_process_documents_successful_workflow(self, orchestrator: DocumentPipelineOrchestrator,
                                                 sample_files: List[Path],
                                                 sample_documents: List[Document]):
        """Test successful end-to-end document processing."""
        with patch.object(orchestrator, '_analyze_files') as mock_analyze, \
             patch.object(orchestrator, '_group_files_by_type') as mock_group, \
             patch.object(orchestrator, '_process_file_group') as mock_process_group:
            
            # Setup mocks
            mock_analyze.return_value = {
                "valid_files": sample_files,
                "errors": [],
                "file_types": {"Text": 2, "PDF": 1},
                "total_size": 1024
            }
            mock_group.return_value = {"Text": sample_files[:2], "PDF": sample_files[2:]}
            mock_process_group.return_value = {
                "documents": sample_documents,
                "errors": []
            }
            
            result = orchestrator.process_documents(sample_files)
            
            # Verify workflow
            mock_analyze.assert_called_once()
            mock_group.assert_called_once_with(sample_files)
            assert mock_process_group.call_count == 2  # Two file type groups
            
            # Check results
            assert len(result["documents"]) == len(sample_documents) * 2  # Two groups
            assert result["errors"] == []
            assert result["stats"]["files_processed"] == len(sample_files)

    def test_process_documents_analysis_errors(self, orchestrator: DocumentPipelineOrchestrator,
                                             sample_files: List[Path]):
        """Test handling of file analysis errors."""
        with patch.object(orchestrator, '_analyze_files') as mock_analyze:
            mock_analyze.return_value = {
                "valid_files": [],
                "errors": ["File not found", "Invalid format"],
                "file_types": {},
                "total_size": 0
            }
            
            result = orchestrator.process_documents(sample_files)
            
            assert result["documents"] == []
            assert len(result["errors"]) == 2
            assert result["stats"]["files_processed"] == 0
            assert result["stats"]["errors_count"] == 2

    def test_process_documents_mixed_success_failure(self, orchestrator: DocumentPipelineOrchestrator,
                                                   sample_files: List[Path],
                                                   sample_documents: List[Document]):
        """Test processing with mixed success and failure."""
        with patch.object(orchestrator, '_analyze_files') as mock_analyze, \
             patch.object(orchestrator, '_group_files_by_type') as mock_group, \
             patch.object(orchestrator, '_process_file_group') as mock_process_group:
            
            mock_analyze.return_value = {
                "valid_files": sample_files[:2],  # Only 2 files valid
                "errors": ["File 3 invalid"],
                "file_types": {"Text": 2},
                "total_size": 512
            }
            mock_group.return_value = {"Text": sample_files[:2]}
            mock_process_group.return_value = {
                "documents": sample_documents,
                "errors": ["Processing warning"]
            }
            
            result = orchestrator.process_documents(sample_files)
            
            assert len(result["documents"]) == len(sample_documents)
            assert len(result["errors"]) == 2  # 1 analysis + 1 processing

    def test_process_documents_fail_fast_mode(self, orchestrator: DocumentPipelineOrchestrator,
                                            sample_files: List[Path]):
        """Test fail-fast error handling mode."""
        # Update config for fail-fast
        orchestrator.config.error_handling.fail_fast = True
        
        with patch.object(orchestrator, '_analyze_files') as mock_analyze, \
             patch.object(orchestrator, '_group_files_by_type') as mock_group, \
             patch.object(orchestrator, '_process_file_group') as mock_process_group:
            
            mock_analyze.return_value = {
                "valid_files": sample_files,
                "errors": [],
                "file_types": {"Text": 1},
                "total_size": 1024
            }
            mock_group.return_value = {"Text": sample_files}
            mock_process_group.side_effect = Exception("Critical error")
            
            with pytest.raises(Exception, match="Critical error"):
                orchestrator.process_documents(sample_files)


class TestStatisticsCalculation:
    """Test statistics calculation."""

    def test_calculate_statistics_basic(self, orchestrator: DocumentPipelineOrchestrator):
        """Test basic statistics calculation."""
        stats = orchestrator._calculate_statistics(
            total_files=5,
            total_documents=10,
            processing_time=2.5,
            error_count=1
        )
        
        assert stats["files_processed"] == 5
        assert stats["documents_created"] == 10
        assert stats["processing_time"] == 2.5
        assert stats["avg_time_per_file"] == 0.5  # 2.5 / 5
        assert stats["errors_count"] == 1
        assert stats["success_rate"] == 80.0  # (5-1)/5 * 100
        assert stats["documents_per_file"] == 2.0  # 10 / 5
        assert "processing_timestamp" in stats

    def test_calculate_statistics_zero_files(self, orchestrator: DocumentPipelineOrchestrator):
        """Test statistics calculation with zero files."""
        stats = orchestrator._calculate_statistics(
            total_files=0,
            total_documents=0,
            processing_time=0.0,
            error_count=0
        )
        
        assert stats["avg_time_per_file"] == 0.0
        assert stats["success_rate"] == 0.0
        assert stats["documents_per_file"] == 0.0

    def test_calculate_statistics_timestamp_format(self, orchestrator: DocumentPipelineOrchestrator):
        """Test that statistics timestamp is in ISO format."""
        stats = orchestrator._calculate_statistics(1, 1, 1.0, 0)
        
        timestamp = stats["processing_timestamp"]
        # Should be able to parse as ISO datetime
        from datetime import datetime
        parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00') if timestamp.endswith('Z') else timestamp)
        assert isinstance(parsed, datetime)


class TestPipelineInformation:
    """Test pipeline information retrieval."""

    def test_get_pipeline_info(self, orchestrator: DocumentPipelineOrchestrator):
        """Test getting pipeline information."""
        info = orchestrator.get_pipeline_info()
        
        assert "orchestrator" in info
        assert "configuration" in info
        assert "error_handling" in info
        
        # Check orchestrator info
        orch_info = info["orchestrator"]
        assert orch_info["version"] == "1.0"
        assert "components" in orch_info
        assert len(orch_info["components"]) == 4
        
        # Check configuration info
        config_info = info["configuration"]
        assert "supported_types" in config_info
        assert "chunking" in config_info
        assert "processing_options" in config_info

    def test_pipeline_info_reflects_configuration(self, orchestrator: DocumentPipelineOrchestrator):
        """Test that pipeline info reflects actual configuration."""
        info = orchestrator.get_pipeline_info()
        
        config = info["configuration"]
        
        # Should match orchestrator config
        assert config["chunking"]["chunk_size"] == orchestrator.config.chunking.chunk_size
        assert config["processing_options"]["pdf_enabled"] == orchestrator.config.processing.enable_pdf_processing


class TestOrchestratorIntegration:
    """Integration tests for DocumentPipelineOrchestrator."""

    def test_end_to_end_with_real_components(self, tmp_path: Path):
        """Test orchestrator with real components (no mocking)."""
        # Create real test files
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content for integration testing.", encoding='utf-8')
        
        # Create orchestrator with realistic config
        config = PipelineConfig(
            processing=ProcessingConfig(
                enable_text_processing=True,
                enable_pdf_processing=False,
                enable_markdown_processing=False
            ),
            chunking=ChunkingConfig(chunk_size=50, chunk_overlap=10),
            error_handling=ErrorHandlingConfig(fail_fast=False)
        )
        
        orchestrator = DocumentPipelineOrchestrator(config)
        
        # Test that components are created correctly
        assert orchestrator.file_analyzer is not None
        assert orchestrator.document_converter is not None
        assert orchestrator.chunking_service is not None
        assert orchestrator.metadata_manager is not None

    def test_orchestrator_performance_tracking(self, orchestrator: DocumentPipelineOrchestrator,
                                             sample_files: List[Path]):
        """Test that orchestrator tracks performance metrics."""
        # Mock successful processing
        with patch.object(orchestrator, '_analyze_files') as mock_analyze, \
             patch.object(orchestrator, '_process_file_group') as mock_process:
            
            mock_analyze.return_value = {
                "valid_files": sample_files,
                "errors": [],
                "file_types": {"Text": len(sample_files)},
                "total_size": 1024
            }
            mock_process.return_value = {"documents": [], "errors": []}
            
            start_time = time.time()
            result = orchestrator.process_documents(sample_files)
            end_time = time.time()
            
            # Should complete quickly and track timing
            assert result["stats"]["processing_time"] > 0
            assert result["stats"]["processing_time"] < (end_time - start_time) + 0.1  # Small tolerance

    def test_orchestrator_respects_configuration_changes(self):
        """Test that orchestrator respects different configurations."""
        config1 = PipelineConfig(
            processing=ProcessingConfig(enable_pdf_processing=True),
            error_handling=ErrorHandlingConfig(fail_fast=True)
        )
        config2 = PipelineConfig(
            processing=ProcessingConfig(enable_pdf_processing=False),
            error_handling=ErrorHandlingConfig(fail_fast=False)
        )
        
        orch1 = DocumentPipelineOrchestrator(config1)
        orch2 = DocumentPipelineOrchestrator(config2)
        
        assert orch1.config.processing.enable_pdf_processing is True
        assert orch2.config.processing.enable_pdf_processing is False
        assert orch1.config.error_handling.fail_fast is True
        assert orch2.config.error_handling.fail_fast is False

    def test_error_propagation_and_isolation(self, orchestrator: DocumentPipelineOrchestrator,
                                           sample_files: List[Path]):
        """Test error propagation and isolation between file groups."""
        with patch.object(orchestrator, '_analyze_files') as mock_analyze, \
             patch.object(orchestrator, '_group_files_by_type') as mock_group, \
             patch.object(orchestrator, '_process_file_group') as mock_process_group:
            
            mock_analyze.return_value = {
                "valid_files": sample_files,
                "errors": [],
                "file_types": {"Text": 2, "PDF": 1},
                "total_size": 1024
            }
            mock_group.return_value = {"Text": sample_files[:2], "PDF": sample_files[2:]}
            
            # First group succeeds, second fails
            def side_effect(file_type, files):
                if file_type == "Text":
                    return {"documents": [Document(id="1", content="content")], "errors": []}
                else:
                    raise Exception("PDF processing failed")
            
            mock_process_group.side_effect = side_effect
            
            result = orchestrator.process_documents(sample_files)
            
            # Should have documents from successful group and errors from failed group
            assert len(result["documents"]) == 1
            assert len(result["errors"]) == 1
            assert "PDF processing failed" in result["errors"][0]
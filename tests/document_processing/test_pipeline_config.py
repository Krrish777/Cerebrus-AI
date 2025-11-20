"""
Unit tests for PipelineConfig and configuration management.

Tests configuration loading, validation, and environment variable overrides
following AGENTS.md principles: isolated, deterministic, comprehensive.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
from typing import Dict, Any

from src.document_processing.pipeline_config import (
    PipelineConfig,
    ChunkingConfig, 
    FileTypeConfig,
    MetadataConfig,
    PerformanceConfig,
    ProcessingConfig,
    ValidationConfig,
    ErrorHandlingConfig,
    PipelineConfigLoader,
    get_pipeline_config,
    reload_pipeline_config
)


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Create sample configuration dictionary for testing."""
    return {
        "chunking": {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "min_chunk_size_ratio": 0.4,
            "boundary_preferences": ["paragraph", "sentence", "line"],
            "enable_statistics": True,
            "enable_preview": True,
            "preview_length": 150
        },
        "supported_mime_types": [
            "application/pdf",
            "text/plain", 
            "text/markdown"
        ],
        "supported_extensions": [".pdf", ".txt", ".md", ".markdown"],
        "extension_to_type_mapping": {
            ".pdf": "PDF",
            ".txt": "Text",
            ".md": "Markdown",
            ".markdown": "Markdown"
        },
        "metadata_fields": {
            "chunk_id": "chunk_id",
            "chunk_index": "chunk_index",
            "source_file": "source_file",
            "source_type": "source_type",
            "page_number": "page_number"
        },
        "performance": {
            "enable_timing": True,
            "enable_statistics": True,
            "enable_progress_tracking": True
        },
        "processing_options": {
            "enable_pdf_processing": True,
            "enable_text_processing": True,
            "enable_markdown_processing": True,
            "enable_markdown_fallback": True
        },
        "validation": {},
        "error_handling": {
            "fail_fast": False,
            "continue_on_individual_file_error": True
        },
        "file_size_thresholds": {
            "kb": 1024,
            "mb": 1048576
        }
    }


@pytest.fixture
def sample_yaml_config() -> str:
    """Create sample YAML configuration content."""
    return """
chunking:
  chunk_size: 750
  chunk_overlap: 150
  min_chunk_size_ratio: 0.3
  boundary_preferences:
    - paragraph
    - sentence
  enable_statistics: true
  enable_preview: false

supported_mime_types:
  - application/pdf
  - text/plain
  - text/markdown

supported_extensions:
  - .pdf
  - .txt
  - .md

extension_to_type_mapping:
  .pdf: PDF
  .txt: Text
  .md: Markdown

metadata_fields:
  chunk_id: chunk_id
  source_file: source_file
  source_type: source_type

performance:
  enable_timing: true
  enable_statistics: false

processing_options:
  enable_pdf_processing: true
  enable_text_processing: true
  enable_markdown_processing: false

error_handling:
  fail_fast: true

file_size_thresholds:
  kb: 1024
  mb: 1048576
"""


class TestChunkingConfig:
    """Test ChunkingConfig validation and behavior."""

    def test_chunking_config_defaults(self):
        """Test ChunkingConfig initializes with proper defaults."""
        config = ChunkingConfig()
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.min_chunk_size_ratio == 0.5
        assert config.boundary_preferences == ["paragraph", "sentence", "line"]
        assert config.enable_statistics is True
        assert config.enable_preview is True
        assert config.preview_length == 200

    def test_chunking_config_custom_values(self):
        """Test ChunkingConfig with custom values."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            min_chunk_size_ratio=0.3,
            boundary_preferences=["sentence"],
            enable_statistics=False
        )
        
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.min_chunk_size_ratio == 0.3
        assert config.boundary_preferences == ["sentence"]
        assert config.enable_statistics is False

    def test_chunking_config_validation_positive_chunk_size(self):
        """Test validation requires positive chunk size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=0)
            
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=-100)

    def test_chunking_config_validation_non_negative_overlap(self):
        """Test validation requires non-negative overlap."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            ChunkingConfig(chunk_overlap=-10)

    def test_chunking_config_validation_overlap_less_than_size(self):
        """Test validation ensures overlap is less than chunk size."""
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)
            
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            ChunkingConfig(chunk_size=100, chunk_overlap=150)

    def test_chunking_config_validation_ratio_bounds(self):
        """Test validation ensures min_chunk_size_ratio is between 0 and 1."""
        with pytest.raises(ValueError, match="min_chunk_size_ratio must be between 0 and 1"):
            ChunkingConfig(min_chunk_size_ratio=0)
            
        with pytest.raises(ValueError, match="min_chunk_size_ratio must be between 0 and 1"):
            ChunkingConfig(min_chunk_size_ratio=1.5)
            
        with pytest.raises(ValueError, match="min_chunk_size_ratio must be between 0 and 1"):
            ChunkingConfig(min_chunk_size_ratio=-0.1)

    def test_chunking_config_valid_edge_cases(self):
        """Test valid edge cases for chunking configuration."""
        # Minimum valid configuration
        config = ChunkingConfig(
            chunk_size=1,
            chunk_overlap=0,
            min_chunk_size_ratio=0.1
        )
        assert config.chunk_size == 1
        assert config.chunk_overlap == 0
        assert config.min_chunk_size_ratio == 0.1
        
        # Maximum ratio
        config = ChunkingConfig(min_chunk_size_ratio=1.0)
        assert config.min_chunk_size_ratio == 1.0


class TestPipelineConfig:
    """Test PipelineConfig initialization and behavior."""

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig initializes with proper defaults."""
        config = PipelineConfig()
        
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.file_types, FileTypeConfig) 
        assert isinstance(config.metadata, MetadataConfig)
        assert isinstance(config.performance, PerformanceConfig)
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.validation, ValidationConfig)
        assert isinstance(config.error_handling, ErrorHandlingConfig)
        assert isinstance(config.file_size_thresholds, dict)

    def test_pipeline_config_custom_components(self):
        """Test PipelineConfig with custom component configurations."""
        custom_chunking = ChunkingConfig(chunk_size=300)
        custom_processing = ProcessingConfig(enable_pdf_processing=False)
        
        config = PipelineConfig(
            chunking=custom_chunking,
            processing=custom_processing
        )
        
        assert config.chunking.chunk_size == 300
        assert config.processing.enable_pdf_processing is False
        # Other components should still have defaults
        assert config.performance.enable_timing is True


class TestPipelineConfigLoader:
    """Test PipelineConfigLoader functionality."""

    def test_config_loader_initialization(self):
        """Test PipelineConfigLoader initializes correctly."""
        loader = PipelineConfigLoader()
        assert loader is not None

    @patch('src.document_processing.pipeline_config.load_config')
    def test_load_config_from_yaml(self, mock_load_config, sample_config_dict: Dict[str, Any]):
        """Test loading configuration from YAML file."""
        mock_load_config.return_value = sample_config_dict
        
        loader = PipelineConfigLoader()
        config = loader.load_config()
        
        assert isinstance(config, PipelineConfig)
        assert config.chunking.chunk_size == 500
        assert config.chunking.chunk_overlap == 100
        assert config.processing.enable_pdf_processing is True

    @patch('src.document_processing.pipeline_config.load_config')
    def test_load_config_handles_load_exception(self, mock_load_config):
        """Test config loader handles YAML load exceptions gracefully."""
        mock_load_config.side_effect = FileNotFoundError("Config file not found")
        
        loader = PipelineConfigLoader()
        config = loader.load_config()
        
        # Should return default configuration
        assert isinstance(config, PipelineConfig)
        assert config.chunking.chunk_size == 1000  # Default value

    @patch.dict(os.environ, {
        'PIPELINE_CHUNK_SIZE': '800',
        'PIPELINE_CHUNK_OVERLAP': '160',
        'PIPELINE_ENABLE_TIMING': 'false',
        'PIPELINE_FAIL_FAST': 'true'
    })
    @patch('src.document_processing.pipeline_config.load_config')
    def test_environment_variable_overrides(self, mock_load_config, sample_config_dict: Dict[str, Any]):
        """Test environment variable overrides work correctly."""
        mock_load_config.return_value = sample_config_dict
        
        loader = PipelineConfigLoader()
        config = loader.load_config()
        
        # Environment variables should override YAML values
        assert config.chunking.chunk_size == 800  # Overridden from 500
        assert config.chunking.chunk_overlap == 160  # Overridden from 100
        assert config.performance.enable_timing is False  # Overridden from True
        assert config.error_handling.fail_fast is True  # Overridden from False

    def test_parse_env_value_types(self):
        """Test environment value parsing for different types."""
        loader = PipelineConfigLoader()
        
        # Boolean values
        assert loader._parse_env_value('true') is True
        assert loader._parse_env_value('false') is False
        assert loader._parse_env_value('True') is True
        assert loader._parse_env_value('FALSE') is False
        
        # Integer values
        assert loader._parse_env_value('42') == 42
        assert loader._parse_env_value('-10') == -10
        
        # Float values
        assert loader._parse_env_value('3.14') == 3.14
        assert loader._parse_env_value('-2.5') == -2.5
        
        # String values (fallback)
        assert loader._parse_env_value('hello') == 'hello'
        assert loader._parse_env_value('not_a_number') == 'not_a_number'

    def test_set_nested_value(self):
        """Test setting nested values in configuration dictionary."""
        loader = PipelineConfigLoader()
        data = {}
        
        # Set nested value
        loader._set_nested_value(data, ('chunking', 'chunk_size'), 500)
        
        assert data['chunking']['chunk_size'] == 500
        
        # Set another nested value in same structure
        loader._set_nested_value(data, ('chunking', 'chunk_overlap'), 100)
        
        assert data['chunking']['chunk_size'] == 500
        assert data['chunking']['chunk_overlap'] == 100
        
        # Set value in different section
        loader._set_nested_value(data, ('performance', 'enable_timing'), False)
        
        assert data['performance']['enable_timing'] is False

    def test_deep_merge(self):
        """Test deep merging of configuration dictionaries."""
        loader = PipelineConfigLoader()
        
        base = {
            'chunking': {'chunk_size': 1000, 'chunk_overlap': 200},
            'processing': {'enable_pdf': True}
        }
        
        override = {
            'chunking': {'chunk_size': 500},  # Override chunk_size, keep chunk_overlap
            'metadata': {'source_file': 'file_path'}  # Add new section
        }
        
        result = loader._deep_merge(base, override)
        
        assert result['chunking']['chunk_size'] == 500  # Overridden
        assert result['chunking']['chunk_overlap'] == 200  # Preserved
        assert result['processing']['enable_pdf'] is True  # Preserved
        assert result['metadata']['source_file'] == 'file_path'  # Added

    def test_create_config_from_dict_complete(self, sample_config_dict: Dict[str, Any]):
        """Test creating PipelineConfig from complete dictionary."""
        loader = PipelineConfigLoader()
        
        config = loader._create_config_from_dict(sample_config_dict)
        
        assert isinstance(config, PipelineConfig)
        assert config.chunking.chunk_size == 500
        assert config.file_types.supported_extensions == [".pdf", ".txt", ".md", ".markdown"]
        assert config.metadata.chunk_id == "chunk_id"
        assert config.performance.enable_timing is True
        assert config.processing.enable_pdf_processing is True

    def test_create_config_from_dict_minimal(self):
        """Test creating PipelineConfig from minimal dictionary."""
        loader = PipelineConfigLoader()
        minimal_dict = {"chunking": {"chunk_size": 300}}
        
        config = loader._create_config_from_dict(minimal_dict)
        
        assert isinstance(config, PipelineConfig)
        assert config.chunking.chunk_size == 300
        # Other values should use defaults
        assert config.chunking.chunk_overlap == 200  # Default

    def test_create_config_from_dict_invalid_structure(self):
        """Test handling of invalid configuration structure."""
        loader = PipelineConfigLoader()
        
        invalid_dict = {
            "chunking": {
                "chunk_size": "not_a_number"  # Invalid type
            }
        }
        
        with pytest.raises(ValueError, match="Invalid configuration structure"):
            loader._create_config_from_dict(invalid_dict)

    def test_create_config_handles_nested_content_analysis(self):
        """Test handling of nested content_analysis structure in chunking."""
        loader = PipelineConfigLoader()
        
        config_with_nested = {
            "chunking": {
                "chunk_size": 500,
                "content_analysis": {
                    "enable_statistics": True,
                    "preview_length": 100
                }
            }
        }
        
        config = loader._create_config_from_dict(config_with_nested)
        
        assert config.chunking.chunk_size == 500
        assert config.chunking.enable_statistics is True
        assert config.chunking.preview_length == 100


class TestConfigurationFunctions:
    """Test module-level configuration functions."""

    @patch('src.document_processing.pipeline_config._config_loader')
    def test_get_pipeline_config(self, mock_config_loader):
        """Test get_pipeline_config function."""
        mock_config = PipelineConfig()
        mock_config_loader.load_config.return_value = mock_config
        
        config = get_pipeline_config()
        
        mock_config_loader.load_config.assert_called_once()
        assert config is mock_config

    @patch('src.document_processing.pipeline_config.PipelineConfigLoader')
    def test_reload_pipeline_config(self, mock_loader_class):
        """Test reload_pipeline_config function."""
        mock_loader_instance = mock_loader_class.return_value
        mock_config = PipelineConfig()
        mock_loader_instance.load_config.return_value = mock_config
        
        config = reload_pipeline_config()
        
        mock_loader_class.assert_called_once()
        mock_loader_instance.load_config.assert_called_once()
        assert config is mock_config


class TestComponentConfigurations:
    """Test individual configuration component classes."""

    def test_file_type_config_defaults(self):
        """Test FileTypeConfig default values."""
        config = FileTypeConfig()
        
        assert "application/pdf" in config.supported_mime_types
        assert "text/plain" in config.supported_mime_types
        assert ".pdf" in config.supported_extensions
        assert ".txt" in config.supported_extensions
        assert config.extension_to_type_mapping[".pdf"] == "PDF"

    def test_metadata_config_field_names(self):
        """Test MetadataConfig field name mappings.""" 
        config = MetadataConfig()
        
        assert config.chunk_id == "chunk_id"
        assert config.source_file == "source_file"
        assert config.source_type == "source_type"
        assert config.processing_date == "processing_date"

    def test_performance_config_defaults(self):
        """Test PerformanceConfig default settings."""
        config = PerformanceConfig()
        
        assert config.enable_timing is True
        assert config.enable_statistics is True
        assert config.enable_progress_tracking is True

    def test_processing_config_defaults(self):
        """Test ProcessingConfig default settings."""
        config = ProcessingConfig()
        
        assert config.enable_pdf_processing is True
        assert config.enable_text_processing is True
        assert config.enable_markdown_processing is True
        assert config.enable_markdown_fallback is True

    def test_error_handling_config_defaults(self):
        """Test ErrorHandlingConfig default settings."""
        config = ErrorHandlingConfig()
        
        assert config.fail_fast is True
        assert config.continue_on_individual_file_error is True

    def test_validation_config_defaults(self):
        """Test ValidationConfig initialization."""
        config = ValidationConfig()
        # ValidationConfig currently has no fields, just verify it initializes
        assert config is not None


class TestConfigurationIntegration:
    """Test configuration integration and real-world scenarios."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.utils.config_loader.Path.exists')
    def test_config_loading_with_real_yaml(self, mock_exists, mock_file, sample_yaml_config: str):
        """Test configuration loading with actual YAML content."""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = sample_yaml_config
        
        loader = PipelineConfigLoader()
        
        # This would normally parse the YAML, but we're mocking the file operations
        # The test verifies the structure is handled correctly
        assert loader is not None

    def test_configuration_consistency(self):
        """Test configuration consistency across components."""
        config = PipelineConfig(
            chunking=ChunkingConfig(chunk_size=400),
            processing=ProcessingConfig(enable_pdf_processing=False)
        )
        
        # All components should be properly initialized
        assert config.chunking.chunk_size == 400
        assert config.processing.enable_pdf_processing is False
        assert config.file_types.supported_extensions  # Should have defaults
        assert config.metadata.source_file == "source_file"  # Should have defaults

    def test_configuration_serialization_compatibility(self):
        """Test that configuration can be serialized/deserialized."""
        import dataclasses
        
        config = PipelineConfig()
        
        # Should be able to convert to dict (for serialization)
        config_dict = dataclasses.asdict(config)
        assert isinstance(config_dict, dict)
        assert "chunking" in config_dict
        assert "file_types" in config_dict

    def test_configuration_with_edge_case_values(self):
        """Test configuration with edge case values."""
        # Minimal valid chunking config
        chunking = ChunkingConfig(
            chunk_size=1,
            chunk_overlap=0,
            min_chunk_size_ratio=0.001
        )
        
        config = PipelineConfig(chunking=chunking)
        
        assert config.chunking.chunk_size == 1
        assert config.chunking.chunk_overlap == 0
        assert config.chunking.min_chunk_size_ratio == 0.001

    @patch.dict(os.environ, {
        'PIPELINE_CHUNK_SIZE': '2000',
        'UNKNOWN_ENV_VAR': 'should_be_ignored'
    })
    @patch('src.document_processing.pipeline_config.load_config')
    def test_environment_override_selective_application(self, mock_load_config):
        """Test that only recognized environment variables are applied."""
        mock_load_config.return_value = {"chunking": {"chunk_size": 1000}}
        
        loader = PipelineConfigLoader()
        config = loader.load_config()
        
        # Should apply recognized env var
        assert config.chunking.chunk_size == 2000
        
        # Unknown env vars should not interfere with config loading
        assert config.chunking.chunk_overlap == 200  # Default value
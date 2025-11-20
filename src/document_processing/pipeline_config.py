"""
Pipeline Configuration Management

Loads and validates configuration for the document processing pipeline from YAML files
with environment variable overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import os

from src.core.logging import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for document chunking using RecursiveDocumentSplitter."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size_ratio: float = 0.5
    boundary_preferences: List[str] = field(default_factory=lambda: ["paragraph", "sentence", "line"])
    enable_statistics: bool = True
    enable_preview: bool = True
    preview_length: int = 200

    def __post_init__(self):
        """Validate chunking configuration."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")
        if not 0 < self.min_chunk_size_ratio <= 1:
            raise ValueError(f"min_chunk_size_ratio must be between 0 and 1, got {self.min_chunk_size_ratio}")


@dataclass
class FileTypeConfig:
    """Configuration for supported file types and routing."""

    supported_mime_types: List[str] = field(default_factory=lambda: [
        "application/pdf", "text/plain", "text/markdown"
    ])
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".pdf", ".txt", ".text", ".md", ".markdown"
    ])
    extension_to_type_mapping: Dict[str, str] = field(default_factory=lambda: {
        ".pdf": "PDF", ".txt": "Text", ".text": "Text",
        ".md": "Markdown", ".markdown": "Markdown"
    })


@dataclass
class MetadataConfig:
    """Configuration for metadata field names and handling."""

    chunk_id: str = "chunk_id"
    chunk_index: str = "chunk_index"
    source_file: str = "source_file"
    source_type: str = "source_type"
    page_number: str = "page_number"
    content_hash: str = "content_hash"
    start_char: str = "start_char"
    end_char: str = "end_char"
    word_count: str = "word_count"
    line_count: str = "line_count"
    boundary_found: str = "boundary_found"
    boundary_type: str = "boundary_type"
    processing_date: str = "processing_date"


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring and optimization."""

    enable_timing: bool = True
    enable_statistics: bool = True
    enable_progress_tracking: bool = True


@dataclass
class ProcessingConfig:
    """Configuration for document processing options."""

    enable_pdf_processing: bool = True
    enable_text_processing: bool = True
    enable_markdown_processing: bool = True
    enable_markdown_fallback: bool = True


@dataclass
class ValidationConfig:
    """Configuration for input/output validation."""

    validate_inputs: bool = True
    validate_outputs: bool = True
    check_file_existence: bool = True
    validate_metadata: bool = True


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling behavior."""

    fail_fast: bool = True
    continue_on_individual_file_error: bool = True


@dataclass
class PipelineConfig:
    """Main configuration container for the document processing pipeline."""

    # Core components
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    file_types: FileTypeConfig = field(default_factory=FileTypeConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)

    # File size thresholds
    file_size_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "kb": 1024, "mb": 1048576
    })


class PipelineConfigLoader:
    """Loads and validates pipeline configuration from YAML with environment overrides."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration loader.

        :param config_path: Path to the YAML configuration file. If None, uses default path.
        """
        if config_path is None:
            config_path = Path("src/config/document_processing.yaml")

        self.config_path = config_path
        self._config: Optional[PipelineConfig] = None

    def load_config(self) -> PipelineConfig:
        """
        Load configuration from YAML file with environment variable overrides.

        :return: Validated PipelineConfig instance
        :raises FileNotFoundError: if config file doesn't exist
        :raises ValueError: if configuration is invalid
        """
        if self._config is not None:
            return self._config

        logger.info("Loading pipeline configuration from %s", self.config_path)

        try:
            # Use the existing config loader utility
            yaml_data = load_config(str(self.config_path))
        except FileNotFoundError as e:
            logger.warning("Config file not found at %s, using defaults", self.config_path)
            self._config = PipelineConfig()
            return self._config
        except Exception as e:
            logger.warning("Failed to load config from %s (%s), using defaults", self.config_path, e)
            self._config = PipelineConfig()
            return self._config

        try:
            # Apply environment variable overrides
            yaml_data = self._apply_environment_overrides(yaml_data)

            # Create configuration objects
            config = self._create_config_from_dict(yaml_data)

            logger.info("Pipeline configuration loaded successfully")
            self._config = config
            return config

        except Exception as e:
            logger.error("Failed to create configuration from dictionary: %s", e)
            raise ValueError(f"Invalid configuration structure: {e}") from e

    def _apply_environment_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration dictionary.

        :param config_dict: Configuration dictionary from YAML
        :return: Configuration dictionary with environment overrides applied
        """
        overrides = {}

        # Environment variable mapping
        env_mappings = {
            'PIPELINE_CHUNK_SIZE': ('chunking', 'chunk_size'),
            'PIPELINE_CHUNK_OVERLAP': ('chunking', 'chunk_overlap'),
            'PIPELINE_ENABLE_TIMING': ('performance', 'enable_timing'),
            'PIPELINE_FAIL_FAST': ('error_handling', 'fail_fast'),
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                logger.info("Applying environment override: %s = %s", env_var, env_value)
                self._set_nested_value(overrides, config_path, self._parse_env_value(env_value))

        # Deep merge overrides into config
        return self._deep_merge(config_dict, overrides)

    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value to appropriate type.

        :param value: String value from environment
        :return: Parsed value (int, float, bool, or str)
        """
        # Try boolean first
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _set_nested_value(self, data: Dict[str, Any], path: tuple, value: Any) -> None:
        """
        Set a nested value in a dictionary using a path tuple.

        :param data: Dictionary to modify
        :param path: Tuple representing the nested path
        :param value: Value to set
        """
        current = data
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge override dictionary into base dictionary.

        :param base: Base configuration dictionary
        :param override: Override configuration dictionary
        :return: Merged configuration dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _create_config_from_dict(self, config_dict: Dict[str, Any]) -> PipelineConfig:
        """
        Create PipelineConfig instance from configuration dictionary.

        :param config_dict: Configuration dictionary
        :return: Validated PipelineConfig instance
        :raises ValueError: if configuration is invalid
        """
        try:
            # Extract nested configurations
            chunking_dict = config_dict.get('chunking', {}).copy()
            # Handle nested content_analysis structure
            content_analysis = chunking_dict.pop('content_analysis', {})
            chunking_dict.update(content_analysis)

            # Get processing options for ProcessingConfig (not chunking)
            processing_options = config_dict.get('processing_options', {})

            return PipelineConfig(
                chunking=ChunkingConfig(**chunking_dict),
                file_types=FileTypeConfig(
                    supported_mime_types=config_dict.get('supported_mime_types', []),
                    supported_extensions=config_dict.get('supported_extensions', []),
                    extension_to_type_mapping=config_dict.get('extension_to_type_mapping', {})
                ),
                metadata=MetadataConfig(**config_dict.get('metadata_fields', {})),
                performance=PerformanceConfig(**config_dict.get('performance', {})),
                processing=ProcessingConfig(**processing_options),
                validation=ValidationConfig(**config_dict.get('validation', {})),
                error_handling=ErrorHandlingConfig(**config_dict.get('error_handling', {})),
                file_size_thresholds=config_dict.get('file_size_thresholds', {})
            )
        except Exception as e:
            logger.error("Failed to create configuration from dictionary: %s", e)
            raise ValueError(f"Invalid configuration structure: {e}") from e


# Global configuration loader instance
_config_loader = PipelineConfigLoader()


def get_pipeline_config() -> PipelineConfig:
    """
    Get the global pipeline configuration instance.

    :return: Loaded and validated PipelineConfig
    """
    return _config_loader.load_config()


def reload_pipeline_config() -> PipelineConfig:
    """
    Reload configuration from disk (useful for testing or dynamic config updates).

    :return: Freshly loaded PipelineConfig
    """
    global _config_loader
    _config_loader = PipelineConfigLoader()
    return _config_loader.load_config()
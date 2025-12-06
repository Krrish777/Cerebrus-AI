"""
Configuration module for embedding generation.

This module provides configuration classes for embedding settings.
All settings are loaded from config/embeddings.yml.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """
    Model-specific configuration settings.

    Attributes:
        name: Name or path of the Sentence Transformer model.
        device: Device to use for computation (cpu, cuda, mps, etc.).
        normalize_embeddings: Whether to normalize embedding vectors.
        prefix: String to prepend to text before embedding.
    """

    name: str = "BAAI/bge-small-en-v1.5"
    device: Optional[str] = None
    normalize_embeddings: bool = True
    prefix: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate model configuration."""
        if not self.name or not self.name.strip():
            raise ValueError("Model name cannot be empty")
        logger.debug("ModelConfig initialized: model=%s, device=%s", self.name, self.device or "auto")


@dataclass(frozen=True)
class ProcessingConfig:
    """
    Processing and batch configuration settings.

    Attributes:
        batch_size: Number of documents to process in each batch.
        max_retries: Maximum number of retry attempts on failure.
        timeout: Timeout for processing operations in seconds.
    """

    batch_size: int = 32
    max_retries: int = 3
    timeout: int = 300

    def __post_init__(self) -> None:
        """Validate processing configuration."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        logger.debug(
            "ProcessingConfig initialized: batch_size=%d, max_retries=%d, timeout=%d",
            self.batch_size,
            self.max_retries,
            self.timeout,
        )


@dataclass(frozen=True)
class MetadataConfig:
    """
    Metadata handling configuration.

    Attributes:
        fields_to_embed: List of metadata field names to include in embeddings.
        include_in_embedding: Whether to include metadata in the embedding.
    """

    fields_to_embed: List[str] = field(default_factory=list)
    include_in_embedding: bool = False

    def __post_init__(self) -> None:
        """Validate metadata configuration."""
        if not isinstance(self.fields_to_embed, list):
            raise TypeError("fields_to_embed must be a list")
        logger.debug(
            "MetadataConfig initialized: fields_to_embed=%s, include_in_embedding=%s",
            self.fields_to_embed,
            self.include_in_embedding,
        )


@dataclass(frozen=True)
class LoggingConfigEmbedding:
    """
    Logging configuration for embedding operations.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_embeddings: Whether to log embedding vectors (can be verbose).
        log_model_info: Whether to log model information at startup.
    """

    level: str = "INFO"
    log_embeddings: bool = False
    log_model_info: bool = True

    def __post_init__(self) -> None:
        """Validate logging configuration."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}. Must be one of {valid_levels}")


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Main configuration class for embedding generation.

    This class aggregates all embedding-related configuration settings
    and provides methods to load configuration from YAML files.

    Attributes:
        provider: Name of the embedding provider (haystack, openai, cohere, etc.).
        model: Model configuration.
        processing: Processing and batch configuration.
        metadata: Metadata handling configuration.
        logging: Logging configuration.
    """

    provider: str = "haystack"
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    logging_config: LoggingConfigEmbedding = field(default_factory=LoggingConfigEmbedding)

    def __post_init__(self) -> None:
        """Validate main configuration."""
        if not self.provider or not self.provider.strip():
            raise ValueError("Provider cannot be empty")
        
        valid_providers = {"haystack", "openai", "cohere", "huggingface"}
        if self.provider not in valid_providers:
            logger.warning(
                "Provider '%s' is not in the standard list: %s",
                self.provider,
                valid_providers,
            )

        logger.info("EmbeddingConfig initialized: provider=%s", self.provider)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "EmbeddingConfig":
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            EmbeddingConfig instance loaded from the file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the configuration file is invalid.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info("Loading embedding configuration from %s", config_path)

        try:
            with config_path.open("r", encoding="utf-8") as file:
                config_dict = yaml.safe_load(file)

            if not config_dict or "embedding" not in config_dict:
                raise ValueError("Invalid configuration: 'embedding' section not found")

            embedding_config = config_dict["embedding"]

            # Extract model configuration
            model_dict = embedding_config.get("model", {})
            model_config = ModelConfig(
                name=model_dict.get("name", ModelConfig.name),
                device=model_dict.get("device"),
                normalize_embeddings=model_dict.get(
                    "normalize_embeddings", ModelConfig.normalize_embeddings
                ),
                prefix=model_dict.get("prefix"),
            )

            # Extract processing configuration
            processing_dict = embedding_config.get("processing", {})
            processing_config = ProcessingConfig(
                batch_size=processing_dict.get("batch_size", ProcessingConfig.batch_size),
                max_retries=processing_dict.get("max_retries", ProcessingConfig.max_retries),
                timeout=processing_dict.get("timeout", ProcessingConfig.timeout),
            )

            # Extract metadata configuration
            metadata_dict = embedding_config.get("metadata", {})
            metadata_config = MetadataConfig(
                fields_to_embed=metadata_dict.get("fields_to_embed", []),
                include_in_embedding=metadata_dict.get("include_in_embedding", False),
            )

            # Extract logging configuration
            logging_dict = embedding_config.get("logging", {})
            logging_config = LoggingConfigEmbedding(
                level=logging_dict.get("level", "INFO"),
                log_embeddings=logging_dict.get("log_embeddings", False),
                log_model_info=logging_dict.get("log_model_info", True),
            )

            config = cls(
                provider=embedding_config.get("provider", "haystack"),
                model=model_config,
                processing=processing_config,
                metadata=metadata_config,
                logging_config=logging_config,
            )

            logger.info("Successfully loaded embedding configuration")
            return config

        except yaml.YAMLError as error:
            raise ValueError(f"Failed to parse YAML configuration: {error}") from error
        except Exception as error:
            raise ValueError(f"Failed to load configuration: {error}") from error

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "EmbeddingConfig":
        """
        Load configuration from the default or specified path.

        Args:
            config_path: Optional path to configuration file.
                        If None, uses config/embeddings.yml relative to project root.

        Returns:
            EmbeddingConfig instance.
        """
        if config_path is None:
            # Default to config/embeddings.yml in project root
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "embeddings.yml"

        return cls.from_yaml(config_path)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "provider": self.provider,
            "model": {
                "name": self.model.name,
                "device": self.model.device,
                "normalize_embeddings": self.model.normalize_embeddings,
                "prefix": self.model.prefix,
            },
            "processing": {
                "batch_size": self.processing.batch_size,
                "max_retries": self.processing.max_retries,
                "timeout": self.processing.timeout,
            },
            "metadata": {
                "fields_to_embed": self.metadata.fields_to_embed,
                "include_in_embedding": self.metadata.include_in_embedding,
            },
            "logging": {
                "level": self.logging_config.level,
                "log_embeddings": self.logging_config.log_embeddings,
                "log_model_info": self.logging_config.log_model_info,
            },
        }

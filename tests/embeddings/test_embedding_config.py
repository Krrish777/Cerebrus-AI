"""Tests for embedding configuration module."""

from pathlib import Path
from typing import Dict, Any

import pytest
import yaml

from src.embeddings.config.embedding_config import (
    EmbeddingConfig,
    LoggingConfigEmbedding,
    MetadataConfig,
    ModelConfig,
    ProcessingConfig,
)


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_valid_model_config(self):
        """Test creation of valid ModelConfig."""
        config = ModelConfig(
            name="BAAI/bge-small-en-v1.5",
            device="cpu",
            normalize_embeddings=True,
            prefix="query: ",
        )

        assert config.name == "BAAI/bge-small-en-v1.5"
        assert config.device == "cpu"
        assert config.normalize_embeddings is True
        assert config.prefix == "query: "

    def test_model_config_empty_name(self):
        """Test ModelConfig validation for empty model name."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelConfig(
                name="",
                device="cpu",
            )

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()

        assert config.name == "BAAI/bge-small-en-v1.5"
        assert config.device is None
        assert config.normalize_embeddings is True
        assert config.prefix is None


class TestProcessingConfig:
    """Tests for ProcessingConfig class."""

    def test_valid_processing_config(self):
        """Test creation of valid ProcessingConfig."""
        config = ProcessingConfig(
            batch_size=64,
            max_retries=5,
            timeout=600,
        )

        assert config.batch_size == 64
        assert config.max_retries == 5
        assert config.timeout == 600

    def test_processing_config_validation_batch_size(self):
        """Test ProcessingConfig validation for invalid batch_size."""
        with pytest.raises(ValueError, match="must be positive"):
            ProcessingConfig(
                batch_size=0,
                max_retries=3,
                timeout=300,
            )

    def test_processing_config_validation_max_retries(self):
        """Test ProcessingConfig validation for invalid max_retries."""
        with pytest.raises(ValueError, match="cannot be negative"):
            ProcessingConfig(
                batch_size=32,
                max_retries=-1,
                timeout=300,
            )

    def test_processing_config_validation_timeout(self):
        """Test ProcessingConfig validation for invalid timeout."""
        with pytest.raises(ValueError, match="must be positive"):
            ProcessingConfig(
                batch_size=32,
                max_retries=3,
                timeout=0,
            )

    def test_processing_config_defaults(self):
        """Test ProcessingConfig default values."""
        config = ProcessingConfig()

        assert config.batch_size == 32
        assert config.max_retries == 3
        assert config.timeout == 300

class TestMetadataConfig:
    """Tests for MetadataConfig class."""

    def test_valid_metadata_config(self):
        """Test creation of valid MetadataConfig."""
        config = MetadataConfig(
            fields_to_embed=["title", "author"],
            include_in_embedding=True,
        )

        assert config.fields_to_embed == ["title", "author"]
        assert config.include_in_embedding is True

    def test_metadata_config_defaults(self):
        """Test MetadataConfig default values."""
        config = MetadataConfig()

        assert config.fields_to_embed == []
        assert config.include_in_embedding is False


class TestLoggingConfigEmbedding:
    """Tests for LoggingConfigEmbedding class."""

    def test_valid_logging_config(self):
        """Test creation of valid LoggingConfigEmbedding."""
        config = LoggingConfigEmbedding(
            level="DEBUG",
            log_embeddings=True,
            log_model_info=False,
        )

        assert config.level == "DEBUG"
        assert config.log_embeddings is True
        assert config.log_model_info is False

    def test_logging_config_validation_level(self):
        """Test LoggingConfigEmbedding validation for invalid level."""
        with pytest.raises(ValueError, match="Invalid log level"):
            LoggingConfigEmbedding(
                level="INVALID",
                log_embeddings=False,
                log_model_info=True,
            )

    def test_logging_config_defaults(self):
        """Test LoggingConfigEmbedding default values."""
        config = LoggingConfigEmbedding()

        assert config.level == "INFO"
        assert config.log_embeddings is False
        assert config.log_model_info is True


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig class."""

    def test_valid_embedding_config(self):
        """Test creation of valid EmbeddingConfig."""
        model_config = ModelConfig(
            name="BAAI/bge-small-en-v1.5",
            device="cpu",
            normalize_embeddings=True,
            prefix=None,
        )
        processing_config = ProcessingConfig(
            batch_size=32,
            max_retries=3,
            timeout=300,
        )
        metadata_config = MetadataConfig(
            fields_to_embed=["title"],
            include_in_embedding=True,
        )
        logging_config = LoggingConfigEmbedding(
            level="INFO",
            log_embeddings=False,
            log_model_info=True,
        )

        config = EmbeddingConfig(
            provider="haystack",
            model=model_config,
            processing=processing_config,
            metadata=metadata_config,
            logging_config=logging_config,
        )

        assert config.provider == "haystack"
        assert config.model == model_config
        assert config.processing == processing_config
        assert config.metadata == metadata_config
        assert config.logging_config == logging_config

    def test_embedding_config_to_dict(self):
        """Test EmbeddingConfig conversion to dictionary."""
        model_config = ModelConfig(
            name="test-model",
            device="cpu",
            normalize_embeddings=True,
        )
        processing_config = ProcessingConfig(
            batch_size=32,
            max_retries=3,
            timeout=300,
        )
        metadata_config = MetadataConfig(
            fields_to_embed=["title"],
            include_in_embedding=True,
        )
        logging_config = LoggingConfigEmbedding(
            level="INFO",
            log_embeddings=False,
            log_model_info=True,
        )

        config = EmbeddingConfig(
            provider="haystack",
            model=model_config,
            processing=processing_config,
            metadata=metadata_config,
            logging_config=logging_config,
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["provider"] == "haystack"
        assert config_dict["model"]["name"] == "test-model"
        assert config_dict["processing"]["batch_size"] == 32
        assert config_dict["metadata"]["fields_to_embed"] == ["title"]
        assert config_dict["logging"]["level"] == "INFO"

    def test_embedding_config_from_yaml(self, tmp_path):
        """Test EmbeddingConfig loading from YAML file."""
        yaml_content = """
embedding:
  provider: haystack
  model:
    name: BAAI/bge-small-en-v1.5
    device: cpu
    normalize_embeddings: true
    prefix: null
  processing:
    batch_size: 32
    max_retries: 3
    timeout: 300
  metadata:
    fields_to_embed: []
    include_in_embedding: false
  logging:
    level: INFO
    log_embeddings: false
    log_model_info: true
"""
        yaml_path = tmp_path / "test_embeddings.yml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        config = EmbeddingConfig.from_yaml(yaml_path)

        assert config.provider == "haystack"
        assert config.model.name == "BAAI/bge-small-en-v1.5"
        assert config.processing.batch_size == 32
        assert config.metadata.fields_to_embed == []
        assert config.logging_config.level == "INFO"

    def test_embedding_config_from_yaml_file_not_found(self):
        """Test EmbeddingConfig loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            EmbeddingConfig.from_yaml(Path("nonexistent.yml"))

    def test_embedding_config_load_default(self):
        """Test EmbeddingConfig.load() uses default config path."""
        # This will only work if config/embeddings.yml exists
        default_config_path = Path("config/embeddings.yml")
        if default_config_path.exists():
            config = EmbeddingConfig.load()
            assert isinstance(config, EmbeddingConfig)
            assert config.provider == "haystack"

"""Tests for EmbedderFactory."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.embeddings.config.embedding_config import EmbeddingConfig
from src.embeddings.factories.embedder_factory import EmbedderFactory
from src.embeddings.providers.base import EmbeddingProvider
from src.embeddings.services.batch_processor import BatchProcessor
from src.embeddings.services.document_embedder import DocumentEmbedder
from src.embeddings.services.query_embedder import QueryEmbedder


class MockProvider(EmbeddingProvider):
    """Mock provider for testing."""

    def __init__(self, config):
        self._config = config

    def warm_up(self):
        pass

    def embed_documents(self, documents):
        return []

    def embed_query(self, query):
        import numpy as np

        return np.array([0.1, 0.2, 0.3], dtype=np.float32)

    def get_embedding_dimension(self):
        return 384

    def get_model_info(self):
        return {"provider": "mock"}


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock EmbeddingConfig."""
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
    yaml_path = tmp_path / "test_config.yml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    return EmbeddingConfig.from_yaml(yaml_path)


class TestEmbedderFactory:
    """Tests for EmbedderFactory class."""

    def test_register_provider(self):
        """Test registering a custom provider."""
        factory = EmbedderFactory()

        factory.register_provider("mock", MockProvider)

        assert "mock" in factory.get_available_providers()

    def test_get_available_providers(self):
        """Test getting list of available providers."""
        factory = EmbedderFactory()

        providers = factory.get_available_providers()

        assert "haystack" in providers
        assert isinstance(providers, list)

    def test_create_document_embedder(self, mock_config):
        """Test creating a DocumentEmbedder instance (integration test)."""
        factory = EmbedderFactory()

        embedder = factory.create_document_embedder(mock_config)

        assert isinstance(embedder, DocumentEmbedder)
        assert embedder._provider is not None

    def test_create_query_embedder(self, mock_config):
        """Test creating a QueryEmbedder instance (integration test)."""
        factory = EmbedderFactory()

        embedder = factory.create_query_embedder(mock_config)

        assert isinstance(embedder, QueryEmbedder)
        assert embedder._provider is not None

    @patch(
        "src.embeddings.factories.embedder_factory.HaystackEmbeddingProvider"
    )
    def test_create_batch_processor(self, mock_haystack_provider, mock_config):
        """Test creating a BatchProcessor instance."""
        factory = EmbedderFactory()

        processor = factory.create_batch_processor(mock_config)

        assert isinstance(processor, BatchProcessor)

    def test_create_with_unknown_provider(self, mock_config):
        """Test creating embedder with unknown provider raises ValueError."""
        from dataclasses import replace
        
        factory = EmbedderFactory()

        # Create a new config with different provider
        modified_config = replace(mock_config, provider="unknown_provider")

        with pytest.raises(ValueError, match="Unknown provider"):
            factory.create_document_embedder(modified_config)

    def test_create_with_custom_provider(self, mock_config):
        """Test creating embedder with custom registered provider."""
        from dataclasses import replace
        
        factory = EmbedderFactory()
        factory.register_provider("mock", MockProvider)

        # Create a new config with different provider
        modified_config = replace(mock_config, provider="mock")

        embedder = factory.create_document_embedder(modified_config)

        assert isinstance(embedder, DocumentEmbedder)

    def test_provider_receives_model_config(self, mock_config):
        """Test that provider receives the model config (integration test)."""
        factory = EmbedderFactory()

        embedder = factory.create_document_embedder(mock_config)

        # Verify the provider was created and has the right model
        assert embedder._provider is not None
        assert embedder._provider._config.name == mock_config.model.name

    def test_batch_processor_receives_processing_config(self, mock_config):
        """Test that BatchProcessor receives processing config (integration test)."""
        factory = EmbedderFactory()

        processor = factory.create_batch_processor(mock_config)

        assert processor._config == mock_config.processing

    @patch(
        "src.embeddings.factories.embedder_factory.HaystackEmbeddingProvider"
    )
    def test_factory_from_config_file(self, mock_haystack_provider):
        """Test creating embedders from config file."""
        default_config_path = Path("config/embeddings.yml")

        if not default_config_path.exists():
            pytest.skip("Default config file not found")

        config = EmbeddingConfig.load()
        factory = EmbedderFactory()

        doc_embedder = factory.create_document_embedder(config)
        query_embedder = factory.create_query_embedder(config)
        batch_processor = factory.create_batch_processor(config)

        assert isinstance(doc_embedder, DocumentEmbedder)
        assert isinstance(query_embedder, QueryEmbedder)
        assert isinstance(batch_processor, BatchProcessor)

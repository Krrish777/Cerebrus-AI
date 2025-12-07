"""
Tests for ProviderFactory.

Testing provider registration, creation, and validation.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.vector_database.factory import ProviderFactory
from src.vector_database.config.vectordb_config import VectorDatabaseConfig
from src.vector_database.providers.base_provider import BaseVectorDBProvider


class MockProvider(BaseVectorDBProvider):
    """Mock provider for testing."""
    
    def __init__(self, config):
        self.config = config
        self.initialized = False
    
    def initialize(self):
        self.initialized = True
    
    def insert_documents(self, documents, policy="skip"):
        pass
    
    def search(self, query_embedding, top_k=10, filters=None, score_threshold=None):
        pass
    
    def get_document_by_id(self, doc_id):
        pass
    
    def delete_documents(self, doc_ids):
        pass
    
    def count_documents(self):
        pass
    
    def get_collection_stats(self):
        pass
    
    def get_collection_info(self):
        pass
    
    def collection_exists(self):
        pass
    
    def clear_collection(self):
        pass
    
    def close(self):
        pass
    
    def health_check(self):
        pass


class TestProviderFactoryRegistration:
    """Test provider registration."""
    
    def test_register_provider_success(self):
        """Test successful provider registration."""
        ProviderFactory.register_provider("test_mock", MockProvider)
        
        assert "test_mock" in ProviderFactory.list_providers()
        assert ProviderFactory.is_provider_available("test_mock")
    
    def test_register_provider_case_insensitive(self):
        """Test provider names are case-insensitive."""
        ProviderFactory.register_provider("TestProvider", MockProvider)
        
        assert ProviderFactory.is_provider_available("testprovider")
        assert ProviderFactory.is_provider_available("TESTPROVIDER")
    
    def test_register_provider_empty_name_fails(self):
        """Test registration with empty name fails."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            ProviderFactory.register_provider("", MockProvider)
    
    def test_register_provider_non_string_name_fails(self):
        """Test registration with non-string name fails."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            ProviderFactory.register_provider(123, MockProvider)
    
    def test_register_provider_non_class_fails(self):
        """Test registration with non-class fails."""
        with pytest.raises(ValueError, match="must be a class type"):
            ProviderFactory.register_provider("test", "not_a_class")
    
    def test_register_provider_wrong_base_class_fails(self):
        """Test registration with wrong base class fails."""
        class WrongProvider:
            pass
        
        with pytest.raises(ValueError, match="must implement BaseVectorDBProvider"):
            ProviderFactory.register_provider("wrong", WrongProvider)
    
    def test_list_providers_returns_sorted(self):
        """Test list_providers returns sorted list."""
        ProviderFactory.register_provider("zebra", MockProvider)
        ProviderFactory.register_provider("alpha", MockProvider)
        
        providers = ProviderFactory.list_providers()
        assert providers == sorted(providers)
    
    def test_is_provider_available_returns_false_for_unknown(self):
        """Test is_provider_available returns False for unknown provider."""
        assert not ProviderFactory.is_provider_available("nonexistent")


class TestProviderFactoryCreation:
    """Test provider creation."""
    
    def test_create_provider_success(self):
        """Test successful provider creation."""
        config = VectorDatabaseConfig(
            provider="test_mock",
            storage_path=Path("./test_data"),
            collection_name="test",
            embedding_dim=384
        )
        
        ProviderFactory.register_provider("test_mock", MockProvider)
        provider = ProviderFactory.create_provider(config, auto_initialize=False)
        
        assert isinstance(provider, MockProvider)
        assert provider.config == config
        assert not provider.initialized
    
    def test_create_provider_with_auto_initialize(self):
        """Test provider creation with auto-initialization."""
        config = VectorDatabaseConfig(
            provider="test_mock",
            storage_path=Path("./test_data"),
            collection_name="test",
            embedding_dim=384
        )
        
        ProviderFactory.register_provider("test_mock", MockProvider)
        provider = ProviderFactory.create_provider(config, auto_initialize=True)
        
        assert isinstance(provider, MockProvider)
        assert provider.initialized
    
    def test_create_provider_invalid_config_fails(self):
        """Test provider creation with invalid config fails."""
        with pytest.raises(ValueError, match="must be a VectorDatabaseConfig"):
            ProviderFactory.create_provider("not_a_config")
    
    def test_create_provider_unknown_provider_fails(self):
        """Test provider creation with unknown provider fails."""
        config = VectorDatabaseConfig(
            provider="unknown_provider",
            storage_path=Path("./test_data"),
            collection_name="test",
            embedding_dim=384
        )
        
        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderFactory.create_provider(config)
    
    def test_create_provider_shows_available_providers_on_error(self):
        """Test error message shows available providers."""
        config = VectorDatabaseConfig(
            provider="unknown",
            storage_path=Path("./test_data"),
            collection_name="test",
            embedding_dim=384
        )
        
        ProviderFactory.register_provider("alpha", MockProvider)
        ProviderFactory.register_provider("beta", MockProvider)
        
        with pytest.raises(ValueError, match="Available providers"):
            ProviderFactory.create_provider(config)
    
    def test_create_provider_initialization_error_fails(self):
        """Test provider creation fails if initialization fails."""
        class BrokenProvider(MockProvider):
            def initialize(self):
                raise RuntimeError("Initialization failed")
        
        config = VectorDatabaseConfig(
            provider="broken",
            storage_path=Path("./test_data"),
            collection_name="test",
            embedding_dim=384
        )
        
        ProviderFactory.register_provider("broken", BrokenProvider)
        
        with pytest.raises(ValueError, match="Provider creation failed"):
            ProviderFactory.create_provider(config, auto_initialize=True)


class TestProviderFactoryIntegration:
    """Test factory integration with real provider."""
    
    def test_qdrant_provider_auto_registered(self):
        """Test Qdrant provider is auto-registered."""
        assert ProviderFactory.is_provider_available("qdrant")
    
    def test_create_qdrant_provider(self, tmp_path):
        """Test creating Qdrant provider via factory."""
        config = VectorDatabaseConfig(
            provider="qdrant",
            storage_path=tmp_path / "qdrant_data",
            collection_name="test_collection",
            embedding_dim=384
        )
        
        provider = ProviderFactory.create_provider(config, auto_initialize=False)
        
        from src.vector_database.providers.qdrant_provider import QdrantProvider
        assert isinstance(provider, QdrantProvider)
        assert provider.config == config
    
    def test_create_qdrant_provider_with_initialization(self, tmp_path):
        """Test creating and initializing Qdrant provider."""
        config = VectorDatabaseConfig(
            provider="qdrant",
            storage_path=tmp_path / "qdrant_data",
            collection_name="test_collection",
            embedding_dim=384
        )
        
        provider = ProviderFactory.create_provider(config, auto_initialize=True)
        
        # Verify initialization
        assert provider.collection_exists()
        
        # Clean up
        provider.close()

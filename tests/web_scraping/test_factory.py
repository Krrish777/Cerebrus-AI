"""
Tests for Web Scraping Factory.

Tests the factory classes for creating configured web scraping components.
"""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from src.web_scraping.config import WebScrapingConfig
from src.web_scraping.config import ScrapingConfig
from src.web_scraping.config import URLValidationConfig
from src.web_scraping.config import CacheConfig
from src.web_scraping.config import RateLimitConfig
from src.web_scraping.config import ProcessingConfig
from src.web_scraping.config import FirecrawlProviderConfig
from src.web_scraping.config import ContentValidationConfig
from src.web_scraping.components.factory import WebScrapingFactory


class TestWebScrapingFactory:
    """Tests for WebScrapingFactory."""

    @pytest.fixture
    def minimal_config(self, tmp_path: Path) -> WebScrapingConfig:
        """Create minimal web scraping configuration."""
        return WebScrapingConfig(
            scraping=ScrapingConfig(),
            firecrawl=FirecrawlProviderConfig(),
            url_validation=URLValidationConfig(),
            content_validation=ContentValidationConfig(),
            rate_limiting=RateLimitConfig(enabled=False),
            cache=CacheConfig(enabled=False, cache_dir=None),
            processing=ProcessingConfig(),
        )

    @pytest.fixture
    def full_config(self, tmp_path: Path) -> WebScrapingConfig:
        """Create full web scraping configuration."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        return WebScrapingConfig(
            scraping=ScrapingConfig(
                timeout_seconds=30,
                retry_attempts=3,
            ),
            firecrawl=FirecrawlProviderConfig(),
            url_validation=URLValidationConfig(
                allowed_schemes=("https", "http"),
                blocked_domains=("blocked.com",),
            ),
            content_validation=ContentValidationConfig(
                min_content_length=100,
                max_content_length=1000000,
            ),
            rate_limiting=RateLimitConfig(
                enabled=True,
                requests_per_minute=60,
                burst_size=10,
            ),
            cache=CacheConfig(
                enabled=True,
                cache_dir=cache_dir,
                ttl_hours=24,
            ),
            processing=ProcessingConfig(),
        )


class TestCreateComponents(TestWebScrapingFactory):
    """Tests for creating individual components."""

    def test_create_url_validator(self, minimal_config: WebScrapingConfig) -> None:
        """Test creating URL validator."""
        factory = WebScrapingFactory(config=minimal_config)
        
        validator = factory.create_url_validator()
        
        assert validator is not None

    def test_create_content_validator(self, minimal_config: WebScrapingConfig) -> None:
        """Test creating content validator."""
        factory = WebScrapingFactory(config=minimal_config)
        
        validator = factory.create_content_validator()
        
        assert validator is not None

    def test_create_rate_limiter_disabled(self, minimal_config: WebScrapingConfig) -> None:
        """Test creating rate limiter when disabled."""
        factory = WebScrapingFactory(config=minimal_config)
        
        limiter = factory.create_rate_limiter()
        
        # Returns None when rate limiting is disabled
        assert limiter is None

    def test_create_rate_limiter_enabled(self, full_config: WebScrapingConfig) -> None:
        """Test creating rate limiter when enabled."""
        factory = WebScrapingFactory(config=full_config)
        
        limiter = factory.create_rate_limiter()
        
        assert limiter is not None

    def test_create_document_builder(self, minimal_config: WebScrapingConfig) -> None:
        """Test creating document builder."""
        factory = WebScrapingFactory(config=minimal_config)
        
        builder = factory.create_document_builder()
        
        assert builder is not None

    def test_create_cache_manager_enabled(self, full_config: WebScrapingConfig) -> None:
        """Test creating enabled cache manager."""
        factory = WebScrapingFactory(config=full_config)
        
        cache = factory.create_cache_manager()
        
        assert cache is not None

    def test_create_cache_manager_disabled(self, minimal_config: WebScrapingConfig) -> None:
        """Test creating cache manager when disabled."""
        factory = WebScrapingFactory(config=minimal_config)
        
        cache = factory.create_cache_manager()
        
        # Returns None when caching is disabled
        assert cache is None


class TestCreateScraper(TestWebScrapingFactory):
    """Tests for creating scraper providers."""

    def test_create_firecrawl_scraper(self, minimal_config: WebScrapingConfig) -> None:
        """Test creating Firecrawl scraper."""
        factory = WebScrapingFactory(config=minimal_config)
        
        scraper = factory.create_scraper()
        
        assert scraper is not None


class TestCreateOrchestrator(TestWebScrapingFactory):
    """Tests for creating orchestrator."""

    def test_create_orchestrator(self, full_config: WebScrapingConfig) -> None:
        """Test creating orchestrator."""
        factory = WebScrapingFactory(config=full_config)
        
        orchestrator = factory.create_orchestrator()
        
        assert orchestrator is not None

    def test_create_orchestrator_without_chunking(self, full_config: WebScrapingConfig) -> None:
        """Test creating orchestrator without chunking."""
        factory = WebScrapingFactory(config=full_config)
        
        orchestrator = factory.create_orchestrator(enable_chunking=False)
        
        assert orchestrator is not None

    def test_create_chunking_service(self, minimal_config: WebScrapingConfig) -> None:
        """Test creating chunking service."""
        factory = WebScrapingFactory(config=minimal_config)
        
        chunking_service = factory.create_chunking_service()
        
        assert chunking_service is not None


class TestFactoryConfiguration(TestWebScrapingFactory):
    """Tests for factory configuration handling."""

    def test_factory_with_config(self, full_config: WebScrapingConfig) -> None:
        """Test that factory uses provided config."""
        factory = WebScrapingFactory(config=full_config)
        
        config = factory.get_config()
        
        assert config == full_config

    def test_factory_create_default_config(self) -> None:
        """Test factory creates default config when none provided."""
        factory = WebScrapingFactory()
        
        # Should not raise - will use default config
        config = factory.get_config()
        
        assert config is not None


class TestFactoryEdgeCases(TestWebScrapingFactory):
    """Tests for edge cases."""

    def test_multiple_orchestrator_creation(self, full_config: WebScrapingConfig) -> None:
        """Test creating multiple orchestrators."""
        factory = WebScrapingFactory(config=full_config)
        
        orchestrator1 = factory.create_orchestrator()
        orchestrator2 = factory.create_orchestrator()
        
        # Should create separate instances
        assert orchestrator1 is not orchestrator2

    def test_factory_caches_config(self, full_config: WebScrapingConfig) -> None:
        """Test that factory caches configuration."""
        factory = WebScrapingFactory(config=full_config)
        
        config1 = factory.get_config()
        config2 = factory.get_config()
        
        # Should return the same config instance
        assert config1 is config2

"""
Integration Tests for Web Scraping Module.

Tests the integration between components and end-to-end workflows.
"""

from pathlib import Path
from typing import List
from unittest.mock import AsyncMock
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
from src.web_scraping.scraping.orchestrator import DefaultWebScrapingOrchestrator
from src.web_scraping.interfaces import ScrapedContent
from src.web_scraping.interfaces import ScrapeResult


class TestEndToEndWorkflow:
    """Integration tests for end-to-end scraping workflow."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> WebScrapingConfig:
        """Create web scraping configuration for testing."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        return WebScrapingConfig(
            scraping=ScrapingConfig(
                timeout_seconds=30,
                retry_attempts=2,
            ),
            firecrawl=FirecrawlProviderConfig(),
            url_validation=URLValidationConfig(
                allowed_schemes=("https", "http"),
                blocked_domains=("blocked.example.com",),
            ),
            content_validation=ContentValidationConfig(
                min_content_length=10,
                max_content_length=1000000,
            ),
            rate_limiting=RateLimitConfig(
                enabled=True,
                requests_per_minute=600,  # High rate for testing
                burst_size=20,
            ),
            cache=CacheConfig(
                enabled=True,
                cache_dir=cache_dir,
                ttl_hours=24,
            ),
            processing=ProcessingConfig(),
        )

    @pytest.fixture
    def factory(self, config: WebScrapingConfig) -> WebScrapingFactory:
        """Create web scraping factory."""
        return WebScrapingFactory(config=config)


class TestFactoryToOrchestrator(TestEndToEndWorkflow):
    """Tests for factory to orchestrator integration."""

    def test_factory_creates_working_orchestrator(
        self,
        factory: WebScrapingFactory,
    ) -> None:
        """Test that factory creates working orchestrator."""
        orchestrator = factory.create_orchestrator()
        
        assert orchestrator is not None
        assert isinstance(orchestrator, DefaultWebScrapingOrchestrator)


class TestConfigurationIntegration(TestEndToEndWorkflow):
    """Tests for configuration integration."""

    def test_config_flows_through_components(self, tmp_path: Path) -> None:
        """Test that configuration flows through all components."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        config = WebScrapingConfig(
            scraping=ScrapingConfig(timeout_seconds=60),
            firecrawl=FirecrawlProviderConfig(),
            url_validation=URLValidationConfig(
                allowed_schemes=("https",),
                blocked_domains=("spam.com",),
            ),
            content_validation=ContentValidationConfig(
                min_content_length=50,
                max_content_length=500000,
            ),
            rate_limiting=RateLimitConfig(
                enabled=True,
                requests_per_minute=300,
                burst_size=15,
            ),
            cache=CacheConfig(
                enabled=True,
                cache_dir=cache_dir,
                ttl_hours=48,
            ),
            processing=ProcessingConfig(),
        )
        
        factory = WebScrapingFactory(config=config)
        
        # Create components and verify they exist
        url_validator = factory.create_url_validator()
        content_validator = factory.create_content_validator()
        cache_manager = factory.create_cache_manager()
        rate_limiter = factory.create_rate_limiter()
        document_builder = factory.create_document_builder()
        
        assert url_validator is not None
        assert content_validator is not None
        assert cache_manager is not None
        assert rate_limiter is not None
        assert document_builder is not None


class TestModuleImports:
    """Tests for module imports and exports."""

    def test_main_exports(self) -> None:
        """Test that main module exports expected classes."""
        from src.web_scraping import (
            WebScrapingConfig,
            WebScrapingFactory,
            WebScrapingError,
        )
        
        assert WebScrapingConfig is not None
        assert WebScrapingFactory is not None
        assert WebScrapingError is not None

    def test_component_imports(self) -> None:
        """Test that component modules can be imported."""
        from src.web_scraping.validation.url_validator import DefaultURLValidator
        from src.web_scraping.validation.content_validator import DefaultContentValidator
        from src.web_scraping.cache.manager import FileCacheManager
        from src.web_scraping.rate_limiting.limiter import TokenBucketRateLimiter
        from src.web_scraping.processing.document_builder import DefaultDocumentBuilder
        from src.web_scraping.scraping.orchestrator import DefaultWebScrapingOrchestrator
        
        assert DefaultURLValidator is not None
        assert DefaultContentValidator is not None
        assert FileCacheManager is not None
        assert TokenBucketRateLimiter is not None
        assert DefaultDocumentBuilder is not None
        assert DefaultWebScrapingOrchestrator is not None

    def test_interface_imports(self) -> None:
        """Test that interfaces can be imported."""
        from src.web_scraping.interfaces import (
            WebScraper,
            URLValidator,
            ContentValidator,
            CacheManager,
            RateLimiter,
            DocumentBuilder,
            WebScrapingOrchestrator,
            ScrapedContent,
            ScrapeResult,
        )
        
        assert WebScraper is not None
        assert URLValidator is not None
        assert ContentValidator is not None
        assert CacheManager is not None
        assert RateLimiter is not None
        assert DocumentBuilder is not None
        assert WebScrapingOrchestrator is not None
        assert ScrapedContent is not None
        assert ScrapeResult is not None

    def test_exception_imports(self) -> None:
        """Test that exceptions can be imported."""
        from src.web_scraping.exceptions import (
            WebScrapingError,
            ConfigurationError,
            ScrapingProviderError,
            ScrapingTimeoutError,
            URLValidationError,
            ContentValidationError,
            RateLimitExceededError,
            CacheError,
        )
        
        assert WebScrapingError is not None
        assert ConfigurationError is not None
        assert ScrapingProviderError is not None
        assert ScrapingTimeoutError is not None
        assert URLValidationError is not None
        assert ContentValidationError is not None
        assert RateLimitExceededError is not None
        assert CacheError is not None

    def test_config_imports(self) -> None:
        """Test that config classes can be imported."""
        from src.web_scraping.config import (
            WebScrapingConfig,
            ScrapingConfig,
            FirecrawlProviderConfig,
            URLValidationConfig,
            ContentValidationConfig,
            RateLimitConfig,
            CacheConfig,
            ProcessingConfig,
        )
        
        assert WebScrapingConfig is not None
        assert ScrapingConfig is not None
        assert FirecrawlProviderConfig is not None
        assert URLValidationConfig is not None
        assert ContentValidationConfig is not None
        assert RateLimitConfig is not None
        assert CacheConfig is not None
        assert ProcessingConfig is not None

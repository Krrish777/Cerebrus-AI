"""
Tests for Web Scraping Orchestrator.

Tests the orchestrator which coordinates all web scraping components
including scraping providers, validators, cache, rate limiting, and
document building.
"""

from typing import Dict
from typing import List
from unittest.mock import MagicMock

import pytest

from src.web_scraping.scraping.orchestrator import DefaultWebScrapingOrchestrator
from src.web_scraping.interfaces import (
    ScrapedContent,
    ScrapeResult,
    WebScraper,
    URLValidator,
    ContentValidator,
    CacheManager,
    RateLimiter,
    DocumentBuilder,
)
from src.web_scraping.exceptions import (
    ScrapingProviderError,
    URLValidationError,
    ContentValidationError,
    RateLimitExceededError,
)


class TestDefaultWebScrapingOrchestrator:
    """Tests for DefaultWebScrapingOrchestrator implementation."""

    @pytest.fixture
    def mock_scraper(self) -> MagicMock:
        """Create mock scraper."""
        scraper = MagicMock(spec=WebScraper)
        scraper.scrape.return_value = ScrapeResult(
            scraped_content=ScrapedContent(
                url="https://example.com",
                content="Scraped content for testing",
                title="Test Page",
                word_count=4,
            ),
            success=True,
        )
        scraper.provider_name = "mock_provider"
        return scraper

    @pytest.fixture
    def mock_url_validator(self) -> MagicMock:
        """Create mock URL validator."""
        validator = MagicMock(spec=URLValidator)
        validator.validate.return_value = True
        validator.validate_with_errors.return_value = (True, [])
        validator.extract_domain.return_value = "example.com"
        return validator

    @pytest.fixture
    def mock_content_validator(self) -> MagicMock:
        """Create mock content validator."""
        validator = MagicMock(spec=ContentValidator)
        validator.validate.return_value = True
        validator.validate_with_errors.return_value = (True, [])
        return validator

    @pytest.fixture
    def mock_cache(self) -> MagicMock:
        """Create mock cache manager."""
        cache = MagicMock(spec=CacheManager)
        cache.get.return_value = None
        cache.exists.return_value = False
        cache.generate_key.side_effect = lambda url: f"key_{url}"
        return cache

    @pytest.fixture
    def mock_rate_limiter(self) -> MagicMock:
        """Create mock rate limiter."""
        limiter = MagicMock(spec=RateLimiter)
        limiter.acquire.return_value = True
        limiter.time_until_available.return_value = 0.0
        return limiter

    @pytest.fixture
    def mock_document_builder(self) -> MagicMock:
        """Create mock document builder."""
        builder = MagicMock(spec=DocumentBuilder)
        mock_doc = MagicMock()
        mock_doc.content = "Scraped content"
        mock_doc.meta = {"url": "https://example.com"}
        builder.build.return_value = mock_doc
        return builder

    @pytest.fixture
    def orchestrator(
        self,
        mock_scraper: MagicMock,
        mock_url_validator: MagicMock,
        mock_content_validator: MagicMock,
        mock_cache: MagicMock,
        mock_rate_limiter: MagicMock,
        mock_document_builder: MagicMock,
    ) -> DefaultWebScrapingOrchestrator:
        """Create orchestrator with mocked dependencies."""
        return DefaultWebScrapingOrchestrator(
            scraper=mock_scraper,
            url_validator=mock_url_validator,
            content_validator=mock_content_validator,
            document_builder=mock_document_builder,
            cache_manager=mock_cache,
            rate_limiter=mock_rate_limiter,
            enable_chunking=False,  # Disable chunking for unit tests
        )


class TestScrapeOperation(TestDefaultWebScrapingOrchestrator):
    """Tests for single URL scraping."""

    def test_scrape_success(self, orchestrator: DefaultWebScrapingOrchestrator) -> None:
        """Test successful scraping."""
        documents = orchestrator.scrape("https://example.com")
        
        assert len(documents) >= 1

    def test_scrape_validates_url(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
        mock_url_validator: MagicMock,
    ) -> None:
        """Test that URL is validated before scraping."""
        orchestrator.scrape("https://example.com")
        
        mock_url_validator.validate_with_errors.assert_called_once_with("https://example.com")

    def test_scrape_invalid_url_raises(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
        mock_url_validator: MagicMock,
    ) -> None:
        """Test that invalid URL raises URLValidationError."""
        mock_url_validator.validate_with_errors.return_value = (False, ["Invalid URL"])
        
        with pytest.raises(URLValidationError):
            orchestrator.scrape("invalid-url")

    def test_scrape_uses_rate_limiter(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
        mock_rate_limiter: MagicMock,
    ) -> None:
        """Test that rate limiter is used."""
        orchestrator.scrape("https://example.com")
        
        mock_rate_limiter.acquire.assert_called()

    def test_scrape_rate_limit_exceeded_raises(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
        mock_rate_limiter: MagicMock,
    ) -> None:
        """Test that rate limit exceeded raises RateLimitExceededError."""
        mock_rate_limiter.acquire.return_value = False
        mock_rate_limiter.time_until_available.return_value = 5.0
        
        with pytest.raises(RateLimitExceededError):
            orchestrator.scrape("https://example.com")

    def test_scrape_validates_content(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
        mock_content_validator: MagicMock,
    ) -> None:
        """Test that content is validated after scraping."""
        orchestrator.scrape("https://example.com")
        
        mock_content_validator.validate_with_errors.assert_called()

    def test_scrape_invalid_content_raises(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
        mock_content_validator: MagicMock,
    ) -> None:
        """Test that invalid content raises ContentValidationError."""
        mock_content_validator.validate_with_errors.return_value = (False, ["Too short"])
        
        with pytest.raises(ContentValidationError):
            orchestrator.scrape("https://example.com")


class TestCacheIntegration(TestDefaultWebScrapingOrchestrator):
    """Tests for cache integration."""

    def test_scrape_checks_cache_first(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
        mock_cache: MagicMock,
    ) -> None:
        """Test that cache is checked before scraping."""
        orchestrator.scrape("https://example.com")
        
        mock_cache.get.assert_called()

    def test_scrape_returns_cached_content(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
        mock_cache: MagicMock,
        mock_scraper: MagicMock,
    ) -> None:
        """Test that cached content is returned without scraping."""
        cached_data = {
            "url": "https://example.com",
            "content": "Cached content",
            "title": "Cached Title",
            "word_count": 2,
        }
        mock_cache.get.return_value = cached_data
        
        documents = orchestrator.scrape("https://example.com")
        
        # Scraper should not be called when cache hit
        mock_scraper.scrape.assert_not_called()
        assert len(documents) >= 1

    def test_scrape_caches_result(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
        mock_cache: MagicMock,
    ) -> None:
        """Test that successful scrape result is cached."""
        orchestrator.scrape("https://example.com")
        
        mock_cache.set.assert_called()


class TestBatchScraping(TestDefaultWebScrapingOrchestrator):
    """Tests for batch scraping operations."""

    def test_scrape_batch_multiple_urls(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
    ) -> None:
        """Test batch scraping of multiple URLs."""
        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3",
        ]
        
        results = orchestrator.scrape_batch(urls)
        
        assert len(results) == 3
        for url in urls:
            assert url in results
            assert isinstance(results[url], list)

    def test_scrape_batch_partial_failure(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
        mock_scraper: MagicMock,
    ) -> None:
        """Test batch scraping with some failures."""
        # Make second URL fail
        def selective_scrape(url: str) -> ScrapeResult:
            if "page2" in url:
                return ScrapeResult(
                    scraped_content=None,
                    success=False,
                    error_message="Failed",
                )
            return ScrapeResult(
                scraped_content=ScrapedContent(
                    url=url,
                    content="Content",
                    title="Title",
                    word_count=1,
                ),
                success=True,
            )

        mock_scraper.scrape.side_effect = selective_scrape

        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3",
        ]
        
        results = orchestrator.scrape_batch(urls)
        
        assert len(results) == 3
        # page2 should have empty list (failed)
        assert len(results["https://example.com/page2"]) == 0

    def test_scrape_batch_empty_list(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
    ) -> None:
        """Test batch scraping with empty URL list."""
        results = orchestrator.scrape_batch([])
        
        assert results == {}


class TestPreview(TestDefaultWebScrapingOrchestrator):
    """Tests for preview functionality."""

    def test_preview_returns_scraped_content(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
    ) -> None:
        """Test preview returns ScrapedContent."""
        preview = orchestrator.preview("https://example.com")
        
        assert isinstance(preview, ScrapedContent)
        assert preview.url == "https://example.com"

    def test_preview_truncates_content(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
        mock_scraper: MagicMock,
    ) -> None:
        """Test preview truncates long content."""
        # Setup scraper to return long content
        long_content = "x" * 1000
        mock_scraper.scrape.return_value = ScrapeResult(
            scraped_content=ScrapedContent(
                url="https://example.com",
                content=long_content,
                title="Test",
                word_count=1,
            ),
            success=True,
        )
        
        preview = orchestrator.preview("https://example.com")
        
        # Content should be truncated (max 500 + "...")
        assert len(preview.content) <= 503


class TestErrorHandling(TestDefaultWebScrapingOrchestrator):
    """Tests for error handling."""

    def test_scraper_failure_raises(
        self,
        orchestrator: DefaultWebScrapingOrchestrator,
        mock_scraper: MagicMock,
    ) -> None:
        """Test that scraper failures raise ScrapingProviderError."""
        mock_scraper.scrape.return_value = ScrapeResult(
            scraped_content=None,
            success=False,
            error_message="Network error",
        )
        
        with pytest.raises(ScrapingProviderError):
            orchestrator.scrape("https://example.com")


class TestOrchestratorWithChunking(TestDefaultWebScrapingOrchestrator):
    """Tests for orchestrator with chunking service integration."""

    @pytest.fixture
    def mock_chunking_service(self) -> MagicMock:
        """Create mock chunking service."""
        chunking = MagicMock()
        mock_chunks = {"documents": [MagicMock(), MagicMock()]}
        chunking.chunk_documents.return_value = mock_chunks
        return chunking

    @pytest.fixture
    def orchestrator_with_chunking(
        self,
        mock_scraper: MagicMock,
        mock_url_validator: MagicMock,
        mock_content_validator: MagicMock,
        mock_cache: MagicMock,
        mock_rate_limiter: MagicMock,
        mock_document_builder: MagicMock,
        mock_chunking_service: MagicMock,
    ) -> DefaultWebScrapingOrchestrator:
        """Create orchestrator with chunking service."""
        return DefaultWebScrapingOrchestrator(
            scraper=mock_scraper,
            url_validator=mock_url_validator,
            content_validator=mock_content_validator,
            document_builder=mock_document_builder,
            cache_manager=mock_cache,
            rate_limiter=mock_rate_limiter,
            chunking_service=mock_chunking_service,
            enable_chunking=True,
        )

    def test_scrape_with_chunking(
        self,
        orchestrator_with_chunking: DefaultWebScrapingOrchestrator,
        mock_chunking_service: MagicMock,
    ) -> None:
        """Test that scraping with chunking returns chunked documents."""
        documents = orchestrator_with_chunking.scrape("https://example.com")
        
        # Should have called chunking service
        mock_chunking_service.chunk_documents.assert_called()
        assert len(documents) >= 1

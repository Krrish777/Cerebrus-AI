"""
Tests for Scraping Providers.

Tests the base provider interface and Firecrawl provider implementation.
"""

from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from src.web_scraping.config import FirecrawlProviderConfig
from src.web_scraping.config import ScrapingConfig
from src.web_scraping.scraping.providers.base import BaseScraperProvider
from src.web_scraping.scraping.providers.firecrawl import FirecrawlScraper
from src.web_scraping.interfaces import ScrapedContent
from src.web_scraping.interfaces import ScrapeResult
from src.web_scraping.exceptions import ScrapingProviderError
from src.web_scraping.exceptions import ConfigurationError


class TestBaseScraperProvider:
    """Tests for BaseScraperProvider abstract class."""

    @pytest.fixture
    def concrete_provider_class(self) -> type:
        """Create a concrete implementation of BaseScraperProvider for testing."""

        class ConcreteProvider(BaseScraperProvider):
            """Concrete implementation for testing."""

            def _scrape_impl(self, url: str) -> ScrapedContent:
                """Mock implementation."""
                return ScrapedContent(
                    url=url,
                    content="Test content",
                    title="Test Title",
                    word_count=2,
                )

            def validate_config(self) -> bool:
                """Mock implementation."""
                return True

            @property
            def provider_name(self) -> str:
                """Return provider name."""
                return "concrete_test"

        return ConcreteProvider

    @pytest.fixture
    def default_config(self) -> ScrapingConfig:
        """Create default scraping configuration."""
        return ScrapingConfig()

    def test_provider_requires_config(
        self,
        concrete_provider_class: type,
        default_config: ScrapingConfig,
    ) -> None:
        """Test that provider requires config."""
        provider = concrete_provider_class(config=default_config)
        
        assert provider._config == default_config

    def test_provider_name_property(
        self,
        concrete_provider_class: type,
        default_config: ScrapingConfig,
    ) -> None:
        """Test provider name property."""
        provider = concrete_provider_class(config=default_config)
        
        assert provider.provider_name == "concrete_test"

    def test_scrape_returns_result(
        self,
        concrete_provider_class: type,
        default_config: ScrapingConfig,
    ) -> None:
        """Test that scrape returns ScrapeResult."""
        provider = concrete_provider_class(config=default_config)
        
        result = provider.scrape("https://example.com")
        
        assert isinstance(result, ScrapeResult)
        assert result.success is True
        assert result.scraped_content is not None

    def test_retry_on_failure(
        self,
        concrete_provider_class: type,
    ) -> None:
        """Test retry logic on failure."""
        config = ScrapingConfig(
            retry_attempts=2,
            retry_delay_seconds=0,  # No delay for tests
        )
        provider = concrete_provider_class(config=config)
        
        call_count = 0

        def failing_scrape(url: str) -> ScrapedContent:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ScrapingProviderError(
                    message="Temporary failure",
                    provider_name="test",
                )
            return ScrapedContent(
                url=url,
                content="Success",
                title="Title",
                word_count=1,
            )

        provider._scrape_impl = failing_scrape
        
        result = provider.scrape("https://example.com")
        
        assert call_count == 3  # Initial + 2 retries


class TestFirecrawlScraper:
    """Tests for FirecrawlScraper implementation."""

    @pytest.fixture
    def scraping_config(self) -> ScrapingConfig:
        """Create scraping configuration."""
        return ScrapingConfig(timeout_seconds=30)

    @pytest.fixture
    def provider_config(self) -> FirecrawlProviderConfig:
        """Create Firecrawl provider configuration."""
        return FirecrawlProviderConfig(
            api_key_env="FIRECRAWL_API_KEY",
            formats=("markdown",),
        )

    @pytest.fixture
    def scraper(
        self,
        scraping_config: ScrapingConfig,
        provider_config: FirecrawlProviderConfig,
    ) -> FirecrawlScraper:
        """Create FirecrawlScraper instance."""
        return FirecrawlScraper(
            config=scraping_config,
            provider_config=provider_config,
        )


class TestFirecrawlProviderName:
    """Tests for Firecrawl provider name."""

    def test_provider_name(self) -> None:
        """Test Firecrawl provider name."""
        scraper = FirecrawlScraper(
            config=ScrapingConfig(),
            provider_config=FirecrawlProviderConfig(),
        )
        
        assert scraper.provider_name == "firecrawl"


class TestFirecrawlConfigValidation:
    """Tests for Firecrawl configuration validation."""

    def test_validate_config_missing_api_key(self) -> None:
        """Test validation fails without API key."""
        scraper = FirecrawlScraper(
            config=ScrapingConfig(),
            provider_config=FirecrawlProviderConfig(
                api_key_env="NONEXISTENT_API_KEY",
            ),
        )
        
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError):
                scraper.validate_config()

    def test_validate_config_with_api_key(self) -> None:
        """Test validation succeeds with API key."""
        scraper = FirecrawlScraper(
            config=ScrapingConfig(),
            provider_config=FirecrawlProviderConfig(),
        )
        
        with patch.dict("os.environ", {"FIRECRAWL_API_KEY": "test_key"}):
            result = scraper.validate_config()
            assert result is True


class TestFirecrawlScraping:
    """Tests for Firecrawl scraping operations."""

    @pytest.fixture
    def scraper(self) -> FirecrawlScraper:
        """Create FirecrawlScraper instance."""
        return FirecrawlScraper(
            config=ScrapingConfig(),
            provider_config=FirecrawlProviderConfig(),
        )

    @pytest.fixture
    def mock_firecrawl_result(self) -> MagicMock:
        """Create mock Firecrawl API result."""
        result = MagicMock()
        result.markdown = "# Test Page\n\nThis is test content."
        result.metadata = MagicMock()
        result.metadata.title = "Test Page"
        result.metadata.description = "A test page"
        result.links = ["https://example.com/link1"]
        return result

    def test_scrape_success_with_mock(
        self,
        scraper: FirecrawlScraper,
        mock_firecrawl_result: MagicMock,
    ) -> None:
        """Test successful scraping with mocked Firecrawl."""
        # Mock the _get_app method to return a mock app
        mock_app = MagicMock()
        mock_app.scrape_url.return_value = mock_firecrawl_result
        
        with patch.object(scraper, "_get_app", return_value=mock_app):
            content = scraper._scrape_impl("https://example.com")
        
        assert isinstance(content, ScrapedContent)
        assert content.url == "https://example.com"
        assert "Test Page" in content.content or content.title == "Test Page"

    def test_scrape_timeout_handling(
        self,
        scraper: FirecrawlScraper,
    ) -> None:
        """Test handling of timeout errors."""
        mock_app = MagicMock()
        mock_app.scrape_url.side_effect = TimeoutError("Request timed out")
        
        with patch.object(scraper, "_get_app", return_value=mock_app):
            from src.web_scraping.exceptions import ScrapingTimeoutError
            
            with pytest.raises(ScrapingTimeoutError):
                scraper._scrape_impl("https://example.com")


class TestContentExtraction:
    """Tests for content extraction from Firecrawl results."""

    @pytest.fixture
    def scraper(self) -> FirecrawlScraper:
        """Create FirecrawlScraper instance."""
        return FirecrawlScraper(
            config=ScrapingConfig(),
            provider_config=FirecrawlProviderConfig(),
        )

    def test_extract_content_from_markdown_attr(self, scraper: FirecrawlScraper) -> None:
        """Test extracting content from markdown attribute."""
        result = MagicMock()
        result.markdown = "# Test Content"
        result.content = None
        
        content = scraper._extract_content(result)
        
        assert content == "# Test Content"

    def test_extract_content_from_dict(self, scraper: FirecrawlScraper) -> None:
        """Test extracting content from dict result."""
        result = {"markdown": "# Dict Content", "content": "Fallback"}
        
        content = scraper._extract_content(result)
        
        assert content == "# Dict Content"

    def test_extract_title_from_metadata(self, scraper: FirecrawlScraper) -> None:
        """Test extracting title from metadata."""
        result = MagicMock()
        result.metadata = MagicMock()
        result.metadata.title = "Test Title"
        
        title = scraper._extract_title(result)
        
        assert title == "Test Title"

    def test_extract_title_from_dict(self, scraper: FirecrawlScraper) -> None:
        """Test extracting title from dict result."""
        result = {"metadata": {"title": "Dict Title"}}
        
        title = scraper._extract_title(result)
        
        assert title == "Dict Title"

    def test_extract_links(self, scraper: FirecrawlScraper) -> None:
        """Test extracting links."""
        result = MagicMock()
        result.links = ["https://example.com/link1", "https://example.com/link2"]
        
        links = scraper._extract_links(result)
        
        assert len(links) == 2
        assert "https://example.com/link1" in links

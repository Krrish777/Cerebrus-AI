"""
Firecrawl Scraper Provider.

This module provides a Firecrawl-based web scraping implementation.
Uses the Firecrawl API for robust web scraping with JavaScript rendering.

Following AGENTS.md principles:
    - Single responsibility: Firecrawl-specific scraping
    - Dependency injection: API key from environment
    - Defensive: Validates API key before use
"""

import os
from datetime import datetime
from typing import Any
from typing import List
from typing import Optional
from urllib.parse import urlparse

from src.core.logging import get_logger
from src.web_scraping.config import FirecrawlProviderConfig
from src.web_scraping.config import ScrapingConfig
from src.web_scraping.exceptions import ConfigurationError
from src.web_scraping.exceptions import ScrapingProviderError
from src.web_scraping.exceptions import ScrapingTimeoutError
from src.web_scraping.interfaces import ScrapedContent
from src.web_scraping.scraping.providers.base import BaseScraperProvider

logger = get_logger(__name__)


class FirecrawlScraper(BaseScraperProvider):
    """
    Firecrawl-based web scraper implementation.

    Uses the Firecrawl API for web scraping with features like:
    - JavaScript rendering
    - Content extraction
    - Markdown conversion
    - Link extraction

    Example:
        scraping_config = ScrapingConfig()
        provider_config = FirecrawlProviderConfig()
        scraper = FirecrawlScraper(scraping_config, provider_config)
        result = scraper.scrape("https://example.com")
    """

    def __init__(
        self,
        config: ScrapingConfig,
        provider_config: Optional[FirecrawlProviderConfig] = None,
    ) -> None:
        """
        Initialize the Firecrawl scraper.

        Args:
            config: General scraping configuration.
            provider_config: Firecrawl-specific configuration.
        """
        super().__init__(config)
        self._provider_config = provider_config or FirecrawlProviderConfig()
        self._app: Optional[Any] = None
        self._api_key: Optional[str] = None

        logger.debug(
            "Firecrawl scraper initialized with api_key_env=%s, formats=%s",
            self._provider_config.api_key_env,
            self._provider_config.formats,
        )

    @property
    def provider_name(self) -> str:
        """Return the name of the scraping provider."""
        return "firecrawl"

    def validate_config(self) -> bool:
        """
        Validate Firecrawl configuration.

        Returns:
            True if configuration is valid.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        # Get API key from environment
        api_key = os.environ.get(self._provider_config.api_key_env)

        if not api_key:
            raise ConfigurationError(
                message=(
                    f"Firecrawl API key not found. "
                    f"Set the {self._provider_config.api_key_env} environment variable."
                ),
                config_key=self._provider_config.api_key_env,
            )

        # Validate formats
        if not self._provider_config.formats:
            raise ConfigurationError(
                message="At least one output format must be specified",
                config_key="formats",
            )

        logger.debug("Firecrawl configuration validated successfully")
        return True

    def _get_app(self) -> Any:
        """
        Get or create the Firecrawl app instance.

        Returns:
            Firecrawl app instance.

        Raises:
            ConfigurationError: If API key is not available.
            ScrapingProviderError: If Firecrawl cannot be initialized.
        """
        if self._app is not None:
            return self._app

        # Get API key
        self._api_key = os.environ.get(self._provider_config.api_key_env)
        if not self._api_key:
            raise ConfigurationError(
                message=(
                    f"Firecrawl API key not found. "
                    f"Set the {self._provider_config.api_key_env} environment variable."
                ),
                config_key=self._provider_config.api_key_env,
            )

        try:
            # Try to import FirecrawlApp
            try:
                from firecrawl import FirecrawlApp
            except ImportError:
                from firecrawl import Firecrawl as FirecrawlApp

            self._app = FirecrawlApp(api_key=self._api_key)
            logger.debug("Firecrawl app initialized")
            return self._app

        except ImportError as error:
            raise ScrapingProviderError(
                message="Firecrawl package not installed. Install with: pip install firecrawl-py",
                provider_name=self.provider_name,
                original_error=error,
            ) from error

        except Exception as error:
            raise ScrapingProviderError(
                message=f"Failed to initialize Firecrawl: {error}",
                provider_name=self.provider_name,
                original_error=error,
            ) from error

    def _scrape_impl(self, url: str) -> ScrapedContent:
        """
        Implementation of Firecrawl scraping.

        Args:
            url: URL to scrape.

        Returns:
            ScrapedContent with scraped data.

        Raises:
            ScrapingProviderError: If scraping fails.
            ScrapingTimeoutError: If the request times out.
        """
        app = self._get_app()

        try:
            logger.debug("Calling Firecrawl API for URL: %s", url)

            # Call Firecrawl API - use scrape() method (not scrape_url)
            result = app.scrape(url)

            # Extract content
            content = self._extract_content(result)
            title = self._extract_title(result)
            description = self._extract_description(result)
            links = self._extract_links(result) if self._provider_config.include_links else []
            metadata = self._extract_metadata(result, url)

            return ScrapedContent(
                url=url,
                content=content,
                title=title,
                description=description,
                links=links,
                metadata=metadata,
                scraped_at=datetime.now(),
                content_type="text/markdown",
            )

        except TimeoutError as error:
            raise ScrapingTimeoutError(
                message=f"Timeout scraping URL: {url}",
                timeout_seconds=self._config.timeout_seconds,
                provider_name=self.provider_name,
                url=url,
                original_error=error,
            ) from error

        except Exception as error:
            error_str = str(error).lower()

            # Check for timeout-related errors
            if "timeout" in error_str:
                raise ScrapingTimeoutError(
                    message=f"Timeout scraping URL: {url}",
                    timeout_seconds=self._config.timeout_seconds,
                    provider_name=self.provider_name,
                    url=url,
                    original_error=error,
                ) from error

            # Check for rate limit errors
            if "rate" in error_str and "limit" in error_str:
                raise ScrapingProviderError(
                    message=f"Rate limit exceeded for URL: {url}",
                    provider_name=self.provider_name,
                    url=url,
                    original_error=error,
                ) from error

            raise ScrapingProviderError(
                message=f"Failed to scrape URL: {url} - {error}",
                provider_name=self.provider_name,
                url=url,
                original_error=error,
            ) from error

    def _extract_content(self, result: Any) -> str:
        """
        Extract content from Firecrawl result.

        Args:
            result: Firecrawl API result.

        Returns:
            Extracted content string.
        """
        # Try different content fields
        if hasattr(result, "markdown") and result.markdown:
            return result.markdown

        if hasattr(result, "content") and result.content:
            return result.content

        if isinstance(result, dict):
            return result.get("markdown", result.get("content", ""))

        return ""

    def _extract_title(self, result: Any) -> str:
        """
        Extract title from Firecrawl result.

        Args:
            result: Firecrawl API result.

        Returns:
            Extracted title string.
        """
        # Try metadata first
        if hasattr(result, "metadata") and result.metadata:
            metadata = result.metadata
            if hasattr(metadata, "title") and metadata.title:
                return metadata.title
            if isinstance(metadata, dict):
                return metadata.get("title", "")

        if isinstance(result, dict):
            metadata = result.get("metadata", {})
            if isinstance(metadata, dict):
                return metadata.get("title", "")

        return ""

    def _extract_description(self, result: Any) -> str:
        """
        Extract description from Firecrawl result.

        Args:
            result: Firecrawl API result.

        Returns:
            Extracted description string.
        """
        if hasattr(result, "metadata") and result.metadata:
            metadata = result.metadata
            if hasattr(metadata, "description") and metadata.description:
                return metadata.description
            if isinstance(metadata, dict):
                return metadata.get("description", "")

        if isinstance(result, dict):
            metadata = result.get("metadata", {})
            if isinstance(metadata, dict):
                return metadata.get("description", "")

        return ""

    def _extract_links(self, result: Any) -> List[str]:
        """
        Extract links from Firecrawl result.

        Args:
            result: Firecrawl API result.

        Returns:
            List of extracted links.
        """
        if hasattr(result, "links") and result.links:
            return list(result.links)

        if isinstance(result, dict):
            links = result.get("links", [])
            if isinstance(links, list):
                return links

        return []

    def _extract_metadata(self, result: Any, url: str) -> dict:
        """
        Extract metadata from Firecrawl result.

        Args:
            result: Firecrawl API result.
            url: Original URL.

        Returns:
            Metadata dictionary.
        """
        metadata = {
            "source": "firecrawl",
            "domain": urlparse(url).netloc,
        }

        # Extract from result metadata
        if hasattr(result, "metadata") and result.metadata:
            result_metadata = result.metadata
            if hasattr(result_metadata, "__dict__"):
                for key, value in vars(result_metadata).items():
                    if value is not None and not key.startswith("_"):
                        metadata[key] = value
            elif isinstance(result_metadata, dict):
                metadata.update(result_metadata)

        if isinstance(result, dict):
            result_metadata = result.get("metadata", {})
            if isinstance(result_metadata, dict):
                metadata.update(result_metadata)

        return metadata

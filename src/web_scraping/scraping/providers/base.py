"""
Base Scraper Provider.

This module provides the base class for scraping providers.
All provider implementations should extend this class.

Following AGENTS.md principles:
    - Single responsibility: Common provider functionality
    - Extensibility: Easy to add new providers
    - Template method pattern: Common workflow with customizable steps
"""

import time
from abc import abstractmethod
from typing import Any
from typing import Optional

from src.core.logging import get_logger
from src.web_scraping.config import ScrapingConfig
from src.web_scraping.exceptions import ScrapingProviderError
from src.web_scraping.exceptions import ScrapingTimeoutError
from src.web_scraping.interfaces import ScrapeResult
from src.web_scraping.interfaces import ScrapedContent
from src.web_scraping.interfaces import WebScraper

logger = get_logger(__name__)


class BaseScraperProvider(WebScraper):
    """
    Base class for all scraping providers.

    Provides common functionality for scraping:
    - Retry logic with exponential backoff
    - Timing and logging
    - Error handling

    Subclasses must implement:
    - _scrape_impl: The actual scraping logic
    - provider_name: Property returning provider name
    - validate_config: Configuration validation
    """

    def __init__(self, config: ScrapingConfig) -> None:
        """
        Initialize the base provider.

        Args:
            config: Scraping configuration.
        """
        self._config = config
        self._provider_config: Optional[Any] = None
        logger.debug(
            "Base scraper provider initialized with timeout=%ds, retries=%d",
            self._config.timeout_seconds,
            self._config.retry_attempts,
        )

    def scrape(self, url: str) -> ScrapeResult:
        """
        Scrape content from a URL with retry logic.

        Args:
            url: URL to scrape.

        Returns:
            ScrapeResult containing the scraped content or error.
        """
        start_time = time.time()
        last_error: Optional[Exception] = None

        logger.info("Starting scrape for URL: %s", url)

        for attempt in range(self._config.retry_attempts + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_retry_delay(attempt)
                    logger.debug(
                        "Retry attempt %d/%d after %.1fs delay",
                        attempt,
                        self._config.retry_attempts,
                        delay,
                    )
                    time.sleep(delay)

                scraped_content = self._scrape_impl(url)

                duration = time.time() - start_time
                logger.info(
                    "Successfully scraped URL: %s (%.2fs, %d words)",
                    url,
                    duration,
                    scraped_content.word_count,
                )

                return ScrapeResult(
                    scraped_content=scraped_content,
                    success=True,
                    scrape_duration_seconds=duration,
                )

            except ScrapingTimeoutError as error:
                last_error = error
                logger.warning(
                    "Timeout scraping %s (attempt %d/%d): %s",
                    url,
                    attempt + 1,
                    self._config.retry_attempts + 1,
                    str(error),
                )
                # Don't retry timeouts - they're likely to timeout again
                break

            except ScrapingProviderError as error:
                last_error = error
                logger.warning(
                    "Provider error scraping %s (attempt %d/%d): %s",
                    url,
                    attempt + 1,
                    self._config.retry_attempts + 1,
                    str(error),
                )

            except Exception as error:
                last_error = error
                logger.warning(
                    "Unexpected error scraping %s (attempt %d/%d): %s",
                    url,
                    attempt + 1,
                    self._config.retry_attempts + 1,
                    str(error),
                )

        duration = time.time() - start_time
        error_message = str(last_error) if last_error else "Unknown error"

        logger.error(
            "Failed to scrape URL after %d attempts: %s - %s",
            self._config.retry_attempts + 1,
            url,
            error_message,
        )

        return ScrapeResult(
            scraped_content=None,
            success=False,
            error_message=error_message,
            scrape_duration_seconds=duration,
        )

    def configure(self, config: Any) -> None:
        """
        Configure the scraper with provider-specific settings.

        Args:
            config: Provider-specific configuration.
        """
        self._provider_config = config
        logger.debug(
            "%s provider configured",
            self.provider_name,
        )

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with exponential backoff.

        Args:
            attempt: Current attempt number (1-based).

        Returns:
            Delay in seconds.
        """
        # Exponential backoff: delay * 2^(attempt-1)
        base_delay = self._config.retry_delay_seconds
        return base_delay * (2 ** (attempt - 1))

    @abstractmethod
    def _scrape_impl(self, url: str) -> ScrapedContent:
        """
        Implementation of the actual scraping logic.

        Args:
            url: URL to scrape.

        Returns:
            ScrapedContent with scraped data.

        Raises:
            ScrapingProviderError: If scraping fails.
            ScrapingTimeoutError: If the request times out.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the scraping provider."""

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate scraper configuration.

        Returns:
            True if configuration is valid.

        Raises:
            ConfigurationError: If configuration is invalid.
        """

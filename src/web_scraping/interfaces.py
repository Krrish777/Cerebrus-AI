"""
Web Scraping Interfaces Module.

This module defines abstract interfaces for all web scraping components.
Following the Dependency Inversion Principle, high-level modules depend on
these abstractions rather than concrete implementations.

Key Interfaces:
    - WebScraper: Scrapes content from URLs
    - URLValidator: Validates URLs before scraping
    - ContentValidator: Validates scraped content quality
    - DocumentBuilder: Builds Haystack Documents from scraped content
    - CacheManager: Manages cached scraped content
    - RateLimiter: Controls request rate to avoid overloading

These interfaces enable:
    - Easy testing through mock implementations
    - Swappable implementations (e.g., different scraping backends)
    - Clear contracts between components
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from haystack import Document


@dataclass
class ScrapedContent:
    """
    Data class representing scraped web content.

    This is an immutable container for content extracted from a URL.

    Attributes:
        url: Original URL that was scraped.
        content: Main text content extracted from the page.
        title: Page title.
        description: Page meta description.
        links: List of links found on the page.
        metadata: Additional metadata from the page.
        scraped_at: Timestamp when the content was scraped.
        word_count: Number of words in the content.
        content_type: MIME type of the original content.
    """

    url: str
    content: str
    title: str = ""
    description: str = ""
    links: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    scraped_at: datetime = field(default_factory=datetime.now)
    word_count: int = 0
    content_type: str = "text/html"

    def __post_init__(self) -> None:
        """Calculate word count if not provided."""
        if self.word_count == 0 and self.content:
            self.word_count = len(self.content.split())

    def to_dict(self) -> Dict[str, Any]:
        """Convert scraped content to a dictionary for serialization."""
        return {
            "url": self.url,
            "content": self.content,
            "title": self.title,
            "description": self.description,
            "links": self.links,
            "metadata": self.metadata,
            "scraped_at": self.scraped_at.isoformat(),
            "word_count": self.word_count,
            "content_type": self.content_type,
        }


@dataclass
class ScrapeResult:
    """
    Result of a scraping operation.

    Attributes:
        scraped_content: The scraped content, if successful.
        success: Whether the scraping was successful.
        error_message: Error message if scraping failed.
        from_cache: Whether content was retrieved from cache.
        scrape_duration_seconds: Time taken to scrape in seconds.
    """

    scraped_content: Optional[ScrapedContent]
    success: bool
    error_message: str = ""
    from_cache: bool = False
    scrape_duration_seconds: float = 0.0


class WebScraper(ABC):
    """
    Abstract interface for web scraping providers.

    Implementations of this interface handle the actual scraping logic,
    potentially using different backends (Firecrawl, Playwright, etc.).

    Example:
        scraper = FirecrawlScraper(config)
        result = scraper.scrape("https://example.com")
        if result.success:
            print(f"Title: {result.scraped_content.title}")
    """

    @abstractmethod
    def scrape(self, url: str) -> ScrapeResult:
        """
        Scrape content from a URL.

        Args:
            url: URL to scrape.

        Returns:
            ScrapeResult containing the scraped content or error.

        Raises:
            ScrapingProviderError: If scraping fails.
            ScrapingTimeoutError: If the request times out.
        """

    @abstractmethod
    def configure(self, config: Any) -> None:
        """
        Configure the scraper with settings.

        Args:
            config: Provider-specific configuration.
        """

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate scraper configuration.

        Returns:
            True if configuration is valid.

        Raises:
            ConfigurationError: If configuration is invalid.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the scraping provider."""


class URLValidator(ABC):
    """
    Abstract interface for URL validation.

    This interface validates URLs against configured constraints such as:
    - Allowed schemes (http, https)
    - Blocked domains
    - URL length limits
    - Valid TLD requirements

    Example:
        validator = DefaultURLValidator(config)
        is_valid, errors = validator.validate_with_errors(url)
    """

    @abstractmethod
    def validate(self, url: str) -> bool:
        """
        Validate if URL is acceptable for scraping.

        Args:
            url: URL to validate.

        Returns:
            True if valid, False otherwise.
        """

    @abstractmethod
    def validate_with_errors(self, url: str) -> Tuple[bool, List[str]]:
        """
        Validate URL and return validation errors.

        Args:
            url: URL to validate.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """

    @abstractmethod
    def normalize_url(self, url: str) -> str:
        """
        Normalize a URL to a standard format.

        Args:
            url: URL to normalize.

        Returns:
            Normalized URL.

        Raises:
            URLValidationError: If the URL cannot be normalized.
        """

    @abstractmethod
    def extract_domain(self, url: str) -> str:
        """
        Extract the domain from a URL.

        Args:
            url: URL to extract domain from.

        Returns:
            Domain string.
        """


class ContentValidator(ABC):
    """
    Abstract interface for content validation.

    This interface validates scraped content quality:
    - Minimum/maximum content length
    - Minimum word count
    - Content quality checks

    Example:
        validator = DefaultContentValidator(config)
        if validator.validate(content, metadata):
            process(content)
    """

    @abstractmethod
    def validate(self, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Validate scraped content quality.

        Args:
            content: Scraped content.
            metadata: Content metadata.

        Returns:
            True if valid, False otherwise.
        """

    @abstractmethod
    def validate_with_errors(
        self,
        content: str,
        metadata: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Validate content and return validation errors.

        Args:
            content: Scraped content.
            metadata: Content metadata.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """


class DocumentBuilder(ABC):
    """
    Abstract interface for building documents from scraped content.

    This interface creates Haystack Documents with appropriate metadata
    from scraped web content.

    Example:
        builder = DefaultDocumentBuilder()
        document = builder.build(scraped_content)
    """

    @abstractmethod
    def build(self, scraped_content: ScrapedContent) -> Document:
        """
        Build Haystack Document from scraped content.

        Args:
            scraped_content: Scraped content with metadata.

        Returns:
            Haystack Document.
        """

    @abstractmethod
    def build_batch(self, scraped_contents: List[ScrapedContent]) -> List[Document]:
        """
        Build multiple documents from scraped contents.

        Args:
            scraped_contents: List of scraped contents.

        Returns:
            List of Haystack Documents.
        """


class CacheManager(ABC):
    """
    Abstract interface for managing cached scraped content.

    The cache manager handles:
    - Storing scraped content
    - Retrieving cached content by URL
    - Cache expiration and cleanup
    - Cache size management

    Example:
        cache = FileCacheManager(config)
        cache_key = cache.generate_key(url)
        if cache.exists(cache_key):
            content = cache.get(cache_key)
        else:
            content = scraper.scrape(url)
            cache.set(cache_key, content.to_dict())
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached content by key.

        Args:
            key: Cache key.

        Returns:
            Cached content dictionary, or None if not cached.

        Raises:
            CacheError: If cache access fails.
        """

    @abstractmethod
    def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store content in cache.

        Args:
            key: Cache key.
            value: Content dictionary to cache.
            ttl: Optional time-to-live in seconds.

        Raises:
            CacheError: If caching fails.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key.

        Returns:
            True if the key exists in the cache.
        """

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete cached content.

        Args:
            key: Cache key.

        Returns:
            True if the key was deleted, False if it didn't exist.

        Raises:
            CacheError: If deletion fails.
        """

    @abstractmethod
    def cleanup(self) -> int:
        """
        Cleanup expired cache entries.

        Returns:
            Number of entries removed.

        Raises:
            CacheError: If cleanup fails.
        """

    @abstractmethod
    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries removed.

        Raises:
            CacheError: If clearing fails.
        """

    @abstractmethod
    def generate_key(self, url: str) -> str:
        """
        Generate a cache key from a URL.

        Args:
            url: URL to generate key for.

        Returns:
            Cache key string.
        """

    @property
    @abstractmethod
    def cache_size_bytes(self) -> int:
        """Return the total size of the cache in bytes."""

    @property
    @abstractmethod
    def entry_count(self) -> int:
        """Return the number of entries in the cache."""


class RateLimiter(ABC):
    """
    Abstract interface for rate limiting.

    This interface implements rate limiting to prevent overloading
    target servers. Typically uses a token bucket algorithm.

    Example:
        limiter = TokenBucketRateLimiter(config)
        domain = extract_domain(url)
        if limiter.acquire(domain):
            scrape(url)
        else:
            wait(limiter.time_until_available(domain))
    """

    @abstractmethod
    def acquire(self, identifier: str) -> bool:
        """
        Attempt to acquire a rate limit token.

        Args:
            identifier: Identifier for rate limiting (e.g., domain).

        Returns:
            True if allowed, False if rate limited.
        """

    @abstractmethod
    def acquire_blocking(
        self,
        identifier: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquire a token, blocking until available or timeout.

        Args:
            identifier: Identifier for rate limiting (e.g., domain).
            timeout: Maximum time to wait in seconds.

        Returns:
            True if token acquired, False if timeout.
        """

    @abstractmethod
    def time_until_available(self, identifier: str) -> float:
        """
        Get time until a token becomes available.

        Args:
            identifier: Identifier for rate limiting.

        Returns:
            Seconds until next token is available.
        """

    @abstractmethod
    def reset(self, identifier: str) -> None:
        """
        Reset rate limit for an identifier.

        Args:
            identifier: Identifier to reset.
        """

    @abstractmethod
    def reset_all(self) -> None:
        """Reset all rate limits."""


class WebScrapingOrchestrator(ABC):
    """
    Abstract interface for the main web scraping orchestrator.

    This is the high-level interface that coordinates:
    - URL validation
    - Rate limiting
    - Cache checking
    - Content scraping
    - Content validation
    - Document building

    Example:
        orchestrator = DefaultWebScrapingOrchestrator(
            scraper=scraper,
            url_validator=url_validator,
            content_validator=content_validator,
            document_builder=document_builder,
            cache_manager=cache,
            rate_limiter=limiter,
        )
        documents = orchestrator.scrape(url)
    """

    @abstractmethod
    def scrape(self, url: str) -> List[Document]:
        """
        Scrape a URL and return documents.

        This is the main entry point for scraping. It:
        1. Validates the URL
        2. Checks rate limits
        3. Checks cache
        4. Scrapes the content
        5. Validates the content
        6. Builds Haystack Documents

        Args:
            url: URL to scrape.

        Returns:
            List of Haystack Document objects.

        Raises:
            URLValidationError: If the URL is invalid.
            ContentValidationError: If content validation fails.
            ScrapingProviderError: If scraping fails.
            RateLimitExceededError: If rate limit is exceeded.
        """

    @abstractmethod
    def scrape_batch(self, urls: List[str]) -> Dict[str, List[Document]]:
        """
        Scrape multiple URLs.

        Args:
            urls: List of URLs to scrape.

        Returns:
            Dictionary mapping URLs to their document lists.
            Failed URLs map to empty lists.
        """

    @abstractmethod
    def preview(self, url: str) -> ScrapedContent:
        """
        Get a preview of URL content without full processing.

        Args:
            url: URL to preview.

        Returns:
            ScrapedContent with basic information.
        """

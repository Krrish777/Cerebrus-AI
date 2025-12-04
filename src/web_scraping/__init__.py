"""
Web Scraping Module.

This module provides web scraping capabilities for Cerebrus AI.
Supports URL validation, content scraping, caching, rate limiting,
and document creation with chunking.

Key Components:
    - WebScrapingOrchestrator: Main orchestrator coordinating all components
    - FirecrawlScraper: Firecrawl-based web scraping provider
    - DefaultURLValidator: URL validation
    - DefaultContentValidator: Content quality validation
    - FileCacheManager: File-based caching
    - TokenBucketRateLimiter: Rate limiting
    - DefaultDocumentBuilder: Haystack document creation
    - WebScraperComponent: Haystack pipeline component

Example usage:
    from src.web_scraping import create_web_scraper

    scraper = create_web_scraper()
    documents = scraper.scrape("https://example.com")

Or with Haystack pipeline:
    from src.web_scraping import WebScraperComponent

    component = WebScraperComponent()
    result = component.run(urls=["https://example.com"])
"""

# Configuration
from src.web_scraping.config import CacheConfig
from src.web_scraping.config import ContentValidationConfig
from src.web_scraping.config import FirecrawlProviderConfig
from src.web_scraping.config import ProcessingConfig
from src.web_scraping.config import RateLimitConfig
from src.web_scraping.config import ScrapingConfig
from src.web_scraping.config import URLValidationConfig
from src.web_scraping.config import WebScrapingConfig

# Exceptions
from src.web_scraping.exceptions import CacheError
from src.web_scraping.exceptions import ConfigurationError
from src.web_scraping.exceptions import ContentValidationError
from src.web_scraping.exceptions import RateLimitExceededError
from src.web_scraping.exceptions import ScrapingProviderError
from src.web_scraping.exceptions import ScrapingTimeoutError
from src.web_scraping.exceptions import URLValidationError
from src.web_scraping.exceptions import WebScrapingError

# Interfaces
from src.web_scraping.interfaces import CacheManager
from src.web_scraping.interfaces import ContentValidator
from src.web_scraping.interfaces import DocumentBuilder
from src.web_scraping.interfaces import RateLimiter
from src.web_scraping.interfaces import ScrapeResult
from src.web_scraping.interfaces import ScrapedContent
from src.web_scraping.interfaces import URLValidator
from src.web_scraping.interfaces import WebScraper
from src.web_scraping.interfaces import WebScrapingOrchestrator

# Implementations
from src.web_scraping.cache.manager import FileCacheManager
from src.web_scraping.processing.document_builder import DefaultDocumentBuilder
from src.web_scraping.rate_limiting.limiter import TokenBucketRateLimiter
from src.web_scraping.scraping.orchestrator import DefaultWebScrapingOrchestrator
from src.web_scraping.scraping.providers.firecrawl import FirecrawlScraper
from src.web_scraping.validation.content_validator import DefaultContentValidator
from src.web_scraping.validation.url_validator import DefaultURLValidator

# Factory and Components
from src.web_scraping.components.factory import create_web_scraper
from src.web_scraping.components.factory import WebScrapingFactory
from src.web_scraping.components.scraper_component import WebScraperComponent

__all__ = [
    # Configuration
    "WebScrapingConfig",
    "ScrapingConfig",
    "FirecrawlProviderConfig",
    "URLValidationConfig",
    "ContentValidationConfig",
    "RateLimitConfig",
    "CacheConfig",
    "ProcessingConfig",
    # Exceptions
    "WebScrapingError",
    "ConfigurationError",
    "ScrapingProviderError",
    "ScrapingTimeoutError",
    "URLValidationError",
    "ContentValidationError",
    "RateLimitExceededError",
    "CacheError",
    # Interfaces
    "WebScraper",
    "URLValidator",
    "ContentValidator",
    "DocumentBuilder",
    "CacheManager",
    "RateLimiter",
    "WebScrapingOrchestrator",
    "ScrapedContent",
    "ScrapeResult",
    # Implementations
    "FirecrawlScraper",
    "DefaultURLValidator",
    "DefaultContentValidator",
    "DefaultDocumentBuilder",
    "FileCacheManager",
    "TokenBucketRateLimiter",
    "DefaultWebScrapingOrchestrator",
    # Factory and Components
    "WebScrapingFactory",
    "WebScraperComponent",
    "create_web_scraper",
]

__version__ = "2.0.0"
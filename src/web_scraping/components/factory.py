"""
Web Scraping Factory.

This module provides a factory for creating web scraping components.
Handles dependency injection and configuration loading.

Following AGENTS.md principles:
    - Single responsibility: Component creation only
    - Factory pattern: Encapsulates creation logic
    - Dependency injection: Creates properly wired components
"""

from pathlib import Path
from typing import Optional

from src.core.logging import get_logger
from src.document_processing.chunking_service import ChunkingService
from src.document_processing.pipeline_config import PipelineConfig
from src.document_processing.pipeline_config import get_pipeline_config
from src.web_scraping.cache.manager import FileCacheManager
from src.web_scraping.config import WebScrapingConfig
from src.web_scraping.processing.document_builder import DefaultDocumentBuilder
from src.web_scraping.rate_limiting.limiter import TokenBucketRateLimiter
from src.web_scraping.scraping.orchestrator import DefaultWebScrapingOrchestrator
from src.web_scraping.scraping.providers.firecrawl import FirecrawlScraper
from src.web_scraping.validation.content_validator import DefaultContentValidator
from src.web_scraping.validation.url_validator import DefaultURLValidator

logger = get_logger(__name__)


class WebScrapingFactory:
    """
    Factory for creating web scraping components.

    Handles configuration loading and dependency wiring.
    Creates fully configured orchestrator with all dependencies.

    Example:
        factory = WebScrapingFactory()
        orchestrator = factory.create_orchestrator()
        documents = orchestrator.scrape("https://example.com")
    """

    def __init__(
        self,
        config: Optional[WebScrapingConfig] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize the factory.

        Args:
            config: Pre-loaded configuration.
            config_path: Path to configuration file.
        """
        self._config = config
        self._config_path = config_path

        # Cached instances
        self._cached_config: Optional[WebScrapingConfig] = None
        self._cached_pipeline_config: Optional[PipelineConfig] = None

        logger.debug("Web scraping factory initialized")

    def get_config(self) -> WebScrapingConfig:
        """
        Get or load the configuration.

        Returns:
            WebScrapingConfig instance.
        """
        if self._config is not None:
            return self._config

        if self._cached_config is not None:
            return self._cached_config

        if self._config_path is not None:
            self._cached_config = WebScrapingConfig.from_yaml(self._config_path)
        else:
            # Try default config path
            default_path = Path("src/config/web_scraping_config.yml")
            if default_path.exists():
                self._cached_config = WebScrapingConfig.from_yaml(default_path)
            else:
                # Use default configuration
                self._cached_config = WebScrapingConfig.create_default()

        return self._cached_config

    def create_url_validator(self) -> DefaultURLValidator:
        """
        Create a URL validator instance.

        Returns:
            DefaultURLValidator instance.
        """
        config = self.get_config()
        return DefaultURLValidator(config.url_validation)

    def create_content_validator(self) -> DefaultContentValidator:
        """
        Create a content validator instance.

        Returns:
            DefaultContentValidator instance.
        """
        config = self.get_config()
        return DefaultContentValidator(config.content_validation)

    def create_scraper(self) -> FirecrawlScraper:
        """
        Create a web scraper instance.

        Returns:
            FirecrawlScraper instance.
        """
        config = self.get_config()
        return FirecrawlScraper(config.scraping, config.firecrawl)

    def create_cache_manager(self) -> Optional[FileCacheManager]:
        """
        Create a cache manager instance.

        Returns:
            FileCacheManager instance if caching is enabled.
        """
        config = self.get_config()

        if not config.cache.enabled:
            return None

        return FileCacheManager(config.cache)

    def create_rate_limiter(self) -> Optional[TokenBucketRateLimiter]:
        """
        Create a rate limiter instance.

        Returns:
            TokenBucketRateLimiter instance if rate limiting is enabled.
        """
        config = self.get_config()

        if not config.rate_limiting.enabled:
            return None

        return TokenBucketRateLimiter(config.rate_limiting)

    def create_document_builder(self) -> DefaultDocumentBuilder:
        """
        Create a document builder instance.

        Returns:
            DefaultDocumentBuilder instance.
        """
        config = self.get_config()
        return DefaultDocumentBuilder(
            include_links_in_meta=config.processing.extract_links,
        )

    def create_chunking_service(self) -> ChunkingService:
        """
        Create a chunking service instance using existing document_processing.

        Returns:
            ChunkingService instance.
        """
        if self._cached_pipeline_config is None:
            self._cached_pipeline_config = get_pipeline_config()

        return ChunkingService(self._cached_pipeline_config)

    def create_orchestrator(
        self,
        enable_chunking: bool = True,
    ) -> DefaultWebScrapingOrchestrator:
        """
        Create a fully configured web scraping orchestrator.

        Args:
            enable_chunking: Whether to enable document chunking.

        Returns:
            DefaultWebScrapingOrchestrator instance with all dependencies.
        """
        config = self.get_config()

        # Create all components
        scraper = self.create_scraper()
        url_validator = self.create_url_validator()
        content_validator = self.create_content_validator()
        document_builder = self.create_document_builder()
        cache_manager = self.create_cache_manager()
        rate_limiter = self.create_rate_limiter()

        # Create chunking service if enabled
        chunking_service: Optional[ChunkingService] = None
        if enable_chunking and config.processing.use_existing_chunking:
            chunking_service = self.create_chunking_service()

        orchestrator = DefaultWebScrapingOrchestrator(
            scraper=scraper,
            url_validator=url_validator,
            content_validator=content_validator,
            document_builder=document_builder,
            cache_manager=cache_manager,
            rate_limiter=rate_limiter,
            chunking_service=chunking_service,
            enable_chunking=enable_chunking,
        )

        logger.info(
            "Created web scraping orchestrator: "
            "caching=%s, rate_limiting=%s, chunking=%s",
            cache_manager is not None,
            rate_limiter is not None,
            enable_chunking,
        )

        return orchestrator


# Convenience function for quick access
def create_web_scraper(
    config_path: Optional[Path] = None,
    enable_chunking: bool = True,
) -> DefaultWebScrapingOrchestrator:
    """
    Create a web scraping orchestrator with default settings.

    Args:
        config_path: Optional path to configuration file.
        enable_chunking: Whether to enable document chunking.

    Returns:
        Configured DefaultWebScrapingOrchestrator.
    """
    factory = WebScrapingFactory(config_path=config_path)
    return factory.create_orchestrator(enable_chunking=enable_chunking)

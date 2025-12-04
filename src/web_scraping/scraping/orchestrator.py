"""
Web Scraping Orchestrator Implementation.

This module provides the main orchestrator for web scraping operations.
Coordinates all components: validation, rate limiting, caching, scraping,
and document building.

Following AGENTS.md principles:
    - Single responsibility: Orchestration only
    - Dependency injection: All dependencies injected
    - Loose coupling: Uses interfaces, not implementations
    - Integration: Uses existing ChunkingService from document_processing
"""

import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from haystack import Document

from src.core.logging import get_logger
from src.document_processing.chunking_service import ChunkingService
from src.document_processing.pipeline_config import PipelineConfig
from src.document_processing.pipeline_config import get_pipeline_config
from src.web_scraping.exceptions import ContentValidationError
from src.web_scraping.exceptions import RateLimitExceededError
from src.web_scraping.exceptions import ScrapingProviderError
from src.web_scraping.exceptions import URLValidationError
from src.web_scraping.interfaces import CacheManager
from src.web_scraping.interfaces import ContentValidator
from src.web_scraping.interfaces import DocumentBuilder
from src.web_scraping.interfaces import RateLimiter
from src.web_scraping.interfaces import ScrapedContent
from src.web_scraping.interfaces import URLValidator
from src.web_scraping.interfaces import WebScraper
from src.web_scraping.interfaces import WebScrapingOrchestrator

logger = get_logger(__name__)


class DefaultWebScrapingOrchestrator(WebScrapingOrchestrator):
    """
    Default implementation of web scraping orchestration.

    Coordinates the full scraping workflow:
    1. URL validation
    2. Rate limiting
    3. Cache lookup
    4. Content scraping
    5. Content validation
    6. Document building
    7. Chunking (using existing ChunkingService)

    Example:
        orchestrator = DefaultWebScrapingOrchestrator(
            scraper=firecrawl_scraper,
            url_validator=url_validator,
            content_validator=content_validator,
            document_builder=document_builder,
            cache_manager=cache_manager,
            rate_limiter=rate_limiter,
        )
        documents = orchestrator.scrape("https://example.com")
    """

    def __init__(
        self,
        scraper: WebScraper,
        url_validator: URLValidator,
        content_validator: ContentValidator,
        document_builder: DocumentBuilder,
        cache_manager: Optional[CacheManager] = None,
        rate_limiter: Optional[RateLimiter] = None,
        chunking_service: Optional[ChunkingService] = None,
        enable_chunking: bool = True,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            scraper: Web scraper implementation.
            url_validator: URL validator implementation.
            content_validator: Content validator implementation.
            document_builder: Document builder implementation.
            cache_manager: Optional cache manager.
            rate_limiter: Optional rate limiter.
            chunking_service: Optional chunking service (uses existing document_processing).
            enable_chunking: Whether to enable chunking of documents.
        """
        self._scraper = scraper
        self._url_validator = url_validator
        self._content_validator = content_validator
        self._document_builder = document_builder
        self._cache_manager = cache_manager
        self._rate_limiter = rate_limiter
        self._chunking_service = chunking_service
        self._enable_chunking = enable_chunking

        # Lazy-load chunking service if needed
        self._pipeline_config: Optional[PipelineConfig] = None

        logger.debug(
            "Web scraping orchestrator initialized: "
            "caching=%s, rate_limiting=%s, chunking=%s",
            cache_manager is not None,
            rate_limiter is not None,
            enable_chunking,
        )

    def _get_chunking_service(self) -> ChunkingService:
        """
        Get or create the chunking service.

        Returns:
            ChunkingService instance.
        """
        if self._chunking_service is not None:
            return self._chunking_service

        # Use existing document processing ChunkingService
        if self._pipeline_config is None:
            self._pipeline_config = get_pipeline_config()

        self._chunking_service = ChunkingService(self._pipeline_config)
        return self._chunking_service

    def scrape(self, url: str) -> List[Document]:
        """
        Scrape a URL and return documents.

        Args:
            url: URL to scrape.

        Returns:
            List of Haystack Document objects (chunked if enabled).

        Raises:
            URLValidationError: If the URL is invalid.
            ContentValidationError: If content validation fails.
            ScrapingProviderError: If scraping fails.
            RateLimitExceededError: If rate limit is exceeded.
        """
        start_time = time.time()
        logger.info("Starting scrape for URL: %s", url)

        # Step 1: Validate URL
        is_valid, validation_errors = self._url_validator.validate_with_errors(url)
        if not is_valid:
            raise URLValidationError(
                message=f"URL validation failed: {'; '.join(validation_errors)}",
                url=url,
                validation_errors=validation_errors,
            )

        # Step 2: Check rate limit
        domain = self._url_validator.extract_domain(url)
        if self._rate_limiter is not None:
            if not self._rate_limiter.acquire(domain):
                wait_time = self._rate_limiter.time_until_available(domain)
                raise RateLimitExceededError(
                    message=f"Rate limit exceeded for domain: {domain}",
                    identifier=domain,
                    retry_after_seconds=wait_time,
                )

        # Step 3: Check cache
        scraped_content: Optional[ScrapedContent] = None
        from_cache = False

        if self._cache_manager is not None:
            cache_key = self._cache_manager.generate_key(url)
            cached_data = self._cache_manager.get(cache_key)

            if cached_data is not None:
                logger.debug("Cache hit for URL: %s", url)
                scraped_content = self._dict_to_scraped_content(cached_data)
                from_cache = True

        # Step 4: Scrape if not cached
        if scraped_content is None:
            result = self._scraper.scrape(url)

            if not result.success:
                raise ScrapingProviderError(
                    message=f"Scraping failed: {result.error_message}",
                    provider_name=self._scraper.provider_name,
                    url=url,
                )

            scraped_content = result.scraped_content

            # Cache the result
            if self._cache_manager is not None and scraped_content is not None:
                cache_key = self._cache_manager.generate_key(url)
                self._cache_manager.set(cache_key, scraped_content.to_dict())

        # Step 5: Validate content
        if scraped_content is not None:
            is_valid, content_errors = self._content_validator.validate_with_errors(
                scraped_content.content,
                {"url": url},
            )

            if not is_valid:
                raise ContentValidationError(
                    message=f"Content validation failed: {'; '.join(content_errors)}",
                    url=url,
                    content_length=len(scraped_content.content),
                    validation_errors=content_errors,
                )

        # Step 6: Build document
        if scraped_content is None:
            logger.warning("No content scraped from URL: %s", url)
            return []

        document = self._document_builder.build(scraped_content)

        # Add cache status to metadata
        document.meta["from_cache"] = from_cache

        # Step 7: Chunk document if enabled
        documents: List[Document]
        if self._enable_chunking:
            chunking_service = self._get_chunking_service()
            chunk_result = chunking_service.chunk_documents([document])
            documents = chunk_result.get("documents", [document])
        else:
            documents = [document]

        duration = time.time() - start_time
        logger.info(
            "Scrape completed for %s: %d documents in %.2fs (cached=%s)",
            url,
            len(documents),
            duration,
            from_cache,
        )

        return documents

    def scrape_batch(self, urls: List[str]) -> Dict[str, List[Document]]:
        """
        Scrape multiple URLs.

        Args:
            urls: List of URLs to scrape.

        Returns:
            Dictionary mapping URLs to their document lists.
            Failed URLs map to empty lists.
        """
        results: Dict[str, List[Document]] = {}
        start_time = time.time()

        logger.info("Starting batch scrape of %d URLs", len(urls))

        for url in urls:
            try:
                documents = self.scrape(url)
                results[url] = documents

            except (URLValidationError, ContentValidationError) as error:
                logger.warning("Validation error for %s: %s", url, error)
                results[url] = []

            except RateLimitExceededError as error:
                logger.warning("Rate limit for %s: %s", url, error)
                results[url] = []

            except ScrapingProviderError as error:
                logger.warning("Scraping error for %s: %s", url, error)
                results[url] = []

            except Exception as error:
                logger.error("Unexpected error for %s: %s", url, error)
                results[url] = []

        duration = time.time() - start_time
        successful = sum(1 for docs in results.values() if docs)

        logger.info(
            "Batch scrape completed: %d/%d successful in %.2fs",
            successful,
            len(urls),
            duration,
        )

        return results

    def preview(self, url: str) -> ScrapedContent:
        """
        Get a preview of URL content without full processing.

        Args:
            url: URL to preview.

        Returns:
            ScrapedContent with basic information.

        Raises:
            URLValidationError: If the URL is invalid.
            ScrapingProviderError: If scraping fails.
        """
        # Validate URL
        is_valid, validation_errors = self._url_validator.validate_with_errors(url)
        if not is_valid:
            raise URLValidationError(
                message=f"URL validation failed: {'; '.join(validation_errors)}",
                url=url,
                validation_errors=validation_errors,
            )

        # Scrape without caching or chunking
        result = self._scraper.scrape(url)

        if not result.success:
            raise ScrapingProviderError(
                message=f"Preview failed: {result.error_message}",
                provider_name=self._scraper.provider_name,
                url=url,
            )

        if result.scraped_content is None:
            raise ScrapingProviderError(
                message="Preview returned no content",
                provider_name=self._scraper.provider_name,
                url=url,
            )

        # Truncate content for preview
        content = result.scraped_content.content
        max_preview_length = 500
        if len(content) > max_preview_length:
            content = content[:max_preview_length] + "..."

        return ScrapedContent(
            url=result.scraped_content.url,
            content=content,
            title=result.scraped_content.title,
            description=result.scraped_content.description,
            links=result.scraped_content.links[:10],  # Limit links
            metadata={
                **result.scraped_content.metadata,
                "full_content_length": len(result.scraped_content.content),
                "is_preview": True,
            },
            scraped_at=result.scraped_content.scraped_at,
            word_count=result.scraped_content.word_count,
            content_type=result.scraped_content.content_type,
        )

    def _dict_to_scraped_content(self, data: Dict[str, Any]) -> ScrapedContent:
        """
        Convert a dictionary to ScrapedContent.

        Args:
            data: Dictionary from cache.

        Returns:
            ScrapedContent instance.
        """
        from datetime import datetime

        scraped_at = data.get("scraped_at")
        if isinstance(scraped_at, str):
            try:
                scraped_at = datetime.fromisoformat(scraped_at)
            except ValueError:
                scraped_at = datetime.now()
        elif scraped_at is None:
            scraped_at = datetime.now()

        return ScrapedContent(
            url=data.get("url", ""),
            content=data.get("content", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            links=data.get("links", []),
            metadata=data.get("metadata", {}),
            scraped_at=scraped_at,
            word_count=data.get("word_count", 0),
            content_type=data.get("content_type", "text/html"),
        )

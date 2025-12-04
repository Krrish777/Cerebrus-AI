"""
Web Scraper Haystack Component.

This module provides a Haystack-compatible component wrapper for web scraping.
Enables integration with Haystack pipelines.

Following AGENTS.md principles:
    - Single responsibility: Haystack integration only
    - Adapter pattern: Wraps orchestrator for Haystack compatibility
"""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from haystack import Document
from haystack import component

from src.core.logging import get_logger
from src.web_scraping.components.factory import WebScrapingFactory
from src.web_scraping.config import WebScrapingConfig
from src.web_scraping.scraping.orchestrator import DefaultWebScrapingOrchestrator

logger = get_logger(__name__)


@component
class WebScraperComponent:
    """
    Haystack component for web scraping.

    Wraps the WebScrapingOrchestrator to provide Haystack pipeline compatibility.
    Can be used in Haystack pipelines for web content retrieval.

    Example in a Haystack pipeline:
        from haystack import Pipeline
        from src.web_scraping.components import WebScraperComponent

        pipe = Pipeline()
        pipe.add_component("scraper", WebScraperComponent())
        result = pipe.run({"scraper": {"urls": ["https://example.com"]}})
    """

    def __init__(
        self,
        config: Optional[WebScrapingConfig] = None,
        config_path: Optional[Path] = None,
        enable_chunking: bool = True,
    ) -> None:
        """
        Initialize the Haystack web scraper component.

        Args:
            config: Pre-loaded configuration.
            config_path: Path to configuration file.
            enable_chunking: Whether to enable document chunking.
        """
        self._config = config
        self._config_path = config_path
        self._enable_chunking = enable_chunking
        self._orchestrator: Optional[DefaultWebScrapingOrchestrator] = None
        self._factory: Optional[WebScrapingFactory] = None

        logger.debug("Web scraper component initialized")

    def _get_orchestrator(self) -> DefaultWebScrapingOrchestrator:
        """
        Get or create the orchestrator.

        Returns:
            DefaultWebScrapingOrchestrator instance.
        """
        if self._orchestrator is not None:
            return self._orchestrator

        self._factory = WebScrapingFactory(
            config=self._config,
            config_path=self._config_path,
        )

        self._orchestrator = self._factory.create_orchestrator(
            enable_chunking=self._enable_chunking,
        )

        return self._orchestrator

    @component.output_types(documents=List[Document], errors=Dict[str, str])
    def run(
        self,
        urls: List[str],
        continue_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Scrape URLs and return documents.

        Args:
            urls: List of URLs to scrape.
            continue_on_error: Whether to continue if individual URLs fail.

        Returns:
            Dictionary with 'documents' and 'errors' keys.
        """
        orchestrator = self._get_orchestrator()

        all_documents: List[Document] = []
        errors: Dict[str, str] = {}

        logger.info("Scraping %d URLs", len(urls))

        for url in urls:
            try:
                documents = orchestrator.scrape(url)
                all_documents.extend(documents)
                logger.debug("Scraped %d documents from %s", len(documents), url)

            except Exception as error:
                error_msg = str(error)
                errors[url] = error_msg
                logger.warning("Failed to scrape %s: %s", url, error_msg)

                if not continue_on_error:
                    break

        logger.info(
            "Scraping completed: %d documents from %d URLs, %d errors",
            len(all_documents),
            len(urls),
            len(errors),
        )

        return {
            "documents": all_documents,
            "errors": errors,
        }

    def warm_up(self) -> None:
        """
        Warm up the component by initializing the orchestrator.

        Called by Haystack before first run in some scenarios.
        """
        self._get_orchestrator()
        logger.debug("Web scraper component warmed up")

"""
Web Scraping Providers Module.

This module provides scraping provider implementations.
"""

from src.web_scraping.scraping.providers.base import BaseScraperProvider
from src.web_scraping.scraping.providers.firecrawl import FirecrawlScraper

__all__ = [
    "BaseScraperProvider",
    "FirecrawlScraper",
]

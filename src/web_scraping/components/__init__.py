"""
Web Scraping Components Module.

This module provides factory and Haystack component wrapper for web scraping.
"""

from src.web_scraping.components.factory import WebScrapingFactory
from src.web_scraping.components.scraper_component import WebScraperComponent

__all__ = [
    "WebScrapingFactory",
    "WebScraperComponent",
]

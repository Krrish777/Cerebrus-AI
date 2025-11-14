"""
Web Scraping Module for Cerebrus AI

Provides web scraping capabilities using Firecrawl API
"""

# Always available - Firecrawl only scraper
from .firecrawl_only import SimpleWebScraper, create_scraper

# Try to import the advanced scraper (with extra dependencies)
try:
    from .web_scraper import WebScraper, create_web_scraper
    FULL_SCRAPER_AVAILABLE = True
except ImportError:
    WebScraper = None
    create_web_scraper = None
    FULL_SCRAPER_AVAILABLE = False

def create_scraper_auto(api_key=None):
    """Create best available scraper"""
    if FULL_SCRAPER_AVAILABLE:
        try:
            return create_web_scraper(api_key)
        except:
            pass
    return create_scraper(api_key)

__all__ = ['SimpleWebScraper', 'create_scraper', 'create_scraper_auto']

if FULL_SCRAPER_AVAILABLE:
    __all__.extend(['WebScraper', 'create_web_scraper'])
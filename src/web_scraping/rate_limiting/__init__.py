"""
Web Scraping Rate Limiting Module.

This module provides rate limiting for web scraping operations.
"""

from src.web_scraping.rate_limiting.limiter import TokenBucketRateLimiter

__all__ = [
    "TokenBucketRateLimiter",
]

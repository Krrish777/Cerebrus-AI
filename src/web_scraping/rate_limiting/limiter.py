"""
Token Bucket Rate Limiter Implementation.

This module provides a token bucket rate limiter for controlling
request rates to avoid overloading target servers.

Following AGENTS.md principles:
    - Single responsibility: Only rate limiting
    - Thread-safe: Uses threading locks
    - Configurable: All parameters from configuration
"""

import threading
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import Optional

from src.core.logging import get_logger
from src.web_scraping.config import RateLimitConfig
from src.web_scraping.interfaces import RateLimiter

logger = get_logger(__name__)


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.

    Attributes:
        tokens: Current number of available tokens.
        capacity: Maximum number of tokens.
        refill_rate: Tokens added per second.
        last_refill: Timestamp of last refill.
    """

    tokens: float
    capacity: int
    refill_rate: float
    last_refill: float = field(default_factory=time.time)


class TokenBucketRateLimiter(RateLimiter):
    """
    Token bucket rate limiter implementation.

    Uses the token bucket algorithm to control request rates.
    Each identifier (e.g., domain) has its own bucket.

    Example:
        config = RateLimitConfig(
            enabled=True,
            requests_per_minute=30,
            burst_size=10,
        )
        limiter = TokenBucketRateLimiter(config)
        if limiter.acquire("example.com"):
            scrape("https://example.com/page")
    """

    def __init__(self, config: RateLimitConfig) -> None:
        """
        Initialize the rate limiter.

        Args:
            config: Rate limiting configuration.
        """
        self._config = config
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

        # Calculate refill rate (tokens per second)
        self._refill_rate = config.requests_per_minute / 60.0

        logger.debug(
            "Rate limiter initialized: enabled=%s, rpm=%d, burst=%d",
            self._config.enabled,
            self._config.requests_per_minute,
            self._config.burst_size,
        )

    def acquire(self, identifier: str) -> bool:
        """
        Attempt to acquire a rate limit token.

        Args:
            identifier: Identifier for rate limiting (e.g., domain).

        Returns:
            True if allowed, False if rate limited.
        """
        if not self._config.enabled:
            return True

        with self._lock:
            bucket = self._get_or_create_bucket(identifier)
            self._refill_bucket(bucket)

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                logger.debug(
                    "Rate limit token acquired for %s (%.1f remaining)",
                    identifier,
                    bucket.tokens,
                )
                return True

            logger.debug(
                "Rate limit exceeded for %s (%.1f tokens available)",
                identifier,
                bucket.tokens,
            )
            return False

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
        if not self._config.enabled:
            return True

        start_time = time.time()

        while True:
            if self.acquire(identifier):
                return True

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    logger.debug(
                        "Rate limit acquire timed out for %s after %.1fs",
                        identifier,
                        elapsed,
                    )
                    return False

            # Wait for next token
            wait_time = self.time_until_available(identifier)
            wait_time = min(wait_time, 0.1)  # Poll at most every 100ms

            if timeout is not None:
                remaining = timeout - (time.time() - start_time)
                wait_time = min(wait_time, remaining)

            if wait_time > 0:
                time.sleep(wait_time)

    def time_until_available(self, identifier: str) -> float:
        """
        Get time until a token becomes available.

        Args:
            identifier: Identifier for rate limiting.

        Returns:
            Seconds until next token is available.
        """
        if not self._config.enabled:
            return 0.0

        with self._lock:
            bucket = self._get_or_create_bucket(identifier)
            self._refill_bucket(bucket)

            if bucket.tokens >= 1.0:
                return 0.0

            # Calculate time to get one token
            tokens_needed = 1.0 - bucket.tokens
            return tokens_needed / self._refill_rate

    def reset(self, identifier: str) -> None:
        """
        Reset rate limit for an identifier.

        Args:
            identifier: Identifier to reset.
        """
        with self._lock:
            if identifier in self._buckets:
                del self._buckets[identifier]
                logger.debug("Rate limit reset for %s", identifier)

    def reset_all(self) -> None:
        """Reset all rate limits."""
        with self._lock:
            self._buckets.clear()
            logger.debug("All rate limits reset")

    def _get_or_create_bucket(self, identifier: str) -> TokenBucket:
        """
        Get or create a token bucket for an identifier.

        Args:
            identifier: Identifier for the bucket.

        Returns:
            TokenBucket for the identifier.
        """
        if identifier not in self._buckets:
            self._buckets[identifier] = TokenBucket(
                tokens=float(self._config.burst_size),
                capacity=self._config.burst_size,
                refill_rate=self._refill_rate,
            )
            logger.debug(
                "Created token bucket for %s (capacity=%d)",
                identifier,
                self._config.burst_size,
            )

        return self._buckets[identifier]

    def _refill_bucket(self, bucket: TokenBucket) -> None:
        """
        Refill a token bucket based on elapsed time.

        Args:
            bucket: Bucket to refill.
        """
        current_time = time.time()
        elapsed = current_time - bucket.last_refill

        if elapsed > 0:
            # Add tokens based on elapsed time
            new_tokens = elapsed * bucket.refill_rate
            bucket.tokens = min(bucket.tokens + new_tokens, float(bucket.capacity))
            bucket.last_refill = current_time

    def get_bucket_status(self, identifier: str) -> Dict[str, float]:
        """
        Get the status of a token bucket.

        Args:
            identifier: Identifier for the bucket.

        Returns:
            Dictionary with bucket status.
        """
        with self._lock:
            bucket = self._get_or_create_bucket(identifier)
            self._refill_bucket(bucket)

            return {
                "tokens": bucket.tokens,
                "capacity": bucket.capacity,
                "refill_rate": bucket.refill_rate,
                "utilization": 1.0 - (bucket.tokens / bucket.capacity),
            }

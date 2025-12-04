"""
Web Scraping Exceptions Module.

This module defines a hierarchy of exceptions for web scraping operations.
Each exception includes relevant context for debugging and error handling.

Exception Hierarchy:
    WebScrapingError (base)
    ├── ConfigurationError
    ├── ScrapingProviderError
    │   └── ScrapingTimeoutError
    ├── URLValidationError
    ├── ContentValidationError
    ├── RateLimitExceededError
    └── CacheError
"""

from typing import Any
from typing import Optional


class WebScrapingError(Exception):
    """
    Base exception for all web scraping operations.

    Attributes:
        message: Human-readable error description.
        original_error: The underlying exception, if any.
    """

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error description.
            original_error: The underlying exception, if any.
        """
        self.message = message
        self.original_error = original_error
        super().__init__(message)


class ConfigurationError(WebScrapingError):
    """
    Raised when configuration is invalid or cannot be loaded.

    Attributes:
        config_key: The configuration key that caused the error.
        config_value: The invalid value, if applicable.
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the configuration error.

        Args:
            message: Human-readable error description.
            config_key: The configuration key that caused the error.
            config_value: The invalid value, if applicable.
            original_error: The underlying exception, if any.
        """
        self.config_key = config_key
        self.config_value = config_value

        detail_parts = [message]
        if config_key:
            detail_parts.append(f"key={config_key}")
        if config_value:
            detail_parts.append(f"value={config_value}")

        super().__init__(
            message="; ".join(detail_parts),
            original_error=original_error,
        )


class ScrapingProviderError(WebScrapingError):
    """
    Raised when a scraping provider fails to fetch content.

    Attributes:
        provider_name: Name of the provider that failed.
        url: The URL that was being scraped.
        status_code: HTTP status code, if applicable.
    """

    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the scraping provider error.

        Args:
            message: Human-readable error description.
            provider_name: Name of the provider that failed.
            url: The URL that was being scraped.
            status_code: HTTP status code, if applicable.
            original_error: The underlying exception, if any.
        """
        self.provider_name = provider_name
        self.url = url
        self.status_code = status_code
        super().__init__(message=message, original_error=original_error)


class ScrapingTimeoutError(ScrapingProviderError):
    """
    Raised when a scraping operation times out.

    Attributes:
        timeout_seconds: The timeout that was exceeded.
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[int] = None,
        provider_name: Optional[str] = None,
        url: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the timeout error.

        Args:
            message: Human-readable error description.
            timeout_seconds: The timeout that was exceeded.
            provider_name: Name of the provider that failed.
            url: The URL that was being scraped.
            original_error: The underlying exception, if any.
        """
        self.timeout_seconds = timeout_seconds
        super().__init__(
            message=message,
            provider_name=provider_name,
            url=url,
            original_error=original_error,
        )


class URLValidationError(WebScrapingError):
    """
    Raised when URL validation fails.

    Attributes:
        url: The URL that failed validation.
        validation_errors: List of specific validation failures.
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        validation_errors: Optional[list[str]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the URL validation error.

        Args:
            message: Human-readable error description.
            url: The URL that failed validation.
            validation_errors: List of specific validation failures.
            original_error: The underlying exception, if any.
        """
        self.url = url
        self.validation_errors = validation_errors or []
        super().__init__(message=message, original_error=original_error)


class ContentValidationError(WebScrapingError):
    """
    Raised when scraped content fails validation.

    Attributes:
        url: The URL that produced the invalid content.
        content_length: Length of the invalid content.
        validation_errors: List of specific validation failures.
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        content_length: Optional[int] = None,
        validation_errors: Optional[list[str]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the content validation error.

        Args:
            message: Human-readable error description.
            url: The URL that produced the invalid content.
            content_length: Length of the invalid content.
            validation_errors: List of specific validation failures.
            original_error: The underlying exception, if any.
        """
        self.url = url
        self.content_length = content_length
        self.validation_errors = validation_errors or []
        super().__init__(message=message, original_error=original_error)


class RateLimitExceededError(WebScrapingError):
    """
    Raised when rate limit is exceeded.

    Attributes:
        identifier: The rate-limited identifier (e.g., domain).
        retry_after_seconds: Suggested wait time before retry.
    """

    def __init__(
        self,
        message: str,
        identifier: Optional[str] = None,
        retry_after_seconds: Optional[float] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the rate limit error.

        Args:
            message: Human-readable error description.
            identifier: The rate-limited identifier (e.g., domain).
            retry_after_seconds: Suggested wait time before retry.
            original_error: The underlying exception, if any.
        """
        self.identifier = identifier
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message=message, original_error=original_error)


class CacheError(WebScrapingError):
    """
    Raised when a cache operation fails.

    Attributes:
        cache_key: The cache key involved in the error.
        operation: The cache operation that failed (get, set, delete).
    """

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the cache error.

        Args:
            message: Human-readable error description.
            cache_key: The cache key involved in the error.
            operation: The cache operation that failed.
            original_error: The underlying exception, if any.
        """
        self.cache_key = cache_key
        self.operation = operation
        super().__init__(message=message, original_error=original_error)

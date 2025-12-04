"""
URL Validator Implementation.

This module provides URL validation for web scraping operations.
Validates URLs against configured constraints before scraping.

Following AGENTS.md principles:
    - Single responsibility: Only URL validation
    - Dependency injection: Configuration injected via constructor
    - Fail-fast: Validates all constraints upfront
"""

import re
from typing import List
from typing import Optional
from typing import Tuple
from urllib.parse import urlparse

from src.core.logging import get_logger
from src.web_scraping.config import URLValidationConfig
from src.web_scraping.exceptions import URLValidationError
from src.web_scraping.interfaces import URLValidator

logger = get_logger(__name__)

# Common TLDs for validation
VALID_TLDS = frozenset({
    "com", "org", "net", "edu", "gov", "mil", "io", "co", "ai", "app",
    "dev", "xyz", "info", "biz", "me", "us", "uk", "ca", "au", "de",
    "fr", "jp", "cn", "in", "br", "ru", "es", "it", "nl", "se", "no",
    "fi", "dk", "pl", "cz", "at", "ch", "be", "pt", "ie", "nz", "za",
    "mx", "ar", "cl", "kr", "tw", "hk", "sg", "my", "th", "vn", "id",
    "ph", "il", "ae", "sa", "eg", "ng", "ke", "gh", "tz", "ug", "zm",
})


class DefaultURLValidator(URLValidator):
    """
    Default implementation of URL validation.

    Validates URLs against configured constraints:
    - Scheme validation (http/https)
    - Domain blocking
    - URL length limits
    - TLD validation

    Example:
        config = URLValidationConfig(
            allowed_schemes=("http", "https"),
            blocked_domains=("example.com",),
        )
        validator = DefaultURLValidator(config)
        is_valid, errors = validator.validate_with_errors("https://test.com")
    """

    def __init__(self, config: URLValidationConfig) -> None:
        """
        Initialize the URL validator.

        Args:
            config: URL validation configuration.
        """
        self._config = config
        logger.debug(
            "URL validator initialized with allowed_schemes=%s, blocked_domains=%d",
            self._config.allowed_schemes,
            len(self._config.blocked_domains),
        )

    def validate(self, url: str) -> bool:
        """
        Validate if URL is acceptable for scraping.

        Args:
            url: URL to validate.

        Returns:
            True if valid, False otherwise.
        """
        is_valid, _ = self.validate_with_errors(url)
        return is_valid

    def validate_with_errors(self, url: str) -> Tuple[bool, List[str]]:
        """
        Validate URL and return validation errors.

        Args:
            url: URL to validate.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """
        errors: List[str] = []

        # Check if URL is provided
        if not url:
            errors.append("URL cannot be empty")
            return False, errors

        # Check URL length
        if len(url) > self._config.max_url_length:
            errors.append(
                f"URL exceeds maximum length of {self._config.max_url_length} characters"
            )

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as error:
            errors.append(f"Failed to parse URL: {error}")
            return False, errors

        # Validate scheme
        if not parsed.scheme:
            errors.append("URL must include a scheme (http or https)")
        elif parsed.scheme.lower() not in self._config.allowed_schemes:
            errors.append(
                f"URL scheme '{parsed.scheme}' not allowed. "
                f"Allowed schemes: {', '.join(self._config.allowed_schemes)}"
            )

        # Validate host
        if not parsed.netloc:
            errors.append("URL must include a host")
        else:
            # Check blocked domains
            domain = self._extract_domain_from_netloc(parsed.netloc)
            if self._is_blocked_domain(domain):
                errors.append(f"Domain '{domain}' is blocked")

            # Validate TLD if required
            if self._config.require_valid_tld:
                tld = self._extract_tld(domain)
                if tld and tld.lower() not in VALID_TLDS:
                    errors.append(f"Invalid TLD: '{tld}'")

        is_valid = len(errors) == 0

        if not is_valid:
            logger.debug("URL validation failed for %s: %s", url, errors)
        else:
            logger.debug("URL validation passed for %s", url)

        return is_valid, errors

    def normalize_url(self, url: str) -> str:
        """
        Normalize a URL to a standard format.

        Normalization includes:
        - Lowercasing scheme and host
        - Removing default ports
        - Removing trailing slashes from path

        Args:
            url: URL to normalize.

        Returns:
            Normalized URL.

        Raises:
            URLValidationError: If the URL cannot be normalized.
        """
        if not url:
            raise URLValidationError(
                message="Cannot normalize empty URL",
                url=url,
            )

        try:
            parsed = urlparse(url)
        except Exception as error:
            raise URLValidationError(
                message=f"Failed to parse URL for normalization: {error}",
                url=url,
                original_error=error,
            ) from error

        # Normalize scheme
        scheme = parsed.scheme.lower() if parsed.scheme else "https"

        # Normalize host
        host = parsed.netloc.lower() if parsed.netloc else ""

        # Remove default ports
        if host.endswith(":80") and scheme == "http":
            host = host[:-3]
        elif host.endswith(":443") and scheme == "https":
            host = host[:-4]

        # Normalize path
        path = parsed.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        # Reconstruct URL
        normalized = f"{scheme}://{host}{path}"

        # Add query string if present
        if parsed.query:
            normalized += f"?{parsed.query}"

        # Add fragment if present
        if parsed.fragment:
            normalized += f"#{parsed.fragment}"

        logger.debug("Normalized URL: %s -> %s", url, normalized)
        return normalized

    def extract_domain(self, url: str) -> str:
        """
        Extract the domain from a URL.

        Args:
            url: URL to extract domain from.

        Returns:
            Domain string (e.g., "example.com").
        """
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc or ""
            return self._extract_domain_from_netloc(netloc)
        except Exception:
            return ""

    def _extract_domain_from_netloc(self, netloc: str) -> str:
        """
        Extract domain from netloc, removing port and credentials.

        Args:
            netloc: Network location string.

        Returns:
            Domain string.
        """
        # Remove credentials (user:pass@)
        if "@" in netloc:
            netloc = netloc.split("@")[-1]

        # Remove port
        if ":" in netloc:
            netloc = netloc.split(":")[0]

        return netloc.lower()

    def _extract_tld(self, domain: str) -> Optional[str]:
        """
        Extract TLD from domain.

        Args:
            domain: Domain string.

        Returns:
            TLD string or None.
        """
        if not domain:
            return None

        parts = domain.split(".")
        if len(parts) >= 2:
            return parts[-1]
        return None

    def _is_blocked_domain(self, domain: str) -> bool:
        """
        Check if a domain is blocked.

        Args:
            domain: Domain to check.

        Returns:
            True if domain is blocked.
        """
        domain_lower = domain.lower()

        for blocked in self._config.blocked_domains:
            blocked_lower = blocked.lower()

            # Exact match
            if domain_lower == blocked_lower:
                return True

            # Subdomain match (e.g., "sub.blocked.com" matches "blocked.com")
            if domain_lower.endswith(f".{blocked_lower}"):
                return True

        return False

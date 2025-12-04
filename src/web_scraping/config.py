"""
Web Scraping Configuration Module.

This module provides configuration dataclasses for web scraping operations.
Configuration is loaded from YAML files and validated at load time.

Following AGENTS.md principles:
    - Frozen dataclasses for immutability
    - Validation in __post_init__ methods
    - No hard-coded values
    - Clear separation of concerns

Example usage:
    config = WebScrapingConfig.from_yaml(Path("config/web_scraping_config.yml"))
    scraper = FirecrawlScraper(config.providers.firecrawl)
"""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import yaml

from src.core.logging import get_logger
from src.web_scraping.exceptions import ConfigurationError

logger = get_logger(__name__)


@dataclass(frozen=True)
class ScrapingConfig:
    """
    Core scraping operation settings.

    Attributes:
        default_provider: Default scraping provider to use.
        timeout_seconds: Request timeout in seconds.
        retry_attempts: Number of retry attempts on failure.
        retry_delay_seconds: Initial delay between retries in seconds.
        user_agent: User agent string for requests.
    """

    default_provider: str = "firecrawl"
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: int = 2
    user_agent: str = "CerebrusAI/1.0"

    def __post_init__(self) -> None:
        """Validate scraping configuration."""
        if self.timeout_seconds <= 0:
            raise ConfigurationError(
                message="timeout_seconds must be positive",
                config_key="timeout_seconds",
                config_value=str(self.timeout_seconds),
            )
        if self.retry_attempts < 0:
            raise ConfigurationError(
                message="retry_attempts cannot be negative",
                config_key="retry_attempts",
                config_value=str(self.retry_attempts),
            )
        if self.retry_delay_seconds < 0:
            raise ConfigurationError(
                message="retry_delay_seconds cannot be negative",
                config_key="retry_delay_seconds",
                config_value=str(self.retry_delay_seconds),
            )


@dataclass(frozen=True)
class FirecrawlProviderConfig:
    """
    Firecrawl API provider settings.

    Attributes:
        api_key_env: Environment variable name for API key.
        formats: Output formats to request.
        wait_for_content: Whether to wait for dynamic content.
        include_links: Whether to extract links from pages.
    """

    api_key_env: str = "FIRECRAWL_API_KEY"
    formats: Tuple[str, ...] = ("markdown",)
    wait_for_content: bool = True
    include_links: bool = True

    def __post_init__(self) -> None:
        """Validate provider configuration."""
        if not self.api_key_env:
            raise ConfigurationError(
                message="api_key_env cannot be empty",
                config_key="api_key_env",
            )
        if not self.formats:
            raise ConfigurationError(
                message="formats cannot be empty",
                config_key="formats",
            )


@dataclass(frozen=True)
class URLValidationConfig:
    """
    URL validation settings.

    Attributes:
        allowed_schemes: List of allowed URL schemes.
        blocked_domains: List of domains to block.
        max_url_length: Maximum URL length in characters.
        require_valid_tld: Whether to require valid TLD.
    """

    allowed_schemes: Tuple[str, ...] = ("http", "https")
    blocked_domains: Tuple[str, ...] = ()
    max_url_length: int = 2048
    require_valid_tld: bool = True

    def __post_init__(self) -> None:
        """Validate URL validation configuration."""
        if not self.allowed_schemes:
            raise ConfigurationError(
                message="allowed_schemes cannot be empty",
                config_key="allowed_schemes",
            )
        if self.max_url_length <= 0:
            raise ConfigurationError(
                message="max_url_length must be positive",
                config_key="max_url_length",
                config_value=str(self.max_url_length),
            )


@dataclass(frozen=True)
class ContentValidationConfig:
    """
    Content quality validation settings.

    Attributes:
        min_content_length: Minimum content length in characters.
        max_content_length: Maximum content length in characters.
        min_word_count: Minimum word count.
        check_content_quality: Whether to perform quality checks.
    """

    min_content_length: int = 100
    max_content_length: int = 1000000
    min_word_count: int = 10
    check_content_quality: bool = True

    def __post_init__(self) -> None:
        """Validate content validation configuration."""
        if self.min_content_length < 0:
            raise ConfigurationError(
                message="min_content_length cannot be negative",
                config_key="min_content_length",
                config_value=str(self.min_content_length),
            )
        if self.max_content_length <= self.min_content_length:
            raise ConfigurationError(
                message="max_content_length must be greater than min_content_length",
                config_key="max_content_length",
                config_value=str(self.max_content_length),
            )
        if self.min_word_count < 0:
            raise ConfigurationError(
                message="min_word_count cannot be negative",
                config_key="min_word_count",
                config_value=str(self.min_word_count),
            )


@dataclass(frozen=True)
class RateLimitConfig:
    """
    Rate limiting settings using token bucket algorithm.

    Attributes:
        enabled: Whether rate limiting is enabled.
        requests_per_minute: Maximum requests per minute.
        burst_size: Maximum burst size for token bucket.
    """

    enabled: bool = True
    requests_per_minute: int = 30
    burst_size: int = 10

    def __post_init__(self) -> None:
        """Validate rate limit configuration."""
        if self.requests_per_minute <= 0:
            raise ConfigurationError(
                message="requests_per_minute must be positive",
                config_key="requests_per_minute",
                config_value=str(self.requests_per_minute),
            )
        if self.burst_size <= 0:
            raise ConfigurationError(
                message="burst_size must be positive",
                config_key="burst_size",
                config_value=str(self.burst_size),
            )


@dataclass(frozen=True)
class CacheConfig:
    """
    Cache settings for scraped content.

    Attributes:
        enabled: Whether caching is enabled.
        cache_dir: Directory for cache storage.
        ttl_hours: Cache time-to-live in hours.
        max_cache_size_gb: Maximum cache size in gigabytes.
    """

    enabled: bool = True
    cache_dir: Optional[Path] = None
    ttl_hours: int = 24
    max_cache_size_gb: int = 5

    def __post_init__(self) -> None:
        """Validate cache configuration."""
        if self.enabled and self.cache_dir is None:
            raise ConfigurationError(
                message="cache_dir is required when caching is enabled",
                config_key="cache_dir",
            )
        if self.ttl_hours <= 0:
            raise ConfigurationError(
                message="ttl_hours must be positive",
                config_key="ttl_hours",
                config_value=str(self.ttl_hours),
            )
        if self.max_cache_size_gb <= 0:
            raise ConfigurationError(
                message="max_cache_size_gb must be positive",
                config_key="max_cache_size_gb",
                config_value=str(self.max_cache_size_gb),
            )


@dataclass(frozen=True)
class ProcessingConfig:
    """
    Processing settings for scraped content.

    Attributes:
        use_existing_chunking: Whether to use document_processing ChunkingService.
        document_config_path: Path to document processing configuration.
        extract_metadata: Whether to extract page metadata.
        extract_links: Whether to extract links from pages.
        clean_html: Whether to clean HTML content.
        remove_scripts: Whether to remove script tags.
    """

    use_existing_chunking: bool = True
    document_config_path: Optional[Path] = None
    extract_metadata: bool = True
    extract_links: bool = True
    clean_html: bool = True
    remove_scripts: bool = True


@dataclass(frozen=True)
class WebScrapingConfig:
    """
    Main configuration class for web scraping operations.

    This class aggregates all sub-configurations and provides factory methods
    for loading from YAML files.

    Attributes:
        scraping: Core scraping settings.
        firecrawl: Firecrawl provider settings.
        url_validation: URL validation settings.
        content_validation: Content validation settings.
        rate_limiting: Rate limiting settings.
        cache: Cache settings.
        processing: Processing settings.
    """

    scraping: ScrapingConfig
    firecrawl: FirecrawlProviderConfig
    url_validation: URLValidationConfig
    content_validation: ContentValidationConfig
    rate_limiting: RateLimitConfig
    cache: CacheConfig
    processing: ProcessingConfig

    @classmethod
    def from_yaml(cls, config_path: Path) -> "WebScrapingConfig":
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            WebScrapingConfig instance populated from the YAML file.

        Raises:
            ConfigurationError: If the file cannot be read or parsed.
            FileNotFoundError: If the configuration file does not exist.
        """
        if not config_path.exists():
            logger.error("Configuration file not found: %s", config_path)
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info("Loading web scraping configuration from %s", config_path)

        try:
            with config_path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ConfigurationError(
                message=f"Failed to parse YAML configuration: {error}",
                original_error=error,
            ) from error

        return cls._from_dict(data, config_path.parent)

    @classmethod
    def _from_dict(cls, data: dict, base_path: Path) -> "WebScrapingConfig":
        """
        Create configuration from a dictionary.

        Args:
            data: Dictionary containing configuration data.
            base_path: Base path for resolving relative paths.

        Returns:
            WebScrapingConfig instance.
        """
        ws_data = data.get("web_scraping", {})

        # Parse scraping config
        scraping_data = ws_data.get("scraping", {})
        scraping = ScrapingConfig(
            default_provider=scraping_data.get("default_provider", "firecrawl"),
            timeout_seconds=scraping_data.get("timeout", 30),
            retry_attempts=scraping_data.get("retry_attempts", 3),
            retry_delay_seconds=scraping_data.get("retry_delay", 2),
            user_agent=scraping_data.get("user_agent", "CerebrusAI/1.0"),
        )

        # Parse Firecrawl provider config
        providers_data = ws_data.get("providers", {})
        firecrawl_data = providers_data.get("firecrawl", {})
        firecrawl = FirecrawlProviderConfig(
            api_key_env=firecrawl_data.get("api_key_env", "FIRECRAWL_API_KEY"),
            formats=tuple(firecrawl_data.get("formats", ["markdown"])),
            wait_for_content=firecrawl_data.get("wait_for_content", True),
            include_links=firecrawl_data.get("include_links", True),
        )

        # Parse URL validation config
        validation_data = ws_data.get("validation", {})
        url_validation = URLValidationConfig(
            allowed_schemes=tuple(validation_data.get("allowed_schemes", ["http", "https"])),
            blocked_domains=tuple(validation_data.get("blocked_domains", [])),
            max_url_length=validation_data.get("max_url_length", 2048),
            require_valid_tld=validation_data.get("require_valid_tld", True),
        )

        # Parse content validation config
        content_val_data = ws_data.get("content_validation", {})
        content_validation = ContentValidationConfig(
            min_content_length=content_val_data.get("min_content_length", 100),
            max_content_length=content_val_data.get("max_content_length", 1000000),
            min_word_count=content_val_data.get("min_word_count", 10),
            check_content_quality=content_val_data.get("check_content_quality", True),
        )

        # Parse rate limit config
        rate_limit_data = ws_data.get("rate_limiting", {})
        rate_limiting = RateLimitConfig(
            enabled=rate_limit_data.get("enabled", True),
            requests_per_minute=rate_limit_data.get("requests_per_minute", 30),
            burst_size=rate_limit_data.get("burst_size", 10),
        )

        # Parse cache config
        cache_data = ws_data.get("cache", {})
        cache_enabled = cache_data.get("enabled", True)
        cache_dir_str = cache_data.get("cache_dir", "cache/web_scraping")
        cache = CacheConfig(
            enabled=cache_enabled,
            cache_dir=base_path / cache_dir_str if cache_enabled else None,
            ttl_hours=cache_data.get("ttl_hours", 24),
            max_cache_size_gb=cache_data.get("max_cache_size_gb", 5),
        )

        # Parse processing config
        processing_data = ws_data.get("processing", {})
        doc_config_path_str = processing_data.get("document_config_path")
        processing = ProcessingConfig(
            use_existing_chunking=processing_data.get("use_existing_chunking", True),
            document_config_path=base_path / doc_config_path_str if doc_config_path_str else None,
            extract_metadata=processing_data.get("extract_metadata", True),
            extract_links=processing_data.get("extract_links", True),
            clean_html=processing_data.get("clean_html", True),
            remove_scripts=processing_data.get("remove_scripts", True),
        )

        return cls(
            scraping=scraping,
            firecrawl=firecrawl,
            url_validation=url_validation,
            content_validation=content_validation,
            rate_limiting=rate_limiting,
            cache=cache,
            processing=processing,
        )

    @classmethod
    def create_default(cls, cache_dir: Optional[Path] = None) -> "WebScrapingConfig":
        """
        Create a configuration with default values.

        Args:
            cache_dir: Optional directory for cache storage.

        Returns:
            WebScrapingConfig instance with default settings.
        """
        return cls(
            scraping=ScrapingConfig(),
            firecrawl=FirecrawlProviderConfig(),
            url_validation=URLValidationConfig(),
            content_validation=ContentValidationConfig(),
            rate_limiting=RateLimitConfig(),
            cache=CacheConfig(
                enabled=cache_dir is not None,
                cache_dir=cache_dir,
            ) if cache_dir else CacheConfig(enabled=False, cache_dir=None),
            processing=ProcessingConfig(),
        )

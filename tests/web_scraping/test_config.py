"""
Tests for Web Scraping Configuration.

Tests the configuration dataclasses, validation, and YAML loading.
"""

from pathlib import Path
from typing import Generator

import pytest
import yaml

from src.web_scraping.config import CacheConfig
from src.web_scraping.config import ContentValidationConfig
from src.web_scraping.config import FirecrawlProviderConfig
from src.web_scraping.config import ProcessingConfig
from src.web_scraping.config import RateLimitConfig
from src.web_scraping.config import ScrapingConfig
from src.web_scraping.config import URLValidationConfig
from src.web_scraping.config import WebScrapingConfig
from src.web_scraping.exceptions import ConfigurationError


class TestScrapingConfig:
    """Tests for ScrapingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ScrapingConfig()
        
        assert config.default_provider == "firecrawl"
        assert config.timeout_seconds == 30
        assert config.retry_attempts == 3
        assert config.retry_delay_seconds == 2
        assert config.user_agent == "CerebrusAI/1.0"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ScrapingConfig(
            default_provider="playwright",
            timeout_seconds=60,
            retry_attempts=5,
            retry_delay_seconds=10,
            user_agent="CustomAgent/1.0",
        )
        
        assert config.default_provider == "playwright"
        assert config.timeout_seconds == 60
        assert config.retry_attempts == 5
        assert config.retry_delay_seconds == 10
        assert config.user_agent == "CustomAgent/1.0"

    def test_invalid_timeout_raises_error(self) -> None:
        """Test that invalid timeout raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            ScrapingConfig(timeout_seconds=0)
        
        assert "timeout_seconds must be positive" in str(exc_info.value)

    def test_negative_retry_attempts_raises_error(self) -> None:
        """Test that negative retry attempts raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            ScrapingConfig(retry_attempts=-1)
        
        assert "retry_attempts cannot be negative" in str(exc_info.value)

    def test_negative_retry_delay_raises_error(self) -> None:
        """Test that negative retry delay raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            ScrapingConfig(retry_delay_seconds=-1)
        
        assert "retry_delay_seconds cannot be negative" in str(exc_info.value)

    def test_frozen_dataclass(self) -> None:
        """Test that config is immutable."""
        config = ScrapingConfig()
        
        with pytest.raises(AttributeError):
            config.timeout_seconds = 100


class TestFirecrawlProviderConfig:
    """Tests for FirecrawlProviderConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = FirecrawlProviderConfig()
        
        assert config.api_key_env == "FIRECRAWL_API_KEY"
        assert config.formats == ("markdown",)
        assert config.wait_for_content is True
        assert config.include_links is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = FirecrawlProviderConfig(
            api_key_env="CUSTOM_API_KEY",
            formats=("html", "markdown"),
            wait_for_content=False,
            include_links=False,
        )
        
        assert config.api_key_env == "CUSTOM_API_KEY"
        assert config.formats == ("html", "markdown")
        assert config.wait_for_content is False
        assert config.include_links is False

    def test_empty_api_key_env_raises_error(self) -> None:
        """Test that empty api_key_env raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            FirecrawlProviderConfig(api_key_env="")
        
        assert "api_key_env cannot be empty" in str(exc_info.value)

    def test_empty_formats_raises_error(self) -> None:
        """Test that empty formats raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            FirecrawlProviderConfig(formats=())
        
        assert "formats cannot be empty" in str(exc_info.value)


class TestURLValidationConfig:
    """Tests for URLValidationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = URLValidationConfig()
        
        assert config.allowed_schemes == ("http", "https")
        assert config.blocked_domains == ()
        assert config.max_url_length == 2048
        assert config.require_valid_tld is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = URLValidationConfig(
            allowed_schemes=("https",),
            blocked_domains=("blocked.com", "spam.net"),
            max_url_length=1024,
            require_valid_tld=False,
        )
        
        assert config.allowed_schemes == ("https",)
        assert config.blocked_domains == ("blocked.com", "spam.net")
        assert config.max_url_length == 1024
        assert config.require_valid_tld is False

    def test_empty_allowed_schemes_raises_error(self) -> None:
        """Test that empty allowed_schemes raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            URLValidationConfig(allowed_schemes=())
        
        assert "allowed_schemes cannot be empty" in str(exc_info.value)

    def test_invalid_max_url_length_raises_error(self) -> None:
        """Test that invalid max_url_length raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            URLValidationConfig(max_url_length=0)
        
        assert "max_url_length must be positive" in str(exc_info.value)


class TestContentValidationConfig:
    """Tests for ContentValidationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ContentValidationConfig()
        
        assert config.min_content_length == 100
        assert config.max_content_length == 1000000
        assert config.min_word_count == 10
        assert config.check_content_quality is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ContentValidationConfig(
            min_content_length=50,
            max_content_length=500000,
            min_word_count=5,
            check_content_quality=False,
        )
        
        assert config.min_content_length == 50
        assert config.max_content_length == 500000
        assert config.min_word_count == 5
        assert config.check_content_quality is False

    def test_negative_min_content_length_raises_error(self) -> None:
        """Test that negative min_content_length raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            ContentValidationConfig(min_content_length=-1)
        
        assert "min_content_length cannot be negative" in str(exc_info.value)

    def test_max_less_than_min_raises_error(self) -> None:
        """Test that max < min raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            ContentValidationConfig(
                min_content_length=1000,
                max_content_length=500,
            )
        
        assert "max_content_length must be greater than min_content_length" in str(exc_info.value)

    def test_negative_word_count_raises_error(self) -> None:
        """Test that negative min_word_count raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            ContentValidationConfig(min_word_count=-1)
        
        assert "min_word_count cannot be negative" in str(exc_info.value)


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RateLimitConfig()
        
        assert config.enabled is True
        assert config.requests_per_minute == 30
        assert config.burst_size == 10

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = RateLimitConfig(
            enabled=False,
            requests_per_minute=60,
            burst_size=20,
        )
        
        assert config.enabled is False
        assert config.requests_per_minute == 60
        assert config.burst_size == 20

    def test_invalid_requests_per_minute_raises_error(self) -> None:
        """Test that invalid requests_per_minute raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            RateLimitConfig(requests_per_minute=0)
        
        assert "requests_per_minute must be positive" in str(exc_info.value)

    def test_invalid_burst_size_raises_error(self) -> None:
        """Test that invalid burst_size raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            RateLimitConfig(burst_size=0)
        
        assert "burst_size must be positive" in str(exc_info.value)


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values_disabled(self) -> None:
        """Test default configuration with cache disabled."""
        config = CacheConfig(enabled=False, cache_dir=None)
        
        assert config.enabled is False
        assert config.cache_dir is None
        assert config.ttl_hours == 24
        assert config.max_cache_size_gb == 5

    def test_enabled_requires_cache_dir(self) -> None:
        """Test that enabled cache requires cache_dir."""
        with pytest.raises(ConfigurationError) as exc_info:
            CacheConfig(enabled=True, cache_dir=None)
        
        assert "cache_dir is required when caching is enabled" in str(exc_info.value)

    def test_enabled_with_cache_dir(self) -> None:
        """Test enabled cache with cache_dir."""
        config = CacheConfig(
            enabled=True,
            cache_dir=Path("/tmp/cache"),
            ttl_hours=48,
            max_cache_size_gb=10,
        )
        
        assert config.enabled is True
        assert config.cache_dir == Path("/tmp/cache")
        assert config.ttl_hours == 48
        assert config.max_cache_size_gb == 10

    def test_invalid_ttl_hours_raises_error(self) -> None:
        """Test that invalid ttl_hours raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            CacheConfig(enabled=False, cache_dir=None, ttl_hours=0)
        
        assert "ttl_hours must be positive" in str(exc_info.value)

    def test_invalid_max_cache_size_raises_error(self) -> None:
        """Test that invalid max_cache_size_gb raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            CacheConfig(enabled=False, cache_dir=None, max_cache_size_gb=0)
        
        assert "max_cache_size_gb must be positive" in str(exc_info.value)


class TestProcessingConfig:
    """Tests for ProcessingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ProcessingConfig()
        
        assert config.use_existing_chunking is True
        assert config.document_config_path is None
        assert config.extract_metadata is True
        assert config.extract_links is True
        assert config.clean_html is True
        assert config.remove_scripts is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ProcessingConfig(
            use_existing_chunking=False,
            document_config_path=Path("/config/doc.yaml"),
            extract_metadata=False,
            extract_links=False,
            clean_html=False,
            remove_scripts=False,
        )
        
        assert config.use_existing_chunking is False
        assert config.document_config_path == Path("/config/doc.yaml")
        assert config.extract_metadata is False
        assert config.extract_links is False
        assert config.clean_html is False
        assert config.remove_scripts is False


class TestWebScrapingConfig:
    """Tests for WebScrapingConfig main configuration class."""

    @pytest.fixture
    def sample_config(self) -> WebScrapingConfig:
        """Create a sample configuration for testing."""
        return WebScrapingConfig(
            scraping=ScrapingConfig(),
            firecrawl=FirecrawlProviderConfig(),
            url_validation=URLValidationConfig(),
            content_validation=ContentValidationConfig(),
            rate_limiting=RateLimitConfig(),
            cache=CacheConfig(enabled=False, cache_dir=None),
            processing=ProcessingConfig(),
        )

    def test_create_config(self, sample_config: WebScrapingConfig) -> None:
        """Test creating a configuration instance."""
        assert sample_config.scraping.default_provider == "firecrawl"
        assert sample_config.firecrawl.api_key_env == "FIRECRAWL_API_KEY"
        assert sample_config.url_validation.max_url_length == 2048
        assert sample_config.content_validation.min_content_length == 100
        assert sample_config.rate_limiting.requests_per_minute == 30
        assert sample_config.cache.enabled is False
        assert sample_config.processing.use_existing_chunking is True

    def test_create_default(self) -> None:
        """Test creating default configuration."""
        config = WebScrapingConfig.create_default()
        
        assert config.scraping.default_provider == "firecrawl"
        assert config.cache.enabled is False

    def test_create_default_with_cache(self, tmp_path: Path) -> None:
        """Test creating default configuration with cache."""
        config = WebScrapingConfig.create_default(cache_dir=tmp_path)
        
        assert config.cache.enabled is True
        assert config.cache.cache_dir == tmp_path

    def test_from_yaml(self, tmp_path: Path) -> None:
        """Test loading configuration from YAML file."""
        config_data = {
            "web_scraping": {
                "scraping": {
                    "default_provider": "firecrawl",
                    "timeout": 60,
                    "retry_attempts": 5,
                },
                "providers": {
                    "firecrawl": {
                        "api_key_env": "TEST_API_KEY",
                        "formats": ["markdown", "html"],
                    },
                },
                "validation": {
                    "allowed_schemes": ["https"],
                    "max_url_length": 4096,
                },
                "content_validation": {
                    "min_content_length": 50,
                    "max_content_length": 500000,
                },
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 60,
                },
                "cache": {
                    "enabled": True,
                    "cache_dir": "cache",
                    "ttl_hours": 48,
                },
                "processing": {
                    "use_existing_chunking": True,
                },
            },
        }
        
        config_path = tmp_path / "config.yml"
        with config_path.open("w") as f:
            yaml.dump(config_data, f)
        
        config = WebScrapingConfig.from_yaml(config_path)
        
        assert config.scraping.timeout_seconds == 60
        assert config.scraping.retry_attempts == 5
        assert config.firecrawl.api_key_env == "TEST_API_KEY"
        assert config.firecrawl.formats == ("markdown", "html")
        assert config.url_validation.allowed_schemes == ("https",)
        assert config.url_validation.max_url_length == 4096
        assert config.content_validation.min_content_length == 50
        assert config.rate_limiting.requests_per_minute == 60
        assert config.cache.enabled is True
        assert config.cache.ttl_hours == 48

    def test_from_yaml_file_not_found(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            WebScrapingConfig.from_yaml(Path("/nonexistent/config.yml"))

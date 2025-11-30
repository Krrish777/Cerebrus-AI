"""
Tests for YouTube Processing Configuration.

This module tests configuration loading, validation, and dataclasses.
"""

from pathlib import Path
from typing import Generator

import pytest
import yaml

from src.youtube_processing.config import CacheConfig
from src.youtube_processing.config import DownloadConfig
from src.youtube_processing.config import MetadataConfig
from src.youtube_processing.config import RetryConfig
from src.youtube_processing.config import ValidationConfig
from src.youtube_processing.config import YouTubeConfig
from src.youtube_processing.exceptions import ConfigurationError


class TestDownloadConfig:
    """Tests for DownloadConfig dataclass."""

    def test_download_config_creation(self, tmp_path: Path) -> None:
        """Test creating DownloadConfig with valid values."""
        config = DownloadConfig(
            temp_dir=tmp_path,
            audio_quality="best",
            audio_format="m4a",
            audio_bitrate="192",
            max_file_size_mb=500,
            timeout_seconds=300,
        )

        assert config.temp_dir == tmp_path
        assert config.audio_quality == "best"
        assert config.audio_format == "m4a"
        assert config.audio_bitrate == "192"
        assert config.max_file_size_mb == 500
        assert config.timeout_seconds == 300

    def test_download_config_defaults(self, tmp_path: Path) -> None:
        """Test DownloadConfig with default values."""
        config = DownloadConfig(temp_dir=tmp_path)

        assert config.audio_quality == "best"
        assert config.audio_format == "m4a"
        assert config.audio_bitrate == "192"
        assert config.max_file_size_mb == 500
        assert config.timeout_seconds == 300

    def test_download_config_invalid_max_file_size(self, tmp_path: Path) -> None:
        """Test that invalid max_file_size_mb raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            DownloadConfig(temp_dir=tmp_path, max_file_size_mb=0)

        assert "max_file_size_mb must be positive" in str(exc_info.value)

    def test_download_config_invalid_timeout(self, tmp_path: Path) -> None:
        """Test that invalid timeout_seconds raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            DownloadConfig(temp_dir=tmp_path, timeout_seconds=-1)

        assert "timeout_seconds must be positive" in str(exc_info.value)


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_cache_config_enabled(self, tmp_path: Path) -> None:
        """Test CacheConfig when caching is enabled."""
        config = CacheConfig(
            enabled=True,
            cache_dir=tmp_path,
            cleanup_after_processing=False,
            max_cache_size_gb=10,
            cache_ttl_days=7,
        )

        assert config.enabled is True
        assert config.cache_dir == tmp_path
        assert config.cleanup_after_processing is False
        assert config.max_cache_size_gb == 10
        assert config.cache_ttl_days == 7

    def test_cache_config_disabled(self) -> None:
        """Test CacheConfig when caching is disabled."""
        config = CacheConfig(enabled=False)

        assert config.enabled is False
        assert config.cache_dir is None

    def test_cache_config_enabled_without_dir_raises_error(self) -> None:
        """Test that enabling cache without cache_dir raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            CacheConfig(enabled=True, cache_dir=None)

        assert "cache_dir is required when caching is enabled" in str(exc_info.value)

    def test_cache_config_invalid_max_size(self, tmp_path: Path) -> None:
        """Test that invalid max_cache_size_gb raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            CacheConfig(enabled=True, cache_dir=tmp_path, max_cache_size_gb=0)

        assert "max_cache_size_gb must be positive" in str(exc_info.value)


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""

    def test_validation_config_defaults(self) -> None:
        """Test ValidationConfig with default values."""
        config = ValidationConfig()

        assert "youtube.com" in config.allowed_domains
        assert "youtu.be" in config.allowed_domains
        assert config.min_duration_seconds == 5
        assert config.max_duration_seconds == 7200
        assert config.allow_live_streams is False
        assert config.allow_age_restricted is False

    def test_validation_config_custom_domains(self) -> None:
        """Test ValidationConfig with custom allowed domains."""
        config = ValidationConfig(
            allowed_domains=["custom.youtube.com"],
        )

        assert config.allowed_domains == ["custom.youtube.com"]

    def test_validation_config_invalid_duration_range(self) -> None:
        """Test that invalid duration range raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            ValidationConfig(
                min_duration_seconds=1000,
                max_duration_seconds=500,
            )

        assert "max_duration_seconds must be greater than min_duration_seconds" in str(
            exc_info.value
        )

    def test_validation_config_negative_min_duration(self) -> None:
        """Test that negative min_duration raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            ValidationConfig(min_duration_seconds=-1)

        assert "min_duration_seconds cannot be negative" in str(exc_info.value)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_retry_config_defaults(self) -> None:
        """Test RetryConfig with default values."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.delay_seconds == 5
        assert config.exponential_backoff is True

    def test_retry_config_custom_values(self) -> None:
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_attempts=5,
            delay_seconds=10,
            exponential_backoff=False,
        )

        assert config.max_attempts == 5
        assert config.delay_seconds == 10
        assert config.exponential_backoff is False

    def test_retry_config_invalid_max_attempts(self) -> None:
        """Test that max_attempts < 1 raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            RetryConfig(max_attempts=0)

        assert "max_attempts must be at least 1" in str(exc_info.value)

    def test_retry_config_negative_delay(self) -> None:
        """Test that negative delay raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            RetryConfig(delay_seconds=-1)

        assert "delay_seconds cannot be negative" in str(exc_info.value)


class TestYouTubeConfig:
    """Tests for YouTubeConfig dataclass."""

    @pytest.fixture
    def sample_config_data(self) -> dict:
        """Sample configuration data for testing."""
        return {
            "youtube_processing": {
                "download": {
                    "temp_dir": "temp/youtube",
                    "audio_quality": "best",
                    "audio_format": "m4a",
                    "max_file_size_mb": 500,
                    "timeout_seconds": 300,
                },
                "cache": {
                    "enabled": True,
                    "cache_dir": "cache/youtube",
                    "cleanup_after_processing": False,
                    "max_cache_size_gb": 10,
                    "cache_ttl_days": 7,
                },
                "validation": {
                    "allowed_domains": ["youtube.com", "youtu.be"],
                    "min_duration_seconds": 5,
                    "max_duration_seconds": 7200,
                    "allow_live_streams": False,
                },
                "metadata": {
                    "extract_description": True,
                    "extract_tags": True,
                    "max_description_length": 5000,
                },
                "retry": {
                    "max_attempts": 3,
                    "delay_seconds": 5,
                    "exponential_backoff": True,
                },
                "audio_config_path": "config/audio_config.yml",
                "youtube_vocabulary": ["YouTube", "subscribe"],
            }
        }

    @pytest.fixture
    def config_file(
        self, tmp_path: Path, sample_config_data: dict
    ) -> Generator[Path, None, None]:
        """Create a temporary config file for testing."""
        config_path = tmp_path / "youtube_config.yml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(sample_config_data, f)
        yield config_path

    def test_youtube_config_from_yaml(self, config_file: Path) -> None:
        """Test loading YouTubeConfig from YAML file."""
        config = YouTubeConfig.from_yaml(config_file)

        assert config.download.audio_quality == "best"
        assert config.download.audio_format == "m4a"
        assert config.cache.enabled is True
        assert "youtube.com" in config.validation.allowed_domains
        assert config.retry.max_attempts == 3
        assert "YouTube" in config.youtube_vocabulary

    def test_youtube_config_file_not_found(self, tmp_path: Path) -> None:
        """Test that missing config file raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            YouTubeConfig.from_yaml(tmp_path / "nonexistent.yml")

        assert "Configuration file not found" in str(exc_info.value)

    def test_youtube_config_invalid_yaml(self, tmp_path: Path) -> None:
        """Test that invalid YAML raises ConfigurationError."""
        config_path = tmp_path / "invalid.yml"
        config_path.write_text("invalid: yaml: content: :", encoding="utf-8")

        with pytest.raises(ConfigurationError) as exc_info:
            YouTubeConfig.from_yaml(config_path)

        assert "Failed to parse YAML" in str(exc_info.value)

    def test_youtube_config_create_default(self, tmp_path: Path) -> None:
        """Test creating YouTubeConfig with defaults."""
        config = YouTubeConfig.create_default(
            temp_dir=tmp_path / "temp",
            cache_dir=tmp_path / "cache",
        )

        assert config.download.temp_dir == tmp_path / "temp"
        assert config.cache.enabled is True
        assert config.cache.cache_dir == tmp_path / "cache"
        assert config.validation.min_duration_seconds == 5
        assert config.retry.max_attempts == 3

    def test_youtube_config_create_default_no_cache(self, tmp_path: Path) -> None:
        """Test creating YouTubeConfig without cache."""
        config = YouTubeConfig.create_default(temp_dir=tmp_path / "temp")

        assert config.cache.enabled is False
        assert config.cache.cache_dir is None

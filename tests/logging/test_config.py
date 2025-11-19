# ============================================================================
# tests/logging/test_config.py
# Tests for LoggingConfig class
# ============================================================================
"""
Tests for LoggingConfig class.

Validates configuration management and path resolution.
"""
import pytest
import logging
from pathlib import Path
from src.core.logging.config import LoggingConfig


class TestLoggingConfigInitialization:
    """Test LoggingConfig initialization and defaults."""
    
    def test_config_uses_default_logs_dir_when_none_provided(self) -> None:
        """Test that default logs directory is used when not specified."""
        config = LoggingConfig()
        
        assert config.logs_dir.name == "logs"
        assert config.logs_dir.is_absolute()
    
    def test_config_uses_custom_logs_dir_when_provided(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that custom logs directory is used when provided."""
        config = LoggingConfig(logs_dir=temp_logs_dir)
        
        assert config.logs_dir == temp_logs_dir
    
    def test_config_converts_string_path_to_pathlib(self) -> None:
        """Test that string paths are converted to Path objects."""
        config = LoggingConfig(logs_dir="custom_logs")
        
        assert isinstance(config.logs_dir, Path)
        assert str(config.logs_dir) == "custom_logs"
    
    def test_config_generates_log_filename_when_none_provided(self) -> None:
        """Test automatic log filename generation."""
        config = LoggingConfig()
        
        filename = Path(config.log_file_name).name
        assert filename.startswith("app_")
        assert filename.endswith(".log")
    
    def test_config_uses_custom_log_filename_when_provided(
        self, sample_log_file_name: str
    ) -> None:
        """Test that custom log filename is used when provided."""
        config = LoggingConfig(log_file_name=sample_log_file_name)
        
        filename = Path(config.log_file_name).name
        assert filename == sample_log_file_name
    
    @pytest.mark.parametrize("log_level,expected", [
        (logging.DEBUG, logging.DEBUG),
        (logging.INFO, logging.INFO),
        (logging.WARNING, logging.WARNING),
        (logging.ERROR, logging.ERROR),
        (logging.CRITICAL, logging.CRITICAL),
    ])
    def test_config_accepts_various_log_levels(
        self, log_level: int, expected: int
    ) -> None:
        """Test that various log levels are accepted and stored correctly."""
        config = LoggingConfig(log_level=log_level)
        
        assert config.log_level == expected


class TestLoggingConfigProperties:
    """Test LoggingConfig property access and encapsulation."""
    
    def test_logs_dir_property_returns_path(self, temp_logs_dir: Path) -> None:
        """Test that logs_dir property returns Path object."""
        config = LoggingConfig(logs_dir=temp_logs_dir)
        
        assert isinstance(config.logs_dir, Path)
        assert config.logs_dir == temp_logs_dir
    
    def test_log_file_path_combines_dir_and_filename(
        self, temp_logs_dir: Path, sample_log_file_name: str
    ) -> None:
        """Test that log_file_path correctly combines directory and filename."""
        config = LoggingConfig(
            logs_dir=temp_logs_dir,
            log_file_name=sample_log_file_name
        )
        
        expected_path = str(temp_logs_dir / sample_log_file_name)
        assert config.log_file_name == expected_path
    
    def test_file_format_property_returns_format_string(self) -> None:
        """Test that file_format property returns the format string."""
        custom_format = '%(name)s - %(message)s'
        config = LoggingConfig(file_format=custom_format)
        
        assert config.file_format == custom_format
    
    def test_console_format_property_returns_format_string(self) -> None:
        """Test that console_format property returns the format string."""
        custom_format = '%(levelname)s: %(message)s'
        config = LoggingConfig(console_format=custom_format)
        
        assert config.console_format == custom_format


class TestLoggingConfigDirectoryManagement:
    """Test directory creation and management."""
    
    def test_ensure_logs_directory_creates_directory(
        self, tmp_path: Path
    ) -> None:
        """Test that ensure_logs_directory creates the directory."""
        logs_dir = tmp_path / "new_logs"
        config = LoggingConfig(logs_dir=logs_dir)
        
        assert not logs_dir.exists()
        
        config.ensure_logs_dir_exists()
        
        assert logs_dir.exists()
        assert logs_dir.is_dir()
    
    def test_ensure_logs_directory_creates_nested_directories(
        self, tmp_path: Path
    ) -> None:
        """Test that nested directories are created."""
        logs_dir = tmp_path / "level1" / "level2" / "logs"
        config = LoggingConfig(logs_dir=logs_dir)
        
        config.ensure_logs_dir_exists()
        
        assert logs_dir.exists()
        assert logs_dir.is_dir()
    
    def test_ensure_logs_directory_does_not_fail_if_exists(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that no error occurs if directory already exists."""
        config = LoggingConfig(logs_dir=temp_logs_dir)
        
        # Should not raise exception
        config.ensure_logs_dir_exists()
        config.ensure_logs_dir_exists()  # Call twice
        
        assert temp_logs_dir.exists()